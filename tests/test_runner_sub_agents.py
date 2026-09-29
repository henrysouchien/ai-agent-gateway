import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.agent_session_log_records import LogEntry as SessionLogEntry, QueryCursor
from agent_gateway.mcp_activation import McpActivationFold
from agent_gateway import AgentRunner, ToolDispatcher
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.event_log import EventLog
from agent_gateway.session import GatewaySession
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.providers.base import StreamEvent
from agent_gateway.runner_budget import (
  ChildCostAccumulator,
  ObservationOnlyCostAccumulator,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.multi_user.billing import _UsageAggregator
import agent_gateway.runner as gateway_runner
import agent_gateway.runner_sub_agents as runner_sub_agents
import agent_gateway.runner_stream_turn as runner_stream_turn
from agent_gateway.runner_sub_agents import RunnerSubAgentMixin
from agent_gateway.sub_agent_result_evidence import SubAgentResultEvidence
from agent_gateway.task_registry import TaskEntry
from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
)
from gateway_test_support.admitted_authority_test_support import (
  SOURCE_TOOL_ID,
  provenance_of,
  sealed_admitted_task,
)
from gateway_test_support.capability_execution_test_support import (
  stub_bound_capability_execution,
)


class _Provider(ModelProvider):
  name = "child-provider"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return config.get("api_key") == "child-secret"

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=64_000,
      supports_thinking=True,
    )


_DIGEST = "sha256:" + "1" * 64
_OPERATION = AgentOperationRef(
  namespace="agent-operation",
  name="test-child",
  version="1",
  digest=_DIGEST,
)
_LOGICAL_TASK = OrdinaryDelegationTaskRef(
  delegation_id="delegation:test-child",
  operation=_OPERATION,
)
_ATTEMPT = AttemptRef(
  attempt_number=1,
  attempt_id="attempt:test-child:1",
  physical_task_id="task:test-child",
)
_PROVENANCE = TaskResultProvenance(
  admitted_task_digest=_DIGEST,
  model_bind_digest=_DIGEST,
  capability_binding_digest=_DIGEST,
  tool_grant_digest=_DIGEST,
)
_RESULT = ResultRequirement(
  mode="narrative",
  terminal_narrative="required",
  outcome=OutcomeRequirement(required=False, source="none"),
)


def _execution(capability_id: str = "node.explore") -> BoundCapabilityExecution:
  return stub_bound_capability_execution(
    provider=_Provider(),
    model="child-model",
    effort="medium",
    capability_id=capability_id,
    credential_principal="user",
    auth_config={
      "api_key": "child-secret",
    },
  )


class _Dispatcher(ToolDispatcher):
  def __init__(self) -> None:
    self._event_log = EventLog()
    self._session_id = "parent-session"

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return [{
      "name": "web_search",
      "description": "Search the web",
      "input_schema": {"type": "object"},
    }]


class _ParentProvider(ModelProvider):
  name = "parent-provider"

def _approval_dispatcher(session: GatewaySession) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=McpClientManager(),
    event_log=EventLog(),
    session_id="parent-session",
    approval_route=DurableLocalApprovalRoute(
      object(),
      object(),
      session,
    ),
    get_tool_definitions=_Dispatcher().get_tool_definitions,
  )


class _EventLog(EventLog):
  pass


class _ChildRunner:
  instances: list["_ChildRunner"] = []
  approval_result: dict[str, object]


  def __init__(self, **kwargs: Any) -> None:
    self.kwargs = kwargs
    self._runner_id = f"runner-{kwargs['session_id']}"
    self.run_kwargs: dict[str, object] | None = None
    self.closed = False
    self.instances.append(self)

  async def run(self, **kwargs: object) -> None:
    self.run_kwargs = kwargs
    self.kwargs["event_log"].append({
      "type": "stream_complete",
      "usage": {"input_tokens": 11, "output_tokens": 7},
    })

  async def force_close(self, timeout: float = 2.0) -> None:
    assert timeout == 2.0
    self.closed = True


class _FailedRetrievalChildRunner(_ChildRunner):
  """A child whose only granted source-capability retrieval failed."""

  async def run(self, **kwargs: object) -> None:
    self.run_kwargs = kwargs
    self.kwargs["event_log"].append({
      "type": "tool_call_complete",
      "tool_call_id": "call-web-search",
      "tool_name": SOURCE_TOOL_ID,
      "is_error": True,
    })
    self.kwargs["event_log"].append({
      "type": "stream_complete",
      "usage": {"input_tokens": 11, "output_tokens": 7},
    })


_TERMINAL_TOOL = "fms_propose_position_initiation_predecision"
_TERMINAL_RESULT = {
  "status": "staged",
  "proposal_id": "proposal-1",
  "artifact_ref": "artifacts/PCTY/predecision.json",
}


class _TerminalToolChildRunner(_ChildRunner):
  async def run(self, **kwargs: object) -> None:
    self.run_kwargs = kwargs
    assert self.kwargs["terminal_tool_result_ids"] == {_TERMINAL_TOOL}
    self.kwargs["event_log"].append({
      "type": "tool_call_start",
      "tool_name": _TERMINAL_TOOL,
    })
    self.kwargs["event_log"].append({
      "type": "tool_call_complete",
      "tool_name": _TERMINAL_TOOL,
      "result": dict(_TERMINAL_RESULT),
      "dispatch": {"outcome": "ok"},
      "is_error": False,
      "error": None,
      "semantic_error": None,
    })
    self.kwargs["event_log"].append({
      "type": "stream_complete",
      "usage": {"input_tokens": 11, "output_tokens": 7},
    })


class _CancelledAfterTerminalChildRunner(_TerminalToolChildRunner):
  async def run(self, **kwargs: object) -> None:
    await super().run(**kwargs)
    raise asyncio.CancelledError


class _ApprovalChildRunner(_ChildRunner):
  approved = True

  async def run(self, **kwargs: object) -> None:
    self.run_kwargs = kwargs
    dispatcher = self.kwargs["dispatcher"]
    approval_task = asyncio.create_task(
      dispatcher._await_user_approval_via_pending_tools(
        SimpleNamespace(
          approval_id="approval-start-quant",
          tool_call_id="call-start-quant",
          tool_name="start_quant_research",
          tool_args_redacted={"request": {"research_file_id": 42}},
        ),
        SimpleNamespace(
          reason="state mutation requires approval",
          allow_persistent_grant=False,
        ),
        nonce="nonce-start-quant",
        resolved_qualifier="",
        allow_persistent=False,
        timeout_seconds=5,
      )
    )
    session = dispatcher._session
    for _ in range(100):
      if "call-start-quant" in session.approval_queues:
        break
      await asyncio.sleep(0)
    else:
      raise AssertionError("child approval was not registered on parent session")
    session.approval_queues["call-start-quant"].put_nowait({
      "approved": self.approved,
      "allow_tool_type": False,
      "approval_id": "approval-start-quant",
    })
    self.approval_result = await approval_task
    assert session.pending_tools == {}
    assert session.approval_queues == {}
    dispatcher._event_log.append({
      "type": "tool_call_complete",
      "tool_call_id": "call-start-quant",
      "tool_name": "start_quant_research",
      "result": {"approved": self.approved},
    })
    self.kwargs["event_log"].append({
      "type": "stream_complete",
      "usage": {"input_tokens": 11, "output_tokens": 7},
    })


class _SessionLog(AgentSessionLog):
  def __init__(self, text: str) -> None:
    self.text = text

  async def query(self, **kwargs: Any) -> tuple[list[SessionLogEntry], QueryCursor | None]:
    assert kwargs["event_types"] == {"assistant_message"}
    return [SessionLogEntry(seq=41, timestamp=0.0, event={
      "type": "assistant_message",
      "stop_reason": "end_turn",
      "logical_response_id": "logical-test-response",
      "logical_response_segment_ordinal": 0,
      "content_blocks": [{"type": "text", "text": self.text}],
    })], None


def _parent(tmp_path: Path, *, session_log: _SessionLog | AgentSessionLog | None) -> AgentRunner:
  runner = object.__new__(AgentRunner)
  runner._sub_agent_config = None
  runner._provider = _ParentProvider()
  runner._auth_config = {"api_key": "parent"}
  runner._full_session_id = "parent-session"
  runner._log = EventLog()
  runner._stream_stall_timeout = 12.0
  runner._mcp_client = None
  runner._mcp_activation_fold = McpActivationFold()
  runner._get_tool_definitions = lambda: []
  runner._on_tool_result = None
  runner._on_usage = None
  runner._on_late_usage_event = None
  runner._on_tool_timing = None
  runner._usage_user_id = "alice"
  runner._request_id = "req-1"
  runner._billing_mode = "metered"
  runner._rate_table_version = "v1"
  runner._channel = "web"
  runner._usage_ledger_dlq_path = tmp_path / "usage-ledger-dlq.jsonl"
  runner._on_metric = None
  runner._compaction_trigger = 80
  runner._tool_call_timeout = 13.0
  runner._on_max_turns = None
  runner._aggregator = _UsageAggregator(
    user_id="alice",
    session_id="parent-session",
    request_id="req-1",
    channel="web",
    rate_table_version="v1",
    billing_mode="metered",
  )
  runner._max_concurrent_sub_agents = 2
  runner._agent_session_log = session_log
  runner._max_resume_chain_depth = 3
  runner._spill_dir_provider = None
  runner._skill_run_id = "skill-run"
  runner._workspace_dir = str(tmp_path)
  runner._context_surfaces_provider = None
  runner._context_surfaces_static = []
  runner._commercial_usage_producer = None
  runner._batch_id = None
  return runner


def _spawn(parent: AgentRunner, **overrides: Any):
  kwargs = {
    "capability_execution": _execution(),
    "skill_name": "explore",
    "logical_task": _LOGICAL_TASK,
    "attempt": _ATTEMPT,
    "result_requirement": _RESULT,
    "result_provenance": _PROVENANCE,
    "dispatcher": _Dispatcher(),
    "max_turns": 4,
  }
  kwargs.update(overrides)
  return asyncio.run(parent.spawn_sub_agent("research this", **kwargs))


def _resume(parent: AgentRunner, **overrides: Any):
  kwargs = {
    "original_task_id": "task:prior",
    "reconstructed_messages": [{"role": "user", "content": "resume"}],
    "parent_messages": [],
    "capability_execution": _execution(),
    "skill_name": "explore",
    "logical_task": _LOGICAL_TASK,
    "attempt": _ATTEMPT,
    "result_requirement": _RESULT,
    "result_provenance": _PROVENANCE,
    "dispatcher": _Dispatcher(),
    "max_turns": 4,
  }
  kwargs.update(overrides)
  return asyncio.run(parent.resume_sub_agent(**kwargs))


def test_runner_sub_agent_methods_are_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerSubAgentMixin)
  for method_name in ("spawn_sub_agent", "resume_sub_agent"):
    assert getattr(AgentRunner, method_name) is getattr(
      RunnerSubAgentMixin, method_name
    )


def test_spawn_sub_agent_materializes_exact_terminal_message(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _ChildRunner.instances.clear()
  terminal = "A complete research report with a material caveat."
  parent = _parent(tmp_path, session_log=_SessionLog(terminal))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _ChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _spawn(parent)

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  assert result.logical_task == _LOGICAL_TASK
  assert result.attempt == _ATTEMPT
  assert result.values.terminal_narrative is not None
  assert result.values.terminal_narrative.content_chars == len(terminal)
  assert result.values.projection is None
  child = _ChildRunner.instances[0]
  assert [tool["name"] for tool in child.kwargs["get_tool_definitions"]()] == [
    "web_search"
  ]
  assert child.run_kwargs == {
    "messages": [{"role": "user", "content": "research this"}],
    "system_prompt": None,
    "max_turns": 4,
  }
  assert child.closed is True


def test_spawn_sub_agent_never_injects_a_result_submission_tool(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  parent = _parent(tmp_path, session_log=_SessionLog("Done."))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _ChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  _spawn(parent)

  names = {
    tool["name"]
    for tool in _ChildRunner.instances[-1].kwargs["get_tool_definitions"]()
  }
  assert "submit_report" not in names


@pytest.mark.parametrize("method", ["spawn", "resume"])
def test_named_child_projects_accepted_terminal_tool_result(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  method: str,
) -> None:
  _TerminalToolChildRunner.instances.clear()
  admitted = sealed_admitted_task(
    logical_task=_LOGICAL_TASK,
    attempt=_ATTEMPT,
    result_requirement=_RESULT,
    tool_id=_TERMINAL_TOOL,
  )
  parent = _parent(tmp_path, session_log=_SessionLog("must not be read"))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _TerminalToolChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = (
    _spawn(
      parent,
      admitted_task=admitted,
      result_provenance=provenance_of(admitted),
    )
    if method == "spawn"
    else _resume(
      parent,
      admitted_task=admitted,
      result_provenance=provenance_of(admitted),
    )
  )

  assert error is None
  assert result is not None
  assert result.execution.status == "succeeded"
  assert result.values.terminal_narrative is None
  assert result.values.projection is not None
  assert result.values.projection.inline_view == {
    "tool_name": _TERMINAL_TOOL,
    "result": _TERMINAL_RESULT,
  }


def test_resume_short_circuits_prior_terminal_success_before_provider(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _ChildRunner.instances.clear()
  admitted = sealed_admitted_task(
    logical_task=_LOGICAL_TASK,
    attempt=_ATTEMPT,
    result_requirement=_RESULT,
    tool_id=_TERMINAL_TOOL,
  )
  parent = _parent(tmp_path, session_log=_SessionLog("must not be read"))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _ChildRunner)

  result, error = _resume(
    parent,
    admitted_task=admitted,
    result_provenance=provenance_of(admitted),
    prior_evidence=SubAgentResultEvidence(
      usage={"tool_calls": 1},
      tools_used=(_TERMINAL_TOOL,),
      fms_results=({"tool_name": _TERMINAL_TOOL, **_TERMINAL_RESULT},),
      artifact_events=(),
      warning_parts=(),
    ),
    prior_terminal_tool_result={
      "tool_name": _TERMINAL_TOOL,
      "result": _TERMINAL_RESULT,
    },
  )

  assert error is None
  assert result is not None
  assert result.execution.status == "succeeded"
  assert result.values.projection is not None
  assert isinstance(result.values.projection.inline_view, dict)
  assert result.values.projection.inline_view["result"] == _TERMINAL_RESULT
  assert _ChildRunner.instances == []


def test_terminal_success_wins_cancellation_after_durable_tool_event(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  admitted = sealed_admitted_task(
    logical_task=_LOGICAL_TASK,
    attempt=_ATTEMPT,
    result_requirement=_RESULT,
    tool_id=_TERMINAL_TOOL,
  )
  parent = _parent(tmp_path, session_log=_SessionLog("must not be read"))
  monkeypatch.setattr(
    gateway_runner,
    "AgentRunner",
    _CancelledAfterTerminalChildRunner,
  )
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _spawn(
    parent,
    admitted_task=admitted,
    result_provenance=provenance_of(admitted),
  )

  assert error is None
  assert result is not None
  assert result.execution.status == "succeeded"
  assert result.values.projection is not None


def test_spawn_sub_agent_uses_exact_child_skill_run_identity(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _ChildRunner.instances.clear()
  parent = _parent(tmp_path, session_log=_SessionLog("Done."))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _ChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _spawn(parent, skill_run_id="child-skill-run")

  assert error is None
  assert isinstance(result, TaskResult)
  assert _ChildRunner.instances[0].kwargs["skill_run_id"] == "child-skill-run"


class _WedgedChildRunner(_ChildRunner):
  """A child whose inner await never resolves and that emits nothing."""

  cancelled = False

  async def run(self, **kwargs: object) -> None:
    self.run_kwargs = kwargs
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError:
      self.cancelled = True
      raise


class _HeartbeatOnlyChildRunner(_ChildRunner):
  """A live child whose only events, past the gap, are stream heartbeats."""

  cancelled = False

  async def run(self, **kwargs: object) -> None:
    try:
      for _ in range(10):
        await asyncio.sleep(0.02)
        self.kwargs["event_log"].append({
          "type": "heartbeat",
          "elapsed_s": 0,
          "last_progress_s": 0,
          "events": 1,
        })
    except asyncio.CancelledError:
      self.cancelled = True
      raise
    await super().run(**kwargs)


class _ToolInFlightChildRunner(_ChildRunner):
  """A child silent past the gap while one of its tool calls is running."""

  cancelled = False

  async def run(self, **kwargs: object) -> None:
    event_log = self.kwargs["event_log"]
    event_log.append({
      "type": "tool_call_start",
      "tool_call_id": "call-1",
      "tool_name": "web_search",
      "tool_input": {"query": "filings"},
    })
    try:
      await asyncio.sleep(0.2)
    except asyncio.CancelledError:
      self.cancelled = True
      raise
    event_log.append({
      "type": "tool_call_complete",
      "tool_call_id": "call-1",
      "tool_name": "web_search",
      "result": {"results": []},
    })
    await super().run(**kwargs)


@pytest.fixture
def _short_activity_gap(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(runner_sub_agents, "SUB_AGENT_ACTIVITY_GAP", 0.05)
  monkeypatch.setattr(runner_sub_agents, "STREAM_GUARD_POLL_INTERVAL", 0.01)


@pytest.mark.usefixtures("_short_activity_gap")
def test_wedged_child_is_cancelled_by_the_activity_guard(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _ChildRunner.instances.clear()
  parent = _parent(tmp_path, session_log=_SessionLog("never reached"))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _WedgedChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _spawn(parent)

  child = _ChildRunner.instances[-1]
  assert isinstance(child, _WedgedChildRunner)
  assert child.cancelled is True
  assert child.closed is True
  assert error is None
  assert result is not None
  assert result.execution.status == "interrupted"
  assert result.execution.terminal_reason is not None
  assert result.execution.terminal_reason.startswith(
    "stalled: Sub-agent stalled: no activity for"
  )
  errors = [
    entry.event
    for entry in child.kwargs["event_log"].entries
    if entry.event.get("type") == "error"
  ]
  assert [event["error_sub_code"] for event in errors] == ["stalled"]


@pytest.mark.usefixtures("_short_activity_gap")
@pytest.mark.parametrize(
  "child_cls",
  [_HeartbeatOnlyChildRunner, _ToolInFlightChildRunner],
  ids=["heartbeats-only", "tool-in-flight"],
)
def test_live_child_outlasts_the_activity_gap(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  child_cls: type[_ChildRunner],
) -> None:
  _ChildRunner.instances.clear()
  parent = _parent(tmp_path, session_log=_SessionLog("Done."))
  monkeypatch.setattr(gateway_runner, "AgentRunner", child_cls)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _spawn(parent)

  child = _ChildRunner.instances[-1]
  assert getattr(child, "cancelled") is False
  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"


class _ToolInputStreamingProvider(_Provider):
  """A child provider composing a large tool input: deltas only, past the gap."""

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    return None

  def build_request_params(self, **_kwargs: Any) -> dict[str, Any]:
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    deadline = time.monotonic() + 0.3
    while time.monotonic() < deadline:
      await asyncio.sleep(0.01)
      yield StreamEvent(type="tool_use_delta", tool_input_json="x")
    yield StreamEvent(type="text_delta", text="Done.")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "Done."})
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def test_real_child_streaming_tool_input_outlasts_the_activity_gap(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setattr(runner_sub_agents, "SUB_AGENT_ACTIVITY_GAP", 0.1)
  monkeypatch.setattr(runner_sub_agents, "STREAM_GUARD_POLL_INTERVAL", 0.01)
  monkeypatch.setattr(runner_stream_turn, "STREAM_PROGRESS_LOG_INTERVAL", 0.02)
  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.02)
  parent = _parent(tmp_path, session_log=AgentSessionLog(tmp_path / "session.jsonl"))
  execution = stub_bound_capability_execution(
    provider=_ToolInputStreamingProvider(),
    model="child-model",
    effort="medium",
    capability_id="node.explore",
    credential_principal="user",
    auth_config={"api_key": "child-secret"},
  )
  observed: list[object] = []

  result, error = _spawn(
    parent,
    capability_execution=execution,
    on_sub_event=lambda event, _sid: observed.append(event.get("type")),
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  # The stream's only runner-visible output before its text is the guard's
  # heartbeat, and that is what kept the parent from calling the child wedged.
  first_text = observed.index("text_delta")
  assert observed[:first_text]
  assert set(observed[:first_text]) == {"heartbeat"}
  assert "error" not in observed


class _RefusingChildProvider(_Provider):
  """A child provider that delivers text, then stops with ``refusal``."""

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    return None

  def build_request_params(self, **_kwargs: Any) -> dict[str, Any]:
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    yield StreamEvent(type="text_delta", text="Partial findings.")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "Partial findings."})
    yield StreamEvent(
      type="message_end",
      stop_reason="refusal",
      stop_details={
        "type": "refusal",
        "category": "reasoning_extraction",
        "explanation": "The request asks for internal reasoning.",
      },
    )


def test_refused_child_settles_with_refusal_signal_and_keeps_its_narrative(
  tmp_path: Path,
) -> None:
  from agent_gateway.sub_agent_narrative_result import (
    read_task_result_terminal_narrative,
  )
  from agent_gateway.sub_agent_skill_state import classify_child_outcome

  parent = _parent(tmp_path, session_log=AgentSessionLog(tmp_path / "session.jsonl"))
  execution = stub_bound_capability_execution(
    provider=_RefusingChildProvider(),
    model="child-model",
    effort="medium",
    capability_id="node.explore",
    credential_principal="user",
    auth_config={"api_key": "child-secret"},
  )
  observed: list[dict[str, Any]] = []

  result, error = _spawn(
    parent,
    capability_execution=execution,
    on_sub_event=lambda event, _sid: observed.append(dict(event)),
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "interrupted"
  assert result.execution.terminal_reason == "refusal: reasoning_extraction"
  assert read_task_result_terminal_narrative(
    result,
    workspace_dir=str(tmp_path),
  ) == "Partial findings."
  classification = classify_child_outcome(result.model_dump(mode="json"), None)
  assert classification.succeeded is False
  assert classification.error is not None
  assert classification.error["code"] == "refusal: reasoning_extraction"
  types = [event.get("type") for event in observed]
  assert "refusal" in types
  assert not {"error", "run_error"} & set(types)


@pytest.mark.parametrize("method", ["spawn", "resume"])
def test_named_child_budget_wraps_observation_accumulator(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  method: str,
) -> None:
  _ChildRunner.instances.clear()
  parent = _parent(tmp_path, session_log=_SessionLog("Done."))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _ChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = (
    _spawn(
      parent,
      max_budget_usd=6.0,
      cost_observation_threshold_usd=3.0,
    )
    if method == "spawn"
    else _resume(
      parent,
      max_budget_usd=6.0,
      cost_observation_threshold_usd=3.0,
    )
  )

  assert error is None
  assert isinstance(result, TaskResult)
  child_kwargs = _ChildRunner.instances[0].kwargs
  accumulator = child_kwargs["_cost_accumulator"]
  assert child_kwargs["max_budget_usd"] == pytest.approx(6.0)
  assert isinstance(accumulator, ChildCostAccumulator)
  assert isinstance(accumulator._parent, ObservationOnlyCostAccumulator)
  assert accumulator._parent.observation_threshold_usd == pytest.approx(3.0)
  accumulator.add(1.25)
  assert accumulator.total == pytest.approx(1.25)
  assert accumulator._parent.total == pytest.approx(1.25)


@pytest.mark.parametrize("method", ["spawn", "resume"])
def test_legacy_child_without_budget_remains_observation_only(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  method: str,
) -> None:
  _ChildRunner.instances.clear()
  parent = _parent(tmp_path, session_log=_SessionLog("Done."))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _ChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _spawn(parent) if method == "spawn" else _resume(parent)

  assert error is None
  assert isinstance(result, TaskResult)
  child_kwargs = _ChildRunner.instances[0].kwargs
  assert child_kwargs["max_budget_usd"] is None
  assert isinstance(
    child_kwargs["_cost_accumulator"],
    ObservationOnlyCostAccumulator,
  )


@pytest.mark.parametrize("method,approved", [
  ("spawn", True),
  ("spawn", False),
  ("resume", True),
  ("resume", False),
])
def test_child_approval_uses_parent_session_and_parent_visible_child_log(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  method: str,
  approved: bool,
) -> None:
  _ChildRunner.instances.clear()
  parent_events: list[tuple[dict[str, Any], str]] = []
  parent = _parent(tmp_path, session_log=_SessionLog("Done."))
  parent._log = EventLog(
    on_event=lambda event, session_id: parent_events.append(
      (event, session_id)
    ),
    session_id="parent-session",
  )
  parent._log.append({"type": "text_delta", "text": "before child"})
  parent_session = GatewaySession(
    session_id="parent-session",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="owner",
  )
  dispatcher = _approval_dispatcher(parent_session)
  _ApprovalChildRunner.approved = approved
  monkeypatch.setattr(gateway_runner, "AgentRunner", _ApprovalChildRunner)

  result, error = (
    _spawn(parent, dispatcher=dispatcher)
    if method == "spawn"
    else _resume(parent, dispatcher=dispatcher)
  )

  assert error is None
  assert isinstance(result, TaskResult)
  approval_event, delivery_session_id = next(
    item for item in parent_events
    if item[0]["type"] == "tool_approval_request"
  )
  assert approval_event == {
    "type": "tool_approval_request",
    "tool_call_id": "call-start-quant",
    "approval_id": "approval-start-quant",
    "nonce": "nonce-start-quant",
    "tool_name": "start_quant_research",
    "tool_input": {"request": {"research_file_id": 42}},
    "resolved_qualifier": "",
    "reason": "state mutation requires approval",
    "allow_persistent_approval": False,
    "ts": approval_event["ts"],
    "sub_agent_id": "sub0:parent-session",
  }
  assert delivery_session_id == "parent-session"
  assert [entry.seq for entry in parent._log.entries] == [1, 2]
  assert parent._log.entries[1].event == approval_event
  assert dispatcher._event_log is _ChildRunner.instances[0].kwargs[
    "event_log"
  ]
  assert dispatcher._session_id == "parent-session"
  assert _ChildRunner.instances[0].approval_result == {
    "approved": approved,
    "allow_tool_type": False,
    "approval_id": "approval-start-quant",
  }
  assert parent_session.pending_tools == {}
  assert parent_session.approval_queues == {}


def test_spawn_sub_agent_requires_durable_terminal_message_log(
  tmp_path: Path,
) -> None:
  with pytest.raises(
    ValueError,
    match="narrative child execution requires a durable session log",
  ):
    _spawn(_parent(tmp_path, session_log=None))


def test_spawn_sub_agent_rejects_non_node_capability(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match=r"requires a node\.\* capability bind"):
    _spawn(
      _parent(tmp_path, session_log=_SessionLog("Done.")),
      capability_execution=_execution("session.driver"),
    )


def _admitted_entry() -> TaskEntry:
  """A registry entry carrying the authority frozen at admission (B-3)."""

  admitted = sealed_admitted_task(
    logical_task=_LOGICAL_TASK,
    attempt=_ATTEMPT,
    result_requirement=_RESULT,
  )
  return TaskEntry(
    task_id=_ATTEMPT.physical_task_id,
    task_type="agent",
    admitted_task=admitted,
  )


def test_spawn_settlement_derives_the_outcome_from_admitted_authority(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  # The runner-to-constructor handoff (B-3): ``spawn_sub_agent`` passes the
  # entry's admitted task into settlement, so the derivation sees the grant
  # and the bindings frozen at admission. Drop that pass and the settled
  # result carries no outcome at all.
  _ChildRunner.instances.clear()
  entry = _admitted_entry()
  assert entry.admitted_task is not None
  parent = _parent(tmp_path, session_log=_SessionLog("Partial findings."))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _FailedRetrievalChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _spawn(
    parent,
    result_provenance=provenance_of(entry.admitted_task),
    task_entry=entry,
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  assert result.outcome is not None
  assert result.outcome.assessment_source == "mechanically_derived"
  # The grant intersected with the source-capability binding is exactly
  # ``web_search``; its only retrieval failed.
  assert result.outcome.disposition == "insufficient_evidence"
  assert result.outcome.unmet_requirements == (SOURCE_TOOL_ID,)


def test_spawn_settlement_without_an_admitted_entry_derives_no_outcome(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  # The negative control on the same execution: with no admitted authority in
  # scope no assessment occurred, so the settled result stays unqualified.
  # Together with the test above this pins that the outcome is read from the
  # admitted task and never from the ambient tool surface.
  _ChildRunner.instances.clear()
  parent = _parent(tmp_path, session_log=_SessionLog("Partial findings."))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _FailedRetrievalChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _spawn(parent)

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  assert result.outcome is None


def test_resume_settlement_derives_the_outcome_from_admitted_authority(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  # The second settlement site: ``resume_sub_agent`` threads the same admitted
  # authority, so a resumed segment settles with a mechanical qualifier too.
  _ChildRunner.instances.clear()
  entry = _admitted_entry()
  assert entry.admitted_task is not None
  parent = _parent(tmp_path, session_log=_SessionLog("Partial findings."))
  monkeypatch.setattr(gateway_runner, "AgentRunner", _FailedRetrievalChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _EventLog)

  result, error = _resume(
    parent,
    result_provenance=provenance_of(entry.admitted_task),
    task_entry=entry,
  )

  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "succeeded"
  assert result.outcome is not None
  assert result.outcome.assessment_source == "mechanically_derived"
  assert result.outcome.disposition == "insufficient_evidence"
  assert result.outcome.unmet_requirements == (SOURCE_TOOL_ID,)
