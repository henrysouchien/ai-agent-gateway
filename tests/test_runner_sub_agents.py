import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway.mcp_activation import McpActivationFold
from agent_gateway import AgentRunner, ToolDispatcher
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.event_log import EventLog
from agent_gateway.session import GatewaySession
from agent_gateway.providers import ModelInfo, ModelProvider
import agent_gateway.runner as gateway_runner
from agent_gateway.runner_sub_agents import RunnerSubAgentMixin
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
from tests.admitted_authority_test_support import (
  SOURCE_TOOL_ID,
  provenance_of,
  sealed_admitted_task,
)
from tests.capability_execution_test_support import (
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


class _Dispatcher:
  def __init__(self) -> None:
    self._event_log = EventLog()
    self._session_id = "parent-session"

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return [{
      "name": "web_search",
      "description": "Search the web",
      "input_schema": {"type": "object"},
    }]


class _NullMcp:
  def is_mcp_tool(self, _name: str) -> bool:
    return False


def _approval_dispatcher(session: GatewaySession) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcp(),
    event_log=EventLog(),
    session_id="parent-session",
    session=session,
    store=object(),
    policy=object(),
    get_tool_definitions=_Dispatcher().get_tool_definitions,
  )


class _EventLog:
  def __init__(self, *, on_event: Any, session_id: str) -> None:
    self._on_event = on_event
    self.session_id = session_id
    self.entries: list[SimpleNamespace] = []

  def append(self, event: dict[str, Any]) -> None:
    self.entries.append(SimpleNamespace(event=event))
    if self._on_event is not None:
      self._on_event(event, self.session_id)


class _ChildRunner:
  instances: list["_ChildRunner"] = []

  def __init__(self, **kwargs: Any) -> None:
    self.kwargs = kwargs
    self._runner_id = f"runner-{kwargs['session_id']}"
    self.run_kwargs: dict[str, Any] | None = None
    self.closed = False
    self.instances.append(self)

  async def run(self, **kwargs: Any) -> None:
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

  async def run(self, **kwargs: Any) -> None:
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


class _ApprovalChildRunner(_ChildRunner):
  approved = True

  async def run(self, **kwargs: Any) -> None:
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


class _SessionLog:
  def __init__(self, text: str) -> None:
    self.text = text

  async def query(self, **kwargs: Any) -> tuple[list[Any], None]:
    assert kwargs["event_types"] == {"assistant_message"}
    return [SimpleNamespace(seq=41, event={
      "type": "assistant_message",
      "stop_reason": "end_turn",
      "logical_response_id": "logical-test-response",
      "logical_response_segment_ordinal": 0,
      "content_blocks": [{"type": "text", "text": self.text}],
    })], None


def _parent(tmp_path: Path, *, session_log: _SessionLog | None) -> AgentRunner:
  runner = object.__new__(AgentRunner)
  runner._sub_agent_config = None
  runner._provider = SimpleNamespace(name="parent-provider")
  runner._auth_config = {"api_key": "parent"}
  runner._full_session_id = "parent-session"
  runner._log = SimpleNamespace(_on_event=None)
  runner._per_turn_timeout = 11.0
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
  runner._usage_ledger_dlq_path = None
  runner._on_metric = None
  runner._compaction_trigger = 0.8
  runner._tool_call_timeout = 13.0
  runner._on_max_turns = None
  runner._aggregator = object()
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
    "timeout": None,
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
    "timeout": None,
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
