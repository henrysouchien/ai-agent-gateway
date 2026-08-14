import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway import AgentRunner
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.providers import ModelInfo, ModelProvider
import agent_gateway.runner as gateway_runner
from agent_gateway.runner_sub_agents import RunnerSubAgentMixin
from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
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
  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return [{
      "name": "web_search",
      "description": "Search the web",
      "input_schema": {"type": "object"},
    }]


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
  runner._loaded_mcp_servers = set()
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
