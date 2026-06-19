import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_sub_agents import RunnerSubAgentMixin  # noqa: E402


class _FakeChildRunner:
  instances: list["_FakeChildRunner"] = []

  def __init__(self, **kwargs: Any) -> None:
    self.kwargs = kwargs
    self.run_kwargs: dict[str, Any] | None = None
    self.closed_timeout: float | None = None
    self.instances.append(self)

  async def run(self, **kwargs: Any) -> None:
    self.run_kwargs = kwargs
    self.kwargs["event_log"].append({"type": "message", "content": "child result"})

  async def force_close(self, timeout: float = 2.0) -> None:
    self.closed_timeout = timeout


class _FakeEventLog:
  instances: list["_FakeEventLog"] = []

  def __init__(self, *, on_event: Any, session_id: str) -> None:
    self._on_event = on_event
    self.session_id = session_id
    self.entries: list[SimpleNamespace] = []
    self.instances.append(self)

  def append(self, event: dict[str, Any]) -> None:
    self.entries.append(SimpleNamespace(event=event))
    if self._on_event is not None:
      self._on_event(event, self.session_id)


def _parent_runner() -> AgentRunner:
  runner = object.__new__(AgentRunner)
  runner._sub_agent_config = None
  runner._provider = SimpleNamespace(name="parent-provider")
  runner._auth_config = {"api_key": "parent"}
  runner._full_session_id = "parent-session"
  runner._log = SimpleNamespace(_on_event=None)
  runner._max_budget_usd = 10.0
  runner._cost_accumulator = SimpleNamespace(exceeded_reason=None)
  runner._per_turn_timeout = 11.0
  runner._stream_stall_timeout = 12.0
  runner._mcp_client = None
  runner._loaded_mcp_servers = {"server-a"}
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
  runner._agent_session_log = None
  runner._max_resume_chain_depth = 3
  runner._spill_dir_provider = None
  runner._skill_run_id = "skill-run"
  runner._workspace_dir = "/workspace"
  runner._context_surfaces_provider = None
  runner._context_surfaces_static = [{"surface": "test"}]
  return runner


def test_runner_sub_agent_methods_are_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerSubAgentMixin)
  assert gateway_runner.RunnerSubAgentMixin is RunnerSubAgentMixin

  for method_name in ("spawn_sub_agent", "resume_sub_agent"):
    assert getattr(AgentRunner, method_name) is getattr(RunnerSubAgentMixin, method_name)


def test_spawn_sub_agent_resolves_parent_module_helpers(monkeypatch: Any) -> None:
  _FakeChildRunner.instances.clear()
  result_calls: list[dict[str, Any]] = []

  monkeypatch.setattr(gateway_runner, "AgentRunner", _FakeChildRunner)
  monkeypatch.setattr(
    gateway_runner,
    "_derive_sub_agent_id",
    lambda session_id, call_index: f"patched-{session_id}-{call_index}",
  )
  monkeypatch.setattr(gateway_runner, "_user_turn_message", lambda task: {"role": "patched_user", "content": task})
  monkeypatch.setattr(
    gateway_runner,
    "_sub_agent_result_from_log_entries",
    lambda entries, **kwargs: result_calls.append({"entries": list(entries), "kwargs": kwargs}) or {"ok": kwargs},
  )

  result, error = asyncio.run(
    _parent_runner().spawn_sub_agent(
      "review this",
      dispatcher=object(),
      max_turns=4,
      timeout=None,
      call_index=7,
      parent_turn_id="turn-1",
    )
  )

  child = _FakeChildRunner.instances[0]
  assert error is None
  assert result == {"ok": {"timed_out": False, "timeout": None, "budget_exceeded_reason": None}}
  assert child.kwargs["session_id"] == "patched-parent-session-7"
  assert child.kwargs["parent_turn_id"] == "turn-1"
  assert child.kwargs["context_surfaces"] == [{"surface": "test"}]
  assert child.run_kwargs == {
    "messages": [{"role": "patched_user", "content": "review this"}],
    "system_prompt": None,
    "model_override": None,
    "max_turns": 4,
  }
  assert child.closed_timeout == 2.0
  assert [entry.event for entry in result_calls[0]["entries"]] == [{"type": "message", "content": "child result"}]


def test_spawn_sub_agent_resolves_progress_eventlog_and_child_budget_aliases(monkeypatch: Any) -> None:
  _FakeChildRunner.instances.clear()
  _FakeEventLog.instances.clear()
  progress_calls: list[tuple[dict[str, Any], str]] = []
  original_calls: list[tuple[dict[str, Any], str]] = []
  sub_event_calls: list[tuple[dict[str, Any], str]] = []
  accumulator_calls: list[tuple[Any, float]] = []
  task_entry = SimpleNamespace(message_inbox=object())

  class _FakeChildAccumulator:
    def __init__(self, parent: Any, max_budget_usd: float) -> None:
      accumulator_calls.append((parent, max_budget_usd))
      self.exceeded_reason = "child_budget"

  def _make_progress_tracker(entry: Any):
    assert entry is task_entry

    def _progress(event: dict[str, Any], session_id: str) -> None:
      progress_calls.append((event, session_id))

    return _progress

  parent = _parent_runner()
  parent._log = SimpleNamespace(_on_event=lambda event, session_id: original_calls.append((event, session_id)))

  monkeypatch.setattr(gateway_runner, "AgentRunner", _FakeChildRunner)
  monkeypatch.setattr(gateway_runner, "EventLog", _FakeEventLog)
  monkeypatch.setattr(gateway_runner, "ChildCostAccumulator", _FakeChildAccumulator)
  monkeypatch.setattr(gateway_runner, "make_progress_tracker", _make_progress_tracker)
  monkeypatch.setattr(
    gateway_runner,
    "_sub_agent_result_from_log_entries",
    lambda entries, **kwargs: {"entries": [entry.event for entry in entries], "kwargs": kwargs},
  )

  result, error = asyncio.run(
    parent.spawn_sub_agent(
      "track progress",
      dispatcher=object(),
      max_turns=1,
      timeout=None,
      task_entry=task_entry,
      max_budget_usd=1.25,
      on_sub_event=lambda event, session_id: sub_event_calls.append((event, session_id)),
    )
  )

  child = _FakeChildRunner.instances[0]
  assert error is None
  assert _FakeEventLog.instances[0] is child.kwargs["event_log"]
  assert child.kwargs["max_budget_usd"] == 1.25
  assert child.kwargs["_cost_accumulator"].exceeded_reason == "child_budget"
  assert accumulator_calls == [(parent._cost_accumulator, 1.25)]
  assert result == {
    "entries": [{"type": "message", "content": "child result"}],
    "kwargs": {"timed_out": False, "timeout": None, "budget_exceeded_reason": "child_budget"},
  }
  for calls in (progress_calls, original_calls, sub_event_calls):
    assert calls == [({"type": "message", "content": "child result", "sub_agent_id": "sub0:parent-session"}, "sub0:parent-session")]


def test_spawn_sub_agent_timeout_uses_parent_asyncio_alias(monkeypatch: Any) -> None:
  _FakeChildRunner.instances.clear()
  waited: dict[str, float] = {}

  def _wait_for(coro: Any, *, timeout: float) -> Any:
    waited["timeout"] = timeout
    coro.close()
    raise asyncio.TimeoutError

  monkeypatch.setattr(gateway_runner, "AgentRunner", _FakeChildRunner)
  monkeypatch.setattr(gateway_runner, "asyncio", SimpleNamespace(wait_for=_wait_for, TimeoutError=asyncio.TimeoutError))
  monkeypatch.setattr(
    gateway_runner,
    "_sub_agent_result_from_log_entries",
    lambda entries, **kwargs: {"entries": [entry.event for entry in entries], "kwargs": kwargs},
  )

  result, error = asyncio.run(
    _parent_runner().spawn_sub_agent(
      "will timeout",
      dispatcher=object(),
      max_turns=1,
      timeout=2.5,
    )
  )

  child = _FakeChildRunner.instances[0]
  assert error is None
  assert waited == {"timeout": 2.5}
  assert child.closed_timeout == 2.0
  assert result == {
    "entries": [{"type": "error", "error": "Sub-agent timed out after 2.5s"}],
    "kwargs": {"timed_out": True, "timeout": 2.5, "budget_exceeded_reason": None},
  }


def test_resume_sub_agent_resolves_parent_module_helpers(monkeypatch: Any) -> None:
  _FakeChildRunner.instances.clear()
  result_calls: list[dict[str, Any]] = []
  task_entry = SimpleNamespace(delivered_messages=set(), message_inbox=object())
  reconstructed_messages = [
    {"role": "assistant", "content": "previous"},
    {"role": "user", "content": "continue"},
  ]

  monkeypatch.setattr(gateway_runner, "AgentRunner", _FakeChildRunner)
  monkeypatch.setattr(
    gateway_runner,
    "_derive_sub_agent_id",
    lambda session_id, call_index: f"resumed-{session_id}-{call_index}",
  )
  monkeypatch.setattr(
    gateway_runner,
    "_sub_agent_result_from_log_entries",
    lambda entries, **kwargs: result_calls.append({"entries": list(entries), "kwargs": kwargs}) or {"ok": kwargs},
  )

  result, error = asyncio.run(
    _parent_runner().resume_sub_agent(
      original_task_id="bg_1",
      reconstructed_messages=reconstructed_messages,
      parent_messages=[SimpleNamespace(message_id="pm-1"), SimpleNamespace(message_id="pm-2")],
      dispatcher=object(),
      max_turns=5,
      timeout=None,
      call_index=8,
      task_entry=task_entry,
    )
  )

  child = _FakeChildRunner.instances[0]
  assert error is None
  assert task_entry.delivered_messages == {"pm-1", "pm-2"}
  assert child.kwargs["session_id"] == "resumed-parent-session-8"
  assert child.kwargs["message_inbox"] is task_entry.message_inbox
  assert child.run_kwargs == {
    "messages": [{"role": "user", "content": "continue"}],
    "system_prompt": None,
    "model_override": None,
    "max_turns": 5,
    "resume_initial_messages": reconstructed_messages,
  }
  assert result == {
    "ok": {
      "timed_out": False,
      "timeout": None,
      "budget_exceeded_reason": None,
      "original_task_id": "bg_1",
    }
  }
  assert [entry.event for entry in result_calls[0]["entries"]] == [{"type": "message", "content": "child result"}]
