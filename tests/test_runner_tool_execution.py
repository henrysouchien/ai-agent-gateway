import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_tool_execution import RunnerToolExecutionMixin  # noqa: E402


class _Provider:
  name = "stub"

  def get_model_info(self, model: str) -> SimpleNamespace:
    return SimpleNamespace(model_id=model, context_window=200_000, max_output_tokens=8192)

  def estimate_cost(self, model: str, uncached: int, cache_read: int, cache_write: int, output: int) -> SimpleNamespace:
    _ = model, uncached, cache_read, cache_write, output
    return SimpleNamespace(total_usd=0.0, breakdown={})


class _Dispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, call_index
    return {"status": "ok", "echo": dict(tool_input)}, None


class _ExplodingDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_input, call_index
    raise AssertionError(f"dispatch should not be called for excluded tool {tool_name}")


def _run(coro):
  return asyncio.run(coro)


def _runner() -> AgentRunner:
  return AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_Dispatcher(),  # type: ignore[arg-type]
    session_id="test-tool-execution",
    provider=_Provider(),  # type: ignore[arg-type]
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_runner_tool_execution_method_is_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerToolExecutionMixin)
  assert gateway_runner.RunnerToolExecutionMixin is RunnerToolExecutionMixin
  assert AgentRunner._execute_single_tool is RunnerToolExecutionMixin._execute_single_tool


def test_execute_single_tool_resolves_parent_module_helpers(monkeypatch: Any) -> None:
  start_calls: list[dict[str, Any]] = []
  complete_calls: list[dict[str, Any]] = []
  semantic_calls: list[Any] = []

  def _redact(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    assert tool_name == "file_read"
    assert tool_input == {"path": "secret.txt"}
    return {"path": "[patched]"}

  def _start_event(**kwargs: Any) -> dict[str, Any]:
    start_calls.append(kwargs)
    return {
      "type": "patched_start",
      "tool_call_id": kwargs["tool_call_id"],
      "tool_name": kwargs["tool_name"],
      "tool_input": kwargs["tool_input"],
    }

  def _complete_event(**kwargs: Any) -> dict[str, Any]:
    complete_calls.append(kwargs)
    return {
      "type": "patched_complete",
      "tool_call_id": kwargs["tool_call_id"],
      "tool_name": kwargs["tool_name"],
      "result": kwargs["result"],
      "error": kwargs["error"],
      "is_error": kwargs["error"] is not None,
    }

  def _semantic(result: Any) -> None:
    semantic_calls.append(result)
    return None

  monkeypatch.setattr(gateway_runner, "_redact_tool_input_for_event", _redact)
  monkeypatch.setattr(gateway_runner, "resolve_display", lambda name, tool_input: {"name": name, "input": tool_input})
  monkeypatch.setattr(gateway_runner, "_build_tool_call_start_event", _start_event)
  monkeypatch.setattr(gateway_runner, "_build_tool_call_complete_event", _complete_event)
  monkeypatch.setattr(gateway_runner, "classify_semantic_tool_error", _semantic)

  runner = _runner()
  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-1",
      "file_read",
      {"path": "secret.txt"},
      {"tools": []},
      call_index=3,
    )
  )

  assert tool_name == "file_read"
  assert extra_blocks == []
  assert json.loads(live_entry["content"]) == {"status": "ok", "echo": {"path": "secret.txt"}}
  assert start_calls[0]["tool_input"] == {"path": "[patched]"}
  assert start_calls[0]["call_index"] == 3
  assert complete_calls[0]["result"] == {"status": "ok", "echo": {"path": "secret.txt"}}
  assert semantic_calls == [{"status": "ok", "echo": {"path": "secret.txt"}}]

  events = [entry.event for entry in runner._log.entries]
  assert events[0]["type"] == "patched_start"
  assert events[0]["display"] == {"name": "file_read", "input": {"path": "[patched]"}}
  assert events[-1]["type"] == "patched_complete"
  assert events[-1]["final_tool_result_blocks"][0]["content"] == live_entry["content"]


def test_execute_single_tool_keeps_generic_excluded_tools_bare() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-generic-excluded",
    provider=_Provider(),  # type: ignore[arg-type]
    excluded_tools={"run_agent"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-1", "run_agent", {"task": "x"}, {"tools": []})
  )

  assert tool_name == "run_agent"
  assert extra_blocks == []
  payload = json.loads(live_entry["content"])
  assert payload["error"] == {
    "code": "tool_excluded",
    "message": "Tool 'run_agent' is not available in this context",
  }


def test_execute_single_tool_returns_typed_blocker_for_excluded_fms_commit_tool() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-fms-commit-excluded",
    provider=_Provider(),  # type: ignore[arg-type]
    excluded_tools={"fms_persist_business_model"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-1",
      "fms_persist_business_model",
      {"judgment": {"ticker": "PCTY"}},
      {"tools": []},
    )
  )

  assert tool_name == "fms_persist_business_model"
  assert extra_blocks == []
  payload = json.loads(live_entry["content"])
  error = payload["error"]
  assert error["code"] == "tool_excluded"
  assert error["sub_code"] == "requires_interactive_approval"
  assert "BUILD_BLOCKED" in error["message"]
  assert error["data"]["recommended_verdict"] == "BUILD_BLOCKED"
  assert error["data"]["pending_action"] == {
    "code": "persist_business_model",
    "stage": "bm",
    "message": "Run fms_persist_business_model interactively with operator approval, then retry the workflow.",
    "severity": "blocking",
    "target": "fms_persist_business_model",
    "source": "runner_tool_exclusion",
    "metadata": {
      "blocked_tool": "fms_persist_business_model",
      "requires_interactive_approval": True,
      "tool_class": "state_write",
      "resolution": (
        "Run this commit tool in an interactive model-writer/thesis-writer "
        "session with operator approval, then retry the blocked workflow."
      ),
    },
  }

  complete_events = [entry.event for entry in runner._log.entries if entry.event.get("type") == "tool_call_complete"]
  assert complete_events
  assert complete_events[0]["error"]["data"] == error["data"]


def test_execute_single_tool_returns_typed_blocker_for_excluded_thesis_writer_tool() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-fms-thesis-excluded",
    provider=_Provider(),  # type: ignore[arg-type]
    excluded_tools={"fms_report_thesis_consultation"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-1", "fms_report_thesis_consultation", {}, {"tools": []})
  )

  error = json.loads(live_entry["content"])["error"]
  assert error["sub_code"] == "requires_interactive_approval"
  assert error["data"]["pending_action"]["code"] == "report_thesis_consultation"
  assert error["data"]["pending_action"]["stage"] == "diligence"


def test_execute_single_tool_keeps_apply_proposal_exclusions_generic() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-apply-excluded",
    provider=_Provider(),  # type: ignore[arg-type]
    excluded_tools={"apply_patch_proposal"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-1", "apply_patch_proposal", {}, {"tools": []})
  )

  assert json.loads(live_entry["content"])["error"] == {
    "code": "tool_excluded",
    "message": "Tool 'apply_patch_proposal' is not available in this context",
  }
