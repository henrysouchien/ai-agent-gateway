import asyncio
import hashlib
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
import agent_gateway.runner_tool_execution as runner_tool_execution  # noqa: E402
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


class _UnavailableDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, tool_input, call_index
    return None, {
      "code": "tool_unavailable",
      "message": "Tool 'get_quote' is not currently available. Load market-data first.",
      "data": {
        "deferred_tool": True,
        "required_tool_pack": "market-data",
        "required_tool_packs": ["market-data"],
        "next_tool": "load_tools",
        "suggested_call": {
          "tool": "load_tools",
          "args": {"pack": "market-data"},
        },
      },
    }


class _HintedErrorDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, call_index
    assert tool_name == "get_price_target"
    assert tool_input == {"ticker": "MSCI"}
    return None, {
      "code": "mcp_tool_error",
      "sub_code": "unknown",
      "message": "ticker\n Unexpected keyword argument [type=unexpected_keyword_argument]",
      "tool_usage_hint": "Pass research_file_id; do not pass ticker.",
    }


class _ReadableResourceDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, tool_input, call_index
    content = "## Daily note\n\nCaptured markdown.\n"
    content_bytes = content.encode("utf-8")
    digest = hashlib.sha256(content_bytes).hexdigest()
    return {
      "written": True,
      "file": "daily/2026-06-12.md",
      "mode": "append",
      "indexed_chunks": 1,
      "_readable_resource_snapshot": {
        "contract_name": "MarkdownNote",
        "content_type": "text/markdown",
        "content_class": "human_readable",
        "title": "daily/2026-06-12.md",
        "source_path": "daily/2026-06-12.md",
        "content_snapshot_id": f"sha256:{digest}",
        "content_sha256": digest,
        "content_bytes": len(content_bytes),
        "content": content,
        "truncated": False,
        "byte_start": 128,
        "byte_end": 128 + len(content_bytes),
        "tool_name": "memory_write",
      },
    }, None


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


def test_execute_single_tool_emits_readable_resource_event_without_model_leak() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ReadableResourceDispatcher(),  # type: ignore[arg-type]
    session_id="control-run-readable",
    provider=_Provider(),  # type: ignore[arg-type]
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    skill_run_id="skill-run-readable",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-readable",
      "memory_write",
      {"file": "daily/2026-06-12.md", "content": "redacted by event", "mode": "append"},
      {"tools": []},
    )
  )

  assert tool_name == "memory_write"
  assert extra_blocks == []
  model_payload = json.loads(live_entry["content"])
  assert model_payload == {
    "written": True,
    "file": "daily/2026-06-12.md",
    "mode": "append",
    "indexed_chunks": 1,
  }

  events = [entry.event for entry in runner._log.entries]
  complete = next(event for event in events if event.get("type") == "tool_call_complete")
  assert "_readable_resource_snapshot" not in complete["result"]
  resource = next(event for event in events if event.get("type") == "readable_resource_ready")
  assert resource["resource_id"].startswith("rr:")
  assert resource["control_run_id"] == "control-run-readable"
  assert resource["skill_run_id"] == "skill-run-readable"
  assert resource["source_path"] == "daily/2026-06-12.md"
  assert resource["content"] == "## Daily note\n\nCaptured markdown.\n"
  assert resource["content_sha256"] == hashlib.sha256(resource["content"].encode("utf-8")).hexdigest()


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


def test_execute_single_tool_returns_typed_blocker_for_output_file_gated_tool(monkeypatch: Any) -> None:
  monkeypatch.setattr(
    runner_tool_execution,
    "_output_file_gated_tool_names",
    lambda: frozenset({"analyze_stock"}),
  )
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-output-file-gated-excluded",
    provider=_Provider(),  # type: ignore[arg-type]
    excluded_tools={"analyze_stock"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-1", "analyze_stock", {"ticker": "MSCI"}, {"tools": []})
  )

  assert tool_name == "analyze_stock"
  assert extra_blocks == []
  payload = json.loads(live_entry["content"])
  error = payload["error"]
  assert error["code"] == "tool_excluded"
  assert error["sub_code"] == "output_file_gated_tool_excluded"
  assert "output='file'" in error["message"]
  assert error["data"]["output_file_gated"] is True
  assert error["data"]["suggested_tools"] == ["get_quote", "industry_peer_comparison"]
  assert "quantifying-risk" in error["data"]["resolution"]


def test_execute_single_tool_preserves_dispatcher_error_data_for_model_result() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_UnavailableDispatcher(),  # type: ignore[arg-type]
    session_id="test-unavailable-data",
    provider=_Provider(),  # type: ignore[arg-type]
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-1", "get_quote", {}, {"tools": []})
  )

  assert tool_name == "get_quote"
  assert extra_blocks == []
  error = json.loads(live_entry["content"])["error"]
  assert error["code"] == "tool_unavailable"
  assert error["data"] == {
    "deferred_tool": True,
    "required_tool_pack": "market-data",
    "required_tool_packs": ["market-data"],
    "next_tool": "load_tools",
    "suggested_call": {
      "tool": "load_tools",
      "args": {"pack": "market-data"},
    },
  }

  complete_events = [entry.event for entry in runner._log.entries if entry.event.get("type") == "tool_call_complete"]
  assert complete_events
  assert complete_events[-1]["final_tool_result_blocks"][0]["content"] == live_entry["content"]


def test_execute_single_tool_exposes_top_level_tool_usage_hint_in_model_error_data() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_HintedErrorDispatcher(),  # type: ignore[arg-type]
    session_id="test-error-hint",
    provider=_Provider(),  # type: ignore[arg-type]
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-1", "get_price_target", {"ticker": "MSCI"}, {"tools": []})
  )

  assert tool_name == "get_price_target"
  assert extra_blocks == []
  error = json.loads(live_entry["content"])["error"]
  assert error["code"] == "mcp_tool_error"
  assert error["data"] == {"tool_usage_hint": "Pass research_file_id; do not pass ticker."}

  complete_events = [entry.event for entry in runner._log.entries if entry.event.get("type") == "tool_call_complete"]
  assert complete_events
  assert complete_events[-1]["error"]["tool_usage_hint"] == "Pass research_file_id; do not pass ticker."
  assert complete_events[-1]["error"]["data"] == {"tool_usage_hint": "Pass research_file_id; do not pass ticker."}
  assert complete_events[-1]["final_tool_result_blocks"][0]["content"] == live_entry["content"]


def test_execute_single_tool_stops_after_repeated_generic_excluded_tool() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-repeated-generic-excluded",
    provider=_Provider(),  # type: ignore[arg-type]
    excluded_tools={"apply_patch_ops"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  first_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-1", "apply_patch_ops", {"ops": []}, {"tools": []})
  )
  assert json.loads(first_entry["content"])["error"] == {
    "code": "tool_excluded",
    "message": "Tool 'apply_patch_ops' is not available in this context",
  }
  assert not hasattr(runner, "_stop_after_tool_results_reason")

  second_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-2", "apply_patch_ops", {"ops": []}, {"tools": []})
  )

  error = json.loads(second_entry["content"])["error"]
  assert error["code"] == "tool_excluded"
  assert error["sub_code"] == "repeated_tool_excluded"
  assert error["data"]["blocked_tool"] == "apply_patch_ops"
  assert error["data"]["exclusion_count"] == 2
  assert error["data"]["stop_after_tool_results"] is True
  assert runner._stop_after_tool_results_reason == "repeated_tool_excluded"
  assert runner._stop_after_tool_results_tool_name == "apply_patch_ops"


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
