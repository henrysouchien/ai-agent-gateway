import asyncio
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.autonomous_event_channel import snapshot_autonomous_event  # noqa: E402
from agent_gateway.mcp_activation import McpActivationFold  # noqa: E402
from agent_gateway.mcp_client import McpClientManager  # noqa: E402
from agent_gateway.server import ChatRequest  # noqa: E402
from agent_gateway.skill_result_events import extract_fms_results  # noqa: E402
from agent_gateway import AgentRunner, AgentSessionLog, EventLog, ModelInfo, ModelProvider, SessionStore, ToolDispatcher, ToolResultContext  # noqa: E402
from agent_gateway.code_execution import CodeExecutionConfig, DockerBackend, build_code_execution  # noqa: E402
from agent_gateway.providers import StreamEvent  # noqa: E402
from agent_gateway.sub_agent import make_run_agent_handler  # noqa: E402
from agent_gateway.tool_result_compaction import (  # noqa: E402
  annotate_result,
  compact_model_tool_result_entry,
  is_error_tool_result_entry,
  make_error_result,
  project_tool_call_complete_for_stream,
  truncate_model_tool_result_content,
  write_tool_result_spill,
)
from agent_gateway.tool_result_spill import (  # noqa: E402
  SpillCapabilities,
  SpillSink,
  make_tool_result_read_handler,
  read_spill_result,
)
import agent_gateway.tool_result_compaction as tool_result_compaction  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
import agent_gateway.runner_tool_execution as runner_tool_execution  # noqa: E402
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_capability_execution_resolver,
  stub_runner_capability_execution,
)
from gateway_test_support.host_policy import owner_session_host_policy


CAP = 4_000
PAYLOAD_SIZE = 5_200


def _run(coro):
  return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _small_tool_result_cap(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(gateway_runner.MODEL_TOOL_RESULT_MAX_CHARS_ENV, str(CAP))
  monkeypatch.delenv(gateway_runner.SPILL_TRUNCATED_TOOL_RESULTS_ENV, raising=False)


class _NullMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)


class _RecordingProvider(ModelProvider):
  name = "stub"

  def __init__(self, turns: list[list[StreamEvent] | Callable[["_RecordingProvider"], list[StreamEvent]]]) -> None:
    self._turns = list(turns)
    self._stream_index = 0
    self.params_history: list[dict[str, Any]] = []
    self.last_spill_ref: str | None = None
    self.code_read_ok = False

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    params = {
      "model": model,
      "messages": messages,
      "system_prompt": system_prompt,
      "tools": tools,
      "max_tokens": max_tokens,
      **kwargs,
    }
    self.params_history.append(params)
    self._observe_messages(messages)
    return params

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if self._stream_index >= len(self._turns):
      events = _text_turn("")
    else:
      turn = self._turns[self._stream_index]
      self._stream_index += 1
      events = turn(self) if callable(turn) else turn
    for event in events:
      yield event

  def _observe_messages(self, messages: list[dict[str, Any]]) -> None:
    for payload in _tool_result_payloads(messages):
      spill_ref = payload.get("spill_ref")
      if isinstance(spill_ref, str):
        self.last_spill_ref = spill_ref
      if payload.get("stdout") == f"{PAYLOAD_SIZE}\n":
        self.code_read_ok = True


def _tool_result_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  payloads: list[dict[str, Any]] = []
  for message in messages:
    content = message.get("content")
    if not isinstance(content, list):
      continue
    for block in content:
      if not isinstance(block, dict) or block.get("type") != "tool_result":
        continue
      raw_content = block.get("content")
      if not isinstance(raw_content, str):
        continue
      try:
        payload = json.loads(raw_content)
      except Exception:
        continue
      if isinstance(payload, dict):
        payloads.append(payload)
  return payloads


def _model_bound_tool_result_blocks(provider: _RecordingProvider) -> list[dict[str, Any]]:
  for params in reversed(provider.params_history):
    messages = params["messages"]
    if not messages:
      continue
    content = messages[-1].get("content")
    if isinstance(content, list) and any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content):
      return content
  raise AssertionError("No model-bound tool result message captured")


def _tool_use_turn(tool_id: str, tool_name: str, tool_input: dict[str, Any] | None = None) -> list[StreamEvent]:
  payload = dict(tool_input or {})
  return [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name=tool_name,
      tool_input=payload,
      raw_block={"type": "tool_use", "id": tool_id, "name": tool_name, "input": payload},
    ),
    StreamEvent(type="message_end", stop_reason="tool_use"),
  ]


def _text_turn(text: str) -> list[StreamEvent]:
  return [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]


def _tool_def(name: str) -> dict[str, Any]:
  return {"name": name, "description": "", "input_schema": {"type": "object", "properties": {}}}


def _dispatcher(
  event_log: EventLog,
  handlers: dict[str, Any],
  *,
  approval_key_qualifier: Callable[[str, dict[str, Any]], str] | None = None,
) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers=handlers,
    event_log=event_log,
    session_id="sess-spill",
    role="owner",
    approval_key_qualifier=approval_key_qualifier,
  )


def _runner(spill_provider: Callable[[], str] | None) -> AgentRunner:
  event_log = EventLog()
  spill_sink = (
    SpillSink(
      root_provider=spill_provider,
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    )
    if spill_provider is not None
    else None
  )
  return AgentRunner(
    event_log=event_log,
    dispatcher=_dispatcher(event_log, {}),
    session_id="sess-spill",
    capability_execution=stub_runner_capability_execution(
      provider=_RecordingProvider([]),
      auth_config={"api_key": "k"},
      model="stub-model",
      effort="none",
    ),
    user_id="alice",
    request_id="req-spill",
    billing_mode="byok",
    rate_table_version="unknown",
    code_execution_spill_dir_provider=spill_sink,
  )


def _large_content(payload_size: int = PAYLOAD_SIZE) -> str:
  return json.dumps({"status": "success", "payload": "x" * payload_size}, default=str)


class _RecordingLogger:
  def __init__(self) -> None:
    self.infos: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    self.warnings: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

  def info(self, message: str, *args: Any, **kwargs: Any) -> None:
    self.infos.append((message, args, kwargs))

  def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
    self.warnings.append((message, args, kwargs))


def test_runner_owns_no_canvas_source_cap_vocabulary() -> None:
  # The pre-dispatch `invalid_input` cap for emit_canvas_artifact was a second
  # rejection vocabulary competing with the pipeline's own `size_cap` stage. The
  # pipeline owns it alone now: it reports `validation_failed{stage: "size_cap",
  # code: "source_size_cap_exceeded"}` on every lane before any node build
  # (certified in tests/test_canvas_emit_sources_contract.py and by the
  # oversize-source fixture in tests/test_canvas_artifact_pipeline.py), so the
  # runner must carry no canvas-specific cap of its own.
  source = Path(runner_tool_execution.__file__).read_text(encoding="utf-8")

  assert "emit_canvas_artifact" not in source


def test_emit_dashboard_artifact_oversize_guard_returns_error_before_dispatch() -> None:
  runner = _runner(None)
  payload = {"blob": "x" * (256 * 1024)}

  result, tool_name, live_events = _run(
    runner._execute_single_tool(
      "tool-dashboard",
      "emit_dashboard_artifact",
      {"payload": payload},
      {},
    )
  )

  assert tool_name == "emit_dashboard_artifact"
  assert live_events == []
  assert result["is_error"] is True
  content = json.loads(result["content"])
  assert content["error"]["code"] == "invalid_input"
  assert "exceeds 256KB limit" in content["error"]["message"]
  assert [entry.event for entry in runner._log.entries] == []


async def _dispatch_bundle_tool(bundle: Any, tool_name: str, tool_input: dict[str, Any]):
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers=bundle.handlers,
    event_log=EventLog(),
    role="owner",
    approval_key_qualifier=bundle.approval_qualifier,
  )
  return await dispatcher.dispatch(f"{tool_name}_call", tool_name, tool_input)


def test_truncate_keeps_typed_view_summary_behind_a_bulk_collection() -> None:
  """A typed view's headline facts survive even when they serialize after the rows."""

  view = {
    "schema_id": "holdings-view",
    "schema_version": "v2",
    "view_data": {
      "portfolio_name": "CURRENT_PORTFOLIO",
      "total_positions": 32,
      "holdings": [
        {"ticker": f"TICK{index:03d}", "notes": "n" * 9_000}
        for index in range(32)
      ],
      "summary": {"total_value": {"value": "395061.44", "display": "$395,061.44"}},
      "quality": {"pricing": {"availability": "partial", "priced": 30, "unpriced": 2}},
      "sources": [{"source": "positions", "availability": "available"}],
    },
  }
  content = json.dumps(view)
  assert len(content) > CAP

  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="get_holdings_view",
    max_chars=CAP,
  )

  assert was_truncated is True
  assert len(truncated) <= CAP
  projection = json.loads(truncated)["content_projection"]
  view_data = projection["view_data"]
  assert view_data["summary"] == view["view_data"]["summary"]
  assert view_data["quality"] == view["view_data"]["quality"]
  assert view_data["sources"] == view["view_data"]["sources"]
  assert view_data["total_positions"] == 32
  kept = [row for row in view_data["holdings"] if "_elided_items" not in row]
  assert len(kept) < 32
  assert view_data["holdings"][-1] == {"_elided_items": 32 - len(kept)}


def test_truncate_fills_the_budget_for_a_result_that_is_one_large_string() -> None:
  content = "report line. " * 20_000

  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="file_read",
    max_chars=CAP,
  )

  assert was_truncated is True
  assert len(truncated) <= CAP
  projection = json.loads(truncated)["content_projection"]
  assert projection.startswith("report line. ")
  assert "...<elided chars=" in projection
  assert len(projection) > CAP // 2


def test_truncate_tool_result_embeds_spill_pointer_only_when_provided() -> None:
  content = _large_content()

  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="lookup",
    max_chars=CAP,
    spill_ref="spill:v1:lookup_tool-1.json:" + "a" * 64,
  )

  assert was_truncated is True
  assert len(truncated) <= CAP
  payload = json.loads(truncated)
  assert payload["spill_ref"] == "spill:v1:lookup_tool-1.json:" + "a" * 64
  assert "spill_file" not in payload
  assert "spill_abspath" not in payload
  assert "spill_hint" not in payload

  plain, plain_was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="lookup",
    max_chars=CAP,
  )

  assert plain_was_truncated is True
  plain_payload = json.loads(plain)
  assert "spill_ref" not in plain_payload


def test_runner_preserves_tool_result_utility_delegates(tmp_path: Path) -> None:
  assert gateway_runner.AgentRunner._annotate_result({"ok": True}) == annotate_result({"ok": True})
  assert gateway_runner.AgentRunner._make_error_result("tool-1", "bad", "failed") == make_error_result(
    "tool-1",
    "bad",
    "failed",
  )
  assert gateway_runner.AgentRunner._is_error_tool_result_entry({"type": "tool_result"}, '{"error": "bad"}')
  filename, spill_abspath = gateway_runner.AgentRunner._write_tool_result_spill(
    work_dir=str(tmp_path),
    tool_name="lookup",
    tool_use_id="tool-1",
    content='{"ok": true}',
  )
  assert filename == "lookup_tool-1.json"
  assert json.loads(Path(spill_abspath).read_text(encoding="utf-8")) == {"ok": True}


def test_annotate_result_collects_policy_low_match_and_subagent_warnings() -> None:
  result = {
    "ok": True,
    "_interceptor_warnings": ["policy"],
    "low_match_warning": "2/10",
    "warning": "partial",
  }

  annotated = annotate_result(result, tool_name="run_agent")

  assert annotated["_runner_warning"] == (
    "Policy warning: policy | Low match rate detected: 2/10 | Sub-agent warning: partial"
  )
  assert annotated["_runner_warning_detail"] == "2/10"
  assert "_interceptor_warnings" not in result


def test_annotate_result_warns_on_empty_status_error_detail() -> None:
  result = {"status": "error", "error": ""}

  annotated = annotate_result(result, tool_name="get_skill_artifact")

  assert annotated["_runner_warning"] == (
    "Tool get_skill_artifact returned status=error without error detail; "
    "do not retry unchanged input unless required context changed or there is new evidence the failure was transient."
  )
  assert "_runner_warning" not in result


def test_annotate_result_keeps_detailed_status_error_unchanged() -> None:
  result = {"status": "error", "error": {"code": "not_found", "message": "missing"}}

  annotated = annotate_result(result, tool_name="get_skill_artifact")

  assert annotated is result


def test_make_error_result_includes_optional_sub_code() -> None:
  result = make_error_result("tool-1", "invalid_input", "bad payload", sub_code="too_large")

  payload = json.loads(result["content"])
  assert result["is_error"] is True
  assert result["tool_use_id"] == "tool-1"
  assert payload["error"] == {
    "code": "invalid_input",
    "message": "bad payload",
    "sub_code": "too_large",
  }


def test_make_error_result_includes_optional_data() -> None:
  result = make_error_result(
    "tool-1",
    "tool_excluded",
    "requires approval",
    sub_code="requires_interactive_approval",
    data={"recommended_verdict": "BUILD_BLOCKED"},
  )

  payload = json.loads(result["content"])
  assert payload["error"] == {
    "code": "tool_excluded",
    "message": "requires approval",
    "sub_code": "requires_interactive_approval",
    "data": {"recommended_verdict": "BUILD_BLOCKED"},
  }


def test_is_error_tool_result_entry_treats_only_truthy_payload_error_as_error() -> None:
  assert is_error_tool_result_entry({"is_error": True}, '{"ok": true}') is True
  assert is_error_tool_result_entry({"error": {"code": "bad"}}, '{"ok": true}') is True
  assert is_error_tool_result_entry({"type": "tool_result"}, '{"error": "bad"}') is True
  assert is_error_tool_result_entry({"type": "tool_result"}, '{"error": null, "rows": []}') is False
  assert is_error_tool_result_entry({"type": "tool_result"}, "not-json") is False


def test_write_tool_result_spill_direct_helper_uses_uuid_factory(tmp_path: Path) -> None:
  first = SimpleNamespace(hex="a" * 32)
  second = SimpleNamespace(hex="bcdef1234567890")
  calls = iter([first, second])
  (tmp_path / f"lookup_{'a' * 32}.txt").write_text("old", encoding="utf-8")

  filename, spill_abspath = write_tool_result_spill(
    work_dir=str(tmp_path),
    tool_name="lookup",
    tool_use_id=None,
    content="plain text",
    uuid_factory=lambda: next(calls),
  )

  assert filename == f"lookup_{'a' * 32}_bcdef123.txt"
  assert spill_abspath == str(tmp_path / filename)
  assert (tmp_path / filename).read_text(encoding="utf-8") == "plain text"


def test_write_tool_result_spill_default_uuid_is_resolved_at_call_time(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(tool_result_compaction.uuid, "uuid4", lambda: SimpleNamespace(hex="d" * 32))

  filename, _spill_abspath = write_tool_result_spill(
    work_dir=str(tmp_path),
    tool_name="lookup",
    tool_use_id=None,
    content='{"ok": true}',
  )

  assert filename == f"lookup_{'d' * 32}.json"


def test_compact_model_tool_result_entry_helper_spills_live_entry_and_logs(tmp_path: Path) -> None:
  content = _large_content()
  result_entry = {"type": "tool_result", "tool_use_id": "tool-1", "content": content}
  logger = _RecordingLogger()

  live_entry, durable_entry = compact_model_tool_result_entry(
    result_entry,
    tool_name="lookup",
    spill_sink=SpillSink(
      root_provider=lambda: str(tmp_path),
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    ),
    log_session_id="sess-direct",
    logger=logger,
    uuid_factory=lambda: SimpleNamespace(hex="e" * 32),
  )

  spill_files = list(tmp_path.iterdir())
  assert len(spill_files) == 1
  assert json.loads(spill_files[0].read_text(encoding="utf-8")) == json.loads(content)
  live_payload = json.loads(live_entry["content"])
  durable_payload = json.loads(durable_entry["content"])
  assert live_payload["spill_ref"].startswith(
    f"spill:v1:{spill_files[0].name}:"
  )
  assert "spill_file" not in live_payload
  assert "spill_abspath" not in live_payload
  assert "spill_ref" not in durable_payload
  assert logger.warnings == []
  assert logger.infos[0][2]["extra"]["data"]["event"] == "tool_result_compacted"
  assert logger.infos[0][2]["extra"]["data"]["session_id"] == "sess-direct"


def test_business_model_terminal_success_uses_bounded_semantic_projection(
  tmp_path: Path,
) -> None:
  verdict = {
    "skill": "business-model-construction",
    "verdict": "BM_CONSTRUCTED",
    "confidence": "MEDIUM",
    "revision": "pcty-business-model-rev-1",
    "validation": {"large": "v" * PAYLOAD_SIZE},
    "data_gaps": [
      {
        "key": "float_yield",
        "text": "Average daily float yield is not disclosed.",
        "claim_keys": ["interest_income_fy26"],
      }
    ],
    "recommended_next_action": "Run forecast-assumptions.",
  }
  content = json.dumps(
    {
      "status": "staged",
      "gate_code": "PROCEED",
      "artifact_ref": "artifacts/PCTY/business-model.json",
      "proposal_id": "proposal-1",
      "error": None,
      "verdict": verdict,
      "verdict_echo": verdict,
      "readback": {
        "typed_outputs": {
          "business_model_stage_receipt": {
            "status": "accepted",
            "stage_metadata": {"evidence_snapshot": "x" * PAYLOAD_SIZE},
          },
          "business_model": {"segments": ["x" * PAYLOAD_SIZE]},
        }
      },
    }
  )
  result_entry = {
    "type": "tool_result",
    "tool_use_id": "bm-tool-1",
    "content": content,
  }
  logger = _RecordingLogger()

  live_entry, durable_entry = compact_model_tool_result_entry(
    result_entry,
    tool_name="fms_persist_business_model",
    spill_dir_provider=lambda: str(tmp_path),
    log_session_id="sess-bm",
    logger=logger,
  )

  assert result_entry["content"] == content
  assert live_entry == durable_entry
  projection = json.loads(live_entry["content"])
  assert projection == {
    "status": "staged",
    "gate_code": "PROCEED",
    "artifact_ref": "artifacts/PCTY/business-model.json",
    "proposal_id": "proposal-1",
    "verdict": "BM_CONSTRUCTED",
    "confidence": "MEDIUM",
    "revision": "pcty-business-model-rev-1",
    "stage_receipt_status": "accepted",
    "data_gaps": [
      {
        "key": "float_yield",
        "text": "Average daily float yield is not disclosed.",
        "claim_keys": ["interest_income_fy26"],
      }
    ],
    "recommended_next_action": "Run forecast-assumptions.",
  }
  assert "readback" not in projection
  assert "validation" not in projection
  assert len(live_entry["content"]) < CAP
  assert list(tmp_path.iterdir()) == []
  assert logger.infos[0][2]["extra"]["data"] == {
    "event": "tool_result_semantically_compacted",
    "session_id": "sess-bm",
    "tool": "fms_persist_business_model",
    "original_chars": len(content),
    "compacted_chars": len(live_entry["content"]),
  }


def test_compact_spills_live_entry_and_keeps_durable_pointer_free(tmp_path: Path) -> None:
  content = _large_content()
  runner = _runner(lambda: str(tmp_path))
  result_entry = {"type": "tool_result", "tool_use_id": "tool-1", "content": content}

  live_entry, durable_entry = runner._compact_model_tool_result_entry(result_entry, tool_name="lookup")

  spill_files = list(tmp_path.iterdir())
  assert len(spill_files) == 1
  assert json.loads(spill_files[0].read_text(encoding="utf-8")) == json.loads(content)
  live_payload = json.loads(live_entry["content"])
  durable_payload = json.loads(durable_entry["content"])
  assert live_payload["spill_ref"].startswith(
    f"spill:v1:{spill_files[0].name}:"
  )
  assert "spill_file" not in durable_payload
  assert "spill_abspath" not in durable_payload
  assert "spill_hint" not in durable_payload
  assert "spill_ref" not in durable_payload


def test_compact_does_not_spill_untruncated_error_missing_provider_or_disabled(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  small_entry = {"type": "tool_result", "tool_use_id": "small", "content": json.dumps({"ok": True})}
  live_entry, durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    small_entry,
    tool_name="lookup",
  )
  assert live_entry is small_entry
  assert durable_entry is small_entry
  assert list(tmp_path.iterdir()) == []

  content = _large_content()
  error_entry = {"type": "tool_result", "tool_use_id": "err", "content": content, "is_error": True}
  live_entry, durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    error_entry,
    tool_name="lookup",
  )
  error_payload = json.loads(live_entry["content"])
  assert "spill_file" not in error_payload
  assert "spill_ref" not in error_payload
  assert "spill_summary" not in error_payload
  assert live_entry == durable_entry
  assert list(tmp_path.iterdir()) == []

  no_provider_entry = {"type": "tool_result", "tool_use_id": "no-provider", "content": content}
  live_entry, durable_entry = _runner(None)._compact_model_tool_result_entry(
    no_provider_entry,
    tool_name="lookup",
  )
  no_provider_payload = json.loads(live_entry["content"])
  assert "spill_file" not in no_provider_payload
  assert "spill_ref" not in no_provider_payload
  assert "spill_summary" not in no_provider_payload
  assert live_entry == durable_entry

  monkeypatch.setenv(gateway_runner.SPILL_TRUNCATED_TOOL_RESULTS_ENV, "no")
  disabled_entry = {"type": "tool_result", "tool_use_id": "disabled", "content": content}
  live_entry, durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    disabled_entry,
    tool_name="lookup",
  )
  disabled_payload = json.loads(live_entry["content"])
  assert "spill_file" not in disabled_payload
  assert "spill_ref" not in disabled_payload
  assert "spill_summary" not in disabled_payload
  assert live_entry == durable_entry
  assert list(tmp_path.iterdir()) == []


def test_compact_spills_payload_with_falsy_error_field_but_not_truthy(tmp_path: Path) -> None:
  # A successful data payload that merely carries a falsy top-level "error"
  # (e.g. {"error": null, ...}) must still spill; only a truthy "error" marks a
  # genuine error result. Guards the _is_error_tool_result_entry tightening.
  pad = "x" * PAYLOAD_SIZE
  null_error = json.dumps({"error": None, "data": pad}, default=str)
  ok_entry = {"type": "tool_result", "tool_use_id": "ok-null-error", "content": null_error}
  live_entry, durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    ok_entry,
    tool_name="lookup",
  )
  spill_files = list(tmp_path.iterdir())
  assert len(spill_files) == 1
  assert json.loads(spill_files[0].read_text(encoding="utf-8")) == json.loads(null_error)
  assert json.loads(live_entry["content"])["spill_ref"].startswith(
    f"spill:v1:{spill_files[0].name}:"
  )
  assert "spill_ref" not in json.loads(durable_entry["content"])

  real_error = json.dumps({"error": {"code": "bad", "message": "x" * PAYLOAD_SIZE}}, default=str)
  err_entry = {"type": "tool_result", "tool_use_id": "real-error", "content": real_error}
  live_entry, durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    err_entry,
    tool_name="lookup",
  )
  assert "spill_ref" not in json.loads(live_entry["content"])
  assert live_entry == durable_entry
  assert len(list(tmp_path.iterdir())) == 1  # no new spill file from the error case


def test_compact_provider_failure_falls_back_without_exception(tmp_path: Path) -> None:
  def _raise_provider() -> str:
    raise OSError("no work dir")

  content = _large_content()
  result_entry = {"type": "tool_result", "tool_use_id": "tool-1", "content": content}

  live_entry, durable_entry = _runner(_raise_provider)._compact_model_tool_result_entry(
    result_entry,
    tool_name="lookup",
  )

  live_payload = json.loads(live_entry["content"])
  assert live_payload["_runner_truncated"] is True
  assert "spill_ref" not in live_payload
  assert live_entry == durable_entry
  assert list(tmp_path.iterdir()) == []


def test_spill_filename_is_sanitized_and_stays_inside_work_dir(tmp_path: Path) -> None:
  content = _large_content()
  result_entry = {"type": "tool_result", "tool_use_id": "../unsafe/id:1", "content": content}

  live_entry, _durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    result_entry,
    tool_name="bad/tool",
  )

  live_payload = json.loads(live_entry["content"])
  spill_ref = live_payload["spill_ref"]
  filename = spill_ref.split(":", 3)[2]
  spill_path = tmp_path / filename
  assert "/" not in filename
  assert ":" not in filename
  assert spill_path.name == filename
  assert spill_path.resolve().parent == tmp_path.resolve()
  assert json.loads(spill_path.read_text(encoding="utf-8")) == json.loads(content)


def test_missing_tool_use_id_uses_uuid_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(gateway_runner.uuid, "uuid4", lambda: SimpleNamespace(hex="f" * 32))
  result_entry = {"type": "tool_result", "content": _large_content()}

  live_entry, _durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    result_entry,
    tool_name="lookup",
  )

  filename = json.loads(live_entry["content"])["spill_ref"].split(":", 3)[2]
  assert filename == f"lookup_{'f' * 32}.json"
  assert (tmp_path / filename).exists()


def test_existing_spill_file_retries_with_uuid_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  (tmp_path / "lookup_tool-1.json").write_text("old", encoding="utf-8")
  monkeypatch.setattr(gateway_runner.uuid, "uuid4", lambda: SimpleNamespace(hex="1234567890abcdef"))
  content = _large_content()
  result_entry = {"type": "tool_result", "tool_use_id": "tool-1", "content": content}

  live_entry, _durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    result_entry,
    tool_name="lookup",
  )

  filename = json.loads(live_entry["content"])["spill_ref"].split(":", 3)[2]
  assert filename == "lookup_tool-1_12345678.json"
  assert (tmp_path / "lookup_tool-1.json").read_text(encoding="utf-8") == "old"
  assert json.loads((tmp_path / filename).read_text(encoding="utf-8")) == json.loads(content)


def test_code_execution_ensure_work_dir_is_idempotent_and_concurrency_safe(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[Path] = []
  call_lock = threading.Lock()

  def _slow_mkdtemp(prefix: str = "", dir: str | None = None, suffix: str | None = None) -> str:
    _ = suffix
    time.sleep(0.05)
    with call_lock:
      path = Path(dir or str(tmp_path)) / f"{prefix}{len(calls)}"
      calls.append(path)
    path.mkdir()
    return str(path)

  monkeypatch.setattr("agent_gateway.code_execution._handlers.tempfile.mkdtemp", _slow_mkdtemp)
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
  bundle = build_code_execution(
    session,
    config=CodeExecutionConfig(register_docker=False, work_dir_root=str(tmp_path)),
  )

  with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(lambda _idx: bundle.ensure_work_dir(), range(2)))

  assert results[0] == results[1]
  assert bundle.ensure_work_dir() == results[0]
  assert session.code_execution_work_dir == results[0]
  assert len(calls) == 1
  assert Path(results[0]).exists()


def test_runner_spills_large_tool_result_and_exact_reader_is_retry_safe(tmp_path: Path) -> None:
  async def _run_test() -> None:
    payload = "x" * PAYLOAD_SIZE

    async def _big_data(_tool_input: dict[str, Any], **kwargs: Any):
      _ = kwargs
      return {"status": "success", "payload": payload}, None

    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(work_dir_root=str(tmp_path)),
    )
    spill_sink = SpillSink(
      root_provider=bundle.ensure_work_dir,
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    )
    local_handlers = dict(bundle.handlers)
    local_handlers["big_data"] = _big_data
    local_handlers["tool_result_read"] = make_tool_result_read_handler(
      lambda: spill_sink
    )
    event_log = EventLog()
    provider = _RecordingProvider([
      _tool_use_turn("tool-1", "big_data"),
      _text_turn("done"),
    ])
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=_dispatcher(event_log, local_handlers, approval_key_qualifier=bundle.approval_qualifier),
      session_id="sess-spill",
      capability_execution=stub_runner_capability_execution(
        provider=provider,
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      get_tool_definitions=lambda: [
        _tool_def("big_data"),
        _tool_def("tool_result_read"),
        *bundle.tool_definitions,
      ],
      user_id="alice",
      request_id="req-spill",
      billing_mode="byok",
      rate_table_version="unknown",
      code_execution_spill_dir_provider=spill_sink,
    )

    await runner.run(messages=[{"role": "user", "content": "load"}], system_prompt="x", max_turns=2)

    expected_content = json.dumps({"status": "success", "payload": payload}, default=str)
    work_dir = Path(session.code_execution_work_dir or "")
    spill_files = [path for path in work_dir.iterdir() if path.name.startswith("big_data_tool-1")]
    assert len(spill_files) == 1
    assert json.loads(spill_files[0].read_text(encoding="utf-8")) == json.loads(expected_content)

    live_blocks = _model_bound_tool_result_blocks(provider)
    live_payload = json.loads(live_blocks[0]["content"])
    spill_ref = live_payload["spill_ref"]
    assert spill_ref.startswith(f"spill:v1:{spill_files[0].name}:")
    assert "spill_file" not in live_payload
    assert "spill_abspath" not in live_payload

    complete_event = next(entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete")
    durable_payload = json.loads(complete_event["final_tool_result_blocks"][0]["content"])
    assert "spill_file" not in durable_payload
    assert "spill_abspath" not in durable_payload
    assert "spill_hint" not in durable_payload
    assert "spill_ref" not in durable_payload
    assert "spill_summary" not in durable_payload

    first_read = read_spill_result(spill_sink, spill_ref=spill_ref)
    second_read = read_spill_result(spill_sink, spill_ref=spill_ref)
    assert first_read == second_read
    assert json.loads(first_read["content"]) == json.loads(expected_content)

    code = (
      "import json\n"
      f"data = json.load(open({spill_files[0].name!r}))\n"
      "print(len(data['payload']))\n"
    )
    result, error = await _dispatch_bundle_tool(
      bundle,
      "code_execute",
      {"host": "subprocess", "code": code},
    )
    assert error is None
    assert result is not None
    assert result["stdout"] == f"{PAYLOAD_SIZE}\n"

    if DockerBackend().available():
      result, error = await _dispatch_bundle_tool(
        bundle,
        "code_execute",
        {"host": "docker", "code": code},
      )
      assert error is None
      assert result is not None
      assert result["stdout"] == f"{PAYLOAD_SIZE}\n"

  _run(_run_test())


def test_streamed_tool_call_complete_fits_the_autonomous_event_channel(tmp_path: Path) -> None:
  # Local run bg_353 (2026-09-17): a 2.49 MB child `get_financials` result rode
  # `tool_call_complete.result` onto the autonomous event channel, tripped its
  # per-event bound, and killed a run that had already produced its deliverable.
  async def _run_test() -> None:
    payload = "x" * (2 * 1024 * 1024)

    async def _big_data(_tool_input: dict[str, Any], **kwargs: Any):
      _ = kwargs
      return {"status": "success", "payload": payload}, None

    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(work_dir_root=str(tmp_path)),
    )
    spill_sink = SpillSink(
      root_provider=bundle.ensure_work_dir,
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    )
    local_handlers = dict(bundle.handlers)
    local_handlers["big_data"] = _big_data
    event_log = EventLog()
    provider = _RecordingProvider([
      _tool_use_turn("tool-1", "big_data"),
      _text_turn("done"),
    ])
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=_dispatcher(event_log, local_handlers, approval_key_qualifier=bundle.approval_qualifier),
      session_id="sess-spill",
      capability_execution=stub_runner_capability_execution(
        provider=provider,
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      get_tool_definitions=lambda: [_tool_def("big_data"), *bundle.tool_definitions],
      user_id="alice",
      request_id="req-spill",
      billing_mode="byok",
      rate_table_version="unknown",
      code_execution_spill_dir_provider=spill_sink,
    )
    durable_events: list[dict[str, Any]] = []
    append_durable_event = runner._append_durable_event

    async def _capture_durable_event(event: dict[str, Any]) -> Any:
      durable_events.append(event)
      return await append_durable_event(event)

    runner._append_durable_event = _capture_durable_event

    await runner.run(messages=[{"role": "user", "content": "load"}], system_prompt="x", max_turns=2)

    streamed_event = next(
      entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"
    )
    # The bounded transport must accept the streamed projection unchanged.
    snapshot_autonomous_event(streamed_event)
    streamed_result = streamed_event["result"]
    assert streamed_result["spill_ref"].startswith("spill:v1:")
    assert payload not in json.dumps(streamed_event, default=str)

    durable_event = next(
      event for event in durable_events if event.get("type") == "tool_call_complete"
    )
    assert durable_event["result"] == {"status": "success", "payload": payload}

  _run(_run_test())


def test_streamed_tool_call_complete_keeps_the_fms_door_envelope(tmp_path: Path) -> None:
  # A successful `fms_persist_business_model` result is swapped for its
  # model-facing receipt at any size. The live event log is what skill capture,
  # sub-agent result delivery and the `fms_*` door extractors read, so a result
  # the channel frame can carry must still stream as the whole envelope.
  envelope = {
    "status": "success",
    "subcommand": "persist_business_model",
    "mutation_mode": "commit",
    "proposal_id": "prop-1",
    "artifact_ref": "fms://business-model/1",
    "verdict": {"verdict": "accepted", "confidence": "high", "revision": "r3"},
    "verdict_echo": {"verdict": "accepted"},
    "readback": {"typed_outputs": {"business_model_stage_receipt": {"status": "ok"}}},
  }

  async def _run_test() -> None:
    async def _persist(_tool_input: dict[str, Any], **kwargs: Any):
      _ = kwargs
      return dict(envelope), None

    event_log = EventLog()
    provider = _RecordingProvider([
      _tool_use_turn("tool-1", "fms_persist_business_model"),
      _text_turn("done"),
    ])
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=_dispatcher(event_log, {"fms_persist_business_model": _persist}),
      session_id="sess-spill",
      capability_execution=stub_runner_capability_execution(
        provider=provider,
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      get_tool_definitions=lambda: [_tool_def("fms_persist_business_model")],
      user_id="alice",
      request_id="req-spill",
      billing_mode="byok",
      rate_table_version="unknown",
      code_execution_spill_dir_provider=SpillSink(
        root_provider=lambda: str(tmp_path),
        capabilities=SpillCapabilities(code_execute=True, spill_read=True),
      ),
    )

    await runner.run(messages=[{"role": "user", "content": "persist"}], system_prompt="x", max_turns=2)

    captured = extract_fms_results(event_log.entries)
    assert [result["subcommand"] for result in captured] == ["persist_business_model"]
    assert captured[0]["mutation_mode"] == "commit"
    assert captured[0]["verdict_echo"] == {"verdict": "accepted"}

    streamed_event = next(
      entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"
    )
    # The model still sees only the receipt: persisted bulk stays out of context.
    model_block = json.loads(streamed_event["final_tool_result_blocks"][0]["content"])
    assert model_block["stage_receipt_status"] == "ok"
    assert "readback" not in model_block

  _run(_run_test())


def test_streamed_door_result_over_the_frame_keeps_its_staged_envelope(tmp_path: Path) -> None:
  # McpStdio :8211 (2026-09-23): a 2.67 M-char staged `fms_propose_competitive_position`
  # result streamed as the model's compacted copy, whose envelope sat under
  # `content_projection`; the door extractors skipped it and the pipeline gated
  # competitive-position on the refusal the model had already repaired.
  staged_proposal = {"changes": [{"path": f"drivers.{index}", "value": "z" * 4096} for index in range(640)]}
  envelope = {
    "status": "staged",
    "subcommand": "propose_competitive_position",
    "mutation_mode": "preview",
    "gate_code": "PROCEED",
    "artifact_ref": "artifacts/PCTY/competitive-position/proposal.json",
    "staged_proposal": staged_proposal,
  }

  async def _run_test() -> None:
    async def _propose(_tool_input: dict[str, Any], **kwargs: Any):
      _ = kwargs
      return json.loads(json.dumps(envelope)), None

    event_log = EventLog()
    provider = _RecordingProvider([
      _tool_use_turn("tool-1", "fms_propose_competitive_position"),
      _text_turn("done"),
    ])
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=_dispatcher(event_log, {"fms_propose_competitive_position": _propose}),
      session_id="sess-spill",
      capability_execution=stub_runner_capability_execution(
        provider=provider,
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      get_tool_definitions=lambda: [_tool_def("fms_propose_competitive_position")],
      user_id="alice",
      request_id="req-spill",
      billing_mode="byok",
      rate_table_version="unknown",
      code_execution_spill_dir_provider=SpillSink(
        root_provider=lambda: str(tmp_path),
        capabilities=SpillCapabilities(code_execute=True, spill_read=True),
      ),
    )

    await runner.run(messages=[{"role": "user", "content": "propose"}], system_prompt="x", max_turns=2)

    streamed_event = next(
      entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"
    )
    assert len(json.dumps(envelope)) > 2 * 1024 * 1024
    # The bounded transport must accept the streamed projection unchanged.
    snapshot_autonomous_event(streamed_event)
    # The truncation is typed on the result the readers see.
    assert streamed_event["result"]["_runner_truncated"] is True
    assert streamed_event["result"]["spill_ref"].startswith("spill:v1:")

    captured = extract_fms_results(event_log.entries)
    assert [
      (result["status"], result["subcommand"], result["mutation_mode"], result["gate_code"])
      for result in captured
    ] == [("staged", "propose_competitive_position", "preview", "PROCEED")]
    assert captured[0]["artifact_ref"] == envelope["artifact_ref"]

  _run(_run_test())


def test_streamed_projection_of_a_wide_nested_result_fits_the_frame() -> None:
  # Every full-depth rung keeps 16 keys per level, so five levels of 20-key
  # dicts still project to ~11.7 M chars, and 400 wide top-level strings of
  # four-byte characters outgrow a rung sized in characters; the stream must
  # end on a rung the frame carries, with the envelope's top level intact.
  def _nested(depth: int) -> Any:
    return 1 if depth == 0 else {f"k{index}": _nested(depth - 1) for index in range(20)}

  result = {"status": "staged", "subcommand": "propose_x", "mutation_mode": "preview", "tree": _nested(5)}
  result.update({f"f{index}": "\N{GRINNING FACE}" * 3000 for index in range(400)})
  content = json.dumps(result)
  result_entry = {"type": "tool_result", "tool_use_id": "tool-1", "content": content}
  live_content, _ = truncate_model_tool_result_content(content, tool_name="fms_propose_x", max_chars=60_000)
  event = {"type": "tool_call_complete", "tool_name": "fms_propose_x", "result": result}

  streamed = project_tool_call_complete_for_stream(
    event,
    result_entry=result_entry,
    live_entry=dict(result_entry, content=live_content),
  )

  snapshot_autonomous_event(streamed)
  assert extract_fms_results([streamed])[0]["status"] == "staged"


def test_streamed_code_execute_event_projects_when_its_images_outgrow_the_frame(
  tmp_path: Path,
) -> None:
  # `strip_code_execute_base64_hook` rewrites the model-facing content only:
  # every `data_base64` becomes an `[image: …]` placeholder there while
  # `event["result"]` keeps the base64. A default-settings `code_execute` result
  # — 100 KB of stdout plus 5 plots of 500 KB — is therefore ~100 K model chars
  # and ~2.5 MB on the wire, so a size decision taken on the model content
  # streams an event the 2 MiB channel bound refuses.
  stdout = "y" * 100_000
  images = [
    {
      "filename": f"plot-{index}.png",
      "media_type": "image/png",
      "data_base64": "A" * (500 * 1024),
    }
    for index in range(5)
  ]

  async def _run_test() -> None:
    async def _code_execute(_tool_input: dict[str, Any], **kwargs: Any):
      _ = kwargs
      return {
        "status": "success",
        "stdout": stdout,
        "stderr": "",
        "images": [dict(image) for image in images],
      }, None

    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(work_dir_root=str(tmp_path)),
    )
    local_handlers = dict(bundle.handlers)
    local_handlers["code_execute"] = _code_execute

    # Production reaches the bundle's synchronous sanitize hook through the
    # runner's awaited `on_tool_result` seam (`easy.py` `_combined_on_tool_result`).
    async def _sanitize_tool_result(ctx: ToolResultContext) -> None:
      bundle.sanitize_hook(ctx)

    event_log = EventLog()
    provider = _RecordingProvider([
      _tool_use_turn("tool-1", "code_execute", {"code": "print(1)"}),
      _text_turn("done"),
    ])
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=_dispatcher(
        event_log,
        local_handlers,
        approval_key_qualifier=bundle.approval_qualifier,
      ),
      session_id="sess-spill",
      capability_execution=stub_runner_capability_execution(
        provider=provider,
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      get_tool_definitions=lambda: list(bundle.tool_definitions),
      on_tool_result=_sanitize_tool_result,
      user_id="alice",
      request_id="req-spill",
      billing_mode="byok",
      rate_table_version="unknown",
      code_execution_spill_dir_provider=SpillSink(
        root_provider=bundle.ensure_work_dir,
        capabilities=SpillCapabilities(code_execute=True, spill_read=True),
      ),
    )

    durable_events: list[dict[str, Any]] = []
    append_durable_event = runner._append_durable_event

    async def _capture_durable_event(event: dict[str, Any]) -> Any:
      durable_events.append(event)
      return await append_durable_event(event)

    runner._append_durable_event = _capture_durable_event

    await runner.run(messages=[{"role": "user", "content": "plot"}], system_prompt="x", max_turns=2)

    streamed_event = next(
      entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"
    )
    # The hook stripped the images from the model block, so that block alone
    # never reveals how large the event is.
    assert "A" * 1024 not in streamed_event["final_tool_result_blocks"][0]["content"]
    # The bounded transport must accept the streamed projection unchanged.
    snapshot_autonomous_event(streamed_event)
    assert streamed_event["result"]["spill_ref"].startswith("spill:v1:")
    # The images stream elided, each cut marked in place, never as their base64.
    for image in streamed_event["result"]["images"]:
      assert "...<elided chars=" in image["data_base64"]
      assert image["filename"].startswith("plot-")

    durable_event = next(
      event for event in durable_events if event.get("type") == "tool_call_complete"
    )
    assert durable_event["result"]["images"][0]["data_base64"] == "A" * (500 * 1024)

  _run(_run_test())


def test_run_agent_sub_runner_spills_into_parent_work_dir(
  tmp_path: Path,
  owner_session_host_policy,
) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda name: (
    "read" if name == "file_read" else None
  )
  async def _run_test() -> None:
    payload = "x" * PAYLOAD_SIZE

    async def _big_data(_tool_input: dict[str, Any], **kwargs: Any):
      _ = kwargs
      return {"status": "success", "payload": payload}, None

    provider = _RecordingProvider([
      _tool_use_turn(
        "parent-run",
        "run_agent",
        {
          "background": False,
          "objective": "read the big data",
        },
      ),
      _tool_use_turn("sub-big", "file_read"),
      _text_turn("read ok"),
      _text_turn("parent done"),
    ])
    base_resolver = stub_capability_execution_resolver(
      default_provider="stub",
      default_model="stub-model",
      extra_models=(("stub", "stub-model"),),
    )
    capability_execution_resolver = replace(
      base_resolver,
      adapter_resolver=lambda _adapter_id: provider,
    )
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
      role="owner",
    )
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, work_dir_root=str(tmp_path)),
    )
    spill_sink = SpillSink(
      root_provider=bundle.ensure_work_dir,
      capabilities=SpillCapabilities(code_execute=True, spill_read=True),
    )
    local_handlers = dict(bundle.handlers)
    local_handlers["file_read"] = _big_data
    local_handlers["tool_result_read"] = make_tool_result_read_handler(
      lambda: spill_sink
    )
    runner_ref: list[Any] = [None]
    local_handlers["run_agent"] = make_run_agent_handler(
      runner_ref,
      parent_session=session,
      mcp_client=_NullMcpClient(),
      local_tool_handlers=local_handlers,
      capability_execution_resolver=capability_execution_resolver,
      approval_key_qualifier=bundle.approval_qualifier,
    )

    event_log = EventLog()
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=_dispatcher(event_log, local_handlers, approval_key_qualifier=bundle.approval_qualifier),
      session_id="sess-parent",
      capability_execution=stub_runner_capability_execution(
        provider=provider,
        auth_config={"api_key": "k"},
        model="stub-model",
        effort="none",
      ),
      get_tool_definitions=lambda: [
        _tool_def("run_agent"),
        _tool_def("file_read"),
        _tool_def("tool_result_read"),
        *bundle.tool_definitions,
      ],
      user_id="alice",
      request_id="req-spill",
      billing_mode="byok",
      rate_table_version="unknown",
      agent_session_log=AgentSessionLog(
        path=tmp_path / "spill-agent-session.jsonl"
      ),
      workspace_dir=str(tmp_path),
      code_execution_spill_dir_provider=spill_sink,
    )
    runner_ref[0] = runner

    await runner.run(messages=[{"role": "user", "content": "delegate"}], system_prompt="x", max_turns=2)

    assert provider.last_spill_ref is not None
    recovered = read_spill_result(
      spill_sink,
      spill_ref=provider.last_spill_ref,
    )
    assert json.loads(recovered["content"]) == {
      "status": "success",
      "payload": payload,
    }
    run_agent_event = next(
      entry.event
      for entry in event_log.entries
      if entry.event.get("type") == "tool_call_complete" and entry.event.get("tool_name") == "run_agent"
    )
    assert (
      run_agent_event["result"]["settlement_projection"]["execution_status"]
      == "succeeded"
    )
    assert (
      run_agent_event["result"]["parent_materialization"]["kind"]
      == "terminal_narrative_inline_exact"
    )

  _run(_run_test())



