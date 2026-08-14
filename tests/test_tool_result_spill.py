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
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, AgentSessionLog, EventLog, ModelInfo, ModelProvider, SessionStore, ToolDispatcher  # noqa: E402
from agent_gateway.code_execution import CodeExecutionConfig, DockerBackend, build_code_execution  # noqa: E402
from agent_gateway.providers import StreamEvent  # noqa: E402
from agent_gateway.sub_agent import make_run_agent_handler  # noqa: E402
from agent_gateway.tool_result_compaction import (  # noqa: E402
  annotate_result,
  compact_model_tool_result_entry,
  is_error_tool_result_entry,
  make_error_result,
  truncate_model_tool_result_content,
  write_tool_result_spill,
)
import agent_gateway.tool_result_compaction as tool_result_compaction  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from tests.capability_execution_test_support import (  # noqa: E402
  stub_capability_execution_resolver,
  stub_runner_capability_execution,
)


CAP = 4_000
PAYLOAD_SIZE = 5_200


def _run(coro):
  return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _small_tool_result_cap(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(gateway_runner.MODEL_TOOL_RESULT_MAX_CHARS_ENV, str(CAP))
  monkeypatch.delenv(gateway_runner.SPILL_TRUNCATED_TOOL_RESULTS_ENV, raising=False)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  def get_server_for_tool(self, _name: str) -> str | None:
    return None


class _RecordingProvider(ModelProvider):
  name = "stub"

  def __init__(self, turns: list[list[StreamEvent] | Callable[["_RecordingProvider"], list[StreamEvent]]]) -> None:
    self._turns = list(turns)
    self._stream_index = 0
    self.params_history: list[dict[str, Any]] = []
    self.last_spill_file: str | None = None
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
      spill_file = payload.get("spill_file")
      if isinstance(spill_file, str):
        self.last_spill_file = spill_file
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
    code_execution_spill_dir_provider=spill_provider,
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


def test_emit_canvas_artifact_oversize_guard_returns_error_before_dispatch() -> None:
  # INC-4 retired emit_html_artifact; the pre-dispatch oversize guard now covers
  # emit_canvas_artifact (tsx_source vs the kit-contract source cap).
  from agent_gateway.canvas_kit_contract import limits as canvas_kit_limits

  runner = _runner(None)
  source_cap = canvas_kit_limits()["source_max_bytes"]
  tsx_source = "x" * (source_cap + 1)

  result, tool_name, live_events = _run(
    runner._execute_single_tool(
      "tool-canvas",
      "emit_canvas_artifact",
      {"tsx_source": tsx_source},
      {},
    )
  )

  assert tool_name == "emit_canvas_artifact"
  assert live_events == []
  assert result["is_error"] is True
  payload = json.loads(result["content"])
  assert payload["error"]["code"] == "invalid_input"
  assert f"exceeds {source_cap} byte limit" in payload["error"]["message"]
  assert [entry.event for entry in runner._log.entries] == []


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


def test_truncate_tool_result_embeds_spill_pointer_only_when_provided() -> None:
  content = _large_content()

  truncated, was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="lookup",
    max_chars=CAP,
    spill_filename="lookup_tool-1.json",
    spill_abspath="/tmp/ce/lookup_tool-1.json",
  )

  assert was_truncated is True
  assert len(truncated) <= CAP
  payload = json.loads(truncated)
  assert payload["spill_file"] == "lookup_tool-1.json"
  assert payload["spill_abspath"] == "/tmp/ce/lookup_tool-1.json"
  assert "FULL, untruncated result" in payload["spill_hint"]
  assert "pd.read_json('lookup_tool-1.json')" in payload["spill_hint"]

  plain, plain_was_truncated = truncate_model_tool_result_content(
    content,
    tool_name="lookup",
    max_chars=CAP,
  )

  assert plain_was_truncated is True
  plain_payload = json.loads(plain)
  assert "spill_file" not in plain_payload
  assert "spill_abspath" not in plain_payload
  assert "spill_hint" not in plain_payload


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
    spill_dir_provider=lambda: str(tmp_path),
    log_session_id="sess-direct",
    logger=logger,
    uuid_factory=lambda: SimpleNamespace(hex="e" * 32),
  )

  spill_files = list(tmp_path.iterdir())
  assert len(spill_files) == 1
  assert json.loads(spill_files[0].read_text(encoding="utf-8")) == json.loads(content)
  live_payload = json.loads(live_entry["content"])
  durable_payload = json.loads(durable_entry["content"])
  assert live_payload["spill_file"] == spill_files[0].name
  assert live_payload["spill_abspath"] == str(spill_files[0])
  assert "spill_file" not in durable_payload
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
  assert live_payload["spill_file"] == spill_files[0].name
  assert live_payload["spill_abspath"] == str(spill_files[0])
  assert "spill_file" not in durable_payload
  assert "spill_abspath" not in durable_payload
  assert "spill_hint" not in durable_payload


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
  assert "spill_file" not in json.loads(live_entry["content"])
  assert live_entry == durable_entry
  assert list(tmp_path.iterdir()) == []

  no_provider_entry = {"type": "tool_result", "tool_use_id": "no-provider", "content": content}
  live_entry, durable_entry = _runner(None)._compact_model_tool_result_entry(
    no_provider_entry,
    tool_name="lookup",
  )
  assert "spill_file" not in json.loads(live_entry["content"])
  assert live_entry == durable_entry

  monkeypatch.setenv(gateway_runner.SPILL_TRUNCATED_TOOL_RESULTS_ENV, "no")
  disabled_entry = {"type": "tool_result", "tool_use_id": "disabled", "content": content}
  live_entry, durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    disabled_entry,
    tool_name="lookup",
  )
  assert "spill_file" not in json.loads(live_entry["content"])
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
  assert json.loads(live_entry["content"])["spill_file"] == spill_files[0].name
  assert "spill_file" not in json.loads(durable_entry["content"])

  real_error = json.dumps({"error": {"code": "bad", "message": "x" * PAYLOAD_SIZE}}, default=str)
  err_entry = {"type": "tool_result", "tool_use_id": "real-error", "content": real_error}
  live_entry, durable_entry = _runner(lambda: str(tmp_path))._compact_model_tool_result_entry(
    err_entry,
    tool_name="lookup",
  )
  assert "spill_file" not in json.loads(live_entry["content"])
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
  assert "spill_file" not in live_payload
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
  filename = live_payload["spill_file"]
  spill_path = Path(live_payload["spill_abspath"])
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

  filename = json.loads(live_entry["content"])["spill_file"]
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

  filename = json.loads(live_entry["content"])["spill_file"]
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


def test_runner_spills_large_tool_result_and_code_execute_reads_bare_filename(tmp_path: Path) -> None:
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
      code_execution_spill_dir_provider=bundle.ensure_work_dir,
    )

    await runner.run(messages=[{"role": "user", "content": "load"}], system_prompt="x", max_turns=2)

    expected_content = json.dumps({"status": "success", "payload": payload}, default=str)
    work_dir = Path(session.code_execution_work_dir or "")
    spill_files = [path for path in work_dir.iterdir() if path.name.startswith("big_data_tool-1")]
    assert len(spill_files) == 1
    assert json.loads(spill_files[0].read_text(encoding="utf-8")) == json.loads(expected_content)

    live_blocks = _model_bound_tool_result_blocks(provider)
    live_payload = json.loads(live_blocks[0]["content"])
    assert live_payload["spill_file"] == spill_files[0].name
    assert live_payload["spill_abspath"] == str(spill_files[0])

    complete_event = next(entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete")
    durable_payload = json.loads(complete_event["final_tool_result_blocks"][0]["content"])
    assert "spill_file" not in durable_payload
    assert "spill_abspath" not in durable_payload
    assert "spill_hint" not in durable_payload

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


def test_run_agent_sub_runner_spills_into_parent_work_dir(tmp_path: Path) -> None:
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
    local_handlers = dict(bundle.handlers)
    local_handlers["file_read"] = _big_data
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
      code_execution_spill_dir_provider=bundle.ensure_work_dir,
    )
    runner_ref[0] = runner

    await runner.run(messages=[{"role": "user", "content": "delegate"}], system_prompt="x", max_turns=2)

    work_dir = Path(session.code_execution_work_dir or "")
    assert provider.last_spill_file is not None
    spill_path = work_dir / provider.last_spill_file
    assert spill_path.exists()
    assert json.loads(spill_path.read_text(encoding="utf-8")) == {
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


def test_interactive_model_provider_runner_threads_spill_provider(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("MCP_CONFIG_PATH", str(ROOT / "deploy" / "mcp.production.json"))
  import api.agent.interactive.runtime as runtime

  captured: dict[str, Any] = {}

  class _FakeRunner:
    def __init__(self, **kwargs: Any) -> None:
      captured.update(kwargs)

  class _FakeProvider(ModelProvider):
    name = "stub"

    def has_active_credential(self, config: dict[str, Any]) -> bool:
      return bool(config.get("api_key"))

    def get_model_info(self, model: str) -> ModelInfo:
      return ModelInfo(id=model, provider="stub")

  monkeypatch.setattr(runtime, "AgentRunner", _FakeRunner)
  def spill_provider() -> str:
    return "/tmp/spill"
  runner_ref: list[Any] = [None]
  session = SimpleNamespace(
    result_queue=None,
    approval_store=None,
    approval_policy=None,
    user_id="alice",
    session_id="sess-runtime",
    loaded_mcp_servers=set(),
    channel="web",
    role="owner",
  )
  capability_execution = stub_runner_capability_execution(
    provider=_FakeProvider(),
    auth_config={"api_key": "k"},
    model="stub-model",
    effort="none",
  )
  request = SimpleNamespace(
    user_id="alice",
    request_id="req-runtime",
    context={},
    capability_execution=capability_execution,
  )

  runner = runtime._build_model_provider_runner(
    EventLog(),
    "sess-runtime",
    10.0,
    request=request,
    session=session,
    runner_ref=runner_ref,
    capability_execution=capability_execution,
    mcp_client_manager=_NullMcpClient(),
    excluded_tools=set(),
    parent_per_turn_timeout=30,
    build_dispatcher=lambda _ctx: object(),
    get_tool_definitions=lambda: [],
    channel="web",
    on_tool_result=lambda _ctx: None,
    code_execution_spill_dir_provider=spill_provider,
  )

  assert captured["code_execution_spill_dir_provider"] is spill_provider
  assert captured["capability_execution"] is capability_execution
  assert runner_ref[0] is runner
