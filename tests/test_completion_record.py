import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import jsonschema
import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.completion_record import (
  append_completion_record,
  build_completion_record,
  completion_record_schema_path,
)
from agent_gateway.providers import CostEstimate, ModelInfo, ModelProvider, StreamEvent
from agent_gateway.send_prompt import send_prompt
from gateway_test_support.capability_execution_test_support import stub_bound_capability_execution


def _schema() -> dict[str, Any]:
  return json.loads(completion_record_schema_path().read_text(encoding="utf-8"))


def _rows(root: Path) -> list[dict[str, Any]]:
  return [
    json.loads(line)
    for path in sorted(root.glob("*.jsonl"))
    for line in path.read_text(encoding="utf-8").splitlines()
  ]


class _Provider(ModelProvider):
  def __init__(self, events: list[StreamEvent], *, fail: Exception | None = None) -> None:
    self.name = "anthropic"
    self._events = events
    self._fail = fail

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name, supports_thinking=True)

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    return None

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return {"model": kwargs["model"], "max_tokens": kwargs["max_tokens"], "messages": kwargs["messages"]}

  async def stream(self, client: Any, params: dict[str, Any]):
    for event in self._events:
      yield event
    if self._fail is not None:
      raise self._fail

  def estimate_cost(self, model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0, cache_creation_tokens: int = 0) -> CostEstimate:
    return CostEstimate(total=0.0)


def _execution(provider: _Provider):
  return stub_bound_capability_execution(
    provider=provider,
    model="claude-sonnet-4-6",
    effort="none",
    credential_principal="user",
    auth_config={"provider": "anthropic", "max_tokens": 4096, "auth_mode": "api", "api_key": "k"},
  )


def test_send_prompt_records_the_stream_and_returns_the_same_text(tmp_path: Path) -> None:
  thinking = {"type": "thinking", "thinking": "weigh it", "signature": "sig"}
  text_block = {"type": "text", "text": "Gross margin was 46%."}
  provider = _Provider([
    StreamEvent(type="message_start", input_tokens=12, cache_read_tokens=3, cache_creation_tokens=0),
    StreamEvent(type="thinking_end", raw_block=thinking),
    StreamEvent(type="text_delta", text="Gross margin was 46%."),
    StreamEvent(type="text_end", raw_block=text_block),
    StreamEvent(type="usage_update", output_tokens=7),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ])
  text = asyncio.run(send_prompt(
    "What was the gross margin?",
    capability_execution=_execution(provider),
    user_id="alice",
    session_id="newsletter-1",
    completion_record_root=tmp_path,
  ))

  assert text == "Gross margin was 46%."
  [row] = _rows(tmp_path)
  jsonschema.validate(row, _schema())
  assert row["capture_level"] == "stream"
  assert row["outcome"]["status"] == "ok"
  assert row["response"] == {
    "content_blocks": [thinking, text_block],
    "stop_reason": "end_turn",
    "provider_response_id": None,
    "provider_request_id": None,
    "service_tier": None,
    "rate_limit": None,
  }
  assert row["request"]["messages"] == [{"role": "user", "content": "What was the gross margin?"}]
  assert row["usage"]["input_tokens"] == 12 and row["usage"]["output_tokens"] == 7
  assert row["join"]["session_id"] == "newsletter-1"
  assert row["model"]["capability_bind"]["upstream_model"] == "claude-sonnet-4-6"
  assert row["source"]["call_site"] == row["model"]["capability_bind"]["capability_id"]


def test_send_prompt_records_provider_diagnostics_and_reaches_the_usage_event(tmp_path: Path) -> None:
  thinking = {"type": "thinking", "thinking": "weigh it", "signature": "sig"}
  redacted = {"type": "redacted_thinking", "data": "opaque"}
  limits = {"anthropic-ratelimit-requests-remaining": "41", "retry-after": "2"}
  provider = _Provider([
    StreamEvent(
      type="message_start", input_tokens=12, provider_response_id="msg_abc",
      provider_request_id="req_123", rate_limit=limits,
    ),
    StreamEvent(type="thinking_delta", thinking_text="weigh "),
    StreamEvent(type="thinking_delta", thinking_text="it"),
    StreamEvent(type="thinking_end", thinking_text="weigh it", raw_block=thinking),
    StreamEvent(type="thinking_end", raw_block=redacted),
    StreamEvent(type="text_delta", text="ok"),
    StreamEvent(type="usage_update", output_tokens=7),
    StreamEvent(type="message_end", stop_reason="end_turn", service_tier="standard"),
  ])
  usage_events: list[Any] = []
  asyncio.run(send_prompt(
    "q",
    capability_execution=_execution(provider),
    user_id="alice",
    completion_record_root=tmp_path,
    on_usage=usage_events.append,
  ))

  [row] = _rows(tmp_path)
  jsonschema.validate(row, _schema())
  # The thinking blocks themselves (including the redacted one) are in content_blocks.
  assert row["response"] == {
    "content_blocks": [thinking, redacted],
    "stop_reason": "end_turn",
    "provider_response_id": "msg_abc",
    "provider_request_id": "req_123",
    "service_tier": "standard",
    "rate_limit": limits,
  }
  assert row["usage"]["thinking_chars"] == len("weigh it")
  [usage_event] = usage_events
  assert (
    usage_event.provider_response_id, usage_event.provider_request_id,
    usage_event.service_tier, usage_event.thinking_chars,
  ) == ("msg_abc", "req_123", "standard", len("weigh it"))


def test_send_prompt_records_a_failed_attempt_with_no_response(tmp_path: Path) -> None:
  provider = _Provider(
    [StreamEvent(type="message_start", input_tokens=5)],
    fail=RuntimeError("upstream reset"),
  )
  with pytest.raises(RuntimeError, match="upstream reset"):
    asyncio.run(send_prompt(
      "q",
      capability_execution=_execution(provider),
      user_id="alice",
      completion_record_root=tmp_path,
    ))

  [row] = _rows(tmp_path)
  jsonschema.validate(row, _schema())
  assert row["response"] is None
  assert row["outcome"] == {"status": "error", "error_type": "RuntimeError", "error_message": "upstream reset"}
  assert row["usage"]["input_tokens"] == 5


def test_send_prompt_without_a_root_writes_nothing(tmp_path: Path) -> None:
  provider = _Provider([StreamEvent(type="text_delta", text="x"), StreamEvent(type="message_end", stop_reason="end_turn")])
  assert asyncio.run(send_prompt(
    "q", capability_execution=_execution(provider), user_id="alice", completion_record_root=None,
  )) == "x"
  assert list(tmp_path.iterdir()) == []


def test_binary_document_blocks_become_content_addressed_refs() -> None:
  record = build_completion_record(
    source_repo="risk_module",
    call_site="risk.document_ingest",
    provider="anthropic",
    model="m",
    capture_level="wire",
    request={"messages": [{"role": "user", "content": [
      {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "aGVsbG8="}},
      {"type": "text", "text": "extract"},
    ]}]},
    response=None,
    status="error",
    started_at=datetime(2026, 10, 12, tzinfo=timezone.utc),
  )
  document, text = record["request"]["messages"][0]["content"]
  assert document == {"type": "document", "source": {
    "type": "blob_ref",
    "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
    "media_type": "application/pdf",
    "bytes": 5,
  }}
  assert text == {"type": "text", "text": "extract"}
  jsonschema.validate(record, _schema())


def test_unwritable_root_never_fails_the_call(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
  blocker = tmp_path / "not-a-dir"
  blocker.write_text("file", encoding="utf-8")
  provider = _Provider([StreamEvent(type="text_delta", text="y"), StreamEvent(type="message_end", stop_reason="end_turn")])
  with caplog.at_level("ERROR", logger="agent_gateway.completion_record"):
    assert asyncio.run(send_prompt(
      "q", capability_execution=_execution(provider), user_id="alice", completion_record_root=blocker,
    )) == "y"
  assert "completion_record_write_failed" in caplog.text


def test_rows_append_to_the_day_shard_named_by_recorded_at(tmp_path: Path) -> None:
  started = datetime(2026, 10, 11, 23, 59, tzinfo=timezone.utc)
  for recorded in (datetime(2026, 10, 11, 23, 59, 30, tzinfo=timezone.utc), datetime(2026, 10, 12, 0, 0, 1, tzinfo=timezone.utc)):
    append_completion_record(tmp_path, build_completion_record(
      source_repo="edgar_updater", call_site="edgar.earnings_8k.anthropic", provider="anthropic",
      model="m", capture_level="wire", request={}, response={}, status="ok",
      started_at=started, recorded_at=recorded,
    ))
  assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-10-11.jsonl", "2026-10-12.jsonl"]
