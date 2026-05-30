import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AnthropicProvider, ModelInfo
from agent_gateway.providers.anthropic import _format_anthropic_rejection_detail


def _model_info() -> ModelInfo:
  return ModelInfo(id="claude-sonnet-4-6", provider="anthropic")


def _make_anthropic_api_status_error(status_code: int, message: str):
  anthropic = pytest.importorskip("anthropic")
  request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
  response = httpx.Response(status_code, request=request)
  return anthropic.APIStatusError(
    message,
    response=response,
    body={"error": {"message": message}},
  )


def test_normalize_messages_synthetic_tool_result_has_no_internal_tool_name() -> None:
  provider = AnthropicProvider()
  messages = [
    {
      "role": "assistant",
      "content": [
        {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"ticker": "AAPL"}},
      ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "continuing"}]},
  ]

  normalized = provider.normalize_messages(messages, _model_info())

  synthetic_message = normalized[1]
  assert synthetic_message["role"] == "user"
  synthetic_block = synthetic_message["content"][0]
  assert synthetic_block["type"] == "tool_result"
  assert synthetic_block["tool_use_id"] == "tool-1"
  assert synthetic_block["is_error"] is True
  assert "tool_name" not in synthetic_block
  assert "lookup" in synthetic_block["content"]


def test_normalize_messages_removes_replayed_tool_result_tool_name() -> None:
  provider = AnthropicProvider()
  messages = [
    {
      "role": "assistant",
      "content": [
        {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"ticker": "AAPL"}},
      ],
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "tool-1",
          "tool_name": "lookup",
          "content": "{\"ok\": true}",
        },
      ],
    },
  ]

  normalized = provider.normalize_messages(messages, _model_info())

  tool_result_block = normalized[1]["content"][0]
  assert tool_result_block == {
    "type": "tool_result",
    "tool_use_id": "tool-1",
    "content": "{\"ok\": true}",
  }


class _FakeErrorResponse:
  status_code = 400
  headers = {"request-id": "req_123"}

  def __init__(self, body: dict[str, object]):
    self._body = body

  def json(self) -> dict[str, object]:
    return self._body


class _FakeAnthropicError(Exception):
  status_code = 400

  def __init__(self, body: dict[str, object]):
    super().__init__("fake anthropic status error")
    self.body = body
    self.response = _FakeErrorResponse(body)


class _FailingStreamContext:
  def __init__(self, exc: Exception):
    self._exc = exc

  async def __aenter__(self):
    raise self._exc

  async def __aexit__(self, exc_type, exc, tb):
    return False


class _StaticStreamContext:
  def __init__(self, events: list[object]):
    self._events = events

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    return False

  def __aiter__(self):
    return self._iter()

  async def _iter(self):
    for event in self._events:
      yield event


class _FakeMessages:
  def __init__(self, exc: Exception):
    self.exc = exc
    self.kwargs: dict[str, object] | None = None

  def stream(self, **kwargs):
    self.kwargs = kwargs
    return _FailingStreamContext(self.exc)


class _FakeBeta:
  def __init__(self, exc: Exception):
    self.messages = _FakeMessages(exc)


class _FakeClient:
  def __init__(self, exc: Exception):
    self.messages = _FakeMessages(exc)
    self.beta = _FakeBeta(exc)


class _FakeStreamingMessages:
  def __init__(self, events: list[object]):
    self.events = events
    self.kwargs: dict[str, object] | None = None

  def stream(self, **kwargs):
    self.kwargs = kwargs
    return _StaticStreamContext(self.events)


class _FakeStreamingClient:
  def __init__(self, events: list[object]):
    self.messages = _FakeStreamingMessages(events)


async def _drain_stream(provider: AnthropicProvider, client: object, params: dict[str, object]) -> None:
  async for _ in provider.stream(client, params):
    pass


async def _collect_stream_types(provider: AnthropicProvider, client: object, params: dict[str, object]) -> list[str]:
  return [event.type async for event in provider.stream(client, params)]


def test_anthropic_rejection_detail_redacts_sensitive_body_fallback() -> None:
  raw_key = "sk-ant-api03-DETAILKEY123"
  detail = _format_anthropic_rejection_detail(
    _FakeAnthropicError(
      {
        "error": {"type": "invalid_request_error"},
        "api_key": raw_key,
        "authorization": "Bearer secret-token",
      }
    )
  )

  assert detail is not None
  assert "status=400" in detail
  assert "type=invalid_request_error" in detail
  assert raw_key not in detail
  assert "secret-token" not in detail
  assert "[redacted]" in detail


def test_stream_wraps_anthropic_rejection_with_sanitized_context(caplog) -> None:
  provider = AnthropicProvider()
  raw_key = "sk-ant-api03-STREAMDETAILKEY123"
  error = _FakeAnthropicError(
    {
      "error": {
        "type": "invalid_request_error",
        "message": f"context_management cannot be combined with this thinking mode {raw_key}",
      },
      "api_key": raw_key,
    }
  )
  client = _FakeClient(error)
  params = {
    "model": "claude-opus-4-7",
    "max_tokens": 4096,
    "messages": [{"role": "user", "content": "hello"}],
    "tools": [{"name": "lookup"}],
    "thinking": {"type": "adaptive"},
    "context_management": {"edits": []},
    "_provider_auth_mode": "oauth",
  }

  with caplog.at_level(logging.WARNING, logger="agent_gateway.providers.anthropic"):
    with pytest.raises(RuntimeError) as exc_info:
      asyncio.run(_drain_stream(provider, client, params))

  message = str(exc_info.value)
  assert "Anthropic request rejected (stage=stream)" in message
  assert "status=400" in message
  assert "type=invalid_request_error" in message
  assert "context_management cannot be combined with this thinking mode" in message
  assert raw_key not in message
  assert "request_id=req_123" in message
  assert "model=claude-opus-4-7" in message
  assert "auth_mode=oauth" in message
  assert "context_management=enabled" in message
  assert "thinking=adaptive" in message
  assert "messages=1" in message
  assert "tools=1" in message
  assert "compact-2026-01-12" in message
  assert raw_key not in caplog.text


def test_stream_status_200_api_error_remains_retryable() -> None:
  provider = AnthropicProvider()
  error = _make_anthropic_api_status_error(200, "stream failed")
  client = _FakeClient(error)
  params = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 4096,
    "messages": [{"role": "user", "content": "hello"}],
    "tools": [],
  }

  with pytest.raises(Exception) as exc_info:
    asyncio.run(_drain_stream(provider, client, params))

  assert exc_info.value is error
  assert provider.is_retryable_error(error) is True


def test_stream_surfaces_ping_and_silent_thinking_as_internal_heartbeats() -> None:
  provider = AnthropicProvider()
  client = _FakeStreamingClient(
    [
      SimpleNamespace(type="ping"),
      SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking")),
      SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="signature_delta", signature="sig")),
      SimpleNamespace(type="content_block_stop"),
    ]
  )

  types = asyncio.run(_collect_stream_types(provider, client, {"model": "claude-sonnet-4-6", "messages": []}))

  assert types == ["heartbeat", "heartbeat", "heartbeat", "thinking_end", "message_end"]


def test_normalize_messages_drops_orphan_tool_result_message() -> None:
  provider = AnthropicProvider()
  messages = [
    {"role": "user", "content": "Earlier context"},
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "tool-orphan",
          "content": "{\"ok\": true}",
        },
      ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "continuing"}]},
  ]

  normalized = provider.normalize_messages(messages, _model_info())

  assert normalized == [
    {"role": "user", "content": "Earlier context"},
    {"role": "assistant", "content": [{"type": "text", "text": "continuing"}]},
  ]


def test_normalize_messages_filters_unexpected_tool_results_after_tool_use() -> None:
  provider = AnthropicProvider()
  messages = [
    {
      "role": "assistant",
      "content": [
        {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"ticker": "AAPL"}},
      ],
    },
    {
      "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "{\"ok\": true}"},
        {"type": "tool_result", "tool_use_id": "tool-orphan", "content": "{\"stale\": true}"},
      ],
    },
  ]

  normalized = provider.normalize_messages(messages, _model_info())

  assert normalized[1]["content"] == [
    {"type": "tool_result", "tool_use_id": "tool-1", "content": "{\"ok\": true}"},
  ]
