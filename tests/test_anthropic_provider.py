# ruff: noqa: E402

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

from agent_gateway import AnthropicProvider, ModelInfo, ThinkingLevel
import agent_gateway.providers.anthropic as anthropic_provider_module
import agent_gateway.providers.anthropic_helpers as anthropic_helpers
from agent_gateway.providers.anthropic import _MODEL_INFO_BY_TAG, _format_anthropic_rejection_detail


def test_server_tool_usage_preserves_known_billable_units_and_rejects_unknown_positive() -> None:
  usage = SimpleNamespace(server_tool_use=SimpleNamespace(
    web_search_requests=2, web_fetch_requests=3,
  ))
  assert anthropic_provider_module._server_tool_unit_deltas(usage) == {
    "web_fetch": 3, "web_search": 2,
  }

  unknown = SimpleNamespace(server_tool_use={
    "web_search_requests": 1, "future_paid_requests": 2,
  })
  with pytest.raises(ValueError, match="unrecognized separately billed"):
    anthropic_provider_module._server_tool_units(unknown)
  with pytest.raises(ValueError, match="invalid Anthropic"):
    anthropic_provider_module._server_tool_unit_deltas(SimpleNamespace(
      server_tool_use={"web_search_requests": True, "web_fetch_requests": 0},
    ))
  with pytest.raises(ValueError, match="invalid Anthropic"):
    anthropic_provider_module._server_tool_unit_deltas(SimpleNamespace(
      server_tool_use={"web_search_requests": 1.5, "web_fetch_requests": 0},
    ))
  with pytest.raises(ValueError, match="unrecognized separately billed"):
    anthropic_provider_module._server_tool_unit_deltas(SimpleNamespace(
      server_tool_use={
        "web_search_requests": 0, "web_fetch_requests": 0,
        "future_paid_requests": "1",
      },
    ))


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


def test_anthropic_provider_helper_exports_are_parent_aliases() -> None:
  helper_names = (
    "_COMMON_BETA_SLUGS",
    "_COMPACTION_BETA_SLUG",
    "_ERROR_REDACTION",
    "_MAX_ERROR_DETAIL_LEN",
    "_MAX_TOOL_ID_LEN",
    "_MODEL_INFO_BY_TAG",
    "_OAUTH_BETA_SLUGS",
    "_OAUTH_IDENTITY",
    "_SENSITIVE_ERROR_KEY_RE",
    "_SENSITIVE_ERROR_VALUE_RES",
    "_TOOL_ID_RE",
    "_exception_body",
    "_exception_status_code",
    "_format_anthropic_rejection_detail",
    "_has_tool_result_block",
    "_model_info_for_model",
    "_model_matches_tag",
    "_normalize_tool_call_id",
    "_redact_error_body",
    "_response_header",
    "_same_model_message",
    "_stream_request_context",
    "_synthetic_tool_result",
    "_thinking_param",
    "_to_plain_dict",
    "_truncate_error_detail",
  )

  for name in helper_names:
    assert getattr(anthropic_provider_module, name) is getattr(anthropic_helpers, name)


def test_model_info_defaults_derive_thinking_mode_from_supports_thinking() -> None:
  assert ModelInfo(id="stub", provider="test").thinking_mode == "none"
  info = ModelInfo(id="stub", provider="test", supports_thinking=True)

  assert info.thinking_mode == "adaptive"
  assert info.supports_thinking is True


def test_fable_model_info_uses_bundled_rates_and_adaptive_thinking() -> None:
  provider = AnthropicProvider()

  info = provider.get_model_info("claude-fable-5")

  assert info.context_window == 1_000_000
  assert info.max_output_tokens == 32_000
  assert info.input_cost_per_mtok == 10.0
  assert info.output_cost_per_mtok == 50.0
  assert info.cache_read_cost_per_mtok == 1.0
  assert info.cache_write_cost_per_mtok == 12.5
  assert info.supports_thinking is True
  assert info.thinking_mode == "adaptive"


def test_opus48_model_info_uses_bundled_rates_and_adaptive_thinking() -> None:
  provider = AnthropicProvider()

  info = provider.get_model_info("claude-opus-4-8")

  assert info.context_window == 1_000_000
  assert info.max_output_tokens == 32_000
  assert info.input_cost_per_mtok == 5.0
  assert info.output_cost_per_mtok == 25.0
  assert info.cache_read_cost_per_mtok == 0.5
  assert info.cache_write_cost_per_mtok == 6.25
  assert info.supports_thinking is True
  assert info.thinking_mode == "adaptive"


def test_haiku_45_model_info_preserves_no_thinking_with_real_rates() -> None:
  provider = AnthropicProvider()

  info = provider.get_model_info("claude-haiku-4-5")

  assert info.input_cost_per_mtok == 1.0
  assert info.output_cost_per_mtok == 5.0
  assert info.cache_read_cost_per_mtok == 0.1
  assert info.cache_write_cost_per_mtok == 1.25
  assert info.supports_thinking is False
  assert info.thinking_mode == "none"


@pytest.mark.parametrize(
  ("model", "expected"),
  [
    ("claude-fable-5", {"type": "adaptive"}),
    ("claude-opus-4-8", {"type": "adaptive"}),
    ("claude-opus-4-7", {"type": "adaptive"}),
    ("claude-sonnet-4-6", {"type": "adaptive"}),
    ("claude-opus-4-6", {"type": "adaptive"}),
    ("claude-sonnet-4-5", {"type": "enabled", "budget_tokens": 10000}),
    ("claude-opus-4-5", {"type": "enabled", "budget_tokens": 10000}),
    ("claude-sonnet-4", {"type": "enabled", "budget_tokens": 10000}),
    ("claude-haiku-4-5", None),
    ("claude-haiku-4-5-20251001", None),
    ("claude-3.7-sonnet-20250219", None),
    ("claude-3-opus-20240229", None),
  ],
)
def test_thinking_param_matches_existing_model_capability_mapping(
  model: str,
  expected: dict[str, object] | None,
) -> None:
  assert AnthropicProvider.thinking_param(model, 12_000) == expected


def test_thinking_param_mapping_covers_anthropic_model_info_table() -> None:
  mapped_models = {tag for tags, _info in _MODEL_INFO_BY_TAG for tag in tags}

  assert {
    "claude-fable-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-sonnet-4-5",
    "claude-opus-4-5",
    "claude-sonnet-4",
    "claude-haiku-4-5",
    "claude-3",
  } <= mapped_models


def test_unknown_claude_model_defaults_to_adaptive_thinking() -> None:
  provider = AnthropicProvider()

  info = provider.get_model_info("claude-zenith-9")

  assert info.supports_thinking is True
  assert info.thinking_mode == "adaptive"
  assert AnthropicProvider.thinking_param("claude-zenith-9", 4096) == {"type": "adaptive"}


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-3.7-sonnet-20250219"])
def test_known_non_thinking_models_emit_no_thinking_param(model: str) -> None:
  provider = AnthropicProvider()

  params = provider.build_request_params(
    model=model,
    messages=[],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
    thinking_level=ThinkingLevel.HIGH,
  )

  assert "thinking" not in params


def test_fable_omits_thinking_when_disabled_or_below_gate_and_never_sends_disabled() -> None:
  provider = AnthropicProvider()
  disabled = provider.build_request_params(
    model="claude-fable-5",
    messages=[],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
    thinking_level=ThinkingLevel.NONE,
  )
  below_gate = provider.build_request_params(
    model="claude-fable-5",
    messages=[],
    system_prompt=None,
    tools=[],
    max_tokens=1024,
    thinking_level=ThinkingLevel.HIGH,
  )

  assert "thinking" not in disabled
  assert "thinking" not in below_gate
  assert "disabled" not in str(disabled)
  assert "disabled" not in str(below_gate)


def test_fable_request_params_do_not_send_sampling_knobs() -> None:
  provider = AnthropicProvider()

  params = provider.build_request_params(
    model="claude-fable-5",
    messages=[],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
    thinking_level=ThinkingLevel.HIGH,
  )

  assert params["thinking"] == {"type": "adaptive"}
  for key in ("temperature", "top_p", "top_k"):
    assert key not in params


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


def test_stream_separates_provider_ping_from_silent_progress_metadata() -> None:
  provider = AnthropicProvider()
  client = _FakeStreamingClient(
    [
      SimpleNamespace(type="ping"),
      SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking")),
      SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="signature_delta", signature="sig")),
      SimpleNamespace(type="content_block_stop"),
      SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="compaction")),
      SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="compaction_delta", content="summary")),
      SimpleNamespace(type="content_block_stop"),
    ]
  )

  types = asyncio.run(_collect_stream_types(provider, client, {"model": "claude-sonnet-4-6", "messages": []}))

  assert types == [
    "heartbeat",
    "stream_progress",
    "stream_progress",
    "thinking_end",
    "stream_progress",
    "compaction",
    "message_end",
  ]


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


def test_normalize_messages_truncates_history_before_last_compaction_block() -> None:
  provider = AnthropicProvider()
  messages = [
    {"role": "user", "content": "original question"},
    {"role": "assistant", "content": [{"type": "text", "text": "early answer"}]},
    {"role": "user", "content": "follow-up"},
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "summary of everything so far"},
        {"type": "text", "text": "post-compaction answer"},
      ],
    },
    {"role": "user", "content": "next question"},
  ]

  normalized = provider.normalize_messages(messages, _model_info())

  assert len(normalized) == 2
  assert normalized[0]["role"] == "assistant"
  assert normalized[0]["content"][0] == {
    "type": "compaction",
    "content": "summary of everything so far",
  }
  assert normalized[0]["content"][1]["type"] == "text"
  assert normalized[1] == {"role": "user", "content": "next question"}


def test_normalize_messages_truncates_to_last_of_multiple_compaction_blocks() -> None:
  provider = AnthropicProvider()
  messages = [
    {"role": "user", "content": "q1"},
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "first summary"},
        {"type": "text", "text": "a1"},
      ],
    },
    {"role": "user", "content": "q2"},
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "second summary"},
        {"type": "text", "text": "a2"},
      ],
    },
    {"role": "user", "content": "q3"},
  ]

  normalized = provider.normalize_messages(messages, _model_info())

  assert len(normalized) == 2
  assert normalized[0]["content"][0]["content"] == "second summary"
  assert normalized[1] == {"role": "user", "content": "q3"}


def test_normalize_messages_without_compaction_block_is_untouched() -> None:
  provider = AnthropicProvider()
  messages = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    {"role": "user", "content": "follow-up"},
  ]

  normalized = provider.normalize_messages(messages, _model_info())

  assert len(normalized) == 3
  assert normalized[0] == {"role": "user", "content": "question"}


def test_normalize_messages_compaction_keeps_tool_pairing_after_anchor() -> None:
  provider = AnthropicProvider()
  messages = [
    {"role": "user", "content": "big history"},
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "summary"},
        {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"ticker": "AAPL"}},
      ],
    },
    {
      "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": "tool-1", "content": "{\"ok\": true}"},
      ],
    },
  ]

  normalized = provider.normalize_messages(messages, _model_info())

  assert len(normalized) == 2
  assert normalized[0]["content"][0]["type"] == "compaction"
  assert normalized[0]["content"][1]["type"] == "tool_use"
  assert normalized[1]["content"][0]["tool_use_id"] == "tool-1"


def test_truncate_helper_converts_compaction_to_text_for_foreign_providers() -> None:
  from agent_gateway.providers.base import truncate_to_last_compaction

  messages = [
    {"role": "user", "content": "big history"},
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "summary text"},
        {"type": "text", "text": "answer"},
      ],
    },
    {"role": "user", "content": "next"},
  ]

  truncated = truncate_to_last_compaction(messages, compaction_as_text=True)

  assert len(truncated) == 2
  first_block = truncated[0]["content"][0]
  assert first_block["type"] == "text"
  assert "summary text" in first_block["text"]
  assert "[Summary of the earlier conversation]" in first_block["text"]


def test_truncate_helper_drops_orphaned_tool_results_from_anchor_prefix() -> None:
  from agent_gateway.providers.base import truncate_to_last_compaction

  messages = [
    {"role": "user", "content": "q"},
    {
      "role": "assistant",
      "content": [
        {"type": "tool_use", "id": "tool-pre", "name": "lookup", "input": {}},
        {"type": "compaction", "content": "summary"},
        {"type": "tool_use", "id": "tool-post", "name": "lookup", "input": {}},
      ],
    },
    {
      "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": "tool-pre", "content": "stale"},
        {"type": "tool_result", "tool_use_id": "tool-post", "content": "fresh"},
      ],
    },
  ]

  truncated = truncate_to_last_compaction(messages)

  assert truncated[0]["content"][0]["type"] == "compaction"
  follower_results = [b["tool_use_id"] for b in truncated[1]["content"]]
  assert follower_results == ["tool-post"]


def test_truncate_helper_as_text_summary_ends_with_separator() -> None:
  from agent_gateway.providers.base import truncate_to_last_compaction

  messages = [
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "summary"},
        {"type": "text", "text": "answer"},
      ],
    },
  ]

  truncated = truncate_to_last_compaction(messages, compaction_as_text=True)

  assert truncated[0]["content"][0]["text"].endswith("\n\n")
