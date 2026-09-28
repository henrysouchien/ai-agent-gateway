# ruff: noqa: E402

import asyncio
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import httpx2
import pytest

if TYPE_CHECKING:
  # `anthropic` is an optional extra (`pyproject.toml:27`), so the runtime
  # reaches it through `pytest.importorskip` below; this is types only.
  from anthropic import RateLimitError

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AnthropicProvider, ModelInfo, ThinkingLevel
import agent_gateway.providers.anthropic as anthropic_provider_module
import agent_gateway.providers.anthropic_helpers as anthropic_helpers
from agent_gateway.providers.anthropic import _format_anthropic_rejection_detail
from agent_gateway.providers.anthropic_oauth import AnthropicCredentialPool
from agent_gateway.providers import StreamEvent


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
    anthropic_provider_module._server_tool_unit_deltas(unknown)
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
  return AnthropicProvider().get_model_info("claude-sonnet-4-6")


def _cached_tool() -> dict[str, object]:
  return {
    "name": "lookup",
    "description": "Look up a value.",
    "input_schema": {"type": "object", "properties": {}},
    "cache_control": {"type": "ephemeral"},
  }


def _cache_marker_locations(params: dict[str, object]) -> list[tuple[str, int, int | None]]:
  locations: list[tuple[str, int, int | None]] = []
  for section in ("system", "tools"):
    blocks = params.get(section)
    if not isinstance(blocks, list):
      continue
    locations.extend(
      (section, index, None)
      for index, block in enumerate(blocks)
      if isinstance(block, dict) and "cache_control" in block
    )
  messages = params.get("messages")
  if isinstance(messages, list):
    for message_index, message in enumerate(messages):
      if not isinstance(message, dict):
        continue
      content = message.get("content")
      if not isinstance(content, list):
        continue
      locations.extend(
        ("messages", message_index, block_index)
        for block_index, block in enumerate(content)
        if isinstance(block, dict) and "cache_control" in block
      )
  return locations


def test_build_request_params_places_fourth_marker_on_interactive_message_tail() -> None:
  params = AnthropicProvider().build_request_params(
    model="claude-sonnet-4-6",
    messages=[{
      "role": "user",
      "content": [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "last"},
      ],
    }],
    system_prompt=[("static", True), ("dynamic", True)],
    tools=[_cached_tool()],
    max_tokens=4096,
  )

  assert _cache_marker_locations(params) == [
    ("system", 0, None),
    ("system", 1, None),
    ("tools", 0, None),
    ("messages", 0, 1),
  ]
  assert params["messages"][-1]["content"][-1]["cache_control"] == {
    "type": "ephemeral"
  }


def test_build_request_params_places_third_marker_for_sub_agent_shape() -> None:
  params = AnthropicProvider().build_request_params(
    model="claude-sonnet-4-6",
    messages=[{
      "role": "user",
      "content": [{"type": "text", "text": "review this"}],
    }],
    system_prompt="sub-agent system",
    tools=[_cached_tool()],
    max_tokens=4096,
  )

  assert _cache_marker_locations(params) == [
    ("system", 0, None),
    ("tools", 0, None),
    ("messages", 0, 0),
  ]


def test_build_request_params_never_adds_a_fifth_explicit_marker() -> None:
  params = AnthropicProvider().build_request_params(
    model="claude-sonnet-4-6",
    messages=[{
      "role": "user",
      "content": [{"type": "text", "text": "do not mark past the API limit"}],
    }],
    system_prompt=[
      ("first", True),
      ("second", True),
      ("third", True),
    ],
    tools=[_cached_tool()],
    max_tokens=4096,
  )

  assert _cache_marker_locations(params) == [
    ("system", 0, None),
    ("system", 1, None),
    ("system", 2, None),
    ("tools", 0, None),
  ]


@pytest.mark.parametrize(
  "final_block",
  [
    pytest.param({"type": "text", "text": "continue"}, id="text"),
    pytest.param(
      {
        "type": "tool_result",
        "tool_use_id": "tool-1",
        "content": "result",
      },
      id="tool-result",
    ),
  ],
)
def test_build_request_params_marks_cacheable_final_block(
  final_block: dict[str, object],
) -> None:
  params = AnthropicProvider().build_request_params(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": [final_block]}],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
  )

  assert _cache_marker_locations(params) == [("messages", 0, 0)]
  assert params["messages"][0]["content"][0]["cache_control"] == {
    "type": "ephemeral"
  }


def test_build_request_params_normalizes_bare_string_final_content_for_marker() -> None:
  params = AnthropicProvider().build_request_params(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "bare prompt"}],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
  )

  assert params["messages"] == [{
    "role": "user",
    "content": [{
      "type": "text",
      "text": "bare prompt",
      "cache_control": {"type": "ephemeral"},
    }],
  }]


def test_build_request_params_skips_marker_without_cacheable_final_block(
  caplog: pytest.LogCaptureFixture,
) -> None:
  caplog.set_level(logging.DEBUG, logger="agent_gateway.providers.anthropic")

  params = AnthropicProvider().build_request_params(
    model="claude-sonnet-4-6",
    messages=[{
      "role": "user",
      "content": [{"type": "thinking", "thinking": "private"}],
    }],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
  )

  assert _cache_marker_locations(params) == []
  skip_records = [
    record
    for record in caplog.records
    if "final message has no cacheable block" in record.getMessage()
  ]
  assert len(skip_records) == 1


def test_build_request_params_strips_stale_message_markers_before_placement() -> None:
  messages = [
    {
      "role": "user",
      "cache_control": {"type": "ephemeral"},
      "content": [{
        "type": "text",
        "text": "old boundary",
        "cache_control": {"type": "ephemeral"},
      }],
    },
    {
      "role": "assistant",
      "content": [{
        "type": "text",
        "text": "answer",
        "cache_control": {"type": "ephemeral"},
      }],
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "not final",
          "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "authoritative boundary"},
      ],
    },
  ]

  params = AnthropicProvider().build_request_params(
    model="claude-sonnet-4-6",
    messages=messages,
    system_prompt=None,
    tools=[],
    max_tokens=4096,
  )

  assert _cache_marker_locations(params) == [("messages", 2, 1)]
  assert "cache_control" not in params["messages"][0]
  assert "cache_control" in messages[0]
  assert "cache_control" in messages[0]["content"][0]


def test_build_request_params_never_marks_trailing_thinking_block() -> None:
  params = AnthropicProvider().build_request_params(
    model="claude-sonnet-4-6",
    messages=[{
      "role": "user",
      "content": [
        {"type": "text", "text": "cache here"},
        {
          "type": "thinking",
          "thinking": "never cache directly",
          "cache_control": {"type": "ephemeral"},
        },
      ],
    }],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
  )

  assert _cache_marker_locations(params) == [("messages", 0, 0)]
  assert "cache_control" not in params["messages"][0]["content"][1]


def _make_anthropic_api_status_error(
  status_code: int,
  message: str,
  *,
  body: dict[str, object] | None = None,
):
  anthropic = pytest.importorskip("anthropic")
  if body is None:
    body = {"error": {"message": message}}
  request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
  response = httpx2.Response(
    status_code,
    request=request,
    headers={"request-id": "req_123"},
    json=body,
  )
  return anthropic.APIStatusError(
    message,
    response=response,
    body=body,
  )


def test_create_client_accepts_configured_timeout_with_real_sdk() -> None:
  anthropic = pytest.importorskip("anthropic")
  provider = AnthropicProvider()
  client = provider.create_client(
    {"auth_mode": "api", "api_key": "bound-api-key"},
    timeout=37.0,
  )
  try:
    assert isinstance(client, anthropic.AsyncAnthropic)
    assert client.timeout.connect == 5.0
    assert client.timeout.read == 37.0
    assert client.timeout.write == 37.0
    assert client.timeout.pool == 37.0
  finally:
    asyncio.run(provider.close_client(client))


def test_create_client_surfaces_429_instead_of_sleeping_its_retry_after(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A provider-reported reset is raised to the caller, not slept inside the request.

  The gateway owns retry policy (runner_stream_turn's bounded loop plus
  `next_credential`); with the SDK's own retry enabled it obeys this
  `retry-after` verbatim, so `RateLimitError` never reaches the sanitizer and
  the turn dies on the stream watchdog instead of reporting the limit.
  """
  anthropic = pytest.importorskip("anthropic")
  provider = AnthropicProvider()
  requests: list[httpx2.Request] = []
  slept: list[float] = []

  async def record_sleep(seconds: float) -> None:
    slept.append(seconds)

  monkeypatch.setattr(anthropic._base_client.anyio, "sleep", record_sleep)

  def handle_request(request: httpx2.Request) -> httpx2.Response:
    requests.append(request)
    return httpx2.Response(
      429,
      headers={"retry-after": "73443"},
      json={
        "type": "error",
        "error": {"type": "rate_limit_error", "message": "usage limit reached"},
      },
    )

  async def send_request() -> "RateLimitError":
    client = provider.create_client({"auth_mode": "api", "api_key": "bound-api-key"})
    try:
      async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handle_request),
      ) as http_client:
        async with client.with_options(http_client=http_client) as local_client:
          with pytest.raises(anthropic.RateLimitError) as caught:
            await local_client.messages.create(
              model="claude-sonnet-4-6",
              max_tokens=16,
              messages=[{"role": "user", "content": "hello"}],
            )
      return caught.value
    finally:
      await provider.close_client(client)

  error = asyncio.run(send_request())

  assert len(requests) == 1
  assert slept == []
  assert error.response.headers["retry-after"] == "73443"
  assert provider.is_retryable_error(error) is True


def test_raw_httpx2_transport_error_is_retryable() -> None:
  error = httpx2.ReadError(
    "connection closed during stream",
    request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"),
  )

  assert AnthropicProvider().is_retryable_error(error) is True


@pytest.mark.parametrize(
  "custom_headers",
  [
    None,
    "X-Api-Key: ambient-header-key",
    "Authorization: Bearer ambient",
    (
      "X-Api-Key: ambient-header-key\nx-api-key: ambient-lowercase-key\n"
      "Authorization: Bearer ambient\nauthorization: Bearer ambient-lowercase"
    ),
  ],
  ids=["credential-env", "api-key-header", "oauth-header", "mixed-case-headers"],
)
def test_create_client_isolates_bound_credentials_and_routes_concurrently(
  monkeypatch: pytest.MonkeyPatch,
  custom_headers: str | None,
) -> None:
  pytest.importorskip("anthropic")
  barrier = threading.Barrier(2)
  ambient = {
    "ANTHROPIC_API_KEY": "ambient-api-key",
    "ANTHROPIC_AUTH_TOKEN": "ambient-oauth-token",
    "ANTHROPIC_BASE_URL": "https://ambient.invalid/v1",
  }
  if custom_headers is not None:
    ambient["ANTHROPIC_CUSTOM_HEADERS"] = custom_headers
  else:
    monkeypatch.delenv("ANTHROPIC_CUSTOM_HEADERS", raising=False)
  for key, value in ambient.items():
    monkeypatch.setenv(key, value)
  provider = AnthropicProvider()

  def capture_bound_request(config: dict[str, str]) -> httpx2.Request:
    requests = []

    def handle_request(request: httpx2.Request) -> httpx2.Response:
      requests.append(request)
      assert {key: os.environ.get(key) for key in ambient} == ambient
      return httpx2.Response(200, json={
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
      })

    async def send_request() -> None:
      barrier.wait(timeout=5.0)
      client = provider.create_client(config)
      try:
        async with httpx2.AsyncClient(
          transport=httpx2.MockTransport(handle_request),
        ) as http_client:
          async with client.with_options(http_client=http_client) as local_client:
            await local_client.messages.create(
              model="claude-sonnet-4-6",
              max_tokens=16,
              messages=[{"role": "user", "content": "hello"}],
            )
      finally:
        await provider.close_client(client)

    asyncio.run(send_request())
    return requests[0]

  with ThreadPoolExecutor(max_workers=2) as executor:
    api_future = executor.submit(
      capture_bound_request,
      {"auth_mode": "api", "api_key": "bound-api-key"},
    )
    oauth_future = executor.submit(
      capture_bound_request,
      {
        "auth_mode": "oauth",
        "auth_token": "bound-oauth-token",
        "baseURL": "https://bound.anthropic.example",
      },
    )
    api_request = api_future.result(timeout=5.0)
    oauth_request = oauth_future.result(timeout=5.0)

  assert {key: os.environ.get(key) for key in ambient} == ambient
  assert api_request.url == httpx2.URL("https://api.anthropic.com/v1/messages")
  assert api_request.headers.get_list("x-api-key") == ["bound-api-key"]
  assert "authorization" not in api_request.headers
  assert oauth_request.url == httpx2.URL("https://bound.anthropic.example/v1/messages")
  assert oauth_request.headers.get_list("authorization") == ["Bearer bound-oauth-token"]
  assert "x-api-key" not in oauth_request.headers


@pytest.mark.parametrize(
  ("config", "custom_headers", "header", "expected", "omitted_header"),
  [
    (
      {"auth_mode": "oauth", "auth_token": "bound-oauth-token"},
      "Authorization:",
      "authorization",
      "Bearer bound-oauth-token",
      "x-api-key",
    ),
    (
      {"auth_mode": "api", "api_key": "bound-api-key"},
      "Authorization:\nX-Api-Key:",
      "x-api-key",
      "bound-api-key",
      "authorization",
    ),
  ],
  ids=["oauth", "api-key"],
)
def test_create_client_ignores_empty_ambient_auth_headers(
  monkeypatch: pytest.MonkeyPatch,
  config: dict[str, str],
  custom_headers: str,
  header: str,
  expected: str,
  omitted_header: str,
) -> None:
  pytest.importorskip("anthropic")
  monkeypatch.setenv(
    "ANTHROPIC_CUSTOM_HEADERS",
    f"{custom_headers}\nX-Ambient-Trace: preserved",
  )
  provider = AnthropicProvider()

  def handle_request(request: httpx2.Request) -> httpx2.Response:
    assert request.headers.get_list(header) == [expected]
    assert omitted_header not in request.headers
    assert request.headers["x-ambient-trace"] == "preserved"
    return httpx2.Response(200, json={
      "id": "msg_123",
      "type": "message",
      "role": "assistant",
      "model": "claude-sonnet-4-6",
      "content": [{"type": "text", "text": "hello"}],
      "stop_reason": "end_turn",
      "stop_sequence": None,
      "usage": {"input_tokens": 1, "output_tokens": 1},
    })

  async def send_request() -> None:
    async with provider.create_client(config) as client:
      async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handle_request),
      ) as http_client:
        async with client.with_options(http_client=http_client) as local_client:
          result = await local_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
          )
          assert result.content[0].text == "hello"

  asyncio.run(send_request())


@pytest.mark.parametrize(
  ("config", "header", "expected", "omitted_header"),
  [
    (
      {"auth_mode": "api", "api_key": "bound-api-key-debug-secret"},
      "x-api-key",
      "bound-api-key-debug-secret",
      "authorization",
    ),
    (
      {"auth_mode": "oauth", "auth_token": "bound-oauth-debug-secret"},
      "authorization",
      "Bearer bound-oauth-debug-secret",
      "x-api-key",
    ),
  ],
  ids=["api-key", "oauth"],
)
def test_create_client_keeps_bound_credential_out_of_sdk_debug_logs(
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
  config: dict[str, str],
  header: str,
  expected: str,
  omitted_header: str,
) -> None:
  monkeypatch.setenv("ANTHROPIC_LOG", "debug")
  pytest.importorskip("anthropic")
  caplog.set_level(logging.DEBUG, logger="anthropic")
  monkeypatch.setenv(
    "ANTHROPIC_CUSTOM_HEADERS",
    "X-Api-Key: ambient-header-key\nx-api-key: ambient-lowercase-key\n"
    "Authorization: Bearer ambient\nauthorization: Bearer ambient-lowercase",
  )
  provider = AnthropicProvider()

  def handle_request(request: httpx2.Request) -> httpx2.Response:
    assert request.headers.get_list(header) == [expected]
    assert omitted_header not in request.headers
    return httpx2.Response(200, json={
      "id": "msg_123",
      "type": "message",
      "role": "assistant",
      "model": "claude-sonnet-4-6",
      "content": [{"type": "text", "text": "hello"}],
      "stop_reason": "end_turn",
      "stop_sequence": None,
      "usage": {"input_tokens": 1, "output_tokens": 1},
    })

  async def send_request() -> None:
    async with provider.create_client(config) as client:
      async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handle_request),
      ) as http_client:
        async with client.with_options(http_client=http_client) as local_client:
          await local_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
          )

  asyncio.run(send_request())
  assert any("Request options:" in record.getMessage() for record in caplog.records)
  credential = config.get("api_key") or config["auth_token"]
  assert all(credential not in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize(
  "config",
  [
    {"auth_mode": "api", "api_key": "   "},
    {"auth_mode": "oauth", "auth_token": "   "},
  ],
)
def test_provider_rejects_blank_bound_credential_despite_ambient_values(
  monkeypatch: pytest.MonkeyPatch,
  config: dict[str, str],
) -> None:
  monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-api-key")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-oauth-token")
  provider = AnthropicProvider()

  assert provider.has_active_credential(config) is False
  with pytest.raises(RuntimeError, match="No Anthropic .* credential configured"):
    provider.create_client(config)


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
    "_STRUCTURED_OUTPUTS_BETA_SLUG",
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
  default_info = ModelInfo(id="stub", provider="test")
  assert default_info.thinking_mode == "none"
  assert default_info.supports_native_compaction is False
  info = ModelInfo(id="stub", provider="test", supports_thinking=True)

  assert info.thinking_mode == "adaptive"
  assert info.supports_thinking is True


@pytest.mark.parametrize(
  ("model", "expected"),
  [
    ("claude-fable-5", True),
    ("claude-mythos-5", True),
    ("claude-opus-4-8", True),
    ("claude-opus-4-7", True),
    ("claude-sonnet-5", True),
    ("claude-sonnet-4-6", True),
    ("claude-sonnet-4-6-20260615", True),
    ("claude-opus-4-6", True),
    ("claude-sonnet-4-5", False),
    ("claude-opus-4-5", False),
    ("claude-haiku-4-5", False),
    ("claude-3.7-sonnet-20250219", False),
    ("claude-sonnet-4-60", False),
    ("claude-sonnet-4-6x", False),
  ],
)
def test_native_compaction_capability_is_model_specific_and_fail_closed(
  model: str,
  expected: bool,
) -> None:
  info = AnthropicProvider().get_model_info(model)

  assert info.supports_native_compaction is expected


@pytest.mark.parametrize(
  ("model", "expects_native_compaction"),
  [
    ("claude-sonnet-4-6", True),
    ("claude-haiku-4-5", False),
    ("claude-sonnet-4-60", False),
    ("claude-sonnet-4-6x", False),
  ],
)
def test_compaction_request_is_emitted_only_for_supported_models(
  model: str,
  expects_native_compaction: bool,
) -> None:
  params = AnthropicProvider().build_request_params(
    model=model,
    messages=[],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
    compaction_trigger=160_000,
    compaction_instructions="Preserve durable state.",
  )

  assert ("context_management" in params) is expects_native_compaction
  if expects_native_compaction:
    assert params["context_management"] == {
      "edits": [{
        "type": "compact_20260112",
        "trigger": {"type": "input_tokens", "value": 160_000},
        "pause_after_compaction": False,
        "instructions": "Preserve durable state.",
      }],
    }


def test_haiku_normalizes_portable_compaction_anchor_to_text() -> None:
  provider = AnthropicProvider()
  messages = [
    {"role": "user", "content": "superseded history"},
    {
      "role": "assistant",
      "content": [{"type": "compaction", "content": "portable summary"}],
      "provider": "anthropic",
      "model": "claude-haiku-4-5",
      "stop_reason": "compaction",
    },
    {"role": "user", "content": "continue"},
  ]

  normalized = provider.normalize_messages(
    messages,
    provider.get_model_info("claude-haiku-4-5"),
  )

  assert normalized[0]["content"][0]["type"] == "text"
  assert "portable summary" in normalized[0]["content"][0]["text"]
  assert "superseded history" not in str(normalized)


def test_fable_model_info_uses_bundled_rates_and_adaptive_thinking() -> None:
  provider = AnthropicProvider()

  info = provider.get_model_info("claude-fable-5")

  assert info.context_window == 1_000_000
  assert info.max_output_tokens == 128_000
  assert info.input_cost_per_mtok == 10.0
  assert info.output_cost_per_mtok == 50.0
  assert info.cache_read_cost_per_mtok == 1.0
  assert info.cache_write_cost_per_mtok == 12.5
  assert info.supports_thinking is True
  assert info.thinking_mode == "adaptive"


def test_fable_revision_uses_its_exact_cached_token_price() -> None:
  provider = AnthropicProvider()

  assert provider.estimate_cost("claude-fable-5-1", 0, 0, cache_read_tokens=1_000_000).total == 0.25
  assert provider.estimate_cost("claude-fable-5", 0, 0, cache_read_tokens=1_000_000).total == 1.0


def test_anthropic_prefers_specific_model_metadata_over_family(monkeypatch) -> None:
  from dataclasses import replace

  family = AnthropicProvider().get_model_info("claude-fable-5")
  specific = replace(family, id="claude-fable-5-1", supports_native_compaction=False)
  monkeypatch.setattr(anthropic_helpers, "_MODEL_INFO_BY_TAG", [
    (("claude-fable-5",), family),
    (("claude-fable-5-1",), specific),
  ])

  provider = AnthropicProvider()
  assert provider.get_model_info("claude-fable-5").supports_native_compaction is True
  assert provider.get_model_info("claude-fable-5-1").supports_native_compaction is False
  assert provider.get_model_info("claude-fable-5-1-20260911").supports_native_compaction is False


def test_opus48_model_info_uses_bundled_rates_and_adaptive_thinking() -> None:
  provider = AnthropicProvider()

  info = provider.get_model_info("claude-opus-4-8")

  assert info.context_window == 1_000_000
  assert info.max_output_tokens == 128_000
  assert info.input_cost_per_mtok == 5.0
  assert info.output_cost_per_mtok == 25.0
  assert info.cache_read_cost_per_mtok == 0.5
  assert info.cache_write_cost_per_mtok == 6.25
  assert info.supports_thinking is True
  assert info.thinking_mode == "adaptive"


def test_opus5_model_info_uses_bundled_rates_and_adaptive_thinking() -> None:
  provider = AnthropicProvider()

  info = provider.get_model_info("claude-opus-5")

  assert info.context_window == 1_000_000
  assert info.max_output_tokens == 128_000
  assert info.input_cost_per_mtok == 5.0
  assert info.output_cost_per_mtok == 25.0
  assert info.cache_read_cost_per_mtok == 0.5
  assert info.cache_write_cost_per_mtok == 6.25
  assert info.supports_thinking is True
  assert info.thinking_mode == "adaptive"


@pytest.mark.parametrize(
  ("requested", "expected_thinking", "expected_output_config"),
  [
    (ThinkingLevel.NONE, {"type": "disabled"}, None),
    (ThinkingLevel.XHIGH, {"type": "adaptive", "display": "summarized"}, {"effort": "xhigh"}),
    (ThinkingLevel.MAX, {"type": "adaptive", "display": "summarized"}, {"effort": "max"}),
  ],
)
def test_opus5_resolved_effort_emits_complete_payload_pair(
  requested: ThinkingLevel,
  expected_thinking: dict[str, str],
  expected_output_config: dict[str, str] | None,
) -> None:
  provider = AnthropicProvider()
  info = provider.get_model_info("claude-opus-5")
  resolution = provider.resolve_effort(
    requested=requested,
    model=info.id,
    model_info=info,
    max_tokens=4096,
  )

  params = provider.build_request_params(
    model=info.id,
    messages=[],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
    thinking_level=requested,
    effort_resolution=resolution,
  )

  assert params["thinking"] == expected_thinking
  if expected_output_config is None:
    assert "output_config" not in params
  else:
    assert params["output_config"] == expected_output_config


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
    ("claude-fable-5", {"type": "adaptive", "display": "summarized"}),
    ("claude-opus-4-8", {"type": "adaptive", "display": "summarized"}),
    ("claude-opus-4-7", {"type": "adaptive", "display": "summarized"}),
    ("claude-sonnet-4-6", {"type": "adaptive", "display": "summarized"}),
    ("claude-opus-4-6", {"type": "adaptive", "display": "summarized"}),
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




def test_registry_unadmitted_claude_model_is_rejected() -> None:
  provider = AnthropicProvider()

  with pytest.raises(ValueError, match="product model registry does not admit"):
    provider.get_model_info("claude-zenith-9")
  with pytest.raises(ValueError, match="product model registry does not admit"):
    AnthropicProvider.thinking_param("claude-zenith-9", 4096)


def test_thinking_param_defers_foreign_model_ids_to_registry_owner() -> None:
  # No prefix pre-check: a non-claude id routed here is decided by the
  # product model registry (raise), not silently degraded to no-thinking.
  with pytest.raises(ValueError, match="product model registry does not admit"):
    AnthropicProvider.thinking_param("gpt-5.2", 4096)


def _registry_entry(**overrides):
  from agent_gateway.model_registry import ModelRegistryEntry

  fields = {
    "key": "anthropic.claude-nova-6",
    "label": "Nova 6",
    "provider": "anthropic",
    "upstream_model": "claude-nova-6",
    "adapter": "anthropic.messages",
    "protocol_profile": "messages.adaptive",
    "route": "anthropic.public",
    "lifecycle": "active",
    "capabilities": {"session.driver": "user_selectable"},
    "supported_efforts": frozenset({"low", "medium", "high", "xhigh", "max"}),
    "default_effort": "high",
    "features": frozenset({"tools", "streaming"}),
    "reported_identities": frozenset({"claude-nova-6"}),
  }
  fields.update(overrides)
  return ModelRegistryEntry(**fields)


def test_registry_admitted_claude_model_without_row_derives_from_registry(
  monkeypatch,
) -> None:
  # Config-only model addition (plan §8): a registry-admitted model is served
  # before the capability table gains a row, with thinking and effort facts
  # derived from the registry owner — no generic substitution that would drop
  # xhigh/max efforts or misreport disable semantics.
  from agent_gateway.model_registry import ProductModelRegistry
  import agent_gateway.providers.base as provider_base

  entry = _registry_entry()
  monkeypatch.setattr(
    provider_base,
    "INITIAL_MODEL_REGISTRY",
    ProductModelRegistry(
      schema="product-model-registry/v1",
      revision="test",
      models={entry.key: entry},
    ),
  )
  provider = AnthropicProvider()

  info = provider.get_model_info("claude-nova-6")

  assert info.supports_thinking is True
  assert info.thinking_mode == "adaptive"
  compat = info.compat or {}
  assert compat["effort_values"] == ("low", "medium", "high", "xhigh", "max")
  assert compat["thinking_default_effort"] == "high"
  assert compat["thinking_default_when_omitted"] == "on"
  # No "none" effort admitted => thinking cannot be explicitly disabled.
  assert compat["thinking_disable"] == "unsupported"
  assert AnthropicProvider.thinking_param("claude-nova-6", 4096) == {"type": "adaptive", "display": "summarized"}


@pytest.mark.parametrize(
  ("model", "expected_disable", "expected_omitted"),
  [
    ("claude-fable-5", "unsupported", "on"),
    ("claude-opus-5", "disabled", "on"),
    ("claude-sonnet-5", "disabled", "on"),
  ],
)
def test_registry_derivation_reproduces_cataloged_adaptive_compat(
  model: str,
  expected_disable: str,
  expected_omitted: str,
) -> None:
  # Oracle for the derivation rules: for every adaptive model that has BOTH a
  # registry entry and a catalog row, deriving from the registry entry must
  # reproduce the catalog row's thinking compat exactly.
  from agent_gateway.model_registry import INITIAL_MODEL_REGISTRY
  from agent_gateway.providers.anthropic_helpers import (
    _model_info_from_registry_entry,
  )

  entry = next(
    e for e in INITIAL_MODEL_REGISTRY.models.values()
    if e.provider == "anthropic" and e.upstream_model == model
  )
  catalog = AnthropicProvider().get_model_info(model)

  derived = _model_info_from_registry_entry(model, entry)
  assert derived.compat is not None

  assert derived.compat == catalog.compat
  assert derived.compat["thinking_disable"] == expected_disable
  assert derived.compat["thinking_default_when_omitted"] == expected_omitted
  assert derived.thinking_mode == catalog.thinking_mode


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

  assert params["thinking"] == {"type": "adaptive", "display": "summarized"}
  for key in ("temperature", "top_p", "top_k"):
    assert key not in params


def test_strict_tool_schema_is_transformed_without_widening_gateway_contract() -> None:
  pytest.importorskip("anthropic")
  provider = AnthropicProvider()
  gateway_schema = {
    "type": "object",
    "properties": {
      "summary": {"type": "string", "minLength": 1, "maxLength": 20},
      "findings": {
        "type": "array",
        "maxItems": 2,
        "items": {
          "oneOf": [
            {
              "type": "object",
              "properties": {
                "kind": {"type": "string", "const": "finding"},
                "claim": {"type": "string"},
              },
              "required": ["claim"],
            }
          ],
        },
      },
    },
    "required": ["summary"],
  }
  tools = [{
    "name": "structured_write",
    "strict": True,
    "eager_input_streaming": False,
    "input_schema": gateway_schema,
  }]

  params = provider.build_request_params(
    model="claude-opus-4-8",
    messages=[],
    system_prompt=None,
    tools=tools,
    max_tokens=4096,
    thinking_level=ThinkingLevel.HIGH,
  )

  strict_tool = params["tools"][0]
  strict_schema = strict_tool["input_schema"]
  assert strict_tool["strict"] is True
  assert strict_tool["eager_input_streaming"] is False
  assert strict_schema["additionalProperties"] is False
  assert strict_schema["properties"]["findings"]["type"] == "array"
  assert "anyOf" in strict_schema["properties"]["findings"]["items"]
  assert "oneOf" not in strict_schema["properties"]["findings"]["items"]
  assert "maxLength" not in strict_schema["properties"]["summary"]
  assert "maxLength: 20" in strict_schema["properties"]["summary"]["description"]
  assert gateway_schema["properties"]["summary"]["maxLength"] == 20
  assert "oneOf" in gateway_schema["properties"]["findings"]["items"]


def test_non_strict_tools_skip_anthropic_schema_transformation() -> None:
  provider = AnthropicProvider()
  tools = [{
    "name": "lookup",
    "input_schema": {
      "type": "object",
      "properties": {"query": {"type": "string", "maxLength": 20}},
    },
  }]

  params = provider.build_request_params(
    model="claude-opus-4-8",
    messages=[],
    system_prompt=None,
    tools=tools,
    max_tokens=4096,
    thinking_level=ThinkingLevel.HIGH,
  )

  assert params["tools"] is tools
  assert params["tools"][0]["input_schema"]["properties"]["query"]["maxLength"] == 20


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
    self.beta = SimpleNamespace(messages=_FakeStreamingMessages(events))


async def _drain_stream(provider: AnthropicProvider, client: object, params: dict[str, object]) -> None:
  async for _ in provider.stream(client, params):
    pass


async def _collect_stream_types(provider: AnthropicProvider, client: object, params: dict[str, object]) -> list[str]:
  return [event.type async for event in provider.stream(client, params)]


async def _collect_stream_events(
  provider: AnthropicProvider,
  client: object,
  params: dict[str, object],
) -> list[StreamEvent]:
  return [event async for event in provider.stream(client, params)]


def test_tool_use_mapper_emits_raw_input_without_redactor_dependency(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import agent_gateway.runner_tool_audit as runner_tool_audit

  monkeypatch.setattr(
    runner_tool_audit,
    "redact_tool_input_for_event",
    lambda *_args, **_kwargs: (_ for _ in ()).throw(
      AssertionError("provider mapper cannot own history redaction")
    ),
  )
  provider_block = SimpleNamespace(
    type="tool_use",
    id="call-1",
    name="registered_write",
    input={},
  )
  client = _FakeStreamingClient([
    SimpleNamespace(
      type="content_block_start",
      content_block=provider_block,
    ),
    SimpleNamespace(
      type="content_block_delta",
      delta=SimpleNamespace(
        type="input_json_delta",
        partial_json='{\"credential\":\"raw-secret\"}',
      ),
    ),
    SimpleNamespace(type="content_block_stop"),
  ])

  events = asyncio.run(_collect_stream_events(
    AnthropicProvider(),
    client,
    {"model": "claude-sonnet-4-6", "messages": []},
  ))
  tool_event = next(event for event in events if event.type == "tool_use_end")

  assert tool_event.tool_input == {"credential": "raw-secret"}
  assert tool_event.raw_block["input"] == {"credential": "raw-secret"}
  assert provider_block.input == {}


@pytest.mark.parametrize(
  ("model", "auth_mode", "compaction_trigger", "expected_betas"),
  [
    (
      "claude-haiku-4-5",
      "api",
      None,
      [anthropic_provider_module._STRUCTURED_OUTPUTS_BETA_SLUG],
    ),
    (
      "claude-haiku-4-5",
      "oauth",
      None,
      [
        *anthropic_provider_module._OAUTH_BETA_SLUGS,
        anthropic_provider_module._STRUCTURED_OUTPUTS_BETA_SLUG,
      ],
    ),
    (
      "claude-opus-4-8",
      "oauth",
      160_000,
      [
        *anthropic_provider_module._OAUTH_BETA_SLUGS,
        anthropic_provider_module._STRUCTURED_OUTPUTS_BETA_SLUG,
        anthropic_provider_module._COMPACTION_BETA_SLUG,
      ],
    ),
  ],
)
def test_stream_routes_strict_tools_through_structured_outputs_beta(
  model: str,
  auth_mode: str,
  compaction_trigger: int | None,
  expected_betas: list[str],
) -> None:
  provider = AnthropicProvider()
  strict_tool = {
    "name": "structured_write",
    "strict": True,
    "eager_input_streaming": False,
    "input_schema": {
      "type": "object",
      "properties": {"value": {"type": "string"}},
      "required": ["value"],
    },
  }
  params = provider.build_request_params(
    model=model,
    messages=[],
    system_prompt=None,
    tools=[strict_tool],
    max_tokens=4096,
    thinking_level=ThinkingLevel.NONE,
    auth_mode=auth_mode,
    compaction_trigger=compaction_trigger,
  )
  client = _FakeStreamingClient([])

  asyncio.run(_drain_stream(provider, client, params))

  assert client.messages.kwargs is None
  assert client.beta.messages.kwargs is not None
  assert client.beta.messages.kwargs["betas"] == expected_betas
  assert "_provider_auth_mode" not in client.beta.messages.kwargs
  assert client.beta.messages.kwargs["tools"][0]["strict"] is True


def test_anthropic_rejection_detail_redacts_sensitive_body_fallback() -> None:
  raw_key = "sk-ant-api03-DETAILKEY123"
  detail = _format_anthropic_rejection_detail(
    _make_anthropic_api_status_error(
      400,
      "invalid request",
      body={
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
  error = _make_anthropic_api_status_error(
    400,
    "invalid request",
    body={
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
    "thinking": {"type": "adaptive", "display": "summarized"},
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


@pytest.mark.parametrize(
  "error_body",
  [
    pytest.param(
      {
        "type": "error",
        "error": {
          "type": "permission_error",
          "message": "OAuth authentication is currently not allowed for this organization.",
          "details": {"error_code": "oauth_not_allowed_for_organization"},
        },
      },
      id="observed-body",
    ),
    pytest.param(
      {"type": "error", "error": {"type": "permission_error", "message": "oauth_not_allowed_for_organization"}},
      id="error-code-only",
    ),
  ],
)
def test_org_refusing_oauth_parks_that_member_and_rotates_to_the_next(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  error_body: dict[str, object],
) -> None:
  """A 403 org OAuth refusal is per credential: pool siblings in other orgs answer."""
  refused, serving = "sk-ant-oat01-refused-org", "sk-ant-oat01-serving"
  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", refused)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([refused, serving]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "absent-oauth.json"))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  provider = AnthropicProvider()
  client = _FakeClient(_make_anthropic_api_status_error(403, f"Error code: 403 - {error_body}", body=error_body))
  params = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "hello"}],
    "_provider_auth_mode": "oauth",
  }
  with pytest.raises(RuntimeError) as exc_info:
    asyncio.run(_drain_stream(provider, client, params))

  failure = provider.classify_credential_failure(exc_info.value)

  assert failure is not None and failure.kind == "auth"
  assert provider.next_credential({"auth_mode": "oauth", "auth_token": refused}, failure) == {
    "auth_token": serving,
  }
  assert pool.blocked_until(refused) > 0


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


def test_stream_summarized_thinking_streams_deltas_and_stores_summary_beside_signature() -> None:
  provider = AnthropicProvider()
  client = _FakeStreamingClient(
    [
      SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking")),
      SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="Weigh ")),
      SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="the guidance.")),
      SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="signature_delta", signature="sig")),
      SimpleNamespace(type="content_block_stop"),
    ]
  )

  events = asyncio.run(_collect_stream_events(provider, client, {"model": "claude-opus-5", "messages": []}))

  assert [event.thinking_text for event in events if event.type == "thinking_delta"] == ["Weigh ", "the guidance."]
  thinking_end = next(event for event in events if event.type == "thinking_end")
  assert thinking_end.raw_block == {"type": "thinking", "thinking": "Weigh the guidance.", "signature": "sig"}


def test_stream_sums_compaction_usage_iterations() -> None:
  provider = AnthropicProvider()
  client = _FakeStreamingClient(
    [
      SimpleNamespace(
        type="message_start",
        message=SimpleNamespace(
          usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=0,
            cache_creation_input_tokens=1,
            cache_read_input_tokens=2,
          )
        ),
      ),
      SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="end_turn"),
        usage=SimpleNamespace(
          input_tokens=100,
          output_tokens=10,
          cache_creation_input_tokens=1,
          cache_read_input_tokens=2,
          iterations=[
            SimpleNamespace(
              input_tokens=100,
              output_tokens=10,
              cache_creation_input_tokens=1,
              cache_read_input_tokens=2,
            ),
            SimpleNamespace(
              input_tokens=50,
              output_tokens=5,
              cache_creation_input_tokens=3,
              cache_read_input_tokens=4,
            ),
          ],
        ),
      ),
    ]
  )

  events = asyncio.run(_collect_stream_events(provider, client, {"model": "claude-sonnet-4-6", "messages": []}))

  message_start = events[0]
  usage_update = events[1]
  assert message_start.type == "message_start"
  assert message_start.input_tokens == 100
  assert message_start.cache_creation_tokens == 1
  assert message_start.cache_read_tokens == 2
  assert usage_update.type == "usage_update"
  assert usage_update.input_tokens == 50
  assert usage_update.output_tokens == 15
  assert usage_update.cache_creation_tokens == 3
  assert usage_update.cache_read_tokens == 4


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


def test_declared_adapter_support_matches_messages_implementation() -> None:
  from agent_gateway.model_registry import INITIAL_MODEL_REGISTRY

  declaration = AnthropicProvider.adapter_route_support()

  assert declaration is not None
  assert declaration.adapter == "anthropic.messages"
  assert declaration.provider == "anthropic"
  assert declaration.protocol_profiles == frozenset(
    {"messages.standard", "messages.adaptive"}
  )
  assert declaration.routes == frozenset({"anthropic.public"})

  # Admits the packaged public-Messages entries (adaptive and standard).
  for key in (
    "anthropic.claude-opus-5",
    "anthropic.claude-sonnet-5",
    "anthropic.claude-haiku-4-5",
  ):
    assert declaration.supports(INITIAL_MODEL_REGISTRY.require(key)), key

  # The Risk-local SDK adapter's execution identities are a different
  # implementation in a different serving process — never claimed here.
  for key in (
    "anthropic.claude-sonnet-5-sdk",
    "anthropic.claude-haiku-4-5-sdk",
    "anthropic.claude-opus-5-5-oauth",
    "anthropic.claude-sonnet-4-20250514-sdk",
  ):
    assert not declaration.supports(INITIAL_MODEL_REGISTRY.require(key)), key

  # messages.adaptive is a real protocol fact of this implementation: the
  # adaptive-thinking models resolve as thinking-capable.
  assert AnthropicProvider().get_model_info("claude-opus-5").supports_thinking
