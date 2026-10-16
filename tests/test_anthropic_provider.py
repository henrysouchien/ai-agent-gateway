# ruff: noqa: E402

import asyncio
import importlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import httpx
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

from agent_gateway import AnthropicProvider, ModelInfo
from model_authority.thinking import ThinkingLevel
import agent_gateway.providers.anthropic as anthropic_provider_module
from agent_gateway.providers.anthropic import (
  CREDENTIAL_MODEL_FAMILY_FIELD,
  _format_anthropic_rejection_detail,
)
from agent_gateway.providers import anthropic_oauth, anthropic_quota
from agent_gateway.providers.anthropic_oauth import (
  ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES,
  ANTHROPIC_DEPLOYMENT_API_KEY_FIELD,
  CREDENTIAL_AFFINITY_KEY_FIELD,
  CREDENTIAL_PREPARED_FIELD,
  AnthropicRequestBinding,
  anthropic_credential_blocked_until,
  resolve_anthropic_credentials,
)
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
  return AnthropicProvider().get_model_info("claude-sonnet-5")


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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
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
    model="claude-sonnet-5",
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
  headers: dict[str, str] | None = None,
):
  anthropic = pytest.importorskip("anthropic")
  if body is None:
    body = {"error": {"message": message}}
  request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
  response = httpx2.Response(
    status_code,
    request=request,
    headers={"request-id": "req_123", **(headers or {})},
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
              model="claude-sonnet-5",
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
        "model": "claude-sonnet-5",
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
              model="claude-sonnet-5",
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


def test_create_client_serves_each_users_bare_bound_token_across_turns(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  pytest.importorskip("anthropic")
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  for name in (
    "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_AUTH_TOKENS", "CLAUDE_CODE_OAUTH_TOKEN",
    "ANTHROPIC_AUTH_STORE_PATH",
  ):
    monkeypatch.delenv(name, raising=False)
  provider = AnthropicProvider()

  # User 1 turn 1, user 2 turn 1, user 1 turn 2: neither token retires the other.
  for token in ("fixture-user-one-token", "fixture-user-two-token", "fixture-user-one-token"):
    client = provider.create_client({"auth_mode": "oauth", "auth_token": token})
    try:
      assert client.auth_token == token
    finally:
      asyncio.run(provider.close_client(client))


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
      "model": "claude-sonnet-5",
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
            model="claude-sonnet-5",
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
      "model": "claude-sonnet-5",
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
            model="claude-sonnet-5",
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
    ("claude-fable-5-1", True),
    ("claude-opus-5-5", True),
    ("claude-opus-5", True),
    ("claude-sonnet-5", True),
    ("claude-haiku-4-5", False),
    ("claude-haiku-4-5-20251001", False),
  ],
)
def test_native_compaction_capability_is_model_specific(
  model: str,
  expected: bool,
) -> None:
  info = AnthropicProvider().get_model_info(model)

  assert info.supports_native_compaction is expected


@pytest.mark.parametrize(
  ("model", "expects_native_compaction"),
  [
    ("claude-sonnet-5", True),
    ("claude-haiku-4-5", False),
    ("claude-haiku-4-5-20251001", False),
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


def _sonnet_5_5_registry(monkeypatch) -> None:
  # Claude Sonnet 5.5 as model-authority/2.1 data, from the 2026-09-29 live
  # probes: thinking disabled is a 400 and between_tools is the lowest
  # setting; forced tool_choice is a 400 under every thinking setting.
  from model_authority.registry import ModelRegistryEntry, ProductModelRegistry
  from model_authority.schema import SCHEMA, AnthropicMessagesCompat, AnthropicThinking
  import agent_gateway.providers.base as provider_base

  entry = ModelRegistryEntry(
    key="anthropic.claude-sonnet-5-5",
    label="Sonnet 5.5",
    provider="anthropic",
    upstream_model="claude-sonnet-5-5",
    adapter="anthropic.messages",
    protocol_profile="messages.adaptive",
    route="anthropic.public",
    lifecycle="active",
    capabilities={"session.driver": "user_selectable"},
    supported_efforts=frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
    default_effort="high",
    features=frozenset({"tools", "streaming"}),
    reported_identities=frozenset({"claude-sonnet-5-5"}),
    compat=AnthropicMessagesCompat(
      accepts_temperature=False,
      thinking=AnthropicThinking(
        default_when_omitted="on",
        can_disable=False,
        effort_control="output_config",
        effort_when_omitted="high",
        off_value="between_tools",
      ),
      forced_tool_choice_requires_thinking_off=True,
      tool_choice_forcing="never",
      native_compaction=True,
      max_output_tokens=128_000,
      context_window=1_000_000,
    ),
  )
  monkeypatch.setattr(
    provider_base,
    "INITIAL_MODEL_REGISTRY",
    ProductModelRegistry(schema=SCHEMA, revision="test", models={entry.key: entry}),
  )


@pytest.mark.parametrize(
  ("model", "requested", "expected_thinking", "expected_output_config", "effective"),
  [
    # Sonnet 5 accepts thinking disabled (probe: 200).
    ("claude-sonnet-5", ThinkingLevel.NONE, {"type": "disabled"}, None, ThinkingLevel.NONE),
    # Opus 5.5 cannot be turned off: none runs its default (medium).
    ("claude-opus-5-5", ThinkingLevel.NONE, None, None, ThinkingLevel.MEDIUM),
    # Sonnet 5.5: between_tools alone (with display or effort xhigh/max: 400).
    ("claude-sonnet-5-5", ThinkingLevel.NONE, {"type": "between_tools"}, None, ThinkingLevel.NONE),
    (
      "claude-sonnet-5-5",
      ThinkingLevel.XHIGH,
      {"type": "adaptive", "display": "summarized"},
      {"effort": "xhigh"},
      ThinkingLevel.XHIGH,
    ),
  ],
)
def test_effort_payload_follows_the_entry_off_value(
  monkeypatch,
  model: str,
  requested: ThinkingLevel,
  expected_thinking: dict[str, str] | None,
  expected_output_config: dict[str, str] | None,
  effective: ThinkingLevel,
) -> None:
  if model == "claude-sonnet-5-5":
    _sonnet_5_5_registry(monkeypatch)
  provider = AnthropicProvider()
  info = provider.get_model_info(model)
  resolution = provider.resolve_effort(requested=requested, model=model, model_info=info, max_tokens=4096)

  params = provider.build_request_params(
    model=model,
    messages=[],
    system_prompt=None,
    tools=[{"name": "submit", "description": "d", "input_schema": {"type": "object"}}],
    max_tokens=4096,
    thinking_level=requested,
    effort_resolution=resolution,
  )

  assert resolution.effective == effective
  assert params.get("thinking") == expected_thinking
  assert params.get("output_config") == expected_output_config
  assert params["tool_choice"] == (
    {"type": "tool", "name": "submit"} if model == "claude-sonnet-5" else {"type": "auto"}
  )


@pytest.mark.parametrize(
  ("forcing", "requested", "tool_count", "expected"),
  [
    ("any", ThinkingLevel.HIGH, 1, {"type": "tool", "name": "submit"}),
    ("thinking_off_only", ThinkingLevel.NONE, 1, {"type": "tool", "name": "submit"}),
    ("thinking_off_only", ThinkingLevel.HIGH, 1, {"type": "auto"}),
    ("never", ThinkingLevel.NONE, 1, {"type": "auto"}),
    ("any", ThinkingLevel.NONE, 2, {"type": "auto"}),
  ],
)
def test_tool_forcing_follows_compat_and_effective_thinking(
  monkeypatch, forcing, requested, tool_count, expected,
) -> None:
  from dataclasses import replace
  from model_authority.registry import ProductModelRegistry
  from model_authority.schema import SCHEMA, AnthropicMessagesCompat
  import agent_gateway.providers.base as provider_base

  entry = _registry_entry("output_config")
  data = entry.compat.model_dump()
  data["thinking"].update(default_when_omitted="off", can_disable=True, off_value="disabled")
  data.update(tool_choice_forcing=forcing, forced_tool_choice_requires_thinking_off=forcing != "any")
  entry = replace(entry, compat=AnthropicMessagesCompat.model_validate(data))
  monkeypatch.setattr(
    provider_base, "INITIAL_MODEL_REGISTRY",
    ProductModelRegistry(schema=SCHEMA, revision="test", models={entry.key: entry}),
  )
  tools = [
    {"name": name, "description": "d", "input_schema": {"type": "object"}}
    for name in ("submit", "other")[:tool_count]
  ]
  params = AnthropicProvider().build_request_params(
    model=entry.upstream_model, messages=[], system_prompt=None, tools=tools,
    max_tokens=4096, thinking_level=requested,
  )
  assert params["tool_choice"] == expected
  if requested == ThinkingLevel.NONE:
    # Explicit off applies even to a default-off entry.
    assert params["thinking"] == {"type": "disabled"}
    assert "output_config" not in params


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
    ("claude-fable-5-1", {"type": "adaptive", "display": "summarized"}),
    ("claude-opus-5-5", {"type": "adaptive", "display": "summarized"}),
    ("claude-opus-5", {"type": "adaptive", "display": "summarized"}),
    ("claude-sonnet-5", {"type": "adaptive", "display": "summarized"}),
    ("claude-haiku-4-5", None),
    ("claude-haiku-4-5-20251001", None),
  ],
)
def test_thinking_param_matches_existing_model_capability_mapping(
  model: str,
  expected: dict[str, object] | None,
) -> None:
  assert AnthropicProvider.thinking_param(model, 12_000) == expected


@pytest.mark.parametrize(
  "model",
  [
    "claude-zenith-9",
    # Listed by the authority for Risk's anthropic.sdk.messages adapter only.
    "claude-sonnet-4-6",
    # A non-claude id routed here is decided by the authority, not degraded.
    "gpt-5.2",
  ],
)
def test_model_without_a_messages_authority_entry_is_refused(model: str) -> None:
  provider = AnthropicProvider()

  with pytest.raises(ValueError, match="admits no anthropic.messages entry"):
    provider.get_model_info(model)
  with pytest.raises(ValueError, match="admits no anthropic.messages entry"):
    AnthropicProvider.thinking_param(model, 4096)


def _registry_entry(effort_control: Literal["output_config", "budget", "none"]):
  from gateway_test_support.model_defaults import compat_for_profile
  from model_authority.registry import ModelRegistryEntry
  from model_authority.schema import AnthropicThinking

  compat = compat_for_profile("messages.adaptive").model_copy(update={
    "thinking": AnthropicThinking(
      default_when_omitted="on",
      can_disable=False,
      effort_control=effort_control,
      effort_when_omitted="high",
    ),
    "max_output_tokens": 64_000,
    "context_window": 500_000,
  })
  return ModelRegistryEntry(
    key="anthropic.claude-nova-6",
    label="Nova 6",
    provider="anthropic",
    upstream_model="claude-nova-6",
    adapter="anthropic.messages",
    protocol_profile="messages.adaptive",
    route="anthropic.public",
    lifecycle="active",
    capabilities={"session.driver": "user_selectable"},
    supported_efforts=frozenset({"low", "medium", "high", "xhigh", "max"}),
    default_effort="high",
    features=frozenset({"tools", "streaming"}),
    reported_identities=frozenset({"claude-nova-6"}),
    compat=compat,
  )


@pytest.mark.parametrize(
  ("effort_control", "thinking_mode", "expected_thinking"),
  [
    ("output_config", "adaptive", {"type": "adaptive", "display": "summarized"}),
    ("budget", "budget", {"type": "enabled", "budget_tokens": 10000}),
    ("none", "none", None),
  ],
)
def test_new_authority_entry_is_served_from_its_compat_alone(
  monkeypatch,
  effort_control: Literal["output_config", "budget", "none"],
  thinking_mode: str,
  expected_thinking: dict[str, object] | None,
) -> None:
  # Adding a model is an authority data change: an entry with no gateway code
  # and no rate row is served, its limits and thinking shape read from compat.
  from model_authority.registry import ProductModelRegistry
  from model_authority.schema import SCHEMA
  import agent_gateway.providers.base as provider_base

  entry = _registry_entry(effort_control)
  monkeypatch.setattr(
    provider_base,
    "INITIAL_MODEL_REGISTRY",
    ProductModelRegistry(schema=SCHEMA, revision="test", models={entry.key: entry}),
  )

  info = AnthropicProvider().get_model_info("claude-nova-6")

  assert info.thinking_mode == thinking_mode
  assert (info.context_window, info.max_output_tokens) == (500_000, 64_000)
  assert info.effort_values == ("low", "medium", "high", "xhigh", "max")
  assert info.input_cost_per_mtok == info.output_cost_per_mtok == 0.0
  assert AnthropicProvider.thinking_param("claude-nova-6", 12_000) == expected_thinking


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-haiku-4-5-20251001"])
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
    model="claude-opus-5",
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
    model="claude-opus-5",
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
  assert json.loads(synthetic_block["content"]) == {
    "error": {
      "code": "missing_tool_result",
      "message": "No result provided for tool lookup",
      "data": {"next_call": None},
    },
  }


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


def test_normalize_messages_drops_foreign_thought_signature_on_tool_use() -> None:
  # L1 2026-10-13: a Gemini turn replayed after fallback to Anthropic was
  # rejected: `tool_use.thoughtSignature: Extra inputs are not permitted`.
  # The signature is the producing model's; a same-model replay keeps it.
  provider = AnthropicProvider()
  info = _model_info()
  foreign = {
    "role": "assistant",
    "provider": "google_antigravity",
    "model": "gemini-model",
    "content": [
      {
        "type": "tool_use", "id": "call-1", "name": "lookup", "input": {},
        "thoughtSignature": "sig", "thought_signature": "sig",
      },
    ],
  }
  own = {
    "role": "assistant",
    "provider": info.provider,
    "model": info.id,
    "content": [
      {"type": "tool_use", "id": "call-2", "name": "lookup", "input": {}, "thoughtSignature": "sig"},
    ],
  }
  tool_result = lambda call: {  # noqa: E731
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": call, "content": "ok"}],
  }

  normalized = provider.normalize_messages(
    [foreign, tool_result("call-1"), own, tool_result("call-2")], info,
  )

  assert normalized[0]["content"][0] == {
    "type": "tool_use", "id": "call-1", "name": "lookup", "input": {},
  }
  assert normalized[2]["content"][0]["thoughtSignature"] == "sig"


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
    {"model": "claude-sonnet-5", "messages": []},
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
      "claude-opus-5",
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
    "model": "claude-sonnet-5",
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
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", refused)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([refused, serving]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "absent-oauth.json"))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  provider = AnthropicProvider()
  client = _FakeClient(_make_anthropic_api_status_error(403, f"Error code: 403 - {error_body}", body=error_body))
  params = {
    "model": "claude-sonnet-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "hello"}],
    "_provider_auth_mode": "oauth",
  }
  with pytest.raises(RuntimeError) as exc_info:
    asyncio.run(_drain_stream(provider, client, params))

  failure = provider.classify_credential_failure(exc_info.value)

  assert failure is not None and failure.kind == "auth"
  refused_member = resolve_anthropic_credentials().by_token(refused)
  serving_member = resolve_anthropic_credentials().by_token(serving)
  assert refused_member is not None and serving_member is not None
  rotated = provider.next_credential({"auth_mode": "oauth", "auth_token": refused}, failure)
  assert rotated is not None
  assert rotated["auth_mode"] == "oauth"
  assert rotated["api_key"] == ""
  assert rotated["auth_token"] == serving
  assert rotated["credential_source"]["generation_ref"] == serving_member.generation_ref
  assert rotated[CREDENTIAL_PREPARED_FIELD] is True
  # The refusal parks the member in the shared sidecar, for every model.
  assert anthropic_credential_blocked_until(refused_member, model="claude-sonnet-5") > time.time()
  assert anthropic_credential_blocked_until(serving_member, model="claude-sonnet-5") <= time.time()


_QUOTA_PARAMS: dict[str, object] = {
  "model": "claude-sonnet-5",
  "max_tokens": 16,
  "messages": [{"role": "user", "content": "hello"}],
  "_provider_auth_mode": "oauth",
}
_UNIFIED = "anthropic-ratelimit-unified-"


def _binding(
  store: Path,
  identity: str = "acct-a",
  *,
  generation_id: str | None = None,
  affinity_key: str | None = "aff-key",
) -> AnthropicRequestBinding:
  return AnthropicRequestBinding(
    locator=anthropic_quota.locator_for_record(store, identity),
    tier="private",
    sidecar_store_path=store,
    affinity_store_path=store,
    generation_id=generation_id or f"gen-{identity}",
    affinity_key=affinity_key,
  )


def _five_hour_headers(
  *, utilization: float, reset_at: float, status: str = "allowed"
) -> dict[str, str]:
  return {
    f"{_UNIFIED}5h-status": status,
    f"{_UNIFIED}5h-utilization": str(utilization),
    f"{_UNIFIED}5h-reset": str(int(reset_at)),
  }


class _HeaderStreamContext(_StaticStreamContext):
  def __init__(self, headers: dict[str, str]):
    super().__init__([])
    self.response = SimpleNamespace(headers=headers)


class _HeaderStreamingClient:
  """A client whose stream opens with these response headers, bound as `create_client` binds it."""

  def __init__(self, headers: dict[str, str], binding: AnthropicRequestBinding):
    stream = lambda **_kwargs: _HeaderStreamContext(headers)  # noqa: E731
    self.messages = SimpleNamespace(stream=stream)
    self.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    self.hank_anthropic_binding = binding


def test_stream_records_a_header_only_observation_and_affinity_for_the_exact_binding(
  tmp_path: Path,
) -> None:
  store = tmp_path / "oauth.json"
  binding = _binding(store, "acct-a")
  sibling = _binding(store, "acct-b")
  client = _HeaderStreamingClient(
    _five_hour_headers(utilization=0.42, reset_at=time.time() + 3600), binding,
  )

  asyncio.run(_drain_stream(AnthropicProvider(), client, dict(_QUOTA_PARAMS)))

  state = anthropic_quota.load_selection_state(store)
  windows = {window.id: window for window in state.observations[binding.locator.id].windows}
  # The headers describe the five-hour window only: no weekly window is invented.
  assert set(windows) == {"five_hour"}
  assert windows["five_hour"].used_fraction == pytest.approx(0.42)
  assert windows["five_hour"].source == "response_header"
  assert sibling.locator.id not in state.observations
  assert state.affinity["aff-key"].locator_id == binding.locator.id


def test_refresh_then_429_is_attributed_to_the_refreshed_generation(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  anthropic = pytest.importorskip("anthropic")
  store = tmp_path / "oauth.json"
  stale = _binding(store, "acct-stale", generation_id="gen-before")
  refreshed = _binding(store, "acct-a", generation_id="gen-after")
  bindings = {"stale-bearer": stale, "refreshed-bearer": refreshed}
  bound_bearers: list[str] = []

  def bind(config: object, bearer: str) -> AnthropicRequestBinding:
    bound_bearers.append(bearer)
    return bindings[bearer]

  monkeypatch.setattr(
    anthropic_provider_module,
    "ensure_fresh_anthropic_credential",
    lambda token, **_kwargs: "refreshed-bearer",
  )
  monkeypatch.setattr(anthropic_provider_module, "anthropic_request_binding", bind)
  reset_at = time.time() + 2 * 3600
  provider = AnthropicProvider()
  client = provider.create_client({"auth_mode": "oauth", "auth_token": "stale-bearer"})

  def handle_request(request: httpx2.Request) -> httpx2.Response:
    assert request.headers["authorization"] == "Bearer refreshed-bearer"
    return httpx2.Response(
      429,
      headers=_five_hour_headers(utilization=1.0, reset_at=reset_at, status="rejected"),
      json={"type": "error", "error": {"type": "rate_limit_error", "message": "limit"}},
    )

  async def send_request() -> Exception:
    try:
      async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle_request)) as http_client:
        async with client.with_options(http_client=http_client) as local_client:
          with pytest.raises(anthropic.RateLimitError) as caught:
            await _drain_stream(provider, local_client, dict(_QUOTA_PARAMS))
      return caught.value
    finally:
      await provider.close_client(client)

  error = asyncio.run(send_request())

  assert bound_bearers == ["refreshed-bearer"]
  assert client.hank_anthropic_binding is refreshed
  state = anthropic_quota.load_selection_state(store)
  assert stale.locator.id not in state.observations
  window = state.observations[refreshed.locator.id].windows[0]
  assert (window.id, window.used_fraction) == ("five_hour", 1.0)
  assert not state.affinity
  failure = provider.classify_credential_failure(error)
  assert failure is not None
  assert (failure.kind, failure.limit_scope) == ("rate_limit", "shared")
  assert failure.reset_at == pytest.approx(reset_at, abs=2.0)


def test_create_client_does_not_refresh_a_prepared_config(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  pytest.importorskip("anthropic")
  binding = _binding(tmp_path / "oauth.json")
  bound_bearers: list[str] = []

  def refuse_refresh(token: str, **_kwargs: object) -> str:
    raise AssertionError("a prepared config's bearer is already fresh")

  def bind(config: object, bearer: str) -> AnthropicRequestBinding:
    bound_bearers.append(bearer)
    return binding

  monkeypatch.setattr(anthropic_provider_module, "ensure_fresh_anthropic_credential", refuse_refresh)
  monkeypatch.setattr(anthropic_provider_module, "anthropic_request_binding", bind)
  provider = AnthropicProvider()
  client = provider.create_client(
    {"auth_mode": "oauth", "auth_token": "prepared-bearer", CREDENTIAL_PREPARED_FIELD: True},
  )
  try:
    assert client.auth_token == "prepared-bearer"
    assert client.hank_anthropic_binding is binding
    assert bound_bearers == ["prepared-bearer"]
  finally:
    asyncio.run(provider.close_client(client))


@pytest.mark.parametrize(
  ("status", "headers", "body"),
  [
    pytest.param(429, {}, None, id="concurrency-429-without-unified-headers"),
    pytest.param(
      429,
      {
        f"{_UNIFIED}5h-status": "allowed",
        f"{_UNIFIED}7d-status": "allowed",
        f"{_UNIFIED}representative-claim": "five_hour",
      },
      None,
      id="concurrency-429-with-every-window-allowed",
    ),
    pytest.param(
      529,
      {},
      {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
      id="overload",
    ),
  ],
)
def test_concurrency_429_and_overload_classify_as_no_credential_failure(
  status: int,
  headers: dict[str, str],
  body: dict[str, object] | None,
) -> None:
  error = _make_anthropic_api_status_error(status, "rate limit exceeded", body=body, headers=headers)

  assert AnthropicProvider().classify_credential_failure(error) is None


@pytest.mark.parametrize(
  "family,claim",
  [
    ("opus", "seven_day_opus"),
    ("fable", "seven_day_fable"),
    ("fable", "seven_day_oi"),
    ("fable", "7d_oi"),
    ("mythos", "seven_day_mythos"),
  ],
)
def test_quota_429_with_a_family_claim_parks_only_that_model_family(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  family: str,
  claim: str,
) -> None:
  refused, serving = f"sk-ant-oat01-{family}-spent", f"sk-ant-oat01-{family}-serving"
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", refused)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([refused, serving]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "absent-oauth.json"))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  reset_at = time.time() + 5 * 24 * 3600
  error = _make_anthropic_api_status_error(
    429,
    "rate limit",
    headers={
      f"{_UNIFIED}status": "rejected",
      f"{_UNIFIED}representative-claim": claim,
      f"{_UNIFIED}reset": str(int(reset_at)),
    },
  )
  provider = AnthropicProvider()
  model = f"claude-{family}-5"
  params = {**_QUOTA_PARAMS, "model": model}
  with pytest.raises(Exception) as exc_info:
    asyncio.run(_drain_stream(provider, _FakeClient(error), params))
  assert exc_info.value is error

  failure = provider.classify_credential_failure(error)

  assert failure is not None
  assert (failure.kind, failure.limit_scope) == ("rate_limit", family)
  assert failure.reset_at == pytest.approx(reset_at, abs=2.0)
  rotated = provider.next_credential(
    {
      "auth_mode": "oauth",
      "auth_token": refused,
      CREDENTIAL_AFFINITY_KEY_FIELD: "aff-key",
      CREDENTIAL_MODEL_FAMILY_FIELD: family,
    },
    failure,
  )
  assert rotated is not None
  assert rotated["auth_token"] == serving
  assert rotated[CREDENTIAL_PREPARED_FIELD] is True
  assert rotated[CREDENTIAL_AFFINITY_KEY_FIELD] == "aff-key"
  assert rotated[CREDENTIAL_MODEL_FAMILY_FIELD] == family
  refused_member = resolve_anthropic_credentials().by_token(refused)
  assert refused_member is not None
  assert anthropic_credential_blocked_until(refused_member, model=model) > time.time()
  for healthy_family in ("opus", "sonnet", "fable", "mythos"):
    if healthy_family != family:
      assert anthropic_credential_blocked_until(
        refused_member, model=f"claude-{healthy_family}-5",
      ) <= time.time()


def test_an_arbitrary_403_keeps_the_generic_mapping_and_parks_no_quota(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  seat = "sk-ant-oat01-arbitrary-403-seat"
  monkeypatch.syspath_prepend(str(ROOT))
  credentials = importlib.import_module("api.credentials")
  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)
  for name in ("ANTHROPIC_AUTH_MODE", "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_BASE_URL"):
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "user-data"))
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
  monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-arbitrary-403-key")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([seat]))
  provider = AnthropicProvider()

  failure = provider.classify_credential_failure(
    _make_anthropic_api_status_error(403, "Request not allowed"),
  )

  # Not a quota signal: the generic mapping, so no scope and no quota verdict.
  assert failure is not None
  assert (failure.kind, failure.status_code, failure.limit_scope) == ("billing", 403, None)
  # An API key's billing denial still rotates to its seats (Henry 2026-10-13)...
  config = credentials.get_anthropic_config()
  assert config["auth_mode"] == "api"
  rotated = provider.next_credential(config, failure)
  assert rotated is not None
  assert (rotated["auth_mode"], rotated["auth_token"]) == ("oauth", seat)
  key = resolve_anthropic_credentials(
    {ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: config[ANTHROPIC_DEPLOYMENT_API_KEY_FIELD]},
  ).credentials[0]
  state = anthropic_quota.load_selection_state(key.sidecar_store_path)
  assert {
    cooldown.reason for rows in state.cooldowns.values() for cooldown in rows
  } == {"billing"}
  # ...while a seat's 403 is the organisation refusing the client and parks nothing.
  assert provider.next_credential({"auth_mode": "oauth", "auth_token": seat}, failure) is None


def test_a_403_with_billing_text_keeps_its_billing_verdict() -> None:
  failure = AnthropicProvider().classify_credential_failure(
    _make_anthropic_api_status_error(403, "Your credit balance is too low"),
  )

  assert failure is not None
  assert (failure.kind, failure.retryable_with_new_credentials) == ("billing", True)


def test_a_403_reporting_an_exhausted_window_is_a_quota_signal_through_the_stream_rejection() -> None:
  reset_at = time.time() + 3600
  error = _make_anthropic_api_status_error(
    403,
    "Request not allowed",
    headers=_five_hour_headers(utilization=1.0, reset_at=reset_at, status="rejected"),
  )
  provider = AnthropicProvider()
  with pytest.raises(RuntimeError) as exc_info:
    asyncio.run(_drain_stream(provider, _FakeClient(error), dict(_QUOTA_PARAMS)))
  assert exc_info.value is not error

  failure = provider.classify_credential_failure(exc_info.value)

  assert failure is not None
  assert (failure.kind, failure.status_code, failure.limit_scope) == ("rate_limit", 403, "shared")
  assert failure.reset_at == pytest.approx(reset_at, abs=2.0)


@pytest.fixture
def pool_tokens(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[str, str]:
  tokens = ("sk-ant-oat01-prepare-first", "sk-ant-oat01-prepare-second")
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", tokens[0])
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps(list(tokens)))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "absent-oauth.json"))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  # Ranking breaks ties by a session-seeded digest, so resolver order alone does not decide
  # the winner; a higher policy priority makes `first` the explicit choice.
  preferred = resolve_anthropic_credentials().by_token(tokens[0])
  assert preferred is not None
  anthropic_quota.set_account_policy(preferred.sidecar_store_path, preferred.locator, priority=1)
  return tokens


@pytest.fixture
def usage_probes(monkeypatch: pytest.MonkeyPatch) -> list[list[anthropic_quota.ProbeTarget]]:
  """Replace the usage probe with a recorder: no test reaches the live usage endpoint."""

  batches: list[list[anthropic_quota.ProbeTarget]] = []

  async def record(targets, **_kwargs):
    batches.append(list(targets))
    return {}

  monkeypatch.setattr(anthropic_quota, "probe_usage_batch", record)
  return batches


def _pool_config(token: str, *, automatic: bool, **extra: object) -> dict[str, object]:
  member = resolve_anthropic_credentials().by_token(token)
  assert member is not None
  source = {key: value for key, value in member.source_metadata().items() if key != "automatic"}
  if automatic:
    source["automatic"] = True
  return {"auth_mode": "oauth", "auth_token": token, "credential_source": source, **extra}


def _prepare(config: dict[str, object], session_id: str | None = "session-1") -> dict[str, object] | None:
  return asyncio.run(
    AnthropicProvider().prepare_credential(config, model="claude-sonnet-5", session_id=session_id)
  )


def test_prepare_credential_keeps_an_exact_source_and_ranks_the_pool_for_an_automatic_one(
  pool_tokens: tuple[str, str],
  usage_probes: list[list[anthropic_quota.ProbeTarget]],
) -> None:
  first, second = pool_tokens
  assert _prepare(_pool_config(first, automatic=False)) is None
  assert usage_probes == []

  material = _prepare(_pool_config(first, automatic=True))

  assert material is not None
  assert material["auth_token"] == first
  assert material[CREDENTIAL_PREPARED_FIELD] is True
  assert material[CREDENTIAL_AFFINITY_KEY_FIELD] == anthropic_quota.affinity_key("", "session-1")
  assert material[CREDENTIAL_AFFINITY_KEY_FIELD] is not None
  assert material[CREDENTIAL_MODEL_FAMILY_FIELD] == "sonnet"
  # Stateless requests share no affinity.
  stateless = _prepare(_pool_config(first, automatic=True), session_id="no-session")
  assert stateless is not None and stateless[CREDENTIAL_AFFINITY_KEY_FIELD] is None

  # The selection reads the shared sidecar: a cooled account yields to its sibling.
  member = resolve_anthropic_credentials().by_token(first)
  assert member is not None
  anthropic_quota.record_cooldown(
    member.sidecar_store_path,
    member.locator,
    anthropic_quota.Cooldown(scope="shared", until=time.time() + 3600, reason="quota"),
  )
  rerouted = _prepare(_pool_config(first, automatic=True))
  assert rerouted is not None and rerouted["auth_token"] == second


def test_prepare_fallback_refreshes_a_recovered_oauth_grant_on_the_first_client_build(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  usage_probes: list[list[anthropic_quota.ProbeTarget]],
) -> None:
  pytest.importorskip("anthropic")
  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)
  store = tmp_path / "oauth.json"
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(store))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  stale_token, fresh_token = "sk-ant-oat01-expired", "sk-ant-oat01-recovered"
  [record] = anthropic_oauth.upsert_anthropic_oauth_record(
    store,
    anthropic_oauth.AnthropicOAuthRecord(
      identity="recovering@example.com",
      access_token=stale_token,
      refresh_token="sk-ant-ort01-recovering",
      expires_at=0.0,
    ),
  )
  exchanges: list[str] = []

  def post(url: str, body: anthropic_oauth.JSONMapping) -> anthropic_oauth.JSONMapping:
    exchanges.append(str(body["refresh_token"]))
    if len(exchanges) == 1:
      raise anthropic_oauth.AnthropicOAuthError("token endpoint temporarily unavailable")
    return {
      "access_token": fresh_token,
      "refresh_token": "sk-ant-ort01-recovered",
      "expires_in": 28800,
    }

  monkeypatch.setattr(anthropic_oauth, "_post_token_endpoint", post)
  sources = resolve_anthropic_credentials()
  fallback = anthropic_oauth.select_anthropic_credential_source(sources)
  assert fallback is not None and fallback.token == stale_token
  [cooldown] = anthropic_quota.load_selection_state(store).cooldowns[fallback.locator.id]
  assert cooldown.reason == "refresh_failed" and cooldown.until > time.time()
  assert len(exchanges) == 1

  # The endpoint has recovered, but the pool still returns its blocked fallback.
  # Explicit False must replace a prepared flag retained from an earlier build.
  config = _pool_config(stale_token, automatic=True, **{CREDENTIAL_PREPARED_FIELD: True})
  material = _prepare(config)
  assert material is not None
  assert material["auth_token"] == stale_token
  assert material[CREDENTIAL_PREPARED_FIELD] is False
  assert len(exchanges) == 1
  assert usage_probes == []

  provider = AnthropicProvider()
  client = provider.create_client({**config, **material})
  try:
    assert len(exchanges) == 2
    assert client.auth_token == fresh_token
    assert client.hank_anthropic_binding.generation_id == record.generation_id
    [stored] = anthropic_oauth.load_anthropic_oauth_store(store)
    assert stored.access_token == fresh_token
  finally:
    asyncio.run(provider.close_client(client))


@pytest.mark.parametrize("key_blocked", [False, True])
def test_hosted_external_prepare_never_selects_a_setup_token(
  pool_tokens: tuple[str, str],
  usage_probes: list[list[anthropic_quota.ProbeTarget]],
  key_blocked: bool,
) -> None:
  key = anthropic_oauth.register_anthropic_static_credential("sk-ant-api03-hosted")
  assert key is not None
  if key_blocked:
    anthropic_quota.record_cooldown(
      key.sidecar_store_path,
      key.locator,
      anthropic_quota.Cooldown("shared", time.time() + 3600, "quota"),
    )
  config: dict[str, object] = {
    "auth_mode": "api",
    "api_key": key.token,
    "credential_source": {**key.source_metadata(), "automatic": True},
    ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: key.generation_ref,
    "placement": "hosted_external",
  }

  material = _prepare(config)

  assert material is not None
  assert material["auth_mode"] == "api"
  assert material["api_key"] == key.token
  assert material["auth_token"] == ""
  assert material["credential_source"] == {**key.source_metadata(), "automatic": True}


def test_prepare_credential_probes_stale_subscription_accounts_on_the_direct_route_only(
  pool_tokens: tuple[str, str],
  usage_probes: list[list[anthropic_quota.ProbeTarget]],
) -> None:
  first, second = pool_tokens

  assert _prepare(_pool_config(first, automatic=True)) is not None

  assert len(usage_probes) == 1
  members = [resolve_anthropic_credentials().by_token(token) for token in (first, second)]
  assert all(member is not None for member in members)
  assert {target.locator.id for target in usage_probes[0]} == {
    member.locator.id for member in members if member is not None
  }
  usage_probes.clear()

  custom = _pool_config(first, automatic=True, baseURL="https://gateway.example.com")
  assert _prepare(custom) is not None
  assert usage_probes == []


def test_concurrent_prepare_credential_claims_one_usage_probe_without_waiting(
  monkeypatch: pytest.MonkeyPatch,
  pool_tokens: tuple[str, str],
) -> None:
  first, _second = pool_tokens
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([first]))
  member = resolve_anthropic_credentials().by_token(first)
  assert member is not None
  config = _pool_config(first, automatic=True)
  real_probe = anthropic_quota.probe_usage_batch
  requested: list[str] = []

  async def exercise() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
      requested.append(request.headers["authorization"])
      entered.set()
      await release.wait()
      return httpx.Response(200, json={"five_hour": {"utilization": 10}})

    async def probe_through_fixture(targets, **kwargs):
      return await real_probe(targets, transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(anthropic_quota, "probe_usage_batch", probe_through_fixture)
    started_at = time.time()
    first_prepare = asyncio.create_task(AnthropicProvider().prepare_credential(
      config, model="claude-sonnet-5", session_id="first-prepare",
    ))
    try:
      await asyncio.wait_for(entered.wait(), timeout=1.0)
      probe = anthropic_quota.load_selection_state(member.sidecar_store_path).observations[
        member.locator.id
      ].probe
      assert probe.in_progress_until is not None
      assert started_at < probe.in_progress_until
      assert probe.in_progress_until <= (
        time.time() + anthropic_quota.PROBE_BATCH_BUDGET_SECONDS + 0.75
      )
      second_material = await asyncio.wait_for(
        AnthropicProvider().prepare_credential(
          config, model="claude-sonnet-5", session_id="second-prepare",
        ),
        timeout=1.0,
      )
      assert second_material is not None and second_material["auth_token"] == first
      assert not first_prepare.done()
      assert requested == [f"Bearer {first}"]
    finally:
      release.set()
      first_material = await asyncio.wait_for(
        first_prepare, timeout=anthropic_quota.PROBE_BATCH_BUDGET_SECONDS + 1.0,
      )
    assert first_material is not None and first_material["auth_token"] == first

  asyncio.run(exercise())

  assert requested == [f"Bearer {first}"]
  observation = anthropic_quota.load_selection_state(member.sidecar_store_path).observations[
    member.locator.id
  ]
  assert observation.probe.in_progress_until is None
  assert observation.probe.last_usage_at is not None
  assert observation.windows and observation.windows[0].used_fraction == pytest.approx(0.1)


def test_prepare_credential_probes_after_an_in_progress_marker_expires(
  monkeypatch: pytest.MonkeyPatch,
  pool_tokens: tuple[str, str],
) -> None:
  first, _second = pool_tokens
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([first]))
  member = resolve_anthropic_credentials().by_token(first)
  assert member is not None
  anthropic_quota.record_probe_state(
    member.sidecar_store_path,
    member.locator,
    anthropic_quota.UsageProbeState(in_progress_until=time.time() - 1.0),
  )
  real_probe = anthropic_quota.probe_usage_batch
  requested: list[str] = []

  def handle(request: httpx.Request) -> httpx.Response:
    requested.append(request.headers["authorization"])
    return httpx.Response(200, json={"five_hour": {"utilization": 20}})

  async def probe_through_fixture(targets, **kwargs):
    return await real_probe(targets, transport=httpx.MockTransport(handle), **kwargs)

  monkeypatch.setattr(anthropic_quota, "probe_usage_batch", probe_through_fixture)

  material = _prepare(_pool_config(first, automatic=True))

  assert material is not None and material["auth_token"] == first
  assert requested == [f"Bearer {first}"]
  observation = anthropic_quota.load_selection_state(member.sidecar_store_path).observations[
    member.locator.id
  ]
  assert observation.probe.in_progress_until is None
  assert observation.probe.last_usage_at is not None
  assert observation.windows and observation.windows[0].used_fraction == pytest.approx(0.2)


def test_usage_probe_claim_in_another_process_honors_a_live_marker_without_waiting(
  pool_tokens: tuple[str, str],
) -> None:
  first, _second = pool_tokens
  member = resolve_anthropic_credentials().by_token(first)
  assert member is not None
  now = time.time()
  assert anthropic_quota.claim_usage_probe(
    member.sidecar_store_path, member.locator, generation=member.generation_id or None, now=now,
  )
  before = anthropic_quota.load_selection_state(member.sidecar_store_path).observations[
    member.locator.id
  ].probe
  assert before.in_progress_until is not None
  # Keep the claim's logical clock fixed across interpreter startup. The child's
  # elapsed measurement covers only the shared-store claim, not import overhead.
  child = subprocess.run(
    [
      sys.executable,
      "-c",
      """
import json
import sys
import time
from pathlib import Path
from agent_gateway.providers import anthropic_quota

store_path, namespace, identity, generation, now = sys.argv[1:]
started = time.monotonic()
claimed = anthropic_quota.claim_usage_probe(
  Path(store_path),
  anthropic_quota.AccountLocator(namespace=namespace, identity=identity),
  generation=json.loads(generation),
  now=float(now),
)
print(json.dumps({"claimed": claimed, "elapsed": time.monotonic() - started}))
""",
      str(member.sidecar_store_path),
      member.locator.namespace,
      member.locator.identity,
      json.dumps(member.generation_id or None),
      str(now),
    ],
    cwd=PKG_DIR,
    check=True,
    capture_output=True,
    text=True,
    timeout=10.0,
  )

  result = json.loads(child.stdout)
  assert result["claimed"] is False
  assert result["elapsed"] < 1.0
  after = anthropic_quota.load_selection_state(member.sidecar_store_path).observations[
    member.locator.id
  ].probe
  assert after == before


def test_prepare_credential_still_selects_when_the_usage_probe_is_forbidden(
  monkeypatch: pytest.MonkeyPatch,
  pool_tokens: tuple[str, str],
) -> None:
  first, _second = pool_tokens
  real_probe = anthropic_quota.probe_usage_batch
  requested: list[str] = []

  def forbidden(request: httpx.Request) -> httpx.Response:
    requested.append(request.url.host)
    return httpx.Response(403, json={"error": {"type": "permission_error", "message": "nope"}})

  async def probe_through_fixture(targets, **kwargs):
    return await real_probe(targets, transport=httpx.MockTransport(forbidden), **kwargs)

  monkeypatch.setattr(anthropic_quota, "probe_usage_batch", probe_through_fixture)

  material = _prepare(_pool_config(first, automatic=True))

  assert material is not None and material["auth_token"] == first
  assert requested == ["api.anthropic.com", "api.anthropic.com"]
  member = resolve_anthropic_credentials().by_token(first)
  assert member is not None
  probe = anthropic_quota.load_selection_state(member.sidecar_store_path).observations[
    member.locator.id
  ].probe
  assert probe.unavailable_reason == "http_403"
  assert probe.unavailable_until is not None and probe.unavailable_until > time.time()
  # The verdict holds for the credential: the next preparation does not probe again.
  requested.clear()
  assert _prepare(_pool_config(first, automatic=True)) is not None
  assert requested == []


def test_prepare_credential_selects_without_telemetry_when_the_probe_fails(
  monkeypatch: pytest.MonkeyPatch,
  pool_tokens: tuple[str, str],
) -> None:
  first, _second = pool_tokens

  async def broken(targets, **_kwargs):
    raise RuntimeError("usage telemetry exploded")

  monkeypatch.setattr(anthropic_quota, "probe_usage_batch", broken)

  material = _prepare(_pool_config(first, automatic=True))

  assert material is not None and material["auth_token"] == first


def test_prepare_credential_is_a_no_op_for_an_api_key_only_config(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  usage_probes: list[list[anthropic_quota.ProbeTarget]],
) -> None:
  for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_AUTH_TOKENS", "CLAUDE_CODE_OAUTH_TOKEN"):
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "absent-oauth.json"))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))

  assert _prepare({"auth_mode": "api", "api_key": "sk-ant-api03-only"}) is None

  assert usage_probes == []
  assert list(tmp_path.rglob("*.selection.json")) == []


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

  types = asyncio.run(_collect_stream_types(provider, client, {"model": "claude-sonnet-5", "messages": []}))

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


@pytest.mark.parametrize(
  ("start_usage", "deltas", "expected"),
  [
    pytest.param(
      {"input_tokens": 100, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 30},
      [{"output_tokens": 7}], 330, id="output-only",
    ),
    pytest.param(
      {"input_tokens": 100, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 30},
      [None], 330, id="no-final-usage",
    ),
    pytest.param(
      {"input_tokens": 100, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 30},
      [{"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}],
      0, id="explicit-zero",
    ),
    pytest.param(
      {"input_tokens": 100, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 30},
      [{"cache_creation_input_tokens": 0}, {"input_tokens": 20}],
      220, id="partial-components",
    ),
    pytest.param(
      {"input_tokens": 100, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 30},
      [{"input_tokens": 200, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 5}],
      205, id="cache-category-shift",
    ),
    pytest.param({"input_tokens": 0}, [{"output_tokens": 7}], 0, id="zero-start"),
    pytest.param({"input_tokens": 12}, [{"output_tokens": 7}], 12, id="optional-caches"),
    pytest.param(None, [{"input_tokens": 12}], 12, id="late-prompt"),
    pytest.param(None, [{"output_tokens": 7}], None, id="no-prompt"),
    pytest.param(
      {"cache_read_input_tokens": 200}, [{"output_tokens": 7}], None,
      id="missing-required-input",
    ),
    pytest.param(None, [None], None, id="no-usage"),
  ],
)
def test_stream_final_prompt_preserves_same_invocation_components(
  start_usage: dict[str, int] | None,
  deltas: list[dict[str, int] | None],
  expected: int | None,
) -> None:
  provider = AnthropicProvider()
  client = _FakeStreamingClient([
    SimpleNamespace(
      type="message_start",
      message=SimpleNamespace(
        usage=SimpleNamespace(**start_usage) if start_usage is not None else None,
      ),
    ),
    *[
      SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason="end_turn"),
        usage=SimpleNamespace(**usage) if usage is not None else None,
      )
      for usage in deltas
    ],
  ])

  events = asyncio.run(_collect_stream_events(
    provider, client, {"model": "claude-sonnet-5", "messages": []},
  ))

  assert events[-1].type == "message_end"
  assert events[-1].request_input_tokens == expected
  assert events[0].input_tokens == (start_usage or {}).get("input_tokens", 0)
  assert events[0].cache_read_tokens == (start_usage or {}).get("cache_read_input_tokens", 0)
  assert events[0].cache_creation_tokens == (start_usage or {}).get("cache_creation_input_tokens", 0)


@pytest.mark.parametrize("native_compaction", [False, True])
@pytest.mark.parametrize(
  ("last_prompt", "expected", "billed_prompt"),
  [
    pytest.param(
      {"input_tokens": 20, "cache_read_input_tokens": 40, "cache_creation_input_tokens": 5},
      65, (120, 240, 35), id="compacted",
    ),
    pytest.param(
      {"input_tokens": 200, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 5},
      205, (300, 200, 35), id="cache-category-shift",
    ),
    pytest.param(
      {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
      0, (100, 200, 30), id="zero",
    ),
    pytest.param(
      {"cache_read_input_tokens": 40, "cache_creation_input_tokens": 5},
      None, (100, 240, 35), id="missing-required-input",
    ),
    pytest.param({"input_tokens": 20}, 20, (120, 200, 30), id="optional-caches"),
  ],
)
def test_stream_final_iteration_prompt_is_independent_of_repeated_billing_and_wait(
  monkeypatch: pytest.MonkeyPatch,
  native_compaction: bool,
  last_prompt: dict[str, int],
  expected: int | None,
  billed_prompt: tuple[int, int, int],
) -> None:
  acquired: list[int] = []

  async def acquire(*, estimated_tokens: int) -> float:
    acquired.append(estimated_tokens)
    return 1.234

  monkeypatch.setattr(
    anthropic_provider_module, "get_global_token_bucket",
    lambda: SimpleNamespace(acquire=acquire),
  )
  first = {
    "input_tokens": 100, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 30,
    "output_tokens": 10, "server_tool_use": {"web_search_requests": 1},
  }
  last = {
    **last_prompt, "output_tokens": 7, "server_tool_use": {"web_fetch_requests": 2},
  }
  snapshot = SimpleNamespace(
    # The top-level initial prompt must not override the final iteration.
    **first, iterations=[SimpleNamespace(**first), SimpleNamespace(**last)],
  )
  client = _FakeStreamingClient([
    SimpleNamespace(
      type="message_start",
      message=SimpleNamespace(usage=SimpleNamespace(**{**first, "output_tokens": 0})),
    ),
    *[
      SimpleNamespace(
        type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=snapshot,
      )
      for _ in range(2)
    ],
  ])
  params: dict[str, object] = {
    "model": "claude-sonnet-5", "messages": [], "max_tokens": 4096,
  }
  if native_compaction:
    params["context_management"] = {"edits": [{"type": "compact_20260112"}]}

  events = asyncio.run(_collect_stream_events(AnthropicProvider(), client, params))

  assert acquired == [4096]
  assert events[0].provider_wait_ms == 1234
  assert sum(event.provider_wait_ms for event in events) == 1234
  assert events[-1].type == "message_end"
  assert events[-1].request_input_tokens == expected
  assert (
    sum(event.input_tokens for event in events),
    sum(event.cache_read_tokens for event in events),
    sum(event.cache_creation_tokens for event in events),
  ) == billed_prompt
  assert sum(event.output_tokens for event in events) == 17
  assert [event.provider_unit_deltas for event in events if event.type == "usage_update"] == [
    {"web_search": 1, "web_fetch": 2}, {},
  ]
  assert (client.beta.messages.kwargs is not None) is native_compaction
  assert (client.messages.kwargs is not None) is not native_compaction


@pytest.mark.parametrize("iterations", [None, []])
def test_stream_sdk_output_only_delta_keeps_initial_prompt(iterations: list[object] | None) -> None:
  from anthropic.types import RawMessageDeltaEvent

  client = _FakeStreamingClient([
    SimpleNamespace(
      type="message_start",
      message=SimpleNamespace(usage=SimpleNamespace(
        input_tokens=100, cache_read_input_tokens=200, cache_creation_input_tokens=30,
      )),
    ),
    RawMessageDeltaEvent.model_validate({
      "type": "message_delta",
      "delta": {"stop_reason": "end_turn", "stop_sequence": None},
      "usage": {"output_tokens": 7, "iterations": iterations},
    }),
  ])

  events = asyncio.run(_collect_stream_events(
    AnthropicProvider(), client, {"model": "claude-sonnet-5", "messages": []},
  ))

  assert events[-1].request_input_tokens == 330
  assert (
    sum(event.input_tokens for event in events),
    sum(event.cache_read_tokens for event in events),
    sum(event.cache_creation_tokens for event in events),
    sum(event.output_tokens for event in events),
  ) == (100, 200, 30, 7)


@pytest.mark.parametrize(
  ("last_iteration", "expected"),
  [
    pytest.param({"input_tokens": 20, "cache_read_input_tokens": 40}, 60, id="known"),
    pytest.param({"output_tokens": 7}, None, id="incomplete-latest-snapshot"),
  ],
)
def test_stream_iteration_snapshots_replace_prompt_without_scalar_aggregate_leakage(
  last_iteration: dict[str, int],
  expected: int | None,
) -> None:
  initial = {"input_tokens": 100, "cache_read_input_tokens": 200}
  client = _FakeStreamingClient([
    SimpleNamespace(
      type="message_start", message=SimpleNamespace(usage=SimpleNamespace(**initial)),
    ),
    *[
      SimpleNamespace(
        type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=usage,
      )
      for usage in [
        {"iterations": [initial, {"input_tokens": 50, "cache_read_input_tokens": 60}]},
        {"iterations": [initial, last_iteration]},
        {"output_tokens": 7},
        # Without an iteration identity these scalar totals cannot restore the
        # last invocation's missing input or overwrite its known smaller prompt.
        {"input_tokens": 999, "cache_read_input_tokens": 999, "iterations": []},
        None,
      ]
    ],
  ])

  events = asyncio.run(_collect_stream_events(
    AnthropicProvider(), client, {"model": "claude-sonnet-5", "messages": []},
  ))

  assert events[-1].request_input_tokens == expected


def test_stream_message_end_carries_refusal_stop_details() -> None:
  from anthropic.types import RawMessageDeltaEvent

  provider = AnthropicProvider()
  client = _FakeStreamingClient(
    [
      RawMessageDeltaEvent.model_validate(
        {
          "type": "message_delta",
          "delta": {
            "stop_reason": "refusal",
            "stop_sequence": None,
            "stop_details": {
              "type": "refusal",
              "category": "reasoning_extraction",
              "explanation": "The request asks for internal reasoning.",
            },
          },
          "usage": {"output_tokens": 3},
        }
      ),
    ]
  )

  events = asyncio.run(_collect_stream_events(provider, client, {"model": "claude-opus-5", "messages": []}))

  message_end = events[-1]
  assert message_end.type == "message_end"
  assert message_end.stop_reason == "refusal"
  assert message_end.stop_details == {
    "type": "refusal",
    "category": "reasoning_extraction",
    "explanation": "The request asks for internal reasoning.",
  }


def test_stream_message_end_without_stop_details_carries_none() -> None:
  provider = AnthropicProvider()
  client = _FakeStreamingClient(
    [SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=None)]
  )

  events = asyncio.run(_collect_stream_events(provider, client, {"model": "claude-opus-5", "messages": []}))

  message_end = events[-1]
  assert message_end.type == "message_end"
  assert message_end.stop_reason == "end_turn"
  assert message_end.stop_details is None


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
  from model_authority.current import INITIAL_MODEL_REGISTRY

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
