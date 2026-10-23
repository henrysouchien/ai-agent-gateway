from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_gateway.auth import AuthConfig
from agent_gateway.providers import (
  CopilotResponsesProvider,
  GoogleGenerateContentProvider,
  MuseResponsesProvider,
  OllamaLocalProvider,
  installed_adapter_route_support,
)
from agent_gateway.providers import base as provider_base
from gateway_test_support.model_defaults import compat_for_profile
from model_authority.capabilities import GATEWAY_EXECUTED_CAPABILITY_IDS
from model_authority.rates import RateTable
from model_authority.registry import ModelRegistryEntry, ProductModelRegistry
from model_authority.schema import SCHEMA
from model_authority.thinking import ThinkingLevel


_NEW_PROVIDERS = (
  GoogleGenerateContentProvider,
  CopilotResponsesProvider,
  MuseResponsesProvider,
  OllamaLocalProvider,
)

# One gateway-executed entry per installed route. Protocol profiles are the
# ones each adapter declares; admission refuses any other combination.
_ADMITTED_ROUTES = (
  ("google.gemini-fixture", "google", "gemini-fixture", "google.generate_content", "generate_content.standard", "google.public"),
  ("copilot.copilot-fixture", "copilot", "copilot-fixture", "copilot.responses", "responses.reasoning", "copilot.api"),
  ("muse.muse-fixture", "muse", "muse-fixture", "muse.responses", "responses.reasoning", "muse.api"),
  ("ollama.llama-fixture", "ollama", "llama-fixture", "ollama.chat_completions", "chat_completions.standard", "ollama.local"),
  ("ollama.daemon-fixture", "ollama", "daemon-fixture", "ollama.chat_completions", "chat_completions.standard", "ollama.daemon_cloud"),
  ("anthropic.claude-fixture", "anthropic", "claude-fixture", "anthropic.messages", "messages.standard", "anthropic.public"),
  ("codex.codex-fixture", "codex", "codex-fixture", "codex.responses", "codex.reasoning", "codex.chatgpt"),
  ("xai.grok-fixture", "xai", "grok-fixture", "xai.responses", "responses.reasoning", "xai.public"),
)

_OLLAMA_CLOUD_KEY = "ollama.cloud-route"
_TEXT = "hello"
_LOCAL_MODEL = "llama-fixture"
_CLOUD_MODEL = "qwen3-coder-cloud"


@pytest.fixture(autouse=True)
def _unpriced_provider_tables(monkeypatch: pytest.MonkeyPatch) -> None:
  """Copilot and Muse construct against the authority rate table.

  A provider the authority has not priced yet has no file. An empty table
  keeps construction on the same zero-cost path a missing row already takes,
  without inventing a price.
  """

  tables = dict(provider_base.RATE_TABLES)
  for name in ("google", "copilot", "muse", "ollama"):
    tables.setdefault(name, RateTable(
      version="0",
      source="test",
      providers={name: {}},
      _path=Path(name),
    ))
  monkeypatch.setattr(provider_base, "RATE_TABLES", tables)


def _entry(
  key: str,
  provider: str,
  upstream: str,
  adapter: str,
  profile: str,
  route: str,
) -> ModelRegistryEntry:
  return ModelRegistryEntry(
    key=key,
    label=key,
    provider=provider,
    upstream_model=upstream,
    adapter=adapter,
    protocol_profile=profile,
    route=route,
    lifecycle="active",
    capabilities={"session.driver": "user_selectable"},
    supported_efforts=frozenset({"none"}),
    default_effort="none",
    features=frozenset({"streaming"}),
    reported_identities=frozenset({upstream}),
    compat=compat_for_profile(profile),
  )


def _registry(*extra: ModelRegistryEntry) -> ProductModelRegistry:
  entries = [_entry(*row) for row in _ADMITTED_ROUTES]
  entries.extend(extra)
  return ProductModelRegistry(
    schema=SCHEMA,
    revision="provider-expansion-test",
    models={entry.key: entry for entry in entries},
  )


def _admit(registry: ProductModelRegistry) -> frozenset[str]:
  return registry.admit_adapter_support(
    installed_adapter_route_support(),
    executed_capability_ids=GATEWAY_EXECUTED_CAPABILITY_IDS,
  )


def _json_error(message: str, **fields: str) -> bytes:
  return json.dumps({"error": {"message": message, **fields}}).encode()


def _sse(*events: dict[str, Any]) -> bytes:
  return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


def _active_config(provider_name: str) -> dict[str, str]:
  if provider_name == "google":
    return {"auth_token": "test-token"}
  if provider_name == "copilot":
    return {"auth_token": "test-token"}
  if provider_name == "muse":
    return {"api_key": "test-token"}
  return {}


async def _stream_status(
  provider: Any,
  *,
  status: int,
  body: bytes,
  headers: dict[str, str] | None = None,
  model: str = _LOCAL_MODEL,
  config: dict[str, str] | None = None,
) -> Exception:
  """The exception ``stream`` raises for a non-200, including Ollama's stamps."""

  def handle(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, content=body)

  client_config: dict[str, object] = dict(config or _active_config(provider.name))
  client_config["_transport"] = httpx.MockTransport(handle)
  client = provider.create_client(client_config)
  try:
    stream = provider.stream(client, {"model": model})
    try:
      async for _event in stream:
        pass
    except Exception as exc:
      return exc
    raise AssertionError(f"{provider.name} stream did not raise for HTTP {status}")
  finally:
    await provider.close_client(client)


def _classify(provider_cls: type, **kwargs: Any):
  provider = provider_cls()
  exc = asyncio.run(_stream_status(provider, **kwargs))
  assert isinstance(exc, httpx.HTTPStatusError)
  return provider.classify_credential_failure(exc)


def test_installed_routes_are_admitted() -> None:
  assert _admit(_registry()) == frozenset()


def test_undeclared_ollama_cloud_route_is_observed_not_refused() -> None:
  registry = _registry(_entry(
    _OLLAMA_CLOUD_KEY,
    "ollama",
    "qwen-fixture",
    "ollama.chat_completions",
    "chat_completions.standard",
    "ollama.cloud",
  ))

  # The artifact deploys ahead of code: an entry no installed adapter declares
  # is returned for selection to skip, never a failed import.
  assert _admit(registry) == frozenset({_OLLAMA_CLOUD_KEY})


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
def test_429_is_rate_limit(provider_cls: type) -> None:
  failure = _classify(provider_cls, status=429, body=_json_error("slow down"))

  assert failure is not None
  assert failure.kind == "rate_limit"


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
def test_quota_exhausted_429_stays_rate_limit(provider_cls: type) -> None:
  # "quota" is a billing word in the generic classifier. A 429 whose body
  # says quota / RESOURCE_EXHAUSTED is still a rate limit, not a denial.
  failure = _classify(
    provider_cls,
    status=429,
    body=_json_error("Quota exceeded", status="RESOURCE_EXHAUSTED"),
  )

  assert failure is not None
  assert failure.kind == "rate_limit"


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
def test_retry_after_sets_reset_at(provider_cls: type) -> None:
  before = time.time()
  failure = _classify(
    provider_cls,
    status=429,
    body=_json_error("slow down"),
    headers={"Retry-After": "30"},
  )

  assert failure is not None
  assert failure.kind == "rate_limit"
  assert failure.reset_at == pytest.approx(before + 30, abs=2)


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
def test_401_is_auth(provider_cls: type) -> None:
  failure = _classify(provider_cls, status=401, body=_json_error("invalid credential"))

  assert failure is not None
  assert failure.kind == "auth"


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
def test_403_is_billing(provider_cls: type) -> None:
  # A local tag, and a body with no cloud marker, so Ollama's cloud rule
  # does not reclassify this 403 as auth.
  failure = _classify(
    provider_cls,
    status=403,
    body=_json_error("forbidden"),
    model=_LOCAL_MODEL,
  )

  assert failure is not None
  assert failure.kind == "billing"


def test_ollama_cloud_tag_403_is_auth() -> None:
  # ``name:cloud`` or a tag ending in ``-cloud`` on a 403 is an unsigned-in
  # daemon, not a billing denial. The body has no cloud marker, so the
  # decision is the model tag stream stamps from the request.
  failure = _classify(
    OllamaLocalProvider,
    status=403,
    body=_json_error("signin required"),
    model=_CLOUD_MODEL,
  )

  assert failure is not None
  assert failure.kind == "auth"


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
def test_400_is_not_a_credential_failure(provider_cls: type) -> None:
  # The phrase would be a rate limit on a 429. Status 400 is not a hop.
  failure = _classify(
    provider_cls,
    status=400,
    body=_json_error("too many requests"),
  )

  assert failure is None


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
def test_context_length_is_not_rate_limit(provider_cls: type) -> None:
  # Not a 400/429, and the body also says "too many requests", so dropping
  # the context-length guard would classify this as a rate limit.
  failure = _classify(
    provider_cls,
    status=413,
    body=_json_error("prompt too long: too many requests"),
  )

  assert failure is None or failure.kind != "rate_limit"


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
@pytest.mark.parametrize(
  ("status", "retryable"),
  [(503, True), (429, False), (400, False)],
)
def test_status_retryability(provider_cls: type, status: int, retryable: bool) -> None:
  provider = provider_cls()
  exc = asyncio.run(_stream_status(provider, status=status, body=_json_error("unavailable")))

  assert isinstance(exc, httpx.HTTPStatusError)
  assert provider.is_retryable_error(exc) is retryable


@pytest.mark.parametrize("provider_cls", _NEW_PROVIDERS)
def test_connect_error_is_retryable(provider_cls: type) -> None:
  assert provider_cls().is_retryable_error(httpx.ConnectError("connection refused")) is True


def test_google_credential_is_token_or_api_key() -> None:
  provider = GoogleGenerateContentProvider()

  assert provider.has_active_credential({"auth_token": "test-token"}) is True
  assert provider.has_active_credential({"api_key": "test-api-key"}) is True
  assert provider.has_active_credential({}) is False
  assert provider.has_active_credential({"auth_token": "  ", "api_key": ""}) is False


def test_copilot_credential_requires_auth_token() -> None:
  provider = CopilotResponsesProvider()

  assert provider.has_active_credential({"auth_token": "test-token"}) is True
  assert provider.has_active_credential({"api_key": "test-api-key"}) is False
  assert provider.has_active_credential({}) is False
  assert provider.has_active_credential({"auth_token": " "}) is False


def test_muse_credential_requires_api_key() -> None:
  provider = MuseResponsesProvider()

  assert provider.has_active_credential({"api_key": "test-token"}) is True
  assert provider.has_active_credential({"auth_token": "test-token"}) is False
  assert provider.has_active_credential({}) is False
  assert provider.has_active_credential({"api_key": " "}) is False


def test_ollama_is_keyless_when_the_daemon_answers() -> None:
  from agent_gateway.providers.ollama import _reset_presence_cache

  _reset_presence_cache()
  daemon = httpx.MockTransport(lambda _request: httpx.Response(200, json={"version": "0"}))
  assert OllamaLocalProvider().has_active_credential({"_transport": daemon}) is True
  _reset_presence_cache()


@pytest.mark.parametrize("provider", ["google", "copilot", "muse", "ollama"])
def test_auth_config_accepts_provider(provider: str) -> None:
  config = AuthConfig.from_dict({"provider": provider, "billing_mode": "byok"})

  assert config.provider == provider


def test_auth_config_rejects_unknown_provider() -> None:
  with pytest.raises(ValueError, match="Unsupported provider 'fixture'"):
    AuthConfig.from_dict({"provider": "fixture", "billing_mode": "byok"})


def _responses_sse() -> bytes:
  return _sse(
    {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}},
    {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
    {"type": "response.output_text.delta", "delta": _TEXT},
    {"type": "response.completed", "response": {"status": "completed"}},
  )


def _google_sse() -> bytes:
  return _sse({
    "candidates": [{
      "content": {"role": "model", "parts": [{"text": _TEXT}]},
      "finishReason": "STOP",
    }],
  })


def _ollama_sse() -> bytes:
  return _sse({
    "choices": [{"delta": {"content": _TEXT}, "finish_reason": "stop"}],
  })


_SMOKE = (
  (
    GoogleGenerateContentProvider,
    "gemini-fixture",
    {"auth_token": "test-token"},
    "bearer",
    "test-token",
    "/v1beta/models/gemini-fixture:streamGenerateContent",
    _google_sse,
  ),
  (
    GoogleGenerateContentProvider,
    "gemini-fixture",
    {"api_key": "test-api-key"},
    "api-key",
    "test-api-key",
    "/v1beta/models/gemini-fixture:streamGenerateContent",
    _google_sse,
  ),
  (
    CopilotResponsesProvider,
    "copilot-fixture",
    {"auth_token": "test-token"},
    "bearer",
    "test-token",
    "/responses",
    _responses_sse,
  ),
  (
    MuseResponsesProvider,
    "muse-fixture",
    {"api_key": "test-token"},
    "bearer",
    "test-token",
    "/v1/responses",
    _responses_sse,
  ),
  (
    OllamaLocalProvider,
    "llama-fixture",
    {"api_key": ""},
    "none",
    "",
    "/v1/chat/completions",
    _ollama_sse,
  ),
)


def _assert_auth(request: httpx.Request, kind: str, secret: str) -> None:
  if kind == "bearer":
    assert request.headers["authorization"] == f"Bearer {secret}"
    assert "x-goog-api-key" not in request.headers
  elif kind == "api-key":
    assert request.headers["x-goog-api-key"] == secret
    assert "authorization" not in request.headers
  else:
    assert "authorization" not in request.headers


@pytest.mark.parametrize(
  ("provider_cls", "model", "config", "auth_kind", "secret", "path", "body"),
  _SMOKE,
  ids=["google-bearer", "google-api-key", "copilot", "muse", "ollama"],
)
def test_stream_wire(
  provider_cls: type,
  model: str,
  config: dict[str, str],
  auth_kind: str,
  secret: str,
  path: str,
  body,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(provider_base, "INITIAL_MODEL_REGISTRY", _registry())
  seen: dict[str, httpx.Request] = {}

  def handle(request: httpx.Request) -> httpx.Response:
    seen["request"] = request
    return httpx.Response(
      200,
      headers={"content-type": "text/event-stream"},
      content=body(),
    )

  async def collect():
    provider = provider_cls()
    client = provider.create_client({**config, "_transport": httpx.MockTransport(handle)})
    try:
      info = provider.get_model_info(model)
      effort = provider.resolve_effort(
        requested=ThinkingLevel.NONE,
        model=model,
        model_info=info,
        max_tokens=16,
      )
      params = provider.build_request_params(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        system_prompt=None,
        tools=[],
        max_tokens=16,
        effort_resolution=effort,
      )
      return [event async for event in provider.stream(client, params)]
    finally:
      await provider.close_client(client)

  events = asyncio.run(collect())
  request = seen["request"]

  assert request.url.path == path
  _assert_auth(request, auth_kind, secret)
  assert any(
    event.type in {"text_delta", "text_end"} and _TEXT in str(event.text)
    for event in events
  )
  assert any(event.type == "message_end" and event.stop_reason == "end_turn" for event in events)


def test_muse_stream_reports_response_diagnostics_reasoning_tokens_and_reasoning_item() -> None:
  """Muse rides the OpenAI Responses mapper: response.id, service_tier, reasoning_tokens, headers."""
  stream_events: list[dict[str, Any]] = [
    {"type": "response.created", "response": {"id": "resp_muse", "model": "muse-fixture", "status": "in_progress"}},
    {"type": "response.output_item.added",
     "item": {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}},
    {"type": "response.output_item.done",
     "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC",
              "summary": [{"type": "summary_text", "text": "thought"}]}},
    {"type": "response.completed", "response": {
      "id": "resp_muse", "model": "muse-fixture", "status": "completed", "service_tier": "default",
      "usage": {"input_tokens": 4, "output_tokens": 8, "output_tokens_details": {"reasoning_tokens": 6}},
    }},
  ]
  payload = "".join("data: " + json.dumps(item) + "\n\n" for item in stream_events)

  def handle(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={
      "content-type": "text/event-stream",
      "x-request-id": "mreq_1",
      "x-ratelimit-remaining-tokens": "500",
      "retry-after": "2",
    }, content=payload)

  async def collect():
    provider = MuseResponsesProvider()
    client = provider.create_client({"api_key": "test-token", "_transport": httpx.MockTransport(handle)})
    try:
      return [event async for event in provider.stream(client, {"model": "muse-fixture"})]
    finally:
      await provider.close_client(client)

  events = asyncio.run(collect())
  start = next(event for event in events if event.type == "message_start")
  end = next(event for event in events if event.type == "message_end")
  usage = next(event for event in events if event.type == "usage_update")
  thinking = next(event for event in events if event.type == "thinking_end")
  assert (start.provider_response_id, start.provider_request_id) == ("resp_muse", "mreq_1")
  assert start.rate_limit == {"x-ratelimit-remaining-tokens": "500", "retry-after": "2"}
  assert (end.provider_response_id, end.service_tier) == ("resp_muse", "default")
  assert usage.reasoning_tokens == 6
  assert json.loads(thinking.raw_block["signature"])["item"]["encrypted_content"] == "ENC"
  assert thinking.raw_block["thinking"] == "thought"
