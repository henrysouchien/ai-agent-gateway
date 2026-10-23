from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from agent_gateway.auth import AuthConfig, ProviderCredentialFailure
from agent_gateway.providers import installed_adapter_route_support
from agent_gateway.providers import base as provider_base
from agent_gateway.providers.google_antigravity import GoogleAntigravityProvider
from agent_gateway.providers.ollama_cloud import OllamaCloudProvider
from gateway_test_support.model_defaults import compat_for_profile
from model_authority.capabilities import GATEWAY_EXECUTED_CAPABILITY_IDS
from model_authority.registry import ModelRegistryEntry, ProductModelRegistry
from model_authority.schema import SCHEMA
from model_authority.thinking import ThinkingLevel


_PROVIDERS = (
  GoogleAntigravityProvider,
  OllamaCloudProvider,
)

# One gateway-executed entry per installed subscription route. Protocol
# profiles are the ones each adapter declares; admission refuses any other
# combination.
_ADMITTED_ROUTES = (
  (
    "google_antigravity.gemini-fixture",
    "google_antigravity",
    "gemini-fixture",
    "google_antigravity.code_assist",
    "generate_content.standard",
    "google.antigravity",
  ),
  (
    "ollama_cloud.qwen-fixture",
    "ollama_cloud",
    "qwen-fixture",
    "ollama_cloud.chat_completions",
    "chat_completions.standard",
    "ollama.cloud",
  ),
)

_ANTIGRAVITY_MODEL = "gemini-fixture"
_OLLAMA_MODEL = "qwen-fixture"
_TOKEN = "test-token"
_PROJECT = "test-project"
_API_KEY = "test-api-key"
_TEXT = "hi"
_VISIBLE = "visible-answer"
_HIDDEN_THOUGHT = "hidden-thought"
_HIDDEN_REASON = "hidden-reason"
_TOP_THINKING = "top-thinking"
_TOP_REASONING = "top-reasoning"
_OUTPUT_CAP = 65_536
_ANTIGRAVITY_SSE_LINE = (
  'data: {"response": {"candidates": [{"content": {"parts": [{"text": "hi"}]}, '
  '"finishReason": "STOP"}]}}'
)


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


def _registry() -> ProductModelRegistry:
  entries = [_entry(*row) for row in _ADMITTED_ROUTES]
  return ProductModelRegistry(
    schema=SCHEMA,
    revision="subscription-transport-test",
    models={entry.key: entry for entry in entries},
  )


def _json_error(message: str, **fields: str) -> bytes:
  return json.dumps({"error": {"message": message, **fields}}).encode()


def _sse(*events: dict[str, Any]) -> bytes:
  return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


def _active_config(provider_name: str) -> dict[str, object]:
  if provider_name == "google_antigravity":
    return {"auth_token": _TOKEN, "project_id": _PROJECT}
  if provider_name == "ollama_cloud":
    return {"api_key": _API_KEY}
  return {}


def _request_json(request: httpx.Request) -> dict[str, object]:
  parsed = json.loads(request.content)
  assert isinstance(parsed, dict)
  return {str(key): value for key, value in parsed.items()}


async def _stream_status(
  provider: Any,
  *,
  status: int,
  body: bytes,
  headers: dict[str, str] | None = None,
  model: str = _ANTIGRAVITY_MODEL,
  config: dict[str, object] | None = None,
) -> Exception:
  """The exception ``stream`` raises for a non-200, headers included."""

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


def _classify(provider_cls: type, **kwargs: Any) -> ProviderCredentialFailure | None:
  provider = provider_cls()
  exc = asyncio.run(_stream_status(provider, **kwargs))
  assert isinstance(exc, httpx.HTTPStatusError)
  return provider.classify_credential_failure(exc)


def test_fixture_routes_are_admitted() -> None:
  registry = _registry()
  supports = installed_adapter_route_support()
  registry.admit_adapter_support(
    supports,
    executed_capability_ids=GATEWAY_EXECUTED_CAPABILITY_IDS,
  )

  assert len(registry.models) == len(_ADMITTED_ROUTES)
  for entry in registry.models.values():
    support = supports[entry.adapter]
    assert support.supports(entry)
    assert entry.route in support.routes


@pytest.mark.parametrize(
  "body",
  [
    _json_error("slow down"),
    _json_error("Quota exceeded", status="RESOURCE_EXHAUSTED"),
  ],
  ids=["plain", "quota-exhausted"],
)
@pytest.mark.parametrize("provider_cls", _PROVIDERS, ids=["google_antigravity", "ollama_cloud"])
def test_429_is_rate_limit_with_retry_after(provider_cls: type, body: bytes) -> None:
  # "quota" / RESOURCE_EXHAUSTED is a billing word in the generic classifier.
  # On 429 it stays a rate limit, and Retry-After is the reset.
  before = time.time()
  failure = _classify(
    provider_cls,
    status=429,
    body=body,
    headers={"Retry-After": "30"},
  )

  assert failure is not None
  assert failure.kind == "rate_limit"
  assert failure.reset_at == pytest.approx(before + 30, abs=2)


@pytest.mark.parametrize("provider_cls", _PROVIDERS, ids=["google_antigravity", "ollama_cloud"])
def test_401_is_auth(provider_cls: type) -> None:
  failure = _classify(provider_cls, status=401, body=_json_error("invalid credential"))

  assert failure is not None
  assert failure.kind == "auth"


@pytest.mark.parametrize("provider_cls", _PROVIDERS, ids=["google_antigravity", "ollama_cloud"])
def test_403_is_billing(provider_cls: type) -> None:
  failure = _classify(provider_cls, status=403, body=_json_error("forbidden"))

  assert failure is not None
  assert failure.kind == "billing"


@pytest.mark.parametrize("provider_cls", _PROVIDERS, ids=["google_antigravity", "ollama_cloud"])
def test_400_is_not_a_credential_failure(provider_cls: type) -> None:
  # The phrase would be a rate limit on a 429. Status 400 is not a hop.
  failure = _classify(
    provider_cls,
    status=400,
    body=_json_error("too many requests", status="RESOURCE_EXHAUSTED"),
  )

  assert failure is None


@pytest.mark.parametrize(
  ("status", "retryable"),
  [(503, True), (429, False)],
  ids=["503", "429"],
)
@pytest.mark.parametrize("provider_cls", _PROVIDERS, ids=["google_antigravity", "ollama_cloud"])
def test_status_retryability(provider_cls: type, status: int, retryable: bool) -> None:
  provider = provider_cls()
  exc = asyncio.run(_stream_status(provider, status=status, body=_json_error("unavailable")))

  assert isinstance(exc, httpx.HTTPStatusError)
  assert provider.is_retryable_error(exc) is retryable


@pytest.mark.parametrize("provider_cls", _PROVIDERS, ids=["google_antigravity", "ollama_cloud"])
def test_connect_error_is_retryable(provider_cls: type) -> None:
  assert provider_cls().is_retryable_error(httpx.ConnectError("connection refused")) is True


_IN_BAND_400 = {"error": {"code": 400, "message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"}}
_PROGRESS_CHUNK = {"response": {"candidates": [{"content": {"role": "model", "parts": [{"text": "partial"}]}}]}}


def test_antigravity_in_band_400_before_any_progress_is_the_request_answer() -> None:
  provider = GoogleAntigravityProvider()
  exc = asyncio.run(_stream_status(provider, status=200, body=_sse(_IN_BAND_400)))

  assert exc.__class__.__name__ == "_GeminiStatusError"
  assert provider.is_retryable_error(exc) is False


def test_antigravity_in_band_400_after_progress_is_a_dropped_attempt() -> None:
  # PAYC :8256 2026-10-13: 9.8 s of streamed thinking, then INVALID_ARGUMENT;
  # the byte-identical request replayed 200. The attempt is retried, not killed.
  provider = GoogleAntigravityProvider()
  exc = asyncio.run(_stream_status(provider, status=200, body=_sse(_PROGRESS_CHUNK, _IN_BAND_400)))

  assert exc.__class__.__name__ == "_GeminiStatusError"
  assert provider.is_retryable_error(exc) is True
  assert provider.classify_credential_failure(exc) is None


def test_antigravity_credential_requires_token_and_project() -> None:
  provider = GoogleAntigravityProvider()
  both: dict[str, object] = {"auth_token": _TOKEN, "project_id": _PROJECT}
  token_only: dict[str, object] = {"auth_token": _TOKEN}
  project_only: dict[str, object] = {"project_id": _PROJECT}
  blank_token: dict[str, object] = {"auth_token": "  ", "project_id": _PROJECT}
  blank_project: dict[str, object] = {"auth_token": _TOKEN, "project_id": "  "}

  assert provider.has_active_credential(both) is True
  assert provider.has_active_credential(token_only) is False
  assert provider.has_active_credential(project_only) is False
  assert provider.has_active_credential(blank_token) is False
  assert provider.has_active_credential(blank_project) is False
  assert provider.has_active_credential({}) is False


def test_ollama_cloud_credential_is_api_key() -> None:
  provider = OllamaCloudProvider()
  keyed: dict[str, object] = {"api_key": _API_KEY}
  blank: dict[str, object] = {"api_key": " "}
  other: dict[str, object] = {"auth_token": _TOKEN}

  assert provider.has_active_credential(keyed) is True
  assert provider.has_active_credential({}) is False
  assert provider.has_active_credential(blank) is False
  assert provider.has_active_credential(other) is False


def test_antigravity_create_client_without_project_id_names_login() -> None:
  provider = GoogleAntigravityProvider()
  missing: dict[str, object] = {"auth_token": _TOKEN}
  blank: dict[str, object] = {"auth_token": _TOKEN, "project_id": "  "}

  for config in (missing, blank):
    with pytest.raises(RuntimeError, match="agent auth login"):
      provider.create_client(config)


def test_antigravity_stream_wire(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(provider_base, "INITIAL_MODEL_REGISTRY", _registry())
  seen: dict[str, httpx.Request] = {}

  def handle(request: httpx.Request) -> httpx.Response:
    seen["request"] = request
    return httpx.Response(
      200,
      headers={"content-type": "text/event-stream"},
      content=f"{_ANTIGRAVITY_SSE_LINE}\n\n".encode(),
    )

  async def collect() -> list[Any]:
    provider = GoogleAntigravityProvider()
    config: dict[str, object] = {
      "auth_token": _TOKEN,
      "project_id": _PROJECT,
      "_transport": httpx.MockTransport(handle),
    }
    client = provider.create_client(config)
    try:
      info = provider.get_model_info(_ANTIGRAVITY_MODEL)
      effort = provider.resolve_effort(
        requested=ThinkingLevel.NONE,
        model=_ANTIGRAVITY_MODEL,
        model_info=info,
        max_tokens=16,
      )
      params = provider.build_request_params(
        model=_ANTIGRAVITY_MODEL,
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
  body = _request_json(request)
  inner = body["request"]

  assert request.method == "POST"
  assert request.url.path == "/v1internal:streamGenerateContent"
  assert request.url.params["alt"] == "sse"
  assert request.headers["authorization"] == f"Bearer {_TOKEN}"
  assert body["project"] == _PROJECT
  assert isinstance(inner, dict)
  assert "contents" in inner
  contents = inner["contents"]
  assert isinstance(contents, list) and contents
  assert "hello" in json.dumps(contents)
  assert any(
    event.type in {"text_delta", "text_end"} and event.text == _TEXT
    for event in events
  )
  assert any(
    event.type == "message_end" and event.stop_reason == "end_turn"
    for event in events
  )


def test_ollama_cloud_stream_wire(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(provider_base, "INITIAL_MODEL_REGISTRY", _registry())
  seen: dict[str, httpx.Request] = {}

  def handle(request: httpx.Request) -> httpx.Response:
    seen["request"] = request
    return httpx.Response(
      200,
      headers={"content-type": "text/event-stream"},
      content=_sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}),
    )

  async def collect() -> None:
    provider = OllamaCloudProvider()
    config: dict[str, object] = {
      "api_key": _API_KEY,
      "_transport": httpx.MockTransport(handle),
    }
    client = provider.create_client(config)
    try:
      info = provider.get_model_info(_OLLAMA_MODEL)
      effort = provider.resolve_effort(
        requested=ThinkingLevel.NONE,
        model=_OLLAMA_MODEL,
        model_info=info,
        max_tokens=16,
      )
      params = provider.build_request_params(
        model=_OLLAMA_MODEL,
        messages=[
          {"role": "user", "content": "hello"},
          {
            "role": "assistant",
            "content": [
              {"type": "thinking", "thinking": _HIDDEN_THOUGHT},
              {"type": "reasoning", "text": _HIDDEN_REASON},
              {"type": "text", "text": _VISIBLE},
            ],
            "thinking": _TOP_THINKING,
            "reasoning": _TOP_REASONING,
            "reasoning_content": _HIDDEN_REASON,
          },
          {"role": "user", "content": "continue"},
        ],
        system_prompt=None,
        tools=[],
        max_tokens=_OUTPUT_CAP + 4096,
        effort_resolution=effort,
      )
      async for _event in provider.stream(client, params):
        pass
    finally:
      await provider.close_client(client)

  asyncio.run(collect())
  request = seen["request"]
  body = _request_json(request)
  raw = request.content.decode()
  messages = body["messages"]

  assert request.method == "POST"
  assert request.url.path == "/v1/chat/completions"
  assert request.headers["authorization"] == f"Bearer {_API_KEY}"
  assert isinstance(messages, list)
  for message in messages:
    assert isinstance(message, dict)
    assert "thinking" not in message
    assert "reasoning" not in message
    assert "reasoning_content" not in message
  # Omitted, not demoted into content. A strip that only drops the key still
  # fails if the thinking text is copied onto the wire.
  assert _HIDDEN_THOUGHT not in raw
  assert _HIDDEN_REASON not in raw
  assert _TOP_THINKING not in raw
  assert _TOP_REASONING not in raw
  assert _VISIBLE in raw
  assert body["max_completion_tokens"] == _OUTPUT_CAP


@pytest.mark.parametrize("provider", ["google_antigravity", "ollama_cloud"])
def test_auth_config_accepts_provider(provider: str) -> None:
  config = AuthConfig.from_dict({"provider": provider, "billing_mode": "byok"})

  assert config.provider == provider


def test_antigravity_tool_schemas_are_reduced_to_the_wire_subset() -> None:
  """Code Assist 400s on `$defs`/`$ref`, list `type`, `const`,
  `exclusiveMinimum` and `uniqueItems` (live 2026-10-11, 31 violations over
  11 tools). The adapter inlines refs and drops what the Schema proto has no
  field for, instead of passing `input_schema` through."""
  from agent_gateway.providers.google_antigravity import _convert_tools

  tools = [{
    "name": "probe",
    "description": "probe tool",
    "input_schema": {
      "$defs": {
        "Item": {"type": "object", "properties": {"id": {"type": "string"}}},
      },
      "type": "object",
      "properties": {
        "item": {"$ref": "#/$defs/Item"},
        "maybe": {"type": ["string", "null"]},
        "mode": {"const": "fast", "exclusiveMinimum": 0, "uniqueItems": True},
        "flag": {"const": True, "type": "boolean"},
        "level": {"enum": ["low", 3]},
      },
      "required": ["item"],
    },
  }]

  (block,) = _convert_tools(tools)
  (declaration,) = block["functionDeclarations"]
  parameters = declaration["parameters"]
  dumped = json.dumps(parameters)

  assert "$defs" not in dumped
  assert "$ref" not in dumped
  assert "exclusiveMinimum" not in dumped
  assert "uniqueItems" not in dumped
  assert parameters["properties"]["item"] == {
    "type": "object",
    "properties": {"id": {"type": "string"}},
  }
  assert parameters["properties"]["maybe"] == {"type": "string", "nullable": True}
  assert parameters["properties"]["mode"] == {"enum": ["fast"]}
  # The wire's enum is strings only: a boolean const keeps its type and loses
  # the enum, and a mixed enum keeps its strings.
  assert parameters["properties"]["flag"] == {"type": "boolean"}
  assert parameters["properties"]["level"] == {"enum": ["low"]}
  assert parameters["required"] == ["item"]


def test_both_gemini_adapters_reduce_the_fms_persist_shape_to_the_wire_subset() -> None:
  """Live 2026-10-12: Code Assist 400'd on ``patternProperties`` from the
  fms_persist forecast-set declarations eleven turns in, after the first
  sanitizer landed. The shape below is that declaration's period map
  (api/agent/shared/tool_defs/fms_persist.py ``_FORECAST_SET_*``), inlined so the
  package test runs without the product checkout. Both google.generate_content
  adapters share one reducer, so the public wire cannot regress separately."""
  from agent_gateway.providers import google as google_public
  from agent_gateway.providers.google_tool_schema import WIRE_DROPPED_SCHEMA_KEYS
  from agent_gateway.providers.google_antigravity import _convert_tools as antigravity_convert

  period_map = {
    "type": "object",
    "patternProperties": {
      r"^[12][0-9]{3}$": {
        "type": "array",
        "items": {"type": "string", "minLength": 1},
        "minItems": 1,
      },
    },
    "additionalProperties": False,
    "minProperties": 1,
  }
  tools = [{
    "name": "fms_set_forecast",
    "description": "persist a forecast set",
    "input_schema": {
      "type": "object",
      "properties": {"values": period_map, "refs": {"$ref": "#/$defs/Map"}},
      "$defs": {"Map": period_map},
      "required": ["values"],
    },
  }]

  def _keys(node: Any, found: set[str]) -> None:
    if isinstance(node, dict):
      for key, value in node.items():
        found.add(key)
        _keys(value, found)
    elif isinstance(node, list):
      for value in node:
        _keys(value, found)

  for convert in (google_public._convert_tools, antigravity_convert):
    (block,) = convert(tools)
    (declaration,) = block["functionDeclarations"]
    seen: set[str] = set()
    _keys(declaration["parameters"], seen)
    offenders = seen & (WIRE_DROPPED_SCHEMA_KEYS | {"$ref", "const"})
    assert not offenders, f"{convert.__module__} leaves wire-rejected keys: {sorted(offenders)}"
    # The meaningful shape survives: the period map is still a typed object
    # whose array item type is intact.
    refs = declaration["parameters"]["properties"]["refs"]
    assert refs["type"] == "object"
    assert refs["minProperties"] == 1



def test_ollama_cloud_monthly_limit_429_without_retry_after_sets_24h_cooldown(
  tmp_path: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway.providers.credential_pool import (
    get_provider_credential_pool,
    upsert_pool_account,
  )
  from agent_gateway.providers.ollama_cloud import _MONTHLY_LIMIT_COOLDOWN_SECONDS

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  store_path = tmp_path / "ollama_cloud" / "credentials.json"
  upsert_pool_account(
    store_path,
    {"identity": "ollama_cloud:monthly", "api_key": _API_KEY},
    default_prefix="ollama_cloud",
    required_any_keys=("api_key",),
  )
  pool = get_provider_credential_pool("ollama_cloud")
  pool.clear()

  before = time.time()
  failure = _classify(
    OllamaCloudProvider,
    status=429,
    body=_json_error(
      "you (user) have reached your monthly usage limit, add usage credits: https://ollama.com/settings"
    ),
  )

  assert failure is not None
  assert failure.kind == "rate_limit"
  assert failure.status_code == 429
  assert failure.reset_at is not None
  assert failure.reset_at == pytest.approx(before + _MONTHLY_LIMIT_COOLDOWN_SECONDS, abs=5)

  rotated = OllamaCloudProvider().next_credential(
    {
      "provider": "ollama_cloud",
      "auth_mode": "api",
      "api_key": _API_KEY,
      "account_identity": "ollama_cloud:monthly",
      "auth_store_path": str(store_path),
    },
    failure,
  )
  assert rotated is None
  assert pool.blocked_until("ollama_cloud:monthly") == pytest.approx(
    failure.reset_at, abs=1,
  )


def test_ollama_cloud_compacted_and_empty_assistant_turns_never_serialize_nil_content(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(provider_base, "INITIAL_MODEL_REGISTRY", _registry())
  provider = OllamaCloudProvider()
  info = provider.get_model_info(_OLLAMA_MODEL)
  effort = provider.resolve_effort(
    requested=ThinkingLevel.NONE,
    model=_OLLAMA_MODEL,
    model_info=info,
    max_tokens=16,
  )
  params = provider.build_request_params(
    model=_OLLAMA_MODEL,
    messages=[
      {"role": "user", "content": "covered turn"},
      {
        "role": "assistant",
        "content": [{"type": "compaction", "content": "Compacted prefix summary"}],
        "stop_reason": "compaction",
      },
      {
        "role": "assistant",
        "content": [
          {"type": "thinking", "thinking": "planning call"},
          {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "pcty"}},
        ],
      },
      {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}],
      },
      {"role": "assistant", "content": [], "stop_reason": "end_turn"},
      {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "thinking only"}],
        "stop_reason": "end_turn",
      },
      {"role": "user", "content": "continue"},
    ],
    system_prompt="sys",
    tools=[],
    max_tokens=256,
    effort_resolution=effort,
  )
  wire = params["messages"]
  assert all(isinstance(m.get("content"), str) for m in wire)
  assert [m["role"] for m in wire] == ["system", "user", "assistant", "tool", "user"]
  assert "Compacted prefix summary" in wire[1]["content"]
  assert wire[2]["content"] == ""
  assert wire[2]["tool_calls"][0]["id"] == "call_1"


def test_ollama_tool_call_accumulator_preserves_repeated_brace_prefix_deltas() -> None:
  from agent_gateway.providers.ollama import _ToolCallAccumulator

  acc = _ToolCallAccumulator(0)
  deltas = [
    '{"',
    'judgment": ',
    '{"',
    'company": ',
    '{',
    '"name": "PCTY"',
    '}}}',
  ]
  for i, delta in enumerate(deltas):
    raw = {
      "id": "call_1" if i == 0 else "",
      "function": {"name": "fms_persist_business_model" if i == 0 else "", "arguments": delta},
    }
    emitted = acc.apply(raw)
    assert emitted == delta
  assert acc.arguments == '{"judgment": {"company": {"name": "PCTY"}}}'
