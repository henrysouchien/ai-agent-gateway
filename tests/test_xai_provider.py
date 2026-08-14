from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from agent_gateway import AuthConfig, OpenAIProvider, XAIProvider
from agent_gateway._provider_utils import _resolve_provider, resolve_auth_config
from agent_gateway.providers.base import ThinkingLevel
from agent_gateway.providers.xai_helpers import _ResponsesStreamState, map_event, resolve_responses_url
from agent_gateway.providers.xai_oauth import (
  DEFAULT_XAI_OAUTH_CLIENT_ID,
  DEFAULT_XAI_OAUTH_SCOPE,
  save_xai_token_record,
)
from agent_gateway.model_registry import INITIAL_MODEL_REGISTRY
from api.credentials import get_xai_config


def test_xai_is_first_class_provider_not_openai_subclass() -> None:
  provider = XAIProvider()
  assert provider.name == "xai"
  assert not isinstance(provider, OpenAIProvider)
  assert OpenAIProvider not in inspect.getmro(XAIProvider)


def test_auth_config_accepts_xai_credential_family() -> None:
  auth = AuthConfig.from_dict({"provider": "xai", "billing_mode": "byok", "api_key": "xai-test"})
  assert auth.provider == "xai"


def test_resolver_uses_xai_env_api_key(monkeypatch) -> None:
  monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
  provider, provider_name, config = _resolve_provider(
    "xai",
    "grok-4.5",
    None,
    None,
    None,
  )
  assert isinstance(provider, XAIProvider)
  assert provider_name == "xai"
  assert config["api_key"] == "xai-test-key"
  assert {
    "model",
    "model_key",
    "effort",
    "thinking",
    "thinking_enabled_requested",
  }.isdisjoint(config)
  assert provider.has_active_credential(config)


def test_provider_does_not_fall_back_to_process_api_key(monkeypatch) -> None:
  monkeypatch.setenv("XAI_API_KEY", "process-service-secret")
  provider = XAIProvider()
  config = {"auth_mode": "api", "api_key": ""}

  assert provider.has_active_credential(config) is False
  with pytest.raises(RuntimeError, match="No xAI api credential configured"):
    provider.create_client(config)


def test_client_route_does_not_fall_back_to_process_environment(monkeypatch) -> None:
  monkeypatch.setenv("XAI_BASE_URL", "https://ambient.invalid/v1")
  provider = XAIProvider()

  client = provider.create_client({
    "auth_mode": "api",
    "api_key": "bound-api-key",
  })

  assert provider._client_state[client]["endpoint_url"] == (
    "https://api.x.ai/v1/responses"
  )
  asyncio.run(provider.close_client(client))


@pytest.mark.parametrize("key", ["base_url", "baseURL"])
def test_client_route_uses_bound_base_url_alias(key: str) -> None:
  provider = XAIProvider()

  client = provider.create_client({
    "auth_mode": "api",
    "api_key": "bound-api-key",
    key: "https://bound.xai.example/v1/",
  })

  assert provider._client_state[client]["endpoint_url"] == (
    "https://bound.xai.example/v1/responses"
  )
  asyncio.run(provider.close_client(client))


def test_explicit_oauth_access_and_refresh_material_remain_active(tmp_path) -> None:
  provider = XAIProvider()
  config = {
    "auth_mode": "oauth",
    "auth_token": "access-1",
    "refresh_token": "refresh-1",
    "auth_store_path": str(tmp_path / "oauth.json"),
  }

  assert provider.has_active_credential(config) is True
  client = provider.create_client(config)
  state = provider._client_state[client]
  assert state["token"] == "access-1"
  assert state["oauth_record"]["refresh_token"] == "refresh-1"
  assert state["oauth_settings"].store_path == tmp_path / "oauth.json"
  asyncio.run(provider.close_client(client))


def test_explicit_oauth_token_does_not_adopt_process_or_default_store_refresh(
  monkeypatch,
  tmp_path,
) -> None:
  user_data_dir = tmp_path / "service-state"
  store_path = user_data_dir / "xai" / "oauth.json"
  save_xai_token_record(
    store_path,
    {
      "access_token": "service-access",
      "refresh_token": "service-refresh",
      "expires_at": 4_000_000_000,
      "scope": DEFAULT_XAI_OAUTH_SCOPE,
      "issuer": "https://auth.x.ai",
      "client_id": DEFAULT_XAI_OAUTH_CLIENT_ID,
    },
  )
  monkeypatch.setenv("USER_DATA_DIR", str(user_data_dir))
  monkeypatch.setenv("XAI_AUTH_TOKEN", "process-service-access")
  monkeypatch.setenv("XAI_REFRESH_TOKEN", "process-service-refresh")

  provider = XAIProvider()
  client = provider.create_client(
    {
      "auth_mode": "oauth",
      "auth_token": "user-access",
    }
  )
  state = provider._client_state[client]

  assert state["token"] == "user-access"
  assert state["oauth_record"]["refresh_token"] == ""
  assert state["oauth_settings"] is None
  asyncio.run(provider.close_client(client))


def test_xai_config_has_no_selection_defaults_and_registry_owns_models(monkeypatch) -> None:
  for name in ("XAI_MODEL", "XAI_EFFORT", "XAI_THINKING", "ALLOWED_MODELS_XAI"):
    monkeypatch.delenv(name, raising=False)
  config = get_xai_config()
  assert config["base_url"] == "https://api.x.ai/v1"
  assert {
    "model",
    "model_key",
    "effort",
    "thinking",
    "thinking_enabled_requested",
  }.isdisjoint(config)
  xai_models = {
    entry.upstream_model: entry.label
    for entry in INITIAL_MODEL_REGISTRY.models.values()
    if entry.provider == "xai"
    and entry.capabilities.get("session.driver") == "user_selectable"
  }
  assert xai_models == {"grok-4.5": "Grok 4.5"}


def test_xai_effort_and_thinking_env_are_not_auth_authority(monkeypatch) -> None:
  monkeypatch.setenv("XAI_EFFORT", "medium")
  monkeypatch.setenv("XAI_THINKING", "false")
  config = resolve_auth_config(provider="xai")
  assert {
    "model",
    "model_key",
    "effort",
    "thinking",
    "thinking_enabled_requested",
  }.isdisjoint(config)


@pytest.mark.parametrize(
  ("model", "requested", "effective", "enabled", "payload"),
  [
    ("grok-4.5", ThinkingLevel.LOW, ThinkingLevel.LOW, True, {"reasoning": {"effort": "low"}}),
    ("grok-4.5", ThinkingLevel.MEDIUM, ThinkingLevel.MEDIUM, True, {"reasoning": {"effort": "medium"}}),
    ("grok-4.5", ThinkingLevel.HIGH, ThinkingLevel.HIGH, True, {"reasoning": {"effort": "high"}}),
    ("grok-4.3", ThinkingLevel.NONE, ThinkingLevel.NONE, False, {"reasoning": {"effort": "none"}}),
    ("grok-4.3-fast", ThinkingLevel.HIGH, ThinkingLevel.HIGH, True, {"reasoning": {"effort": "high"}}),
  ],
)
def test_effort_resolution_preserves_supported_effort_exactly(
  model, requested, effective, enabled, payload
) -> None:
  provider = XAIProvider()
  resolution = provider.resolve_effort(
    requested=requested,
    model=model,
    model_info=provider.get_model_info(model),
    max_tokens=16_000,
  )
  assert resolution.requested == requested
  assert resolution.effective == effective
  assert resolution.thinking_enabled_effective is enabled
  assert dict(resolution.payload_fragments) == payload


@pytest.mark.parametrize(
  ("model", "requested"),
  [
    # Formerly silently clamped/upgraded; unsupported effort now refuses.
    ("grok-4.5", ThinkingLevel.NONE),
    ("grok-4.5", ThinkingLevel.MINIMAL),
    ("grok-4.5", ThinkingLevel.XHIGH),
    ("grok-4.5", ThinkingLevel.MAX),
    ("grok-4.3-fast", ThinkingLevel.MAX),
    ("grok-build-0.1", ThinkingLevel.HIGH),
    ("grok-4.20-beta-latest-non-reasoning", ThinkingLevel.HIGH),
  ],
)
def test_unsupported_effort_is_refused_never_clamped(model, requested) -> None:
  provider = XAIProvider()
  with pytest.raises(ValueError, match="refused"):
    provider.resolve_effort(
      requested=requested,
      model=model,
      model_info=provider.get_model_info(model),
      max_tokens=16_000,
    )


def _admitted_effort_resolution(
  provider: XAIProvider,
  model: str,
  requested: ThinkingLevel,
) -> object:
  return provider.resolve_effort(
    requested=requested,
    model=model,
    model_info=provider.get_model_info(model),
    max_tokens=16_000,
  )


def test_build_request_refuses_without_admitted_effort_resolution() -> None:
  provider = XAIProvider()
  with pytest.raises(ValueError, match="EffortResolution"):
    provider.build_request_params(
      model="grok-4.5",
      messages=[{"role": "user", "content": "hi"}],
      system_prompt=None,
      tools=[],
      max_tokens=2048,
      thinking_level=ThinkingLevel.HIGH,
    )


def test_build_request_uses_responses_local_history_and_tools() -> None:
  provider = XAIProvider()
  params = provider.build_request_params(
    model="grok-4.5",
    messages=[{"role": "user", "content": "Use the calculator"}],
    system_prompt="Be precise.",
    tools=[{"name": "calculator", "description": "Calculate", "input_schema": {"type": "object"}}],
    max_tokens=2048,
    thinking_level=ThinkingLevel.LOW,
    effort_resolution=_admitted_effort_resolution(
      provider, "grok-4.5", ThinkingLevel.LOW
    ),
  )
  assert params["store"] is False
  assert params["stream"] is True
  assert "previous_response_id" not in params
  assert params["include"] == ["reasoning.encrypted_content"]
  assert params["reasoning"] == {"effort": "low"}
  assert params["tools"][0]["name"] == "calculator"
  assert params["input"][0]["content"][0]["text"] == "Use the calculator"


def test_encrypted_reasoning_is_replayed_from_local_history() -> None:
  provider = XAIProvider()
  reasoning_item = {
    "type": "reasoning",
    "id": "rs_123",
    "summary": [],
    "encrypted_content": "encrypted-xai-reasoning",
  }
  params = provider.build_request_params(
    model="grok-4.5",
    messages=[
      {
        "role": "assistant",
        "provider": "xai",
        "model": "grok-4.5",
        "content": [
          {
            "type": "thinking",
            "thinking": "",
            "signature": json.dumps(reasoning_item),
          }
        ],
      },
      {"role": "user", "content": "Continue"},
    ],
    system_prompt=None,
    tools=[],
    max_tokens=1024,
    effort_resolution=_admitted_effort_resolution(
      provider, "grok-4.5", ThinkingLevel.LOW
    ),
  )
  assert reasoning_item in params["input"]
  assert "previous_response_id" not in params
  assert "tool_choice" not in params
  assert "parallel_tool_calls" not in params


def test_responses_url_uses_xai_v1_default() -> None:
  assert resolve_responses_url(None) == "https://api.x.ai/v1/responses"
  assert resolve_responses_url("https://proxy.example/v1/") == "https://proxy.example/v1/responses"


def test_whole_chunk_tool_call_maps_full_tool_lifecycle() -> None:
  state = _ResponsesStreamState()
  added = map_event(
    {
      "type": "response.output_item.added",
      "item": {
        "type": "function_call",
        "id": "fc_123",
        "call_id": "call_123",
        "name": "lookup",
        "arguments": '{"ticker":"XAI"}',
      },
    },
    state,
  )
  done = map_event(
    {
      "type": "response.output_item.done",
      "item": {
        "type": "function_call",
        "id": "fc_123",
        "call_id": "call_123",
        "name": "lookup",
        "arguments": '{"ticker":"XAI"}',
      },
    },
    state,
  )
  completed = map_event(
    {
      "type": "response.completed",
      "response": {"status": "completed", "usage": {"input_tokens": 3, "output_tokens": 2}},
    },
    state,
  )
  assert [event.type for event in added + done] == ["tool_use_start", "tool_use_delta", "tool_use_end"]
  assert done[0].tool_input == {"ticker": "XAI"}
  assert completed[-1].type == "message_end"
  assert completed[-1].stop_reason == "tool_use"


def test_single_done_event_whole_tool_call_maps_full_lifecycle() -> None:
  state = _ResponsesStreamState()
  events = map_event(
    {
      "type": "response.output_item.done",
      "item": {
        "type": "function_call",
        "id": "fc_456",
        "call_id": "call_456",
        "name": "lookup",
        "arguments": '{"ticker":"GROK"}',
      },
    },
    state,
  )
  assert [event.type for event in events] == ["tool_use_start", "tool_use_delta", "tool_use_end"]
  assert events[-1].tool_input == {"ticker": "GROK"}


def test_error_event_identifies_xai() -> None:
  with pytest.raises(RuntimeError, match="xAI error"):
    map_event({"type": "error", "message": "bad request"}, _ResponsesStreamState())


@pytest.mark.parametrize(
  ("model_id", "expected_window", "expected_trigger"),
  [
    # docs.x.ai values (audited 2026-07-21); grok-latest is an undocumented
    # alias pinned conservatively to grok-4.5's 500k.
    ("grok-4.5", 500_000, 400_000),
    ("grok-4.5-latest", 500_000, 400_000),
    ("grok-latest", 500_000, 400_000),
    ("grok-build-0.1", 256_000, 204_800),
    ("grok-4.3", 1_000_000, 800_000),
    ("grok-4.20-beta-latest-reasoning", 1_000_000, 800_000),
  ],
)
def test_xai_windows_match_official_docs(model_id: str, expected_window: int, expected_trigger: int) -> None:
  from agent_gateway.runner_limits import effective_compaction_trigger

  provider = XAIProvider()
  info = provider.get_model_info(model_id)

  assert info.context_window == expected_window
  assert effective_compaction_trigger(160_000, info) == expected_trigger


def test_xai_reasoning_summary_streams_what_it_persists() -> None:
  """xAI delegates reasoning mapping to codex_helpers._map_event.

  There is no xAI-specific reasoning handler -- xai_helpers imports the codex mapper and
  forwards every non-error event to it -- so a separator change in that mapper is an xAI
  behavior change too. This test exists so that fact stays visible: the streamed thinking
  text and the durable block must agree here for the same reason they must in codex.
  """
  item = {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC"}
  events: list[dict] = [{"type": "response.output_item.added", "item": item}]
  for index, text in enumerate(("first", "second")):
    events.append({"type": "response.reasoning_summary_part.added", "item_id": "rs_1",
                   "summary_index": index, "part": {"type": "summary_text", "text": ""}})
    events.append({"type": "response.reasoning_summary_text.delta", "item_id": "rs_1",
                   "summary_index": index, "delta": text})
    events.append({"type": "response.reasoning_summary_part.done", "item_id": "rs_1",
                   "summary_index": index, "part": {"type": "summary_text", "text": text}})
  events.append({"type": "response.output_item.done",
                 "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC",
                          "summary": [{"type": "summary_text", "text": "first"},
                                      {"type": "summary_text", "text": "second"}]}})

  state = _ResponsesStreamState()
  streamed: list[str] = []
  durable = None
  for raw in events:
    for mapped in map_event(raw, state):
      if mapped.type == "thinking_delta":
        streamed.append(mapped.thinking_text or "")
      elif mapped.type == "thinking_end":
        durable = mapped.thinking_text or ""

  assert durable == "first\n\nsecond"
  assert "".join(streamed) == durable
