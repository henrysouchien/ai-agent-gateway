from __future__ import annotations

import inspect
import json

import pytest

from agent_gateway import AuthConfig, OpenAIProvider, XAIProvider
from agent_gateway._provider_utils import _resolve_provider, resolve_auth_config
from agent_gateway.providers.base import ThinkingLevel
from agent_gateway.providers.xai_helpers import _ResponsesStreamState, map_event, resolve_responses_url
from api.agent.shared.model_resolution import get_allowed_models, get_model_display_names
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
  provider, provider_name, config = _resolve_provider("xai", None, None, None, None)
  assert isinstance(provider, XAIProvider)
  assert provider_name == "xai"
  assert config["model"] == "grok-4.5"
  assert provider.has_active_credential(config)


def test_xai_config_and_catalog_defaults(monkeypatch) -> None:
  for name in ("XAI_MODEL", "XAI_EFFORT", "XAI_THINKING", "ALLOWED_MODELS_XAI"):
    monkeypatch.delenv(name, raising=False)
  config = get_xai_config()
  assert config["model"] == "grok-4.5"
  assert config["base_url"] == "https://api.x.ai/v1"
  assert {"grok-4.5", "grok-4.3"} <= get_allowed_models("xai")
  assert get_model_display_names("xai")["grok-4.5"] == "Grok 4.5"


def test_xai_effort_and_thinking_env_must_agree(monkeypatch) -> None:
  monkeypatch.setenv("XAI_EFFORT", "medium")
  monkeypatch.setenv("XAI_THINKING", "false")
  with pytest.raises(ValueError, match="conflicting XAI_effort"):
    resolve_auth_config(provider="xai")


@pytest.mark.parametrize(
  ("model", "requested", "effective", "enabled", "payload"),
  [
    ("grok-4.5", ThinkingLevel.NONE, ThinkingLevel.LOW, True, {"reasoning": {"effort": "low"}}),
    ("grok-4.5", ThinkingLevel.MINIMAL, ThinkingLevel.LOW, True, {"reasoning": {"effort": "low"}}),
    ("grok-4.5", ThinkingLevel.MEDIUM, ThinkingLevel.MEDIUM, True, {"reasoning": {"effort": "medium"}}),
    ("grok-4.5", ThinkingLevel.XHIGH, ThinkingLevel.HIGH, True, {"reasoning": {"effort": "high"}}),
    ("grok-4.3", ThinkingLevel.NONE, ThinkingLevel.NONE, False, {"reasoning": {"effort": "none"}}),
    ("grok-4.3-fast", ThinkingLevel.MAX, ThinkingLevel.HIGH, True, {"reasoning": {"effort": "high"}}),
    ("grok-build-0.1", ThinkingLevel.HIGH, ThinkingLevel.NONE, False, {}),
    ("grok-4.20-beta-latest-non-reasoning", ThinkingLevel.HIGH, ThinkingLevel.NONE, False, {}),
  ],
)
def test_effort_resolution_table(model, requested, effective, enabled, payload) -> None:
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


def test_build_request_uses_responses_local_history_and_tools() -> None:
  provider = XAIProvider()
  params = provider.build_request_params(
    model="grok-4.5",
    messages=[{"role": "user", "content": "Use the calculator"}],
    system_prompt="Be precise.",
    tools=[{"name": "calculator", "description": "Calculate", "input_schema": {"type": "object"}}],
    max_tokens=2048,
    thinking_level=ThinkingLevel.NONE,
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
