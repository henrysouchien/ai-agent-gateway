from __future__ import annotations

import pytest
from dataclasses import replace
from pathlib import Path

from agent_gateway import resolve_auth_config
from model_authority.thinking import ThinkingLevel
from model_authority.thinking import EffortResolution
from agent_gateway.providers.anthropic import AnthropicProvider
from agent_gateway.providers.codex import CodexProvider
from agent_gateway.providers.openai import OpenAIProvider
from agent_gateway.runner_auth import merge_refreshed_auth_config
from agent_gateway.runner import AgentRunner
from agent_gateway.runner_state import normalized_run_config
from agent_gateway.runner_streaming import effective_stream_stall_timeout
from agent_gateway.server import ChatRequest
from agent_gateway.skills import parse_skill_file
from model_authority.schema import AnthropicMessagesCompat


def test_auth_config_cannot_select_effort_or_thinking() -> None:
  with pytest.raises(ValueError, match="not model-selection authority"):
    resolve_auth_config(
      auth_config={"api_key": "k", "effort": "medium"}
    )


def test_legacy_thinking_is_rejected_at_run_and_refresh_boundaries() -> None:
  with pytest.raises(ValueError, match="must not contain model selection"):
    normalized_run_config(
      {"api_key": "k", "thinking": True},
      upstream_model="claude-sonnet-5",
      effort="high",
    )

  credential_config = {
    "provider": "anthropic",
    "auth_mode": "api",
    "api_key": "k",
    "max_tokens": 16_000,
  }
  refreshed = merge_refreshed_auth_config(
    credential_config,
    {"api_key": "new"},
  )
  assert refreshed["api_key"] == "new"
  assert {
    "model",
    "model_key",
    "effort",
    "thinking",
    "thinking_enabled_requested",
  }.isdisjoint(refreshed)
  with pytest.raises(ValueError, match="must not contain model selection"):
    merge_refreshed_auth_config(
      credential_config,
      {"api_key": "new", "thinking": False},
    )


def test_sonnet5_none_always_emits_disabled_below_gate() -> None:
  provider = AnthropicProvider()
  info = provider.get_model_info("claude-sonnet-5")
  resolved = provider.resolve_effort(
    requested=ThinkingLevel.NONE, model=info.id, model_info=info, max_tokens=1024
  )
  assert resolved == EffortResolution(ThinkingLevel.NONE, ThinkingLevel.NONE, False, {"thinking": {"type": "disabled"}})


def test_below_gate_uses_omitted_default_capability() -> None:
  provider = AnthropicProvider()
  sonnet = provider.get_model_info("claude-sonnet-5")
  opus55 = provider.get_model_info("claude-opus-5-5")
  # The same model with thinking off when omitted, as the authority can declare it.
  assert isinstance(sonnet.compat, AnthropicMessagesCompat)
  off_by_default = replace(sonnet, compat=sonnet.compat.model_copy(update={
    "thinking": sonnet.compat.thinking.model_copy(update={
      "default_when_omitted": "off", "effort_when_omitted": "none",
    }),
  }))

  def below_gate(info):
    resolution = provider.resolve_effort(
      requested=ThinkingLevel.HIGH, model=info.id, model_info=info, max_tokens=1024
    )
    return (resolution.effective, resolution.thinking_enabled_effective, dict(resolution.payload_fragments))

  assert below_gate(sonnet) == (ThinkingLevel.HIGH, True, {})
  assert below_gate(opus55) == (ThinkingLevel.MEDIUM, True, {})
  assert below_gate(off_by_default) == (ThinkingLevel.NONE, False, {})


def test_fable_none_is_effectively_on_and_xhigh_clamps_to_supported_efforts() -> None:
  provider = AnthropicProvider()
  fable = provider.get_model_info("claude-fable-5")
  opus = provider.get_model_info("claude-opus-5")
  # An entry whose supported efforts stop at high, as the authority can declare it.
  high_ceiling = replace(opus, effort_values=("none", "low", "medium", "high"))
  assert provider.resolve_effort(
    requested=ThinkingLevel.NONE, model=fable.id, model_info=fable, max_tokens=4096
  ).effective is ThinkingLevel.HIGH
  clamped = provider.resolve_effort(
    requested=ThinkingLevel.XHIGH, model=high_ceiling.id, model_info=high_ceiling, max_tokens=4096
  )
  assert clamped.effective is ThinkingLevel.HIGH
  assert clamped.payload_fragments["output_config"] == {"effort": "high"}


def test_openai_runtime_compat_cannot_disable_responses_effort() -> None:
  provider = OpenAIProvider()
  info = provider.get_model_info("gpt-5.6")
  resolved = provider.resolve_effort(
    requested=ThinkingLevel.MAX,
    model=info.id,
    model_info=info,
    max_tokens=4096,
    compat={"supportsReasoningEffort": False},
  )
  assert resolved.payload_fragments == {"reasoning": {"effort": "max"}}
  assert resolved.thinking_enabled_effective is True
  assert effective_stream_stall_timeout(
    None, config={"effort": "max"}, model_info=info, max_tokens=4096, effort_resolution=resolved
  ) == 300

  runner = AgentRunner.__new__(AgentRunner)
  runner._effort_resolution = resolved
  assert runner.effort_introspection == {
    "requested": "max",
    "effective": "max",
    "thinking_enabled_effective": True,
  }


def test_gpt56_specific_rows_and_max_payload_are_distinct() -> None:
  openai = OpenAIProvider()
  codex = CodexProvider()
  cases = [
    (openai, "gpt-5.6"),
    (openai, "gpt-5.6-sol"),
    (codex, "gpt-5.6-sol"),
    (codex, "gpt-5.6-terra"),
    (codex, "gpt-5.6-luna"),
  ]
  infos = [(provider, provider.get_model_info(model)) for provider, model in cases]
  assert [info.id for _provider, info in infos] == [model for _provider, model in cases]
  assert all(
    provider.resolve_effort(
      requested=ThinkingLevel.MAX, model=info.id, model_info=info, max_tokens=4096
    ).payload_fragments == {"reasoning": {"effort": "max"}}
    for provider, info in infos
  )


def test_codex_reasoning_deep_merge_preserves_summary() -> None:
  provider = CodexProvider()
  params = provider.build_request_params(
    model="gpt-5.6-terra",
    messages=[],
    system_prompt=None,
    tools=[],
    max_tokens=4096,
    thinking_level=ThinkingLevel.MAX,
    reasoning_summary="auto",
  )
  assert params["reasoning"] == {"summary": "auto", "effort": "max"}
  assert "effort_requested" not in params
  assert "effort_effective" not in params


def test_chat_request_validates_and_canonicalizes_effort() -> None:
  request = ChatRequest(
    messages=[],
    model_key="openai.gpt-5-6",
    effort=" XHIGH ",
  )
  assert request.effort == "xhigh"
  with pytest.raises(ValueError):
    ChatRequest(
      messages=[],
      model_key="openai.gpt-5-6",
      effort="",
    )


def test_skill_effort_conflict_and_positional_boundary(tmp_path: Path) -> None:
  path = tmp_path / "skill.md"
  path.write_text("---\nname: effort-skill\neffort: medium\nthinking: true\n---\nBody\n")
  with pytest.raises(ValueError, match="conflicting"):
    parse_skill_file(path)
