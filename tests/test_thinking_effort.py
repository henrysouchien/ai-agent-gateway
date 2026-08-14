from __future__ import annotations

import pytest
from dataclasses import fields
from pathlib import Path

from agent_gateway import EffortResolution, ThinkingLevel, resolve_auth_config
from agent_gateway.providers.anthropic import AnthropicProvider
from agent_gateway.providers.codex import CodexProvider
from agent_gateway.providers.openai import OpenAIProvider
from agent_gateway.runner_auth import merge_refreshed_auth_config
from agent_gateway.runner import AgentRunner
from agent_gateway.runner_state import normalized_run_config
from agent_gateway.runner_streaming import effective_stream_stall_timeout
from agent_gateway.server import ChatRequest
from agent_gateway.skills import parse_skill_file
from agent_gateway.thinking import parse_effort, resolve_effort_pair


@pytest.mark.parametrize("raw", ["none", "MINIMAL", " low ", "medium", "high", "xhigh", "max"])
def test_parse_effort_accepts_all_canonical_levels(raw: str) -> None:
  assert parse_effort(raw) is ThinkingLevel(raw.strip().lower())


@pytest.mark.parametrize(
  ("effort", "thinking", "expected"),
  [
    (None, True, ThinkingLevel.HIGH),
    (None, False, ThinkingLevel.NONE),
    ("high", True, ThinkingLevel.HIGH),
    ("none", False, ThinkingLevel.NONE),
    ("medium", None, ThinkingLevel.MEDIUM),
  ],
)
def test_dual_key_agreement_matrix(effort, thinking, expected) -> None:
  assert resolve_effort_pair(effort=effort, thinking=thinking) is expected


@pytest.mark.parametrize(("effort", "thinking"), [("medium", True), ("medium", False), ("none", True)])
def test_dual_key_conflicts_raise(effort, thinking) -> None:
  with pytest.raises(ValueError, match="conflicting"):
    resolve_effort_pair(effort=effort, thinking=thinking)


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
  opus = provider.get_model_info("claude-opus-4-8")
  sonnet_resolution = provider.resolve_effort(
    requested=ThinkingLevel.HIGH, model=sonnet.id, model_info=sonnet, max_tokens=1024
  )
  opus_resolution = provider.resolve_effort(
    requested=ThinkingLevel.HIGH, model=opus.id, model_info=opus, max_tokens=1024
  )
  assert (sonnet_resolution.effective, sonnet_resolution.thinking_enabled_effective) == (ThinkingLevel.HIGH, True)
  assert (opus_resolution.effective, opus_resolution.thinking_enabled_effective) == (ThinkingLevel.NONE, False)


def test_fable_none_is_effectively_on_and_opus46_xhigh_clamps() -> None:
  provider = AnthropicProvider()
  fable = provider.get_model_info("claude-fable-5")
  opus46 = provider.get_model_info("claude-opus-4-6")
  assert provider.resolve_effort(
    requested=ThinkingLevel.NONE, model=fable.id, model_info=fable, max_tokens=4096
  ).effective is ThinkingLevel.HIGH
  assert provider.resolve_effort(
    requested=ThinkingLevel.XHIGH, model=opus46.id, model_info=opus46, max_tokens=4096
  ).effective is ThinkingLevel.HIGH


def test_openai_runtime_compat_cannot_disable_responses_effort() -> None:
  provider = OpenAIProvider()
  info = provider.get_model_info("gpt-5.6-terra")
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
  provider = OpenAIProvider()
  infos = [provider.get_model_info(model) for model in ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna")]
  assert [info.id for info in infos] == ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
  assert all(
    provider.resolve_effort(
      requested=ThinkingLevel.MAX, model=info.id, model_info=info, max_tokens=4096
    ).payload_fragments == {"reasoning": {"effort": "max"}}
    for info in infos
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


def test_profile_config_does_not_duplicate_effort_authority() -> None:
  from api.agent.profiles import ProfileConfig

  assert "effort" not in {field.name for field in fields(ProfileConfig)}


def test_skill_effort_conflict_and_positional_boundary(tmp_path: Path) -> None:
  path = tmp_path / "skill.md"
  path.write_text("---\nname: effort-skill\neffort: medium\nthinking: true\n---\nBody\n")
  with pytest.raises(ValueError, match="conflicting"):
    parse_skill_file(path)


def test_provider_env_effort_does_not_enter_credential_config(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from api.credentials import get_anthropic_config

  monkeypatch.setenv("ANTHROPIC_EFFORT", "medium")
  monkeypatch.setenv("ANTHROPIC_THINKING", "false")
  config = get_anthropic_config()
  assert "effort" not in config
  assert "thinking" not in config


def test_anthropic_sdk_floor_is_locked() -> None:
  root = Path(__file__).resolve().parents[3]
  assert 'anthropic = ["anthropic>=0.93.0"]' in (root / "packages/agent-gateway/pyproject.toml").read_text()
  assert "anthropic>=0.93.0" in (root / "packages/agent-gateway/requirements-dev.in").read_text()
