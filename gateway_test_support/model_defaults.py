"""Packaged model-selection defaults, read from the shipped authority.

Tests that exercise *default* resolution derive their expectations here so a
catalog revision changes no consumer's expectations.
"""

from __future__ import annotations

from dataclasses import dataclass

from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from model_authority.schema import (
  AnthropicMessagesCompat,
  AnthropicThinking,
  FamilyCompat,
  GoogleGenerateContentCompat,
  OpenAIChatCompletionsCompat,
  OpenAIResponsesCompat,
  ReasoningControl,
  protocol_family,
)


@dataclass(frozen=True)
class CapabilityDefault:
  capability_id: str
  model_key: str
  upstream_model: str
  effort: str
  registry_revision: str
  policy_revision: str


def capability_default(capability_id: str) -> CapabilityDefault:
  """Packaged policy default for a capability whose default kind is 'model'; raises for inherit_parent."""
  default = INITIAL_MODEL_SELECTION_POLICY.capabilities[capability_id].default
  if default.kind != "model" or default.model_key is None or default.effort is None:
    raise ValueError(
      f"{capability_id} has no packaged model default (kind={default.kind!r})"
    )
  return CapabilityDefault(
    capability_id=capability_id,
    model_key=default.model_key,
    upstream_model=INITIAL_MODEL_REGISTRY.require(default.model_key).upstream_model,
    effort=default.effort,
    registry_revision=INITIAL_MODEL_REGISTRY.revision,
    policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
  )


SESSION_DRIVER = capability_default("session.driver")
QUANT_WORKER = capability_default("investment.quant_worker")


def compat_for_profile(protocol_profile: str) -> FamilyCompat:
  """A compat block of ``protocol_profile``'s family for hand-built test entries.

  Limits are the adapters' conservative defaults; reasoning admits every
  effort, so the entry's ``supported_efforts`` alone decides what a bind may
  request.
  """
  family = protocol_family(protocol_profile)
  if family == "anthropic.messages":
    return AnthropicMessagesCompat(
      accepts_temperature=False,
      thinking=AnthropicThinking(
        default_when_omitted="off",
        can_disable=True,
        effort_control="output_config",
        effort_when_omitted="none",
      ),
      forced_tool_choice_requires_thinking_off=True,
      native_compaction=False,
      max_output_tokens=16_384,
      context_window=200_000,
    )
  if family == "openai.responses":
    return OpenAIResponsesCompat(
      reasoning_control=ReasoningControl(
        param="reasoning.effort",
        off_value="none",
        values=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
      ),
      reasoning_summary=False,
      function_tools=True,
      max_output_tokens=16_384,
      context_window=200_000,
    )
  if family == "openai.chat_completions":
    return OpenAIChatCompletionsCompat(
      accepts_temperature=True,
      reasoning_control=None,
      output_token_param="max_completion_tokens",
      sampling_required=False,
      json_mode="none",
      max_output_tokens=16_384,
      context_window=200_000,
    )
  return GoogleGenerateContentCompat(max_output_tokens=16_384, context_window=200_000)
