from __future__ import annotations

from .base import ModelInfo, ThinkingLevel

# Legacy protocol metadata for identifiers predating the product registry.
# CodexProvider overlays current prices and limits from rates/codex.json;
# new identifiers need only registry YAML and a rate row, not this table.
_MODEL_INFO_BY_TAG: list[tuple[tuple[str, ...], ModelInfo]] = [
  *[
    (
      (model_id,),
      ModelInfo(
        id=model_id,
        provider="codex",
        context_window=400_000,  # ChatGPT backend enforces ~370-385k input (probed live 2026-07-21)
        max_output_tokens=128_000,
        supports_thinking=True,
        supports_vision=True,
      ),
    )
    for model_id in (
      "gpt-5.6-sol",
      "gpt-5.6-terra",
      "gpt-5.6-luna",
      "gpt-5.6",
    )
  ],
  (
    ("gpt-5.5",),
    ModelInfo(
      id="gpt-5.5",
      provider="codex",
      context_window=400_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
    ),
  ),
  (
    ("gpt-5.1",),
    ModelInfo(
      id="gpt-5.1",
      provider="codex",
      context_window=400_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
    ),
  ),
  (
    ("gpt-5.1-codex-max",),
    ModelInfo(
      id="gpt-5.1-codex-max",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
    ),
  ),
  (
    ("gpt-5.1-codex-mini",),
    ModelInfo(
      id="gpt-5.1-codex-mini",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
    ),
  ),
  (
    ("gpt-5.2",),
    ModelInfo(
      id="gpt-5.2",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
    ),
  ),
  (
    ("gpt-5.2-codex",),
    ModelInfo(
      id="gpt-5.2-codex",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
    ),
  ),
  (
    ("gpt-5.3-codex",),
    ModelInfo(
      id="gpt-5.3-codex",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
    ),
  ),
  (
    ("gpt-5.3-codex-spark",),
    ModelInfo(
      id="gpt-5.3-codex-spark",
      provider="codex",
      context_window=128_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=False,
    ),
  ),
  (
    ("gpt-5.4",),
    ModelInfo(
      id="gpt-5.4",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
    ),
  ),
]

_EFFORT_VALUES_BY_MODEL = {
  "gpt-5.6": ("none", "low", "medium", "high", "xhigh", "max"),
  "gpt-5.6-sol": ("none", "low", "medium", "high", "xhigh", "max"),
  "gpt-5.6-terra": ("none", "low", "medium", "high", "xhigh", "max"),
  "gpt-5.6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
  "gpt-5.5": ("none", "low", "medium", "high", "xhigh"),
  "gpt-5.4": ("none", "low", "medium", "high", "xhigh"),
  "gpt-5.3-codex": ("low", "medium", "high", "xhigh"),
  "gpt-5.3-codex-spark": ("low", "medium", "high", "xhigh"),
  "gpt-5.2": ("low", "medium", "high"),
  "gpt-5.2-codex": ("low", "medium", "high"),
  "gpt-5.1": ("none", "low", "medium", "high"),
  "gpt-5.1-codex-max": ("none", "low", "medium", "high", "xhigh"),
  "gpt-5.1-codex-mini": ("medium", "high"),
}
for _tags, _info in _MODEL_INFO_BY_TAG:
  _values = _EFFORT_VALUES_BY_MODEL[_info.id]
  _info.compat = {
    "supportsReasoningEffort": True,
    "reasoningEffortValues": _values,
    "reasoningEffortDefault": "none" if _info.id in {"gpt-5.4", "gpt-5.1"} else "medium",
    "omitEqualsNone": False,
  }


def _model_matches_tag(model_id: str, tag: str) -> bool:
  candidates = [model_id, model_id.rsplit("/", 1)[-1]]
  return any(candidate == tag or candidate.startswith(f"{tag}-") for candidate in candidates)


def _map_reasoning_effort(level: ThinkingLevel) -> str | None:
  if level == ThinkingLevel.NONE:
    return None
  if level == ThinkingLevel.MINIMAL:
    return "minimal"
  if level == ThinkingLevel.LOW:
    return "low"
  if level == ThinkingLevel.MEDIUM:
    return "medium"
  return "high"


def _clamp_reasoning_effort(model_id: str, effort: str) -> str:
  identifier = model_id.rsplit("/", 1)[-1]
  if (
    identifier.startswith("gpt-5.2")
    or identifier.startswith("gpt-5.3")
    or identifier.startswith("gpt-5.4")
    or identifier.startswith("gpt-5.5")
  ) and effort == "minimal":
    return "low"
  if identifier == "gpt-5.1-codex-mini":
    return "high" if effort == "high" else "medium"
  return effort
