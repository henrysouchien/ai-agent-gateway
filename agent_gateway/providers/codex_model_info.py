from __future__ import annotations

from .base import ModelInfo, ThinkingLevel

_MODEL_INFO_BY_TAG: list[tuple[tuple[str, ...], ModelInfo]] = [
  (
    ("gpt-5.5",),
    ModelInfo(
      id="gpt-5.5",
      provider="codex",
      context_window=1_050_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=5.00,
      output_cost_per_mtok=30.00,
      cache_read_cost_per_mtok=0.50,
    ),
  ),
  (
    ("gpt-5.1",),
    ModelInfo(
      id="gpt-5.1",
      provider="codex",
      context_window=272_000,
      max_output_tokens=128_000,
      supports_thinking=True,
      supports_vision=True,
      input_cost_per_mtok=1.25,
      output_cost_per_mtok=10.0,
      cache_read_cost_per_mtok=0.125,
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
      input_cost_per_mtok=1.25,
      output_cost_per_mtok=10.0,
      cache_read_cost_per_mtok=0.125,
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
      input_cost_per_mtok=0.25,
      output_cost_per_mtok=2.0,
      cache_read_cost_per_mtok=0.025,
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
      input_cost_per_mtok=1.75,
      output_cost_per_mtok=14.0,
      cache_read_cost_per_mtok=0.175,
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
      input_cost_per_mtok=1.75,
      output_cost_per_mtok=14.0,
      cache_read_cost_per_mtok=0.175,
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
      input_cost_per_mtok=1.75,
      output_cost_per_mtok=14.0,
      cache_read_cost_per_mtok=0.175,
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
      input_cost_per_mtok=0.0,
      output_cost_per_mtok=0.0,
      cache_read_cost_per_mtok=0.0,
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
      input_cost_per_mtok=2.5,
      output_cost_per_mtok=15.0,
      cache_read_cost_per_mtok=0.25,
    ),
  ),
]


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
