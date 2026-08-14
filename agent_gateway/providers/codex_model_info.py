from __future__ import annotations

from .base import ModelInfo, ThinkingLevel

# context_window derivation (two intentional sources — do NOT unify blindly):
#   400_000  = ChatGPT-backend enforced ceiling, probed live 2026-07-21
#              (~370k accepted / 385k rejected, rounded up). Applies to the
#              gpt-5.6 family, gpt-5.5, gpt-5.1 — the models actually runnable
#              on the ChatGPT backend.
#   272_000  = OpenAI published catalog context for the codex-specialized
#              rows (5.1-codex-*, 5.2/-codex, 5.3-codex, 5.4). NOT backend-probed.
#   128_000  = gpt-5.3-codex-spark catalog context.
# WHY THIS MATTERS: effective_compaction_trigger = max(window*0.8, 160k). The
# trigger-above-wall defect (the one fixed for grok in fac666326) only bites
# when the registered window EXCEEDS the real enforced ceiling. The 272k rows
# are conservative (below any plausible real ceiling → early trigger, no hedge
# dependency), so they are SAFE as-is. Do not "fix" them upward to match 400k
# without a live probe of each model's real ceiling — that would reintroduce
# the exact defect. To unify, probe each row live and lower/keep, never raise.
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
        input_cost_per_mtok=input_cost,
        output_cost_per_mtok=output_cost,
        cache_read_cost_per_mtok=cache_cost,
      ),
    )
    for model_id, input_cost, output_cost, cache_cost in (
      ("gpt-5.6-sol", 5.00, 30.00, 0.50),
      ("gpt-5.6-terra", 2.50, 15.00, 0.25),
      ("gpt-5.6-luna", 1.00, 6.00, 0.10),
      ("gpt-5.6", 5.00, 30.00, 0.50),
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
      context_window=400_000,
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
