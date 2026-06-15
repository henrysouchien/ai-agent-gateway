import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
for path in (str(ROOT), str(PKG_DIR)):
  if path not in sys.path:
    sys.path.insert(0, path)

from agent_gateway.providers.base import ModelInfo
from agent_gateway.runner import (
  COMPACTION_TRIGGER_PCT,
  MODEL_CONTEXT_LIMIT,
  _effective_compaction_trigger,
  _model_context_window,
)


def _model(context_window: int) -> ModelInfo:
  return ModelInfo(id="m", provider="anthropic", context_window=context_window)


def test_disabled_trigger_stays_none() -> None:
  assert _effective_compaction_trigger(None, _model(1_000_000)) is None


def test_large_window_scales_trigger_up() -> None:
  # 160k legacy trigger on a 1M-window model -> 80% of 1M, not 160k.
  trig = _effective_compaction_trigger(160_000, _model(1_000_000))
  assert trig == int(1_000_000 * COMPACTION_TRIGGER_PCT / 100)
  assert trig > 160_000


def test_small_window_keeps_configured_floor() -> None:
  # 200k window: 80% == 160k, equal to the configured floor.
  assert _effective_compaction_trigger(160_000, _model(200_000)) == 160_000


def test_configured_value_is_a_floor_when_window_pct_is_lower() -> None:
  # A configured trigger higher than the window-derived value wins (never
  # compacts later than legacy on a tiny window).
  assert _effective_compaction_trigger(180_000, _model(200_000)) == 180_000


def test_missing_window_falls_back_to_default_limit() -> None:
  class _NoWindow:
    pass

  assert _model_context_window(_NoWindow()) == MODEL_CONTEXT_LIMIT
  assert _effective_compaction_trigger(160_000, _NoWindow()) == int(
    MODEL_CONTEXT_LIMIT * COMPACTION_TRIGGER_PCT / 100
  )
