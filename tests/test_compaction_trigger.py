import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
for path in (str(ROOT), str(PKG_DIR)):
  if path not in sys.path:
    sys.path.insert(0, path)

from agent_gateway.providers.base import ModelInfo  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_limits import (  # noqa: E402
  COMPACTION_TRIGGER_PCT,
  CONTEXT_PRESSURE_REMINDER_PCT,
  MODEL_CONTEXT_LIMIT,
  context_pressure_reminder_decision,
  effective_compaction_trigger,
  estimate_tokens,
  model_context_window,
  system_prompt_estimate_text,
  token_breakdown_snapshot,
  token_estimate_snapshot,
)


def test_context_pressure_decision_exact_threshold_and_band_hysteresis() -> None:
  below = context_pressure_reminder_decision(
    est_tokens=599,
    context_limit=1_000,
    next_threshold_pct=CONTEXT_PRESSURE_REMINDER_PCT,
  )
  at_initial = context_pressure_reminder_decision(
    est_tokens=600,
    context_limit=1_000,
    next_threshold_pct=below.next_threshold_pct,
  )
  below_next = context_pressure_reminder_decision(
    est_tokens=699,
    context_limit=1_000,
    next_threshold_pct=at_initial.next_threshold_pct,
  )
  at_next = context_pressure_reminder_decision(
    est_tokens=700,
    context_limit=1_000,
    next_threshold_pct=below_next.next_threshold_pct,
  )

  assert (below.reminder_pct, below.next_threshold_pct) == (None, 60)
  assert (at_initial.reminder_pct, at_initial.next_threshold_pct) == (60, 70)
  assert (below_next.reminder_pct, below_next.next_threshold_pct) == (None, 70)
  assert (at_next.reminder_pct, at_next.next_threshold_pct) == (70, 80)


def test_context_pressure_decision_jump_and_jitter_do_not_rearm() -> None:
  initial = context_pressure_reminder_decision(
    est_tokens=650,
    context_limit=1_000,
    next_threshold_pct=60,
  )
  below_next = context_pressure_reminder_decision(
    est_tokens=749,
    context_limit=1_000,
    next_threshold_pct=initial.next_threshold_pct,
  )
  jump = context_pressure_reminder_decision(
    est_tokens=899,
    context_limit=1_000,
    next_threshold_pct=below_next.next_threshold_pct,
  )
  drop = context_pressure_reminder_decision(
    est_tokens=500,
    context_limit=1_000,
    next_threshold_pct=jump.next_threshold_pct,
  )
  jitter = context_pressure_reminder_decision(
    est_tokens=989,
    context_limit=1_000,
    next_threshold_pct=drop.next_threshold_pct,
  )
  reset = context_pressure_reminder_decision(
    est_tokens=650,
    context_limit=1_000,
    next_threshold_pct=60,
  )

  assert (initial.reminder_pct, initial.next_threshold_pct) == (65, 75)
  assert (below_next.reminder_pct, below_next.next_threshold_pct) == (
    None,
    75,
  )
  assert (jump.reminder_pct, jump.next_threshold_pct) == (89, 99)
  assert (drop.reminder_pct, drop.next_threshold_pct) == (None, 99)
  assert (jitter.reminder_pct, jitter.next_threshold_pct) == (None, 99)
  assert (reset.reminder_pct, reset.next_threshold_pct) == (65, 75)


def test_context_pressure_decision_uses_exact_integer_boundary_math() -> None:
  context_limit = 10**30 + 7
  threshold_tokens = (60 * context_limit + 99) // 100

  below = context_pressure_reminder_decision(
    est_tokens=threshold_tokens - 1,
    context_limit=context_limit,
    next_threshold_pct=60,
  )
  at_or_above = context_pressure_reminder_decision(
    est_tokens=threshold_tokens,
    context_limit=context_limit,
    next_threshold_pct=60,
  )

  assert below.reminder_pct is None
  assert at_or_above.reminder_pct == 60


def _model(context_window: int) -> ModelInfo:
  return ModelInfo(id="m", provider="anthropic", context_window=context_window)


def test_disabled_trigger_stays_none() -> None:
  assert effective_compaction_trigger(None, _model(1_000_000)) is None


def test_large_window_scales_trigger_up() -> None:
  # 160k legacy trigger on a 1M-window model -> 80% of 1M, not 160k.
  trig = effective_compaction_trigger(160_000, _model(1_000_000))
  assert trig == int(1_000_000 * COMPACTION_TRIGGER_PCT / 100)
  assert trig > 160_000


def test_small_window_keeps_configured_floor() -> None:
  # 200k window: 80% == 160k, equal to the configured floor.
  assert effective_compaction_trigger(160_000, _model(200_000)) == 160_000


def test_configured_value_is_a_floor_when_window_pct_is_lower() -> None:
  # A configured trigger higher than the window-derived value wins (never
  # compacts later than legacy on a tiny window).
  assert effective_compaction_trigger(180_000, _model(200_000)) == 180_000


def test_missing_window_falls_back_to_default_limit() -> None:
  class _NoWindow:
    pass

  assert model_context_window(_NoWindow()) == MODEL_CONTEXT_LIMIT
  assert effective_compaction_trigger(160_000, _NoWindow()) == int(
    MODEL_CONTEXT_LIMIT * COMPACTION_TRIGGER_PCT / 100
  )


def test_runner_preserves_private_limit_helper_aliases() -> None:
  assert gateway_runner._model_context_window is model_context_window
  assert gateway_runner._effective_compaction_trigger is effective_compaction_trigger
  assert gateway_runner._token_estimate_snapshot is token_estimate_snapshot


def test_system_prompt_estimate_text_matches_runner_list_prompt_joining() -> None:
  assert system_prompt_estimate_text("plain") == "plain"
  assert system_prompt_estimate_text([("static", True), ("", True), ("dynamic", False)]) == "static\n\ndynamic"
  assert system_prompt_estimate_text(None) == ""


def test_token_estimate_snapshot_serializes_messages_tools_and_counts_tokens() -> None:
  messages = [{"role": "user", "content": "hello"}]
  tools = [{"name": "tool", "input_schema": {"type": "object"}}]

  snapshot = token_estimate_snapshot(
    system_text="system",
    messages=messages,
    tools=tools,
  )

  assert snapshot.system_text == "system"
  assert snapshot.system_chars == len("system")
  assert snapshot.tools_chars == len(snapshot.tools_text)
  assert snapshot.est_system_tokens == estimate_tokens("system")
  assert snapshot.est_messages_tokens == estimate_tokens(snapshot.messages_text)
  assert snapshot.est_tools_tokens == estimate_tokens(snapshot.tools_text)
  assert snapshot.est_total_tokens == (
    snapshot.est_system_tokens
    + snapshot.est_messages_tokens
    + snapshot.est_tools_tokens
  )
  assert snapshot.message_count == 1
  assert snapshot.tool_count == 1


def test_token_estimate_snapshot_uses_last_compaction_for_message_text() -> None:
  messages = [
    {"role": "user", "content": "old user turn"},
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "old assistant text"},
        {"type": "compaction", "content": "summary"},
      ],
    },
    {"role": "user", "content": "new user turn"},
  ]

  snapshot = token_estimate_snapshot(
    system_text="system",
    messages=messages,
    tools=[],
  )

  assert "old user turn" not in snapshot.messages_text
  assert "old assistant text" not in snapshot.messages_text
  assert "summary" in snapshot.messages_text
  assert "new user turn" in snapshot.messages_text
  assert snapshot.message_count == len(messages)


def test_token_estimate_snapshot_omits_empty_tools_text_and_tokens() -> None:
  snapshot = token_estimate_snapshot(
    system_text="system",
    messages=[],
    tools=[],
  )

  assert snapshot.tools_text == ""
  assert snapshot.tools_chars == 0
  assert snapshot.est_tools_tokens == 0
  assert snapshot.tool_count == 0


def test_token_breakdown_snapshot_matches_runner_proportional_math() -> None:
  messages = [{"role": "user", "content": "hello"}]
  messages_chars = len(json.dumps(messages, default=str))
  total_chars = 10 + 30 + messages_chars

  snapshot = token_breakdown_snapshot(
    input_tokens=1000,
    system_chars=10,
    tools_chars=30,
    messages=messages,
  )

  assert snapshot is not None
  assert snapshot.input_tokens == 1000
  assert snapshot.pct_system == round(10 / total_chars * 100)
  assert snapshot.pct_tools == round(30 / total_chars * 100)
  assert snapshot.pct_messages == round(messages_chars / total_chars * 100)
  assert snapshot.est_system_tokens == round(1000 * 10 / total_chars)
  assert snapshot.est_tools_tokens == round(1000 * 30 / total_chars)
  assert snapshot.est_messages_tokens == (
    1000
    - snapshot.est_system_tokens
    - snapshot.est_tools_tokens
  )


def test_token_breakdown_snapshot_skips_non_positive_input_tokens() -> None:
  assert token_breakdown_snapshot(
    input_tokens=0,
    system_chars=10,
    tools_chars=30,
    messages=[{"role": "user", "content": "hello"}],
  ) is None
  assert token_breakdown_snapshot(
    input_tokens=-1,
    system_chars=10,
    tools_chars=30,
    messages=[{"role": "user", "content": "hello"}],
  ) is None
