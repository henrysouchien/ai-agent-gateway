"""Quota observations, parsing, the pure ranker, the selection sidecar and the usage probe.

Everything runs on an injected clock (`NOW`), temporary stores and
`httpx.MockTransport`; nothing sleeps longer than a few hundredths of a second and
nothing touches the network.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import stat
import time
import types
from typing import Any

import httpx
import pytest

from agent_gateway.providers import anthropic_quota as quota
from agent_gateway.providers.anthropic_oauth import ANTHROPIC_OAUTH_BETA
from agent_gateway.providers.anthropic_quota import (
  AccountLocator,
  AccountObservation,
  AccountPolicy,
  Affinity,
  Cooldown,
  ProbeTarget,
  QuotaWindow,
  RankCandidate,
  UsageProbeState,
)

NOW = 1_760_000_000.0
HOUR = 3600.0
WEEK = 7 * 86400.0
FIVE_HOURS = 5 * HOUR


@pytest.fixture(autouse=True)
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(quota, "time", types.SimpleNamespace(time=lambda: NOW))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def win(
  window_id: str,
  used: float | None,
  *,
  reset_in: float | None = None,
  age: float = 0.0,
  scope: str = "shared",
  status: str | None = None,
) -> QuotaWindow:
  return QuotaWindow(
    id=window_id,
    scope=scope,
    used_fraction=used,
    reset_at=None if reset_in is None else NOW + reset_in,
    duration_s=FIVE_HOURS if window_id == "five_hour" else WEEK,
    observed_at=NOW - age,
    source="usage",
    status=status,
  )


def five(used: float | None, **kw: Any) -> QuotaWindow:
  return win("five_hour", used, **kw)


def week(used: float | None, **kw: Any) -> QuotaWindow:
  return win("seven_day", used, **kw)


def cand(
  name: str,
  *,
  tier: str = "private",
  order: int | None = None,
  priority: int = 0,
  reserve: float = 10.0,
  windows: tuple[QuotaWindow, ...] = (),
  cooldowns: tuple[Cooldown, ...] = (),
) -> RankCandidate:
  return RankCandidate(
    locator=AccountLocator("store:/test", name),
    tier=tier,
    order=ord(name[0]) if order is None else order,
    policy=AccountPolicy(priority=priority, reserve_pct=reserve),
    observation=AccountObservation(windows=windows),
    cooldowns=cooldowns,
  )


def rank(
  candidates: list[RankCandidate],
  *,
  family: str | None = None,
  affinity: Affinity | None = None,
  seed: str | None = None,
  now: float = NOW,
) -> quota.RankDecision:
  return quota.rank_candidates(
    candidates, now=now, family=family, affinity=affinity, session_seed=seed
  )


def names(decision: quota.RankDecision) -> list[str]:
  return [c.locator.identity for c in decision.ordered]


def warm(name: str, *, idle: float = 100.0) -> Affinity:
  return Affinity(AccountLocator("store:/test", name).id, NOW - idle)


def iso(epoch: float) -> str:
  return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# contract surface
# ---------------------------------------------------------------------------


def test_public_surface_is_exactly_the_contract() -> None:
  assert set(quota.__all__) == {
    "SELECTION_SIDECAR_SUFFIX", "SELECTION_STATE_VERSION", "DEFAULT_PRIORITY",
    "DEFAULT_RESERVE_PCT", "SHORT_WINDOW_HEAT", "WARM_IDLE_SECONDS",
    "USAGE_FRESH_TTL_SECONDS", "USAGE_UNAVAILABLE_SECONDS",
    "USAGE_RETRY_AFTER_CAP_SECONDS", "LAST_GOOD_MAX_AGE_SECONDS",
    "AFFINITY_PRUNE_SECONDS", "PROBE_BATCH_BUDGET_SECONDS", "PROBE_MAX_CONCURRENCY",
    "ANTHROPIC_USAGE_URL", "REASON_WARM", "REASON_QUOTA_DRAIN",
    "REASON_SHORT_WINDOW_HOT", "REASON_RESERVE", "REASON_ORG_FALLBACK",
    "REASON_QUOTA_UNAVAILABLE", "REASON_ALL_BLOCKED_PROBE", "REASON_ORDERED",
    "selection_sidecar_path", "AccountLocator", "locator_for_record",
    "locator_for_env_token", "AccountPolicy", "QuotaWindow", "Cooldown",
    "UsageProbeState", "AccountObservation", "Affinity", "SelectionState",
    "load_selection_state", "update_selection_state", "set_account_policy",
    "reset_account_policy", "record_observation", "record_cooldown",
    "record_probe_state", "claim_usage_probe", "record_success", "affinity_key", "model_family",
    "claim_scope",
    "parse_rate_limit_headers", "parse_usage_report", "usage_probe_allowed", "window_exhausted",
    "ProbeTarget", "ProbeResult", "probe_usage_batch", "RankCandidate",
    "RankDecision", "rank_candidates", "describe_account_usage",
  }
  assert all(hasattr(quota, name) for name in quota.__all__)


def test_constants_match_the_contract() -> None:
  assert quota.SELECTION_SIDECAR_SUFFIX == ".selection.json"
  assert quota.SHORT_WINDOW_HEAT == 0.85
  assert quota.WARM_IDLE_SECONDS == 3600.0
  assert quota.USAGE_FRESH_TTL_SECONDS == 300.0
  assert quota.DEFAULT_RESERVE_PCT == 10.0
  assert quota.PROBE_BATCH_BUDGET_SECONDS == 2.0
  assert quota.PROBE_MAX_CONCURRENCY == 4
  assert quota.ANTHROPIC_USAGE_URL == "https://api.anthropic.com/api/oauth/usage"


@pytest.mark.parametrize(
  "status,exhausted",
  [
    (None, False),
    ("", False),
    ("  ", False),
    ("allow", False),
    ("allowed", False),
    (" Allow_Warning ", False),
    ("rejected", True),
    ("throttled", True),
    ("future_provider_status", True),
    (" DENIED ", True),
  ],
)
def test_window_exhausted_treats_every_present_non_allow_status_as_exhausted(
  status: str | None, exhausted: bool
) -> None:
  assert quota.window_exhausted(status) is exhausted
  candidate = cand("A", windows=(week(None, reset_in=HOUR, status=status),))
  decision = rank([candidate])
  assert decision.blocked == (((candidate, NOW + HOUR),) if exhausted else ())
  assert names(decision) == ([] if exhausted else ["A"])


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


def test_locator_ids_are_opaque_namespaced_and_deterministic(tmp_path: Path) -> None:
  store_a, store_b = tmp_path / "a" / "oauth.json", tmp_path / "b" / "oauth.json"
  first = quota.locator_for_record(store_a, "same@example.com")
  assert first.id == quota.locator_for_record(store_a, "same@example.com").id
  assert first.id != quota.locator_for_record(store_b, "same@example.com").id
  assert first.id.startswith("acct_") and len(first.id) == len("acct_") + 16
  assert "example.com" not in first.id and str(tmp_path) not in first.id
  assert first.namespace == "store:" + str(store_a.resolve())


def test_env_token_locator_never_carries_the_token() -> None:
  token = "sk-ant-oat01-secret-token"
  locator = quota.locator_for_env_token(token)
  assert locator.namespace == "env"
  assert token not in locator.identity and token not in locator.id
  assert locator.identity == hashlib.sha256(token.encode()).hexdigest()


def test_affinity_key_is_none_without_a_real_session() -> None:
  assert quota.affinity_key("u1", None) is None
  assert quota.affinity_key("u1", "") is None
  assert quota.affinity_key("u1", "   ") is None
  assert quota.affinity_key("u1", "no-session") is None


def test_affinity_key_hashes_scope_session_and_provider() -> None:
  key = quota.affinity_key("u1", "sess-123")
  assert key == hashlib.sha256("u1\0sess-123\0anthropic".encode()).hexdigest()
  assert key != quota.affinity_key("u2", "sess-123")
  assert key != quota.affinity_key("u1", "sess-124")
  assert key != quota.affinity_key("u1", "sess-123", "other")
  assert "sess-123" not in (key or "")


# ---------------------------------------------------------------------------
# model family and routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
  "model,family",
  [
    ("claude-opus-4-1-20250805", "opus"),
    ("claude-sonnet-4-5-20250929", "sonnet"),
    ("claude-3-5-haiku-latest", "haiku"),
    ("claude-fable-5-1", "fable"),
    ("claude-mythos-5", "mythos"),
    ("Opus", "opus"),
    ("gpt-5", None),
    ("", None),
  ],
)
def test_model_family(model: str, family: str | None) -> None:
  assert quota.model_family(model) == family


@pytest.mark.parametrize(
  "claim,scope",
  [
    (None, "shared"),
    ("", "shared"),
    ("five_hour", "shared"),
    ("5h", "shared"),
    ("seven_day", "shared"),
    ("7d", "shared"),
    ("seven_day_opus", "opus"),
    ("seven_day_sonnet", "sonnet"),
    ("seven_day_haiku", "haiku"),
    ("seven_day_fable", "fable"),
    ("seven_day_mythos", "mythos"),
    ("7d_oi", "fable"),
    ("seven_day_oi", "fable"),
    ("SEVEN-DAY-OI", "fable"),
    ("anthropic-ratelimit-unified-7d_oi", "fable"),
    ("unrecognized_limit", "shared"),
    ("seven_day_opuslike", "shared"),
  ],
)
def test_claim_scope_uses_family_and_window_mappings(claim: str | None, scope: str) -> None:
  assert quota.claim_scope(claim) == scope


@pytest.mark.parametrize(
  "base_url,allowed",
  [
    (None, True),
    ("", True),
    ("https://api.anthropic.com", True),
    ("https://api.anthropic.com/v1", True),
    ("https://api.anthropic.com:443", True),
    ("http://api.anthropic.com", False),
    ("https://api.anthropic.com:8443", False),
    ("https://api.anthropic.com.evil.example", False),
    ("https://api.anthropic.com@evil.example", False),
    ("https://evil.example", False),
    ("http://127.0.0.1:8123", False),
  ],
)
def test_usage_probe_allowed_is_the_direct_authority_only(base_url: str | None, allowed: bool) -> None:
  assert quota.usage_probe_allowed(base_url) is allowed


# ---------------------------------------------------------------------------
# header parsing
# ---------------------------------------------------------------------------


def test_headers_describe_only_the_windows_present() -> None:
  headers = {
    "anthropic-ratelimit-unified-5h-utilization": "0.42",
    "anthropic-ratelimit-unified-5h-reset": str(int(NOW + 3600)),
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
  }
  windows = quota.parse_rate_limit_headers(headers, now=NOW)
  assert [w.id for w in windows] == ["five_hour"]
  w = windows[0]
  assert (w.scope, w.used_fraction, w.reset_at, w.duration_s) == ("shared", 0.42, NOW + 3600, FIVE_HOURS)
  assert (w.observed_at, w.source, w.status) == (NOW, "response_header", "allowed")


def test_headers_parse_both_windows_iso_resets_and_any_header_case() -> None:
  headers = httpx.Headers(
    {
      "Anthropic-Ratelimit-Unified-5H-Utilization": "0.1",
      "Anthropic-Ratelimit-Unified-7D-Utilization": "0.9",
      "Anthropic-Ratelimit-Unified-7D-Reset": iso(NOW + 86400),
    }
  )
  by_id = {w.id: w for w in quota.parse_rate_limit_headers(headers, now=NOW)}
  assert by_id["five_hour"].used_fraction == 0.1 and by_id["five_hour"].reset_at is None
  assert by_id["seven_day"].used_fraction == 0.9
  assert by_id["seven_day"].reset_at == pytest.approx(NOW + 86400)
  assert by_id["seven_day"].duration_s == WEEK


def test_headers_model_scoped_weekly_window() -> None:
  headers = {"anthropic-ratelimit-unified-7d_oi-utilization": "0.5"}
  (w,) = quota.parse_rate_limit_headers(headers, now=NOW)
  assert (w.id, w.scope, w.used_fraction) == ("seven_day_fable", "fable", 0.5)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "", "abc", " "])
def test_header_unknown_values_are_never_zero(raw: str) -> None:
  headers = {"anthropic-ratelimit-unified-5h-utilization": raw}
  assert quota.parse_rate_limit_headers(headers, now=NOW) == ()


def test_header_fractions_clamp_and_rejected_status_keeps_unknown_use() -> None:
  high = quota.parse_rate_limit_headers({"anthropic-ratelimit-unified-5h-utilization": "1.7"}, now=NOW)
  low = quota.parse_rate_limit_headers({"anthropic-ratelimit-unified-5h-utilization": "-0.3"}, now=NOW)
  assert high[0].used_fraction == 1.0 and low[0].used_fraction == 0.0
  (rejected,) = quota.parse_rate_limit_headers(
    {
      "anthropic-ratelimit-unified-7d-status": "rejected",
      "anthropic-ratelimit-unified-7d-reset": str(int(NOW + 600)),
    },
    now=NOW,
  )
  assert rejected.used_fraction is None and rejected.status == "rejected"
  assert rejected.reset_at == NOW + 600


def test_header_reset_small_numbers_are_delays_and_nonpositive_is_unknown() -> None:
  (delay,) = quota.parse_rate_limit_headers(
    {
      "anthropic-ratelimit-unified-5h-utilization": "0.2",
      "anthropic-ratelimit-unified-5h-reset": "120",
    },
    now=NOW,
  )
  assert delay.reset_at == NOW + 120
  (none,) = quota.parse_rate_limit_headers(
    {
      "anthropic-ratelimit-unified-5h-utilization": "0.2",
      "anthropic-ratelimit-unified-5h-reset": "0",
    },
    now=NOW,
  )
  assert none.reset_at is None


# ---------------------------------------------------------------------------
# usage report parsing
# ---------------------------------------------------------------------------


def test_usage_report_percentages_become_fractions() -> None:
  payload = {
    "five_hour": {"utilization": 40.0, "resets_at": iso(NOW + 3600)},
    "seven_day": {"utilization": 12.5, "resets_at": iso(NOW + 86400)},
    "seven_day_opus": None,
    "seven_day_sonnet": {"utilization": 80, "resets_at": None},
  }
  by_id = {w.id: w for w in quota.parse_usage_report(payload, now=NOW)}
  assert set(by_id) == {"five_hour", "seven_day", "seven_day_sonnet"}
  assert by_id["five_hour"].used_fraction == pytest.approx(0.4)
  assert by_id["five_hour"].reset_at == pytest.approx(NOW + 3600)
  assert by_id["seven_day"].used_fraction == pytest.approx(0.125)
  assert by_id["seven_day_sonnet"].scope == "sonnet"
  assert by_id["seven_day_sonnet"].reset_at is None
  assert all(w.source == "usage" and w.observed_at == NOW for w in by_id.values())
  assert by_id["five_hour"].duration_s == FIVE_HOURS and by_id["seven_day"].duration_s == WEEK


def test_usage_report_generic_limits_fill_missing_buckets_and_scoped_weekly() -> None:
  payload = {
    "five_hour": None,
    "seven_day": None,
    "limits": [
      {"kind": "session", "percent": 30, "resets_at": iso(NOW + 100)},
      {"kind": "weekly_all", "percent": 55, "resets_at": iso(NOW + 200)},
      {
        "kind": "weekly_scoped",
        "percent": 100,
        "resets_at": iso(NOW + 300),
        "scope": {"model": {"display_name": "Opus 5"}},
        "is_active": False,
      },
      {"kind": "weekly_scoped", "percent": 20, "scope": {"model": {"display_name": "Fable"}}},
      {"kind": "weekly_scoped", "percent": 20, "scope": {"model": {}}},
      {"kind": "weekly_scoped", "percent": "nan", "scope": {"model": {"display_name": "Haiku"}}},
    ],
  }
  by_id = {w.id: w for w in quota.parse_usage_report(payload, now=NOW)}
  assert set(by_id) == {"five_hour", "seven_day", "seven_day_opus-5", "seven_day_fable"}
  assert by_id["five_hour"].used_fraction == pytest.approx(0.3)
  assert by_id["seven_day"].used_fraction == pytest.approx(0.55)
  assert (by_id["seven_day_opus-5"].scope, by_id["seven_day_opus-5"].used_fraction) == ("opus", 1.0)
  assert by_id["seven_day_fable"].scope == "fable"


def test_usage_report_legacy_bucket_wins_over_duplicate_scoped_row() -> None:
  payload = {
    "seven_day_opus": {"utilization": 10},
    "limits": [
      {"kind": "weekly_scoped", "percent": 90, "scope": {"model": {"display_name": "Opus"}}}
    ],
  }
  windows = quota.parse_usage_report(payload, now=NOW)
  assert [(w.id, w.used_fraction) for w in windows] == [("seven_day_opus", pytest.approx(0.1))]


@pytest.mark.parametrize("raw", ["nan", float("inf"), None, "abc", True])
def test_usage_report_unknown_utilization_is_omitted_not_zero(raw: Any) -> None:
  payload = {"five_hour": {"utilization": raw, "resets_at": iso(NOW + 60)}}
  assert quota.parse_usage_report(payload, now=NOW) == ()


def test_usage_report_clamps_and_tolerates_junk() -> None:
  payload = {"five_hour": {"utilization": 250}, "seven_day": {"utilization": -5}}
  by_id = {w.id: w.used_fraction for w in quota.parse_usage_report(payload, now=NOW)}
  assert by_id == {"five_hour": 1.0, "seven_day": 0.0}
  not_a_mapping: Any = []
  assert quota.parse_usage_report(not_a_mapping, now=NOW) == ()
  assert quota.parse_usage_report({"limits": "nope"}, now=NOW) == ()


# ---------------------------------------------------------------------------
# ranker: formula
# ---------------------------------------------------------------------------


def test_required_drain_formula_examples() -> None:
  measured = quota._Measured
  # OMP example: 92% used, resets in 15 minutes -> 0.08 / 0.25 = 0.32 per hour.
  assert quota._required_drain(measured(week(0.92, reset_in=900), 0.92), NOW) == pytest.approx(0.32)
  # 55% used, resets in 144h -> 0.45 / 144 = 0.003125 per hour.
  assert quota._required_drain(
    measured(week(0.55, reset_in=144 * HOUR), 0.55), NOW
  ) == pytest.approx(0.003125)
  # No reset clock: the full window remains.
  assert quota._required_drain(measured(week(0.45), 0.45), NOW) == pytest.approx(0.55 / 168)
  assert quota._required_drain(measured(five(0.5), 0.5), NOW) == pytest.approx(0.5 / 5)
  # One-minute floor on an imminent reset; reset beyond the duration is capped.
  assert quota._required_drain(measured(week(0.5, reset_in=10), 0.5), NOW) == pytest.approx(30.0)
  assert quota._required_drain(
    measured(week(0.5, reset_in=2 * WEEK), 0.5), NOW
  ) == pytest.approx(0.5 / 168)
  assert quota._required_drain(measured(week(1.0, reset_in=900), 1.0), NOW) == 0.0


# ---------------------------------------------------------------------------
# ranker: spec section 6 table
# ---------------------------------------------------------------------------


def test_row_sooner_expiring_allowance_wins() -> None:
  a = cand("A", windows=(five(0.1, reset_in=3 * HOUR), week(0.4, reset_in=6 * HOUR)))
  b = cand("B", windows=(five(0.1, reset_in=3 * HOUR), week(0.2, reset_in=96 * HOUR)))
  decision = rank([b, a])
  assert names(decision) == ["A", "B"]
  assert decision.reason == quota.REASON_QUOTA_DRAIN and decision.tier == "private"


def test_row_hot_short_window_beats_weekly_drain() -> None:
  def pool(a_short: float) -> list[RankCandidate]:
    return [
      cand("A", windows=(five(a_short, reset_in=3 * HOUR), week(0.4, reset_in=6 * HOUR))),
      cand("B", windows=(five(0.30, reset_in=3 * HOUR), week(0.2, reset_in=96 * HOUR))),
    ]

  hot = rank(pool(0.86))
  assert names(hot) == ["B", "A"] and hot.reason == quota.REASON_QUOTA_DRAIN
  assert names(rank(pool(0.84))) == ["A", "B"]


def test_heat_boundary_equality_is_hot() -> None:
  def pool(a_short: float) -> list[RankCandidate]:
    return [
      cand("A", windows=(five(a_short), week(0.4, reset_in=6 * HOUR))),
      cand("B", windows=(five(0.30), week(0.2, reset_in=96 * HOUR))),
    ]

  assert names(rank(pool(0.85))) == ["B", "A"]
  assert names(rank(pool(0.8499))) == ["A", "B"]
  both_hot = rank(
    [
      cand("A", reserve=0.0, windows=(five(0.9),)),
      cand("B", reserve=0.0, windows=(five(0.95),)),
    ]
  )
  assert both_hot.reason == quota.REASON_SHORT_WINDOW_HOT


def test_row_reserve_prefers_sibling_but_never_refuses_the_last_account() -> None:
  a = cand("A", windows=(week(0.92, reset_in=HOUR),))
  b = cand("B", windows=(week(0.60, reset_in=96 * HOUR),))
  assert names(rank([a, b])) == ["B", "A"]
  org = cand("C", tier="org", windows=(week(0.1, reset_in=96 * HOUR),))
  only_a = rank([a, org])
  assert names(only_a) == ["A"] and only_a.tier == "private"
  assert only_a.reason == quota.REASON_RESERVE


@pytest.mark.parametrize("short_used", [0.85, 0.89])
def test_reserve_precedes_short_window_heat(short_used: float) -> None:
  hot_outside_reserve = cand(
    "A", windows=(five(short_used, reset_in=HOUR), week(0.2, reset_in=96 * HOUR))
  )
  nonhot_in_reserve = cand(
    "B", windows=(five(0.2, reset_in=HOUR), week(0.95, reset_in=HOUR))
  )
  for pool in ([hot_outside_reserve, nonhot_in_reserve], [nonhot_in_reserve, hot_outside_reserve]):
    decision = rank(pool)
    assert names(decision) == ["A", "B"]
    assert decision.reason == quota.REASON_SHORT_WINDOW_HOT


@pytest.mark.parametrize(
  "used,reserve,in_reserve",
  [
    (0.90, 10.0, True),  # remaining 0.09999999999999998 <= 0.10
    (0.89, 10.0, False),
    (0.85, 15.0, True),  # remaining 0.15000000000000002 <= 0.15
    (0.84, 15.0, False),
    (0.999, 0.0, False),  # reserve disabled
  ],
)
def test_reserve_boundary_equality(used: float, reserve: float, in_reserve: bool) -> None:
  a = cand("A", reserve=reserve, windows=(week(used, reset_in=60),))
  b = cand("B", windows=(week(0.5, reset_in=96 * HOUR),))
  expected = ["B", "A"] if in_reserve else ["A", "B"]
  assert names(rank([a, b])) == expected


def test_row_warm_affinity_survives_better_drain_until_idle() -> None:
  a = cand("A", windows=(week(0.5, reset_in=96 * HOUR),))
  b = cand("B", windows=(week(0.1, reset_in=24 * HOUR),))
  assert names(rank([a, b])) == ["B", "A"]
  kept = rank([a, b], affinity=warm("A", idle=3599.0))
  assert names(kept) == ["A", "B"] and kept.reason == quota.REASON_WARM
  assert names(rank([a, b], affinity=warm("A", idle=3600.0))) == ["B", "A"]
  assert names(rank([a, b], affinity=warm("A", idle=7200.0))) == ["B", "A"]


def test_row_reserve_eviction_needs_a_measured_non_reserve_sibling() -> None:
  in_reserve = (week(0.92, reset_in=HOUR),)
  healthy = (week(0.50, reset_in=96 * HOUR),)
  a = cand("A", windows=in_reserve)
  evicted = rank([a, cand("B", windows=healthy)], affinity=warm("A"))
  assert names(evicted) == ["B", "A"] and evicted.reason == quota.REASON_QUOTA_DRAIN
  # A stale or missing sibling measurement cannot evict.
  for sibling in (
    cand("B", windows=(week(0.50, reset_in=96 * HOUR, age=400),)),
    cand("B"),
    cand("B", windows=(week(0.95, reset_in=HOUR),)),  # also inside reserve
  ):
    kept = rank([a, sibling], affinity=warm("A"))
    assert names(kept)[0] == "A" and kept.reason == quota.REASON_WARM
  # Stale reserve evidence on the warm account, or reserve disabled, cannot evict either.
  stale_a = cand("A", windows=(week(0.92, reset_in=HOUR, age=400),))
  assert names(rank([stale_a, cand("B", windows=healthy)], affinity=warm("A")))[0] == "A"
  no_reserve = cand("A", reserve=0.0, windows=in_reserve)
  assert names(rank([no_reserve, cand("B", windows=healthy)], affinity=warm("A")))[0] == "A"


def test_warm_affinity_is_per_tier_and_requires_a_usable_account() -> None:
  private = cand("A", windows=(week(0.5, reset_in=96 * HOUR),))
  org = cand("O", tier="org", windows=(week(0.1, reset_in=24 * HOUR),))
  assert names(rank([private, org], affinity=warm("O"))) == ["A"]
  blocked = cand("B", cooldowns=(Cooldown("shared", NOW + 60, "quota"),))
  assert names(rank([blocked, private], affinity=warm("B"))) == ["A"]
  assert names(rank([private], affinity=warm("gone"))) == ["A"]


def test_row_unknown_usage_keeps_ordered_selection() -> None:
  pool = [
    cand("A", priority=0),
    cand("B", priority=2),
    cand("C", priority=1),
  ]
  decision = rank(pool)
  assert names(decision) == ["B", "C", "A"]
  assert decision.reason == quota.REASON_QUOTA_UNAVAILABLE
  equal = [cand("A"), cand("B"), cand("C")]
  assert names(rank(list(reversed(equal)))) == ["A", "B", "C"]


def test_row_opus_exhaustion_leaves_sonnet_eligible() -> None:
  opus_exhausted = win("seven_day_opus", 1.0, reset_in=2 * HOUR, scope="opus")
  a = cand("A", windows=(five(0.1), week(0.2), opus_exhausted))
  b = cand("B", windows=(five(0.1), week(0.5)))
  opus = rank([a, b], family="opus")
  assert names(opus) == ["B"]
  assert opus.blocked == ((a, NOW + 2 * HOUR),)
  assert sorted(names(rank([a, b], family="sonnet"))) == ["A", "B"]
  assert sorted(names(rank([a, b], family=None))) == ["A", "B"]


def test_shared_exhaustion_blocks_every_family_and_latest_reset_wins() -> None:
  a = cand(
    "A",
    windows=(
      five(1.0, reset_in=1000),
      win("seven_day_opus", 1.0, reset_in=7000, scope="opus"),
    ),
    cooldowns=(Cooldown("shared", NOW + 5000, "quota"),),
  )
  b = cand("B")
  for family, until in ((None, 5000), ("sonnet", 5000), ("opus", 7000)):
    decision = rank([a, b], family=family)
    assert names(decision) == ["B"]
    assert decision.blocked == ((a, NOW + until),)


def test_block_requires_a_fresh_future_reset() -> None:
  stale = cand("A", windows=(five(1.0, reset_in=1000, age=400),))
  assert names(rank([stale])) == ["A"]
  past = cand("A", windows=(five(1.0, reset_in=-1),))
  assert names(rank([past])) == ["A"]
  no_reset = cand("A", windows=(five(1.0),))
  assert names(rank([no_reset])) == ["A"]
  status_only = cand("A", windows=(win("seven_day", None, reset_in=600, status="rejected"),))
  assert rank([status_only]).blocked == ((status_only, NOW + 600),)
  expired_cooldown = cand("A", cooldowns=(Cooldown("shared", NOW, "quota"),))
  assert names(rank([expired_cooldown])) == ["A"]


def test_expired_and_stale_windows_are_unknown_never_measured() -> None:
  # A would win every measured comparison (priority, drain) if its window counted.
  expired = cand("A", reserve=0.0, priority=5, windows=(week(0.1, reset_in=-60),))
  stale = cand("S", reserve=0.0, priority=5, windows=(week(0.0, reset_in=HOUR, age=301),))
  measured = cand("B", windows=(week(0.5, reset_in=96 * HOUR),))
  assert names(rank([expired, stale, measured])) == ["B", "A", "S"]
  just_fresh = cand("F", windows=(week(0.0, reset_in=HOUR, age=300),))
  assert names(rank([measured, just_fresh]))[0] == "F"


def test_private_tier_precedes_org_and_org_is_a_fallback() -> None:
  private_blocked = cand("A", cooldowns=(Cooldown("shared", NOW + 60, "auth"),))
  org = cand("O", tier="org")
  decision = rank([private_blocked, org])
  assert (names(decision), decision.tier, decision.reason) == (["O"], "org", quota.REASON_ORG_FALLBACK)
  assert decision.blocked == ((private_blocked, NOW + 60),)
  in_reserve = cand("A", windows=(week(0.95, reset_in=HOUR),))
  healthy_org = cand("O", tier="org", windows=(week(0.1, reset_in=HOUR),))
  assert names(rank([healthy_org, in_reserve])) == ["A"]
  only_org = rank([cand("O", tier="org")])
  assert only_org.reason == quota.REASON_ORDERED and only_org.tier == "org"


def test_all_blocked_sets_earliest_reset_fallback_with_stable_ties() -> None:
  soon = cand("B", cooldowns=(Cooldown("shared", NOW + 100, "quota"),))
  later = cand("A", tier="org", cooldowns=(Cooldown("shared", NOW + 900, "quota"),))
  tie_first = cand("C", cooldowns=(Cooldown("shared", NOW + 100, "rate_limit"),))
  decision = rank([later, tie_first, soon])
  assert decision.ordered == () and decision.tier is None
  assert decision.reason == quota.REASON_ALL_BLOCKED_PROBE
  assert decision.fallback == soon  # order 'B' < 'C'
  assert [(c.locator.identity, until) for c, until in decision.blocked] == [
    ("B", NOW + 100),
    ("C", NOW + 100),
    ("A", NOW + 900),
  ]
  assert rank([]).fallback is None and rank([]).ordered == ()


def test_scoped_cooldown_applies_only_to_its_family() -> None:
  a = cand("A", cooldowns=(Cooldown("opus", NOW + 100, "quota"),))
  assert names(rank([a], family="opus")) == []
  assert names(rank([a], family="sonnet")) == ["A"]
  assert names(rank([a], family=None)) == ["A"]


def test_most_pressured_weekly_window_selects_the_drain_basis() -> None:
  reset = 100 * HOUR
  a = cand(
    "A",
    windows=(
      week(0.1, reset_in=reset),
      win("seven_day_opus", 0.5, reset_in=reset, scope="opus"),
    ),
  )
  b = cand("B", windows=(week(0.3, reset_in=reset),))
  # Opus window is the more pressured one: drain 0.5/100 = 0.005 < B's 0.7/100 = 0.007.
  assert names(rank([a, b], family="opus")) == ["B", "A"]
  # Without the opus scope only the shared window applies: 0.9/100 = 0.009 > 0.007.
  assert names(rank([a, b], family="sonnet")) == ["A", "B"]
  assert names(rank([a, b], family=None)) == ["A", "B"]


def test_all_windows_participate_in_reserve() -> None:
  a = cand("A", windows=(five(0.97, reset_in=HOUR), week(0.1, reset_in=HOUR)))
  b = cand("B", windows=(five(0.2, reset_in=HOUR), week(0.1, reset_in=96 * HOUR)))
  assert names(rank([a, b])) == ["B", "A"]


def test_clockless_windows_use_the_full_duration() -> None:
  a = cand("A", windows=(week(0.5),))
  b = cand("B", windows=(week(0.5, reset_in=24 * HOUR),))
  assert names(rank([a, b])) == ["B", "A"]  # 0.5/24 > 0.5/168


def test_priority_precedes_weekly_drain_within_measured_cohort() -> None:
  a = cand("A", priority=1, windows=(week(0.5, reset_in=160 * HOUR),))
  b = cand("B", priority=0, windows=(week(0.1, reset_in=HOUR),))
  assert names(rank([b, a])) == ["A", "B"]


def test_session_seeded_tie_order_is_deterministic() -> None:
  pool = [cand(n) for n in "ABCDE"]
  expected = sorted(
    pool, key=lambda c: hashlib.sha256(f"seed-1\0{c.locator.id}".encode()).hexdigest()
  )
  for _ in range(3):
    random.Random(1).shuffle(pool)
    assert names(rank(pool, seed="seed-1")) == [c.locator.identity for c in expected]
  assert names(rank(list(reversed(pool)), seed=None)) == list("ABCDE")


# ---------------------------------------------------------------------------
# ranker: total order
# ---------------------------------------------------------------------------


def _mixed_pool(seed: int, size: int = 8) -> list[RankCandidate]:
  rng = random.Random(seed)
  pool = []
  for index in range(size):
    windows: list[QuotaWindow] = []
    if rng.random() < 0.7:
      windows.append(week(rng.choice([0.0, 0.3, 0.5, 0.9, 0.9]), reset_in=rng.choice([None, 600.0, 36 * HOUR, 120 * HOUR]), age=rng.choice([0.0, 0.0, 400.0])))
    if rng.random() < 0.7:
      windows.append(five(rng.choice([0.1, 0.5, 0.85, 0.9]), reset_in=rng.choice([None, 900.0, 3 * HOUR]), age=rng.choice([0.0, 0.0, 400.0])))
    pool.append(
      cand(
        chr(ord("A") + index),
        priority=rng.choice([0, 0, 1]),
        reserve=rng.choice([10.0, 10.0, 0.0]),
        windows=tuple(windows),
      )
    )
  return pool


@pytest.mark.parametrize("seed", [None, "session-a"])
def test_comparator_is_a_transitive_total_order(seed: str | None) -> None:
  for trial in range(6):
    pool = _mixed_pool(trial)
    full = names(rank(pool, seed=seed))
    assert sorted(full) == sorted(c.locator.identity for c in pool)
    position = {name: index for index, name in enumerate(full)}
    for first, second in itertools.combinations(pool, 2):
      winner = names(rank([first, second], seed=seed))[0]
      expected = min(first.locator.identity, second.locator.identity, key=position.__getitem__)
      assert winner == expected


@pytest.mark.parametrize("seed", [None, "session-a"])
def test_ranking_is_stable_under_input_permutation(seed: str | None) -> None:
  for trial in range(4):
    pool = _mixed_pool(100 + trial)
    reference = names(rank(pool, seed=seed))
    rng = random.Random(trial)
    for _ in range(20):
      shuffled = list(pool)
      rng.shuffle(shuffled)
      assert names(rank(shuffled, seed=seed)) == reference


def test_three_way_missing_window_cases_are_permutation_stable() -> None:
  pool = [
    cand("A", windows=(week(0.4, reset_in=6 * HOUR),)),
    cand("B", windows=(five(0.4, reset_in=HOUR),)),
    cand("C"),
    cand("D", windows=(five(0.4, reset_in=HOUR), week(0.4, reset_in=6 * HOUR))),
  ]
  reference = names(rank(pool))
  for permutation in itertools.permutations(pool):
    assert names(rank(list(permutation))) == reference
  # Measured accounts precede the unknown one; missing window groups sort after present ones.
  assert reference[-1] == "C" and reference[0] == "D"


# ---------------------------------------------------------------------------
# sidecar
# ---------------------------------------------------------------------------


def _store(tmp_path: Path, name: str = "oauth.json") -> Path:
  return tmp_path / name


def _loc(store: Path, identity: str = "acct@example.com") -> AccountLocator:
  return quota.locator_for_record(store, identity)


def test_sidecar_name_derives_from_the_exact_store_filename(tmp_path: Path) -> None:
  assert quota.selection_sidecar_path(tmp_path / "oauth.json").name == "oauth.json.selection.json"
  assert quota.selection_sidecar_path(tmp_path / "custom.json").name == "custom.json.selection.json"


def test_missing_sidecar_is_an_empty_state(tmp_path: Path) -> None:
  state = quota.load_selection_state(_store(tmp_path), now=NOW)
  assert (state.policies, state.observations, state.cooldowns, state.affinity, state.problems) == ({}, {}, {}, {}, ())


def test_distinct_custom_stores_in_one_directory_are_independent(tmp_path: Path) -> None:
  one, two = _store(tmp_path, "one.json"), _store(tmp_path, "two.json")
  locator = quota.AccountLocator("store:shared-name", "same")
  quota.set_account_policy(one, locator, priority=7)
  quota.record_success(one, locator, "k1", now=NOW)
  quota.record_observation(one, locator, [five(0.4)], now=NOW)
  quota.record_cooldown(one, locator, Cooldown("shared", NOW + 60, "quota"))
  state_two = quota.load_selection_state(two, now=NOW)
  assert state_two.policies == {} and state_two.affinity == {} and state_two.observations == {}
  assert state_two.cooldowns == {}
  assert quota.selection_sidecar_path(one).exists() and not quota.selection_sidecar_path(two).exists()
  quota.set_account_policy(two, locator, priority=1)
  assert quota.load_selection_state(one, now=NOW).policies[locator.id].priority == 7
  assert quota.load_selection_state(two, now=NOW).policies[locator.id].priority == 1


def test_sidecar_is_mode_0600_with_its_own_lock_file(tmp_path: Path) -> None:
  store = _store(tmp_path, "custom.json")
  quota.record_success(store, _loc(store), "key", now=NOW)
  sidecar = quota.selection_sidecar_path(store)
  assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
  assert (tmp_path / ".custom.json.selection.json.lock").exists()
  assert not (tmp_path / ".custom.json.lock").exists()


def test_policy_roundtrip_defaults_reset_and_never_pruned(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  assert quota.set_account_policy(store, locator, priority=3) == AccountPolicy(3, 10.0)
  assert quota.set_account_policy(store, locator, reserve_pct=0) == AccountPolicy(3, 0.0)
  assert quota.set_account_policy(store, locator, reserve_pct=12.5, priority=-1) == AccountPolicy(-1, 12.5)
  far_future = NOW + 10 * 365 * 86400
  assert quota.load_selection_state(store, now=far_future).policies[locator.id] == AccountPolicy(-1, 12.5)
  quota.update_selection_state(store, lambda raw: None, now=far_future)
  assert quota.load_selection_state(store, now=far_future).policies[locator.id] == AccountPolicy(-1, 12.5)
  quota.reset_account_policy(store, locator)
  assert locator.id not in quota.load_selection_state(store, now=NOW).policies
  quota.reset_account_policy(store, locator)  # idempotent


@pytest.mark.parametrize("priority", ["1", True, 1.5, []])
def test_policy_rejects_non_integer_priority(tmp_path: Path, priority: Any) -> None:
  store = _store(tmp_path)
  with pytest.raises(ValueError):
    quota.set_account_policy(store, _loc(store), priority=priority)
  assert not quota.selection_sidecar_path(store).exists()


@pytest.mark.parametrize("reserve", [-0.1, 100.1, float("nan"), float("inf"), True, "5", []])
def test_policy_rejects_malformed_reserve(tmp_path: Path, reserve: Any) -> None:
  store = _store(tmp_path)
  with pytest.raises(ValueError):
    quota.set_account_policy(store, _loc(store), reserve_pct=reserve)
  assert not quota.selection_sidecar_path(store).exists()


def test_header_merge_never_renews_another_window_and_late_probe_is_ignored(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.record_observation(
    store,
    locator,
    [five(0.4, age=100, reset_in=HOUR), week(0.2, age=100, reset_in=48 * HOUR)],
    now=NOW,
  )
  header_five = QuotaWindow("five_hour", "shared", 0.6, NOW + 2 * HOUR, FIVE_HOURS, NOW, "response_header")
  quota.record_observation(store, locator, [header_five], now=NOW)
  windows = {
    w.id: w for w in quota.load_selection_state(store, now=NOW).observations[locator.id].windows
  }
  assert windows["five_hour"].used_fraction == 0.6 and windows["five_hour"].source == "response_header"
  assert windows["seven_day"].observed_at == NOW - 100  # a 5h header did not renew 7d
  # A probe that started before the header lands late: it cannot overwrite the newer 5h
  # window, but it can still refresh the 7d window it measured more recently.
  late_probe = [five(0.1, age=50), week(0.3, age=50, reset_in=48 * HOUR)]
  quota.record_observation(store, locator, late_probe, now=NOW)
  windows = {
    w.id: w for w in quota.load_selection_state(store, now=NOW).observations[locator.id].windows
  }
  assert windows["five_hour"].used_fraction == 0.6
  assert (windows["seven_day"].used_fraction, windows["seven_day"].observed_at) == (0.3, NOW - 50)


def test_observation_merge_is_per_window_scope_pair(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.record_observation(store, locator, [week(0.2), win("seven_day", 0.9, scope="opus")], now=NOW)
  quota.record_observation(store, locator, [week(0.3, age=-5)], now=NOW)
  windows = quota.load_selection_state(store, now=NOW).observations[locator.id].windows
  assert {(w.scope, w.used_fraction) for w in windows} == {("shared", 0.3), ("opus", 0.9)}


def test_cooldown_keeps_the_later_until_per_scope(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.record_cooldown(store, locator, Cooldown("shared", NOW + 100, "quota"))
  quota.record_cooldown(store, locator, Cooldown("shared", NOW + 50, "rate_limit"))
  quota.record_cooldown(store, locator, Cooldown("opus", NOW + 70, "quota"))
  rows = quota.load_selection_state(store, now=NOW).cooldowns[locator.id]
  assert {(c.scope, c.until, c.reason) for c in rows} == {
    ("shared", NOW + 100, "quota"),
    ("opus", NOW + 70, "quota"),
  }
  quota.record_cooldown(store, locator, Cooldown("shared", NOW + 500, "auth"))
  rows = quota.load_selection_state(store, now=NOW).cooldowns[locator.id]
  assert {(c.scope, c.until, c.reason) for c in rows}.issuperset({("shared", NOW + 500, "auth")})
  assert len(rows) == 2


def test_probe_state_replaces_but_keeps_last_usage_time(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.record_probe_state(store, locator, UsageProbeState(next_probe_at=NOW + 300, last_usage_at=NOW - 10))
  failure = UsageProbeState(next_probe_at=NOW + 600, unavailable_until=NOW + 600, unavailable_reason="network", generation="g")
  quota.record_probe_state(store, locator, failure)
  probe = quota.load_selection_state(store, now=NOW).observations[locator.id].probe
  assert probe == UsageProbeState(NOW + 600, NOW + 600, "network", "g", NOW - 10)


def test_success_writes_affinity_only_for_a_real_key(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.record_success(store, locator, None, now=NOW)
  assert not quota.selection_sidecar_path(store).exists()
  key = quota.affinity_key("u1", "session-1")
  assert key is not None
  quota.record_success(store, locator, key, now=NOW)
  assert quota.load_selection_state(store, now=NOW).affinity[key] == Affinity(locator.id, NOW)


def test_pruning_drops_stale_data_but_not_policies(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.set_account_policy(store, locator, priority=2)

  def mutate(raw: dict) -> None:
    def window(window_id: str, observed_at: float) -> dict:
      return {
        "id": window_id, "scope": "shared", "used_fraction": 0.5, "reset_at": None,
        "duration_s": WEEK, "observed_at": observed_at, "source": "usage", "status": None,
      }

    raw["observations"]["acct_old"] = {"windows": [window("seven_day", NOW - 86401)], "probe": {}}
    raw["observations"]["acct_mixed"] = {
      "windows": [window("seven_day", NOW - 86401), window("five_hour", NOW - 86399)],
      "probe": {},
    }
    raw["cooldowns"]["acct_cd"] = [
      {"scope": "shared", "until": NOW - 1, "reason": "quota"},
      {"scope": "opus", "until": NOW + 10, "reason": "quota"},
    ]
    raw["cooldowns"]["acct_cd_dead"] = [{"scope": "shared", "until": NOW, "reason": "quota"}]
    raw["affinity"]["old"] = {"locator_id": "acct_x", "last_success_at": NOW - 30 * 86400 - 1}
    raw["affinity"]["new"] = {"locator_id": "acct_x", "last_success_at": NOW - 29 * 86400}
    raw["policies"]["acct_gone"] = {"priority": 4, "reserve_pct": 0.0}

  state = quota.update_selection_state(store, mutate, now=NOW)
  assert "acct_old" not in state.observations
  assert [w.id for w in state.observations["acct_mixed"].windows] == ["five_hour"]
  assert [c.scope for c in state.cooldowns["acct_cd"]] == ["opus"] and "acct_cd_dead" not in state.cooldowns
  assert set(state.affinity) == {"new"}
  assert state.policies["acct_gone"] == AccountPolicy(4, 0.0) and locator.id in state.policies
  on_disk = json.loads(quota.selection_sidecar_path(store).read_text())
  assert "acct_old" not in on_disk["observations"] and "old" not in on_disk["affinity"]


def test_unknown_fields_are_reported_and_ignored(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.selection_sidecar_path(store).write_text(
    json.dumps(
      {
        "version": 1,
        "mystery": 1,
        "policies": {locator.id: {"priority": 5, "reserve_pct": 20.0, "color": "red"}},
        "affinity": {"k": {"locator_id": "acct_x", "last_success_at": NOW, "extra": True}},
      }
    )
  )
  state = quota.load_selection_state(store, now=NOW)
  assert state.policies[locator.id] == AccountPolicy(5, 20.0)
  assert state.affinity["k"] == Affinity("acct_x", NOW)
  assert {"unknown_field:mystery", "unknown_field:color", "unknown_field:extra"} <= set(state.problems)


def test_invalid_rows_are_dropped_not_fatal(tmp_path: Path) -> None:
  store = _store(tmp_path)
  quota.selection_sidecar_path(store).write_text(
    json.dumps(
      {
        "version": 1,
        "policies": {"a": {"priority": "high"}, "b": {"priority": 1, "reserve_pct": 5}},
        "affinity": {"k": "oops"},
      }
    )
  )
  state = quota.load_selection_state(store, now=NOW)
  assert set(state.policies) == {"b"} and state.affinity == {}
  assert "invalid_row:policies" in state.problems and "invalid_row:affinity" in state.problems


def test_huge_sidecar_reserve_is_an_invalid_row_not_a_load_failure(tmp_path: Path) -> None:
  store = _store(tmp_path)
  quota.selection_sidecar_path(store).write_text(
    json.dumps(
      {
        "version": 1,
        "policies": {
          "huge": {"priority": 1, "reserve_pct": 10**400},
          "healthy": {"priority": 2, "reserve_pct": 15},
        },
      }
    )
  )
  state = quota.load_selection_state(store, now=NOW)
  assert state.policies == {"healthy": AccountPolicy(2, 15.0)}
  assert "invalid_row:policies" in state.problems


@pytest.mark.parametrize("content", ["{not json", "[]", '"text"', "", '{"version": 99}', "\x00\x01"])
def test_corrupt_sidecar_degrades_and_heals_on_the_next_write(tmp_path: Path, content: str) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  sidecar = quota.selection_sidecar_path(store)
  sidecar.write_text(content)
  state = quota.load_selection_state(store, now=NOW)
  assert state.problems and state.policies == {} and state.affinity == {}
  quota.record_success(store, locator, "key", now=NOW)  # never raises
  healed = quota.load_selection_state(store, now=NOW)
  assert healed.problems == () and healed.affinity["key"].locator_id == locator.id
  assert sidecar.with_name(sidecar.name + ".corrupt").read_text() == content


def test_undecodable_bytes_are_corrupt_not_fatal(tmp_path: Path) -> None:
  store = _store(tmp_path)
  quota.selection_sidecar_path(store).write_bytes(b"\xff\xfe\x00bad")
  assert "corrupt_sidecar" in quota.load_selection_state(store, now=NOW).problems


def test_unwritable_sidecar_degrades_without_raising(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.set_account_policy(store, locator, priority=1)

  def deny(path: Path, payload: Any) -> None:
    raise OSError("read-only file system")

  monkeypatch.setattr(quota, "atomic_write_json", deny)
  state = quota.update_selection_state(store, lambda raw: raw["policies"].clear(), now=NOW)
  assert "unwritable" in state.problems and state.policies[locator.id].priority == 1
  quota.record_success(store, locator, "k", now=NOW)
  quota.record_observation(store, locator, [five(0.1)], now=NOW)
  quota.record_cooldown(store, locator, Cooldown("shared", NOW + 60, "quota"))
  quota.record_probe_state(store, locator, UsageProbeState(next_probe_at=NOW))
  with pytest.raises(OSError):
    quota.set_account_policy(store, locator, priority=2)  # the operator is told
  assert quota.load_selection_state(store, now=NOW).policies[locator.id].priority == 1


@pytest.mark.parametrize("simulate_read_error", [False, True], ids=["chmod", "permission-error"])
def test_unreadable_sidecar_aborts_mutations_and_preserves_policy(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, simulate_read_error: bool
) -> None:
  if not simulate_read_error and os.geteuid() == 0:
    pytest.skip("root bypasses chmod read permissions; the PermissionError case covers that runner")
  store = _store(tmp_path)
  locator_a, locator_b = _loc(store, "A"), _loc(store, "B")
  policy_a = AccountPolicy(3, 12.5)
  quota.set_account_policy(store, locator_a, priority=policy_a.priority, reserve_pct=policy_a.reserve_pct)
  sidecar = quota.selection_sidecar_path(store)
  original_mode = stat.S_IMODE(sidecar.stat().st_mode)
  original_bytes = sidecar.read_bytes()
  original_read_text = Path.read_text
  mutated = False

  def deny_sidecar_read(path: Path, *args: Any, **kwargs: Any) -> str:
    if path == sidecar:
      raise PermissionError("selection sidecar is unreadable")
    return original_read_text(path, *args, **kwargs)

  def mutate(raw: dict[str, Any]) -> None:
    nonlocal mutated
    mutated = True
    raw.clear()

  try:
    sidecar.chmod(0o000)
    with monkeypatch.context() as patch:
      if simulate_read_error:
        patch.setattr(Path, "read_text", deny_sidecar_read)
      with pytest.raises(OSError):
        quota.set_account_policy(store, locator_b, priority=7)
      with pytest.raises(OSError):
        quota.reset_account_policy(store, locator_a)
      state = quota.update_selection_state(store, mutate, now=NOW)
      assert "unreadable_sidecar" in state.problems
      assert not mutated
      assert stat.S_IMODE(sidecar.stat().st_mode) == 0o000
  finally:
    sidecar.chmod(original_mode)
  assert sidecar.read_bytes() == original_bytes
  state = quota.load_selection_state(store, now=NOW)
  assert state.policies == {locator_a.id: policy_a}
  assert locator_b.id not in state.policies


def test_lock_failure_degrades_without_raising(tmp_path: Path) -> None:
  blocker = tmp_path / "not-a-directory"
  blocker.write_text("x")
  state = quota.update_selection_state(blocker / "oauth.json", lambda raw: None, now=NOW)
  assert "unwritable" in state.problems
  quota.record_success(blocker / "oauth.json", AccountLocator("n", "i"), "k", now=NOW)


def test_sidecar_holds_no_secret_material(tmp_path: Path) -> None:
  store = _store(tmp_path)
  token = "sk-ant-oat01-very-secret"
  locator = quota.locator_for_env_token(token)
  quota.set_account_policy(store, locator, priority=1)
  quota.record_observation(store, locator, [five(0.1)], now=NOW)
  quota.record_success(store, locator, quota.affinity_key("u", "s"), now=NOW)
  assert token not in quota.selection_sidecar_path(store).read_text()


def test_updates_preserve_unrelated_rows(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator_a, locator_b = _loc(store, "a"), _loc(store, "b")
  quota.set_account_policy(store, locator_a, priority=1)
  quota.record_observation(store, locator_b, [five(0.2)], now=NOW)
  quota.set_account_policy(store, locator_b, priority=2)
  state = quota.load_selection_state(store, now=NOW)
  assert state.policies[locator_a.id].priority == 1 and state.policies[locator_b.id].priority == 2
  assert state.observations[locator_b.id].windows[0].used_fraction == 0.2


# ---------------------------------------------------------------------------
# CLI view
# ---------------------------------------------------------------------------


def test_describe_account_usage_reads_cached_state_only(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  empty = quota.describe_account_usage(quota.load_selection_state(store, now=NOW), locator, now=NOW)
  assert empty == {
    "account": locator.id, "priority": 0, "reserve_pct": 10.0, "policy_source": "default",
    "windows": [], "usage": "unknown", "unavailable_reason": None, "cooldowns": [],
  }
  quota.set_account_policy(store, locator, priority=2, reserve_pct=5)
  quota.record_observation(
    store,
    locator,
    [five(0.4, age=60, reset_in=HOUR), week(0.9, age=400, reset_in=-10)],
    now=NOW,
  )
  quota.record_cooldown(store, locator, Cooldown("opus", NOW + 90, "quota"))
  view = quota.describe_account_usage(quota.load_selection_state(store, now=NOW), locator, now=NOW)
  assert (view["priority"], view["reserve_pct"], view["policy_source"]) == (2, 5.0, "configured")
  windows = {w["id"]: w for w in view["windows"]}
  assert windows["five_hour"] == {
    "id": "five_hour", "scope": "shared", "used_pct": 40.0, "resets_in_s": HOUR,
    "age_s": 60.0, "source": "usage", "fresh": True,
  }
  assert windows["seven_day"]["used_pct"] is None and windows["seven_day"]["fresh"] is False
  assert view["usage"] == "fresh"
  assert view["cooldowns"] == [{"scope": "opus", "reason": "quota", "remaining_s": 90.0}]
  stale = quota.describe_account_usage(quota.load_selection_state(store, now=NOW + 1000), locator, now=NOW + 1000)
  assert stale["usage"] == "stale"


def test_describe_reports_unavailable_probe(tmp_path: Path) -> None:
  store = _store(tmp_path)
  locator = _loc(store)
  quota.record_probe_state(
    store, locator, UsageProbeState(next_probe_at=NOW + 600, unavailable_until=NOW + 600, unavailable_reason="http_403")
  )
  view = quota.describe_account_usage(quota.load_selection_state(store, now=NOW), locator, now=NOW)
  assert (view["usage"], view["unavailable_reason"]) == ("unavailable", "http_403")
  later = quota.describe_account_usage(quota.load_selection_state(store, now=NOW + 700), locator, now=NOW + 700)
  assert later["usage"] == "unknown" and later["unavailable_reason"] is None


# ---------------------------------------------------------------------------
# usage probe
# ---------------------------------------------------------------------------


def _target(name: str, *, bearer: str | None = None, generation: str | None = "gen-1", base_url: str | None = None) -> ProbeTarget:
  return ProbeTarget(AccountLocator("store:/probe", name), bearer or f"token-{name}", generation, base_url)


def _probe(handler: Any, targets: list[ProbeTarget], **kw: Any) -> dict[str, quota.ProbeResult]:
  return asyncio.run(
    quota.probe_usage_batch(targets, transport=httpx.MockTransport(handler), now=NOW, **kw)
  )


def _usage_body() -> dict:
  return {
    "five_hour": {"utilization": 40.0, "resets_at": iso(NOW + 3600)},
    "seven_day": {"utilization": 20.0, "resets_at": iso(NOW + 86400)},
  }


def test_probe_success_parses_windows_and_sends_bearer_and_beta_headers() -> None:
  seen: list[httpx.Request] = []

  def handler(request: httpx.Request) -> httpx.Response:
    seen.append(request)
    return httpx.Response(200, json=_usage_body())

  target = _target("A", bearer="bearer-A")
  result = _probe(handler, [target])[target.locator.id]
  (request,) = seen
  assert request.method == "GET" and str(request.url) == quota.ANTHROPIC_USAGE_URL
  assert request.headers["authorization"] == "Bearer bearer-A"
  assert request.headers["anthropic-beta"] == ANTHROPIC_OAUTH_BETA
  assert {w.id: w.used_fraction for w in result.windows} == {
    "five_hour": pytest.approx(0.4),
    "seven_day": pytest.approx(0.2),
  }
  probe = result.probe
  assert probe.last_usage_at == NOW and probe.unavailable_until is None and probe.unavailable_reason is None
  assert probe.next_probe_at is not None
  assert NOW + 300 * 0.75 <= probe.next_probe_at <= NOW + 300 * 1.25
  assert "bearer-A" not in repr(result)


def test_probe_empty_usage_report_is_a_success_with_no_windows() -> None:
  target = _target("A")
  result = _probe(lambda r: httpx.Response(200, json={"five_hour": None, "seven_day": None}), [target])[target.locator.id]
  assert result.windows == () and result.probe.last_usage_at == NOW


def test_huge_usage_utilization_does_not_discard_a_healthy_batch_result() -> None:
  malformed, healthy = _target("huge"), _target("healthy")

  def handler(request: httpx.Request) -> httpx.Response:
    if request.headers["authorization"] == "Bearer token-huge":
      return httpx.Response(
        200,
        json={"seven_day": {"utilization": 10**400, "resets_at": iso(NOW + 3 * 86400)}},
      )
    return httpx.Response(200, json=_usage_body())

  results = _probe(handler, [malformed, healthy])
  assert set(results) == {malformed.locator.id, healthy.locator.id}
  assert results[malformed.locator.id].windows == ()
  assert results[malformed.locator.id].probe.last_usage_at == NOW
  healthy_result = results[healthy.locator.id]
  assert {window.id: window.used_fraction for window in healthy_result.windows} == {
    "five_hour": pytest.approx(0.4),
    "seven_day": pytest.approx(0.2),
  }
  assert healthy_result.probe.last_usage_at == NOW


@pytest.mark.parametrize("jitter", [1.25, 0.75], ids=["longer-than-320s", "shorter-than-ttl"])
def test_exhausted_usage_stays_blocked_until_its_jittered_probe_deadline(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, jitter: float
) -> None:
  def fixed_jitter(lower: float, upper: float) -> float:
    assert lower <= jitter <= upper
    return jitter

  monkeypatch.setattr(quota.random, "uniform", fixed_jitter)
  store = _store(tmp_path)
  target = ProbeTarget(_loc(store), "test-bearer", "g1", None)
  reset_at = NOW + 3 * 86400
  result = _probe(
    lambda request: httpx.Response(
      200, json={"seven_day": {"utilization": 100, "resets_at": iso(reset_at)}}
    ),
    [target],
  )[target.locator.id]
  next_probe_at = result.probe.next_probe_at
  assert next_probe_at == NOW + quota.USAGE_FRESH_TTL_SECONDS * jitter
  assert next_probe_at is not None
  quota.record_observation(store, target.locator, result.windows, now=NOW)
  quota.record_probe_state(store, target.locator, result.probe)
  state = quota.load_selection_state(store, now=NOW + 320)
  candidate = RankCandidate(
    locator=target.locator,
    tier="private",
    order=0,
    policy=AccountPolicy(),
    observation=state.observations[target.locator.id],
    cooldowns=(),
  )
  decision = rank([candidate], now=NOW + 320)
  if jitter > 1:
    assert next_probe_at > NOW + 320
    assert decision.ordered == ()
    assert decision.reason == quota.REASON_ALL_BLOCKED_PROBE
    assert decision.blocked == ((candidate, reset_at),)
    assert decision.fallback == candidate
  else:
    assert next_probe_at < NOW + 320
    assert decision.ordered == (candidate,)
    assert decision.blocked == ()
  assert rank([candidate], now=next_probe_at).blocked == ((candidate, reset_at),)
  expired = rank([candidate], now=next_probe_at + 0.001)
  assert expired.ordered == (candidate,)
  assert expired.blocked == ()


@pytest.mark.parametrize("status,reason", [(401, "http_401"), (403, "http_403")])
def test_probe_auth_refusal_is_unavailable_for_an_hour_with_generation(status: int, reason: str) -> None:
  target = _target("A", generation="gen-7")
  result = _probe(lambda r: httpx.Response(status), [target])[target.locator.id]
  assert result.windows == ()
  assert result.probe == UsageProbeState(
    next_probe_at=NOW + 3600, unavailable_until=NOW + 3600, unavailable_reason=reason, generation="gen-7"
  )


@pytest.mark.parametrize(
  "retry_after,wait",
  [("120", 120.0), ("99999", 3600.0), (None, 300.0), ("garbage", 300.0), ("-5", 300.0), ("0", 300.0)],
)
def test_probe_429_respects_capped_retry_after_without_retrying(retry_after: str | None, wait: float) -> None:
  calls: list[int] = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(1)
    return httpx.Response(429, headers={} if retry_after is None else {"retry-after": retry_after})

  target = _target("A")
  result = _probe(handler, [target])[target.locator.id]
  assert len(calls) == 1
  assert result.windows == ()
  assert (result.probe.unavailable_reason, result.probe.unavailable_until) == ("http_429", NOW + wait)


def test_probe_429_accepts_an_http_date() -> None:
  stamp = datetime.fromtimestamp(NOW + 90, timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
  target = _target("A")
  result = _probe(lambda r: httpx.Response(429, headers={"retry-after": stamp}), [target])[target.locator.id]
  assert result.probe.unavailable_until == NOW + 90


def test_probe_timeout_and_transport_errors_retain_last_good_and_retry_in_five_minutes() -> None:
  def handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)

  target = _target("A")
  result = _probe(handler, [target])[target.locator.id]
  assert result.windows == ()
  assert result.probe.unavailable_reason == "network" and result.probe.next_probe_at == NOW + 300

  def refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)

  assert _probe(refuse, [target])[target.locator.id].probe.unavailable_reason == "network"


@pytest.mark.parametrize(
  "response,reason,seconds",
  [
    (httpx.Response(500), "network", 300.0),
    (httpx.Response(529), "network", 300.0),
    (httpx.Response(400), "network", 300.0),
    (httpx.Response(404), "route", 3600.0),
    (httpx.Response(200, text="not json"), "parse", 300.0),
    (httpx.Response(200, json={"hello": "world"}), "parse", 300.0),
    (httpx.Response(200, json=[1, 2]), "parse", 300.0),
  ],
)
def test_probe_other_failures(response: httpx.Response, reason: str, seconds: float) -> None:
  target = _target("A")
  result = _probe(lambda r: response, [target])[target.locator.id]
  assert result.windows == ()
  assert (result.probe.unavailable_reason, result.probe.next_probe_at) == (reason, NOW + seconds)


def test_probe_makes_no_network_call_when_the_route_is_not_allowed() -> None:
  calls: list[httpx.Request] = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(request)
    return httpx.Response(200, json=_usage_body())

  custom = _target("A", base_url="http://127.0.0.1:4567")
  direct = _target("B", base_url="https://api.anthropic.com")
  results = _probe(handler, [custom, direct])
  assert len(calls) == 1 and calls[0].headers["authorization"] == "Bearer token-B"
  assert results[custom.locator.id].windows == ()
  assert results[custom.locator.id].probe == UsageProbeState(
    next_probe_at=NOW + 3600, unavailable_until=NOW + 3600, unavailable_reason="route", generation="gen-1"
  )
  assert results[direct.locator.id].windows

  calls.clear()
  only_custom = _probe(handler, [custom])
  assert calls == [] and list(only_custom) == [custom.locator.id]


def test_probe_coalesces_duplicate_targets_and_handles_no_targets() -> None:
  calls: list[int] = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(1)
    return httpx.Response(200, json=_usage_body())

  target = _target("A")
  assert len(_probe(handler, [target, target])) == 1 and len(calls) == 1
  assert _probe(handler, []) == {}


def test_probe_batch_budget_cancels_unfinished_probes() -> None:
  started: list[str] = []
  cancelled: list[str] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    bearer = request.headers["authorization"]
    started.append(bearer)
    if bearer.endswith("fast"):
      return httpx.Response(200, json=_usage_body())
    try:
      await asyncio.sleep(30)
    except asyncio.CancelledError:
      cancelled.append(bearer)
      raise
    return httpx.Response(200, json=_usage_body())

  fast, slow, queued = _target("fast"), _target("slow"), _target("queued")
  begun = time.monotonic()
  results = _probe(handler, [fast, slow, queued], budget_s=0.1, max_concurrency=2)
  assert time.monotonic() - begun < 5
  assert set(results) == {fast.locator.id}
  assert set(cancelled) == {"Bearer token-slow", "Bearer token-queued"}
  # With one slot, the queued probe never even starts and still has no entry.
  started.clear()
  cancelled.clear()
  results = _probe(handler, [slow, queued, fast], budget_s=0.1, max_concurrency=1)
  assert results == {} and started == ["Bearer token-slow"] and cancelled == ["Bearer token-slow"]


def test_probe_budget_of_zero_makes_no_request() -> None:
  calls: list[int] = []

  def handler(request: httpx.Request) -> httpx.Response:
    calls.append(1)
    return httpx.Response(200, json=_usage_body())

  assert _probe(handler, [_target("A")], budget_s=0.0) == {}
  assert calls == []


def test_probe_respects_the_concurrency_cap() -> None:
  in_flight = 0
  peak = 0

  async def handler(request: httpx.Request) -> httpx.Response:
    nonlocal in_flight, peak
    in_flight += 1
    peak = max(peak, in_flight)
    await asyncio.sleep(0.01)
    in_flight -= 1
    return httpx.Response(200, json=_usage_body())

  targets = [_target(f"T{i}") for i in range(6)]
  results = _probe(handler, targets, budget_s=5.0, max_concurrency=2)
  assert len(results) == 6 and peak == 2


def test_probe_caller_cancellation_cancels_children() -> None:
  cancelled: list[int] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    try:
      await asyncio.sleep(30)
    except asyncio.CancelledError:
      cancelled.append(1)
      raise
    return httpx.Response(200)

  async def scenario() -> None:
    task = asyncio.ensure_future(
      quota.probe_usage_batch(
        [_target("A")], transport=httpx.MockTransport(handler), now=NOW, budget_s=10.0
      )
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

  asyncio.run(scenario())
  assert cancelled == [1]


def test_probe_result_feeds_the_sidecar(tmp_path: Path) -> None:
  store = _store(tmp_path)
  target = ProbeTarget(quota.locator_for_record(store, "acct"), "bearer-xyz", "g1", None)
  result = _probe(lambda r: httpx.Response(200, json=_usage_body()), [target])[target.locator.id]
  quota.record_observation(store, target.locator, result.windows, now=NOW)
  quota.record_probe_state(store, target.locator, result.probe)
  state = quota.load_selection_state(store, now=NOW)
  assert len(state.observations[target.locator.id].windows) == 2
  assert state.observations[target.locator.id].probe.last_usage_at == NOW
  assert "bearer-xyz" not in quota.selection_sidecar_path(store).read_text()
