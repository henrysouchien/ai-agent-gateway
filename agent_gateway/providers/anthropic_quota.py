"""Quota observations, usage parsing, the pure account ranker and the selection sidecar.

Owns four things for Anthropic OAuth/setup-token pools: normalized quota windows,
header and `/api/oauth/usage` parsing, the pure candidate ranker, and the per-store
`<store>.selection.json` sidecar (account policy, observations, probe backoff and
in-progress markers, scoped cooldowns and session affinity). It holds no secrets and does no credential
membership or refresh; those stay in `anthropic_oauth.py`. The mechanism and the
numeric examples are specified in
`docs/design/anthropic-quota-aware-account-selection-spec.md` sections 4-6; the
header/usage parsing and the drain/pressure formulas adapt OMP's
`packages/ai/src/usage/claude.ts` and `auth/usage-report.ts` at
`can1357/oh-my-pi@83c9df0ab94e7c408be66498cfc55b567ca6ed2d`.

Missing telemetry is never zero: an unknown value is `None`, a past reset makes its
window unknown, and a stale window never creates a block, evicts affinity or wins a
measured comparison. Successful usage stays fresh until its scheduled next probe;
independent header windows use `USAGE_FRESH_TTL_SECONDS`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, TypeGuard
from urllib.parse import urlsplit

import httpx

from .anthropic_oauth import ANTHROPIC_OAUTH_BETA
from .credential_pool import atomic_write_json, store_write_lock

__all__ = (
  "SELECTION_SIDECAR_SUFFIX",
  "SELECTION_STATE_VERSION",
  "DEFAULT_PRIORITY",
  "DEFAULT_RESERVE_PCT",
  "SHORT_WINDOW_HEAT",
  "WARM_IDLE_SECONDS",
  "USAGE_FRESH_TTL_SECONDS",
  "USAGE_UNAVAILABLE_SECONDS",
  "USAGE_RETRY_AFTER_CAP_SECONDS",
  "LAST_GOOD_MAX_AGE_SECONDS",
  "AFFINITY_PRUNE_SECONDS",
  "PROBE_BATCH_BUDGET_SECONDS",
  "PROBE_MAX_CONCURRENCY",
  "ANTHROPIC_USAGE_URL",
  "REASON_WARM",
  "REASON_QUOTA_DRAIN",
  "REASON_SHORT_WINDOW_HOT",
  "REASON_RESERVE",
  "REASON_ORG_FALLBACK",
  "REASON_QUOTA_UNAVAILABLE",
  "REASON_ALL_BLOCKED_PROBE",
  "REASON_ORDERED",
  "selection_sidecar_path",
  "AccountLocator",
  "locator_for_record",
  "locator_for_env_token",
  "AccountPolicy",
  "QuotaWindow",
  "Cooldown",
  "UsageProbeState",
  "AccountObservation",
  "Affinity",
  "SelectionState",
  "load_selection_state",
  "update_selection_state",
  "set_account_policy",
  "reset_account_policy",
  "record_observation",
  "record_cooldown",
  "record_probe_state",
  "claim_usage_probe",
  "record_success",
  "affinity_key",
  "model_family",
  "claim_scope",
  "window_exhausted",
  "parse_rate_limit_headers",
  "parse_usage_report",
  "usage_probe_allowed",
  "ProbeTarget",
  "ProbeResult",
  "probe_usage_batch",
  "RankCandidate",
  "RankDecision",
  "rank_candidates",
  "describe_account_usage",
)

LOGGER = logging.getLogger(__name__)

SELECTION_SIDECAR_SUFFIX = ".selection.json"
SELECTION_STATE_VERSION = 1
DEFAULT_PRIORITY = 0
DEFAULT_RESERVE_PCT = 10.0
SHORT_WINDOW_HEAT = 0.85
WARM_IDLE_SECONDS = 3600.0
USAGE_FRESH_TTL_SECONDS = 300.0
USAGE_UNAVAILABLE_SECONDS = 3600.0
USAGE_RETRY_AFTER_CAP_SECONDS = 3600.0
LAST_GOOD_MAX_AGE_SECONDS = 86400.0
AFFINITY_PRUNE_SECONDS = 30 * 86400.0
PROBE_BATCH_BUDGET_SECONDS = 2.0
PROBE_MAX_CONCURRENCY = 4
ANTHROPIC_USAGE_URL = "https://api.anthropic.com/api/oauth/usage"

REASON_WARM = "warm"
REASON_QUOTA_DRAIN = "quota_drain"
REASON_SHORT_WINDOW_HOT = "short_window_hot"
REASON_RESERVE = "reserve"
REASON_ORG_FALLBACK = "org_fallback"
REASON_QUOTA_UNAVAILABLE = "quota_unavailable"
REASON_ALL_BLOCKED_PROBE = "all_blocked_probe"
REASON_ORDERED = "ordered"

_SHARED = "shared"
_FIVE_HOUR = "five_hour"
_SEVEN_DAY = "seven_day"
_FIVE_HOUR_SECONDS = 5 * 3600.0
_SEVEN_DAY_SECONDS = 7 * 86400.0
_TRANSIENT_PROBE_SECONDS = 300.0
_RETRY_AFTER_DEFAULT_SECONDS = 300.0
_USAGE_JITTER = 0.25
_ENDPOINT_ABSENT_STATUSES = frozenset({404, 405, 410, 501})
# A reset value this small predates the decade, so it is a delay, not an instant
# (the convention `providers/base.py` applies to the same headers).
_EPOCH_SECONDS_FLOOR = 1_000_000_000.0
_FAMILY_TOKENS = ("opus", "sonnet", "haiku", "fable", "mythos")
_UNWRITABLE = "unwritable"

# Header window suffix -> (window id, scope, duration). `7d_oi` is OMP's model-scoped
# weekly header, which it maps to the Fable family.
_HEADER_WINDOWS = (
  ("5h", _FIVE_HOUR, _SHARED, _FIVE_HOUR_SECONDS),
  ("7d", _SEVEN_DAY, _SHARED, _SEVEN_DAY_SECONDS),
  ("7d_oi", "seven_day_fable", "fable", _SEVEN_DAY_SECONDS),
)


# ---------------------------------------------------------------------------
# Identity and value types
# ---------------------------------------------------------------------------


def selection_sidecar_path(store_path: Path) -> Path:
  return store_path.with_name(store_path.name + SELECTION_SIDECAR_SUFFIX)


@dataclass(frozen=True)
class AccountLocator:
  namespace: str
  identity: str

  @property
  def id(self) -> str:
    digest = hashlib.sha256(f"{self.namespace}\0{self.identity}".encode("utf-8")).hexdigest()
    return "acct_" + digest[:16]


def locator_for_record(store_path: Path, identity: str) -> AccountLocator:
  return AccountLocator(namespace="store:" + str(store_path.resolve()), identity=identity)


def locator_for_env_token(token: str) -> AccountLocator:
  return AccountLocator(
    namespace="env", identity=hashlib.sha256(token.encode("utf-8")).hexdigest()
  )


@dataclass(frozen=True)
class AccountPolicy:
  priority: int = DEFAULT_PRIORITY
  reserve_pct: float = DEFAULT_RESERVE_PCT


@dataclass(frozen=True)
class QuotaWindow:
  id: str
  scope: str
  used_fraction: float | None
  reset_at: float | None
  duration_s: float
  observed_at: float
  source: str
  status: str | None = None


@dataclass(frozen=True)
class Cooldown:
  scope: str
  until: float
  reason: str


@dataclass(frozen=True)
class UsageProbeState:
  next_probe_at: float | None = None
  unavailable_until: float | None = None
  unavailable_reason: str | None = None
  generation: str | None = None
  last_usage_at: float | None = None
  in_progress_until: float | None = None


@dataclass(frozen=True)
class AccountObservation:
  windows: tuple[QuotaWindow, ...] = ()
  probe: UsageProbeState = UsageProbeState()


@dataclass(frozen=True)
class Affinity:
  locator_id: str
  last_success_at: float


@dataclass(frozen=True)
class SelectionState:
  policies: Mapping[str, AccountPolicy]
  observations: Mapping[str, AccountObservation]
  cooldowns: Mapping[str, tuple[Cooldown, ...]]
  affinity: Mapping[str, Affinity]
  problems: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Value normalization and parsing
# ---------------------------------------------------------------------------


def _finite(value: Any) -> float | None:
  """A finite float from a number or numeric string; everything else is unknown."""

  if isinstance(value, bool):
    return None
  if not isinstance(value, (int, float, str)):
    return None
  try:
    number = float(value)
  except (ValueError, OverflowError):
    return None
  return number if math.isfinite(number) else None


def _clamp01(value: float | None) -> float | None:
  return None if value is None else min(1.0, max(0.0, value))


def _percent_fraction(value: Any) -> float | None:
  number = _finite(value)
  return None if number is None else _clamp01(number / 100.0)


def _fraction(value: Any) -> float | None:
  return _clamp01(_finite(value))


def _timestamp(value: Any) -> float | None:
  """UTC epoch seconds from an epoch number or an ISO-8601 string."""

  number = _finite(value)
  if number is not None:
    return number if number > 0 else None
  if not isinstance(value, str):
    return None
  text = value.strip()
  if not text:
    return None
  try:
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
  except ValueError:
    return None
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=timezone.utc)
  return _finite(parsed.timestamp())


def _header_reset(value: Any, *, now: float) -> float | None:
  number = _finite(value)
  if number is not None:
    if number <= 0:
      return None
    return number if number > _EPOCH_SECONDS_FLOOR else now + number
  return _timestamp(value)


def _text(value: Any) -> str | None:
  if not isinstance(value, str):
    return None
  stripped = value.strip()
  return stripped or None


def window_exhausted(status: str | None) -> bool:
  """A reported non-allow status is quota exhaustion, including new provider statuses."""

  normalized = (status or "").strip().lower()
  return bool(normalized) and not normalized.startswith("allow")


def model_family(model: str) -> str | None:
  lowered = model.lower() if isinstance(model, str) else ""
  for token in _FAMILY_TOKENS:
    if token in lowered:
      return token
  return None


def claim_scope(claim: str | None) -> str:
  """The representative claim's family or known window scope, else the shared account."""

  tokens = re.split(r"[^a-z0-9]+", (claim or "").lower())
  for family in _FAMILY_TOKENS:
    if family in tokens:
      return family
  normalized = f"_{'_'.join(tokens)}_"
  # Match scoped windows before their shared prefixes, deriving provider aliases
  # from the same suffix/id mapping used to parse response headers.
  for suffix, window_id, scope, _duration in reversed(_HEADER_WINDOWS):
    aliases = (suffix, window_id, suffix.replace("7d", _SEVEN_DAY, 1))
    if any(f"_{alias}_" in normalized for alias in aliases):
      return scope
  return _SHARED


def parse_rate_limit_headers(headers: Mapping[str, str], *, now: float) -> tuple[QuotaWindow, ...]:
  """Windows described by `anthropic-ratelimit-unified-*` headers; only those present.

  Utilization is a fraction in the headers (OMP multiplies it by 100). A window with
  neither a utilization nor a status-with-reset describes nothing and is omitted.
  """

  lowered = {str(name).lower(): value for name, value in headers.items()}
  windows: list[QuotaWindow] = []
  for suffix, window_id, scope, duration in _HEADER_WINDOWS:
    prefix = f"anthropic-ratelimit-unified-{suffix}-"
    used = _fraction(lowered.get(prefix + "utilization"))
    reset = _header_reset(lowered.get(prefix + "reset"), now=now)
    status = _text(lowered.get(prefix + "status"))
    if used is None and not (status is not None and reset is not None):
      continue
    windows.append(
      QuotaWindow(
        id=window_id,
        scope=scope,
        used_fraction=used,
        reset_at=reset,
        duration_s=duration,
        observed_at=now,
        source="response_header",
        status=status,
      )
    )
  return tuple(windows)


def _slug(name: str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _usage_bucket(bucket: Any) -> tuple[float, float | None] | None:
  """(used fraction, reset) of a usage bucket; OMP builds a limit only with a utilization."""

  if not isinstance(bucket, Mapping):
    return None
  used = _percent_fraction(bucket.get("utilization"))
  if used is None:
    return None
  return used, _timestamp(bucket.get("resets_at"))


def _usage_limit_entries(raw: Any) -> list[tuple[str, tuple[float, float | None], str | None]]:
  if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
    return []
  entries: list[tuple[str, tuple[float, float | None], str | None]] = []
  for entry in raw:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("kind"), str):
      continue
    used = _percent_fraction(entry.get("percent"))
    if used is None:
      continue
    scope = entry.get("scope")
    model = scope.get("model") if isinstance(scope, Mapping) else None
    display = _text(model.get("display_name")) if isinstance(model, Mapping) else None
    entries.append((entry["kind"], (used, _timestamp(entry.get("resets_at"))), display))
  return entries


def parse_usage_report(payload: Mapping[str, Any], *, now: float) -> tuple[QuotaWindow, ...]:
  """Windows of a `GET /api/oauth/usage` body.

  Utilization is a percentage. The legacy per-model weekly buckets may be null; scoped
  weekly caps arrive in `limits[]` as `weekly_scoped` rows named by model display name.
  """

  if not isinstance(payload, Mapping):
    return ()
  entries = _usage_limit_entries(payload.get("limits"))
  first_kind: dict[str, tuple[float, float | None]] = {}
  for kind, bucket, _display in entries:
    first_kind.setdefault(kind, bucket)

  def build(window_id: str, scope: str, duration: float, bucket: tuple[float, float | None]) -> QuotaWindow:
    return QuotaWindow(
      id=window_id,
      scope=scope,
      used_fraction=bucket[0],
      reset_at=bucket[1],
      duration_s=duration,
      observed_at=now,
      source="usage",
    )

  windows: list[QuotaWindow] = []
  five_hour = _usage_bucket(payload.get("five_hour")) or first_kind.get("session")
  if five_hour is not None:
    windows.append(build(_FIVE_HOUR, _SHARED, _FIVE_HOUR_SECONDS, five_hour))
  seven_day = _usage_bucket(payload.get("seven_day")) or first_kind.get("weekly_all")
  if seven_day is not None:
    windows.append(build(_SEVEN_DAY, _SHARED, _SEVEN_DAY_SECONDS, seven_day))
  for family in ("opus", "sonnet"):
    bucket = _usage_bucket(payload.get(f"seven_day_{family}"))
    if bucket is not None:
      windows.append(build(f"seven_day_{family}", family, _SEVEN_DAY_SECONDS, bucket))
  seen = {window.id for window in windows}
  for kind, bucket, display in entries:
    if kind != "weekly_scoped" or display is None:
      continue
    slug = _slug(display)
    window_id = f"seven_day_{slug}"
    if not slug or window_id in seen:
      continue
    seen.add(window_id)
    windows.append(build(window_id, model_family(display) or slug, _SEVEN_DAY_SECONDS, bucket))
  return tuple(windows)


def usage_probe_allowed(base_url: str | None) -> bool:
  """True only for the direct `https://api.anthropic.com` authority (None/empty = default)."""

  if base_url is None or not base_url.strip():
    return True
  try:
    parts = urlsplit(base_url.strip())
    return (
      parts.scheme == "https"
      and parts.hostname == "api.anthropic.com"
      and parts.port in (None, 443)
      and parts.username is None
      and parts.password is None
    )
  except ValueError:
    return False


def affinity_key(user_scope: str, full_session_id: str | None, provider: str = "anthropic") -> str | None:
  session = (full_session_id or "").strip()
  if not session or session == "no-session":
    return None
  return hashlib.sha256("\0".join((user_scope, session, provider)).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sidecar: raw dict <-> typed state
# ---------------------------------------------------------------------------

_WINDOW_FIELDS = frozenset(
  {"id", "scope", "used_fraction", "reset_at", "duration_s", "observed_at", "source", "status"}
)
_PROBE_FIELDS = frozenset(
  {"next_probe_at", "unavailable_until", "unavailable_reason", "generation", "last_usage_at", "in_progress_until"}
)
_TOP_FIELDS = frozenset({"version", "policies", "observations", "cooldowns", "affinity"})


def _empty_raw() -> dict[str, Any]:
  return {
    "version": SELECTION_STATE_VERSION,
    "policies": {},
    "observations": {},
    "cooldowns": {},
    "affinity": {},
  }


def _empty_probe_raw() -> dict[str, Any]:
  return asdict(UsageProbeState())


def _unknown_fields(row: Mapping[str, Any], allowed: frozenset[str], problems: list[str]) -> None:
  for name in sorted(set(row) - allowed):
    problems.append(f"unknown_field:{name}")


def _clean_window(row: Any, problems: list[str]) -> dict[str, Any] | None:
  if not isinstance(row, Mapping):
    problems.append("invalid_row:windows")
    return None
  _unknown_fields(row, _WINDOW_FIELDS, problems)
  window_id, scope, source = _text(row.get("id")), _text(row.get("scope")), _text(row.get("source"))
  duration, observed = _finite(row.get("duration_s")), _finite(row.get("observed_at"))
  if window_id is None or scope is None or source is None or observed is None:
    problems.append("invalid_row:windows")
    return None
  if duration is None or duration <= 0:
    problems.append("invalid_row:windows")
    return None
  return {
    "id": window_id,
    "scope": scope,
    "used_fraction": _fraction(row.get("used_fraction")),
    "reset_at": _timestamp(row.get("reset_at")),
    "duration_s": duration,
    "observed_at": observed,
    "source": source,
    "status": _text(row.get("status")),
  }


def _clean_probe(row: Any, problems: list[str]) -> dict[str, Any]:
  if not isinstance(row, Mapping):
    return _empty_probe_raw()
  _unknown_fields(row, _PROBE_FIELDS, problems)
  return {
    "next_probe_at": _finite(row.get("next_probe_at")),
    "unavailable_until": _finite(row.get("unavailable_until")),
    "unavailable_reason": _text(row.get("unavailable_reason")),
    "generation": _text(row.get("generation")),
    "last_usage_at": _finite(row.get("last_usage_at")),
    "in_progress_until": _finite(row.get("in_progress_until")),
  }


def _clean_policy(row: Any, problems: list[str]) -> dict[str, Any] | None:
  if not isinstance(row, Mapping):
    problems.append("invalid_row:policies")
    return None
  _unknown_fields(row, frozenset({"priority", "reserve_pct"}), problems)
  priority = row.get("priority", DEFAULT_PRIORITY)
  reserve = row.get("reserve_pct", DEFAULT_RESERVE_PCT)
  reserve_value = _finite(reserve) if not isinstance(reserve, str) else None
  if (
    isinstance(priority, bool)
    or not isinstance(priority, int)
    or reserve_value is None
    or not 0.0 <= reserve_value <= 100.0
  ):
    problems.append("invalid_row:policies")
    return None
  return {"priority": priority, "reserve_pct": reserve_value}


def _clean_cooldown(row: Any, problems: list[str]) -> dict[str, Any] | None:
  if not isinstance(row, Mapping):
    problems.append("invalid_row:cooldowns")
    return None
  _unknown_fields(row, frozenset({"scope", "until", "reason"}), problems)
  scope, reason, until = _text(row.get("scope")), _text(row.get("reason")), _finite(row.get("until"))
  if scope is None or reason is None or until is None:
    problems.append("invalid_row:cooldowns")
    return None
  return {"scope": scope, "until": until, "reason": reason}


def _clean_affinity(row: Any, problems: list[str]) -> dict[str, Any] | None:
  if not isinstance(row, Mapping):
    problems.append("invalid_row:affinity")
    return None
  _unknown_fields(row, frozenset({"locator_id", "last_success_at"}), problems)
  locator_id, last = _text(row.get("locator_id")), _finite(row.get("last_success_at"))
  if locator_id is None or last is None:
    problems.append("invalid_row:affinity")
    return None
  return {"locator_id": locator_id, "last_success_at": last}


def _section(data: Mapping[str, Any], name: str, problems: list[str]) -> Mapping[str, Any]:
  value = data.get(name, {})
  if not isinstance(value, Mapping):
    problems.append(f"invalid_section:{name}")
    return {}
  return value


def _clean_raw(data: Any) -> tuple[dict[str, Any], list[str]]:
  """Canonicalize a decoded sidecar: known fields only, valid rows only, problems reported."""

  problems: list[str] = []
  raw = _empty_raw()
  if not isinstance(data, Mapping):
    return raw, ["corrupt_sidecar"]
  version = data.get("version")
  if isinstance(version, bool) or version != SELECTION_STATE_VERSION:
    return raw, ["unsupported_version"]
  _unknown_fields(data, _TOP_FIELDS, problems)

  for key, row in _section(data, "policies", problems).items():
    policy = _clean_policy(row, problems)
    if policy is not None:
      raw["policies"][str(key)] = policy

  for key, row in _section(data, "observations", problems).items():
    if not isinstance(row, Mapping):
      problems.append("invalid_row:observations")
      continue
    _unknown_fields(row, frozenset({"windows", "probe"}), problems)
    rows = row.get("windows", [])
    windows: list[dict[str, Any]] = []
    for window_row in rows if isinstance(rows, list) else []:
      window = _clean_window(window_row, problems)
      if window is not None:
        windows.append(window)
    raw["observations"][str(key)] = {"windows": windows, "probe": _clean_probe(row.get("probe"), problems)}

  for key, rows in _section(data, "cooldowns", problems).items():
    if not isinstance(rows, list):
      problems.append("invalid_row:cooldowns")
      continue
    cooldowns = [c for c in (_clean_cooldown(r, problems) for r in rows) if c is not None]
    if cooldowns:
      raw["cooldowns"][str(key)] = cooldowns

  for key, row in _section(data, "affinity", problems).items():
    affinity = _clean_affinity(row, problems)
    if affinity is not None:
      raw["affinity"][str(key)] = affinity
  return raw, list(dict.fromkeys(problems))


def _prune(raw: dict[str, Any], now: float) -> None:
  """Drop last-good windows older than 24h, idle affinity older than 30d, expired cooldowns.

  Policies are never pruned.
  """

  cutoff = now - LAST_GOOD_MAX_AGE_SECONDS
  for locator_id, observation in list(raw["observations"].items()):
    observation["windows"] = [w for w in observation["windows"] if w["observed_at"] >= cutoff]
    probe = observation["probe"]
    probe_live = any(
      probe[name] is not None and probe[name] >= cutoff
      for name in ("next_probe_at", "unavailable_until", "last_usage_at", "in_progress_until")
    )
    if not observation["windows"] and not probe_live:
      del raw["observations"][locator_id]
  for locator_id, rows in list(raw["cooldowns"].items()):
    live = [row for row in rows if row["until"] > now]
    if live:
      raw["cooldowns"][locator_id] = live
    else:
      del raw["cooldowns"][locator_id]
  affinity_cutoff = now - AFFINITY_PRUNE_SECONDS
  raw["affinity"] = {
    key: row for key, row in raw["affinity"].items() if row["last_success_at"] >= affinity_cutoff
  }


def _state_from_raw(raw: Mapping[str, Any], problems: Sequence[str]) -> SelectionState:
  return SelectionState(
    policies={key: AccountPolicy(**row) for key, row in raw["policies"].items()},
    observations={
      key: AccountObservation(
        windows=tuple(QuotaWindow(**window) for window in row["windows"]),
        probe=UsageProbeState(**row["probe"]),
      )
      for key, row in raw["observations"].items()
    },
    cooldowns={
      key: tuple(Cooldown(**row) for row in rows) for key, rows in raw["cooldowns"].items()
    },
    affinity={key: Affinity(**row) for key, row in raw["affinity"].items()},
    problems=tuple(problems),
  )


def _read_sidecar(path: Path) -> tuple[dict[str, Any], list[str]]:
  try:
    data = json.loads(path.read_text(encoding="utf-8"))
  except FileNotFoundError:
    return _empty_raw(), []
  except OSError as exc:
    LOGGER.warning("Anthropic selection sidecar %s is unreadable (%s); using the ordered pool", path, type(exc).__name__)
    return _empty_raw(), ["unreadable_sidecar"]
  except ValueError:
    LOGGER.warning("Anthropic selection sidecar %s is corrupt; using the ordered pool", path)
    return _empty_raw(), ["corrupt_sidecar"]
  raw, problems = _clean_raw(data)
  if problems:
    LOGGER.warning("Anthropic selection sidecar %s reported %s", path, ", ".join(problems))
  return raw, problems


def load_selection_state(store_path: Path, *, now: float | None = None) -> SelectionState:
  """The sidecar's state; never raises. Missing, corrupt or unreadable yields an empty state."""

  moment = time.time() if now is None else now
  raw, problems = _read_sidecar(selection_sidecar_path(store_path))
  _prune(raw, moment)
  return _state_from_raw(raw, problems)


def update_selection_state(
  store_path: Path,
  mutate: Callable[[dict], None],
  *,
  now: float | None = None,
) -> SelectionState:
  """Locked read, mutate, prune and atomic replace; a write failure never raises."""

  moment = time.time() if now is None else now
  path = selection_sidecar_path(store_path)
  before: SelectionState | None = None
  problems: list[str] = []
  try:
    with store_write_lock(path):
      raw, problems = _read_sidecar(path)
      _prune(raw, moment)
      before = _state_from_raw(raw, problems)
      if "unreadable_sidecar" in problems:
        LOGGER.warning("Anthropic selection sidecar %s cannot be read; leaving it unchanged", path)
        return before
      if "corrupt_sidecar" in problems or "unsupported_version" in problems:
        # Keep the unreadable bytes for the operator instead of silently overwriting them.
        try:
          os.replace(path, path.with_name(path.name + ".corrupt"))
        except OSError:
          pass
      mutate(raw)
      raw, _ = _clean_raw(raw)
      _prune(raw, moment)
      atomic_write_json(path, raw)
  except OSError as exc:
    LOGGER.warning(
      "Anthropic selection sidecar %s could not be written (%s); selection continues without it",
      path,
      type(exc).__name__,
    )
    state = before if before is not None else load_selection_state(store_path, now=moment)
    return SelectionState(
      policies=state.policies,
      observations=state.observations,
      cooldowns=state.cooldowns,
      affinity=state.affinity,
      problems=(*state.problems, _UNWRITABLE),
    )
  return _state_from_raw(raw, problems)


def _require_written(state: SelectionState, store_path: Path) -> None:
  if _UNWRITABLE in state.problems or "unreadable_sidecar" in state.problems:
    raise OSError(f"cannot write {selection_sidecar_path(store_path)}")


def _is_number(value: Any) -> bool:
  return isinstance(value, (int, float)) and not isinstance(value, bool)


def set_account_policy(
  store_path: Path,
  locator: AccountLocator,
  *,
  priority: int | None = None,
  reserve_pct: float | None = None,
) -> AccountPolicy:
  """Operator boundary: ValueError on malformed input, OSError when the sidecar cannot be written."""

  if priority is not None and (isinstance(priority, bool) or not isinstance(priority, int)):
    raise ValueError("priority must be an integer")
  if reserve_pct is not None:
    value = _finite(reserve_pct) if _is_number(reserve_pct) else None
    if value is None or not 0.0 <= value <= 100.0:
      raise ValueError("reserve_pct must be a finite number between 0 and 100")
    reserve_pct = value

  def mutate(raw: dict) -> None:
    current = raw["policies"].get(locator.id, {})
    raw["policies"][locator.id] = {
      "priority": current.get("priority", DEFAULT_PRIORITY) if priority is None else priority,
      "reserve_pct": current.get("reserve_pct", DEFAULT_RESERVE_PCT) if reserve_pct is None else reserve_pct,
    }

  state = update_selection_state(store_path, mutate)
  _require_written(state, store_path)
  return state.policies[locator.id]


def reset_account_policy(store_path: Path, locator: AccountLocator) -> None:
  def mutate(raw: dict) -> None:
    raw["policies"].pop(locator.id, None)

  state = update_selection_state(store_path, mutate)
  _require_written(state, store_path)


def record_observation(
  store_path: Path,
  locator: AccountLocator,
  windows: Sequence[QuotaWindow],
  *,
  now: float | None = None,
) -> None:
  """Per-window merge: a window replaces its (id, scope) twin only when strictly newer."""

  incoming = [row for row in (_clean_window(asdict(w), []) for w in windows) if row is not None]
  if not incoming:
    return

  def mutate(raw: dict) -> None:
    observation = raw["observations"].setdefault(
      locator.id, {"windows": [], "probe": _empty_probe_raw()}
    )
    stored = {(row["id"], row["scope"]): row for row in observation["windows"]}
    for row in incoming:
      existing = stored.get((row["id"], row["scope"]))
      if existing is None or row["observed_at"] > existing["observed_at"]:
        stored[(row["id"], row["scope"])] = row
    observation["windows"] = list(stored.values())

  update_selection_state(store_path, mutate, now=now)


def record_cooldown(store_path: Path, locator: AccountLocator, cooldown: Cooldown) -> None:
  """Keeps the later `until` per scope."""

  row = _clean_cooldown(asdict(cooldown), [])
  if row is None:
    return

  def mutate(raw: dict) -> None:
    rows = raw["cooldowns"].setdefault(locator.id, [])
    for index, existing in enumerate(rows):
      if existing["scope"] == row["scope"]:
        if row["until"] > existing["until"]:
          rows[index] = row
        return
    rows.append(row)

  update_selection_state(store_path, mutate)


def record_probe_state(store_path: Path, locator: AccountLocator, probe: UsageProbeState) -> None:
  """Replaces the probe state; `last_usage_at` never moves backwards or to unknown."""

  row = _clean_probe(asdict(probe), [])

  def mutate(raw: dict) -> None:
    observation = raw["observations"].setdefault(
      locator.id, {"windows": [], "probe": _empty_probe_raw()}
    )
    previous = observation["probe"]["last_usage_at"]
    if previous is not None and (row["last_usage_at"] is None or previous > row["last_usage_at"]):
      row["last_usage_at"] = previous
    observation["probe"] = row

  update_selection_state(store_path, mutate)


def claim_usage_probe(
  store_path: Path,
  locator: AccountLocator,
  *,
  generation: str | None = None,
  now: float | None = None,
) -> bool:
  """Claim a due probe under the metadata lock, without waiting for another probe.

  The marker expires after the bounded batch plus a small cancellation margin.
  Auxiliary-state failures still permit telemetry and cannot prevent inference.
  """

  moment = time.time() if now is None else now
  claimed = False

  def mutate(raw: dict) -> None:
    nonlocal claimed
    observation = raw["observations"].setdefault(
      locator.id, {"windows": [], "probe": _empty_probe_raw()}
    )
    probe = observation["probe"]
    if probe["in_progress_until"] is not None and probe["in_progress_until"] > moment:
      return
    if probe["next_probe_at"] is not None and probe["next_probe_at"] > moment:
      return
    if probe["unavailable_until"] is not None and probe["unavailable_until"] > moment:
      credential_bound = probe["unavailable_reason"] in {"http_401", "http_403"}
      if not credential_bound or probe["generation"] == generation:
        return
    probe["in_progress_until"] = moment + PROBE_BATCH_BUDGET_SECONDS + 0.5
    claimed = True

  state = update_selection_state(store_path, mutate, now=moment)
  return claimed or _UNWRITABLE in state.problems or "unreadable_sidecar" in state.problems


def record_success(
  store_path: Path,
  locator: AccountLocator,
  key: str | None,
  *,
  now: float | None = None,
) -> None:
  if key is None:
    return
  moment = time.time() if now is None else now

  def mutate(raw: dict) -> None:
    raw["affinity"][key] = {"locator_id": locator.id, "last_success_at": moment}

  update_selection_state(store_path, mutate, now=moment)


# ---------------------------------------------------------------------------
# Usage probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeTarget:
  locator: AccountLocator
  bearer: str
  generation: str | None
  base_url: str | None


@dataclass(frozen=True)
class ProbeResult:
  windows: tuple[QuotaWindow, ...]
  probe: UsageProbeState


def _unavailable(moment: float, seconds: float, reason: str, generation: str | None) -> ProbeResult:
  until = moment + seconds
  return ProbeResult(
    windows=(),
    probe=UsageProbeState(
      next_probe_at=until,
      unavailable_until=until,
      unavailable_reason=reason,
      generation=generation,
    ),
  )


def _retry_after_seconds(value: str | None, *, moment: float) -> float:
  """Retry-After capped at an hour; absent, invalid or non-positive means five minutes."""

  text = (value or "").strip()
  seconds = _finite(text)
  if seconds is None and text:
    try:
      seconds = parsedate_to_datetime(text).timestamp() - moment
    except (TypeError, ValueError):
      seconds = None
  if seconds is None or seconds <= 0:
    return _RETRY_AFTER_DEFAULT_SECONDS
  return min(seconds, USAGE_RETRY_AFTER_CAP_SECONDS)


def _looks_like_usage(payload: Any) -> TypeGuard[Mapping[str, Any]]:
  return isinstance(payload, Mapping) and any(
    key in payload for key in ("five_hour", "seven_day", "limits", "extra_usage", "spend")
  )


async def _fetch_usage(client: httpx.AsyncClient, target: ProbeTarget, moment: float) -> ProbeResult:
  generation = target.generation
  try:
    response = await client.get(
      ANTHROPIC_USAGE_URL,
      headers={
        "Authorization": f"Bearer {target.bearer}",
        "anthropic-beta": ANTHROPIC_OAUTH_BETA,
        "Accept": "application/json",
      },
    )
  except Exception as exc:
    LOGGER.warning("Anthropic usage probe for %s failed (%s)", target.locator.id, type(exc).__name__)
    return _unavailable(moment, _TRANSIENT_PROBE_SECONDS, "network", generation)

  status = response.status_code
  if status in (401, 403):
    return _unavailable(moment, USAGE_UNAVAILABLE_SECONDS, f"http_{status}", generation)
  if status == 429:
    wait = _retry_after_seconds(response.headers.get("retry-after"), moment=moment)
    return _unavailable(moment, wait, "http_429", generation)
  if status in _ENDPOINT_ABSENT_STATUSES:
    return _unavailable(moment, USAGE_UNAVAILABLE_SECONDS, "route", generation)
  if status != 200:
    LOGGER.warning("Anthropic usage probe for %s returned HTTP %s", target.locator.id, status)
    return _unavailable(moment, _TRANSIENT_PROBE_SECONDS, "network", generation)
  try:
    payload = response.json()
  except ValueError:
    payload = None
  if not _looks_like_usage(payload):
    return _unavailable(moment, _TRANSIENT_PROBE_SECONDS, "parse", generation)
  jitter = random.uniform(1.0 - _USAGE_JITTER, 1.0 + _USAGE_JITTER)
  return ProbeResult(
    windows=parse_usage_report(payload, now=moment),
    probe=UsageProbeState(
      next_probe_at=moment + USAGE_FRESH_TTL_SECONDS * jitter,
      last_usage_at=moment,
    ),
  )


async def probe_usage_batch(
  targets: Sequence[ProbeTarget],
  *,
  transport: httpx.AsyncBaseTransport | None = None,
  now: float | None = None,
  budget_s: float = PROBE_BATCH_BUDGET_SECONDS,
  max_concurrency: int = PROBE_MAX_CONCURRENCY,
) -> dict[str, ProbeResult]:
  """Probe usage for `targets` within one total budget (queueing included).

  Probes unfinished at the budget are cancelled and have no entry. A route that is not
  the direct Anthropic authority gets an unavailable verdict without any network call.
  """

  moment = time.time() if now is None else now
  results: dict[str, ProbeResult] = {}
  pending_targets: dict[str, ProbeTarget] = {}
  for target in targets:
    locator_id = target.locator.id
    if locator_id in results or locator_id in pending_targets:
      continue
    if usage_probe_allowed(target.base_url):
      pending_targets[locator_id] = target
    else:
      results[locator_id] = _unavailable(moment, USAGE_UNAVAILABLE_SECONDS, "route", target.generation)
  if not pending_targets or budget_s <= 0:
    return results

  semaphore = asyncio.Semaphore(max(1, max_concurrency))

  async def run(client: httpx.AsyncClient, target: ProbeTarget) -> ProbeResult:
    async with semaphore:
      return await _fetch_usage(client, target, moment)

  async with httpx.AsyncClient(
    transport=transport, timeout=httpx.Timeout(budget_s), follow_redirects=False
  ) as client:
    tasks = {
      asyncio.ensure_future(run(client, target)): locator_id
      for locator_id, target in pending_targets.items()
    }
    try:
      done, _ = await asyncio.wait(set(tasks), timeout=budget_s)
    finally:
      unfinished = [task for task in tasks if not task.done()]
      for task in unfinished:
        task.cancel()
      if unfinished:
        await asyncio.gather(*unfinished, return_exceptions=True)
    for task in done:
      results[tasks[task]] = task.result()
  return results


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankCandidate:
  locator: AccountLocator
  tier: str
  order: int
  policy: AccountPolicy
  observation: AccountObservation
  cooldowns: tuple[Cooldown, ...]


@dataclass(frozen=True)
class RankDecision:
  ordered: tuple[RankCandidate, ...]
  blocked: tuple[tuple[RankCandidate, float], ...]
  tier: str | None
  reason: str
  fallback: RankCandidate | None


@dataclass(frozen=True)
class _Evaluation:
  candidate: RankCandidate
  blocked_until: float | None
  measured: bool
  in_reserve: bool
  hot: bool
  key: tuple[Any, ...]


def _q(value: float) -> float:
  """Fixed-precision numeric key so equality at a boundary is decided deterministically."""

  return round(value, 9)


def _is_fresh(window: QuotaWindow, now: float, probe: UsageProbeState) -> bool:
  if (
    window.source == "usage"
    and probe.last_usage_at == window.observed_at
    and probe.next_probe_at is not None
    and probe.unavailable_reason is None
  ):
    return now <= probe.next_probe_at
  return now - window.observed_at <= USAGE_FRESH_TTL_SECONDS


def _is_expired(window: QuotaWindow, now: float) -> bool:
  return window.reset_at is not None and window.reset_at <= now


def _is_exhausted(window: QuotaWindow) -> bool:
  if window.used_fraction is not None and _q(window.used_fraction) >= 1.0:
    return True
  return window_exhausted(window.status)


def _applies(scope: str, family: str | None) -> bool:
  return scope == _SHARED or (family is not None and scope == family)


def _hours_left(window: QuotaWindow, now: float) -> float:
  remaining = window.duration_s if window.reset_at is None else min(window.reset_at - now, window.duration_s)
  return max(1.0 / 60.0, remaining / 3600.0)


@dataclass(frozen=True)
class _Measured:
  """A fresh, unexpired window with a known used fraction."""

  window: QuotaWindow
  used: float


def _required_drain(measured: _Measured, now: float) -> float:
  remaining = 1.0 - measured.used
  return 0.0 if remaining <= 0 else remaining / _hours_left(measured.window, now)


def _pressure(measured: _Measured, now: float) -> tuple[float, float, str, str]:
  """OMP's bucket pressure: used fraction per elapsed hour, with used as the tie-break."""

  window = measured.window
  pressure = measured.used
  if window.reset_at is not None:
    remaining = min(max(window.reset_at - now, 0.0), window.duration_s)
    elapsed_hours = (window.duration_s - remaining) / 3600.0
    if elapsed_hours > 0:
      pressure = measured.used / elapsed_hours
  return (_q(pressure), _q(measured.used), window.id, window.scope)


def _most_pressured(windows: Sequence[_Measured], now: float) -> _Measured | None:
  return max(windows, key=lambda measured: _pressure(measured, now), default=None)


def _group_key(measured: _Measured | None, now: float) -> tuple[int, float, float]:
  if measured is None:
    return (1, 0.0, 0.0)
  return (0, -_q(_required_drain(measured, now)), _q(measured.used))


def _tie_digest(session_seed: str | None, locator: AccountLocator) -> str:
  if session_seed is None:
    return ""
  return hashlib.sha256(f"{session_seed}\0{locator.id}".encode("utf-8")).hexdigest()


def _evaluate(
  candidate: RankCandidate,
  *,
  now: float,
  family: str | None,
  session_seed: str | None,
) -> _Evaluation:
  blocks = [c.until for c in candidate.cooldowns if c.until > now and _applies(c.scope, family)]
  measured_windows: list[_Measured] = []
  for window in candidate.observation.windows:
    if (
      not _applies(window.scope, family)
      or _is_expired(window, now)
      or not _is_fresh(window, now, candidate.observation.probe)
    ):
      continue
    if window.reset_at is not None and _is_exhausted(window):
      blocks.append(window.reset_at)
    if window.used_fraction is not None:
      measured_windows.append(_Measured(window, window.used_fraction))

  reserve = candidate.policy.reserve_pct
  in_reserve = (
    reserve > 0
    and bool(measured_windows)
    and _q(1.0 - max(m.used for m in measured_windows)) <= _q(reserve / 100.0)
  )
  weekly = _most_pressured([m for m in measured_windows if m.window.id.startswith(_SEVEN_DAY)], now)
  short = _most_pressured([m for m in measured_windows if m.window.id == _FIVE_HOUR], now)
  hot = short is not None and _q(short.used) >= SHORT_WINDOW_HEAT
  measured = bool(measured_windows)
  key = (
    1 if in_reserve else 0,
    1 if hot else 0,
    0 if measured else 1,
    -candidate.policy.priority,
    *_group_key(weekly, now),
    *_group_key(short, now),
    _tie_digest(session_seed, candidate.locator),
    candidate.order,
    candidate.locator.id,
  )
  return _Evaluation(
    candidate=candidate,
    blocked_until=max(blocks) if blocks else None,
    measured=measured,
    in_reserve=in_reserve,
    hot=hot,
    key=key,
  )


def rank_candidates(
  candidates: Sequence[RankCandidate],
  *,
  now: float,
  family: str | None,
  affinity: Affinity | None,
  session_seed: str | None,
) -> RankDecision:
  """Pure spec section 6 ranking: tier, blocks, warm affinity, total-order comparator."""

  evaluations = [
    _evaluate(candidate, now=now, family=family, session_seed=session_seed)
    for candidate in candidates
  ]
  blocked_pairs = sorted(
    ((e.blocked_until, e.candidate) for e in evaluations if e.blocked_until is not None),
    key=lambda pair: (pair[0], pair[1].order, pair[1].locator.id),
  )
  blocked = tuple((candidate, until) for until, candidate in blocked_pairs)
  usable = [e for e in evaluations if e.blocked_until is None]
  if not usable:
    fallback = blocked[0][0] if blocked else None
    return RankDecision(
      ordered=(), blocked=blocked, tier=None, reason=REASON_ALL_BLOCKED_PROBE, fallback=fallback
    )

  tier = min({e.candidate.tier for e in usable}, key=lambda name: (name != "private", name))
  in_tier = sorted((e for e in usable if e.candidate.tier == tier), key=lambda e: e.key)

  ordered = in_tier
  warm = False
  if affinity is not None and now - affinity.last_success_at < WARM_IDLE_SECONDS:
    kept = next((e for e in in_tier if e.candidate.locator.id == affinity.locator_id), None)
    if kept is not None:
      evicted = kept.in_reserve and any(
        e is not kept and e.measured and not e.in_reserve for e in in_tier
      )
      if not evicted:
        warm = True
        ordered = [kept, *(e for e in in_tier if e is not kept)]

  top = ordered[0]
  has_private = any(c.tier == "private" for c in candidates)
  if warm:
    reason = REASON_WARM
  elif tier != "private" and has_private:
    reason = REASON_ORG_FALLBACK
  elif top.in_reserve:
    reason = REASON_RESERVE
  elif top.hot:
    reason = REASON_SHORT_WINDOW_HOT
  elif len(ordered) == 1:
    reason = REASON_ORDERED
  elif top.measured:
    reason = REASON_QUOTA_DRAIN
  else:
    reason = REASON_QUOTA_UNAVAILABLE
  return RankDecision(
    ordered=tuple(e.candidate for e in ordered),
    blocked=blocked,
    tier=tier,
    reason=reason,
    fallback=None,
  )


# ---------------------------------------------------------------------------
# CLI view
# ---------------------------------------------------------------------------


def describe_account_usage(state: SelectionState, locator: AccountLocator, *, now: float) -> dict[str, Any]:
  """Cached-data view of one account; never refreshes or selects."""

  policy = state.policies.get(locator.id)
  effective = policy or AccountPolicy()
  observation = state.observations.get(locator.id, AccountObservation())
  windows: list[dict[str, Any]] = []
  any_fresh = False
  for window in observation.windows:
    expired = _is_expired(window, now)
    fresh = _is_fresh(window, now, observation.probe) and not expired
    any_fresh = any_fresh or fresh
    windows.append(
      {
        "id": window.id,
        "scope": window.scope,
        "used_pct": (
          None
          if window.used_fraction is None or expired
          else round(window.used_fraction * 100.0, 2)
        ),
        "resets_in_s": None if window.reset_at is None else max(0.0, window.reset_at - now),
        "age_s": max(0.0, now - window.observed_at),
        "source": window.source,
        "fresh": fresh,
      }
    )
  probe = observation.probe
  unavailable = probe.unavailable_until is not None and probe.unavailable_until > now
  if unavailable:
    usage = "unavailable"
  elif any_fresh:
    usage = "fresh"
  elif windows:
    usage = "stale"
  else:
    usage = "unknown"
  return {
    "account": locator.id,
    "priority": effective.priority,
    "reserve_pct": effective.reserve_pct,
    "policy_source": "configured" if policy is not None else "default",
    "windows": windows,
    "usage": usage,
    "unavailable_reason": probe.unavailable_reason if unavailable else None,
    "cooldowns": [
      {"scope": c.scope, "reason": c.reason, "remaining_s": c.until - now}
      for c in state.cooldowns.get(locator.id, ())
      if c.until > now
    ],
  }
