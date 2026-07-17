from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from typing import Any, Mapping


_TRADE_EXECUTE_TO_PREVIEW_TOOLS = {
  "execute_basket_trade": frozenset({"preview_basket_trade"}),
  "execute_trade": frozenset({"preview_trade"}),
  "execute_option_trade": frozenset({"preview_option_trade"}),
  "execute_futures_roll": frozenset({"preview_futures_roll"}),
}

_SUMMARY_ALIAS_FIELDS = frozenset({
  "approval_summary",
  "approvalSummary",
  "preview_summary",
  "previewSummary",
  "trade_preview",
  "tradePreview",
})

# This is operator-visible approval evidence, not a generic audit redaction.
# Keep it to the economics required to decide the trade and avoid account IDs
# or raw broker preview payloads.
_SUMMARY_DATA_FIELDS = (
  "ticker",
  "side",
  "quantity",
  "order_type",
  "time_in_force",
  "limit_price",
  "stop_price",
  "estimated_price",
  "estimated_total",
  "estimated_commission",
  "pre_trade_weight",
  "post_trade_weight",
)

_SUMMARY_METADATA_FIELDS = (
  "expires_at",
  "broker_provider",
)

DEFAULT_APPROVAL_EXPIRY_SECONDS = 600.0
MIN_APPROVAL_EXPIRY_SECONDS = 0.1
TRADE_PREVIEW_APPROVAL_DISPATCH_MARGIN_SECONDS = 30.0


def enrich_trade_approval_args(
  tool_name: str,
  approval_args: Mapping[str, Any] | None,
  *,
  event_log: Any | None = None,
) -> dict[str, Any]:
  """Attach trusted decision evidence to irreversible approval payloads."""

  enriched = dict(approval_args or {})
  normalized_tool = _policy_tool_name(tool_name)
  if normalized_tool == "apply_patch_proposal":
    undo_token_id = _text(
      enriched.get("source_model_writer_undo_token_id")
    )
    undo_effect = _text(enriched.get("source_model_writer_undo_effect"))
    undo_expires_at = enriched.get("source_model_writer_undo_expires_at")
    if (
      undo_token_id
      and undo_effect == "retired_after_apply"
      and isinstance(undo_expires_at, (int, float))
      and not isinstance(undo_expires_at, bool)
      and math.isfinite(undo_expires_at)
    ):
      consequence = (
        "Applying this proposal permanently retires the bound model-writer Undo "
        f"receipt {undo_token_id}. Deny this apply to preserve the restore option."
      )
      enriched["consequence"] = consequence
      enriched["approval_summary"] = {
        "proposal_id": _text(enriched.get("proposal_id")),
        "model_writer_undo": {
          "status": "will_be_retired_by_apply",
          "undo_token_id": undo_token_id,
          "undo_expires_at": float(undo_expires_at),
        },
        "operator_choice": (
          "Approve to promote the Thesis proposal, or deny and use "
          "fms_undo_model_writer_commit before the receipt expires."
        ),
      }
  preview_tools = _TRADE_EXECUTE_TO_PREVIEW_TOOLS.get(normalized_tool)
  if not preview_tools:
    return enriched

  for field in _SUMMARY_ALIAS_FIELDS:
    enriched.pop(field, None)

  preview_id = _text(enriched.get("preview_id"))
  preview_ids = _preview_ids_from_args(enriched)
  if not preview_id and not preview_ids:
    return enriched

  summary = _find_trade_preview_summary(
    event_log=event_log,
    preview_id=preview_id,
    preview_ids=preview_ids,
    preview_tools=preview_tools,
  )
  if summary is not None:
    enriched["approval_summary"] = summary
  return enriched


def effective_trade_approval_expiry_seconds(
  tool_name: str,
  approval_args: Mapping[str, Any] | None,
  *,
  requested_expiry_seconds: float | int | None,
  max_wait_seconds: float | int | None,
  now: datetime | None = None,
) -> float:
  """Return the approval window after applying trade-preview freshness caps."""

  requested = _positive_float(requested_expiry_seconds, DEFAULT_APPROVAL_EXPIRY_SECONDS)
  if _policy_tool_name(tool_name) not in _TRADE_EXECUTE_TO_PREVIEW_TOOLS:
    return max(MIN_APPROVAL_EXPIRY_SECONDS, requested)

  max_wait = _positive_float(max_wait_seconds, requested)
  effective = min(requested, max_wait)

  args = approval_args or {}
  summary = args.get("approval_summary") if isinstance(args, Mapping) else None
  if not isinstance(summary, Mapping):
    return max(MIN_APPROVAL_EXPIRY_SECONDS, effective)

  remaining = _seconds_until_expiry(summary.get("expires_at"), now=now)
  if remaining is None:
    return max(MIN_APPROVAL_EXPIRY_SECONDS, effective)

  preview_safe_window = remaining - TRADE_PREVIEW_APPROVAL_DISPATCH_MARGIN_SECONDS
  return max(MIN_APPROVAL_EXPIRY_SECONDS, min(effective, preview_safe_window))


def _find_trade_preview_summary(
  *,
  event_log: Any | None,
  preview_id: str,
  preview_ids: list[str],
  preview_tools: frozenset[str],
) -> dict[str, Any] | None:
  entries = getattr(event_log, "entries", None)
  if not entries:
    return None

  for entry in reversed(list(entries)):
    event = getattr(entry, "event", entry)
    if not isinstance(event, Mapping):
      continue
    if event.get("type") != "tool_call_complete" or event.get("error") is not None:
      continue
    if _policy_tool_name(_text(event.get("tool_name"))) not in preview_tools:
      continue
    result = event.get("result")
    if not isinstance(result, Mapping):
      continue
    if preview_ids:
      summary = _basket_preview_summary_from_result(result, expected_preview_ids=preview_ids)
    else:
      summary = _preview_summary_from_result(result, expected_preview_id=preview_id)
    if summary is not None:
      return summary
  return None


def _preview_summary_from_result(result: Mapping[str, Any], *, expected_preview_id: str) -> dict[str, Any] | None:
  data = result.get("data")
  if not isinstance(data, Mapping):
    data = result

  preview_id = _text(data.get("preview_id") if isinstance(data, Mapping) else None)
  if not preview_id or preview_id != expected_preview_id:
    return None

  metadata = result.get("metadata")
  if not isinstance(metadata, Mapping):
    metadata = {}

  if _is_expired(metadata.get("expires_at")):
    return None

  summary: dict[str, Any] = {"preview_id": preview_id}
  for field in _SUMMARY_DATA_FIELDS:
    _copy_if_present(summary, data, field)
  for field in _SUMMARY_METADATA_FIELDS:
    _copy_if_present(summary, metadata, field)

  validation = data.get("validation") if isinstance(data, Mapping) else None
  if isinstance(validation, Mapping):
    compact_validation = {}
    for field in ("is_valid", "warnings", "errors"):
      _copy_if_present(compact_validation, validation, field)
    if compact_validation:
      summary["validation"] = compact_validation

  return summary


def _basket_preview_summary_from_result(result: Mapping[str, Any], *, expected_preview_ids: list[str]) -> dict[str, Any] | None:
  source = result.get("snapshot")
  if not isinstance(source, Mapping):
    source = result

  if _is_expired(source.get("expires_at")):
    return None

  result_preview_ids = _preview_ids_from_args(source)
  if not result_preview_ids:
    legs = source.get("preview_legs")
    if isinstance(legs, list):
      result_preview_ids = [_text(leg.get("preview_id")) for leg in legs if isinstance(leg, Mapping) and _text(leg.get("preview_id"))]

  if set(result_preview_ids) != set(expected_preview_ids):
    return None

  raw_legs = source.get("preview_legs")
  if not isinstance(raw_legs, list):
    raw_legs = source.get("legs")
  requested = set(expected_preview_ids)
  matching_legs = [
    leg
    for leg in raw_legs
    if isinstance(leg, Mapping) and _text(leg.get("preview_id")) in requested
  ] if isinstance(raw_legs, list) else []
  if matching_legs and not _all_legs_have_fresh_expiry(matching_legs):
    return None

  summary: dict[str, Any] = {"preview_ids": list(expected_preview_ids)}
  expires_at = _earliest_expiry([leg.get("expires_at") for leg in matching_legs])
  if expires_at is not None:
    summary["expires_at"] = expires_at.isoformat()
  for field in (
    "basket_name",
    "action",
    "total_estimated_cost",
    "total_estimated_proceeds",
    "net_estimated_cash",
    "gross_estimated_notional",
    "total_legs",
    "buy_legs",
    "sell_legs",
    "skipped_legs",
    "skipped_tickers",
    "warnings",
  ):
    _copy_if_present(summary, source, field)

  legs_summary = [_basket_leg_summary(leg) for leg in matching_legs]
  if legs_summary:
    summary["legs"] = legs_summary

  return summary


def _all_legs_have_fresh_expiry(legs: list[Mapping[str, Any]]) -> bool:
  for leg in legs:
    expires_at = leg.get("expires_at")
    if expires_at in (None, "") or _is_expired(expires_at):
      return False
  return True


def _basket_leg_summary(leg: Mapping[str, Any]) -> dict[str, Any]:
  summary: dict[str, Any] = {}
  for field in (
    "ticker",
    "side",
    "quantity",
    "estimated_price",
    "estimated_total",
    "preview_id",
    "pre_trade_weight",
    "post_trade_weight",
    "target_weight",
    "status",
    "error",
  ):
    _copy_if_present(summary, leg, field)
  return summary


def _preview_ids_from_args(args: Mapping[str, Any]) -> list[str]:
  raw = args.get("preview_ids")
  if raw is None:
    return []
  if isinstance(raw, str):
    text = raw.strip()
    if not text:
      return []
    try:
      parsed = json.loads(text)
    except json.JSONDecodeError:
      parsed = [part.strip() for part in text.split(",")]
    raw = parsed
  if not isinstance(raw, (list, tuple, set, frozenset)):
    return []
  return [_text(value) for value in raw if _text(value)]


def _is_expired(value: Any) -> bool:
  expires_at = _parse_datetime(value)
  if expires_at is None:
    return False
  return _as_utc(expires_at) <= datetime.now(UTC)


def _seconds_until_expiry(value: Any, *, now: datetime | None = None) -> float | None:
  expires_at = _parse_datetime(value)
  if expires_at is None:
    return None
  current = datetime.now(UTC) if now is None else _as_utc(now)
  return (_as_utc(expires_at) - current).total_seconds()


def _earliest_expiry(values: list[Any]) -> datetime | None:
  parsed = [expires_at for expires_at in (_parse_datetime(value) for value in values) if expires_at is not None]
  if not parsed:
    return None
  return min(_as_utc(value) for value in parsed)


def _as_utc(value: datetime) -> datetime:
  if value.tzinfo is None:
    return value.replace(tzinfo=UTC)
  return value.astimezone(UTC)


def _parse_datetime(value: Any) -> datetime | None:
  if value is None or value == "":
    return None
  if isinstance(value, datetime):
    return value
  raw = str(value).strip()
  if not raw:
    return None
  try:
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
  except ValueError:
    return None


def _positive_float(value: Any, default: float) -> float:
  try:
    parsed = float(value)
  except (TypeError, ValueError):
    return default
  return parsed if parsed > 0 else default


def _copy_if_present(target: dict[str, Any], source: Mapping[str, Any], key: str) -> None:
  value = source.get(key)
  if value is not None and value != "":
    target[key] = value


def _text(value: Any) -> str:
  return str(value or "").strip()


def _policy_tool_name(tool_name: str) -> str:
  if tool_name.startswith("mcp__"):
    parts = tool_name.split("__", 2)
    if len(parts) == 3:
      return parts[2]
  return tool_name


__all__ = ["effective_trade_approval_expiry_seconds", "enrich_trade_approval_args"]
