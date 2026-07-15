from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import HTTPException

from agent_gateway.skill_context import current_skill
from agent_gateway.artifact_paths import canonicalize_ticker

from . import batches
from .runs_helpers import _session_owner_user_id

VALUATION_READY_SKILL = "valuation-ready"
VALUATION_READY_TEMPLATE = "valuation-ready"
EXPLICIT_TICKER_SOURCE = "explicit_ticker"

_DILIGENCE_TRACKS_MODULE_NAMES = frozenset({"agent", "agent.skills", "agent.skills.diligence_tracks"})


VALUATION_READY_BATCH_DISPATCH_TOOL_DEF: dict[str, Any] = {
  "name": "valuation_ready_batch_dispatch",
  "description": "Start a gateway-local valuation-ready batch for one explicit ticker.",
  "input_schema": {
    "type": "object",
    "properties": {
      "ticker": {
        "type": "string",
        "description": "Explicit public-company ticker symbol to run through the valuation-ready workflow.",
      },
      "budget_usd": {
        "type": "number",
        "description": "Optional total batch budget. Defaults to the valuation-ready catalog suggestion.",
        "minimum": 0,
      },
      "max_concurrency": {
        "type": "integer",
        "description": "Optional max concurrent ticker runs. Single-ticker dispatch defaults to 1.",
        "minimum": 1,
      },
    },
    "required": ["ticker"],
    "additionalProperties": False,
  },
}


VALUATION_READY_BATCH_READ_TOOL_DEF: dict[str, Any] = {
  "name": "valuation_ready_batch_read",
  "description": "Read a gateway-local valuation-ready batch digest, verdict matrix, candidates, and failures.",
  "input_schema": {
    "type": "object",
    "properties": {
      "batch_id": {
        "type": "integer",
        "description": "Gateway control-plane batch id returned by valuation_ready_batch_dispatch.",
        "minimum": 1,
      },
      "top_n": {
        "type": "integer",
        "description": "Maximum candidate rows to include.",
        "minimum": 1,
        "maximum": 100,
        "default": 10,
      },
    },
    "required": ["batch_id"],
    "additionalProperties": False,
  },
}


ToolHandler = Callable[[dict[str, Any]], Awaitable[tuple[Any | None, dict[str, Any] | None]]]


def make_valuation_ready_skill_tool_bundle(*, app_state: Any, session: Any) -> dict[str, Any]:
  return {
    "skill_name": VALUATION_READY_SKILL,
    "tool_definitions": [
      VALUATION_READY_BATCH_DISPATCH_TOOL_DEF,
      VALUATION_READY_BATCH_READ_TOOL_DEF,
    ],
    "handlers": {
      "valuation_ready_batch_dispatch": _make_dispatch_handler(app_state=app_state, session=session),
      "valuation_ready_batch_read": _make_read_handler(session=session),
    },
  }


def _unsupported_runtime_error() -> dict[str, Any]:
  return {
    "code": "unsupported_runtime",
    "message": (
      "valuation-ready batch initiation is available only inside the gateway-hosted "
      "interactive valuation-ready skill runtime."
    ),
    "details": {
      "verdict": "INSUFFICIENT_DATA",
      "reason": "unsupported_runtime",
      "supported_runtime": "gateway_interactive",
    },
  }


def _guard_active_skill() -> dict[str, Any] | None:
  if current_skill() == VALUATION_READY_SKILL:
    return None
  return _unsupported_runtime_error()


def _normalize_ticker(value: Any) -> str | None:
  try:
    return canonicalize_ticker(value)
  except ValueError:
    return None


def _valuation_ready_defaults() -> dict[str, Any]:
  try:
    from agent.skills.diligence_tracks import batch_workflow_defaults
  except ModuleNotFoundError as exc:
    if exc.name not in _DILIGENCE_TRACKS_MODULE_NAMES:
      raise
    from api.agent.skills.diligence_tracks import batch_workflow_defaults

  return batch_workflow_defaults(VALUATION_READY_TEMPLATE)


def _dispatch_spec(tool_input: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  ticker = _normalize_ticker((tool_input or {}).get("ticker"))
  if ticker is None:
    return None, {
      "code": "invalid_ticker",
      "message": "valuation_ready_batch_dispatch requires a valid explicit ticker.",
      "details": {"verdict": "INSUFFICIENT_DATA", "reason": "missing_or_invalid_ticker"},
    }
  defaults = _valuation_ready_defaults()
  budget_raw = (tool_input or {}).get("budget_usd")
  if budget_raw is None:
    budget_usd = float(defaults.get("suggested_budget_usd_per_name") or 25.0)
  else:
    try:
      budget_usd = float(budget_raw)
    except (TypeError, ValueError):
      return None, {
        "code": "invalid_budget",
        "message": "budget_usd must be numeric when provided.",
        "details": {"verdict": "INSUFFICIENT_DATA", "reason": "invalid_budget"},
      }
    if budget_usd < 0:
      return None, {
        "code": "invalid_budget",
        "message": "budget_usd must be non-negative.",
        "details": {"verdict": "INSUFFICIENT_DATA", "reason": "invalid_budget"},
      }
  max_concurrency_raw = (tool_input or {}).get("max_concurrency")
  if max_concurrency_raw is None:
    max_concurrency = 1
  else:
    try:
      max_concurrency = int(max_concurrency_raw)
    except (TypeError, ValueError):
      return None, {
        "code": "invalid_max_concurrency",
        "message": "max_concurrency must be an integer when provided.",
        "details": {"verdict": "INSUFFICIENT_DATA", "reason": "invalid_max_concurrency"},
      }
    if max_concurrency < 1:
      return None, {
        "code": "invalid_max_concurrency",
        "message": "max_concurrency must be at least 1.",
        "details": {"verdict": "INSUFFICIENT_DATA", "reason": "invalid_max_concurrency"},
      }
  return {
    "source": EXPLICIT_TICKER_SOURCE,
    "universe": [ticker],
    "pipeline_template": VALUATION_READY_TEMPLATE,
    "budget_usd": budget_usd,
    "max_concurrency": max_concurrency,
  }, None


def _make_dispatch_handler(*, app_state: Any, session: Any) -> ToolHandler:
  async def _handle(tool_input: dict[str, Any], **_: Any) -> tuple[Any | None, dict[str, Any] | None]:
    unsupported = _guard_active_skill()
    if unsupported is not None:
      return None, unsupported
    spec, error = _dispatch_spec(tool_input or {})
    if error is not None:
      return None, error
    assert spec is not None
    try:
      payload = await batches.dispatch_batch_in_process(
        spec,
        app_state=app_state,
        user_id=_session_owner_user_id(session),
        user_email=getattr(session, "user_email", None),
        channel=getattr(session, "channel", None),
      )
    except batches._active_batch_error_type() as exc:
      return None, {"code": "active_batch_conflict", "message": str(exc)}
    except ValueError as exc:
      return None, {"code": "invalid_batch_spec", "message": str(exc)}
    except Exception as exc:
      return None, {"code": "batch_dispatch_failed", "message": str(exc)}
    return {
      **payload,
      "source": EXPLICIT_TICKER_SOURCE,
      "pipeline_template": VALUATION_READY_TEMPLATE,
      "ticker": spec["universe"][0],
      "budget_usd": spec["budget_usd"],
      "max_concurrency": spec["max_concurrency"],
    }, None

  return _handle


def _make_read_handler(*, session: Any) -> ToolHandler:
  async def _handle(tool_input: dict[str, Any], **_: Any) -> tuple[Any | None, dict[str, Any] | None]:
    unsupported = _guard_active_skill()
    if unsupported is not None:
      return None, unsupported
    try:
      batch_id = int((tool_input or {}).get("batch_id"))
    except (TypeError, ValueError):
      return None, {"code": "invalid_batch_id", "message": "batch_id must be a positive integer."}
    if batch_id < 1:
      return None, {"code": "invalid_batch_id", "message": "batch_id must be a positive integer."}
    try:
      top_n = int((tool_input or {}).get("top_n") or 10)
    except (TypeError, ValueError):
      return None, {"code": "invalid_top_n", "message": "top_n must be an integer when provided."}
    top_n = max(1, min(top_n, 100))
    try:
      return batches.read_batch_for_user(
        batch_id,
        user_id=_session_owner_user_id(session),
        top_n=top_n,
      ), None
    except HTTPException as exc:
      return None, {"code": "batch_read_failed", "message": str(exc.detail)}
    except Exception as exc:
      return None, {"code": "batch_read_failed", "message": str(exc)}

  return _handle


__all__ = [
  "EXPLICIT_TICKER_SOURCE",
  "VALUATION_READY_BATCH_DISPATCH_TOOL_DEF",
  "VALUATION_READY_BATCH_READ_TOOL_DEF",
  "VALUATION_READY_SKILL",
  "VALUATION_READY_TEMPLATE",
  "make_valuation_ready_skill_tool_bundle",
]
