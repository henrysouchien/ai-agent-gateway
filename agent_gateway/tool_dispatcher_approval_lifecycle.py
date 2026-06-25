from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any

from . import approval_settings
from .approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest as PolicyApprovalRequest,
  sha256_args,
  utc_now,
)
from .approval_enrichment import effective_trade_approval_expiry_seconds, enrich_trade_approval_args


async def await_user_approval_via_pending_tools(
  *,
  session: Any | None,
  approval_store: Any | None,
  event_log: Any | None,
  request: PolicyApprovalRequest,
  decision: PolicyApprovalDecision,
  nonce: str,
  resolved_qualifier: str,
  allow_persistent: bool,
  timeout_seconds: float,
  log: logging.Logger,
) -> dict[str, Any] | None:
  if session is None:
    return None
  approval_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
  session.pending_tools[request.tool_call_id] = {
    "approval_id": request.approval_id,
    "nonce": nonce,
    "requested_at": int(time.time()),
    "status": "approval_pending",
    "tool_name": request.tool_name,
    "resolved_qualifier": resolved_qualifier,
  }
  session.approval_queues[request.tool_call_id] = approval_queue

  approval_event = {
    "type": "tool_approval_request",
    "tool_call_id": request.tool_call_id,
    "approval_id": request.approval_id,
    "nonce": nonce,
    "tool_name": request.tool_name,
    "tool_input": request.tool_args_redacted,
    "resolved_qualifier": resolved_qualifier,
    "reason": decision.reason,
    "allow_persistent_approval": allow_persistent and decision.allow_persistent_grant,
    "ts": time.time(),
  }
  if event_log is not None:
    event_log.append(approval_event)
  session_log = getattr(session, "agent_session_log", None)
  if session_log is not None:
    try:
      await session_log.append(approval_event)
    except Exception:
      log.warning("Failed to persist approval request event for %s", request.tool_call_id, exc_info=True)

  try:
    return await asyncio.wait_for(approval_queue.get(), timeout=max(0.1, timeout_seconds))
  except asyncio.TimeoutError:
    if approval_store is not None:
      try:
        latest = await approval_store.get(request.approval_id)
        if latest is not None and latest.state == "pending_user":
          await approval_store.transition_state(
            latest.approval_id,
            "expired",
            expected_state_version=latest.state_version,
            decision_reason="Timed out waiting for user approval",
          )
      except Exception:
        log.warning("Failed to expire timed-out approval request %s", request.approval_id, exc_info=True)
    return None
  finally:
    session.pending_tools.pop(request.tool_call_id, None)
    session.approval_queues.pop(request.tool_call_id, None)


def redact_for_approval_request(
  tool_name: str,
  tool_input: dict[str, Any],
  *,
  event_log: Any | None,
  enrich_trade_approval_args_fn: Any = enrich_trade_approval_args,
  sha256_args_fn: Any = sha256_args,
) -> tuple[dict[str, Any], str]:
  try:
    from agent.shared.tool_redaction import get_audit_hmac_secret, get_audit_hmac_key_id, hmac_value, redact_tool_input

    secret = get_audit_hmac_secret()
    key_id = get_audit_hmac_key_id()
    redacted = redact_tool_input(
      tool_name,
      tool_input,
      deployment_secret=secret,
      key_id=key_id,
      redaction_scope="fresh_raw",
    )
    redacted = enrich_trade_approval_args_fn(tool_name, redacted, event_log=event_log)
    args_hash = hmac_value(tool_input, deployment_secret=secret, key_id=key_id)
    return redacted, args_hash
  except Exception:
    return {}, sha256_args_fn(tool_input)


def approval_transport_input(
  tool_name: str,
  tool_input: dict[str, Any],
  *,
  event_log: Any | None,
  enrich_trade_approval_args_fn: Any = enrich_trade_approval_args,
) -> dict[str, Any]:
  return enrich_trade_approval_args_fn(tool_name, dict(tool_input), event_log=event_log)


def effective_trade_approval_decision(
  tool_name: str,
  tool_args_redacted: dict[str, Any],
  decision: PolicyApprovalDecision,
  *,
  effective_trade_approval_expiry_seconds_fn: Any = effective_trade_approval_expiry_seconds,
  approval_wait_seconds_fn: Any = approval_settings.approval_wait_seconds,
  utc_now_fn: Any = utc_now,
) -> PolicyApprovalDecision:
  expiry_seconds = effective_trade_approval_expiry_seconds_fn(
    tool_name,
    tool_args_redacted,
    requested_expiry_seconds=decision.expiry_seconds,
    max_wait_seconds=approval_wait_seconds_fn(),
    now=utc_now_fn(),
  )
  if expiry_seconds == decision.expiry_seconds:
    return decision
  return replace(decision, expiry_seconds=expiry_seconds)
