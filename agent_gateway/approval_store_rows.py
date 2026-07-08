from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from .approval_policy import ApprovalRequest, DelegationGrant, PersistentGrant, utc_now


def dt_to_text(value: datetime | None) -> str | None:
  if value is None:
    return None
  if value.tzinfo is None:
    value = value.replace(tzinfo=UTC)
  return value.astimezone(UTC).isoformat()


def dt_from_text(value: str | None) -> datetime | None:
  if value is None:
    return None
  parsed = datetime.fromisoformat(value)
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=UTC)
  return parsed.astimezone(UTC)


def json_dumps(value: Any) -> str:
  return json.dumps(value, sort_keys=True, default=str)


def json_loads(value: str | None) -> Any:
  if not value:
    return None
  return json.loads(value)


def request_to_row(request: ApprovalRequest) -> dict[str, Any]:
  return {
    "approval_id": request.approval_id,
    "tool_call_id": request.tool_call_id,
    "parent_approval_id": request.parent_approval_id,
    "approval_chain_id": request.approval_chain_id,
    "delegation_id": request.delegation_id,
    "request_id": request.request_id,
    "session_id": request.session_id,
    "run_id": request.run_id,
    "skill": request.skill,
    "user_id": request.user_id,
    "profile": request.profile,
    "channel": request.channel,
    "tool_name": request.tool_name,
    "tool_class": request.tool_class,
    "tool_args_redacted": json_dumps(request.tool_args_redacted),
    "args_hash": request.args_hash,
    "reason": request.reason,
    "blast_radius_summary": request.blast_radius_summary,
    "state": request.state,
    "state_version": request.state_version,
    "requested_at": dt_to_text(request.requested_at),
    "decided_at": dt_to_text(request.decided_at),
    "expires_at": dt_to_text(request.expires_at),
    "decider_id": request.decider_id,
    "decider_role": request.decider_role,
    "decision": request.decision,
    "decision_reason": request.decision_reason,
    "required_decider_count": request.required_decider_count,
    "eligible_decider_count": request.eligible_decider_count,
    "votes_received_count": request.votes_received_count,
    "args_predicate": json_dumps(request.args_predicate) if request.args_predicate is not None else None,
    "chain_trust_window_seconds": request.chain_trust_window_seconds,
    "route_target": request.route_target,
    "route_target_type": request.route_target_type,
    "external_callback_id": request.external_callback_id,
    "policy_id": request.policy_id,
    "policy_version": request.policy_version,
    "policy_bundle_hash": request.policy_bundle_hash,
    "persistent_grant_scope": request.persistent_grant_scope,
    "tenant_id": request.tenant_id,
    "model_id": request.model_id,
    "model_version": request.model_version,
    "system_prompt_hash": request.system_prompt_hash,
    "tool_schema_version": request.tool_schema_version,
    "mcp_server_version": request.mcp_server_version,
    "notification_policy": request.notification_policy,
  }


def row_to_request(row: sqlite3.Row) -> ApprovalRequest:
  return ApprovalRequest(
    approval_id=str(row["approval_id"]),
    tool_call_id=str(row["tool_call_id"]),
    parent_approval_id=row["parent_approval_id"],
    approval_chain_id=str(row["approval_chain_id"]),
    delegation_id=row["delegation_id"] if "delegation_id" in row.keys() else None,
    request_id=str(row["request_id"]),
    session_id=row["session_id"],
    run_id=row["run_id"],
    skill=row["skill"] if "skill" in row.keys() else None,
    user_id=str(row["user_id"]),
    profile=str(row["profile"]),
    channel=str(row["channel"]),
    tool_name=str(row["tool_name"]),
    tool_class=str(row["tool_class"]),  # type: ignore[arg-type]
    tool_args_redacted=json_loads(row["tool_args_redacted"]) or {},
    args_hash=str(row["args_hash"]),
    reason=row["reason"],
    blast_radius_summary=str(row["blast_radius_summary"]),
    state=str(row["state"]),  # type: ignore[arg-type]
    state_version=int(row["state_version"] or 0),
    requested_at=dt_from_text(row["requested_at"]) or utc_now(),
    decided_at=dt_from_text(row["decided_at"]),
    expires_at=dt_from_text(row["expires_at"]),
    decider_id=row["decider_id"],
    decider_role=row["decider_role"],
    decision=row["decision"],  # type: ignore[arg-type]
    decision_reason=row["decision_reason"],
    required_decider_count=int(row["required_decider_count"] or 1),
    eligible_decider_count=int(row["eligible_decider_count"] or 1),
    votes_received_count=int(row["votes_received_count"] or 0),
    args_predicate=json_loads(row["args_predicate"]),
    chain_trust_window_seconds=row["chain_trust_window_seconds"],
    route_target=row["route_target"],
    route_target_type=row["route_target_type"],
    external_callback_id=row["external_callback_id"],
    policy_id=str(row["policy_id"]),
    policy_version=str(row["policy_version"]),
    policy_bundle_hash=str(row["policy_bundle_hash"]),
    persistent_grant_scope=row["persistent_grant_scope"],
    tenant_id=row["tenant_id"],
    model_id=row["model_id"],
    model_version=row["model_version"],
    system_prompt_hash=row["system_prompt_hash"],
    tool_schema_version=row["tool_schema_version"],
    mcp_server_version=row["mcp_server_version"],
    notification_policy=row["notification_policy"] if "notification_policy" in row.keys() else "auto",
  )


def row_to_grant(row: sqlite3.Row) -> PersistentGrant:
  return PersistentGrant(
    grant_id=str(row["grant_id"]),
    user_id=str(row["user_id"]),
    tool_name=str(row["tool_name"]),
    scope_hint=str(row["scope_hint"]),
    args_predicate=json_loads(row["args_predicate"]),
    granted_at=dt_from_text(row["granted_at"]) or utc_now(),
    expires_at=dt_from_text(row["expires_at"]),
    revoked_at=dt_from_text(row["revoked_at"]),
    granted_via_approval_id=str(row["granted_via_approval_id"]),
    policy_id=str(row["policy_id"]),
  )


def row_to_delegation_grant(row: sqlite3.Row) -> DelegationGrant:
  return DelegationGrant(
    delegation_id=str(row["delegation_id"]),
    delegator_user_id=str(row["delegator_user_id"]),
    delegator_run_id=row["delegator_run_id"],
    delegator_session_id=row["delegator_session_id"],
    delegator_profile=str(row["delegator_profile"]),
    delegator_channel=str(row["delegator_channel"]),
    bound_excel_session_id=str(row["bound_excel_session_id"]),
    bound_relay_request_id=str(row["bound_relay_request_id"]),
    bound_workbook=row["bound_workbook"],
    tool_class_ceiling=frozenset(json_loads(row["tool_class_ceiling"]) or []),  # type: ignore[arg-type]
    args_predicate=json_loads(row["args_predicate"]),
    window_seconds=int(row["window_seconds"]),
    exclude_external_write_bypass=bool(row["exclude_external_write_bypass"]),
    created_at=dt_from_text(row["created_at"]) or utc_now(),
    expires_at=dt_from_text(row["expires_at"]),
    revoked_at=dt_from_text(row["revoked_at"]),
    consumed_at=dt_from_text(row["consumed_at"]),
  )


__all__ = [
  "dt_from_text",
  "dt_to_text",
  "json_dumps",
  "json_loads",
  "request_to_row",
  "row_to_delegation_grant",
  "row_to_grant",
  "row_to_request",
]
