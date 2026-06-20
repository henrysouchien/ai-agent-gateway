from __future__ import annotations

import copy
import hashlib
import hmac
import logging
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Literal

from .approval_policy import ApprovalRequest, ApprovalVote, PersistentGrant, ToolClass, canonical_json, decision_latency_ms, utc_now
from .skill_context import current_skill
from .tool_redaction import redact_tool_input as default_redact_tool_input


log = logging.getLogger("agent_gateway.approval_audit")

AuditEventType = Literal[
  "request_created",
  "auto_approved",
  "auto_denied",
  "user_hold_started",
  "external_route_started",
  "external_callback_received",
  "vote_recorded",
  "approved",
  "denied",
  "expired",
  "persistent_grant_used",
  "persistent_grant_created",
  "persistent_grant_revoked",
  "tool_executed_success",
  "tool_executed_error",
  "tool_cancelled",
]


class AuditBuildError(RuntimeError):
  """Sanitized audit-entry construction failure."""


@dataclass(frozen=True, kw_only=True)
class ApprovalAuditEntry:
  entry_id: str
  approval_id: str
  request_id: str
  tool_call_id: str
  parent_approval_id: str | None
  approval_chain_id: str
  pending_tools_nonce: str | None
  ts: datetime
  event_type: AuditEventType
  user_id: str
  profile: str
  channel: str
  session_id: str | None
  run_id: str | None
  skill: str | None
  tool_name: str
  tool_class: ToolClass
  tool_args_redacted: dict[str, Any]
  args_hash: str
  args_hash_version: str
  decider_id: str | None
  decider_role: str | None
  decision_reason: str | None
  decision_latency_ms: int | None
  outcome: Literal["success", "tool_error", "cancelled"] | None
  error_summary: str | None
  policy_id: str
  policy_version: str
  policy_bundle_hash: str
  model_id: str | None
  model_version: str | None
  system_prompt_hash: str | None
  tool_schema_version: str | None
  mcp_server_version: str | None
  tenant_id: str | None = None
  retention_class: Literal["dev", "operational", "compliance"] = "operational"
  legal_hold: bool = False

  def to_json_dict(self) -> dict[str, Any]:
    payload = asdict(self)
    payload["ts"] = self.ts.isoformat()
    return payload


def _args_hash_version(args_hash: str, *, key_id: str) -> str:
  if args_hash.startswith("hmac-sha256-v1:"):
    parts = args_hash.split(":", 2)
    if len(parts) == 3 and parts[1]:
      return f"{parts[0]}:{parts[1]}"
  if len(args_hash) == 64 and all(char in "0123456789abcdef" for char in args_hash):
    return "sha256"
  return f"hmac-sha256-v1:{key_id}"


def build_audit_entry(
  *,
  raw_tool_args: dict[str, Any],
  deployment_secret: bytes,
  key_id: str,
  event_type: AuditEventType,
  request: ApprovalRequest,
  pending_tools_nonce: str | None = None,
  vote: ApprovalVote | None = None,
  grant: PersistentGrant | None = None,
  outcome: Literal["success", "tool_error", "cancelled"] | None = None,
  error_summary: str | None = None,
  skill: str | None = None,
  retention_class: Literal["dev", "operational", "compliance"] = "operational",
  legal_hold: bool = False,
  tool_input_redactor: Callable[..., dict[str, Any]] | None = None,
) -> ApprovalAuditEntry:
  _alias = raw_tool_args
  del raw_tool_args
  args_copy: dict[str, Any] | None = None
  args_hash: str | None = None
  args_hash_version = f"hmac-sha256-v1:{key_id}"
  tool_args_redacted: dict[str, Any] | None = None
  caught: BaseException | None = None
  try:
    if _alias:
      args_copy = dict(_alias)
      digest = hmac.new(
        deployment_secret,
        canonical_json(args_copy).encode("utf-8"),
        hashlib.sha256,
      ).hexdigest()
      args_hash = f"hmac-sha256-v1:{key_id}:{digest}"
      redactor = tool_input_redactor or default_redact_tool_input
      tool_args_redacted = redactor(
        request.tool_name,
        args_copy,
        deployment_secret=deployment_secret,
        key_id=key_id,
        redaction_scope="fresh_raw",
      )
    else:
      tool_args_redacted = copy.deepcopy(request.tool_args_redacted) if request.tool_args_redacted else {}
      if request.args_hash:
        args_hash = request.args_hash
        args_hash_version = _args_hash_version(args_hash, key_id=key_id)
      else:
        args_copy = {}
        digest = hmac.new(
          deployment_secret,
          canonical_json(args_copy).encode("utf-8"),
          hashlib.sha256,
        ).hexdigest()
        args_hash = f"hmac-sha256-v1:{key_id}:{digest}"
  except BaseException as exc:
    caught = exc
  finally:
    if caught is not None and caught.__traceback__ is not None:
      traceback.clear_frames(caught.__traceback__)
    if args_copy is not None:
      args_copy.clear()
    del args_copy
    del _alias
    deployment_secret = b""

  if caught is not None:
    del caught
    raise AuditBuildError(
      "build_audit_entry failed (raw args, secret, and original exception detail redacted)"
    ) from None

  grant_tool_name = grant.tool_name if grant is not None else request.tool_name
  resolved_skill = skill if skill is not None else getattr(request, "skill", None)
  return ApprovalAuditEntry(
    entry_id=str(uuid.uuid4()),
    approval_id=request.approval_id,
    request_id=request.request_id,
    tool_call_id=request.tool_call_id,
    parent_approval_id=request.parent_approval_id,
    approval_chain_id=request.approval_chain_id,
    pending_tools_nonce=pending_tools_nonce,
    ts=utc_now(),
    event_type=event_type,
    user_id=request.user_id,
    profile=request.profile,
    channel=request.channel,
    session_id=request.session_id,
    run_id=request.run_id,
    skill=resolved_skill,
    tool_name=grant_tool_name,
    tool_class=request.tool_class,
    tool_args_redacted=tool_args_redacted or {},
    args_hash=args_hash or "",
    args_hash_version=args_hash_version,
    decider_id=(vote.decider_id if vote is not None else request.decider_id),
    decider_role=(vote.decider_role if vote is not None else request.decider_role),
    decision_reason=(vote.decision_reason if vote is not None else request.decision_reason),
    decision_latency_ms=decision_latency_ms(request),
    outcome=outcome,
    error_summary=error_summary,
    policy_id=request.policy_id,
    policy_version=request.policy_version,
    policy_bundle_hash=request.policy_bundle_hash,
    model_id=request.model_id,
    model_version=request.model_version,
    system_prompt_hash=request.system_prompt_hash,
    tool_schema_version=request.tool_schema_version,
    mcp_server_version=request.mcp_server_version,
    tenant_id=request.tenant_id,
    retention_class=retention_class,
    legal_hold=legal_hold,
  )


class ApprovalAuditEmitter:
  def __init__(
    self,
    *,
    writer: Any,
    deployment_secret: bytes,
    key_id: str,
    tool_input_redactor: Callable[..., dict[str, Any]] | None = None,
  ) -> None:
    self._writer = writer
    self._deployment_secret = deployment_secret
    self._key_id = key_id
    self._tool_input_redactor = tool_input_redactor

  async def emit_audit_for_lifecycle_event(
    self,
    *,
    event_type: AuditEventType,
    request: ApprovalRequest,
    raw_tool_args: dict[str, Any],
    vote: ApprovalVote | None = None,
    grant: PersistentGrant | None = None,
    outcome: Literal["success", "tool_error", "cancelled"] | None = None,
    error_summary: str | None = None,
    pending_tools_nonce: str | None = None,
    skill: str | None = None,
  ) -> None:
    entry = build_audit_entry(
      raw_tool_args=raw_tool_args,
      deployment_secret=self._deployment_secret,
      key_id=self._key_id,
      event_type=event_type,
      request=request,
      vote=vote,
      grant=grant,
      outcome=outcome,
      error_summary=error_summary,
      pending_tools_nonce=pending_tools_nonce,
      skill=skill if skill is not None else current_skill(),
      tool_input_redactor=self._tool_input_redactor,
    )
    del raw_tool_args
    await self._write(entry, pre_execution=event_type not in {"tool_executed_success", "tool_executed_error", "tool_cancelled"})

  async def emit_execution_outcome(
    self,
    *,
    request: ApprovalRequest,
    raw_tool_args: dict[str, Any],
    outcome: Literal["success", "tool_error", "cancelled"],
    error_summary: str | None = None,
  ) -> None:
    event_type: AuditEventType = {
      "success": "tool_executed_success",
      "tool_error": "tool_executed_error",
      "cancelled": "tool_cancelled",
    }[outcome]  # type: ignore[assignment]
    await self.emit_audit_for_lifecycle_event(
      event_type=event_type,
      request=request,
      raw_tool_args=raw_tool_args,
      outcome=outcome,
      error_summary=error_summary,
    )

  async def emit_grant_event(
    self,
    *,
    event_type: AuditEventType,
    grant: PersistentGrant,
    request: ApprovalRequest | None = None,
  ) -> None:
    if request is None:
      return
    await self.emit_audit_for_lifecycle_event(
      event_type=event_type,
      request=request,
      raw_tool_args={},
      grant=grant,
    )

  async def _write(self, entry: ApprovalAuditEntry, *, pre_execution: bool) -> None:
    try:
      await self._writer.write(entry)
    except Exception:
      if pre_execution and entry.tool_class in {"state_write", "external_write", "portfolio_config", "irreversible"}:
        raise
      log.warning("Approval audit write failed for %s/%s", entry.approval_id, entry.event_type, exc_info=True)
