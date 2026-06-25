from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import os
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Protocol

from .approval_policy import (
  ApprovalRequest,
  ApprovalState,
  ApprovalVote,
  DelegationGrant,
  PersistentGrant,
  utc_now,
)


TERMINAL_STATES = frozenset({"auto_approved", "auto_denied", "approved", "denied", "expired"})


def _dt_to_text(value: datetime | None) -> str | None:
  if value is None:
    return None
  if value.tzinfo is None:
    value = value.replace(tzinfo=UTC)
  return value.astimezone(UTC).isoformat()


def _dt_from_text(value: str | None) -> datetime | None:
  if value is None:
    return None
  parsed = datetime.fromisoformat(value)
  if parsed.tzinfo is None:
    parsed = parsed.replace(tzinfo=UTC)
  return parsed.astimezone(UTC)


def _json_dumps(value: Any) -> str:
  return json.dumps(value, sort_keys=True, default=str)


def _json_loads(value: str | None) -> Any:
  if not value:
    return None
  return json.loads(value)


class ApprovalRequestStore(Protocol):
  async def create(self, request: ApprovalRequest) -> ApprovalRequest: ...
  async def get(self, approval_id: str) -> ApprovalRequest | None: ...
  async def get_by_tool_call_id(self, tool_call_id: str) -> ApprovalRequest | None: ...
  async def update_request(self, request: ApprovalRequest) -> ApprovalRequest: ...
  async def transition_state(
    self,
    approval_id: str,
    state: ApprovalState,
    *,
    expected_state_version: int | None = None,
    route_target_type: str | None = None,
    route_target: str | None = None,
    expires_at: datetime | None = None,
    decider_id: str | None = None,
    decider_role: str | None = None,
    decision: str | None = None,
    decision_reason: str | None = None,
  ) -> ApprovalRequest: ...
  async def record_vote(self, approval_id: str, vote: ApprovalVote) -> ApprovalRequest: ...
  async def create_persistent_grant(self, grant: PersistentGrant) -> PersistentGrant: ...
  async def find_persistent_grant(
    self,
    *,
    user_id: str,
    tool_name: str,
    scope_hint: str,
    now: datetime | None = None,
  ) -> PersistentGrant | None: ...
  async def revoke_persistent_grant(self, grant_id: str, *, revoked_at: datetime | None = None) -> None: ...
  async def create_delegation_grant(self, grant: DelegationGrant) -> DelegationGrant: ...
  async def get_delegation_grant(self, delegation_id: str) -> DelegationGrant | None: ...
  async def claim_delegation_grant(
    self,
    *,
    delegation_id: str,
    bound_relay_request_id: str,
    bound_excel_session_id: str,
    now: datetime | None = None,
  ) -> DelegationGrant | None: ...
  async def revoke_delegation_grant(self, delegation_id: str, *, revoked_at: datetime | None = None) -> None: ...
  async def expire_pending(self, *, now: datetime | None = None) -> int: ...


class SQLiteApprovalStore:
  """SQLite-backed approval store with atomic vote recording."""

  def __init__(self, path: str | os.PathLike[str] = "data/gateway/approvals.sqlite3", *, audit_emitter: Any | None = None) -> None:
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._audit_emitter = audit_emitter
    self._lock = asyncio.Lock()
    self._init_schema()

  @property
  def audit_emitter(self) -> Any | None:
    return self._audit_emitter

  def _connect(self) -> sqlite3.Connection:
    conn = sqlite3.connect(self.path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

  @contextmanager
  def _connection(self) -> Iterator[sqlite3.Connection]:
    conn = self._connect()
    try:
      with conn:
        yield conn
    finally:
      conn.close()

  def _init_schema(self) -> None:
    with self._connection() as conn:
      conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS approval_requests (
          approval_id TEXT PRIMARY KEY,
          tool_call_id TEXT NOT NULL,
          parent_approval_id TEXT,
          approval_chain_id TEXT NOT NULL,
          delegation_id TEXT,
          request_id TEXT NOT NULL,
          session_id TEXT,
          run_id TEXT,
          user_id TEXT NOT NULL,
          profile TEXT NOT NULL,
          channel TEXT NOT NULL,
          tool_name TEXT NOT NULL,
          tool_class TEXT NOT NULL,
          tool_args_redacted TEXT NOT NULL,
          args_hash TEXT NOT NULL,
          reason TEXT,
          blast_radius_summary TEXT NOT NULL,
          state TEXT NOT NULL,
          state_version INTEGER NOT NULL DEFAULT 0,
          requested_at TEXT NOT NULL,
          decided_at TEXT,
          expires_at TEXT,
          decider_id TEXT,
          decider_role TEXT,
          decision TEXT,
          decision_reason TEXT,
          required_decider_count INTEGER NOT NULL DEFAULT 1,
          eligible_decider_count INTEGER NOT NULL DEFAULT 1,
          votes_received_count INTEGER NOT NULL DEFAULT 0,
          args_predicate TEXT,
          chain_trust_window_seconds INTEGER,
          route_target TEXT,
          route_target_type TEXT,
          external_callback_id TEXT,
          policy_id TEXT NOT NULL,
          policy_version TEXT NOT NULL,
          policy_bundle_hash TEXT NOT NULL,
          persistent_grant_scope TEXT,
          tenant_id TEXT,
          model_id TEXT,
          model_version TEXT,
          system_prompt_hash TEXT,
          tool_schema_version TEXT,
          mcp_server_version TEXT,
          skill TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_approval_requests_tool_call_id
          ON approval_requests(tool_call_id);
        CREATE INDEX IF NOT EXISTS idx_approval_requests_state_expires
          ON approval_requests(state, expires_at);

        CREATE TABLE IF NOT EXISTS approval_votes (
          vote_id TEXT PRIMARY KEY,
          approval_id TEXT NOT NULL REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
          decider_id TEXT NOT NULL,
          decider_role TEXT,
          decision TEXT NOT NULL,
          decision_reason TEXT,
          decided_at TEXT NOT NULL,
          external_callback_id TEXT,
          UNIQUE(approval_id, decider_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_approval_votes_external_callback
          ON approval_votes(external_callback_id)
          WHERE external_callback_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS persistent_grants (
          grant_id TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          tool_name TEXT NOT NULL,
          scope_hint TEXT NOT NULL,
          args_predicate TEXT,
          granted_at TEXT NOT NULL,
          expires_at TEXT,
          revoked_at TEXT,
          granted_via_approval_id TEXT NOT NULL,
          policy_id TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_persistent_grants_lookup
          ON persistent_grants(user_id, tool_name, scope_hint, revoked_at, expires_at);

        CREATE TABLE IF NOT EXISTS delegation_grants (
          delegation_id TEXT PRIMARY KEY,
          delegator_user_id TEXT NOT NULL,
          delegator_run_id TEXT,
          delegator_session_id TEXT,
          delegator_profile TEXT NOT NULL,
          delegator_channel TEXT NOT NULL,
          bound_excel_session_id TEXT NOT NULL,
          bound_relay_request_id TEXT NOT NULL,
          bound_workbook TEXT,
          tool_class_ceiling TEXT NOT NULL,
          args_predicate TEXT,
          window_seconds INTEGER NOT NULL,
          exclude_external_write_bypass INTEGER NOT NULL DEFAULT 1,
          created_at TEXT NOT NULL,
          expires_at TEXT,
          revoked_at TEXT,
          consumed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_delegation_grants_lookup
          ON delegation_grants(delegator_user_id, bound_excel_session_id, bound_relay_request_id);
        """
      )
      columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(approval_requests)").fetchall()}
      if "skill" not in columns:
        conn.execute("ALTER TABLE approval_requests ADD COLUMN skill TEXT")
      if "delegation_id" not in columns:
        conn.execute("ALTER TABLE approval_requests ADD COLUMN delegation_id TEXT")

  async def create(self, request: ApprovalRequest) -> ApprovalRequest:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          INSERT INTO approval_requests (
            approval_id, tool_call_id, parent_approval_id, approval_chain_id,
            delegation_id, request_id, session_id, run_id, user_id, profile, channel,
            tool_name, tool_class, tool_args_redacted, args_hash, reason,
            blast_radius_summary, state, state_version, requested_at, decided_at,
            expires_at, decider_id, decider_role, decision, decision_reason,
            required_decider_count, eligible_decider_count, votes_received_count,
            args_predicate, chain_trust_window_seconds, route_target, route_target_type,
            external_callback_id, policy_id, policy_version, policy_bundle_hash,
            persistent_grant_scope, tenant_id, model_id, model_version,
            system_prompt_hash, tool_schema_version, mcp_server_version, skill
          ) VALUES (
            :approval_id, :tool_call_id, :parent_approval_id, :approval_chain_id,
            :delegation_id, :request_id, :session_id, :run_id, :user_id, :profile, :channel,
            :tool_name, :tool_class, :tool_args_redacted, :args_hash, :reason,
            :blast_radius_summary, :state, :state_version, :requested_at, :decided_at,
            :expires_at, :decider_id, :decider_role, :decision, :decision_reason,
            :required_decider_count, :eligible_decider_count, :votes_received_count,
            :args_predicate, :chain_trust_window_seconds, :route_target, :route_target_type,
            :external_callback_id, :policy_id, :policy_version, :policy_bundle_hash,
            :persistent_grant_scope, :tenant_id, :model_id, :model_version,
            :system_prompt_hash, :tool_schema_version, :mcp_server_version, :skill
          )
          """,
          self._request_to_row(request),
        )
        conn.commit()
    await self._emit("request_created", request)
    return request

  async def get(self, approval_id: str) -> ApprovalRequest | None:
    with self._connection() as conn:
      row = conn.execute(
        "SELECT * FROM approval_requests WHERE approval_id = ?",
        (approval_id,),
      ).fetchone()
    return self._row_to_request(row) if row is not None else None

  async def get_by_tool_call_id(self, tool_call_id: str) -> ApprovalRequest | None:
    with self._connection() as conn:
      row = conn.execute(
        "SELECT * FROM approval_requests WHERE tool_call_id = ? ORDER BY requested_at DESC LIMIT 1",
        (tool_call_id,),
      ).fetchone()
    return self._row_to_request(row) if row is not None else None

  async def update_request(self, request: ApprovalRequest) -> ApprovalRequest:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          UPDATE approval_requests SET
            tool_args_redacted = :tool_args_redacted,
            args_hash = :args_hash,
            reason = :reason,
            blast_radius_summary = :blast_radius_summary,
            required_decider_count = :required_decider_count,
            eligible_decider_count = :eligible_decider_count,
            args_predicate = :args_predicate,
            chain_trust_window_seconds = :chain_trust_window_seconds,
            route_target = :route_target,
            route_target_type = :route_target_type,
            policy_id = :policy_id,
            policy_version = :policy_version,
            policy_bundle_hash = :policy_bundle_hash,
            persistent_grant_scope = :persistent_grant_scope
          WHERE approval_id = :approval_id
          """,
          self._request_to_row(request),
        )
        conn.commit()
    return request

  async def transition_state(
    self,
    approval_id: str,
    state: ApprovalState,
    *,
    expected_state_version: int | None = None,
    route_target_type: str | None = None,
    route_target: str | None = None,
    expires_at: datetime | None = None,
    decider_id: str | None = None,
    decider_role: str | None = None,
    decision: str | None = None,
    decision_reason: str | None = None,
  ) -> ApprovalRequest:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if current_row is None:
          raise KeyError(f"approval request not found: {approval_id}")
        current = self._row_to_request(current_row)
        if expected_state_version is not None and current.state_version != expected_state_version:
          raise RuntimeError("approval request state_version changed")
        decided_at = utc_now() if state in TERMINAL_STATES else current.decided_at
        terminal_decision = decision
        if terminal_decision is None and state in {"approved", "denied", "auto_approved", "auto_denied", "expired"}:
          terminal_decision = state
        updated = replace(
          current,
          state=state,
          state_version=current.state_version + 1,
          route_target_type=route_target_type if route_target_type is not None else current.route_target_type,
          route_target=route_target if route_target is not None else current.route_target,
          expires_at=expires_at if expires_at is not None else current.expires_at,
          decided_at=decided_at,
          decider_id=decider_id if decider_id is not None else current.decider_id,
          decider_role=decider_role if decider_role is not None else current.decider_role,
          decision=terminal_decision,  # type: ignore[arg-type]
          decision_reason=decision_reason if decision_reason is not None else current.decision_reason,
        )
        conn.execute(
          """
          UPDATE approval_requests SET
            state = :state,
            state_version = :state_version,
            route_target_type = :route_target_type,
            route_target = :route_target,
            expires_at = :expires_at,
            decided_at = :decided_at,
            decider_id = :decider_id,
            decider_role = :decider_role,
            decision = :decision,
            decision_reason = :decision_reason
          WHERE approval_id = :approval_id
          """,
          self._request_to_row(updated),
        )
        conn.commit()
    await self._emit(self._event_type_for_state(updated.state), updated)
    return updated

  async def record_vote(self, approval_id: str, vote: ApprovalVote) -> ApprovalRequest:
    emitted_vote = False
    terminal_event: str | None = None
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        current_row = conn.execute(
          "SELECT * FROM approval_requests WHERE approval_id = ?",
          (approval_id,),
        ).fetchone()
        if current_row is None:
          raise KeyError(f"approval request not found: {approval_id}")
        current = self._row_to_request(current_row)
        if current.state in TERMINAL_STATES:
          conn.commit()
          return current

        existing = conn.execute(
          "SELECT * FROM approval_votes WHERE approval_id = ? AND decider_id = ?",
          (approval_id, vote.decider_id),
        ).fetchone()
        if existing is None:
          conn.execute(
            """
            INSERT INTO approval_votes (
              vote_id, approval_id, decider_id, decider_role, decision,
              decision_reason, decided_at, external_callback_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
              vote.vote_id,
              vote.approval_id,
              vote.decider_id,
              vote.decider_role,
              vote.decision,
              vote.decision_reason,
              _dt_to_text(vote.decided_at),
              vote.external_callback_id,
            ),
          )
          emitted_vote = True

        counts = conn.execute(
          """
          SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN decision = 'approved' THEN 1 ELSE 0 END) AS approved_count,
            SUM(CASE WHEN decision = 'denied' THEN 1 ELSE 0 END) AS denied_count
          FROM approval_votes
          WHERE approval_id = ?
          """,
          (approval_id,),
        ).fetchone()
        total = int(counts["total"] or 0)
        approved_count = int(counts["approved_count"] or 0)
        denied_count = int(counts["denied_count"] or 0)
        terminal_state: ApprovalState | None = None
        if denied_count >= current.required_decider_count:
          terminal_state = "denied"
        elif approved_count >= current.required_decider_count:
          terminal_state = "approved"

        updated = replace(current, votes_received_count=total)
        if terminal_state is not None:
          updated = replace(
            updated,
            state=terminal_state,
            state_version=current.state_version + 1,
            decided_at=vote.decided_at,
            decider_id=vote.decider_id,
            decider_role=vote.decider_role,
            decision=terminal_state,
            decision_reason=vote.decision_reason,
          )
          terminal_event = terminal_state

        conn.execute(
          """
          UPDATE approval_requests SET
            state = :state,
            state_version = :state_version,
            votes_received_count = :votes_received_count,
            decided_at = :decided_at,
            decider_id = :decider_id,
            decider_role = :decider_role,
            decision = :decision,
            decision_reason = :decision_reason
          WHERE approval_id = :approval_id
          """,
          self._request_to_row(updated),
        )
        conn.commit()
    if emitted_vote:
      await self._emit("vote_recorded", updated, vote=vote)
    if terminal_event is not None:
      await self._emit(terminal_event, updated, vote=vote)
    return updated

  async def create_persistent_grant(self, grant: PersistentGrant) -> PersistentGrant:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          INSERT INTO persistent_grants (
            grant_id, user_id, tool_name, scope_hint, args_predicate,
            granted_at, expires_at, revoked_at, granted_via_approval_id, policy_id
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            grant.grant_id,
            grant.user_id,
            grant.tool_name,
            grant.scope_hint,
            _json_dumps(grant.args_predicate) if grant.args_predicate is not None else None,
            _dt_to_text(grant.granted_at),
            _dt_to_text(grant.expires_at),
            _dt_to_text(grant.revoked_at),
            grant.granted_via_approval_id,
            grant.policy_id,
          ),
        )
        conn.commit()
    await self._emit_grant(
      "persistent_grant_created",
      grant,
      request=await self.get(grant.granted_via_approval_id),
    )
    return grant

  async def find_persistent_grant(
    self,
    *,
    user_id: str,
    tool_name: str,
    scope_hint: str,
    now: datetime | None = None,
  ) -> PersistentGrant | None:
    now_text = _dt_to_text(now or utc_now())
    with self._connection() as conn:
      row = conn.execute(
        """
        SELECT * FROM persistent_grants
        WHERE user_id = ?
          AND tool_name = ?
          AND scope_hint = ?
          AND revoked_at IS NULL
          AND (expires_at IS NULL OR expires_at > ?)
        ORDER BY granted_at DESC
        LIMIT 1
        """,
        (user_id, tool_name, scope_hint, now_text),
      ).fetchone()
    return self._row_to_grant(row) if row is not None else None

  async def revoke_persistent_grant(self, grant_id: str, *, revoked_at: datetime | None = None) -> None:
    when = revoked_at or utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
          "SELECT * FROM persistent_grants WHERE grant_id = ?",
          (grant_id,),
        ).fetchone()
        conn.execute(
          "UPDATE persistent_grants SET revoked_at = ? WHERE grant_id = ?",
          (_dt_to_text(when), grant_id),
        )
        conn.commit()
    if row is not None:
      grant = self._row_to_grant(row)
      await self._emit_grant(
        "persistent_grant_revoked",
        grant,
        request=await self.get(grant.granted_via_approval_id),
      )

  async def create_delegation_grant(self, grant: DelegationGrant) -> DelegationGrant:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          INSERT INTO delegation_grants (
            delegation_id, delegator_user_id, delegator_run_id, delegator_session_id,
            delegator_profile, delegator_channel, bound_excel_session_id,
            bound_relay_request_id, bound_workbook, tool_class_ceiling,
            args_predicate, window_seconds, exclude_external_write_bypass,
            created_at, expires_at, revoked_at, consumed_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            grant.delegation_id,
            grant.delegator_user_id,
            grant.delegator_run_id,
            grant.delegator_session_id,
            grant.delegator_profile,
            grant.delegator_channel,
            grant.bound_excel_session_id,
            grant.bound_relay_request_id,
            grant.bound_workbook,
            _json_dumps(sorted(grant.tool_class_ceiling)),
            _json_dumps(grant.args_predicate) if grant.args_predicate is not None else None,
            grant.window_seconds,
            1 if grant.exclude_external_write_bypass else 0,
            _dt_to_text(grant.created_at),
            _dt_to_text(grant.expires_at),
            _dt_to_text(grant.revoked_at),
            _dt_to_text(grant.consumed_at),
          ),
        )
        conn.commit()
    return grant

  async def get_delegation_grant(self, delegation_id: str) -> DelegationGrant | None:
    with self._connection() as conn:
      row = conn.execute(
        "SELECT * FROM delegation_grants WHERE delegation_id = ?",
        (delegation_id,),
      ).fetchone()
    return self._row_to_delegation_grant(row) if row is not None else None

  async def claim_delegation_grant(
    self,
    *,
    delegation_id: str,
    bound_relay_request_id: str,
    bound_excel_session_id: str,
    now: datetime | None = None,
  ) -> DelegationGrant | None:
    now_value = now or utc_now()
    now_text = _dt_to_text(now_value)
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
          """
          UPDATE delegation_grants
          SET consumed_at = :now
          WHERE delegation_id = :delegation_id
            AND bound_relay_request_id = :rid
            AND bound_excel_session_id = :sid
            AND consumed_at IS NULL
            AND revoked_at IS NULL
            AND (expires_at IS NULL OR expires_at > :now)
          """,
          {
            "now": now_text,
            "delegation_id": delegation_id,
            "rid": bound_relay_request_id,
            "sid": bound_excel_session_id,
          },
        )
        if cursor.rowcount == 1:
          row = conn.execute(
            "SELECT * FROM delegation_grants WHERE delegation_id = ?",
            (delegation_id,),
          ).fetchone()
          conn.commit()
          return self._row_to_delegation_grant(row) if row is not None else None
        conn.commit()
    return None

  async def revoke_delegation_grant(self, delegation_id: str, *, revoked_at: datetime | None = None) -> None:
    when = revoked_at or utc_now()
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          "UPDATE delegation_grants SET revoked_at = ? WHERE delegation_id = ?",
          (_dt_to_text(when), delegation_id),
        )
        conn.commit()

  async def expire_pending(self, *, now: datetime | None = None) -> int:
    now_value = now or utc_now()
    expired: list[ApprovalRequest] = []
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
          """
          SELECT * FROM approval_requests
          WHERE state IN ('pending_user', 'routed_external')
            AND expires_at IS NOT NULL
            AND expires_at <= ?
          """,
          (_dt_to_text(now_value),),
        ).fetchall()
        for row in rows:
          current = self._row_to_request(row)
          updated = replace(
            current,
            state="expired",
            state_version=current.state_version + 1,
            decided_at=now_value,
            decision="expired",
          )
          conn.execute(
            """
            UPDATE approval_requests SET
              state = :state,
              state_version = :state_version,
              decided_at = :decided_at,
              decision = :decision
            WHERE approval_id = :approval_id
            """,
            self._request_to_row(updated),
          )
          expired.append(updated)
        conn.commit()
    for request in expired:
      await self._emit("expired", request)
    return len(expired)

  async def _emit(self, event_type: str, request: ApprovalRequest, **kwargs: Any) -> None:
    if self._audit_emitter is None:
      return
    emit = getattr(self._audit_emitter, "emit_audit_for_lifecycle_event", None)
    if emit is None:
      return
    kwargs.setdefault("skill", request.skill)
    await emit(event_type=event_type, request=request, raw_tool_args={}, **kwargs)

  async def _emit_grant(
    self,
    event_type: str,
    grant: PersistentGrant,
    *,
    request: ApprovalRequest | None = None,
  ) -> None:
    if self._audit_emitter is None:
      return
    emit = getattr(self._audit_emitter, "emit_grant_event", None)
    if emit is None:
      return
    await emit(event_type=event_type, grant=grant, request=request)

  @staticmethod
  def _event_type_for_state(state: str) -> str:
    return {
      "auto_approved": "auto_approved",
      "auto_denied": "auto_denied",
      "pending_user": "user_hold_started",
      "routed_external": "external_route_started",
      "approved": "approved",
      "denied": "denied",
      "expired": "expired",
    }.get(state, state)

  @staticmethod
  def _request_to_row(request: ApprovalRequest) -> dict[str, Any]:
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
      "tool_args_redacted": _json_dumps(request.tool_args_redacted),
      "args_hash": request.args_hash,
      "reason": request.reason,
      "blast_radius_summary": request.blast_radius_summary,
      "state": request.state,
      "state_version": request.state_version,
      "requested_at": _dt_to_text(request.requested_at),
      "decided_at": _dt_to_text(request.decided_at),
      "expires_at": _dt_to_text(request.expires_at),
      "decider_id": request.decider_id,
      "decider_role": request.decider_role,
      "decision": request.decision,
      "decision_reason": request.decision_reason,
      "required_decider_count": request.required_decider_count,
      "eligible_decider_count": request.eligible_decider_count,
      "votes_received_count": request.votes_received_count,
      "args_predicate": _json_dumps(request.args_predicate) if request.args_predicate is not None else None,
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
    }

  @staticmethod
  def _row_to_request(row: sqlite3.Row) -> ApprovalRequest:
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
      tool_args_redacted=_json_loads(row["tool_args_redacted"]) or {},
      args_hash=str(row["args_hash"]),
      reason=row["reason"],
      blast_radius_summary=str(row["blast_radius_summary"]),
      state=str(row["state"]),  # type: ignore[arg-type]
      state_version=int(row["state_version"] or 0),
      requested_at=_dt_from_text(row["requested_at"]) or utc_now(),
      decided_at=_dt_from_text(row["decided_at"]),
      expires_at=_dt_from_text(row["expires_at"]),
      decider_id=row["decider_id"],
      decider_role=row["decider_role"],
      decision=row["decision"],  # type: ignore[arg-type]
      decision_reason=row["decision_reason"],
      required_decider_count=int(row["required_decider_count"] or 1),
      eligible_decider_count=int(row["eligible_decider_count"] or 1),
      votes_received_count=int(row["votes_received_count"] or 0),
      args_predicate=_json_loads(row["args_predicate"]),
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
    )

  @staticmethod
  def _row_to_grant(row: sqlite3.Row) -> PersistentGrant:
    return PersistentGrant(
      grant_id=str(row["grant_id"]),
      user_id=str(row["user_id"]),
      tool_name=str(row["tool_name"]),
      scope_hint=str(row["scope_hint"]),
      args_predicate=_json_loads(row["args_predicate"]),
      granted_at=_dt_from_text(row["granted_at"]) or utc_now(),
      expires_at=_dt_from_text(row["expires_at"]),
      revoked_at=_dt_from_text(row["revoked_at"]),
      granted_via_approval_id=str(row["granted_via_approval_id"]),
      policy_id=str(row["policy_id"]),
    )

  @staticmethod
  def _row_to_delegation_grant(row: sqlite3.Row) -> DelegationGrant:
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
      tool_class_ceiling=frozenset(_json_loads(row["tool_class_ceiling"]) or []),  # type: ignore[arg-type]
      args_predicate=_json_loads(row["args_predicate"]),
      window_seconds=int(row["window_seconds"]),
      exclude_external_write_bypass=bool(row["exclude_external_write_bypass"]),
      created_at=_dt_from_text(row["created_at"]) or utc_now(),
      expires_at=_dt_from_text(row["expires_at"]),
      revoked_at=_dt_from_text(row["revoked_at"]),
      consumed_at=_dt_from_text(row["consumed_at"]),
    )


async def expire_pending_loop(store: ApprovalRequestStore, *, interval_seconds: float = 30.0) -> None:
  while True:
    await asyncio.sleep(interval_seconds)
    await store.expire_pending()
