from __future__ import annotations

import asyncio
from contextlib import contextmanager
import os
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Protocol

from . import approval_store_rows as _rows
from .approval_notifications import (
  ApprovalNotificationDestinationResolver,
  ApprovalNotificationSender,
  approval_notification_policy_for_request,
  maybe_await,
  normalize_approval_notification_destinations,
  render_approval_notification_message,
)
from .approval_policy import (
  ApprovalRequest,
  ApprovalState,
  ApprovalVote,
  DelegationGrant,
  PersistentGrant,
  utc_now,
)


TERMINAL_STATES = frozenset({"auto_approved", "auto_denied", "approved", "denied", "expired"})


_dt_to_text = _rows.dt_to_text
_dt_from_text = _rows.dt_from_text
_json_dumps = _rows.json_dumps
_json_loads = _rows.json_loads


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

  def __init__(
    self,
    path: str | os.PathLike[str] = "data/gateway/approvals.sqlite3",
    *,
    audit_emitter: Any | None = None,
    notification_destination_resolver: ApprovalNotificationDestinationResolver | None = None,
    notification_sender: ApprovalNotificationSender | None = None,
  ) -> None:
    self.path = Path(path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._audit_emitter = audit_emitter
    self._notification_destination_resolver = notification_destination_resolver
    self._notification_sender = notification_sender
    self._notification_delivery_task: asyncio.Task | None = None
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
          skill TEXT,
          notification_policy TEXT NOT NULL DEFAULT 'auto'
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

        CREATE TABLE IF NOT EXISTS approval_notification_outbox (
          approval_id TEXT NOT NULL REFERENCES approval_requests(approval_id) ON DELETE CASCADE,
          channel TEXT NOT NULL,
          destination TEXT NOT NULL,
          state TEXT NOT NULL,
          message TEXT NOT NULL,
          dedupe_key TEXT NOT NULL,
          attempt_count INTEGER NOT NULL DEFAULT 0,
          last_error TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          sent_at TEXT,
          PRIMARY KEY (approval_id, channel, destination)
        );
        CREATE INDEX IF NOT EXISTS idx_approval_notification_outbox_state
          ON approval_notification_outbox(state, updated_at);
        """
      )
      columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(approval_requests)").fetchall()}
      if "skill" not in columns:
        conn.execute("ALTER TABLE approval_requests ADD COLUMN skill TEXT")
      if "delegation_id" not in columns:
        conn.execute("ALTER TABLE approval_requests ADD COLUMN delegation_id TEXT")
      if "notification_policy" not in columns:
        conn.execute("ALTER TABLE approval_requests ADD COLUMN notification_policy TEXT NOT NULL DEFAULT 'auto'")

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
            system_prompt_hash, tool_schema_version, mcp_server_version, skill,
            notification_policy
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
            :system_prompt_hash, :tool_schema_version, :mcp_server_version, :skill,
            :notification_policy
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
    return self._row_to_request_with_projection(row) if row is not None else None

  async def get_by_tool_call_id(self, tool_call_id: str) -> ApprovalRequest | None:
    with self._connection() as conn:
      row = conn.execute(
        "SELECT * FROM approval_requests WHERE tool_call_id = ? ORDER BY requested_at DESC LIMIT 1",
        (tool_call_id,),
      ).fetchone()
    return self._row_to_request_with_projection(row) if row is not None else None

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
            persistent_grant_scope = :persistent_grant_scope,
            notification_policy = :notification_policy
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

  async def enqueue_pending_approval_notification(self, request: ApprovalRequest) -> dict[str, Any] | None:
    if request.state != "pending_user":
      return None
    now_text = _dt_to_text(utc_now())
    if approval_notification_policy_for_request(request) == "skip":
      async with self._lock:
        with self._connection() as conn:
          conn.execute("BEGIN IMMEDIATE")
          self._insert_notification_row(
            conn,
            approval_id=request.approval_id,
            channel="",
            destination="",
            state="skipped_policy",
            message="",
            now_text=now_text,
          )
          projection = self._notification_projection_for_conn(conn, request.approval_id)
          conn.commit()
      return projection

    destinations = await self._resolve_notification_destinations(request)
    if not destinations:
      async with self._lock:
        with self._connection() as conn:
          conn.execute("BEGIN IMMEDIATE")
          self._insert_notification_row(
            conn,
            approval_id=request.approval_id,
            channel="",
            destination="",
            state="skipped_no_destination",
            message="",
            now_text=now_text,
          )
          projection = self._notification_projection_for_conn(conn, request.approval_id)
          conn.commit()
      return projection

    message = render_approval_notification_message(request)
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for destination in destinations:
          self._insert_notification_row(
            conn,
            approval_id=request.approval_id,
            channel=destination.channel,
            destination=destination.destination,
            state="pending",
            message=message,
            now_text=now_text,
          )
        projection = self._notification_projection_for_conn(conn, request.approval_id)
        conn.commit()
    return projection

  async def deliver_pending_approval_notifications(self, *, limit: int = 50) -> int:
    if self._notification_sender is None:
      return 0
    rows = await self._claim_pending_notification_rows(limit=max(1, int(limit)))
    delivered_or_failed = 0
    for row in rows:
      now_text = _dt_to_text(utc_now())
      try:
        await maybe_await(self._notification_sender(row))
      except Exception as exc:
        await self._mark_notification_failed(row, type(exc).__name__, now_text=now_text)
      else:
        await self._mark_notification_sent(row, now_text=now_text)
      delivered_or_failed += 1
    return delivered_or_failed

  def schedule_approval_notification_delivery(self, *, limit: int = 50) -> bool:
    if self._notification_sender is None:
      return False
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      return False
    if self._notification_delivery_task is not None and not self._notification_delivery_task.done():
      return False
    self._notification_delivery_task = loop.create_task(
      self.deliver_pending_approval_notifications(limit=limit)
    )
    return True

  async def get_approval_notification_projection(self, approval_id: str) -> dict[str, Any] | None:
    with self._connection() as conn:
      return self._notification_projection_for_conn(conn, approval_id)

  async def retry_failed_approval_notifications(self, approval_id: str) -> dict[str, Any]:
    now_text = _dt_to_text(utc_now())
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
          """
          UPDATE approval_notification_outbox
          SET state = 'pending',
              updated_at = ?,
              last_error = NULL
          WHERE approval_id = ?
            AND state = 'failed_retryable'
            AND EXISTS (
              SELECT 1
              FROM approval_requests
              WHERE approval_requests.approval_id = approval_notification_outbox.approval_id
                AND approval_requests.state = 'pending_user'
            )
          """,
          (now_text, approval_id),
        )
        projection = self._notification_projection_for_conn(conn, approval_id)
        conn.commit()
    return {
      "approval_id": approval_id,
      "requeued": int(cursor.rowcount or 0),
      "notification": projection,
    }

  async def list_approval_notification_outbox(self, approval_id: str | None = None) -> list[dict[str, Any]]:
    with self._connection() as conn:
      if approval_id is None:
        rows = conn.execute(
          "SELECT * FROM approval_notification_outbox ORDER BY created_at ASC"
        ).fetchall()
      else:
        rows = conn.execute(
          "SELECT * FROM approval_notification_outbox WHERE approval_id = ? ORDER BY created_at ASC",
          (approval_id,),
        ).fetchall()
    return [dict(row) for row in rows]

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

  async def _resolve_notification_destinations(self, request: ApprovalRequest):
    if self._notification_destination_resolver is None:
      return []
    raw_destinations = await maybe_await(self._notification_destination_resolver(request))
    return normalize_approval_notification_destinations(raw_destinations or [])

  async def _claim_pending_notification_rows(self, *, limit: int) -> list[dict[str, Any]]:
    now_text = _dt_to_text(utc_now())
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = [
          dict(row)
          for row in conn.execute(
            """
            SELECT outbox.*
            FROM approval_notification_outbox AS outbox
            JOIN approval_requests AS approvals
              ON approvals.approval_id = outbox.approval_id
            WHERE outbox.state = 'pending'
              AND approvals.state = 'pending_user'
            ORDER BY outbox.created_at ASC
            LIMIT ?
            """,
            (limit,),
          ).fetchall()
        ]
        for row in rows:
          conn.execute(
            """
            UPDATE approval_notification_outbox
            SET state = 'processing',
                updated_at = ?,
                attempt_count = attempt_count + 1
            WHERE approval_id = ?
              AND channel = ?
              AND destination = ?
              AND state = 'pending'
            """,
            (now_text, row["approval_id"], row["channel"], row["destination"]),
          )
        conn.commit()
    return rows

  @staticmethod
  def _insert_notification_row(
    conn: sqlite3.Connection,
    *,
    approval_id: str,
    channel: str,
    destination: str,
    state: str,
    message: str,
    now_text: str | None,
  ) -> None:
    conn.execute(
      """
      INSERT OR IGNORE INTO approval_notification_outbox (
        approval_id, channel, destination, state, message, dedupe_key,
        attempt_count, last_error, created_at, updated_at, sent_at
      ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, NULL)
      """,
      (
        approval_id,
        channel,
        destination,
        state,
        message,
        f"approval:{approval_id}:{channel}:{destination}",
        now_text,
        now_text,
      ),
    )

  async def _mark_notification_sent(self, row: dict[str, Any], *, now_text: str | None) -> None:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          UPDATE approval_notification_outbox
          SET state = 'sent',
              updated_at = ?,
              sent_at = ?,
              last_error = NULL
          WHERE approval_id = ?
            AND channel = ?
            AND destination = ?
            AND state IN ('pending', 'processing')
          """,
          (now_text, now_text, row["approval_id"], row["channel"], row["destination"]),
        )
        conn.commit()

  async def _mark_notification_failed(self, row: dict[str, Any], error: str, *, now_text: str | None) -> None:
    async with self._lock:
      with self._connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
          """
          UPDATE approval_notification_outbox
          SET state = 'failed_retryable',
              updated_at = ?,
              last_error = ?
          WHERE approval_id = ?
            AND channel = ?
            AND destination = ?
            AND state IN ('pending', 'processing')
          """,
          (now_text, error[:500], row["approval_id"], row["channel"], row["destination"]),
        )
        conn.commit()

  def _row_to_request_with_projection(self, row: sqlite3.Row) -> ApprovalRequest:
    request = self._row_to_request(row)
    with self._connection() as conn:
      notification = self._notification_projection_for_conn(conn, request.approval_id)
    if notification is None:
      return request
    return replace(request, notification=notification)

  @staticmethod
  def _notification_projection_for_conn(conn: sqlite3.Connection, approval_id: str) -> dict[str, Any] | None:
    rows = conn.execute(
      """
      SELECT state, channel, sent_at
      FROM approval_notification_outbox
      WHERE approval_id = ?
      """,
      (approval_id,),
    ).fetchall()
    if not rows:
      return None
    states = {"pending" if str(row["state"]) == "processing" else str(row["state"]) for row in rows}
    state_order = [
      "sent",
      "failed_retryable",
      "failed_terminal",
      "pending",
      "skipped_no_destination",
      "skipped_policy",
    ]
    state = next((candidate for candidate in state_order if candidate in states), None)
    if state is None:
      return None
    channels: list[str] = []
    for row in rows:
      channel = str(row["channel"] or "")
      if not channel or channel not in {"telegram", "email", "push"}:
        continue
      row_state = "pending" if row["state"] == "processing" else row["state"]
      if state == "sent" and row_state != "sent":
        continue
      if state != "sent" and row_state != state:
        continue
      if channel not in channels:
        channels.append(channel)
    projection: dict[str, Any] = {"state": state, "channels": channels}
    sent_values = sorted(str(row["sent_at"]) for row in rows if row["sent_at"])
    if sent_values:
      projection["last_sent_at"] = sent_values[-1]
    return projection

  @staticmethod
  def _request_to_row(request: ApprovalRequest) -> dict[str, Any]:
    return _rows.request_to_row(request)

  @staticmethod
  def _row_to_request(row: sqlite3.Row) -> ApprovalRequest:
    return _rows.row_to_request(row)

  @staticmethod
  def _row_to_grant(row: sqlite3.Row) -> PersistentGrant:
    return _rows.row_to_grant(row)

  @staticmethod
  def _row_to_delegation_grant(row: sqlite3.Row) -> DelegationGrant:
    return _rows.row_to_delegation_grant(row)


async def expire_pending_loop(store: ApprovalRequestStore, *, interval_seconds: float = 30.0) -> None:
  while True:
    await asyncio.sleep(interval_seconds)
    await store.expire_pending()
