"""Gateway-owned undo lifetimes, independent of short-lived authentication sessions.

Only an explicit conversation close or a successful restore permits reclamation.
Unknown tokens (including results not observed by this gateway) remain preserved.
"""
from __future__ import annotations

from contextlib import closing
import functools
import hashlib
import inspect
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .approval_continuation import RootContinueOutcome
from .session import AuthManager, GatewaySession, session_owner_user_id

log = logging.getLogger(__name__)


def conversation_id(
  session: GatewaySession | None,
  *,
  log_path: Path | None = None,
  session_id: str = "",
) -> str:
  """The durable log identity, with the runner's current log taking precedence."""
  path = log_path
  if path is None and session is not None:
    log = session.agent_session_log
    path = log.path if log is not None else session.resume_log_path
  return Path(path).stem if path is not None else (
    session.session_id if session is not None else session_id
  )


def owner_key(tenant_id: Any, owner_user_id: str) -> str:
  return json.dumps([tenant_id, owner_user_id])


def owner_id(session: GatewaySession) -> str:
  return owner_key(session.tenant_id, session_owner_user_id(session))


def restore_tokens(value: Any) -> set[str]:
  """Read token-bearing structured results, including MCP JSON text blocks."""
  tokens: set[str] = set()
  if isinstance(value, dict):
    for key, child in value.items():
      if key in {"restore_token", "undo_token"} and isinstance(child, str) and child:
        tokens.add(child)
      elif key == "_undo_meta" and isinstance(child, str) and child:
        tokens.add(child)
      else:
        tokens.update(restore_tokens(child))
  elif isinstance(value, (list, tuple)):
    for child in value:
      tokens.update(restore_tokens(child))
  elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
    try:
      tokens.update(restore_tokens(json.loads(value)))
    except (ValueError, TypeError):
      pass
  return tokens


class UndoTokenLifetimeRegistry:
  """Persist token holders and explicit endings; no TTL or count eviction."""

  def __init__(self, path: Path) -> None:
    self.path = path
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
      db.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
          owner_id TEXT NOT NULL, conversation_id TEXT NOT NULL,
          terminal INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (owner_id, conversation_id)
        );
        CREATE TABLE IF NOT EXISTS tokens (
          owner_id TEXT NOT NULL, token_hash TEXT NOT NULL,
          conversation_id TEXT NOT NULL, consumed INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (owner_id, token_hash, conversation_id)
        );
        CREATE TABLE IF NOT EXISTS sessions (
          owner_id TEXT NOT NULL, session_id TEXT NOT NULL,
          conversation_id TEXT NOT NULL,
          PRIMARY KEY (owner_id, session_id, conversation_id)
        );
      """)

  @staticmethod
  def _key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()

  def record_tool_result(
    self, *, owner: str, conversation: str, tool_name: str,
    tool_input: Any, result: Any, error: Any = None,
  ) -> None:
    issued = restore_tokens(result)
    successful = error is None and not (
      isinstance(result, dict)
      and (result.get("isError") is True or result.get("success") is False
           or result.get("restored") is False or result.get("error"))
    )
    restoring = tool_name.split("__")[-1].startswith(("restore_", "undo"))
    consumed = restore_tokens(tool_input) if restoring and successful else set()
    if not issued and not consumed:
      return
    with closing(sqlite3.connect(self.path)) as db, db:
      db.execute("INSERT OR IGNORE INTO conversations VALUES (?, ?, 0)", (owner, conversation))
      db.executemany(
        "INSERT OR IGNORE INTO tokens VALUES (?, ?, ?, "
        "COALESCE((SELECT MAX(consumed) FROM tokens WHERE owner_id = ? AND token_hash = ?), 0))",
        [(owner, self._key(token), conversation, owner, self._key(token)) for token in issued],
      )
      db.executemany(
        "UPDATE tokens SET consumed = 1 WHERE owner_id = ? AND token_hash = ?",
        [(owner, self._key(token)) for token in consumed],
      )

  def begin_close(self, *, owner: str, conversation: str) -> None:
    # Stop new attachments before awaiting cancellation, but preserve backups
    # until every live holder has actually released the conversation.
    with closing(sqlite3.connect(self.path)) as db, db:
      db.execute(
        "INSERT INTO conversations VALUES (?, ?, 2) "
        "ON CONFLICT(owner_id, conversation_id) DO UPDATE "
        "SET terminal = CASE WHEN terminal = 1 THEN 1 ELSE 2 END",
        (owner, conversation),
      )

  def close_conversation(self, *, owner: str, conversation: str) -> None:
    with closing(sqlite3.connect(self.path)) as db, db:
      db.execute(
        "INSERT INTO conversations VALUES (?, ?, 1) "
        "ON CONFLICT(owner_id, conversation_id) DO UPDATE SET terminal = 1",
        (owner, conversation),
      )

  def is_closed(self, *, owner: str, conversation: str) -> bool:
    with closing(sqlite3.connect(self.path)) as db:
      row = db.execute(
        "SELECT terminal FROM conversations WHERE owner_id = ? AND conversation_id = ?",
        (owner, conversation),
      ).fetchone()
    return bool(row and row[0])

  def attach_session(self, *, owner: str, session_id: str, conversation: str) -> None:
    """Remember which gateway sessions ran a conversation, so a parked approval
    (which only names its session) can be traced back after the session expires."""
    with closing(sqlite3.connect(self.path)) as db, db:
      db.execute(
        "INSERT OR IGNORE INTO sessions VALUES (?, ?, ?)", (owner, session_id, conversation),
      )

  def session_ids(self, *, owner: str, conversation: str) -> set[str]:
    with closing(sqlite3.connect(self.path)) as db:
      rows = db.execute(
        "SELECT session_id FROM sessions WHERE owner_id = ? AND conversation_id = ?",
        (owner, conversation),
      ).fetchall()
    return {row[0] for row in rows}

  def is_session_closed(self, *, owner: str, session_id: str) -> bool:
    with closing(sqlite3.connect(self.path)) as db:
      row = db.execute(
        "SELECT 1 FROM sessions s JOIN conversations c "
        "ON c.owner_id = s.owner_id AND c.conversation_id = s.conversation_id "
        "WHERE s.owner_id = ? AND s.session_id = ? AND c.terminal != 0 LIMIT 1",
        (owner, session_id),
      ).fetchone()
    return row is not None

  def fork_conversation(self, *, owner: str, source: str, target: str) -> bool:
    with closing(sqlite3.connect(self.path)) as db, db:
      if db.execute(
        "SELECT 1 FROM conversations WHERE owner_id = ? AND conversation_id = ?",
        (owner, target),
      ).fetchone():
        return True
      source_row = db.execute(
        "SELECT terminal FROM conversations WHERE owner_id = ? AND conversation_id = ?",
        (owner, source),
      ).fetchone()
      if source_row and source_row[0]:
        return False
      db.execute("INSERT OR IGNORE INTO conversations VALUES (?, ?, 0)", (owner, target))
      db.execute(
        "INSERT OR IGNORE INTO tokens SELECT owner_id, token_hash, ?, consumed "
        "FROM tokens WHERE owner_id = ? AND conversation_id = ?",
        (target, owner, source),
      )
    return True

  def statuses(self, tokens: list[str], *, owner: str, conversation: str | None = None) -> dict[str, str]:
    statuses: dict[str, str] = {}
    with closing(sqlite3.connect(self.path)) as db:
      for token in tokens:
        rows = db.execute(
          "SELECT t.consumed, c.terminal, t.conversation_id FROM tokens t "
          "JOIN conversations c USING (owner_id, conversation_id) "
          "WHERE t.owner_id = ? AND t.token_hash = ?",
          (owner, self._key(token)),
        ).fetchall()
        if not rows or (conversation is not None and not any(row[2] == conversation for row in rows)):
          status = "unknown"
        elif all(row[0] for row in rows):
          status = "consumed"
        elif any(not row[0] and row[1] != 1 for row in rows):
          status = "active"
        else:
          status = "terminal"
        statuses[token] = status
    return statuses


def observe_runner_results(runtime: Any, session: GatewaySession, registry: UndoTokenLifetimeRegistry) -> None:
  """Install at construction so ordinary, client and delegated results share it."""
  build_runner = runtime.build_runner
  if session.fork_source_log_path is not None:
    if not registry.fork_conversation(
      owner=owner_id(session), source=session.fork_source_log_path.stem,
      target=conversation_id(session),
    ):
      raise HTTPException(status_code=409, detail="Source conversation is closed")

  def build_with_lifetimes(*args: Any, **kwargs: Any) -> Any:
    runner = build_runner(*args, **kwargs)
    registry.attach_session(
      owner=owner_id(session), session_id=session.session_id,
      conversation=conversation_id(session),
    )
    # The product wraps the runner (e.g. UserScopedRunner) and the wrapper only
    # forwards reads; the hook the real runner calls lives on the innermost one.
    inner = runner
    while getattr(inner, "_runner", None) is not None or getattr(inner, "_inner", None) is not None:
      inner = getattr(inner, "_runner", None) or getattr(inner, "_inner")
    previous = getattr(inner, "_on_tool_result", None)

    async def on_result(ctx: Any) -> Any:
      try:
        registry.record_tool_result(
          owner=owner_id(session), conversation=conversation_id(session),
          tool_name=ctx.tool_name, tool_input=ctx.tool_input,
          result=ctx.result, error=ctx.error,
        )
      except Exception:
        log.warning("Undo token lifetime recording failed", exc_info=True)
      if previous is None:
        return None
      outcome = previous(ctx)
      return await outcome if inspect.isawaitable(outcome) else outcome

    inner._on_tool_result = on_result
    if inner is not runner:
      runner._on_tool_result = on_result
    return runner

  # Runtime factories receive started_at by signature inspection.
  runtime.build_runner = functools.wraps(build_runner)(build_with_lifetimes)


async def cancel_parked_approvals(
  store: Any, policy: Any, *, tenant_storage_id: str, session_ids: set[str],
) -> int:
  """Deny durable approvals still parked for a closed conversation's sessions.

  A parked approval outlives its live session and any other session of the
  owner can decide it; denying it here means no decision can resume the turn.
  """
  if store is None or not session_ids:
    return 0
  cancelled = 0
  for request in await store.list_parked_requests(tenant_storage_id):
    if request.session_id not in session_ids:
      continue
    identity = dict(
      expected_tool_call_id=request.tool_call_id, expected_user_id=request.user_id,
      expected_request_id=request.request_id, expected_run_id=request.run_id,
      expected_session_id=request.session_id, expected_channel=request.channel,
    )
    try:
      await store.fence_persistent_grants_for_cancellation(request.approval_id, **identity)
      updated, transitioned, _ = await store.terminalize_pending_for_cancellation(
        request.approval_id, **identity,
        decider_id=tenant_storage_id, decision_reason="conversation_closed",
      )
      await store.revoke_persistent_grants_for_approval(request.approval_id)
      if transitioned and policy is not None:
        await policy.on_resolve(request=updated)
      cancelled += int(transitioned)
    except Exception:
      log.warning(
        "Closing conversation could not cancel parked approval %s",
        request.approval_id, exc_info=True,
      )
  return cancelled


def guard_root_continue(hook: Any, app: Any, tenant_id: Any) -> Any:
  """Refuse to cold-restore a root turn whose conversation has been closed."""
  if hook is None or getattr(hook, "_undo_lifetime_guarded", False):
    return hook

  async def guarded(continue_request: Any) -> Any:
    store = getattr(app.state, "gateway_approval_store", None)
    record = await store.get(continue_request.approval_id) if store is not None else None
    if record is not None and app.state.undo_token_lifetimes.is_session_closed(
      owner=owner_key(tenant_id, continue_request.tenant_storage_id),
      session_id=record.session_id,
    ):
      return RootContinueOutcome("not_restorable")
    return await hook(continue_request)

  setattr(guarded, "_undo_lifetime_guarded", True)
  return guarded


class UndoStatusRequest(BaseModel):
  tokens: list[str]
  conversation_id: str | None = None


def create_undo_token_router(
  *, auth: AuthManager, registry: UndoTokenLifetimeRegistry, end_conversation: Any,
) -> APIRouter:
  router = APIRouter()

  def authenticate(request: Request) -> GatewaySession:
    token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
    return auth.verify_token(token)

  @router.post("/workbook-undo/status")
  async def undo_status(request: Request, payload: UndoStatusRequest) -> dict[str, Any]:
    session = authenticate(request)
    return {
      "conversation_id": conversation_id(session),
      "statuses": registry.statuses(payload.tokens, owner=owner_id(session), conversation=payload.conversation_id),
    }

  @router.post("/chat/close")
  async def close_chat(request: Request) -> dict[str, Any]:
    session = authenticate(request)
    conversation = conversation_id(session)
    await end_conversation(session, conversation)
    return {"conversation_id": conversation, "status": "terminal"}

  @router.delete("/chat/conversations/{conversation}")
  async def delete_conversation(request: Request, conversation: str) -> dict[str, Any]:
    session = authenticate(request)
    await end_conversation(session, conversation)
    return {"conversation_id": conversation, "status": "terminal"}

  return router
