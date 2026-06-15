from __future__ import annotations

import asyncio
import hashlib
import inspect
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Literal, Optional, Set

import jwt
from fastapi import HTTPException

from .events import DEFAULT_SCHEMA_VERSION
from .event_log import EventLog
from .session_event_history import SessionEventHistory

if TYPE_CHECKING:
  from .multi_user.billing import SessionUsageSummary


JWT_ALGORITHM = "HS256"
JWT_HS256_MIN_SECRET_BYTES = 32
OnSessionExpiry = Callable[["GatewaySession"], Awaitable[None] | None]
_RESERVED_USER_IDS = {"_default"}


def _normalize_required_user_id(user_id: str | None) -> str:
  normalized = str(user_id or "").strip()
  if not normalized:
    raise ValueError("user_id is required")
  if normalized in _RESERVED_USER_IDS:
    raise ValueError("user_id '_default' is reserved")
  return normalized


@dataclass
class StreamSubscriber:
  """One connected client reading a session's active turn stream."""

  subscriber_id: str
  connected_at: float
  last_sent_seq: int
  queue: asyncio.Queue
  client_label: str | None = None
  pump_task: asyncio.Task[Any] | None = None
  disconnect_reason: str | None = None


@dataclass
class SessionStream:
  """Live and grace-window state for the current chat turn."""

  event_log: EventLog
  runner_task: asyncio.Task[Any] | None
  subscribers: Dict[str, StreamSubscriber] = field(default_factory=dict)
  transcript_written_seqs: set[int] = field(default_factory=set)
  cleanup_handle: asyncio.TimerHandle | None = None

  @property
  def is_running(self) -> bool:
    return self.runner_task is not None and not self.runner_task.done()


@dataclass
class GatewaySession:
  """Mutable per-user runtime state.

  A session owns approval queues, approved tool types, loaded MCP servers, and
  code execution state in addition to the authentication metadata used by
  `AuthManager`.
  """

  session_id: str
  api_key_hash: str
  created_at: int
  expires_at: int
  user_id: str
  user_email: str | None = None
  risk_user_id: int = 0
  role: Literal["owner", "invite"] = "owner"
  kind: Literal["chat", "control"] = "chat"
  auth_config: dict[str, Any] | None = None
  channel: Optional[str] = None
  is_public: bool = False
  schema_version: int = DEFAULT_SCHEMA_VERSION
  stream_active: bool = False
  active_turn: Optional[SessionStream] = None
  cached_usage: SessionUsageSummary | None = None
  pending_tools: Dict[str, Dict] = field(default_factory=dict)
  approved_tool_types: Set[str] = field(default_factory=set)
  loaded_mcp_servers: Set[str] = field(default_factory=set)
  approval_queues: Dict[str, asyncio.Queue] = field(default_factory=dict)
  approval_store: Any | None = None
  approval_policy: Any | None = None
  approval_expire_pending_task: asyncio.Task[Any] | None = None
  tool_sequence: int = 0
  result_queue: Optional[asyncio.Queue] = None
  code_execution_work_dir: Optional[str] = None
  background_tasks: Dict[str, Any] = field(default_factory=dict)
  control_chat_tasks: Dict[str, asyncio.Task[Any]] = field(default_factory=dict)
  event_history: SessionEventHistory = field(default_factory=SessionEventHistory)
  initial_message: str = ""
  _expiring: bool = False


class SessionStore:
  """In-memory session registry with TTL-based cleanup."""

  def __init__(self, ttl: int = 3600) -> None:
    self.ttl = ttl
    self.sessions: Dict[str, GatewaySession] = {}
    self._on_expiry: OnSessionExpiry | None = None
    self._on_expiry_hooks: list[OnSessionExpiry] = []

  def set_on_expiry(self, hook: OnSessionExpiry) -> None:
    """Replace the session-expiry cleanup hook.

    Prefer ``add_on_expiry`` when composing multiple cleanup owners.
    """
    self._on_expiry = hook
    self._on_expiry_hooks = [hook]

  def add_on_expiry(self, hook: OnSessionExpiry) -> None:
    """Register an additional session-expiry cleanup hook."""
    self._on_expiry_hooks.append(hook)

  def create_session(
    self,
    api_key_hash: str,
    *,
    user_id: str,
    user_email: str | None = None,
    risk_user_id: int = 0,
    role: Literal["owner", "invite"] = "owner",
    kind: Literal["chat", "control"] = "chat",
    auth_config: dict[str, Any] | None = None,
    ttl_seconds: int | None = None,
    schema_version: int = DEFAULT_SCHEMA_VERSION,
  ) -> GatewaySession:
    now = int(time.time())
    ttl = self.ttl if ttl_seconds is None else int(ttl_seconds)
    session_id = f"sess_{uuid.uuid4().hex}"
    normalized_user_id = _normalize_required_user_id(user_id)
    session = GatewaySession(
      session_id=session_id,
      api_key_hash=api_key_hash,
      created_at=now,
      expires_at=now + ttl,
      user_id=normalized_user_id,
      user_email=user_email,
      risk_user_id=risk_user_id,
      role=role,
      kind=kind,
      auth_config=dict(auth_config) if auth_config is not None else None,
      schema_version=int(schema_version),
      result_queue=asyncio.Queue(),
    )
    self.sessions[session_id] = session
    return session

  def get_session(self, session_id: str) -> Optional[GatewaySession]:
    return self.sessions.get(session_id)

  def expire_session(self, session_id: str) -> None:
    session = self.sessions.get(session_id)
    if session is None or session._expiring:
      return
    session._expiring = True
    self.sessions.pop(session_id, None)
    if self._on_expiry_hooks or self._on_expiry is not None:
      try:
        loop = asyncio.get_running_loop()
      except RuntimeError:
        pass
      else:
        loop.create_task(self._safe_on_expiry(session))
        return
    self._cleanup_session_files(session)

  def cleanup_expired(self) -> None:
    now = int(time.time())
    expired_ids = [session_id for session_id, session in self.sessions.items() if session.expires_at <= now]
    for session_id in expired_ids:
      self.expire_session(session_id)

  async def expire_session_async(self, session_id: str) -> None:
    session = self.sessions.get(session_id)
    if session is None or session._expiring:
      return
    session._expiring = True
    self.sessions.pop(session_id, None)
    await self._safe_on_expiry(session)

  async def cleanup_expired_async(self) -> None:
    now = int(time.time())
    expired_ids = [session_id for session_id, session in self.sessions.items() if session.expires_at <= now]
    for session_id in expired_ids:
      await self.expire_session_async(session_id)

  async def _safe_on_expiry(self, session: GatewaySession) -> None:
    hooks = list(self._on_expiry_hooks)
    if not hooks and self._on_expiry is not None:
      hooks = [self._on_expiry]
    for hook in hooks:
      try:
        result = hook(session)
        if inspect.isawaitable(result):
          await result
      except Exception:
        pass
    self._cleanup_session_files(session)

  @staticmethod
  def _cleanup_session_files(session: GatewaySession) -> None:
    if session.code_execution_work_dir:
      shutil.rmtree(session.code_execution_work_dir, ignore_errors=True)
      session.code_execution_work_dir = None


class AuthManager:
  """Issue and verify JWT session tokens for the gateway HTTP API."""

  def __init__(self, secret: str, valid_keys: Set[str], session_store: SessionStore) -> None:
    if len(secret.encode("utf-8")) < JWT_HS256_MIN_SECRET_BYTES:
      raise ValueError(
        f"JWT signing secret must be at least {JWT_HS256_MIN_SECRET_BYTES} bytes for {JWT_ALGORITHM}"
      )
    self._secret = secret
    self._valid_keys = set(valid_keys)
    self.session_store = session_store

  @staticmethod
  def hash_api_key(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return digest[:16]

  @staticmethod
  def get_bearer_token(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
      raise HTTPException(status_code=401, detail="Missing Authorization header")
    return authorization.split(" ", 1)[1]

  def validate_api_key(self, api_key: str) -> None:
    if not api_key:
      raise HTTPException(status_code=401, detail="Missing API key")
    if self._valid_keys and api_key not in self._valid_keys:
      raise HTTPException(status_code=401, detail="Invalid API key")

  def issue_token(self, session: GatewaySession) -> str:
    payload = {
      "session_id": session.session_id,
      "api_key_hash": session.api_key_hash,
      "created_at": session.created_at,
      "expires_at": session.expires_at,
      "user_id": session.user_id,
      "user_email": session.user_email,
      "risk_user_id": session.risk_user_id,
      "role": session.role,
      "channel": session.channel,
      "is_public": session.is_public,
      "schema_version": session.schema_version,
    }
    return jwt.encode(payload, self._secret, algorithm=JWT_ALGORITHM)

  def _decode_token(self, token: str) -> dict[str, Any]:
    try:
      payload = jwt.decode(token, self._secret, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
      raise HTTPException(status_code=401, detail="Invalid session token") from exc

    session_id = payload.get("session_id")
    api_key_hash = payload.get("api_key_hash")
    expires_at = payload.get("expires_at")

    if not session_id or not api_key_hash or not expires_at:
      raise HTTPException(status_code=401, detail="Invalid session payload")

    now = int(time.time())
    if now >= int(expires_at):
      self.session_store.expire_session(session_id)
      raise HTTPException(status_code=401, detail="Session expired")

    session = self.session_store.get_session(session_id)
    if not session or session.api_key_hash != api_key_hash:
      raise HTTPException(status_code=401, detail="Unknown session")

    token_user_id = str(payload.get("user_id") or "").strip()
    if not token_user_id:
      raise HTTPException(status_code=401, detail="Invalid session user_id")
    try:
      token_user_id = _normalize_required_user_id(token_user_id)
    except ValueError as exc:
      raise HTTPException(status_code=401, detail=str(exc)) from exc
    if token_user_id != session.user_id:
      raise HTTPException(status_code=401, detail="Session user mismatch")
    token_user_email = payload.get("user_email")
    if token_user_email != session.user_email:
      raise HTTPException(status_code=401, detail="Session user email mismatch")

    token_risk_user_id = payload.get("risk_user_id", session.risk_user_id)
    try:
      token_risk_user_id = int(token_risk_user_id)
    except (TypeError, ValueError) as exc:
      raise HTTPException(status_code=401, detail="Invalid session risk_user_id") from exc
    if token_risk_user_id != session.risk_user_id:
      raise HTTPException(status_code=401, detail="Session risk user mismatch")

    token_role = payload.get("role", session.role)
    if token_role not in {"owner", "invite"}:
      raise HTTPException(status_code=401, detail="Invalid session role")
    if token_role != session.role:
      raise HTTPException(status_code=401, detail="Session role mismatch")

    channel = payload.get("channel")
    is_public = payload.get("is_public", False)
    payload["risk_user_id"] = token_risk_user_id
    payload["role"] = token_role
    payload["channel"] = channel if isinstance(channel, str) else None
    payload["is_public"] = is_public if isinstance(is_public, bool) else False
    session.channel = payload["channel"]
    session.is_public = payload["is_public"]
    token_schema_version = payload.get("schema_version", session.schema_version)
    try:
      token_schema_version = int(token_schema_version)
    except (TypeError, ValueError) as exc:
      raise HTTPException(status_code=401, detail="Invalid session schema_version") from exc
    if token_schema_version != session.schema_version:
      raise HTTPException(status_code=401, detail="Session schema version mismatch")
    payload["schema_version"] = token_schema_version

    return payload

  def verify_token_with_payload(self, token: str) -> tuple[GatewaySession, dict[str, Any]]:
    payload = self._decode_token(token)
    session_id = str(payload["session_id"])
    session = self.session_store.get_session(session_id)
    if session is None:
      raise HTTPException(status_code=401, detail="Unknown session")
    return session, payload

  def verify_token(self, token: str) -> GatewaySession:
    session, _payload = self.verify_token_with_payload(token)

    return session


__all__ = ["AuthManager", "GatewaySession", "SessionStore"]
