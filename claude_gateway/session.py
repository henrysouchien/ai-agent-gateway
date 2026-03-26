from __future__ import annotations

import asyncio
import hashlib
import shutil
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Set

import jwt
from fastapi import HTTPException


JWT_ALGORITHM = "HS256"
OnSessionExpiry = Callable[["Session"], Awaitable[None]]


@dataclass
class Session:
  session_id: str
  api_key_hash: str
  created_at: int
  expires_at: int
  stream_active: bool = False
  pending_tools: Dict[str, Dict] = field(default_factory=dict)
  approved_tool_types: Set[str] = field(default_factory=set)
  loaded_mcp_servers: Set[str] = field(default_factory=set)
  approval_queues: Dict[str, asyncio.Queue] = field(default_factory=dict)
  tool_sequence: int = 0
  result_queue: Optional[asyncio.Queue] = None
  code_execution_work_dir: Optional[str] = None
  background_tasks: Dict[str, Any] = field(default_factory=dict)
  _expiring: bool = False


class SessionStore:
  def __init__(self, ttl: int = 3600) -> None:
    self.ttl = ttl
    self.sessions: Dict[str, Session] = {}
    self._on_expiry: OnSessionExpiry | None = None

  def set_on_expiry(self, hook: OnSessionExpiry) -> None:
    self._on_expiry = hook

  def create_session(self, api_key_hash: str) -> Session:
    now = int(time.time())
    session_id = f"sess_{uuid.uuid4().hex}"
    session = Session(
      session_id=session_id,
      api_key_hash=api_key_hash,
      created_at=now,
      expires_at=now + self.ttl,
      result_queue=asyncio.Queue(),
    )
    self.sessions[session_id] = session
    return session

  def get_session(self, session_id: str) -> Optional[Session]:
    return self.sessions.get(session_id)

  def expire_session(self, session_id: str) -> None:
    session = self.sessions.get(session_id)
    if session is None or session._expiring:
      return
    session._expiring = True
    self.sessions.pop(session_id, None)
    if self._on_expiry is not None:
      try:
        loop = asyncio.get_running_loop()
      except RuntimeError:
        pass
      else:
        loop.create_task(self._safe_on_expiry(session))
        return
    if session.code_execution_work_dir:
      shutil.rmtree(session.code_execution_work_dir, ignore_errors=True)

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
    if self._on_expiry is not None:
      await self._safe_on_expiry(session)
      return
    if session.code_execution_work_dir:
      shutil.rmtree(session.code_execution_work_dir, ignore_errors=True)

  async def cleanup_expired_async(self) -> None:
    now = int(time.time())
    expired_ids = [session_id for session_id, session in self.sessions.items() if session.expires_at <= now]
    for session_id in expired_ids:
      await self.expire_session_async(session_id)

  async def _safe_on_expiry(self, session: Session) -> None:
    if self._on_expiry is None:
      return
    try:
      await self._on_expiry(session)
    except Exception:
      pass


class AuthManager:
  def __init__(self, secret: str, valid_keys: Set[str], session_store: SessionStore) -> None:
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

  def issue_token(self, session: Session) -> str:
    payload = {
      "session_id": session.session_id,
      "api_key_hash": session.api_key_hash,
      "created_at": session.created_at,
      "expires_at": session.expires_at,
    }
    return jwt.encode(payload, self._secret, algorithm=JWT_ALGORITHM)

  def verify_token(self, token: str) -> Session:
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

    return session


__all__ = ["AuthManager", "Session", "SessionStore"]
