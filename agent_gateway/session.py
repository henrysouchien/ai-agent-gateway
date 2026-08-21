from __future__ import annotations

import asyncio
import hashlib
import inspect
import shutil
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, Literal, Mapping, Optional, Set

import jwt
from fastapi import HTTPException

from agent_workflow_contracts import WorkflowContentPage

from .capability_binding import (
  CAPABILITY_IDS,
  CapabilityId,
  CredentialHandle,
  CredentialPrincipal,
  ModelSelectionIntent,
)
from .events import DEFAULT_SCHEMA_VERSION
from .event_log import EventLog
from .mcp_activation import McpActivationFold
from .session_event_history import SessionEventHistory
from .session_capabilities import normalize_session_capabilities

if TYPE_CHECKING:
  from .multi_user.billing import SessionUsageSummary


JWT_ALGORITHM = "HS256"
JWT_HS256_MIN_SECRET_BYTES = 32
SESSION_EXPIRY_RETRY_SECONDS = 0.05
OnSessionExpiry = Callable[["GatewaySession"], Awaitable[None] | None]
SessionExpiryBlocker = Callable[["GatewaySession"], int]
_RESERVED_USER_IDS = {"_default"}
WorkflowOutputReader = Callable[[str, str, int], Awaitable[WorkflowContentPage]]


def _normalize_required_user_id(user_id: str | None) -> str:
  normalized = str(user_id or "").strip()
  if not normalized:
    raise ValueError("user_id is required")
  if normalized in _RESERVED_USER_IDS:
    raise ValueError("user_id '_default' is reserved")
  return normalized


def _normalize_optional_identity_value(value: str | None) -> str | None:
  normalized = value.strip() if isinstance(value, str) else value
  return normalized or None


def session_owner_user_id(session: "GatewaySession") -> str:
  """Return the canonical event-bus owner for a trusted session."""

  owner_user_id = str(
    getattr(session, "owner_user_id", None)
    or getattr(session, "user_id", None)
    or ""
  ).strip()
  if not owner_user_id:
    raise ValueError("session owner_user_id is required")
  return owner_user_id


def _normalize_required_tenant_id(tenant_id: str | None) -> str:
  normalized = str(tenant_id or "").strip()
  if not normalized:
    raise ValueError("tenant_id is required to bind session credentials")
  return normalized


def _session_credential_provider(
  session: "GatewaySession",
  provider: str | None,
) -> str:
  normalized = str(provider or "").strip().lower()
  configured = ""
  if isinstance(session.auth_config, dict):
    configured = str(session.auth_config.get("provider") or "").strip().lower()
  if normalized and configured and normalized != configured:
    raise ValueError("credential provider does not match session auth_config")
  if not normalized:
    normalized = configured
  if not normalized:
    raise ValueError("credential provider is required to bind session credentials")
  return normalized


def _session_actor_id(session: "GatewaySession") -> str:
  owner_user_id = getattr(session, "owner_user_id", None)
  user_id = getattr(session, "user_id", None)
  normalized = str(owner_user_id or user_id or "").strip()
  if not normalized:
    raise ValueError("session owner identity is required to bind user credentials")
  return normalized


def _normalize_session_aliases(
  *,
  owner_user_id: str,
  raw_user_id: str | None,
  user_slug: str | None,
  user_email: str | None,
  user_aliases: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
  aliases: list[str] = []

  def add(value: str | None) -> None:
    normalized = value.strip() if isinstance(value, str) else value
    if normalized and normalized not in aliases:
      aliases.append(normalized)

  add(owner_user_id)
  if user_aliases:
    for alias in user_aliases:
      add(str(alias))
  add(raw_user_id)
  add(user_slug)
  add(user_email.strip().lower() if isinstance(user_email, str) else user_email)
  return tuple(aliases)


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
  research_file_id: int | None = None
  research_file_activity_lease: Any | None = field(default=None, repr=False)
  runtime_owner: Any | None = field(default=None, repr=False)
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
  role: Literal["owner", "invite"] = "invite"
  capabilities: frozenset[str] = field(default_factory=frozenset)
  model_entitled_capabilities: frozenset[str] = field(default_factory=frozenset)
  model_entitled_keys: frozenset[str] = field(default_factory=frozenset)
  kind: Literal["chat", "control"] = "chat"
  auth_config: dict[str, Any] | None = field(default=None, repr=False)
  tenant_id: str | None = None
  session_credential_handle: CredentialHandle | None = None
  allow_service_for_interactive: bool = False
  capability_run_overrides: Mapping[CapabilityId, ModelSelectionIntent] = field(
    default_factory=lambda: MappingProxyType({})
  )
  channel: Optional[str] = None
  owner_user_id: str | None = None
  raw_user_id: str | None = None
  user_slug: str | None = None
  user_aliases: tuple[str, ...] = field(default_factory=tuple)
  identity_status: str | None = None
  is_public: bool = False
  schema_version: int = DEFAULT_SCHEMA_VERSION
  stream_active: bool = False
  active_turn: Optional[SessionStream] = None
  cached_usage: SessionUsageSummary | None = None
  pending_tools: Dict[str, Dict] = field(default_factory=dict)
  approved_tool_types: Set[str] = field(default_factory=set)
  # The single writer of MCP load state (T3-I12). The loaded-server set, the
  # deferred-tool set and the dispatcher allowlist are all derived from this
  # fold by `mcp_activation.derive_live_surface`; none of them is stored, so
  # none of them can disagree with another.
  mcp_activation_fold: McpActivationFold = field(default_factory=McpActivationFold)
  loaded_local_tools: Set[str] = field(default_factory=set)
  approval_queues: Dict[str, asyncio.Queue] = field(default_factory=dict)
  approval_store: Any | None = None
  approval_policy: Any | None = None
  max_budget_usd: float | None = None
  approval_expire_pending_task: asyncio.Task[Any] | None = None
  tool_sequence: int = 0
  result_queue: Optional[asyncio.Queue] = None
  code_execution_work_dir: Optional[str] = None
  background_tasks: Dict[str, Any] = field(default_factory=dict)
  control_chat_tasks: Dict[str, asyncio.Task[Any]] = field(default_factory=dict)
  event_history: SessionEventHistory = field(default_factory=SessionEventHistory)
  initial_message: str = ""
  dispatch_scope: dict[str, Any] | None = None
  stage_skill_route: dict[str, str] | None = None
  purpose: str | None = None
  learn_memory_nudge_turns: int = 0
  learn_skill_nudge_iters: int = 0
  learning_fork_ledger: Any | None = field(default=None, repr=False)
  learning_fork_registry: Any | None = field(default=None, repr=False)
  workflow_output_reader: WorkflowOutputReader | None = field(
    default=None,
    repr=False,
  )
  _commercial_dispatch_owner: object | None = field(default=None, repr=False)
  _capability_selections_bound: bool = field(default=False, repr=False)
  _expired: bool = field(default=False, repr=False)
  _expiry_hooks_complete: bool = field(default=False, repr=False)
  _final_expiry_hook_index: int = field(default=0, repr=False)
  _expiry_task: asyncio.Task[None] | None = field(default=None, repr=False)
  _expiry_retry_handle: asyncio.TimerHandle | None = field(
    default=None,
    repr=False,
  )
  _expiring: bool = False


def bind_session_capability_selections(
  session: GatewaySession,
  *,
  run_overrides: Mapping[CapabilityId, ModelSelectionIntent],
) -> None:
  """Freeze normalized role selections onto one gateway session exactly once."""

  if session._capability_selections_bound:
    raise ValueError("session capability selections are already bound")
  normalized = dict(run_overrides)
  for capability_id in normalized:
    if capability_id not in CAPABILITY_IDS:
      raise ValueError(f"unknown capability_id: {capability_id!r}")
    if capability_id in {"session.driver", "node.fork"}:
      raise ValueError(f"{capability_id} cannot be selected at session init")
  for capability_id, intent in normalized.items():
    if not isinstance(intent, ModelSelectionIntent):
      raise TypeError(
        f"{capability_id} model override must be stable-key selection intent"
      )
  session.capability_run_overrides = MappingProxyType(normalized)
  session._capability_selections_bound = True


def bind_session_credentials(
  session: GatewaySession,
  *,
  tenant_id: str,
  credential_principal: CredentialPrincipal,
  provider: str | None = None,
  allow_service_for_interactive: bool = False,
  handle_id: str | None = None,
) -> CredentialHandle:
  """Attach secret-free credential provenance to a gateway session.

  The principal must come from a trusted resolver or server-owned policy. It is
  never inferred from the credential payload, billing mode, or channel.
  """

  normalized_tenant_id = _normalize_required_tenant_id(tenant_id)
  if getattr(session, "session_credential_handle", None) is not None:
    raise ValueError("session credentials are already bound")
  existing_tenant_id = getattr(session, "tenant_id", None)
  if existing_tenant_id is not None and existing_tenant_id != normalized_tenant_id:
    raise ValueError("credential tenant does not match existing session tenant_id")
  actor_id = (
    _session_actor_id(session)
    if credential_principal == "user"
    else None
  )
  handle = CredentialHandle(
    handle_id=handle_id or f"cred_{uuid.uuid4().hex}",
    provider=_session_credential_provider(session, provider),
    principal=credential_principal,
    tenant_id=normalized_tenant_id,
    actor_id=actor_id,
  )
  session.tenant_id = normalized_tenant_id
  session.session_credential_handle = handle
  session.allow_service_for_interactive = bool(allow_service_for_interactive)
  return handle


def attach_session_credential_handle(
  session: GatewaySession,
  handle: CredentialHandle,
  *,
  allow_service_for_interactive: bool = False,
) -> None:
  """Attach an existing immutable handle, preserving its opaque identity."""

  if getattr(session, "session_credential_handle", None) is not None:
    raise ValueError("session credentials are already bound")
  provider = _session_credential_provider(session, None)
  if handle.provider != provider:
    raise ValueError("credential handle provider does not match session auth_config")
  existing_tenant_id = getattr(session, "tenant_id", None)
  if existing_tenant_id is not None and existing_tenant_id != handle.tenant_id:
    raise ValueError("credential handle tenant does not match session tenant_id")
  expected_actor = _session_actor_id(session)
  if handle.principal == "user" and handle.actor_id != expected_actor:
    raise ValueError("user credential handle actor does not match session owner")
  session.tenant_id = handle.tenant_id
  session.session_credential_handle = handle
  session.allow_service_for_interactive = bool(allow_service_for_interactive)


class SessionStore:
  """In-memory session registry with TTL-based cleanup."""

  def __init__(self, ttl: int = 3600) -> None:
    self.ttl = ttl
    self.sessions: Dict[str, GatewaySession] = {}
    self._session_kinds: Dict[str, Literal["chat", "control"]] = {}
    self._on_expiry: OnSessionExpiry | None = None
    self._on_expiry_hooks: list[OnSessionExpiry] = []
    self._on_final_expiry_hooks: list[OnSessionExpiry] = []
    self._expiry_blockers: list[SessionExpiryBlocker] = []

  def set_on_expiry(self, hook: OnSessionExpiry) -> None:
    """Replace the session-expiry cleanup hook.

    Prefer ``add_on_expiry`` when composing multiple cleanup owners.
    """
    self._on_expiry = hook
    self._on_expiry_hooks = [hook]

  def add_on_expiry(self, hook: OnSessionExpiry) -> None:
    """Register immediate cleanup run exactly once when expiry begins."""
    self._on_expiry_hooks.append(hook)

  def add_on_final_expiry(self, hook: OnSessionExpiry) -> None:
    """Register teardown deferred until every runtime owner is finished."""

    self._on_final_expiry_hooks.append(hook)

  def add_expiry_blocker(self, blocker: SessionExpiryBlocker) -> None:
    """Register a side-effect-free runtime-ownership count for expiry."""

    if not callable(blocker):
      raise TypeError("session expiry blocker must be callable")
    self._expiry_blockers.append(blocker)

  def create_session(
    self,
    api_key_hash: str,
    *,
    user_id: str,
    user_email: str | None = None,
    risk_user_id: int = 0,
    role: Literal["owner", "invite"] = "invite",
    capabilities: frozenset[str] | set[str] | tuple[str, ...] | list[str] = frozenset(),
    model_entitled_capabilities: frozenset[str] = frozenset(),
    model_entitled_keys: frozenset[str] = frozenset(),
    kind: Literal["chat", "control"] = "chat",
    auth_config: dict[str, Any] | None = None,
    ttl_seconds: int | None = None,
    schema_version: int = DEFAULT_SCHEMA_VERSION,
    owner_user_id: str | None = None,
    raw_user_id: str | None = None,
    user_slug: str | None = None,
    user_aliases: tuple[str, ...] | list[str] | None = None,
    identity_status: str | None = None,
    tenant_id: str | None = None,
    credential_principal: CredentialPrincipal | None = None,
    session_credential_handle: CredentialHandle | None = None,
    allow_service_for_interactive: bool = False,
  ) -> GatewaySession:
    now = int(time.time())
    ttl = self.ttl if ttl_seconds is None else int(ttl_seconds)
    session_id = f"sess_{uuid.uuid4().hex}"
    normalized_user_id = _normalize_required_user_id(user_id)
    try:
      normalized_risk_user_id = int(risk_user_id or 0)
    except (TypeError, ValueError):
      normalized_risk_user_id = 0
    normalized_owner_user_id = _normalize_optional_identity_value(owner_user_id)
    if normalized_owner_user_id is None:
      normalized_owner_user_id = str(normalized_risk_user_id) if normalized_risk_user_id > 0 else normalized_user_id
    normalized_raw_user_id = _normalize_optional_identity_value(raw_user_id) or normalized_user_id
    normalized_user_slug = _normalize_optional_identity_value(user_slug)
    if (
      normalized_user_slug is None
      and normalized_risk_user_id > 0
      and normalized_raw_user_id != normalized_owner_user_id
      and not normalized_raw_user_id.isdecimal()
    ):
      normalized_user_slug = normalized_raw_user_id
    normalized_identity_status = (
      _normalize_optional_identity_value(identity_status)
      or ("fallback_canonical" if normalized_risk_user_id > 0 else "legacy_user_id_fallback")
    )
    session = GatewaySession(
      session_id=session_id,
      api_key_hash=api_key_hash,
      created_at=now,
      expires_at=now + ttl,
      user_id=normalized_user_id,
      user_email=user_email,
      risk_user_id=normalized_risk_user_id,
      role=role,
      capabilities=normalize_session_capabilities(capabilities),
      model_entitled_capabilities=frozenset(model_entitled_capabilities),
      model_entitled_keys=frozenset(model_entitled_keys),
      kind=kind,
      auth_config=dict(auth_config) if auth_config is not None else None,
      owner_user_id=normalized_owner_user_id,
      raw_user_id=normalized_raw_user_id,
      user_slug=normalized_user_slug,
      user_aliases=_normalize_session_aliases(
        owner_user_id=normalized_owner_user_id,
        raw_user_id=normalized_raw_user_id,
        user_slug=normalized_user_slug,
        user_email=user_email,
        user_aliases=user_aliases,
      ),
      identity_status=normalized_identity_status,
      tenant_id=(
        _normalize_required_tenant_id(tenant_id)
        if tenant_id is not None
        else None
      ),
      allow_service_for_interactive=bool(allow_service_for_interactive),
      schema_version=int(schema_version),
      result_queue=asyncio.Queue(),
    )
    if credential_principal is not None and session_credential_handle is not None:
      raise ValueError(
        "provide credential_principal or session_credential_handle, not both"
      )
    if credential_principal is not None:
      bind_session_credentials(
        session,
        tenant_id=_normalize_required_tenant_id(tenant_id),
        credential_principal=credential_principal,
        allow_service_for_interactive=allow_service_for_interactive,
      )
    elif session_credential_handle is not None:
      if tenant_id is not None and session_credential_handle.tenant_id != session.tenant_id:
        raise ValueError("credential handle tenant does not match session tenant_id")
      attach_session_credential_handle(
        session,
        session_credential_handle,
        allow_service_for_interactive=allow_service_for_interactive,
      )
    self.sessions[session_id] = session
    self._session_kinds[session_id] = kind
    return session

  def get_session(self, session_id: str) -> Optional[GatewaySession]:
    session = self.sessions.get(session_id)
    if session is None or session._expired:
      return None
    return session

  def visible_sessions_snapshot(self) -> tuple[GatewaySession, ...]:
    """Return sessions still usable at one captured wall-clock instant."""

    now = int(time.time())
    return tuple(
      session
      for session in tuple(self.sessions.values())
      if not session._expired and session.expires_at > now
    )

  def expire_session(self, session_id: str) -> None:
    session = self.sessions.get(session_id)
    if session is None:
      return
    session._expired = True
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      if (
        self._expiry_blocked(session)
        or self._immediate_expiry_hooks()
        or self._on_final_expiry_hooks
      ):
        return
      session._expiring = True
      self.sessions.pop(session_id, None)
      self._session_kinds.pop(session_id, None)
      self._cleanup_session_files(session)
      return
    task = session._expiry_task
    if task is None or task.done():
      task = loop.create_task(
        self._expire_session_async(session),
        name=f"{session.session_id}:expiry",
      )
      session._expiry_task = task

  def cleanup_expired(self) -> None:
    now = int(time.time())
    expired_ids = [session_id for session_id, session in self.sessions.items() if session.expires_at <= now]
    for session_id in expired_ids:
      self.expire_session(session_id)

  async def expire_session_async(self, session_id: str) -> None:
    session = self.sessions.get(session_id)
    if session is None:
      return
    session._expired = True
    task = session._expiry_task
    if task is None or task.done():
      task = asyncio.create_task(
        self._expire_session_async(session),
        name=f"{session.session_id}:expiry",
      )
      session._expiry_task = task
    if task is not asyncio.current_task():
      await asyncio.shield(task)

  async def cleanup_expired_async(self) -> None:
    now = int(time.time())
    expired_ids = [session_id for session_id, session in self.sessions.items() if session.expires_at <= now]
    for session_id in expired_ids:
      await self.expire_session_async(session_id)

  async def _expire_session_async(self, session: GatewaySession) -> None:
    handle = session._expiry_retry_handle
    if handle is not None:
      handle.cancel()
      session._expiry_retry_handle = None
    if not session._expiry_hooks_complete:
      await self._safe_expiry_hooks(session, self._immediate_expiry_hooks())
      session._expiry_hooks_complete = True
    if self._expiry_blocked(session):
      self._schedule_expiry_retry(session)
      return
    session._expiring = True
    if not await self._run_final_expiry_hooks(session):
      self._schedule_expiry_retry(session)
      return
    self._cleanup_session_files(session)
    self.sessions.pop(session.session_id, None)
    self._session_kinds.pop(session.session_id, None)

  def _schedule_expiry_retry(self, session: GatewaySession) -> None:
    existing = session._expiry_retry_handle
    if existing is not None and not existing.cancelled():
      return
    loop = asyncio.get_running_loop()

    def retry() -> None:
      session._expiry_retry_handle = None
      if self.sessions.get(session.session_id) is session:
        self.expire_session(session.session_id)

    session._expiry_retry_handle = loop.call_later(
      SESSION_EXPIRY_RETRY_SECONDS,
      retry,
    )

  def _immediate_expiry_hooks(self) -> tuple[OnSessionExpiry, ...]:
    hooks = list(self._on_expiry_hooks)
    if not hooks and self._on_expiry is not None:
      hooks = [self._on_expiry]
    return tuple(hooks)

  async def _safe_expiry_hooks(
    self,
    session: GatewaySession,
    hooks: tuple[OnSessionExpiry, ...],
  ) -> None:
    for hook in hooks:
      try:
        result = hook(session)
        if inspect.isawaitable(result):
          await result
      except Exception:
        pass

  async def _run_final_expiry_hooks(self, session: GatewaySession) -> bool:
    """Advance retry-safe final teardown without skipping a failed hook."""

    hooks = tuple(self._on_final_expiry_hooks)
    while session._final_expiry_hook_index < len(hooks):
      hook = hooks[session._final_expiry_hook_index]
      try:
        result = hook(session)
        if inspect.isawaitable(result):
          await result
      except Exception:
        return False
      session._final_expiry_hook_index += 1
    return True

  def _expiry_blocked(self, session: GatewaySession) -> bool:
    for blocker in tuple(self._expiry_blockers):
      try:
        count = blocker(session)
      except Exception:
        return True
      if type(count) is not int or count < 0 or count != 0:
        return True
    return False

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
      "capabilities": sorted(session.capabilities),
      "model_entitled_capabilities": sorted(session.model_entitled_capabilities),
      "model_entitled_keys": sorted(session.model_entitled_keys),
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

    if "kind" in payload or "session_class" in payload:
      raise HTTPException(status_code=401, detail="Session class claims are forbidden")

    session_id = payload.get("session_id")
    api_key_hash = payload.get("api_key_hash")
    expires_at = payload.get("expires_at")

    required_claims = (
      "risk_user_id",
      "role",
      "capabilities",
      "model_entitled_capabilities",
      "model_entitled_keys",
      "channel",
      "is_public",
      "schema_version",
    )
    if (
      not session_id
      or not api_key_hash
      or not expires_at
      or any(claim not in payload for claim in required_claims)
    ):
      raise HTTPException(status_code=401, detail="Invalid session payload")

    now = int(time.time())
    if now >= int(expires_at):
      self.session_store.expire_session(session_id)
      raise HTTPException(status_code=401, detail="Session expired")

    session = self.session_store.get_session(session_id)
    if not session or session.api_key_hash != api_key_hash:
      raise HTTPException(status_code=401, detail="Unknown session")
    stored_kind = self.session_store._session_kinds.get(session_id)
    if (
      stored_kind not in {"chat", "control"}
      or session.kind != stored_kind
    ):
      raise HTTPException(status_code=401, detail="Session kind integrity check failed")

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

    token_risk_user_id = payload["risk_user_id"]
    try:
      token_risk_user_id = int(token_risk_user_id)
    except (TypeError, ValueError) as exc:
      raise HTTPException(status_code=401, detail="Invalid session risk_user_id") from exc
    if token_risk_user_id != session.risk_user_id:
      raise HTTPException(status_code=401, detail="Session risk user mismatch")

    token_role = payload["role"]
    if token_role not in {"owner", "invite"}:
      raise HTTPException(status_code=401, detail="Invalid session role")
    if token_role != session.role:
      raise HTTPException(status_code=401, detail="Session role mismatch")

    try:
      token_capabilities = normalize_session_capabilities(payload["capabilities"])
    except ValueError as exc:
      raise HTTPException(status_code=401, detail="Invalid session capabilities") from exc
    if token_capabilities != session.capabilities:
      raise HTTPException(status_code=401, detail="Session capabilities mismatch")
    token_model_capabilities = frozenset(
      str(value or "").strip()
      for value in payload["model_entitled_capabilities"]
      if str(value or "").strip()
    )
    token_model_keys = frozenset(
      str(value or "").strip()
      for value in payload["model_entitled_keys"]
      if str(value or "").strip()
    )
    if token_model_capabilities != session.model_entitled_capabilities:
      raise HTTPException(status_code=401, detail="Session model capability entitlements mismatch")
    if token_model_keys != session.model_entitled_keys:
      raise HTTPException(status_code=401, detail="Session model key entitlements mismatch")

    channel = payload["channel"]
    is_public = payload["is_public"]
    payload["risk_user_id"] = token_risk_user_id
    payload["role"] = token_role
    payload["capabilities"] = sorted(token_capabilities)
    payload["model_entitled_capabilities"] = sorted(token_model_capabilities)
    payload["model_entitled_keys"] = sorted(token_model_keys)
    payload["channel"] = channel if isinstance(channel, str) else None
    payload["is_public"] = is_public if isinstance(is_public, bool) else False
    session.channel = payload["channel"]
    session.is_public = payload["is_public"]
    token_schema_version = payload["schema_version"]
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


__all__ = [
  "AuthManager",
  "GatewaySession",
  "SessionStore",
  "bind_session_capability_selections",
  "session_owner_user_id",
]
