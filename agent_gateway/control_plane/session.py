from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from agent_gateway.auth import (
  ChannelMismatchError,
  CredentialsResolver,
  CredentialsTimeoutError,
  MissingUserIdError,
)
from agent_gateway.session import AuthManager


CONTROL_SESSION_TTL_SECONDS = 900


class ControlSessionRequest(BaseModel):
  api_key: str = Field(..., min_length=1)
  user_id: str | None = None
  user_email: str | None = None
  context: dict[str, Any] = Field(default_factory=dict)


class ControlSessionIdentityResponse(BaseModel):
  owner_user_id: str
  user_slug: str | None = None
  aliases: list[str] = Field(default_factory=list)
  identity_status: str


class ControlSessionResponse(BaseModel):
  session_token: str
  session_id: str
  expires_at: int
  kind: Literal["control"]
  user_id: str
  risk_user_id: int = 0
  user_slug: str | None = None
  user_email: str | None = None
  channel: str | None = None
  identity: ControlSessionIdentityResponse


def _normalize_user_id(user_id: str | None) -> str | None:
  normalized = user_id.strip() if isinstance(user_id, str) else user_id
  if normalized == "":
    return None
  if normalized == "_default":
    raise MissingUserIdError("user_id '_default' is reserved; supply a stable end-user id.")
  return normalized


def _error_response(
  exc: Exception,
  *,
  user_id: str | None = None,
  timeout_seconds: float | None = None,
) -> JSONResponse:
  if isinstance(exc, CredentialsTimeoutError):
    payload: dict[str, Any] = {"error": "credentials_timeout", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    if timeout_seconds is not None:
      payload["timeout_seconds"] = timeout_seconds
    return JSONResponse(payload, status_code=504)

  if isinstance(exc, MissingUserIdError):
    payload = {"error": "missing_user_id", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    return JSONResponse(payload, status_code=400)

  if isinstance(exc, ChannelMismatchError):
    payload = {"error": "channel_mismatch", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    return JSONResponse(payload, status_code=401)

  status_code = getattr(exc, "status_code", 500)
  detail = getattr(exc, "detail", None)
  payload = {
    "error": "auth_failed" if status_code < 500 else "credentials_unavailable",
    "message": str(detail) if detail is not None else str(exc),
  }
  if user_id is not None:
    payload["user_id"] = user_id
  return JSONResponse(payload, status_code=status_code)


def _claimed_channel(payload: ControlSessionRequest) -> str | None:
  raw = payload.context.get("channel") if isinstance(payload.context, dict) else None
  if isinstance(raw, str) and raw.strip():
    return raw.strip().lower()
  return None


def _normalize_channel(channel: str | None) -> str | None:
  normalized = channel.strip().lower() if isinstance(channel, str) else channel
  if normalized == "":
    return None
  return normalized


def _resolver_contract_error(message: str, *, user_id: str | None) -> JSONResponse:
  payload: dict[str, Any] = {"error": "credential_resolver_invalid", "message": message}
  if user_id is not None:
    payload["user_id"] = user_id
  return JSONResponse(payload, status_code=400)


def _user_identity_api() -> Any | None:
  api_dir = Path(__file__).resolve().parents[4] / "api"
  if api_dir.exists() and str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))
  try:
    return importlib.import_module("user_identity")
  except ModuleNotFoundError as exc:
    if exc.name not in {"user_identity", "api"}:
      raise
    try:
      return importlib.import_module("api.user_identity")
    except ModuleNotFoundError as nested_exc:
      if nested_exc.name not in {"user_identity", "api"}:
        raise
      return None


def _fallback_control_identity(
  *,
  user_id: str,
  risk_user_id: int,
  user_email: str | None,
  role: str | None,
  channel: str | None,
) -> Any:
  owner_user_id = str(risk_user_id) if risk_user_id > 0 else user_id
  user_slug = user_id if risk_user_id > 0 and user_id != owner_user_id and not user_id.isdecimal() else None
  aliases = [owner_user_id]
  if user_slug is not None:
    aliases.append(user_slug)
  if user_email:
    aliases.append(user_email.strip().lower())
  return SimpleNamespace(
    owner_user_id=owner_user_id,
    user_slug=user_slug,
    risk_user_id=risk_user_id,
    user_email=user_email,
    aliases=tuple(aliases),
    raw_user_id=user_id,
    role=role,
    channel=channel,
    identity_status="fallback_canonical" if risk_user_id > 0 else "legacy_user_id_fallback",
  )


def _resolve_control_identity(
  *,
  user_id: str,
  risk_user_id: int,
  user_email: str | None,
  role: str | None,
  channel: str | None,
) -> Any:
  api = _user_identity_api()
  if api is None:
    return _fallback_control_identity(
      user_id=user_id,
      risk_user_id=risk_user_id,
      user_email=user_email,
      role=role,
      channel=channel,
    )
  resolver = getattr(api, "resolve_canonical_user_identity", None)
  if not callable(resolver):
    return _fallback_control_identity(
      user_id=user_id,
      risk_user_id=risk_user_id,
      user_email=user_email,
      role=role,
      channel=channel,
    )
  return resolver(
    user_id,
    risk_user_id=risk_user_id,
    user_email=user_email,
    role=role,
    channel=channel,
  )


def _identity_response(identity: Any) -> ControlSessionIdentityResponse:
  return ControlSessionIdentityResponse(
    owner_user_id=str(identity.owner_user_id),
    user_slug=identity.user_slug,
    aliases=[str(alias) for alias in identity.aliases],
    identity_status=str(identity.identity_status),
  )


def build_session_router(
  *,
  auth: AuthManager,
  credentials_resolver: CredentialsResolver | None,
  resolver_timeout_seconds: float,
  approval_store: Any | None = None,
  approval_policy: Any | None = None,
) -> APIRouter:
  router = APIRouter()

  @router.post("/session", response_model=ControlSessionResponse)
  async def control_session(payload: ControlSessionRequest) -> ControlSessionResponse | JSONResponse:
    auth.validate_api_key(payload.api_key)
    try:
      resolved_user_id = _normalize_user_id(payload.user_id)
    except MissingUserIdError as exc:
      return _error_response(exc)

    resolved_user_email = payload.user_email.strip() if isinstance(payload.user_email, str) else payload.user_email
    if resolved_user_email == "":
      resolved_user_email = None

    resolved_auth_config: dict[str, Any] | None = None
    resolved_channel: str | None = None
    resolved_risk_user_id = 0
    resolved_role: Literal["owner", "invite"] = "owner"

    if credentials_resolver is not None:
      try:
        result = await asyncio.wait_for(
          credentials_resolver(payload.api_key, payload),
          timeout=resolver_timeout_seconds,
        )
      except asyncio.TimeoutError:
        return _error_response(
          CredentialsTimeoutError(
            f"Credential resolution for user '{resolved_user_id or 'unresolved'}' timed out after "
            f"{resolver_timeout_seconds:.1f}s. Check the resolver latency or raise resolver_timeout_seconds."
          ),
          user_id=resolved_user_id,
          timeout_seconds=resolver_timeout_seconds,
        )
      except Exception as exc:
        return _error_response(exc, user_id=resolved_user_id)

      try:
        resolved_user_id = _normalize_user_id(result.user_id)
      except MissingUserIdError as exc:
        return _error_response(exc)
      if resolved_user_id is None:
        return _error_response(MissingUserIdError("credential resolver must return user_id."))

      resolved_channel = _normalize_channel(result.channel)
      resolved_user_email = result.user_email or resolved_user_email
      try:
        resolved_risk_user_id = int(result.risk_user_id) if result.risk_user_id is not None else 0
      except (TypeError, ValueError):
        resolved_risk_user_id = 0
      if resolved_risk_user_id <= 0:
        return _resolver_contract_error(
          "credential resolver must return a positive risk_user_id.",
          user_id=resolved_user_id,
        )
      if result.role not in {"owner", "invite"}:
        return _resolver_contract_error(
          "credential resolver must return role='owner' or role='invite'.",
          user_id=resolved_user_id,
        )
      resolved_role = result.role
      resolved_auth_config = result.auth_config.to_dict()

      claimed_channel = _claimed_channel(payload)
      if claimed_channel is not None and resolved_channel is not None and claimed_channel != resolved_channel:
        return _error_response(
          ChannelMismatchError(
            f"context.channel={claimed_channel!r} does not match key channel={resolved_channel!r}. "
            "The API key is bound to a specific channel; the request must claim the same channel."
          ),
          user_id=resolved_user_id,
        )
    elif resolved_user_id is None:
      return _error_response(
        MissingUserIdError("user_id is required in /control/session when no credentials resolver is configured.")
      )
    else:
      resolved_channel = _claimed_channel(payload)

    try:
      identity = _resolve_control_identity(
        user_id=resolved_user_id,
        risk_user_id=resolved_risk_user_id,
        user_email=resolved_user_email,
        role=resolved_role,
        channel=resolved_channel,
      )
    except (ValueError, SystemExit) as exc:
      return _resolver_contract_error(str(exc), user_id=resolved_user_id)

    session = auth.session_store.create_session(
      api_key_hash=AuthManager.hash_api_key(payload.api_key),
      user_id=resolved_user_id,
      user_email=identity.user_email,
      risk_user_id=identity.risk_user_id,
      role=resolved_role,
      kind="control",
      auth_config=resolved_auth_config,
      ttl_seconds=CONTROL_SESSION_TTL_SECONDS,
      owner_user_id=str(identity.owner_user_id),
      raw_user_id=str(identity.raw_user_id),
      user_slug=identity.user_slug,
      user_aliases=tuple(str(alias) for alias in identity.aliases),
      identity_status=str(identity.identity_status),
    )
    session.channel = resolved_channel
    session.is_public = False
    session.approval_store = approval_store
    session.approval_policy = approval_policy

    return ControlSessionResponse(
      session_token=auth.issue_token(session),
      session_id=session.session_id,
      expires_at=session.expires_at,
      kind="control",
      user_id=identity.owner_user_id,
      risk_user_id=identity.risk_user_id,
      user_slug=identity.user_slug,
      user_email=identity.user_email,
      channel=session.channel,
      identity=_identity_response(identity),
    )

  return router


__all__ = [
  "CONTROL_SESSION_TTL_SECONDS",
  "ControlSessionIdentityResponse",
  "ControlSessionRequest",
  "ControlSessionResponse",
  "build_session_router",
]
