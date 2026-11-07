from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Callable, Literal

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from agent_gateway.user_identity_mapping import StorageIdentityFinding

from agent_gateway.auth import (
  CredentialsResolver,
)
from agent_gateway.auth import (
  ChannelMismatchError,
  CredentialsTimeoutError,
  MissingUserIdError,
)
from agent_gateway.subject_assertion import SubjectVerificationUnavailable
from agent_gateway.approval_route import bind_session_approval_route
from hank_model_transport.providers.anthropic_oauth import ANTHROPIC_USER_SCOPE_FIELD
from agent_gateway.session import AuthManager
from agent_gateway.ttl_policy import (
  CONTROL_SESSION_TTL,
  TtlResolver,
  default_ttl_resolver,
  resolve_ttl_seconds,
)


CONTROL_SESSION_TTL_SECONDS = 900
McpAuthority = Annotated[int, Field(strict=True, gt=0)] | None


class ControlSessionRequest(BaseModel):
  api_key: str = Field(..., min_length=1)
  user_id: str | None = None
  user_email: str | None = None
  request_id: str | None = None
  subject_assertion: str | None = None
  context: dict[str, Any] = Field(default_factory=dict)


class ControlSessionIdentityResponse(BaseModel):
  model_config = ConfigDict(extra="forbid")

  owner_user_id: str
  mcp_authority: McpAuthority
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
  # Entitlements the resolver supplied that name no registered capability:
  # they grant nothing and are reported so the operator can correct the grant.
  unknown_model_entitled_capabilities: list[str] = Field(default_factory=list)
  identity: ControlSessionIdentityResponse


def _normalize_user_id(user_id: str | None) -> str | None:
  normalized = user_id.strip() if isinstance(user_id, str) else user_id
  if normalized == "":
    return None
  return normalized


def _error_response(
  exc: Exception,
  *,
  user_id: str | None = None,
) -> JSONResponse:
  if isinstance(exc, CredentialsTimeoutError):
    payload: dict[str, Any] = {"error": "credentials_timeout", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
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

  if isinstance(exc, SubjectVerificationUnavailable):
    return JSONResponse(exc.error_body(user_id=user_id), status_code=exc.status_code)

  status_code = getattr(exc, "status_code", 500)
  payload = {
    "error": "auth_failed" if status_code < 500 else "credentials_unavailable",
    "message": (
      "Authentication failed"
      if status_code < 500
      else "Credential resolver unavailable"
    ),
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
  del message
  payload: dict[str, Any] = {
    "error": "credential_resolver_invalid",
    "message": "Credential resolver returned invalid identity metadata",
  }
  if user_id is not None:
    payload["user_id"] = user_id
  return JSONResponse(payload, status_code=400)




@dataclass(frozen=True)
class _FallbackControlIdentity:
  owner_user_id: str
  mcp_authority: int | None
  user_slug: str | None
  risk_user_id: int
  user_email: str | None
  aliases: tuple[str, ...]
  raw_user_id: str
  role: str | None
  channel: str | None
  identity_status: str


def _fallback_control_identity(
  *,
  user_id: str,
  risk_user_id: int,
  user_email: str | None,
  role: str | None,
  channel: str | None,
) -> _FallbackControlIdentity:
  owner_user_id = str(risk_user_id) if risk_user_id > 0 else user_id
  user_slug = user_id if risk_user_id > 0 and user_id != owner_user_id and not user_id.isdecimal() else None
  aliases = [owner_user_id]
  if user_slug is not None:
    aliases.append(user_slug)
  if user_email:
    aliases.append(user_email.strip().lower())
  return _FallbackControlIdentity(
    owner_user_id=owner_user_id,
    mcp_authority=None,
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
  identity_resolver: Callable[..., Any] | None = None,
) -> Any:
  if identity_resolver is None:
    return _fallback_control_identity(
      user_id=user_id,
      risk_user_id=risk_user_id,
      user_email=user_email,
      role=role,
      channel=channel,
    )
  return identity_resolver(
    user_id,
    # 0 is this layer's unresolved sentinel; non-None states an id to the canonical resolver.
    risk_user_id=risk_user_id if risk_user_id > 0 else None,
    user_email=user_email,
    role=role,
    channel=channel,
  )


def _identity_response(
  identity: Any,
  *,
  mcp_authority: int | None,
) -> ControlSessionIdentityResponse:
  return ControlSessionIdentityResponse(
    owner_user_id=str(identity.owner_user_id),
    mcp_authority=mcp_authority,
    user_slug=identity.user_slug,
    aliases=[str(alias) for alias in identity.aliases],
    identity_status=str(identity.identity_status),
  )


def build_session_router(
  *,
  auth: AuthManager,
  credentials_resolver: CredentialsResolver | None,
  tenant_id: str | None,
  allow_service_credentials_for_interactive: bool,
  identity_resolver: Callable[..., Any] | None = None,
  ttl_resolver: TtlResolver = default_ttl_resolver,
  approval_store: Any | None = None,
  approval_policy: Any | None = None,
) -> APIRouter:
  router = APIRouter()

  @router.post("/session", response_model=ControlSessionResponse)
  async def control_session(
    payload: ControlSessionRequest,
    response: Response,
  ) -> ControlSessionResponse | JSONResponse:
    auth.validate_api_key(payload.api_key)
    try:
      resolved_user_id = (
        payload.user_id
        if credentials_resolver is None and identity_resolver is None
        else _normalize_user_id(payload.user_id)
      )
    except MissingUserIdError as exc:
      return _error_response(exc)

    resolved_user_email = payload.user_email.strip() if isinstance(payload.user_email, str) else payload.user_email
    if resolved_user_email == "":
      resolved_user_email = None

    resolved_auth_config: dict[str, Any] | None = None
    resolved_channel: str | None = None
    resolved_risk_user_id = 0
    resolved_mcp_authority: int | None = None
    resolved_role: Literal["owner", "invite"] = "invite"
    resolved_capabilities: frozenset[str] = frozenset()
    resolved_model_entitled_capabilities: frozenset[str] = frozenset()
    resolved_unknown_model_entitled_capabilities: frozenset[str] = frozenset()
    resolved_model_entitled_keys: frozenset[str] = frozenset()
    resolved_enrolled_user_key_providers: frozenset[str] = frozenset()
    resolved_unavailable_user_key_providers: frozenset[str] = frozenset()
    resolved_credential_principal = None
    resolved_allow_service_for_interactive = (
      allow_service_credentials_for_interactive
    )

    if credentials_resolver is not None:
      try:
        result = await credentials_resolver(payload.api_key, payload)
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
      resolved_mcp_authority = resolved_risk_user_id
      if result.role not in {"owner", "invite"}:
        return _resolver_contract_error(
          "credential resolver must return role='owner' or role='invite'.",
          user_id=resolved_user_id,
        )
      resolved_role = result.role
      resolved_capabilities = result.capabilities
      resolved_model_entitled_capabilities = result.model_entitled_capabilities
      resolved_unknown_model_entitled_capabilities = (
        result.unknown_model_entitled_capabilities
      )
      resolved_model_entitled_keys = result.model_entitled_keys
      resolved_enrolled_user_key_providers = result.enrolled_user_key_providers
      resolved_unavailable_user_key_providers = result.unavailable_user_key_providers
      resolved_auth_config = result.auth_config.to_dict()
      # Stamp the validated run owner's scope so anthropic_oauth seat admission keys on it even when the resolver omitted it.
      resolved_auth_config[ANTHROPIC_USER_SCOPE_FIELD] = str(resolved_risk_user_id)
      resolved_credential_principal = result.credential_principal
      resolved_allow_service_for_interactive = (
        result.allow_service_for_interactive
      )

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
        identity_resolver=identity_resolver,
      )
    except (ValueError, SystemExit) as exc:
      return _resolver_contract_error(str(exc), user_id=resolved_user_id)

    try:
      session = auth.session_store.create_session(
        api_key_hash=AuthManager.hash_api_key(payload.api_key),
        user_id=resolved_user_id,
        user_email=identity.user_email,
        risk_user_id=identity.risk_user_id,
        role=resolved_role,
        capabilities=resolved_capabilities,
        model_entitled_capabilities=resolved_model_entitled_capabilities,
        model_entitled_keys=resolved_model_entitled_keys,
        enrolled_user_key_providers=resolved_enrolled_user_key_providers,
        unavailable_user_key_providers=resolved_unavailable_user_key_providers,
        kind="control",
        auth_config=resolved_auth_config,
        ttl_seconds=resolve_ttl_seconds(
          ttl_resolver, resolved_user_id, CONTROL_SESSION_TTL, CONTROL_SESSION_TTL_SECONDS,
        ),
        owner_user_id=identity.owner_user_id,
        raw_user_id=identity.raw_user_id,
        user_slug=identity.user_slug,
        user_aliases=tuple(str(alias) for alias in identity.aliases),
        identity_status=str(identity.identity_status),
        tenant_id=tenant_id,
        credential_principal=(
          resolved_credential_principal if tenant_id is not None else None
        ),
        allow_service_for_interactive=resolved_allow_service_for_interactive,
      )
    except StorageIdentityFinding as exc:
      return _resolver_contract_error(str(exc), user_id=resolved_user_id)
    session.channel = resolved_channel
    session.is_public = False
    bind_session_approval_route(session, approval_store, approval_policy)

    response.headers["Cache-Control"] = "private, no-store"
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
      unknown_model_entitled_capabilities=sorted(
        resolved_unknown_model_entitled_capabilities
      ),
      identity=_identity_response(identity, mcp_authority=resolved_mcp_authority),
    )

  return router


__all__ = [
  "CONTROL_SESSION_TTL_SECONDS",
  "ControlSessionIdentityResponse",
  "ControlSessionRequest",
  "ControlSessionResponse",
  "McpAuthority",
  "build_session_router",
]
