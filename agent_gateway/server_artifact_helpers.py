from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Mapping

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .auth import ChannelMismatchError, CredentialsTimeoutError, CrossUserReuseError, MissingUserIdError, NoCredentialError
from .session import AuthManager
from .claim_signing_authority import GatewayClaimSigningAuthority
from .server_models import (
  _AGENT_API_CLAIM_AUDIENCE,
  _AGENT_API_CLAIM_CLOCK_SKEW_SECONDS,
  _AGENT_API_CLAIM_HEADERS,
  _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT,
  _AGENT_API_CLAIM_NONCE_HEX_LENGTH,
)

log = logging.getLogger("agent_gateway.server_artifact_helpers")

def _model_to_dict(model: Any) -> Dict[str, Any]:
  if hasattr(model, "model_dump"):
    return model.model_dump()
  return model.dict()


def _normalize_prefix(prefix: str) -> str:
  cleaned = (prefix or "").strip()
  if not cleaned or cleaned == "/":
    return ""
  return "/" + cleaned.strip("/")


def _route_path(prefix: str, suffix: str) -> str:
  normalized = _normalize_prefix(prefix)
  return f"{normalized}{suffix}" if normalized else suffix


def _default_control_skills_dir() -> Path:
  configured = os.getenv("AGENT_GATEWAY_SKILLS_DIR", "").strip()
  if configured:
    return Path(configured).expanduser()
  return Path(__file__).resolve().parent / "_no_control_skills"


def _default_autonomous_api_dir() -> Path:
  return Path(__file__).resolve().parents[3] / "api"


def _default_autonomous_log_dir() -> Path | None:
  explicit = os.getenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", "").strip()
  if explicit:
    return Path(explicit).expanduser()
  gateway_log_dir = os.getenv("GATEWAY_LOG_DIR", "").strip()
  if gateway_log_dir:
    return Path(gateway_log_dir).expanduser() / "autonomous"
  legacy_agents_log_dir = os.getenv("AGENTS_MCP_LOG_DIR", "").strip()
  if legacy_agents_log_dir:
    return Path(legacy_agents_log_dir).expanduser()
  return None


def _resolve_compaction_trigger(runtime_val: int | None, config_val: int | None) -> int | None:
  """Resolve compaction trigger: runtime overrides config. 0 or negative = explicitly disable."""
  raw = runtime_val if runtime_val is not None else config_val
  if raw is None or raw <= 0:
    return None
  return raw


def _sanitize_for_json(obj: Any) -> Any:
  if isinstance(obj, float) and not math.isfinite(obj):
    return None
  if isinstance(obj, dict):
    return {key: _sanitize_for_json(value) for key, value in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_sanitize_for_json(value) for value in obj]
  if isinstance(obj, (set, frozenset)):
    return [_sanitize_for_json(value) for value in obj]
  return obj


def _json_dumps(payload: Dict[str, Any]) -> str:
  sanitized = _sanitize_for_json(payload)
  return bytes(JSONResponse(content=sanitized).body).decode("utf-8")


def _claim_ttl_ceiling_seconds() -> int:
  raw = os.getenv("AGENT_API_CLAIM_MAX_TTL_SECONDS", "").strip()
  if not raw:
    return _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT
  try:
    value = int(raw)
  except ValueError:
    return _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT
  return value if value > 0 else _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT


def _verify_signed_user_claim(request: Request) -> dict[str, Any]:
  claim_headers = _extract_agent_claim_headers(request.headers)
  if claim_headers is None:
    raise HTTPException(status_code=401, detail="Signed user claim required")

  authority = getattr(
    request.app.state,
    "gateway_claim_signing_authority",
    None,
  )
  if type(authority) is not GatewayClaimSigningAuthority:
    raise HTTPException(
      status_code=503,
      detail="Agent API signed claim verifier not configured",
    )

  verified = authority.verify_user_claim(
    claim_headers,
    ttl_ceiling=_claim_ttl_ceiling_seconds(),
  )
  if verified is None:
    raise HTTPException(status_code=401, detail="Invalid signed user claim")
  return verified


def _artifact_auth_dependency(request: Request) -> str:
  authorization = request.headers.get("Authorization")
  if authorization is not None:
    token = AuthManager.get_bearer_token(authorization)
    auth_manager = getattr(request.app.state, "auth", None)
    if auth_manager is None:
      raise HTTPException(status_code=503, detail="Gateway auth manager unavailable")
    session, _claims = auth_manager.verify_token_with_payload(token)
    risk_user_id = int(getattr(session, "risk_user_id", 0) or 0)
    if risk_user_id > 0:
      return str(risk_user_id)
    return session.user_id

  claim = _verify_signed_user_claim(request)
  return str(claim["user_id"])


def _extract_agent_claim_headers(headers: Mapping[str, Any]) -> dict[str, str] | None:
  claim_headers: dict[str, str] = {}
  for field_name, header_name in _AGENT_API_CLAIM_HEADERS.items():
    value = headers.get(header_name)
    if value is None:
      return None
    claim_headers[field_name] = str(value)
  return claim_headers


def _verify_agent_claim_headers(
  hmac_key: str,
  claim_headers: Mapping[str, str],
  *,
  ttl_ceiling: int,
  now: int | None = None,
) -> dict[str, Any] | None:
  if claim_headers.get("audience") != _AGENT_API_CLAIM_AUDIENCE:
    return None
  try:
    issued_at = int(claim_headers.get("issued_at", ""))
    expiry = int(claim_headers.get("expiry", ""))
  except (TypeError, ValueError):
    return None

  current_time = int(time.time()) if now is None else int(now)
  if issued_at > current_time + _AGENT_API_CLAIM_CLOCK_SKEW_SECONDS:
    return None
  if current_time > expiry:
    return None
  if expiry - issued_at > ttl_ceiling:
    return None

  user_id = str(claim_headers.get("user_id") or "")
  user_email = str(claim_headers.get("user_email") or "")
  nonce = str(claim_headers.get("nonce") or "")
  signature = str(claim_headers.get("signature") or "")
  if not user_id or not user_email:
    return None
  if len(nonce) != _AGENT_API_CLAIM_NONCE_HEX_LENGTH:
    return None
  try:
    bytes.fromhex(nonce)
  except ValueError:
    return None

  canonical = f"{_AGENT_API_CLAIM_AUDIENCE}\n{issued_at}\n{expiry}\n{user_id}\n{user_email}\n{nonce}".encode("utf-8")
  expected = hmac.new(hmac_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
  if not hmac.compare_digest(expected, signature):
    return None
  return {
    **dict(claim_headers),
    "issued_at": issued_at,
    "expiry": expiry,
    "user_id": user_id,
    "user_email": user_email,
  }


def _normalize_request_user_id(user_id: str | None) -> str | None:
  normalized = user_id.strip() if isinstance(user_id, str) else user_id
  if normalized == "":
    return None
  if normalized == "_default":
    raise MissingUserIdError("user_id '_default' is reserved; supply a stable end-user id.")
  return normalized


def _resolver_contract_payload(message: str, *, user_id: str | None = None) -> tuple[int, Dict[str, Any]]:
  del message
  payload: Dict[str, Any] = {
    "error": "credential_resolver_invalid",
    "message": "Credential resolver returned invalid identity metadata",
  }
  if user_id is not None:
    payload["user_id"] = user_id
  return 400, payload


def _error_payload(
  exc: Exception,
  *,
  user_id: str | None = None,
  session_id: str | None = None,
  request_user: str | None = None,
  session_user: str | None = None,
  timeout_seconds: float | None = None,
) -> tuple[int, Dict[str, Any]]:
  if isinstance(exc, CredentialsTimeoutError):
    payload: Dict[str, Any] = {
      "error": "credentials_timeout",
      "message": str(exc),
    }
    if user_id is not None:
      payload["user_id"] = user_id
    if timeout_seconds is not None:
      payload["timeout_seconds"] = timeout_seconds
    return 504, payload

  if isinstance(exc, MissingUserIdError):
    payload = {"error": "missing_user_id", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    if session_id is not None:
      payload["session_id"] = session_id
    return 400, payload

  if isinstance(exc, CrossUserReuseError):
    payload = {"error": "cross_user_reuse", "message": str(exc)}
    if session_id is not None:
      payload["session_id"] = session_id
    if session_user is not None:
      payload["session_user"] = session_user
    if request_user is not None:
      payload["request_user"] = request_user
    return 401, payload

  if isinstance(exc, NoCredentialError):
    payload = {"error": "credentials_unavailable", "message": str(exc), "reason": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    return 401, payload

  if isinstance(exc, ChannelMismatchError):
    payload = {"error": "channel_mismatch", "message": str(exc)}
    if user_id is not None:
      payload["user_id"] = user_id
    return 400, payload

  if isinstance(exc, HTTPException):
    payload = {
      "error": "auth_failed",
      "message": "Authentication failed",
    }
    if user_id is not None:
      payload["user_id"] = user_id
    return exc.status_code, payload

  payload = {
    "error": "credentials_unavailable",
    "message": "Credential resolver unavailable",
    "reason": "credential_resolver_failed",
  }
  if user_id is not None:
    payload["user_id"] = user_id
  return 500, payload
