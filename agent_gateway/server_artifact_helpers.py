from __future__ import annotations

import hashlib
import hmac
import json as json_mod
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from .artifact_paths import ArtifactPath
from .auth import ChannelMismatchError, CredentialsTimeoutError, CrossUserReuseError, MissingUserIdError, NoCredentialError
from .session import AuthManager
from .server_models import (
  _AGENT_API_CLAIM_AUDIENCE,
  _AGENT_API_CLAIM_CLOCK_SKEW_SECONDS,
  _AGENT_API_CLAIM_HEADERS,
  _AGENT_API_CLAIM_MAX_TTL_SECONDS_DEFAULT,
  _AGENT_API_CLAIM_NONCE_HEX_LENGTH,
  _ARTIFACT_ORIGIN_FILTER_VALUES,
  _ARTIFACT_ORIGIN_VALUES,
  _ARTIFACT_VISIBILITY_FILTER_VALUES,
  _ARTIFACT_VISIBILITY_VALUES,
)

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
  return JSONResponse(content=sanitized).body.decode("utf-8")


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

  hmac_key = os.getenv("AGENT_API_USER_CLAIM_HMAC_KEY", "").strip()
  if not hmac_key:
    raise HTTPException(
      status_code=503,
      detail="Agent API signed claim verifier not configured (AGENT_API_USER_CLAIM_HMAC_KEY not set)",
    )

  verified = _verify_agent_claim_headers(
    hmac_key,
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


def _artifact_json_response(
  artifact: ArtifactPath,
  *,
  user_id: str,
  filters: dict[str, Any],
) -> JSONResponse:
  path = _assert_artifact_path_still_safe(artifact)
  if not path.is_file():
    raise HTTPException(status_code=404, detail="Artifact not found")
  payload = _artifact_payload_from_path(path)
  payload = _decorate_artifact_payload(payload, user_id=user_id)
  if not _artifact_payload_matches_filters(payload, filters=filters):
    raise HTTPException(status_code=404, detail="Artifact not found")
  return JSONResponse(content=payload, headers=_file_cache_headers(path))


def _artifact_payload_from_path(path: Path) -> dict[str, Any]:
  try:
    with path.open("r", encoding="utf-8") as handle:
      payload = json_mod.load(handle)
  except json_mod.JSONDecodeError as exc:
    raise HTTPException(
      status_code=500,
      detail=f"Artifact JSON is invalid at line {exc.lineno}, column {exc.colno}: {exc.msg}",
    ) from exc
  if not isinstance(payload, dict):
    raise HTTPException(status_code=500, detail="Artifact payload must be a JSON object")
  return payload


def _decorate_artifact_payload(payload: dict[str, Any], *, user_id: str) -> dict[str, Any]:
  decorated = dict(payload)
  fields = _artifact_effective_fields(decorated, user_id=user_id)
  decorated.update(fields)
  return decorated


def _artifact_effective_fields(payload: dict[str, Any], *, user_id: str) -> dict[str, Any]:
  raw_research_file_id = payload.get("research_file_id")
  research_file_id = _int_or_none(raw_research_file_id)
  classification = _artifact_research_file_classification(user_id=user_id, research_file_id=research_file_id)
  if classification is None:
    classification = _artifact_sidecar_classification(
      payload,
      unresolved_research_file=_artifact_research_file_id_token_present(raw_research_file_id),
    )
  return {
    "research_file_id": research_file_id,
    "control_run_id": str(payload.get("control_run_id") or "").strip() or None,
    "has_research_file": research_file_id is not None,
    **classification,
  }


def _artifact_research_file_classification(*, user_id: str, research_file_id: int | None) -> dict[str, Any] | None:
  if research_file_id is None:
    return None
  try:
    from research.repository import get_repository_factory

    row = get_repository_factory().get(user_id).get_file(int(research_file_id))
  except Exception:
    return None
  if row is None:
    return None
  return {
    "origin_kind": _artifact_origin_kind(row.get("origin_kind")),
    "visibility": _artifact_visibility(row.get("visibility")),
    "origin_ref": _artifact_origin_ref(row.get("origin_ref")),
    "classification_source": "research_file",
  }


def _artifact_sidecar_classification(
  payload: dict[str, Any],
  *,
  unresolved_research_file: bool,
) -> dict[str, Any]:
  if payload.get("origin_kind") is None and payload.get("visibility") is None and payload.get("origin_ref") is None:
    if unresolved_research_file:
      return {
        "origin_kind": "import",
        "visibility": "archived",
        "origin_ref": None,
        "classification_source": "unresolved_research_file",
      }
    return {
      "origin_kind": "product",
      "visibility": "default",
      "origin_ref": None,
      "classification_source": "legacy_default",
    }
  try:
    return {
      "origin_kind": _artifact_origin_kind(payload.get("origin_kind")),
      "visibility": _artifact_visibility(payload.get("visibility")),
      "origin_ref": _artifact_origin_ref(payload.get("origin_ref")),
      "classification_source": "sidecar",
    }
  except ValueError:
    return {
      "origin_kind": "import",
      "visibility": "archived",
      "origin_ref": None,
      "classification_source": "invalid_sidecar",
    }


def _artifact_origin_kind(value: object | None) -> str:
  normalized = str(value if value is not None else "product").strip().lower()
  if normalized not in _ARTIFACT_ORIGIN_VALUES:
    raise ValueError("invalid artifact origin_kind")
  return normalized


def _artifact_origin_kind_filter(value: object | None) -> str:
  normalized = str(value if value is not None else "all").strip().lower()
  if normalized not in _ARTIFACT_ORIGIN_FILTER_VALUES:
    raise HTTPException(status_code=400, detail="origin_kind filter is invalid")
  return normalized


def _artifact_visibility(value: object | None) -> str:
  normalized = str(value if value is not None else "default").strip().lower()
  if normalized not in _ARTIFACT_VISIBILITY_VALUES:
    raise ValueError("invalid artifact visibility")
  return normalized


def _artifact_visibility_filter(value: object | None) -> str:
  normalized = str(value if value is not None else "default").strip().lower()
  if normalized not in _ARTIFACT_VISIBILITY_FILTER_VALUES:
    raise HTTPException(status_code=400, detail="visibility filter is invalid")
  return normalized


def _artifact_origin_ref(value: object | None) -> dict[str, Any] | None:
  if value is None:
    return None
  if isinstance(value, dict):
    return value or None
  if isinstance(value, str) and value.strip():
    parsed = json_mod.loads(value)
    if isinstance(parsed, dict):
      return parsed or None
  if isinstance(value, str) and not value.strip():
    return None
  raise ValueError("invalid artifact origin_ref")


def _artifact_request_filters(request: Request) -> dict[str, Any]:
  return {
    "research_file_id": _query_int_or_none(request, "research_file_id"),
    "control_run_id": _query_str_or_none(request, "control_run_id"),
    "visibility": _artifact_visibility_filter(request.query_params.get("visibility")),
    "origin_kind": _artifact_origin_kind_filter(request.query_params.get("origin_kind")),
  }


def _artifact_payload_matches_filters(payload: dict[str, Any], *, filters: dict[str, Any]) -> bool:
  research_file_id = filters.get("research_file_id")
  if research_file_id is not None and payload.get("research_file_id") != int(research_file_id):
    return False
  control_run_id = filters.get("control_run_id")
  if control_run_id is not None and payload.get("control_run_id") != control_run_id:
    return False
  origin_kind = filters.get("origin_kind")
  if origin_kind != "all" and payload.get("origin_kind") != origin_kind:
    return False
  visibility = filters.get("visibility")
  if visibility != "all" and payload.get("visibility") != visibility:
    return False
  return True


def _int_or_none(value: Any) -> int | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, int):
    return value
  if isinstance(value, str):
    text = value.strip()
    if not text:
      return None
    try:
      return int(text)
    except ValueError:
      return None
  return None


def _artifact_research_file_id_token_present(value: Any) -> bool:
  if value is None:
    return False
  if isinstance(value, str):
    return bool(value.strip())
  return True


def _query_int_or_none(request: Request, name: str) -> int | None:
  raw = request.query_params.get(name)
  if raw is None or str(raw).strip() == "":
    return None
  try:
    return int(str(raw).strip())
  except ValueError as exc:
    raise HTTPException(status_code=422, detail=f"{name} must be an integer") from exc


def _query_str_or_none(request: Request, name: str) -> str | None:
  raw = request.query_params.get(name)
  if raw is None:
    return None
  return str(raw).strip() or None


def _assert_artifact_path_still_safe(artifact: ArtifactPath) -> Path:
  try:
    resolved = artifact.path.resolve()
    resolved.relative_to(artifact.workspace_root.resolve())
  except ValueError as exc:
    raise HTTPException(status_code=400, detail="Unsafe artifact path") from exc
  return resolved


def _file_cache_headers(path: Path) -> dict[str, str]:
  stat = path.stat()
  return {
    "Cache-Control": "private, max-age=0",
    "ETag": f'W/"{stat.st_mtime_ns:x}-{stat.st_size:x}"',
  }


def _letter_filename(ticker: str, artifact_id: str) -> str:
  date = artifact_id[:10] if len(artifact_id) >= 10 else artifact_id
  return f"LP-letter-{ticker}-{date}.docx"


def _normalize_request_user_id(user_id: str | None) -> str | None:
  normalized = user_id.strip() if isinstance(user_id, str) else user_id
  if normalized == "":
    return None
  if normalized == "_default":
    raise MissingUserIdError("user_id '_default' is reserved; supply a stable end-user id.")
  return normalized


def _resolver_contract_payload(message: str, *, user_id: str | None = None) -> tuple[int, Dict[str, Any]]:
  payload: Dict[str, Any] = {"error": "credential_resolver_invalid", "message": message}
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
      "message": str(exc.detail) if exc.detail is not None else "Authentication failed",
    }
    if user_id is not None:
      payload["user_id"] = user_id
    return exc.status_code, payload

  payload = {"error": "credentials_unavailable", "message": str(exc), "reason": str(exc)}
  if user_id is not None:
    payload["user_id"] = user_id
  return 500, payload
