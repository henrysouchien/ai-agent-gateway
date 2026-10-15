from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import time
from typing import Any

from .ttl_policy import AGENT_API_CLAIM_TTL, TtlResolver, resolve_ttl_seconds

from hank_contracts.agent_api import (
  AGENT_API_CLAIM_AUDIENCE as _AGENT_API_CLAIM_AUDIENCE,
  AGENT_API_CLAIM_TTL_SECONDS_DEFAULT as _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT,
  AGENT_API_CLAIM_ENV_VARS as _AGENT_API_CLAIM_ENV_VARS,
  canonical_agent_api_claim_bytes,
)


def _runtime_module() -> Any:
  for module_name in ("agent_gateway.autonomous_runner", "autonomous_runner"):
    module = sys.modules.get(module_name)
    if module is not None:
      return module
  return sys.modules[__name__]


def _runtime_attr(name: str, default: Any) -> Any:
  return getattr(_runtime_module(), name, default)


def _time_time() -> float:
  return _runtime_attr("time", time).time()


def get_agent_api_claim_ttl_seconds(
  user_id: str | None = None,
  ttl_resolver: TtlResolver | None = None,
) -> int:
  raw = os.getenv("AGENT_API_CLAIM_TTL_SECONDS", "").strip()
  default_ttl = _runtime_attr("_AGENT_API_CLAIM_TTL_SECONDS_DEFAULT", _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT)
  try:
    ttl_seconds = int(raw)
  except ValueError:
    ttl_seconds = default_ttl
  fallback = ttl_seconds if ttl_seconds > 0 else default_ttl
  if ttl_resolver is None or user_id is None:
    return fallback
  return resolve_ttl_seconds(
    ttl_resolver, user_id, AGENT_API_CLAIM_TTL, fallback,
  )


def sign_user_claim(
  hmac_key: str,
  *,
  user_id: str,
  user_email: str | None,
  ttl_seconds: int,
) -> dict[str, str]:
  if ttl_seconds <= 0:
    raise ValueError("ttl_seconds must be positive")

  audience = _runtime_attr("_AGENT_API_CLAIM_AUDIENCE", _AGENT_API_CLAIM_AUDIENCE)
  env_vars = _runtime_attr("_AGENT_API_CLAIM_ENV_VARS", _AGENT_API_CLAIM_ENV_VARS)
  issued_at = int(_time_time())
  expiry = issued_at + ttl_seconds
  nonce = secrets.token_hex(16)
  normalized_email = user_email or ""
  canonical = canonical_agent_api_claim_bytes(
    audience=audience,
    issued_at=issued_at,
    expiry=expiry,
    user_id=user_id,
    user_email=normalized_email,
    nonce=nonce,
  )
  signature = hmac.new(hmac_key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()

  return {
    env_vars["audience"]: audience,
    env_vars["issued_at"]: str(issued_at),
    env_vars["expiry"]: str(expiry),
    env_vars["user_id"]: user_id,
    env_vars["user_email"]: normalized_email,
    env_vars["nonce"]: nonce,
    env_vars["signature"]: signature,
  }


__all__ = [
  "get_agent_api_claim_ttl_seconds",
  "sign_user_claim",
]
