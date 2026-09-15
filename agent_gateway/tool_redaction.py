from __future__ import annotations

import hashlib
import logging
import hmac
import json
import os
import re
import sys
from typing import Any

from .secret_boundary import sanitize_boundary_value, sanitization_failure_tool_input

log = logging.getLogger("agent_gateway.tool_redaction")


HMAC_KEY_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


_ABSENT_HOST_REDACTION_MODULES = frozenset({
  "agent",
  "agent.shared",
  "agent.shared.tool_redaction",
})


def resolve_redaction_provider() -> Any:
  """Resolve the audit redaction provider once, at import time.

  Prefers the host `agent.shared.tool_redaction` module and selects this
  gateway-local implementation only when the host package is cleanly absent.
  Any other import failure raises, so a broken host redaction install stops
  the process at startup instead of silently degrading per approval call.
  """
  try:
    from agent.shared import tool_redaction as host_redaction
  except ModuleNotFoundError as exc:
    if exc.name not in _ABSENT_HOST_REDACTION_MODULES:
      raise
    return sys.modules[__name__]
  return host_redaction


def get_audit_hmac_secret() -> bytes:
  secret = os.getenv("GATEWAY_AUDIT_HMAC_SECRET", "").strip()
  if not secret:
    log.warning(
      "GATEWAY_AUDIT_HMAC_SECRET is not set; audit args-hash HMAC is keyed "
      "on the public dev-secret constant | failure=true"
    )
    secret = "dev-secret"
  return secret.encode("utf-8")


def get_audit_hmac_key_id() -> str:
  key_id = os.getenv("GATEWAY_AUDIT_HMAC_KEY_ID", "key-1").strip() or "key-1"
  validate_hmac_key_id(key_id)
  return key_id


def validate_hmac_key_id(key_id: str) -> None:
  if not HMAC_KEY_ID_RE.fullmatch(key_id):
    raise ValueError("GATEWAY_AUDIT_HMAC_KEY_ID must match [A-Za-z0-9_.-]{1,64}")


def hmac_value(value: Any, *, deployment_secret: bytes, key_id: str | None = None) -> str:
  resolved_key_id = key_id or get_audit_hmac_key_id()
  validate_hmac_key_id(resolved_key_id)
  digest = hmac.new(
    deployment_secret,
    _canonical_leaf(value).encode("utf-8"),
    hashlib.sha256,
  ).hexdigest()
  return f"hmac-sha256-v1:{resolved_key_id}:{digest}"


def redact_tool_input(
  tool_name: str,
  tool_input: dict[str, Any] | None,
  *,
  server_name: str | None = None,
  deployment_secret: bytes,
  redaction_scope: str = "fresh_raw",
  key_id: str | None = None,
) -> dict[str, Any]:
  _ = tool_name, server_name, deployment_secret, redaction_scope, key_id
  sanitized = sanitize_boundary_value(tool_input or {}, sink="tool_input")
  return sanitized if isinstance(sanitized, dict) else sanitization_failure_tool_input()


def _canonical_leaf(value: Any) -> str:
  if isinstance(value, bytes):
    return value.hex()
  if isinstance(value, (str, int, float, bool)) or value is None:
    return str(value)
  return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
