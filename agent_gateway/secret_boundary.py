from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import logging
import re
from typing import Any

from .capability_execution import BoundCapabilityExecution
from .ui_blocks_metrics import record as record_package_counter


REDACTED_SECRET = "<redacted-secret>"
SANITIZATION_FAILED = "<secret-sanitization-failed>"
UNSUPPORTED_VALUE = "<unsupported-boundary-value>"

_CREDENTIAL_FIELDS = frozenset({
  "access_token",
  "api_key",
  "auth_token",
  "client_secret",
  "id_token",
  "refresh_token",
})
_MIN_SUBSTRING_SECRET_LENGTH = 8
_MAX_DEPTH = 32
_MAX_NODES = 100_000
_LOG = logging.getLogger("agent_gateway.secret_boundary")

# These patterns intentionally recognize credential material, not suspicious key
# names. Short examples such as ``sk-example`` and ordinary prose remain intact.
_HIGH_CONFIDENCE_PATTERNS = (
  re.compile(r"\bsk-ant-(?:api\d{2}|oat\d{2})-[A-Za-z0-9_-]{16,}\b"),
  re.compile(r"\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{24,}\b"),
  re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.IGNORECASE),
  re.compile(
    r"-----BEGIN (?:[A-Z0-9]+ )?PRIVATE KEY-----.*?"
    r"-----END (?:[A-Z0-9]+ )?PRIVATE KEY-----",
    re.DOTALL,
  ),
)


class SecretBoundary:
  """Lifecycle-local secret knowledge for model and persistence boundaries.

  The object is intentionally neither serializable nor backed by process-global
  registration. A runner owns it for exactly the lifetime of one bound capability
  execution.
  """

  __slots__ = ("_exact_values",)

  def __init__(self, exact_values: tuple[str, ...] = ()) -> None:
    self._exact_values = tuple(
      sorted(
        {
          value
          for value in exact_values
          if isinstance(value, str)
          and value.strip()
        },
        key=len,
        reverse=True,
      )
    )

  @classmethod
  def from_capability_execution(
    cls,
    execution: BoundCapabilityExecution,
  ) -> "SecretBoundary":
    if not isinstance(execution, BoundCapabilityExecution):
      raise TypeError("secret registration requires BoundCapabilityExecution")
    return cls.from_auth_config(execution.auth_config)

  @classmethod
  def from_auth_config(
    cls,
    auth_config: Mapping[str, Any],
  ) -> "SecretBoundary":
    """Register exact active credential material from a typed auth mapping."""

    if not isinstance(auth_config, Mapping):
      raise TypeError("secret registration requires an auth config mapping")
    values = tuple(
      value.strip()
      for key, value in auth_config.items()
      if key in _CREDENTIAL_FIELDS
      and isinstance(value, str)
      and value.strip()
    )
    return cls(values)

  def __reduce__(self) -> Any:
    raise TypeError("SecretBoundary is lifecycle-local and not serializable")

  def __getstate__(self) -> Any:
    raise TypeError("SecretBoundary is lifecycle-local and not serializable")

  def extended_for_capability_execution(
    self,
    execution: BoundCapabilityExecution,
  ) -> "SecretBoundary":
    refreshed = type(self).from_capability_execution(execution)
    return type(self)((*self._exact_values, *refreshed._exact_values))

  def sanitize(self, value: Any, *, sink: str) -> Any:
    del sink  # Sink is explicit at call sites and reserved for value-free metrics.
    remaining = [_MAX_NODES]
    return self._sanitize(value, depth=0, remaining=remaining, active=set())

  def _sanitize(
    self,
    value: Any,
    *,
    depth: int,
    remaining: list[int],
    active: set[int],
  ) -> Any:
    remaining[0] -= 1
    if remaining[0] < 0 or depth > _MAX_DEPTH:
      _observe_sanitization_failure()
      return SANITIZATION_FAILED
    if value is None or isinstance(value, (bool, int, float)):
      return value
    if isinstance(value, str):
      return self._sanitize_string(value)
    if isinstance(value, bytes):
      return UNSUPPORTED_VALUE

    container_id = id(value)
    if container_id in active:
      _observe_sanitization_failure()
      return SANITIZATION_FAILED
    if isinstance(value, Mapping):
      active.add(container_id)
      try:
        sanitized: dict[Any, Any] = {}
        for raw_key, raw_value in value.items():
          key = (
            self._sanitize_string(raw_key)
            if isinstance(raw_key, str)
            else raw_key
            if raw_key is None or isinstance(raw_key, (bool, int, float))
            else UNSUPPORTED_VALUE
          )
          if key in sanitized:
            _observe_sanitization_failure()
            return SANITIZATION_FAILED
          sanitized[key] = self._sanitize(
            raw_value,
            depth=depth + 1,
            remaining=remaining,
            active=active,
          )
        return sanitized
      finally:
        active.remove(container_id)
    if isinstance(value, (list, tuple, set, frozenset)):
      active.add(container_id)
      try:
        return [
          self._sanitize(
            item,
            depth=depth + 1,
            remaining=remaining,
            active=active,
          )
          for item in value
        ]
      finally:
        active.remove(container_id)
    return UNSUPPORTED_VALUE

  def _sanitize_string(self, value: str) -> str:
    sanitized = value
    for secret in self._exact_values:
      if len(secret) >= _MIN_SUBSTRING_SECRET_LENGTH:
        sanitized = sanitized.replace(secret, REDACTED_SECRET)
      elif sanitized == secret:
        sanitized = REDACTED_SECRET
    for pattern in _HIGH_CONFIDENCE_PATTERNS:
      sanitized = pattern.sub(REDACTED_SECRET, sanitized)
    return sanitized


def sanitize_boundary_value(
  value: Any,
  *,
  sink: str,
  boundary: SecretBoundary | None = None,
) -> Any:
  """Return a safe projection, using a fixed tombstone on sanitizer failure."""

  try:
    return (boundary or SecretBoundary()).sanitize(value, sink=sink)
  except Exception:
    _observe_sanitization_failure()
    return SANITIZATION_FAILED


def _observe_sanitization_failure() -> None:
  """Emit value-free evidence that a boundary projection failed closed."""

  record_package_counter("secret_boundary_sanitization_failed")
  _LOG.warning(
    "Secret boundary sanitization failed; fixed tombstone emitted"
  )


def sanitize_context_boundary_value(
  context: Any,
  value: Any,
  *,
  sink: str,
) -> Any:
  """Use a ToolResultContext's runner-owned boundary without persisting it."""

  sanitizer = getattr(context, "boundary_sanitizer", None)
  if callable(sanitizer):
    try:
      return sanitizer(value, sink)
    except Exception:
      _observe_sanitization_failure()
      return SANITIZATION_FAILED
  return sanitize_boundary_value(value, sink=sink)


def sanitization_failure_tool_input() -> dict[str, str]:
  return {"_boundary_error": SANITIZATION_FAILED}


def sanitization_failure_tool_block(
  block_type: str,
  *,
  correlation_value: Any = None,
) -> dict[str, Any]:
  """Return a structurally valid, value-free typed block tombstone."""

  del correlation_value
  safe_id = "boundary-sanitization-failed"
  if block_type in {"tool_use", "server_tool_use"}:
    return {
      "type": block_type,
      "id": safe_id,
      "name": "boundary_sanitization_failed",
      "input": sanitization_failure_tool_input(),
    }
  return {
    "type": "tool_result",
    "tool_use_id": safe_id,
    "content": SANITIZATION_FAILED,
    "is_error": True,
  }


def sanitize_approval_decision_projection(
  decision: Any,
  *,
  sink: str,
  boundary: SecretBoundary | None = None,
) -> tuple[Any, dict[str, Any] | None]:
  """Separate a policy decision's durable projection from raw execution args."""

  raw_modified = getattr(decision, "modified_tool_args", None)
  raw_modified = dict(raw_modified) if isinstance(raw_modified, Mapping) else None
  projected_fields = sanitize_boundary_value(
    {
      "reason": getattr(decision, "reason", ""),
      "route_target": getattr(decision, "route_target", None),
      "route_target_type": getattr(decision, "route_target_type", None),
      "persistent_grant_scope_hint": getattr(
        decision,
        "persistent_grant_scope_hint",
        None,
      ),
      "redacted_args_for_audit": getattr(
        decision,
        "redacted_args_for_audit",
        None,
      ),
      "args_predicate": getattr(decision, "args_predicate", None),
      "policy_id": getattr(decision, "policy_id", "single-user"),
      "policy_version": getattr(decision, "policy_version", "1"),
      "grant_reference": getattr(decision, "grant_reference", None),
    },
    sink=sink,
    boundary=boundary,
  )
  if not isinstance(projected_fields, dict):
    projected_fields = {
      "reason": SANITIZATION_FAILED,
      "route_target": None,
      "route_target_type": None,
      "persistent_grant_scope_hint": None,
      "redacted_args_for_audit": None,
      "args_predicate": None,
      "policy_id": "boundary-failure",
      "policy_version": "0",
      "grant_reference": None,
    }
  updates = {
    "reason": (
      projected_fields["reason"]
      if isinstance(projected_fields.get("reason"), str)
      else SANITIZATION_FAILED
    ),
    "route_target": (
      projected_fields.get("route_target")
      if isinstance(projected_fields.get("route_target"), str)
      else None
    ),
    "route_target_type": (
      projected_fields.get("route_target_type")
      if isinstance(projected_fields.get("route_target_type"), str)
      else None
    ),
    "persistent_grant_scope_hint": (
      projected_fields.get("persistent_grant_scope_hint")
      if isinstance(projected_fields.get("persistent_grant_scope_hint"), str)
      else None
    ),
    "redacted_args_for_audit": (
      projected_fields.get("redacted_args_for_audit")
      if isinstance(projected_fields.get("redacted_args_for_audit"), dict)
      else None
    ),
    "args_predicate": (
      projected_fields.get("args_predicate")
      if isinstance(projected_fields.get("args_predicate"), dict)
      else None
    ),
    "policy_id": (
      projected_fields.get("policy_id")
      if isinstance(projected_fields.get("policy_id"), str)
      else "boundary-failure"
    ),
    "policy_version": (
      projected_fields.get("policy_version")
      if isinstance(projected_fields.get("policy_version"), str)
      else "0"
    ),
    "grant_reference": (
      projected_fields.get("grant_reference")
      if isinstance(projected_fields.get("grant_reference"), str)
      else None
    ),
    "modified_tool_args": None,
  }
  try:
    return replace(decision, **updates), raw_modified
  except Exception:
    _observe_sanitization_failure()
    fallback_updates = {
      **updates,
      "reason": SANITIZATION_FAILED,
      "route_target": None,
      "route_target_type": None,
      "persistent_grant_scope_hint": None,
      "redacted_args_for_audit": None,
      "args_predicate": None,
      "grant_reference": None,
    }
    return replace(decision, **fallback_updates), raw_modified


def sanitize_tool_event(
  event: Mapping[str, Any],
  *,
  sink: str,
  boundary: SecretBoundary | None = None,
) -> dict[str, Any]:
  """Sanitize only typed tool-derived fields, leaving ordinary prose alone."""

  projected = dict(event)
  event_type = str(projected.get("type") or "")
  fields: tuple[str, ...] = ()
  if event_type in {
    "tool_call_start",
    "tool_call_interrupted",
    "tool_execute_request",
    "tool_approval_request",
  }:
    fields = ("tool_input", "display")
  elif event_type == "tool_call_complete":
    fields = (
      "result",
      "error",
      "semantic_error",
      "final_tool_result_blocks",
    )
  elif event_type == "tool_output_chunk":
    fields = ("text", "content")
  elif event_type in {"error", "run_error", "stream_retry"}:
    fields = ("error", "message", "detail", "reason")
  elif event_type in {"readable_resource_ready", "artifact_ready"}:
    fields = ("content", "artifact", "metadata")

  for field in fields:
    if field in projected:
      projected[field] = sanitize_boundary_value(
        projected[field],
        sink=sink,
        boundary=boundary,
      )

  blocks_field = None
  if event_type == "assistant_message":
    blocks_field = "content_blocks"
  elif event_type == "runtime_guard":
    blocks_field = "draft_content_blocks"
  if blocks_field is not None and isinstance(projected.get(blocks_field), list):
    projected[blocks_field] = _sanitize_typed_blocks(
      projected[blocks_field],
      sink=sink,
      boundary=boundary,
    )
  if event_type == "user_message" and isinstance(projected.get("content"), list):
    projected["content"] = _sanitize_typed_blocks(
      projected["content"],
      sink=sink,
      boundary=boundary,
    )
  return projected


def _sanitize_typed_blocks(
  blocks: list[Any],
  *,
  sink: str,
  boundary: SecretBoundary | None,
) -> list[Any]:
  projected: list[Any] = []
  for block in blocks:
    if not isinstance(block, Mapping):
      projected.append(block)
      continue
    block_type = str(block.get("type") or "")
    if block_type not in {"tool_use", "server_tool_use", "tool_result"}:
      projected.append(dict(block))
      continue
    safe_block = sanitize_boundary_value(
      block,
      sink=sink,
      boundary=boundary,
    )
    if isinstance(safe_block, dict):
      projected.append(safe_block)
      continue
    _observe_sanitization_failure()
    projected.append(
      sanitization_failure_tool_block(
        block_type,
        correlation_value=(
          block.get("id")
          if block_type in {"tool_use", "server_tool_use"}
          else block.get("tool_use_id")
        ),
      )
    )
  return projected


__all__ = [
  "REDACTED_SECRET",
  "SANITIZATION_FAILED",
  "SecretBoundary",
  "sanitize_boundary_value",
  "sanitize_approval_decision_projection",
  "sanitize_context_boundary_value",
  "sanitize_tool_event",
  "sanitization_failure_tool_block",
  "sanitization_failure_tool_input",
]
