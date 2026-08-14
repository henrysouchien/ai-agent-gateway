from __future__ import annotations

import asyncio
import math
import uuid
from contextvars import ContextVar
from datetime import timedelta
from numbers import Real
from typing import Any

from .approval_policy import DelegationGrant, utc_now


DEFAULT_MESSAGE_EXCEL_AGENT_TIMEOUT_SECONDS = 300.0
ALLOWED_DELEGATION_TOOL_CLASSES = frozenset({"read", "pure_transform", "artifact_write", "state_write"})
# Sentinel distinguishing "caller omitted this field" from an explicit null so
# chat selection intent is forwarded (or refused) on presence, never guessed.
_ABSENT: Any = object()
# Chat selection intent keys copied from caller tool_input by presence: stable
# keys flow through to relay admission/delivery; raw 'model' is refused typed.
CHAT_SELECTION_INTENT_KEYS = ("model", "model_key", "effort", "catalog_revision")


def chat_selection_intent_kwargs(tool_input: dict[str, Any]) -> dict[str, Any]:
  """Extract chat model-selection intent for mint_and_submit by key presence."""
  return {key: tool_input[key] for key in CHAT_SELECTION_INTENT_KEYS if key in tool_input}
# Keeps the public helper signature stable while preserving the interactive handler's
# historical behavior of forwarding timeout_s to relay.submit.
_SUBMIT_TIMEOUT_SECONDS: ContextVar[int | None] = ContextVar("_SUBMIT_TIMEOUT_SECONDS", default=None)


def _validate_relay_restart_exceptions(
  value: tuple[type[Exception], ...],
) -> tuple[type[Exception], ...]:
  if not isinstance(value, tuple) or not value or any(
    not isinstance(exception_type, type)
    or not issubclass(exception_type, Exception)
    for exception_type in value
  ):
    raise TypeError("relay_restart_exceptions must contain only Exception types")
  return value


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
  payload: dict[str, Any] = {"code": code, "message": message}
  payload.update(details)
  return payload


def _as_non_empty_str(value: Any) -> str | None:
  if not isinstance(value, str):
    return None
  stripped = value.strip()
  return stripped or None


def _is_positive_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_number(value: Any) -> bool:
  return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def validate_requested_ceiling(value: Any) -> tuple[frozenset[str] | None, dict[str, Any] | None]:
  if not isinstance(value, (list, tuple, set, frozenset)) or isinstance(value, (str, bytes)):
    return None, _error("invalid_input", "'approve_tool_classes' must be an array")

  ceiling: set[str] = set()
  for item in value:
    if not isinstance(item, str) or not item.strip():
      return None, _error("invalid_input", "'approve_tool_classes' entries must be non-empty strings")
    tool_class = item.strip()
    if tool_class not in ALLOWED_DELEGATION_TOOL_CLASSES:
      return None, _error(
        "invalid_input",
        f"Unsupported delegated approval tool class '{tool_class}'",
        allowed_tool_classes=sorted(ALLOWED_DELEGATION_TOOL_CLASSES),
      )
    ceiling.add(tool_class)

  return frozenset(ceiling), None


def _connected_workbooks(workbooks: list[dict[str, Any]]) -> list[dict[str, Any]]:
  connected: list[dict[str, Any]] = []
  for workbook in workbooks:
    if not isinstance(workbook, dict):
      continue
    if workbook.get("detached"):
      continue
    if workbook.get("live") is False:
      continue
    if _as_non_empty_str(workbook.get("gateway_session_id")) is None:
      continue
    connected.append(workbook)
  return connected


def _workbook_label(workbook: dict[str, Any]) -> str | None:
  return _as_non_empty_str(workbook.get("name"))


def _target_matches(workbook: dict[str, Any], requested: str) -> bool:
  candidates = (
    workbook.get("name"),
    workbook.get("session"),
    workbook.get("gateway_session_id"),
  )
  return any(isinstance(candidate, str) and candidate.strip() == requested for candidate in candidates)


def _workbook_names(workbooks: list[dict[str, Any]]) -> list[str]:
  names: list[str] = []
  for workbook in workbooks:
    name = _workbook_label(workbook)
    if name is not None:
      names.append(name)
  return names


def _validate_mint_inputs(
  *,
  text: Any,
  workbook: Any,
  force_compaction: Any,
  window_seconds: Any,
  args_predicate: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  parsed_text = _as_non_empty_str(text)
  if parsed_text is None:
    return None, _error("invalid_input", "'text' is required")

  parsed_workbook: str | None = None
  if workbook is not None:
    parsed_workbook = _as_non_empty_str(workbook)
    if parsed_workbook is None:
      return None, _error("invalid_input", "'workbook' must be a non-empty string")

  if not isinstance(force_compaction, bool):
    return None, _error("invalid_input", "'force_compaction' must be a boolean")

  if not _is_positive_int(window_seconds):
    return None, _error("invalid_input", "'window_seconds' must be a positive integer")

  if args_predicate is not None and not isinstance(args_predicate, dict):
    return None, _error("invalid_input", "'args_predicate' must be an object or null")

  return {
    "text": parsed_text,
    "workbook": parsed_workbook,
    "force_compaction": force_compaction,
    "window_seconds": window_seconds,
    "args_predicate": args_predicate,
  }, None


async def mint_and_submit(
  *,
  relay: Any,
  approval_store: Any,
  user_id: str,
  gateway_session_id: str,
  text: Any,
  workbook: Any = None,
  force_compaction: Any = False,
  window_seconds: Any = None,
  args_predicate: Any = None,
  delegator_profile: str = "orchestrator",
  delegator_run_id: str | None = None,
  default_window_seconds: int = 600,
  default_ceiling: frozenset[str] = ALLOWED_DELEGATION_TOOL_CLASSES,
  relay_timeout_seconds: Any = None,
  seed_history: Any = None,
  model: Any = _ABSENT,
  model_key: Any = _ABSENT,
  effort: Any = _ABSENT,
  catalog_revision: Any = _ABSENT,
  relay_restart_exceptions: tuple[type[Exception], ...],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Discover a target workbook, mint its grant, and submit one Excel agent turn."""

  restart_exception_types = _validate_relay_restart_exceptions(
    relay_restart_exceptions
  )

  if model is not _ABSENT:
    # Same admission rule as the relay (relay_requests.enqueue): raw 'model' is
    # retired selection intent and must be refused typed before any grant is
    # minted, never silently dropped into a server-default turn.
    return None, _error(
      "chat_model_not_accepted",
      "'model' is not accepted; pass 'model_key' with a stable model key "
      "from the session's eligible model choices, or omit it for the server default",
    )

  parsed_ceiling, error = validate_requested_ceiling(default_ceiling)
  if error is not None:
    return None, error
  assert parsed_ceiling is not None

  parsed_relay_timeout_seconds: int | None = None
  if relay_timeout_seconds is not None:
    if not _is_positive_number(relay_timeout_seconds):
      return None, _error("invalid_input", "'relay_timeout_seconds' must be a positive number")
    parsed_relay_timeout_seconds = max(1, int(math.ceil(float(relay_timeout_seconds))))

  mint_inputs, error = _validate_mint_inputs(
    text=text,
    workbook=workbook,
    force_compaction=force_compaction,
    window_seconds=default_window_seconds if window_seconds is None else window_seconds,
    args_predicate=args_predicate,
  )
  if error is not None:
    return None, error
  assert mint_inputs is not None

  created_delegation_id: str | None = None
  try:
    listed = await relay.list_workbooks(gateway_session_id, user_id)
    if not isinstance(listed, list):
      return None, _error("relay_error", "Excel relay returned an invalid workbook list")
    connected = _connected_workbooks(listed)

    requested_workbook = mint_inputs["workbook"]
    if requested_workbook is not None:
      matches = [item for item in connected if _target_matches(item, requested_workbook)]
      if not matches:
        return None, _error(
          "no_excel_session",
          f"No connected Excel workbook matches '{requested_workbook}'",
          workbooks=_workbook_names(connected),
        )
      if len(matches) > 1:
        return None, _error(
          "ambiguous_target",
          f"Multiple connected Excel workbooks match '{requested_workbook}'",
          workbooks=_workbook_names(matches),
        )
      target_workbook = matches[0]
    else:
      if not connected:
        return None, _error("no_excel_session", "No connected Excel workbook session for this user")
      if len(connected) > 1:
        return None, _error(
          "ambiguous_target",
          "Multiple Excel workbooks are connected. Specify 'workbook'.",
          workbooks=_workbook_names(connected),
        )
      target_workbook = connected[0]

    target_excel_session_id = _as_non_empty_str(target_workbook.get("gateway_session_id"))
    if target_excel_session_id is None:
      return None, _error("relay_error", "Excel relay workbook is missing gateway_session_id")
    target_workbook_name = _workbook_label(target_workbook)

    request_id = uuid.uuid4().hex
    delegation_id = uuid.uuid4().hex
    created_at = utc_now()
    grant_window_seconds = mint_inputs["window_seconds"]
    grant = DelegationGrant(
      delegation_id=delegation_id,
      delegator_user_id=user_id,
      delegator_session_id=gateway_session_id,
      delegator_run_id=delegator_run_id,
      delegator_profile=delegator_profile,
      delegator_channel="excel",
      bound_excel_session_id=target_excel_session_id,
      bound_relay_request_id=request_id,
      bound_workbook=target_workbook_name,
      tool_class_ceiling=parsed_ceiling,
      args_predicate=mint_inputs["args_predicate"],
      window_seconds=grant_window_seconds,
      exclude_external_write_bypass=True,
      created_at=created_at,
      expires_at=created_at + timedelta(seconds=grant_window_seconds),
    )
    await approval_store.create_delegation_grant(grant)
    created_delegation_id = delegation_id

    resolved_relay_timeout_seconds = parsed_relay_timeout_seconds
    if resolved_relay_timeout_seconds is None:
      resolved_relay_timeout_seconds = _SUBMIT_TIMEOUT_SECONDS.get()
    if resolved_relay_timeout_seconds is None:
      resolved_relay_timeout_seconds = max(1, int(math.ceil(DEFAULT_MESSAGE_EXCEL_AGENT_TIMEOUT_SECONDS)))
    submit_tool_input = {
      "text": mint_inputs["text"],
      "force_compaction": mint_inputs["force_compaction"],
      "delegation_id": delegation_id,
    }
    if seed_history is not None:
      submit_tool_input["seed_history"] = seed_history
    # Stable model-selection intent flows through to relay admission/delivery
    # exactly like every other chat path; it is never silently discarded here.
    for selection_key, selection_value in (
      ("model_key", model_key),
      ("effort", effort),
      ("catalog_revision", catalog_revision),
    ):
      if selection_value is not _ABSENT:
        submit_tool_input[selection_key] = selection_value
    submitted = await relay.submit(
      tool_name="send_chat_message",
      tool_input=submit_tool_input,
      timeout=resolved_relay_timeout_seconds,
      target_session=target_excel_session_id,
      gateway_session_id=gateway_session_id,
      user_id=user_id,
      kind="chat",
      request_id=request_id,
    )
    if not isinstance(submitted, dict) or submitted.get("request_id") != request_id:
      await approval_store.revoke_delegation_grant(delegation_id)
      created_delegation_id = None
      return None, _error("relay_error", "Excel relay did not accept the pre-reserved request_id")

    return {
      "request_id": request_id,
      "delegation_id": delegation_id,
      "excel_session_id": target_excel_session_id,
      "workbook": target_workbook_name,
    }, None
  except Exception as exc:
    if created_delegation_id is not None:
      try:
        await approval_store.revoke_delegation_grant(created_delegation_id)
      except Exception:
        return None, _error(
          "internal_error",
          "Failed to revoke unsubmitted delegation grant",
        )
    if isinstance(exc, restart_exception_types):
      return None, _error(
        "relay_restart_in_progress",
        "Excel MCP relay restart in progress; retry after gateway restart",
      )
    return None, _error("internal_error", "Excel relay submission failed")


async def poll_result(
  *,
  relay: Any,
  request_id: str,
  excel_session_id: str,
  user_id: str,
  timeout_s: float = DEFAULT_MESSAGE_EXCEL_AGENT_TIMEOUT_SECONDS,
  poll_interval_seconds: float = 1.0,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  """Poll one Excel relay request as the bound Excel session."""

  if not _is_positive_number(timeout_s):
    return None, _error("invalid_input", "'timeout_s' must be a positive number")
  timeout_seconds = float(timeout_s)

  try:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    poll_interval = max(0.0, float(poll_interval_seconds))
    while True:
      # The relay records the inflight request's owner as the TARGET Excel session
      # (not the orchestrator), so poll as that session to pass _validate_owner_locked.
      # Same-user, server-side trusted call.
      status, response = await relay.result(
        request_id, gateway_session_id=excel_session_id, user_id=user_id
      )
      if not isinstance(response, dict):
        return None, _error("relay_error", "Excel relay returned an invalid result response", status=status)

      state = response.get("state")
      if state == "done":
        payload = response.get("result")
        if isinstance(payload, dict):
          return dict(payload), None
        return {"result": payload}, None

      if state in {"failed", "timeout"}:
        return None, _error(
          "excel_agent_failed" if state == "failed" else "timeout",
          (
            "Excel agent request failed"
            if state == "failed"
            else "Excel relay request timed out"
          ),
          request_id=request_id,
          status=status,
        )

      if status != "ok":
        return None, _error(
          "relay_error",
          "Excel relay result returned an invalid status",
          request_id=request_id,
        )

      if state != "pending":
        return None, _error(
          "relay_error",
          "Excel relay returned an invalid result state",
          request_id=request_id,
        )

      remaining = deadline - loop.time()
      if remaining <= 0:
        return None, _error(
          "timeout",
          "Timed out waiting for Excel agent response",
          request_id=request_id,
        )
      await asyncio.sleep(min(poll_interval, remaining))
  except Exception:
    return None, _error(
      "internal_error",
      "Excel relay result polling failed",
      request_id=request_id,
    )


def make_message_excel_agent_handler(
  *,
  relay: Any,
  approval_store: Any,
  user_id: str,
  gateway_session_id: str,
  delegator_profile: str = "orchestrator",
  delegator_run_id: str | None = None,
  default_window_seconds: int = 600,
  default_ceiling: frozenset[str] = frozenset({"read", "pure_transform", "artifact_write", "state_write"}),
  poll_interval_seconds: float = 1.0,
  relay_restart_exceptions: tuple[type[Exception], ...],
):
  """Build a standalone message_excel_agent relay dispatch handler."""

  restart_exception_types = _validate_relay_restart_exceptions(
    relay_restart_exceptions
  )

  async def _handle_message_excel_agent(tool_input: dict[str, Any], **kwargs: Any):
    _ = kwargs
    if not isinstance(tool_input, dict):
      return None, _error("invalid_input", "tool_input must be an object")

    mint_inputs, error = _validate_mint_inputs(
      text=tool_input.get("text"),
      workbook=tool_input.get("workbook"),
      force_compaction=tool_input.get("force_compaction", False),
      window_seconds=tool_input.get("window_seconds", default_window_seconds),
      args_predicate=tool_input.get("args_predicate"),
    )
    if error is not None:
      return None, error
    assert mint_inputs is not None

    timeout_s = tool_input.get("timeout_s", DEFAULT_MESSAGE_EXCEL_AGENT_TIMEOUT_SECONDS)
    if not _is_positive_number(timeout_s):
      return None, _error("invalid_input", "'timeout_s' must be a positive number")
    timeout_seconds = float(timeout_s)
    relay_timeout_seconds = max(1, int(math.ceil(timeout_seconds)))

    token = _SUBMIT_TIMEOUT_SECONDS.set(relay_timeout_seconds)
    try:
      submitted, error = await mint_and_submit(
        relay=relay,
        approval_store=approval_store,
        user_id=user_id,
        gateway_session_id=gateway_session_id,
        text=mint_inputs["text"],
        workbook=mint_inputs["workbook"],
        force_compaction=mint_inputs["force_compaction"],
        window_seconds=mint_inputs["window_seconds"],
        args_predicate=mint_inputs["args_predicate"],
        delegator_profile=delegator_profile,
        delegator_run_id=delegator_run_id,
        default_window_seconds=default_window_seconds,
        default_ceiling=default_ceiling,
        relay_timeout_seconds=relay_timeout_seconds,
        seed_history=tool_input.get("seed_history"),
        relay_restart_exceptions=restart_exception_types,
        **chat_selection_intent_kwargs(tool_input),
      )
    finally:
      _SUBMIT_TIMEOUT_SECONDS.reset(token)
    if error is not None:
      return None, error
    assert submitted is not None

    request_id = submitted["request_id"]
    delegation_id = submitted["delegation_id"]
    result, error = await poll_result(
      relay=relay,
      request_id=request_id,
      excel_session_id=submitted["excel_session_id"],
      user_id=user_id,
      timeout_s=timeout_seconds,
      poll_interval_seconds=poll_interval_seconds,
    )
    if error is not None:
      if error.get("request_id") == request_id:
        error["delegation_id"] = delegation_id
      return None, error
    assert result is not None
    result["delegation_id"] = delegation_id
    result["request_id"] = request_id
    return result, None

  return _handle_message_excel_agent


def make_message_excel_agent_tool_def() -> dict[str, Any]:
  return {
    "name": "message_excel_agent",
    "description": (
      "Drive the user's live Excel taskpane agent for one turn under a single-use "
      "delegated approval grant, routed through the Excel relay."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "text": {
          "type": "string",
          "description": "Message to send to the Excel taskpane agent.",
        },
        "workbook": {
          "type": "string",
          "description": "Workbook name, workbook session, or Excel gateway session to target.",
        },
        "force_compaction": {
          "type": "boolean",
          "description": "Ask the taskpane agent to compact before handling this turn.",
          "default": False,
        },
        "window_seconds": {
          "type": "integer",
          "minimum": 1,
          "description": "Delegated approval grant validity window in seconds.",
        },
        "args_predicate": {
          "type": ["object", "null"],
          "description": "Optional argument predicate that scopes delegated auto-approval.",
        },
        "model_key": {
          "type": "string",
          "description": (
            "Optional per-turn model selection for the Excel agent. Pass a stable "
            "model key exactly as returned by the session's eligible model choices; "
            "omit it to use the server default. Raw provider model IDs and "
            "provider:model aliases are refused with a typed error, never "
            "silently dropped."
          ),
        },
        "effort": {
          "type": "string",
          "description": (
            "Optional effort level supported by the selected model_key. Requires "
            "model_key; unsupported efforts are refused with a typed error. Omit "
            "it for the model's default effort."
          ),
        },
        "catalog_revision": {
          "type": "string",
          "description": (
            "Optional model-catalog revision observed when choosing model_key. "
            "Requires model_key. Concurrency context so stale selections can be "
            "detected; never authority over the server's catalog. Omit it when "
            "no revision was observed."
          ),
        },
        "timeout_s": {
          "type": "number",
          "exclusiveMinimum": 0,
          "description": "Maximum seconds to wait for the Excel agent relay result.",
        },
      },
      "required": ["text"],
    },
  }
