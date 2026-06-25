from __future__ import annotations

from typing import Any, Callable, Dict

AGENT_TELEMETRY_RUN_ID_ENV = "AGENT_TELEMETRY_RUN_ID"
AGENT_TELEMETRY_REQUEST_ID_ENV = "AGENT_TELEMETRY_REQUEST_ID"
AGENT_TELEMETRY_TOOL_CALL_ID_ENV = "AGENT_TELEMETRY_TOOL_CALL_ID"

AGENT_TELEMETRY_RUN_ID_HEADER = "X-Agent-Telemetry-Run-Id"
AGENT_TELEMETRY_REQUEST_ID_HEADER = "X-Agent-Telemetry-Request-Id"
AGENT_TELEMETRY_TOOL_CALL_ID_HEADER = "X-Agent-Telemetry-Tool-Call-Id"

AGENT_TELEMETRY_ENV_TO_HEADER = (
  (AGENT_TELEMETRY_RUN_ID_ENV, AGENT_TELEMETRY_RUN_ID_HEADER),
  (AGENT_TELEMETRY_REQUEST_ID_ENV, AGENT_TELEMETRY_REQUEST_ID_HEADER),
  (AGENT_TELEMETRY_TOOL_CALL_ID_ENV, AGENT_TELEMETRY_TOOL_CALL_ID_HEADER),
)


def _clean_text(value: Any) -> str | None:
  text = str(value or "").strip()
  return text or None


def apply_agent_telemetry_env(
  env: Dict[str, str],
  *,
  tool_call_id: Any,
  request_id: Any = None,
  run_id: Any = None,
) -> Dict[str, str]:
  for env_name, _header_name in AGENT_TELEMETRY_ENV_TO_HEADER:
    env.pop(env_name, None)

  resolved_tool_call_id = _clean_text(tool_call_id)
  if resolved_tool_call_id:
    env[AGENT_TELEMETRY_TOOL_CALL_ID_ENV] = resolved_tool_call_id

  resolved_request_id = _clean_text(request_id)
  if resolved_request_id:
    env[AGENT_TELEMETRY_REQUEST_ID_ENV] = resolved_request_id

  resolved_run_id = _clean_text(run_id)
  if resolved_run_id:
    env[AGENT_TELEMETRY_RUN_ID_ENV] = resolved_run_id

  return env


def apply_agent_telemetry_env_from_tool_ctx(
  env: Dict[str, str],
  tool_ctx: Any | None,
) -> Dict[str, str]:
  if tool_ctx is None:
    return env
  return apply_agent_telemetry_env(
    env,
    tool_call_id=getattr(tool_ctx, "tool_call_id", None),
    request_id=getattr(tool_ctx, "request_id", None),
    run_id=getattr(tool_ctx, "run_id", None),
  )


def make_prepare_env_with_agent_telemetry(
  prepare_env: Callable[[Dict[str, str]], Dict[str, str] | None] | None,
  tool_ctx: Any | None,
) -> Callable[[Dict[str, str]], Dict[str, str]]:
  def wrapped(env: Dict[str, str]) -> Dict[str, str]:
    if prepare_env is not None:
      prepared = prepare_env(env)
      if prepared is not None:
        env = prepared
    return apply_agent_telemetry_env_from_tool_ctx(env, tool_ctx)

  return wrapped


__all__ = [
  "AGENT_TELEMETRY_ENV_TO_HEADER",
  "AGENT_TELEMETRY_REQUEST_ID_ENV",
  "AGENT_TELEMETRY_REQUEST_ID_HEADER",
  "AGENT_TELEMETRY_RUN_ID_ENV",
  "AGENT_TELEMETRY_RUN_ID_HEADER",
  "AGENT_TELEMETRY_TOOL_CALL_ID_ENV",
  "AGENT_TELEMETRY_TOOL_CALL_ID_HEADER",
  "apply_agent_telemetry_env",
  "apply_agent_telemetry_env_from_tool_ctx",
  "make_prepare_env_with_agent_telemetry",
]
