from __future__ import annotations

import asyncio
import importlib.util
import ipaddress
import math
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse, urlunparse

import httpx

DEFAULT_TIMEOUT_SECONDS = 300.0
_ADDIN_DISPATCH_MODULE_NAMES = frozenset({"api", "api.addin_dispatch"})


def _fallback_derive_gateway_base_url(backend_url: str) -> str:
  parsed = urlparse(backend_url)
  if not (parsed.scheme and parsed.netloc):
    return backend_url.strip().rstrip("/")
  segments = [segment for segment in parsed.path.split("/") if segment]
  if len(segments) >= 3 and segments[-3:-1] == ["api", "mcp"]:
    segments = segments[:-3]
  new_path = "/" + "/".join(segments) if segments else ""
  return urlunparse((parsed.scheme, parsed.netloc, new_path, "", "", "")).rstrip("/")


def _fallback_classify_addin_dispatch_error(error: BaseException | str) -> str:
  message = str(error).strip().lower()
  if not message:
    return "dispatch_failed"
  if "addin_dispatch_api_key is not set" in message:
    return "api_key_missing"
  if "no active excel taskpane connection" in message:
    return "no_taskpane_registered"
  if "taskpane disconnected" in message or "targeted workbook is disconnected" in message:
    return "taskpane_disconnected"
  if "chat_relay_disabled" in message or "chat relay is disabled" in message:
    return "chat_relay_disabled"
  if "focus_lost" in message or (
    "lost focus" in message and ("workbook" in message or "multiple workbooks" in message)
  ):
    return "focus_lost"
  if "taskpane_tool_timeout" in message:
    return "taskpane_tool_timeout"
  if "401" in message or "unauthorized" in message or "forbidden" in message or "403" in message:
    return "auth_failed"
  if "connection refused" in message or "couldn't connect" in message or "connecterror" in message:
    return "gateway_unreachable"
  if "timed out" in message or "timeout" in message:
    return "dispatch_timeout"
  return "dispatch_failed"


def _load_addin_dispatch_helpers() -> tuple[Callable[[str], str], Callable[[BaseException | str], str]]:
  try:
    from api.addin_dispatch import _derive_gateway_base_url, classify_addin_dispatch_error

    return _derive_gateway_base_url, classify_addin_dispatch_error
  except ModuleNotFoundError as exc:
    if exc.name not in _ADDIN_DISPATCH_MODULE_NAMES:
      raise
    pass

  helper_path = Path(__file__).resolve().parents[3] / "api" / "addin_dispatch.py"
  if not helper_path.exists():
    return _fallback_derive_gateway_base_url, _fallback_classify_addin_dispatch_error
  spec = importlib.util.spec_from_file_location("_autonomous_excel_addin_dispatch", helper_path)
  if spec is None or spec.loader is None:
    return _fallback_derive_gateway_base_url, _fallback_classify_addin_dispatch_error
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module._derive_gateway_base_url, module.classify_addin_dispatch_error


_derive_gateway_base_url, _classify_addin_dispatch_error = _load_addin_dispatch_helpers()

try:
  from excel_mcp._gateway_session import (
    default_tls_verify as _gateway_default_tls_verify,
    get_session_token as _get_session_token,
  )
except Exception:  # pragma: no cover - exercised when excel-mcp is not installed.
  _gateway_default_tls_verify = None
  _get_session_token = None


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
  payload: dict[str, Any] = {"code": code, "message": message}
  payload.update(details)
  return payload


def _classified_error(
  message: str,
  *,
  code: str | None = None,
  status_code: int | None = None,
  request_id: str | None = None,
  delegation_id: str | None = None,
  state: str | None = None,
  payload: Any = None,
) -> dict[str, Any]:
  classification_source = message if payload is None else f"{message} {payload}"
  reason = _classify_addin_dispatch_error(classification_source)
  details: dict[str, Any] = {"reason": reason}
  if status_code is not None:
    details["status_code"] = status_code
  if request_id is not None:
    details["request_id"] = request_id
  if delegation_id is not None:
    details["delegation_id"] = delegation_id
  if state is not None:
    details["state"] = state
  if payload is not None:
    details["payload"] = payload
  return _error(code or reason, message, **details)


def _is_positive_number(value: Any) -> bool:
  return (
    isinstance(value, (int, float))
    and not isinstance(value, bool)
    and math.isfinite(float(value))
    and float(value) > 0
  )


def _normalize_gateway_url(raw: str) -> str:
  candidate = str(raw or "").strip()
  if not candidate:
    return ""
  if "://" not in candidate:
    candidate = f"https://{candidate}"
  return candidate.rstrip("/")


def _is_loopback_host(host: str | None) -> bool:
  if not host:
    return False
  if host.lower() in {"localhost", "::1"}:
    return True
  try:
    return ipaddress.ip_address(host).is_loopback
  except ValueError:
    return False


def _fallback_default_tls_verify(gateway_base_url: str) -> bool:
  parsed = urlparse(_normalize_gateway_url(gateway_base_url))
  return not _is_loopback_host(parsed.hostname)


def _resolve_tls_verify(gateway_base_url: str, tls_verify: bool | None) -> bool:
  if tls_verify is not None:
    return bool(tls_verify)
  if _gateway_default_tls_verify is not None:
    return bool(_gateway_default_tls_verify(gateway_base_url))
  return _fallback_default_tls_verify(gateway_base_url)


def _gateway_api_base_url(base_url: str) -> str:
  stripped = base_url.rstrip("/")
  return stripped if stripped.endswith("/api") else f"{stripped}/api"


def _api_url(base_url: str, path: str) -> str:
  return f"{_gateway_api_base_url(base_url)}/{path.lstrip('/')}"


def _response_payload(response: httpx.Response) -> Any:
  try:
    return response.json()
  except Exception:
    text = getattr(response, "text", "")
    return text if text else None


def _response_error_message(response: httpx.Response, *, operation: str) -> tuple[str, Any]:
  payload = _response_payload(response)
  status_code = getattr(response, "status_code", "unknown")
  if isinstance(payload, dict):
    detail = payload.get("message") or payload.get("error") or payload.get("detail") or payload
  elif payload is not None:
    detail = payload
  else:
    detail = ""
  suffix = f": {detail}" if detail else ""
  return f"{operation} HTTP {status_code}{suffix}", payload


async def _inline_exchange_session_token(
  *,
  client: httpx.AsyncClient,
  api_key: str,
  gateway_base_url: str,
) -> str:
  response = await client.post(
    _api_url(gateway_base_url, "chat/init"),
    json={
      "api_key": api_key,
      "user_id": "mcp-subprocess",
      "context": {"channel": "mcp"},
    },
  )
  if response.status_code >= 400:
    message, payload = _response_error_message(response, operation="Gateway /api/chat/init")
    raise RuntimeError(message if payload is None else f"{message}")
  payload = response.json()
  token = str(payload.get("session_token") or "").strip()
  if not token:
    raise RuntimeError("Gateway /api/chat/init did not return session_token")
  return token


async def _exchange_session_token(
  *,
  client: httpx.AsyncClient,
  api_key: str,
  gateway_base_url: str,
  tls_verify: bool,
) -> str:
  if _get_session_token is not None:
    return await asyncio.to_thread(
      _get_session_token,
      api_key,
      gateway_base_url=gateway_base_url,
      tls_verify=tls_verify,
    )
  return await _inline_exchange_session_token(
    client=client,
    api_key=api_key,
    gateway_base_url=gateway_base_url,
  )


def _dispatch_payload(tool_input: dict[str, Any]) -> dict[str, Any] | None:
  text = tool_input.get("text")
  if not isinstance(text, str) or not text.strip():
    return None

  payload: dict[str, Any] = {"text": text}
  for name in ("workbook", "force_compaction", "window_seconds", "args_predicate"):
    if name in tool_input:
      payload[name] = tool_input[name]
  return payload


def make_autonomous_message_excel_agent_handler(
  *,
  gateway_url: str,
  gateway_api_key: str,
  poll_interval_seconds: float = 1.0,
  tls_verify: bool | None = None,
):
  """Build a local message_excel_agent handler backed by gateway orchestration HTTP."""

  gateway_base_url = _derive_gateway_base_url(_normalize_gateway_url(gateway_url))
  api_key = str(gateway_api_key or "").strip()
  verify = _resolve_tls_verify(gateway_base_url, tls_verify)
  poll_interval = max(0.0, float(poll_interval_seconds))

  async def _handle_message_excel_agent(tool_input: dict[str, Any], **kwargs: Any):
    _ = kwargs
    try:
      if not gateway_base_url:
        return None, _error("invalid_config", "gateway_url is required")
      if not api_key:
        return None, _error("invalid_config", "gateway_api_key is required")
      if not isinstance(tool_input, dict):
        return None, _error("invalid_input", "tool_input must be an object")

      timeout_s = tool_input.get("timeout_s", DEFAULT_TIMEOUT_SECONDS)
      if not _is_positive_number(timeout_s):
        return None, _error("invalid_input", "'timeout_s' must be a positive number")
      timeout_seconds = float(timeout_s)

      payload = _dispatch_payload(tool_input)
      if payload is None:
        return None, _error("invalid_input", "'text' is required")

      async with httpx.AsyncClient(verify=verify, timeout=timeout_seconds) as client:
        token = await _exchange_session_token(
          client=client,
          api_key=api_key,
          gateway_base_url=gateway_base_url,
          tls_verify=verify,
        )
        headers = {"Authorization": f"Bearer {token}"}
        submit_response = await client.post(
          _api_url(gateway_base_url, "orchestration/excel-dispatch"),
          json=payload,
          headers=headers,
        )
        if submit_response.status_code >= 400:
          message, response_payload = _response_error_message(
            submit_response,
            operation="Excel dispatch submit",
          )
          return None, _classified_error(
            message,
            status_code=submit_response.status_code,
            payload=response_payload,
          )

        submitted = submit_response.json()
        request_id = str(submitted.get("request_id") or "").strip()
        delegation_id = str(submitted.get("delegation_id") or "").strip()
        if not request_id or not delegation_id:
          return None, _error(
            "dispatch_failed",
            "Excel dispatch submit did not return request_id and delegation_id",
            payload=submitted,
          )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        status_url = _api_url(
          gateway_base_url,
          f"orchestration/excel-dispatch/{quote(request_id, safe='')}",
        )
        while True:
          status_response = await client.get(
            status_url,
            params={"delegation_id": delegation_id},
            headers=headers,
          )
          if status_response.status_code >= 400:
            message, response_payload = _response_error_message(
              status_response,
              operation="Excel dispatch status",
            )
            return None, _classified_error(
              message,
              status_code=status_response.status_code,
              request_id=request_id,
              delegation_id=delegation_id,
              payload=response_payload,
            )

          status_payload = status_response.json()
          state = status_payload.get("state")
          if state == "done":
            result_payload = status_payload.get("result")
            if isinstance(result_payload, dict):
              result = dict(result_payload)
            else:
              result = {"result": result_payload}
            result["delegation_id"] = delegation_id
            result["request_id"] = request_id
            return result, None

          if state in {"failed", "timeout"}:
            error_payload = status_payload.get("error")
            message = ""
            if isinstance(error_payload, dict):
              message = str(error_payload.get("message") or error_payload.get("error") or "")
            elif error_payload is not None:
              message = str(error_payload)
            if not message:
              message = f"Excel dispatch request {state}"
            return None, _classified_error(
              message,
              code="excel_agent_failed" if state == "failed" else "timeout",
              request_id=request_id,
              delegation_id=delegation_id,
              state=str(state),
              payload=status_payload,
            )

          if state != "pending":
            return None, _error(
              "dispatch_failed",
              f"Excel dispatch returned unknown state '{state}'",
              request_id=request_id,
              delegation_id=delegation_id,
              payload=status_payload,
            )

          remaining = deadline - loop.time()
          if remaining <= 0:
            return None, _classified_error(
              "Timed out waiting for Excel agent response",
              code="timeout",
              request_id=request_id,
              delegation_id=delegation_id,
            )
          await asyncio.sleep(min(poll_interval, remaining))
    except Exception as exc:
      return None, _classified_error(str(exc) or exc.__class__.__name__)

  return _handle_message_excel_agent


__all__ = ["make_autonomous_message_excel_agent_handler"]
