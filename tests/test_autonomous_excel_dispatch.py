from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request

from agent_gateway import autonomous_excel_dispatch as dispatch
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.control_plane.orchestration import build_orchestration_router


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def _response(method: str, url: str, status_code: int, payload: dict[str, Any]) -> httpx.Response:
  return httpx.Response(
    status_code,
    json=payload,
    request=httpx.Request(method, url),
  )


class _FakeGatewayHttp:
  def __init__(self, responses: list[httpx.Response], *, default_get: httpx.Response | None = None) -> None:
    self.responses = list(responses)
    self.default_get = default_get
    self.requests: list[dict[str, Any]] = []
    self.client_kwargs: list[dict[str, Any]] = []

  def client_factory(self, **kwargs: Any):
    self.client_kwargs.append(dict(kwargs))
    return _FakeAsyncClient(self)

  def next_response(self, method: str) -> httpx.Response:
    if self.responses:
      return self.responses.pop(0)
    if method == "GET" and self.default_get is not None:
      return self.default_get
    raise AssertionError(f"No fake {method} response configured")


class _FakeAsyncClient:
  def __init__(self, transport: _FakeGatewayHttp) -> None:
    self.transport = transport

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    return False

  async def post(
    self,
    url: str,
    *,
    json: dict[str, Any],
    headers: dict[str, str] | None = None,
  ) -> httpx.Response:
    self.transport.requests.append(
      {
        "method": "POST",
        "url": url,
        "json": dict(json),
        "headers": dict(headers or {}),
      }
    )
    return self.transport.next_response("POST")

  async def get(
    self,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
  ) -> httpx.Response:
    self.transport.requests.append(
      {
        "method": "GET",
        "url": url,
        "params": dict(params or {}),
        "headers": dict(headers or {}),
      }
    )
    return self.transport.next_response("GET")


def test_handler_exchanges_key_posts_and_polls_until_done(monkeypatch) -> None:
  base = "https://gateway.example"
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        200,
        {"request_id": "request-1", "delegation_id": "delegation-1"},
      ),
      _response(
        "GET",
        f"{base}/api/orchestration/excel-dispatch/request-1",
        200,
        {"state": "pending"},
      ),
      _response(
        "GET",
        f"{base}/api/orchestration/excel-dispatch/request-1",
        200,
        {
          "state": "done",
          "result": {
            "text": "complete",
            "escalations": [{"approval_id": "approval-1"}],
          },
        },
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(
    handler(
      {
        "text": "Update the model",
        "workbook": "Budget.xlsx",
        "force_compaction": True,
        "window_seconds": 120,
        "args_predicate": {"sheet": "Budget"},
      }
    )
  )

  assert error is None
  assert result == {
    "text": "complete",
    "escalations": [{"approval_id": "approval-1"}],
    "delegation_id": "delegation-1",
    "request_id": "request-1",
  }
  assert transport.client_kwargs == [{"verify": False, "timeout": 300.0}]
  assert transport.requests[0] == {
    "method": "POST",
    "url": f"{base}/api/chat/init",
    "json": {
      "api_key": "api-key-1",
      "user_id": "mcp-subprocess",
      "context": {"channel": "mcp"},
    },
    "headers": {},
  }
  assert transport.requests[1] == {
    "method": "POST",
    "url": f"{base}/api/orchestration/excel-dispatch",
    "json": {
      "text": "Update the model",
      "workbook": "Budget.xlsx",
      "force_compaction": True,
      "window_seconds": 120,
      "args_predicate": {"sheet": "Budget"},
    },
    "headers": {"Authorization": "Bearer jwt-token"},
  }
  get_requests = transport.requests[2:]
  assert [request["method"] for request in get_requests] == ["GET", "GET"]
  assert all(request["url"] == f"{base}/api/orchestration/excel-dispatch/request-1" for request in get_requests)
  assert all(request["params"] == {"delegation_id": "delegation-1"} for request in get_requests)
  assert all(request["headers"] == {"Authorization": "Bearer jwt-token"} for request in get_requests)


def test_handler_forwards_stable_selection_intent_keys_to_dispatch_payload(monkeypatch) -> None:
  base = "https://gateway.example"
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        200,
        {"request_id": "request-1", "delegation_id": "delegation-1"},
      ),
      _response(
        "GET",
        f"{base}/api/orchestration/excel-dispatch/request-1",
        200,
        {"state": "done", "result": {"text": "complete"}},
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(
    handler(
      {
        "text": "Update the model",
        "model_key": "anthropic.claude-sonnet-5",
        "effort": "high",
        "catalog_revision": "2026-08-13.1",
      }
    )
  )

  assert error is None
  assert result is not None
  submit_json = transport.requests[1]["json"]
  assert submit_json["model_key"] == "anthropic.claude-sonnet-5"
  assert submit_json["effort"] == "high"
  assert submit_json["catalog_revision"] == "2026-08-13.1"
  assert "model" not in submit_json


def test_handler_omitted_selection_intent_sends_no_selection_keys(monkeypatch) -> None:
  base = "https://gateway.example"
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        200,
        {"request_id": "request-1", "delegation_id": "delegation-1"},
      ),
      _response(
        "GET",
        f"{base}/api/orchestration/excel-dispatch/request-1",
        200,
        {"state": "done", "result": {"text": "complete"}},
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(handler({"text": "Update the model"}))

  assert error is None
  assert result is not None
  submit_json = transport.requests[1]["json"]
  assert submit_json == {"text": "Update the model"}
  for selection_key in ("model", "model_key", "effort", "catalog_revision"):
    assert selection_key not in submit_json


class _SelectionFakeRelay:
  """Minimal relay double for end-to-end orchestration route tests."""

  def __init__(self) -> None:
    self.list_calls: list[dict[str, Any]] = []
    self.submissions: list[dict[str, Any]] = []

  async def list_workbooks(
    self,
    gateway_session_id: str | None = None,
    user_id: str | None = None,
  ) -> list[dict[str, Any]]:
    self.list_calls.append({"gateway_session_id": gateway_session_id, "user_id": user_id})
    return [
      {
        "name": "Budget.xlsx",
        "session": "workbook-session-1",
        "gateway_session_id": "excel-session-1",
        "detached": False,
      }
    ]

  async def submit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    self.submissions.append({"args": args, "kwargs": dict(kwargs)})
    return {"request_id": kwargs.get("request_id")}


def test_handler_raw_model_is_refused_typed_by_orchestration_route(
  monkeypatch,
  tmp_path: Path,
) -> None:
  """End-to-end: the handler forwards raw 'model' instead of dropping it, and the
  hardened orchestration route refuses it typed with HTTP 400
  chat_model_not_accepted before any relay submission."""

  base = "https://gateway.example"
  monkeypatch.setenv("EXCEL_ORCHESTRATION_DEV", "1")

  relay = _SelectionFakeRelay()
  app = FastAPI()
  app.state.gateway_approval_store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")

  class _AuthContext:
    user_id = "alice"
    session_id = "orchestrator-session"
    profile = "autonomous"

  def _authenticate(request: Request) -> _AuthContext:
    if request.headers.get("Authorization") != "Bearer helper-jwt":
      raise HTTPException(status_code=401, detail="missing token")
    return _AuthContext()

  app.include_router(
    build_orchestration_router(
      relay=relay,
      authenticate=_authenticate,
      relay_restart_exceptions=(RuntimeError,),
    )
  )

  real_async_client = httpx.AsyncClient

  def _client_factory(**kwargs: Any) -> httpx.AsyncClient:
    del kwargs
    return real_async_client(transport=httpx.ASGITransport(app=app), base_url=base)

  monkeypatch.setattr(
    dispatch,
    "_get_session_token",
    lambda api_key, *, gateway_base_url, tls_verify: "helper-jwt",
  )
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", _client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(
    handler({"text": "Update the model", "model": "claude-sonnet-4-6"})
  )

  assert result is None
  assert error is not None
  assert error["code"] == "chat_model_not_accepted"
  assert error["status_code"] == 400
  assert error["payload"]["code"] == "chat_model_not_accepted"
  assert relay.submissions == []


def test_handler_prefers_imported_gateway_session_helper(monkeypatch) -> None:
  base = "https://gateway.example"
  calls: list[dict[str, Any]] = []

  def _fake_get_session_token(api_key: str, *, gateway_base_url: str, tls_verify: bool) -> str:
    calls.append(
      {
        "api_key": api_key,
        "gateway_base_url": gateway_base_url,
        "tls_verify": tls_verify,
      }
    )
    return "helper-jwt"

  transport = _FakeGatewayHttp(
    [
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        200,
        {"request_id": "request-1", "delegation_id": "delegation-1"},
      ),
      _response(
        "GET",
        f"{base}/api/orchestration/excel-dispatch/request-1",
        200,
        {"state": "done", "result": {"text": "complete"}},
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", _fake_get_session_token)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(handler({"text": "Update the model"}))

  assert error is None
  assert result == {
    "text": "complete",
    "delegation_id": "delegation-1",
    "request_id": "request-1",
  }
  assert calls == [
    {
      "api_key": "api-key-1",
      "gateway_base_url": base,
      "tls_verify": False,
    }
  ]
  assert transport.requests[0]["url"] == f"{base}/api/orchestration/excel-dispatch"
  assert transport.requests[0]["headers"] == {"Authorization": "Bearer helper-jwt"}


def test_handler_returns_classified_error_on_submit_http_error(monkeypatch) -> None:
  base = "https://gateway.example"
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        401,
        {"message": "Unauthorized"},
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(handler({"text": "Update the model"}))

  assert result is None
  assert error is not None
  assert error["reason"] == "auth_failed"
  assert error["status_code"] == 401
  assert [request["method"] for request in transport.requests] == ["POST", "POST"]


def test_handler_preserves_retryable_relay_restart_submit_error(monkeypatch) -> None:
  base = "https://gateway.example"
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        503,
        {
          "code": "relay_restart_in_progress",
          "message": "Excel MCP relay restart in progress",
        },
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(handler({"text": "Update the model"}))

  assert result is None
  assert error is not None
  assert error["code"] == "relay_restart_in_progress"
  assert error["reason"] == "relay_restart_in_progress"
  assert error["status_code"] == 503
  assert error["payload"]["code"] == "relay_restart_in_progress"


def test_handler_classifies_focus_lost_status_error(monkeypatch) -> None:
  base = "https://gateway.example"
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        200,
        {"request_id": "request-1", "delegation_id": "delegation-1"},
      ),
      _response(
        "GET",
        f"{base}/api/orchestration/excel-dispatch/request-1",
        200,
        {
          "state": "failed",
          "error": {
            "code": "focus_lost",
            "message": "Bring the workbook to the front and retry.",
          },
        },
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(handler({"text": "Update the model"}))

  assert result is None
  assert error is not None
  assert error["code"] == "excel_agent_failed"
  assert error["reason"] == "focus_lost"
  assert error["state"] == "failed"
  assert error["payload"]["error"]["code"] == "focus_lost"


def test_handler_classifies_taskpane_tool_timeout_status_error(monkeypatch) -> None:
  base = "https://gateway.example"
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        200,
        {"request_id": "request-1", "delegation_id": "delegation-1"},
      ),
      _response(
        "GET",
        f"{base}/api/orchestration/excel-dispatch/request-1",
        200,
        {
          "state": "failed",
          "error": {
            "code": "taskpane_tool_timeout",
            "message": "Check workbook state before retrying.",
          },
        },
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(handler({"text": "Update the model"}))

  assert result is None
  assert error is not None
  assert error["code"] == "excel_agent_failed"
  assert error["reason"] == "taskpane_tool_timeout"
  assert error["state"] == "failed"
  assert error["payload"]["error"]["code"] == "taskpane_tool_timeout"


def test_handler_classifies_chat_relay_disabled_status_error(monkeypatch) -> None:
  base = "https://gateway.example"
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        200,
        {"request_id": "request-1", "delegation_id": "delegation-1"},
      ),
      _response(
        "GET",
        f"{base}/api/orchestration/excel-dispatch/request-1",
        200,
        {
          "state": "failed",
          "error": {
            "code": "chat_relay_disabled",
            "message": "Excel chat relay is disabled in this taskpane bundle.",
          },
        },
      ),
    ]
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(handler({"text": "Update the model"}))

  assert result is None
  assert error is not None
  assert error["code"] == "excel_agent_failed"
  assert error["reason"] == "chat_relay_disabled"
  assert error["state"] == "failed"
  assert error["payload"]["error"]["code"] == "chat_relay_disabled"


def test_handler_returns_timeout_when_status_never_done(monkeypatch) -> None:
  base = "https://gateway.example"
  pending = _response(
    "GET",
    f"{base}/api/orchestration/excel-dispatch/request-1",
    200,
    {"state": "pending"},
  )
  transport = _FakeGatewayHttp(
    [
      _response("POST", f"{base}/api/chat/init", 200, {"session_token": "jwt-token"}),
      _response(
        "POST",
        f"{base}/api/orchestration/excel-dispatch",
        200,
        {"request_id": "request-1", "delegation_id": "delegation-1"},
      ),
      pending,
    ],
    default_get=pending,
  )
  monkeypatch.setattr(dispatch, "_get_session_token", None)
  monkeypatch.setattr(dispatch.httpx, "AsyncClient", transport.client_factory)

  handler = dispatch.make_autonomous_message_excel_agent_handler(
    gateway_url=base,
    gateway_api_key="api-key-1",
    poll_interval_seconds=0.001,
    tls_verify=False,
  )
  result, error = _run(handler({"text": "Update the model", "timeout_s": 0.001}))

  assert result is None
  assert error is not None
  assert error["code"] == "timeout"
  assert error["reason"] == "dispatch_timeout"
  assert error["request_id"] == "request-1"
  assert error["delegation_id"] == "delegation-1"
  assert any(request["method"] == "GET" for request in transport.requests)
