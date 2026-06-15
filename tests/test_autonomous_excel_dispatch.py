from __future__ import annotations

import asyncio
from typing import Any

import httpx

from agent_gateway import autonomous_excel_dispatch as dispatch


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
