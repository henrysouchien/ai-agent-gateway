from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from agent_gateway.approval_policy import DelegationGrant, utc_now
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.control_plane.orchestration import build_orchestration_router


@dataclass(frozen=True)
class _AuthContext:
  user_id: str
  session_id: str
  profile: str | None = None


class _FakeRelay:
  def __init__(
    self,
    *,
    workbooks: list[dict[str, Any]] | None = None,
    result_status: str = "ok",
    result_response: dict[str, Any] | None = None,
  ) -> None:
    self.workbooks = list(workbooks or [])
    self.result_status = result_status
    self.result_response = result_response or {"state": "pending"}
    self.list_calls: list[dict[str, Any]] = []
    self.submissions: list[dict[str, Any]] = []
    self.result_calls: list[dict[str, Any]] = []

  async def list_workbooks(
    self,
    gateway_session_id: str | None = None,
    user_id: str | None = None,
  ) -> list[dict[str, Any]]:
    self.list_calls.append({"gateway_session_id": gateway_session_id, "user_id": user_id})
    return list(self.workbooks)

  async def submit(
    self,
    tool_name: str,
    tool_input: dict[str, Any],
    timeout: int,
    target_session: str | None = None,
    *,
    gateway_session_id: str | None = None,
    user_id: str | None = None,
    kind: str = "tool",
    request_id: str | None = None,
  ) -> dict[str, Any]:
    self.submissions.append(
      {
        "tool_name": tool_name,
        "tool_input": dict(tool_input),
        "timeout": timeout,
        "target_session": target_session,
        "gateway_session_id": gateway_session_id,
        "user_id": user_id,
        "kind": kind,
        "request_id": request_id,
      }
    )
    return {"request_id": request_id}

  async def result(
    self,
    request_id: str,
    *,
    gateway_session_id: str | None = None,
    user_id: str | None = None,
  ) -> tuple[str, dict[str, Any]]:
    self.result_calls.append(
      {"request_id": request_id, "gateway_session_id": gateway_session_id, "user_id": user_id}
    )
    return self.result_status, dict(self.result_response)


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def _headers(*, user_id: str = "alice", session_id: str = "orchestrator-session") -> dict[str, str]:
  return {
    "Authorization": "Bearer test-token",
    "x-user-id": user_id,
    "x-session-id": session_id,
  }


def _make_app(tmp_path: Path, relay: _FakeRelay) -> tuple[FastAPI, SQLiteApprovalStore]:
  app = FastAPI()
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  app.state.gateway_approval_store = store

  def _authenticate(request: Request) -> _AuthContext:
    if request.headers.get("Authorization") != "Bearer test-token":
      raise HTTPException(status_code=401, detail="missing test token")
    return _AuthContext(
      user_id=request.headers.get("x-user-id", "alice"),
      session_id=request.headers.get("x-session-id", "orchestrator-session"),
      profile=request.headers.get("x-profile"),
    )

  app.include_router(build_orchestration_router(relay=relay, authenticate=_authenticate))
  return app, store


def _create_delegation_grant(
  store: SQLiteApprovalStore,
  *,
  delegation_id: str = "delegation-1",
  delegator_user_id: str = "alice",
  request_id: str = "request-1",
  excel_session_id: str = "excel-session-1",
) -> DelegationGrant:
  created_at = utc_now()
  grant = DelegationGrant(
    delegation_id=delegation_id,
    delegator_user_id=delegator_user_id,
    delegator_run_id=None,
    delegator_session_id="orchestrator-session",
    delegator_profile="autonomous",
    delegator_channel="excel",
    bound_excel_session_id=excel_session_id,
    bound_relay_request_id=request_id,
    bound_workbook="Budget.xlsx",
    tool_class_ceiling=frozenset({"read", "pure_transform", "artifact_write", "state_write"}),
    args_predicate=None,
    window_seconds=300,
    created_at=created_at,
    expires_at=created_at + timedelta(seconds=300),
  )
  return _run(store.create_delegation_grant(grant))


def test_post_dev_gate_mints_grant_and_submits(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setenv("EXCEL_ORCHESTRATION_DEV", "1")
  relay = _FakeRelay(
    workbooks=[
      {
        "name": "Budget.xlsx",
        "session": "workbook-session-1",
        "gateway_session_id": "excel-session-1",
        "detached": False,
      }
    ]
  )
  app, store = _make_app(tmp_path, relay)

  with TestClient(app) as client:
    response = client.post(
      "/api/orchestration/excel-dispatch",
      headers=_headers(user_id="alice"),
      json={
        "text": "Update the model",
        "workbook": "Budget.xlsx",
        "force_compaction": True,
        "window_seconds": 120,
        "args_predicate": {"sheet": "Budget"},
        "timeout_s": 1,
        "delegator_user_id": "mallory",
      },
    )

  assert response.status_code == 200, response.text
  body = response.json()
  assert set(body) == {"request_id", "delegation_id"}

  grant = _run(store.get_delegation_grant(body["delegation_id"]))
  assert grant is not None
  assert grant.delegator_user_id == "alice"
  assert grant.delegator_session_id == "orchestrator-session"
  assert grant.delegator_run_id is None
  assert grant.delegator_profile == "autonomous"
  assert grant.bound_excel_session_id == "excel-session-1"
  assert grant.bound_relay_request_id == body["request_id"]
  assert grant.bound_workbook == "Budget.xlsx"
  assert grant.args_predicate == {"sheet": "Budget"}
  assert grant.window_seconds == 120

  assert relay.list_calls == [
    {"gateway_session_id": "orchestrator-session", "user_id": "alice"}
  ]
  assert len(relay.submissions) == 1
  submission = relay.submissions[0]
  assert submission["tool_name"] == "send_chat_message"
  assert submission["target_session"] == "excel-session-1"
  assert submission["gateway_session_id"] == "orchestrator-session"
  assert submission["user_id"] == "alice"
  assert submission["kind"] == "chat"
  assert submission["request_id"] == body["request_id"]
  assert submission["tool_input"] == {
    "text": "Update the model",
    "force_compaction": True,
    "delegation_id": body["delegation_id"],
  }
  assert relay.result_calls == []


def test_post_gate_off_returns_403(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.delenv("EXCEL_ORCHESTRATION_DEV", raising=False)
  relay = _FakeRelay(
    workbooks=[
      {
        "name": "Budget.xlsx",
        "session": "workbook-session-1",
        "gateway_session_id": "excel-session-1",
        "detached": False,
      }
    ]
  )
  app, _store = _make_app(tmp_path, relay)

  with TestClient(app) as client:
    response = client.post(
      "/api/orchestration/excel-dispatch",
      headers=_headers(),
      json={"text": "Update the model"},
    )

  assert response.status_code == 403
  assert relay.list_calls == []
  assert relay.submissions == []


def test_get_gate_off_returns_403(monkeypatch, tmp_path: Path) -> None:
  # Even with a valid grant + matching request_id + correct delegator, the GET must
  # 403 when the dev flag is off, and must not poll the relay.
  monkeypatch.delenv("EXCEL_ORCHESTRATION_DEV", raising=False)
  relay = _FakeRelay()
  app, store = _make_app(tmp_path, relay)
  _create_delegation_grant(store, delegation_id="delegation-1", request_id="request-1")

  with TestClient(app) as client:
    response = client.get(
      "/api/orchestration/excel-dispatch/request-1",
      params={"delegation_id": "delegation-1"},
      headers=_headers(user_id="alice"),
    )

  assert response.status_code == 403
  assert relay.result_calls == []


def test_get_authorizes_by_delegation_chain_and_polls_bound_excel_session(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setenv("EXCEL_ORCHESTRATION_DEV", "1")
  relay = _FakeRelay(
    result_response={
      "state": "done",
      "result": {
        "text": "complete",
        "escalations": [{"approval_id": "approval-1"}],
      },
    }
  )
  app, store = _make_app(tmp_path, relay)
  _create_delegation_grant(store, delegation_id="delegation-1", request_id="request-1")

  with TestClient(app) as client:
    response = client.get(
      "/api/orchestration/excel-dispatch/request-1",
      params={"delegation_id": "delegation-1"},
      headers=_headers(user_id="alice"),
    )

  assert response.status_code == 200, response.text
  assert response.json() == {
    "state": "done",
    "result": {
      "text": "complete",
      "escalations": [{"approval_id": "approval-1"}],
    },
  }
  assert relay.result_calls == [
    {"request_id": "request-1", "gateway_session_id": "excel-session-1", "user_id": "alice"}
  ]


def test_get_rejects_wrong_delegator(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setenv("EXCEL_ORCHESTRATION_DEV", "1")
  relay = _FakeRelay()
  app, store = _make_app(tmp_path, relay)
  _create_delegation_grant(
    store,
    delegation_id="delegation-1",
    delegator_user_id="alice",
    request_id="request-1",
  )

  with TestClient(app) as client:
    response = client.get(
      "/api/orchestration/excel-dispatch/request-1",
      params={"delegation_id": "delegation-1"},
      headers=_headers(user_id="bob"),
    )

  assert response.status_code == 403
  assert relay.result_calls == []


def test_get_rejects_mismatched_request_id(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setenv("EXCEL_ORCHESTRATION_DEV", "1")
  relay = _FakeRelay()
  app, store = _make_app(tmp_path, relay)
  _create_delegation_grant(store, delegation_id="delegation-1", request_id="other-request")

  with TestClient(app) as client:
    response = client.get(
      "/api/orchestration/excel-dispatch/request-1",
      params={"delegation_id": "delegation-1"},
      headers=_headers(user_id="alice"),
    )

  assert response.status_code == 404
  assert relay.result_calls == []


def test_get_requires_delegation_id(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setenv("EXCEL_ORCHESTRATION_DEV", "1")
  relay = _FakeRelay()
  app, _store = _make_app(tmp_path, relay)

  with TestClient(app) as client:
    response = client.get(
      "/api/orchestration/excel-dispatch/request-1",
      headers=_headers(user_id="alice"),
    )

  assert response.status_code == 400
  assert relay.result_calls == []
