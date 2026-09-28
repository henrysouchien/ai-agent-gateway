from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient


PKG_DIR = Path(__file__).resolve().parents[2]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_preferences import ApprovalPreferenceStore
from agent_gateway.control_plane.approvals import build_approvals_router


def _client(tmp_path: Path, *, kind: str = "control") -> TestClient:
  session = SimpleNamespace(kind=kind, owner_user_id="alice", channel="excel")
  auth = SimpleNamespace(verify_token=lambda _token: session)
  app = FastAPI()
  app.include_router(
    build_approvals_router(auth=auth),  # type: ignore[arg-type]
    prefix="/api/control",
  )
  app.state.gateway_approval_preference_store = ApprovalPreferenceStore(
    tmp_path / "approval-preferences.sqlite3"
  )
  return TestClient(app)


def test_unset_preference_reads_as_the_product_default(tmp_path: Path) -> None:
  with _client(tmp_path) as client:
    response = client.get(
      "/api/control/approvals/preference",
      headers={"Authorization": "Bearer t"},
    )

  assert response.status_code == 200
  assert response.json()["preference"] == {
    "user_id": "alice",
    "preference": "auto_approve_all_but_trades",
    "updated_at": None,
    "source": "default",
  }


def test_setting_the_opt_out_is_durable_and_readable(tmp_path: Path) -> None:
  with _client(tmp_path) as client:
    headers = {"Authorization": "Bearer t"}
    written = client.put(
      "/api/control/approvals/preference",
      json={"preference": "request_user_approval"},
      headers=headers,
    )
    read_back = client.get("/api/control/approvals/preference", headers=headers)

  assert written.status_code == 200
  assert written.json()["preference"]["preference"] == "request_user_approval"
  assert read_back.json()["preference"]["preference"] == "request_user_approval"
  assert read_back.json()["preference"]["source"] == "stored"


def test_an_unsupported_preference_is_refused(tmp_path: Path) -> None:
  with _client(tmp_path) as client:
    headers = {"Authorization": "Bearer t"}
    response = client.put(
      "/api/control/approvals/preference",
      json={"preference": "approve_everything"},
      headers=headers,
    )
    read_back = client.get("/api/control/approvals/preference", headers=headers)

  assert response.status_code == 400
  assert read_back.json()["preference"]["preference"] == "auto_approve_all_but_trades"


def test_a_chat_bearer_cannot_set_another_channel_policy(tmp_path: Path) -> None:
  with _client(tmp_path, kind="chat") as client:
    response = client.put(
      "/api/control/approvals/preference",
      json={"preference": "request_user_approval"},
      headers={"Authorization": "Bearer t"},
    )

  assert response.status_code == 401
