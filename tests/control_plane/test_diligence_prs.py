from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.control_plane import diligence_prs as diligence_prs_module


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def _merge_payload(**overrides: Any) -> dict[str, Any]:
  payload = {
    "confirm_merge": True,
    "expected_ticker": "msft",
    "expected_workspace_id": "batch_7_MSFT",
    "expected_proposal_ids": ["proposal-1", "proposal-2"],
    "expected_research_file_id": 42,
    "expected_handoff_id": 9,
  }
  payload.update(overrides)
  return payload


def test_merge_diligence_pr_route_delegates_to_s4m_handler(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}

  def handler_factory(session: Any):
    captured["session"] = session

    async def handler(tool_input: dict[str, Any]):
      captured["tool_input"] = tool_input
      return {
        "tool": "merge_diligence_pr",
        "mutation_mode": "apply",
        "status": "success",
        "pr_id": tool_input["pr_id"],
        "pr": {"state": "merged"},
      }, None

    return handler

  monkeypatch.setattr(diligence_prs_module, "_merge_handler_for_session", handler_factory)

  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 200, response.text
  assert response.json()["pr"]["state"] == "merged"
  assert captured["session"].user_id == "tui-user"
  assert captured["tool_input"] == {
    "pr_id": "dpr_7_MSFT_abc",
    "confirm_merge": True,
    "expected_ticker": "MSFT",
    "expected_workspace_id": "batch_7_MSFT",
    "expected_proposal_ids": ["proposal-1", "proposal-2"],
    "expected_research_file_id": 42,
    "expected_handoff_id": 9,
    "process_outbox": True,
  }


def test_merge_diligence_pr_route_returns_conflict_for_blocked_merge(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  async def handler(_tool_input: dict[str, Any]):
    return {
      "tool": "merge_diligence_pr",
      "mutation_mode": "apply",
      "status": "blocked",
      "pr_id": "dpr_7_MSFT_abc",
      "blockers": [{"code": "base_hash_stale"}],
    }, None

  monkeypatch.setattr(diligence_prs_module, "_merge_handler_for_session", lambda _session: handler)

  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 409, response.text
  assert response.json()["blockers"] == [{"code": "base_hash_stale"}]


def test_merge_diligence_pr_route_maps_missing_pr_to_not_found(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  async def handler(_tool_input: dict[str, Any]):
    return None, {
      "code": "diligence_pr_not_found",
      "message": "diligence PR not found",
      "pr_id": "dpr_missing",
    }

  monkeypatch.setattr(diligence_prs_module, "_merge_handler_for_session", lambda _session: handler)

  response = client.post(
    "/api/control/diligence-prs/dpr_missing/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 404, response.text
  assert response.json()["code"] == "diligence_pr_not_found"


def test_merge_diligence_pr_route_requires_confirmation_and_expected_metadata(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  calls = 0

  def handler_factory(_session: Any):
    nonlocal calls
    calls += 1
    raise AssertionError("invalid requests must not reach the merge handler")

  monkeypatch.setattr(diligence_prs_module, "_merge_handler_for_session", handler_factory)
  url = "/api/control/diligence-prs/dpr_7_MSFT_abc/merge"
  headers = _headers(test_control_session)

  confirmation = client.post(url, headers=headers, json=_merge_payload(confirm_merge=False))
  missing_workspace = client.post(url, headers=headers, json=_merge_payload(expected_workspace_id=""))
  duplicate_proposals = client.post(
    url,
    headers=headers,
    json=_merge_payload(expected_proposal_ids=["proposal-1", "proposal-1"]),
  )
  unexpected = client.post(url, headers=headers, json={**_merge_payload(), "receipt": {}})

  assert confirmation.status_code == 400
  assert missing_workspace.status_code == 422
  assert duplicate_proposals.status_code == 422
  assert unexpected.status_code == 422
  assert calls == 0


def test_merge_diligence_pr_route_requires_owner_role(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  session = control_plane_app.state.auth.session_store.get_session(test_control_session["session_id"])
  assert session is not None
  session.role = "invite"
  invite_payload = dict(test_control_session)
  invite_payload["session_token"] = control_plane_app.state.auth.issue_token(session)

  def handler_factory(_session: Any):
    raise AssertionError("invite sessions must not reach the merge handler")

  monkeypatch.setattr(diligence_prs_module, "_merge_handler_for_session", handler_factory)

  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(invite_payload),
    json=_merge_payload(),
  )

  assert response.status_code == 403, response.text
  assert response.json()["detail"] == "Owner control session required to merge a diligence PR"
