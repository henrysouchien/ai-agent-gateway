from __future__ import annotations

from fastapi.testclient import TestClient

from agent_gateway.control_plane.middleware import CONTROL_PLANE_VERSION_HEADER


def test_control_health_shape_and_registered_endpoints(
  client: TestClient,
  control_health_url: str,
) -> None:
  response = client.get(control_health_url)

  assert response.status_code == 200
  assert response.headers[CONTROL_PLANE_VERSION_HEADER] == "1"
  payload = response.json()
  assert payload["status"] == "ok"
  assert payload["version"] == "1"
  assert "GET /api/control/health" in payload["endpoints"]
  assert "POST /api/control/session" in payload["endpoints"]


def test_control_version_header_is_control_prefix_only(
  client: TestClient,
  control_health_url: str,
) -> None:
  control_missing = client.get("/api/control/missing")
  root_health = client.get("/api/health")

  assert control_missing.status_code == 404
  assert control_missing.headers[CONTROL_PLANE_VERSION_HEADER] == "1"
  assert root_health.status_code == 200
  assert CONTROL_PLANE_VERSION_HEADER not in root_health.headers
  assert root_health.json()["status"] == "ok"
  assert "package" in root_health.json()


def test_approval_delivery_fatal_state_fails_health(
  client: TestClient,
  control_health_url: str,
) -> None:
  coordinator = (
    client.app.state.autonomous_approval_delivery_coordinator
  )
  coordinator._fatal_error = "injected approval recovery failure"

  control_health = client.get(control_health_url)
  root_health = client.get("/api/health")

  assert control_health.status_code == 200
  assert control_health.json()["status"] == "error"
  assert root_health.status_code == 503
  assert root_health.json()["status"] == "error"


def test_approval_delivery_coordinator_follows_app_lifespan(
  control_plane_app,
) -> None:
  coordinator = (
    control_plane_app.state.autonomous_approval_delivery_coordinator
  )
  assert coordinator._task is None

  with TestClient(control_plane_app):
    assert coordinator._task is not None
    assert not coordinator._task.done()

  assert coordinator._task is None
