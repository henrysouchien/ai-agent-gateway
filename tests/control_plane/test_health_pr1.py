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
