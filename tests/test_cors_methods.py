from __future__ import annotations

from fastapi.testclient import TestClient

from agent_gateway.server_models import GatewayServerConfig


_ACTIVE_ROUTE_METHODS = [
  "GET",
  "POST",
  "PUT",
  "DELETE",
  "PATCH",
  "OPTIONS",
]


def test_gateway_cors_default_matches_active_route_methods() -> None:
  assert GatewayServerConfig().cors_allow_methods == _ACTIVE_ROUTE_METHODS


def test_gateway_cors_admits_preference_put_and_delete(make_test_app) -> None:
  app = make_test_app()

  with TestClient(app) as client:
    for method in ("PUT", "DELETE"):
      response = client.options(
        "/api/model-preferences/session.driver",
        headers={
          "Origin": "http://localhost:3002",
          "Access-Control-Request-Method": method,
          "Access-Control-Request-Headers": "authorization,content-type",
        },
      )
      assert response.status_code == 200, response.text
      assert response.text == "OK"
      assert response.headers["access-control-allow-origin"] == (
        "http://localhost:3002"
      )
      advertised = {
        item.strip()
        for item in response.headers["access-control-allow-methods"].split(",")
      }
      assert advertised == set(_ACTIVE_ROUTE_METHODS)


def test_gateway_cors_rejects_an_undeclared_method(make_test_app) -> None:
  app = make_test_app()

  with TestClient(app) as client:
    response = client.options(
      "/api/model-preferences/session.driver",
      headers={
        "Origin": "http://localhost:3002",
        "Access-Control-Request-Method": "TRACE",
      },
    )

  assert response.status_code == 400
  assert response.text == "Disallowed CORS method"
