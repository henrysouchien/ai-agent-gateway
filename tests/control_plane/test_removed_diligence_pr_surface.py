from __future__ import annotations


def test_diligence_pr_routes_are_not_registered(control_plane_app) -> None:
  route_paths = {
    str(getattr(route, "path", ""))
    for route in control_plane_app.routes
  }

  assert not any("/control/diligence-prs" in path for path in route_paths)
