from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.routing import APIRoute
from pydantic import BaseModel


CONTROL_PLANE_VERSION = "1"


class ControlHealthResponse(BaseModel):
  status: str
  version: str
  endpoints: list[str]


def _control_route_entries(request: Request, route_prefix: str) -> list[str]:
  control_prefix = f"{route_prefix.rstrip('/')}/control" if route_prefix else "/control"
  entries: list[str] = []
  for route in request.app.routes:
    if not isinstance(route, APIRoute):
      continue
    if route.path != control_prefix and not route.path.startswith(f"{control_prefix}/"):
      continue
    methods = sorted(method for method in route.methods if method not in {"HEAD", "OPTIONS"})
    entries.extend(f"{method} {route.path}" for method in methods)
  return sorted(entries)


def build_health_router(*, route_prefix: str) -> APIRouter:
  router = APIRouter()

  @router.get("/health", response_model=ControlHealthResponse)
  async def control_health(request: Request) -> ControlHealthResponse:
    return ControlHealthResponse(
      status="ok",
      version=CONTROL_PLANE_VERSION,
      endpoints=_control_route_entries(request, route_prefix),
    )

  return router


__all__ = ["CONTROL_PLANE_VERSION", "ControlHealthResponse", "build_health_router"]
