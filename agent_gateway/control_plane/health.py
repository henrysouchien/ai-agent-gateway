from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fastapi import APIRouter, Request
from fastapi.routing import APIRoute
from pydantic import BaseModel

from ..autonomous_claim_broker import frame_cap_metrics as claim_broker_frame_cap_metrics
from ..autonomous_event_channel import frame_cap_metrics as event_channel_frame_cap_metrics
from ..autonomous_frame_metrics import FrameCapSnapshot


CONTROL_PLANE_VERSION = "1"


class FrameCapsHealth(BaseModel):
  claim_broker: FrameCapSnapshot
  event_channel: FrameCapSnapshot


class ControlHealthResponse(BaseModel):
  status: str
  version: str
  endpoints: list[str]
  frame_caps: FrameCapsHealth


def _control_route_entries(request: Request, route_prefix: str) -> list[str]:
  control_prefix = f"{route_prefix.rstrip('/')}/control" if route_prefix else "/control"
  entries: list[str] = []

  def iter_api_routes(
    routes: Iterable[Any],
    prefix: str = "",
  ) -> Iterable[tuple[APIRoute, str]]:
    for route in routes:
      if isinstance(route, APIRoute):
        yield route, f"{prefix}{route.path}"
        continue
      original_router = getattr(route, "original_router", None)
      nested_routes = getattr(original_router, "routes", None)
      if nested_routes is None:
        continue
      include_context = getattr(route, "include_context", None)
      nested_prefix = str(getattr(include_context, "prefix", "") or "")
      yield from iter_api_routes(nested_routes, f"{prefix}{nested_prefix}")

  for route, path in iter_api_routes(request.app.routes):
    if path != control_prefix and not path.startswith(f"{control_prefix}/"):
      continue
    # FastAPI always populates an APIRoute's method set; the Optional comes from
    # starlette's base Route, where a mounted route may carry none.
    assert route.methods is not None
    methods = sorted(method for method in route.methods if method not in {"HEAD", "OPTIONS"})
    entries.extend(f"{method} {path}" for method in methods)
  return sorted(entries)


def build_health_router(*, route_prefix: str) -> APIRouter:
  router = APIRouter()

  @router.get("/health", response_model=ControlHealthResponse)
  async def control_health(request: Request) -> ControlHealthResponse:
    approval_delivery_coordinator = getattr(
      request.app.state,
      "autonomous_approval_delivery_coordinator",
      None,
    )
    approval_delivery_fatal_error = getattr(
      approval_delivery_coordinator,
      "fatal_error",
      None,
    )
    return ControlHealthResponse(
      status=(
        "error"
        if approval_delivery_fatal_error is not None
        else "ok"
      ),
      version=CONTROL_PLANE_VERSION,
      endpoints=_control_route_entries(request, route_prefix),
      frame_caps=FrameCapsHealth(
        claim_broker=claim_broker_frame_cap_metrics.snapshot(),
        event_channel=event_channel_frame_cap_metrics.snapshot(),
      ),
    )

  return router


__all__ = [
  "CONTROL_PLANE_VERSION",
  "FrameCapsHealth",
  "ControlHealthResponse",
  "build_health_router",
]
