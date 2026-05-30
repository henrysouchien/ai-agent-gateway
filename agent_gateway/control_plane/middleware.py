from __future__ import annotations

from collections.abc import Iterable

from fastapi import FastAPI, Request

from .health import CONTROL_PLANE_VERSION


CONTROL_PLANE_VERSION_HEADER = "X-Control-Plane-Version"


def _matches_prefix(path: str, prefixes: Iterable[str]) -> bool:
  for prefix in prefixes:
    normalized = "/" + prefix.strip("/")
    if path == normalized or path.startswith(f"{normalized}/"):
      return True
  return False


def add_control_plane_version_header_middleware(
  app: FastAPI,
  *,
  path_prefixes: Iterable[str],
) -> None:
  prefixes = tuple(path_prefixes)

  @app.middleware("http")
  async def control_plane_version_header(request: Request, call_next):
    response = await call_next(request)
    if _matches_prefix(request.url.path, prefixes):
      response.headers[CONTROL_PLANE_VERSION_HEADER] = CONTROL_PLANE_VERSION
    return response


__all__ = ["CONTROL_PLANE_VERSION_HEADER", "add_control_plane_version_header_middleware"]
