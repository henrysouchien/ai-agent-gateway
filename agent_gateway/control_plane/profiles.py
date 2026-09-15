from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agent_gateway.session import AuthManager


log = logging.getLogger("agent_gateway.control_plane.profiles")


class ProfileMetadataResponse(BaseModel):
  name: str
  model: str | None = None
  channel_context: str | None = None


class ProfilesListResponse(BaseModel):
  profiles: list[ProfileMetadataResponse]


def _profile_response_from_loader(
  profile_loader: Callable[[str], Any],
  name: str,
) -> ProfileMetadataResponse | None:
  try:
    profile = profile_loader(name)
  except Exception:
    log.warning("profile %s failed to load; omitting from listing", name, exc_info=True)
    return None
  if not profile.supports_autonomous_execution:
    return None

  return ProfileMetadataResponse(
    name=profile.name,
    model=getattr(profile, "model", None) if isinstance(getattr(profile, "model", None), str) else None,
    channel_context=profile.channel_context,
  )


def _list_profile_metadata(
  *,
  profile_names_provider: Callable[[], Iterable[str]] | None,
  profile_loader: Callable[[str], Any] | None,
) -> list[ProfileMetadataResponse]:
  if profile_names_provider is None or profile_loader is None:
    return []

  entries: list[ProfileMetadataResponse] = []
  for name in profile_names_provider():
    response = _profile_response_from_loader(profile_loader, name)
    if response is not None:
      entries.append(response)
  return sorted(entries, key=lambda entry: entry.name)


def _require_bearer_session(request: Request, auth: AuthManager) -> None:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  auth.verify_token(token)


def build_profiles_router(
  *,
  auth: AuthManager,
  profile_names_provider: Callable[[], Iterable[str]] | None = None,
  profile_loader: Callable[[str], Any] | None = None,
) -> APIRouter:
  if (profile_names_provider is None) != (profile_loader is None):
    raise ValueError("control profile names provider and loader must be configured together")

  router = APIRouter(prefix="/profiles")

  @router.get("", response_model=ProfilesListResponse)
  async def list_profiles(request: Request) -> ProfilesListResponse:
    _require_bearer_session(request, auth)
    return ProfilesListResponse(profiles=_list_profile_metadata(
      profile_names_provider=profile_names_provider,
      profile_loader=profile_loader,
    ))

  return router


__all__ = ["ProfileMetadataResponse", "ProfilesListResponse", "build_profiles_router"]
