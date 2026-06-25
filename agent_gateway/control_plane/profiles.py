from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agent_gateway.session import AuthManager

_AGENT_PROFILES_MODULE_NAMES = frozenset({"agent", "agent.profiles"})


class ProfileMetadataResponse(BaseModel):
  name: str
  model: str | None = None
  channel_context: str | None = None


class ProfilesListResponse(BaseModel):
  profiles: list[ProfileMetadataResponse]


def _profiles_api() -> Any:
  api_dir = Path(__file__).resolve().parents[4] / "api"
  if api_dir.exists() and str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))
  try:
    return importlib.import_module("agent.profiles")
  except ModuleNotFoundError as exc:
    if exc.name not in _AGENT_PROFILES_MODULE_NAMES:
      raise
    return importlib.import_module("api.agent.profiles")


def _profile_response_from_module(package_name: str, name: str) -> ProfileMetadataResponse | None:
  try:
    module = importlib.import_module(f"{package_name}.{name}")
  except Exception:
    return None

  get_profile = getattr(module, "get_profile", None)
  if not callable(get_profile):
    return None

  try:
    profile = get_profile()
  except Exception:
    return None

  profile_name = getattr(profile, "name", None)
  if not isinstance(profile_name, str) or not profile_name.strip():
    return None

  return ProfileMetadataResponse(
    name=profile_name.strip(),
    model=getattr(profile, "model", None) if isinstance(getattr(profile, "model", None), str) else None,
    channel_context=getattr(profile, "channel_context", None)
    if isinstance(getattr(profile, "channel_context", None), str)
    else None,
  )


def _list_profile_metadata() -> list[ProfileMetadataResponse]:
  profiles_pkg = _profiles_api()
  package_paths = getattr(profiles_pkg, "__path__", None)
  if package_paths is None:
    return []

  entries: list[ProfileMetadataResponse] = []
  for module_info in sorted(pkgutil.iter_modules(package_paths), key=lambda entry: entry.name):
    if module_info.ispkg or module_info.name.startswith("_") or module_info.name == "prompt_loader":
      continue
    response = _profile_response_from_module(profiles_pkg.__name__, module_info.name)
    if response is not None:
      entries.append(response)
  return sorted(entries, key=lambda entry: entry.name)


def _require_bearer_session(request: Request, auth: AuthManager) -> None:
  token = AuthManager.get_bearer_token(request.headers.get("Authorization"))
  auth.verify_token(token)


def build_profiles_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter(prefix="/profiles")

  @router.get("", response_model=ProfilesListResponse)
  async def list_profiles(request: Request) -> ProfilesListResponse:
    _require_bearer_session(request, auth)
    return ProfilesListResponse(profiles=_list_profile_metadata())

  return router


__all__ = ["ProfileMetadataResponse", "ProfilesListResponse", "build_profiles_router"]
