from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Request

from agent_gateway.auth import CredentialsResolver
from agent_gateway.session import AuthManager

from .approvals import build_approvals_router
from .artifacts import build_artifacts_router
from .batches import build_batches_router
from .diligence_prs import build_diligence_prs_router
from .events import build_events_router
from .health import build_health_router
from .profiles import build_profiles_router
from .readable_resources import build_readable_resources_router
from .runs import build_runs_router
from .schedules import build_schedules_router
from .session import build_session_router
from .skills import build_skills_router


def create_control_plane_router(
  *,
  auth: AuthManager,
  credentials_resolver: CredentialsResolver | None,
  resolver_timeout_seconds: float,
  route_prefix: str,
  skills_dir: Path,
  artifact_auth_dependency: Callable[[Request], str],
  autonomous_registry: Any | None = None,
  agent_schedule_store: Any | None = None,
  agent_schedule_runner: Any | None = None,
  approval_store: Any | None = None,
  approval_policy: Any | None = None,
  dispatch_scope_validator: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> APIRouter:
  router = APIRouter()
  router.include_router(
    build_session_router(
      auth=auth,
      credentials_resolver=credentials_resolver,
      resolver_timeout_seconds=resolver_timeout_seconds,
      approval_store=approval_store,
      approval_policy=approval_policy,
    )
  )
  router.include_router(build_profiles_router(auth=auth))
  router.include_router(build_skills_router(auth=auth, skills_dir=skills_dir))
  router.include_router(build_schedules_router(
    auth=auth,
    agent_schedule_store=agent_schedule_store,
    agent_schedule_runner=agent_schedule_runner,
    dispatch_scope_validator=dispatch_scope_validator,
  ))
  router.include_router(build_runs_router(
    auth=auth,
    autonomous_registry=autonomous_registry,
    skills_dir=skills_dir,
    dispatch_scope_validator=dispatch_scope_validator,
  ))
  router.include_router(build_batches_router(auth=auth))
  router.include_router(build_diligence_prs_router(auth=auth))
  router.include_router(build_approvals_router(auth=auth, autonomous_registry=autonomous_registry))
  router.include_router(build_events_router(auth=auth))
  router.include_router(build_readable_resources_router(
    auth=auth,
    autonomous_registry=autonomous_registry,
  ))
  router.include_router(build_artifacts_router(
    artifact_auth_dependency=artifact_auth_dependency,
    autonomous_registry=autonomous_registry,
  ))
  router.include_router(build_health_router(route_prefix=route_prefix))
  return router


__all__ = ["create_control_plane_router"]
