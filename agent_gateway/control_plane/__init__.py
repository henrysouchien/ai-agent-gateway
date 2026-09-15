from __future__ import annotations

from typing import Any, Callable, Iterable

from fastapi import APIRouter

from agent_gateway.auth import CredentialsResolver
from agent_gateway.control_skill_catalog import ControlSkillCatalog
from agent_gateway.session import AuthManager

from .approvals import build_approvals_router
from .batches import build_batches_router
from .events import build_events_router
from .health import build_health_router
from .profiles import build_profiles_router
from .readable_resources import build_readable_resources_router
from .runs import build_runs_router
from .schedules import OperatorScheduleBackend, build_schedules_router
from .session import build_session_router
from .skills import build_skills_router


def create_control_plane_router(
  *,
  auth: AuthManager,
  credentials_resolver: CredentialsResolver | None,
  resolver_timeout_seconds: float,
  tenant_id: str | None,
  allow_service_credentials_for_interactive: bool,
  route_prefix: str,
  control_skill_catalog: ControlSkillCatalog,
  control_profile_names_provider: Callable[[], Iterable[str]] | None = None,
  control_profile_loader: Callable[[str], Any] | None = None,
  identity_resolver: Callable[..., Any] | None = None,
  autonomous_registry: Any | None = None,
  agent_schedule_store_for: Any | None = None,
  agent_schedule_runner: Any | None = None,
  operator_schedule_backend: OperatorScheduleBackend | None = None,
  approval_store: Any | None = None,
  approval_policy: Any | None = None,
  dispatch_scope_validator: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> APIRouter:
  router = APIRouter()
  session_router = build_session_router(
    auth=auth,
    credentials_resolver=credentials_resolver,
    resolver_timeout_seconds=resolver_timeout_seconds,
    tenant_id=tenant_id,
    identity_resolver=identity_resolver,
    allow_service_credentials_for_interactive=(
      allow_service_credentials_for_interactive
    ),
    approval_store=approval_store,
    approval_policy=approval_policy,
  )
  router.include_router(session_router)
  router.include_router(build_profiles_router(
    auth=auth,
    profile_names_provider=control_profile_names_provider,
    profile_loader=control_profile_loader,
  ))
  router.include_router(
    build_skills_router(auth=auth, catalog=control_skill_catalog)
  )
  router.include_router(build_schedules_router(
    auth=auth,
    operator_schedule_backend=operator_schedule_backend,
    agent_schedule_store_for=agent_schedule_store_for,
    agent_schedule_runner=agent_schedule_runner,
    dispatch_scope_validator=dispatch_scope_validator,
  ))
  router.include_router(build_runs_router(
    auth=auth,
    autonomous_registry=autonomous_registry,
    dispatch_scope_validator=dispatch_scope_validator,
    control_profile_loader=control_profile_loader,
  ))
  router.include_router(build_batches_router(auth=auth))
  router.include_router(build_approvals_router(auth=auth, autonomous_registry=autonomous_registry))
  router.include_router(build_events_router(auth=auth))
  router.include_router(build_readable_resources_router(
    auth=auth,
    autonomous_registry=autonomous_registry,
  ))
  router.include_router(build_health_router(route_prefix=route_prefix))
  return router


__all__ = ["create_control_plane_router"]
