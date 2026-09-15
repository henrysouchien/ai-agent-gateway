"""Server-owned semantic capability routing for ordinary child admission.

The historical child-tool-scope receipt treated concrete tool names declared
by a skill file as executable authority.  Ordinary delegation now follows the
same boundary as workflows: an immutable operation declares semantic needs,
the live server catalog selects compatible routes, and the resulting exact
``ToolGrant`` is persisted inside ``AdmittedTask``.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from agent_workflow_contracts import (
  AdmittedToolRoute,
  AgentOperationSnapshot,
  ExecutionIdentity,
  OperationUnavailable,
  ResolvedAuthority,
  ToolGrant,
  ToolGrantEntry,
  sha256_digest,
)
from agent_workflow_contracts.models import CatalogToolEffect

from .capability_resolution import (
  OperationDeclaration,
  resolve_operation_authority,
  snapshot_platform_catalog,
)
from .policy_imports import (
  load_server_policy_module,
  resolve_server_policy_tool_class,
)


ADMITTED_TASK_METADATA_KEY = "admitted_task"


class OperationToolAdmissionError(ValueError):
  """The live runtime cannot bind an operation's semantic requirements."""


ToolEffectResolver = Callable[
  [str, str | None, bool],
  CatalogToolEffect | None,
]


def _tool_grant(
  *,
  grant_id: str,
  entries: tuple[ToolGrantEntry, ...],
) -> ToolGrant:
  payload = {
    "grant_id": grant_id,
    "tools": [entry.model_dump(mode="json") for entry in entries],
  }
  return ToolGrant(
    grant_id=grant_id,
    tools=entries,
    digest=sha256_digest(payload),
  )


def _normalized_effect(raw: object) -> CatalogToolEffect | None:
  value = str(raw or "").strip().lower()
  if value in {"read", "pure_transform", "read_only", "support"}:
    return "read"
  if value in {"preview", "artifact_write"}:
    return "propose"
  if value == "state_write":
    return "write"
  if value == "external_write":
    return "external_effect"
  return None


def _server_owned_effect(
  tool_id: str,
  server_id: str | None,
  is_local: bool,
) -> CatalogToolEffect | None:
  if is_local:
    policy = load_server_policy_module()
    get_local_effect = (
      getattr(policy, "get_local_tool_effect", None)
      if policy is not None
      else None
    )
    raw = get_local_effect(tool_id) if callable(get_local_effect) else None
  else:
    raw = resolve_server_policy_tool_class(
      tool_id,
      runtime_server=server_id,
      default="",
    )
  return _normalized_effect(raw)


def admit_operation_tools(
  operation: AgentOperationSnapshot,
  *,
  grant_id: str,
  operation_tool_ids: Iterable[str],
  definitions: Iterable[Mapping[str, Any]],
  local_tool_handlers: Mapping[str, Any],
  mcp_client: Any,
  effect_resolver: ToolEffectResolver | None = None,
  identity: ExecutionIdentity | None = None,
  exclusions: Iterable[str] = (),
) -> ResolvedAuthority | OperationUnavailable:
  """Resolve one ordinary-delegation operation's authority (B-6).

  A thin adapter over :func:`resolve_operation_authority`: it names the
  candidate ids (the operation's declared ceiling, intersected with the
  definitions the parent can actually offer this child) and hands the exact
  declaration to the one resolver.  Nothing here decides satisfaction, and
  nothing re-derives a route the catalog snapshot did not describe.

  The Left is returned, not raised: an operation the platform cannot authorize
  is a visible :class:`OperationUnavailable`, and the ``operation_unavailable``
  wire code at the ``run_agent`` boundary is exactly its projection.
  """

  if not isinstance(operation, AgentOperationSnapshot):
    raise TypeError("operation must be an AgentOperationSnapshot")
  exact_ceiling = frozenset(operation_tool_ids)
  candidate_ids = tuple(sorted({
    tool_id
    for definition in definitions
    if (tool_id := str(definition.get("name") or "").strip())
    and tool_id in exact_ceiling
  }))
  declaration = OperationDeclaration(
    operation_name=operation.operation.name,
    grant_id=grant_id,
    workspace_scope=operation.workspace_scope,
    required_capabilities=operation.required_capabilities,
    tool_ceiling=exact_ceiling,
  )
  return resolve_operation_authority(
    declaration,
    catalog=snapshot_platform_catalog(
      tool_ids=candidate_ids,
      local_tool_handlers=local_tool_handlers,
      mcp_client=mcp_client,
      effect_resolver=effect_resolver or _server_owned_effect,
    ),
    identity=identity,
    exclusions=exclusions,
  )


def parse_tool_grant(raw: object) -> ToolGrant:
  """Validate one persisted canonical ToolGrant, including its digest."""

  try:
    grant = ToolGrant.model_validate(raw)
  except Exception as exc:
    raise OperationToolAdmissionError("invalid persisted ToolGrant") from exc
  expected = sha256_digest({
    "grant_id": grant.grant_id,
    "tools": [entry.model_dump(mode="json") for entry in grant.tools],
  })
  if grant.digest != expected:
    raise OperationToolAdmissionError("persisted ToolGrant digest mismatch")
  return grant


def reissue_tool_grant(grant: ToolGrant, *, grant_id: str) -> ToolGrant:
  """Issue the same exact authority under a new admitted attempt identity."""

  validated = parse_tool_grant(grant)
  return _tool_grant(grant_id=grant_id, entries=validated.tools)


def dispatcher_scopes_from_admitted_routes(
  grant: ToolGrant,
  routes: tuple[AdmittedToolRoute, ...],
) -> tuple[frozenset[str], frozenset[str], dict[str, set[str]]]:
  """Project dispatcher scopes from persisted authority without live reads."""

  grant = parse_tool_grant(grant)
  granted_tool_ids = tuple(entry.tool_id for entry in grant.tools)
  route_tool_ids = tuple(route.tool_id for route in routes)
  if route_tool_ids != granted_tool_ids:
    raise OperationToolAdmissionError(
      "admitted tool routes do not match the exact ordered ToolGrant"
    )
  local_tool_ids = frozenset(
    route.tool_id for route in routes if route.origin == "local"
  )
  mcp_scope: dict[str, set[str]] = {}
  for route in routes:
    if route.origin != "mcp":
      continue
    assert route.server_id is not None
    mcp_scope.setdefault(route.server_id, set()).add(route.tool_id)
  return frozenset(granted_tool_ids), local_tool_ids, mcp_scope


__all__ = [
  "ADMITTED_TASK_METADATA_KEY",
  "OperationToolAdmissionError",
  "admit_operation_tools",
  "dispatcher_scopes_from_admitted_routes",
  "parse_tool_grant",
  "reissue_tool_grant",
]
