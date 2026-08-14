"""Server-owned semantic capability routing for ordinary child admission.

The historical child-tool-scope receipt treated concrete tool names declared
by a skill file as executable authority.  Ordinary delegation now follows the
same boundary as workflows: an immutable operation declares semantic needs,
the live server catalog selects compatible routes, and the resulting exact
``ToolGrant`` is persisted inside ``AdmittedTask``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from agent_workflow_contracts import (
  AgentOperationSnapshot,
  CapabilityBinding,
  ToolGrant,
  ToolGrantEntry,
  sha256_digest,
)

from .policy_imports import (
  load_server_policy_module,
  resolve_server_policy_tool_class,
)
from .semantic_capabilities import (
  SemanticCapabilityCompilationError,
  SemanticToolRoute,
  compile_semantic_capabilities,
)


ADMITTED_TASK_METADATA_KEY = "admitted_task"


class OperationToolAdmissionError(ValueError):
  """The live runtime cannot bind an operation's semantic requirements."""


@dataclass(frozen=True, slots=True)
class OperationToolAdmission:
  tool_grant: ToolGrant
  capability_bindings: tuple[CapabilityBinding, ...]
  tool_ids: frozenset[str]
  mcp_tools_by_server: Mapping[str, frozenset[str]]


ToolEffectResolver = Callable[[str, str | None, bool], str | None]


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


def _normalized_effect(raw: str | None) -> str | None:
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
) -> str | None:
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


def _allowed_effects(workspace_scope: str) -> frozenset[str]:
  if workspace_scope == "read_only":
    return frozenset({"read"})
  if workspace_scope == "workspace_write":
    return frozenset({"read", "propose", "write"})
  if workspace_scope == "model_write":
    return frozenset({"read", "propose", "write", "external_effect"})
  raise OperationToolAdmissionError(
    f"unknown operation workspace scope: {workspace_scope}"
  )


def _tool_route_facts(
  *,
  operation_tool_ids: frozenset[str],
  definitions: Iterable[Mapping[str, Any]],
  local_tool_handlers: Mapping[str, Any],
  mcp_client: Any,
  effect_resolver: ToolEffectResolver,
) -> tuple[SemanticToolRoute, ...]:
  is_mcp_tool = getattr(mcp_client, "is_mcp_tool", None)
  get_server = getattr(mcp_client, "get_server_for_tool", None)
  get_original = getattr(mcp_client, "get_original_tool_name", None)
  routes: dict[str, SemanticToolRoute] = {}
  for definition in definitions:
    tool_id = str(definition.get("name") or "").strip()
    if (
      not tool_id
      or tool_id not in operation_tool_ids
    ):
      continue
    is_local = tool_id in local_tool_handlers
    is_mcp = bool(callable(is_mcp_tool) and is_mcp_tool(tool_id))
    if is_local == is_mcp:
      # Ambiguous or unroutable definitions cannot become authority.
      continue
    server_id = (
      str(get_server(tool_id) or "").strip()
      if is_mcp and callable(get_server)
      else None
    )
    if is_mcp and not server_id:
      continue
    policy_tool_id = (
      str(get_original(tool_id) or tool_id).strip()
      if is_mcp and callable(get_original)
      else tool_id
    )
    effect = effect_resolver(policy_tool_id, server_id, is_local)
    if effect is None:
      continue
    routes[tool_id] = SemanticToolRoute(
      tool_id=tool_id,
      effect=effect,
      server_id=server_id,
    )
  return tuple(routes[name] for name in sorted(routes))


def semantic_tool_routes(
  tool_ids: Iterable[str],
  *,
  local_tool_handlers: Mapping[str, Any],
  mcp_client: Any,
  effect_resolver: ToolEffectResolver | None = None,
) -> tuple[SemanticToolRoute, ...]:
  """Prove exact private tool IDs against trusted live routing facts."""

  names = frozenset(tool_ids)
  return _tool_route_facts(
    operation_tool_ids=names,
    definitions=({"name": name} for name in sorted(names)),
    local_tool_handlers=local_tool_handlers,
    mcp_client=mcp_client,
    effect_resolver=effect_resolver or _server_owned_effect,
  )


def admit_operation_tools(
  operation: AgentOperationSnapshot,
  *,
  grant_id: str,
  operation_tool_ids: Iterable[str],
  definitions: Iterable[Mapping[str, Any]],
  local_tool_handlers: Mapping[str, Any],
  mcp_client: Any,
  effect_resolver: ToolEffectResolver | None = None,
) -> OperationToolAdmission:
  """Bind semantic requirements to live routes and compile an exact grant."""

  if not isinstance(operation, AgentOperationSnapshot):
    raise TypeError("operation must be an AgentOperationSnapshot")
  resolver = effect_resolver or _server_owned_effect
  exact_ceiling = frozenset(operation_tool_ids)
  route_facts = _tool_route_facts(
    operation_tool_ids=exact_ceiling,
    definitions=definitions,
    local_tool_handlers=local_tool_handlers,
    mcp_client=mcp_client,
    effect_resolver=resolver,
  )
  allowed_effects = _allowed_effects(operation.workspace_scope)
  scoped_routes = tuple(
    route for route in route_facts if route.effect in allowed_effects
  )
  try:
    compiled = compile_semantic_capabilities(
      operation.required_capabilities,
      grant_id=grant_id,
      tool_routes=scoped_routes,
    )
  except SemanticCapabilityCompilationError as exc:
    raise OperationToolAdmissionError(str(exc)) from exc
  return OperationToolAdmission(
    tool_grant=compiled.tool_grant,
    capability_bindings=compiled.capability_bindings,
    tool_ids=compiled.tool_ids,
    mcp_tools_by_server=compiled.mcp_tools_by_server,
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


def scopes_from_tool_grant(
  grant: ToolGrant,
  *,
  local_tool_handlers: Mapping[str, Any],
  mcp_client: Any,
) -> tuple[frozenset[str], dict[str, set[str]]]:
  """Resolve dispatcher scopes from an already-admitted exact grant."""

  grant = parse_tool_grant(grant)
  is_mcp_tool = getattr(mcp_client, "is_mcp_tool", None)
  get_server = getattr(mcp_client, "get_server_for_tool", None)
  mcp_scope: dict[str, set[str]] = {}
  for entry in grant.tools:
    if entry.tool_id in local_tool_handlers:
      continue
    if not callable(is_mcp_tool) or not is_mcp_tool(entry.tool_id):
      raise OperationToolAdmissionError(
        f"admitted tool {entry.tool_id!r} is no longer live"
      )
    server = str(get_server(entry.tool_id) or "").strip() if callable(get_server) else ""
    if not server:
      raise OperationToolAdmissionError(
        f"admitted tool {entry.tool_id!r} has no live server route"
      )
    mcp_scope.setdefault(server, set()).add(entry.tool_id)
  return frozenset(entry.tool_id for entry in grant.tools), mcp_scope


__all__ = [
  "ADMITTED_TASK_METADATA_KEY",
  "OperationToolAdmission",
  "OperationToolAdmissionError",
  "admit_operation_tools",
  "parse_tool_grant",
  "reissue_tool_grant",
  "scopes_from_tool_grant",
]
