"""Inert compiler for exact registered MCP tool descriptors."""

from __future__ import annotations

from dataclasses import dataclass

from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  RegisteredToolServerDescriptor,
  ToolRegistrationCatalog,
  ToolRegistrationDeclaration,
  UnknownToolRegistrationError,
  validate_registered_tool_server_descriptor,
  validate_tool_registration_catalog,
  validate_tool_registration_declaration,
)

from .tool_definition import (
  LiveToolRouteBinding,
  validate_live_tool_route_binding,
)


class RegisteredMcpToolCompilationError(ValueError):
  """An exact live MCP route cannot join its static registration."""


class UnknownRegisteredMcpToolDescriptorError(LookupError):
  """An exposed live MCP name has no compiled exact descriptor."""


UNREGISTERED_ROUTE_REASON = "live MCP route has no exact static registration"
UNREGISTERED_ROUTE_ERROR_CODE = "mcp_tool_not_registered"


@dataclass(frozen=True, slots=True)
class RefusedMcpRoute:
  """One live MCP route refused because it has no exact static registration.

  The route stays out of the model-visible surface; this record is what
  readiness and the typed call error report.
  """

  logical_server_id: str
  logical_name: str
  exposed_name: str
  transport_server_id: str
  route_kind: str
  reason: str

  def call_error(self) -> dict[str, str]:
    """Return the typed error a call to this refused tool receives."""

    return {
      "code": UNREGISTERED_ROUTE_ERROR_CODE,
      "message": (
        f"Tool '{self.exposed_name}' on server '{self.logical_server_id}' "
        f"is refused: {self.reason}."
      ),
    }

  def as_readiness(self) -> dict[str, str]:
    """Return the route as listed in gateway readiness."""

    return {
      "logical_server_id": self.logical_server_id,
      "logical_name": self.logical_name,
      "exposed_name": self.exposed_name,
      "transport_server_id": self.transport_server_id,
      "route_kind": self.route_kind,
      "reason": self.reason,
    }


@dataclass(frozen=True, slots=True)
class RegisteredMcpToolCompilation:
  """Descriptors for registered live routes and the routes refused for lack of one."""

  descriptors: tuple["RegisteredMcpToolDescriptor", ...]
  refused: tuple[RefusedMcpRoute, ...]


@dataclass(frozen=True, slots=True)
class RegisteredMcpToolDescriptor:
  """One exact static MCP registration joined to one exact live route."""

  declaration: ToolRegistrationDeclaration
  server: RegisteredToolServerDescriptor
  live_binding: LiveToolRouteBinding

  def __post_init__(self) -> None:
    declaration = validate_tool_registration_declaration(self.declaration)
    server = validate_registered_tool_server_descriptor(self.server)
    binding = validate_live_tool_route_binding(self.live_binding)
    identity = RegisteredToolIdentity(
      route_kind="mcp",
      logical_server_id=binding.logical_server_id,
      logical_name=binding.logical_name,
    )
    if declaration.identity != identity:
      raise RegisteredMcpToolCompilationError(
        "live MCP route does not match its static declaration"
      )
    if server.logical_server_id != binding.logical_server_id:
      raise RegisteredMcpToolCompilationError(
        "live MCP route does not match its static server"
      )
    if server.transport_server_id != binding.transport_server_id:
      raise RegisteredMcpToolCompilationError(
        "live MCP transport does not match its static server"
      )
    object.__setattr__(self, "declaration", declaration)
    object.__setattr__(self, "server", server)
    object.__setattr__(self, "live_binding", binding)

  @property
  def identity(self) -> RegisteredToolIdentity:
    """Return the exact static identity derived from the live route."""

    return self.declaration.identity

  @property
  def registration_key(self) -> str:
    """Return the exact internal key for this registered identity."""

    return self.identity.registration_key

  @property
  def exposed_name(self) -> str:
    """Return the exact provider-facing name on the live route."""

    return self.live_binding.exposed_name

  def materialize_provider_definition(self) -> dict[str, object]:
    """Return a fresh mutable provider definition."""

    return self.live_binding.materialize_provider_definition()


def _identity_order(
  descriptor: RegisteredMcpToolDescriptor,
) -> tuple[str, str]:
  server_id = descriptor.identity.logical_server_id
  assert server_id is not None
  return (server_id, descriptor.identity.logical_name)


def compile_registered_mcp_tool_descriptors(
  tool_registration_catalog: ToolRegistrationCatalog,
  manager_route_bindings: tuple[LiveToolRouteBinding, ...],
) -> RegisteredMcpToolCompilation:
  """Join static registrations to the manager's exact immutable route tuple.

  ``manager_route_bindings`` is the tuple returned by
  ``McpClientManager.get_server_tool_route_bindings``. This pure join
  revalidates its value shape and its coherence with the static catalog; it
  does not reconstruct or attest the manager topology that produced it.

  A route with no exact static registration is contained per route: it is
  returned in ``refused`` and every other route still compiles. Any other
  incoherence (duplicate identity, a registration that disagrees with its
  route) still raises ``RegisteredMcpToolCompilationError``.
  """

  catalog = validate_tool_registration_catalog(tool_registration_catalog)
  if type(manager_route_bindings) is not tuple:
    raise TypeError("manager_route_bindings must be an exact tuple")

  descriptors: list[RegisteredMcpToolDescriptor] = []
  refused: list[RefusedMcpRoute] = []
  identities: set[RegisteredToolIdentity] = set()
  registration_keys: set[str] = set()
  exposed_names: set[str] = set()
  for raw_binding in manager_route_bindings:
    binding = validate_live_tool_route_binding(raw_binding)
    identity = RegisteredToolIdentity(
      route_kind="mcp",
      logical_server_id=binding.logical_server_id,
      logical_name=binding.logical_name,
    )
    try:
      declaration = catalog.by_identity(identity)
      server = catalog.server(binding.logical_server_id)
    except UnknownToolRegistrationError:
      refused.append(RefusedMcpRoute(
        logical_server_id=binding.logical_server_id,
        logical_name=binding.logical_name,
        exposed_name=binding.exposed_name,
        transport_server_id=binding.transport_server_id,
        route_kind=binding.route_kind,
        reason=UNREGISTERED_ROUTE_REASON,
      ))
      continue
    descriptor = RegisteredMcpToolDescriptor(
      declaration=declaration,
      server=server,
      live_binding=binding,
    )
    if descriptor.identity in identities:
      raise RegisteredMcpToolCompilationError(
        "duplicate live MCP registration identity"
      )
    if descriptor.registration_key in registration_keys:
      raise RegisteredMcpToolCompilationError(
        "duplicate live MCP registration key"
      )
    if descriptor.exposed_name in exposed_names:
      raise RegisteredMcpToolCompilationError(
        "duplicate live MCP exposed name"
      )
    identities.add(descriptor.identity)
    registration_keys.add(descriptor.registration_key)
    exposed_names.add(descriptor.exposed_name)
    descriptors.append(descriptor)

  descriptors.sort(key=_identity_order)
  refused.sort(key=lambda route: (route.logical_server_id, route.logical_name))
  return RegisteredMcpToolCompilation(tuple(descriptors), tuple(refused))


__all__ = [
  "RefusedMcpRoute",
  "RegisteredMcpToolCompilation",
  "RegisteredMcpToolCompilationError",
  "RegisteredMcpToolDescriptor",
  "UNREGISTERED_ROUTE_ERROR_CODE",
  "UNREGISTERED_ROUTE_REASON",
  "UnknownRegisteredMcpToolDescriptorError",
  "compile_registered_mcp_tool_descriptors",
]
