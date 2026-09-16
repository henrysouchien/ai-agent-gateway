from __future__ import annotations

import ast
import asyncio
import inspect
import json
from dataclasses import FrozenInstanceError
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace

import pytest

from agent_gateway.approval_policy import sha256_args
from agent_gateway.mcp_client import (
  McpClientManager,
  RegisteredMcpPlannedToolCall,
  _ServerState,
)
from agent_gateway.mcp_client_connections import (
  McpClientSession,
  McpToolCallResult,
)
from agent_gateway.tool_dispatcher_helpers import TrustedToolPlan
from agent_gateway.tool_policy_registry import (
  PlanDecision,
  PreparedToolCall,
  ToolPolicyImplementationRegistry,
  ToolPolicyResultError,
)
from agent_gateway.tool_definition import (
  LiveToolRouteBinding,
  OriginatedToolDefinition,
)
from agent_gateway.tool_registration import (
  RegisteredMcpToolCompilationError,
  RegisteredMcpToolDescriptor,
  UnknownRegisteredMcpToolDescriptorError,
  compile_registered_mcp_tool_descriptors,
)
from agent_workflow_contracts.tool_registration import (
  McpInputPreparationRoute,
  RegisteredToolIdentity,
  RegisteredToolServerDescriptor,
  ToolApprovalPolicy,
  ToolIntrinsicSemantics,
  ToolRegistrationCatalog,
  ToolRegistrationDeclaration,
  VersionedPolicyRef,
)


ROOT = Path(__file__).resolve().parents[1]


def _policy(kind: str) -> VersionedPolicyRef:
  return VersionedPolicyRef(
    kind=kind,  # type: ignore[arg-type]
    policy_id=f"{kind}.identity",
    version="1",
  )




def _declaration(
  server_id: str,
  logical_name: str,
  *,
  idempotent: bool = True,
  effect: str = "read",
) -> ToolRegistrationDeclaration:
  return ToolRegistrationDeclaration(
    identity=RegisteredToolIdentity(
      route_kind="mcp",
      logical_server_id=server_id,
      logical_name=logical_name,
    ),
    semantics=ToolIntrinsicSemantics(
      effect=effect,  # type: ignore[arg-type]
      idempotent=idempotent,
      semantic_capability="market-data.read/v1",
      approval=ToolApprovalPolicy(mode="never"),
      audience="ordinary",
      redaction_policy=_policy("redaction"),
      planning_policy=_policy("planning"),
      input_preparation_policy=_policy("input_preparation"),
      outcome_policy=_policy("outcome"),
      source_identity_policy=_policy("source_identity"),
    ),
  )


def _server(
  logical_server_id: str,
  *,
  transport_server_id: str | None = None,
  default_timeout_seconds: float = 30,
  per_tool_timeout_seconds: dict[str, float] | None = None,
) -> RegisteredToolServerDescriptor:
  return RegisteredToolServerDescriptor(
    logical_server_id=logical_server_id,
    transport_server_id=transport_server_id or logical_server_id,
    default_timeout_seconds=default_timeout_seconds,
    per_tool_timeout_seconds=per_tool_timeout_seconds or {},
    session_injection_policy=_policy("session_injection"),
  )


def _binding(
  *,
  server_id: str,
  exposed_name: str,
  logical_name: str,
  transport_server_id: str,
  provider_original_name: str,
  route_kind: str = "physical",
  description: str = "Read market data.",
) -> LiveToolRouteBinding:
  return LiveToolRouteBinding(
    originated_definition=OriginatedToolDefinition(
      definition={
        "name": exposed_name,
        "description": description,
        "input_schema": {
          "type": "object",
          "properties": {"ticker": {"type": "string"}},
          "required": ["ticker"],
        },
      },
      origin="mcp",
      server_id=server_id,
    ),
    route_kind=route_kind,  # type: ignore[arg-type]
    logical_name=logical_name,
    transport_server_id=transport_server_id,
    provider_original_name=provider_original_name,
    provider_id="provider",
  )


def _catalog(
  declarations: tuple[ToolRegistrationDeclaration, ...],
  servers: tuple[RegisteredToolServerDescriptor, ...],
) -> ToolRegistrationCatalog:
  return ToolRegistrationCatalog(
    declarations=declarations,
    servers=servers,
  )


def test_compiler_joins_physical_and_logical_routes_exactly() -> None:
  physical_declaration = _declaration("provider-mcp", "provider_quote")
  logical_declaration = _declaration("market-data-mcp", "get_quote")
  catalog = _catalog(
    (logical_declaration, physical_declaration),
    (
      _server("market-data-mcp", transport_server_id="provider-mcp"),
      _server("provider-mcp"),
    ),
  )
  physical = _binding(
    server_id="provider-mcp",
    exposed_name="provider__provider_quote",
    logical_name="provider_quote",
    transport_server_id="provider-mcp",
    provider_original_name="provider_quote",
  )
  logical = _binding(
    server_id="market-data-mcp",
    exposed_name="get_quote",
    logical_name="get_quote",
    transport_server_id="provider-mcp",
    provider_original_name="provider_quote",
    route_kind="logical",
  )

  descriptors = compile_registered_mcp_tool_descriptors(
    catalog,
    (physical, logical),
  )

  assert tuple(descriptor.identity for descriptor in descriptors) == (
    logical_declaration.identity,
    physical_declaration.identity,
  )
  assert tuple(descriptor.exposed_name for descriptor in descriptors) == (
    "get_quote",
    "provider__provider_quote",
  )
  assert descriptors[0].server.transport_server_id == "provider-mcp"
  assert descriptors[0].live_binding.provider_original_name == "provider_quote"


def test_same_bare_name_on_distinct_servers_is_identity_preserving() -> None:
  declaration_a = _declaration("a-mcp", "lookup")
  declaration_b = _declaration("b-mcp", "lookup")
  binding_a = _binding(
    server_id="a-mcp",
    exposed_name="a__lookup",
    logical_name="lookup",
    transport_server_id="a-mcp",
    provider_original_name="lookup",
  )
  binding_b = _binding(
    server_id="b-mcp",
    exposed_name="b__lookup",
    logical_name="lookup",
    transport_server_id="b-mcp",
    provider_original_name="lookup",
  )

  descriptors = compile_registered_mcp_tool_descriptors(
    _catalog(
      (declaration_b, declaration_a),
      (_server("b-mcp"), _server("a-mcp")),
    ),
    (binding_b, binding_a),
  )

  assert tuple(
    (descriptor.identity.logical_server_id, descriptor.identity.logical_name)
    for descriptor in descriptors
  ) == (("a-mcp", "lookup"), ("b-mcp", "lookup"))
  assert tuple(descriptor.exposed_name for descriptor in descriptors) == (
    "a__lookup",
    "b__lookup",
  )
  assert descriptors[0].registration_key != descriptors[1].registration_key


def test_declared_but_offline_routes_remain_valid_and_unbound() -> None:
  online = _declaration("online-mcp", "online_tool")
  offline = _declaration("offline-mcp", "offline_tool")
  catalog = _catalog(
    (offline, online),
    (_server("offline-mcp"), _server("online-mcp")),
  )
  binding = _binding(
    server_id="online-mcp",
    exposed_name="online_tool",
    logical_name="online_tool",
    transport_server_id="online-mcp",
    provider_original_name="online_tool",
  )

  descriptors = compile_registered_mcp_tool_descriptors(catalog, (binding,))

  assert tuple(descriptor.identity for descriptor in descriptors) == (
    online.identity,
  )
  assert catalog.resolve_mcp("offline-mcp", "offline_tool").identity == (
    offline.identity
  )
  assert compile_registered_mcp_tool_descriptors(catalog, ()) == ()


class _CatalogMcpSession(McpClientSession):
  async def call_tool(
    self,
    name: str,
    arguments: dict[str, object],
    *,
    read_timeout_seconds: timedelta,
    meta: dict[str, object] | None = None,
  ) -> McpToolCallResult:
    raise AssertionError("catalog topology must not execute its transport session")


def _server_state(
  *,
  server_id: str,
  tool_name: str,
  description: str,
  prefix: str,
) -> _ServerState:
  definition = OriginatedToolDefinition(
    definition={
      "name": tool_name,
      "description": description,
      "input_schema": {"type": "object"},
    },
    origin="mcp",
    server_id=server_id,
  ).materialize()
  return _ServerState(
    name=server_id,
    session=_CatalogMcpSession(),
    exit_contexts=[],
    tool_prefix=prefix,
    tool_definitions=[definition],
    tool_names={tool_name},
    config={"type": "stdio", "command": server_id},
  )


def _install_physical_live_tool(
  manager: McpClientManager,
  *,
  server_id: str,
  tool_name: str,
) -> None:
  manager._servers = {
    **manager._servers,
    server_id: _server_state(
      server_id=server_id,
      tool_name=tool_name,
      description="Registered live tool.",
      prefix="",
    ),
  }




def test_manager_binds_only_live_registered_routes_and_allows_offline_rows() -> None:
  online = _declaration("online-mcp", "online_tool")
  offline = _declaration("offline-mcp", "offline_tool")
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"online-mcp", "offline-mcp"},
    tool_registration_catalog=_catalog(
      (offline, online),
      (_server("offline-mcp"), _server("online-mcp")),
    ),
  )
  _install_physical_live_tool(
    manager,
    server_id="online-mcp",
    tool_name="online_tool",
  )

  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: "online-mcp"
  )

  descriptor = manager.get_registered_mcp_tool_descriptor("online_tool")
  assert descriptor.identity == online.identity
  first = manager.get_tool_definitions()
  second = manager.get_tool_definitions()
  assert first == second == [{
    "name": "online_tool",
    "description": "Registered live tool.",
    "input_schema": {"type": "object"},
  }]
  assert first is not second
  with pytest.raises(
    UnknownRegisteredMcpToolDescriptorError,
    match="unknown registered live MCP tool",
  ):
    manager.get_registered_mcp_tool_descriptor("offline_tool")


def test_manager_resolves_sdk_alias_and_same_bare_names_to_exact_descriptors() -> None:
  declaration_a = _declaration("a-mcp", "lookup", effect="read")
  declaration_b = _declaration("b-mcp", "lookup", effect="state_write")
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"a-mcp", "b-mcp"},
    server_aliases={"legacy-a": "a-mcp"},
    tool_registration_catalog=_catalog(
      (declaration_a, declaration_b),
      (_server("a-mcp"), _server("b-mcp")),
    ),
  )


  manager._servers = {
    "a-mcp": _server_state(
      server_id="a-mcp",
      tool_name="lookup",
      description="Lookup from a-mcp.",
      prefix="a_",
    ),
    "b-mcp": _server_state(
      server_id="b-mcp",
      tool_name="lookup",
      description="Lookup from b-mcp.",
      prefix="b_",
    ),
  }
  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: None,
  )

  descriptor_a = manager.get_registered_mcp_tool_descriptor_for_sdk_tool(
    "mcp__legacy-a__lookup"
  )
  descriptor_b = manager.get_registered_mcp_tool_descriptor_for_sdk_tool(
    "mcp__b-mcp__lookup"
  )

  assert descriptor_a.identity == declaration_a.identity
  assert descriptor_a.exposed_name == "a_lookup"
  assert descriptor_a.declaration.semantics.effect == "read"
  assert descriptor_b.identity == declaration_b.identity
  assert descriptor_b.exposed_name == "b_lookup"
  assert descriptor_b.declaration.semantics.effect == "state_write"
















def test_manager_rejects_an_unknown_live_route_during_topology_compile() -> None:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"research-mcp"},
    tool_registration_catalog=_catalog(
      (_declaration("research-mcp", "declared_tool"),),
      (_server("research-mcp"),),
    ),
  )
  _install_physical_live_tool(
    manager,
    server_id="research-mcp",
    tool_name="undeclared_tool",
  )

  with pytest.raises(
    RegisteredMcpToolCompilationError,
    match="no exact static registration",
  ):
    manager._apply_collision_filtering(
      policy_server_for_tool=lambda _name: "research-mcp"
    )


def test_manager_uses_registered_per_tool_timeout() -> None:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"research-mcp"},
    tool_registration_catalog=_catalog(
      (_declaration("research-mcp", "filings_read"),),
      (_server(
        "research-mcp",
        default_timeout_seconds=41,
        per_tool_timeout_seconds={"filings_read": 73},
      ),),
    ),
  )
  _install_physical_live_tool(
    manager,
    server_id="research-mcp",
    tool_name="filings_read",
  )
  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: "research-mcp"
  )

  assert manager._timeout_for_tool(
    "research-mcp",
    "filings_read",
    "filings_read",
  ) == 73


def test_manager_uses_logical_registration_timeout_on_physical_transport() -> None:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"market-data-mcp"},
    logical_server_routes={"market-data-mcp": "provider-mcp"},
    logical_tool_aliases={
      "market-data-mcp": {"fetch_financials": "provider_fetch"},
    },
    tool_registration_catalog=_catalog(
      (_declaration("market-data-mcp", "fetch_financials"),),
      (_server(
        "market-data-mcp",
        transport_server_id="provider-mcp",
        default_timeout_seconds=41,
        per_tool_timeout_seconds={"fetch_financials": 73},
      ),),
    ),
  )
  _install_physical_live_tool(
    manager,
    server_id="provider-mcp",
    tool_name="provider_fetch",
  )

  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: "market-data-mcp",
  )

  descriptor = manager.get_registered_mcp_tool_descriptor(
    "fetch_financials"
  )
  assert descriptor.identity.logical_server_id == "market-data-mcp"
  assert descriptor.live_binding.transport_server_id == "provider-mcp"
  assert descriptor.live_binding.provider_original_name == "provider_fetch"
  assert manager.get_server_for_tool("fetch_financials") == "market-data-mcp"
  assert manager.get_original_tool_name("fetch_financials") == "provider_fetch"
  assert manager._timeout_for_tool(
    "provider-mcp",
    "fetch_financials",
    "provider_fetch",
  ) == 73




def test_manager_rebuilds_and_clears_registered_descriptor_cache() -> None:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"research-mcp"},
    tool_registration_catalog=_catalog(
      (_declaration("research-mcp", "filings_read"),),
      (_server("research-mcp"),),
    ),
  )
  _install_physical_live_tool(
    manager,
    server_id="research-mcp",
    tool_name="filings_read",
  )

  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: "research-mcp"
  )
  first = manager.get_registered_mcp_tool_descriptor("filings_read")
  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: "research-mcp"
  )
  second = manager.get_registered_mcp_tool_descriptor("filings_read")

  assert second == first
  assert second is not first

  asyncio.run(manager.shutdown())
  with pytest.raises(
    UnknownRegisteredMcpToolDescriptorError,
    match="unknown registered live MCP tool",
  ):
    manager.get_registered_mcp_tool_descriptor("filings_read")


@pytest.mark.parametrize(
  ("effect", "idempotent", "caller_allows"),
  (
    ("read", True, False),
    ("read", False, True),
    ("state_write", True, True),
  ),
)
def test_manager_replay_controls_can_narrow_but_never_widen_registration(
  effect: str,
  idempotent: bool,
  caller_allows: bool,
) -> None:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"research-mcp"},
    tool_registration_catalog=_catalog(
      (_declaration(
        "research-mcp",
        "filings_read",
        effect=effect,
        idempotent=idempotent,
      ),),
      (_server("research-mcp"),),
    ),
  )
  _install_physical_live_tool(
    manager,
    server_id="research-mcp",
    tool_name="filings_read",
  )
  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: "research-mcp"
  )
  replay_attempts: list[str] = []
  future_reconnects: list[str] = []

  async def fail_once(**_kwargs: object) -> object:
    raise EOFError("connection closed")

  async def unexpected_replay(**_kwargs: object) -> object:
    replay_attempts.append("replay")
    return object()

  async def reconnect_for_future(**_kwargs: object) -> bool:
    future_reconnects.append("reconnect")
    return True

  manager._call_tool_once = fail_once  # type: ignore[method-assign]
  manager._retry_stdio_tool_call_after_reconnect = (  # type: ignore[method-assign]
    unexpected_replay
  )
  manager._reconnect_stdio_server_for_future = (
    reconnect_for_future
  )

  result, error = asyncio.run(manager.call_tool(
    "filings_read",
    PreparedToolCall({}),
    allow_uncertain_replay=caller_allows,
  ))

  assert result is None
  assert error is not None
  assert replay_attempts == []
  assert future_reconnects == ["reconnect"]


def test_manager_allows_registered_idempotent_read_replay() -> None:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"research-mcp"},
    tool_registration_catalog=_catalog(
      (_declaration("research-mcp", "filings_read"),),
      (_server("research-mcp"),),
    ),
  )
  _install_physical_live_tool(
    manager,
    server_id="research-mcp",
    tool_name="filings_read",
  )
  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: "research-mcp"
  )
  replay_attempts: list[str] = []

  async def fail_once(**_kwargs: object) -> object:
    raise EOFError("connection closed")

  async def replay(**_kwargs: object) -> object:
    replay_attempts.append("replay")
    return SimpleNamespace(
      isError=False,
      structuredContent={"status": "ok"},
      content=[],
    )

  manager._call_tool_once = fail_once  # type: ignore[method-assign]
  manager._retry_stdio_tool_call_after_reconnect = replay  # type: ignore[method-assign]
  manager._translate_provider_symbol = (
    lambda *_args, **_kwargs: (_ for _ in ()).throw(
      AssertionError("registered calls must already be prepared")
    )
  )

  result, error = asyncio.run(manager.call_tool(
    "filings_read",
    PreparedToolCall({}),
    allow_uncertain_replay=True,
  ))

  assert error is None
  assert result == {"status": "ok"}
  assert replay_attempts == ["replay"]


def test_compiler_detaches_every_input_and_materializes_fresh_definitions() -> None:
  declaration = _declaration("research-mcp", "filings_read")
  server = _server("research-mcp")
  catalog = _catalog((declaration,), (server,))
  binding = _binding(
    server_id="research-mcp",
    exposed_name="filings_read",
    logical_name="filings_read",
    transport_server_id="research-mcp",
    provider_original_name="filings_read",
  )

  descriptor = compile_registered_mcp_tool_descriptors(
    catalog,
    (binding,),
  )[0]

  assert descriptor.declaration is not catalog.declarations[0]
  assert descriptor.server is not catalog.servers[0]
  assert descriptor.live_binding is not binding
  assert descriptor.live_binding.originated_definition is not (
    binding.originated_definition
  )
  with pytest.raises(FrozenInstanceError):
    descriptor.server = server  # type: ignore[misc]
  first = descriptor.materialize_provider_definition()
  second = descriptor.materialize_provider_definition()
  first["name"] = "mutated"
  first_schema = first["input_schema"]
  second_schema = second["input_schema"]
  assert isinstance(first_schema, dict)
  assert isinstance(second_schema, dict)
  first_required = first_schema["required"]
  assert isinstance(first_required, list)
  first_required.append("other")
  assert second["name"] == "filings_read"
  assert second_schema["required"] == ["ticker"]


def test_compiler_detaches_hostile_mapping_proxy_backing() -> None:
  declaration = _declaration("research-mcp", "filings_read")
  catalog = _catalog((declaration,), (_server("research-mcp"),))
  binding = _binding(
    server_id="research-mcp",
    exposed_name="filings_read",
    logical_name="filings_read",
    transport_server_id="research-mcp",
    provider_original_name="filings_read",
  )
  schema_backing = {"type": "object"}
  definition_backing = {
    "name": "filings_read",
    "input_schema": MappingProxyType(schema_backing),
  }
  object.__setattr__(
    binding.originated_definition,
    "definition",
    MappingProxyType(definition_backing),
  )

  descriptor = compile_registered_mcp_tool_descriptors(catalog, (binding,))[0]
  definition_backing["name"] = "mutated"
  schema_backing["type"] = "array"

  assert descriptor.exposed_name == "filings_read"
  assert descriptor.materialize_provider_definition()["input_schema"] == {
    "type": "object"
  }


@pytest.mark.parametrize("bindings", [[], {}, "binding", iter(())])
def test_compiler_requires_an_exact_binding_tuple(bindings: object) -> None:
  with pytest.raises(TypeError, match="exact tuple"):
    compile_registered_mcp_tool_descriptors(
      _catalog((), ()),
      bindings,  # type: ignore[arg-type]
    )


def test_compiler_rejects_unknown_or_non_mcp_static_routes() -> None:
  binding = _binding(
    server_id="research-mcp",
    exposed_name="filings_read",
    logical_name="filings_read",
    transport_server_id="research-mcp",
    provider_original_name="filings_read",
  )
  local_declaration = ToolRegistrationDeclaration(
    identity=RegisteredToolIdentity(
      route_kind="local_handler",
      logical_name="filings_read",
    ),
    semantics=_declaration("research-mcp", "unused").semantics,
  )
  catalog = _catalog(
    (local_declaration,),
    (_server("research-mcp"),),
  )

  with pytest.raises(
    RegisteredMcpToolCompilationError,
    match="no exact static registration",
  ):
    compile_registered_mcp_tool_descriptors(catalog, (binding,))


def test_descriptor_rejects_non_mcp_static_identity() -> None:
  local_declaration = ToolRegistrationDeclaration(
    identity=RegisteredToolIdentity(
      route_kind="local_handler",
      logical_name="filings_read",
    ),
    semantics=_declaration("research-mcp", "unused").semantics,
  )
  binding = _binding(
    server_id="research-mcp",
    exposed_name="filings_read",
    logical_name="filings_read",
    transport_server_id="research-mcp",
    provider_original_name="filings_read",
  )

  with pytest.raises(RegisteredMcpToolCompilationError, match="declaration"):
    RegisteredMcpToolDescriptor(
      declaration=local_declaration,
      server=_server("research-mcp"),
      live_binding=binding,
    )


def test_compiler_rejects_static_transport_mismatch() -> None:
  declaration = _declaration("market-data-mcp", "get_quote")
  catalog = _catalog(
    (declaration,),
    (_server("market-data-mcp", transport_server_id="wrong-mcp"),),
  )
  binding = _binding(
    server_id="market-data-mcp",
    exposed_name="get_quote",
    logical_name="get_quote",
    transport_server_id="provider-mcp",
    provider_original_name="provider_quote",
    route_kind="logical",
  )

  with pytest.raises(RegisteredMcpToolCompilationError, match="transport"):
    compile_registered_mcp_tool_descriptors(catalog, (binding,))


def test_compiler_accepts_manager_same_server_physical_and_logical_routes() -> None:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"same-mcp"},
    logical_server_routes={"same-mcp": "same-mcp"},
    logical_tool_aliases={
      "same-mcp": {"alias_tool": "provider_tool"},
    },
    provider_ids_by_server={"same-mcp": "provider"},
  )
  server_state = _server_state(
    server_id="same-mcp",
    tool_name="provider_tool",
    description="Provider definition.",
    prefix="",
  )
  server_state.config = None
  manager._servers = {"same-mcp": server_state}
  manager._apply_collision_filtering(
    policy_server_for_tool=lambda _name: "same-mcp"
  )
  manager_bindings = manager.get_server_tool_route_bindings({"same-mcp"})
  catalog = _catalog(
    (
      _declaration("same-mcp", "provider_tool"),
      _declaration("same-mcp", "alias_tool"),
    ),
    (_server("same-mcp"),),
  )

  descriptors = compile_registered_mcp_tool_descriptors(
    catalog,
    manager_bindings,
  )

  assert tuple(descriptor.identity.logical_name for descriptor in descriptors) == (
    "alias_tool",
    "provider_tool",
  )
  assert {
    descriptor.identity.logical_name: descriptor.live_binding.route_kind
    for descriptor in descriptors
  } == {
    "alias_tool": "logical",
    "provider_tool": "physical",
  }
  assert all(
    descriptor.server.logical_server_id
    == descriptor.server.transport_server_id
    == "same-mcp"
    for descriptor in descriptors
  )


def test_compiler_rejects_duplicate_identity_and_exposed_name() -> None:
  declaration = _declaration("research-mcp", "filings_read")
  catalog = _catalog((declaration,), (_server("research-mcp"),))
  binding = _binding(
    server_id="research-mcp",
    exposed_name="filings_read",
    logical_name="filings_read",
    transport_server_id="research-mcp",
    provider_original_name="filings_read",
  )
  with pytest.raises(RegisteredMcpToolCompilationError, match="identity"):
    compile_registered_mcp_tool_descriptors(catalog, (binding, binding))

  declaration_a = _declaration("a-mcp", "shared")
  declaration_b = _declaration("b-mcp", "shared")
  binding_a = _binding(
    server_id="a-mcp",
    exposed_name="shared",
    logical_name="shared",
    transport_server_id="a-transport",
    provider_original_name="provider_a",
    route_kind="logical",
  )
  binding_b = _binding(
    server_id="b-mcp",
    exposed_name="shared",
    logical_name="shared",
    transport_server_id="b-transport",
    provider_original_name="provider_b",
    route_kind="logical",
  )
  collision_catalog = _catalog(
    (declaration_a, declaration_b),
    (
      _server("a-mcp", transport_server_id="a-transport"),
      _server("b-mcp", transport_server_id="b-transport"),
    ),
  )
  with pytest.raises(RegisteredMcpToolCompilationError, match="exposed"):
    compile_registered_mcp_tool_descriptors(
      collision_catalog,
      (binding_a, binding_b),
    )


def test_compiler_has_an_explicit_registration_key_collision_guard(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  declarations = (
    _declaration("a-mcp", "tool_a"),
    _declaration("b-mcp", "tool_b"),
  )
  catalog = _catalog(
    declarations,
    (_server("a-mcp"), _server("b-mcp")),
  )
  bindings = (
    _binding(
      server_id="a-mcp",
      exposed_name="tool_a",
      logical_name="tool_a",
      transport_server_id="a-mcp",
      provider_original_name="tool_a",
    ),
    _binding(
      server_id="b-mcp",
      exposed_name="tool_b",
      logical_name="tool_b",
      transport_server_id="b-mcp",
      provider_original_name="tool_b",
    ),
  )
  monkeypatch.setattr(
    RegisteredMcpToolDescriptor,
    "registration_key",
    property(lambda _self: "tool-registration:sha256:" + "0" * 64),
  )

  with pytest.raises(RegisteredMcpToolCompilationError, match="registration key"):
    compile_registered_mcp_tool_descriptors(catalog, bindings)


def test_compiler_rejects_malformed_or_forged_bindings() -> None:
  declaration = _declaration("research-mcp", "filings_read")
  catalog = _catalog((declaration,), (_server("research-mcp"),))
  with pytest.raises(TypeError, match="exact contract"):
    compile_registered_mcp_tool_descriptors(
      catalog,
      (object(),),  # type: ignore[arg-type]
    )

  binding = _binding(
    server_id="research-mcp",
    exposed_name="filings_read",
    logical_name="filings_read",
    transport_server_id="research-mcp",
    provider_original_name="filings_read",
  )
  object.__setattr__(binding, "logical_name", " padded ")
  with pytest.raises(ValueError, match="trimmed"):
    compile_registered_mcp_tool_descriptors(catalog, (binding,))

  forged = _binding(
    server_id="research-mcp",
    exposed_name="filings_read",
    logical_name="filings_read",
    transport_server_id="research-mcp",
    provider_original_name="filings_read",
  )
  object.__setattr__(
    forged.originated_definition,
    "definition",
    {"name": "filings_read"},
  )
  with pytest.raises(TypeError, match="frozen mapping"):
    compile_registered_mcp_tool_descriptors(catalog, (forged,))


def test_compiler_requires_an_exact_detachable_static_catalog() -> None:
  with pytest.raises(TypeError, match="exact ToolRegistrationCatalog"):
    compile_registered_mcp_tool_descriptors(
      object(),  # type: ignore[arg-type]
      (),
    )

  declaration = _declaration("research-mcp", "filings_read")
  catalog = _catalog((declaration,), (_server("research-mcp"),))
  object.__setattr__(catalog, "declarations", [declaration])
  with pytest.raises(TypeError, match="exact tuple"):
    compile_registered_mcp_tool_descriptors(catalog, ())


def test_module_direct_source_dependencies_are_exact_and_not_root_reexported() -> None:
  source = (ROOT / "agent_gateway" / "tool_registration.py").read_text(
    encoding="utf-8"
  )
  allowed_absolute = {
    "__future__": {"annotations"},
    "dataclasses": {"dataclass"},
    "agent_workflow_contracts.tool_registration": {
      "RegisteredToolIdentity",
      "RegisteredToolServerDescriptor",
      "ToolRegistrationCatalog",
      "ToolRegistrationDeclaration",
      "UnknownToolRegistrationError",
      "validate_registered_tool_server_descriptor",
      "validate_tool_registration_catalog",
      "validate_tool_registration_declaration",
    },
  }
  allowed_relative = {
    "tool_definition": {
      "LiveToolRouteBinding",
      "validate_live_tool_route_binding",
    },
  }
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
      pytest.fail(f"unexpected import statement: {ast.unparse(node)}")
    if not isinstance(node, ast.ImportFrom):
      continue
    assert all(alias.asname is None for alias in node.names)
    if node.level == 0:
      assert node.module in allowed_absolute
      assert {alias.name for alias in node.names} == allowed_absolute[node.module]
    else:
      assert node.level == 1
      assert node.module in allowed_relative
      assert {alias.name for alias in node.names} == allowed_relative[node.module]

  package_init = (ROOT / "agent_gateway" / "__init__.py").read_text(
    encoding="utf-8"
  )
  assert "tool_registration" not in package_init
  assert "mcp_client" not in source
  assert "agent." not in source
