from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.mcp_client_connections import McpToolCallResult


ALIASES = {
  "check_market_cap": ("fmp_market_cap_check", {"symbol": "MSFT"}),
  "describe_market_data_endpoint": ("fmp_describe", {"endpoint": "income_statement"}),
  "fetch_company_profile": ("fmp_profile", {"symbol": "MSFT"}),
  "fetch_financials": (
    "fmp_fetch",
    {
      "endpoint": "income_statement",
      "symbol": "MSFT",
      "period": "annual",
      "limit": 3,
      "columns": "date,revenue,grossProfit",
    },
  ),
  "list_market_data_endpoints": ("fmp_list_endpoints", {"category": "financials"}),
  "search_companies": ("fmp_search", {"query": "Microsoft", "limit": 3}),
}
IDENTITY_TOOLS = {
  "get_market_context": {"symbol": "MSFT"},
  "get_news": {"symbols": "MSFT"},
}

class _UnusedMcpSession:
  async def call_tool(
    self,
    name: str,
    arguments: dict[str, object],
    *,
    read_timeout_seconds: timedelta,
    meta: dict[str, object] | None = None,
  ) -> McpToolCallResult:
    _ = name, arguments, read_timeout_seconds, meta
    raise AssertionError("physical session calls are intercepted by these tests")


def _manager() -> McpClientManager:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"fmp-mcp", "market-data-mcp"},
    logical_server_routes={"market-data-mcp": "fmp-mcp"},
    logical_tool_aliases={
      "market-data-mcp": {
        alias_name: original_name
        for alias_name, (original_name, _tool_input) in ALIASES.items()
      },
    },
    provider_ids_by_server={"fmp-mcp": "fmp"},
  )
  tool_definitions = [
    {
      "name": original_name,
      "description": f"Physical definition for {original_name}",
      "input_schema": {
        "type": "object",
        "properties": {key: {"type": "string"} for key in tool_input},
      },
    }
    for original_name, tool_input in ALIASES.values()
  ]
  manager._servers = {
    "fmp-mcp": mcp_client_module._ServerState(
      name="fmp-mcp",
      session=_UnusedMcpSession(),
      exit_contexts=[],
      tool_definitions=tool_definitions,
      tool_names={tool["name"] for tool in tool_definitions},
      tool_prefix="",
      config=None,
    )
  }

  alias_names = set(ALIASES)
  manager._apply_collision_filtering(
    policy_server_for_tool=lambda tool_name: (
      "market-data-mcp" if tool_name in alias_names else "fmp-mcp"
    )
  )
  return manager


def _retirement_manager() -> McpClientManager:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"market-data-mcp"},
    logical_server_routes={"market-data-mcp": "fmp-mcp"},
    logical_tool_aliases={
      "market-data-mcp": {
        alias_name: original_name
        for alias_name, (original_name, _tool_input) in ALIASES.items()
      },
    },
    provider_ids_by_server={"fmp-mcp": "fmp"},
  )
  physical_tools = {
    original_name: tool_input
    for original_name, tool_input in ALIASES.values()
  } | IDENTITY_TOOLS
  manager._servers = {
    "fmp-mcp": mcp_client_module._ServerState(
      name="fmp-mcp",
      session=_UnusedMcpSession(),
      exit_contexts=[],
      tool_definitions=[
        {
          "name": tool_name,
          "description": "Physical market-data definition",
          "input_schema": {
            "type": "object",
            "properties": {key: {"type": "string"} for key in tool_input},
          },
        }
        for tool_name, tool_input in physical_tools.items()
      ],
      tool_names=set(physical_tools),
      tool_prefix="",
      config=None,
    )
  }
  logical_names = set(ALIASES) | set(IDENTITY_TOOLS)
  manager._apply_collision_filtering(
    policy_server_for_tool=lambda tool_name: (
      "market-data-mcp" if tool_name in logical_names else None
    )
  )
  return manager


def _two_logical_manager() -> McpClientManager:
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"physical-mcp", "logical-z-mcp", "logical-a-mcp"},
    logical_server_routes={
      "logical-z-mcp": "physical-mcp",
      "logical-a-mcp": "physical-mcp",
    },
    logical_tool_aliases={
      "logical-z-mcp": {"z_alias": "provider_z"},
      "logical-a-mcp": {"a_alias": "provider_a"},
    },
    provider_ids_by_server={"physical-mcp": "provider"},
  )
  manager._servers = {
    "physical-mcp": mcp_client_module._ServerState(
      name="physical-mcp",
      session=_UnusedMcpSession(),
      exit_contexts=[],
      tool_prefix="",
      tool_definitions=[
        {
          "name": "provider_a",
          "description": "Provider A.",
          "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "string"}},
          },
        },
        {
          "name": "provider_z",
          "description": "Provider Z.",
          "input_schema": {
            "type": "object",
            "properties": {"z": {"type": "string"}},
          },
        },
      ],
      tool_names={"provider_a", "provider_z"},
      config=None,
    ),
  }
  owner_by_alias = {
    "z_alias": "logical-z-mcp",
    "a_alias": "logical-a-mcp",
  }
  manager._apply_collision_filtering(
    policy_server_for_tool=owner_by_alias.get,
  )
  return manager


@pytest.mark.parametrize(
  ("alias_name", "original_name", "tool_input"),
  [
    (alias_name, original_name, tool_input)
    for alias_name, (original_name, tool_input) in sorted(ALIASES.items())
  ],
)
def test_logical_alias_behavior_matches_branded_original(
  monkeypatch: pytest.MonkeyPatch,
  alias_name: str,
  original_name: str,
  tool_input: dict[str, object],
) -> None:
  manager = _manager()
  calls: list[dict[str, object]] = []

  async def fake_call_tool_once(**kwargs):
    calls.append({
      "server": kwargs["server"],
      "original_name": kwargs["original_name"],
      "tool_input": kwargs["tool_input"],
    })
    return SimpleNamespace(
      isError=False,
      structuredContent={
        "status": "ok",
        "tool": kwargs["original_name"],
        "input": kwargs["tool_input"],
      },
      content=None,
    )

  monkeypatch.setattr(manager, "_call_tool_once", fake_call_tool_once)
  monkeypatch.setattr(
    manager,
    "_translate_provider_symbol",
    lambda _server, _name, payload: payload,
  )

  branded_result, branded_error = asyncio.run(manager.call_tool(original_name, dict(tool_input)))
  alias_result, alias_error = asyncio.run(manager.call_tool(alias_name, dict(tool_input)))

  assert branded_error is None
  assert alias_error is None
  assert alias_result == branded_result
  assert calls[0]["server"] is calls[1]["server"]
  assert calls[0]["original_name"] == calls[1]["original_name"] == original_name
  assert calls[0]["tool_input"] == calls[1]["tool_input"] == tool_input
  assert manager.get_server_for_tool(original_name) == "fmp-mcp"
  assert manager.get_server_for_tool(alias_name) == "market-data-mcp"
  assert manager.get_provider_id_for_tool(original_name) == "fmp"
  assert manager.get_provider_id_for_tool(alias_name) == "fmp"
  assert manager.get_provider_id_for_tool(f"mcp__fmp-mcp__{original_name}") == "fmp"
  assert manager.get_provider_id_for_tool(f"mcp__market-data-mcp__{alias_name}") == "fmp"
  assert manager.get_provider_id_for_tool(f"mcp__market-data-mcp__{original_name}") is None


def test_logical_alias_catalog_is_additive_and_schema_identical() -> None:
  manager = _manager()

  branded_records = manager.get_server_tool_definition_records({"fmp-mcp"})
  logical_records = manager.get_server_tool_definition_records({
    "market-data-mcp"
  })
  assert all(record.origin == "mcp" for record in branded_records)
  assert all(record.server_id == "fmp-mcp" for record in branded_records)
  assert all(record.origin == "mcp" for record in logical_records)
  assert all(
    record.server_id == "market-data-mcp" for record in logical_records
  )
  assert manager.get_server_tool_definitions({"fmp-mcp"}) == [
    record.materialize() for record in branded_records
  ]
  assert manager.get_server_tool_definitions({"market-data-mcp"}) == [
    record.materialize() for record in logical_records
  ]
  combined_records = manager.get_server_tool_definition_records({
    "fmp-mcp",
    "market-data-mcp",
  })
  assert combined_records[:len(branded_records)] == branded_records
  assert combined_records[len(branded_records):] == logical_records

  branded_definitions = {
    tool["name"]: tool
    for tool in manager.get_server_tool_definitions({"fmp-mcp"})
  }
  logical_definitions = {
    tool["name"]: tool
    for tool in manager.get_server_tool_definitions({"market-data-mcp"})
  }

  assert set(branded_definitions) == {original for original, _args in ALIASES.values()}
  assert set(logical_definitions) == set(ALIASES)
  assert manager.get_server_names() == {"fmp-mcp", "market-data-mcp"}
  assert manager.get_server_catalog()["fmp-mcp"]["tools"] == sorted(branded_definitions)
  assert manager.get_server_catalog()["market-data-mcp"]["tools"] == sorted(ALIASES)

  for alias_name, (original_name, _tool_input) in ALIASES.items():
    assert (
      logical_definitions[alias_name]["input_schema"]
      == branded_definitions[original_name]["input_schema"]
    )
    assert manager.resolve_tool_name("market-data-mcp", original_name) == alias_name
    assert manager.get_original_tool_name(alias_name) == original_name

  assert manager.get_provider_id_for_tool("unknown_tool") is None


def test_logical_route_bindings_preserve_alias_transport_and_provider_identity() -> None:
  manager = _manager()
  branded_before = manager.get_server_tool_definitions({"fmp-mcp"})
  logical_before = manager.get_server_tool_definitions({"market-data-mcp"})

  branded = manager.get_server_tool_route_bindings({"fmp-mcp"})
  logical = manager.get_server_tool_route_bindings({"market-data-mcp"})
  combined = manager.get_server_tool_route_bindings({
    "fmp-mcp",
    "market-data-mcp",
  })

  assert tuple(binding.exposed_name for binding in branded) == tuple(
    definition["name"] for definition in branded_before
  )
  assert all(binding.route_kind == "physical" for binding in branded)
  assert all(binding.logical_server_id == "fmp-mcp" for binding in branded)
  assert all(binding.transport_server_id == "fmp-mcp" for binding in branded)
  assert all(binding.provider_id == "fmp" for binding in branded)

  assert tuple(binding.exposed_name for binding in logical) == tuple(
    definition["name"] for definition in logical_before
  )
  assert all(binding.route_kind == "logical" for binding in logical)
  assert all(
    binding.logical_server_id == "market-data-mcp" for binding in logical
  )
  assert all(binding.transport_server_id == "fmp-mcp" for binding in logical)
  assert all(binding.provider_id == "fmp" for binding in logical)
  for binding in logical:
    assert binding.logical_name == binding.exposed_name
    assert binding.provider_original_name == ALIASES[binding.exposed_name][0]
    assert manager.get_original_tool_name(binding.exposed_name) == (
      binding.provider_original_name
    )
    assert manager.get_policy_tool_name(binding.exposed_name) == (
      binding.logical_name
    )

  assert manager.get_policy_tool_name("fetch_financials") == "fetch_financials"
  assert manager.get_original_tool_name("fetch_financials") == "fmp_fetch"
  assert manager.get_policy_tool_name("fmp_fetch") == "fmp_fetch"
  assert manager.get_policy_tool_name("unknown") is None

  assert combined[:len(branded)] == branded
  assert combined[len(branded):] == logical
  assert manager.get_server_tool_definitions({"fmp-mcp"}) == branded_before
  assert manager.get_server_tool_definitions({
    "market-data-mcp"
  }) == logical_before


def test_policy_tool_name_fails_closed_on_corrupt_or_ambiguous_provenance() -> None:
  manager = _manager()
  manager._tool_to_server["fetch_financials"] = "wrong-mcp"

  assert manager.get_policy_tool_name("fetch_financials") is None


def test_logical_route_binding_order_uses_manager_mapping_not_requested_set() -> None:
  manager = _two_logical_manager()

  bindings = manager.get_server_tool_route_bindings({
    "logical-a-mcp",
    "logical-z-mcp",
  })

  assert tuple(binding.logical_server_id for binding in bindings) == (
    "logical-z-mcp",
    "logical-a-mcp",
  )
  assert tuple(binding.exposed_name for binding in bindings) == (
    "z_alias",
    "a_alias",
  )
  assert tuple(binding.provider_original_name for binding in bindings) == (
    "provider_z",
    "provider_a",
  )


def test_alias_generation_is_immutable_detached_and_cleared_on_shutdown() -> None:
  manager = _two_logical_manager()
  generation = manager._logical_alias_generation
  provenance = generation[0]
  current_physical = next(
    definition
    for definition in manager._servers["physical-mcp"].tool_definitions
    if definition["name"] == provenance.provider_original_name
  )

  assert type(generation) is tuple
  assert provenance.physical_definition.materialize() == current_physical
  assert provenance.logical_definition.materialize()["name"] == (
    provenance.exposed_name
  )
  current_physical["description"] = "Drifted after generation."
  assert provenance.physical_definition.materialize()["description"] != (
    current_physical["description"]
  )
  with pytest.raises(FrozenInstanceError):
    provenance.transport_server_id = "other-mcp"  # type: ignore[misc]
  with pytest.raises(TypeError):
    provenance.physical_definition.definition["name"] = "changed"  # type: ignore[index]

  asyncio.run(manager.shutdown())
  assert manager._logical_alias_generation == ()


def test_transport_only_logical_route_bindings_preserve_identity_tools() -> None:
  manager = _retirement_manager()
  bindings = manager.get_server_tool_route_bindings({"market-data-mcp"})

  assert {binding.exposed_name for binding in bindings} == (
    set(ALIASES) | set(IDENTITY_TOOLS)
  )
  assert all(binding.route_kind == "logical" for binding in bindings)
  assert all(
    binding.logical_server_id == "market-data-mcp" for binding in bindings
  )
  assert all(binding.transport_server_id == "fmp-mcp" for binding in bindings)
  assert all(binding.provider_id == "fmp" for binding in bindings)
  for binding in bindings:
    if binding.exposed_name in IDENTITY_TOOLS:
      assert binding.logical_name == binding.provider_original_name
    else:
      assert binding.provider_original_name == ALIASES[binding.exposed_name][0]


@pytest.mark.parametrize(
  ("corrupt", "match"),
  [
    ("owner", "owner"),
    ("route_missing", "transport mapping is missing"),
    ("transport_missing", "transport mapping is incoherent"),
    ("original_missing", "original mapping is missing"),
    ("original_malformed", "original mapping"),
  ],
)
def test_logical_route_bindings_refuse_live_map_corruption(
  corrupt: str,
  match: str,
) -> None:
  manager = _manager()
  exposed_name = next(iter(ALIASES))
  if corrupt == "owner":
    manager._tool_to_server[exposed_name] = "fmp-mcp"
  elif corrupt == "route_missing":
    manager._logical_server_routes.pop("market-data-mcp")
  elif corrupt == "transport_missing":
    manager._logical_server_routes["market-data-mcp"] = "missing-mcp"
  elif corrupt == "original_missing":
    manager._dispatch_to_original.pop(exposed_name)
  else:
    manager._dispatch_to_original[exposed_name] = " padded "

  with pytest.raises(ValueError, match=match):
    manager.get_server_tool_route_bindings({"market-data-mcp"})


@pytest.mark.parametrize(
  ("corrupt", "match"),
  [
    ("wrong_transport", "route diverges from alias generation"),
    ("wrong_existing_original", "original mapping"),
    ("physical_missing", "physical provenance"),
    ("physical_duplicate", "physical provenance"),
    ("physical_description", "physical definition diverges"),
    ("surface_missing", "surface provenance"),
    ("surface_duplicate", "surface provenance"),
    ("surface_schema", "definition diverges from surface"),
    ("transport_schema", "physical definition diverges"),
    ("generation_missing", "unique alias generation"),
    ("generation_duplicate", "unique alias generation"),
    ("logical_duplicate", "duplicate MCP live exposed route"),
  ],
)
def test_logical_route_bindings_require_exact_unique_provenance(
  corrupt: str,
  match: str,
) -> None:
  manager = _two_logical_manager()
  logical_server = "logical-z-mcp"
  exposed_name = "z_alias"
  provider_original_name = "provider_z"
  transport_state = manager._servers["physical-mcp"]
  physical_definition = next(
    definition
    for definition in transport_state.tool_definitions
    if definition["name"] == provider_original_name
  )
  logical_definition = manager._logical_tool_definitions[logical_server][0]
  surface_definition = next(
    definition
    for definition in manager._tool_definitions
    if definition["name"] == exposed_name
  )

  if corrupt == "wrong_transport":
    manager._servers["wrong-mcp"] = mcp_client_module._ServerState(
      name="wrong-mcp",
      session=_UnusedMcpSession(),
      exit_contexts=[],
      tool_prefix="",
      tool_definitions=[json.loads(json.dumps(physical_definition))],
      tool_names={provider_original_name},
      config=None,
    )
    manager._logical_server_routes[logical_server] = "wrong-mcp"
  elif corrupt == "wrong_existing_original":
    manager._dispatch_to_original[exposed_name] = "provider_a"
  elif corrupt == "physical_missing":
    transport_state.tool_definitions.remove(physical_definition)
  elif corrupt == "physical_duplicate":
    transport_state.tool_definitions.append(dict(physical_definition))
  elif corrupt == "physical_description":
    physical_definition["description"] = "Description-only drift."
  elif corrupt == "surface_missing":
    manager._tool_definitions.remove(surface_definition)
  elif corrupt == "surface_duplicate":
    manager._tool_definitions.append(dict(surface_definition))
  elif corrupt == "surface_schema":
    surface_index = manager._tool_definitions.index(surface_definition)
    divergent_surface = json.loads(json.dumps(surface_definition))
    divergent_surface["input_schema"] = {"type": "array"}
    manager._tool_definitions[surface_index] = divergent_surface
  elif corrupt == "transport_schema":
    physical_definition["input_schema"] = {"type": "array"}
  elif corrupt == "generation_missing":
    manager._logical_alias_generation = tuple(
      item
      for item in manager._logical_alias_generation
      if item.exposed_name != exposed_name
    )
  elif corrupt == "generation_duplicate":
    manager._logical_alias_generation = (
      *manager._logical_alias_generation,
      next(
        item
        for item in manager._logical_alias_generation
        if item.exposed_name == exposed_name
      ),
    )
  else:
    manager._logical_tool_definitions[logical_server].append(
      dict(logical_definition)
    )

  with pytest.raises(ValueError, match=match):
    manager.get_server_tool_route_bindings({logical_server})


def test_retirement_catalog_exposes_only_the_full_logical_surface() -> None:
  manager = _retirement_manager()
  expected_names = set(ALIASES) | set(IDENTITY_TOOLS)

  assert {tool["name"] for tool in manager.get_tool_definitions()} == expected_names
  assert {
    tool["name"]
    for tool in manager.get_server_tool_definitions({"market-data-mcp"})
  } == expected_names
  assert manager.get_server_tool_definitions({"fmp-mcp"}) == []
  assert manager.get_server_names() == {"market-data-mcp"}
  assert set(manager.get_server_catalog()) == {"market-data-mcp"}
  assert manager.get_server_catalog()["market-data-mcp"] == {
    "tool_count": len(expected_names),
    "tools": sorted(expected_names),
  }

  for alias_name, (physical_name, _tool_input) in ALIASES.items():
    assert manager.get_server_for_tool(alias_name) == "market-data-mcp"
    assert manager.get_server_for_tool(physical_name) is None
    assert manager.is_mcp_tool(physical_name) is False
  for identity_name in IDENTITY_TOOLS:
    assert manager.get_server_for_tool(identity_name) == "market-data-mcp"
    assert manager.get_original_tool_name(identity_name) == identity_name

  # The transport registration and its native catalog remain intact for dispatch.
  assert set(manager._servers) == {"fmp-mcp"}
  assert manager._servers["fmp-mcp"].tool_names == {
    physical_name for physical_name, _tool_input in ALIASES.values()
  } | set(IDENTITY_TOOLS)
  assert manager.get_startup_diagnostics() == {}


def test_retirement_dispatch_preserves_physical_session_and_provider_identity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  manager = _retirement_manager()
  calls: list[tuple[object, str, dict[str, object]]] = []

  async def fake_call_tool_once(**kwargs):
    calls.append((kwargs["server"], kwargs["original_name"], kwargs["tool_input"]))
    return SimpleNamespace(
      isError=False,
      structuredContent={"tool": kwargs["original_name"]},
      content=None,
    )

  monkeypatch.setattr(manager, "_call_tool_once", fake_call_tool_once)
  monkeypatch.setattr(
    manager,
    "_translate_provider_symbol",
    lambda _server, _name, payload: payload,
  )

  renamed_result, renamed_error = asyncio.run(
    manager.call_tool("fetch_financials", {"symbol": "MSFT"})
  )
  identity_result, identity_error = asyncio.run(
    manager.call_tool("get_news", {"symbols": "MSFT"})
  )
  physical_result, physical_error = asyncio.run(
    manager.call_tool(ALIASES["fetch_financials"][0], {"symbol": "MSFT"})
  )

  assert renamed_error is None
  assert identity_error is None
  assert physical_result is None
  assert physical_error is not None
  assert physical_error["code"] == "unknown_tool"
  assert renamed_result == {"tool": ALIASES["fetch_financials"][0]}
  assert identity_result == {"tool": "get_news"}
  assert calls[0][0] is calls[1][0] is manager._servers["fmp-mcp"]
  assert [call[1] for call in calls] == [ALIASES["fetch_financials"][0], "get_news"]
  assert manager.get_provider_id_for_tool("fetch_financials") == "fmp"
  assert manager.get_provider_id_for_tool("get_news") == "fmp"
  assert manager.get_provider_id_for_tool("mcp__market-data-mcp__get_news") == "fmp"
  assert manager.get_provider_id_for_tool(
    f"mcp__fmp-mcp__{ALIASES['fetch_financials'][0]}"
  ) is None


def test_logical_aliases_survive_shared_policy_import_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    mcp_client_module,
    "load_server_policy_helpers",
    lambda: (None, None, None),
  )
  manager = McpClientManager(
    config_path=None,
    allowed_servers={"fmp-mcp", "market-data-mcp"},
    logical_server_routes={"market-data-mcp": "fmp-mcp"},
    logical_tool_aliases={
      "market-data-mcp": {
        alias_name: original_name
        for alias_name, (original_name, _tool_input) in ALIASES.items()
      },
    },
  )
  manager._servers = {
    "fmp-mcp": mcp_client_module._ServerState(
      name="fmp-mcp",
      session=_UnusedMcpSession(),
      exit_contexts=[],
      tool_definitions=[
        {"name": original_name, "description": original_name, "input_schema": {}}
        for original_name, _tool_input in ALIASES.values()
      ],
      tool_names={original_name for original_name, _tool_input in ALIASES.values()},
      tool_prefix="",
      config=None,
    )
  }

  manager._apply_collision_filtering()

  assert {
    alias_name: manager.get_server_for_tool(alias_name)
    for alias_name in ALIASES
  } == {alias_name: "market-data-mcp" for alias_name in ALIASES}
  assert {
    alias_name: manager.get_original_tool_name(alias_name)
    for alias_name in ALIASES
  } == {
    alias_name: original_name
    for alias_name, (original_name, _tool_input) in ALIASES.items()
  }


def test_namespaced_provider_route_does_not_depend_on_live_catalog_startup() -> None:
  manager = McpClientManager(
    config_path=None,
    logical_server_routes={"market-data-mcp": "fmp-mcp"},
    logical_tool_aliases={
      "market-data-mcp": {
        alias_name: original_name
        for alias_name, (original_name, _tool_input) in ALIASES.items()
      },
    },
    provider_ids_by_server={"fmp-mcp": "fmp"},
  )

  assert manager.get_provider_id_for_tool("mcp__fmp-mcp__fmp_fetch") == "fmp"
  assert manager.get_provider_id_for_tool("mcp__market-data-mcp__fetch_financials") == "fmp"
  assert manager.get_provider_id_for_tool("mcp__market-data-mcp__fmp_fetch") is None
  assert manager.get_provider_id_for_tool("fetch_financials") is None




def test_logical_server_request_starts_only_the_physical_transport(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path,
) -> None:
  config_path = tmp_path / "mcp.json"
  config_path.write_text(json.dumps({
    "mcpServers": {
      "fmp-mcp": {"command": "physical-fmp", "type": "stdio"},
    },
  }))
  manager = McpClientManager(
    config_path=config_path,
    allowed_servers={"market-data-mcp"},
    logical_server_routes={"market-data-mcp": "fmp-mcp"},
  )
  connected: list[str] = []

  async def fake_connect(name, _config):
    connected.append(name)
    return None

  monkeypatch.setattr(manager, "_connect_or_warn", fake_connect)
  monkeypatch.setattr(manager, "_apply_collision_filtering", lambda: None)

  asyncio.run(manager.startup(allowed_servers={"market-data-mcp"}))

  assert connected == ["fmp-mcp"]
  assert manager.get_startup_diagnostics() == {}
