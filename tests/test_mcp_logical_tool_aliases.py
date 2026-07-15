from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import McpClientManager


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
    "fmp-mcp": SimpleNamespace(
      name="fmp-mcp",
      session=object(),
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
    "fmp-mcp": SimpleNamespace(
      name="fmp-mcp",
      session=object(),
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
  monkeypatch.setattr(manager, "_translate_provider_symbol", lambda _name, payload: payload)

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
  monkeypatch.setattr(manager, "_translate_provider_symbol", lambda _name, payload: payload)

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
  monkeypatch.setattr(mcp_client_module, "load_server_policy_module", lambda: None)
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
    "fmp-mcp": SimpleNamespace(
      name="fmp-mcp",
      session=object(),
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


def test_committed_catalog_diff_pins_the_additive_alias_set() -> None:
  repo_root = Path(__file__).resolve().parents[3]
  evidence = json.loads(
    (repo_root / "docs/qa/provider-port-phase6-tool-catalog-diff.json").read_text()
  )

  assert evidence["phase"] == "6-retirement"
  assert evidence["logical_server"] == "market-data-mcp"
  assert evidence["logical_server_keys_after"] == 1
  assert evidence["branded_rows_removed"] == len(ALIASES)
  assert evidence["logical_market_data_rows_after"] == 21
  assert set(ALIASES) <= set(evidence["neutral_tools"])
  assert not {
    original_name
    for original_name, _tool_input in ALIASES.values()
  } & set(evidence["neutral_tools"])
  assert evidence["input_schema_changes"] == 0
  assert evidence["effect_class_changes"] == 0
  assert evidence["channel_tier_changes"] == 0


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
