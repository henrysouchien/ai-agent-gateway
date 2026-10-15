"""The activation fold and its derivations (T3-I12 / D-B7-1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GATEWAY_DIR = Path(__file__).resolve().parents[1]
if str(GATEWAY_DIR) not in sys.path:
  sys.path.insert(0, str(GATEWAY_DIR))

from agent_gateway.mcp_activation import (  # noqa: E402
  McpActivationError,
  McpActivationFold,
  derive_live_surface,
  live_tool_surface,
)


SERVER_CATALOG = {
  "portfolio-reads-mcp": {"tools": ["get_positions", "get_returns"]},
  "market-data-mcp": {"tools": ["screen_stocks", "compare_peers"]},
}


class _Profile:
  def __init__(
    self,
    *,
    core_mcp_tools=None,
    mcp_tool_ceilings=None,
    unscoped_active_mcp_servers=frozenset(),
    denied_mcp_servers=frozenset(),
  ) -> None:
    self.core_mcp_tools = core_mcp_tools or {}
    self.mcp_tool_ceilings = mcp_tool_ceilings or {}
    self.unscoped_active_mcp_servers = unscoped_active_mcp_servers
    self.denied_mcp_servers = denied_mcp_servers


def _surface(fold, *, profile=None, denied=None):
  return derive_live_surface(
    profile=profile,
    server_catalog=SERVER_CATALOG,
    activation_fold=fold,
    denied_mcp_servers=denied,
  )


def test_fold_is_append_only_and_orders_its_records() -> None:
  fold = McpActivationFold()
  first = fold.record("market-data-mcp", tools=["screen_stocks"], source="load_tools")
  second = fold.record("market-data-mcp", tools=["compare_peers"], source="load_tools")

  assert fold.records == (first, second)
  assert fold.activated_servers == frozenset({"market-data-mcp"})
  assert fold.granted_tools("market-data-mcp") == frozenset(
    {"screen_stocks", "compare_peers"}
  )
  assert not hasattr(fold, "remove")
  assert not hasattr(fold, "difference_update")


def test_fold_rejects_an_empty_server_id() -> None:
  with pytest.raises(McpActivationError):
    McpActivationFold().record("   ", tools=["screen_stocks"])


def test_whole_server_activation_is_distinct_from_a_scoped_one() -> None:
  fold = McpActivationFold()
  fold.record("market-data-mcp", tools=None, source="run_agent")

  assert fold.whole_servers == frozenset({"market-data-mcp"})
  assert fold.granted_tools("market-data-mcp") == frozenset()


def test_empty_fold_surface_is_every_catalog_server_under_the_profile_ceiling() -> None:
  # No channel tier withholds a server: both connected servers are live. The
  # registered ceiling still scopes portfolio-reads; market-data registers no
  # ceiling and advertises its whole tool list.
  profile = _Profile(core_mcp_tools={"portfolio-reads-mcp": {"get_positions"}})

  surface = _surface(McpActivationFold(), profile=profile)

  assert surface.active_servers == frozenset(SERVER_CATALOG)
  assert surface.allowed_mcp_tools_by_server == {
    "portfolio-reads-mcp": frozenset({"get_positions"}),
    "market-data-mcp": frozenset({"screen_stocks", "compare_peers"}),
  }
  assert surface.deferred_mcp_tools == frozenset({"get_returns"})
  assert surface.deferred_mcp_tool_ids == frozenset({
    "mcp__portfolio-reads-mcp__get_returns",
  })


def test_live_surface_detaches_and_freezes_nested_authority_inputs() -> None:
  catalog = {
    "research-corpus-mcp": {
      "tools": ["corpus_search"],
      "metadata": {"labels": ["research"]},
    },
  }
  allowed = {"research-corpus-mcp": {"corpus_search"}}

  surface = live_tool_surface(
    active_servers={"research-corpus-mcp"},
    server_catalog=catalog,
    allowed_mcp_tools_by_server=allowed,
  )
  catalog["research-corpus-mcp"]["tools"].append("corpus_write")
  allowed["research-corpus-mcp"].add("corpus_write")

  assert surface.server_catalog["research-corpus-mcp"]["tools"] == (
    "corpus_search",
  )
  assert surface.allowed_mcp_tools_by_server["research-corpus-mcp"] == (
    frozenset({"corpus_search"})
  )
  with pytest.raises(TypeError):
    surface.server_catalog["other-mcp"] = {}  # type: ignore[index]
  with pytest.raises(TypeError):
    surface.server_catalog["research-corpus-mcp"]["metadata"]["extra"] = True
  with pytest.raises(TypeError):
    surface.allowed_mcp_tools_by_server["research-corpus-mcp"] = frozenset()  # type: ignore[index]


def test_one_activation_moves_advertised_and_allowed_together() -> None:
  # The desync class: a pack could leave the deferred set while the allowlist
  # never learned the grant. One fold makes the two the same fact.
  profile = _Profile(core_mcp_tools={
    "portfolio-reads-mcp": {"get_positions"},
    "market-data-mcp": set(),
  })
  fold = McpActivationFold()
  fold.record("market-data-mcp", tools=["screen_stocks"], source="load_tools")

  surface = _surface(fold, profile=profile)

  assert "market-data-mcp" in surface.active_servers
  assert surface.allowed_mcp_tools_by_server["market-data-mcp"] == frozenset(
    {"screen_stocks"}
  )
  assert "screen_stocks" not in surface.deferred_mcp_tools
  assert "compare_peers" in surface.deferred_mcp_tools


def test_every_advertised_tool_is_an_allowed_tool_for_any_fold() -> None:
  profile = _Profile(core_mcp_tools={"portfolio-reads-mcp": {"get_positions"}})
  fold = McpActivationFold()
  fold.record("market-data-mcp", tools=["screen_stocks"], source="load_tools")
  fold.record("portfolio-reads-mcp", tools=None, source="run_agent")

  surface = _surface(fold, profile=profile)

  advertised = {
    tool_name
    for server_name in surface.active_servers
    for tool_name in SERVER_CATALOG[server_name]["tools"]
    if tool_name not in surface.deferred_mcp_tools
  }
  allowed = {
    tool_name
    for tool_names in surface.allowed_mcp_tools_by_server.values()
    for tool_name in tool_names
  }
  assert advertised <= allowed


def test_a_denied_server_is_absent_from_the_surface_even_when_activated() -> None:
  profile = _Profile(core_mcp_tools={"portfolio-reads-mcp": {"get_positions"}})
  fold = McpActivationFold()
  fold.record("market-data-mcp", tools=["screen_stocks"], source="load_tools")

  surface = _surface(fold, profile=profile, denied={"market-data-mcp"})

  assert surface.active_servers == frozenset({"portfolio-reads-mcp"})
  assert "market-data-mcp" not in surface.allowed_mcp_tools_by_server
  assert "market-data-mcp" not in surface.server_catalog


def test_a_profileless_surface_advertises_every_active_server_tool() -> None:
  fold = McpActivationFold()
  fold.record("market-data-mcp", tools=["screen_stocks"], source="load_tools")

  surface = _surface(fold, profile=None)

  assert surface.deferred_mcp_tools == frozenset()
  assert surface.allowed_mcp_tools_by_server["market-data-mcp"] == frozenset(
    {"screen_stocks", "compare_peers"}
  )


def test_an_unscoped_active_server_defers_nothing() -> None:
  profile = _Profile(
    core_mcp_tools={"portfolio-reads-mcp": {"get_positions"}},
    unscoped_active_mcp_servers=frozenset({"portfolio-reads-mcp"}),
  )

  surface = _surface(McpActivationFold(), profile=profile)

  assert surface.allowed_mcp_tools_by_server["portfolio-reads-mcp"] == frozenset(
    {"get_positions", "get_returns"}
  )
  assert "get_returns" not in surface.deferred_mcp_tools


def test_a_registered_ceiling_bounds_every_grant_on_every_route() -> None:
  # The invite ceiling: a profile that may reach only `get_current_model` on
  # the portfolio server keeps that bound whether a load grant names other
  # tools, or a run_agent activation takes the whole server. Default-visible
  # (`core_mcp_tools`) is a different fact and grants still widen it.
  catalog = {
    "portfolio-reads-mcp": {"tools": ["get_current_model", "get_positions", "execute_trade"]},
    "market-data-mcp": {"tools": ["screen_stocks", "compare_peers"]},
  }
  profile = _Profile(
    core_mcp_tools={"portfolio-reads-mcp": {"get_current_model"}, "market-data-mcp": set()},
    mcp_tool_ceilings={"portfolio-reads-mcp": {"get_current_model"}},
  )
  fold = McpActivationFold()
  fold.record("portfolio-reads-mcp", tools=["execute_trade", "get_positions"], source="load_tools")
  fold.record("market-data-mcp", tools=["screen_stocks"], source="load_tools")

  surface = derive_live_surface(profile=profile, server_catalog=catalog, activation_fold=fold)

  assert surface.allowed_mcp_tools_by_server == {
    "portfolio-reads-mcp": frozenset({"get_current_model"}),
    "market-data-mcp": frozenset({"screen_stocks"}),
  }
  # The load grant widened nothing past the ceiling; the granted names are
  # visible (a granted name is never deferred) but ceilinged out of allowed,
  # so the dispatcher refuses them even though the prompt may mention them.
  assert surface.deferred_mcp_tools == {"compare_peers"}
  # Ungranted default-visible truth: compare_peers is catalog-only and deferred.
  assert "compare_peers" not in surface.allowed_mcp_tools_by_server["market-data-mcp"]

  fold.record("portfolio-reads-mcp", tools=None, source="run_agent")
  surface = derive_live_surface(profile=profile, server_catalog=catalog, activation_fold=fold)

  assert surface.allowed_mcp_tools_by_server["portfolio-reads-mcp"] == frozenset({"get_current_model"})
  assert surface.deferred_mcp_tools == {"compare_peers"}
