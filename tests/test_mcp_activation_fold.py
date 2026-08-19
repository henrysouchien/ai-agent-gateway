"""The activation fold and its derivations (T3-I12 / D-B7-1)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GATEWAY_DIR = ROOT / "packages" / "agent-gateway"
if str(GATEWAY_DIR) not in sys.path:
  sys.path.insert(0, str(GATEWAY_DIR))

from agent_gateway.event_adapter import (  # noqa: E402
  CONTROL_V1_FIELD_PROJECTION,
  CONTROL_V1_WIRE_EVENT_TYPES,
  V1_FIELD_PROJECTION,
  V1_WIRE_EVENT_TYPES,
)
from agent_gateway.mcp_activation import (  # noqa: E402
  MCP_SERVER_ACTIVATED_EVENT,
  McpActivationError,
  McpActivationFold,
  derive_live_surface,
  fold_mcp_activations,
  mcp_server_activated_event,
)
from agent_gateway.secret_boundary import sanitize_tool_event  # noqa: E402


CHANNEL_TIERS = {
  None: {"always": {"portfolio-reads-mcp"}, "defer": {"market-data-mcp"}},
  "web": {"always": set(), "defer": {"market-data-mcp", "portfolio-reads-mcp"}},
}

SERVER_CATALOG = {
  "portfolio-reads-mcp": {"tools": ["get_positions", "get_returns"]},
  "market-data-mcp": {"tools": ["screen_stocks", "compare_peers"]},
}


class _Profile:
  def __init__(
    self,
    *,
    core_mcp_tools=None,
    unscoped_active_mcp_servers=frozenset(),
    denied_mcp_servers=frozenset(),
  ) -> None:
    self.core_mcp_tools = core_mcp_tools or {}
    self.unscoped_active_mcp_servers = unscoped_active_mcp_servers
    self.denied_mcp_servers = denied_mcp_servers


def _surface(fold, *, profile=None, channel=None, denied=None):
  return derive_live_surface(
    profile=profile,
    channel_context=channel,
    channel_tiers=CHANNEL_TIERS,
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


def test_the_fold_replays_exactly_from_its_durable_events() -> None:
  live = McpActivationFold()
  live.record("market-data-mcp", tools=["screen_stocks"], source="load_tools")
  live.record("portfolio-reads-mcp", tools=None, source="run_agent")

  events = [
    mcp_server_activated_event(
      server_id="market-data-mcp",
      tools=["screen_stocks"],
      source="load_tools",
    ),
    mcp_server_activated_event(
      server_id="portfolio-reads-mcp",
      tools=None,
      source="run_agent",
    ),
  ]

  assert fold_mcp_activations(events) == live


def test_replay_skips_refusals_and_foreign_events() -> None:
  events = [
    {"type": "tool_call_complete", "server_id": "market-data-mcp"},
    mcp_server_activated_event(
      server_id="market-data-mcp",
      source="run_agent",
      error={"code": "mcp_server_denied", "message": "denied by profile"},
    ),
    mcp_server_activated_event(
      server_id="portfolio-reads-mcp",
      tools=["get_positions"],
      source="load_tools",
    ),
  ]

  fold = fold_mcp_activations(events)

  assert fold.activated_servers == frozenset({"portfolio-reads-mcp"})


def test_activation_is_session_log_only_with_no_wire_projection() -> None:
  # D-B7-1: no client needs the record, so it is deliberately absent from both
  # wire vocabularies. An older binary replaying a newer log ignores it.
  assert MCP_SERVER_ACTIVATED_EVENT not in V1_WIRE_EVENT_TYPES
  assert MCP_SERVER_ACTIVATED_EVENT not in CONTROL_V1_WIRE_EVENT_TYPES
  assert MCP_SERVER_ACTIVATED_EVENT not in V1_FIELD_PROJECTION
  assert MCP_SERVER_ACTIVATED_EVENT not in CONTROL_V1_FIELD_PROJECTION


def test_the_secret_boundary_sanitizes_the_activation_refusal_message() -> None:
  event = mcp_server_activated_event(
    server_id="market-data-mcp",
    source="run_agent",
    error={
      "code": "mcp_server_unavailable",
      "message": "token sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKKLLLL",
    },
  )

  sanitized = sanitize_tool_event(event, sink="session_log")

  assert "sk-ant-api03-AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIIIJJJJKKKKLLLL" not in str(
    sanitized["error"]
  )
  assert sanitized["server_id"] == "market-data-mcp"


def test_empty_fold_surface_is_the_tier_always_set_under_the_profile_ceiling() -> None:
  profile = _Profile(core_mcp_tools={"portfolio-reads-mcp": {"get_positions"}})

  surface = _surface(McpActivationFold(), profile=profile)

  assert surface.active_servers == frozenset({"portfolio-reads-mcp"})
  assert surface.allowed_mcp_tools_by_server == {
    "portfolio-reads-mcp": frozenset({"get_positions"}),
  }
  assert surface.deferred_mcp_tools == frozenset(
    {"get_returns", "screen_stocks", "compare_peers"}
  )
  assert surface.deferred_mcp_tool_ids == frozenset({
    "mcp__portfolio-reads-mcp__get_returns",
    "mcp__market-data-mcp__screen_stocks",
    "mcp__market-data-mcp__compare_peers",
  })


def test_one_activation_moves_advertised_and_allowed_together() -> None:
  # The desync class: a pack could leave the deferred set while the allowlist
  # never learned the grant. One fold makes the two the same fact.
  profile = _Profile(core_mcp_tools={"portfolio-reads-mcp": {"get_positions"}})
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
  assert surface.deferred_mcp_tools == frozenset({"screen_stocks", "compare_peers"})
