from __future__ import annotations

import pytest

from agent_gateway.active_skill_authority import resolve_active_skill_authority
from agent_gateway.capability_resolution import CapabilityResolutionInputError


def test_declared_inline_grant_is_independent_of_skill_name() -> None:
  kwargs = {
    "declared_live_tool_routes": {"code_execute": "code_execute"},
    "inline_mode_exception_routes": {"code_execute": "code_execute"},
    "mode_denied_tools": {"code_execute", "run_bash"},
  }

  original = resolve_active_skill_authority("sourced-teardown", **kwargs)
  renamed = resolve_active_skill_authority("renamed-skill", **kwargs)

  assert original == renamed
  assert original.granted == frozenset({"code_execute"})
  assert original.denied == frozenset({"run_bash"})


def test_removing_inline_exception_revokes_live_declared_tool() -> None:
  authority = resolve_active_skill_authority(
    "sourced-teardown",
    declared_live_tool_routes={"code_execute": "code_execute"},
    inline_mode_exception_routes={},
    mode_denied_tools={"code_execute"},
  )

  assert authority.granted == frozenset()
  assert authority.denied == frozenset({"code_execute"})


def test_market_scan_grant_uses_exact_claim_server_and_policy() -> None:
  authority = resolve_active_skill_authority(
    "market-scan",
    declared_live_tool_routes={
      "mcp__idea-workbench-mcp__start_investment_run": "start_investment_run",
      "mcp__idea-workbench-mcp__start_quant_research": "start_quant_research",
      "mcp__other-server__get_investment_run": "get_investment_run",
    },
    inline_mode_exception_routes={
      "mcp__idea-workbench-mcp__start_quant_research": "start_quant_research",
      "mcp__other-server__get_investment_run": "get_investment_run",
    },
  )

  assert authority.granted == frozenset({"start_investment_run"})
  assert "get_investment_run" not in authority.granted
  assert "start_quant_research" not in authority.granted


def test_market_scan_wrong_server_same_bare_route_is_not_claim_granted() -> None:
  authority = resolve_active_skill_authority(
    "market-scan",
    declared_live_tool_routes={
      "mcp__other-workbench__start_investment_run": "start_investment_run",
    },
  )

  assert authority.granted == frozenset()


def test_inline_route_must_be_an_exact_declared_live_route() -> None:
  authority = resolve_active_skill_authority(
    "renamed-skill",
    declared_live_tool_routes={
      "mcp__gsheets-mcp__gsheets_write_range": "gsheets_write_range",
    },
    inline_mode_exception_routes={
      "mcp__other-sheets__gsheets_write_range": "gsheets_write_range",
    },
  )

  assert authority.granted == frozenset()


def test_ambiguous_live_exposed_routes_refuse_loudly() -> None:
  with pytest.raises(CapabilityResolutionInputError) as error:
    resolve_active_skill_authority(
      "market-scan",
      declared_live_tool_routes={
        "mcp__idea-workbench-mcp__get_investment_run": "get_investment_run",
        "mcp__other-server__get_investment_run": "get_investment_run",
      },
    )

  assert "get_investment_run" in str(error.value)


def test_malformed_inline_route_table_refuses_loudly() -> None:
  with pytest.raises(CapabilityResolutionInputError) as error:
    resolve_active_skill_authority(
      "renamed-skill",
      declared_live_tool_routes={"code_execute": "code_execute"},
      inline_mode_exception_routes={"code_execute": ""},
    )

  assert "code_execute" in str(error.value)
