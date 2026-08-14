import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway import AgentRunner, EventLog, ToolDispatcher  # noqa: E402
from agent_gateway.runner_skill_gate import (  # noqa: E402
  default_tool_definitions,
  effective_excluded_tools,
  filter_excluded_tool_definitions,
  is_report_door_clear_event,
  normalize_skill_deny,
  normalize_skill_report_doors,
)
from tests.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _McpClientWithTools:
  def __init__(self, tools: list[dict[str, Any]]) -> None:
    self._tools = tools

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return self._tools


class _StubProvider:
  name = "stub"


def _capability_execution():
  return stub_runner_capability_execution(
    provider=_StubProvider(),
    model="stub-model",
    effort="none",
    auth_config={"api_key": "k"},
  )


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-skill-gate",
  )


def _make_runner(tool_names: list[str] | None = None) -> AgentRunner:
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-skill-gate",
    capability_execution=_capability_execution(),
    excluded_tools={"always_hidden"},
    get_tool_definitions=lambda: [{"name": name} for name in (tool_names or [])],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def _make_runner_with_mcp_tools(tools: list[dict[str, Any]]) -> AgentRunner:
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-skill-gate",
    capability_execution=_capability_execution(),
    mcp_client=_McpClientWithTools(tools),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_skill_gate_filter_helpers_normalize_inputs() -> None:
  tools = [{"name": "read"}, {"name": "write"}, {"name": "always_hidden"}]

  assert effective_excluded_tools({"always_hidden"}, set()) == {"always_hidden"}
  assert effective_excluded_tools({"always_hidden"}, {"write"}) == {"always_hidden", "write"}
  assert effective_excluded_tools({"always_hidden"}, set(), {"always_hidden"}) == set()
  assert effective_excluded_tools(
    {"always_hidden"},
    {"always_hidden"},
    {"always_hidden"},
  ) == {"always_hidden"}
  assert filter_excluded_tool_definitions(tools, {"write"}) == [{"name": "read"}, {"name": "always_hidden"}]
  assert filter_excluded_tool_definitions(tools, set()) == tools
  assert normalize_skill_report_doors({" report ": " skill ", "": "skip", "empty": " "}) == {"report": "skill"}
  assert normalize_skill_report_doors(None) is None
  assert normalize_skill_report_doors(["not", "a", "dict"]) == {}
  assert normalize_skill_deny(" write ") == {"write"}
  assert normalize_skill_deny([" read ", "", None, "write"]) == {"read", "write"}
  assert normalize_skill_deny({"write", " read "}) == {"read", "write"}
  assert normalize_skill_deny({"bad": "mapping"}) is None


def test_default_tool_definitions_prefers_callback_and_copies_result_list() -> None:
  callback_tools = [{"name": "read"}]
  mcp_tools = [{"name": "write"}]

  result = default_tool_definitions(lambda: callback_tools, _McpClientWithTools(mcp_tools))

  assert result == callback_tools
  assert result is not callback_tools
  result.append({"name": "mutated"})
  assert callback_tools == [{"name": "read"}]


def test_default_tool_definitions_falls_back_to_mcp_client_and_empty_list() -> None:
  mcp_tools = [{"name": "write"}]

  assert default_tool_definitions(None, _McpClientWithTools(mcp_tools)) is mcp_tools
  assert default_tool_definitions(None, None) == []
  assert gateway_runner._default_tool_definitions is default_tool_definitions
  assert _make_runner_with_mcp_tools(mcp_tools)._default_tool_definitions() is mcp_tools


def test_report_door_clear_event_helper_matches_success_contract() -> None:
  success_event = {
    "type": "tool_call_complete",
    "tool_name": "fms_report",
    "error": None,
    "result": {"status": "STAGED", "subcommand": "report", "mutation_mode": "preview"},
  }

  assert is_report_door_clear_event(success_event, expected_skill="skill", success_statuses={"staged"})
  assert not is_report_door_clear_event(success_event, expected_skill=None, success_statuses={"staged"})
  assert not is_report_door_clear_event(
    {**success_event, "error": {"code": "bad"}},
    expected_skill="skill",
    success_statuses={"staged"},
  )
  assert not is_report_door_clear_event(
    {**success_event, "result": {"status": "error", "subcommand": "report", "mutation_mode": "preview"}},
    expected_skill="skill",
    success_statuses={"staged"},
  )
  assert not is_report_door_clear_event(
    {**success_event, "result": {"status": "staged", "subcommand": "report", "mutation_mode": "apply"}},
    expected_skill="skill",
    success_statuses={"staged"},
  )

  model_writer_event = {
    "type": "tool_call_complete",
    "tool_name": "fms_persist_business_model",
    "error": None,
    "result": {"status": "staged", "subcommand": "persist_business_model", "mutation_mode": "model_writer"},
  }
  assert is_report_door_clear_event(model_writer_event, expected_skill="skill", success_statuses={"staged"})
  assert not is_report_door_clear_event(
    {**model_writer_event, "tool_name": "fms_persist_dcf_relative_valuation"},
    expected_skill="skill",
    success_statuses={"staged"},
  )


def test_runner_skill_gate_delegates_filter_and_activate() -> None:
  runner = _make_runner(["read", "write", "always_hidden"])
  base_kwargs: dict[str, Any] = {}

  assert runner._default_tool_definitions() == [{"name": "read"}, {"name": "write"}, {"name": "always_hidden"}]

  runner._activate_skill_report_doors({" report ": " skill "})
  assert runner._active_skill_report_doors == {"report": "skill"}
  runner._activate_skill_report_doors(None)
  assert runner._active_skill_report_doors == {"report": "skill"}
  runner._activate_skill_report_doors("reset")
  assert runner._active_skill_report_doors == {}

  assert runner._effective_excluded_tools() == {"always_hidden"}
  assert runner._filter_excluded_tool_definitions([{"name": "read"}, {"name": "always_hidden"}]) == [{"name": "read"}]

  runner._activate_skill_allow([" always_hidden ", ""], base_kwargs)
  assert runner._active_skill_allow == {"always_hidden"}
  assert runner._effective_excluded_tools() == set()
  assert base_kwargs["tools"] == [
    {"name": "read"},
    {"name": "write"},
    {"name": "always_hidden"},
  ]

  runner._activate_skill_deny([" write ", ""], base_kwargs)

  assert runner._active_skill_deny == {"write"}
  assert base_kwargs["tools"] == [{"name": "read"}, {"name": "always_hidden"}]

  runner._activate_skill_deny(["always_hidden"], base_kwargs)
  assert runner._effective_excluded_tools() == {"always_hidden"}
  assert base_kwargs["tools"] == [{"name": "read"}, {"name": "write"}]

  unchanged_tools = base_kwargs["tools"]
  runner._activate_skill_deny({"bad": "mapping"}, base_kwargs)
  assert base_kwargs["tools"] is unchanged_tools
