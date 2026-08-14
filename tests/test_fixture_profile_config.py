# ruff: noqa: E402

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

# entry.py validates PRODUCT_ID at import time. Keep collection hermetic in CI.
os.environ.setdefault("PRODUCT_ID", "hank-test")

from agent.profiles import load_profile
from agent.shared.tool_handlers import _build_local_tool_handlers
from agent_gateway import AgentRunner, EventLog, ToolDispatcher
from agent_gateway.fixture_gate import FIXTURE_APPROVAL_TOOL_NAME, FIXTURE_MODEL_ID
from agent_gateway.providers.fixture import FixtureProvider
from api.agent.autonomous import entry as autonomous_entry
from tests.capability_execution_test_support import (
  stub_capability_execution_resolver,
  stub_runner_capability_execution,
)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_server_for_tool(self, _name: str) -> str | None:
    return None

  async def call_tool(self, name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}


def _construct_runner_from_runtime_config(config: Any, *, session_id: str) -> AgentRunner:
  event_log = EventLog()
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log,
    session_id=session_id,
  )
  return AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id=session_id,
    capability_execution=stub_runner_capability_execution(
      provider=FixtureProvider(),
      auth_config={
        "auth_mode": "none",
        "api_key": "",
        "auth_token": "",
        "max_tokens": config.max_tokens,
      },
      model=FIXTURE_MODEL_ID,
      effort="none",
    ),
    get_tool_definitions=lambda: [],
    client_timeout=config.client_timeout,
    max_tokens_override=config.max_tokens,
    per_turn_timeout=config.per_turn_timeout,
    max_budget_usd=config.max_budget_usd,
    max_concurrent_sub_agents=config.max_concurrent_sub_agents,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    channel="cli",
  )


def _fixture_session_driver_execution():
  resolver = stub_capability_execution_resolver(
    default_provider="fixture",
    default_model=FIXTURE_MODEL_ID,
    run_mode="autonomous",
  )
  return resolver.resolve("session.driver")


def test_fixture_profile_budgets_validate_through_runner_config(monkeypatch) -> None:
  monkeypatch.setenv("APP_ENV", "test")
  monkeypatch.delenv("ENVIRONMENT", raising=False)
  monkeypatch.delenv("AGENT_GATEWAY_ENV", raising=False)
  monkeypatch.delenv("NODE_ENV", raising=False)

  profile = load_profile("_fixture")
  execution = _fixture_session_driver_execution()
  run_config = autonomous_entry._run_once_config(
    profile,
    session_driver_execution=execution,
  )
  dev_config = autonomous_entry._dev_config(
    profile,
    session_driver_execution=execution,
  )

  assert run_config.max_budget_usd == profile.max_budget_usd > 0
  assert dev_config.max_budget_usd == profile.dev_max_budget_usd > 0

  _construct_runner_from_runtime_config(run_config, session_id="fixture-profile-run")
  _construct_runner_from_runtime_config(dev_config, session_id="fixture-profile-dev")


def test_fixture_profile_carries_and_builds_fixture_gate_handler(monkeypatch) -> None:
  monkeypatch.setenv("APP_ENV", "test")
  monkeypatch.delenv("ENVIRONMENT", raising=False)
  monkeypatch.delenv("AGENT_GATEWAY_ENV", raising=False)
  monkeypatch.delenv("NODE_ENV", raising=False)

  profile = load_profile("_fixture")
  handlers = _build_local_tool_handlers(
    "cli",
    set(),
    enabled_local_tool_names=profile.local_tool_names,
  )

  assert profile.local_tool_names == {FIXTURE_APPROVAL_TOOL_NAME}
  assert set(handlers) == {FIXTURE_APPROVAL_TOOL_NAME}
  result, error = asyncio.run(handlers[FIXTURE_APPROVAL_TOOL_NAME]({"reason": "fixture regression"}))

  assert error is None
  assert result == {
    "ok": True,
    "tool": FIXTURE_APPROVAL_TOOL_NAME,
    "approved_input": {"reason": "fixture regression"},
  }


def test_fixture_profile_timeout_defaults_to_live_qa_window(monkeypatch) -> None:
  monkeypatch.setenv("APP_ENV", "test")
  monkeypatch.delenv("ENVIRONMENT", raising=False)
  monkeypatch.delenv("AGENT_GATEWAY_ENV", raising=False)
  monkeypatch.delenv("NODE_ENV", raising=False)
  monkeypatch.delenv("FIXTURE_PROFILE_TIMEOUT_SECONDS", raising=False)
  monkeypatch.delenv("AGENT_GATEWAY_FIXTURE_PROFILE_TIMEOUT_SECONDS", raising=False)

  profile = load_profile("_fixture")
  execution = _fixture_session_driver_execution()
  run_config = autonomous_entry._run_once_config(
    profile,
    session_driver_execution=execution,
  )
  dev_config = autonomous_entry._dev_config(
    profile,
    session_driver_execution=execution,
  )

  assert profile.timeout_seconds == 300
  assert profile.dev_timeout == 300
  assert run_config.timeout_seconds == 300
  assert dev_config.timeout_seconds == 300


def test_fixture_profile_timeout_can_be_overridden_for_fast_tests(monkeypatch) -> None:
  monkeypatch.setenv("APP_ENV", "test")
  monkeypatch.delenv("ENVIRONMENT", raising=False)
  monkeypatch.delenv("AGENT_GATEWAY_ENV", raising=False)
  monkeypatch.delenv("NODE_ENV", raising=False)
  monkeypatch.setenv("FIXTURE_PROFILE_TIMEOUT_SECONDS", "42")
  monkeypatch.delenv("AGENT_GATEWAY_FIXTURE_PROFILE_TIMEOUT_SECONDS", raising=False)

  profile = load_profile("_fixture")
  execution = _fixture_session_driver_execution()
  run_config = autonomous_entry._run_once_config(
    profile,
    session_driver_execution=execution,
  )
  dev_config = autonomous_entry._dev_config(
    profile,
    session_driver_execution=execution,
  )

  assert profile.timeout_seconds == 42
  assert profile.dev_timeout == 42
  assert run_config.timeout_seconds == 42
  assert dev_config.timeout_seconds == 42
