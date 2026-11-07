from __future__ import annotations
from agent_gateway.registry_scope import NoRegistryScope, RegistryScope
from gateway_test_support.run_context_test_support import run_context_for_test
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
from gateway_test_support.dispatcher_factory_test_support import (
  _allow_interceptor,
  _deps,
  _local_handler,
  _never_needs_approval,
  _request_approval,
  _session,
)

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from agent_gateway.approval_policy import RunContext
from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway.event_log import EventLog
from agent_gateway.dispatcher_factory import (
  DispatcherConstructionError,
  GatewayDispatcherDeps,
  InvocationPrincipal,
  build_tool_dispatcher,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.tool_dispatcher_helpers import LocalToolHandler

ROOT = Path(__file__).resolve().parents[3]













def _legacy_base_snapshot(dispatcher: ToolDispatcher) -> dict[str, Any]:
  snapshot = dict(dispatcher.__dict__)
  for field_name in ("_secret_boundary", "_boundary_event_log"):
    owned = snapshot.pop(field_name)
    snapshot[f"{field_name}_type"] = type(owned)
  return snapshot




def test_principal_derives_all_identity_from_session_without_loose_inputs() -> None:
  session = _session(kind="chat", channel="excel")

  def qualifier(_name: str, _args: dict[str, Any]) -> str:
    return "scope"

  principal = InvocationPrincipal.from_session(
    session,
    approval_key_qualifier=qualifier,
  )

  assert principal.session is session
  assert principal.session_kind == "chat"
  assert principal.user_id == "42"
  assert principal.risk_user_id == 42
  assert principal.role == "owner"
  assert principal.capabilities == frozenset()
  assert principal.channel == "excel"
  assert principal.approval_key_qualifier is qualifier
  with pytest.raises(FrozenInstanceError):
    principal.user_id = "attacker"  # type: ignore[misc]


def test_integrity_guard_uses_a_real_raise_after_session_kind_changes() -> None:
  session = _session()
  principal = InvocationPrincipal.from_session(session)
  session.kind = "control"

  with pytest.raises(DispatcherConstructionError, match="does not match"):
    build_tool_dispatcher(
      _deps(),
      capability_bind=TEST_CAPABILITY_BIND,
      principal=principal,
      profile="chat_embedded",
      run_context=run_context_for_test(registry_scope=NoRegistryScope()),
      event_log=None,
      session_id=session.session_id,
      request_approval=None,
      needs_approval=None,
      local_tool_handlers={},
    )


def test_integrity_guard_survives_optimized_python_mode() -> None:
  package_dir = Path(__file__).resolve().parents[1]
  script = """
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
from agent_gateway.registry_scope import NoRegistryScope
from gateway_test_support.run_context_test_support import run_context_for_test
from types import SimpleNamespace
from agent_gateway.dispatcher_factory import (
  DispatcherConstructionError,
  GatewayDispatcherDeps,
  InvocationPrincipal,
  build_tool_dispatcher,
)

session = SimpleNamespace(
  kind="chat",
  user_id="42",
  risk_user_id=42,
  role="owner",
  channel="web",
)
principal = InvocationPrincipal.from_session(session)
session.kind = "control"
try:
  build_tool_dispatcher(
    GatewayDispatcherDeps(None, None, None, frozenset()),
    capability_bind=TEST_CAPABILITY_BIND,
    principal=principal,
    profile="chat_embedded",
    run_context=run_context_for_test(registry_scope=NoRegistryScope()),
    event_log=None,
    session_id="session",
    request_approval=None,
    needs_approval=None,
    local_tool_handlers={},
  )
except DispatcherConstructionError:
  raise SystemExit(0)
raise SystemExit(1)
"""
  environment = dict(os.environ)
  environment["PYTHONPATH"] = str(package_dir)

  completed = subprocess.run(
    [sys.executable, "-O", "-c", script],
    check=False,
    env=environment,
  )

  assert completed.returncode == 0








def test_chat_embedded_golden_attribute_snapshot_matches_easy_inline() -> None:
  session = _session()
  mcp_client = McpClientManager(config_path=None)
  local_handlers: dict[str, LocalToolHandler] = {
    "local": _local_handler,
  }
  needs_approval = _never_needs_approval
  request_approval = _request_approval
  event_log = EventLog()

  def qualifier(_name: str, _args: dict[str, Any]) -> str:
    return "qualifier"

  def get_tool_definitions() -> list[dict[str, Any]]:
    return [{"name": "local"}]

  mcp_session_servers = {"session-aware-mcp"}
  commercial_work_start = object()

  def commercial_recheck(_context: Any) -> None:
    return None

  commercial_servers = frozenset({"portfolio-trades-mcp"})

  expected = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=mcp_client,
    local_tool_handlers=local_handlers,
    needs_approval=needs_approval,
    request_approval=request_approval,
    event_log=event_log,
    session_id=session.session_id,
    mcp_session_inject_servers=mcp_session_servers,
    approval_key_qualifier=qualifier,
    get_tool_definitions=get_tool_definitions,
    commercial_work_start=commercial_work_start,
    commercial_irreversible_recheck=commercial_recheck,
    commercial_mcp_servers=commercial_servers,
    run_context=run_context_for_test(registry_scope=RegistryScope('run', 'test-run', '/tmp/test-store')),
  )
  actual = build_tool_dispatcher(
    _deps(
      mcp_client=mcp_client,
      mcp_meta_inject_servers=frozenset(),
    ),
    capability_bind=TEST_CAPABILITY_BIND,
    principal=InvocationPrincipal.from_session(
      session,
      approval_key_qualifier=qualifier,
    ),
    profile="chat_embedded",
    run_context=expected.run_context,
    local_tool_handlers=local_handlers,
    needs_approval=needs_approval,
    request_approval=request_approval,
    event_log=event_log,
    session_id=session.session_id,
    mcp_session_inject_servers=mcp_session_servers,
    get_tool_definitions=get_tool_definitions,
    commercial_work_start=commercial_work_start,
    commercial_irreversible_recheck=commercial_recheck,
    commercial_mcp_servers=commercial_servers,
  )

  assert isinstance(actual, ToolDispatcher)
  assert _legacy_base_snapshot(actual) == _legacy_base_snapshot(expected)
  assert actual._session is None
  assert actual._user_id is None


