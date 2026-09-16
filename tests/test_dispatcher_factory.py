from __future__ import annotations

import inspect
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
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
from agent_gateway.session import GatewaySession
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.tool_dispatcher_helpers import (
  ApprovalDecision,
  ApprovalRequest,
  InterceptContext,
  InterceptDecision,
  LocalToolHandler,
  ToolResult,
)

ROOT = Path(__file__).resolve().parents[3]







def _never_needs_approval(
  _name: str,
  _tool_input: object,
  _qualifier: str,
) -> bool:
  return False


async def _local_handler(_tool_input: object) -> ToolResult:
  return {"ok": True}, None


async def _request_approval(
  _request: ApprovalRequest,
) -> ApprovalDecision | None:
  return None


async def _allow_interceptor(
  _context: InterceptContext,
) -> InterceptDecision:
  return InterceptDecision(action="allow")




def _session(
  *,
  kind: str = "chat",
  channel: str = "web",
) -> GatewaySession:
  return GatewaySession(
    session_id=f"session-{kind}-{channel}",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="42",
    risk_user_id=42,
    role="owner",
    kind=kind,  # type: ignore[arg-type]
    channel=channel,
    auth_config={"api_key": "secret"},
    approved_tool_types={"read"},
  )


def _deps(
  *,
  mcp_client: McpClientManager | None = None,
  approval_store: Any = None,
  approval_policy: Any = None,
  mcp_meta_inject_servers: frozenset[str] = frozenset({
    "portfolio-reads-mcp"
  }),
) -> GatewayDispatcherDeps:
  return GatewayDispatcherDeps(
    mcp_client=mcp_client or McpClientManager(config_path=None),
    approval_store=approval_store,
    approval_policy=approval_policy,
    mcp_meta_inject_servers=mcp_meta_inject_servers,
  )






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
  assert {
    name
    for name, parameter in inspect.signature(InvocationPrincipal).parameters.items()
    if parameter.default is inspect.Parameter.empty
  } == {"session"}
  assert all(
    not field.init
    for field in fields(InvocationPrincipal)
    if field.name in {"session_kind", "user_id", "risk_user_id", "role", "capabilities", "channel"}
  )
  with pytest.raises(FrozenInstanceError):
    principal.user_id = "attacker"  # type: ignore[misc]


def test_integrity_guard_uses_a_real_raise_after_session_kind_changes() -> None:
  session = _session()
  principal = InvocationPrincipal.from_session(session)
  session.kind = "control"

  with pytest.raises(DispatcherConstructionError, match="does not match"):
    build_tool_dispatcher(
      _deps(),
      principal=principal,
      profile="chat_embedded",
      event_log=None,
      session_id=session.session_id,
      request_approval=None,
      needs_approval=None,
      approved_tool_types=set(),
      local_tool_handlers={},
    )


def test_integrity_guard_survives_optimized_python_mode() -> None:
  package_dir = Path(__file__).resolve().parents[1]
  script = """
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
    principal=principal,
    profile="chat_embedded",
    event_log=None,
    session_id="session",
    request_approval=None,
    needs_approval=None,
    approved_tool_types=set(),
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
  cache_denied = frozenset({"never-cache"})
  commercial_work_start = object()

  def commercial_recheck(_context: Any) -> None:
    return None

  commercial_servers = frozenset({"portfolio-trades-mcp"})

  expected = ToolDispatcher(
    mcp_client=mcp_client,
    local_tool_handlers=local_handlers,
    needs_approval=needs_approval,
    request_approval=request_approval,
    approved_tool_types=session.approved_tool_types,
    event_log=event_log,
    session_id=session.session_id,
    mcp_session_inject_servers=mcp_session_servers,
    approval_key_qualifier=qualifier,
    session_cache_denied_tools=cache_denied,
    get_tool_definitions=get_tool_definitions,
    commercial_work_start=commercial_work_start,
    commercial_irreversible_recheck=commercial_recheck,
    commercial_mcp_servers=commercial_servers,
  )
  actual = build_tool_dispatcher(
    _deps(
      mcp_client=mcp_client,
      mcp_meta_inject_servers=frozenset(),
    ),
    principal=InvocationPrincipal.from_session(
      session,
      approval_key_qualifier=qualifier,
    ),
    profile="chat_embedded",
    local_tool_handlers=local_handlers,
    needs_approval=needs_approval,
    request_approval=request_approval,
    approved_tool_types=session.approved_tool_types,
    event_log=event_log,
    session_id=session.session_id,
    mcp_session_inject_servers=mcp_session_servers,
    session_cache_denied_tools=cache_denied,
    get_tool_definitions=get_tool_definitions,
    commercial_work_start=commercial_work_start,
    commercial_irreversible_recheck=commercial_recheck,
    commercial_mcp_servers=commercial_servers,
  )

  assert isinstance(actual, ToolDispatcher)
  assert _legacy_base_snapshot(actual) == _legacy_base_snapshot(expected)
  assert actual._session is None
  assert actual._user_id is None


