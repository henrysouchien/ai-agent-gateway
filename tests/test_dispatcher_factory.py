from __future__ import annotations

import inspect
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_gateway.dispatcher_factory import (
  DispatcherConstructionError,
  GatewayDispatcherDeps,
  InvocationPrincipal,
  build_tool_dispatcher,
)
from agent_gateway.session import GatewaySession
from agent_gateway.tool_dispatcher import ToolDispatcher

ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))

from agent.interactive.tool_dispatcher import ExcelToolDispatcher  # noqa: E402
from excel_mcp.relay import ChannelType  # noqa: E402


class _McpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_server_for_tool(self, _name: str) -> str | None:
    return None

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": name}


class _ChannelRegistry:
  def __init__(self, *, active_channels: list[Any] | None = None) -> None:
    self._active_channels = list(active_channels or [])

  def get_active_channels(self) -> list[Any]:
    return list(self._active_channels)

  def get_channel_for_tool(self, _tool_name: str) -> None:
    return None


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
  mcp_client: Any | None = None,
  approval_store: Any = None,
  approval_policy: Any = None,
  mcp_meta_inject_servers: frozenset[str] = frozenset({
    "portfolio-reads-mcp"
  }),
) -> GatewayDispatcherDeps:
  return GatewayDispatcherDeps(
    mcp_client=mcp_client or _McpClient(),
    approval_store=approval_store,
    approval_policy=approval_policy,
    mcp_meta_inject_servers=mcp_meta_inject_servers,
  )


def _build(
  *,
  profile: str,
  session: GatewaySession,
  deps: GatewayDispatcherDeps,
  **overrides: Any,
) -> ToolDispatcher | ExcelToolDispatcher:
  values: dict[str, Any] = {
    "event_log": None,
    "session_id": session.session_id,
    "request_approval": None,
    "needs_approval": lambda *_args, **_kwargs: False,
    "approved_tool_types": session.approved_tool_types,
    "local_tool_handlers": {},
    "channel_registry": _ChannelRegistry(),
    "execute_addin": lambda *_args, **_kwargs: None,
    "channel_context": session.channel,
    "tool_packs": {},
  }
  if profile == "interactive":
    values.update({
      "excel_wrap": True,
      "_excel_tool_dispatcher_cls": ExcelToolDispatcher,
    })
  values.update(overrides)
  return build_tool_dispatcher(
    deps,
    principal=InvocationPrincipal.from_session(session),
    profile=profile,  # type: ignore[arg-type]
    **values,
  )


def _base(
  dispatcher: ToolDispatcher | ExcelToolDispatcher,
) -> ToolDispatcher:
  if isinstance(dispatcher, ExcelToolDispatcher):
    return dispatcher._base
  return dispatcher


def _legacy_base_snapshot(dispatcher: ToolDispatcher) -> dict[str, Any]:
  snapshot = dict(dispatcher.__dict__)
  for field_name in ("_secret_boundary", "_boundary_event_log"):
    owned = snapshot.pop(field_name)
    snapshot[f"{field_name}_type"] = type(owned)
  return snapshot


def _wrapper_snapshot(dispatcher: ExcelToolDispatcher) -> dict[str, Any]:
  return {
    key: value
    for key, value in dispatcher.__dict__.items()
    if key not in {"_base", "_addin_lock"}
  }


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
  package_dir = ROOT / "packages" / "agent-gateway"
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


@pytest.mark.parametrize(
  "profile,kind,channel",
  [
    ("chat_embedded", "chat", "web"),
    ("interactive", "chat", "excel"),
  ],
)
def test_server_owned_meta_injection_and_prompt_policy_reach_all_profiles(
  profile: str,
  kind: str,
  channel: str,
) -> None:
  meta_servers = frozenset({"portfolio-reads-mcp", "research-corpus-mcp"})
  dispatcher = _build(
    profile=profile,
    session=_session(kind=kind, channel=channel),
    deps=_deps(
      mcp_meta_inject_servers=meta_servers,
    ),
  )
  base = _base(dispatcher)

  assert base._mcp_meta_inject_servers == meta_servers
  assert base._should_avoid_permission_prompts is False


def test_store_and_policy_mapping_is_profile_conditional() -> None:
  store = object()
  policy = object()
  deps = _deps(
    approval_store=store,
    approval_policy=policy,
  )

  chat = _base(_build(
    profile="chat_embedded",
    session=_session(),
    deps=deps,
  ))
  interactive = _base(_build(
    profile="interactive",
    session=_session(channel="excel"),
    deps=deps,
  ))
  assert chat._session is None
  assert chat._approval_store is None
  assert chat._approval_policy is None
  assert interactive._approval_store is store
  assert interactive._approval_policy is policy


def test_dependency_partition_and_per_runtime_wiring() -> None:
  dependency_fields = {field.name for field in fields(GatewayDispatcherDeps)}
  assert dependency_fields == {
    "mcp_client",
    "approval_store",
    "approval_policy",
    "mcp_meta_inject_servers",
  }
  assert dependency_fields.isdisjoint({
    "interceptors",
    "get_tool_definitions",
    "commercial_mcp_servers",
    "mcp_session_inject_servers",
  })

  interceptor = object()

  def get_tool_definitions() -> list[dict[str, Any]]:
    return []

  commercial_servers = frozenset({"portfolio-trades-mcp"})
  session_inject_servers = {"legacy-session-mcp"}
  dispatcher = _base(_build(
    profile="chat_embedded",
    session=_session(),
    deps=_deps(),
    interceptors=[interceptor],
    get_tool_definitions=get_tool_definitions,
    commercial_mcp_servers=commercial_servers,
    mcp_session_inject_servers=session_inject_servers,
  ))

  assert dispatcher._interceptors == [interceptor]
  assert dispatcher._get_tool_definitions is get_tool_definitions
  assert dispatcher._commercial_mcp_servers == commercial_servers
  assert dispatcher._mcp_session_inject_servers is session_inject_servers


def test_chat_embedded_golden_attribute_snapshot_matches_easy_inline() -> None:
  session = _session()
  mcp_client = _McpClient()
  local_handlers = {"local": lambda _args: ({"ok": True}, None)}

  def needs_approval(
    _name: str,
    _tool_input: dict[str, Any],
    _qualifier: str,
  ) -> bool:
    return False

  def request_approval(_request: Any) -> None:
    return None

  event_log = object()

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


@pytest.mark.parametrize("channel", ["excel", "web"])
@pytest.mark.asyncio
async def test_interactive_golden_attribute_snapshot_and_wrapper_parity(
  channel: str,
) -> None:
  session = _session(channel=channel)
  mcp_client = _McpClient()
  local_handlers = {"local": lambda _args: ({"ok": True}, None)}

  def needs_approval(
    _name: str,
    _tool_input: dict[str, Any],
    _qualifier: str,
  ) -> bool:
    return False

  def request_approval(_request: Any) -> None:
    return None

  event_log = object()

  def qualifier(_name: str, _args: dict[str, Any]) -> str:
    return "qualifier"

  interceptors = [object()]
  meta_servers = frozenset({"portfolio-reads-mcp"})
  identity_overrides = {"research-corpus-mcp": 7}
  cache_denied = frozenset({"never-cache"})
  store = object()
  policy = object()
  run_context = object()

  def get_tool_definitions() -> list[dict[str, Any]]:
    return [{"name": "local"}]

  allowed_tools = {"research-corpus-mcp": {"corpus_search"}}

  def describe_scope(_server: str | None, _tool: str) -> str:
    return "blocked"

  commercial_work_start = object()

  def commercial_recheck(_context: Any) -> None:
    return None

  commercial_servers = frozenset({"portfolio-trades-mcp"})
  channel_registry = _ChannelRegistry(
    active_channels=[
      SimpleNamespace(
        channel_type=ChannelType.EXCEL,
        tool_names={"excel_only_tool"},
      )
    ]
  )
  def execute_addin(_request: Any) -> None:
    return None

  tool_packs = {"market-data": {"tools": {"get_quote"}}}

  expected_base = ToolDispatcher(
    mcp_client=mcp_client,
    local_tool_handlers=local_handlers,
    needs_approval=needs_approval,
    request_approval=request_approval,
    approved_tool_types=session.approved_tool_types,
    event_log=event_log,
    approval_key_qualifier=qualifier,
    interceptors=interceptors,
    session_id=session.session_id,
    user_id=session.user_id,
    risk_user_id=session.risk_user_id,
    channel=channel or session.channel,
    role=session.role,
    mcp_meta_inject_servers=meta_servers,
    mcp_identity_overrides=identity_overrides,
    credentials_resolver_active=True,
    session_cache_denied_tools=cache_denied,
    session=session,
    store=store,
    policy=policy,
    run_context=run_context,
    get_tool_definitions=get_tool_definitions,
    allowed_mcp_tools_by_server=allowed_tools,
    mcp_scope_context="profile",
    describe_mcp_scope_block=describe_scope,
    commercial_work_start=commercial_work_start,
    commercial_irreversible_recheck=commercial_recheck,
    commercial_mcp_servers=commercial_servers,
  )
  expected = ExcelToolDispatcher(
    base=expected_base,
    channel_registry=channel_registry,
    execute_addin=execute_addin,
    channel_context=channel,
    tool_packs=tool_packs,
  )
  actual = build_tool_dispatcher(
    _deps(
      mcp_client=mcp_client,
      approval_store=store,
      approval_policy=policy,
      mcp_meta_inject_servers=meta_servers,
    ),
    principal=InvocationPrincipal.from_session(
      session,
      approval_key_qualifier=qualifier,
    ),
    profile="interactive",
    local_tool_handlers=local_handlers,
    needs_approval=needs_approval,
    request_approval=request_approval,
    approved_tool_types=session.approved_tool_types,
    event_log=event_log,
    interceptors=interceptors,
    session_id=session.session_id,
    session_cache_denied_tools=cache_denied,
    session=session,
    store=store,
    policy=policy,
    run_context=run_context,
    get_tool_definitions=get_tool_definitions,
    mcp_identity_overrides=identity_overrides,
    credentials_resolver_active=True,
    allowed_mcp_tools_by_server=allowed_tools,
    mcp_scope_context="profile",
    describe_mcp_scope_block=describe_scope,
    commercial_work_start=commercial_work_start,
    commercial_irreversible_recheck=commercial_recheck,
    commercial_mcp_servers=commercial_servers,
    excel_wrap=True,
    channel_registry=channel_registry,
    execute_addin=execute_addin,
    channel_context=channel,
    tool_packs=tool_packs,
    _excel_tool_dispatcher_cls=ExcelToolDispatcher,
  )

  assert isinstance(actual, ExcelToolDispatcher)
  assert _legacy_base_snapshot(actual._base) == _legacy_base_snapshot(
    expected._base
  )
  assert _wrapper_snapshot(actual) == _wrapper_snapshot(expected)
  assert actual._channel_context == channel

  if channel == "web":
    result, error = await actual.dispatch(
      "call-1",
      "excel_only_tool",
      {},
    )
    assert result is None
    assert error == {
      "code": "tool_unavailable",
      "message": "Tool 'excel_only_tool' is not available from this channel",
    }
