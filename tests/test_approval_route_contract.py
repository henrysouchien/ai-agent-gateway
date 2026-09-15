# ruff: noqa: E402

"""The route contract itself: what can be spelled, and what cannot.

S11 a durable-local route cannot be spelled without the session its ledger row
binds to; S12 a dispatcher with a live route has exactly one session, and a
dispatcher with no route has no ledger; S13 one writer in the gateway
construction funnel.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
import time
from collections.abc import Callable
from dataclasses import fields
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Literal

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import McpClientManager, ToolDispatcher
from agent_gateway.approval_route import (
  DurableLocalApprovalRoute,
  NoApprovalRoute,
  ParentDelegatedApprovalRoute,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelChild,
)
from agent_gateway.dispatcher_factory import (
  GatewayDispatcherDeps,
  InvocationPrincipal,
  build_tool_dispatcher,
)
from agent_gateway.tool_dispatcher import LocalToolHandler
from agent_gateway.tool_dispatcher_helpers import ToolResult
from agent_gateway.session import GatewaySession
from agent_gateway.single_user_policy import SingleUserApprovalPolicy


class _NullMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  async def call_tool(
    self,
    name: str,
    tool_input: object,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    _ = (
      name,
      tool_input,
      meta,
      abort_event,
      gateway_session,
      allow_uncertain_replay,
      trusted_dispatch_scope,
    )
    raise AssertionError("no MCP call is expected")


def _session(session_id: str = "sess-route") -> GatewaySession:
  now = int(time.time())
  return GatewaySession(
    session_id=session_id,
    api_key_hash="hash",
    created_at=now,
    expires_at=now + 600,
    user_id="alice",
    role="owner",
    channel="web",
  )


def _store(tmp_path: Path) -> SQLiteApprovalStore:
  return SQLiteApprovalStore(tmp_path / "approvals.sqlite3")


def _policy() -> SingleUserApprovalPolicy:
  return SingleUserApprovalPolicy()

def _invoke_constructor(
  constructor: Callable[..., object],
  *args: object,
) -> object:
  """Call a constructor dynamically so invalid-runtime contract cases stay invalid."""

  return constructor(*args)


# --- S11 -------------------------------------------------------------------


def test_durable_route_cannot_be_spelled_without_a_session(
  tmp_path: Path,
) -> None:
  store = _store(tmp_path)
  policy = _policy()
  session = _session()

  with pytest.raises(TypeError):
    _invoke_constructor(DurableLocalApprovalRoute, store, policy)
  with pytest.raises(TypeError):
    _invoke_constructor(DurableLocalApprovalRoute, store, policy, None)
  with pytest.raises(TypeError):
    _invoke_constructor(DurableLocalApprovalRoute, store, policy, object())
  with pytest.raises(ValueError):
    _invoke_constructor(DurableLocalApprovalRoute, None, policy, session)
  with pytest.raises(ValueError):
    _invoke_constructor(DurableLocalApprovalRoute, store, None, session)

  route = DurableLocalApprovalRoute(store, policy, session)
  assert route.store is store
  assert route.policy is policy
  assert route.session is session


def test_delegated_route_requires_its_exact_channel_and_session() -> None:
  session = _session()
  channel = object.__new__(AutonomousApprovalChannelChild)

  with pytest.raises(TypeError):
    _invoke_constructor(ParentDelegatedApprovalRoute, channel, None)
  with pytest.raises(TypeError):
    _invoke_constructor(ParentDelegatedApprovalRoute, None, session)

  route = ParentDelegatedApprovalRoute(channel, session)
  assert route.channel is channel
  assert route.session is session


def test_no_approval_route_is_the_only_sessionless_variant() -> None:
  assert [f.name for f in dataclasses.fields(NoApprovalRoute)] == []
  assert {f.name for f in dataclasses.fields(DurableLocalApprovalRoute)} == {
    "store",
    "policy",
    "session",
  }
  assert {f.name for f in dataclasses.fields(ParentDelegatedApprovalRoute)} == {
    "channel",
    "session",
  }


# --- S12 -------------------------------------------------------------------


def _planned_handler(events: list[str]) -> LocalToolHandler:
  """A real PLANNING_IDENTITY handler over a real ARTIFACT_ONLY ChangeSet."""

  from api.fms.core.change_set import (
    ArtifactOnlyPlan,
    ArtifactPayload,
    BaseRevision,
    CanonicalPayload,
    ChangeSet,
    CommitStrategy,
    DomainResultRef,
    EffectCriticality,
    EffectKind,
    EffectSpec,
    InlinePayload,
    IntentRef,
    ProducerRef,
    ReviewKind,
    ReviewRequirement,
    TargetRef,
    TargetScope,
  )

  def canonical(value: object) -> CanonicalPayload:
    return CanonicalPayload.from_value(value)

  def inline(value: object) -> InlinePayload:
    return InlinePayload("v1", "application/json", canonical(value).content)

  artifact_path = "artifacts/TEST/planned.json"
  change_set = ChangeSet(
    "v1",
    "",
    "",
    ProducerRef("test", "research_producer", "alice", "run-1"),
    TargetRef("TEST", 7, "workspace/TEST", TargetScope.WORKSPACE),
    (BaseRevision("workbook", "models/TEST.xlsx", "a" * 64),),
    IntentRef("planned_write", canonical({"x": 1})),
    DomainResultRef("test", "v1", inline({"ok": True})),
    (
      EffectSpec(
        "artifact",
        EffectKind.ARTIFACT_REFUSAL_ONLY,
        EffectCriticality.REQUIRED,
        (),
        ArtifactPayload(
          artifact_path,
          canonical({"status": "planned"}),
          inline({"status": "planned"}),
        ),
      ),
    ),
    (),
    ReviewRequirement(ReviewKind.NONE, None),
    CommitStrategy.ARTIFACT_ONLY,
    ArtifactOnlyPlan(artifact_path),
  )
  prepared = SimpleNamespace(change_set=change_set)

  class _PlannedHandler:
    PLANNING_IDENTITY = "change_set"

    async def __call__(
      self,
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> Any:
      raise AssertionError("planned tools must not execute through the legacy handler")

    async def plan_change(
      self,
      _tool_input: dict[str, Any],
      *,
      call_index: int,
      tool_ctx: Any,
    ) -> tuple[Any, Any]:
      _ = call_index, tool_ctx
      events.append("plan")
      return change_set, prepared

    async def execute_prepared_change(
      self,
      *_args: Any,
      **_kwargs: Any,
    ) -> Any:
      events.append("execute")
      return {"ok": True}, None

  return _PlannedHandler()


def test_live_route_is_the_only_session_carrier(tmp_path: Path) -> None:
  store = _store(tmp_path)
  policy = _policy()
  session_a = _session("sess-a")
  session_b = _session("sess-b")
  route = DurableLocalApprovalRoute(store, policy, session_a)

  with pytest.raises(ValueError):
    ToolDispatcher(
      role="owner",
      mcp_client=_NullMcp(),
      local_tool_handlers={},
      needs_approval=lambda *_a: True,
      approval_route=route,
      session=session_b,
    )
  with pytest.raises(ValueError):
    ToolDispatcher(
      role="owner",
      mcp_client=_NullMcp(),
      local_tool_handlers={},
      needs_approval=lambda *_a: True,
      approval_route=route,
      session=session_a,
    )

  dispatcher = ToolDispatcher(
    role="owner",
    mcp_client=_NullMcp(),
    local_tool_handlers={},
    needs_approval=lambda *_a: True,
    approval_route=route,
  )
  assert dispatcher._session is session_a
  assert dispatcher._approval_store is store
  assert dispatcher._approval_policy is policy


def test_dispatcher_without_a_route_has_no_ledger_and_refuses_at_the_door(
  tmp_path: Path,
) -> None:
  events: list[str] = []
  _store(tmp_path)
  dispatcher = ToolDispatcher(
    role="owner",
    mcp_client=_NullMcp(),
    local_tool_handlers={"planned": _planned_handler(events)},
    needs_approval=lambda *_a: True,
    request_approval=None,
  )

  assert isinstance(dispatcher._approval_route, NoApprovalRoute)
  assert dispatcher._session is None
  assert dispatcher._approval_store is None
  assert dispatcher._approval_policy is None

  result, error = asyncio.run(
    dispatcher.dispatch("call-1", "planned", {"x": 1}, call_index=3)
  )

  assert result is None
  assert error is not None
  assert error["code"] == "approval_route_absent"
  assert events == ["plan"]
  # No row exists anywhere: the door refused before any ledger write.
  import sqlite3

  with sqlite3.connect(tmp_path / "approvals.sqlite3") as conn:
    assert conn.execute("SELECT COUNT(*) FROM approval_requests").fetchone()[0] == 0


# --- S13 -------------------------------------------------------------------


class _CapturingDispatcher:
  captured: dict[str, object] = {}

  def __init__(self, **kwargs: object) -> None:
    type(self).captured = dict(kwargs)


def test_gateway_construction_funnel_has_one_route_writer(
  tmp_path: Path,
) -> None:
  # Process resources, never a run's authority: deps carry handles, not a route.
  assert "approval_route" not in {
    dep_field.name for dep_field in fields(GatewayDispatcherDeps)
  }

  store = _store(tmp_path)
  policy = _policy()
  # Deliberately real, so a NoApprovalRoute for chat_embedded cannot be an
  # accident of this process happening to hold nothing.
  deps = GatewayDispatcherDeps(
    mcp_client=_NullMcp(),
    approval_store=store,
    approval_policy=policy,
    mcp_meta_inject_servers=frozenset(),
  )

  def _build(
    profile: Literal["chat_embedded", "interactive"],
  ) -> tuple[dict[str, object], GatewaySession]:
    session = _session(f"sess-{profile}")
    principal = InvocationPrincipal.from_session(session)
    build_tool_dispatcher(
      deps,
      principal=principal,
      profile=profile,
      event_log=None,
      session_id=session.session_id,
      request_approval=None,
      needs_approval=lambda *_a: False,
      approved_tool_types=set(),
      local_tool_handlers={},
      _tool_dispatcher_cls=_CapturingDispatcher,
      _excel_tool_dispatcher_cls=lambda **kwargs: kwargs["base"],
    )
    return dict(_CapturingDispatcher.captured), principal.session

  chat_kwargs, _chat_session = _build("chat_embedded")
  assert isinstance(chat_kwargs["approval_route"], NoApprovalRoute)
  assert not {"store", "policy", "session"} & chat_kwargs.keys()

  interactive_kwargs, interactive_session = _build("interactive")
  interactive_route = interactive_kwargs["approval_route"]
  assert isinstance(interactive_route, DurableLocalApprovalRoute)
  assert interactive_route == DurableLocalApprovalRoute(
    store,
    policy,
    interactive_session,
  )
  assert interactive_route.session is interactive_session
  assert "store" not in interactive_kwargs
  assert "policy" not in interactive_kwargs
