"""An agent tool's call into a product route presents the gateway's delegated bearer.

The dispatcher injects `session_token` into MCP meta for the servers that call back
into product routes. For a gateway-minted session that bearer is the session's
delegated tool-callback token (`purpose=agent_tool_callback`, one per turn), so the
route knows an agent called; a session without an issuer (a runtime child restored
from launch secrets) presents its own token, whose stored purpose names the runtime.
Design: docs/design/agent-tool-callback-caller-kind-2026-11-06.md (C1 0fd2b3c7).
"""
from __future__ import annotations

import asyncio
from typing import Any, Mapping

import jwt
import pytest

from agent_gateway.approval_policy import RunContext
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.registry_scope import RegistryScope
from agent_gateway.session import (
  AGENT_TOOL_CALLBACK_PURPOSE,
  AuthManager,
  JWT_ALGORITHM,
  SessionStore,
)
from agent_gateway.tool_dispatcher import ToolDispatcher
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
from gateway_test_support.host_policy import owner_session_host_policy  # noqa: F401

_JWT_SECRET = "jwt-secret-with-at-least-32-bytes"
_SERVER = "portfolio-producers-mcp"
_TOOL = "record_model_acceptance_decision"


class _CallbackMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)
    self.metas: list[dict[str, Any] | None] = []

  def is_mcp_tool(self, name: str) -> bool:
    return name == _TOOL

  def get_server_for_tool(self, name: str) -> str | None:
    return _SERVER if name == _TOOL else None

  async def call_tool(
    self,
    name: str,
    tool_input: object,
    meta: dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: Mapping[str, object] | None = None,
  ) -> tuple[object | None, dict[str, object] | None]:
    _ = tool_input, abort_event, gateway_session, allow_uncertain_replay, trusted_dispatch_scope
    self.metas.append(meta)
    return {"ok": True}, None

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


def _dispatcher(session, mcp: _CallbackMcp, *, request_id: str) -> ToolDispatcher:
  return ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=mcp,
    session_id=session.session_id,
    user_id=session.user_id,
    risk_user_id=session.risk_user_id,
    channel="excel",
    role="owner",
    mcp_meta_inject_servers=frozenset({_SERVER}),
    session=session,
    run_context=RunContext(
      user_id=session.user_id, request_id=request_id, session_id=session.session_id,
      channel="excel", registry_scope=RegistryScope("run", "test-run", "/tmp/test-store"),
    ),
  )


def _dispatch(dispatcher: ToolDispatcher, call_id: str) -> None:
  result, error = asyncio.run(
    dispatcher.dispatch(call_id, _TOOL, {}, advertised_tool_names=frozenset({_TOOL}))
  )
  assert error is None, error
  assert result == {"ok": True}


def test_meta_carries_the_delegated_bearer_once_per_turn(owner_session_host_policy) -> None:  # noqa: F811
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(api_key_hash="hash", user_id="alice", risk_user_id=101, role="owner")
  session.channel = "excel"
  session.caller_kind = "human"
  own = auth.issue_token(session)
  mcp = _CallbackMcp()

  turn_one = _dispatcher(session, mcp, request_id="turn-1")
  _dispatch(turn_one, "call-1")
  _dispatch(turn_one, "call-2")
  _dispatch(_dispatcher(session, mcp, request_id="turn-2"), "call-3")

  bearers = [meta["session_token"] for meta in mcp.metas if meta is not None]
  assert len(bearers) == 3
  assert own not in bearers
  assert bearers[0] == bearers[1] != bearers[2]
  for bearer, turn in zip(bearers, ("turn-1", "turn-1", "turn-2")):
    claims = jwt.decode(bearer, _JWT_SECRET, algorithms=[JWT_ALGORITHM])
    assert claims["purpose"] == AGENT_TOOL_CALLBACK_PURPOSE
    assert claims["turn_id"] == turn
    assert (claims["user_id"], claims["risk_user_id"], claims["role"]) == ("alice", 101, "owner")
    verified, kind = auth.verify_token_caller_kind(bearer)
    assert verified is session and kind == "agent"
  assert auth.verify_token_caller_kind(own)[1] == "human"
  assert mcp.metas[0] is not None and mcp.metas[0]["user_id"] == "101"


def test_a_session_without_an_issuer_presents_its_own_token(owner_session_host_policy) -> None:  # noqa: F811
  """A runtime child restored from launch secrets: no issuer, its own token (whose
  stored purpose already names the runtime)."""
  store = SessionStore(ttl=3600)
  session = store.create_session(api_key_hash="hash", user_id="alice", risk_user_id=101, role="owner")
  session.session_token = "launch-secret-token"
  assert session.tool_callback_token_issuer is None
  mcp = _CallbackMcp()

  _dispatch(_dispatcher(session, mcp, request_id="turn-1"), "call-1")

  assert mcp.metas[0] is not None
  assert mcp.metas[0]["session_token"] == "launch-secret-token"


@pytest.mark.parametrize("turn_id", ["", "   "])
def test_an_empty_turn_id_cannot_mint(turn_id: str) -> None:
  store = SessionStore(ttl=3600)
  auth = AuthManager(secret=_JWT_SECRET, valid_keys={"gateway-key"}, session_store=store)
  session = store.create_session(api_key_hash="hash", user_id="alice", risk_user_id=101, role="owner")
  with pytest.raises(ValueError, match="turn id"):
    auth.issue_tool_callback_token(session, turn_id=turn_id)
