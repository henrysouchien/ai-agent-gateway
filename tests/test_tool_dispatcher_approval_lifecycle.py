# ruff: noqa: E402

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import EventLog, ToolDispatcher
from agent_gateway import policy_imports
from agent_gateway import tool_dispatcher as dispatcher_module
from agent_gateway import tool_dispatcher_approval_lifecycle as lifecycle_helpers


class _NullMcp:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_server_for_tool(self, _name: str) -> str | None:
    return None

  async def call_tool(self, _name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    return {"ok": True}, None


class _PrefixedMcp(_NullMcp):
  def is_mcp_tool(self, name: str) -> bool:
    return name == "trades_execute_trade"

  def get_server_for_tool(self, name: str) -> str | None:
    return "portfolio-trades-mcp" if name == "trades_execute_trade" else None

  def get_original_tool_name(self, name: str) -> str:
    return "execute_trade" if name == "trades_execute_trade" else name


class _NoMcpLookup:
  async def call_tool(self, _name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    return {"ok": True}, None


class _SessionLog:
  def __init__(self) -> None:
    self.events: list[dict[str, Any]] = []

  async def append(self, event: dict[str, Any]) -> None:
    self.events.append(dict(event))


def _request() -> SimpleNamespace:
  return SimpleNamespace(
    approval_id="approval-1",
    tool_call_id="call-1",
    tool_name="place_order",
    tool_args_redacted={"ticker": "MSFT"},
  )


def _decision() -> SimpleNamespace:
  return SimpleNamespace(
    reason="needs approval",
    allow_persistent_grant=True,
  )


def test_tool_dispatcher_resolve_tool_class_uses_policy_owner(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"execute_trade"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-trades-mcp"
    if tool_name == "execute_trade"
    else None,
    get_tool_class=lambda server_name, tool_name: "irreversible"
    if (server_name, tool_name) == ("portfolio-trades-mcp", "execute_trade")
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )

  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("execute_trade") == "irreversible"


def test_tool_dispatcher_resolve_tool_class_preserves_parent_policy_resolver_seam(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls: list[dict[str, Any]] = []

  def fake_resolve_server_policy_tool_class(tool_name: str, **kwargs: Any) -> str:
    calls.append({"tool_name": tool_name, **kwargs})
    return "artifact_write"

  monkeypatch.setattr(
    dispatcher_module,
    "resolve_server_policy_tool_class",
    fake_resolve_server_policy_tool_class,
  )
  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("emit_artifact") == "artifact_write"
  assert calls == [
    {
      "tool_name": "emit_artifact",
      "policy_tool_name": "emit_artifact",
      "runtime_server": None,
      "default": "",
    }
  ]


def test_tool_dispatcher_resolve_tool_class_does_not_require_mcp_lookup_methods(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"record_workflow_action"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-reads-mcp"
    if tool_name == "record_workflow_action"
    else None,
    get_tool_class=lambda server_name, tool_name: "state_write"
    if (server_name, tool_name) == ("portfolio-reads-mcp", "record_workflow_action")
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )

  dispatcher = ToolDispatcher(mcp_client=_NoMcpLookup(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("record_workflow_action") == "state_write"


def test_tool_dispatcher_resolve_tool_class_uses_original_name_for_prefixed_mcp_tool(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"execute_trade"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-trades-mcp"
    if tool_name == "execute_trade"
    else None,
    get_tool_class=lambda server_name, tool_name: "irreversible"
    if (server_name, tool_name) == ("portfolio-trades-mcp", "execute_trade")
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )

  dispatcher = ToolDispatcher(mcp_client=_PrefixedMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("trades_execute_trade") == "irreversible"


def test_tool_dispatcher_resolve_tool_class_uses_policy_owner_after_split_runtime_server(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"execute_trade"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-trades-mcp"
    if tool_name == "execute_trade"
    else None,
    get_tool_class=lambda server_name, tool_name: "irreversible"
    if (server_name, tool_name) == ("portfolio-trades-mcp", "execute_trade")
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )

  dispatcher = ToolDispatcher(mcp_client=_PrefixedMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("trades_execute_trade") == "irreversible"


def test_tool_dispatcher_resolve_tool_class_falls_back_when_policy_modules_absent(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_import_module(name: str) -> Any:
    if name == "agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'agent'", name="agent")
    if name == "api.agent.shared.server_policies":
      raise ModuleNotFoundError("No module named 'api'", name="api")
    raise AssertionError(f"unexpected import: {name}")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  assert dispatcher._resolve_tool_class("unmapped_tool") == "state_write"


def test_tool_dispatcher_resolve_tool_class_raises_when_policy_import_dependency_breaks(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fake_import_module(_name: str) -> Any:
    raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    dispatcher._resolve_tool_class("execute_trade")


def test_tool_dispatcher_resolve_tool_class_raises_when_policy_module_lacks_class_helper(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  policy_module = SimpleNamespace(
    get_forbidden_tools_for_session=lambda _session: frozenset({"execute_trade"}),
    get_server_for_policy_tool=lambda tool_name: "portfolio-reads-mcp"
    if tool_name == "execute_trade"
    else None,
  )

  monkeypatch.setattr(
    policy_imports.importlib,
    "import_module",
    lambda name: policy_module if name == "agent.shared.server_policies" else None,
  )
  dispatcher = ToolDispatcher(mcp_client=_NullMcp(), local_tool_handlers={}, event_log=EventLog())

  with pytest.raises(AttributeError, match="get_tool_class"):
    dispatcher._resolve_tool_class("execute_trade")


async def _wait_for_queue(session: SimpleNamespace, tool_call_id: str) -> asyncio.Queue:
  for _ in range(100):
    queue = session.approval_queues.get(tool_call_id)
    if queue is not None:
      return queue
    await asyncio.sleep(0)
  raise AssertionError("approval queue was not registered")


def test_pending_tool_helper_registers_event_waits_and_cleans_up() -> None:
  async def scenario() -> None:
    session_log = _SessionLog()
    session = SimpleNamespace(pending_tools={}, approval_queues={}, agent_session_log=session_log)
    event_log = EventLog()
    request = _request()
    task = asyncio.create_task(
      lifecycle_helpers.await_user_approval_via_pending_tools(
        session=session,
        approval_store=None,
        event_log=event_log,
        request=request,
        decision=_decision(),
        nonce="nonce-1",
        resolved_qualifier="qual-1",
        allow_persistent=True,
        timeout_seconds=5,
        log=logging.getLogger("test"),
      )
    )

    queue = await _wait_for_queue(session, request.tool_call_id)
    assert session.pending_tools[request.tool_call_id] == {
      "approval_id": "approval-1",
      "nonce": "nonce-1",
      "requested_at": session.pending_tools[request.tool_call_id]["requested_at"],
      "status": "approval_pending",
      "tool_name": "place_order",
      "resolved_qualifier": "qual-1",
    }
    assert event_log.entries[0].event["type"] == "tool_approval_request"
    assert session_log.events[0]["allow_persistent_approval"] is True

    await queue.put({"approved": True, "allow_tool_type": False})

    assert await task == {"approved": True, "allow_tool_type": False}
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_pending_tool_helper_expires_store_request_on_timeout() -> None:
  class Store:
    def __init__(self) -> None:
      self.transitions: list[dict[str, Any]] = []

    async def get(self, approval_id: str) -> SimpleNamespace:
      assert approval_id == "approval-1"
      return SimpleNamespace(approval_id=approval_id, state="pending_user", state_version=7)

    async def transition_state(self, approval_id: str, state: str, **kwargs: Any) -> SimpleNamespace:
      self.transitions.append({"approval_id": approval_id, "state": state, **kwargs})
      return SimpleNamespace(approval_id=approval_id, state=state, state_version=8)

  async def scenario() -> None:
    store = Store()
    session = SimpleNamespace(pending_tools={}, approval_queues={}, agent_session_log=None)

    result = await lifecycle_helpers.await_user_approval_via_pending_tools(
      session=session,
      approval_store=store,
      event_log=None,
      request=_request(),
      decision=_decision(),
      nonce="nonce-1",
      resolved_qualifier="",
      allow_persistent=False,
      timeout_seconds=0,
      log=logging.getLogger("test"),
    )

    assert result is None
    assert store.transitions == [
      {
        "approval_id": "approval-1",
        "state": "expired",
        "expected_state_version": 7,
        "decision_reason": "Timed out waiting for user approval",
      }
    ]
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  asyncio.run(scenario())


def test_tool_dispatcher_pending_approval_wrapper_threads_instance_state(monkeypatch) -> None:
  captured: dict[str, Any] = {}

  async def fake_await_user_approval_via_pending_tools(**kwargs: Any) -> dict[str, bool]:
    captured.update(kwargs)
    return {"approved": True}

  monkeypatch.setattr(
    lifecycle_helpers,
    "await_user_approval_via_pending_tools",
    fake_await_user_approval_via_pending_tools,
  )
  session = SimpleNamespace(
    pending_tools={},
    approval_queues={},
    approval_store="store",
    approval_policy="policy",
  )
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={},
    event_log=EventLog(),
    session=session,
  )
  request = _request()
  decision = _decision()

  result = asyncio.run(
    dispatcher._await_user_approval_via_pending_tools(
      request,
      decision,
      nonce="nonce-1",
      resolved_qualifier="qual-1",
      allow_persistent=True,
      timeout_seconds=15,
    )
  )

  assert result == {"approved": True}
  assert captured["session"] is session
  assert captured["approval_store"] == "store"
  assert captured["event_log"] is dispatcher._event_log
  assert captured["request"] is request
  assert captured["decision"] is decision
  assert captured["nonce"] == "nonce-1"
  assert captured["resolved_qualifier"] == "qual-1"
  assert captured["allow_persistent"] is True
  assert captured["timeout_seconds"] == 15


def test_tool_dispatcher_run_approval_lifecycle_wrapper_threads_instance_state(monkeypatch) -> None:
  captured: dict[str, Any] = {}

  async def fake_run_approval_lifecycle(**kwargs: Any) -> dict[str, bool]:
    captured.update(kwargs)
    return {"approved": True}

  monkeypatch.setattr(
    lifecycle_helpers,
    "run_approval_lifecycle",
    fake_run_approval_lifecycle,
  )
  session = SimpleNamespace(
    pending_tools={},
    approval_queues={},
    approval_store="store",
    approval_policy="policy",
  )
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcp(),
    local_tool_handlers={},
    event_log=EventLog(),
    session=session,
  )

  result = asyncio.run(
    dispatcher._run_approval_lifecycle(
      tool_call_id="call-1",
      tool_name="place_order",
      tool_input={"ticker": "MSFT"},
      qualifier="qual-1",
      reason="needs approval",
      allow_persistent=True,
    )
  )

  assert result == {"approved": True}
  assert captured["store"] == "store"
  assert captured["policy"] == "policy"
  assert captured["session"] is session
  assert captured["tool_call_id"] == "call-1"
  assert captured["tool_name"] == "place_order"
  assert captured["tool_input"] == {"ticker": "MSFT"}
  assert captured["qualifier"] == "qual-1"
  assert captured["reason"] == "needs approval"
  assert captured["allow_persistent"] is True
  assert captured["resolve_run_context_fn"].__self__ is dispatcher
  assert captured["redact_for_approval_request_fn"].__self__ is dispatcher
  assert captured["resolve_tool_class_fn"].__self__ is dispatcher
  assert captured["effective_trade_approval_decision_fn"].__self__ is dispatcher
  assert captured["await_user_approval_via_pending_tools_fn"].__self__ is dispatcher
  assert callable(captured["current_skill_fn"])
  assert callable(captured["approval_queue_timeout_seconds_fn"])
