from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))

from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog, SessionStore  # noqa: E402
from agent_gateway import policy_imports, sdk_runner_approval  # noqa: E402
from agent_gateway.approval_policy import ApprovalDecision as PolicyApprovalDecision, ApprovalRequest, ApprovalRequestPayload, RunContext  # noqa: E402
from agent_gateway.approval_store import SQLiteApprovalStore  # noqa: E402
from agent_gateway.approvals import _record_vote_and_unblock  # noqa: E402
from agent_gateway.batch_approval_projection import (  # noqa: E402
  BatchApprovalProjectionRegistry,
  BatchApprovalScope,
)
from agent_gateway.providers.agent_sdk import SDK_PINNED_VERSION  # noqa: E402
from agent_gateway.runner import _ACTIVE_SKILL_DENY_RESULT_KEY, _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY  # noqa: E402
from agent_gateway.skill_context import clear_current_skill, current_skill, set_current_skill  # noqa: E402


def _run(coro):
  return asyncio.run(coro)


class _PermissionResultAllow:
  behavior = "allow"

  def __init__(
    self,
    *,
    updated_input: dict[str, Any] | None = None,
    updated_permissions: list[Any] | None = None,
  ) -> None:
    self.updated_input = updated_input
    self.updated_permissions = updated_permissions


class _PermissionResultDeny:
  behavior = "deny"

  def __init__(self, *, message: str = "", interrupt: bool = False) -> None:
    self.message = message
    self.interrupt = interrupt


class _HookMatcher:
  def __init__(self, *, hooks: list[Any]) -> None:
    self.hooks = hooks


class _ClaudeAgentOptions:
  def __init__(self, **kwargs: Any) -> None:
    self.kwargs = kwargs


class _AsyncMessages:
  def __init__(self, messages: list[Any]) -> None:
    self.messages = list(messages)
    self.closed = False

  def __aiter__(self):
    return self

  async def __anext__(self):
    if not self.messages:
      raise StopAsyncIteration
    message = self.messages.pop(0)
    if isinstance(message, BaseException):
      raise message
    return message

  async def aclose(self) -> None:
    self.closed = True
    self.messages.clear()


def _sdk_result_message() -> types.SimpleNamespace:
  return types.SimpleNamespace(
    duration_ms=1,
    num_turns=1,
    usage={
      "input_tokens": 0,
      "output_tokens": 0,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0,
    },
    total_cost_usd=0.0,
  )


def _install_fake_agent_sdk(
  monkeypatch: pytest.MonkeyPatch,
  iterator_factory: Callable[[Any, Any], Any] | None = None,
) -> types.SimpleNamespace:
  state = types.SimpleNamespace(options=[], prompts=[])

  def _query(prompt: Any, options: Any):
    state.prompts.append(prompt)
    state.options.append(options)
    if iterator_factory is not None:
      return iterator_factory(prompt, options)
    return _AsyncMessages([_sdk_result_message()])

  module = types.ModuleType("claude_agent_sdk")
  module.__version__ = SDK_PINNED_VERSION
  module.HookMatcher = _HookMatcher
  module.ClaudeAgentOptions = _ClaudeAgentOptions
  module.PermissionResultAllow = _PermissionResultAllow
  module.PermissionResultDeny = _PermissionResultDeny
  module.query = _query
  monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
  return state


def _make_runner(
  *,
  event_log: EventLog | None = None,
  disallowed_tools: list[str] | None = None,
  on_tool_result: Any | None = None,
) -> AgentSDKRunner:
  return AgentSDKRunner(
    event_log=event_log or EventLog(),
    session_id="sess-sdk-enforce",
    sdk_config=AgentSDKConfig(
      api_key="k",
      model="claude-sonnet-4-6",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    system_prompt="test",
    disallowed_tools=list(disallowed_tools or []),
    on_tool_result=on_tool_result,
  )


def test_sdk_runner_static_disallowed_tool_denied_without_approval(monkeypatch: pytest.MonkeyPatch) -> None:
  state = _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner(disallowed_tools=["file_write"])

  _run(runner.run([{"role": "user", "content": "hello"}]))

  assert state.options
  callback = state.options[0].kwargs["can_use_tool"]
  assert getattr(callback, "__self__", None) is runner
  assert getattr(callback, "__func__", None) is AgentSDKRunner._can_use_tool_callback

  denied = _run(callback("file_write", {"path": "x"}, None))
  assert denied.behavior == "deny"
  assert denied.message == "Tool 'file_write' is not available in this context"

  allowed = _run(callback("file_read", {"path": "x"}, None))
  assert allowed.behavior == "allow"


def test_sdk_runner_stale_prefixed_mcp_tool_denied_without_approval(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_fake_agent_sdk(monkeypatch)
  from agent.shared import server_policies

  monkeypatch.setattr(
    server_policies,
    "get_server_for_policy_tool",
    lambda tool_name: "portfolio-trades-mcp" if tool_name == "execute_trade" else None,
  )
  runner = _make_runner()

  denied = _run(
    runner._can_use_tool_callback(
      "mcp__portfolio-reads-mcp__execute_trade",
      {"preview_id": "p1"},
      None,
    )
  )

  assert denied.behavior == "deny"
  assert "policy owner for 'execute_trade' is 'portfolio-trades-mcp'" in denied.message


def test_sdk_runner_stale_prefixed_mcp_tool_policy_import_drift_fails_loud(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)

  def fake_import_module(_name: str):
    raise ModuleNotFoundError("No module named 'broken_dependency'", name="broken_dependency")

  monkeypatch.setattr(policy_imports.importlib, "import_module", fake_import_module)
  runner = _make_runner()

  with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
    _run(
      runner._can_use_tool_callback(
        "mcp__portfolio-reads-mcp__execute_trade",
        {"preview_id": "p1"},
        None,
      )
    )


@pytest.mark.parametrize(
  "tool_name",
  [
    "merge_diligence_pr",
    "mcp__portfolio-proposals-local__merge_diligence_pr",
  ],
)
@pytest.mark.parametrize("lifecycle_configured", [False, True])
def test_sdk_runner_promotion_saga_requires_owner_control_route_before_approval(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
  lifecycle_configured: bool,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  calls: list[str] = []

  class Store:
    async def create(self, _request: ApprovalRequest) -> ApprovalRequest:
      calls.append("store")
      raise AssertionError("promotion must be refused before approval persistence")

  class Policy:
    async def decide(self, **_kwargs: Any) -> PolicyApprovalDecision:
      calls.append("policy")
      return PolicyApprovalDecision(
        outcome="auto_approve",
        reason="custom policy attempted automatic promotion",
      )

  runner = _make_runner()
  if lifecycle_configured:
    runner._session = SimpleNamespace(
      session_id="sess-sdk-enforce",
      user_id="alice",
      channel="web",
      role="owner",
      pending_tools={},
      approval_queues={},
    )
    runner._approval_store = Store()
    runner._approval_policy = Policy()

  denied = _run(
    runner._can_use_tool_callback(
      tool_name,
      {"pr_id": "dpr-1", "confirm_merge": True},
      None,
    )
  )

  assert denied.behavior == "deny"
  assert "[owner_control_route_required]" in denied.message
  assert "authenticated owner control-plane route" in denied.message.lower()
  assert calls == []


def test_sdk_runner_catalog_constraint_failure_denies_before_lifecycle(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)

  def unavailable_constraint(_tool_name: str) -> str:
    raise RuntimeError("catalog dependency failed")

  monkeypatch.setattr(
    sdk_runner_approval,
    "constraint_for_catalog_tool",
    unavailable_constraint,
  )
  runner = _make_runner()

  denied = _run(
    runner._can_use_tool_callback(
      "get_portfolio_summary",
      {},
      None,
    )
  )

  assert denied.behavior == "deny"
  assert "[approval_constraint_unavailable]" in denied.message


def test_sdk_runner_relay_policy_denial_uses_machine_readable_message(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _install_fake_agent_sdk(monkeypatch)

  class _Policy:
    policy_bundle_hash = "test-policy"

    async def decide(
      self,
      *,
      payload: ApprovalRequestPayload,
      request: ApprovalRequest,
      run_context: RunContext,
    ):
      _ = payload, request, run_context
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="Tool requires approval",
        route_target_type="pending_tools",
        expiry_seconds=600,
        allow_persistent_grant=True,
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

    async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
      _ = grant_id, reason

    def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
      _ = decider_role, tool_class
      return True

  async def _case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    policy = _Policy()
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    runner = AgentSDKRunner(
      event_log=EventLog(),
      session_id=session.session_id,
      sdk_config=AgentSDKConfig(
        api_key="k",
        model="claude-sonnet-4-6",
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      system_prompt="test",
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id=session.session_id,
        channel="web",
      ),
    )

    callback_task = asyncio.create_task(runner._can_use_tool_callback("file_write", {"path": "x"}, None))
    for _ in range(100):
      if session.pending_tools:
        break
      await asyncio.sleep(0.001)
    else:
      raise AssertionError("approval request was not queued")

    tool_call_id, pending = next(iter(session.pending_tools.items()))
    await _record_vote_and_unblock(
      target_session=session,
      pending_entry=pending,
      tool_call_id=tool_call_id,
      nonce=pending["nonce"],
      decider_id="alice",
      decider_role="owner",
      approved=False,
      allow_tool_type=False,
      reason=None,
      app_state=SimpleNamespace(gateway_approval_store=store, gateway_approval_policy=policy),
      denied_by="relay_policy",
    )
    denied = await callback_task

    assert denied.behavior == "deny"
    assert denied.message.startswith("[relay_policy_denied]")
    assert "relay chat policy" in denied.message
    assert "taskpane composer" in denied.message

  _run(_case())


def test_sdk_batch_admission_cancel_before_pending_publish_aborts_durable_row(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _install_fake_agent_sdk(monkeypatch)

  class _Policy:
    policy_bundle_hash = "sdk-batch-admission-policy"

    def __init__(self) -> None:
      self.request: ApprovalRequest | None = None

    async def decide(
      self,
      *,
      payload: ApprovalRequestPayload,
      request: ApprovalRequest,
      run_context: RunContext,
    ) -> PolicyApprovalDecision:
      _ = payload, run_context
      self.request = request
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="Tool requires approval",
        route_target_type="pending_tools",
        expiry_seconds=600,
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

  async def _case() -> None:
    store = SQLiteApprovalStore(tmp_path / "sdk-batch-admission.sqlite3")
    policy = _Policy()
    registry = BatchApprovalProjectionRegistry()
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
    )
    session.channel = "tui"
    scope = BatchApprovalScope(
      batch_id=77,
      owner_user_id="alice",
      channel="tui",
      store=store,
      policy=policy,
      registry=registry,
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    runner = AgentSDKRunner(
      event_log=EventLog(),
      session_id=session.session_id,
      sdk_config=AgentSDKConfig(
        api_key="k",
        model="claude-sonnet-4-6",
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      system_prompt="test",
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id="batch_77",
        run_id="batch_77",
        session_id=session.session_id,
        channel="tui",
      ),
    )
    pending_committed = asyncio.Event()

    async def pause_before_pending_publish(
      request: ApprovalRequest,
      decision: PolicyApprovalDecision,
      *,
      nonce: str,
      batch_admission: Any | None = None,
    ) -> None:
      _ = request, decision, nonce, batch_admission
      pending_committed.set()
      await asyncio.Event().wait()

    runner._await_user_approval_via_pending_tools = pause_before_pending_publish  # type: ignore[method-assign]
    callback_task = asyncio.create_task(
      runner._can_use_tool_callback("file_write", {"path": "x"}, None)
    )
    await pending_committed.wait()
    callback_task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await callback_task

    assert policy.request is not None
    stored = await store.get(policy.request.approval_id)
    assert stored is not None
    assert stored.state == "denied"
    assert session.pending_tools == {}
    assert session.approval_queues == {}
    assert registry.projections_for_batch(owner_user_id="alice", batch_id=77) == []
    assert registry._admission_gates[("alice", 77)].active == 0

  _run(_case())


def test_sdk_runner_policy_modified_input_behavior_is_unchanged(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  modified = {"path": "normalized"}
  resolved: list[ApprovalRequest] = []

  class _Policy:
    policy_bundle_hash = "test-policy"

    async def decide(
      self,
      *,
      payload: ApprovalRequestPayload,
      request: ApprovalRequest,
      run_context: RunContext,
    ):
      _ = request, run_context
      assert payload.tool_args == {"path": "raw"}
      return PolicyApprovalDecision(
        outcome="auto_approve",
        reason="normalized input",
        modified_tool_args=modified,
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      resolved.append(request)

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
  )
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id=session.session_id,
    sdk_config=AgentSDKConfig(
      api_key="k",
      model="claude-sonnet-4-6",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    system_prompt="test",
    session=session,
    store=store,
    policy=_Policy(),
  )
  original = {"path": "raw"}

  allowed = _run(runner._can_use_tool_callback("file_write", original, None))

  assert allowed.behavior == "allow"
  assert allowed.updated_input == modified
  assert original == {"path": "raw"}
  assert len(resolved) == 1
  assert resolved[0].state == "auto_approved"


def test_sdk_runner_trade_approval_record_includes_preview_summary(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  preview_expires_at = datetime.now(UTC) + timedelta(seconds=120)

  class _Policy:
    policy_bundle_hash = "test-policy"

    async def decide(
      self,
      *,
      payload: ApprovalRequestPayload,
      request: ApprovalRequest,
      run_context: RunContext,
    ):
      _ = payload, request, run_context
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="Tool requires approval",
        route_target_type="pending_tools",
        expiry_seconds=600,
        allow_persistent_grant=True,
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

    async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
      _ = grant_id, reason

    def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
      _ = decider_role, tool_class
      return True

  event_log = EventLog()
  event_log.append(
    {
      "type": "tool_call_complete",
      "tool_call_id": "preview-1",
      "tool_name": "mcp__portfolio-reads-mcp__preview_trade",
      "result": {
        "status": "success",
        "metadata": {
          "account_id": "acct-1",
          "expires_at": preview_expires_at.isoformat(),
          "broker_provider": "ibkr",
        },
        "data": {
          "preview_id": "p1",
          "ticker": "SGOV",
          "side": "BUY",
          "quantity": 10,
          "order_type": "Market",
          "time_in_force": "Day",
          "estimated_price": 100.25,
          "estimated_total": 1002.5,
          "estimated_commission": 0.0,
          "validation": {"is_valid": True, "warnings": []},
        },
      },
      "error": None,
    }
  )

  async def _case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    policy = _Policy()
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    runner = AgentSDKRunner(
      event_log=event_log,
      session_id=session.session_id,
      sdk_config=AgentSDKConfig(
        api_key="k",
        model="claude-sonnet-4-6",
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      system_prompt="test",
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id=session.session_id,
        channel="web",
      ),
    )

    callback_task = asyncio.create_task(
      runner._can_use_tool_callback("mcp__portfolio-trades-mcp__execute_trade", {"preview_id": "p1"}, None)
    )
    for _ in range(100):
      if session.pending_tools:
        break
      await asyncio.sleep(0.001)
    else:
      raise AssertionError("approval request was not queued")

    tool_call_id, pending = next(iter(session.pending_tools.items()))
    request = await store.get(str(pending["approval_id"]))
    assert request is not None
    assert request.tool_args_redacted["preview_id"] == "p1"
    assert request.tool_args_redacted["approval_summary"]["ticker"] == "SGOV"
    assert request.tool_args_redacted["approval_summary"]["quantity"] == 10
    assert request.tool_args_redacted["approval_summary"]["estimated_total"] == 1002.5
    assert request.expires_at is not None
    approval_window = (request.expires_at - request.requested_at).total_seconds()
    assert 85 <= approval_window <= 91

    await _record_vote_and_unblock(
      target_session=session,
      pending_entry=pending,
      tool_call_id=tool_call_id,
      nonce=pending["nonce"],
      decider_id="alice",
      decider_role="owner",
      approved=False,
      allow_tool_type=False,
      reason="test",
      app_state=SimpleNamespace(gateway_approval_store=store, gateway_approval_policy=policy),
    )
    denied = await callback_task
    assert denied.behavior == "deny"

  _run(_case())


def test_sdk_runner_opted_in_skill_result_activates_write_tool_deny(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_fake_agent_sdk(monkeypatch)
  contexts = []

  async def _on_tool_result(ctx: Any):
    contexts.append(ctx)
    return []

  runner = _make_runner(on_tool_result=_on_tool_result)
  result = {
    "skill": "phase0-agent",
    "content": "Do the work.",
    _ACTIVE_SKILL_DENY_RESULT_KEY: ["file_write"],
  }

  hook_result = _run(
    runner._post_tool_use_hook(
      {
        "tool_name": "invoke_skill",
        "tool_input": {"skill_name": "phase0-agent"},
        "result": json.dumps(result),
      },
      "tool-invoke",
      None,
    )
  )

  assert hook_result == {}
  assert runner._active_skill_deny == {"file_write"}
  assert contexts[0].result == {"skill": "phase0-agent", "content": "Do the work."}
  assert _ACTIVE_SKILL_DENY_RESULT_KEY not in contexts[0].result

  denied = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))
  assert denied.behavior == "deny"
  assert denied.message == "Tool 'file_write' is not available in this context"


def test_sdk_runner_legacy_skill_result_does_not_activate_active_skill_gate(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner(disallowed_tools=["run_bash"])

  result = {"skill": "legacy-agent", "content": "Legacy skill body."}
  normalized, error = runner._normalize_tool_result(
    {
      "type": "tool_result",
      "tool_use_id": "tool-invoke",
      "content": json.dumps(result),
    }
  )

  assert error is None
  assert normalized == result
  assert runner._active_skill_deny == set()

  static_denied = _run(runner._can_use_tool_callback("run_bash", {"command": "date"}, None))
  assert static_denied.behavior == "deny"
  assert static_denied.message == "Tool 'run_bash' is not available in this context"

  dynamic_allowed = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))
  assert dynamic_allowed.behavior == "allow"


def test_sdk_runner_active_skill_deny_replaces_existing_set(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()
  runner._active_skill_deny = {"file_write", "run_bash"}

  runner._activate_skill_deny(["file_write"])

  assert runner._active_skill_deny == {"file_write"}

  runner._activate_skill_deny([])

  assert runner._active_skill_deny == set()


def test_sdk_runner_report_door_clears_active_skill_gate(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()
  runner._active_skill_deny = {"emit_html_artifact"}
  set_current_skill("sniff-test")

  try:
    invoke_result = {
      "skill": "sniff-test",
      "content": "Do the sniff test.",
      _ACTIVE_SKILL_DENY_RESULT_KEY: ["emit_html_artifact"],
      _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY: {"fms_report_sniff_test": "sniff-test"},
    }
    stripped = runner._consume_private_tool_result_fields(invoke_result)
    assert stripped == {"skill": "sniff-test", "content": "Do the sniff test."}
    assert runner._active_skill_deny == {"emit_html_artifact"}
    assert runner._active_skill_report_doors == {"fms_report_sniff_test": "sniff-test"}

    result = {
      "status": "staged",
      "subcommand": "report_sniff_test",
      "mutation_mode": "preview",
    }
    runner._clear_active_skill_if_report_door_completed(
      tool_name="fms_report_sniff_test",
      result=result,
      error=None,
    )

    assert runner._active_skill_deny == set()
    assert current_skill() is None
  finally:
    clear_current_skill()


def test_sdk_runner_model_writer_terminal_door_clears_build_model_deny(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()
  runner._active_skill_deny = {"build_model"}
  runner._active_skill_report_doors = {"fms_persist_business_model": "business-model-construction"}
  set_current_skill("business-model-construction")

  try:
    denied = _run(runner._can_use_tool_callback("build_model", {}, None))
    assert denied.behavior == "deny"

    cleared = runner._clear_active_skill_if_report_door_completed(
      tool_name="fms_persist_business_model",
      result={
        "status": "staged",
        "subcommand": "persist_business_model",
        "mutation_mode": "model_writer",
      },
      error=None,
    )

    assert cleared is True
    assert runner._active_skill_deny == set()
    assert runner._active_skill_report_doors == {}
    assert current_skill() is None

    allowed = _run(runner._can_use_tool_callback("build_model", {}, None))
    assert allowed.behavior == "allow"
  finally:
    clear_current_skill()


def test_sdk_runner_model_writer_mid_skill_build_model_deny_holds(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()
  runner._active_skill_deny = {"build_model"}
  runner._active_skill_report_doors = {"fms_persist_business_model": "business-model-construction"}
  set_current_skill("business-model-construction")

  try:
    denied = _run(runner._can_use_tool_callback("build_model", {}, None))
    assert denied.behavior == "deny"

    cleared = runner._clear_active_skill_if_report_door_completed(
      tool_name="fms_report_build_model",
      result={
        "status": "staged",
        "subcommand": "report_build_model",
        "mutation_mode": "preview",
      },
      error=None,
    )

    assert cleared is False
    assert runner._active_skill_deny == {"build_model"}
    assert runner._active_skill_report_doors == {"fms_persist_business_model": "business-model-construction"}
    assert current_skill() == "business-model-construction"
  finally:
    clear_current_skill()


def test_sdk_runner_report_door_semantic_error_does_not_clear_active_skill_gate(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()
  runner._active_skill_deny = {"emit_html_artifact"}
  runner._active_skill_report_doors = {"fms_report_sniff_test": "sniff-test"}
  set_current_skill("sniff-test")

  try:
    cleared = runner._clear_active_skill_if_report_door_completed(
      tool_name="fms_report_sniff_test",
      result={
        "status": "error",
        "subcommand": "report_sniff_test",
        "mutation_mode": "preview",
        "message": "judgment rejected",
      },
      error=None,
    )

    assert cleared is False
    assert runner._active_skill_deny == {"emit_html_artifact"}
    assert runner._active_skill_report_doors == {"fms_report_sniff_test": "sniff-test"}
    assert current_skill() == "sniff-test"
  finally:
    clear_current_skill()


def test_sdk_runner_active_skill_deny_clears_on_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_fake_agent_sdk(monkeypatch)
  success_runner = _make_runner()
  success_runner._active_skill_deny.add("file_write")

  async def _run_success() -> None:
    set_current_skill("phase0-agent")
    await success_runner.run([{"role": "user", "content": "hello"}])
    assert current_skill() is None

  _run(_run_success())
  assert success_runner._active_skill_deny == set()

  _install_fake_agent_sdk(
    monkeypatch,
    iterator_factory=lambda _prompt, _options: _AsyncMessages([RuntimeError("sdk failed")]),
  )
  error_runner = _make_runner()
  error_runner._active_skill_deny.add("file_write")

  async def _run_error() -> None:
    set_current_skill("phase0-agent")
    with pytest.raises(RuntimeError, match="sdk failed"):
      await error_runner.run([{"role": "user", "content": "hello"}])
    assert current_skill() is None

  _run(_run_error())
  assert error_runner._active_skill_deny == set()
