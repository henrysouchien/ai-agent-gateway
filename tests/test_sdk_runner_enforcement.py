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
from agent_gateway.runner import (  # noqa: E402
  _ACTIVE_SKILL_ALLOW_RESULT_KEY,
  _ACTIVE_SKILL_DENY_RESULT_KEY,
  _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY,
)
from agent_gateway.skill_context import clear_current_skill, current_skill, set_current_skill  # noqa: E402
from tests.sdk_capability_execution_test_support import stub_sdk_capability_execution  # noqa: E402


def _run(coro):
  return asyncio.run(coro)


def test_sdk_approval_context_uses_canonical_session_owner_identity() -> None:
  session = SimpleNamespace(
    user_id="henry",
    owner_user_id="1",
    channel="cli",
    role="owner",
  )

  resolved = sdk_runner_approval.resolve_run_context(
    run_context=RunContext(
      user_id="henry",
      request_id="request-1",
      session_id="session-1",
      profile="chat",
      channel="cli",
    ),
    usage_user_id="henry",
    session=session,
    approval_policy=SimpleNamespace(policy_bundle_hash="policy-1"),
    request_id="request-1",
    session_id="session-1",
    channel="cli",
  )

  assert resolved.user_id == "1"


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
    subtype="success",
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
  mcp_server_configs: dict[str, Any] | None = None,
  on_tool_result: Any | None = None,
  run_context: RunContext | None = None,
  skill_run_id: str | None = None,
  max_tokens_override: int | None = None,
  api_key: str = "test-secret",
) -> AgentSDKRunner:
  return AgentSDKRunner(
    event_log=event_log or EventLog(),
    session_id="sess-sdk-enforce",
    sdk_config=AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=stub_sdk_capability_execution(api_key=api_key),
    system_prompt="test",
    disallowed_tools=list(disallowed_tools or []),
    mcp_server_configs=mcp_server_configs,
    on_tool_result=on_tool_result,
    run_context=run_context,
    skill_run_id=skill_run_id,
    max_tokens_override=max_tokens_override,
  )


def test_sdk_post_tool_use_replaces_model_output_with_sanitized_projection() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-CODEX-SDK-8f21d7"
  runner = _make_runner(api_key=secret)

  hook_result = _run(
    runner._post_tool_use_hook(
      {
        "tool_name": "lookup",
        "tool_input": {"query": "ordinary"},
        "result": json.dumps({"status": "ok", "credential": secret}),
      },
      "tool-secret",
      None,
    )
  )

  serialized = json.dumps(hook_result)
  assert secret not in serialized
  assert "<redacted-secret>" in serialized


def test_sdk_post_tool_failure_blocks_raw_secret_from_model_continuation() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-CODEX-SDK-ERROR-8f21d7"
  runner = _make_runner(api_key=secret)

  hook_result = _run(
    runner._post_tool_use_failure_hook(
      {
        "tool_name": "lookup",
        "tool_input": {"query": "ordinary"},
        "error": f"provider rejected credential {secret}",
      },
      "tool-secret-error",
      None,
    )
  )

  serialized = json.dumps(hook_result)
  assert hook_result["decision"] == "block"
  assert secret not in serialized
  assert "<redacted-secret>" in serialized


def test_sdk_runner_projects_max_tokens_override_to_pinned_sdk_environment() -> None:
  runner = _make_runner(max_tokens_override=32000)

  assert runner._max_tokens_override == 32000
  assert runner._credential_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"


def test_sdk_runner_denies_forged_same_server_tool_outside_advertised_stage_scope(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)

  class _StageMcpConfigs(dict[str, Any]):
    sdk_admission_enforced = True
    advertised_mcp_tool_ids_by_server = {
      "research-corpus-mcp": {
        "mcp__research-corpus-mcp__filings_read",
      }
    }

  runner = _make_runner(
    mcp_server_configs=_StageMcpConfigs({
      "research-corpus-mcp": {"command": "research-corpus"},
    })
  )

  allowed = _run(
    runner._can_use_tool_callback(
      "mcp__research-corpus-mcp__filings_read",
      {},
      None,
    )
  )
  denied = _run(
    runner._can_use_tool_callback(
      "mcp__research-corpus-mcp__transcripts_read",
      {},
      None,
    )
  )

  assert allowed.behavior == "allow"
  assert denied.behavior == "deny"
  assert "not available in this context" in denied.message


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


def test_sdk_runner_executes_report_admission_with_same_carried_run_identity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  state = _install_fake_agent_sdk(monkeypatch)
  run_context = RunContext(
    user_id="alice",
    request_id="request-sdk-report",
    session_id="sess-sdk-report",
    run_id="skill-run-sdk-report",
  )
  runner = _make_runner(
    run_context=run_context,
    skill_run_id="skill-run-sdk-report",
  )

  _run(runner.run([{"role": "user", "content": "report the build"}]))

  callback = state.options[0].kwargs["can_use_tool"]
  decision = _run(callback("fms_report_build_model", {}, None))
  assert decision.behavior == "allow"
  assert runner._resolve_run_context().run_id == "skill-run-sdk-report"


def test_sdk_runner_report_admission_fails_closed_without_carrier(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()

  decision = _run(
    runner._can_use_tool_callback("fms_report_build_model", {}, None)
  )

  assert decision.behavior == "deny"
  assert "[run_identity_required]" in decision.message


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
    "promote_reviewed_change",
  ],
)
@pytest.mark.parametrize("lifecycle_configured", [False, True])
def test_sdk_runner_promotion_saga_requires_owner_control_route_before_approval(
  monkeypatch: pytest.MonkeyPatch,
  tool_name: str,
  lifecycle_configured: bool,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  monkeypatch.setattr(
    sdk_runner_approval,
    "constraint_for_catalog_tool",
    lambda _tool_name: "fresh_human_owner",
  )
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
      {"change_id": "change-1", "confirm": True},
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


def test_sdk_runner_user_denial_uses_ordinary_message(
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
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      capability_execution=stub_sdk_capability_execution(),
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
    )
    denied = await callback_task

    assert denied.behavior == "deny"
    assert denied.message == "user denied"
    assert not denied.interrupt

  _run(_case())


def test_sdk_runner_approval_expiry_interrupts_turn_instead_of_reading_as_denial(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _install_fake_agent_sdk(monkeypatch)

  class _Policy:
    policy_bundle_hash = "test-policy"

    def __init__(self) -> None:
      self.request: ApprovalRequest | None = None

    async def decide(
      self,
      *,
      payload: ApprovalRequestPayload,
      request: ApprovalRequest,
      run_context: RunContext,
    ):
      _ = payload, run_context
      self.request = request
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="Tool requires approval",
        expiry_seconds=0.2,
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
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      capability_execution=stub_sdk_capability_execution(),
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

    # Nobody ever votes: the queue wait must expire.
    denied = await runner._can_use_tool_callback("file_write", {"path": "x"}, None)

    assert denied.behavior == "deny"
    assert denied.interrupt is True
    assert "approval_timeout" in denied.message
    assert "user denied" not in denied.message
    assert session.pending_tools == {}
    assert session.approval_queues == {}

    assert policy.request is not None
    stored = await store.get(policy.request.approval_id)
    assert stored is not None
    assert stored.state == "expired"

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
    session.batch_stage_run_seq = 3
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
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      capability_execution=stub_sdk_capability_execution(),
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


def test_sdk_projected_pending_tool_binds_stage_identity_to_projection_and_event() -> None:
  async def _case() -> None:
    events: list[dict[str, Any]] = []
    request = SimpleNamespace(
      approval_id="approval-sdk-batch",
      tool_call_id="tool-sdk-batch",
      tool_name="file_write",
      tool_args_redacted={"path": "model.xlsx"},
    )
    session = SimpleNamespace(
      pending_tools={},
      approval_queues={},
      batch_stage_run_seq=3,
    )

    class _Admission:
      def publish_pending(self) -> None:
        pending = session.pending_tools[request.tool_call_id]
        assert pending["stage_run_seq"] == 3
        session.approval_queues[request.tool_call_id].put_nowait(
          {"approved": False}
        )

    result = await sdk_runner_approval.await_user_approval_via_pending_tools(
      session=session,
      approval_store=None,
      request=request,
      decision=SimpleNamespace(
        reason="review required",
        allow_persistent_grant=False,
      ),
      nonce="nonce-sdk-batch",
      append_event_fn=events.append,
      timeout_seconds=5,
      log=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
      batch_admission=_Admission(),
    )

    assert result == ("denied", {"approved": False})
    assert events[0]["stage_run_seq"] == 3
    assert session.pending_tools == {}
    assert session.approval_queues == {}

  _run(_case())


@pytest.mark.parametrize("stage_run_seq", [None, 0, -1, True, "3"])
def test_sdk_projected_pending_tool_rejects_invalid_stage_identity(
  stage_run_seq: object,
) -> None:
  session = SimpleNamespace(
    pending_tools={},
    approval_queues={},
    batch_stage_run_seq=stage_run_seq,
  )

  with pytest.raises(
    ValueError,
    match="stage_run_seq must be a positive integer",
  ):
    _run(
      sdk_runner_approval.await_user_approval_via_pending_tools(
        session=session,
        approval_store=None,
        request=SimpleNamespace(
          approval_id="approval-sdk-batch",
          tool_call_id="tool-sdk-batch",
          tool_name="file_write",
          tool_args_redacted={},
        ),
        decision=SimpleNamespace(
          reason="review required",
          allow_persistent_grant=False,
        ),
        nonce="nonce-sdk-batch",
        append_event_fn=lambda _event: None,
        timeout_seconds=5,
        log=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
        batch_admission=object(),
      )
    )
  assert session.pending_tools == {}
  assert session.approval_queues == {}


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
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=stub_sdk_capability_execution(),
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


def test_sdk_approval_persists_exact_secret_safe_projection_but_policy_receives_raw_input(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  secret = "CUSTOM-ACTIVE-CREDENTIAL-SDK-APPROVAL-8f21d7"
  original = {
    "path": "/Users/alice/Documents/report.xlsx",
    "credential": secret,
    "api_key_set": True,
    "note": "Ordinary api_key discussion and sk-example text.",
  }

  class _Policy:
    policy_bundle_hash = "test-policy"

    def __init__(self) -> None:
      self.raw_args: dict[str, Any] | None = None
      self.request: ApprovalRequest | None = None

    async def decide(
      self,
      *,
      payload: ApprovalRequestPayload,
      request: ApprovalRequest,
      run_context: RunContext,
    ) -> PolicyApprovalDecision:
      _ = run_context
      self.raw_args = dict(payload.tool_args)
      self.request = request
      return PolicyApprovalDecision(
        outcome="auto_approve",
        reason="approved",
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  policy = _Policy()
  session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
  )
  runner = AgentSDKRunner(
    event_log=EventLog(),
    session_id=session.session_id,
    sdk_config=AgentSDKConfig(
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=stub_sdk_capability_execution(api_key=secret),
    system_prompt="test",
    session=session,
    store=store,
    policy=policy,
  )

  allowed = _run(runner._can_use_tool_callback("file_write", original, None))

  assert allowed.behavior == "allow"
  assert policy.raw_args == original
  assert policy.request is not None
  expected_projection = {
    **original,
    "credential": "<redacted-secret>",
  }
  assert policy.request.tool_args_redacted == expected_projection
  stored = _run(store.get(policy.request.approval_id))
  assert stored is not None
  assert stored.tool_args_redacted == expected_projection
  assert secret not in json.dumps(stored.tool_args_redacted)
  assert original["credential"] == secret


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
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      capability_execution=stub_sdk_capability_execution(),
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


def test_sdk_runner_opted_in_skill_result_activates_exact_allow_and_write_deny(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  contexts = []

  async def _on_tool_result(ctx: Any):
    contexts.append(ctx)
    return []

  runner = _make_runner(
    on_tool_result=_on_tool_result,
    disallowed_tools=["start_investment_run"],
  )
  result = {
    "skill": "phase0-agent",
    "content": "Do the work.",
    _ACTIVE_SKILL_ALLOW_RESULT_KEY: ["start_investment_run"],
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
  assert runner._active_skill_allow == {"start_investment_run"}
  assert runner._active_skill_deny == {"file_write"}
  assert contexts[0].result == {"skill": "phase0-agent", "content": "Do the work."}
  assert _ACTIVE_SKILL_ALLOW_RESULT_KEY not in contexts[0].result
  assert _ACTIVE_SKILL_DENY_RESULT_KEY not in contexts[0].result

  allowed = _run(runner._can_use_tool_callback("start_investment_run", {}, None))
  assert allowed.behavior == "allow"

  denied = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))
  assert denied.behavior == "deny"
  assert denied.message == "Tool 'file_write' is not available in this context"

  runner._activate_skill_deny(["start_investment_run"])
  denied_again = _run(runner._can_use_tool_callback("start_investment_run", {}, None))
  assert denied_again.behavior == "deny"


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
  runner._active_skill_deny = {"emit_canvas_artifact"}
  set_current_skill("sniff-test")

  try:
    invoke_result = {
      "skill": "sniff-test",
      "content": "Do the sniff test.",
      _ACTIVE_SKILL_DENY_RESULT_KEY: ["emit_canvas_artifact"],
      _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY: {"fms_report_sniff_test": "sniff-test"},
    }
    stripped = runner._consume_private_tool_result_fields(invoke_result)
    assert stripped == {"skill": "sniff-test", "content": "Do the sniff test."}
    assert runner._active_skill_deny == {"emit_canvas_artifact"}
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
  runner._active_skill_deny = {"emit_canvas_artifact"}
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
    assert runner._active_skill_deny == {"emit_canvas_artifact"}
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
