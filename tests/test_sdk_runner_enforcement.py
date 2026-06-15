from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog, SessionStore
from agent_gateway.approval_policy import ApprovalDecision as PolicyApprovalDecision, ApprovalRequest, ApprovalRequestPayload, RunContext
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.approvals import _record_vote_and_unblock
from agent_gateway.providers.agent_sdk import SDK_PINNED_VERSION
from agent_gateway.runner import _ACTIVE_SKILL_DENY_RESULT_KEY, _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY
from agent_gateway.skill_context import clear_current_skill, current_skill, set_current_skill


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
