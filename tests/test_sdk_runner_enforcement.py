from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))



from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog, SessionStore  # noqa: E402
from agent_gateway import sdk_runner_approval  # noqa: E402
from agent_gateway.approval_policy import ApprovalDecision as PolicyApprovalDecision, ApprovalRequest, ApprovalRequestPayload, ApprovalState, RunContext, sha256_args  # noqa: E402
from agent_gateway.approval_route import (
  DurableLocalApprovalRoute,
  NoApprovalRoute,
  ParentDelegatedApprovalRoute,
)
from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelChild,
)
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
from agent_gateway.skill_limits import ActiveSkillAdmission, SkillExecutionLimits  # noqa: E402
from agent_gateway.sdk_runner_stream import ToolCallInfo  # noqa: E402
from agent_gateway.mcp_client import RegisteredMcpRawPatchAuthorization  # noqa: E402
from agent_gateway.tool_dispatch_classification import ToolResultSettlement  # noqa: E402
from agent_gateway.tool_policy_registry import PlanDecision, PreparedToolCall  # noqa: E402
from gateway_test_support.sdk_capability_execution_test_support import stub_sdk_capability_execution  # noqa: E402


def _skill_admission(name: str) -> ActiveSkillAdmission:
  return ActiveSkillAdmission(
    name,
    SkillExecutionLimits(None, None, None),
  )


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
  module.__dict__.update({
    "__version__": SDK_PINNED_VERSION,
    "HookMatcher": _HookMatcher,
    "ClaudeAgentOptions": _ClaudeAgentOptions,
    "PermissionResultAllow": _PermissionResultAllow,
    "PermissionResultDeny": _PermissionResultDeny,
    "query": _query,
  })
  monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
  return state


def _make_runner(
  *,
  event_log: EventLog | None = None,
  disallowed_tools: list[str] | None = None,
  mcp_server_configs: dict[str, Any] | None = None,
  registered_mcp_descriptor_for_sdk_tool: Callable[[str], Any] | None = None,
  prepare_registered_mcp_tool_call_for_sdk_tool: Callable[..., Any] | None = None,
  registered_approval_overlay: Callable[..., bool] | None = None,
  redact_registered_mcp_tool_input_for_sdk_tool: Callable[..., Any] | None = None,
  settle_registered_mcp_tool_result_for_sdk_tool: Callable[..., Any] | None = None,
  on_tool_result: Any | None = None,
  run_context: RunContext | None = None,
  skill_run_id: str | None = None,
  max_tokens_override: int | None = None,
  api_key: str = "test-secret",
  approval_lifecycle: Literal["required", "not_required"] = "required",
) -> AgentSDKRunner:
  if (
    registered_mcp_descriptor_for_sdk_tool is not None
    and redact_registered_mcp_tool_input_for_sdk_tool is None
  ):
    redact_registered_mcp_tool_input_for_sdk_tool = _identity_registered_redaction
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
    registered_mcp_descriptor_for_sdk_tool=(
      registered_mcp_descriptor_for_sdk_tool
    ),
    prepare_registered_mcp_tool_call_for_sdk_tool=(
      prepare_registered_mcp_tool_call_for_sdk_tool
    ),
    registered_approval_overlay=registered_approval_overlay,
    redact_registered_mcp_tool_input_for_sdk_tool=(
      redact_registered_mcp_tool_input_for_sdk_tool
    ),
    settle_registered_mcp_tool_result_for_sdk_tool=(
      settle_registered_mcp_tool_result_for_sdk_tool
    ),
    on_tool_result=on_tool_result,
    run_context=run_context,
    skill_run_id=skill_run_id,
    max_tokens_override=max_tokens_override,
    approval_lifecycle=approval_lifecycle,
  )


def _registered_settlement(*_args: Any) -> ToolResultSettlement:
  return ToolResultSettlement("ok")


def _identity_registered_redaction(_tool_name: str, tool_input: dict[str, Any]):
  return dict(tool_input)


def _seed_tool_call(
  runner: AgentSDKRunner,
  tool_call_id: str,
  tool_name: str,
  tool_input: dict[str, Any],
) -> None:
  runner._pending_tool_calls[tool_call_id] = ToolCallInfo(
    tool_call_id,
    tool_name,
    tool_input,
    0.0,
    dict(tool_input),
  )


def test_sdk_runner_unconfigured_lifecycle_denies_without_explicit_opt_in(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """S8: an SDK run that reached no local ledger denies; it never fails open.

  `_make_runner()` supplies no route, so this is the un-configured path: the
  runner holds `NoApprovalRoute`, no store, no policy and no session.
  """

  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()

  assert isinstance(runner._approval_route, NoApprovalRoute)
  assert runner._approval_store is None
  assert runner._approval_policy is None

  denied = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))

  assert denied.behavior == "deny"
  assert "[approval_route_absent]" in denied.message


def test_sdk_runner_denies_on_a_route_whose_ledger_it_does_not_own(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """S8, the other half: a live-but-not-local route is still not an allow.

  The SDK runtime evaluates policy and writes the ledger in-process, so only a
  durable-local route configures it. A parent-delegated route is a real route
  this runtime cannot serve — the honest answer is a refusal, not an allow.
  """

  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()
  session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
  )
  runner._session = session
  runner._approval_route = ParentDelegatedApprovalRoute(
    object.__new__(AutonomousApprovalChannelChild),
    session,
  )

  denied = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))

  assert denied.behavior == "deny"
  assert "[approval_route_absent]" in denied.message
  assert runner._approval_store is None
  assert runner._approval_policy is None


def test_sdk_runner_explicit_not_required_lifecycle_allows_without_store(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner(approval_lifecycle="not_required")

  allowed = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))

  assert allowed.behavior == "allow"


def test_sdk_runner_requires_complete_registered_mcp_owners() -> None:
  with pytest.raises(ValueError, match="descriptor, preparation, redaction"):
    _make_runner(
      registered_mcp_descriptor_for_sdk_tool=lambda _tool_name: object(),
      settle_registered_mcp_tool_result_for_sdk_tool=_registered_settlement,
    )


def test_sdk_runner_requires_callable_registered_approval_overlay() -> None:
  with pytest.raises(TypeError, match="registered_approval_overlay"):
    _make_runner(registered_approval_overlay=object())  # type: ignore[arg-type]


def test_sdk_runner_registered_mode_builtin_callback_stays_catalogless(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)

  def descriptor_for(tool_name: str) -> Any:
    raise AssertionError(
      f"builtin {tool_name!r} must not resolve through the MCP manager"
    )

  def prepare_registered_call(tool_name: str, *_args: Any) -> Any:
    raise AssertionError(
      f"builtin {tool_name!r} must not prepare through the MCP manager"
    )

  runner = _make_runner(
    registered_mcp_descriptor_for_sdk_tool=descriptor_for,
    prepare_registered_mcp_tool_call_for_sdk_tool=prepare_registered_call,
    settle_registered_mcp_tool_result_for_sdk_tool=_registered_settlement,
    approval_lifecycle="not_required",
  )

  allowed = _run(
    runner._can_use_tool_callback("file_write", {"path": "x"}, None)
  )

  assert allowed.behavior == "allow"


def test_sdk_runner_registered_mode_sdk_local_callback_stays_catalogless(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  local_tool_id = "mcp__gateway-tools__load_tools"

  class LocalMcpConfig(dict[str, Any]):
    catalogless_mcp_tool_ids = {local_tool_id}

  def descriptor_for(tool_name: str) -> Any:
    raise AssertionError(
      f"SDK-local tool {tool_name!r} must not resolve through the MCP manager"
    )

  def prepare_registered_call(tool_name: str, *_args: Any) -> Any:
    raise AssertionError(
      f"SDK-local tool {tool_name!r} must not prepare through the MCP manager"
    )

  runner = _make_runner(
    mcp_server_configs=LocalMcpConfig({"gateway-tools": {"type": "sdk"}}),
    registered_mcp_descriptor_for_sdk_tool=descriptor_for,
    prepare_registered_mcp_tool_call_for_sdk_tool=prepare_registered_call,
    settle_registered_mcp_tool_result_for_sdk_tool=_registered_settlement,
    approval_lifecycle="not_required",
  )

  allowed = _run(
    runner._can_use_tool_callback(local_tool_id, {}, None)
  )

  assert allowed.behavior == "allow"


def test_sdk_registered_read_returns_manager_prepared_input_without_approval(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  identity = SimpleNamespace(
    logical_name="business_model_validate",
    materialize=lambda: {
      "route_kind": "mcp",
      "logical_server_id": "model-engine",
      "logical_name": "business_model_validate",
    },
  )
  descriptor = SimpleNamespace(
    identity=identity,
    declaration=SimpleNamespace(semantics=SimpleNamespace(effect="read")),
  )
  scopes: list[object] = []

  def prepare(
    _tool_name: str,
    _input: Any,
    scope: object,
    _overlay: object,
  ) -> Any:
    scopes.append(scope)
    return SimpleNamespace(
      descriptor=descriptor,
      prepared_call=PreparedToolCall({"ticker": "BRK.B"}),
      planning=PlanDecision("none"),
      prepared_authorization=None,
      approval_required=False,
      approval_reuse_key=None,
    )

  runner = _make_runner(
    registered_mcp_descriptor_for_sdk_tool=lambda _tool_name: descriptor,
    prepare_registered_mcp_tool_call_for_sdk_tool=prepare,
    settle_registered_mcp_tool_result_for_sdk_tool=_registered_settlement,
  )
  runner._session = SimpleNamespace(
    dispatch_scope={"portfolio_id": "portfolio-1"}
  )

  allowed = _run(runner._can_use_tool_callback(
    "mcp__model-engine__business_model_validate",
    {"ticker": "BRK/B"},
    None,
  ))

  assert allowed.behavior == "allow"
  assert allowed.updated_input == {"ticker": "BRK.B"}
  assert scopes == [{
    "portfolio_id": "portfolio-1",
    "user_id": "alice",
  }]


def test_sdk_registered_not_required_lifecycle_returns_prepared_input(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  descriptor = SimpleNamespace(
    identity=SimpleNamespace(logical_name="manage_proxy_cache"),
    declaration=SimpleNamespace(
      semantics=SimpleNamespace(effect="portfolio_config")
    ),
  )
  runner = _make_runner(
    registered_mcp_descriptor_for_sdk_tool=lambda _tool_name: descriptor,
    prepare_registered_mcp_tool_call_for_sdk_tool=(
      lambda _tool_name, _input, _scope, _overlay: SimpleNamespace(
        descriptor=descriptor,
        prepared_call=PreparedToolCall({"action": "invalidate"}),
        planning=PlanDecision("none"),
        prepared_authorization=None,
        approval_required=True,
        approval_reuse_key=None,
      )
    ),
    settle_registered_mcp_tool_result_for_sdk_tool=_registered_settlement,
    approval_lifecycle="not_required",
  )
  runner._session = SimpleNamespace(dispatch_scope=None)

  allowed = _run(runner._can_use_tool_callback(
    "mcp__portfolio-config-mcp__manage_proxy_cache",
    {"action": "invalidate", "ignored": True},
    None,
  ))

  assert allowed.behavior == "allow"
  assert allowed.updated_input == {"action": "invalidate"}


def test_sdk_registered_write_uses_exact_reuse_and_prepared_identity(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  identity = SimpleNamespace(
    logical_name="manage_proxy_cache",
  )
  descriptor = SimpleNamespace(
    identity=identity,
    declaration=SimpleNamespace(
      semantics=SimpleNamespace(effect="portfolio_config")
    ),
  )
  captured: dict[str, Any] = {}
  lifecycle_input = {"action": "invalidate"}

  async def lifecycle(**kwargs: Any) -> dict[str, Any]:
    captured.update(kwargs)
    return {
      "approved": True,
      "tool_input": lifecycle_input,
      "policy_modified_tool_args": True,
    }

  monkeypatch.setattr(
    sdk_runner_approval._approval_lifecycle_helpers,
    "run_approval_lifecycle",
    lifecycle,
  )
  runner = _make_runner(
    registered_mcp_descriptor_for_sdk_tool=lambda _tool_name: descriptor,
    prepare_registered_mcp_tool_call_for_sdk_tool=(
      lambda _tool_name, _input, _scope, _overlay: SimpleNamespace(
        descriptor=descriptor,
        prepared_call=PreparedToolCall({"action": "invalidate"}),
        planning=PlanDecision("none"),
        prepared_authorization=None,
        approval_required=True,
        approval_reuse_key="tool-approval-cache:v1:sha256:exact",
      )
    ),
    settle_registered_mcp_tool_result_for_sdk_tool=_registered_settlement,
  )
  runner._session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
  )
  runner._approval_route = DurableLocalApprovalRoute(
    object(),
    object(),
    runner._session,
  )

  allowed = _run(runner._can_use_tool_callback(
    "mcp__portfolio-config-mcp__manage_proxy_cache",
    {"action": "invalidate", "ignored": True},
    None,
  ))

  assert allowed.behavior == "allow"
  assert allowed.updated_input == lifecycle_input
  assert captured["tool_input"] == {"action": "invalidate"}
  assert captured["approval_reuse_mode"] == "exact"
  assert captured["approval_reuse_key"] == (
    "tool-approval-cache:v1:sha256:exact"
  )


def test_sdk_registered_plan_hands_one_durable_payload_to_same_authorized_ref(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  descriptor = SimpleNamespace(
    identity=SimpleNamespace(logical_name="apply_patch_ops"),
    declaration=SimpleNamespace(
      semantics=SimpleNamespace(effect="state_write")
    ),
  )
  prepared_input = {"research_file_id": 7, "ops": []}
  prepared_authorization = RegisteredMcpRawPatchAuthorization(
    approval_identity={"identity_source": "reviewed_change_binding"},
    approval_arguments=prepared_input,
    approval_arguments_hash=sha256_args(prepared_input),
    prepared_payload=b'{"prepared":true}',
  )
  captured: dict[str, Any] = {}
  authorized_ids: list[str] = []

  async def lifecycle(**kwargs: Any) -> dict[str, Any]:
    captured.update(kwargs)
    return {
      "approved": True,
      "tool_input": {**prepared_input, "ops": [{"forged": True}]},
    }

  def materialize_authorized_input(
    tool_call_id: str,
  ) -> dict[str, object]:
    authorized_ids.append(tool_call_id)
    return {
      **prepared_input,
      "authorization_ref": f"approval-ref:v1:{tool_call_id}",
    }

  monkeypatch.setattr(
    sdk_runner_approval._approval_lifecycle_helpers,
    "run_approval_lifecycle",
    lifecycle,
  )
  runner = _make_runner(
    registered_mcp_descriptor_for_sdk_tool=lambda _tool_name: descriptor,
    prepare_registered_mcp_tool_call_for_sdk_tool=(
      lambda _tool_name, _input, _scope, _overlay: SimpleNamespace(
        descriptor=descriptor,
        prepared_call=PreparedToolCall(prepared_input),
        planning=PlanDecision("prepared_plan", prepared_plan={"plan": 1}),
        prepared_authorization=prepared_authorization,
        approval_required=True,
        approval_reuse_key="tool-approval-cache:v1:sha256:plan",
        materialize_authorized_input=materialize_authorized_input,
      )
    ),
    settle_registered_mcp_tool_result_for_sdk_tool=_registered_settlement,
  )
  runner._session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
  )
  runner._approval_route = DurableLocalApprovalRoute(
    object(),
    object(),
    runner._session,
  )

  allowed = _run(runner._can_use_tool_callback(
    "mcp__portfolio-writes-mcp__apply_patch_ops",
    prepared_input,
    None,
  ))

  assert allowed.behavior == "allow"
  assert len(authorized_ids) == 1
  assert captured["tool_call_id"] == authorized_ids[0]
  assert captured["approval_identity"] == {
    "identity_source": "reviewed_change_binding"
  }
  assert captured["prepared_authorization_payload"] == b'{"prepared":true}'
  assert captured["approval_args_hash"] == sha256_args(prepared_input)
  assert allowed.updated_input["ops"] == []
  assert allowed.updated_input["authorization_ref"] == (
    f"approval-ref:v1:{authorized_ids[0]}"
  )


def test_sdk_post_tool_use_does_not_reproject_a_settled_result() -> None:
  # The result belongs to the tool that produced it; the hook only rewrites the
  # model's copy for the gateway's own signals, never to re-scan a payload.
  runner = _make_runner(api_key="CUSTOM-ACTIVE-CREDENTIAL-CODEX-SDK-8f21d7")
  _seed_tool_call(
    runner,
    "tool-plain",
    "lookup",
    {"query": "ordinary"},
  )

  hook_result = _run(
    runner._post_tool_use_hook(
      {
        "tool_name": "lookup",
        "tool_input": {"query": "ordinary"},
        "result": json.dumps({"status": "ok", "answer": 42}),
      },
      "tool-plain",
      None,
    )
  )

  assert hook_result == {}


def test_sdk_post_tool_failure_blocks_raw_secret_from_model_continuation() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-CODEX-SDK-ERROR-8f21d7"
  runner = _make_runner(api_key=secret)
  _seed_tool_call(
    runner,
    "tool-secret-error",
    "lookup",
    {"query": "ordinary"},
  )

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
    }),
    approval_lifecycle="not_required",
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
  runner = _make_runner(
    disallowed_tools=["file_write"],
    approval_lifecycle="not_required",
  )

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
    approval_lifecycle="not_required",
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
  from agent_gateway import policy_imports

  monkeypatch.setattr(
    policy_imports,
    "_server_policy",
    SimpleNamespace(
      get_forbidden_tools_for_session=lambda _session: frozenset(),
      get_server_for_policy_tool=lambda name: (
        "portfolio-trades-mcp" if name == "execute_trade" else None
      ),
    ),
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






def test_sdk_approval_refuses_inline_limit_mismatch_before_policy(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  policy_calls: list[bool] = []

  class Policy:
    async def decide(self, **_kwargs: Any) -> PolicyApprovalDecision:
      policy_calls.append(True)
      raise AssertionError("mismatched admission must precede policy")

  runner = _make_runner(
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      skill="quant-research",
      admitted_skill_execution_limits=SkillExecutionLimits(
        20,
        32_000,
        20.0,
      ),
    )
  )
  runner._session = SessionStore(ttl=3600).create_session(
    api_key_hash="hash",
    user_id="alice",
  )
  runner._approval_route = DurableLocalApprovalRoute(
    object(),
    Policy(),
    runner._session,
  )
  token = set_current_skill(ActiveSkillAdmission(
    "quant-research",
    SkillExecutionLimits(19, 32_000, 20.0),
  ))
  try:
    denied = _run(
      runner._can_use_tool_callback("file_write", {"path": "x"}, None)
    )
  finally:
    from agent_gateway.skill_context import reset_current_skill

    reset_current_skill(token)

  assert denied.behavior == "deny"
  assert "[skill_admission_mismatch]" in denied.message
  assert policy_calls == []


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
      approval_route=DurableLocalApprovalRoute(
        store,
        policy,
        session,
      ),
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
      if callback_task.done():
        await callback_task
      await asyncio.sleep(0.001)
    else:
      raise AssertionError("approval request was not queued")

    tool_call_id, pending = next(iter(session.pending_tools.items()))
    approval_events = [
      entry.event
      for entry in runner._log.entries
      if entry.event.get("type") == "tool_approval_request"
    ]
    assert len(approval_events) == 1
    assert approval_events[0]["tool_call_id"] == tool_call_id
    notification_rows = await store.list_approval_notification_outbox(
      str(pending["approval_id"])
    )
    assert len(notification_rows) == 1
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
      approval_route=DurableLocalApprovalRoute(
        store,
        policy,
        session,
      ),
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


@pytest.mark.parametrize(
  ("winner_state", "approved", "expected_behavior"),
  [
    ("approved", True, "allow"),
    ("denied", False, "deny"),
  ],
)
def test_sdk_runner_approval_timeout_uses_shared_durable_winner(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  winner_state: ApprovalState,
  approved: bool,
  expected_behavior: str,
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
    ) -> PolicyApprovalDecision:
      _ = payload, request, run_context
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="Tool requires approval",
        expiry_seconds=0.01,
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

  async def _case() -> None:
    store = SQLiteApprovalStore(tmp_path / f"winner-{winner_state}.sqlite3")
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
      approval_route=DurableLocalApprovalRoute(
        store,
        _Policy(),
        session,
      ),
    )

    callback_task = asyncio.create_task(
      runner._can_use_tool_callback("file_write", {"path": "x"}, None)
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
    await store.transition_state(
      request.approval_id,
      winner_state,
      expected_state_version=request.state_version,
      decider_id="alice",
      decider_role="owner",
      decision_reason="race winner",
    )
    await asyncio.sleep(0.12)
    session.approval_queues[tool_call_id].put_nowait({
      "approval_id": request.approval_id,
      "approved": approved,
      "allow_tool_type": False,
    })

    result = await callback_task
    assert result.behavior == expected_behavior
    assert not getattr(result, "interrupt", False)
    assert session.pending_tools == {}
    assert session.approval_queues == {}

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
      approval_route=DurableLocalApprovalRoute(
        store,
        policy,
        session,
      ),
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
      resolved_qualifier: str,
      allow_persistent: bool,
      timeout_seconds: float,
      batch_admission: Any | None = None,
    ) -> None:
      _ = (
        request,
        decision,
        nonce,
        resolved_qualifier,
        allow_persistent,
        timeout_seconds,
        batch_admission,
      )
      pending_committed.set()
      await asyncio.Event().wait()

    runner._await_user_approval_via_pending_tools = pause_before_pending_publish
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
    assert await store.list_approval_notification_outbox(
      policy.request.approval_id
    ) == []
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
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    ),
    capability_execution=stub_sdk_capability_execution(),
    system_prompt="test",
    session=session,
    approval_route=DurableLocalApprovalRoute(
      store,
      _Policy(),
      session,
    ),
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
    approval_route=DurableLocalApprovalRoute(
      store,
      policy,
      session,
    ),
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
    identity = SimpleNamespace(
      logical_name="execute_trade",
      materialize=lambda: {
        "route_kind": "mcp",
        "logical_server_id": "portfolio-trades-mcp",
        "logical_name": "execute_trade",
      },
    )
    descriptor = SimpleNamespace(
      identity=identity,
      declaration=SimpleNamespace(
        semantics=SimpleNamespace(effect="irreversible")
      ),
    )
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
      approval_route=DurableLocalApprovalRoute(
        store,
        policy,
        session,
      ),
      registered_mcp_descriptor_for_sdk_tool=(
        lambda _tool_name: descriptor
      ),
      prepare_registered_mcp_tool_call_for_sdk_tool=(
        lambda _tool_name, tool_input, _scope, _overlay: SimpleNamespace(
          descriptor=descriptor,
          prepared_call=PreparedToolCall(tool_input),
          planning=PlanDecision("none"),
          prepared_authorization=None,
          approval_required=True,
          approval_reuse_key=None,
        )
      ),
      redact_registered_mcp_tool_input_for_sdk_tool=(
        lambda _tool_name, tool_input: dict(tool_input)
      ),
      settle_registered_mcp_tool_result_for_sdk_tool=(
        _registered_settlement
      ),
      run_context=RunContext(
        user_id="alice",
        request_id="request-1",
        session_id=session.session_id,
        channel="web",
      ),
    )

    callback_task = asyncio.create_task(
      runner._can_use_tool_callback(
        "mcp__portfolio-trades-mcp__provider_execute_trade",
        {"preview_id": "p1"},
        None,
      )
    )
    for _ in range(100):
      if session.pending_tools:
        break
      if callback_task.done():
        await callback_task
      await asyncio.sleep(0.001)
    else:
      raise AssertionError("approval request was not queued")

    tool_call_id, pending = next(iter(session.pending_tools.items()))
    request = await store.get(str(pending["approval_id"]))
    assert request is not None
    assert request.tool_name == "execute_trade"
    assert request.tool_class == "irreversible"
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
    approval_lifecycle="not_required",
  )
  result = {
    "skill": "phase0-agent",
    "content": "Do the work.",
    _ACTIVE_SKILL_ALLOW_RESULT_KEY: ["start_investment_run"],
    _ACTIVE_SKILL_DENY_RESULT_KEY: ["file_write"],
  }
  _seed_tool_call(
    runner,
    "tool-invoke",
    "invoke_skill",
    {"skill_name": "phase0-agent"},
  )

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

  assert hook_result["continue_"] is False
  assert hook_result["hookSpecificOutput"]["updatedMCPToolOutput"] == {
    "skill": "phase0-agent",
    "content": "Do the work.",
  }
  assert _ACTIVE_SKILL_ALLOW_RESULT_KEY not in json.dumps(hook_result)
  assert _ACTIVE_SKILL_DENY_RESULT_KEY not in json.dumps(hook_result)
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
  runner = _make_runner(
    disallowed_tools=["run_bash"],
    approval_lifecycle="not_required",
  )

  result = {"skill": "legacy-agent", "content": "Legacy skill body."}
  _seed_tool_call(
    runner,
    "tool-invoke",
    "invoke_skill",
    {"skill_name": "legacy-agent"},
  )
  hook_result = _run(
    runner._post_tool_use_hook(
      {
        "tool_name": "invoke_skill",
        "tool_input": {"skill_name": "legacy-agent"},
        "result": json.dumps(result),
      },
      "tool-invoke",
      None,
    )
  )

  assert hook_result == {}
  assert runner._active_skill_deny == set()

  static_denied = _run(runner._can_use_tool_callback("run_bash", {"command": "date"}, None))
  assert static_denied.behavior == "deny"
  assert static_denied.message == "Tool 'run_bash' is not available in this context"

  dynamic_allowed = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))
  assert dynamic_allowed.behavior == "allow"


def test_sdk_runner_foreign_result_cannot_activate_active_skill_gate(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner(
    disallowed_tools=["start_investment_run"],
    approval_lifecycle="not_required",
  )
  _seed_tool_call(
    runner,
    "tool-foreign",
    "mcp__foreign__lookup",
    {},
  )

  hook_result = _run(
    runner._post_tool_use_hook(
      {
        "tool_name": "mcp__foreign__lookup",
        "tool_input": {},
        "result": json.dumps({
          "status": "ok",
          _ACTIVE_SKILL_ALLOW_RESULT_KEY: ["start_investment_run"],
          _ACTIVE_SKILL_DENY_RESULT_KEY: ["file_write"],
          _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY: {
            "fms_report_sniff_test": "sniff-test",
          },
        }),
      },
      "tool-foreign",
      None,
    )
  )

  assert runner._active_skill_allow == set()
  assert runner._active_skill_deny == set()
  assert runner._active_skill_report_doors == {}
  assert hook_result["hookSpecificOutput"]["updatedMCPToolOutput"] == {
    "status": "ok",
  }
  assert "continue_" not in hook_result


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
  set_current_skill(_skill_admission("sniff-test"))

  try:
    invoke_result = {
      "skill": "sniff-test",
      "content": "Do the sniff test.",
      _ACTIVE_SKILL_DENY_RESULT_KEY: ["emit_canvas_artifact"],
      _ACTIVE_SKILL_REPORT_DOORS_RESULT_KEY: {"fms_report_sniff_test": "sniff-test"},
    }
    stripped = runner._consume_private_tool_result_fields(
      invoke_result,
      tool_name="invoke_skill",
    )
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
  runner = _make_runner(approval_lifecycle="not_required")
  runner._active_skill_deny = {"build_model"}
  runner._active_skill_report_doors = {"fms_persist_business_model": "business-model-construction"}
  set_current_skill(_skill_admission("business-model-construction"))

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
  set_current_skill(_skill_admission("business-model-construction"))

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
  set_current_skill(_skill_admission("sniff-test"))

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
    set_current_skill(_skill_admission("phase0-agent"))
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
    set_current_skill(_skill_admission("phase0-agent"))
    with pytest.raises(RuntimeError, match="sdk failed"):
      await error_runner.run([{"role": "user", "content": "hello"}])
    assert current_skill() is None

  _run(_run_error())
  assert error_runner._active_skill_deny == set()
