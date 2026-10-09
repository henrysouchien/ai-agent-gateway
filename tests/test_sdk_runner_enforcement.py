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
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))



from agent_gateway import AgentSDKConfig, AgentSDKRunner, EventLog, SessionStore  # noqa: E402
from agent_gateway import sdk_runner_approval  # noqa: E402
from agent_gateway.admission import (
  admit_root_turn,
  capture_root_turn,
  record_root_turn_admission,
  root_context,
)
from agent_gateway.approval_policy import ApprovalDecision as PolicyApprovalDecision, ApprovalRequest, ApprovalRequestPayload, ApprovalState, RunContext, sha256_args  # noqa: E402
from agent_gateway.approval_route import (
  DurableLocalApprovalRoute,
  NoApprovalRoute,
  ParentDelegatedApprovalRoute,
  bind_session_approval_route,
)
from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelChild,
)
from agent_gateway.approval_store import SQLiteApprovalStore  # noqa: E402
from agent_gateway.approvals import ApprovalActionError, _record_vote_and_unblock  # noqa: E402
from agent_gateway.batch_approval_projection import (  # noqa: E402
  BatchApprovalProjectionRegistry,
  BatchApprovalScope,
)
from agent_gateway.providers.agent_sdk import SDK_PINNED_VERSION  # noqa: E402
from agent_gateway.runner import _ACTIVE_SKILL_ALLOW_RESULT_KEY  # noqa: E402
from agent_gateway.skill_context import clear_current_skill, current_skill, set_current_skill  # noqa: E402
from agent_gateway.skill_limits import ActiveSkillAdmission, SkillExecutionLimits  # noqa: E402
from agent_gateway.single_user_policy import SingleUserApprovalPolicy  # noqa: E402
from agent_gateway.sdk_runner_stream import ToolCallInfo  # noqa: E402
from agent_gateway.sdk_runner import _SDKSegmentQueryIterator  # noqa: E402
from agent_gateway.mcp_client import RegisteredMcpRawPatchAuthorization  # noqa: E402
from agent_gateway.tool_dispatch_classification import ToolResultSettlement  # noqa: E402
from agent_gateway.tool_dispatcher_approval_lifecycle import ApprovalPark  # noqa: E402
from agent_gateway.tool_policy_registry import PlanDecision, PreparedToolCall  # noqa: E402
from gateway_test_support.sdk_capability_execution_test_support import stub_sdk_capability_execution  # noqa: E402


def _skill_admission(name: str) -> ActiveSkillAdmission:
  return ActiveSkillAdmission(
    name,
    SkillExecutionLimits(None),
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
  state = types.SimpleNamespace(options=[], prompts=[], clients=[])

  class _ClaudeSDKClient:
    def __init__(self, *, options: _ClaudeAgentOptions) -> None:
      self.options = options
      self.iterator: Any | None = None
      self.connected = False
      self.disconnected = False
      state.clients.append(self)

    async def connect(self) -> None:
      self.connected = True

    async def query(self, prompt: Any, session_id: str = "default") -> None:
      state.prompts.append(prompt)
      state.options.append(self.options)
      self.iterator = (
        iterator_factory(prompt, self.options)
        if iterator_factory is not None
        else _AsyncMessages([_sdk_result_message()])
      )

    def receive_response(self) -> Any:
      assert self.iterator is not None
      return self.iterator

    async def disconnect(self) -> None:
      if self.disconnected:
        return
      if self.iterator is not None:
        await self.iterator.aclose()
      self.disconnected = True

  state.client_type = _ClaudeSDKClient

  module = types.ModuleType("claude_agent_sdk")
  module.__dict__.update({
    "__version__": SDK_PINNED_VERSION,
    "HookMatcher": _HookMatcher,
    "ClaudeAgentOptions": _ClaudeAgentOptions,
    "PermissionResultAllow": _PermissionResultAllow,
    "PermissionResultDeny": _PermissionResultDeny,
    "ClaudeSDKClient": _ClaudeSDKClient,
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
  redact_registered_mcp_tool_input_for_sdk_tool: Callable[..., Any] | None = None,
  settle_registered_mcp_tool_result_for_sdk_tool: Callable[..., Any] | None = None,
  on_tool_result: Any | None = None,
  run_context: RunContext | None = None,
  skill_run_id: str | None = None,
  max_tokens_override: int | None = None,
  api_key: str = "test-secret",
  approval_store_path: Path | None = None,
) -> AgentSDKRunner:
  if (
    registered_mcp_descriptor_for_sdk_tool is not None
    and redact_registered_mcp_tool_input_for_sdk_tool is None
  ):
    redact_registered_mcp_tool_input_for_sdk_tool = _identity_registered_redaction
  session = None
  approval_route = None
  if approval_store_path is not None:
    # Only a live order asks; every class these tests call is approved by the
    # real policy through the durable-local ledger.
    store = SQLiteApprovalStore(approval_store_path)
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
    )
    approval_route = DurableLocalApprovalRoute(
      store,
      SingleUserApprovalPolicy(store=store),
      session,
    )
  route_kwargs: dict[str, Any] = (
    {}
    if approval_route is None
    else {"session": session, "approval_route": approval_route}
  )
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
    **route_kwargs,
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


def test_sdk_runner_without_a_decider_refuses_only_a_live_order(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """GA23: with no route, a non-money call is settled; a live order is refused.

  `_make_runner()` supplies no route, so the runner holds `NoApprovalRoute`,
  no store, no policy and no session.
  """

  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner()

  assert isinstance(runner._approval_route, NoApprovalRoute)
  assert runner._approval_store is None
  assert runner._approval_policy is None

  allowed = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))
  assert allowed.behavior == "allow"

  monkeypatch.setattr(
    runner,
    "_resolve_sdk_approval_identity",
    lambda _tool_name: ("execute_trade", "irreversible"),
  )
  denied = _run(runner._can_use_tool_callback("execute_trade", {"ticker": "MSFT"}, None))

  assert denied.behavior == "deny"
  assert "[approval_route_absent]" in denied.message


def test_sdk_runner_parent_delegated_non_money_call_reaches_the_class_check(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A catalogless non-money call on a parent-delegated route is settled.

  The delegated lifecycle applies the class check: everything outside the
  money boundary runs under the run owner's authority with no card.
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

  allowed = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))

  assert allowed.behavior == "allow"
  assert runner._approval_store is None
  assert runner._approval_policy is None


def test_sdk_runner_requires_complete_registered_mcp_owners() -> None:
  with pytest.raises(ValueError, match="descriptor, preparation, redaction"):
    _make_runner(
      registered_mcp_descriptor_for_sdk_tool=lambda _tool_name: object(),
      settle_registered_mcp_tool_result_for_sdk_tool=_registered_settlement,
    )


def test_sdk_runner_registered_mode_builtin_callback_stays_catalogless(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
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
    approval_store_path=tmp_path / "approvals.sqlite3",
  )

  allowed = _run(
    runner._can_use_tool_callback("file_write", {"path": "x"}, None)
  )

  assert allowed.behavior == "allow"


def test_sdk_runner_registered_mode_sdk_local_callback_stays_catalogless(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
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
    approval_store_path=tmp_path / "approvals.sqlite3",
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
      lambda _tool_name, _input, _scope: SimpleNamespace(
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
      lambda _tool_name, _input, _scope: SimpleNamespace(
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
  tmp_path: Path,
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
    approval_store_path=tmp_path / "approvals.sqlite3",
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


def test_sdk_runner_static_disallowed_tool_denied_without_approval(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  state = _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner(
    disallowed_tools=["file_write"],
    approval_store_path=tmp_path / "approvals.sqlite3",
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
  tmp_path: Path,
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
    approval_store_path=tmp_path / "approvals.sqlite3",
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


@pytest.mark.parametrize(
  "ending",
  ["normal", "eof", "transport_error", "cancel", "disconnect", "rebuild", "failed_result"],
)
@pytest.mark.parametrize("root_capture", ["absent", "current", "superseded"])
def test_sdk_runner_transport_end_drains_detached_permission_before_release(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  ending: str,
  root_capture: str,
) -> None:
  class _Policy:
    policy_id = "sdk-detached-permission-policy"
    policy_version = "1"
    policy_bundle_hash = "test-policy"

    def __init__(self) -> None:
      self.decision_started = asyncio.Event()
      self.release_decision = asyncio.Event()

    async def decide(
      self,
      *,
      payload: ApprovalRequestPayload,
      request: ApprovalRequest,
      run_context: RunContext,
    ) -> PolicyApprovalDecision:
      if root_capture == "superseded" and not self.decision_started.is_set():
        self.decision_started.set()
        await self.release_decision.wait()
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="Tool requires approval",
        expiry_seconds=600,
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      pass

    def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
      return True

  async def _case() -> None:
    published = asyncio.Event()
    replacement_published = asyncio.Event()
    publication_count = 0
    receiving = asyncio.Event()
    transport_end = asyncio.Event()
    handler_cancelled = asyncio.Event()
    finish_handler = asyncio.Event()
    permission_results: list[Any] = []
    transport_releases: list[tuple[bool, bool]] = []
    summaries: list[bool] = []
    transport_error = RuntimeError("transport failed")
    store = SQLiteApprovalStore(tmp_path / f"lifetime-{ending}.sqlite3")
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
    )
    policy = _Policy()
    execution = stub_sdk_capability_execution()
    route = bind_session_approval_route(session, store, policy)

    async def record_turn():
      context = root_context(
        session, withheld_tools=frozenset(), research_file_id=None,
      )
      turn = await record_root_turn_admission(store, admit_root_turn(
        context,
        execution=execution,
        tool_grant={
          "tool_ids": context.tool_ids,
          "withheld_tools": sorted(context.withheld),
        },
        approval_route=route,
      ))
      capture_root_turn(session, turn)
      return turn

    root_a = await record_turn() if root_capture != "absent" else None
    root_b = None

    class _ApprovalLog(EventLog):
      def append(self, event: dict[str, Any]) -> Any:
        nonlocal publication_count
        entry = super().append(event)
        if event.get("type") == "tool_approval_request":
          published.set()
          publication_count += 1
          if publication_count == 2:
            replacement_published.set()
        return entry

    class _DetachedPermissionIterator(_AsyncMessages):
      def __init__(self, options: _ClaudeAgentOptions) -> None:
        super().__init__([])
        self.options = options
        self.segment = len(iterators)
        self.handler: asyncio.Task[Any] | None = None

      async def __anext__(self) -> Any:
        async def permission_handler() -> None:
          try:
            result = await self.options.kwargs["can_use_tool"](
              "file_write",
              {"path": "x"},
              None,
            )
            permission_results.append(result)
          except asyncio.CancelledError:
            handler_cancelled.set()
            await finish_handler.wait()
            raise

        self.handler = asyncio.create_task(permission_handler())
        await published.wait()
        receiving.set()
        await transport_end.wait()
        if ending == "rebuild":
          if self.segment == 0:
            runner._disallowed_tools.append("Read")
          else:
            await self.handler
          result_message = _sdk_result_message()
          result_message.session_id = "sdk-lifetime-session"
          return result_message
        if ending == "normal":
          return _sdk_result_message()
        if ending == "failed_result":
          result_message = _sdk_result_message()
          result_message.subtype = "error_during_execution"
          result_message.is_error = True
          result_message.result = "provider failed"
          return result_message
        if ending == "transport_error":
          raise transport_error
        raise StopAsyncIteration

      async def aclose(self) -> None:
        assert self.handler is not None
        transport_releases.append((
          self.handler.done(),
          not session.pending_tools and not session.approval_queues,
        ))
        await super().aclose()
        transport_end.set()

    iterators: list[_DetachedPermissionIterator] = []

    def query(_prompt: Any, options: _ClaudeAgentOptions) -> _DetachedPermissionIterator:
      iterator = _DetachedPermissionIterator(options)
      iterators.append(iterator)
      return iterator

    _install_fake_agent_sdk(monkeypatch, query)
    event_log = _ApprovalLog()
    runner = AgentSDKRunner(
      event_log=event_log,
      session_id=session.session_id,
      sdk_config=AgentSDKConfig(
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      capability_execution=execution,
      system_prompt="test",
      session=session,
      approval_route=route,
      on_session_summary=lambda _summary: summaries.append(
        all(iterator.handler is not None and iterator.handler.done() for iterator in iterators)
      ),
    )
    run_task = asyncio.create_task(
      runner.run(messages=[{"role": "user", "content": "write"}])
    )
    disconnects: list[asyncio.Task[None]] = []
    new_call_id: str | None = None
    new_pending: dict[str, Any] | None = None
    try:
      if root_capture == "superseded":
        await asyncio.wait_for(policy.decision_started.wait(), timeout=5)
        root_b = await record_turn()
        assert root_a is not None
        assert root_b.turn_id != root_a.turn_id
        policy.release_decision.set()
      await asyncio.wait_for(receiving.wait(), timeout=5)
      tool_call_id, pending = next(iter(session.pending_tools.items()))
      approval_id = str(pending["approval_id"])
      nonce = pending["nonce"]
      approval_event = next(
        entry.event for entry in event_log.entries
        if entry.event.get("type") == "tool_approval_request"
        and entry.event["tool_call_id"] == tool_call_id
      )
      assert approval_event["approval_id"] == approval_id
      assert approval_event["nonce"] == nonce
      if root_a is not None:
        assert session.owner_user_id is not None
        assert await SQLiteApprovalStore(store.path)._locate_root_turn_for_approval(
          tenant_storage_id=session.owner_user_id, approval_id=approval_id,
        ) == (session.session_id, tool_call_id, nonce, root_a.turn_id)
      if ending == "cancel":
        run_task.cancel()
      elif ending == "disconnect":
        disconnects = [
          asyncio.create_task(runner.on_disconnect()),
          asyncio.create_task(runner.on_disconnect()),
        ]
      else:
        transport_end.set()
      await asyncio.wait_for(handler_cancelled.wait(), timeout=5)
      assert not run_task.done()
      assert all(not task.done() for task in disconnects)
      assert transport_releases == []
      assert summaries == []
      assert session.pending_tools == {}
      assert session.approval_queues == {}
      assert permission_results == []
      if ending == "rebuild":
        assert len(iterators) == 1
      if ending == "disconnect":
        run_task.cancel()
      finish_handler.set()
      if ending == "rebuild":
        await asyncio.wait_for(replacement_published.wait(), timeout=5)
        new_call_id, new_pending = next(iter(session.pending_tools.items()))
        assert new_call_id != tool_call_id
        await _record_vote_and_unblock(
          target_session=session,
          pending_entry=new_pending,
          tool_call_id=new_call_id,
          nonce=new_pending["nonce"],
          decider_id="alice",
          decider_role="owner",
          approved=True,
          allow_tool_type=False,
          reason=None,
          app_state=SimpleNamespace(
            gateway_approval_store=store,
            gateway_approval_policy=policy,
          ),
        )
      if disconnects:
        await asyncio.gather(*disconnects)
      if root_a is not None and ending != "failed_result":
        with pytest.raises(ApprovalPark) as parked:
          await asyncio.wait_for(run_task, timeout=5)
        assert type(parked.value) is ApprovalPark
        assert (
          parked.value.job_id,
          parked.value.approval_id,
          parked.value.original_tool_call_id,
          parked.value.nonce,
          parked.value.root_turn_id,
        ) == (
          session.session_id, approval_id, tool_call_id, nonce, root_a.turn_id,
        )
        if ending == "transport_error":
          assert parked.value.__cause__ is transport_error
      elif ending in {"cancel", "disconnect"}:
        with pytest.raises(asyncio.CancelledError):
          await asyncio.wait_for(run_task, timeout=5)
      elif ending in {"normal", "rebuild"}:
        await asyncio.wait_for(run_task, timeout=5)
      else:
        with pytest.raises(RuntimeError) as raised:
          await asyncio.wait_for(run_task, timeout=5)
        if ending == "transport_error":
          assert raised.value is transport_error
        if ending == "failed_result":
          assert str(raised.value) == "agent_sdk_result_error: provider failed"
      assert iterators[0].handler is not None
      assert iterators[0].handler.cancelled()
      assert transport_releases == [(True, True)] * (2 if ending == "rebuild" else 1)
      assert summaries == [True]
      if ending == "rebuild":
        assert len(permission_results) == 1
        assert permission_results[0].behavior == "allow"
        assert iterators[1].handler is not None
        assert not iterators[1].handler.cancelled()
      else:
        assert permission_results == []
      stored = await store.get(approval_id)
      assert stored is not None
      assert stored.approval_id == approval_id
      assert stored.state == "pending_user"
      assert stored.tool_call_id == tool_call_id
      if root_a is not None:
        terminal_events = [
          entry.event for entry in event_log.entries
          if entry.event.get("type") == "stream_complete"
        ]
        if ending == "failed_result":
          assert terminal_events == []
          assert [
            entry.event["error"] for entry in event_log.entries
            if entry.event.get("type") == "error"
          ] == ["agent_sdk_result_error: provider failed"]
        else:
          assert len(terminal_events) == 1
          terminal = terminal_events[0]
          assert terminal["terminal_disposition"] == "awaiting_approval"
          assert {key: terminal[key] for key in (
            "job_id", "approval_id", "original_tool_call_id", "root_turn_id",
          )} == {
            "job_id": session.session_id,
            "approval_id": approval_id,
            "original_tool_call_id": tool_call_id,
            "root_turn_id": root_a.turn_id,
          }
          assert "nonce" not in terminal
          assert nonce not in json.dumps(terminal)
        assert session.owner_user_id is not None
        reopened = SQLiteApprovalStore(store.path)
        located = await reopened._locate_request_for_decision(
          tenant_storage_id=session.owner_user_id,
          approval_id=approval_id,
        )
        assert located is not None
        durable_request, locator, original_nonce = located
        assert durable_request.approval_id == approval_id
        assert durable_request.state == "pending_user"
        assert original_nonce == nonce
        assert locator.job_id == session.session_id
        assert locator.root_turn_id == root_a.turn_id
        assert locator.executable_binding_state == "not_restorable"
        assert await reopened.locate_request(
          tenant_storage_id=session.owner_user_id, approval_id=approval_id,
        ) == locator
        assert await reopened._locate_root_turn_for_approval(
          tenant_storage_id=session.owner_user_id, approval_id=approval_id,
        ) == (session.session_id, tool_call_id, nonce, root_a.turn_id)
        if ending == "rebuild":
          assert new_call_id is not None and new_pending is not None
          replacement = await reopened.get(str(new_pending["approval_id"]))
          assert replacement is not None
          assert replacement.state == "approved"
          assert await reopened._locate_root_turn_for_approval(
            tenant_storage_id=session.owner_user_id,
            approval_id=replacement.approval_id,
          ) == (
            session.session_id, new_call_id, new_pending["nonce"], root_a.turn_id,
          )
        if root_b is not None:
          later_turn = await reopened._read_root_turn_for_executor(
            job_id=session.session_id,
            turn_id=root_b.turn_id,
            tenant_storage_id=session.owner_user_id,
          )
          assert later_turn is not None
          assert later_turn.executable_refs == ()
    finally:
      finish_handler.set()
      transport_end.set()
      tasks: list[asyncio.Task[Any]] = [run_task, *disconnects]
      tasks.extend(iterator.handler for iterator in iterators if iterator.handler is not None)
      for task in tasks:
        if not task.done():
          task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

  _run(_case())


@pytest.mark.parametrize("startup_phase", ["connect", "query"])
def test_sdk_runner_disconnect_drains_owned_client_startup_before_resource_release(
  monkeypatch: pytest.MonkeyPatch,
  startup_phase: str,
) -> None:
  state = _install_fake_agent_sdk(monkeypatch)

  async def _case() -> None:
    startup_started = asyncio.Event()
    startup_cleaning_up = asyncio.Event()
    finish_startup_cleanup = asyncio.Event()
    startup_settled = asyncio.Event()
    client_close_states: list[bool] = []

    async def pause_startup() -> None:
      try:
        startup_started.set()
        await asyncio.Event().wait()
      finally:
        startup_cleaning_up.set()
        await finish_startup_cleanup.wait()
        startup_settled.set()

    class _StartingClient(state.client_type):
      async def connect(self) -> None:
        await super().connect()
        if startup_phase == "connect":
          await pause_startup()

      async def query(self, prompt: Any, session_id: str = "default") -> None:
        if startup_phase == "query":
          await pause_startup()
        await super().query(prompt, session_id)

      async def disconnect(self) -> None:
        client_close_states.append(startup_settled.is_set())
        await super().disconnect()

    monkeypatch.setattr(sys.modules["claude_agent_sdk"], "ClaudeSDKClient", _StartingClient)
    runner = _make_runner()
    run_task = asyncio.create_task(
      runner.run(messages=[{"role": "user", "content": "hello"}])
    )
    disconnects: list[asyncio.Task[None]] = []
    try:
      await asyncio.wait_for(startup_started.wait(), timeout=5)
      disconnects.append(asyncio.create_task(runner.on_disconnect()))
      await asyncio.wait_for(startup_cleaning_up.wait(), timeout=5)
      disconnects.append(asyncio.create_task(runner.on_disconnect()))
      await asyncio.sleep(0)
      disconnects[0].cancel()
      await asyncio.sleep(0)
      assert not run_task.done()
      assert all(not task.done() for task in disconnects)
      assert not state.clients[0].disconnected
      finish_startup_cleanup.set()
      with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(disconnects[0], timeout=5)
      await asyncio.wait_for(disconnects[1], timeout=5)
      with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(run_task, timeout=5)
      assert startup_settled.is_set()
      assert all(client_close_states)
      assert state.clients[0].disconnected
      assert state.prompts == []
    finally:
      finish_startup_cleanup.set()
      tasks: list[asyncio.Task[Any]] = [run_task, *disconnects]
      for task in tasks:
        if not task.done():
          task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

  _run(_case())


@pytest.mark.parametrize("cleanup_raises", [False, True])
def test_sdk_runner_public_client_close_is_a_shared_resource_barrier(
  monkeypatch: pytest.MonkeyPatch,
  cleanup_raises: bool,
) -> None:
  state = _install_fake_agent_sdk(monkeypatch)

  async def _case() -> None:
    close_started = asyncio.Event()
    finish_close = asyncio.Event()
    close_error = RuntimeError("client disconnect failed")

    class _ClosingClient(state.client_type):
      async def disconnect(self) -> None:
        close_started.set()
        await finish_close.wait()
        await super().disconnect()
        if cleanup_raises:
          raise close_error

    monkeypatch.setattr(sys.modules["claude_agent_sdk"], "ClaudeSDKClient", _ClosingClient)
    runner = _make_runner()
    run_task = asyncio.create_task(
      runner.run(messages=[{"role": "user", "content": "hello"}])
    )
    disconnects: list[asyncio.Task[None]] = []
    try:
      await asyncio.wait_for(close_started.wait(), timeout=5)
      disconnects = [
        asyncio.create_task(runner.on_disconnect()),
        asyncio.create_task(runner.on_disconnect()),
      ]
      await asyncio.sleep(0)
      assert not run_task.done()
      assert all(not task.done() for task in disconnects)
      assert not state.clients[0].disconnected
      finish_close.set()
      results = await asyncio.gather(run_task, *disconnects, return_exceptions=True)
      if cleanup_raises:
        assert all(result is close_error for result in results)
      else:
        assert results == [None, None, None]
      assert state.clients[0].disconnected
    finally:
      finish_close.set()
      tasks: list[asyncio.Task[Any]] = [run_task, *disconnects]
      for task in tasks:
        if not task.done():
          task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

  _run(_case())


def test_sdk_runner_external_disconnect_drains_self_closing_permission_handler(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  state = _install_fake_agent_sdk(monkeypatch)

  async def _case() -> None:
    transport_closing = asyncio.Event()
    finish_transport_close = asyncio.Event()
    handler_cleaning_up = asyncio.Event()
    finish_handler_cleanup = asyncio.Event()
    store = SQLiteApprovalStore(tmp_path / "self-closing-permission.sqlite3")
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
    )

    class _Policy:
      policy_bundle_hash = "test-policy"

      async def decide(
        self,
        *,
        payload: ApprovalRequestPayload,
        request: ApprovalRequest,
        run_context: RunContext,
      ) -> PolicyApprovalDecision:
        try:
          await runner.on_disconnect()
          await asyncio.Event().wait()
        finally:
          await runner.on_disconnect()
          handler_cleaning_up.set()
          await finish_handler_cleanup.wait()
        return PolicyApprovalDecision(
          outcome="request_user_approval",
          reason="Tool requires approval",
          expiry_seconds=600,
        )

      async def on_resolve(self, *, request: ApprovalRequest) -> None:
        pass

    class _ClosingClient(state.client_type):
      async def disconnect(self) -> None:
        transport_closing.set()
        await finish_transport_close.wait()
        await super().disconnect()

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
      approval_route=DurableLocalApprovalRoute(store, _Policy(), session),
    )
    client = _ClosingClient(options=_ClaudeAgentOptions())
    await client.connect()
    segment = _SDKSegmentQueryIterator(
      runner=runner,
      sdk_module=sys.modules["claude_agent_sdk"],
      initial_prompt="test",
      options_kwargs={},
    )
    segment._client = client
    runner._query_iter = segment
    handler = asyncio.create_task(
      runner._can_use_tool_callback("file_write", {"path": "x"}, None)
    )
    external: asyncio.Task[None] | None = None
    second_external: asyncio.Task[None] | None = None
    try:
      await asyncio.wait_for(handler_cleaning_up.wait(), timeout=5)
      external = asyncio.create_task(runner.on_disconnect())
      second_external = asyncio.create_task(runner.on_disconnect())
      await asyncio.sleep(0)
      assert not external.done()
      assert not second_external.done()
      assert not handler.done()
      assert not transport_closing.is_set()
      assert not client.disconnected
      finish_handler_cleanup.set()
      await asyncio.wait_for(transport_closing.wait(), timeout=5)
      assert handler.done()
      assert not external.done()
      assert not second_external.done()
      assert not client.disconnected
      finish_transport_close.set()
      await asyncio.wait_for(
        asyncio.gather(external, second_external),
        timeout=5,
      )
      assert client.disconnected
      with pytest.raises(asyncio.CancelledError):
        await handler
    finally:
      finish_transport_close.set()
      finish_handler_cleanup.set()
      tasks: list[asyncio.Task[Any]] = [handler]
      if external is not None:
        tasks.append(external)
      if second_external is not None:
        tasks.append(second_external)
      for task in tasks:
        if not task.done():
          task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

  _run(_case())


@pytest.mark.parametrize("cleanup_raises", [False, True])
def test_sdk_runner_disconnect_drain_survives_reentrant_cleanup_and_caller_cancellation(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  cleanup_raises: bool,
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
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="Tool requires approval",
        expiry_seconds=600,
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      pass

  async def _case() -> None:
    published = asyncio.Event()
    cleanup_started = asyncio.Event()
    finish_cleanup = asyncio.Event()
    cleanup_error = RuntimeError("permission cleanup failed")
    store = SQLiteApprovalStore(tmp_path / "reentrant-lifetime.sqlite3")
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
    )

    class _ApprovalLog(EventLog):
      def append(self, event: dict[str, Any]) -> Any:
        entry = super().append(event)
        if event.get("type") == "tool_approval_request":
          published.set()
        return entry

    runner = AgentSDKRunner(
      event_log=_ApprovalLog(),
      session_id=session.session_id,
      sdk_config=AgentSDKConfig(
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      ),
      capability_execution=stub_sdk_capability_execution(),
      system_prompt="test",
      session=session,
      approval_route=DurableLocalApprovalRoute(store, _Policy(), session),
    )
    original_wait = runner._await_user_approval_via_pending_tools

    async def wait_with_cleanup(*args: Any, **kwargs: Any) -> Any:
      try:
        return await original_wait(*args, **kwargs)
      finally:
        await runner.on_disconnect()
        cleanup_started.set()
        await finish_cleanup.wait()
        if cleanup_raises:
          raise cleanup_error

    runner._await_user_approval_via_pending_tools = wait_with_cleanup
    handler = asyncio.create_task(
      runner._can_use_tool_callback("file_write", {"path": "x"}, None)
    )
    disconnects: list[asyncio.Task[None]] = []
    try:
      await asyncio.wait_for(published.wait(), timeout=5)
      approval_id = str(next(iter(session.pending_tools.values()))["approval_id"])
      disconnects.append(asyncio.create_task(runner.on_disconnect()))
      await asyncio.wait_for(cleanup_started.wait(), timeout=5)
      disconnects.append(asyncio.create_task(runner.on_disconnect()))
      await asyncio.sleep(0)
      disconnects[0].cancel()
      await asyncio.sleep(0)
      disconnects[0].cancel()
      await asyncio.sleep(0)
      assert not handler.done()
      assert all(not task.done() for task in disconnects)
      assert session.pending_tools == {}
      assert session.approval_queues == {}
      finish_cleanup.set()
      if cleanup_raises:
        for task in [handler, *disconnects]:
          with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(task, timeout=5)
          assert raised.value is cleanup_error
      else:
        with pytest.raises(asyncio.CancelledError):
          await asyncio.wait_for(disconnects[0], timeout=5)
        await asyncio.wait_for(disconnects[1], timeout=5)
        with pytest.raises(asyncio.CancelledError):
          await handler
      stored = await store.get(approval_id)
      assert stored is not None
      assert stored.state == "pending_user"
    finally:
      finish_cleanup.set()
      tasks = [handler, *disconnects]
      for task in tasks:
        if not task.done():
          task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

  _run(_case())


@pytest.mark.parametrize(
  ("winner_state", "approved", "expected_behavior"),
  [
    ("approved", True, "allow"),
    ("denied", False, "deny"),
  ],
)
def test_sdk_runner_connected_vote_settles_original_callback_once(
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
        expiry_seconds=600,
      )

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

    def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
      return True

  async def _case() -> None:
    published = asyncio.Event()

    class _ApprovalLog(EventLog):
      def append(self, event: dict[str, Any]) -> Any:
        entry = super().append(event)
        if event.get("type") == "tool_approval_request":
          published.set()
        return entry

    store = SQLiteApprovalStore(tmp_path / f"winner-{winner_state}.sqlite3")
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
    )
    policy = _Policy()
    runner = AgentSDKRunner(
      event_log=_ApprovalLog(),
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
    )

    callback_task = asyncio.create_task(
      runner._can_use_tool_callback("file_write", {"path": "x"}, None)
    )
    await asyncio.wait_for(published.wait(), timeout=5)

    tool_call_id, pending = next(iter(session.pending_tools.items()))
    request = await store.get(str(pending["approval_id"]))
    assert request is not None
    vote: dict[str, Any] = {
      "target_session": session,
      "pending_entry": pending,
      "tool_call_id": tool_call_id,
      "nonce": pending["nonce"],
      "decider_id": "alice",
      "decider_role": "owner",
      "approved": approved,
      "allow_tool_type": False,
      "reason": None,
      "app_state": SimpleNamespace(
        gateway_approval_store=store,
        gateway_approval_policy=policy,
      ),
    }
    await _record_vote_and_unblock(**vote)
    with pytest.raises(ApprovalActionError) as duplicate:
      await _record_vote_and_unblock(**vote)
    assert duplicate.value.status_code == 409

    result = await callback_task
    assert result.behavior == expected_behavior
    assert not getattr(result, "interrupt", False)
    assert session.pending_tools == {}
    assert session.approval_queues == {}
    stored = await store.get(request.approval_id)
    assert stored is not None
    assert stored.state == winner_state

  _run(_case())


def test_sdk_batch_admission_disconnect_before_pending_publish_aborts_durable_row(
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
    await runner.on_disconnect()
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
        lambda _tool_name, tool_input, _scope: SimpleNamespace(
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


def test_sdk_runner_opted_in_skill_result_activates_exact_allow(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  contexts = []

  async def _on_tool_result(ctx: Any):
    contexts.append(ctx)
    return []

  runner = _make_runner(
    on_tool_result=_on_tool_result,
    disallowed_tools=["start_investment_run"],
    approval_store_path=tmp_path / "approvals.sqlite3",
  )
  result = {
    "skill": "phase0-agent",
    "content": "Do the work.",
    _ACTIVE_SKILL_ALLOW_RESULT_KEY: ["start_investment_run"],
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
  assert runner._active_skill_allow == {"start_investment_run"}
  assert contexts[0].result == {"skill": "phase0-agent", "content": "Do the work."}
  assert _ACTIVE_SKILL_ALLOW_RESULT_KEY not in contexts[0].result

  allowed = _run(runner._can_use_tool_callback("start_investment_run", {}, None))
  assert allowed.behavior == "allow"

  # A skill no longer subtracts tools: an unlisted tool keeps the run's surface.
  file_write = _run(runner._can_use_tool_callback("file_write", {"path": "x"}, None))
  assert file_write.behavior == "allow"


def test_sdk_runner_legacy_skill_result_does_not_activate_active_skill_gate(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  _install_fake_agent_sdk(monkeypatch)
  runner = _make_runner(
    disallowed_tools=["run_bash"],
    approval_store_path=tmp_path / "approvals.sqlite3",
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
  assert runner._active_skill_allow == set()

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
        }),
      },
      "tool-foreign",
      None,
    )
  )

  assert runner._active_skill_allow == set()
  assert hook_result["hookSpecificOutput"]["updatedMCPToolOutput"] == {
    "status": "ok",
  }
  assert "continue_" not in hook_result


def test_sdk_runner_active_skill_allow_clears_on_success_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_fake_agent_sdk(monkeypatch)
  success_runner = _make_runner()
  success_runner._active_skill_allow.add("file_write")

  async def _run_success() -> None:
    set_current_skill(_skill_admission("phase0-agent"))
    await success_runner.run([{"role": "user", "content": "hello"}])
    assert current_skill() is None

  _run(_run_success())
  assert success_runner._active_skill_allow == set()

  _install_fake_agent_sdk(
    monkeypatch,
    iterator_factory=lambda _prompt, _options: _AsyncMessages([RuntimeError("sdk failed")]),
  )
  error_runner = _make_runner()
  error_runner._active_skill_allow.add("file_write")

  async def _run_error() -> None:
    set_current_skill(_skill_admission("phase0-agent"))
    with pytest.raises(RuntimeError, match="sdk failed"):
      await error_runner.run([{"role": "user", "content": "hello"}])
    assert current_skill() is None

  _run(_run_error())
  assert error_runner._active_skill_allow == set()
