from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent.shared.tool_policy_implementations import (
  product_redaction_context_factory,
)

from agent_gateway.session import SessionStore
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.approval_route import DurableLocalApprovalRoute
from agent_gateway.approval_policy import (
  ApprovalDecision,
  ApprovalRequestPayload,
  DelegationGrant,
  PersistentGrant,
  RunContext,
  apply_decision_to_request,
  approval_reuse_scope_authorized,
  build_approval_request,
  utc_now,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.approvals import _record_vote_and_unblock_locked
from agent_gateway.single_user_policy import (
  DelegationApprovalPolicy,
  SingleUserApprovalPolicy,
)
from agent_gateway.tool_dispatcher import ToolDispatcher
from agent_gateway.tool_dispatcher_helpers import InterceptDecision
from agent_gateway.tool_policy_registry import (
  PlanDecision,
  PreparedToolCall,
  RedactionResult,
  SourceIdentityResult,
  ToolPolicyImplementation,
  ToolPolicyImplementationRegistry,
  reject_policy_parameters,
)
from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  ToolApprovalPolicy,
  ToolIntrinsicSemantics,
  ToolRegistrationCatalog,
  ToolRegistrationDeclaration,
  VersionedPolicyRef,
)

class _NoMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def _request(
  *,
  mode: str = "legacy",
  key: str | None = None,
  tool_call_id: str = "call-1",
):
  return build_approval_request(
    tool_call_id=tool_call_id,
    tool_name="registered_write",
    tool_class="state_write",
    tool_args_redacted={"ticker": "AAPL"},
    args_hash="args-hash",
    run_context=RunContext(user_id="alice", request_id="request-1"),
    approval_reuse_mode=mode,  # type: ignore[arg-type]
    approval_reuse_key=key,
  )


@pytest.mark.parametrize(
  ("mode", "key"),
  [
    ("exact", None),
    ("exact", ""),
    ("exact", " padded "),
    ("legacy", "key"),
    ("disabled", "key"),
  ],
)
def test_request_rejects_invalid_reuse_pairs(mode: str, key: str | None) -> None:
  with pytest.raises(ValueError):
    _request(mode=mode, key=key)


def test_exact_reuse_round_trips_and_is_immutable(tmp_path: Any) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  request = _request(mode="exact", key="registered:safe-key")

  _run(store.create(request))
  loaded = _run(store.get(request.approval_id))

  assert loaded is not None
  assert loaded.approval_reuse_mode == "exact"
  assert loaded.approval_reuse_key == "registered:safe-key"
  with pytest.raises(ValueError, match="identity is immutable"):
    _run(store.update_request(replace(loaded, approval_reuse_key="other-key")))


def test_exact_lookup_cannot_use_a_legacy_scope_collision(tmp_path: Any) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
  policy = SingleUserApprovalPolicy(store=store)
  legacy_source = replace(
    _request(),
    approval_id="legacy-source",
    approval_chain_id="legacy-source",
    state="approved",
    decision="approved",
    persistent_grant_scope="registered:safe-key",
  )
  _run(store.create(legacy_source))
  _run(store.create_persistent_grant(PersistentGrant(
    grant_id="legacy-grant",
    user_id="alice",
    tool_name="registered_write",
    scope_hint="registered:safe-key",
    args_predicate=None,
    granted_at=utc_now(),
    expires_at=None,
    revoked_at=None,
    granted_via_approval_id=legacy_source.approval_id,
    policy_id=policy.policy_id,
  )))
  request = _request(
    mode="exact",
    key="registered:safe-key",
    tool_call_id="call-exact",
  )
  payload = ApprovalRequestPayload(
    request.approval_id,
    request.tool_name,
    request.tool_class,
    {"ticker": "AAPL"},
  )

  decision = _run(policy.decide(
    payload=payload,
    request=request,
    run_context=RunContext(user_id="alice", request_id="request-1"),
  ))

  assert decision.outcome == "request_user_approval"
  assert decision.persistent_grant_scope_hint == "registered:safe-key"

  exact_source = replace(
    _request(
      mode="exact",
      key="registered:safe-key",
      tool_call_id="exact-source-call",
    ),
    approval_id="exact-source",
    approval_chain_id="exact-source",
    state="approved",
    decision="approved",
    persistent_grant_scope="registered:safe-key",
  )
  _run(store.create(exact_source))
  _run(store.create_persistent_grant(PersistentGrant(
    grant_id="exact-grant",
    user_id="alice",
    tool_name="registered_write",
    scope_hint="registered:safe-key",
    args_predicate=None,
    granted_at=utc_now(),
    expires_at=None,
    revoked_at=None,
    granted_via_approval_id=exact_source.approval_id,
    policy_id=policy.policy_id,
  )))

  reused = _run(policy.decide(
    payload=payload,
    request=request,
    run_context=RunContext(user_id="alice", request_id="request-1"),
  ))

  assert reused.outcome == "auto_approve"
  assert reused.grant_reference == "exact-grant"
  applied = apply_decision_to_request(request, reused)
  assert applied.authorization_mode == "PERSISTENT_GRANT"
  assert applied.grant_reference == "exact-grant"
  assert applied.persistent_grant_scope == "registered:safe-key"


@pytest.mark.parametrize(
  ("mode", "key", "scope"),
  [
    ("disabled", None, "legacy:scope"),
    ("exact", "registered:safe-key", "registered:other-key"),
  ],
)
def test_policy_cannot_attach_a_mismatched_reuse_grant(
  mode: str,
  key: str | None,
  scope: str,
) -> None:
  applied = apply_decision_to_request(
    _request(mode=mode, key=key),
    ApprovalDecision(
      outcome="auto_approve",
      reason="hostile policy grant",
      persistent_grant_scope_hint=scope,
      grant_reference="wrong-grant",
    ),
  )

  assert applied.authorization_mode == "HUMAN"
  assert applied.grant_reference is None
  assert applied.persistent_grant_scope is None


@pytest.mark.parametrize(
  ("mode", "key", "scope"),
  [
    ("legacy", None, "legacy:scope"),
    ("exact", "registered:safe-key", "registered:safe-key"),
  ],
)
def test_policy_cannot_offer_persistence_without_authorizing_it(
  mode: str,
  key: str | None,
  scope: str,
) -> None:
  applied = apply_decision_to_request(
    _request(mode=mode, key=key),
    ApprovalDecision(
      outcome="request_user_approval",
      reason="approve once",
      allow_persistent_grant=False,
      persistent_grant_scope_hint=scope,
    ),
  )

  assert applied.persistent_grant_scope is None


def test_exact_session_cache_hit_records_the_exact_key(tmp_path: Any) -> None:
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")

  class Policy:
    policy_id = "test-policy"
    policy_version = "1"

    async def decide(self, **_kwargs: Any) -> Any:
      raise AssertionError("an exact session cache hit cannot consult policy")

    async def on_resolve(self, *, request: Any) -> None:
      assert request.cache_reference == "registered:safe-key"

  dispatcher = ToolDispatcher(
    mcp_client=_NoMcp(),
    approval_route=DurableLocalApprovalRoute(
      store,
      Policy(),
      SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice"),
    ),
    run_context=RunContext(user_id="alice", request_id="request-cache"),
  )

  result = _run(dispatcher._run_approval_lifecycle(
    tool_call_id="call-cache",
    tool_name="registered_write",
    tool_input={"ticker": "AAPL"},
    qualifier="legacy-qualifier-must-not-appear",
    reason="",
    allow_persistent=True,
    approval_reuse_mode="exact",
    approval_reuse_key="registered:safe-key",
    approval_args_redacted={"ticker": "AAPL"},
    approval_args_hash="args-hash",
    session_cache_approved=True,
  ))

  request = result["request"]
  assert request.authorization_mode == "CACHE_HIT"
  assert request.cache_reference == "registered:safe-key"
  assert _run(store.get(request.approval_id)) == request


def test_resume_cannot_change_approval_reuse_identity(tmp_path: Any) -> None:
  dispatcher = ToolDispatcher(
    mcp_client=_NoMcp(),
    approval_route=DurableLocalApprovalRoute(
      SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
      SingleUserApprovalPolicy(),
      SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice"),
    ),
    run_context=RunContext(user_id="alice", request_id="request-resume"),
  )

  with pytest.raises(RuntimeError, match="constraint changed"):
    _run(dispatcher._run_approval_lifecycle(
      tool_call_id="call-resume",
      tool_name="registered_write",
      tool_input={"ticker": "AAPL"},
      qualifier="",
      reason="",
      allow_persistent=False,
      approval_reuse_mode="disabled",
      resume_approval_request=_request(
        mode="exact",
        key="registered:safe-key",
        tool_call_id="call-resume",
      ),
      approval_args_redacted={"ticker": "AAPL"},
      approval_args_hash="args-hash",
    ))


def test_client_allow_tool_type_cannot_enable_disabled_reuse(tmp_path: Any) -> None:
  async def case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request = replace(
      _request(mode="disabled", tool_call_id="call-disabled"),
      state="pending_user",
    )
    await store.create(request)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
    session = SessionStore(ttl=3600).create_session(
      api_key_hash="hash",
      user_id="alice",
    )
    session.approval_queues[request.tool_call_id] = queue
    pending = {
      "approval_id": request.approval_id,
      "nonce": "nonce-1",
      "status": "approval_pending",
      "tool_name": request.tool_name,
    }

    await _record_vote_and_unblock_locked(
      target_session=session,
      pending_entry=pending,
      tool_call_id=request.tool_call_id,
      nonce="nonce-1",
      decider_id="alice",
      decider_role="owner",
      approved=True,
      allow_tool_type=True,
      reason="approved once",
      app_state=SimpleNamespace(),
      authoritative_store=store,
      authoritative_policy=SingleUserApprovalPolicy(store=store),
    )

    assert (await queue.get())["allow_tool_type"] is False
    assert pending["allow_tool_type"] is False
    assert await store.find_persistent_grant(
      user_id="alice",
      tool_name=request.tool_name,
      scope_hint="forged",
      approval_reuse_mode="disabled",
    ) is None

  _run(case())


def test_disabled_reuse_skips_lookup_and_disallows_persistence() -> None:
  class Store:
    async def find_persistent_grant(self, **_kwargs: Any) -> Any:
      raise AssertionError("disabled reuse cannot consult persistent grants")

  policy = SingleUserApprovalPolicy(store=Store())
  request = _request(mode="disabled")
  payload = ApprovalRequestPayload(
    request.approval_id,
    request.tool_name,
    request.tool_class,
    {"ticker": "AAPL"},
  )

  decision = _run(policy.decide(
    payload=payload,
    request=request,
    run_context=RunContext(user_id="alice", request_id="request-1"),
  ))

  assert decision.outcome == "request_user_approval"
  assert decision.allow_persistent_grant is False
  assert decision.persistent_grant_scope_hint is None
  assert approval_reuse_scope_authorized(
    replace(request, persistent_grant_scope="forged-client-scope")
  ) is False


def test_registered_reuse_never_uses_delegation_auto_approval() -> None:
  now = utc_now()
  delegation = DelegationGrant(
    delegation_id="delegation-1",
    delegator_user_id="alice",
    delegator_run_id="run-1",
    delegator_session_id="session-1",
    delegator_profile="chat",
    delegator_channel="web",
    bound_excel_session_id="excel-1",
    bound_relay_request_id="relay-1",
    bound_workbook=None,
    tool_class_ceiling=frozenset({"state_write"}),
    args_predicate=None,
    window_seconds=600,
    created_at=now,
  )
  run_context = RunContext(
    user_id="alice",
    request_id="request-1",
    delegation=delegation,
  )
  policy = DelegationApprovalPolicy(base=SingleUserApprovalPolicy())
  request = _request(mode="exact", key="registered:safe-key")
  payload = ApprovalRequestPayload(
    request.approval_id,
    request.tool_name,
    request.tool_class,
    {"ticker": "AAPL"},
  )

  decision = _run(policy.decide(
    payload=payload,
    request=request,
    run_context=run_context,
  ))

  assert decision.outcome == "request_user_approval"
  assert decision.policy_id == "single-user"


def _ref(kind: str) -> VersionedPolicyRef:
  return VersionedPolicyRef(kind=kind, policy_id=kind, version="v1")  # type: ignore[arg-type]


def _implementation(kind: str, callback: Any) -> ToolPolicyImplementation:
  return ToolPolicyImplementation(
    kind,  # type: ignore[arg-type]
    kind,
    "v1",
    reject_policy_parameters,
    callback,
  )


def test_dynamic_ask_disables_registered_exact_reuse() -> None:
  declaration = ToolRegistrationDeclaration(
    RegisteredToolIdentity(
      route_kind="local_handler",
      logical_server_id=None,
      logical_name="registered_write",
    ),
    ToolIntrinsicSemantics(
      effect="state_write",
      idempotent=False,
      semantic_capability="test.write/v1",
      approval=ToolApprovalPolicy("always", cache_key=_ref("approval_cache_key")),
      audience="ordinary",
      redaction_policy=_ref("redaction"),
      planning_policy=_ref("planning"),
      input_preparation_policy=_ref("input_preparation"),
      outcome_policy=_ref("outcome"),
      source_identity_policy=_ref("source_identity"),
    ),
  )
  registry = ToolPolicyImplementationRegistry((
    _implementation("approval_cache_key", lambda _ref, _call: "registered:safe-key"),
    _implementation("approval_predicate", lambda _ref, _call: True),
    _implementation("redaction", lambda _ref, call: RedactionResult(call.prepared_input)),
    _implementation("planning", lambda _ref, _call: PlanDecision("none")),
    _implementation("input_preparation", lambda _ref, call: PreparedToolCall(call.raw_input)),
    _implementation("outcome", lambda _ref, _call: "ok"),
    _implementation("source_identity", lambda _ref, _call: SourceIdentityResult(())),
    _implementation("session_injection", lambda _ref, _call: None),
  ))


  async def ask(_context: Any) -> InterceptDecision:
    return InterceptDecision("ask", message="inspect this write")

  async def handler(tool_input: dict[str, Any], **_kwargs: Any):
    return tool_input, None

  dispatcher = ToolDispatcher(
    mcp_client=_NoMcp(),
    local_tool_handlers={"registered_write": handler},
    needs_approval=lambda *_args: True,
    interceptors=(ask,),
    approval_route=DurableLocalApprovalRoute(
      SimpleNamespace(),
      SimpleNamespace(),
      SessionStore(ttl=3600).create_session(
        api_key_hash="hash",
        user_id="alice",
        role="owner",
      ),
    ),
    role="owner",
    tool_registration_catalog=ToolRegistrationCatalog((declaration,), ()),
    tool_policy_implementations=registry,
    approval_predicate_context_factory=lambda _declaration, _prepared: None,
    redaction_context_factory=product_redaction_context_factory,
  )
  captured: dict[str, Any] = {}

  async def lifecycle(**kwargs: Any) -> dict[str, Any]:
    captured.update(kwargs)
    return {"approved": True, "allow_tool_type": True, "tool_input": kwargs["tool_input"]}

  dispatcher._run_approval_lifecycle = lifecycle
  result = _run(dispatcher.dispatch_prepared(
    "call-dynamic",
    "registered_write",
    PreparedToolCall({"value": 1}),
  ))

  assert result == ({"value": 1}, None)
  assert captured["approval_reuse_mode"] == "disabled"
  assert captured["approval_reuse_key"] is None
  assert captured["allow_persistent"] is False
