import asyncio
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_policy import (
  ApprovalRequest,
  ApprovalRequestPayload,
  DelegationGrant,
  PersistentGrant,
  RunContext,
  ToolClass,
  build_approval_request,
  utc_now,
)
from agent_gateway.approval_resolver import resolve_policy
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.single_user_policy import DelegationApprovalPolicy, SingleUserApprovalPolicy


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def _store(tmp_path: Path) -> SQLiteApprovalStore:
  return SQLiteApprovalStore(tmp_path / "approvals.sqlite3")


def _grant(**overrides: Any) -> DelegationGrant:
  values: dict[str, Any] = {
    "delegation_id": "delegation-1",
    "delegator_user_id": "alice",
    "delegator_run_id": "run-1",
    "delegator_session_id": "chat-session-1",
    "delegator_profile": "chat",
    "delegator_channel": "web",
    "bound_excel_session_id": "excel-session-1",
    "bound_relay_request_id": "relay-request-1",
    "bound_workbook": "Budget.xlsx",
    "tool_class_ceiling": frozenset({"state_write"}),
    "args_predicate": None,
    "window_seconds": 600,
    "exclude_external_write_bypass": True,
    "created_at": utc_now(),
    "expires_at": None,
    "revoked_at": None,
    "consumed_at": None,
  }
  values.update(overrides)
  return DelegationGrant(**values)


def _run_context(*, delegation: DelegationGrant | None = None) -> RunContext:
  return RunContext(
    user_id="alice",
    request_id="request-1",
    session_id="excel-session-1",
    run_id="excel-run-1",
    profile="chat",
    channel="excel",
    delegation=delegation,
  )


def _request_and_payload(
  *,
  run_context: RunContext,
  tool_class: ToolClass = "state_write",
  tool_name: str = "update_position",
  tool_args: dict[str, Any] | None = None,
) -> tuple[ApprovalRequest, ApprovalRequestPayload]:
  args = dict(tool_args or {})
  request = build_approval_request(
    tool_call_id=f"tool-{tool_class}-{tool_name}",
    tool_name=tool_name,
    tool_class=tool_class,
    tool_args_redacted=dict(args),
    args_hash=f"hash-{tool_class}-{tool_name}",
    run_context=run_context,
  )
  payload = ApprovalRequestPayload(
    request.approval_id,
    request.tool_name,
    request.tool_class,
    dict(args),
  )
  return request, payload


def _policy(*, store: SQLiteApprovalStore | None = None) -> DelegationApprovalPolicy:
  return DelegationApprovalPolicy(base=SingleUserApprovalPolicy(store=store))


def test_no_delegation_passthrough_matches_base_for_state_write() -> None:
  async def _case() -> None:
    base = SingleUserApprovalPolicy()
    policy = DelegationApprovalPolicy(base=base)
    run_context = _run_context()
    request, payload = _request_and_payload(run_context=run_context)

    expected = await base.decide(payload=payload, request=request, run_context=run_context)
    actual = await policy.decide(payload=payload, request=request, run_context=run_context)

    assert actual == expected
    assert actual.outcome == "request_user_approval"

  _run(_case())


def test_delegated_state_write_in_ceiling_auto_approves_within_window() -> None:
  async def _case() -> None:
    delegation = _grant(tool_class_ceiling=frozenset({"state_write"}))
    run_context = _run_context(delegation=delegation)
    request, payload = _request_and_payload(run_context=run_context)

    decision = await _policy().decide(payload=payload, request=request, run_context=run_context)

    assert decision.outcome == "auto_approve"
    assert decision.reason == "delegation grant matched"
    assert decision.policy_id == "delegation"

  _run(_case())


def test_delegated_state_write_not_in_ceiling_requests_user_approval() -> None:
  async def _case() -> None:
    delegation = _grant(tool_class_ceiling=frozenset({"read"}))
    run_context = _run_context(delegation=delegation)
    request, payload = _request_and_payload(run_context=run_context)

    decision = await _policy().decide(payload=payload, request=request, run_context=run_context)

    assert decision.outcome == "request_user_approval"
    assert decision.policy_id == "delegation"

  _run(_case())


def test_delegated_external_write_ignores_matching_persistent_grant(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    base = SingleUserApprovalPolicy(store=store)
    policy = DelegationApprovalPolicy(base=base)
    await store.create_persistent_grant(
      PersistentGrant(
        grant_id="persistent-grant-1",
        user_id="alice",
        tool_name="execute_trade",
        scope_hint="external_write:execute_trade:AAPL",
        args_predicate=None,
        granted_at=utc_now(),
        expires_at=utc_now() + timedelta(days=1),
        revoked_at=None,
        granted_via_approval_id="prior-approval",
        policy_id="single-user",
      )
    )

    non_delegated_context = _run_context()
    request, payload = _request_and_payload(
      run_context=non_delegated_context,
      tool_class="external_write",
      tool_name="execute_trade",
      tool_args={"ticker": "AAPL"},
    )
    base_decision = await base.decide(payload=payload, request=request, run_context=non_delegated_context)
    assert base_decision.outcome == "auto_approve"

    delegated_context = _run_context(
      delegation=_grant(tool_class_ceiling=frozenset({"external_write"})),
    )
    delegated_request, delegated_payload = _request_and_payload(
      run_context=delegated_context,
      tool_class="external_write",
      tool_name="execute_trade",
      tool_args={"ticker": "AAPL"},
    )
    delegated_decision = await policy.decide(
      payload=delegated_payload,
      request=delegated_request,
      run_context=delegated_context,
    )

    assert delegated_decision.outcome == "request_user_approval"
    assert delegated_decision.policy_id == "delegation"

  _run(_case())


def test_resolve_policy_wraps_default_and_blocks_delegated_external_write_persistent_grant(
  tmp_path: Path,
  monkeypatch: Any,
) -> None:
  async def _case() -> None:
    monkeypatch.delenv("GATEWAY_APPROVAL_POLICY_CLASS", raising=False)
    store = _store(tmp_path)
    policy = resolve_policy(store=store)
    assert isinstance(policy, DelegationApprovalPolicy)
    await store.create_persistent_grant(
      PersistentGrant(
        grant_id="persistent-grant-1",
        user_id="alice",
        tool_name="execute_trade",
        scope_hint="external_write:execute_trade:AAPL",
        args_predicate=None,
        granted_at=utc_now(),
        expires_at=utc_now() + timedelta(days=1),
        revoked_at=None,
        granted_via_approval_id="prior-approval",
        policy_id="single-user",
      )
    )

    delegated_context = _run_context(
      delegation=_grant(tool_class_ceiling=frozenset({"external_write"})),
    )
    request, payload = _request_and_payload(
      run_context=delegated_context,
      tool_class="external_write",
      tool_name="execute_trade",
      tool_args={"ticker": "AAPL"},
    )

    decision = await policy.decide(payload=payload, request=request, run_context=delegated_context)

    assert decision.outcome == "request_user_approval"
    assert decision.policy_id == "delegation"

  _run(_case())


def test_delegated_portfolio_config_and_irreversible_request_user_approval() -> None:
  async def _case() -> None:
    policy = _policy()
    for tool_class in ("portfolio_config", "irreversible"):
      run_context = _run_context(
        delegation=_grant(tool_class_ceiling=frozenset({tool_class})),
      )
      request, payload = _request_and_payload(
        run_context=run_context,
        tool_class=tool_class,  # type: ignore[arg-type]
        tool_name=f"{tool_class}_tool",
      )

      decision = await policy.decide(payload=payload, request=request, run_context=run_context)

      assert decision.outcome == "request_user_approval"
      assert decision.policy_id == "delegation"

  _run(_case())


def test_delegated_args_predicate_must_match() -> None:
  async def _case() -> None:
    policy = _policy()
    delegation = _grant(
      tool_class_ceiling=frozenset({"state_write"}),
      args_predicate={"ticker": "MSFT"},
    )

    mismatch_context = _run_context(delegation=delegation)
    mismatch_request, mismatch_payload = _request_and_payload(
      run_context=mismatch_context,
      tool_args={"ticker": "AAPL"},
    )
    mismatch = await policy.decide(
      payload=mismatch_payload,
      request=mismatch_request,
      run_context=mismatch_context,
    )

    match_context = _run_context(delegation=delegation)
    match_request, match_payload = _request_and_payload(
      run_context=match_context,
      tool_args={"ticker": "MSFT"},
    )
    match = await policy.decide(payload=match_payload, request=match_request, run_context=match_context)

    assert mismatch.outcome == "request_user_approval"
    assert match.outcome == "auto_approve"

  _run(_case())


def test_delegated_expired_window_requests_user_approval() -> None:
  async def _case() -> None:
    delegation = _grant(
      created_at=utc_now() - timedelta(days=1),
      window_seconds=1,
      tool_class_ceiling=frozenset({"state_write"}),
    )
    run_context = _run_context(delegation=delegation)
    request, payload = _request_and_payload(run_context=run_context)

    decision = await _policy().decide(payload=payload, request=request, run_context=run_context)

    assert decision.outcome == "request_user_approval"
    assert decision.policy_id == "delegation"

  _run(_case())


def test_build_request_and_store_round_trip_delegation_id(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    delegation = _grant(delegation_id="delegation-roundtrip")
    run_context = _run_context(delegation=delegation)
    request, _payload = _request_and_payload(run_context=run_context)

    assert request.delegation_id == "delegation-roundtrip"

    await store.create(request)
    stored = await store.get(request.approval_id)

    assert stored is not None
    assert stored.delegation_id == "delegation-roundtrip"

  _run(_case())
