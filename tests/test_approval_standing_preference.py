# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest


PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_policy import (
  ApprovalRequestPayload,
  RunContext,
  ToolClass,
  build_approval_request,
)
from agent_gateway.approval_preferences import ApprovalPreferenceStore
from agent_gateway.single_user_policy import SingleUserApprovalPolicy


def _decide(
  policy: SingleUserApprovalPolicy,
  *,
  tool_class: ToolClass,
  tool_name: str,
  user_id: str = "alice",
) -> Any:
  run_context = RunContext(
    user_id=user_id,
    request_id="request-1",
    session_id="chat-session-1",
    profile="analyst",
    channel="excel",
  )
  request = build_approval_request(
    tool_call_id=f"tool-{tool_class}-{tool_name}",
    tool_name=tool_name,
    tool_class=tool_class,
    tool_args_redacted={},
    args_hash=f"hash-{tool_name}",
    run_context=run_context,
  )
  payload = ApprovalRequestPayload(
    request.approval_id,
    request.tool_name,
    request.tool_class,
    {},
  )
  return asyncio.run(
    policy.decide(payload=payload, request=request, run_context=run_context)
  )


@pytest.mark.parametrize(
  ("tool_class", "tool_name"),
  [
    ("state_write", "memory_store"),
    ("external_write", "gsheets_write_range"),
    ("artifact_write", "write_cells"),
    ("portfolio_config", "set_risk_profile"),
  ],
)
def test_unset_preference_settles_every_class_but_the_money_boundary(
  tmp_path: Path,
  tool_class: ToolClass,
  tool_name: str,
) -> None:
  """The product default: no agent-driven surface draws a card for a write."""

  policy = SingleUserApprovalPolicy(
    preference_store=ApprovalPreferenceStore(tmp_path / "approval-preferences.sqlite3")
  )

  decision = _decide(policy, tool_class=tool_class, tool_name=tool_name)

  assert decision.outcome == "auto_approve"
  assert decision.reason == "Standing approval preference: auto_approve_all_but_trades"


@pytest.mark.parametrize("tool_name", ["execute_trade", "cancel_order"])
def test_money_boundary_asks_whatever_the_preference_says(
  tmp_path: Path,
  tool_name: str,
) -> None:
  store = ApprovalPreferenceStore(tmp_path / "approval-preferences.sqlite3")
  store.put(user_id="alice", preference="auto_approve_all_but_trades")
  policy = SingleUserApprovalPolicy(preference_store=store)

  decision = _decide(policy, tool_class="irreversible", tool_name=tool_name)

  assert decision.outcome == "request_user_approval"
  assert decision.allow_persistent_grant is False


def test_opt_out_restores_the_prompt_for_a_write(tmp_path: Path) -> None:
  """Both states are supported: a user who wants to be asked is asked."""

  store = ApprovalPreferenceStore(tmp_path / "approval-preferences.sqlite3")
  store.put(user_id="alice", preference="request_user_approval")
  policy = SingleUserApprovalPolicy(preference_store=store)

  decision = _decide(policy, tool_class="state_write", tool_name="memory_store")

  assert decision.outcome == "request_user_approval"


def test_preference_is_per_user_not_global(tmp_path: Path) -> None:
  store = ApprovalPreferenceStore(tmp_path / "approval-preferences.sqlite3")
  store.put(user_id="alice", preference="request_user_approval")
  policy = SingleUserApprovalPolicy(preference_store=store)

  alice = _decide(policy, tool_class="state_write", tool_name="memory_store", user_id="alice")
  bob = _decide(policy, tool_class="state_write", tool_name="memory_store", user_id="bob")

  assert alice.outcome == "request_user_approval"
  assert bob.outcome == "auto_approve"


def test_stored_preference_survives_a_new_store_over_the_same_file(tmp_path: Path) -> None:
  path = tmp_path / "approval-preferences.sqlite3"
  ApprovalPreferenceStore(path).put(user_id="alice", preference="request_user_approval")

  reopened = ApprovalPreferenceStore(path).get(user_id="alice")

  assert reopened.preference == "request_user_approval"
  assert reopened.source == "stored"


def test_an_unsupported_preference_is_refused_at_the_write(tmp_path: Path) -> None:
  store = ApprovalPreferenceStore(tmp_path / "approval-preferences.sqlite3")

  with pytest.raises(ValueError, match="unsupported approval preference"):
    store.put(user_id="alice", preference="approve_everything")

  assert store.get(user_id="alice").preference == "auto_approve_all_but_trades"
