import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.excel_dispatch import (
  make_message_excel_agent_handler,
  make_message_excel_agent_tool_def,
  mint_and_submit,
  poll_result,
  validate_requested_ceiling,
)


def test_message_excel_agent_tool_def_advertises_stable_selection_keys() -> None:
  """The documented tool contract must expose the selection triple the handlers
  thread through, teaching stable keys and omission-equals-server-default."""

  properties = make_message_excel_agent_tool_def()["input_schema"]["properties"]

  assert "model" not in properties
  assert properties["model_key"]["type"] == "string"
  assert "stable" in properties["model_key"]["description"]
  assert "server default" in properties["model_key"]["description"]
  assert "typed error" in properties["model_key"]["description"]
  assert properties["effort"]["type"] == "string"
  assert "Requires model_key" in properties["effort"]["description"]
  assert properties["catalog_revision"]["type"] == "string"
  assert "Requires model_key" in properties["catalog_revision"]["description"]
  assert "never authority" in properties["catalog_revision"]["description"]


class _RelayRestartInProgress(RuntimeError):
  pass


class _RelayRestartLockUnavailable(RuntimeError):
  pass


class _RestartMessageLookalike(RuntimeError):
  pass


_RELAY_RESTART_EXCEPTIONS = (
  _RelayRestartInProgress,
  _RelayRestartLockUnavailable,
)


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def _store(tmp_path: Path) -> SQLiteApprovalStore:
  return SQLiteApprovalStore(tmp_path / "approvals.sqlite3")


def _delegation_count(tmp_path: Path) -> int:
  with sqlite3.connect(tmp_path / "approvals.sqlite3") as conn:
    row = conn.execute("SELECT COUNT(*) FROM delegation_grants").fetchone()
  return int(row[0])


class FakeRelay:
  def __init__(
    self,
    *,
    workbooks: list[dict[str, Any]],
    results: list[tuple[str, dict[str, Any]]] | None = None,
    repeat_last_result: bool = True,
  ) -> None:
    self.workbooks = workbooks
    self.results = list(results or [("ok", {"state": "done", "result": {"text": "ok"}})])
    self.repeat_last_result = repeat_last_result
    self.list_calls: list[dict[str, Any]] = []
    self.submissions: list[dict[str, Any]] = []
    self.result_calls: list[dict[str, Any]] = []
    self.owner_session: str | None = None

  async def list_workbooks(self, gateway_session_id: str | None = None, user_id: str | None = None) -> list[dict[str, Any]]:
    self.list_calls.append({"gateway_session_id": gateway_session_id, "user_id": user_id})
    return list(self.workbooks)

  async def submit(
    self,
    tool_name: str,
    tool_input: dict[str, Any],
    timeout: int,
    target_session: str | None = None,
    *,
    gateway_session_id: str | None = None,
    user_id: str | None = None,
    kind: str = "tool",
    request_id: str | None = None,
  ) -> dict[str, Any]:
    self.submissions.append(
      {
        "tool_name": tool_name,
        "tool_input": dict(tool_input),
        "timeout": timeout,
        "target_session": target_session,
        "gateway_session_id": gateway_session_id,
        "user_id": user_id,
        "kind": kind,
        "request_id": request_id,
      }
    )
    # The real relay records the inflight owner as the resolved target Excel session.
    self.owner_session = target_session
    return {"request_id": request_id}

  async def result(
    self,
    request_id: str,
    *,
    gateway_session_id: str | None = None,
    user_id: str | None = None,
  ) -> tuple[str, dict[str, Any]]:
    self.result_calls.append({"request_id": request_id, "gateway_session_id": gateway_session_id, "user_id": user_id})
    if self.owner_session is not None and gateway_session_id != self.owner_session:
      # Mirror the real relay's _validate_owner_locked: only the inflight request's
      # owner (the target Excel session) may poll its result.
      raise PermissionError(f"result owner mismatch: {gateway_session_id} != {self.owner_session}")
    if self.results:
      result = self.results.pop(0)
      if self.repeat_last_result and not self.results:
        self.results.append(result)
      return result
    return "ok", {"state": "pending"}


class RestartBlockedRelay(FakeRelay):
  def __init__(
    self,
    *args: Any,
    exception_type: type[Exception],
    exception_message: str = "relay_restart_in_progress",
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.exception_type = exception_type
    self.exception_message = exception_message

  async def submit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
    await super().submit(*args, **kwargs)
    raise self.exception_type(self.exception_message)


def _handler(relay: FakeRelay, store: SQLiteApprovalStore, *, poll_interval_seconds: float = 0.001):
  return make_message_excel_agent_handler(
    relay=relay,
    approval_store=store,
    user_id="alice",
    gateway_session_id="orchestrator-session-1",
    delegator_profile="orchestrator",
    delegator_run_id="run-1",
    poll_interval_seconds=poll_interval_seconds,
    relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
  )


def test_mint_and_submit_happy_path_returns_ids_and_bound_grant(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ]
    )

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Update the model",
      workbook="Budget.xlsx",
      force_compaction=True,
      window_seconds=120,
      args_predicate={"sheet": "Budget"},
      delegator_profile="autonomous",
      delegator_run_id="run-2",
      default_ceiling=frozenset({"read", "pure_transform"}),
      relay_timeout_seconds=42,
      seed_history=[{"role": "user", "content": "seed"}],
      model_key="anthropic.claude-sonnet-5",
      effort="high",
      catalog_revision="2026-08-13.1",
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert error is None
    assert submitted is not None
    assert isinstance(submitted["request_id"], str)
    assert isinstance(submitted["delegation_id"], str)
    assert submitted["excel_session_id"] == "excel-session-1"
    assert submitted["workbook"] == "Budget.xlsx"

    stored = await store.get_delegation_grant(submitted["delegation_id"])
    assert stored is not None
    assert stored.delegator_user_id == "alice"
    assert stored.delegator_session_id == "orchestrator-session-1"
    assert stored.delegator_run_id == "run-2"
    assert stored.delegator_profile == "autonomous"
    assert stored.delegator_channel == "excel"
    assert stored.bound_excel_session_id == "excel-session-1"
    assert stored.bound_relay_request_id == submitted["request_id"]
    assert stored.bound_workbook == "Budget.xlsx"
    assert stored.tool_class_ceiling == frozenset({"read", "pure_transform"})
    assert stored.args_predicate == {"sheet": "Budget"}
    assert stored.window_seconds == 120
    assert stored.exclude_external_write_bypass is True

    assert len(relay.submissions) == 1
    submission = relay.submissions[0]
    assert submission["tool_name"] == "send_chat_message"
    assert submission["target_session"] == "excel-session-1"
    assert submission["gateway_session_id"] == "orchestrator-session-1"
    assert submission["user_id"] == "alice"
    assert submission["kind"] == "chat"
    assert submission["request_id"] == submitted["request_id"]
    assert submission["timeout"] == 42
    assert submission["tool_input"] == {
      "text": "Update the model",
      "force_compaction": True,
      "delegation_id": submitted["delegation_id"],
      "seed_history": [{"role": "user", "content": "seed"}],
      "model_key": "anthropic.claude-sonnet-5",
      "effort": "high",
      "catalog_revision": "2026-08-13.1",
    }
    assert relay.result_calls == []

  _run(_case())


def test_mint_and_submit_refuses_raw_model_typed_without_grant_or_submit(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ]
    )

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Update the model",
      model="claude-sonnet-4-6",
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error is not None
    assert error["code"] == "chat_model_not_accepted"
    assert "model_key" in error["message"]
    assert _delegation_count(tmp_path) == 0
    assert relay.submissions == []

  _run(_case())


def test_mint_and_submit_refuses_explicit_null_raw_model(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ]
    )

    # Presence of the retired key is refused even with a null value, matching
    # the relay admission predicate ("model" in tool_input).
    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Update the model",
      model=None,
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error is not None
    assert error["code"] == "chat_model_not_accepted"
    assert relay.submissions == []

  _run(_case())


def test_message_handler_threads_stable_selection_keys_to_relay_payload(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ]
    )
    handler = _handler(relay, store)

    result, error = await handler(
      {
        "text": "Update the model",
        "model_key": "anthropic.claude-sonnet-5",
        "effort": "high",
        "catalog_revision": "2026-08-13.1",
      }
    )

    assert error is None
    assert result is not None
    assert len(relay.submissions) == 1
    tool_input = relay.submissions[0]["tool_input"]
    assert tool_input["model_key"] == "anthropic.claude-sonnet-5"
    assert tool_input["effort"] == "high"
    assert tool_input["catalog_revision"] == "2026-08-13.1"
    assert "model" not in tool_input

  _run(_case())


def test_message_handler_refuses_raw_model_typed(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ]
    )
    handler = _handler(relay, store)

    result, error = await handler({"text": "Update the model", "model": "claude-sonnet-4-6"})

    assert result is None
    assert error is not None
    assert error["code"] == "chat_model_not_accepted"
    assert _delegation_count(tmp_path) == 0
    assert relay.submissions == []

  _run(_case())


@pytest.mark.parametrize("restart_exception_type", _RELAY_RESTART_EXCEPTIONS)
def test_mint_and_submit_returns_typed_restart_error_and_revokes_grant(
  tmp_path: Path,
  restart_exception_type: type[Exception],
) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = RestartBlockedRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ],
      exception_type=restart_exception_type,
    )

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Update the model",
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error == {
      "code": "relay_restart_in_progress",
      "message": "Excel MCP relay restart in progress; retry after gateway restart",
    }
    assert len(relay.submissions) == 1
    delegation_id = relay.submissions[0]["tool_input"]["delegation_id"]
    grant = await store.get_delegation_grant(delegation_id)
    assert grant is not None
    assert grant.revoked_at is not None

  _run(_case())


def test_mint_and_submit_does_not_classify_restart_message_lookalike(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    secret = "CUSTOM-ACTIVE-CREDENTIAL-excel-submit-8f21d7"
    store = _store(tmp_path)
    relay = RestartBlockedRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ],
      exception_type=_RestartMessageLookalike,
      exception_message=f"relay_restart_in_progress {secret}",
    )

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Update the model",
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error == {
      "code": "internal_error",
      "message": "Excel relay submission failed",
    }
    assert secret not in str(error)
    delegation_id = relay.submissions[0]["tool_input"]["delegation_id"]
    grant = await store.get_delegation_grant(delegation_id)
    assert grant is not None
    assert grant.revoked_at is not None

  _run(_case())


def test_mint_and_submit_revoke_failure_response_is_value_free(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    secret = "CUSTOM-ACTIVE-CREDENTIAL-excel-revoke-8f21d7"
    store = _store(tmp_path)
    relay = RestartBlockedRelay(
      workbooks=[{
        "name": "Budget.xlsx",
        "gateway_session_id": "excel-session-1",
      }],
      exception_type=_RestartMessageLookalike,
      exception_message="submit failed",
    )

    async def fail_revoke(_delegation_id: str) -> None:
      raise RuntimeError(secret)

    monkeypatch.setattr(store, "revoke_delegation_grant", fail_revoke)
    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Update the model",
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error == {
      "code": "internal_error",
      "message": "Failed to revoke unsubmitted delegation grant",
    }
    assert secret not in str(error)

  _run(_case())


@pytest.mark.parametrize(
  "restart_exceptions",
  [(), (RuntimeError("not-a-type"),), (KeyboardInterrupt,)],
)
def test_excel_dispatch_requires_explicit_exception_type_binding(
  restart_exceptions: tuple[Any, ...],
) -> None:
  with pytest.raises(
    TypeError,
    match="relay_restart_exceptions must contain only Exception types",
  ):
    make_message_excel_agent_handler(
      relay=object(),
      approval_store=object(),
      user_id="alice",
      gateway_session_id="session-1",
      relay_restart_exceptions=restart_exceptions,
    )


@pytest.mark.parametrize("restart_exception_type", _RELAY_RESTART_EXCEPTIONS)
def test_message_handler_preserves_typed_restart_error_and_revocation(
  tmp_path: Path,
  restart_exception_type: type[Exception],
) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = RestartBlockedRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ],
      exception_type=restart_exception_type,
    )
    handler = _handler(relay, store)

    result, error = await handler({"text": "Update the model"})

    assert result is None
    assert error == {
      "code": "relay_restart_in_progress",
      "message": "Excel MCP relay restart in progress; retry after gateway restart",
    }
    delegation_id = relay.submissions[0]["tool_input"]["delegation_id"]
    grant = await store.get_delegation_grant(delegation_id)
    assert grant is not None
    assert grant.revoked_at is not None
    assert relay.result_calls == []

  _run(_case())


def test_validate_requested_ceiling_accepts_allowed_subset() -> None:
  ceiling, error = validate_requested_ceiling(["state_write", "read", "read"])

  assert error is None
  assert ceiling == frozenset({"state_write", "read"})


def test_mint_and_submit_rejects_disallowed_ceiling_without_grant_or_submit(tmp_path: Path) -> None:
  async def _case(default_ceiling: frozenset[str]) -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
        }
      ]
    )

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Update the model",
      default_ceiling=default_ceiling,
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error is not None
    assert error["code"] == "invalid_input"
    assert _delegation_count(tmp_path) == 0
    assert relay.submissions == []

  _run(_case(frozenset({"read", "external_write"})))
  _run(_case(frozenset({"read", "unknown"})))


def test_mint_and_submit_ambiguous_target_returns_error_without_grant_or_submit(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {"name": "Budget.xlsx", "session": "workbook-session-1", "gateway_session_id": "excel-session-1", "detached": False},
        {"name": "Forecast.xlsx", "session": "workbook-session-2", "gateway_session_id": "excel-session-2", "detached": False},
      ]
    )

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Summarize the workbook",
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error is not None
    assert error["code"] == "ambiguous_target"
    assert _delegation_count(tmp_path) == 0
    assert relay.submissions == []
    assert relay.result_calls == []

  _run(_case())


def test_mint_and_submit_ignores_explicitly_non_live_workbooks(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "detached": False,
          "live": False,
        },
      ]
    )

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Summarize the workbook",
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error is not None
    assert error["code"] == "no_excel_session"
    assert _delegation_count(tmp_path) == 0
    assert relay.submissions == []
    assert relay.result_calls == []

  _run(_case())


def test_mint_and_submit_not_connected_returns_error_without_grant_or_submit(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {"name": "Budget.xlsx", "session": "workbook-session-1", "gateway_session_id": "excel-session-1", "detached": False}
      ]
    )

    submitted, error = await mint_and_submit(
      relay=relay,
      approval_store=store,
      user_id="alice",
      gateway_session_id="orchestrator-session-1",
      text="Summarize",
      workbook="Missing.xlsx",
      relay_restart_exceptions=_RELAY_RESTART_EXCEPTIONS,
    )

    assert submitted is None
    assert error is not None
    assert error["code"] == "no_excel_session"
    assert _delegation_count(tmp_path) == 0
    assert relay.submissions == []
    assert relay.result_calls == []

  _run(_case())


def test_poll_result_done_returns_payload_and_polls_as_excel_session() -> None:
  async def _case() -> None:
    relay = FakeRelay(
      workbooks=[],
      results=[
        (
          "ok",
          {
            "state": "done",
            "result": {
              "text": "I updated the workbook.",
              "escalations": [{"approval_id": "approval-1"}],
            },
          },
        )
      ],
    )
    relay.owner_session = "excel-session-1"

    result, error = await poll_result(
      relay=relay,
      request_id="request-1",
      excel_session_id="excel-session-1",
      user_id="alice",
      timeout_s=1.0,
      poll_interval_seconds=0.001,
    )

    assert error is None
    assert result == {
      "text": "I updated the workbook.",
      "escalations": [{"approval_id": "approval-1"}],
    }
    assert relay.result_calls == [
      {"request_id": "request-1", "gateway_session_id": "excel-session-1", "user_id": "alice"}
    ]

  _run(_case())


def test_poll_result_failed_returns_error() -> None:
  async def _case() -> None:
    secret = "CUSTOM-ACTIVE-CREDENTIAL-excel-result-8f21d7"
    relay = FakeRelay(
      workbooks=[],
      results=[("ok", {"state": "failed", "error": {"message": secret}})],
    )
    relay.owner_session = "excel-session-1"

    result, error = await poll_result(
      relay=relay,
      request_id="request-1",
      excel_session_id="excel-session-1",
      user_id="alice",
      timeout_s=1.0,
      poll_interval_seconds=0.001,
    )

    assert result is None
    assert error is not None
    assert error["code"] == "excel_agent_failed"
    assert error["message"] == "Excel agent request failed"
    assert error["request_id"] == "request-1"
    assert "delegation_id" not in error
    assert secret not in str(error)

  _run(_case())


def test_poll_result_exception_is_value_free_and_keeps_request_id() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-excel-poll-8f21d7"

  class ExplodingRelay(FakeRelay):
    async def result(self, *_args: Any, **_kwargs: Any):
      raise RuntimeError(secret)

  result, error = _run(poll_result(
    relay=ExplodingRelay(workbooks=[]),
    request_id="request-canary",
    excel_session_id="excel-session-1",
    user_id="alice",
    timeout_s=1.0,
  ))

  assert result is None
  assert error == {
    "code": "internal_error",
    "message": "Excel relay result polling failed",
    "request_id": "request-canary",
  }
  assert secret not in str(error)


def test_poll_result_pending_until_deadline_returns_timeout_error() -> None:
  async def _case() -> None:
    relay = FakeRelay(
      workbooks=[],
      results=[("ok", {"state": "pending"})],
    )
    relay.owner_session = "excel-session-1"

    result, error = await poll_result(
      relay=relay,
      request_id="request-1",
      excel_session_id="excel-session-1",
      user_id="alice",
      timeout_s=0.003,
      poll_interval_seconds=0.001,
    )

    assert result is None
    assert error is not None
    assert error["code"] == "timeout"
    assert error["request_id"] == "request-1"
    assert len(relay.result_calls) >= 1
    assert all(call["gateway_session_id"] == "excel-session-1" for call in relay.result_calls)

  _run(_case())


def test_happy_path_discovers_single_workbook_mints_grant_and_returns_result(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {
          "name": "Budget.xlsx",
          "session": "workbook-session-1",
          "gateway_session_id": "excel-session-1",
          "active": True,
          "connected_at": 1.0,
          "detached": False,
          "detach_grace_deadline": None,
        }
      ],
      results=[
        (
          "ok",
          {
            "state": "done",
            "result": {
              "text": "I updated the workbook.",
              "diagnostics": {"used_compaction": False},
            },
          },
        )
      ],
    )
    handler = _handler(relay, store)

    result, error = await handler({"text": "Update the model"})

    assert error is None
    assert result is not None
    assert result["text"] == "I updated the workbook."
    assert result["diagnostics"] == {"used_compaction": False}
    assert isinstance(result["delegation_id"], str)
    assert isinstance(result["request_id"], str)

    stored = await store.get_delegation_grant(result["delegation_id"])
    assert stored is not None
    assert stored.delegator_user_id == "alice"
    assert stored.delegator_session_id == "orchestrator-session-1"
    assert stored.delegator_run_id == "run-1"
    assert stored.delegator_profile == "orchestrator"
    assert stored.delegator_channel == "excel"
    assert stored.bound_excel_session_id == "excel-session-1"
    assert stored.bound_relay_request_id == result["request_id"]
    assert stored.bound_workbook == "Budget.xlsx"
    assert stored.tool_class_ceiling == frozenset({"read", "pure_transform", "artifact_write", "state_write"})
    assert stored.window_seconds == 600
    assert stored.exclude_external_write_bypass is True
    assert stored.created_at is not None
    assert stored.expires_at is not None

    assert len(relay.submissions) == 1
    submission = relay.submissions[0]
    assert submission["tool_name"] == "send_chat_message"
    assert submission["target_session"] == "excel-session-1"
    assert submission["gateway_session_id"] == "orchestrator-session-1"
    assert submission["user_id"] == "alice"
    assert submission["kind"] == "chat"
    assert submission["request_id"] == result["request_id"]
    assert submission["tool_input"] == {
      "text": "Update the model",
      "force_compaction": False,
      "delegation_id": result["delegation_id"],
    }
    assert relay.result_calls == [
      {"request_id": result["request_id"], "gateway_session_id": "excel-session-1", "user_id": "alice"}
    ]

  _run(_case())


def test_ambiguous_workbooks_without_target_returns_error_without_grant_or_submit(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {"name": "Budget.xlsx", "session": "workbook-session-1", "gateway_session_id": "excel-session-1", "detached": False},
        {"name": "Forecast.xlsx", "session": "workbook-session-2", "gateway_session_id": "excel-session-2", "detached": False},
      ]
    )
    handler = _handler(relay, store)

    result, error = await handler({"text": "Summarize the workbook"})

    assert result is None
    assert error is not None
    assert error["code"] == "ambiguous_target"
    assert _delegation_count(tmp_path) == 0
    assert relay.submissions == []

  _run(_case())


def test_specified_workbook_not_connected_returns_error_without_grant_or_submit(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {"name": "Budget.xlsx", "session": "workbook-session-1", "gateway_session_id": "excel-session-1", "detached": False}
      ]
    )
    handler = _handler(relay, store)

    result, error = await handler({"text": "Summarize", "workbook": "Missing.xlsx"})

    assert result is None
    assert error is not None
    assert error["code"] == "no_excel_session"
    assert _delegation_count(tmp_path) == 0
    assert relay.submissions == []

  _run(_case())


def test_failed_relay_result_returns_error(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {"name": "Budget.xlsx", "session": "workbook-session-1", "gateway_session_id": "excel-session-1", "detached": False}
      ],
      results=[("ok", {"state": "failed", "error": {"message": "approval denied"}})],
    )
    handler = _handler(relay, store)

    result, error = await handler({"text": "Try a risky action"})

    assert result is None
    assert error is not None
    assert error["code"] == "excel_agent_failed"
    assert error["message"] == "Excel agent request failed"
    assert len(relay.submissions) == 1

  _run(_case())


def test_pending_result_until_tiny_deadline_returns_timeout_error(tmp_path: Path) -> None:
  async def _case() -> None:
    store = _store(tmp_path)
    relay = FakeRelay(
      workbooks=[
        {"name": "Budget.xlsx", "session": "workbook-session-1", "gateway_session_id": "excel-session-1", "detached": False}
      ],
      results=[("ok", {"state": "pending"})],
    )
    handler = _handler(relay, store, poll_interval_seconds=0.001)

    result, error = await handler({"text": "Keep working", "timeout_s": 0.003})

    assert result is None
    assert error is not None
    assert error["code"] == "timeout"
    assert len(relay.submissions) == 1
    assert len(relay.result_calls) >= 1

  _run(_case())
