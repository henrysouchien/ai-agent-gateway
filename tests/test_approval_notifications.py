import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.approval_notifications import ApprovalNotificationDestination  # noqa: E402
from agent_gateway.approval_policy import (  # noqa: E402
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest,
  ApprovalRequestPayload,
  RunContext,
  build_approval_request,
  utc_now,
)
from agent_gateway.approval_store import SQLiteApprovalStore  # noqa: E402
from agent_gateway.approvals import _approval_request_to_dict  # noqa: E402
from agent_gateway.tool_dispatcher_approval_lifecycle import run_approval_lifecycle  # noqa: E402


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def _request(
  *,
  tool_class: str = "external_write",
  tool_name: str = "execute_trade",
  notification_policy: str = "auto",
  tool_args_redacted: dict[str, Any] | None = None,
) -> ApprovalRequest:
  request = build_approval_request(
    tool_call_id=f"tool-{tool_name}",
    tool_name=tool_name,
    tool_class=tool_class,  # type: ignore[arg-type]
    tool_args_redacted=tool_args_redacted or {"ticker": "SGOV"},
    args_hash="hash-approval",
    run_context=RunContext(
      user_id="alice",
      request_id="request-1",
      session_id="session-1",
      run_id="bg-1",
      profile="analyst",
      channel="web",
    ),
  )
  return replace(request, notification_policy=notification_policy)  # type: ignore[arg-type]


async def _persist_pending(store: SQLiteApprovalStore, request: ApprovalRequest) -> ApprovalRequest:
  await store.create(request)
  return await store.transition_state(
    request.approval_id,
    "pending_user",
    expires_at=utc_now(),
    expected_state_version=request.state_version,
  )


def test_pending_interrupt_approval_without_destination_records_skipped_projection(tmp_path: Path) -> None:
  async def _case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request = await _persist_pending(store, _request(tool_class="irreversible"))

    projection = await store.enqueue_pending_approval_notification(request)
    stored = await store.get(request.approval_id)
    rows = await store.list_approval_notification_outbox(request.approval_id)

    assert projection == {"state": "skipped_no_destination", "channels": []}
    assert stored is not None
    assert stored.notification == projection
    assert len(rows) == 1
    assert rows[0]["state"] == "skipped_no_destination"
    assert rows[0]["channel"] == ""
    assert rows[0]["destination"] == ""

  _run(_case())


def test_non_interrupting_pending_approval_records_skipped_policy(tmp_path: Path) -> None:
  async def _case() -> None:
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    request = await _persist_pending(store, _request(tool_class="state_write"))

    projection = await store.enqueue_pending_approval_notification(request)

    assert projection == {"state": "skipped_policy", "channels": []}

  _run(_case())


def test_notification_outbox_dedupes_destinations_and_keeps_message_redacted(tmp_path: Path) -> None:
  async def _case() -> None:
    def resolver(_request: ApprovalRequest):
      return [
        ApprovalNotificationDestination(channel="telegram", destination="chat-1"),
        {"channel": "telegram", "destination": "chat-1"},
        {"channel": "telegram", "destination": "chat-unverified", "verified": False},
      ]

    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3", notification_destination_resolver=resolver)
    request = await _persist_pending(
      store,
      _request(
        tool_class="irreversible",
        tool_args_redacted={
          "account_id": "acct-private",
          "preview_id": "preview-private",
          "order_id": "order-private",
          "token": "secret-token",
          "path": "/private/path/model.xlsx",
        },
      ),
    )

    await store.enqueue_pending_approval_notification(request)
    await store.enqueue_pending_approval_notification(request)

    rows = await store.list_approval_notification_outbox(request.approval_id)
    projection = await store.get_approval_notification_projection(request.approval_id)

    assert len(rows) == 1
    assert rows[0]["state"] == "pending"
    assert rows[0]["channel"] == "telegram"
    assert projection == {"state": "pending", "channels": ["telegram"]}
    message = rows[0]["message"]
    assert "execute_trade" in message
    for private_value in ("acct-private", "preview-private", "order-private", "secret-token", "/private/path"):
      assert private_value not in message

  _run(_case())


def test_delivery_success_and_failure_preserve_pending_approval_state(tmp_path: Path) -> None:
  async def _case() -> None:
    sent_rows: list[dict[str, Any]] = []

    async def sender(row: dict[str, Any]) -> None:
      sent_rows.append(row)

    success_store = SQLiteApprovalStore(
      tmp_path / "success.sqlite3",
      notification_destination_resolver=lambda _request: [
        ApprovalNotificationDestination(channel="telegram", destination="chat-1")
      ],
      notification_sender=sender,
    )
    success_request = await _persist_pending(success_store, _request(tool_class="irreversible"))
    await success_store.enqueue_pending_approval_notification(success_request)

    assert await success_store.deliver_pending_approval_notifications() == 1
    success_projection = await success_store.get_approval_notification_projection(success_request.approval_id)
    success_record = await success_store.get(success_request.approval_id)

    assert len(sent_rows) == 1
    assert success_projection is not None
    assert success_projection["state"] == "sent"
    assert success_projection["channels"] == ["telegram"]
    assert success_projection["last_sent_at"]
    assert success_record is not None
    assert success_record.state == "pending_user"

    async def failing_sender(_row: dict[str, Any]) -> None:
      raise RuntimeError("temporary telegram outage")

    failed_store = SQLiteApprovalStore(
      tmp_path / "failed.sqlite3",
      notification_destination_resolver=lambda _request: [
        ApprovalNotificationDestination(channel="telegram", destination="chat-1")
      ],
      notification_sender=failing_sender,
    )
    failed_request = await _persist_pending(failed_store, _request(tool_class="irreversible"))
    await failed_store.enqueue_pending_approval_notification(failed_request)

    assert await failed_store.deliver_pending_approval_notifications() == 1
    failed_projection = await failed_store.get_approval_notification_projection(failed_request.approval_id)
    failed_record = await failed_store.get(failed_request.approval_id)

    assert failed_projection == {"state": "failed_retryable", "channels": ["telegram"]}
    assert failed_record is not None
    assert failed_record.state == "pending_user"

  _run(_case())


def test_retry_failed_approval_notifications_requeues_without_duplicate_sends(tmp_path: Path) -> None:
  async def _case() -> None:
    send_attempts: list[dict[str, Any]] = []

    async def flaky_sender(row: dict[str, Any]) -> None:
      send_attempts.append(row)
      if len(send_attempts) == 1:
        raise RuntimeError("temporary telegram outage")

    store = SQLiteApprovalStore(
      tmp_path / "approvals.sqlite3",
      notification_destination_resolver=lambda _request: [
        ApprovalNotificationDestination(channel="telegram", destination="chat-1")
      ],
      notification_sender=flaky_sender,
    )
    request = await _persist_pending(store, _request(tool_class="irreversible"))
    await store.enqueue_pending_approval_notification(request)

    assert await store.deliver_pending_approval_notifications() == 1
    assert await store.deliver_pending_approval_notifications() == 0
    failed_rows = await store.list_approval_notification_outbox(request.approval_id)
    assert failed_rows[0]["state"] == "failed_retryable"
    assert failed_rows[0]["attempt_count"] == 1
    assert failed_rows[0]["last_error"] == "RuntimeError"

    retry = await store.retry_failed_approval_notifications(request.approval_id)
    assert retry == {
      "approval_id": request.approval_id,
      "requeued": 1,
      "notification": {"state": "pending", "channels": ["telegram"]},
    }
    requeued_rows = await store.list_approval_notification_outbox(request.approval_id)
    assert requeued_rows[0]["state"] == "pending"
    assert requeued_rows[0]["attempt_count"] == 1
    assert requeued_rows[0]["last_error"] is None

    assert await store.deliver_pending_approval_notifications() == 1
    sent_rows = await store.list_approval_notification_outbox(request.approval_id)
    sent_projection = await store.get_approval_notification_projection(request.approval_id)
    assert sent_rows[0]["state"] == "sent"
    assert sent_rows[0]["attempt_count"] == 2
    assert sent_projection is not None
    assert sent_projection["state"] == "sent"
    assert sent_projection["channels"] == ["telegram"]
    assert len(send_attempts) == 2

    retry_after_success = await store.retry_failed_approval_notifications(request.approval_id)
    assert retry_after_success["requeued"] == 0
    assert retry_after_success["notification"] is not None
    assert retry_after_success["notification"]["state"] == "sent"

  _run(_case())


def test_notification_retry_and_delivery_skip_resolved_approvals(tmp_path: Path) -> None:
  async def _case() -> None:
    sent_rows: list[dict[str, Any]] = []

    async def sender(row: dict[str, Any]) -> None:
      sent_rows.append(row)

    sent_store = SQLiteApprovalStore(
      tmp_path / "sent.sqlite3",
      notification_destination_resolver=lambda _request: [
        ApprovalNotificationDestination(channel="telegram", destination="chat-1")
      ],
      notification_sender=sender,
    )
    sent_request = await _persist_pending(sent_store, _request(tool_class="irreversible"))
    await sent_store.enqueue_pending_approval_notification(sent_request)
    await sent_store.transition_state(
      sent_request.approval_id,
      "approved",
      expected_state_version=sent_request.state_version,
    )

    assert await sent_store.deliver_pending_approval_notifications() == 0
    assert sent_rows == []
    pending_rows = await sent_store.list_approval_notification_outbox(sent_request.approval_id)
    assert pending_rows[0]["state"] == "pending"

    async def failing_sender(_row: dict[str, Any]) -> None:
      raise RuntimeError("temporary telegram outage")

    failed_store = SQLiteApprovalStore(
      tmp_path / "failed.sqlite3",
      notification_destination_resolver=lambda _request: [
        ApprovalNotificationDestination(channel="telegram", destination="chat-1")
      ],
      notification_sender=failing_sender,
    )
    failed_request = await _persist_pending(failed_store, _request(tool_class="irreversible"))
    await failed_store.enqueue_pending_approval_notification(failed_request)
    assert await failed_store.deliver_pending_approval_notifications() == 1
    await failed_store.transition_state(
      failed_request.approval_id,
      "denied",
      expected_state_version=failed_request.state_version,
    )

    retry = await failed_store.retry_failed_approval_notifications(failed_request.approval_id)
    assert retry["requeued"] == 0
    assert retry["notification"] == {"state": "failed_retryable", "channels": ["telegram"]}
    failed_rows = await failed_store.list_approval_notification_outbox(failed_request.approval_id)
    assert failed_rows[0]["state"] == "failed_retryable"

  _run(_case())


def test_lifecycle_persistence_failure_emits_no_notification_intent() -> None:
  async def _case() -> None:
    class _FailingStore:
      def __init__(self) -> None:
        self.notifications: list[ApprovalRequest] = []

      async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        return request

      async def update_request(self, request: ApprovalRequest) -> ApprovalRequest:
        return request

      async def transition_state(self, *_args: Any, **_kwargs: Any) -> ApprovalRequest:
        raise RuntimeError("approval persistence failed")

      async def enqueue_pending_approval_notification(self, request: ApprovalRequest) -> None:
        self.notifications.append(request)

    class _Policy:
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
          route_target_type="pending_tools",
          expiry_seconds=600,
        )

    store = _FailingStore()

    with pytest.raises(RuntimeError, match="approval persistence failed"):
      await run_approval_lifecycle(
        store=store,
        policy=_Policy(),
        session=object(),
        tool_call_id="tool-1",
        tool_name="execute_trade",
        tool_input={"preview_id": "raw-preview"},
        qualifier="execute_trade",
        reason="needs approval",
        allow_persistent=False,
        resolve_run_context_fn=lambda: RunContext(user_id="alice", request_id="request-1"),
        current_skill_fn=lambda: None,
        redact_for_approval_request_fn=lambda _tool_name, _tool_input: ({}, "hash"),
        resolve_tool_class_fn=lambda _tool_name: "irreversible",
        effective_trade_approval_decision_fn=lambda _tool_name, _tool_args, decision: decision,
        await_user_approval_via_pending_tools_fn=lambda *_args, **_kwargs: None,
        approval_queue_timeout_seconds_fn=lambda _seconds: 0.01,
      )

    assert store.notifications == []

  _run(_case())


def test_approval_request_to_dict_preserves_projection_and_strips_internal_policy() -> None:
  request = replace(
    _request(tool_class="irreversible", notification_policy="interrupt"),
    state="pending_user",
    notification={"state": "sent", "channels": ["telegram"], "last_sent_at": "2026-07-03T15:00:50Z"},
  )

  payload = _approval_request_to_dict(request)

  assert payload["notification"] == {
    "state": "sent",
    "channels": ["telegram"],
    "last_sent_at": "2026-07-03T15:00:50Z",
  }
  assert "notification_policy" not in payload
