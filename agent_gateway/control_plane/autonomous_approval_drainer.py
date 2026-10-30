from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, TypedDict

from agent_gateway.approval_policy import ApprovalRequest
from agent_gateway.autonomous_runner import (
  AutonomousRegistry,
  AutonomousTask,
)

from .autonomous_approval_delivery import (
  autonomous_run_accepts_approval_decisions,
  deliver_autonomous_approval_outbox,
  fail_quarantined_autonomous_approval_owner,
  require_matching_autonomous_delivery,
)


log = logging.getLogger(
  "agent_gateway.autonomous_approval_drainer"
)
AUTONOMOUS_APPROVAL_DRAIN_FAILURE_BACKOFF_SECONDS = 1.0
AUTONOMOUS_APPROVAL_DRAIN_SHUTDOWN_SECONDS = 5.0

class _AutonomousApprovalDelivery(TypedDict):
  delivery_sequence: int
  approval_id: str
  tool_call_id: str
  nonce: str
  task_id: str
  control_run_id: str
  session_id: str
  channel_id: str
  approved: bool
  decided_at_ns: int
  last_attempt_ns: int | None
  state: str
  attempt_count: int


class _ReadAutonomousApprovalDeliveryRecoveryWindow(Protocol):
  async def __call__(self) -> dict[str, int]: ...


class _SelectPendingAutonomousApprovalDeliveries(Protocol):
  async def __call__(
    self,
    *,
    after_sequence: int,
    through_sequence: int | None,
  ) -> list[_AutonomousApprovalDelivery]: ...


class _GetApprovalRequest(Protocol):
  async def __call__(
    self,
    approval_id: str,
  ) -> ApprovalRequest | None: ...


class _GetAutonomousApprovalDelivery(Protocol):
  async def __call__(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
  ) -> _AutonomousApprovalDelivery | None: ...


class _WriteAutonomousApprovalDeliveryFailure(Protocol):
  async def __call__(
    self,
    approval_id: str,
    *,
    tool_call_id: str,
    nonce: str,
    error: str,
  ) -> _AutonomousApprovalDelivery: ...


class PermanentAutonomousApprovalDeliveryError(RuntimeError):
  """The durable row cannot ever be delivered to its recorded owner."""


def wake_autonomous_approval_delivery(app_state: Any) -> bool:
  coordinator = getattr(
    app_state,
    "autonomous_approval_delivery_coordinator",
    None,
  )
  wake = getattr(coordinator, "wake", None)
  if not callable(wake):
    return False
  return wake() is True


class AutonomousApprovalDeliveryCoordinator:
  """Event-driven, bounded recovery for durable child approval delivery."""

  def __init__(
    self,
    *,
    store: Any,
    registry: AutonomousRegistry,
    shutdown_timeout_seconds: float = (
      AUTONOMOUS_APPROVAL_DRAIN_SHUTDOWN_SECONDS
    ),
  ) -> None:
    self._store = store
    self._registry = registry
    self._shutdown_timeout_seconds = float(
      shutdown_timeout_seconds
    )
    self._wake_event = asyncio.Event()
    self._wake_generation = 0
    self._task: asyncio.Task[None] | None = None
    self._retry_handle: asyncio.TimerHandle | None = None
    self._drain_failure_count = 0
    self._fatal_error: str | None = None
    self._closing = False

  @property
  def fatal_error(self) -> str | None:
    """The latest drain error; None after a successful drain."""
    return self._fatal_error

  def start(self) -> None:
    if self._closing:
      raise RuntimeError(
        "autonomous approval drainer is already closed"
      )
    if self._task is not None:
      raise RuntimeError(
        "autonomous approval drainer is already started"
      )
    self._task = asyncio.create_task(
      self._run(),
      name="autonomous-approval-delivery-drainer",
    )
    self.wake()

  def wake(self) -> bool:
    if self._closing:
      return False
    self._wake_generation += 1
    self._wake_event.set()
    return True

  async def shutdown(self) -> None:
    task = self._task
    if task is None:
      self._closing = True
      return
    self._closing = True
    retry_handle = self._retry_handle
    self._retry_handle = None
    if retry_handle is not None:
      retry_handle.cancel()
    self._wake_event.set()
    try:
      await asyncio.wait_for(
        task,
        timeout=self._shutdown_timeout_seconds,
      )
    except TimeoutError:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)
      log.error(
        "Autonomous approval drainer exceeded its shutdown bound"
      )
    finally:
      self._task = None

  async def _run(self) -> None:
    while True:
      await self._wake_event.wait()
      observed_generation = self._wake_generation
      self._wake_event.clear()
      if self._closing:
        return
      try:
        await self.drain_once()
      except asyncio.CancelledError:
        raise
      except Exception as exc:
        log.exception(
          "Bounded autonomous approval delivery drain failed"
        )
        self._record_drain_failure(exc)
      else:
        self._drain_failure_count = 0
        if self._fatal_error is not None:
          log.info("Autonomous approval delivery drain recovered")
          self._fatal_error = None
      if (
        observed_generation != self._wake_generation
        and not self._closing
      ):
        self._wake_event.set()

  def _record_drain_failure(self, failure: Exception) -> None:
    self._drain_failure_count += 1
    # Reported, never latched: the durable outbox still holds every row, so
    # the drainer keeps retrying with backoff and health clears on success.
    self._fatal_error = (
      "autonomous approval recovery failing after "
      f"{self._drain_failure_count} attempts: "
      f"{type(failure).__name__}: {failure}"
    )
    self._arm_retry(
      min(
        AUTONOMOUS_APPROVAL_DRAIN_FAILURE_BACKOFF_SECONDS
        * (2 ** min(self._drain_failure_count - 1, 16)),
        16.0,
      )
    )

  async def drain_once(self) -> int:
    recovery_window_reader: _ReadAutonomousApprovalDeliveryRecoveryWindow | None = getattr(
      self._store,
      "autonomous_approval_delivery_recovery_window",
      None,
    )
    selector: _SelectPendingAutonomousApprovalDeliveries | None = getattr(
      self._store,
      "list_pending_autonomous_approval_deliveries",
      None,
    )
    if (
      not callable(recovery_window_reader)
      or not callable(selector)
    ):
      raise RuntimeError(
        "autonomous approval bounded recovery selector is unavailable"
      )
    recovery_window = await recovery_window_reader()
    if (
      not isinstance(recovery_window, dict)
      or set(recovery_window)
      != {"high_water", "observed_at_ns"}
    ):
      raise RuntimeError(
        "autonomous approval recovery window is invalid"
      )
    high_water = recovery_window["high_water"]
    observed_at_ns = recovery_window["observed_at_ns"]
    if (
      type(high_water) is not int
      or high_water < 0
      or type(observed_at_ns) is not int
      or observed_at_ns < 1
    ):
      raise RuntimeError(
        "autonomous approval recovery window is invalid"
      )
    after_sequence = 0
    attempted = 0
    unrecorded: Exception | None = None
    undelivered: Exception | None = None
    while after_sequence < high_water and not self._closing:
      deliveries = await selector(
        after_sequence=after_sequence,
        through_sequence=high_water,
      )
      if not isinstance(deliveries, list):
        raise RuntimeError(
          "autonomous approval pending selector returned invalid data"
        )
      if not deliveries:
        break
      for delivery in deliveries:
        if self._closing:
          break
        if not isinstance(delivery, dict):
          raise RuntimeError(
            "autonomous approval pending row is malformed"
          )
        delivery_sequence = delivery.get("delivery_sequence")
        if (
          type(delivery_sequence) is not int
          or not after_sequence < delivery_sequence <= high_water
        ):
          raise RuntimeError(
            "autonomous approval pending selector broke ordering"
          )
        after_sequence = delivery_sequence
        attempted += 1
        try:
          await self._deliver_one(delivery)
        except asyncio.CancelledError:
          raise
        except PermanentAutonomousApprovalDeliveryError as exc:
          # A row that can never be delivered is quarantined by the store, and
          # only then does its owner run fail, so it never waits on a decision
          # that cannot arrive. A quarantine the store did not take is
          # reported at drain level and fails nothing.
          if await self._quarantine_failure(delivery, exc) is None:
            unrecorded = exc
          else:
            await fail_quarantined_autonomous_approval_owner(
              self._registry,
              task_id=delivery.get("task_id"),
              failure=exc,
            )
        except Exception as exc:
          # A failed publish is recorded by the store and the row stays
          # pending, so the drain retries it; a failure the store did not
          # record is reported at drain level.
          try:
            outcome = await self._record_failure(delivery, exc)
          except asyncio.CancelledError:
            raise
          except Exception as record_exc:
            log.exception(
              "Autonomous approval delivery failure could not be recorded"
            )
            unrecorded = record_exc
            continue
          if outcome is not None and outcome.get("state") == "pending":
            undelivered = exc
    if unrecorded is not None:
      raise RuntimeError(
        "autonomous approval delivery failure was not recorded by the store"
      ) from unrecorded
    if undelivered is not None:
      raise RuntimeError(
        "autonomous approval delivery failed and remains pending"
      ) from undelivered
    return attempted

  def _arm_retry(self, delay_seconds: float) -> None:
    if self._closing:
      return
    bounded_delay = max(0.001, min(float(delay_seconds), 30.0))
    loop = asyncio.get_running_loop()
    scheduled_at = loop.time() + bounded_delay
    current = self._retry_handle
    if (
      current is not None
      and not current.cancelled()
      and current.when() <= scheduled_at
    ):
      return
    if current is not None:
      current.cancel()
    self._retry_handle = loop.call_later(
      bounded_delay,
      self._retry_wake,
    )

  def _retry_wake(self) -> None:
    self._retry_handle = None
    self.wake()

  async def _deliver_one(
    self,
    delivery: _AutonomousApprovalDelivery,
  ) -> None:
    if not isinstance(delivery, dict):
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval pending row is malformed"
      )
    task_id = delivery.get("task_id")
    if type(task_id) is not str or not task_id:
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval pending row has no task authority"
      )
    record = self._registry._tasks.get(task_id)
    if type(record) is not AutonomousTask:
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval owner task is unavailable"
      )
    if not autonomous_run_accepts_approval_decisions(record):
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval owner task is not active"
      )
    owner_user_id = str(record.owner_user_id or "").strip()
    if owner_user_id:
      live_role = getattr(self._registry, "live_user_role", None)
      lookup = getattr(self._registry, "_mcp_user_key_lookup", None)
      if callable(live_role):
        role = live_role(owner_user_id, getattr(record, "user_email", None))
        if role is None and lookup is not None:
          raise PermanentAutonomousApprovalDeliveryError(
            f"User {owner_user_id!r} is no longer admitted on the roster; approval delivery stopped"
          )
      elif callable(lookup):
        try:
          entry = lookup(owner_user_id, getattr(record, "user_email", None))
        except SystemExit:
          entry = None
        if not isinstance(entry, dict):
          raise PermanentAutonomousApprovalDeliveryError(
            f"User {owner_user_id!r} is no longer admitted on the roster; approval delivery stopped"
          )
    approval_id = delivery.get("approval_id")
    tool_call_id = delivery.get("tool_call_id")
    nonce = delivery.get("nonce")
    approved = delivery.get("approved")
    if (
      type(approval_id) is not str
      or not approval_id
      or type(tool_call_id) is not str
      or not tool_call_id
      or type(nonce) is not str
      or not nonce
      or type(approved) is not bool
    ):
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval pending row has invalid decision authority"
      )
    get_request: _GetApprovalRequest | None = getattr(self._store, "get", None)
    if not callable(get_request):
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval request lookup is unavailable"
      )
    request_record = await get_request(approval_id)
    if request_record is None:
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval pending row has no durable request"
      )
    # A row is delivered only to the run it was created for; that match is
    # require_matching_autonomous_delivery's, and a mismatch is corruption, so the
    # row is quarantined rather than retried.
    try:
      require_matching_autonomous_delivery(
        record,
        delivery,
        request_record,
        approval_id=approval_id,
        tool_call_id=tool_call_id,
        nonce=nonce,
        approved=approved,
      )
    except (RuntimeError, ValueError) as exc:
      raise PermanentAutonomousApprovalDeliveryError(str(exc)) from exc
    await deliver_autonomous_approval_outbox(
      registry=self._registry,
      store=self._store,
      record=record,
      request_record=request_record,
      delivery=delivery,
      approval_id=approval_id,
      tool_call_id=tool_call_id,
      nonce=nonce,
      approved=approved,
      user_id=str(record.owner_user_id or "").strip(),
      channel=request_record.channel,
    )

  async def _record_failure(
    self,
    delivery: _AutonomousApprovalDelivery,
    failure: Exception,
  ) -> _AutonomousApprovalDelivery | None:
    """Record one delivery failure; None means the row is gone, so there is nothing to retry.

    A recording failure raises to the caller, which reports it at drain level.
    """
    record_failure: _WriteAutonomousApprovalDeliveryFailure | None = getattr(
      self._store,
      "record_autonomous_approval_delivery_failure",
      None,
    )
    if not callable(record_failure):
      raise RuntimeError(
        "autonomous approval delivery failure recorder is unavailable"
      )
    get_delivery: _GetAutonomousApprovalDelivery | None = getattr(
      self._store,
      "get_autonomous_approval_delivery",
      None,
    )
    if callable(get_delivery):
      current = await get_delivery(
        delivery.get("approval_id"),
        tool_call_id=delivery.get("tool_call_id"),
        nonce=delivery.get("nonce"),
      )
      if current is None or current.get("state") == "acknowledged":
        return current
      prior_attempt_count = delivery.get("attempt_count")
      current_attempt_count = current.get("attempt_count")
      if (
        type(prior_attempt_count) is int
        and type(current_attempt_count) is int
        and current_attempt_count > prior_attempt_count
      ):
        return current
    outcome = await record_failure(
      delivery.get("approval_id"),
      tool_call_id=delivery.get("tool_call_id"),
      nonce=delivery.get("nonce"),
      error=f"{type(failure).__name__}: {failure}",
    )
    if not isinstance(outcome, dict):
      raise RuntimeError(
        "autonomous approval delivery failure recorder returned invalid data"
      )
    return outcome

  async def _quarantine_failure(
    self,
    delivery: _AutonomousApprovalDelivery,
    failure: Exception,
  ) -> _AutonomousApprovalDelivery | None:
    if not isinstance(delivery, dict):
      return None
    quarantine: _WriteAutonomousApprovalDeliveryFailure | None = getattr(
      self._store,
      "quarantine_autonomous_approval_delivery",
      None,
    )
    if not callable(quarantine):
      log.error(
        "Autonomous approval delivery quarantine is unavailable"
      )
      return None
    try:
      outcome = await quarantine(
        delivery.get("approval_id"),
        tool_call_id=delivery.get("tool_call_id"),
        nonce=delivery.get("nonce"),
        error=f"{type(failure).__name__}: {failure}",
      )
      return outcome if isinstance(outcome, dict) else None
    except Exception:
      log.exception(
        "Autonomous approval delivery could not be quarantined"
      )
      return None


__all__ = [
  "AUTONOMOUS_APPROVAL_DRAIN_SHUTDOWN_SECONDS",
  "AutonomousApprovalDeliveryCoordinator",
  "PermanentAutonomousApprovalDeliveryError",
  "wake_autonomous_approval_delivery",
]
