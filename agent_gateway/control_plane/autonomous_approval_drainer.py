from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from agent_gateway.autonomous_runner import (
  AutonomousRegistry,
  AutonomousTask,
)

from .autonomous_approval_delivery import (
  autonomous_run_accepts_approval_decisions,
  deliver_autonomous_approval_outbox,
)


log = logging.getLogger(
  "agent_gateway.autonomous_approval_drainer"
)
AUTONOMOUS_APPROVAL_DRAIN_BATCH_LIMIT = 64
AUTONOMOUS_APPROVAL_DRAIN_SHUTDOWN_SECONDS = 5.0
AUTONOMOUS_APPROVAL_FALLBACK_RETRY_MAX_ATTEMPTS = 5
AUTONOMOUS_APPROVAL_FALLBACK_RETRY_WINDOW_SECONDS = 60.0


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
    batch_limit: int = AUTONOMOUS_APPROVAL_DRAIN_BATCH_LIMIT,
    shutdown_timeout_seconds: float = (
      AUTONOMOUS_APPROVAL_DRAIN_SHUTDOWN_SECONDS
    ),
    fallback_retry_base_seconds: float = 1.0,
  ) -> None:
    if (
      isinstance(batch_limit, bool)
      or not isinstance(batch_limit, int)
      or not 1 <= batch_limit <= 256
    ):
      raise ValueError(
        "autonomous approval drain batch_limit must be 1..256"
      )
    if (
      isinstance(fallback_retry_base_seconds, bool)
      or not isinstance(fallback_retry_base_seconds, (int, float))
      or not 0 < fallback_retry_base_seconds <= 5
    ):
      raise ValueError(
        "autonomous approval fallback retry base must be >0 and <=5"
      )
    if (
      isinstance(shutdown_timeout_seconds, bool)
      or not isinstance(shutdown_timeout_seconds, (int, float))
      or not 0 < shutdown_timeout_seconds <= 30
    ):
      raise ValueError(
        "autonomous approval drain shutdown timeout must be >0 and <=30"
      )
    self._store = store
    self._registry = registry
    self._batch_limit = batch_limit
    self._shutdown_timeout_seconds = float(
      shutdown_timeout_seconds
    )
    self._fallback_retry_base_seconds = float(
      fallback_retry_base_seconds
    )
    self._wake_event = asyncio.Event()
    self._wake_generation = 0
    self._task: asyncio.Task[None] | None = None
    self._retry_handle: asyncio.TimerHandle | None = None
    self._fallback_failures: dict[
      tuple[object, object, object],
      tuple[int, float],
    ] = {}
    self._drain_failure_count = 0
    self._drain_first_failure_at: float | None = None
    self._fatal_error: str | None = None
    self._closing = False

  @property
  def fatal_error(self) -> str | None:
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
    if self._closing or self._fatal_error is not None:
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
        self._drain_first_failure_at = None
      if (
        observed_generation != self._wake_generation
        and not self._closing
        and self._fatal_error is None
      ):
        self._wake_event.set()

  def _record_drain_failure(self, failure: Exception) -> None:
    loop = asyncio.get_running_loop()
    now = loop.time()
    if self._drain_first_failure_at is None:
      self._drain_first_failure_at = now
    self._drain_failure_count += 1
    exhausted = (
      self._drain_failure_count
      >= AUTONOMOUS_APPROVAL_FALLBACK_RETRY_MAX_ATTEMPTS
      or now - self._drain_first_failure_at
      >= AUTONOMOUS_APPROVAL_FALLBACK_RETRY_WINDOW_SECONDS
    )
    if exhausted:
      self._fatal_error = (
        "autonomous approval recovery failed closed after "
        f"{self._drain_failure_count} attempts: "
        f"{type(failure).__name__}: {failure}"
      )
      retry_handle = self._retry_handle
      self._retry_handle = None
      if retry_handle is not None:
        retry_handle.cancel()
      log.critical(self._fatal_error)
      return
    self._arm_retry(
      min(
        self._fallback_retry_base_seconds
        * (2 ** (self._drain_failure_count - 1)),
        16.0,
      )
    )

  async def drain_once(self) -> int:
    recovery_window_reader = getattr(
      self._store,
      "autonomous_approval_delivery_recovery_window",
      None,
    )
    selector = getattr(
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
    retry_delays: list[float] = []
    while after_sequence < high_water and not self._closing:
      deliveries = await selector(
        limit=self._batch_limit,
        after_sequence=after_sequence,
        through_sequence=high_water,
      )
      if not isinstance(deliveries, list):
        raise RuntimeError(
          "autonomous approval pending selector returned invalid data"
        )
      if len(deliveries) > self._batch_limit:
        raise RuntimeError(
          "autonomous approval pending selector exceeded its bound"
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
        next_attempt_ns = delivery.get("next_attempt_ns")
        if type(next_attempt_ns) is not int or next_attempt_ns < 1:
          raise RuntimeError(
            "autonomous approval pending retry time is invalid"
          )
        if next_attempt_ns > observed_at_ns:
          retry_delays.append(
            (next_attempt_ns - observed_at_ns) / 1_000_000_000
          )
          continue
        attempted += 1
        try:
          await self._deliver_one(delivery)
          self._clear_fallback_failure(delivery)
        except asyncio.CancelledError:
          raise
        except PermanentAutonomousApprovalDeliveryError as exc:
          quarantined = await self._quarantine_failure(
            delivery,
            exc,
          )
          if quarantined is not None:
            self._clear_fallback_failure(delivery)
            await self._fail_quarantined_owner(
              quarantined,
              exc,
            )
          else:
            await self._fallback_retry_or_fail(delivery, exc)
        except Exception as exc:
          outcome = await self._record_failure(delivery, exc)
          if outcome is not None:
            if outcome.get("state") == "quarantined":
              self._clear_fallback_failure(delivery)
              await self._fail_quarantined_owner(outcome, exc)
            elif outcome.get("state") == "pending":
              last_attempt_ns = outcome.get("last_attempt_ns")
              retry_at_ns = outcome.get("next_attempt_ns")
              if (
                type(last_attempt_ns) is int
                and type(retry_at_ns) is int
                and retry_at_ns > last_attempt_ns
              ):
                retry_delays.append(
                  (retry_at_ns - last_attempt_ns)
                  / 1_000_000_000
                )
                self._clear_fallback_failure(delivery)
              else:
                await self._fallback_retry_or_fail(
                  delivery,
                  exc,
                )
            elif outcome.get("state") in {
              "published",
              "acknowledged",
            }:
              self._clear_fallback_failure(delivery)
            else:
              await self._fallback_retry_or_fail(delivery, exc)
          else:
            await self._fallback_retry_or_fail(delivery, exc)
      if len(deliveries) < self._batch_limit:
        break
    if retry_delays and not self._closing:
      self._arm_retry(min(retry_delays))
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

  @staticmethod
  def _delivery_identity(
    delivery: dict[str, Any],
  ) -> tuple[object, object, object]:
    return (
      delivery.get("approval_id"),
      delivery.get("tool_call_id"),
      delivery.get("nonce"),
    )

  def _clear_fallback_failure(
    self,
    delivery: dict[str, Any],
  ) -> None:
    self._fallback_failures.pop(
      self._delivery_identity(delivery),
      None,
    )

  async def _fallback_retry_or_fail(
    self,
    delivery: dict[str, Any],
    failure: Exception,
  ) -> None:
    identity = self._delivery_identity(delivery)
    now = asyncio.get_running_loop().time()
    prior_count, first_failure_at = self._fallback_failures.get(
      identity,
      (0, now),
    )
    failure_count = prior_count + 1
    self._fallback_failures[identity] = (
      failure_count,
      first_failure_at,
    )
    exhausted = (
      failure_count
      >= AUTONOMOUS_APPROVAL_FALLBACK_RETRY_MAX_ATTEMPTS
      or now - first_failure_at
      >= AUTONOMOUS_APPROVAL_FALLBACK_RETRY_WINDOW_SECONDS
    )
    if exhausted:
      self._fallback_failures.pop(identity, None)
      await self._fail_quarantined_owner(delivery, failure)
      return
    delay = min(
      self._fallback_retry_base_seconds
      * (2 ** (failure_count - 1)),
      16.0,
    )
    self._arm_retry(delay)

  async def _deliver_one(
    self,
    delivery: dict[str, Any],
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
    expected_run_identity = {
      "task_id": record.task_id,
      "control_run_id": record.control_run_id,
      "session_id": record.session_id,
      "channel_id": record.channel_id,
    }
    if any(
      delivery.get(field_name) != expected
      for field_name, expected in expected_run_identity.items()
    ):
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval pending row changed run authority"
      )
    if not autonomous_run_accepts_approval_decisions(record):
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval owner task is not active"
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
    get_request = getattr(self._store, "get", None)
    if not callable(get_request):
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval request lookup is unavailable"
      )
    request_record = await get_request(approval_id)
    if request_record is None:
      raise PermanentAutonomousApprovalDeliveryError(
        "autonomous approval pending row has no durable request"
      )
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
      channel=record.channel,
    )

  async def _record_failure(
    self,
    delivery: object,
    failure: Exception,
  ) -> dict[str, Any] | None:
    if not isinstance(delivery, dict):
      log.error(
        "Malformed autonomous approval pending row could not be recorded"
      )
      return None
    record_failure = getattr(
      self._store,
      "record_autonomous_approval_delivery_failure",
      None,
    )
    if not callable(record_failure):
      log.error(
        "Autonomous approval delivery failure recorder unavailable"
      )
      return None
    try:
      get_delivery = getattr(
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
      return outcome if isinstance(outcome, dict) else None
    except Exception:
      log.exception(
        "Autonomous approval delivery failure could not be recorded"
      )
      return None

  async def _quarantine_failure(
    self,
    delivery: object,
    failure: Exception,
  ) -> dict[str, Any] | None:
    if not isinstance(delivery, dict):
      return None
    quarantine = getattr(
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

  async def _fail_quarantined_owner(
    self,
    delivery: dict[str, Any],
    failure: Exception,
  ) -> None:
    fail_owner = getattr(
      self._registry,
      "fail_autonomous_approval_delivery",
      None,
    )
    if not callable(fail_owner):
      log.error(
        "Autonomous approval quarantine could not fail its owner run"
      )
      return
    outcome = fail_owner(
      delivery.get("task_id"),
      error=f"{type(failure).__name__}: {failure}",
    )
    if inspect.isawaitable(outcome):
      await outcome


__all__ = [
  "AUTONOMOUS_APPROVAL_DRAIN_BATCH_LIMIT",
  "AUTONOMOUS_APPROVAL_DRAIN_SHUTDOWN_SECONDS",
  "AUTONOMOUS_APPROVAL_FALLBACK_RETRY_MAX_ATTEMPTS",
  "AUTONOMOUS_APPROVAL_FALLBACK_RETRY_WINDOW_SECONDS",
  "AutonomousApprovalDeliveryCoordinator",
  "PermanentAutonomousApprovalDeliveryError",
  "wake_autonomous_approval_delivery",
]
