"""Deterministic trigger and interactive lifecycle wiring for learning forks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Any
import uuid

from .fork_ledger import ForkLedger, ReceiptClaim
from .fork_task_registry import (
  ForkLaunchDecision,
  ForkTaskRegistry,
  learn_fork_enabled,
)
from .task_registry import TaskNotification


log = logging.getLogger("agent_gateway.learning_fork_trigger")

DEFAULT_LEARN_MEMORY_NUDGE_TURNS = 10
DEFAULT_LEARN_SKILL_NUDGE_ITERS = 10
LEARNING_RECEIPT_CLAIM_LIMIT = 5


@dataclass(frozen=True, slots=True)
class LearningForkTriggerDecision:
  """Pure trigger outcome after one completed foreground user turn."""

  should_submit: bool
  memory_turns: int
  skill_iters: int
  reason: str


@dataclass(frozen=True, slots=True)
class LearningReceiptDelivery:
  """Claims surfaced into exactly one notification-bearing turn."""

  ledger: ForkLedger
  claims: tuple[ReceiptClaim, ...]


_process_instance_id = f"gateway-{uuid.uuid4().hex}"
_process_ledger: ForkLedger | None = None
_process_registry: ForkTaskRegistry | None = None


def _nonnegative_env_int(name: str, default: int) -> int:
  raw = os.getenv(name)
  if raw is None:
    return default
  try:
    value = int(raw)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{name} must be a non-negative integer") from exc
  if value < 0:
    raise ValueError(f"{name} must be a non-negative integer")
  return value


def learn_memory_nudge_turns() -> int:
  return _nonnegative_env_int(
    "HANK_LEARN_MEMORY_NUDGE_TURNS",
    DEFAULT_LEARN_MEMORY_NUDGE_TURNS,
  )


def learn_skill_nudge_iters() -> int:
  return _nonnegative_env_int(
    "HANK_LEARN_SKILL_NUDGE_ITERS",
    DEFAULT_LEARN_SKILL_NUDGE_ITERS,
  )


def evaluate_learning_fork_trigger(
  *,
  memory_turns: int,
  skill_iters: int,
  tool_calling_iters: int,
  foreground_memory_write: bool,
  completed: bool,
  real_final_response: bool,
  errored: bool,
  aborted: bool,
  cancelled: bool,
  enabled: bool,
  memory_threshold: int,
  skill_threshold: int,
) -> LearningForkTriggerDecision:
  """Advance Hermes counters and decide whether one combined fork is due."""

  for value, field_name in (
    (memory_turns, "memory_turns"),
    (skill_iters, "skill_iters"),
    (tool_calling_iters, "tool_calling_iters"),
    (memory_threshold, "memory_threshold"),
    (skill_threshold, "skill_threshold"),
  ):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
      raise ValueError(f"{field_name} must be a non-negative integer")

  next_memory = memory_turns
  next_skill = skill_iters
  if completed and not (errored or aborted or cancelled):
    next_memory += 1
    next_skill += tool_calling_iters
    if foreground_memory_write:
      next_memory = 0

  if not enabled:
    reason = "disabled"
  elif not completed:
    reason = "not_completed"
  elif errored:
    reason = "errored"
  elif aborted:
    reason = "aborted"
  elif cancelled:
    reason = "cancelled"
  elif not real_final_response:
    reason = "no_final_response"
  elif memory_threshold == 0 or skill_threshold == 0:
    reason = "counter_disabled"
  elif next_memory < memory_threshold:
    reason = "memory_counter"
  elif next_skill < skill_threshold:
    reason = "skill_counter"
  else:
    return LearningForkTriggerDecision(
      should_submit=True,
      memory_turns=next_memory,
      skill_iters=next_skill,
      reason="tripped",
    )
  return LearningForkTriggerDecision(
    should_submit=False,
    memory_turns=next_memory,
    skill_iters=next_skill,
    reason=reason,
  )


def owner_operated_interactive_analyst(runner: Any, session: Any) -> bool:
  """Default-on cohort; non-owner and BYOK sessions are excluded."""

  execution = getattr(runner, "_capability_execution", None)
  bind = getattr(execution, "bind", None)
  run_context = getattr(getattr(runner, "_dispatcher", None), "_run_context", None)
  return bool(
    getattr(session, "role", None) == "owner"
    and getattr(runner, "_billing_mode", None) != "byok"
    and getattr(bind, "credential_principal", None) == "service"
    and getattr(bind, "run_mode", None) == "interactive"
    and getattr(run_context, "profile", None) == "analyst"
    and not bool(getattr(runner, "_fork_mode", False))
  )


def _gateway_session(runner: Any) -> Any | None:
  return (
    getattr(runner, "_gateway_session", None)
    or getattr(getattr(runner, "_dispatcher", None), "_session", None)
  )


def _owner(session: Any) -> str:
  owner = str(
    getattr(session, "owner_user_id", None)
    or getattr(session, "user_id", None)
    or ""
  ).strip()
  if not owner:
    raise ValueError("learning fork owner is required")
  return owner


def _emit_metric(runner: Any, event: str) -> None:
  callback = getattr(runner, "_on_metric", None)
  if callable(callback):
    try:
      callback(event, 1)
    except Exception:
      log.warning("Learning-fork metric sink failed", exc_info=True)


def _registry_telemetry(event: str, fields: Mapping[str, Any]) -> None:
  log.info("Learning-fork registry event %s: %s", event, dict(fields))


def _default_ledger_path(owner: str) -> Path:
  import memory

  user_data_dir = Path(memory.get_user_data_dir(owner))
  return user_data_dir.parent.parent / "fork-ledger.sqlite3"


def _runtime_for_runner(
  runner: Any,
  *,
  create: bool,
) -> tuple[ForkLedger, ForkTaskRegistry] | None:
  global _process_ledger, _process_registry

  session = _gateway_session(runner)
  injected_ledger = getattr(session, "learning_fork_ledger", None)
  injected_registry = getattr(session, "learning_fork_registry", None)
  if isinstance(injected_ledger, ForkLedger) and isinstance(
    injected_registry,
    ForkTaskRegistry,
  ):
    return injected_ledger, injected_registry
  if _process_ledger is not None and _process_registry is not None:
    return _process_ledger, _process_registry
  if not create or session is None:
    return None

  from .runner_fork_agents import spawn_learning_fork

  ledger = ForkLedger(
    _default_ledger_path(_owner(session)),
    process_instance_id=_process_instance_id,
  )
  ledger.reconcile_startup(
    live_process_instance_ids={_process_instance_id},
  )
  registry = ForkTaskRegistry(
    ledger,
    spawn_fork=spawn_learning_fork,
    telemetry=_registry_telemetry,
    owner_operated_interactive=True,
  )
  _process_ledger = ledger
  _process_registry = registry
  return ledger, registry


def claim_learning_receipts(runner: Any) -> LearningReceiptDelivery | None:
  """Claim pending receipts and enqueue only their one-line summaries."""

  session = _gateway_session(runner)
  if session is None or not owner_operated_interactive_analyst(runner, session):
    return None
  try:
    runtime = _runtime_for_runner(runner, create=True)
    if runtime is None:
      return None
    ledger, _registry = runtime
    claims = ledger.claim_pending_receipts(
      session_id=str(session.session_id),
      owner=_owner(session),
      claiming_turn_id=str(getattr(runner, "_request_id", "") or uuid.uuid4().hex),
      limit=LEARNING_RECEIPT_CLAIM_LIMIT,
    )
    deduped: list[ReceiptClaim] = []
    seen: set[str] = set()
    for claim in claims:
      if claim.fork_id in seen:
        ledger.revert_receipt_claim(
          fork_id=claim.fork_id,
          claim_token=claim.claim_token,
        )
        continue
      seen.add(claim.fork_id)
      try:
        notification = TaskNotification(
          task_id=f"learning-fork:{claim.fork_id}",
          agent_name="self-learning",
          event="completed",
          summary=claim.receipt_text,
          timestamp=time.time(),
          payload={"receipt": claim.receipt_text},
        )
        queued = runner._notification_queue.push(notification)
      except Exception:
        queued = False
        log.warning(
          "Learning-fork receipt notification failed for %s",
          claim.fork_id,
          exc_info=True,
        )
      if not queued:
        ledger.revert_receipt_claim(
          fork_id=claim.fork_id,
          claim_token=claim.claim_token,
        )
        continue
      deduped.append(claim)
    if not deduped:
      return None
    _emit_metric(runner, "learning_fork_receipts_claimed")
    return LearningReceiptDelivery(ledger=ledger, claims=tuple(deduped))
  except Exception:
    _emit_metric(runner, "learning_fork_receipt_claim_failed")
    log.warning("Learning-fork receipt claim failed", exc_info=True)
    return None


def settle_learning_receipts(
  runner: Any,
  delivery: LearningReceiptDelivery | None,
  *,
  success: bool,
) -> None:
  """CAS-ack a successful bearing turn, otherwise return claims to pending."""

  if delivery is None:
    return
  for claim in delivery.claims:
    try:
      changed = (
        delivery.ledger.ack_receipt(
          fork_id=claim.fork_id,
          claim_token=claim.claim_token,
        )
        if success
        else delivery.ledger.revert_receipt_claim(
          fork_id=claim.fork_id,
          claim_token=claim.claim_token,
        )
      )
      if not changed:
        raise RuntimeError("learning receipt CAS did not change one row")
    except Exception:
      _emit_metric(runner, "learning_fork_receipt_settlement_failed")
      log.warning(
        "Learning-fork receipt settlement failed for %s",
        claim.fork_id,
        exc_info=True,
      )


def submit_learning_fork_after_turn(
  runner: Any,
  *,
  handoff: Any,
  tool_calling_iters: int,
  foreground_memory_write: bool,
  completed: bool,
  real_final_response: bool,
  errored: bool,
  aborted: bool,
  cancelled: bool,
) -> ForkLaunchDecision | None:
  """Best-effort post-delivery trigger handoff into the process registry."""

  session = _gateway_session(runner)
  if session is None:
    return None
  try:
    eligible = owner_operated_interactive_analyst(runner, session)
    enabled = learn_fork_enabled(
      owner_operated_interactive=eligible,
    )
    decision = evaluate_learning_fork_trigger(
      memory_turns=int(getattr(session, "learn_memory_nudge_turns", 0)),
      skill_iters=int(getattr(session, "learn_skill_nudge_iters", 0)),
      tool_calling_iters=tool_calling_iters,
      foreground_memory_write=foreground_memory_write,
      completed=completed,
      real_final_response=real_final_response,
      errored=errored,
      aborted=aborted,
      cancelled=cancelled,
      enabled=enabled,
      memory_threshold=learn_memory_nudge_turns(),
      skill_threshold=learn_skill_nudge_iters(),
    )
    session.learn_memory_nudge_turns = decision.memory_turns
    session.learn_skill_nudge_iters = decision.skill_iters
    if not decision.should_submit or handoff is None:
      return None

    runtime = _runtime_for_runner(runner, create=True)
    if runtime is None:
      return None
    ledger, registry = runtime
    from .runner_fork_agents import LearningForkWorkItem

    fork_id = f"learn-{uuid.uuid4().hex}"
    work_item = LearningForkWorkItem(
      parent=runner,
      handoff=handoff,
      ledger=ledger,
      session_id=str(session.session_id),
      owner=_owner(session),
      user_id=str(getattr(session, "user_id", "") or _owner(session)),
    )
    launch = registry.submit(
      fork_id=fork_id,
      session_id=str(session.session_id),
      owner=_owner(session),
      handoff=work_item,
    )
    if launch.launched:
      session.learn_skill_nudge_iters = 0
    return launch
  except Exception:
    _emit_metric(runner, "learning_fork_trigger_failed")
    log.warning("Learning-fork trigger failed", exc_info=True)
    return None


def successful_memory_write_from_events(
  entries: Sequence[Any],
  *,
  fork: bool,
) -> bool:
  """Return whether this owner wrote memory successfully in its own event log."""

  for raw in entries:
    event = raw.event if hasattr(raw, "event") else raw
    if not isinstance(event, Mapping):
      continue
    if bool(event.get("fork")) != fork:
      continue
    if (
      event.get("type") == "tool_call_complete"
      and event.get("tool_name") == "memory_write"
      and not bool(event.get("is_error"))
      and event.get("error") is None
    ):
      return True
  return False


__all__ = [
  "DEFAULT_LEARN_MEMORY_NUDGE_TURNS",
  "DEFAULT_LEARN_SKILL_NUDGE_ITERS",
  "LearningForkTriggerDecision",
  "LearningReceiptDelivery",
  "claim_learning_receipts",
  "evaluate_learning_fork_trigger",
  "learn_memory_nudge_turns",
  "learn_skill_nudge_iters",
  "owner_operated_interactive_analyst",
  "settle_learning_receipts",
  "submit_learning_fork_after_turn",
  "successful_memory_write_from_events",
]
