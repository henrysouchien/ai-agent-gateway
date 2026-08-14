from __future__ import annotations

import asyncio
from copy import deepcopy
import fcntl
import logging
import socket
import sys
import time
from typing import Any, Dict, List

from .agent_session_log_records import EVENT_SCHEMA_VERSION
from .product_config import gateway_product_id
from .runner_introspection import format_exc as _format_exc
from .runner_session_events import (
  build_assistant_message_event as _build_assistant_message_event,
  build_attach_event as _build_attach_event,
  build_detach_event as _build_detach_event,
  build_error_event as _build_error_event,
  build_interrupted_event as _build_interrupted_event,
  build_operator_pause_event as _build_operator_pause_event,
  build_orphan_tool_call_interrupted_events as _build_orphan_tool_call_interrupted_events,
  build_run_error_event as _build_run_error_event,
  build_skill_result_failure_event as _build_skill_result_failure_event,
  build_skill_run_started_event as _build_skill_run_started_event,
  build_stream_retry_event as _build_stream_retry_event,
  build_user_message_event as _build_user_message_event,
  build_workflow_output_attached_event as _build_workflow_output_attached_event,
  durable_event_payload as _durable_event_payload,
  release_write_lease as _release_write_lease,
  shutdown_interrupted_reason as _shutdown_interrupted_reason,
  write_lease_metadata as _write_lease_metadata,
)
from .skill_lifecycle import (
  SKILL_RESULT_CORE_FIELDS,
  TopLevelSkillCompletionPlan,
  TopLevelSkillLifecycleMetadata,
  WriterLeaseAlreadyHeldError,
  drain_owned_lifecycle_task,
)
from .skill_completion_wal import (
  SkillCompletionEffectConflict,
  SkillCompletionWal,
  SkillCompletionWalCorruptError,
  TopLevelSkillCompletionEffectPlan,
  apply_completion_effect,
  envelope_digest,
)
from .runner_tool_audit import get_tool_risk_value as _get_tool_risk_value
from .secret_boundary import sanitize_tool_event
from .task_registry import ParentMessage, TaskEntry, TaskRegistry, TaskState
from .workflow_output_attachment import WorkflowOutputAttachment


log = logging.getLogger("agent_gateway.runner")

_DURABLE_TOP_LEVEL_ENVELOPE_FIELDS = frozenset({
  "runner_id",
  "role",
  "sub_agent_id",
  "product_id",
  "event_schema_version",
})
_TOP_LEVEL_SKILL_LIFECYCLE_ORIGIN_FIELD = "lifecycle_origin"
_TOP_LEVEL_SKILL_LIFECYCLE_ORIGIN = "top_level"


def _runner_attr(instance: Any, name: str, fallback: Any) -> Any:
  for cls in type(instance).__mro__:
    module = sys.modules.get(getattr(cls, "__module__", ""))
    if module is not None and getattr(module, "AgentRunner", None) is cls:
      return getattr(module, name, fallback)
  module = sys.modules.get("agent_gateway.runner")
  if module is None:
    return fallback
  return getattr(module, name, fallback)


def _exact_value_match(expected: Any, actual: Any) -> bool:
  if type(actual) is not type(expected):
    return False
  if isinstance(expected, dict):
    return (
      actual.keys() == expected.keys()
      and all(
        _exact_value_match(value, actual[field_name])
        for field_name, value in expected.items()
      )
    )
  if isinstance(expected, list):
    return (
      len(actual) == len(expected)
      and all(
        _exact_value_match(expected_item, actual_item)
        for expected_item, actual_item in zip(expected, actual)
      )
    )
  return bool(actual == expected)


class RunnerSessionLifecycleMixin:
  def _durable_event_for_append(
    self,
    event: Dict[str, Any],
  ) -> Dict[str, Any] | None:
    if self._agent_session_log is None or self._runner_id is None:
      return None
    durable_event_payload = _runner_attr(self, "_durable_event_payload", _durable_event_payload)
    product_id = _runner_attr(self, "gateway_product_id", gateway_product_id)()
    safe_event = sanitize_tool_event(
      event,
      sink="durable_event",
      boundary=getattr(self, "_secret_boundary", None),
    )
    return durable_event_payload(
      safe_event,
      runner_id=self._runner_id,
      role=self._role,
      sub_agent_id=self._sub_agent_id,
      product_id=product_id,
    )

  async def _append_durable_event(self, event: Dict[str, Any]) -> Any | None:
    payload = self._durable_event_for_append(deepcopy(event))
    if payload is None:
      return None
    entry = await self._agent_session_log.append(payload)
    self._last_durable_seq = max(
      int(getattr(self, "_last_durable_seq", 0) or 0),
      entry.seq,
    )
    return entry

  async def _confirm_durable_skill_event(
    self,
    event: Dict[str, Any],
  ) -> Dict[str, Any] | None:
    durable_log = getattr(self, "_agent_session_log", None)
    event_type = event.get("type")
    skill_run_id = event.get("skill_run_id")
    if (
      durable_log is None
      or event_type not in {
        "skill_run_started",
        "skill_result_captured",
      }
      or not isinstance(skill_run_id, str)
      or not skill_run_id
    ):
      return None
    expected = self._durable_event_for_append(deepcopy(event))
    if expected is None:
      return None
    expected["event_schema_version"] = EVENT_SCHEMA_VERSION
    entries, _ = await durable_log.query(
      event_types={event_type},
      contains_text=skill_run_id,
      order="asc",
    )
    matches = [
      entry
      for entry in entries
      if (
        entry.event.get("type") == event_type
        and entry.event.get("skill_run_id") == skill_run_id
      )
    ]
    if len(matches) > 1:
      raise RuntimeError(
        f"Duplicate durable {event_type} events for skill lifecycle "
        f"{skill_run_id}"
      )
    if not matches:
      return None
    confirmed = matches[0].event
    if not _exact_value_match(expected, confirmed):
      raise RuntimeError(
        f"Durable {event_type} envelope mismatch for skill lifecycle "
        f"{skill_run_id}"
      )
    return deepcopy(confirmed)

  def _append_durable_event_sync(self, event: Dict[str, Any]) -> Any | None:
    """Commit a terminal receipt without yielding after settlement proof."""

    payload = self._durable_event_for_append(deepcopy(event))
    if payload is None:
      return None
    append_sync = getattr(self._agent_session_log, "append_sync", None)
    if not callable(append_sync):
      raise RuntimeError(
        "durable session log does not support synchronous terminal settlement"
      )
    entry = append_sync(payload)
    self._last_durable_seq = max(
      int(getattr(self, "_last_durable_seq", 0) or 0),
      entry.seq,
    )
    return entry

  @staticmethod
  def _require_append_acknowledgement(
    entry: Any,
    event: Dict[str, Any],
    *,
    target: str,
  ) -> None:
    appended = getattr(entry, "event", None)
    if not isinstance(appended, dict):
      raise RuntimeError(
        f"{target} append returned no event acknowledgement"
      )
    for field_name, expected in event.items():
      if (
        field_name not in appended
        or not _exact_value_match(
          expected,
          appended[field_name],
        )
      ):
        raise RuntimeError(
          f"{target} append acknowledgement mismatch for "
          f"{field_name}"
        )

  def _preflight_top_level_skill_run(
    self,
    messages: List[Dict[str, Any]],
    resume_initial_messages: List[Dict[str, Any]] | None,
  ) -> Dict[str, Any] | None:
    lifecycle = getattr(
      self,
      "_top_level_skill_lifecycle",
      None,
    )
    if lifecycle is None:
      return None
    if not isinstance(
      lifecycle,
      TopLevelSkillLifecycleMetadata,
    ):
      raise RuntimeError(
        "Top-level skill lifecycle metadata was not retained exactly"
      )
    if self._agent_session_log is None:
      raise RuntimeError(
        "Top-level named-skill runs require a durable session log"
      )
    if self._role != "writer":
      raise RuntimeError(
        "Top-level named-skill lifecycle is restricted to the writer"
      )
    if resume_initial_messages is not None:
      raise RuntimeError(
        "Top-level named-skill lifecycle cannot run on resume"
      )
    if (
      len(messages) != 1
      or not isinstance(messages[0], dict)
      or messages[0].get("role") != "user"
    ):
      raise RuntimeError(
        "Top-level named-skill lifecycle requires exactly one fresh "
        "incoming user message"
      )
    policy = getattr(
      self,
      "_top_level_skill_result_policy",
      None,
    )
    if (
      policy is None
      or not callable(getattr(policy, "prepare", None))
      or not callable(
        getattr(policy, "prepare_system_prompt", None)
      )
    ):
      raise RuntimeError(
        "Top-level named-skill lifecycle requires a result policy"
      )
    if self._skill_run_id != lifecycle.skill_run_id:
      raise RuntimeError(
        "Top-level named-skill citation and lifecycle run IDs diverged"
      )
    return dict(messages[0])

  async def _prepare_top_level_skill_system_prompt(
    self,
    proposed_prompt: Any,
  ) -> Any:
    if self._top_level_skill_lifecycle is None:
      return proposed_prompt
    policy = self._top_level_skill_result_policy
    if policy is None:
      raise RuntimeError(
        "Top-level skill result policy is unavailable"
      )
    return await policy.prepare_system_prompt(
      deepcopy(proposed_prompt)
    )

  def _top_level_server_terminal_cause(self) -> str | None:
    policy = self._top_level_skill_result_policy
    if policy is None:
      return None
    cause = getattr(policy, "server_terminal_cause", None)
    if cause is None:
      return None
    if cause not in {
      "caller_cancellation",
      "shutdown",
      "timeout",
    }:
      raise RuntimeError(
        f"Invalid top-level server terminal cause {cause!r}"
      )
    return cause

  async def _top_level_skill_entries(
    self,
    *,
    event_types: set[str],
  ) -> list[Any]:
    lifecycle = self._top_level_skill_lifecycle
    if lifecycle is None or self._agent_session_log is None:
      return []
    entries, _ = await self._agent_session_log.query(
      event_types=event_types,
      contains_text=lifecycle.skill_run_id,
      order="asc",
    )
    return [
      entry
      for entry in entries
      if entry.event.get("skill_run_id")
      == lifecycle.skill_run_id
    ]

  async def _assert_fresh_top_level_skill_lifecycle(self) -> None:
    lifecycle = self._top_level_skill_lifecycle
    if lifecycle is None:
      return
    entries = await self._top_level_skill_entries(
      event_types={
        "skill_run_started",
        "skill_result_captured",
      },
    )
    if entries:
      event_types = ", ".join(
        str(entry.event.get("type"))
        for entry in entries
      )
      raise RuntimeError(
        "Stale top-level skill lifecycle already exists for "
        f"{lifecycle.skill_run_id}: {event_types}"
      )

  async def _confirm_durable_top_level_skill_event(
    self,
    event: Dict[str, Any],
  ) -> bool:
    lifecycle = self._top_level_skill_lifecycle
    if lifecycle is None:
      return False
    confirmed = await self._confirm_durable_skill_event(
      deepcopy(event)
    )
    if confirmed is None:
      return False
    lifecycle.require_event_identity(
      confirmed,
      event_type=str(event.get("type")),
    )
    return True

  def _project_top_level_skill_event(
    self,
    event: Dict[str, Any],
  ) -> None:
    if event.get("type") == "skill_run_started":
      if self._top_level_skill_started_projected:
        return
    elif event.get("type") == "skill_result_captured":
      if self._top_level_skill_result_projected:
        return
    else:
      raise RuntimeError(
        f"Unsupported top-level skill event {event.get('type')!r}"
      )
    projected_event = deepcopy(event)
    projected_entry = self._append(projected_event)
    if projected_entry is None:
      raise RuntimeError(
        f"Live EventLog rejected top-level {event.get('type')} projection"
      )
    self._require_append_acknowledgement(
      projected_entry,
      event,
      target=f"live {event.get('type')}",
    )
    if event.get("type") == "skill_run_started":
      self._top_level_skill_started_projected = True
    else:
      self._top_level_skill_result_projected = True

  async def _persist_top_level_skill_started(
    self,
    event: Dict[str, Any],
  ) -> bool:
    await self._assert_fresh_top_level_skill_lifecycle()
    try:
      durable_entry = await self._append_durable_event(
        deepcopy(event)
      )
      if durable_entry is None:
        raise RuntimeError(
          "Durable top-level skill-start target is unavailable"
        )
      expected = self._expected_durable_top_level_event(event)
      acknowledged = getattr(durable_entry, "event", None)
      if not _exact_value_match(expected, acknowledged):
        raise RuntimeError(
          "Durable skill_run_started append acknowledgement is not "
          "the exact writer envelope"
        )
    except BaseException as exc:
      confirmed = await self._confirm_durable_top_level_skill_event(
        event
      )
      if not confirmed:
        raise exc
    self._top_level_skill_started_committed = True
    self._top_level_skill_started_event = deepcopy(event)
    self._project_top_level_skill_event(deepcopy(event))
    return True

  async def _emit_top_level_skill_started(self) -> bool:
    lifecycle = self._top_level_skill_lifecycle
    if lifecycle is None:
      return False
    if self._top_level_skill_started_task is None:
      event = _runner_attr(
        self,
        "_build_skill_run_started_event",
        _build_skill_run_started_event,
      )(
        lifecycle,
        started_at=_runner_attr(self, "time", time).time(),
      )
      event[_TOP_LEVEL_SKILL_LIFECYCLE_ORIGIN_FIELD] = (
        _TOP_LEVEL_SKILL_LIFECYCLE_ORIGIN
      )
      self._top_level_skill_started_event = deepcopy(event)
      self._top_level_skill_started_task = asyncio.create_task(
        self._persist_top_level_skill_started(deepcopy(event)),
        name=(
          f"top-level-skill:{lifecycle.skill_run_id}:started"
        ),
      )
    try:
      return bool(
        await asyncio.shield(
          self._top_level_skill_started_task
        )
      )
    except asyncio.CancelledError:
      await drain_owned_lifecycle_task(
        self._top_level_skill_started_task
      )
      raise

  async def _build_top_level_skill_result_event(
    self,
    terminal_event: Dict[str, Any],
  ) -> Dict[str, Any]:
    lifecycle = self._top_level_skill_lifecycle
    policy = self._top_level_skill_result_policy
    if lifecycle is None or policy is None:
      raise RuntimeError(
        "Top-level skill result policy is unavailable"
      )
    plan = await policy.prepare(
      self._log,
      deepcopy(terminal_event),
    )
    if not isinstance(plan, TopLevelSkillCompletionPlan):
      raise RuntimeError(
        "Top-level skill policy did not return a completion plan"
      )
    prepared_terminal = plan.terminal_event
    self._prepared_top_level_terminal_event = deepcopy(
      prepared_terminal
    )
    self._top_level_skill_completion_effect_plan = deepcopy(
      plan.effect
    )
    return lifecycle.normalize_result_event(
      deepcopy(plan.result_event)
    )

  async def _assert_no_top_level_skill_result(self) -> None:
    entries = await self._top_level_skill_entries(
      event_types={"skill_result_captured"},
    )
    if entries:
      lifecycle = self._top_level_skill_lifecycle
      raise RuntimeError(
        "Stale top-level skill_result_captured already exists for "
        f"{lifecycle.skill_run_id}"
      )

  async def _prepare_top_level_skill_result(
    self,
    terminal_event: Dict[str, Any],
  ) -> Dict[str, Any] | None:
    if self._top_level_skill_lifecycle is None:
      return None
    if not self._top_level_skill_started_committed:
      raise RuntimeError(
        "Cannot capture top-level skill result before start"
      )
    if self._top_level_skill_result_committed:
      raise RuntimeError(
        "Top-level skill result was already committed"
      )
    await self._assert_no_top_level_skill_result()
    event = await self._build_top_level_skill_result_event(
      terminal_event
    )
    canonical = self._top_level_skill_lifecycle.normalize_result_event(
      event
    )
    self._top_level_skill_result_event = deepcopy(canonical)
    return deepcopy(canonical)

  def _bind_top_level_terminal_identity(
    self,
    event: Dict[str, Any],
    *,
    lifecycle: TopLevelSkillLifecycleMetadata | None = None,
  ) -> Dict[str, Any]:
    lifecycle = lifecycle or self._top_level_skill_lifecycle
    if lifecycle is None:
      raise RuntimeError(
        "Top-level terminal identity requires lifecycle metadata"
      )
    terminal = deepcopy(event)
    if terminal.get("type") not in {"error", "stream_complete"}:
      raise RuntimeError(
        "Top-level terminal must be error or stream_complete"
      )
    for field_name, expected in lifecycle.identity_fields().items():
      actual = terminal.get(field_name, expected)
      if (
        actual != expected
        or type(actual) is not type(expected)
      ):
        raise RuntimeError(
          f"Top-level terminal identity mismatch for {field_name}"
        )
      terminal[field_name] = expected
    return terminal

  def _expected_durable_top_level_event(
    self,
    event: Dict[str, Any],
  ) -> Dict[str, Any]:
    if "event_schema_version" in event:
      if event.get("event_schema_version") != EVENT_SCHEMA_VERSION:
        raise RuntimeError(
          "Prepared durable top-level envelope has an invalid schema"
        )
      return deepcopy(event)
    payload = self._durable_event_for_append(deepcopy(event))
    if payload is None:
      raise RuntimeError(
        "Durable top-level lifecycle target is unavailable"
      )
    payload["event_schema_version"] = EVENT_SCHEMA_VERSION
    return payload

  def _append_exact_durable_top_level_envelope_sync(
    self,
    event: Dict[str, Any],
  ) -> Any:
    if event.get("event_schema_version") != EVENT_SCHEMA_VERSION:
      raise RuntimeError(
        "Exact durable top-level envelope has an invalid schema"
      )
    append_sync = getattr(
      self._agent_session_log,
      "append_sync",
      None,
    )
    if not callable(append_sync):
      raise RuntimeError(
        "Durable session log lacks synchronous exact append"
      )
    entry = append_sync(deepcopy(event))
    self._last_durable_seq = entry.seq
    return entry

  def _exact_top_level_durable_event_sync(
    self,
    event: Dict[str, Any],
  ) -> Dict[str, Any] | None:
    query_sync = getattr(
      self._agent_session_log,
      "query_sync",
      None,
    )
    if not callable(query_sync):
      raise RuntimeError(
        "Durable session log lacks synchronous exact readback"
      )
    skill_run_id = event.get("skill_run_id")
    if not isinstance(skill_run_id, str) or not skill_run_id:
      raise RuntimeError(
        "Top-level durable envelope lacks skill_run_id"
      )
    entries, _ = query_sync(
      event_types={str(event.get("type"))},
      contains_text=skill_run_id,
      order="asc",
    )
    lifecycle_matches = [
      entry.event
      for entry in entries
      if entry.event.get("skill_run_id") == skill_run_id
    ]
    if len(lifecycle_matches) > 1:
      raise RuntimeError(
        "Duplicate durable top-level lifecycle envelopes for "
        f"{skill_run_id} and {event.get('type')}"
      )
    if not lifecycle_matches:
      return None
    expected = self._expected_durable_top_level_event(event)
    if not _exact_value_match(
      expected,
      lifecycle_matches[0],
    ):
      raise RuntimeError(
        "Conflicting durable top-level lifecycle envelope for "
        f"{skill_run_id} and {event.get('type')}"
      )
    return deepcopy(lifecycle_matches[0])

  def _append_and_readback_top_level_event_sync(
    self,
    event: Dict[str, Any],
  ) -> Dict[str, Any]:
    confirmed = self._exact_top_level_durable_event_sync(event)
    if confirmed is not None:
      return confirmed

    last_failure: BaseException | None = None
    for _attempt in range(2):
      try:
        entry = self._append_exact_durable_top_level_envelope_sync(
          event
        )
        if entry is None:
          raise RuntimeError(
            "Durable top-level lifecycle append is unavailable"
          )
        self._require_append_acknowledgement(
          entry,
          event,
          target=f"durable {event.get('type')}",
        )
      except BaseException as exc:
        last_failure = exc
      confirmed = self._exact_top_level_durable_event_sync(event)
      if confirmed is not None:
        return confirmed
      if last_failure is None:
        last_failure = RuntimeError(
          "Durable top-level lifecycle readback is missing"
        )

    assert last_failure is not None
    raise last_failure

  def _top_level_completion_wal(self) -> SkillCompletionWal:
    if self._agent_session_log is None:
      raise RuntimeError(
        "Top-level completion WAL requires a durable session log"
      )
    return SkillCompletionWal(self._agent_session_log.path)

  def _require_top_level_admission_fence(self) -> Any:
    admission = getattr(
      self,
      "_top_level_skill_admission",
      None,
    )
    if admission is None:
      raise RuntimeError(
        "Top-level completion requires a pre-acquired admission"
      )
    admission.validate_fence()
    return admission

  def _settled_wal_is_replaceable(
    self,
    record: Dict[str, Any],
  ) -> bool:
    if record.get("record_type") != "settled":
      return False
    query_sync = getattr(
      self._agent_session_log,
      "query_sync",
      None,
    )
    if not callable(query_sync):
      raise RuntimeError(
        "Durable session log lacks synchronous WAL proof"
      )
    skill_run_id = record["skill_run_id"]
    entries, _ = query_sync(
      event_types={
        "skill_result_captured",
        "error",
        "stream_complete",
      },
      contains_text=skill_run_id,
      order="asc",
    )
    results = [
      entry.event
      for entry in entries
      if (
        entry.event.get("type") == "skill_result_captured"
        and entry.event.get("skill_run_id") == skill_run_id
      )
    ]
    terminals = [
      entry.event
      for entry in entries
      if (
        entry.event.get("type") in {"error", "stream_complete"}
        and entry.event.get("skill_run_id") == skill_run_id
      )
    ]
    if len(results) == 1 and len(terminals) == 1:
      try:
        result_wrapper = self._durable_top_level_wrapper(
          results[0]
        )
        terminal_wrapper = self._durable_top_level_wrapper(
          terminals[0]
        )
      except SkillCompletionWalCorruptError:
        return False
    else:
      result_wrapper = None
      terminal_wrapper = None
    return (
      len(results) == 1
      and len(terminals) == 1
      and _exact_value_match(
        result_wrapper,
        terminal_wrapper,
      )
      and envelope_digest(results[0])
      == record["result_digest"]
      and envelope_digest(terminals[0])
      == record["terminal_digest"]
    )

  def _completion_intent_record(
    self,
    *,
    lifecycle: TopLevelSkillLifecycleMetadata,
    durable_result: Dict[str, Any],
    durable_terminal: Dict[str, Any],
    effect: TopLevelSkillCompletionEffectPlan,
    fence: Dict[str, Any],
  ) -> Dict[str, Any]:
    return {
      "record_type": "intent",
      "skill_run_id": lifecycle.skill_run_id,
      "lifecycle": lifecycle.identity_fields(),
      "result": deepcopy(durable_result),
      "terminal": deepcopy(durable_terminal),
      "effect": effect.durable_payload(),
      "fence": deepcopy(fence),
    }

  @staticmethod
  def _durable_top_level_wrapper(
    event: Dict[str, Any],
  ) -> Dict[str, Any]:
    wrapper = {
      field_name: deepcopy(event[field_name])
      for field_name in _DURABLE_TOP_LEVEL_ENVELOPE_FIELDS
      if field_name in event
    }
    if wrapper.get("event_schema_version") != EVENT_SCHEMA_VERSION:
      raise SkillCompletionWalCorruptError(
        "Completion intent durable envelope has an invalid schema"
      )
    if (
      type(wrapper.get("runner_id")) is not str
      or not wrapper["runner_id"]
      or wrapper.get("role") != "writer"
      or "sub_agent_id" in wrapper
    ):
      raise SkillCompletionWalCorruptError(
        "Completion intent durable writer identity is invalid"
      )
    if (
      "product_id" in wrapper
      and (
        type(wrapper["product_id"]) is not str
        or not wrapper["product_id"]
      )
    ):
      raise SkillCompletionWalCorruptError(
        "Completion intent durable product identity is invalid"
      )
    return wrapper

  def _tolerate_unknown_durable_result_fields(
    self,
    event: Dict[str, Any],
    *,
    source: str,
  ) -> None:
    """Log-and-ignore unknown fields on a durable result read back.

    Durable session logs outlive any single deployed generation, so a
    recovered result may carry envelope fields written by a newer (or
    older) generation's schema. Recovery must stay forward-compatible
    on read: unknown fields are dropped with a structured notice while
    the CORE result fields remain strictly validated (missing core
    fields, invalid values, and identity mismatches still hard-fail).
    """
    unexpected = set(event) - (
      SKILL_RESULT_CORE_FIELDS
      | _DURABLE_TOP_LEVEL_ENVELOPE_FIELDS
    )
    if not unexpected:
      return
    notices = getattr(
      self,
      "_unknown_durable_result_field_notices",
      None,
    )
    if notices is None:
      notices = set()
      self._unknown_durable_result_field_notices = notices
    notice_key = (
      source,
      event.get("skill_run_id"),
      tuple(sorted(unexpected)),
    )
    if notice_key in notices:
      return
    notices.add(notice_key)
    log.warning(
      "[%s] Ignoring unknown durable result fields on recovery "
      "source=%s skill_run_id=%s skill=%s unknown_fields=%s "
      "(core result fields remain strictly validated)",
      self._sid,
      source,
      event.get("skill_run_id"),
      event.get("skill"),
      sorted(unexpected),
    )

  def _logical_result_from_completion_intent(
    self,
    event: Dict[str, Any],
    *,
    lifecycle: TopLevelSkillLifecycleMetadata,
  ) -> Dict[str, Any]:
    self._tolerate_unknown_durable_result_fields(
      event,
      source="completion_intent",
    )
    self._durable_top_level_wrapper(event)
    logical = {
      field_name: deepcopy(event[field_name])
      for field_name in SKILL_RESULT_CORE_FIELDS
      if field_name in event
    }
    try:
      return lifecycle.normalize_result_event(logical)
    except (RuntimeError, TypeError, ValueError) as exc:
      raise SkillCompletionWalCorruptError(
        "Completion intent result envelope is invalid"
      ) from exc

  def _logical_terminal_from_completion_intent(
    self,
    event: Dict[str, Any],
    *,
    lifecycle: TopLevelSkillLifecycleMetadata,
  ) -> Dict[str, Any]:
    self._durable_top_level_wrapper(event)
    event_type = event.get("type")
    if event_type not in {"error", "stream_complete"}:
      raise SkillCompletionWalCorruptError(
        "Completion intent terminal type is invalid"
      )
    try:
      lifecycle.require_event_identity(
        event,
        event_type=event_type,
      )
    except RuntimeError as exc:
      raise SkillCompletionWalCorruptError(
        "Completion intent terminal identity is invalid"
      ) from exc
    logical = {
      field_name: deepcopy(value)
      for field_name, value in event.items()
      if field_name not in _DURABLE_TOP_LEVEL_ENVELOPE_FIELDS
    }
    try:
      return self._bind_top_level_terminal_identity(
        logical,
        lifecycle=lifecycle,
      )
    except (RuntimeError, TypeError, ValueError) as exc:
      raise SkillCompletionWalCorruptError(
        "Completion intent terminal envelope is invalid"
      ) from exc

  def _validate_existing_completion_intent(
    self,
    *,
    current: Dict[str, Any],
    expected: Dict[str, Any],
    lifecycle: TopLevelSkillLifecycleMetadata,
    logical_result: Dict[str, Any],
    logical_terminal: Dict[str, Any],
    effect: TopLevelSkillCompletionEffectPlan,
    current_fence: Dict[str, Any],
  ) -> None:
    if current.get("lifecycle") != lifecycle.identity_fields():
      raise SkillCompletionWalCorruptError(
        "Completion intent lifecycle identity is inconsistent"
      )
    durable_result = current.get("result")
    durable_terminal = current.get("terminal")
    if (
      type(durable_result) is not dict
      or type(durable_terminal) is not dict
    ):
      raise SkillCompletionWalCorruptError(
        "Completion intent durable envelopes are missing"
      )
    recovered_result = self._logical_result_from_completion_intent(
      durable_result,
      lifecycle=lifecycle,
    )
    recovered_terminal = self._logical_terminal_from_completion_intent(
      durable_terminal,
      lifecycle=lifecycle,
    )
    if (
      not _exact_value_match(recovered_result, logical_result)
      or not _exact_value_match(
        recovered_terminal,
        logical_terminal,
      )
      or not _exact_value_match(
        current.get("effect"),
        effect.durable_payload(),
      )
      or not _exact_value_match(
        self._durable_top_level_wrapper(durable_result),
        self._durable_top_level_wrapper(durable_terminal),
      )
    ):
      raise SkillCompletionWalCorruptError(
        "Completion intent exact envelopes are inconsistent"
      )
    intent_fence = current.get("fence")
    if not isinstance(intent_fence, dict):
      raise SkillCompletionWalCorruptError(
        "Completion intent fence is missing"
      )
    if _exact_value_match(intent_fence, current_fence):
      comparable = deepcopy(current)
      comparable.pop("checksum", None)
      expected_with_schema = deepcopy(expected)
      expected_with_schema["schema_version"] = 1
      if not _exact_value_match(comparable, expected_with_schema):
        raise SkillCompletionWalCorruptError(
          "Completion WAL changed within the held lease generation"
        )
      return
    if (
      type(intent_fence.get("generation")) is not int
      or intent_fence["generation"] >= current_fence["generation"]
    ):
      raise SkillCompletionWalCorruptError(
        "Completion WAL carries a stale or future fence"
      )

  def _store_completion_tombstone(
    self,
    *,
    wal: SkillCompletionWal,
    record_type: str,
    skill_run_id: str,
    durable_result: Dict[str, Any],
    durable_terminal: Dict[str, Any],
    reason: str,
  ) -> Dict[str, Any]:
    admission = self._require_top_level_admission_fence()
    return wal.store(
      {
        "record_type": record_type,
        "skill_run_id": skill_run_id,
        "result_digest": envelope_digest(durable_result),
        "terminal_digest": envelope_digest(durable_terminal),
        "fence": admission.fence,
        "reason": reason,
      }
    )

  def _append_completion_ambiguity_marker_sync(
    self,
    *,
    lifecycle: TopLevelSkillLifecycleMetadata,
    intent: Dict[str, Any],
  ) -> Dict[str, Any]:
    effect = intent["effect"]
    marker = {
      "type": "runtime_guard",
      "guard": "top_level_skill_completion_ambiguous",
      "message": (
        "Top-level skill completion effect target matches neither "
        "the durable before nor after digest"
      ),
      **lifecycle.identity_fields(),
      "effect_kind": effect["kind"],
      "effect_target": effect["target"],
      "effect_before_digest": effect["before_digest"],
      "effect_after_digest": effect["after_digest"],
      "intent_fence": deepcopy(intent["fence"]),
      **self._durable_top_level_wrapper(intent["result"]),
    }
    return self._append_and_readback_top_level_event_sync(marker)

  def _commit_top_level_skill_plan_sync(
    self,
    *,
    lifecycle: TopLevelSkillLifecycleMetadata,
    result: Dict[str, Any],
    terminal: Dict[str, Any],
    effect: TopLevelSkillCompletionEffectPlan,
    project_live: bool,
    existing_intent: Dict[str, Any] | None = None,
    settlement_reason: str = "exact_result_and_terminal_committed",
    durable_wrapper: Dict[str, Any] | None = None,
    durable_result_override: Dict[str, Any] | None = None,
    durable_terminal_override: Dict[str, Any] | None = None,
  ) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the cancellation-masked WAL/effect/result/terminal segment.

    ``durable_result_override``/``durable_terminal_override`` let
    recovery reuse an envelope that already exists in the durable log
    (possibly carrying unknown fields written by another generation)
    instead of re-deriving a core-only envelope that could never match
    the committed bytes on exact readback. Overrides must agree with
    the validated logical events on every known field.
    """

    if (
      durable_result_override is not None
      or durable_terminal_override is not None
    ) and durable_wrapper is None:
      raise RuntimeError(
        "Durable envelope overrides require a recovery wrapper"
      )
    admission = self._require_top_level_admission_fence()
    canonical_result = lifecycle.normalize_result_event(
      deepcopy(result)
    )
    canonical_terminal = self._bind_top_level_terminal_identity(
      terminal,
      lifecycle=lifecycle,
    )
    if durable_result_override is not None:
      override_core = {
        field_name: durable_result_override[field_name]
        for field_name in SKILL_RESULT_CORE_FIELDS
        if field_name in durable_result_override
      }
      if not _exact_value_match(canonical_result, override_core):
        raise SkillCompletionWalCorruptError(
          "Durable result override diverges from its logical core"
        )
    if durable_terminal_override is not None:
      override_logical = {
        field_name: value
        for field_name, value in durable_terminal_override.items()
        if field_name not in _DURABLE_TOP_LEVEL_ENVELOPE_FIELDS
      }
      if not _exact_value_match(canonical_terminal, override_logical):
        raise SkillCompletionWalCorruptError(
          "Durable terminal override diverges from its logical event"
        )
    if durable_wrapper is None:
      proposed_durable_result = (
        self._expected_durable_top_level_event(canonical_result)
      )
      proposed_durable_terminal = (
        self._expected_durable_top_level_event(canonical_terminal)
      )
    else:
      exact_wrapper = self._durable_top_level_wrapper(
        durable_wrapper
      )
      proposed_durable_result = (
        deepcopy(durable_result_override)
        if durable_result_override is not None
        else {
          **canonical_result,
          **deepcopy(exact_wrapper),
        }
      )
      proposed_durable_terminal = (
        deepcopy(durable_terminal_override)
        if durable_terminal_override is not None
        else {
          **canonical_terminal,
          **deepcopy(exact_wrapper),
        }
      )
    result_wrapper = self._durable_top_level_wrapper(
      proposed_durable_result
    )
    terminal_wrapper = self._durable_top_level_wrapper(
      proposed_durable_terminal
    )
    if not _exact_value_match(result_wrapper, terminal_wrapper):
      raise RuntimeError(
        "Top-level result and terminal durable wrappers diverged"
      )
    wal = self._top_level_completion_wal()
    intent = self._completion_intent_record(
      lifecycle=lifecycle,
      durable_result=proposed_durable_result,
      durable_terminal=proposed_durable_terminal,
      effect=effect,
      fence=admission.fence,
    )
    loaded = wal.load()
    if (
      existing_intent is not None
      and not _exact_value_match(existing_intent, loaded)
    ):
      raise SkillCompletionWalCorruptError(
        "Completion WAL changed after recovery inspection"
      )
    current = loaded
    if current is not None:
      if current.get("record_type") == "settled":
        if not self._settled_wal_is_replaceable(current):
          raise SkillCompletionWalCorruptError(
            "Settled completion WAL lacks exact durable proof"
          )
        current = None
      elif current.get("record_type") == "ambiguous":
        raise SkillCompletionEffectConflict(
          "Completion WAL is blocked by an ambiguous effect"
        )
      elif current.get("record_type") == "intent":
        self._validate_existing_completion_intent(
          current=current,
          expected=intent,
          lifecycle=lifecycle,
          logical_result=canonical_result,
          logical_terminal=canonical_terminal,
          effect=effect,
          current_fence=admission.fence,
        )
    if current is None:
      current = wal.store(intent)
    self._exact_top_level_durable_event_sync(current["result"])
    self._exact_top_level_durable_event_sync(current["terminal"])
    self._require_top_level_admission_fence()
    try:
      apply_completion_effect(
        current["effect"],
        expected_workspace=self._workspace_dir,
      )
    except SkillCompletionEffectConflict:
      self._append_completion_ambiguity_marker_sync(
        lifecycle=lifecycle,
        intent=current,
      )
      self._store_completion_tombstone(
        wal=wal,
        record_type="ambiguous",
        skill_run_id=lifecycle.skill_run_id,
        durable_result=current["result"],
        durable_terminal=current["terminal"],
        reason="effect_target_digest_conflict",
      )
      raise
    self._require_top_level_admission_fence()
    durable_result = self._append_and_readback_top_level_event_sync(
      current["result"]
    )
    durable_terminal = self._append_and_readback_top_level_event_sync(
      current["terminal"]
    )
    self._store_completion_tombstone(
      wal=wal,
      record_type="settled",
      skill_run_id=lifecycle.skill_run_id,
      durable_result=durable_result,
      durable_terminal=durable_terminal,
      reason=settlement_reason,
    )
    if project_live:
      self._top_level_skill_result_event = deepcopy(
        canonical_result
      )
      self._top_level_skill_result_committed = True
      self._project_top_level_skill_event(
        deepcopy(canonical_result)
      )
      self._deferred_top_level_terminal_event = deepcopy(
        canonical_terminal
      )
      self._top_level_skill_terminal_committed = True
      if not getattr(self._log, "has_terminal", False):
        self._append(deepcopy(canonical_terminal))
      terminal_entries = [
        entry.event
        for entry in getattr(self._log, "entries", ())
        if (
          isinstance(getattr(entry, "event", None), dict)
          and entry.event.get("type")
          in {"error", "stream_complete"}
        )
      ]
      if (
        len(terminal_entries) != 1
        or not _exact_value_match(
          terminal_entries[0],
          canonical_terminal,
        )
      ):
        # Operator ruling 2026-08-03: a projection discrepancy in the JSON
        # session log is a diary problem — warn, never veto completed work.
        log.warning(
          "Live EventLog terminal projection is not exact "
          "(%d terminal entries) — proceeding; bookkeeping never vetoes "
          "completed work",
          len(terminal_entries),
        )
      self._deferred_top_level_terminal_flushed = True
    return durable_result, durable_terminal

  def _top_level_skill_failure_result_event(
    self,
    *,
    failure: BaseException,
    failure_code: str,
  ) -> Dict[str, Any]:
    """Preserve prepared evidence while changing only failure semantics."""

    lifecycle = self._top_level_skill_lifecycle
    if lifecycle is None:
      raise RuntimeError(
        "Top-level skill failure receipt requires lifecycle metadata"
      )
    prepared = self._top_level_skill_result_event
    error = (
      f"{failure_code}: {type(failure).__name__}: {failure}"
    )
    if prepared is None:
      event = _runner_attr(
        self,
        "_build_skill_result_failure_event",
        _build_skill_result_failure_event,
      )(
        lifecycle,
        error=error,
      )
    else:
      event = deepcopy(prepared)
      event.update(
        {
          "exit_code": 1,
          "outcome": "error",
          "status": "error",
          "error": error,
        }
      )
    return lifecycle.normalize_result_event(event)

  @staticmethod
  def _top_level_skill_failure_terminal_event(
    *,
    proposed_terminal: Dict[str, Any],
    failure: BaseException,
    failure_code: str,
  ) -> Dict[str, Any]:
    if proposed_terminal.get("type") == "error":
      terminal = deepcopy(proposed_terminal)
      terminal.setdefault("error_type", failure_code)
      terminal.setdefault(
        "error",
        f"{failure_code}: {type(failure).__name__}: {failure}",
      )
      return terminal
    return {
      "type": "error",
      "error_type": failure_code,
      "error": (
        f"{failure_code}: {type(failure).__name__}: {failure}"
      ),
    }

  async def _build_and_persist_final_top_level_skill_result(
    self,
    terminal_event: Dict[str, Any],
  ) -> bool:
    lifecycle = self._top_level_skill_lifecycle
    if lifecycle is None:
      return False
    failure = self._top_level_skill_result_failure
    effective_terminal = deepcopy(terminal_event)
    event = (
      deepcopy(self._top_level_skill_result_event)
      if self._top_level_skill_result_event is not None
      else None
    )
    if failure is None:
      try:
        event = await self._build_top_level_skill_result_event(
          terminal_event
        )
      except BaseException as exc:
        failure = exc
        self._top_level_skill_result_failure = exc
        self._top_level_skill_result_failure_code = (
          "top_level_skill_result_policy_failed"
        )
        policy = self._top_level_skill_result_policy
        prepared_terminal = (
          policy.prepared_terminal_event
          if policy is not None
          else None
        )
        if isinstance(prepared_terminal, dict):
          self._prepared_top_level_terminal_event = deepcopy(
            prepared_terminal
          )
          effective_terminal = deepcopy(prepared_terminal)
      else:
        prepared_terminal = self._prepared_top_level_terminal_event
        if isinstance(prepared_terminal, dict):
          effective_terminal = deepcopy(prepared_terminal)
        self._top_level_skill_result_event = deepcopy(event)
    if failure is not None:
      failure_code = (
        self._top_level_skill_result_failure_code
        or "top_level_skill_result_policy_failed"
      )
      event = self._top_level_skill_failure_result_event(
        failure=failure,
        failure_code=failure_code,
      )
      deferred_terminal = self._deferred_top_level_terminal_event
      effective_terminal = (
        deepcopy(deferred_terminal)
        if deferred_terminal is not None
        else self._top_level_skill_failure_terminal_event(
          proposed_terminal=terminal_event,
          failure=failure,
          failure_code=failure_code,
        )
      )
      effect = TopLevelSkillCompletionEffectPlan.noop()
    else:
      effect = self._top_level_skill_completion_effect_plan
      if not isinstance(
        effect,
        TopLevelSkillCompletionEffectPlan,
      ):
        raise RuntimeError(
          "Top-level completion plan is missing its effect"
        )
    if event is None:
      raise RuntimeError(
        "Top-level skill finalization produced no result event"
      )
    effective_terminal = self._bind_top_level_terminal_identity(
      effective_terminal,
      lifecycle=lifecycle,
    )
    self._defer_top_level_terminal_event(effective_terminal)
    self._top_level_skill_result_event = deepcopy(event)
    self._commit_top_level_skill_plan_sync(
      lifecycle=lifecycle,
      result=deepcopy(event),
      terminal=deepcopy(effective_terminal),
      effect=effect,
      project_live=True,
    )
    return True

  async def _finalize_top_level_skill_result(
    self,
    *,
    run_error: BaseException | None,
    clean_detach_reason: str,
    terminal_event: Dict[str, Any] | None = None,
  ) -> bool:
    lifecycle = self._top_level_skill_lifecycle
    if (
      lifecycle is None
      or not self._top_level_skill_started_committed
      or self._top_level_skill_result_committed
    ):
      return False
    if self._top_level_skill_result_task is None:
      if terminal_event is not None:
        final_terminal_event = deepcopy(terminal_event)
      elif self._top_level_server_terminal_cause() is not None:
        cause = self._top_level_server_terminal_cause()
        final_terminal_event = {
          "type": "stream_complete",
          "terminal_disposition": "interrupted",
          "reason": cause,
          "server_terminal_cause": cause,
        }
      elif run_error is not None:
        final_terminal_event = {
          "type": "error",
          "error_type": type(run_error).__name__,
          "error": _runner_attr(
            self,
            "_format_exc",
            _format_exc,
          )(run_error),
        }
      elif clean_detach_reason != "completed":
        final_terminal_event = {
          "type": "stream_complete",
          "terminal_disposition": "interrupted",
          "reason": clean_detach_reason,
        }
      else:
        final_terminal_event = {
          "type": "error",
          "error": (
            "Top-level skill run ended without committing its "
            "required result"
          ),
        }
      self._top_level_skill_result_task = asyncio.create_task(
        self._build_and_persist_final_top_level_skill_result(
          final_terminal_event
        ),
        name=(
          f"top-level-skill:{lifecycle.skill_run_id}:result"
        ),
      )
    try:
      return bool(
        await asyncio.shield(
          self._top_level_skill_result_task
        )
      )
    except asyncio.CancelledError:
      return bool(
        await drain_owned_lifecycle_task(
          self._top_level_skill_result_task
        )
      )

  def _defer_top_level_terminal_event(
    self,
    event: Dict[str, Any],
  ) -> bool:
    lifecycle = self._top_level_skill_lifecycle
    if lifecycle is None:
      return False
    event = self._bind_top_level_terminal_identity(
      event,
      lifecycle=lifecycle,
    )
    event_type = event.get("type")
    if event_type not in {"error", "stream_complete"}:
      raise RuntimeError(
        "Top-level named-skill terminal must be an error or "
        "stream_complete event"
      )
    deferred = self._deferred_top_level_terminal_event
    if deferred is not None:
      if _exact_value_match(deferred, event):
        return True
      raise RuntimeError(
        "Top-level named-skill run attempted to emit more than one "
        "terminal event"
      )
    self._deferred_top_level_terminal_event = deepcopy(event)
    return True

  def _defer_prepared_top_level_terminal_event(
    self,
    proposed_event: Dict[str, Any],
  ) -> bool:
    prepared = self._prepared_top_level_terminal_event
    return self._defer_top_level_terminal_event(
      deepcopy(
        prepared
        if isinstance(prepared, dict)
        else proposed_event
      )
    )

  async def _flush_deferred_top_level_terminal_event(self) -> bool:
    event = self._deferred_top_level_terminal_event
    if event is None or self._deferred_top_level_terminal_flushed:
      return False
    if (
      not self._top_level_skill_result_committed
      or not self._top_level_skill_result_projected
    ):
      raise RuntimeError(
        "Cannot flush top-level terminal event before its required "
        "skill result is committed and projected"
      )
    if self._top_level_skill_terminal_committed:
      if self._exact_top_level_durable_event_sync(event) is None:
        raise RuntimeError(
          "Committed top-level terminal lacks exact durable readback"
        )
    else:
      durable_entry = await self._append_durable_event(
        deepcopy(event)
      )
      if durable_entry is None:
        raise RuntimeError(
          "Durable top-level terminal target is unavailable"
        )
      self._require_append_acknowledgement(
        durable_entry,
        event,
        target="durable top-level terminal",
      )
      self._top_level_skill_terminal_committed = True
    if not getattr(self._log, "has_terminal", False):
      self._append(deepcopy(event))
      if not getattr(self._log, "has_terminal", False):
        raise RuntimeError(
          "Live EventLog rejected deferred top-level terminal event"
        )
    terminal_entries = [
      entry
      for entry in getattr(self._log, "entries", ())
      if isinstance(getattr(entry, "event", None), dict)
      and entry.event.get("type") in {"error", "stream_complete"}
    ]
    if (
      len(terminal_entries) != 1
      or not _exact_value_match(
        terminal_entries[0].event,
        event,
      )
    ):
      raise RuntimeError(
        "Live EventLog did not acknowledge the exact deferred "
        "top-level terminal event"
      )
    self._deferred_top_level_terminal_flushed = True
    return True

  async def _rebuild_task_registry_from_log(self) -> None:
    if self._task_registry_rebuilt:
      return
    async with self._task_registry_rebuild_lock:
      if self._task_registry_rebuilt:
        return
      if self._agent_session_log is None:
        self._task_registry_rebuilt = True
        return

      entries, _ = await self._agent_session_log.query(
        event_types={
          "task_registered",
          "task_completed",
          "agent_completion",
          "parent_message_sent",
          "skill_result_captured",
        },
        order="desc",
      )
      events: list[dict[str, Any]] = []
      registered_task_ids: set[str] = set()
      max_retained = max(0, getattr(self._task_registry, "_max_retained", 50))
      for entry in entries:
        event = entry.event
        task_id = str(event.get("task_id") or "")
        if not task_id:
          continue
        durable_event = dict(event)
        durable_event["_durable_seq"] = entry.seq
        events.append(durable_event)
        if event.get("type") == "task_registered":
          registered_task_ids.add(task_id)
          if len(registered_task_ids) >= max_retained:
            break
      self._task_registry.load_from_events(events)
      if self._role == "writer":
        required_skill_results = [
          task_entry
          for task_entry in self._task_registry.list_tasks()
          if (
            "required_skill_lifecycle" in task_entry.metadata
            and task_entry.completion_persistence_state == "committed"
            and task_entry.state in {
              TaskState.COMPLETED,
              TaskState.FAILED,
              TaskState.KILLED,
            }
          )
        ]
        if required_skill_results:
          async def _strictly_settle_required_skill_result(
            task_entry: TaskEntry,
          ) -> None:
            lifecycle = self._required_skill_lifecycle(task_entry)
            if lifecycle is None:
              return
            existing = await self._durable_required_skill_result_event(
              task_entry,
              lifecycle,
            )
            task_entry.required_skill_result_settled = (
              existing is not None
            )
            if existing is None:
              await self._ensure_required_skill_result_settled(
                task_entry
              )

          await asyncio.gather(*(
            _strictly_settle_required_skill_result(task_entry)
            for task_entry in required_skill_results
          ))
      self._task_registry_rebuilt = True

  async def _lookup_task_in_log(self, task_id: str) -> TaskEntry | None:
    if self._agent_session_log is None:
      return None
    entries, _ = await self._agent_session_log.query(
      event_types={
        "task_registered",
        "task_completed",
        "agent_completion",
        "parent_message_sent",
        "skill_result_captured",
      },
      order="desc",
    )
    events: list[dict[str, Any]] = []
    found_registration = False
    for entry in entries:
      event = entry.event
      if event.get("task_id") != task_id:
        continue
      durable_event = dict(event)
      durable_event["_durable_seq"] = entry.seq
      events.append(durable_event)
      if event.get("type") == "task_registered":
        found_registration = True
        break
    if not found_registration:
      return None
    registry_type = _runner_attr(self, "TaskRegistry", TaskRegistry)
    registry = registry_type(max_retained=max(1, getattr(self._task_registry, "_max_retained", 50)))
    registry.load_from_events(events)
    return registry.get(task_id)

  async def _emit_attach_event(self) -> None:
    time_module = _runner_attr(self, "time", time)
    socket_module = _runner_attr(self, "socket", socket)
    entry = await self._append_durable_event(
      _runner_attr(self, "_build_attach_event", _build_attach_event)(
        gateway_session_id=self._gateway_session_id,
        started_at=time_module.time(),
        client_kind=self._client_kind,
        hostname=socket_module.gethostname(),
      )
    )
    self._durable_attach_emitted = entry is not None

  async def _append_user_message_event(self, message: Dict[str, Any]) -> Any | None:
    return await self._append_durable_event(
      _runner_attr(self, "_build_user_message_event", _build_user_message_event)(
        content=message.get("content"),
        client_kind=self._client_kind,
        received_at=_runner_attr(self, "time", time).time(),
      )
    )

  async def _append_assistant_message_event(
    self,
    *,
    content_blocks: List[Dict[str, Any]],
    stop_reason: str | None,
    model: str,
    usage: Dict[str, Any],
    parent_messages: list[ParentMessage] | None = None,
    consumer_turn: int | None = None,
    logical_response_id: str | None = None,
    logical_response_segment_ordinal: int | None = None,
    continued_from_assistant_message_seq: int | None = None,
    workflow_output_attachments: list[
      WorkflowOutputAttachment
    ] | None = None,
  ) -> Any | None:
    consumptions: list[dict[str, Any]] = []
    if parent_messages:
      if type(consumer_turn) is not int or consumer_turn <= 0:
        raise RuntimeError(
          "parent message consumption requires a positive turn"
        )
      consumptions = await self._parent_message_consumption_bindings(
        parent_messages,
        consumer_turn=consumer_turn,
      )
    event = _runner_attr(
      self,
      "_build_assistant_message_event",
      _build_assistant_message_event,
    )(
      content_blocks=content_blocks,
      stop_reason=stop_reason,
      model=model,
      provider=getattr(self._provider, "name", None),
      usage=usage,
      logical_response_id=logical_response_id,
      logical_response_segment_ordinal=(
        logical_response_segment_ordinal
      ),
      continued_from_assistant_message_seq=(
        continued_from_assistant_message_seq
      ),
      workflow_output_attachments=workflow_output_attachments,
    )
    if consumptions:
      event["parent_message_consumptions"] = consumptions
    entry = await self._append_durable_event(
      event
    )
    if entry is not None:
      self._last_assistant_message_seq = entry.seq
    for attachment in workflow_output_attachments or ():
      attachment_event = _runner_attr(
        self,
        "_build_workflow_output_attached_event",
        _build_workflow_output_attached_event,
      )(
        attachment=attachment,
        assistant_message_seq=(entry.seq if entry is not None else None),
      )
      await self._append_durable_event(attachment_event)
      self._append(attachment_event)
    if entry is not None:
      for attachment in workflow_output_attachments or ():
        if (
          self._pending_workflow_output_attachments.get(
            attachment.workflow_run_id
          )
          == attachment
        ):
          self._pending_workflow_output_attachments.pop(
            attachment.workflow_run_id,
            None,
          )
    return entry

  async def _parent_message_consumption_bindings(
    self,
    parent_messages: list[ParentMessage],
    *,
    consumer_turn: int,
  ) -> list[dict[str, Any]]:
    """Validate exact sent facts before binding them to an assistant event."""

    durable_log = self._agent_session_log
    if durable_log is None:
      return []
    entries, _ = await durable_log.query(
      event_types={"parent_message_sent"},
      order="asc",
    )
    sent_by_identity: dict[tuple[str, str], list[Any]] = {}
    for entry in entries:
      task_id = entry.event.get("task_id")
      message_id = entry.event.get("message_id")
      if type(task_id) is str and type(message_id) is str:
        sent_by_identity.setdefault((task_id, message_id), []).append(entry)

    bindings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for message in parent_messages:
      if (
        message.task_id is None
        or type(message.sent_seq) is not int
        or message.sent_seq <= 0
      ):
        raise RuntimeError(
          "parent message consumption lacks durable acceptance identity"
        )
      identity = (message.task_id, message.message_id)
      sent_entries = sent_by_identity.get(identity, [])
      if (
        identity in seen
        or len(sent_entries) != 1
        or sent_entries[0].seq != message.sent_seq
        or sent_entries[0].event.get("message") != message.text
        or sent_entries[0].event.get("sub_agent_id") != self._sub_agent_id
      ):
        raise RuntimeError(
          "parent message consumption lacks one exact durable sent fact"
        )
      seen.add(identity)
      bindings.append({
        "task_id": message.task_id,
        "message_id": message.message_id,
        "parent_message_seq": message.sent_seq,
        "consumer_turn": consumer_turn,
      })
    return bindings

  async def _materialize_parent_message_consumption_audits(
    self,
    *,
    assistant_message_seq: int | None = None,
    expected_messages: list[ParentMessage] | None = None,
  ) -> None:
    """Validate assistant-bound consumption and append missing audit events."""

    durable_log = self._agent_session_log
    if durable_log is None:
      return
    entries, _ = await durable_log.query(
      event_types={
        "assistant_message",
        "parent_message_sent",
        "parent_message_consumed",
      },
      order="asc",
    )
    sent_by_identity: dict[tuple[str, str], list[Any]] = {}
    consumed_by_identity: dict[tuple[str, str], list[Any]] = {}
    assistant_entries: list[Any] = []
    for entry in entries:
      event = entry.event
      event_type = event.get("type")
      if event_type == "assistant_message":
        if (
          event.get("sub_agent_id") == self._sub_agent_id
          and (
            assistant_message_seq is None
            or entry.seq == assistant_message_seq
          )
        ):
          assistant_entries.append(entry)
        continue
      task_id = event.get("task_id")
      message_id = event.get("message_id")
      if type(task_id) is not str or type(message_id) is not str:
        continue
      bucket = (
        sent_by_identity
        if event_type == "parent_message_sent"
        else consumed_by_identity
      )
      bucket.setdefault((task_id, message_id), []).append(entry)

    bound: dict[tuple[str, str], tuple[Any, dict[str, Any]]] = {}
    for assistant in assistant_entries:
      raw_bindings = assistant.event.get(
        "parent_message_consumptions",
        [],
      )
      if not isinstance(raw_bindings, list):
        raise RuntimeError(
          "assistant parent-message consumption bindings are invalid"
        )
      for binding in raw_bindings:
        if (
          not isinstance(binding, dict)
          or set(binding) != {
            "task_id",
            "message_id",
            "parent_message_seq",
            "consumer_turn",
          }
          or type(binding.get("task_id")) is not str
          or type(binding.get("message_id")) is not str
          or type(binding.get("parent_message_seq")) is not int
          or binding["parent_message_seq"] <= 0
          or type(binding.get("consumer_turn")) is not int
          or binding["consumer_turn"] <= 0
        ):
          raise RuntimeError(
            "assistant parent-message consumption binding is malformed"
          )
        identity = (binding["task_id"], binding["message_id"])
        if identity in bound:
          raise RuntimeError(
            "parent message is bound to multiple assistant responses"
          )
        sent_entries = sent_by_identity.get(identity, [])
        if (
          len(sent_entries) != 1
          or sent_entries[0].seq != binding["parent_message_seq"]
          or sent_entries[0].seq >= assistant.seq
          or sent_entries[0].event.get("sub_agent_id")
          != assistant.event.get("sub_agent_id")
          or assistant.event.get("role") != "sub_agent"
          or type(assistant.event.get("runner_id")) is not str
        ):
          raise RuntimeError(
            "assistant consumption binding lacks exact sent/child lineage"
          )
        bound[identity] = (assistant, binding)

    if expected_messages is not None:
      expected = {
        (message.task_id, message.message_id): message.sent_seq
        for message in expected_messages
      }
      actual = {
        identity: binding["parent_message_seq"]
        for identity, (_assistant, binding) in bound.items()
      }
      if expected != actual:
        raise RuntimeError(
          "assistant consumption binding does not match consumed messages"
        )

    for identity, (assistant, binding) in bound.items():
      consumed_entries = consumed_by_identity.get(identity, [])
      if len(consumed_entries) > 1:
        raise RuntimeError(
          "parent message has duplicate durable consumption audits"
        )
      if consumed_entries:
        consumed = consumed_entries[0]
        event = consumed.event
        if (
          consumed.seq <= assistant.seq
          or event.get("parent_message_seq")
          != binding["parent_message_seq"]
          or event.get("assistant_message_seq") != assistant.seq
          or event.get("consumer_turn") != binding["consumer_turn"]
          or event.get("runner_id")
          != assistant.event.get("runner_id")
          or event.get("role") != assistant.event.get("role")
          or event.get("sub_agent_id")
          != assistant.event.get("sub_agent_id")
        ):
          raise RuntimeError(
            "parent message consumption audit conflicts with assistant fact"
          )
        continue

      event = {
        "type": "parent_message_consumed",
        "task_id": identity[0],
        "message_id": identity[1],
        "parent_message_seq": binding["parent_message_seq"],
        "consumer_turn": binding["consumer_turn"],
        "assistant_message_seq": assistant.seq,
        "consumed_at": _runner_attr(self, "time", time).time(),
        "runner_id": assistant.event["runner_id"],
        "role": assistant.event["role"],
        "sub_agent_id": assistant.event["sub_agent_id"],
      }
      appended = await self._append_durable_event(event)
      if appended is None:
        raise RuntimeError(
          "parent message consumption audit was not durably appended"
        )
      self._require_append_acknowledgement(
        appended,
        event,
        target="parent message consumption",
      )
      consumed_by_identity[identity] = [appended]

  async def _acknowledge_parent_messages_consumed(
    self,
    parent_messages: list[ParentMessage],
    *,
    consumer_turn: int,
  ) -> None:
    """Materialize the audit projection of the atomic assistant binding."""

    if self._agent_session_log is None or not parent_messages:
      return
    assistant_message_seq = self._last_assistant_message_seq
    if type(assistant_message_seq) is not int or assistant_message_seq <= 0:
      raise RuntimeError(
        "parent message consumption requires a durable assistant message"
      )
    if type(consumer_turn) is not int or consumer_turn <= 0:
      raise RuntimeError("parent message consumption requires a positive turn")
    await self._materialize_parent_message_consumption_audits(
      assistant_message_seq=assistant_message_seq,
      expected_messages=parent_messages,
    )

  async def _emit_stream_retry_event(self, *, attempt: int, error: str) -> None:
    event = _runner_attr(self, "_build_stream_retry_event", _build_stream_retry_event)(attempt=attempt, error=error)
    await self._append_durable_event(event)
    self._append(event)

  async def _emit_error_event(self, error: str) -> None:
    event = _runner_attr(self, "_build_error_event", _build_error_event)(error)
    try:
      await self._call_on_before_stream_complete(event)
    except Exception as exc:
      log.warning(
        "[%s] terminal error hook failed while preserving error closure: %s",
        self._sid,
        exc,
      )
    if self._top_level_skill_lifecycle is not None:
      self._defer_top_level_terminal_event(event)
      return
    await self._append_durable_event(event)
    self._append(event)

  async def _emit_run_error_event(
    self,
    exc: BaseException,
    *,
    phase: str = "run",
    server_terminal_cause: str | None = None,
  ) -> None:
    event = _runner_attr(
      self,
      "_build_run_error_event",
      _build_run_error_event,
    )(
      phase=phase,
      error_type=(
        "ServerTerminalCause"
        if server_terminal_cause is not None
        else type(exc).__name__
      ),
      error=(
        server_terminal_cause
        if server_terminal_cause is not None
        else _runner_attr(
          self,
          "_format_exc",
          _format_exc,
        )(exc)
      ),
    )
    if server_terminal_cause is not None:
      event["reason"] = server_terminal_cause
      event["server_terminal_cause"] = server_terminal_cause
    entry = await self._append_durable_event(event)
    if self._top_level_skill_lifecycle is not None:
      if entry is None:
        raise RuntimeError(
          "Top-level skill durable run-error target is unavailable"
        )
      self._require_append_acknowledgement(
        entry,
        event,
        target="durable top-level run_error",
      )

  async def _emit_interrupted_event(
    self,
    reason: str,
    *,
    runner_id: str | None = None,
    role: str | None = None,
    last_completed_seq: int | None = None,
    recovered_by_runner_id: str | None = None,
    recovered_at: float | None = None,
    extra_fields: Dict[str, Any] | None = None,
  ) -> None:
    event = _runner_attr(
      self,
      "_build_interrupted_event",
      _build_interrupted_event,
    )(
      reason=reason,
      runner_id=runner_id or self._runner_id,
      role=role or self._role,
      last_completed_seq=(
        self._last_durable_seq
        if last_completed_seq is None
        else last_completed_seq
      ),
      recovered_by_runner_id=recovered_by_runner_id,
      recovered_at=recovered_at,
      extra_fields=extra_fields,
    )
    if (
      self._top_level_skill_lifecycle is not None
      and recovered_by_runner_id is None
      and recovered_at is None
    ):
      deferred = self._deferred_top_level_interrupted_event
      if deferred is not None:
        if _exact_value_match(deferred, event):
          return
        raise RuntimeError(
          "Top-level named-skill run attempted to emit more than "
          "one interruption marker"
        )
      self._deferred_top_level_interrupted_event = deepcopy(event)
      return
    entry = await self._append_durable_event(event)
    if self._top_level_skill_lifecycle is not None:
      if entry is None:
        raise RuntimeError(
          "Top-level skill durable interruption target is unavailable"
        )
      self._require_append_acknowledgement(
        entry,
        event,
        target="durable top-level interrupted",
      )

  async def _flush_deferred_top_level_interrupted_event(
    self,
  ) -> bool:
    if self._top_level_skill_lifecycle is None:
      return False
    if (
      not self._top_level_skill_result_committed
      or not self._top_level_skill_result_projected
      or not self._deferred_top_level_terminal_flushed
    ):
      raise RuntimeError(
        "Cannot flush top-level interruption or detach before its "
        "exact result and terminal event"
      )
    event = self._deferred_top_level_interrupted_event
    if event is None:
      return False
    if self._deferred_top_level_interrupted_flushed:
      return False
    entry = await self._append_durable_event(
      deepcopy(event)
    )
    if entry is None:
      raise RuntimeError(
        "Top-level skill durable interruption target is unavailable"
      )
    self._require_append_acknowledgement(
      entry,
      event,
      target="durable top-level interrupted",
    )
    self._deferred_top_level_interrupted_flushed = True
    return True

  def _shutdown_interrupted_reason(self) -> tuple[str, Dict[str, Any]]:
    if self._shutdown_signal_provider is None:
      return "graceful_shutdown", {}
    try:
      signal_payload = self._shutdown_signal_provider()
    except Exception as exc:
      log.warning("[%s] shutdown signal provider failed (non-fatal): %s", self._sid, exc)
      return "graceful_shutdown", {}
    return _runner_attr(self, "_shutdown_interrupted_reason", _shutdown_interrupted_reason)(signal_payload)

  def _lifecycle_from_durable_start(
    self,
    event: Dict[str, Any],
  ) -> TopLevelSkillLifecycleMetadata:
    try:
      if (
        event.get(_TOP_LEVEL_SKILL_LIFECYCLE_ORIGIN_FIELD)
        != _TOP_LEVEL_SKILL_LIFECYCLE_ORIGIN
      ):
        raise ValueError(
          "durable start is not an explicit top-level lifecycle"
        )
      self._durable_top_level_wrapper(event)
      lifecycle = TopLevelSkillLifecycleMetadata(
        skill_run_id=event["skill_run_id"],
        skill=event["skill"],
        scope=event["scope"],
        ticker=event["ticker"],
        portfolio_id=event["portfolio_id"],
      )
      lifecycle.require_event_identity(
        event,
        event_type="skill_run_started",
      )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
      raise SkillCompletionWalCorruptError(
        "Durable top-level skill start has invalid lifecycle identity"
      ) from exc
    return lifecycle

  def _logical_result_from_recovery_log(
    self,
    event: Dict[str, Any],
    *,
    lifecycle: TopLevelSkillLifecycleMetadata,
  ) -> Dict[str, Any]:
    self._tolerate_unknown_durable_result_fields(
      event,
      source="recovery_log",
    )
    logical = {
      field_name: deepcopy(event[field_name])
      for field_name in SKILL_RESULT_CORE_FIELDS
      if field_name in event
    }
    try:
      return lifecycle.normalize_result_event(logical)
    except (RuntimeError, TypeError, ValueError) as exc:
      raise SkillCompletionWalCorruptError(
        "Recovered top-level result is invalid"
      ) from exc

  def _logical_terminal_from_recovery_log(
    self,
    event: Dict[str, Any],
    *,
    lifecycle: TopLevelSkillLifecycleMetadata,
  ) -> Dict[str, Any]:
    event_type = event.get("type")
    if event_type not in {"error", "stream_complete"}:
      raise SkillCompletionWalCorruptError(
        "Recovered top-level terminal type is invalid"
      )
    try:
      lifecycle.require_event_identity(
        event,
        event_type=event_type,
      )
    except RuntimeError as exc:
      raise SkillCompletionWalCorruptError(
        "Recovered top-level terminal identity is invalid"
      ) from exc
    logical = {
      field_name: deepcopy(value)
      for field_name, value in event.items()
      if field_name not in _DURABLE_TOP_LEVEL_ENVELOPE_FIELDS
    }
    try:
      return self._bind_top_level_terminal_identity(
        logical,
        lifecycle=lifecycle,
      )
    except (RuntimeError, TypeError, ValueError) as exc:
      raise SkillCompletionWalCorruptError(
        "Recovered top-level terminal is invalid"
      ) from exc

  @staticmethod
  def _recovery_terminal_for_result(
    result: Dict[str, Any],
  ) -> Dict[str, Any]:
    if (
      result.get("exit_code") == 0
      and result.get("outcome") == "success"
    ):
      return {
        "type": "stream_complete",
        "terminal_disposition": "completed",
        "reason": "recovered_after_result_commit",
      }
    error = result.get("error")
    if not isinstance(error, str) or not error:
      error = (
        "Recovered orphaned top-level skill run without a "
        "durable terminal"
      )
    return {
      "type": "error",
      "error_type": "top_level_skill_recovered_orphan",
      "error": error,
    }

  @staticmethod
  def _recovery_interrupted_envelope(
    *,
    lifecycle: TopLevelSkillLifecycleMetadata,
    start_entry: Any,
  ) -> Dict[str, Any]:
    start_event = start_entry.event
    runner_id = start_event.get("runner_id")
    if not isinstance(runner_id, str) or not runner_id:
      runner_id = (
        "recovered_top_level_skill:"
        f"{lifecycle.skill_run_id}"
      )
    return {
      "type": "interrupted",
      "reason": "recovered_on_attach",
      "runner_id": runner_id,
      "role": "writer",
      "last_completed_seq": start_entry.seq,
      **lifecycle.identity_fields(),
      "recovery_kind": "top_level_skill_orphan_reconciled",
      "event_schema_version": EVENT_SCHEMA_VERSION,
    }

  def _recover_top_level_skill_lifecycles_sync(self) -> None:
    """Reconcile every old top-level start before the new attach."""

    wal = self._top_level_completion_wal()
    wal_record = wal.load()
    if (
      wal_record is not None
      and wal_record.get("record_type") == "ambiguous"
    ):
      raise SkillCompletionEffectConflict(
        "Completion WAL is blocked by an ambiguous effect"
      )
    if (
      wal_record is not None
      and wal_record.get("record_type") == "settled"
      and not self._settled_wal_is_replaceable(wal_record)
    ):
      raise SkillCompletionWalCorruptError(
        "Settled completion WAL lacks exact durable proof"
      )

    query_sync = getattr(self._agent_session_log, "query_sync", None)
    if not callable(query_sync):
      raise RuntimeError(
        "Durable session log lacks synchronous lifecycle recovery"
      )
    entries, _ = query_sync(
      event_types={
        "skill_run_started",
        "skill_result_captured",
        "error",
        "stream_complete",
        "interrupted",
      },
      order="asc",
    )
    starts: dict[str, Any] = {}
    results: dict[str, list[Any]] = {}
    terminals: dict[str, list[Any]] = {}
    interruptions: dict[str, list[Any]] = {}
    for entry in entries:
      event = entry.event
      skill_run_id = event.get("skill_run_id")
      if not isinstance(skill_run_id, str) or not skill_run_id:
        continue
      event_type = event.get("type")
      if event_type == "skill_run_started":
        if (
          event.get(_TOP_LEVEL_SKILL_LIFECYCLE_ORIGIN_FIELD)
          != _TOP_LEVEL_SKILL_LIFECYCLE_ORIGIN
        ):
          continue
        if skill_run_id in starts:
          raise SkillCompletionWalCorruptError(
            "Duplicate durable top-level skill starts"
          )
        starts[skill_run_id] = entry
      elif event_type == "skill_result_captured":
        results.setdefault(skill_run_id, []).append(entry)
      elif event_type in {"error", "stream_complete"}:
        terminals.setdefault(skill_run_id, []).append(entry)
      elif (
        event_type == "interrupted"
        and event.get("recovery_kind")
        == "top_level_skill_orphan_reconciled"
      ):
        interruptions.setdefault(skill_run_id, []).append(entry)

    wal_skill_run_id = (
      wal_record.get("skill_run_id")
      if wal_record is not None
      else None
    )
    if (
      wal_record is not None
      and wal_skill_run_id not in starts
    ):
      raise SkillCompletionWalCorruptError(
        "Completion WAL has no matching durable skill start"
      )

    for skill_run_id, start_entry in starts.items():
      lifecycle = self._lifecycle_from_durable_start(
        start_entry.event
      )
      start_wrapper = self._durable_top_level_wrapper(
        start_entry.event
      )
      lifecycle_results = results.get(skill_run_id, [])
      lifecycle_terminals = terminals.get(skill_run_id, [])
      lifecycle_interruptions = interruptions.get(
        skill_run_id,
        [],
      )
      if (
        len(lifecycle_results) > 1
        or len(lifecycle_terminals) > 1
        or len(lifecycle_interruptions) > 1
      ):
        raise SkillCompletionWalCorruptError(
          "Duplicate durable top-level lifecycle envelopes"
        )
      for durable_entry in (
        *lifecycle_results,
        *lifecycle_terminals,
      ):
        recovered_wrapper = self._durable_top_level_wrapper(
          durable_entry.event
        )
        if not _exact_value_match(
          start_wrapper,
          recovered_wrapper,
        ):
          raise SkillCompletionWalCorruptError(
            "Top-level lifecycle durable wrapper drifted"
          )
      expected_interruption = self._recovery_interrupted_envelope(
        lifecycle=lifecycle,
        start_entry=start_entry,
      )
      if lifecycle_interruptions and not _exact_value_match(
        lifecycle_interruptions[0].event,
        expected_interruption,
      ):
        raise SkillCompletionWalCorruptError(
          "Recovered top-level interruption marker is not exact"
        )

      matching_wal = (
        wal_record
        if wal_skill_run_id == skill_run_id
        else None
      )
      has_result = bool(lifecycle_results)
      has_terminal = bool(lifecycle_terminals)
      recovered_logical_result = (
        self._logical_result_from_recovery_log(
          lifecycle_results[0].event,
          lifecycle=lifecycle,
        )
        if has_result
        else None
      )
      recovered_logical_terminal = (
        self._logical_terminal_from_recovery_log(
          lifecycle_terminals[0].event,
          lifecycle=lifecycle,
        )
        if has_terminal
        else None
      )
      if (
        matching_wal is None
        and has_result
        and has_terminal
      ):
        continue
      if (
        matching_wal is not None
        and matching_wal.get("record_type") == "settled"
      ):
        if (
          matching_wal.get("reason")
          == "recovered_exact_result_and_terminal_committed"
          and not lifecycle_interruptions
        ):
          self._append_and_readback_top_level_event_sync(
            expected_interruption
          )
        continue
      if lifecycle_interruptions:
        raise SkillCompletionWalCorruptError(
          "Recovery marker precedes exact result and terminal proof"
        )

      if (
        matching_wal is not None
        and matching_wal.get("record_type") == "intent"
      ):
        if matching_wal.get("lifecycle") != lifecycle.identity_fields():
          raise SkillCompletionWalCorruptError(
            "Completion intent does not match its durable skill start"
          )
        durable_result = matching_wal["result"]
        durable_terminal = matching_wal["terminal"]
        if (
          not _exact_value_match(
            start_wrapper,
            self._durable_top_level_wrapper(durable_result),
          )
          or not _exact_value_match(
            start_wrapper,
            self._durable_top_level_wrapper(durable_terminal),
          )
        ):
          raise SkillCompletionWalCorruptError(
            "Completion intent durable wrapper does not match its start"
          )
        logical_result = self._logical_result_from_completion_intent(
          durable_result,
          lifecycle=lifecycle,
        )
        logical_terminal = self._logical_terminal_from_completion_intent(
          durable_terminal,
          lifecycle=lifecycle,
        )
        effect = (
          TopLevelSkillCompletionEffectPlan.from_durable_payload(
            matching_wal["effect"],
            expected_workspace=self._workspace_dir,
          )
        )
        existing_intent = matching_wal
        durable_result_override = None
        durable_terminal_override = None
      else:
        if has_result:
          assert recovered_logical_result is not None
          logical_result = recovered_logical_result
        else:
          if has_terminal:
            assert recovered_logical_terminal is not None
            logical_terminal = recovered_logical_terminal
            if (
              logical_terminal.get("type") == "stream_complete"
              and logical_terminal.get("terminal_disposition")
              != "interrupted"
            ):
              raise SkillCompletionWalCorruptError(
                "Successful terminal exists without its required result"
              )
            recovery_error = str(
              logical_terminal.get("error")
              or logical_terminal.get("reason")
              or "orphaned top-level skill run"
            )
          else:
            recovery_error = (
              "Recovered orphaned top-level skill run without a "
              "durable completion intent"
            )
          logical_result = _runner_attr(
            self,
            "_build_skill_result_failure_event",
            _build_skill_result_failure_event,
          )(
            lifecycle,
            error=recovery_error,
          )
        if has_terminal:
          assert recovered_logical_terminal is not None
          logical_terminal = recovered_logical_terminal
        else:
          logical_terminal = self._recovery_terminal_for_result(
            logical_result
          )
        effect = TopLevelSkillCompletionEffectPlan.noop()
        existing_intent = None
        durable_result_override = (
          lifecycle_results[0].event if has_result else None
        )
        durable_terminal_override = (
          lifecycle_terminals[0].event if has_terminal else None
        )

      self._commit_top_level_skill_plan_sync(
        lifecycle=lifecycle,
        result=logical_result,
        terminal=logical_terminal,
        effect=effect,
        project_live=False,
        existing_intent=existing_intent,
        settlement_reason=(
          "recovered_exact_result_and_terminal_committed"
        ),
        durable_wrapper=start_wrapper,
        durable_result_override=durable_result_override,
        durable_terminal_override=durable_terminal_override,
      )
      self._append_and_readback_top_level_event_sync(
        expected_interruption
      )

  async def _emit_detach_event(self, reason: str) -> None:
    if not self._durable_attach_emitted:
      return
    event = _runner_attr(
      self,
      "_build_detach_event",
      _build_detach_event,
    )(
      reason=reason,
      ended_at=_runner_attr(self, "time", time).time(),
    )
    entry = await self._append_durable_event(event)
    if self._top_level_skill_lifecycle is not None:
      if entry is None:
        raise RuntimeError(
          "Top-level skill durable detach target is unavailable"
        )
      self._require_append_acknowledgement(
        entry,
        event,
        target="durable top-level detach",
      )
    if entry is not None:
      self._top_level_skill_detach_committed = True

  async def _emit_operator_pause_event(self, safe_boundary: str) -> None:
    event = _runner_attr(self, "_build_operator_pause_event", _build_operator_pause_event)(safe_boundary)
    self._append(event)
    await self._emit_interrupted_event("operator_pause", extra_fields={"safe_boundary": safe_boundary})

  async def _acquire_writer_lease_and_recover(self) -> None:
    if self._agent_session_log is None or self._role != "writer":
      return

    admission = getattr(self, "_top_level_skill_admission", None)
    if admission is not None:
      if admission.state != "transferred":
        raise RuntimeError(
          "Top-level skill admission was not transferred to AgentRunner"
        )
      if self._write_lease_file is None:
        raise RuntimeError(
          "Transferred top-level skill admission lost its lease"
        )
      self._recover_top_level_skill_lifecycles_sync()
      await self._assert_fresh_top_level_skill_lifecycle()
    else:
      fcntl_module = _runner_attr(self, "fcntl", fcntl)
      lease_file = self._agent_session_log.write_lease_path.open("a+b")
      try:
        fcntl_module.flock(lease_file.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
      except BlockingIOError as exc:
        lease_file.close()
        raise WriterLeaseAlreadyHeldError(f"Writer lease already held for {self._agent_session_log.path}") from exc
      self._write_lease_file = lease_file

    last_known_safe_seq = 0
    safe_entries, _ = await self._agent_session_log.query(
      event_types={"detach", "interrupted"},
      role="writer",
      order="desc",
      limit=1,
    )
    if safe_entries:
      last_known_safe_seq = safe_entries[0].seq
      self._last_durable_seq = last_known_safe_seq

    prior_writer_runner_id: str | None = None
    writer_lifecycle, _ = await self._agent_session_log.query(
      event_types={"attach", "detach", "interrupted"},
      role="writer",
      order="desc",
      limit=1,
    )
    if writer_lifecycle and writer_lifecycle[0].event.get("type") == "attach":
      prior_writer_runner_id = str(writer_lifecycle[0].event.get("runner_id") or "")

    orphan_entries, _ = await self._agent_session_log.query(
      event_types={"tool_call_start", "tool_call_complete", "tool_call_interrupted"},
      after_seq=last_known_safe_seq + 1,
      order="asc",
    )
    discovered_at = _runner_attr(self, "time", time).time()
    for synthetic_event in _runner_attr(
      self,
      "_build_orphan_tool_call_interrupted_events",
      _build_orphan_tool_call_interrupted_events,
    )(
      orphan_entries,
      discovered_at=discovered_at,
      tool_risk_for_tool=_runner_attr(self, "_get_tool_risk_value", _get_tool_risk_value),
    ):
      await self._append_durable_event(synthetic_event)

    if prior_writer_runner_id:
      await self._emit_interrupted_event(
        "recovered_on_attach",
        runner_id=prior_writer_runner_id,
        role="writer",
        last_completed_seq=last_known_safe_seq,
        recovered_by_runner_id=self._runner_id,
        recovered_at=discovered_at,
      )

  def _write_lease_metadata(self) -> None:
    if self._agent_session_log is None or self._role != "writer" or self._runner_id is None:
      return
    time_module = _runner_attr(self, "time", time)
    socket_module = _runner_attr(self, "socket", socket)
    _runner_attr(self, "_write_lease_metadata", _write_lease_metadata)(
      self._agent_session_log,
      role=self._role,
      runner_id=self._runner_id,
      gateway_session_id=self._gateway_session_id,
      started_at=time_module.time(),
      hostname=socket_module.gethostname(),
    )

  def _release_write_lease(self) -> None:
    if getattr(self, "_writer_lease_poisoned", False):
      self._deferred_write_lease_release = True
      return
    agent_session_log = getattr(self, "_agent_session_log", None)
    durable_append_futures = tuple(
      getattr(
        agent_session_log,
        "pending_append_futures",
        (),
      )
      if agent_session_log is not None
      else ()
    )
    pending_operations = (
      *getattr(
        self,
        "_pending_background_completion_appends",
        set(),
      ),
      *getattr(
        self,
        "_pending_background_initializations",
        set(),
      ),
    )
    unsettled_operations = tuple(
      task
      for task in (*pending_operations, *durable_append_futures)
      if not task.done()
    )
    if self._write_lease_file is not None and unsettled_operations:
      self._deferred_write_lease_release = True
      lease_waiters = getattr(
        self,
        "_write_lease_settlement_waiters",
        None,
      )
      if lease_waiters is None:
        lease_waiters = set()
        self._write_lease_settlement_waiters = lease_waiters

      def _release_after_settlement(operation: Any) -> None:
        lease_waiters.discard(operation)
        self._release_write_lease()

      for operation in unsettled_operations:
        if operation not in lease_waiters:
          lease_waiters.add(operation)
          operation.add_done_callback(_release_after_settlement)
      return
    self._deferred_write_lease_release = False
    admission = getattr(self, "_top_level_skill_admission", None)
    if admission is not None:
      admission.release()
      self._write_lease_file = None
    else:
      _runner_attr(self, "_release_write_lease", _release_write_lease)(
        self._write_lease_file,
        clear_write_lease_file=lambda: setattr(self, "_write_lease_file", None),
      )

  async def _await_write_lease_handoff(self) -> bool:
    """Drain every tracked append before handing off the writer lease."""

    if self._write_lease_file is None:
      return False
    while True:
      agent_session_log = self._agent_session_log
      candidates = (
        *getattr(
          self,
          "_pending_background_completion_appends",
          set(),
        ),
        *getattr(
          self,
          "_pending_background_initializations",
          set(),
        ),
        *(
          getattr(
            agent_session_log,
            "pending_append_futures",
            (),
          )
          if agent_session_log is not None
          else ()
        ),
      )
      pending_by_identity = {
        id(operation): operation
        for operation in candidates
        if not operation.done()
      }
      if not pending_by_identity:
        break

      async def _drain_pending() -> None:
        await asyncio.gather(
          *pending_by_identity.values(),
          return_exceptions=True,
        )

      drain_task = asyncio.create_task(
        _drain_pending(),
        name="writer-lease:drain-pending-appends",
      )
      await drain_owned_lifecycle_task(drain_task)

    # A poison flag fences release while operations are unresolved. At this
    # point every tracked operation has settled and the final durable markers
    # have been written, so recovery may safely take over on the next lease.
    self._writer_lease_poisoned = False
    self._release_write_lease()
    if self._write_lease_file is not None:
      raise RuntimeError(
        "Writer lease remained held after settlement handoff"
      )
    return True

  def _publish_top_level_skill_settlement(
    self,
    error: BaseException | None,
  ) -> bool:
    if self._top_level_skill_lifecycle is None:
      return False
    if self._top_level_skill_settlement_complete.is_set():
      return False
    settlement_error = error
    try:
      missing: list[str] = []
      if self._write_lease_file is not None:
        missing.append("lease_handoff")
      if self._top_level_skill_started_committed:
        if not self._top_level_skill_result_committed:
          missing.append("result")
        if not self._top_level_skill_result_projected:
          missing.append("result_projection")
        elif self._top_level_skill_result_event is None:
          missing.append("result_projection_exact")
        else:
          lifecycle = self._top_level_skill_lifecycle
          live_results = [
            entry.event
            for entry in getattr(self._log, "entries", ())
            if isinstance(getattr(entry, "event", None), dict)
            and entry.event.get("type") == "skill_result_captured"
            and entry.event.get("skill_run_id")
            == lifecycle.skill_run_id
          ]
          if (
            len(live_results) != 1
            or not _exact_value_match(
              live_results[0],
              self._top_level_skill_result_event,
            )
          ):
            missing.append("result_projection_exact")
        if not self._deferred_top_level_terminal_flushed:
          missing.append("terminal")
        if (
          self._deferred_top_level_interrupted_event is not None
          and not self._deferred_top_level_interrupted_flushed
        ):
          missing.append("interrupted")
        if (
          self._durable_attach_emitted
          and not self._top_level_skill_detach_committed
        ):
          missing.append("detach")
      if missing:
        incomplete_error = RuntimeError(
          "Cannot publish incomplete top-level settlement: "
          + ", ".join(missing)
        )
        if settlement_error is not None:
          incomplete_error.__cause__ = settlement_error
        settlement_error = incomplete_error
    except BaseException as publish_error:
      if settlement_error is not None:
        publish_error.__cause__ = settlement_error
      settlement_error = publish_error
    finally:
      self._top_level_skill_settlement_error = settlement_error
      self._top_level_skill_settlement_complete.set()
    return True
