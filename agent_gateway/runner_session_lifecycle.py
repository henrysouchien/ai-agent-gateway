from __future__ import annotations

import fcntl
import logging
import socket
import sys
import time
from typing import Any, Dict, List

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
  build_stream_retry_event as _build_stream_retry_event,
  build_user_message_event as _build_user_message_event,
  durable_event_payload as _durable_event_payload,
  release_write_lease as _release_write_lease,
  shutdown_interrupted_reason as _shutdown_interrupted_reason,
  write_lease_metadata as _write_lease_metadata,
)
from .runner_tool_audit import get_tool_risk_value as _get_tool_risk_value
from .task_registry import TaskEntry, TaskRegistry


log = logging.getLogger("agent_gateway.runner")


def _runner_attr(instance: Any, name: str, fallback: Any) -> Any:
  for cls in type(instance).__mro__:
    module = sys.modules.get(getattr(cls, "__module__", ""))
    if module is not None and getattr(module, "AgentRunner", None) is cls:
      return getattr(module, name, fallback)
  module = sys.modules.get("agent_gateway.runner")
  if module is None:
    return fallback
  return getattr(module, name, fallback)


class RunnerSessionLifecycleMixin:
  async def _append_durable_event(self, event: Dict[str, Any]) -> Any | None:
    if self._agent_session_log is None or self._runner_id is None:
      return None
    durable_event_payload = _runner_attr(self, "_durable_event_payload", _durable_event_payload)
    product_id = _runner_attr(self, "gateway_product_id", gateway_product_id)()
    payload = durable_event_payload(
      event,
      runner_id=self._runner_id,
      role=self._role,
      sub_agent_id=self._sub_agent_id,
      product_id=product_id,
    )
    entry = await self._agent_session_log.append(payload)
    self._last_durable_seq = entry.seq
    return entry

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
        event_types={"task_registered", "task_completed", "parent_message_sent"},
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
        events.append(dict(event))
        if event.get("type") == "task_registered":
          registered_task_ids.add(task_id)
          if len(registered_task_ids) >= max_retained:
            break
      self._task_registry.load_from_events(events)
      self._task_registry_rebuilt = True

  async def _lookup_task_in_log(self, task_id: str) -> TaskEntry | None:
    if self._agent_session_log is None:
      return None
    entries, _ = await self._agent_session_log.query(
      event_types={"task_registered", "task_completed", "parent_message_sent"},
      order="desc",
    )
    events: list[dict[str, Any]] = []
    found_registration = False
    for entry in entries:
      event = entry.event
      if event.get("task_id") != task_id:
        continue
      events.append(dict(event))
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

  async def _append_user_message_event(self, message: Dict[str, Any]) -> None:
    await self._append_durable_event(
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
    usage: Dict[str, int],
  ) -> None:
    entry = await self._append_durable_event(
      _runner_attr(self, "_build_assistant_message_event", _build_assistant_message_event)(
        content_blocks=content_blocks,
        stop_reason=stop_reason,
        model=model,
        provider=getattr(self._provider, "name", None),
        usage=usage,
      )
    )
    if entry is not None:
      self._last_assistant_message_seq = entry.seq

  async def _emit_stream_retry_event(self, *, attempt: int, error: str) -> None:
    event = _runner_attr(self, "_build_stream_retry_event", _build_stream_retry_event)(attempt=attempt, error=error)
    await self._append_durable_event(event)
    self._append(event)

  async def _emit_error_event(self, error: str) -> None:
    event = _runner_attr(self, "_build_error_event", _build_error_event)(error)
    await self._call_on_before_stream_complete(event)
    await self._append_durable_event(event)
    self._append(event)

  async def _emit_run_error_event(self, exc: BaseException, *, phase: str = "run") -> None:
    await self._append_durable_event(
      _runner_attr(self, "_build_run_error_event", _build_run_error_event)(
        phase=phase,
        error_type=type(exc).__name__,
        error=_runner_attr(self, "_format_exc", _format_exc)(exc),
      )
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
    await self._append_durable_event(
      _runner_attr(self, "_build_interrupted_event", _build_interrupted_event)(
        reason=reason,
        runner_id=runner_id or self._runner_id,
        role=role or self._role,
        last_completed_seq=self._last_durable_seq if last_completed_seq is None else last_completed_seq,
        recovered_by_runner_id=recovered_by_runner_id,
        recovered_at=recovered_at,
        extra_fields=extra_fields,
      )
    )

  def _shutdown_interrupted_reason(self) -> tuple[str, Dict[str, Any]]:
    if self._shutdown_signal_provider is None:
      return "graceful_shutdown", {}
    try:
      signal_payload = self._shutdown_signal_provider()
    except Exception as exc:
      log.warning("[%s] shutdown signal provider failed (non-fatal): %s", self._sid, exc)
      return "graceful_shutdown", {}
    return _runner_attr(self, "_shutdown_interrupted_reason", _shutdown_interrupted_reason)(signal_payload)

  async def _emit_detach_event(self, reason: str) -> None:
    if not self._durable_attach_emitted:
      return
    await self._append_durable_event(
      _runner_attr(self, "_build_detach_event", _build_detach_event)(
        reason=reason,
        ended_at=_runner_attr(self, "time", time).time(),
      )
    )

  async def _emit_operator_pause_event(self, safe_boundary: str) -> None:
    event = _runner_attr(self, "_build_operator_pause_event", _build_operator_pause_event)(safe_boundary)
    self._append(event)
    await self._emit_interrupted_event("operator_pause", extra_fields={"safe_boundary": safe_boundary})

  async def _acquire_writer_lease_and_recover(self) -> None:
    if self._agent_session_log is None or self._role != "writer":
      return

    fcntl_module = _runner_attr(self, "fcntl", fcntl)
    lease_file = self._agent_session_log.write_lease_path.open("a+b")
    try:
      fcntl_module.flock(lease_file.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
    except BlockingIOError as exc:
      lease_file.close()
      raise RuntimeError(f"Writer lease already held for {self._agent_session_log.path}") from exc
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
    _runner_attr(self, "_release_write_lease", _release_write_lease)(
      self._write_lease_file,
      clear_write_lease_file=lambda: setattr(self, "_write_lease_file", None),
    )
