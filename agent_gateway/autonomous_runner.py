from __future__ import annotations

import asyncio
import json
import logging
import os  # noqa: F401 - compatibility alias for autonomous_runner_state
import secrets
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping

from .autonomous_capability_handoff import AutonomousCapabilityBindingResolver
from .autonomous_approval_channel import (
  AutonomousApprovalChannelAuthority,
  AutonomousApprovalChannelParent,
  AutonomousApprovalChannelProtocolError,
  AutonomousApprovalDecision,
)
from .autonomous_approval_ack import (
  require_durable_autonomous_approval_acknowledgement,
)
from .autonomous_control_files import (
  AutonomousControlAppendError,
  adopt_open_json_record_file,
  append_closed_json_record,
  append_open_json_record,
  fsync_owned_file_directory,
  iter_closed_json_records,
  require_appendable_owned_file,
)
from .autonomous_control_contract import (
  AUTONOMOUS_OPERATOR_AGGREGATE_BYTES_LIMIT,
  AUTONOMOUS_OPERATOR_MESSAGE_LIMIT,
  AUTONOMOUS_OPERATOR_RECORD_FIELDS,
  encode_closed_control_record,
)
from .autonomous_launch_envelope import AutonomousControlAuthority
from .claim_signing_authority import GatewayClaimSigningAuthority
from .capability_binding import CredentialHandle
from .autonomous_event_channel import (
  AutonomousEventRecord,
  ReceivedAutonomousEventStream,
)
from .autonomous_runner_claims import (
  _AGENT_API_CLAIM_AUDIENCE as _AGENT_API_CLAIM_AUDIENCE,
  _AGENT_API_CLAIM_ENV_VARS as _AGENT_API_CLAIM_ENV_VARS,
  _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT as _AGENT_API_CLAIM_TTL_SECONDS_DEFAULT,
  get_agent_api_claim_ttl_seconds as get_agent_api_claim_ttl_seconds,  # noqa: F401 - compatibility alias
  sign_user_claim as sign_user_claim,  # noqa: F401 - compatibility alias
)
from .autonomous_runner_state import (
  AUTONOMOUS_TERMINAL_STATES,
  _ACTIVE_AUTONOMOUS_PROCESS_STATES,
  _AUTONOMOUS_MANIFEST_FILE_RE as _AUTONOMOUS_MANIFEST_FILE_RE,
  _AUTONOMOUS_RUN_FILE_RE as _AUTONOMOUS_RUN_FILE_RE,
  _AUTONOMOUS_TASK_ID_RE as _AUTONOMOUS_TASK_ID_RE,
  _ManifestTrackedList as _ManifestTrackedList,
  _REHYDRATED_ACTIVE_STATES,
  _REHYDRATION_INTERRUPTED_ERROR as _REHYDRATION_INTERRUPTED_ERROR,
  _RUN_SEQUENCE_CURSOR_FILE as _RUN_SEQUENCE_CURSOR_FILE,
  _SKIP_WARNED as _SKIP_WARNED,
  _SKIP_WARNED_FILE as _SKIP_WARNED_FILE,
  _SKIP_WARNED_LOADED_DIRS as _SKIP_WARNED_LOADED_DIRS,
  _TASK_MANIFEST_VERSION as _TASK_MANIFEST_VERSION,
  AutonomousRegistryStateMixin,
  AutonomousTask,
  is_root_run_event,
  is_root_terminal_event,
)
from . import autonomous_runner_events as _runner_events
from . import autonomous_runner_status as _runner_status
from . import autonomous_runner_commands as _runner_commands
from .autonomous_runner_start import (
  _SPAWN_CLEANUP_GRACE_SEC as _SPAWN_CLEANUP_GRACE_SEC,
  _get_process_group_id as _get_process_group_id,
  _signal_process_group as _signal_process_group,
  AutonomousRegistryStartMixin,
)
from .autonomous_run_lock import AutonomousRunMutationLock
from .ui_blocks_metrics import record as record_package_counter

_STATUS_TAIL_LINES = 40
_POST_EXIT_SETTLE_SECONDS = 5.0
_AUTONOMOUS_PROFILE_NAME_RE = _runner_commands._AUTONOMOUS_PROFILE_NAME_RE
_LOGGER = logging.getLogger(__name__)
_APPROVAL_DECISION_AUTONOMOUS_STATES = {"running", "approval_pending", "remediating"}

SkillResumeAllowedResolver = Callable[[str], bool]


def _required_control_text(
  value: Any,
  *,
  field_name: str,
  max_length: int = 512,
) -> str:
  if (
    type(value) is not str
    or not value
    or value != value.strip()
    or len(value) > max_length
    or "\x00" in value
  ):
    raise ValueError(f"{field_name} must be a canonical non-empty string")
  return value


def _control_endpoint_identity(
  record: AutonomousTask,
  *,
  endpoint: str,
) -> tuple[Path, int, int]:
  authority = record.control_authority
  if type(authority) is not AutonomousControlAuthority:
    raise RuntimeError(
      "autonomous control append requires exact signed authority"
    )
  path = getattr(record, f"{endpoint}_path")
  signed_path = getattr(authority, f"{endpoint}_path")
  expected_device = getattr(authority, f"{endpoint}_device")
  expected_inode = getattr(authority, f"{endpoint}_inode")
  if (
    not isinstance(path, Path)
    or signed_path != str(path)
    or isinstance(expected_device, bool)
    or not isinstance(expected_device, int)
    or expected_device < 0
    or isinstance(expected_inode, bool)
    or not isinstance(expected_inode, int)
    or expected_inode <= 0
  ):
    raise RuntimeError(
      "autonomous control append endpoint does not match signed authority"
    )
  return path, expected_device, expected_inode


def _require_control_append_endpoint(
  record: AutonomousTask,
  *,
  endpoint: str,
) -> None:
  path, expected_device, expected_inode = _control_endpoint_identity(
    record,
    endpoint=endpoint,
  )
  require_appendable_owned_file(
    path,
    expected_device=expected_device,
    expected_inode=expected_inode,
  )


def _control_record_snapshot(
  record: AutonomousTask,
  *,
  endpoint: str,
  kind: str,
  fields: frozenset[str],
) -> tuple[list[dict[str, Any]], int]:
  path, expected_device, expected_inode = _control_endpoint_identity(
    record,
    endpoint=endpoint,
  )
  records = list(iter_closed_json_records(
    path,
    expected_device=expected_device,
    expected_inode=expected_inode,
    kind=kind,
    fields=fields,
  ))
  expected_authority = {
    "task_id": record.task_id,
    "control_run_id": record.control_run_id,
    "session_id": record.session_id,
    "channel_id": record.channel_id,
  }
  for payload in records:
    if any(
      payload.get(field_name) != expected
      for field_name, expected in expected_authority.items()
    ):
      raise RuntimeError(
        "autonomous control record does not match signed run authority"
      )
  visible_stat = os.lstat(path)
  if (
    visible_stat.st_dev != expected_device
    or visible_stat.st_ino != expected_inode
  ):
    raise RuntimeError(
      "autonomous control file identity changed during inspection"
    )
  return records, visible_stat.st_size


def _append_control_record(
  record: AutonomousTask,
  *,
  endpoint: str,
  payload: dict[str, Any],
) -> None:
  path, expected_device, expected_inode = _control_endpoint_identity(
    record,
    endpoint=endpoint,
  )
  append_closed_json_record(
    path,
    expected_device=expected_device,
    expected_inode=expected_inode,
    payload=payload,
  )


def normalize_autonomous_profile(profile: str) -> str:
  return _runner_commands.normalize_autonomous_profile(
    profile,
    profile_name_re=_AUTONOMOUS_PROFILE_NAME_RE,
  )


def resolve_autonomous_owner_run_limit(
  owner_user_id: str,
  in_flight_count: int,
) -> int | None:
  """Billing attachment seam; U-D1 leaves every owner unlimited for now."""
  _ = owner_user_id, in_flight_count
  return None


def _record_owner_user_id(record: AutonomousTask) -> str:
  owner_user_id = str(record.owner_user_id or "").strip()
  if not owner_user_id:
    raise ValueError("autonomous run owner_user_id is required")
  return owner_user_id


class AutonomousRegistry(AutonomousRegistryStartMixin, AutonomousRegistryStateMixin):
  def __init__(
    self,
    *,
    api_dir: Path,
    tenant_id: str | None = None,
    python_executable: str | None = None,
    log_dir: Path | None = None,
    max_running: int = 2,
    user_event_bus: Any | None = None,
    approval_store: Any | None = None,
    service_provider_handles: Mapping[str, CredentialHandle] | None = None,
    autonomous_capability_binding_resolver: (
      AutonomousCapabilityBindingResolver | None
    ) = None,
    skill_resume_allowed_resolver: SkillResumeAllowedResolver | None = None,
    claim_signing_authority: (
      GatewayClaimSigningAuthority | None
    ) = None,
    owner_run_limit_resolver: Callable[[str, int], int | None] = (
      resolve_autonomous_owner_run_limit
    ),
  ) -> None:
    self._api_dir = Path(api_dir)
    self._tenant_id = str(tenant_id or "").strip() or None
    self._python = python_executable or sys.executable
    self._log_dir = (log_dir or Path("~/.cache/agent-gateway/autonomous").expanduser()).expanduser()
    self._max_running = max_running
    self._owner_run_limit_resolver = owner_run_limit_resolver
    self._user_event_bus = user_event_bus
    if approval_store is not None:
      required_store_methods = (
        "get",
        "get_autonomous_approval_delivery",
        "acknowledge_autonomous_approval_delivery",
      )
      if any(
        not callable(getattr(approval_store, method_name, None))
        for method_name in required_store_methods
      ):
        raise TypeError(
          "approval_store does not implement autonomous delivery authority"
        )
    self._approval_store = approval_store
    self._service_provider_handles = dict(service_provider_handles or {})
    self._autonomous_capability_binding_resolver = (
      autonomous_capability_binding_resolver
    )
    if (
      skill_resume_allowed_resolver is not None
      and not callable(skill_resume_allowed_resolver)
    ):
      raise TypeError("skill_resume_allowed_resolver must be callable or None")
    self._skill_resume_allowed_resolver = skill_resume_allowed_resolver
    if (
      claim_signing_authority is not None
      and type(claim_signing_authority)
      is not GatewayClaimSigningAuthority
    ):
      raise TypeError(
        "claim_signing_authority must be exact"
      )
    self._claim_signing_authority = claim_signing_authority
    self._tasks: dict[str, AutonomousTask] = {}
    self.run_mutation_lock = AutonomousRunMutationLock(self._log_dir)
    self._seq = self._initial_task_seq()
    self._slot_lock = asyncio.Lock()
    self._reserved_slots = 0
    self._cleanup_uncommitted_spill_starts()
    self.rehydrate()

  def set_user_event_bus(self, user_event_bus: Any | None) -> None:
    self._user_event_bus = user_event_bus

  def _next_task_id(self) -> str:
    task_id = f"bg_{self._seq}"
    self._seq += 1
    self._write_sequence_cursor()
    return task_id

  def _build_cmd(
    self,
    *,
    profile: str,
    mode: str,
    task: str | None,
    skill: str | None,
    context: str | None,
    pack: str | None = None,
    deliver: bool = True,
    ticker: str | None = None,
    max_budget_usd: float | None = None,
  ) -> list[str]:
    return _runner_commands.build_autonomous_cmd(
      python_executable=self._python,
      profile=profile,
      mode=mode,
      task=task,
      skill=skill,
      pack=pack,
      deliver=deliver,
      context=context,
      ticker=ticker,
      max_budget_usd=max_budget_usd,
      normalize_autonomous_profile_func=normalize_autonomous_profile,
    )

  def _start_payload(self, record: AutonomousTask) -> dict[str, Any]:
    return {
      "task_id": record.task_id,
      "run_id": record.control_run_id,
      "mode": record.mode,
      "pack": record.pack,
      "deliver": record.deliver,
      "log_path": str(record.log_path),
      "started_at": int(record.started_at),
      "cmd": list(record.cmd),
    }

  def _event_for_record(self, record: AutonomousTask, event: dict[str, Any]) -> dict[str, Any]:
    return _runner_events.event_for_record(record, event)

  def _replay_seed_events_for_record(self, record: AutonomousTask) -> list[dict[str, Any]]:
    return _runner_events.replay_seed_events_for_record(
      record,
      event_for_record_func=self._event_for_record,
    )

  def _record_replay_buffer_terminated(self, record: AutonomousTask) -> bool:
    return _runner_events.record_replay_buffer_terminated(
      record,
      terminal_states=AUTONOMOUS_TERMINAL_STATES,
      rehydrated_active_states=_REHYDRATED_ACTIVE_STATES,
    )

  async def _seed_replay_buffer_for_record(
    self,
    record: AutonomousTask,
    *,
    strict: bool = False,
  ) -> None:
    if self._user_event_bus is None:
      return
    seed = getattr(self._user_event_bus, "seed_replay_buffer", None)
    if not callable(seed):
      return
    if strict:
      await seed(
        _record_owner_user_id(record),
        record.control_run_id,
        self._replay_seed_events_for_record(record),
        terminated=self._record_replay_buffer_terminated(record),
      )
      return
    try:
      await seed(
        _record_owner_user_id(record),
        record.control_run_id,
        self._replay_seed_events_for_record(record),
        terminated=self._record_replay_buffer_terminated(record),
      )
    except Exception:
      pass

  def _event_duplicate_key(self, event: dict[str, Any]) -> tuple[str, str] | None:
    return _runner_events.event_duplicate_key(event)

  def _event_already_recorded(self, record: AutonomousTask, event: dict[str, Any]) -> bool:
    return _runner_events.event_already_recorded(
      record,
      event,
      event_duplicate_key_func=self._event_duplicate_key,
    )

  def _operator_inbox_record_for_message_id(
    self,
    record: AutonomousTask,
    message_id: str,
  ) -> dict[str, Any] | None:
    return _runner_events.operator_inbox_record_for_message_id(record, message_id)

  def _parent_message_event(
    self,
    record: AutonomousTask,
    *,
    message_id: str,
    text: str,
    user_id: str,
    sent_at: float,
  ) -> dict[str, Any]:
    return _runner_events.parent_message_event(
      record,
      message_id=message_id,
      text=text,
      user_id=user_id,
      sent_at=sent_at,
      operator_inbox_record_for_message_id_func=self._operator_inbox_record_for_message_id,
      event_for_record_func=self._event_for_record,
    )

  async def _persist_and_publish_parent_message_event(
    self,
    record: AutonomousTask,
    *,
    message_id: str,
    text: str,
    user_id: str,
    sent_at: float,
  ) -> None:
    event = self._parent_message_event(
      record,
      message_id=message_id,
      text=text,
      user_id=user_id,
      sent_at=sent_at,
    )
    await self._record_and_publish_event(record, event)

  async def _record_and_publish_event(
    self,
    record: AutonomousTask,
    event: dict[str, Any],
    *,
    strict: bool = False,
  ) -> dict[str, Any] | None:
    async with record.event_record_lock:
      event_copy = self._event_for_record(record, event)
      if self._event_already_recorded(record, event_copy):
        return None
      try:
        await self._append_event_evidence(record, event_copy)
      except asyncio.CancelledError:
        raise
      except BaseException as exc:
        if not self._handle_event_append_failure(
          record,
          exc,
          strict=strict,
        ):
          return None
        raise
      if record.event_lines is None:
        record.event_lines = []
      if self._user_event_bus is None:
        record.event_lines.append(event_copy)
        return event_copy
      await self._seed_replay_buffer_for_record(record, strict=strict)
      if strict:
        await self._user_event_bus.publish(
          user_id=_record_owner_user_id(record),
          control_run_id=record.control_run_id,
          event=event_copy,
        )
        record.event_lines.append(event_copy)
        return event_copy
      record.event_lines.append(event_copy)
      try:
        await self._user_event_bus.publish(
          user_id=_record_owner_user_id(record),
          control_run_id=record.control_run_id,
          event=event_copy,
        )
      except Exception:
        pass
      return event_copy

  @staticmethod
  def _append_event_evidence_sync(
    record: AutonomousTask,
    event: dict[str, Any],
  ) -> None:
    if record.events_path is None:
      raise RuntimeError("autonomous events path is unavailable")
    if record.events_device is None or record.events_inode is None:
      file_stat, created = adopt_open_json_record_file(record.events_path)
      record.events_device = file_stat.st_dev
      record.events_inode = file_stat.st_ino
      if created:
        fsync_owned_file_directory(record.events_path)
    append_open_json_record(
      record.events_path,
      expected_device=record.events_device,
      expected_inode=record.events_inode,
      payload=event,
    )

  async def _append_event_evidence(
    self,
    record: AutonomousTask,
    event: dict[str, Any],
  ) -> None:
    worker = asyncio.create_task(
      asyncio.to_thread(self._append_event_evidence_sync, record, event)
    )
    cancelled = False
    while True:
      try:
        await asyncio.shield(worker)
        break
      except asyncio.CancelledError:
        if worker.cancelled():
          raise
        if worker.done():
          break
        cancelled = True
    if cancelled:
      raise asyncio.CancelledError

  def _handle_event_append_failure(
    self,
    record: AutonomousTask,
    error: BaseException,
    *,
    strict: bool,
  ) -> bool:
    live = record.proc is not None and record.proc.returncode is None
    if not live:
      record.events_evidence_status = "unreadable"
      _LOGGER.critical(
        "Autonomous event evidence append degraded for process-less run %s",
        record.task_id,
        exc_info=(type(error), error, error.__traceback__),
      )
      return False
    already_fenced = record.cancellation_requested and (
      isinstance(record.error, str)
      and record.error.startswith("Autonomous event evidence append failure")
    )
    record.cancellation_requested = True
    record.error = f"Autonomous event evidence append failure: {error}"
    self._close_owner_lifeline(record)
    if not already_fenced:
      try:
        self._signal_owned_process_group(record, signal.SIGTERM)
      except RuntimeError as signal_error:
        record.error = (
          f"{record.error}; process-group fence failed: {signal_error}"
        )
    return strict or not already_fenced

  async def _publish_run_state(self, record: AutonomousTask, state: str) -> None:
    await self._record_and_publish_event(
      record,
      {
        "type": "run_state_changed",
        "run_id": record.control_run_id,
        "control_run_id": record.control_run_id,
        "state": state,
        "ts": int(time.time()),
      },
    )

  async def _cleanup_run_buffer(self, record: AutonomousTask) -> None:
    if self._user_event_bus is None:
      return
    try:
      await self._user_event_bus.cleanup_run(record.user_id, record.control_run_id)
    except Exception:
      pass

  def _terminal_state_for_record(self, record: AutonomousTask) -> str:
    if record.state == "killed":
      return "cancelled"
    if record.state == "interrupted":
      return "interrupted"
    if record.state in {"budget_limited", "budget_exceeded"}:
      return "budget_limited"
    if record.state == "blocked":
      return "blocked"
    if record.state in {"completed", "finished"}:
      return "completed"
    if record.state == "failed":
      return "failed"
    return "running"

  def _is_active_process_state(self, record: AutonomousTask) -> bool:
    return record.state in _ACTIVE_AUTONOMOUS_PROCESS_STATES

  def _has_terminal_run_state(self, record: AutonomousTask, state: str) -> bool:
    for event in record.event_lines or ():
      if (
        is_root_run_event(event)
        and event.get("type") == "run_state_changed"
        and event.get("state") == state
      ):
        return True
    return False

  def _status_payload(self, record: AutonomousTask) -> dict[str, Any]:
    return _runner_status.status_payload(
      record,
      tail_lines_func=_runner_status.tail_lines,
      status_tail_lines=_STATUS_TAIL_LINES,
    )

  def _get(self, task_id: str) -> AutonomousTask:
    record = self._tasks.get(task_id)
    if record is None:
      raise ValueError(f"Unknown task_id: {task_id}")
    return record

  @staticmethod
  def _close_approval_channel(record: AutonomousTask) -> None:
    endpoint = record.approval_channel
    record.approval_channel = None
    if endpoint is not None:
      endpoint.close()

  def _find_by_control_run_id(self, control_run_id: str) -> AutonomousTask | None:
    record = self._tasks.get(control_run_id)
    if record is not None:
      return record
    return next(
      (task for task in self._tasks.values() if task.control_run_id == control_run_id),
      None,
    )

  def live_process_count(self) -> int:
    return sum(
      1
      for record in self._tasks.values()
      if record.proc is not None and record.proc.returncode is None
    )

  def live_process_count_for_owner(self, owner_user_id: str) -> int:
    owner = str(owner_user_id or "").strip()
    if not owner:
      raise ValueError("owner_user_id is required for capacity accounting")
    return sum(
      1
      for record in self._tasks.values()
      if record.owner_user_id == owner
      and record.proc is not None
      and record.proc.returncode is None
    )

  def owner_capacity(self, owner_user_id: str) -> dict[str, int | str | None]:
    owner = str(owner_user_id or "").strip()
    in_flight_count = self.live_process_count_for_owner(owner)
    return {
      "owner_user_id": owner,
      "in_flight_count": in_flight_count,
      "limit": self._owner_run_limit_resolver(
        owner,
        in_flight_count,
      ),
    }

  def _append_control_record_or_fence(
    self,
    record: AutonomousTask,
    *,
    endpoint: str,
    payload: dict[str, Any],
  ) -> None:
    try:
      _append_control_record(
        record,
        endpoint=endpoint,
        payload=payload,
      )
    except AutonomousControlAppendError as exc:
      if exc.stream_recovered:
        raise
      record.cancellation_requested = True
      record.error = (
        f"Unrecoverable autonomous {endpoint} append failure"
      )
      self._close_owner_lifeline(record)
      if record.proc is not None and record.proc.returncode is None:
        try:
          self._signal_owned_process_group(
            record,
            signal.SIGTERM,
          )
        except RuntimeError as signal_error:
          record.error = (
            f"{record.error}; process-group fence failed: "
            f"{signal_error}"
          )
      raise

  async def _drain_event_channel(
    self,
    task_id: str,
  ) -> ReceivedAutonomousEventStream:
    record = self._tasks.get(task_id)
    if record is None or record.event_channel is None:
      raise RuntimeError("autonomous event channel owner is unavailable")
    endpoint = record.event_channel
    first_receive = True
    try:
      while True:
        if record.cancellation_requested:
          raise asyncio.CancelledError(
            "autonomous event channel drain cancelled by run owner"
          )
        if first_receive:
          result = await asyncio.to_thread(
            endpoint.receive_next,
            unbounded_stream=True,
          )
          first_receive = False
        else:
          result = await asyncio.to_thread(endpoint.receive_next)
        if isinstance(result, AutonomousEventRecord):
          record.event_channel_records.append(result)
          if result.event.get("type") == (
            "approval_delivery_acknowledged"
          ):
            async with record.approval_decision_lock:
              await (
                require_durable_autonomous_approval_acknowledgement(
                  store=self._approval_store,
                  record=record,
                  event=result.event,
                )
              )
          projected_event = await self._record_and_publish_event(
            record,
            result.event,
            strict=True,
          )
          if projected_event is None:
            raise RuntimeError(
              "autonomous event channel record was suppressed as a duplicate"
            )
          record.event_channel_projected_events.append(projected_event)
          continue
        if not isinstance(result, ReceivedAutonomousEventStream):
          raise RuntimeError(
            "autonomous event channel returned an unknown receive result"
          )
        if (
          len(result.records) != len(record.event_channel_records)
          or any(
            received is not delivered
            for received, delivered in zip(
              result.records,
              record.event_channel_records,
              strict=True,
            )
          )
        ):
          raise RuntimeError(
            "autonomous event channel stream identity differs from delivered records"
          )
        record.event_channel_stream = result
        if record.cancellation_requested:
          raise asyncio.CancelledError(
            "autonomous event channel acknowledgement cancelled by run owner"
          )
        record.event_channel_ack_started = True
        acknowledgement = await asyncio.to_thread(
          endpoint.acknowledge,
          result,
        )
        record.event_channel_acknowledgement = acknowledgement
        return result
    except BaseException:
      try:
        endpoint.close()
      except Exception:
        _LOGGER.exception(
          "Autonomous event channel close failed after receive failure"
        )
      try:
        self._close_approval_channel(record)
      except Exception:
        _LOGGER.exception(
          "Autonomous approval channel close failed after event failure"
        )
      if record.proc is not None:
        try:
          self._signal_owned_process_group(record, signal.SIGTERM)
        except RuntimeError as signal_error:
          record.error = (
            "autonomous event channel failed and process-group ownership "
            f"could not be confirmed: {signal_error}"
          )
      raise

  async def _read_owned_process_sentinel_status(
    self,
    record: AutonomousTask,
  ) -> int:
    if record.proc is None:
      raise RuntimeError("autonomous process sentinel is unavailable")
    stderr = getattr(record.proc, "stderr", None)
    if stderr is None:
      # Test doubles model the target process directly. Production sentinels
      # always own a private stderr status pipe.
      return await record.proc.wait()
    line = await stderr.readline()
    if not line:
      raise RuntimeError(
        "autonomous process sentinel closed before reporting child status"
      )
    if len(line) > 4096:
      raise RuntimeError("autonomous process sentinel status exceeds 4096 bytes")
    try:
      payload = json.loads(line)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise RuntimeError("autonomous process sentinel status is malformed") from exc
    if not isinstance(payload, dict) or payload.get("version") != 1:
      raise RuntimeError("autonomous process sentinel status version is invalid")
    kind = payload.get("kind")
    if kind == "error":
      if set(payload) != {"error", "kind", "version"}:
        raise RuntimeError("autonomous process sentinel error status is malformed")
      error = payload.get("error")
      if not isinstance(error, str) or not error:
        raise RuntimeError("autonomous process sentinel error is invalid")
      if error == "child_output_projection_failed" or error.startswith(
        "credential_boundary_registration_failed:"
      ):
        record_package_counter("secret_boundary_sanitization_failed")
      raise RuntimeError(f"autonomous process sentinel failed: {error}")
    if kind != "exited" or set(payload) != {
      "kind",
      "returncode",
      "version",
    }:
      raise RuntimeError("autonomous process sentinel exit status is malformed")
    returncode = payload.get("returncode")
    if (
      isinstance(returncode, bool)
      or not isinstance(returncode, int)
      or returncode < -255
      or returncode > 255
    ):
      raise RuntimeError(
        "autonomous process sentinel child returncode is invalid"
      )
    return returncode

  def _checked_manifest_write(
    self,
    record: AutonomousTask,
    *,
    phase: str,
  ) -> bool:
    try:
      return bool(
        self._write_task_manifest(
          record,
          checked=True,
        )
      )
    except Exception:
      _LOGGER.exception(
        "Autonomous task manifest writer raised during %s for %s",
        phase,
        record.task_id,
      )
      return False

  def _durably_remove_or_quarantine_terminal_manifest(
    self,
    record: AutonomousTask,
  ) -> bool:
    try:
      if self._delete_task_manifest(record.task_id):
        return True
    except Exception:
      _LOGGER.exception(
        "Autonomous task manifest deletion raised for fenced run %s",
        record.task_id,
      )
    try:
      return bool(self._quarantine_task_manifest(record.task_id))
    except Exception:
      _LOGGER.exception(
        "Autonomous task manifest quarantine raised for fenced run %s",
        record.task_id,
      )
      return False

  def _commit_terminal_manifest_or_fence(
    self,
    record: AutonomousTask,
  ) -> bool:
    record.terminal_manifest_committed = False
    intended_state = record.state
    if self._checked_manifest_write(
      record,
      phase="terminal outcome commit",
    ):
      record.terminal_manifest_committed = True
      return True

    record.state = "failed"
    record.terminal_reason = None
    record.error = (
      "autonomous terminal manifest commit failed; "
      f"run fenced from {intended_state} outcome"
    )
    record.completed_at = record.completed_at or time.time()
    if self._checked_manifest_write(
      record,
      phase="terminal persistence-failure commit",
    ):
      record.terminal_manifest_committed = True
      return True

    record.error = (
      "autonomous terminal manifest commit failed and the fenced failure "
      "outcome could not be durably persisted"
    )
    if self._durably_remove_or_quarantine_terminal_manifest(record):
      _LOGGER.critical(
        "Removed or quarantined stale active manifest for unhealthy "
        "autonomous run %s after terminal persistence failure",
        record.task_id,
      )
    else:
      record.error = (
        f"{record.error}; stale active manifest removal/quarantine also failed"
      )
      _LOGGER.critical(
        "Autonomous run %s is unhealthy: terminal outcome could not be "
        "persisted and its stale active manifest could not be fenced",
        record.task_id,
      )
    return False

  @staticmethod
  def _child_failure_disposition(
    record: AutonomousTask,
    exit_code: int | None,
  ) -> str | None:
    """The child's own failure, always led by its exit code when one exists.

    Teardown paths may have stashed gateway-side context in ``record.error``
    before the terminal disposition runs; that context is kept, but it must
    never displace the child's observed exit code.
    """
    child_failure = record.error
    if exit_code in (None, 0):
      return child_failure
    exit_text = f"Process exited with code {exit_code}"
    if child_failure is None:
      return exit_text
    if exit_text in child_failure:
      return child_failure
    return f"{exit_text}; {child_failure}"

  async def _reap(self, task_id: str) -> None:
    """Supervised reap: a child process exit must always settle the record.

    ``_reap_owned_process`` observes the owned process and commits the
    terminal outcome. If it raises anything unexpected (e.g. an OS-level
    signalling failure during cleanup), nothing else will ever flip the run
    record out of its active state, so the crash is converted into a forced
    terminal failure instead of a permanently "running" zombie.
    """
    try:
      await self._reap_owned_process(task_id)
    except asyncio.CancelledError:
      raise
    except BaseException as reaper_crash:
      _LOGGER.exception(
        "Autonomous reaper crashed for %s; forcing terminal disposition",
        task_id,
      )
      await self._force_terminal_disposition_after_reaper_crash(
        task_id,
        reaper_crash,
      )

  async def _force_terminal_disposition_after_reaper_crash(
    self,
    task_id: str,
    reaper_crash: BaseException,
  ) -> None:
    record = self._tasks.get(task_id)
    if record is None:
      return
    # Best-effort teardown of whatever the crashed reaper left open. The
    # terminal manifest commit below is the one step that must happen.
    self._close_owner_lifeline(record)
    try:
      self._close_approval_channel(record)
    except Exception:
      _LOGGER.exception(
        "Autonomous approval channel close failed after reaper crash"
      )
    if record.event_channel is not None:
      try:
        record.event_channel.close()
      except Exception:
        _LOGGER.exception(
          "Autonomous event channel close failed after reaper crash"
        )
    if record.proc is not None and record.proc.returncode is None:
      try:
        self._signal_owned_process_group(record, signal.SIGKILL)
      except Exception:
        _LOGGER.exception(
          "Autonomous process-group kill failed after reaper crash"
        )
    observer_tasks = [
      task
      for task in (record.sentinel_status_task, record.event_channel_task)
      if task is not None
    ]
    for task in observer_tasks:
      if not task.done():
        task.cancel()
    if observer_tasks:
      await asyncio.gather(*observer_tasks, return_exceptions=True)
    if record.claim_broker is not None:
      try:
        record.claim_broker.close()
      except Exception:
        _LOGGER.exception(
          "Autonomous claim broker close failed after reaper crash"
        )
      finally:
        record.claim_broker = None
    if record.exit_code is None and record.proc is not None:
      observed_returncode = record.proc.returncode
      if isinstance(observed_returncode, int):
        record.exit_code = observed_returncode
    if self._is_active_process_state(record):
      record.state = "failed"
      # The child's disposition must not be masked by the reaper's own crash.
      child_failure = self._child_failure_disposition(
        record,
        record.exit_code,
      )
      crash_failure = (
        f"reaper crashed: {type(reaper_crash).__name__}: {reaper_crash}"
      )
      if child_failure is None:
        record.error = crash_failure
      else:
        record.error = f"{child_failure}; {crash_failure}"
      record.completed_at = time.time()
    else:
      record.completed_at = record.completed_at or time.time()
    terminal_manifest_committed = self._commit_terminal_manifest_or_fence(
      record
    )
    if record.log_handle is not None:
      record.log_handle.close()
      record.log_handle = None
    await self._release_slot(record)
    if terminal_manifest_committed:
      try:
        terminal_state = self._terminal_state_for_record(record)
        if (
          terminal_state != "running"
          and not self._has_terminal_run_state(record, terminal_state)
        ):
          await self._publish_run_state(record, terminal_state)
        if terminal_state != "running":
          await self._cleanup_run_buffer(record)
      except Exception:
        _LOGGER.exception(
          "Autonomous terminal run-state publish failed after reaper crash"
        )

  async def _reap_owned_process(self, task_id: str) -> None:
    record = self._tasks.get(task_id)
    if record is None or record.proc is None:
      return
    process_error: BaseException | None = None
    channel_error: BaseException | None = None
    cleanup_error: Exception | None = None
    exit_code: int | None = None

    def close_approval_channel() -> None:
      nonlocal cleanup_error
      try:
        self._close_approval_channel(record)
      except Exception as exc:
        if cleanup_error is None:
          cleanup_error = exc

    def close_event_channel() -> None:
      nonlocal cleanup_error
      if record.event_channel is None:
        return
      try:
        record.event_channel.close()
      except Exception as exc:
        if cleanup_error is None:
          cleanup_error = exc

    def interrupt_event_channel() -> None:
      nonlocal cleanup_error
      if record.event_channel is None:
        return
      try:
        record.event_channel.interrupt()
      except Exception as exc:
        if cleanup_error is None:
          cleanup_error = exc

    process_status_task = asyncio.create_task(
      self._read_owned_process_sentinel_status(record)
    )
    record.sentinel_status_task = process_status_task
    channel_task = record.event_channel_task
    wait_tasks: set[asyncio.Task[Any]] = {process_status_task}
    if channel_task is None:
      channel_error = RuntimeError(
        "autonomous event channel drain task is unavailable"
      )
    else:
      wait_tasks.add(channel_task)

    process_settled = False
    channel_settled = channel_task is None
    group_fence_attempted = False
    group_fence_confirmed = False

    def harvest(tasks: set[asyncio.Task[Any]]) -> None:
      nonlocal channel_error, channel_settled, exit_code
      nonlocal process_error, process_settled
      for task in tasks:
        if task is process_status_task:
          process_settled = True
          try:
            exit_code = process_status_task.result()
          except BaseException as exc:
            if process_error is None:
              process_error = exc
        elif task is channel_task:
          channel_settled = True
          try:
            channel_task.result()
          except BaseException as exc:
            if channel_error is None:
              channel_error = exc

    accepted_cancellation = (
      record.cancellation_requested
      and not record.event_channel_ack_started
    )
    while not accepted_cancellation:
      pending = {
        task
        for task in wait_tasks
        if not (
          (task is process_status_task and process_settled)
          or (task is channel_task and channel_settled)
        )
      }
      if not pending:
        break
      done, _ = await asyncio.wait(
        pending,
        return_when=asyncio.FIRST_COMPLETED,
      )
      # Harvest every completion in this wake before applying precedence.
      harvest(done)
      accepted_cancellation = (
        record.cancellation_requested
        and not record.event_channel_ack_started
      )
      if accepted_cancellation:
        break
      if process_settled and process_error is None:
        group_fence_attempted = True
        try:
          group_fence_confirmed = self._signal_owned_process_group(
            record,
            signal.SIGKILL,
          )
        except RuntimeError as exc:
          if cleanup_error is None:
            cleanup_error = exc
      if channel_error is not None:
        break
      if process_settled:
        if process_error is None:
          if not channel_settled and channel_task is not None:
            try:
              await asyncio.wait_for(
                asyncio.shield(channel_task),
                timeout=_POST_EXIT_SETTLE_SECONDS,
              )
            except asyncio.TimeoutError:
              interrupt_event_channel()
              await asyncio.gather(channel_task, return_exceptions=True)
            except BaseException:
              # The result is harvested below so channel errors retain their
              # prescribed precedence over a valid process settlement.
              pass
            if channel_task.done():
              harvest({channel_task})
        break

    if (
      accepted_cancellation
      or process_error is not None
      or channel_error is not None
    ):
      close_approval_channel()
      interrupt_event_channel()
      if record.proc.returncode is None:
        try:
          self._signal_owned_process_group(record, signal.SIGTERM)
        except RuntimeError as exc:
          if cleanup_error is None:
            cleanup_error = exc
      pending = {task for task in wait_tasks if not task.done()}
      if pending:
        done, _ = await asyncio.wait(
          pending,
          timeout=_SPAWN_CLEANUP_GRACE_SEC,
        )
        harvest(done)

    sentinel_stderr = getattr(record.proc, "stderr", None)
    self._close_owner_lifeline(record)
    sentinel_was_live_for_cleanup = record.proc.returncode is None
    if not group_fence_attempted and record.proc.returncode is None:
      try:
        group_fence_confirmed = self._signal_owned_process_group(
          record,
          signal.SIGKILL,
        )
      except RuntimeError as exc:
        if cleanup_error is None:
          cleanup_error = exc
    elif (
      not group_fence_attempted
      and sentinel_stderr is not None
      and process_error is None
      and not accepted_cancellation
    ):
      process_error = RuntimeError(
        "autonomous process sentinel exited before final group cleanup"
      )

    try:
      await asyncio.wait_for(
        record.proc.wait(),
        timeout=_SPAWN_CLEANUP_GRACE_SEC,
      )
    except asyncio.TimeoutError:
      if cleanup_error is None:
        cleanup_error = RuntimeError(
          "autonomous process sentinel did not exit after final group cleanup"
        )
    except Exception as exc:
      if cleanup_error is None:
        cleanup_error = exc

    if (
      sentinel_was_live_for_cleanup
      and not group_fence_confirmed
      and record.proc.returncode != -signal.SIGKILL
      and cleanup_error is None
    ):
      cleanup_error = RuntimeError(
        "autonomous process sentinel was not live for final group cleanup"
      )

    close_event_channel()
    close_approval_channel()
    for task in wait_tasks:
      if not task.done():
        task.cancel()
    await asyncio.gather(*wait_tasks, return_exceptions=True)
    harvest({task for task in wait_tasks if task.done()})

    if record.claim_broker is not None:
      try:
        record.claim_broker.close()
      except Exception as exc:
        if cleanup_error is None:
          cleanup_error = exc
      finally:
        record.claim_broker = None

    record.exit_code = exit_code
    if self._is_active_process_state(record):
      if (
        record.cancellation_requested
        and cleanup_error is None
      ):
        # Cancellation owns the terminal disposition once accepted while the
        # run is active. Tearing down the owned process group intentionally
        # aborts both the sentinel status pipe and unacknowledged event stream;
        # those protocol errors are expected. Cleanup failures remain fatal so
        # "killed" is never reported while an owned process may still be live.
        record.state = "killed"
        record.error = record.error or "Process terminated by user"
      elif cleanup_error is not None:
        record.state = "failed"
        # A cleanup failure must not mask why the child itself failed; the
        # child's disposition is the diagnostic that matters to the caller.
        child_failure = self._child_failure_disposition(record, exit_code)
        if child_failure is None:
          record.error = f"reaper failed: {cleanup_error}"
        else:
          record.error = (
            f"{child_failure}; reaper also failed: {cleanup_error}"
          )
      elif channel_error is not None:
        record.state = "failed"
        # The event channel tearing down is expected collateral when the
        # child process dies before sending any events; the child's own
        # disposition is the diagnostic that matters and must not be masked.
        child_failure = self._child_failure_disposition(record, exit_code)
        channel_failure = (
          "autonomous event channel failed: "
          f"{type(channel_error).__name__}: {channel_error}"
        )
        if child_failure is None:
          record.error = channel_failure
        else:
          record.error = f"{child_failure}; {channel_failure}"
      elif process_error is not None:
        record.state = "failed"
        child_failure = self._child_failure_disposition(record, exit_code)
        if child_failure is None:
          record.error = f"reaper failed: {process_error}"
        else:
          record.error = (
            f"{child_failure}; reaper also failed: {process_error}"
          )
      elif exit_code == 0:
        record.state = "completed"
      else:
        record.state = "failed"
        record.error = record.error or f"Process exited with code {exit_code}"
      record.completed_at = time.time()
    else:
      record.completed_at = record.completed_at or time.time()

    approval_delivery_quarantined = (
      record.approval_delivery_quarantined
      or (
        record.state == "failed"
        and isinstance(record.error, str)
        and record.error.startswith(
          "approval_delivery_quarantined:"
        )
      )
    )
    if (
      process_error is None
      and channel_error is None
      and cleanup_error is None
      and not approval_delivery_quarantined
    ):
      self._apply_terminal_event_state(record)
    terminal_manifest_committed = self._commit_terminal_manifest_or_fence(
      record
    )

    if record.log_handle is not None:
      record.log_handle.close()
      record.log_handle = None

    await self._release_slot(record)
    close_event_channel()
    close_approval_channel()
    if terminal_manifest_committed:
      terminal_state = self._terminal_state_for_record(record)
      if (
        terminal_state != "running"
        and not self._has_terminal_run_state(record, terminal_state)
      ):
        await self._publish_run_state(record, terminal_state)
      if terminal_state != "running":
        await self._cleanup_run_buffer(record)

  def status(self, task_id: str) -> dict[str, Any]:
    return self._status_payload(self._get(task_id))

  async def wait(self, task_id: str, *, timeout_sec: int = 600) -> dict[str, Any]:
    record = self._get(task_id)
    if self._is_active_process_state(record) and record.reaper_task is not None:
      try:
        await asyncio.wait_for(asyncio.shield(record.reaper_task), timeout=float(timeout_sec))
      except asyncio.TimeoutError:
        pass
    return self._status_payload(record)

  def logs(self, task_id: str, *, tail: int = 200) -> dict[str, Any]:
    record = self._get(task_id)
    lines, total_lines = _runner_status.tail_lines(record.log_path, int(tail))
    return {
      "task_id": record.task_id,
      "log_path": str(record.log_path),
      "lines": lines,
      "total_lines": total_lines,
    }

  async def send_operator_message(
    self,
    control_run_id: str,
    *,
    user_id: str,
    channel: str | None = None,
    message: str,
    message_id: str | None = None,
  ) -> dict[str, Any]:
    record = self._find_by_control_run_id(control_run_id)
    if record is None:
      raise ValueError(f"Unknown control_run_id: {control_run_id}")
    if _record_owner_user_id(record) != user_id:
      raise PermissionError("Run not found")

    normalized_channel = channel.strip().lower() if isinstance(channel, str) and channel.strip() else None
    if record.channel is not None and normalized_channel != record.channel:
      raise PermissionError("Run not found")

    if record.state not in {"running", "waiting", "approval_pending"} or (
      record.proc is not None and record.proc.returncode is not None
    ):
      raise RuntimeError("Autonomous run is not accepting messages")
    if record.event_lines is not None and any(
      is_root_terminal_event(event)
      and event.get("type") == "stream_complete"
      for event in record.event_lines
    ):
      raise RuntimeError("Autonomous run is no longer accepting messages")

    text = message.strip() if isinstance(message, str) else ""
    if not text:
      raise ValueError("message is required")

    if record.operator_inbox_path is None:
      raise RuntimeError("Autonomous operator inbox unavailable")

    async with record.operator_message_lock:
      resolved_message_id = message_id.strip() if isinstance(message_id, str) and message_id.strip() else None
      resolved_message_id = resolved_message_id or f"op_{secrets.token_hex(8)}"
      inbox_records, inbox_bytes = _control_record_snapshot(
        record,
        endpoint="operator_inbox",
        kind="operator_message",
        fields=AUTONOMOUS_OPERATOR_RECORD_FIELDS,
      )
      records_by_message_id: dict[str, dict[str, Any]] = {}
      for inbox_record in inbox_records:
        existing_message_id = _required_control_text(
          inbox_record.get("message_id"),
          field_name="message_id",
        )
        _required_control_text(
          inbox_record.get("text"),
          field_name="text",
          max_length=64 * 1024,
        )
        existing_sent_at_ns = inbox_record.get("sent_at_ns")
        if (
          isinstance(existing_sent_at_ns, bool)
          or not isinstance(existing_sent_at_ns, int)
          or existing_sent_at_ns <= 0
        ):
          raise RuntimeError(
            "autonomous operator record sent_at_ns is invalid"
          )
        prior = records_by_message_id.get(existing_message_id)
        if prior is not None and prior != inbox_record:
          raise RuntimeError(
            "autonomous operator message id was reused with different content"
          )
        records_by_message_id[existing_message_id] = inbox_record

      existing_inbox_record = records_by_message_id.get(
        resolved_message_id
      )
      if existing_inbox_record is not None:
        if existing_inbox_record["text"] != text:
          raise RuntimeError(
            "autonomous operator message id was reused with different content"
          )
        existing_sent_at_ns = int(existing_inbox_record["sent_at_ns"])
        await self._persist_and_publish_parent_message_event(
          record,
          message_id=resolved_message_id,
          text=text,
          user_id=user_id,
          sent_at=existing_sent_at_ns / 1_000_000_000,
        )
        record.delivered_messages.add(resolved_message_id)
        return {
          "task_id": record.task_id,
          "run_id": record.control_run_id,
          "message_id": resolved_message_id,
          "delivery_status": "duplicate",
        }
      if resolved_message_id in record.delivered_messages:
        raise RuntimeError(
          "autonomous operator inbox lost a delivered message"
        )
      if (
        len(records_by_message_id)
        >= AUTONOMOUS_OPERATOR_MESSAGE_LIMIT
      ):
        raise RuntimeError(
          "autonomous operator inbox exceeds its message quota"
        )

      sent_at_ns = time.time_ns()
      inbox_record = {
        "version": 1,
        "kind": "operator_message",
        "task_id": record.task_id,
        "control_run_id": record.control_run_id,
        "session_id": record.session_id,
        "channel_id": record.channel_id,
        "message_id": resolved_message_id,
        "text": text,
        "sent_at_ns": sent_at_ns,
      }
      encoded_inbox_record = encode_closed_control_record(inbox_record)
      if (
        inbox_bytes + len(encoded_inbox_record)
        > AUTONOMOUS_OPERATOR_AGGREGATE_BYTES_LIMIT
      ):
        raise RuntimeError(
          "autonomous operator inbox exceeds its aggregate byte quota"
        )
      self._append_control_record_or_fence(
        record,
        endpoint="operator_inbox",
        payload=inbox_record,
      )

      await self._persist_and_publish_parent_message_event(
        record,
        message_id=resolved_message_id,
        text=text,
        user_id=user_id,
        sent_at=sent_at_ns / 1_000_000_000,
      )
      record.delivered_messages.add(resolved_message_id)
      return {
        "task_id": record.task_id,
        "run_id": record.control_run_id,
        "message_id": resolved_message_id,
        "delivery_status": "delivered",
      }

  async def send_approval_decision(
    self,
    control_run_id: str,
    *,
    user_id: str,
    channel: str | None = None,
    approval_id: str,
    tool_call_id: str,
    nonce: str,
    approved: bool,
    decided_at_ns: int,
    delivery_sequence: int,
    publication_transaction: Callable[[], ContextManager[None]],
    sent_reconciliation: Callable[[], ContextManager[None]],
  ) -> dict[str, Any]:
    record = self._find_by_control_run_id(control_run_id)
    if record is None:
      raise ValueError(f"Unknown control_run_id: {control_run_id}")
    if _record_owner_user_id(record) != user_id:
      raise PermissionError("Run not found")

    normalized_channel = channel.strip().lower() if isinstance(channel, str) and channel.strip() else None
    if record.channel is not None and normalized_channel != record.channel:
      raise PermissionError("Run not found")

    if record.state not in _APPROVAL_DECISION_AUTONOMOUS_STATES or (
      record.proc is not None and record.proc.returncode is not None
    ):
      raise RuntimeError("Autonomous run is not running")
    endpoint = getattr(record, "approval_channel", None)
    launch_nonce = getattr(record, "launch_nonce", None)
    if (
      type(endpoint) is not AutonomousApprovalChannelParent
      or type(launch_nonce) is not str
    ):
      raise RuntimeError(
        "Autonomous approval channel unavailable"
      )

    normalized_approval_id = _required_control_text(
      approval_id,
      field_name="approval_id",
    )
    normalized_tool_call_id = _required_control_text(
      tool_call_id,
      field_name="tool_call_id",
    )
    normalized_nonce = _required_control_text(
      nonce,
      field_name="nonce",
    )
    if not isinstance(approved, bool):
      raise ValueError("approved must be a bool")
    if (
      isinstance(decided_at_ns, bool)
      or not isinstance(decided_at_ns, int)
      or decided_at_ns <= 0
    ):
      raise ValueError("decided_at_ns must be a positive integer")
    if type(delivery_sequence) is not int or delivery_sequence < 1:
      raise ValueError(
        "delivery_sequence must be a positive integer"
      )
    if not callable(publication_transaction):
      raise TypeError("publication_transaction is required")
    if not callable(sent_reconciliation):
      raise TypeError("sent_reconciliation is required")
    decision = AutonomousApprovalDecision(
      authority=AutonomousApprovalChannelAuthority(
        launch_nonce=launch_nonce,
        task_id=record.task_id,
        control_run_id=record.control_run_id,
        session_id=record.session_id,
        channel_id=record.channel_id,
      ),
      delivery_sequence=delivery_sequence,
      approval_id=normalized_approval_id,
      tool_call_id=normalized_tool_call_id,
      nonce=normalized_nonce,
      approved=approved,
      decided_at_ns=decided_at_ns,
    )
    async with record.approval_decision_lock:
      try:
        endpoint.require_sent(decision)
      except AutonomousApprovalChannelProtocolError:
        already_sent = False
      else:
        already_sent = True

      def publish() -> str:
        try:
          with publication_transaction():
            if record.state not in (
              _APPROVAL_DECISION_AUTONOMOUS_STATES
            ) or (
              record.proc is not None
              and record.proc.returncode is not None
            ):
              raise RuntimeError(
                "Autonomous run is not running"
              )
            endpoint.send(decision)
        except BaseException as publication_error:
          try:
            endpoint.require_sent(decision)
          except AutonomousApprovalChannelProtocolError:
            raise publication_error
          with sent_reconciliation():
            pass
          return "reconciled"
        return "duplicate" if already_sent else "delivered"

      publication_task = asyncio.create_task(
        asyncio.to_thread(publish),
        name=(
          "autonomous-approval-publication:"
          f"{record.task_id}:{delivery_sequence}"
        ),
      )
      try:
        delivery_status = await asyncio.shield(publication_task)
      except asyncio.CancelledError:
        await publication_task
        raise

    await self._record_and_publish_event(
      record,
      {
        "type": "approval_decision_sent",
        "task_id": record.task_id,
        "run_id": record.control_run_id,
        "control_run_id": record.control_run_id,
        "approval_id": normalized_approval_id,
        "tool_call_id": normalized_tool_call_id,
        "approved": approved,
        "allow_tool_type": False,
        "decider": {
          "user_id": user_id,
        },
        "sent_at": decided_at_ns / 1_000_000_000,
      },
    )
    return {
      "task_id": record.task_id,
      "run_id": record.control_run_id,
      "approval_id": normalized_approval_id,
      "tool_call_id": normalized_tool_call_id,
      "delivery_status": delivery_status,
    }

  async def fail_autonomous_approval_delivery(
    self,
    task_id: str,
    *,
    error: str,
  ) -> bool:
    """Irrevocably fail one active run after approval delivery quarantine."""

    if (
      type(task_id) is not str
      or _AUTONOMOUS_TASK_ID_RE.fullmatch(task_id) is None
    ):
      return False
    record = self._tasks.get(task_id)
    if record is None or not self._is_active_process_state(record):
      return False
    normalized_error = _required_control_text(
      error,
      field_name="approval delivery error",
    )
    record.cancellation_requested = True
    record.approval_delivery_quarantined = True
    record.state = "failed"
    record.terminal_reason = None
    record.error = (
      "approval_delivery_quarantined: "
      f"{normalized_error}"
    )
    record.completed_at = record.completed_at or time.time()
    committed = self._commit_terminal_manifest_or_fence(record)
    self._close_owner_lifeline(record)
    try:
      self._close_approval_channel(record)
    except Exception as exc:
      record.error = (
        f"{record.error}; approval channel close failed: {exc}"
      )
      committed = (
        self._commit_terminal_manifest_or_fence(record)
        and committed
      )
    if record.proc is not None and record.proc.returncode is None:
      try:
        self._signal_owned_process_group(record, signal.SIGTERM)
      except RuntimeError as exc:
        record.error = (
          "approval_delivery_quarantined: "
          f"{normalized_error}; process termination failed: {exc}"
        )
        committed = (
          self._commit_terminal_manifest_or_fence(record)
          and committed
        )
    if committed:
      await self._publish_run_state(record, "failed")
    return True

  async def cancel(self, task_id: str) -> dict[str, Any]:
    record = self._get(task_id)
    if self._is_active_process_state(record):
      if record.event_channel_ack_started:
        if record.reaper_task is None:
          raise RuntimeError(
            "acknowledged autonomous run has no direct-process reaper"
          )
        await asyncio.shield(record.reaper_task)
        return self._status_payload(record)
      if not record.event_channel_ack_started:
        record.cancellation_requested = True
        record.error = record.error or "Process terminated by user"
        self._checked_manifest_write(
          record,
          phase="cancellation request",
        )
        self._close_owner_lifeline(record)
        try:
          self._close_approval_channel(record)
        except Exception as exc:
          record.error = (
            f"{record.error}; approval channel close failed: {exc}"
          )
        if record.event_channel is not None:
          record.event_channel.interrupt()
        if record.proc is not None and record.proc.returncode is None:
          try:
            self._signal_owned_process_group(record, signal.SIGTERM)
          except RuntimeError as exc:
            # A signalling failure (e.g. Darwin EPERM against a mid-exit
            # group, or the child already gone) must not fail the cancel
            # request; the reaper settles the terminal disposition below.
            record.error = (
              f"{record.error}; process-group terminate failed: {exc}"
            )
      if record.reaper_task is None:
        raise RuntimeError("active autonomous run has no direct-process reaper")
      try:
        await asyncio.wait_for(
          asyncio.shield(record.reaper_task),
          timeout=_SPAWN_CLEANUP_GRACE_SEC,
        )
      except asyncio.TimeoutError:
        try:
          self._signal_owned_process_group(record, signal.SIGKILL)
        except RuntimeError as exc:
          record.error = (
            f"{record.error}; process-group kill failed: {exc}"
          )
        try:
          await asyncio.wait_for(
            asyncio.shield(record.reaper_task),
            timeout=_SPAWN_CLEANUP_GRACE_SEC,
          )
        except asyncio.TimeoutError as exc:
          raise RuntimeError(
            "cancelled autonomous run did not settle after final process-group kill"
          ) from exc
    return self._status_payload(record)

  async def shutdown(self, *, grace_sec: float = 10.0) -> None:
    live_records = [
      record
      for record in self._tasks.values()
      if record.proc is not None and record.proc.returncode is None
    ]

    for record in live_records:
      if self._is_active_process_state(record):
        if record.event_channel_ack_started:
          continue
        record.cancellation_requested = True
        record.error = record.error or "Process terminated during gateway shutdown"
        self._checked_manifest_write(
          record,
          phase="gateway shutdown request",
        )
      self._close_owner_lifeline(record)
      try:
        self._close_approval_channel(record)
      except Exception:
        _LOGGER.exception(
          "Autonomous approval channel close failed during shutdown"
        )
      try:
        self._signal_owned_process_group(record, signal.SIGTERM)
      except RuntimeError as exc:
        record.state = "failed"
        record.error = f"process-group ownership failure during shutdown: {exc}"
        self._checked_manifest_write(
          record,
          phase="gateway shutdown ownership failure",
        )

    waiters = [record.reaper_task for record in live_records if record.reaper_task is not None]
    if waiters:
      done, pending = await asyncio.wait(waiters, timeout=grace_sec)
      if pending:
        for record in live_records:
          if record.proc is not None and record.proc.returncode is None:
            try:
              self._signal_owned_process_group(record, signal.SIGKILL)
            except RuntimeError:
              pass
        await asyncio.gather(*pending, return_exceptions=True)
      else:
        await asyncio.gather(*done, return_exceptions=True)

    for record in self._tasks.values():
      if record.log_handle is not None:
        record.log_handle.close()
        record.log_handle = None
      if record.event_channel is not None:
        try:
          record.event_channel.close()
        except Exception:
          _LOGGER.exception(
            "Autonomous event channel final close failed"
          )
      try:
        self._close_approval_channel(record)
      except Exception:
        _LOGGER.exception(
          "Autonomous approval channel final close failed"
        )
      self._close_owner_lifeline(record)


__all__ = ["AutonomousRegistry", "AutonomousTask", "normalize_autonomous_profile"]
