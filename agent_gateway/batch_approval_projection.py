from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal


def normalize_approval_channel(value: str | None) -> str | None:
  normalized = str(value or "").strip().lower()
  return normalized or None


@dataclass(frozen=True)
class ApprovalProjection:
  approval_id: str
  batch_id: int
  run_id: str
  stage_run_seq: int | None
  owner_user_id: str
  channel: str | None
  tool_call_id: str
  nonce: str
  session_id: str
  lifecycle: Literal["live"]
  session: Any
  store: Any | None
  policy: Any | None


@dataclass(frozen=True)
class BoundApprovalAuthorizationSubject:
  """A request bound to a pre-fence admission but not published live yet."""

  approval_id: str
  batch_id: int
  run_id: str
  owner_user_id: str
  channel: str | None
  tool_call_id: str
  session_id: str
  request: Any
  session: Any
  store: Any
  policy: Any | None


@dataclass(frozen=True)
class _BatchApprovalCarrier:
  batch_id: int
  run_id: str
  owner_user_id: str
  channel: str | None
  session: Any
  store: Any | None
  policy: Any | None


@dataclass
class _BatchAdmissionGate:
  closed: bool = False
  active: int = 0
  close_when_drained: bool = False
  drained: asyncio.Event = field(default_factory=asyncio.Event)
  admissions: dict[int, "BatchApprovalAdmission"] = field(default_factory=dict)

  def __post_init__(self) -> None:
    self.drained.set()


@dataclass
class BatchApprovalAdmission:
  """One approval producer admitted before its first durable mutation."""

  registry: "BatchApprovalProjectionRegistry"
  key: tuple[str, int]
  carrier: _BatchApprovalCarrier
  released: bool = False
  published: bool = False
  request: Any | None = None
  store: Any | None = None

  def bind_request(self, *, request: Any, store: Any) -> None:
    if self.released:
      raise RuntimeError("batch approval admission is already released")
    expected_session_id = str(
      getattr(self.carrier.session, "session_id", "") or ""
    )
    if not all((
      str(getattr(request, "user_id", "") or "") == self.carrier.owner_user_id,
      str(getattr(request, "request_id", "") or "") == self.carrier.run_id,
      str(getattr(request, "run_id", "") or "") == self.carrier.run_id,
      str(getattr(request, "session_id", "") or "") == expected_session_id,
      normalize_approval_channel(getattr(request, "channel", None))
      == self.carrier.channel,
    )):
      raise ValueError("batch approval durable identity mismatch")
    if not callable(getattr(store, "abort_unpublished_approval", None)):
      raise RuntimeError("batch approval store cannot abort unpublished approvals")
    if not callable(
      getattr(store, "fence_persistent_grants_for_cancellation", None)
    ) or not callable(getattr(store, "revoke_persistent_grants_for_approval", None)):
      raise RuntimeError(
        "batch approval store cannot quarantine persistent approval grants"
      )
    self.request = request
    self.store = store

  def publish_pending(self) -> None:
    if self.released:
      raise RuntimeError("batch approval admission is already released")
    try:
      self.registry._publish_admitted_carrier(self)
      self.published = True
    finally:
      self.release()

  async def abort_unpublished(self) -> None:
    if self.published or self.request is None or self.store is None:
      return
    approval_id = str(getattr(self.request, "approval_id", "") or "").strip()
    tool_call_id = str(getattr(self.request, "tool_call_id", "") or "").strip()
    abort = getattr(self.store, "abort_unpublished_approval", None)
    if not callable(abort) or not approval_id or not tool_call_id:
      raise RuntimeError("batch approval admission cannot be aborted safely")
    for attempt in range(2):
      try:
        result = await abort(
          approval_id,
          expected_tool_call_id=tool_call_id,
          expected_user_id=self.carrier.owner_user_id,
          expected_request_id=self.carrier.run_id,
          expected_run_id=self.carrier.run_id,
          expected_session_id=str(
            getattr(self.carrier.session, "session_id", "") or ""
          ),
          expected_channel=self.carrier.channel,
          decision_reason="Batch approval admission aborted before publication",
        )
        if (
          not isinstance(result, tuple)
          or len(result) != 3
          or result[2] is not True
        ):
          raise RuntimeError("batch approval abort durable identity mismatch")
        break
      except KeyError:
        break
      except asyncio.CancelledError:
        if attempt == 1:
          raise
    pending = getattr(self.carrier.session, "pending_tools", {}).get(tool_call_id)
    if isinstance(pending, dict) and str(pending.get("approval_id") or "") == approval_id:
      self.carrier.session.pending_tools.pop(tool_call_id, None)
      self.carrier.session.approval_queues.pop(tool_call_id, None)

  def release(self) -> None:
    if self.released:
      return
    self.released = True
    self.registry._release_admission(self)


class BatchApprovalProjectionRegistry:
  """Process-local routing references for durable batch approvals."""

  def __init__(self) -> None:
    self._carriers: dict[tuple[str, int, int], _BatchApprovalCarrier] = {}
    self._projections: dict[str, ApprovalProjection] = {}
    self._batch_admission_fences: set[tuple[str, int]] = set()
    self._batch_fence_projection_snapshots: dict[
      tuple[str, int],
      dict[str, ApprovalProjection],
    ] = {}
    self._batch_fence_projection_history: dict[
      tuple[str, int],
      dict[str, ApprovalProjection],
    ] = {}
    self._admission_gates: dict[tuple[str, int], _BatchAdmissionGate] = {}
    self._shutdown_started = False

  def register_session(
    self,
    *,
    batch_id: int,
    owner_user_id: str,
    channel: str | None,
    session: Any,
    store: Any | None = None,
    policy: Any | None = None,
  ) -> _BatchApprovalCarrier:
    normalized_batch_id = int(batch_id)
    if normalized_batch_id < 1:
      raise ValueError("batch_id must be a positive integer")
    normalized_owner = str(owner_user_id or "").strip()
    if not normalized_owner:
      raise ValueError("batch approval owner_user_id is required")
    batch_key = (normalized_owner, normalized_batch_id)
    if self._shutdown_started:
      raise RuntimeError("batch approval projection registry is shutting down")
    if batch_key in self._batch_admission_fences:
      raise RuntimeError("batch approval projection admission is fenced")
    session_user_id = str(getattr(session, "user_id", "") or "").strip()
    if session_user_id != normalized_owner:
      raise ValueError("batch approval session owner mismatch")
    normalized_channel = normalize_approval_channel(channel)
    session_channel = normalize_approval_channel(getattr(session, "channel", None))
    if session_channel != normalized_channel:
      raise ValueError("batch approval session channel mismatch")
    carrier = _BatchApprovalCarrier(
      batch_id=normalized_batch_id,
      run_id=f"batch_{normalized_batch_id}",
      owner_user_id=normalized_owner,
      channel=normalized_channel,
      session=session,
      store=store if store is not None else getattr(session, "approval_store", None),
      policy=policy if policy is not None else getattr(session, "approval_policy", None),
    )
    key = (normalized_owner, normalized_batch_id, id(session))
    existing = self._carriers.get(key)
    if existing is not None and (
      existing != carrier
      or existing.store is not carrier.store
      or existing.policy is not carrier.policy
    ):
      raise ValueError("conflicting batch approval session registration")
    prior_projections = dict(self._projections)
    try:
      self._sync_carrier(carrier)
    except BaseException:
      self._projections = prior_projections
      raise
    self._carriers[key] = carrier
    self._admission_gates.setdefault(batch_key, _BatchAdmissionGate())
    return carrier

  def acquire_admission(
    self,
    *,
    batch_id: int,
    owner_user_id: str,
    session: Any,
  ) -> BatchApprovalAdmission:
    key = (str(owner_user_id or "").strip(), int(batch_id))
    carrier_key = (key[0], key[1], id(session))
    carrier = self._carriers.get(carrier_key)
    if carrier is None or carrier.session is not session:
      raise RuntimeError("batch approval session is not registered")
    gate = self._admission_gates.setdefault(key, _BatchAdmissionGate())
    if self._shutdown_started:
      raise RuntimeError("batch approval projection registry is shutting down")
    if gate.closed or key in self._batch_admission_fences:
      raise RuntimeError("batch approval projection admission is fenced")
    admission = BatchApprovalAdmission(
      registry=self,
      key=key,
      carrier=carrier,
    )
    gate.active += 1
    gate.admissions[id(admission)] = admission
    gate.drained.clear()
    return admission

  async def freeze_batch(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> tuple[ApprovalProjection, ...]:
    """Fence one batch and return its complete immutable live projection set."""
    initial = self.close_batch_admission(
      owner_user_id=owner_user_id,
      batch_id=batch_id,
    )
    try:
      drained = await self.snapshot_after_batch_drain(
        owner_user_id=owner_user_id,
        batch_id=batch_id,
      )
    except BaseException:
      if not self._shutdown_started:
        self.release_batch_fence(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
          rollback=True,
        )
      raise
    return self.merge_projection_sets(initial, drained)

  def close_batch_admission(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> tuple[ApprovalProjection, ...]:
    """Fence one batch without waiting for already-admitted producers."""
    normalized_owner = str(owner_user_id or "").strip()
    normalized_batch_id = int(batch_id)
    key = (normalized_owner, normalized_batch_id)
    if self._shutdown_started:
      raise RuntimeError("batch approval projection registry is shutting down")
    if key in self._batch_admission_fences:
      raise RuntimeError("batch approval projection admission is already fenced")
    gate = self._admission_gates.setdefault(key, _BatchAdmissionGate())
    prior_projections = {
      approval_id: projection
      for approval_id, projection in self._projections.items()
      if projection.owner_user_id == normalized_owner
      and projection.batch_id == normalized_batch_id
    }
    gate.closed = True
    self._batch_admission_fences.add(key)
    try:
      for carrier in tuple(self._carriers.values()):
        if carrier.owner_user_id == normalized_owner and carrier.batch_id == normalized_batch_id:
          self._sync_carrier(carrier, force=True)
    except BaseException:
      if not self._shutdown_started:
        self._restore_batch_projections(key, prior_projections)
        self._batch_admission_fences.discard(key)
        gate.closed = False
      raise
    self._batch_fence_projection_snapshots[key] = prior_projections
    self._batch_fence_projection_history[key] = {}
    return self._projection_snapshot_for_batch(key)

  async def snapshot_after_batch_drain(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> tuple[ApprovalProjection, ...]:
    """Wait for admitted producers after fencing, then force-resync carriers."""
    key = (str(owner_user_id or "").strip(), int(batch_id))
    if key not in self._batch_admission_fences:
      raise RuntimeError("batch approval projection admission is not fenced")
    gate = self._admission_gates.setdefault(key, _BatchAdmissionGate())
    await gate.drained.wait()
    if self._shutdown_started:
      raise RuntimeError("batch approval projection registry is shutting down")
    return self.snapshot_fenced_batch(
      owner_user_id=key[0],
      batch_id=key[1],
    )

  def release_batch_fence(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
    rollback: bool = False,
  ) -> None:
    key = (str(owner_user_id or "").strip(), int(batch_id))
    prior_projections = self._batch_fence_projection_snapshots.pop(key, None)
    self._batch_fence_projection_history.pop(key, None)
    if rollback and prior_projections is not None:
      self._restore_batch_projections(key, prior_projections)
    self._batch_admission_fences.discard(key)
    gate = self._admission_gates.get(key)
    if gate is not None and not self._shutdown_started:
      gate.closed = False

  def close_admission_for_shutdown(
    self,
  ) -> dict[tuple[str, int], tuple[ApprovalProjection, ...]]:
    """Close admission globally and capture approvals already published."""
    self._shutdown_started = True
    keys = self.batch_keys() | set(self._admission_gates)
    self._batch_admission_fences.update(keys)
    for key in keys:
      self._admission_gates.setdefault(key, _BatchAdmissionGate()).closed = True
      self._batch_fence_projection_history.setdefault(key, {})
    for carrier in tuple(self._carriers.values()):
      self._sync_carrier(carrier, force=True)
    return {
      key: self._projection_snapshot_for_batch(key)
      for key in sorted(keys)
    }

  async def snapshot_after_shutdown_drain(
    self,
  ) -> dict[tuple[str, int], tuple[ApprovalProjection, ...]]:
    """Drain admitted producers, then resync the final projection set."""
    keys = self.batch_keys() | set(self._admission_gates)
    await asyncio.gather(*(
      self._admission_gates.setdefault(key, _BatchAdmissionGate()).drained.wait()
      for key in keys
    ))
    for carrier in tuple(self._carriers.values()):
      self._sync_carrier(carrier, force=True)
    keys = (
      self.batch_keys()
      | set(self._admission_gates)
      | set(self._batch_fence_projection_history)
    )
    return {
      key: self.merge_projection_sets(
        tuple(self._batch_fence_projection_history.pop(key, {}).values()),
        self._projection_snapshot_for_batch(key),
      )
      for key in sorted(keys)
    }

  async def begin_shutdown(self) -> dict[tuple[str, int], tuple[ApprovalProjection, ...]]:
    """Close admission, drain producers, and return the merged snapshot."""
    before = self.close_admission_for_shutdown()
    after = await self.snapshot_after_shutdown_drain()
    return self._merge_projection_snapshots(before, after)

  def projections_for_owner(
    self,
    *,
    owner_user_id: str,
    channel: str | None,
  ) -> list[ApprovalProjection]:
    normalized_owner = str(owner_user_id or "").strip()
    normalized_channel = normalize_approval_channel(channel)
    for carrier in tuple(self._carriers.values()):
      if carrier.owner_user_id == normalized_owner and carrier.channel == normalized_channel:
        self._sync_carrier(carrier)
    return sorted(
      (
        projection
        for projection in self._projections.values()
        if projection.owner_user_id == normalized_owner
        and projection.channel == normalized_channel
      ),
      key=lambda projection: (projection.batch_id, projection.approval_id),
    )

  def projections_for_batch(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> list[ApprovalProjection]:
    normalized_owner = str(owner_user_id or "").strip()
    normalized_batch_id = int(batch_id)
    for carrier in tuple(self._carriers.values()):
      if (
        carrier.owner_user_id == normalized_owner
        and carrier.batch_id == normalized_batch_id
      ):
        self._sync_carrier(carrier)
    return sorted(
      (
        projection
        for projection in self._projections.values()
        if projection.owner_user_id == normalized_owner
        and projection.batch_id == normalized_batch_id
      ),
      key=lambda projection: projection.approval_id,
    )

  def snapshot_fenced_batch(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> tuple[ApprovalProjection, ...]:
    """Force-resync a fenced batch while teardown still owns its carriers."""
    key = (str(owner_user_id or "").strip(), int(batch_id))
    if not self._shutdown_started and key not in self._batch_admission_fences:
      raise RuntimeError("batch approval projection admission is not fenced")
    for carrier in tuple(self._carriers.values()):
      if (carrier.owner_user_id, carrier.batch_id) == key:
        self._sync_carrier(carrier, force=True)
    return self.merge_projection_sets(
      tuple(self._batch_fence_projection_history.pop(key, {}).values()),
      self._projection_snapshot_for_batch(key),
    )

  def bound_authorization_subjects_for_fenced_batch(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> tuple[BoundApprovalAuthorizationSubject, ...]:
    """Snapshot bound pre-fence requests for cancellation authorization."""
    key = (str(owner_user_id or "").strip(), int(batch_id))
    if not self._shutdown_started and key not in self._batch_admission_fences:
      raise RuntimeError("batch approval projection admission is not fenced")
    gate = self._admission_gates.get(key)
    if gate is None:
      return ()
    subjects: dict[str, BoundApprovalAuthorizationSubject] = {}
    for admission in tuple(gate.admissions.values()):
      request = admission.request
      store = admission.store
      if request is None or store is None:
        continue
      approval_id = str(getattr(request, "approval_id", "") or "").strip()
      tool_call_id = str(getattr(request, "tool_call_id", "") or "").strip()
      session_id = str(
        getattr(admission.carrier.session, "session_id", "") or ""
      ).strip()
      if not approval_id or not tool_call_id or not session_id:
        raise RuntimeError("bound batch approval identity is unavailable")
      subjects[approval_id] = BoundApprovalAuthorizationSubject(
        approval_id=approval_id,
        batch_id=key[1],
        run_id=admission.carrier.run_id,
        owner_user_id=key[0],
        channel=admission.carrier.channel,
        tool_call_id=tool_call_id,
        session_id=session_id,
        request=request,
        session=admission.carrier.session,
        store=store,
        policy=admission.carrier.policy,
      )
    return tuple(sorted(subjects.values(), key=lambda item: item.approval_id))

  def batch_keys(self) -> set[tuple[str, int]]:
    return {
      (carrier.owner_user_id, carrier.batch_id)
      for carrier in self._carriers.values()
    } | {
      (projection.owner_user_id, projection.batch_id)
      for projection in self._projections.values()
    } | set(self._batch_admission_fences)

  def find_projection(
    self,
    *,
    run_id: str,
    approval_id: str,
    owner_user_id: str,
    channel: str | None,
  ) -> ApprovalProjection | None:
    normalized_approval_id = str(approval_id or "").strip()
    if not normalized_approval_id:
      return None
    self.projections_for_owner(owner_user_id=owner_user_id, channel=channel)
    projection = self._projections.get(normalized_approval_id)
    if projection is None or projection.run_id != str(run_id or "").strip():
      return None
    if projection.owner_user_id != str(owner_user_id or "").strip():
      return None
    if projection.channel != normalize_approval_channel(channel):
      return None
    return projection

  def all_projections(self) -> list[ApprovalProjection]:
    """Return every live projection after synchronizing its carrier."""
    for carrier in tuple(self._carriers.values()):
      self._sync_carrier(carrier)
    return sorted(
      self._projections.values(),
      key=lambda projection: (
        projection.owner_user_id,
        projection.batch_id,
        projection.approval_id,
      ),
    )

  def close_batch(self, *, owner_user_id: str, batch_id: int) -> None:
    normalized_owner = str(owner_user_id or "").strip()
    normalized_batch_id = int(batch_id)
    key = (normalized_owner, normalized_batch_id)
    gate = self._admission_gates.get(key)
    if gate is not None and gate.active > 0:
      gate.closed = True
      gate.close_when_drained = True
      self._batch_admission_fences.add(key)
      return
    self._close_batch_now(key)

  def _close_batch_now(self, key: tuple[str, int]) -> None:
    normalized_owner, normalized_batch_id = key
    self._carriers = {
      carrier_key: carrier
      for carrier_key, carrier in self._carriers.items()
      if not (
        carrier.owner_user_id == normalized_owner
        and carrier.batch_id == normalized_batch_id
      )
    }
    self._projections = {
      approval_id: projection
      for approval_id, projection in self._projections.items()
      if not (
        projection.owner_user_id == normalized_owner
        and projection.batch_id == normalized_batch_id
      )
    }
    self._batch_admission_fences.discard(key)
    self._batch_fence_projection_snapshots.pop(key, None)
    self._batch_fence_projection_history.pop(key, None)
    self._admission_gates.pop(key, None)

  def clear(self) -> None:
    self._carriers.clear()
    self._projections.clear()
    self._batch_admission_fences.clear()
    self._batch_fence_projection_snapshots.clear()
    self._batch_fence_projection_history.clear()
    self._admission_gates.clear()

  def _publish_admitted_carrier(self, admission: BatchApprovalAdmission) -> None:
    if admission.registry is not self:
      raise RuntimeError("batch approval admission belongs to another registry")
    gate = self._admission_gates.get(admission.key)
    if gate is None or gate.active < 1:
      raise RuntimeError("batch approval admission is not active")
    prior_projections = dict(self._projections)
    try:
      self._sync_carrier(admission.carrier, force=True)
      request = admission.request
      approval_id = str(getattr(request, "approval_id", "") or "").strip()
      projection = self._projections.get(approval_id)
      if (
        request is None
        or projection is None
        or projection.session is not admission.carrier.session
        or not approval_record_matches_projection(request, projection)
      ):
        raise RuntimeError("batch approval publication identity is unavailable")
      fence_history = self._batch_fence_projection_history.get(admission.key)
      if fence_history is not None:
        fence_history[approval_id] = projection
    except BaseException:
      self._projections = prior_projections
      raise

  def _release_admission(self, admission: BatchApprovalAdmission) -> None:
    gate = self._admission_gates.get(admission.key)
    if gate is None or gate.active < 1:
      raise RuntimeError("batch approval admission accounting is invalid")
    registered = gate.admissions.pop(id(admission), None)
    if registered is not admission:
      raise RuntimeError("batch approval admission registration is invalid")
    gate.active -= 1
    if gate.active == 0:
      gate.drained.set()
      if gate.close_when_drained and not self._shutdown_started:
        self._close_batch_now(admission.key)

  def _restore_batch_projections(
    self,
    key: tuple[str, int],
    prior_projections: dict[str, ApprovalProjection],
  ) -> None:
    owner_user_id, batch_id = key
    self._projections = {
      approval_id: projection
      for approval_id, projection in self._projections.items()
      if not (
        projection.owner_user_id == owner_user_id
        and projection.batch_id == batch_id
      )
    }
    self._projections.update(prior_projections)

  def _projection_snapshot_for_batch(
    self,
    key: tuple[str, int],
  ) -> tuple[ApprovalProjection, ...]:
    owner_user_id, batch_id = key
    return tuple(sorted(
      (
        projection
        for projection in self._projections.values()
        if projection.owner_user_id == owner_user_id
        and projection.batch_id == batch_id
      ),
      key=lambda projection: projection.approval_id,
    ))

  @staticmethod
  def _merge_projection_snapshots(
    *snapshots: dict[tuple[str, int], tuple[ApprovalProjection, ...]],
  ) -> dict[tuple[str, int], tuple[ApprovalProjection, ...]]:
    merged: dict[tuple[str, int], dict[str, ApprovalProjection]] = {}
    for snapshot in snapshots:
      for key, projections in snapshot.items():
        target = merged.setdefault(key, {})
        for projection in projections:
          target[projection.approval_id] = projection
    return {
      key: tuple(sorted(values.values(), key=lambda item: item.approval_id))
      for key, values in merged.items()
    }

  @staticmethod
  def merge_projection_sets(
    *projection_sets: tuple[ApprovalProjection, ...],
  ) -> tuple[ApprovalProjection, ...]:
    merged: dict[str, ApprovalProjection] = {}
    for projections in projection_sets:
      for projection in projections:
        merged[projection.approval_id] = projection
    return tuple(sorted(merged.values(), key=lambda item: item.approval_id))

  def _sync_carrier(
    self,
    carrier: _BatchApprovalCarrier,
    *,
    force: bool = False,
  ) -> None:
    if not force and (
      self._shutdown_started
      or (carrier.owner_user_id, carrier.batch_id) in self._batch_admission_fences
    ):
      return
    session_id = str(getattr(carrier.session, "session_id", "") or "").strip()
    if not session_id:
      raise ValueError("batch approval session_id is required")
    live_ids: set[str] = set()
    pending_tools = getattr(carrier.session, "pending_tools", {})
    for tool_call_id, pending in pending_tools.items():
      if not isinstance(pending, dict) or pending.get("status") != "approval_pending":
        continue
      approval_id = str(pending.get("approval_id") or "").strip()
      nonce = str(pending.get("nonce") or "").strip()
      normalized_tool_call_id = str(tool_call_id or "").strip()
      if not approval_id or not nonce or not normalized_tool_call_id:
        continue
      live_ids.add(approval_id)
      stage_run_seq = pending.get("stage_run_seq")
      try:
        normalized_stage_run_seq = int(stage_run_seq) if stage_run_seq is not None else None
      except (TypeError, ValueError):
        raise ValueError("batch approval stage_run_seq must be an integer") from None
      projection = ApprovalProjection(
        approval_id=approval_id,
        batch_id=carrier.batch_id,
        run_id=carrier.run_id,
        stage_run_seq=normalized_stage_run_seq,
        owner_user_id=carrier.owner_user_id,
        channel=carrier.channel,
        tool_call_id=normalized_tool_call_id,
        nonce=nonce,
        session_id=session_id,
        lifecycle="live",
        session=carrier.session,
        store=carrier.store,
        policy=carrier.policy,
      )
      existing = self._projections.get(approval_id)
      if existing is not None and (
        existing != projection
        or existing.session is not carrier.session
        or existing.store is not carrier.store
        or existing.policy is not carrier.policy
      ):
        raise ValueError("conflicting batch approval identity registration")
      self._projections[approval_id] = projection
    self._projections = {
      approval_id: projection
      for approval_id, projection in self._projections.items()
      if projection.session is not carrier.session or approval_id in live_ids
    }


def approval_record_matches_projection(
  record: Any,
  projection: ApprovalProjection | BoundApprovalAuthorizationSubject,
) -> bool:
  """Fail-closed durable/live identity join for a projected approval."""
  return all((
    str(getattr(record, "approval_id", "") or "") == projection.approval_id,
    str(getattr(record, "tool_call_id", "") or "") == projection.tool_call_id,
    str(getattr(record, "user_id", "") or "") == projection.owner_user_id,
    str(getattr(record, "request_id", "") or "") == projection.run_id,
    str(getattr(record, "run_id", "") or "") == projection.run_id,
    str(getattr(record, "session_id", "") or "") == projection.session_id,
    normalize_approval_channel(getattr(record, "channel", None)) == projection.channel,
  ))


@dataclass(frozen=True)
class BatchApprovalScope:
  batch_id: int
  owner_user_id: str
  channel: str | None
  store: Any
  policy: Any
  registry: BatchApprovalProjectionRegistry

  @property
  def run_id(self) -> str:
    return f"batch_{self.batch_id}"

  def register_session(self, session: Any) -> _BatchApprovalCarrier:
    return self.registry.register_session(
      batch_id=self.batch_id,
      owner_user_id=self.owner_user_id,
      channel=self.channel,
      session=session,
      store=self.store,
      policy=self.policy,
    )

  def acquire_admission(self, session: Any) -> BatchApprovalAdmission:
    return self.registry.acquire_admission(
      batch_id=self.batch_id,
      owner_user_id=self.owner_user_id,
      session=session,
    )


_ACTIVE_BATCH_APPROVAL_SCOPE: ContextVar[BatchApprovalScope | None] = ContextVar(
  "active_batch_approval_scope",
  default=None,
)


def current_batch_approval_scope() -> BatchApprovalScope | None:
  return _ACTIVE_BATCH_APPROVAL_SCOPE.get()


def acquire_batch_approval_admission(
  session: Any | None,
) -> BatchApprovalAdmission | None:
  if session is None:
    return None
  scope = getattr(session, "batch_approval_scope", None)
  if scope is None:
    return None
  if not isinstance(scope, BatchApprovalScope):
    raise RuntimeError("batch approval session scope is invalid")
  return scope.acquire_admission(session)


async def abort_unpublished_batch_approval_admission(
  admission: BatchApprovalAdmission,
) -> None:
  cleanup_task = asyncio.create_task(admission.abort_unpublished())
  while not cleanup_task.done():
    try:
      await asyncio.shield(cleanup_task)
    except asyncio.CancelledError:
      continue
  cleanup_task.result()


@contextmanager
def bind_batch_approval_scope(scope: BatchApprovalScope) -> Iterator[None]:
  token = _ACTIVE_BATCH_APPROVAL_SCOPE.set(scope)
  try:
    yield
  finally:
    _ACTIVE_BATCH_APPROVAL_SCOPE.reset(token)


__all__ = [
  "ApprovalProjection",
  "BoundApprovalAuthorizationSubject",
  "approval_record_matches_projection",
  "acquire_batch_approval_admission",
  "abort_unpublished_batch_approval_admission",
  "BatchApprovalAdmission",
  "BatchApprovalProjectionRegistry",
  "BatchApprovalScope",
  "bind_batch_approval_scope",
  "current_batch_approval_scope",
  "normalize_approval_channel",
]
