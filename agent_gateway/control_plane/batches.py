from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from agent_gateway.approvals import (
  ApprovalActionError,
  _cancel_pending_approval_and_unblock,
  _release_cancelled_approval,
)
from agent_gateway.batch_approval_projection import (
  BatchApprovalProjectionRegistry,
  BatchApprovalScope,
  approval_record_matches_projection,
  bind_batch_approval_scope,
)
from agent_gateway.fixture_gate import fixture_provider_available, fixture_unavailable_message
from agent_gateway.runtime_auth_context import bind_authenticated_runtime_auth_config
from agent_gateway.session import AuthManager

from .corpus_readiness import (
  CORPUS_REQUIREMENTS_FIELD,
  CorpusReadinessGateError,
  require_corpus_readiness,
)
from .runs import _require_bearer_session, _require_control_session, _session_owner_user_id

_BATCH_CONTROLLER_MODULE_NAMES = frozenset({"agent", "agent.batch", "agent.batch.controller"})
_BATCH_REGISTRY_MODULE_NAMES = frozenset({"agent", "agent.batch", "agent.batch.registry"})
_BATCH_WORKFLOW_MODULE_NAMES = frozenset({"agent", "agent.skills", "agent.skills.diligence_tracks"})
_MEMORY_MODULE_NAMES = frozenset({"memory"})
log = logging.getLogger("agent_gateway.control_plane.batches")


@dataclass(frozen=True)
class _BatchCancellationFence:
  projections: tuple[Any, ...]
  authorization_subjects: tuple[Any, ...]


class BatchTaskRegistry:
  def __init__(self) -> None:
    self._tasks: dict[tuple[str, int], asyncio.Task[Any]] = {}
    self._monitors: dict[tuple[str, int], asyncio.Task[Any]] = {}
    self._cancel_admission_fences: set[tuple[str, int]] = set()
    self._shutdown_started = False
    self.approval_projections = BatchApprovalProjectionRegistry()

  def start(self, *, owner_user_id: str, batch_id: int, task: asyncio.Task[Any], registry: Any) -> None:
    key = self._key(owner_user_id, batch_id)
    if self._shutdown_started or key in self._cancel_admission_fences:
      if not task.done():
        task.cancel()
      raise RuntimeError("batch task admission is closed")
    existing = self._tasks.get(key)
    if existing is not None and not existing.done():
      raise RuntimeError("batch task is already registered")
    self._tasks[key] = task
    self._monitors[key] = asyncio.create_task(
      self._consume(key, task, registry)
    )

  def assert_accepting(self) -> None:
    if self._shutdown_started:
      raise RuntimeError("batch task registry is shutting down")

  def begin_batch_cancellation(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> _BatchCancellationFence:
    key = self._key(owner_user_id, batch_id)
    self.assert_accepting()
    if key in self._cancel_admission_fences:
      raise RuntimeError("batch cancellation is already in progress")
    self._cancel_admission_fences.add(key)
    try:
      projections = self.approval_projections.close_batch_admission(
        owner_user_id=key[0],
        batch_id=key[1],
      )
      bound_subjects = (
        self.approval_projections.bound_authorization_subjects_for_fenced_batch(
          owner_user_id=key[0],
          batch_id=key[1],
        )
      )
      authorization_by_id = {
        subject.approval_id: subject
        for subject in bound_subjects
      }
      authorization_by_id.update({
        projection.approval_id: projection
        for projection in projections
      })
      return _BatchCancellationFence(
        projections=projections,
        authorization_subjects=tuple(sorted(
          authorization_by_id.values(),
          key=lambda subject: subject.approval_id,
        )),
      )
    except BaseException:
      self._cancel_admission_fences.discard(key)
      self.approval_projections.release_batch_fence(
        owner_user_id=key[0],
        batch_id=key[1],
        rollback=True,
      )
      raise

  def abort_batch_cancellation(self, *, owner_user_id: str, batch_id: int) -> None:
    key = self._key(owner_user_id, batch_id)
    self._cancel_admission_fences.discard(key)
    self.approval_projections.release_batch_fence(
      owner_user_id=key[0],
      batch_id=key[1],
      rollback=True,
    )

  def finish_batch_cancellation(self, *, owner_user_id: str, batch_id: int) -> None:
    key = self._key(owner_user_id, batch_id)
    self._cancel_admission_fences.discard(key)
    self.approval_projections.release_batch_fence(
      owner_user_id=key[0],
      batch_id=key[1],
    )

  def get(self, *, owner_user_id: str, batch_id: int) -> asyncio.Task[Any] | None:
    return self._tasks.get(self._key(owner_user_id, batch_id))

  async def cancel(self, *, owner_user_id: str, batch_id: int) -> bool:
    key = self._key(owner_user_id, batch_id)
    task = self._tasks.get(key)
    monitor = self._monitors.get(key)
    if task is None:
      if key not in self._cancel_admission_fences:
        self.approval_projections.close_batch(
          owner_user_id=key[0],
          batch_id=key[1],
        )
      return False
    if not task.done():
      task.cancel()
    if monitor is not None:
      await asyncio.gather(monitor, return_exceptions=True)
    return True

  def live_count(self) -> int:
    return sum(1 for task in self._tasks.values() if not task.done())

  async def shutdown(self, *, app_state: Any | None = None) -> None:
    _ = app_state
    self._shutdown_started = True
    projection_snapshots_before_cancel = (
      self.approval_projections.close_admission_for_shutdown()
    )
    for task in tuple(self._tasks.values()):
      if not task.done():
        task.cancel()
    projection_snapshots_after_drain = (
      await self.approval_projections.snapshot_after_shutdown_drain()
    )
    projection_snapshots = self.approval_projections._merge_projection_snapshots(
      projection_snapshots_before_cancel,
      projection_snapshots_after_drain,
    )
    approval_error = await self._quarantine_shutdown_approvals_until_stable(
      projection_snapshots,
    )
    while self._monitors:
      monitors = tuple(self._monitors.values())
      await asyncio.gather(*monitors, return_exceptions=True)
    late_approval_error = await self._quarantine_shutdown_approvals_until_stable({})
    if approval_error is None:
      approval_error = late_approval_error
    self._tasks.clear()
    self._monitors.clear()
    self._cancel_admission_fences.clear()
    self.approval_projections.clear()
    if approval_error is not None:
      log.error(
        "Batch approval shutdown quarantine failed: %s",
        approval_error.payload.get("error"),
      )

  async def _quarantine_shutdown_approvals_until_stable(
    self,
    projection_snapshots: dict[tuple[str, int], tuple[Any, ...]],
  ) -> ApprovalActionError | None:
    seeds = dict(projection_snapshots)
    approval_error: ApprovalActionError | None = None
    stable_empty_passes = 0
    while stable_empty_passes < 2:
      found_projection = False
      batch_keys = (
        set(self._tasks)
        | set(seeds)
        | self.approval_projections.batch_keys()
      )
      for owner_user_id, batch_id in sorted(batch_keys):
        projections = self.approval_projections.merge_projection_sets(
          seeds.pop((owner_user_id, batch_id), ()),
          self.approval_projections.snapshot_fenced_batch(
            owner_user_id=owner_user_id,
            batch_id=batch_id,
          ),
        )
        if not projections:
          continue
        found_projection = True
        try:
          warning = await _deny_batch_pending_approvals_for_cancel(
            projections=projections,
            owner_user_id=owner_user_id,
            batch_id=batch_id,
            authenticated=None,
            app_state=None,
            reason="gateway_shutdown",
            trusted_lifecycle_authority=True,
          )
          if approval_error is None:
            approval_error = warning
        except ApprovalActionError as exc:
          if approval_error is None:
            approval_error = exc
        except asyncio.CancelledError:
          if approval_error is None:
            approval_error = ApprovalActionError(
              503,
              {"error": "Batch approval shutdown quarantine was interrupted"},
            )
        except Exception:
          log.exception(
            "Unexpected batch approval shutdown quarantine failure for batch %s",
            batch_id,
          )
          if approval_error is None:
            approval_error = ApprovalActionError(
              503,
              {"error": "Batch approval shutdown quarantine failed"},
            )
      if found_projection:
        stable_empty_passes = 0
        continue
      stable_empty_passes += 1
      if stable_empty_passes < 2:
        await asyncio.sleep(0)
    return approval_error

  async def _consume(
    self,
    key: tuple[str, int],
    task: asyncio.Task[Any],
    registry: Any,
  ) -> None:
    owner_user_id, batch_id = key
    try:
      try:
        await task
      except asyncio.CancelledError:
        _set_status_if_not_terminal(registry, batch_id, "cancelled")
      except Exception as exc:
        _set_status_if_not_terminal(registry, batch_id, "failed", error=str(exc))
    finally:
      if self._tasks.get(key) is task:
        self._tasks.pop(key, None)
      self._monitors.pop(key, None)
      if not self._shutdown_started and key not in self._cancel_admission_fences:
        self.approval_projections.close_batch(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
        )
      close = getattr(registry, "close", None)
      if callable(close):
        close()

  @staticmethod
  def _key(owner_user_id: str, batch_id: int) -> tuple[str, int]:
    owner = str(owner_user_id or "").strip()
    normalized_batch_id = int(batch_id)
    if not owner or normalized_batch_id < 1:
      raise ValueError("batch task owner and positive batch_id are required")
    return owner, normalized_batch_id


def build_batches_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter(prefix="/batches")

  @router.post("")
  async def dispatch_batch(request: Request, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    return await _dispatch_batch_for_authenticated(request, payload, authenticated)

  @router.get("")
  async def list_batches(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
  ) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    registry = _registry_for_user(owner_user_id)
    try:
      rows = registry.list_batches(owner_user_id, limit=limit)
      batches = [registry.get_batch_digest(int(row["batch_id"])) for row in rows]
      return {"batches": batches}
    finally:
      registry.close()

  @router.get("/workflows")
  async def list_batch_workflows(request: Request) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    return {"workflows": _batch_workflow_catalog()}

  @router.post("/fixture")
  async def seed_fixture_batch(
    request: Request,
    payload: dict[str, Any] | None = Body(default=None),
  ) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    if payload is not None and not isinstance(payload, dict):
      raise HTTPException(status_code=422, detail="fixture batch spec must be an object")
    body = payload or {}
    if body.get("dev_mode") is not True:
      raise HTTPException(status_code=403, detail="fixture batch seed requires dev_mode=true")
    if not fixture_provider_available():
      raise HTTPException(status_code=403, detail=fixture_unavailable_message("fixture batch seed"))
    fixture = str(body.get("fixture") or "mixed").strip().lower()
    if fixture != "mixed":
      raise HTTPException(status_code=422, detail="unknown fixture batch: expected 'mixed'")

    registry = _registry_for_user(owner_user_id)
    try:
      try:
        batch_id = _seed_fixture_batch(
          registry,
          user_id=owner_user_id,
          host=socket.gethostname(),
          pid=os.getpid(),
        )
      except _active_batch_error_type() as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
      return _batch_detail_payload(registry, batch_id, top_n=10)
    finally:
      registry.close()

  @router.get("/{batch_id}")
  async def get_batch(
    request: Request,
    batch_id: int,
    top_n: int = Query(default=10, ge=1, le=100),
  ) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    return read_batch_for_user(
      batch_id,
      user_id=_session_owner_user_id(authenticated),
      top_n=top_n,
    )

  @router.delete("/{batch_id}")
  async def cancel_batch(request: Request, batch_id: int) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    registry = _registry_for_user(owner_user_id)
    try:
      _require_batch_owner(registry, batch_id, owner_user_id)
      task_registry = _task_registry(request)
      try:
        cancellation_fence = task_registry.begin_batch_cancellation(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
        )
      except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
      try:
        await _preflight_batch_approval_cancellation(
          projections=cancellation_fence.authorization_subjects,
          authenticated=authenticated,
        )
      except ApprovalActionError as exc:
        task_registry.abort_batch_cancellation(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
        )
        raise HTTPException(
          status_code=exc.status_code,
          detail=exc.payload,
        ) from exc
      except asyncio.CancelledError:
        task_registry.abort_batch_cancellation(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
        )
        raise
      except Exception as exc:
        task_registry.abort_batch_cancellation(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
        )
        raise HTTPException(
          status_code=503,
          detail={"error": "Batch approval cancellation preflight failed"},
        ) from exc
      completion_task = asyncio.create_task(
        _complete_authorized_batch_cancellation(
          task_registry=task_registry,
          registry=registry,
          projections=cancellation_fence.projections,
          owner_user_id=owner_user_id,
          batch_id=batch_id,
          authenticated=authenticated,
          app_state=request.app.state,
        )
      )
      try:
        try:
          approval_error, batch_digest = await asyncio.shield(completion_task)
        except asyncio.CancelledError as cancellation:
          while not completion_task.done():
            try:
              await asyncio.shield(completion_task)
            except asyncio.CancelledError:
              continue
          try:
            completion_task.result()
          except BaseException:
            log.exception(
              "Batch cancellation completion failed after caller cancellation for batch %s",
              batch_id,
            )
          raise cancellation
        if approval_error is not None:
          raise HTTPException(
            status_code=approval_error.status_code,
            detail=approval_error.payload,
          ) from approval_error
        return {"batch": batch_digest}
      finally:
        task_registry.finish_batch_cancellation(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
        )
    finally:
      registry.close()

  @router.post("/{batch_id}/retry-failed")
  async def retry_failed_batch(request: Request, batch_id: int) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    owner_user_id = _session_owner_user_id(authenticated)
    source_registry = _registry_for_user(owner_user_id)
    try:
      prior = _require_batch_owner(source_registry, batch_id, owner_user_id)
      failures = source_registry.get_batch_failures(batch_id)
      tickers = sorted({str(row.get("ticker") or "").strip().upper() for row in failures if str(row.get("ticker") or "").strip()})
      if not tickers:
        raise HTTPException(status_code=422, detail="batch has no failed tickers to retry")
      spec = _retry_spec(prior, tickers)
    finally:
      source_registry.close()

    dispatch_payload = dict(spec)
    return await _dispatch_batch_for_authenticated(request, dispatch_payload, authenticated)

  return router


async def _dispatch_batch_for_authenticated(request: Request, payload: dict[str, Any], authenticated: Any) -> dict[str, Any]:
  auth_config = _authenticated_batch_auth_config(authenticated)
  gateway_config = getattr(request.app.state, "gateway_config", None)
  if getattr(gateway_config, "credentials_resolver", None) is not None and not _has_active_credential(auth_config):
    raise HTTPException(status_code=503, detail="authenticated batch credential unavailable")
  try:
    return await dispatch_batch_in_process(
      payload,
      app_state=request.app.state,
      user_id=_session_owner_user_id(authenticated),
      user_email=authenticated.user_email,
      auth_config=auth_config,
      channel=getattr(authenticated, "channel", None),
    )
  except _active_batch_error_type() as exc:
    raise HTTPException(status_code=409, detail=str(exc)) from exc
  except CorpusReadinessGateError as exc:
    raise HTTPException(status_code=exc.status_code, detail=exc.to_payload()) from exc
  except RuntimeError as exc:
    raise HTTPException(status_code=409, detail=str(exc)) from exc
  except ValueError as exc:
    raise HTTPException(status_code=422, detail=str(exc)) from exc


async def dispatch_batch_in_process(
  payload: dict[str, Any],
  *,
  app_state: Any,
  user_id: str,
  user_email: str | None = None,
  auth_config: dict[str, Any] | None = None,
  channel: str | None = None,
) -> dict[str, Any]:
  if not isinstance(payload, dict):
    raise ValueError("batch spec must be an object")
  if auth_config is not None:
    auth_config = dict(auth_config)
    if not _has_active_credential(auth_config):
      raise ValueError("authenticated batch credential unavailable")
  payload, corpus_readiness = await require_corpus_readiness(
    payload,
    app_state=app_state,
  )
  registry = _registry_for_user(user_id)
  try:
    task_registry = _task_registry_for_state(app_state)
    task_registry.assert_accepting()
    batch_id, _user_id, _user_email = _controller().acquire_batch_run(
      payload,
      registry=registry,
      host=socket.gethostname(),
      pid=os.getpid(),
      user_id=user_id,
      user_email=user_email,
    )
    approval_store = getattr(app_state, "gateway_approval_store", None)
    approval_policy = getattr(app_state, "gateway_approval_policy", None)
    scope = (
      BatchApprovalScope(
        batch_id=batch_id,
        owner_user_id=user_id,
        channel=channel,
        store=approval_store,
        policy=approval_policy,
        registry=task_registry.approval_projections,
      )
      if approval_store is not None and approval_policy is not None
      else None
    )
    # asyncio tasks copy their creation context. Bind only while creating the
    # batch task so credentials stay in memory, isolated from sibling batches,
    # and never enter the persisted spec/registry/event payloads.
    with bind_authenticated_runtime_auth_config(auth_config):
      if scope is None:
        task = asyncio.create_task(
          _controller().run_acquired_batch(
            batch_id,
            payload,
            registry=registry,
            _identity=(user_id, user_email),
            _on_finalize=lambda event: _publish_batch_terminal_event(app_state, event),
          )
        )
      else:
        # asyncio tasks copy their creation context. Bind only for construction so
        # nested stages inherit this batch's routing scope without process globals.
        with bind_batch_approval_scope(scope):
          task = asyncio.create_task(
            _controller().run_acquired_batch(
              batch_id,
              payload,
              registry=registry,
              _identity=(user_id, user_email),
              _on_finalize=lambda event: _publish_batch_terminal_event(app_state, event),
            )
          )
    task_registry.start(
      owner_user_id=user_id,
      batch_id=batch_id,
      task=task,
      registry=registry,
    )
    response: dict[str, Any] = {"batch_id": batch_id, "status": "running"}
    if corpus_readiness is not None:
      response["corpus_readiness"] = corpus_readiness
    return response
  except Exception:
    registry.close()
    raise


async def _complete_authorized_batch_cancellation(
  *,
  task_registry: BatchTaskRegistry,
  registry: Any,
  projections: tuple[Any, ...],
  owner_user_id: str,
  batch_id: int,
  authenticated: Any,
  app_state: Any,
) -> tuple[ApprovalActionError | None, dict[str, Any]]:
  """Finish quarantine and task teardown after the mutation boundary."""
  approval_error: ApprovalActionError | None = None
  task = task_registry.get(owner_user_id=owner_user_id, batch_id=batch_id)
  if task is not None and not task.done():
    registry.transition_status_if_current(
      batch_id,
      "cancelling",
      expected_statuses={"running", "remediating"},
    )
  await task_registry.cancel(owner_user_id=owner_user_id, batch_id=batch_id)
  try:
    drained_projections = (
      await task_registry.approval_projections.snapshot_after_batch_drain(
        owner_user_id=owner_user_id,
        batch_id=batch_id,
      )
    )
  except asyncio.CancelledError:
    raise
  except Exception as exc:
    drained_projections = ()
    approval_error = ApprovalActionError(
      503,
      {"error": "Batch approval admission drain failed"},
    )
    log.exception(
      "Batch approval admission drain failed for batch %s",
      batch_id,
      exc_info=exc,
    )
  try:
    warning = await _deny_batch_pending_approvals_for_cancel(
      projections=projections,
      owner_user_id=owner_user_id,
      batch_id=batch_id,
      authenticated=authenticated,
      app_state=app_state,
      reason="batch_cancelled",
      trusted_lifecycle_authority=False,
    )
    if approval_error is None:
      approval_error = warning
  except ApprovalActionError as exc:
    if approval_error is None:
      approval_error = exc
  except asyncio.CancelledError:
    raise
  except Exception:
    log.exception(
      "Unexpected batch approval cancellation quarantine failure for batch %s",
      batch_id,
    )
    if approval_error is None:
      approval_error = ApprovalActionError(
        503,
        {"error": "Batch approval cancellation quarantine failed"},
      )
  late_passes = 0
  late_seed = drained_projections
  while late_passes < 2:
    late_projections = task_registry.approval_projections.merge_projection_sets(
      late_seed,
      task_registry.approval_projections.snapshot_fenced_batch(
        owner_user_id=owner_user_id,
        batch_id=batch_id,
      ),
    )
    late_seed = ()
    if not late_projections:
      late_passes += 1
      if late_passes < 2:
        await asyncio.sleep(0)
      continue
    late_passes = 0
    try:
      warning = await _deny_batch_pending_approvals_for_cancel(
        projections=late_projections,
        owner_user_id=owner_user_id,
        batch_id=batch_id,
        authenticated=None,
        app_state=app_state,
        reason="batch_cancelled",
        trusted_lifecycle_authority=True,
      )
      if approval_error is None:
        approval_error = warning
    except ApprovalActionError as exc:
      if approval_error is None:
        approval_error = exc
    except asyncio.CancelledError:
      if approval_error is None:
        approval_error = ApprovalActionError(
          503,
          {"error": "Late batch approval quarantine was interrupted"},
        )
    except Exception:
      log.exception(
        "Unexpected late batch approval quarantine failure for batch %s",
        batch_id,
      )
      if approval_error is None:
        approval_error = ApprovalActionError(
          503,
          {"error": "Late batch approval quarantine failed"},
        )
  task_registry.approval_projections.close_batch(
    owner_user_id=owner_user_id,
    batch_id=batch_id,
  )
  return approval_error, registry.get_batch_digest(batch_id)


async def _deny_batch_pending_approvals_for_cancel(
  *,
  projections: tuple[Any, ...],
  owner_user_id: str,
  batch_id: int,
  authenticated: Any | None,
  app_state: Any,
  reason: str,
  trusted_lifecycle_authority: bool,
) -> ApprovalActionError | None:
  run_id = f"batch_{int(batch_id)}"
  processed: dict[str, tuple[Any, dict[str, Any]]] = {}
  first_error: ApprovalActionError | None = None
  identity_warning: ApprovalActionError | None = None

  def release_processed() -> None:
    for projection, pending_entry in processed.values():
      _release_cancelled_approval(
        target_session=projection.session,
        pending_entry=pending_entry,
        tool_call_id=projection.tool_call_id,
        approval_id=projection.approval_id,
      )

  candidates = [
    projection
    for projection in projections
    if projection.run_id == run_id
  ]
  for projection in candidates:
    pending_entry = projection.session.pending_tools.get(projection.tool_call_id)
    if not isinstance(pending_entry, dict):
      pending_entry = {
        "approval_id": projection.approval_id,
        "nonce": projection.nonce,
        "status": "approval_pending",
      }
    pending_entry["_approval_cancel_requested"] = True
    processed[projection.approval_id] = (projection, pending_entry)

  for projection in candidates:
    _, pending_entry = processed[projection.approval_id]
    if projection.store is None or projection.policy is None:
      if first_error is None:
        first_error = ApprovalActionError(
          503,
          {"error": "Approval subsystem unavailable", "approval_id": projection.approval_id},
        )
      continue
    try:
      result = await _cancel_pending_approval_and_unblock(
        target_session=projection.session,
        pending_entry=pending_entry,
        tool_call_id=projection.tool_call_id,
        nonce=projection.nonce,
        decider_id=(
          "gateway_shutdown"
          if authenticated is None
          else owner_user_id
        ),
        decider_role=(
          "system"
          if authenticated is None
          else getattr(authenticated, "role", None)
        ),
        reason=reason,
        app_state=app_state,
        authoritative_store=projection.store,
        authoritative_policy=projection.policy,
        expected_owner_user_id=projection.owner_user_id,
        expected_request_id=projection.run_id,
        expected_run_id=projection.run_id,
        expected_session_id=projection.session_id,
        expected_channel=projection.channel,
        system_override=trusted_lifecycle_authority,
        role_authorization_prechecked=(
          authenticated is not None and not trusted_lifecycle_authority
        ),
        release_queue=False,
      )
    except asyncio.CancelledError:
      release_processed()
      raise
    except ApprovalActionError as exc:
      if exc.status_code == 403 and authenticated is not None:
        exc = ApprovalActionError(
          503,
          {
            "error": "Batch approval authorization changed after preflight",
            "approval_id": projection.approval_id,
          },
        )
      if first_error is None:
        first_error = exc
      continue
    except Exception:
      if first_error is None:
        first_error = ApprovalActionError(
          503,
          {
            "error": "Batch approval cancellation quarantine failed",
            "approval_id": projection.approval_id,
          },
        )
      continue
    if not result["identity_matches"] and identity_warning is None:
      identity_warning = ApprovalActionError(
        409,
        {
          "error": "Batch approval durable identity mismatch quarantined",
          "approval_id": projection.approval_id,
        },
      )
  release_processed()
  if first_error is not None:
    raise first_error
  return identity_warning


async def _preflight_batch_approval_cancellation(
  *,
  projections: tuple[Any, ...],
  authenticated: Any,
) -> None:
  """Authorize user-driven lifecycle teardown before mutating live state."""
  first_error: BaseException | None = None
  for projection in projections:
    try:
      await _authorize_batch_projection_for_user_cancellation(
        projection=projection,
        authenticated=authenticated,
      )
    except asyncio.CancelledError:
      raise
    except ApprovalActionError as exc:
      if exc.status_code == 403:
        raise
      if first_error is None:
        first_error = exc
    except Exception as exc:
      if first_error is None:
        first_error = exc
  if first_error is not None:
    raise first_error


async def _authorize_batch_projection_for_user_cancellation(
  *,
  projection: Any,
  authenticated: Any,
) -> None:
  if projection.store is None or projection.policy is None:
    raise ApprovalActionError(
      503,
      {"error": "Approval subsystem unavailable", "approval_id": projection.approval_id},
    )
  request_record = await projection.store.get(projection.approval_id)
  if request_record is None:
    request_record = getattr(projection, "request", None)
  if request_record is None or not approval_record_matches_projection(
    request_record,
    projection,
  ):
    return
  if not projection.policy.role_authorized_for_class(
    decider_role=getattr(authenticated, "role", None),
    tool_class=request_record.tool_class,
  ):
    raise ApprovalActionError(
      403,
      {
        "error": "Role is not authorized to cancel this approval",
        "approval_id": projection.approval_id,
        "tool_class": request_record.tool_class,
      },
    )


def _authenticated_batch_auth_config(authenticated: Any) -> dict[str, Any] | None:
  auth_config = getattr(authenticated, "auth_config", None)
  if auth_config is None:
    return None
  to_dict = getattr(auth_config, "to_dict", None)
  if callable(to_dict):
    auth_config = to_dict()
  if not isinstance(auth_config, dict):
    raise HTTPException(status_code=503, detail="authenticated batch credential unavailable")
  return dict(auth_config)


def _has_active_credential(auth_config: dict[str, Any] | None) -> bool:
  if not auth_config:
    return False
  if str(auth_config.get("provider") or "").strip().lower() == "fixture":
    return True
  auth_mode = str(auth_config.get("auth_mode") or "").strip().lower()
  if auth_mode == "oauth":
    return bool(str(auth_config.get("auth_token") or "").strip())
  if auth_mode == "api":
    return bool(str(auth_config.get("api_key") or "").strip())
  return bool(
    str(auth_config.get("auth_token") or "").strip()
    or str(auth_config.get("api_key") or "").strip()
  )


def read_batch_for_user(batch_id: int, *, user_id: str, top_n: int = 10) -> dict[str, Any]:
  registry = _registry_for_user(user_id)
  try:
    _require_batch_owner(registry, batch_id, user_id)
    return _batch_detail_payload(registry, batch_id, top_n=top_n)
  finally:
    registry.close()


def _retry_spec(batch_row: dict[str, Any], tickers: list[str]) -> dict[str, Any]:
  try:
    spec = json.loads(str(batch_row.get("spec_json") or "{}"))
  except json.JSONDecodeError as exc:
    raise HTTPException(status_code=422, detail="prior batch spec is not valid JSON") from exc
  if not isinstance(spec, dict):
    raise HTTPException(status_code=422, detail="prior batch spec is not an object")
  spec = dict(spec)
  _set_retry_universe(spec, tickers)
  return spec


def _set_retry_universe(spec: dict[str, Any], tickers: list[str]) -> None:
  spec["universe"] = list(tickers)
  requirements = spec.get(CORPUS_REQUIREMENTS_FIELD)
  if not isinstance(requirements, list):
    return
  retry_tickers = {str(ticker or "").strip().upper() for ticker in tickers}
  spec[CORPUS_REQUIREMENTS_FIELD] = [
    dict(requirement)
    for requirement in requirements
    if isinstance(requirement, dict)
    and str(requirement.get("ticker") or "").strip().upper() in retry_tickers
  ]


def _require_batch_owner(registry: Any, batch_id: int, user_id: str) -> dict[str, Any]:
  try:
    digest = registry.get_batch_digest(batch_id)
  except KeyError as exc:
    raise HTTPException(status_code=404, detail="Batch not found") from exc
  if str(digest.get("user_id") or "") != str(user_id):
    raise HTTPException(status_code=404, detail="Batch not found")
  return digest


def _batch_detail_payload(registry: Any, batch_id: int, *, top_n: int) -> dict[str, Any]:
  batch = registry.get_batch_digest(batch_id)
  return {
    "batch": batch,
    "verdict_matrix": registry.get_batch_verdict_matrix(batch_id),
    "candidates": registry.get_batch_candidates(batch_id, top_n),
    "failures": _annotate_batch_failures(registry.get_batch_failures(batch_id), batch=batch),
    "diligence_prs": _list_diligence_prs(registry, batch_id),
  }


def _list_diligence_prs(registry: Any, batch_id: int) -> list[dict[str, Any]]:
  list_prs = getattr(registry, "list_diligence_prs", None)
  if not callable(list_prs):
    return []
  return list_prs(batch_id)


def _annotate_batch_failures(failures: list[dict[str, Any]], *, batch: dict[str, Any]) -> list[dict[str, Any]]:
  return [_annotate_batch_failure(row, batch=batch) for row in failures]


def _annotate_batch_failure(row: dict[str, Any], *, batch: dict[str, Any]) -> dict[str, Any]:
  annotated = dict(row)
  if str(annotated.get("error") or "").strip() != "skipped_existing_unroutable":
    return annotated
  ticker = str(annotated.get("ticker") or "").strip().upper()
  annotated["repair_hint"] = (
    "This ticker already has a research file that could not be routed for this batch source. "
    "Rerun it by calling start_diligence_batch with gates.force_rerun_existing=true."
  )
  annotated["retry_spec"] = _force_rerun_retry_spec(batch, ticker=ticker)
  return annotated


def _force_rerun_retry_spec(batch: dict[str, Any], *, ticker: str) -> dict[str, Any]:
  try:
    spec = json.loads(str(batch.get("spec_json") or "{}"))
  except json.JSONDecodeError:
    spec = {}
  if not isinstance(spec, dict):
    spec = {}
  retry_spec = dict(spec)
  _set_retry_universe(retry_spec, [ticker] if ticker else [])
  gates = retry_spec.get("gates") if isinstance(retry_spec.get("gates"), dict) else {}
  retry_spec["gates"] = {**gates, "force_rerun_existing": True}
  retry_spec["force_rerun_existing"] = True
  return retry_spec


def _seed_fixture_batch(registry: Any, *, user_id: str, host: str, pid: int | None) -> int:
  now = time.time()
  spec = {
    "fixture": "mixed",
    "source": "_fixture",
    "universe": ["MSFT", "AAPL", "TSLA"],
    "budget_usd": 0.0,
    "dev_mode": True,
  }
  batch_id = registry.acquire_batch(
    user_id=user_id,
    host=host,
    spec=spec,
    budget_usd=0.0,
    pid=pid,
    now=now,
  )
  registry.allocate_run(
    batch_id=batch_id,
    ticker="MSFT",
    stage="quality",
    skill="business-quality-assessment",
    status="completed",
    dispatch_status="reported",
    result_status="staged",
    gate_code="PROCEED",
    confidence=0.82,
    composite=0.74,
    proposal_id="fixture-proposal-msft",
    proposal_expires_at=now + 3600,
    artifact_id="fixture-artifact-msft",
    artifact_ref="fixtures/batches/msft-quality.json",
    cost_usd=0.0,
    started_at=now,
    finished_at=now + 1,
  )
  registry.allocate_run(
    batch_id=batch_id,
    ticker="AAPL",
    stage="valuation",
    skill="dcf-relative-valuation",
    status="rejected",
    dispatch_status="reported",
    result_status="noop",
    gate_code="STOP",
    confidence=0.58,
    composite=0.31,
    cost_usd=0.0,
    started_at=now + 1,
    finished_at=now + 2,
  )
  registry.allocate_run(
    batch_id=batch_id,
    ticker="TSLA",
    stage="industry",
    skill="industry-landscape",
    status="failed",
    dispatch_status="failed",
    result_status="error",
    gate_code="INSUFFICIENT_DATA",
    confidence=0.0,
    composite=0.0,
    cost_usd=0.0,
    error="fixture failure for renderer QA",
    started_at=now + 2,
    finished_at=now + 3,
  )
  registry.set_status(batch_id, "completed", now=now + 3)
  return batch_id


def _set_status_if_not_terminal(registry: Any, batch_id: int, status: str, *, error: str | None = None) -> None:
  try:
    transition = getattr(registry, "transition_status_if_current", None)
    if callable(transition):
      transition(
        batch_id,
        status,
        expected_statuses={"running", "cancelling", "remediating"},
        error=error,
      )
      return
    current_status = str(registry.get_batch_digest(batch_id).get("status") or "")
    if current_status not in {"completed", "failed", "cancelled", "budget_limited", "blocked"}:
      registry.set_status(batch_id, status, error=error)
  except Exception:
    return


async def _publish_batch_terminal_event(app_state: Any, event: dict[str, Any]) -> None:
  user_event_bus = getattr(app_state, "user_event_bus", None)
  if user_event_bus is None:
    return
  batch_id = event.get("batch_id")
  control_run_id = str(event.get("control_run_id") or f"batch_{batch_id}")
  user_id = str(event.get("user_id") or "").strip()
  if not user_id:
    return
  payload = dict(event)
  payload.setdefault("type", "run_state_changed")
  payload.setdefault("run_id", control_run_id)
  payload.setdefault("control_run_id", control_run_id)
  payload.setdefault("ts", time.time())
  await user_event_bus.publish(user_id=user_id, control_run_id=control_run_id, event=payload)


def _registry_for_user(user_id: str):
  try:
    from memory import get_workspace_dir
    from agent.batch.registry import BatchRegistry
  except ModuleNotFoundError as exc:
    if exc.name not in _MEMORY_MODULE_NAMES | _BATCH_REGISTRY_MODULE_NAMES:
      raise
    from api.memory import get_workspace_dir
    from api.agent.batch.registry import BatchRegistry

  return BatchRegistry(Path(get_workspace_dir(user_id)) / "batch_registry.db")


def _read_only_registry_for_user(user_id: str):
  """Open the already-initialized batch registry without bootstrap writes."""

  try:
    from memory import get_workspace_path
    from agent.batch.registry import BatchRegistry
  except ModuleNotFoundError as exc:
    if exc.name not in _MEMORY_MODULE_NAMES | _BATCH_REGISTRY_MODULE_NAMES:
      raise
    from api.memory import get_workspace_path
    from api.agent.batch.registry import BatchRegistry

  return BatchRegistry(
    Path(get_workspace_path(user_id)) / "batch_registry.db",
    read_only=True,
  )


def _controller():
  try:
    from agent.batch import controller
  except ModuleNotFoundError as exc:
    if exc.name not in _BATCH_CONTROLLER_MODULE_NAMES:
      raise
    from api.agent.batch import controller

  return controller


def _batch_workflow_catalog() -> dict[str, Any]:
  try:
    from agent.skills.diligence_tracks import batch_workflow_catalog
  except ModuleNotFoundError as exc:
    if exc.name not in _BATCH_WORKFLOW_MODULE_NAMES:
      raise
    from api.agent.skills.diligence_tracks import batch_workflow_catalog

  return batch_workflow_catalog()


def _active_batch_error_type():
  try:
    from agent.batch.registry import ActiveBatchError
  except ModuleNotFoundError as exc:
    if exc.name not in _BATCH_REGISTRY_MODULE_NAMES:
      raise
    from api.agent.batch.registry import ActiveBatchError

  return ActiveBatchError


def _task_registry(request: Request) -> BatchTaskRegistry:
  return _task_registry_for_state(request.app.state)


def _task_registry_for_state(app_state: Any) -> BatchTaskRegistry:
  registry = getattr(app_state, "batch_task_registry", None)
  if registry is None:
    registry = BatchTaskRegistry()
    app_state.batch_task_registry = registry
  return registry


__all__ = [
  "BatchTaskRegistry",
  "build_batches_router",
  "dispatch_batch_in_process",
  "read_batch_for_user",
]
