from __future__ import annotations

import asyncio
from copy import deepcopy
import hashlib
import json
import logging
import os
import socket
import time
import uuid
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
from agent_gateway.capability_binding import (
  AuthContext,
  CapabilityBind,
  CredentialHandle,
)
from agent_gateway.capability_execution import (
  BoundCapabilityExecution,
  CapabilityExecutionResolver,
  MaterializedCredential,
)
from agent_gateway.model_registry import (
  ProductModelRegistry,
  ProductModelSelectionPolicy,
)
from agent_gateway.claim_signing_authority import (
  GatewayClaimSigningAuthority,
)
from agent_gateway.fixture_gate import fixture_provider_available, fixture_unavailable_message
from agent_gateway.role_validation import require_exact_role
from agent_gateway.session import AuthManager, GatewaySession

from .corpus_readiness import (
  CORPUS_REQUIREMENTS_FIELD,
  CorpusReadinessGateError,
  require_corpus_readiness,
)
from .runs import _require_bearer_session, _require_control_session, _session_owner_user_id

_BATCH_CONTROLLER_MODULE_NAMES = frozenset({"agent", "agent.batch", "agent.batch.controller"})
_BATCH_FINALIZATION_MODULE_NAMES = frozenset({
  "agent",
  "agent.batch",
  "agent.batch.controller_finalization",
})
_BATCH_REGISTRY_MODULE_NAMES = frozenset({"agent", "agent.batch", "agent.batch.registry"})
_BATCH_WORKFLOW_MODULE_NAMES = frozenset({"agent", "agent.skills", "agent.skills.diligence_tracks"})
_MEMORY_MODULE_NAMES = frozenset({"memory"})
_TERMINAL_BATCH_STATUSES = frozenset({
  "completed",
  "failed",
  "cancelled",
  "budget_limited",
  "blocked",
})
_ACTIVE_BATCH_STATUSES = frozenset({"running", "cancelling", "remediating"})
_DURABLE_CORPUS_REJECTION_CODES = frozenset({
  "invalid_corpus_requirements",
  "missing_corpus_requirements",
})
log = logging.getLogger("agent_gateway.control_plane.batches")
_SERVICE_BATCH_SESSION_LIFETIME_SECONDS = 24 * 60 * 60


@dataclass
class _BatchCancellationProgress:
  boundary_crossed: bool = False


@dataclass(frozen=True)
class _BatchCancellationFence:
  projections: tuple[Any, ...]
  authorization_subjects: tuple[Any, ...]
  progress: _BatchCancellationProgress


class _BatchDispatchNotAdmitted(RuntimeError):
  def __init__(
    self,
    status_code: int,
    detail: Any,
    *,
    durable_rejection: bool = False,
  ) -> None:
    self.status_code = int(status_code)
    self.detail = detail
    self.durable_rejection = bool(durable_rejection)
    super().__init__(str(detail))


class _BatchDispatchValidationError(ValueError):
  pass


class _BatchDispatchReconciliationError(RuntimeError):
  pass


class _BatchCancellationAlreadyInProgress(RuntimeError):
  def __init__(
    self,
    *,
    boundary_crossed: bool,
  ) -> None:
    self.boundary_crossed = bool(boundary_crossed)
    super().__init__("batch cancellation is already in progress")


class BatchTaskRegistry:
  def __init__(self) -> None:
    self._tasks: dict[tuple[str, int], asyncio.Task[Any]] = {}
    self._monitors: dict[tuple[str, int], asyncio.Task[Any]] = {}
    self._cancel_admission_fences: set[tuple[str, int]] = set()
    self._cancellation_progress: dict[
      tuple[str, int],
      _BatchCancellationProgress,
    ] = {}
    self._shutdown_started = False
    self.approval_projections = BatchApprovalProjectionRegistry()

  def start(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
    task: asyncio.Task[Any],
    registry: Any,
    app_state: Any | None = None,
  ) -> None:
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
      self._consume(
        key,
        task,
        registry,
        app_state=app_state,
      )
    )

  def assert_accepting(self) -> None:
    if self._shutdown_started:
      raise RuntimeError("batch task registry is shutting down")

  def has_admitted_batch(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> bool:
    key = self._key(owner_user_id, batch_id)
    return key in self._tasks or key in self._monitors

  def begin_batch_cancellation(
    self,
    *,
    owner_user_id: str,
    batch_id: int,
  ) -> _BatchCancellationFence:
    key = self._key(owner_user_id, batch_id)
    self.assert_accepting()
    if key in self._cancel_admission_fences:
      progress = self._cancellation_progress.get(key)
      if progress is None:
        raise RuntimeError("batch cancellation progress is unavailable")
      raise _BatchCancellationAlreadyInProgress(
        boundary_crossed=progress.boundary_crossed,
      )
    progress = _BatchCancellationProgress()
    self._cancel_admission_fences.add(key)
    self._cancellation_progress[key] = progress
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
        progress=progress,
      )
    except BaseException:
      self._cancel_admission_fences.discard(key)
      self._cancellation_progress.pop(key, None)
      self.approval_projections.release_batch_fence(
        owner_user_id=key[0],
        batch_id=key[1],
        rollback=True,
      )
      raise

  def abort_batch_cancellation(self, *, owner_user_id: str, batch_id: int) -> None:
    key = self._key(owner_user_id, batch_id)
    self._cancel_admission_fences.discard(key)
    self._cancellation_progress.pop(key, None)
    self.approval_projections.release_batch_fence(
      owner_user_id=key[0],
      batch_id=key[1],
      rollback=True,
    )

  def finish_batch_cancellation(self, *, owner_user_id: str, batch_id: int) -> None:
    key = self._key(owner_user_id, batch_id)
    self._cancel_admission_fences.discard(key)
    self._cancellation_progress.pop(key, None)
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

  async def publish_terminal_event(
    self,
    *,
    app_state: Any,
    event: dict[str, Any],
  ) -> bool:
    batch_id = event.get("batch_id")
    owner_user_id = str(event.get("user_id") or "").strip()
    if (
      isinstance(batch_id, bool)
      or not isinstance(batch_id, int)
      or not owner_user_id
    ):
      raise ValueError("batch terminal event requires owner_user_id and positive batch_id")
    self._key(owner_user_id, batch_id)
    return await _publish_batch_terminal_event(app_state, event)

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
    self._cancellation_progress.clear()
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
    *,
    app_state: Any | None = None,
  ) -> None:
    owner_user_id, batch_id = key
    try:
      try:
        await task
      except asyncio.CancelledError:
        _set_status_if_not_terminal(registry, batch_id, "cancelled")
      except Exception as exc:
        _set_status_if_not_terminal(registry, batch_id, "failed", error=str(exc))
      if (
        app_state is not None
        and not self._shutdown_started
        and key not in self._cancel_admission_fences
      ):
        digest = registry.get_batch_digest(batch_id)
        if str(digest.get("status") or "") not in _TERMINAL_BATCH_STATUSES:
          _set_status_if_not_terminal(
            registry,
            batch_id,
            "failed",
            error="batch_task_exited_without_terminal_state",
          )
          digest = registry.get_batch_digest(batch_id)
        if str(digest.get("status") or "") not in _TERMINAL_BATCH_STATUSES:
          raise RuntimeError("batch task monitor did not reach a terminal state")
        event = _batch_terminal_event_payload_from_digest(
          batch_id,
          digest,
        )
        await self.publish_terminal_event(
          app_state=app_state,
          event=event,
        )
    except Exception:
      log.exception(
        "Batch task terminal publication failed batch_id=%s owner_user_id=%s",
        batch_id,
        owner_user_id,
      )
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
    return await _dispatch_batch_for_authenticated(
      request,
      payload,
      authenticated,
      dispatch_key=_required_batch_dispatch_key(request),
    )

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
        _resolver, fixture_execution = _batch_capability_execution_context(
          app_state=request.app.state,
          user_id=owner_user_id,
          authenticated_session=authenticated,
        )
        batch_id = _seed_fixture_batch(
          registry,
          user_id=owner_user_id,
          host=socket.gethostname(),
          pid=os.getpid(),
          capability_bind=fixture_execution.bind,
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
      existing_batch = _require_batch_owner(
        registry,
        batch_id,
        owner_user_id,
      )
      if str(existing_batch.get("status") or "") in _TERMINAL_BATCH_STATUSES:
        return {"batch": existing_batch}
      task_registry = _task_registry(request)
      try:
        cancellation_fence = task_registry.begin_batch_cancellation(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
        )
      except _BatchCancellationAlreadyInProgress as exc:
        raise HTTPException(
          status_code=409,
          detail=str(exc),
          headers=(
            {"X-Batch-Cancellation-Committed": "true"}
            if exc.boundary_crossed
            else None
          ),
        ) from exc
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
      cancellation_progress = cancellation_fence.progress
      completion_task = asyncio.create_task(
        _complete_authorized_batch_cancellation(
          task_registry=task_registry,
          registry=registry,
          projections=cancellation_fence.projections,
          owner_user_id=owner_user_id,
          batch_id=batch_id,
          authenticated=authenticated,
          app_state=request.app.state,
          progress=cancellation_progress,
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
        except Exception as exc:
          raise HTTPException(
            status_code=503,
            detail={"error": "Batch cancellation completion failed"},
            headers=(
              {"X-Batch-Cancellation-Committed": "true"}
              if cancellation_progress.boundary_crossed
              else None
            ),
          ) from exc
        if approval_error is not None:
          raise HTTPException(
            status_code=approval_error.status_code,
            detail=approval_error.payload,
            headers=(
              {"X-Batch-Cancellation-Committed": "true"}
              if cancellation_progress.boundary_crossed
              else None
            ),
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
      required_bind = _batch_bind_from_row(prior)
    finally:
      source_registry.close()

    dispatch_payload = dict(spec)
    return await _dispatch_batch_for_authenticated(
      request,
      dispatch_payload,
      authenticated,
      dispatch_key=f"batch-retry-failed-{batch_id}",
      required_bind=required_bind,
    )

  return router


def _batch_capability_execution_context(
  *,
  app_state: Any,
  user_id: str,
  authenticated_session: GatewaySession | None,
  required_bind: CapabilityBind | None = None,
) -> tuple[CapabilityExecutionResolver, BoundCapabilityExecution]:
  """Resolve one exact server-owned batch execution before task admission."""

  actor_id = str(user_id or "").strip()
  if not actor_id:
    raise ValueError("batch user_id is required")
  gateway_config = getattr(app_state, "gateway_config", None)
  if gateway_config is None:
    raise RuntimeError("gateway batch capability configuration is unavailable")

  selection_policy = getattr(
    gateway_config,
    "model_selection_policy",
    None,
  )
  model_registry = getattr(
    gateway_config,
    "model_registry",
    None,
  )
  if not isinstance(selection_policy, ProductModelSelectionPolicy):
    raise RuntimeError(
      "batch model-selection policy is not configured"
    )
  if not isinstance(model_registry, ProductModelRegistry):
    raise RuntimeError(
      "batch product model registry is not configured"
    )

  tenant_id = str(getattr(gateway_config, "tenant_id", None) or "").strip()
  if not tenant_id:
    raise RuntimeError("tenant_id is required for capability-bound batch")

  service_handles = dict(
    getattr(gateway_config, "service_provider_handles", {}) or {}
  )
  user_handles: dict[str, CredentialHandle] = {}
  run_scoped_user_providers: set[str] = set()
  session_handle: CredentialHandle | None = None

  if authenticated_session is not None:
    if not isinstance(authenticated_session, GatewaySession):
      raise TypeError(
        "authenticated batch dispatch requires a GatewaySession"
      )
    session_owner = _session_owner_user_id(authenticated_session)
    if session_owner != actor_id:
      raise RuntimeError(
        "authenticated batch session owner does not match the batch user"
      )
    session_tenant = str(authenticated_session.tenant_id or "").strip()
    if session_tenant and session_tenant != tenant_id:
      raise RuntimeError("batch session tenant does not match gateway tenant")
    session_handle = authenticated_session.session_credential_handle
    if (
      authenticated_session.auth_config is not None
      and session_handle is None
    ):
      raise RuntimeError(
        "authenticated batch session credential provenance is missing"
      )
    if session_handle is not None:
      if session_handle.tenant_id != tenant_id:
        raise RuntimeError(
          "batch session credential tenant does not match gateway tenant"
        )
      if session_handle.principal == "user":
        user_handles[session_handle.provider] = session_handle
        run_scoped_user_providers.add(session_handle.provider)
      else:
        service_handles[session_handle.provider] = session_handle
      if not isinstance(authenticated_session.auth_config, dict):
        raise RuntimeError(
          "selected batch session credential has no credential material"
        )

  session_driver_policy = selection_policy.capabilities["session.driver"]

  auth_context = AuthContext(
    run_mode="batch",
    actor_id=actor_id,
    tenant_id=tenant_id,
    user_provider_handles=user_handles,
    service_provider_handles=service_handles,
    entitled_capabilities=frozenset({"session.driver"}),
    entitled_model_keys=session_driver_policy.allowed_model_keys,
    run_scoped_user_providers=frozenset(run_scoped_user_providers),
  )
  service_materializer = getattr(
    gateway_config,
    "service_auth_config_resolver",
    None,
  )
  adapter_resolver = getattr(
    gateway_config,
    "capability_adapter_resolver",
    None,
  )
  if not callable(adapter_resolver):
    raise RuntimeError("batch capability adapter resolver is not configured")

  def _materialize(handle: CredentialHandle) -> MaterializedCredential:
    if session_handle is not None and handle is session_handle:
      if authenticated_session is None:
        raise RuntimeError(
          "batch session credential material is unavailable"
        )
      auth_config = authenticated_session.auth_config
      if not isinstance(auth_config, dict):
        raise RuntimeError(
          "selected batch session credential has no credential material"
        )
      return MaterializedCredential(
        handle=handle,
        auth_config=auth_config,
      )
    if handle.principal != "service":
      raise RuntimeError(
        "non-session batch user credential handles are not supported"
      )
    if service_materializer is None:
      raise RuntimeError(
        f"batch service credential materializer is missing for {handle.provider}"
      )
    materialized = service_materializer(handle)
    if not isinstance(materialized, MaterializedCredential):
      raise RuntimeError(
        "batch service credential materializer must return MaterializedCredential"
      )
    if materialized.handle is not handle:
      raise RuntimeError(
        "batch service credential materializer returned a different handle"
      )
    return materialized

  resolver = CapabilityExecutionResolver(
    registry=model_registry,
    selection_policy=selection_policy,
    auth_context=auth_context,
    credential_materializer=_materialize,
    adapter_resolver=adapter_resolver,
    trusted_channel=(
      authenticated_session.channel
      if authenticated_session is not None
      else None
    ),
  )
  execution = (
    resolver.materialize_bind(required_bind)
    if required_bind is not None
    else resolver.resolve("session.driver")
  )
  if execution.bind.run_mode != "batch":
    raise RuntimeError("batch session.driver resolved with the wrong run mode")
  execution.validate()
  return resolver, execution


def build_service_batch_session(
  *,
  user_id: str,
  user_email: str | None,
  role: str,
  tenant_id: str | None,
  channel: str | None,
  session_driver_execution: BoundCapabilityExecution,
  risk_user_id: int | None = None,
  raw_user_id: str | None = None,
  user_slug: str | None = None,
  user_aliases: tuple[str, ...] | list[str] | None = None,
  identity_status: str | None = None,
) -> GatewaySession:
  """Build the only parent session permitted for sessionless batch dispatch."""
  normalized_user_id = str(user_id or "").strip()
  if not normalized_user_id:
    raise ValueError("service batch user_id is required")
  normalized_risk_user_id = int(risk_user_id or 0)
  if normalized_risk_user_id <= 0:
    try:
      normalized_risk_user_id = int(normalized_user_id)
    except ValueError:
      normalized_risk_user_id = 0
  if normalized_risk_user_id < 0:
    normalized_risk_user_id = 0
  exact_role = require_exact_role(role)
  if not isinstance(
    session_driver_execution,
    BoundCapabilityExecution,
  ):
    raise TypeError(
      "service batch execution must be BoundCapabilityExecution"
    )
  session_driver_execution.validate()
  if session_driver_execution.bind.capability_id != "session.driver":
    raise ValueError(
      "service batch execution must bind session.driver"
    )
  auth_config = dict(session_driver_execution.auth_config)
  if (
    str(auth_config.get("provider") or "").strip()
    != session_driver_execution.bind.provider
  ):
    raise ValueError(
      "service batch auth provider does not match its execution bind"
    )
  created_at = int(time.time())
  return GatewaySession(
    session_id=f"service_batch_{uuid.uuid4().hex}",
    api_key_hash=hashlib.sha256(
      f"service-batch:{normalized_user_id}".encode("utf-8")
    ).hexdigest(),
    created_at=created_at,
    expires_at=(
      created_at + _SERVICE_BATCH_SESSION_LIFETIME_SECONDS
    ),
    user_id=normalized_user_id,
    user_email=user_email,
    risk_user_id=normalized_risk_user_id,
    role=exact_role,
    kind="control",
    auth_config=auth_config,
    tenant_id=tenant_id,
    session_credential_handle=None,
    channel=channel,
    owner_user_id=normalized_user_id,
    raw_user_id=raw_user_id,
    user_slug=user_slug,
    user_aliases=tuple(user_aliases or (normalized_user_id,)),
    identity_status=identity_status,
    dispatch_scope=None,
  )


def _batch_parent_session(
  *,
  user_id: str,
  user_email: str | None,
  role: str,
  tenant_id: str | None,
  channel: str | None,
  authenticated_session: GatewaySession | None,
  session_driver_execution: BoundCapabilityExecution,
) -> GatewaySession:
  """Bind admitted batch authority to its canonical durable owner."""

  normalized_user_id = str(user_id or "").strip()
  if not normalized_user_id:
    raise ValueError("batch parent user_id is required")
  if (
    authenticated_session is not None
    and authenticated_session.user_id == normalized_user_id
  ):
    return authenticated_session
  return build_service_batch_session(
    user_id=normalized_user_id,
    user_email=user_email,
    role=require_exact_role(role),
    tenant_id=tenant_id,
    channel=channel,
    session_driver_execution=session_driver_execution,
    risk_user_id=(
      authenticated_session.risk_user_id
      if authenticated_session is not None
      else None
    ),
    raw_user_id=(
      authenticated_session.raw_user_id
      if authenticated_session is not None
      else None
    ),
    user_slug=(
      authenticated_session.user_slug
      if authenticated_session is not None
      else None
    ),
    user_aliases=(
      authenticated_session.user_aliases
      if authenticated_session is not None
      else None
    ),
    identity_status=(
      authenticated_session.identity_status
      if authenticated_session is not None
      else None
    ),
  )


async def _dispatch_batch_for_authenticated(
  request: Request,
  payload: dict[str, Any],
  authenticated: Any,
  *,
  dispatch_key: str,
  required_bind: CapabilityBind | None = None,
) -> dict[str, Any]:
  try:
    return await dispatch_batch_in_process(
      payload,
      app_state=request.app.state,
      user_id=_session_owner_user_id(authenticated),
      user_email=authenticated.user_email,
      role=require_exact_role(getattr(authenticated, "role", None)),
      channel=getattr(authenticated, "channel", None),
      authenticated_session=authenticated,
      dispatch_key=dispatch_key,
      required_bind=required_bind,
    )
  except _active_batch_error_type() as exc:
    raise HTTPException(
      status_code=409,
      detail=str(exc),
    ) from exc
  except _BatchDispatchNotAdmitted as exc:
    raise HTTPException(
      status_code=exc.status_code,
      detail=exc.detail,
      headers=(
        {"X-Batch-Dispatch-Admitted": "false"}
        if exc.durable_rejection
        else None
      ),
    ) from exc
  except _BatchDispatchReconciliationError as exc:
    raise HTTPException(status_code=503, detail=str(exc)) from exc
  except CorpusReadinessGateError as exc:
    raise HTTPException(
      status_code=exc.status_code,
      detail=exc.to_payload(),
    ) from exc
  except RuntimeError as exc:
    raise HTTPException(status_code=409, detail=str(exc)) from exc
  except ValueError as exc:
    raise HTTPException(status_code=422, detail=str(exc)) from exc


async def dispatch_batch_in_process(
  payload: dict[str, Any],
  *,
  app_state: Any,
  user_id: str,
  dispatch_key: str,
  user_email: str | None = None,
  role: str | None = None,
  channel: str | None = None,
  authenticated_session: GatewaySession | None = None,
  required_bind: CapabilityBind | None = None,
) -> dict[str, Any]:
  if not isinstance(payload, dict):
    raise ValueError("batch spec must be an object")
  if not isinstance(dispatch_key, str) or not dispatch_key.strip():
    raise ValueError("batch dispatch requires a non-empty dispatch_key")
  exact_role = (
    require_exact_role(getattr(authenticated_session, "role", None))
    if authenticated_session is not None
    else require_exact_role(role)
  )
  if role is not None and require_exact_role(role) != exact_role:
    raise ValueError("batch dispatch role does not match authenticated session")
  if required_bind is not None:
    if not isinstance(required_bind, CapabilityBind):
      raise TypeError("required_bind must be CapabilityBind")
    if required_bind.capability_id != "session.driver":
      raise ValueError("batch retry requires a session.driver bind")
    if required_bind.run_mode != "batch":
      raise ValueError("batch retry requires a batch-mode bind")
  dispatch_request_spec = deepcopy(payload)
  replay_registry = _registry_for_user(user_id)
  try:
    task_registry = _task_registry_for_state(app_state)
    replay = replay_registry.lookup_batch_dispatch(
      user_id=user_id,
      dispatch_key=dispatch_key,
      request_spec=dispatch_request_spec,
    )
    if replay is not None:
      if isinstance(replay, _batch_dispatch_rejection_record_type()):
        raise _BatchDispatchNotAdmitted(
          replay.status_code,
          replay.detail,
          durable_rejection=True,
        )
      return await _reconcile_batch_dispatch_replay(
        registry=replay_registry,
        task_registry=task_registry,
        app_state=app_state,
        user_id=user_id,
        batch_id=replay.batch_id,
      )
    await _reconcile_active_batches_before_fresh_admission(
      registry=replay_registry,
      task_registry=task_registry,
      app_state=app_state,
      user_id=user_id,
    )
  finally:
    replay_registry.close()
  gateway_config = getattr(app_state, "gateway_config", None)
  if (
    authenticated_session is not None
    and getattr(gateway_config, "credentials_resolver", None) is not None
    and not _has_active_credential(
      _authenticated_batch_auth_config(authenticated_session)
    )
  ):
    raise _BatchDispatchNotAdmitted(
      503,
      "authenticated batch credential unavailable",
    )
  try:
    (
      capability_execution_resolver,
      session_driver_execution,
    ) = _batch_capability_execution_context(
      app_state=app_state,
      user_id=user_id,
      authenticated_session=authenticated_session,
      required_bind=required_bind,
    )
  except ValueError as exc:
    raise _BatchDispatchNotAdmitted(422, str(exc)) from exc
  except RuntimeError as exc:
    raise _BatchDispatchNotAdmitted(503, str(exc)) from exc
  try:
    payload, corpus_readiness = await require_corpus_readiness(
      payload,
      app_state=app_state,
    )
  except CorpusReadinessGateError as exc:
    if (
      exc.status_code != 422
      or exc.code not in _DURABLE_CORPUS_REJECTION_CODES
    ):
      raise
    rejection_registry = _registry_for_user(user_id)
    try:
      return await _record_or_reconcile_batch_dispatch_rejection(
        registry=rejection_registry,
        task_registry=_task_registry_for_state(app_state),
        app_state=app_state,
        user_id=user_id,
        dispatch_key=dispatch_key,
        request_spec=dispatch_request_spec,
        status_code=exc.status_code,
        detail=exc.to_payload(),
      )
    finally:
      rejection_registry.close()
  registry = _registry_for_user(user_id)
  try:
    task_registry = _task_registry_for_state(app_state)
    try:
      task_registry.assert_accepting()
    except RuntimeError as exc:
      raise _BatchDispatchNotAdmitted(409, str(exc)) from exc
    try:
      batch_id = _acquire_and_start_batch(
        payload,
        app_state=app_state,
        registry=registry,
        task_registry=task_registry,
        dispatch_key=dispatch_key,
        dispatch_request_spec=dispatch_request_spec,
        user_id=user_id,
        user_email=user_email,
        role=exact_role,
        channel=channel,
        authenticated_session=authenticated_session,
        capability_execution_resolver=capability_execution_resolver,
        session_driver_execution=session_driver_execution,
      )
    except _batch_dispatch_replay_type() as exc:
      response = await _reconcile_batch_dispatch_replay(
        registry=registry,
        task_registry=task_registry,
        app_state=app_state,
        user_id=user_id,
        batch_id=int(exc.batch_id),
      )
      registry.close()
      return response
    except _batch_dispatch_rejected_type() as exc:
      raise _BatchDispatchNotAdmitted(
        exc.status_code,
        exc.detail,
        durable_rejection=True,
      ) from exc
    except _BatchDispatchValidationError as exc:
      response = await _record_or_reconcile_batch_dispatch_rejection(
        registry=registry,
        task_registry=task_registry,
        app_state=app_state,
        user_id=user_id,
        dispatch_key=dispatch_key,
        request_spec=dispatch_request_spec,
        status_code=422,
        detail=str(exc),
      )
      registry.close()
      return response
    response: dict[str, Any] = {
      "batch_id": batch_id,
      "status": "running",
      "replayed": False,
    }
    if corpus_readiness is not None:
      response["corpus_readiness"] = corpus_readiness
    return response
  except Exception:
    registry.close()
    raise


async def _record_or_reconcile_batch_dispatch_rejection(
  *,
  registry: Any,
  task_registry: BatchTaskRegistry,
  app_state: Any,
  user_id: str,
  dispatch_key: str,
  request_spec: dict[str, Any],
  status_code: int,
  detail: Any,
) -> dict[str, Any]:
  outcome = registry.record_batch_dispatch_rejection(
    user_id=user_id,
    dispatch_key=dispatch_key,
    request_spec=request_spec,
    status_code=status_code,
    detail=detail,
  )
  if isinstance(outcome, _batch_dispatch_rejection_record_type()):
    raise _BatchDispatchNotAdmitted(
      outcome.status_code,
      outcome.detail,
      durable_rejection=True,
    )
  return await _reconcile_batch_dispatch_replay(
    registry=registry,
    task_registry=task_registry,
    app_state=app_state,
    user_id=user_id,
    batch_id=outcome.batch_id,
  )


async def _reconcile_active_batches_before_fresh_admission(
  *,
  registry: Any,
  task_registry: BatchTaskRegistry,
  app_state: Any,
  user_id: str,
) -> None:
  for active in registry.list_active_batches(user_id):
    batch_id = int(active["batch_id"])
    if task_registry.has_admitted_batch(
      owner_user_id=user_id,
      batch_id=batch_id,
    ):
      continue
    _set_status_if_not_terminal(
      registry,
      batch_id,
      "failed",
      error="batch_dispatch_orphaned_before_fresh_admission",
    )
    digest = registry.get_batch_digest(batch_id)
    if str(digest.get("status") or "") not in _TERMINAL_BATCH_STATUSES:
      raise _BatchDispatchReconciliationError(
        "active orphan could not be terminalized before fresh admission"
      )
    try:
      await task_registry.publish_terminal_event(
        app_state=app_state,
        event=_batch_terminal_event_payload_from_digest(
          batch_id,
          digest,
        ),
      )
    except Exception:
      log.exception(
        "Fresh-admission orphan terminal publication failed batch_id=%s",
        batch_id,
      )


async def _reconcile_batch_dispatch_replay(
  *,
  registry: Any,
  task_registry: BatchTaskRegistry,
  app_state: Any,
  user_id: str,
  batch_id: int,
) -> dict[str, Any]:
  digest = registry.get_batch_digest(batch_id)
  if (
    str(digest.get("status") or "") in _ACTIVE_BATCH_STATUSES
    and not task_registry.has_admitted_batch(
      owner_user_id=user_id,
      batch_id=batch_id,
    )
  ):
    _set_status_if_not_terminal(
      registry,
      batch_id,
      "failed",
      error="batch_dispatch_replay_orphaned_after_gateway_restart",
    )
    digest = registry.get_batch_digest(batch_id)
    if str(digest.get("status") or "") not in _TERMINAL_BATCH_STATUSES:
      raise _BatchDispatchReconciliationError(
        "orphaned batch replay could not be terminalized"
      )
    event = _batch_terminal_event_payload_from_digest(
      batch_id,
      digest,
    )
    try:
      await task_registry.publish_terminal_event(
        app_state=app_state,
        event=event,
      )
    except Exception:
      log.exception(
        "Orphaned replay terminal publication failed batch_id=%s",
        batch_id,
      )
  return {
    "batch_id": batch_id,
    "status": str(digest.get("status") or ""),
    "replayed": True,
  }


def _acquire_and_start_batch(
  payload: dict[str, Any],
  *,
  app_state: Any,
  registry: Any,
  task_registry: BatchTaskRegistry,
  dispatch_key: str,
  dispatch_request_spec: dict[str, Any],
  user_id: str,
  user_email: str | None,
  role: str,
  channel: str | None,
  authenticated_session: GatewaySession | None,
  capability_execution_resolver: Any,
  session_driver_execution: Any,
) -> int:
  """Atomically bridge durable admission to process-local task ownership.

  This is intentionally synchronous: gateway workers use one event loop per
  process, so no same-process replay can observe the admitted row before
  ``task_registry.start`` records its task.
  """

  parent_session = _batch_parent_session(
    user_id=user_id,
    user_email=user_email,
    role=role,
    tenant_id=str(
      getattr(app_state.gateway_config, "tenant_id", None) or ""
    ) or None,
    channel=channel,
    authenticated_session=authenticated_session,
    session_driver_execution=session_driver_execution,
  )
  if type(parent_session) is not GatewaySession:
    raise TypeError("batch admission parent must be exact GatewaySession")
  claim_signing_authority = getattr(
    app_state,
    "gateway_claim_signing_authority",
    None,
  )
  if type(claim_signing_authority) is not GatewayClaimSigningAuthority:
    raise RuntimeError(
      "batch admission claim-signing authority is unavailable"
    )
  claim_signer = claim_signing_authority.bind_user(
    user_id=parent_session.user_id,
    user_email=parent_session.user_email,
  )
  storage_root = getattr(app_state, "autonomous_storage_root", None)
  if not isinstance(storage_root, Path):
    raise RuntimeError("batch autonomous storage root is unavailable")

  try:
    batch_id, _user_id, _user_email = _controller().acquire_batch_run(
      payload,
      registry=registry,
      host=socket.gethostname(),
      pid=os.getpid(),
      dispatch_key=dispatch_key,
      dispatch_request_spec=dispatch_request_spec,
      capability_bind=session_driver_execution.bind.receipt(),
      user_id=user_id,
      user_email=user_email,
    )
  except _batch_dispatch_replay_type():
    raise
  except _batch_dispatch_rejected_type():
    raise
  except (TypeError, ValueError) as exc:
    raise _BatchDispatchValidationError(str(exc)) from exc
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

  run_id = f"batch_{batch_id}"

  def _create_batch_task() -> asyncio.Task[Any]:
    async def _captured_run_admission_factory(
      *,
      task_id: str,
      session_driver_execution: BoundCapabilityExecution,
    ) -> Any:
      return await _controller().admit_in_process_runtime_authority(
        parent_session=parent_session,
        origin="batch" if scope is not None else "service",
        run_id=run_id,
        task_id=task_id,
        tool_call_id=None,
      session_driver_execution=session_driver_execution,
        claim_signer=claim_signer,
        storage_root=storage_root,
      )

    return asyncio.create_task(
      _run_acquired_batch_containing_system_exit(
        _controller(),
        batch_id,
        payload,
        registry=registry,
        _identity=(user_id, user_email),
        _on_finalize=None,
        capability_execution_resolver=capability_execution_resolver,
        session_driver_execution=session_driver_execution,
        captured_run_admission_factory=(
          _captured_run_admission_factory
        ),
      )
    )

  if scope is None:
    task = _create_batch_task()
  else:
    # asyncio tasks copy their creation context. Bind only for construction so
    # nested stages inherit this batch's approval scope without process globals.
    with bind_batch_approval_scope(scope):
      task = _create_batch_task()
  task_registry.start(
    owner_user_id=user_id,
    batch_id=batch_id,
    task=task,
    registry=registry,
    app_state=app_state,
  )
  return batch_id


async def _run_acquired_batch_containing_system_exit(
  controller: Any,
  batch_id: int,
  payload: dict[str, Any],
  **kwargs: Any,
) -> Any:
  try:
    return await controller.run_acquired_batch(batch_id, payload, **kwargs)
  except SystemExit as exc:
    log.error(
      "Batch %s exited during in-process execution: %s",
      batch_id,
      exc,
    )
    return None


async def _complete_authorized_batch_cancellation(
  *,
  task_registry: BatchTaskRegistry,
  registry: Any,
  projections: tuple[Any, ...],
  owner_user_id: str,
  batch_id: int,
  authenticated: Any,
  app_state: Any,
  progress: _BatchCancellationProgress,
) -> tuple[ApprovalActionError | None, dict[str, Any]]:
  """Finish quarantine and task teardown after the mutation boundary."""
  approval_error: ApprovalActionError | None = None
  task = task_registry.get(owner_user_id=owner_user_id, batch_id=batch_id)
  if task is not None and not task.done():
    transitioned = registry.transition_status_if_current(
      batch_id,
      "cancelling",
      expected_statuses={"running", "remediating"},
    )
    if transitioned is True:
      progress.boundary_crossed = True
  cancelled_task = await task_registry.cancel(
    owner_user_id=owner_user_id,
    batch_id=batch_id,
  )
  if cancelled_task:
    progress.boundary_crossed = True
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
    if projections:
      progress.boundary_crossed = True
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
    try:
      late_projections = task_registry.approval_projections.merge_projection_sets(
        late_seed,
        task_registry.approval_projections.snapshot_fenced_batch(
          owner_user_id=owner_user_id,
          batch_id=batch_id,
        ),
      )
    except asyncio.CancelledError:
      if approval_error is None:
        approval_error = ApprovalActionError(
          503,
          {"error": "Late batch approval quarantine was interrupted"},
        )
      break
    except Exception:
      log.exception(
        "Late batch approval quarantine snapshot failed for batch %s",
        batch_id,
      )
      if approval_error is None:
        approval_error = ApprovalActionError(
          503,
          {"error": "Late batch approval quarantine failed"},
        )
      break
    late_seed = ()
    if not late_projections:
      late_passes += 1
      if late_passes < 2:
        await asyncio.sleep(0)
      continue
    late_passes = 0
    try:
      progress.boundary_crossed = True
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
  try:
    task_registry.approval_projections.close_batch(
      owner_user_id=owner_user_id,
      batch_id=batch_id,
    )
  except asyncio.CancelledError:
    if approval_error is None:
      approval_error = ApprovalActionError(
        503,
        {"error": "Batch approval quarantine close was interrupted"},
      )
  except Exception:
    log.exception(
      "Batch approval quarantine close failed for batch %s",
      batch_id,
    )
    if approval_error is None:
      approval_error = ApprovalActionError(
        503,
        {"error": "Batch approval quarantine close failed"},
      )
  batch_digest = registry.get_batch_digest(batch_id)
  if str(batch_digest.get("status") or "") not in _TERMINAL_BATCH_STATUSES:
    transition = getattr(registry, "transition_status_if_current", None)
    if callable(transition):
      transitioned = transition(
        batch_id,
        "cancelled",
        expected_statuses={"running", "cancelling", "remediating"},
      )
      if transitioned is True:
        progress.boundary_crossed = True
    else:
      registry.set_status(batch_id, "cancelled")
    batch_digest = registry.get_batch_digest(batch_id)
    if str(batch_digest.get("status") or "") in _TERMINAL_BATCH_STATUSES:
      progress.boundary_crossed = True
  else:
    progress.boundary_crossed = True
  if str(batch_digest.get("status") or "") not in _TERMINAL_BATCH_STATUSES:
    raise RuntimeError("batch cancellation did not reach a terminal state")
  terminal_event = _batch_terminal_event_payload_from_digest(
    batch_id,
    batch_digest,
  )
  await task_registry.publish_terminal_event(
    app_state=app_state,
    event=terminal_event,
  )
  return approval_error, batch_digest


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


def _batch_bind_from_row(batch_row: dict[str, Any]) -> CapabilityBind:
  """Require the original complete execution bind for same-admission retry."""

  raw_bind = batch_row.get("capability_bind")
  if raw_bind is None:
    raise HTTPException(
      status_code=409,
      detail="prior batch has no durable capability bind; retry is blocked",
    )
  try:
    return CapabilityBind.from_receipt(raw_bind)
  except (TypeError, ValueError) as exc:
    raise HTTPException(
      status_code=409,
      detail="prior batch durable capability bind is invalid; retry is blocked",
    ) from exc


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
    "proposals": registry.get_batch_proposals(batch_id),
    "failures": _annotate_batch_failures(registry.get_batch_failures(batch_id), batch=batch),
  }


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


def _seed_fixture_batch(
  registry: Any,
  *,
  user_id: str,
  host: str,
  pid: int | None,
  capability_bind: CapabilityBind,
) -> int:
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
    capability_bind=capability_bind.receipt(),
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
    if current_status not in _TERMINAL_BATCH_STATUSES:
      registry.set_status(batch_id, status, error=error)
  except Exception:
    return


async def _publish_batch_terminal_event(app_state: Any, event: dict[str, Any]) -> bool:
  user_event_bus = getattr(app_state, "user_event_bus", None)
  if user_event_bus is None:
    raise RuntimeError("batch terminal event bus authority is unavailable")
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
  publish_terminal = getattr(
    user_event_bus,
    "publish_terminal_if_absent",
    None,
  )
  if not callable(publish_terminal):
    raise RuntimeError("batch terminal event bus authority is unavailable")
  return await publish_terminal(
    user_id,
    control_run_id,
    payload,
  )


def terminal_batch_event_for_user(
  run_id: str,
  *,
  user_id: str,
) -> dict[str, Any] | None:
  if not run_id.startswith("batch_"):
    return None
  raw_batch_id = run_id.removeprefix("batch_")
  if not raw_batch_id.isdigit() or str(int(raw_batch_id)) != raw_batch_id:
    raise HTTPException(status_code=404, detail="Run not found")
  batch_id = int(raw_batch_id)
  if batch_id < 1:
    raise HTTPException(status_code=404, detail="Run not found")
  detail = read_batch_for_user(batch_id, user_id=user_id)
  batch = detail.get("batch")
  if not isinstance(batch, dict):
    raise HTTPException(status_code=503, detail="Batch event authority unavailable")
  status = str(batch.get("status") or "").strip().lower()
  if status not in _TERMINAL_BATCH_STATUSES:
    return None
  return _batch_terminal_event_payload_from_digest(
    batch_id,
    batch,
  )


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


def _batch_terminal_event_payload_from_digest(
  batch_id: int,
  digest: dict[str, Any],
) -> dict[str, Any]:
  try:
    from agent.batch.controller_finalization import (
      batch_terminal_event_payload_from_digest,
    )
  except ModuleNotFoundError as exc:
    if exc.name not in _BATCH_FINALIZATION_MODULE_NAMES:
      raise
    from api.agent.batch.controller_finalization import (
      batch_terminal_event_payload_from_digest,
    )

  return batch_terminal_event_payload_from_digest(batch_id, digest)


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


def _batch_dispatch_replay_type():
  try:
    from agent.batch.registry import BatchDispatchReplay
  except ModuleNotFoundError as exc:
    if exc.name not in _BATCH_REGISTRY_MODULE_NAMES:
      raise
    from api.agent.batch.registry import BatchDispatchReplay

  return BatchDispatchReplay


def _batch_dispatch_rejected_type():
  try:
    from agent.batch.registry import BatchDispatchRejected
  except ModuleNotFoundError as exc:
    if exc.name not in _BATCH_REGISTRY_MODULE_NAMES:
      raise
    from api.agent.batch.registry import BatchDispatchRejected

  return BatchDispatchRejected


def _batch_dispatch_rejection_record_type():
  try:
    from agent.batch.registry import BatchDispatchRejection
  except ModuleNotFoundError as exc:
    if exc.name not in _BATCH_REGISTRY_MODULE_NAMES:
      raise
    from api.agent.batch.registry import BatchDispatchRejection

  return BatchDispatchRejection


def _required_batch_dispatch_key(request: Request) -> str:
  raw_key = request.headers.get("Idempotency-Key")
  if raw_key is None:
    raise HTTPException(
      status_code=400,
      detail="Idempotency-Key is required for batch dispatch",
    )
  if (
    not raw_key
    or raw_key != raw_key.strip()
    or len(raw_key) > 128
    or any(ord(character) < 33 or ord(character) > 126 for character in raw_key)
  ):
    raise HTTPException(
      status_code=422,
      detail="Idempotency-Key must contain 1-128 visible ASCII characters",
    )
  return raw_key


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
