from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from agent_gateway.fixture_gate import fixture_provider_available, fixture_unavailable_message
from agent_gateway.session import AuthManager

from .runs import _require_bearer_session, _require_control_session

_BATCH_CONTROLLER_MODULE_NAMES = frozenset({"agent", "agent.batch", "agent.batch.controller"})
_BATCH_REGISTRY_MODULE_NAMES = frozenset({"agent", "agent.batch", "agent.batch.registry"})
_BATCH_WORKFLOW_MODULE_NAMES = frozenset({"agent", "agent.skills", "agent.skills.diligence_tracks"})
_MEMORY_MODULE_NAMES = frozenset({"memory"})


class BatchTaskRegistry:
  def __init__(self) -> None:
    self._tasks: dict[int, asyncio.Task[Any]] = {}

  def start(self, batch_id: int, task: asyncio.Task[Any], registry: Any) -> None:
    self._tasks[batch_id] = task
    task.add_done_callback(lambda finished: asyncio.create_task(self._consume(batch_id, finished, registry)))

  def get(self, batch_id: int) -> asyncio.Task[Any] | None:
    return self._tasks.get(batch_id)

  async def shutdown(self) -> None:
    tasks = list(self._tasks.values())
    for task in tasks:
      if not task.done():
        task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)
    self._tasks.clear()

  async def _consume(self, batch_id: int, task: asyncio.Task[Any], registry: Any) -> None:
    try:
      try:
        await task
      except asyncio.CancelledError:
        _set_status_if_not_terminal(registry, batch_id, "cancelled")
      except Exception as exc:
        _set_status_if_not_terminal(registry, batch_id, "failed", error=str(exc))
    finally:
      self._tasks.pop(batch_id, None)
      close = getattr(registry, "close", None)
      if callable(close):
        close()


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
    registry = _registry_for_user(authenticated.user_id)
    try:
      rows = registry.list_batches(authenticated.user_id, limit=limit)
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

    registry = _registry_for_user(authenticated.user_id)
    try:
      try:
        batch_id = _seed_fixture_batch(
          registry,
          user_id=authenticated.user_id,
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
    return read_batch_for_user(batch_id, user_id=authenticated.user_id, top_n=top_n)

  @router.delete("/{batch_id}")
  async def cancel_batch(request: Request, batch_id: int) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    registry = _registry_for_user(authenticated.user_id)
    try:
      _require_batch_owner(registry, batch_id, authenticated.user_id)
      registry.set_status(batch_id, "cancelling")
      return {"batch": registry.get_batch_digest(batch_id)}
    finally:
      registry.close()

  @router.post("/{batch_id}/retry-failed")
  async def retry_failed_batch(request: Request, batch_id: int) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    source_registry = _registry_for_user(authenticated.user_id)
    try:
      prior = _require_batch_owner(source_registry, batch_id, authenticated.user_id)
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
  try:
    return await dispatch_batch_in_process(
      payload,
      app_state=request.app.state,
      user_id=authenticated.user_id,
      user_email=authenticated.user_email,
    )
  except _active_batch_error_type() as exc:
    raise HTTPException(status_code=409, detail=str(exc)) from exc
  except ValueError as exc:
    raise HTTPException(status_code=422, detail=str(exc)) from exc


async def dispatch_batch_in_process(
  payload: dict[str, Any],
  *,
  app_state: Any,
  user_id: str,
  user_email: str | None = None,
) -> dict[str, Any]:
  if not isinstance(payload, dict):
    raise ValueError("batch spec must be an object")
  registry = _registry_for_user(user_id)
  try:
    batch_id, _user_id, _user_email = _controller().acquire_batch_run(
      payload,
      registry=registry,
      host=socket.gethostname(),
      pid=os.getpid(),
      user_id=user_id,
      user_email=user_email,
    )
    task = asyncio.create_task(
      _controller().run_acquired_batch(
        batch_id,
        payload,
        registry=registry,
        _identity=(user_id, user_email),
        _on_finalize=lambda event: _publish_batch_terminal_event(app_state, event),
      )
    )
    _task_registry_for_state(app_state).start(batch_id, task, registry)
    return {"batch_id": batch_id, "status": "running"}
  except Exception:
    registry.close()
    raise


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
  spec["universe"] = tickers
  return spec


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
  retry_spec["universe"] = [ticker] if ticker else []
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
