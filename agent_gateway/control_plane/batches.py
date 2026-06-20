from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from agent_gateway.session import AuthManager

from .runs import _require_bearer_session, _require_control_session


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
    if not isinstance(payload, dict):
      raise HTTPException(status_code=422, detail="batch spec must be an object")
    registry = _registry_for_user(authenticated.user_id)
    try:
      batch_id, _user_id, _user_email = _controller().acquire_batch_run(
        payload,
        registry=registry,
        host=socket.gethostname(),
        pid=os.getpid(),
        user_id=authenticated.user_id,
        user_email=authenticated.user_email,
      )
    except _active_batch_error_type() as exc:
      registry.close()
      raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
      registry.close()
      raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
      registry.close()
      raise

    task = asyncio.create_task(
      _controller().run_acquired_batch(
        batch_id,
        payload,
        registry=registry,
        _identity=(authenticated.user_id, authenticated.user_email),
        _on_finalize=lambda event: _publish_batch_terminal_event(request.app.state, event),
      )
    )
    _task_registry(request).start(batch_id, task, registry)
    return {"batch_id": batch_id, "status": "running"}

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

  @router.get("/{batch_id}")
  async def get_batch(
    request: Request,
    batch_id: int,
    top_n: int = Query(default=10, ge=1, le=100),
  ) -> dict[str, Any]:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    registry = _registry_for_user(authenticated.user_id)
    try:
      _require_batch_owner(registry, batch_id, authenticated.user_id)
      return {
        "batch": registry.get_batch_digest(batch_id),
        "verdict_matrix": registry.get_batch_verdict_matrix(batch_id),
        "candidates": registry.get_batch_candidates(batch_id, top_n),
        "failures": registry.get_batch_failures(batch_id),
      }
    finally:
      registry.close()

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
    return await dispatch_batch(request, dispatch_payload)

  return router


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
  except ModuleNotFoundError:
    from api.memory import get_workspace_dir
    from api.agent.batch.registry import BatchRegistry

  return BatchRegistry(Path(get_workspace_dir(user_id)) / "batch_registry.db")


def _controller():
  try:
    from agent.batch import controller
  except ModuleNotFoundError:
    from api.agent.batch import controller

  return controller


def _active_batch_error_type():
  try:
    from agent.batch.registry import ActiveBatchError
  except ModuleNotFoundError:
    from api.agent.batch.registry import ActiveBatchError

  return ActiveBatchError


def _task_registry(request: Request) -> BatchTaskRegistry:
  registry = getattr(request.app.state, "batch_task_registry", None)
  if registry is None:
    registry = BatchTaskRegistry()
    request.app.state.batch_task_registry = registry
  return registry


__all__ = ["BatchTaskRegistry", "build_batches_router"]
