from __future__ import annotations

import asyncio
import inspect
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.control_plane import batches as batches_module
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from api.agent.batch.registry import BatchRegistry


API_KEY = "batch-control-key"


class FakeActiveBatchError(RuntimeError):
  def __init__(self, batch_id: int) -> None:
    self.batch_id = batch_id
    super().__init__(f"active batch already exists: {batch_id}")


class FakeBatchController:
  def __init__(self) -> None:
    self.acquire_calls: list[dict[str, Any]] = []
    self.run_calls: list[dict[str, Any]] = []
    self.raise_active: bool = False
    self.wait_for_cancel: bool = False

  def acquire_batch_run(
    self,
    spec: dict[str, Any],
    *,
    registry: BatchRegistry,
    host: str,
    pid: int | None = None,
    user_id: str | None = None,
    user_email: str | None = None,
  ) -> tuple[int, str, str | None]:
    if self.raise_active:
      raise FakeActiveBatchError(99)
    resolved_user_id = user_id or "alice"
    batch_id = registry.acquire_batch(
      user_id=resolved_user_id,
      host=host,
      spec=spec,
      budget_usd=float(spec.get("budget_usd") or spec.get("budget") or 0),
      pid=pid,
    )
    self.acquire_calls.append(
      {
        "batch_id": batch_id,
        "spec": dict(spec),
        "user_id": resolved_user_id,
        "user_email": user_email,
      }
    )
    return batch_id, resolved_user_id, user_email

  async def run_acquired_batch(
    self,
    batch_id: int,
    spec: dict[str, Any],
    *,
    registry: BatchRegistry,
    _identity: tuple[str, str | None] | None = None,
    _on_finalize=None,
  ) -> dict[str, Any]:
    self.run_calls.append({"batch_id": batch_id, "spec": dict(spec), "identity": _identity})
    if self.wait_for_cancel:
      for _ in range(50):
        if str(registry.get_batch_digest(batch_id).get("status") or "") == "cancelling":
          registry.set_status(batch_id, "cancelled")
          await self._notify_finalize(batch_id, registry, _on_finalize)
          return {"batch_id": batch_id, "status": "cancelled"}
        await asyncio.sleep(0.01)

    run_seq = registry.allocate_run(
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
      cost_usd=0.125,
      finished_at=time.time(),
    )
    assert run_seq == 1
    registry.add_spent(batch_id, 0.125)
    registry.set_status(batch_id, "completed")
    await self._notify_finalize(batch_id, registry, _on_finalize)
    return {"batch_id": batch_id, "status": "completed"}

  async def _notify_finalize(self, batch_id: int, registry: BatchRegistry, callback) -> None:
    if callback is None:
      return
    digest = registry.get_batch_digest(batch_id)
    result = callback(
      {
        "type": "run_state_changed",
        "run_id": f"batch_{batch_id}",
        "control_run_id": f"batch_{batch_id}",
        "batch_id": batch_id,
        "state": digest["status"],
        "status": digest["status"],
        "user_id": digest["user_id"],
        "cost_usd": digest["cost_usd"],
        "ts": time.time(),
      }
    )
    if inspect.isawaitable(result):
      await result


def _make_app():
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="batch-control-test-secret-0123456789",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )


@pytest.fixture
def fake_batch_control(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeBatchController:
  controller = FakeBatchController()
  registry_paths: dict[str, Path] = {}

  def registry_for_user(user_id: str) -> BatchRegistry:
    path = registry_paths.setdefault(user_id, tmp_path / f"{user_id}.db")
    return BatchRegistry(path)

  monkeypatch.setattr(batches_module, "_controller", lambda: controller)
  monkeypatch.setattr(batches_module, "_active_batch_error_type", lambda: FakeActiveBatchError)
  monkeypatch.setattr(batches_module, "_registry_for_user", registry_for_user)
  return controller


def _control_session(client: TestClient, user_id: str = "alice") -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "user_email": f"{user_id}@example.com", "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def _batch_spec(*, universe: list[str] | None = None) -> dict[str, Any]:
  return {
    "source": "quality_screen",
    "universe": universe or ["MSFT"],
    "budget_usd": 1.0,
    "max_concurrency": 2,
  }


def _wait_for_batch_status(
  client: TestClient,
  headers: dict[str, str],
  batch_id: int,
  expected: set[str],
) -> dict[str, Any]:
  last_payload: dict[str, Any] | None = None
  for _ in range(50):
    response = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    assert response.status_code == 200, response.text
    last_payload = response.json()
    status = str(last_payload["batch"]["status"])
    if status in expected:
      return last_payload
    time.sleep(0.02)
  raise AssertionError(f"batch {batch_id} never reached {expected}: {last_payload}")


def test_control_batches_dispatch_list_and_get(fake_batch_control: FakeBatchController) -> None:
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "running"
    batch_id = int(payload["batch_id"])

    detail = _wait_for_batch_status(client, headers, batch_id, {"completed"})
    batch = detail["batch"]
    assert batch["user_id"] == "alice"
    assert batch["status"] == "completed"
    assert batch["cost_usd"] == pytest.approx(0.125)
    assert batch["total_spent_usd"] == pytest.approx(0.125)
    assert detail["verdict_matrix"] == [
      {
        "ticker": "MSFT",
        "skill": "business-quality-assessment",
        "dispatch_status": "reported",
        "gate_code": "PROCEED",
        "confidence": 0.82,
        "composite": 0.74,
        "result_status": "staged",
      }
    ]

    list_response = client.get("/api/control/batches", headers=headers)
    assert list_response.status_code == 200, list_response.text
    batches = list_response.json()["batches"]
    assert [item["batch_id"] for item in batches] == [batch_id]
    assert batches[0]["cost_usd"] == pytest.approx(0.125)
    assert fake_batch_control.run_calls[0]["identity"] == ("alice", "alice@example.com")


def test_control_batches_active_batch_conflict_maps_to_409(fake_batch_control: FakeBatchController) -> None:
  fake_batch_control.raise_active = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())

  assert response.status_code == 409
  assert "active batch already exists" in response.json()["detail"]


def test_control_batches_cancel_sets_cancelling(fake_batch_control: FakeBatchController) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    assert response.status_code == 200, response.text
    batch_id = int(response.json()["batch_id"])

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)
    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["batch"]["status"] == "cancelling"

    detail = _wait_for_batch_status(client, headers, batch_id, {"cancelled"})
    assert detail["batch"]["status"] == "cancelled"


def test_batch_task_registry_cancelled_task_finishes_cancelled(tmp_path: Path) -> None:
  async def run_case() -> None:
    registry = BatchRegistry(tmp_path / "cancelled-task.db")
    batch_id = registry.acquire_batch(
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )

    async def cancelled_task() -> None:
      raise asyncio.CancelledError

    task = asyncio.create_task(cancelled_task())
    task_registry = batches_module.BatchTaskRegistry()
    await task_registry._consume(batch_id, task, registry)

    digest = registry.get_batch_digest(batch_id)
    assert digest["status"] == "cancelled"
    assert digest["error"] is None
    registry.close()

  asyncio.run(run_case())


def test_control_batches_digest_allows_budget_limited(
  fake_batch_control: FakeBatchController,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  registry_path = tmp_path / "alice.db"

  def registry_for_user(_user_id: str) -> BatchRegistry:
    return BatchRegistry(registry_path)

  monkeypatch.setattr(batches_module, "_registry_for_user", registry_for_user)
  source_registry = BatchRegistry(registry_path)
  batch_id = source_registry.acquire_batch(
    user_id="alice",
    host="test-host",
    spec=_batch_spec(universe=["AAPL", "MSFT"]),
    budget_usd=1.0,
    pid=None,
  )
  source_registry.allocate_run(
    batch_id=batch_id,
    ticker="AAPL",
    stage="quality",
    skill="business-quality-assessment",
    status="completed",
    dispatch_status="reported",
    gate_code="PROCEED",
    cost_usd=0.75,
  )
  source_registry.allocate_run(
    batch_id=batch_id,
    ticker="MSFT",
    stage="quality",
    skill="business-quality-assessment",
    status="skipped",
    dispatch_status="skipped",
    error="budget",
  )
  source_registry.add_spent(batch_id, 0.75)
  source_registry.set_status(batch_id, "budget_limited")
  source_registry.close()

  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    detail_response = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    list_response = client.get("/api/control/batches", headers=headers)

  assert detail_response.status_code == 200, detail_response.text
  detail = detail_response.json()
  assert detail["batch"]["status"] == "budget_limited"
  assert detail["batch"]["error"] is None
  assert detail["batch"]["counts_by_status"] == {"completed": 1, "skipped": 1}
  assert detail["failures"] == [{"ticker": "MSFT", "skill": "business-quality-assessment", "status": "skipped", "error": "budget"}]
  assert list_response.status_code == 200, list_response.text
  assert list_response.json()["batches"][0]["status"] == "budget_limited"


def test_control_batches_retry_failed_uses_failed_tickers_only(
  fake_batch_control: FakeBatchController,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  registry_path = tmp_path / "alice.db"

  def registry_for_user(_user_id: str) -> BatchRegistry:
    return BatchRegistry(registry_path)

  monkeypatch.setattr(batches_module, "_registry_for_user", registry_for_user)
  source_registry = BatchRegistry(registry_path)
  source_batch_id = source_registry.acquire_batch(
    user_id="alice",
    host="test-host",
    spec=_batch_spec(universe=["AAPL", "MSFT"]),
    budget_usd=1.0,
    pid=None,
  )
  source_registry.allocate_run(
    batch_id=source_batch_id,
    ticker="AAPL",
    stage="quality",
    skill="business-quality-assessment",
    status="failed",
    dispatch_status="failed",
    error="boom",
  )
  source_registry.allocate_run(
    batch_id=source_batch_id,
    ticker="MSFT",
    stage="quality",
    skill="business-quality-assessment",
    status="rejected",
    dispatch_status="reported",
    gate_code="STOP",
  )
  source_registry.set_status(source_batch_id, "completed")
  source_registry.close()

  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post(f"/api/control/batches/{source_batch_id}/retry-failed", headers=headers)

  assert response.status_code == 200, response.text
  retry_batch_id = int(response.json()["batch_id"])
  assert retry_batch_id != source_batch_id
  assert fake_batch_control.acquire_calls[-1]["spec"]["universe"] == ["AAPL"]


def test_publish_batch_terminal_event_uses_batch_control_run_id() -> None:
  published: list[dict[str, Any]] = []

  class FakeBus:
    async def publish(self, *, user_id: str, control_run_id: str, event: dict[str, Any]) -> None:
      published.append({"user_id": user_id, "control_run_id": control_run_id, "event": dict(event)})

  asyncio.run(
    batches_module._publish_batch_terminal_event(
      SimpleNamespace(user_event_bus=FakeBus()),
      {
        "type": "run_state_changed",
        "batch_id": 42,
        "state": "completed",
        "user_id": "alice",
      },
    )
  )

  assert published == [
    {
      "user_id": "alice",
      "control_run_id": "batch_42",
      "event": {
        "type": "run_state_changed",
        "batch_id": 42,
        "state": "completed",
        "user_id": "alice",
        "run_id": "batch_42",
        "control_run_id": "batch_42",
        "ts": published[0]["event"]["ts"],
      },
    }
  ]
