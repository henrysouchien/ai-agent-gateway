from __future__ import annotations

# ruff: noqa: E402

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import inspect
import json
import sys
import threading
import time
import uuid
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

from agent_gateway import EventLog, ToolDispatcher
from agent_gateway import approvals as approvals_module
from agent_gateway.approval_policy import (
  ApprovalDecision as PolicyApprovalDecision,
  ApprovalRequest,
  RunContext,
  utc_now,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.batch_approval_projection import (
  BatchApprovalScope,
  current_batch_approval_scope,
)
from agent_gateway.capability_binding import (
  CredentialHandle,
)
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.control_plane import batches as batches_module
from agent_gateway.control_plane.valuation_ready_tools import make_valuation_ready_skill_tool_bundle
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.claim_signing_authority import (
  GatewayClaimSigningAuthority,
)
from agent_gateway.session import GatewaySession
from agent_gateway.skill_context import clear_current_skill, reset_current_skill, set_current_skill
from api.agent.batch.registry import (
  ActiveBatchError,
  BatchDispatchRecord,
  BatchRegistry,
)
from tests.capability_execution_test_support import (
  stub_capability_execution_resolver,
)


API_KEY = "batch-control-key"
_BATCH_CAPABILITY_RESOLVER = stub_capability_execution_resolver(
  default_provider="anthropic",
  default_model="claude-sonnet-4-6",
  default_effort="none",
  run_mode="batch",
)
_MODEL_REGISTRY = _BATCH_CAPABILITY_RESOLVER.registry
_MODEL_SELECTION_POLICY = _BATCH_CAPABILITY_RESOLVER.selection_policy


class _BatchTestProvider(ModelProvider):
  def __init__(self, name: str = "anthropic") -> None:
    self.name = name

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=16_000,
      supports_thinking=False,
    )


_BATCH_TEST_PROVIDER = _BatchTestProvider("anthropic")
_OPENAI_BATCH_TEST_PROVIDER = _BatchTestProvider("openai")
_SERVICE_HANDLE = CredentialHandle(
  handle_id="control-batches-service-anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="test-product",
  actor_id=None,
)
_SERVICE_MATERIAL = MaterializedCredential(
  handle=_SERVICE_HANDLE,
  auth_config={
    "provider": "anthropic",
    "api_key": "control-batches-service-secret",
  },
)


def _service_materializer(handle: CredentialHandle) -> MaterializedCredential:
  if handle is not _SERVICE_HANDLE:
    raise RuntimeError("unexpected control-batches credential handle")
  return _SERVICE_MATERIAL


def _adapter_resolver(adapter: str) -> ModelProvider:
  if adapter == "test.anthropic":
    return _BATCH_TEST_PROVIDER
  if adapter == "test.openai":
    return _OPENAI_BATCH_TEST_PROVIDER
  raise RuntimeError(f"unexpected control-batches adapter: {adapter}")


def _gateway_config(*, mcp_client: Any = None) -> SimpleNamespace:
  return SimpleNamespace(
    tenant_id="test-product",
    model_registry=_MODEL_REGISTRY,
    model_selection_policy=_MODEL_SELECTION_POLICY,
    service_provider_handles={"anthropic": _SERVICE_HANDLE},
    service_auth_config_resolver=_service_materializer,
    capability_adapter_resolver=_adapter_resolver,
    default_provider=_BATCH_TEST_PROVIDER,
    mcp_client=mcp_client,
  )


def _gateway_session(
  *,
  owner_user_id: str = "1",
  user_id: str = "alice",
  user_email: str = "alice@example.com",
  channel: str = "tui",
) -> GatewaySession:
  now = int(time.time())
  session = GatewaySession(
    session_id=f"control-batches-{uuid.uuid4().hex}",
    api_key_hash="control-batches-hash",
    created_at=now,
    expires_at=now + 60,
    user_id=user_id,
    owner_user_id=owner_user_id,
    user_email=user_email,
    risk_user_id=int(owner_user_id),
    kind="chat",
    channel=channel,
    auth_config={
      "provider": "anthropic",
      "api_key": "control-batches-session-secret",
    },
    tenant_id="test-product",
    session_credential_handle=_SERVICE_HANDLE,
  )
  return session


def test_batch_admission_remains_fail_closed_without_claim_authority() -> None:
  session = _gateway_session()
  app_state = SimpleNamespace(
    gateway_config=_gateway_config(),
    gateway_claim_signing_authority=None,
  )

  with pytest.raises(
    RuntimeError,
    match="batch admission claim-signing authority is unavailable",
  ):
    batches_module._acquire_and_start_batch(
      {},
      app_state=app_state,
      registry=object(),
      task_registry=batches_module.BatchTaskRegistry(),
      dispatch_key="missing-authority",
      dispatch_request_spec={},
      user_id=session.user_id,
      user_email=session.user_email,
      role=session.role,
      channel=session.channel,
      authenticated_session=session,
      capability_execution_resolver=object(),
      session_driver_execution=object(),
    )


def _run(coro):
  return asyncio.run(coro)


class FakeActiveBatchError(RuntimeError):
  def __init__(self, batch_id: int) -> None:
    self.batch_id = batch_id
    super().__init__(f"active batch already exists: {batch_id}")


class ReadyCorpusMcpClient:
  def __init__(self, *, ready: bool = True) -> None:
    self.ready = ready
    self.calls: list[tuple[str, dict[str, Any]]] = []

  def get_server_for_tool(self, name: str) -> str | None:
    return "research-corpus-mcp" if name == "check_corpus_readiness" else None

  async def call_tool(self, name: str, tool_input: dict[str, Any]):
    self.calls.append((name, dict(tool_input)))
    return {
      "status": "success",
      **tool_input,
      "ready": self.ready,
      "found_filings": list(tool_input["required_filings"]) if self.ready else [],
      "found_transcripts": list(tool_input["required_transcripts"]) if self.ready else [],
      "missing_filings": [] if self.ready else list(tool_input["required_filings"]),
      "missing_transcripts": [] if self.ready else list(tool_input["required_transcripts"]),
      "unavailable_filings": [],
      "unavailable_transcripts": [],
    }, None


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
    capability_bind: dict[str, Any],
    pid: int | None = None,
    dispatch_key: str | None = None,
    dispatch_request_spec: dict[str, Any] | None = None,
    user_id: str | None = None,
    user_email: str | None = None,
  ) -> tuple[int, str, str | None]:
    if self.raise_active:
      raise FakeActiveBatchError(99)
    resolved_user_id = user_id or "alice"
    batch_id = _acquire_test_batch(registry,
      user_id=resolved_user_id,
      host=host,
      spec=spec,
      capability_bind=capability_bind,
      dispatch_request_spec=dispatch_request_spec,
      budget_usd=float(spec.get("budget_usd") or spec.get("budget") or 0),
      pid=pid,
      dispatch_key=dispatch_key,
    )
    self.acquire_calls.append(
      {
        "batch_id": batch_id,
        "spec": dict(spec),
        "user_id": resolved_user_id,
        "user_email": user_email,
        "dispatch_key": dispatch_key,
        "capability_bind": dict(capability_bind),
      }
    )
    return batch_id, resolved_user_id, user_email

  async def run_acquired_batch(
    self,
    batch_id: int,
    spec: dict[str, Any],
    *,
    registry: BatchRegistry,
    capability_execution_resolver: Any,
    session_driver_execution: Any,
    captured_run_admission_factory: Any,
    _identity: tuple[str, str | None] | None = None,
    _on_finalize=None,
  ) -> dict[str, Any]:
    assert capability_execution_resolver.auth_context.run_mode == "batch"
    assert session_driver_execution.bind.run_mode == "batch"
    self.run_calls.append({
      "batch_id": batch_id,
      "spec": dict(spec),
      "identity": _identity,
      "capability_execution_resolver": capability_execution_resolver,
      "session_driver_execution": session_driver_execution,
      "captured_run_admission_factory": (
        captured_run_admission_factory
      ),
    })
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


def _make_app(*, mcp_client: Any = None):
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda *_args: None,
      capability_execution=request.capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      default_provider=_BATCH_TEST_PROVIDER,
      jwt_secret="batch-control-test-secret-0123456789",
      valid_api_keys={API_KEY},
      tenant_id="test-product",
      model_registry=_MODEL_REGISTRY,
      model_selection_policy=_MODEL_SELECTION_POLICY,
      service_provider_handles={"anthropic": _SERVICE_HANDLE},
      service_auth_config_resolver=_service_materializer,
      capability_adapter_resolver=_adapter_resolver,
      build_chat_runtime=_build_chat_runtime,
      mcp_client=mcp_client,
      claim_signing_authority=GatewayClaimSigningAuthority(
        "batch-control-test-claim-key-at-least-32-bytes"
      ),
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


def _headers(
  session_payload: dict[str, Any],
  *,
  idempotency_key: str | None = None,
) -> dict[str, str]:
  return {
    "Authorization": f"Bearer {session_payload['session_token']}",
    "Idempotency-Key": idempotency_key or f"batch-test-{uuid.uuid4().hex}",
  }


def _batch_spec(*, universe: list[str] | None = None) -> dict[str, Any]:
  return {
    "source": "quality_screen",
    "universe": universe or ["MSFT"],
    "budget_usd": 1.0,
    "max_concurrency": 2,
  }


def _acquire_test_batch(
  registry: BatchRegistry,
  **kwargs: Any,
) -> int:
  capability_bind = kwargs.pop("capability_bind", None)
  if capability_bind is None:
    owner_user_id = str(kwargs.get("user_id") or "alice")
    _resolver, execution = (
      batches_module._batch_capability_execution_context(
        app_state=SimpleNamespace(gateway_config=_gateway_config()),
        user_id=owner_user_id,
        authenticated_session=None,
      )
    )
    capability_bind = execution.bind.receipt()
  return BatchRegistry.acquire_batch(
    registry,
    capability_bind=capability_bind,
    **kwargs,
  )


def _install_batch_pending_approval(
  *,
  app: Any,
  batch_id: int,
  owner_user_id: str = "alice",
  channel: str = "tui",
  request_id_override: str | None = None,
  durable_owner_user_id: str | None = None,
  tool_class: str = "state_write",
  persistent_grant_scope: str | None = None,
  stage_run_seq: int = 3,
) -> tuple[ApprovalRequest, asyncio.Queue, SimpleNamespace]:
  suffix = uuid.uuid4().hex
  approval_id = f"approval-batch-{suffix}"
  tool_call_id = f"tool-batch-{suffix}"
  session_id = f"batch-stage-{suffix}"
  run_id = f"batch_{batch_id}"
  request_record = ApprovalRequest(
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    parent_approval_id=None,
    approval_chain_id=approval_id,
    request_id=request_id_override or run_id,
    session_id=session_id,
    run_id=run_id,
    user_id=durable_owner_user_id or owner_user_id,
    profile="analyst",
    channel=channel,
    tool_name="memory_write",
    tool_class=tool_class,
    tool_args_redacted={"file": "notes/test.md"},
    args_hash=f"cancel-hash-{suffix}",
    reason="requires approval",
    blast_radius_summary="state_write:memory_write",
    approval_constraint="standard",
    state="pending_user",
    requested_at=utc_now(),
    persistent_grant_scope=persistent_grant_scope,
  )
  _run(app.state.gateway_approval_store.create(request_record))
  queue: asyncio.Queue = asyncio.Queue(maxsize=1)
  batch_session = SimpleNamespace(
    session_id=session_id,
    user_id=owner_user_id,
    channel=channel,
    approval_store=app.state.gateway_approval_store,
    approval_policy=app.state.gateway_approval_policy,
    pending_tools={
      tool_call_id: {
        "approval_id": approval_id,
        "nonce": "cancel-nonce",
        "status": "approval_pending",
        "stage_run_seq": stage_run_seq,
      }
    },
    approval_queues={tool_call_id: queue},
  )
  app.state.batch_task_registry.approval_projections.register_session(
    batch_id=batch_id,
    owner_user_id=owner_user_id,
    channel=channel,
    session=batch_session,
  )
  return request_record, queue, batch_session


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
        "semantic_result": None,
        "execution_termination": None,
      }
    ]
    assert detail["proposals"] == []
    assert set(detail) == {"batch", "verdict_matrix", "candidates", "proposals", "failures"}

    list_response = client.get("/api/control/batches", headers=headers)
    assert list_response.status_code == 200, list_response.text
    batches = list_response.json()["batches"]
    assert [item["batch_id"] for item in batches] == [batch_id]
    assert batches[0]["cost_usd"] == pytest.approx(0.125)
    assert fake_batch_control.run_calls[0]["identity"] == ("alice", "alice@example.com")


def test_control_batch_dispatch_replays_same_key_without_duplicate_task(
  fake_batch_control: FakeBatchController,
) -> None:
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(
      _control_session(client),
      idempotency_key="batch-replay-test",
    )
    first = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )
    assert first.status_code == 200, first.text
    batch_id = int(first.json()["batch_id"])
    _wait_for_batch_status(client, headers, batch_id, {"completed"})

    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert replay.status_code == 200, replay.text
  assert replay.json() == {
    "batch_id": batch_id,
    "status": "completed",
    "replayed": True,
  }
  assert len(fake_batch_control.acquire_calls) == 1
  assert len(fake_batch_control.run_calls) == 1


def test_control_batch_dispatch_replay_preserves_locally_admitted_task(
  fake_batch_control: FakeBatchController,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(
      _control_session(client),
      idempotency_key="batch-live-local-replay",
    )
    first = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert first.status_code == 200, first.text
  assert replay.status_code == 200, replay.text
  assert replay.json() == {
    "batch_id": int(first.json()["batch_id"]),
    "status": "running",
    "replayed": True,
  }
  assert len(fake_batch_control.acquire_calls) == 1
  assert len(fake_batch_control.run_calls) == 1


def test_control_batch_dispatch_key_rejects_payload_drift(
  fake_batch_control: FakeBatchController,
) -> None:
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(
      _control_session(client),
      idempotency_key="batch-conflict-test",
    )
    first = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )
    assert first.status_code == 200, first.text
    _wait_for_batch_status(
      client,
      headers,
      int(first.json()["batch_id"]),
      {"completed"},
    )

    conflict = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(universe=["ADI"]),
    )

  assert conflict.status_code == 409, conflict.text
  assert "different spec" in conflict.json()["detail"]
  assert len(fake_batch_control.acquire_calls) == 1
  assert len(fake_batch_control.run_calls) == 1


def test_control_batch_dispatch_requires_idempotency_key(
  fake_batch_control: FakeBatchController,
) -> None:
  app = _make_app()
  with TestClient(app) as client:
    session = _control_session(client)
    response = client.post(
      "/api/control/batches",
      headers={"Authorization": f"Bearer {session['session_token']}"},
      json=_batch_spec(),
    )

  assert response.status_code == 400, response.text
  assert response.json()["detail"] == (
    "Idempotency-Key is required for batch dispatch"
  )
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batch_post_admission_value_error_is_not_attested_unadmitted(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  original_handoff = batches_module._acquire_and_start_batch
  handoff_calls = 0

  def fail_once_after_handoff(*args: Any, **kwargs: Any) -> int:
    nonlocal handoff_calls
    handoff_calls += 1
    batch_id = original_handoff(*args, **kwargs)
    if handoff_calls == 1:
      raise ValueError("post-admission task handoff response failed")
    return batch_id

  monkeypatch.setattr(
    batches_module,
    "_acquire_and_start_batch",
    fail_once_after_handoff,
  )
  with TestClient(app) as client:
    headers = _headers(
      _control_session(client),
      idempotency_key="batch-post-admission-value-error",
    )
    first = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )
    registry = batches_module._registry_for_user("alice")
    admitted = registry.list_batches("alice")
    registry.close()
    second = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert first.status_code == 422, first.text
  assert "X-Batch-Dispatch-Admitted" not in first.headers
  assert len(admitted) == 1
  assert second.status_code == 200, second.text
  assert second.json()["batch_id"] == admitted[0]["batch_id"]
  assert second.json()["replayed"] is True
  assert len(fake_batch_control.run_calls) == 1


def test_control_batch_dispatch_replay_survives_gateway_restart(
  fake_batch_control: FakeBatchController,
) -> None:
  key = "batch-restart-replay-test"
  first_app = _make_app()
  with TestClient(first_app) as client:
    headers = _headers(_control_session(client), idempotency_key=key)
    first = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )
    assert first.status_code == 200, first.text
    batch_id = int(first.json()["batch_id"])
    _wait_for_batch_status(client, headers, batch_id, {"completed"})

  restarted_app = _make_app()
  with TestClient(restarted_app) as client:
    headers = _headers(_control_session(client), idempotency_key=key)
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert replay.status_code == 200, replay.text
  assert replay.json()["batch_id"] == batch_id
  assert replay.json()["status"] == "completed"
  assert replay.json()["replayed"] is True
  assert len(fake_batch_control.acquire_calls) == 1
  assert len(fake_batch_control.run_calls) == 1


def test_exact_replay_precedes_all_mutable_fresh_admission_gates(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  key = "batch-replay-before-mutable-gates"
  request_spec = _batch_spec()
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="old-gateway",
    spec={**request_spec, "batch_date": "2026-07-25"},
    dispatch_request_spec=request_spec,
    budget_usd=1.0,
    pid=999_999,
    dispatch_key=key,
  )
  registry.set_status(batch_id, "completed")
  registry.close()

  def forbidden_gate(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("exact replay reached a fresh-admission gate")

  async def forbidden_readiness(*_args: Any, **_kwargs: Any) -> Any:
    forbidden_gate()

  task_registry = batches_module.BatchTaskRegistry()
  monkeypatch.setattr(batches_module, "_has_active_credential", forbidden_gate)
  monkeypatch.setattr(
    batches_module,
    "_batch_capability_execution_context",
    forbidden_gate,
  )
  monkeypatch.setattr(
    batches_module,
    "require_corpus_readiness",
    forbidden_readiness,
  )
  monkeypatch.setattr(task_registry, "assert_accepting", forbidden_gate)
  app_state = SimpleNamespace(
    gateway_config=SimpleNamespace(credentials_resolver=object()),
    batch_task_registry=task_registry,
  )

  result = asyncio.run(
    batches_module.dispatch_batch_in_process(
      request_spec,
      app_state=app_state,
      user_id="alice",
      dispatch_key=key,
      user_email="alice@example.com",
      authenticated_session=_gateway_session(),
    )
  )

  assert result == {
    "batch_id": batch_id,
    "status": "completed",
    "replayed": True,
  }
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batch_dispatch_replay_terminalizes_fresh_foreign_orphan(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    batches_module.socket,
    "gethostname",
    lambda: "local-gateway",
  )
  key = "batch-fresh-foreign-replay"
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="foreign-gateway",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=999_999,
    now=time.time(),
    dispatch_key=key,
  )
  registry.close()

  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client), idempotency_key=key)
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert replay.status_code == 200, replay.text
  assert replay.json() == {
    "batch_id": batch_id,
    "status": "failed",
    "replayed": True,
  }
  reopened = batches_module._registry_for_user("alice")
  try:
    digest = reopened.get_batch_digest(batch_id)
  finally:
    reopened.close()
  assert digest["status"] == "failed"
  assert digest["error"] == (
    "batch_dispatch_replay_orphaned_after_gateway_restart"
  )
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batch_dispatch_replay_terminalizes_same_host_pid_reuse_orphan(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    batches_module.socket,
    "gethostname",
    lambda: "local-gateway",
  )
  key = "batch-same-host-pid-reuse-replay"
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="local-gateway",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=batches_module.os.getpid(),
    now=time.time(),
    dispatch_key=key,
  )
  registry.close()

  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client), idempotency_key=key)
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert replay.status_code == 200, replay.text
  assert replay.json() == {
    "batch_id": batch_id,
    "status": "failed",
    "replayed": True,
  }
  reopened = batches_module._registry_for_user("alice")
  try:
    digest = reopened.get_batch_digest(batch_id)
  finally:
    reopened.close()
  assert digest["error"] == (
    "batch_dispatch_replay_orphaned_after_gateway_restart"
  )
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batch_dispatch_replay_fails_closed_on_malformed_owner(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(
    batches_module.socket,
    "gethostname",
    lambda: "local-gateway",
  )
  key = "batch-malformed-owner-replay"
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="local-gateway",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=batches_module.os.getpid(),
    now=time.time(),
    dispatch_key=key,
  )
  with registry._db_lock:
    conn = registry._ensure_db()
    conn.execute(
      "UPDATE batches SET host = 12345 WHERE batch_id = ?",
      (batch_id,),
    )
    conn.commit()
  registry.close()

  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client), idempotency_key=key)
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert replay.status_code == 200, replay.text
  assert replay.json() == {
    "batch_id": batch_id,
    "status": "failed",
    "replayed": True,
  }
  reopened = batches_module._registry_for_user("alice")
  try:
    digest = reopened.get_batch_digest(batch_id)
  finally:
    reopened.close()
  assert digest["error"] == (
    "batch_dispatch_replay_orphaned_after_gateway_restart"
  )
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_validation_rejection_race_reconciles_admitted_winner(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  request_spec = _batch_spec()
  dispatch_key = "batch-admission-wins-validation-race"

  class RaceRegistry:
    def __init__(self) -> None:
      self.rejection_calls = 0
      self.close_calls = 0

    def lookup_batch_dispatch(self, **_kwargs: Any) -> None:
      return None

    def list_active_batches(self, _user_id: str) -> list[dict[str, Any]]:
      return []

    def record_batch_dispatch_rejection(
      self,
      **_kwargs: Any,
    ) -> BatchDispatchRecord:
      self.rejection_calls += 1
      return BatchDispatchRecord(
        batch_id=17,
        dispatch_key=dispatch_key,
        request_spec=request_spec,
      )

    def get_batch_digest(self, _batch_id: int) -> dict[str, Any]:
      return {
        "batch_id": 17,
        "user_id": "alice",
        "status": "completed",
        "cost_usd": 0.0,
      }

    def close(self) -> None:
      self.close_calls += 1

  registry = RaceRegistry()
  monkeypatch.setattr(
    batches_module,
    "_registry_for_user",
    lambda _user_id: registry,
  )
  monkeypatch.setattr(
    batches_module,
    "_batch_capability_execution_context",
    lambda **_kwargs: (object(), object()),
  )
  monkeypatch.setattr(
    batches_module,
    "require_corpus_readiness",
    lambda payload, **_kwargs: asyncio.sleep(
      0,
      result=(payload, None),
    ),
  )

  def fail_validation(*_args: Any, **_kwargs: Any) -> int:
    raise batches_module._BatchDispatchValidationError("invalid request")

  monkeypatch.setattr(
    batches_module,
    "_acquire_and_start_batch",
    fail_validation,
  )
  app_state = SimpleNamespace(
    gateway_config=SimpleNamespace(credentials_resolver=None),
    batch_task_registry=batches_module.BatchTaskRegistry(),
  )

  result = asyncio.run(
    batches_module.dispatch_batch_in_process(
      request_spec,
      app_state=app_state,
      user_id="alice",
      role="owner",
      dispatch_key=dispatch_key,
      user_email="alice@example.com",
    )
  )

  assert result == {
    "batch_id": 17,
    "status": "completed",
    "replayed": True,
  }
  assert registry.rejection_calls == 1
  assert registry.close_calls == 2


@pytest.mark.parametrize(
  ("owner_host", "owner_pid"),
  [
    ("foreign-gateway", 999_999),
    ("local-gateway", batches_module.os.getpid()),
  ],
  # Explicit ids: the second value is the collecting process's pid, which
  # differs per xdist worker — a pid-bearing auto-id makes workers collect
  # different test ids and aborts the run.
  ids=("foreign-owner", "local-owner"),
)
def test_fresh_key_terminalizes_unowned_active_batch_before_admission(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
  owner_host: str,
  owner_pid: int,
) -> None:
  monkeypatch.setattr(
    batches_module.socket,
    "gethostname",
    lambda: "local-gateway",
  )
  registry = batches_module._registry_for_user("alice")
  orphan_id = _acquire_test_batch(registry,
    user_id="alice",
    host=owner_host,
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=owner_pid,
  )
  registry.close()

  app = _make_app()
  with TestClient(app) as client:
    response = client.post(
      "/api/control/batches",
      headers=_headers(
        _control_session(client),
        idempotency_key=f"fresh-after-orphan-{owner_host}",
      ),
      json=_batch_spec(universe=["ADI"]),
    )

  assert response.status_code == 200, response.text
  assert int(response.json()["batch_id"]) != orphan_id
  reopened = batches_module._registry_for_user("alice")
  try:
    orphan = reopened.get_batch_digest(orphan_id)
  finally:
    reopened.close()
  assert orphan["status"] == "failed"
  assert orphan["error"] == (
    "batch_dispatch_orphaned_before_fresh_admission"
  )
  assert len(fake_batch_control.acquire_calls) == 1
  assert len(fake_batch_control.run_calls) == 1


def test_fresh_key_preserves_locally_owned_active_batch(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  registry = batches_module._registry_for_user("alice")
  active_id = _acquire_test_batch(registry,
    user_id="alice",
    host="local-gateway",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=batches_module.os.getpid(),
  )
  registry.close()
  task_registry = batches_module.BatchTaskRegistry()
  monkeypatch.setattr(
    task_registry,
    "has_admitted_batch",
    lambda *, owner_user_id, batch_id: (
      owner_user_id == "alice" and batch_id == active_id
    ),
  )
  app_state = SimpleNamespace(
    gateway_config=_gateway_config(),
    batch_task_registry=task_registry,
    gateway_approval_store=None,
    gateway_approval_policy=None,
    gateway_claim_signing_authority=GatewayClaimSigningAuthority(
      "local-active-batch-test-key-at-least-32-bytes"
    ),
    autonomous_storage_root=Path(
      "/tmp/agent-gateway-local-active-batch-test"
    ),
  )

  with pytest.raises(ActiveBatchError) as exc_info:
    asyncio.run(
      batches_module.dispatch_batch_in_process(
        _batch_spec(universe=["ADI"]),
        app_state=app_state,
        user_id="alice",
        role="owner",
        dispatch_key="fresh-while-local-batch-live",
        user_email="alice@example.com",
      )
    )

  assert exc_info.value.batch_id == active_id
  reopened = batches_module._registry_for_user("alice")
  try:
    assert reopened.get_batch_digest(active_id)["status"] == "running"
    assert len(reopened.list_batches("alice")) == 1
  finally:
    reopened.close()
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batch_dispatch_replay_terminalizes_restart_orphan(
  fake_batch_control: FakeBatchController,
) -> None:
  key = "batch-restart-orphan-test"
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=999_999,
    dispatch_key=key,
  )
  registry.close()

  restarted_app = _make_app()
  with TestClient(restarted_app) as client:
    headers = _headers(_control_session(client), idempotency_key=key)
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert replay.status_code == 200, replay.text
  assert replay.json() == {
    "batch_id": batch_id,
    "status": "failed",
    "replayed": True,
  }
  reopened = batches_module._registry_for_user("alice")
  try:
    digest = reopened.get_batch_digest(batch_id)
  finally:
    reopened.close()
  assert digest["status"] == "failed"
  assert digest["error"] == (
    "batch_dispatch_replay_orphaned_after_gateway_restart"
  )
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batch_dispatch_replay_fails_if_orphan_stays_active(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  key = "batch-restart-orphan-transition-failure"
  registry = batches_module._registry_for_user("alice")
  _acquire_test_batch(registry,
    user_id="alice",
    host="",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=999_999,
    dispatch_key=key,
  )
  registry.close()
  monkeypatch.setattr(
    BatchRegistry,
    "transition_status_if_current",
    lambda *_args, **_kwargs: None,
  )

  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client), idempotency_key=key)
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert replay.status_code == 503, replay.text
  assert replay.json()["detail"] == (
    "orphaned batch replay could not be terminalized"
  )
  assert "X-Batch-Dispatch-Admitted" not in replay.headers
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_in_process_batch_task_contains_system_exit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def run_case() -> None:
    class SystemExitController:
      def acquire_batch_run(self, _payload, **_kwargs):
        return 17, "alice", "alice@example.com"

      async def run_acquired_batch(self, *_args, **_kwargs):
        raise SystemExit("malformed MCP config")

    registry = SimpleNamespace(
      close=lambda: None,
      list_active_batches=lambda _user_id: [],
      lookup_batch_dispatch=lambda **_kwargs: None,
    )
    captured_tasks: list[asyncio.Task[Any]] = []
    task_registry = batches_module.BatchTaskRegistry()
    original_start = task_registry.start

    def capture_start(**kwargs) -> None:
      captured_tasks.append(kwargs["task"])
      original_start(**kwargs)

    monkeypatch.setattr(task_registry, "start", capture_start)
    monkeypatch.setattr(batches_module, "_controller", lambda: SystemExitController())
    monkeypatch.setattr(batches_module, "_registry_for_user", lambda _user_id: registry)
    monkeypatch.setattr(
      batches_module,
      "require_corpus_readiness",
      lambda payload, **_kwargs: asyncio.sleep(0, result=(payload, None)),
    )
    app_state = SimpleNamespace(
      gateway_config=_gateway_config(),
      batch_task_registry=task_registry,
      gateway_approval_store=None,
      gateway_approval_policy=None,
      gateway_claim_signing_authority=GatewayClaimSigningAuthority(
        "batch-system-exit-test-key-at-least-32-bytes"
      ),
      autonomous_storage_root=Path(
        "/tmp/agent-gateway-batch-system-exit-test"
      ),
    )

    result = await batches_module.dispatch_batch_in_process(
      _batch_spec(),
      app_state=app_state,
      user_id="alice",
      role="owner",
      dispatch_key="batch-system-exit-test",
      user_email="alice@example.com",
    )
    assert result == {
      "batch_id": 17,
      "status": "running",
      "replayed": False,
    }
    assert len(captured_tasks) == 1
    assert await captured_tasks[0] is None
    await asyncio.sleep(0)
    assert not asyncio.current_task().cancelled()

  asyncio.run(run_case())


def test_in_process_batch_task_preserves_cancellation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def run_case() -> None:
    started = asyncio.Event()

    class CancelledController:
      def acquire_batch_run(self, _payload, **_kwargs):
        return 18, "alice", "alice@example.com"

      async def run_acquired_batch(self, *_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    registry = SimpleNamespace(
      close=lambda: None,
      list_active_batches=lambda _user_id: [],
      lookup_batch_dispatch=lambda **_kwargs: None,
    )
    captured_tasks: list[asyncio.Task[Any]] = []
    task_registry = batches_module.BatchTaskRegistry()
    original_start = task_registry.start

    def capture_start(**kwargs) -> None:
      captured_tasks.append(kwargs["task"])
      original_start(**kwargs)

    monkeypatch.setattr(task_registry, "start", capture_start)
    monkeypatch.setattr(batches_module, "_controller", lambda: CancelledController())
    monkeypatch.setattr(batches_module, "_registry_for_user", lambda _user_id: registry)
    monkeypatch.setattr(
      batches_module,
      "require_corpus_readiness",
      lambda payload, **_kwargs: asyncio.sleep(0, result=(payload, None)),
    )
    app_state = SimpleNamespace(
      gateway_config=_gateway_config(),
      batch_task_registry=task_registry,
      gateway_approval_store=None,
      gateway_approval_policy=None,
      gateway_claim_signing_authority=GatewayClaimSigningAuthority(
        "batch-cancellation-test-key-at-least-32-bytes"
      ),
      autonomous_storage_root=Path(
        "/tmp/agent-gateway-batch-cancellation-test"
      ),
    )

    await batches_module.dispatch_batch_in_process(
      _batch_spec(),
      app_state=app_state,
      user_id="alice",
      role="owner",
      dispatch_key="batch-admission-failure-test",
      user_email="alice@example.com",
    )
    assert len(captured_tasks) == 1
    await started.wait()
    captured_tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
      await captured_tasks[0]

  asyncio.run(run_case())


def test_control_batches_use_canonical_session_owner_identity(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(batches_module, "_session_owner_user_id", lambda _session: "1")
  app = _make_app()

  with TestClient(app) as client:
    headers = _headers(_control_session(client, user_id="henry"))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    assert response.status_code == 200, response.text
    batch_id = int(response.json()["batch_id"])
    detail = _wait_for_batch_status(client, headers, batch_id, {"completed"})
    listed = client.get("/api/control/batches", headers=headers)

  assert detail["batch"]["user_id"] == "1"
  assert fake_batch_control.run_calls[0]["identity"] == ("1", "henry@example.com")
  assert [row["batch_id"] for row in listed.json()["batches"]] == [batch_id]


def test_batch_detail_adds_force_rerun_hint_for_existing_unroutable(tmp_path: Path) -> None:
  registry = BatchRegistry(tmp_path / "batch.db")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="test-host",
    spec={
      "source": "quality_screen",
      "universe": ["MSFT", "AAPL"],
      "budget_usd": 3.0,
      "max_concurrency": 2,
      "gates": {"skip_source_mismatch": True},
    },
    budget_usd=3.0,
    pid=None,
  )
  registry.allocate_run(
    batch_id=batch_id,
    ticker="MSFT",
    stage="pipeline",
    skill=None,
    status="skipped",
    dispatch_status="skipped",
    item_source="quality_screen",
    error="skipped_existing_unroutable",
  )

  detail = batches_module._batch_detail_payload(registry, batch_id, top_n=10)

  assert detail["failures"] == [
    {
      "ticker": "MSFT",
      "skill": None,
      "status": "skipped",
      "error": "skipped_existing_unroutable",
      "repair_hint": (
        "This ticker already has a research file that could not be routed for this batch source. "
        "Rerun it by calling start_diligence_batch with gates.force_rerun_existing=true."
      ),
      "retry_spec": {
        "source": "quality_screen",
        "universe": ["MSFT"],
        "budget_usd": 3.0,
        "max_concurrency": 2,
        "gates": {"skip_source_mismatch": True, "force_rerun_existing": True},
        "force_rerun_existing": True,
      },
    }
  ]
  assert set(detail) == {"batch", "verdict_matrix", "candidates", "proposals", "failures"}
  registry.close()


def test_retry_spec_narrows_persisted_corpus_requirements_to_failed_tickers() -> None:
  prior = {
    "spec_json": json.dumps({
      "source": "explicit_ticker",
      "universe": ["MSFT", "ADI"],
      "pipeline_template": "valuation-ready",
      "corpus_requirements": [
        {
          "ticker": "MSFT",
          "required_filings": ["2025-FY"],
          "required_transcripts": ["2026-Q2"],
        },
        {
          "ticker": "ADI",
          "required_filings": ["2025-FY"],
          "required_transcripts": ["2026-Q3"],
        },
      ],
    })
  }

  retry = batches_module._retry_spec(prior, ["ADI"])

  assert retry["universe"] == ["ADI"]
  assert retry["corpus_requirements"] == [{
    "ticker": "ADI",
    "required_filings": ["2025-FY"],
    "required_transcripts": ["2026-Q3"],
  }]


def test_gateway_local_valuation_ready_dispatch_uses_batch_helper(
  fake_batch_control: FakeBatchController,
) -> None:
  async def run_case() -> None:
    mcp_client = ReadyCorpusMcpClient()
    app_state = SimpleNamespace(
      user_event_bus=None,
      gateway_config=_gateway_config(mcp_client=mcp_client),
      gateway_claim_signing_authority=GatewayClaimSigningAuthority(
        "valuation-ready-dispatch-test-key-at-least-32-bytes"
      ),
      autonomous_storage_root=Path(
        "/tmp/agent-gateway-valuation-ready-dispatch-test"
      ),
    )
    session = _gateway_session()
    bundle = make_valuation_ready_skill_tool_bundle(app_state=app_state, session=session)
    dispatch = bundle["handlers"]["valuation_ready_batch_dispatch"]
    read = bundle["handlers"]["valuation_ready_batch_read"]
    dispatch_schema = bundle["tool_definitions"][0]["input_schema"]
    assert dispatch_schema["required"] == [
      "ticker",
      "required_filings",
      "required_transcripts",
    ]

    token = set_current_skill("valuation-ready")
    try:
      result, error = await dispatch({
        "ticker": "adi",
        "required_filings": ["2025-FY", "2026-Q2"],
        "required_transcripts": ["2026-Q1", "2026-Q2"],
      }, tool_ctx=SimpleNamespace(tool_call_id="valuation-ready-dispatch-1"))
      assert error is None
      assert result["status"] == "running"
      assert result["source"] == "explicit_ticker"
      assert result["pipeline_template"] == "valuation-ready"
      assert result["ticker"] == "ADI"
      assert result["max_concurrency"] == 1
      assert result["corpus_readiness"]["status"] == "ready"
      batch_id = int(result["batch_id"])
      task_registry = getattr(app_state, "batch_task_registry")
      task = task_registry.get(owner_user_id="1", batch_id=batch_id)
      assert task is not None
      await task

      detail, read_error = await read({"batch_id": batch_id})
      assert read_error is None
      assert detail["batch"]["status"] == "completed"
    finally:
      reset_current_skill(token)

  _run(run_case())
  assert fake_batch_control.acquire_calls[0]["spec"]["source"] == "explicit_ticker"
  assert fake_batch_control.acquire_calls[0]["spec"]["pipeline_template"] == "valuation-ready"
  assert fake_batch_control.acquire_calls[0]["spec"]["universe"] == ["ADI"]
  assert fake_batch_control.acquire_calls[0]["spec"]["corpus_requirements"] == [{
    "ticker": "ADI",
    "required_filings": ["2025-FY", "2026-Q2"],
    "required_transcripts": ["2026-Q1", "2026-Q2"],
  }]
  assert fake_batch_control.run_calls[0]["spec"] == fake_batch_control.acquire_calls[0]["spec"]
  assert fake_batch_control.run_calls[0]["identity"] == ("1", "alice@example.com")


def test_batch_admission_to_task_handoff_is_structurally_synchronous() -> None:
  handoff = batches_module._acquire_and_start_batch
  source = inspect.getsource(handoff)

  assert not inspect.iscoroutinefunction(handoff)
  assert "asyncio.create_task(" in source
  assert "async def _captured_run_admission_factory(" in source
  assert source.index("acquire_batch_run") < source.rindex("task_registry.start")


def test_gateway_local_valuation_ready_dispatch_propagates_session_channel(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  captured: dict[str, Any] = {}

  async def fake_dispatch(_spec, **kwargs):
    captured.update(kwargs)
    return {"batch_id": 7, "status": "running"}

  monkeypatch.setattr(batches_module, "dispatch_batch_in_process", fake_dispatch)
  app_state = SimpleNamespace()
  session = SimpleNamespace(
    session_id="valuation-session-1",
    user_id="henry",
    owner_user_id="1",
    risk_user_id=1,
    user_email="henry@example.com",
    channel="telegram",
  )
  bundle = make_valuation_ready_skill_tool_bundle(app_state=app_state, session=session)

  async def run_case() -> None:
    token = set_current_skill("valuation-ready")
    try:
      result, error = await bundle["handlers"]["valuation_ready_batch_dispatch"](
        {
          "ticker": "PCTY",
          "required_filings": ["2025-FY"],
          "required_transcripts": ["2026-Q2"],
        },
        tool_ctx=SimpleNamespace(tool_call_id="valuation-ready-dispatch-channel"),
      )
      assert error is None
      assert result["batch_id"] == 7
    finally:
      reset_current_skill(token)

  _run(run_case())
  assert captured["user_id"] == "1"
  assert captured["channel"] == "telegram"
  assert captured["dispatch_key"].startswith("valuation-ready-")


def test_gateway_local_valuation_ready_dispatch_scopes_key_to_tool_call(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  dispatch_keys: list[str] = []

  async def fake_dispatch(_spec, **kwargs):
    dispatch_keys.append(str(kwargs["dispatch_key"]))
    return {"batch_id": len(dispatch_keys), "status": "running"}

  monkeypatch.setattr(batches_module, "dispatch_batch_in_process", fake_dispatch)
  session = SimpleNamespace(
    session_id="valuation-session-idempotency",
    user_id="henry",
    owner_user_id="1",
    risk_user_id=1,
    user_email="henry@example.com",
    channel="cli",
  )
  dispatch = make_valuation_ready_skill_tool_bundle(
    app_state=SimpleNamespace(),
    session=session,
  )["handlers"]["valuation_ready_batch_dispatch"]
  async def run_case() -> None:
    token = set_current_skill("valuation-ready")
    try:
      for tool_call_id, ticker in (
        ("tool-call-1", "PCTY"),
        ("tool-call-1", "ADI"),
        ("tool-call-2", "PCTY"),
      ):
        result, error = await dispatch(
          {
            "ticker": ticker,
            "required_filings": ["2025-FY"],
            "required_transcripts": ["2026-Q2"],
          },
          tool_ctx=SimpleNamespace(tool_call_id=tool_call_id),
        )
        assert error is None
        assert result["status"] == "running"
    finally:
      reset_current_skill(token)

  _run(run_case())
  assert dispatch_keys[0] == dispatch_keys[1]
  assert dispatch_keys[2] != dispatch_keys[0]


def test_gateway_local_valuation_ready_dispatch_blocks_before_batch_admission_on_corpus_gap(
  fake_batch_control: FakeBatchController,
) -> None:
  async def run_case() -> None:
    mcp_client = ReadyCorpusMcpClient(ready=False)
    app_state = SimpleNamespace(
      user_event_bus=None,
      gateway_config=_gateway_config(mcp_client=mcp_client),
    )
    session = _gateway_session()
    bundle = make_valuation_ready_skill_tool_bundle(app_state=app_state, session=session)

    token = set_current_skill("valuation-ready")
    try:
      result, error = await bundle["handlers"]["valuation_ready_batch_dispatch"]({
        "ticker": "PCTY",
        "required_filings": ["2025-FY"],
        "required_transcripts": ["2026-Q2"],
      }, tool_ctx=SimpleNamespace(tool_call_id="valuation-ready-dispatch-gap"))
    finally:
      reset_current_skill(token)

    assert result is None
    assert error["code"] == "corpus_not_ready"
    assert error["details"]["readiness"]["missing_transcripts"] == ["2026-Q2"]

  _run(run_case())
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_gateway_local_valuation_ready_dispatch_refuses_unsupported_runtime(
  fake_batch_control: FakeBatchController,
) -> None:
  async def run_case() -> None:
    app_state = SimpleNamespace(user_event_bus=None)
    session = SimpleNamespace(user_id="alice", user_email="alice@example.com")
    bundle = make_valuation_ready_skill_tool_bundle(app_state=app_state, session=session)
    dispatch = bundle["handlers"]["valuation_ready_batch_dispatch"]
    clear_current_skill()

    result, error = await dispatch(
      {"ticker": "ADI"},
      tool_ctx=SimpleNamespace(tool_call_id="valuation-ready-unsupported"),
    )

    assert result is None
    assert error["code"] == "unsupported_runtime"
    assert error["details"]["verdict"] == "INSUFFICIENT_DATA"
    assert error["details"]["reason"] == "unsupported_runtime"

  _run(run_case())
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batches_active_batch_conflict_maps_to_409(fake_batch_control: FakeBatchController) -> None:
  fake_batch_control.raise_active = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())

  assert response.status_code == 409
  assert "X-Batch-Dispatch-Admitted" not in response.headers
  assert "active batch already exists" in response.json()["detail"]


def test_control_batches_capability_failure_attests_not_admitted(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def fail_capability_resolution(**_kwargs: Any) -> Any:
    raise RuntimeError("batch capability configuration unavailable")

  monkeypatch.setattr(
    batches_module,
    "_batch_capability_execution_context",
    fail_capability_resolution,
  )
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )

  assert response.status_code == 503, response.text
  assert "X-Batch-Dispatch-Admitted" not in response.headers
  assert response.json()["detail"] == (
    "batch capability configuration unavailable"
  )
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batches_malformed_field_type_attests_not_admitted(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  validation_calls = 0

  def reject_malformed_spec(*_args: Any, **_kwargs: Any) -> Any:
    nonlocal validation_calls
    validation_calls += 1
    raise TypeError("max_concurrency must be an integer")

  monkeypatch.setattr(
    fake_batch_control,
    "acquire_batch_run",
    reject_malformed_spec,
  )
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post(
      "/api/control/batches",
      headers=headers,
      json={**_batch_spec(), "max_concurrency": {"invalid": True}},
    )
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json={**_batch_spec(), "max_concurrency": {"invalid": True}},
    )

  assert response.status_code == 422, response.text
  assert response.headers["X-Batch-Dispatch-Admitted"] == "false"
  assert response.json()["detail"] == "max_concurrency must be an integer"
  assert replay.status_code == 422, replay.text
  assert replay.headers["X-Batch-Dispatch-Admitted"] == "false"
  assert replay.json()["detail"] == "max_concurrency must be an integer"
  assert validation_calls == 1
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batches_durably_rejects_intrinsic_corpus_request_error(
  fake_batch_control: FakeBatchController,
) -> None:
  app = _make_app()
  key = "batch-missing-corpus-requirements"
  invalid_spec = {
    "source": "explicit_ticker",
    "universe": ["PCTY"],
    "pipeline_template": "valuation-ready",
    "budget_usd": 25.0,
  }
  with TestClient(app) as client:
    headers = _headers(
      _control_session(client),
      idempotency_key=key,
    )
    first = client.post(
      "/api/control/batches",
      headers=headers,
      json=invalid_spec,
    )
    replay = client.post(
      "/api/control/batches",
      headers=headers,
      json=invalid_spec,
    )
    drift = client.post(
      "/api/control/batches",
      headers=headers,
      json={**invalid_spec, "universe": ["ADI"]},
    )

  expected_detail = {
    "code": "missing_corpus_requirements",
    "message": (
      "valuation-ready batches require exact filing and transcript "
      "period declarations."
    ),
    "details": {"pipeline_template": "valuation-ready"},
  }
  assert first.status_code == 422, first.text
  assert first.headers["X-Batch-Dispatch-Admitted"] == "false"
  assert first.json()["detail"] == expected_detail
  assert replay.status_code == 422, replay.text
  assert replay.headers["X-Batch-Dispatch-Admitted"] == "false"
  assert replay.json()["detail"] == expected_detail
  assert drift.status_code == 409, drift.text
  assert "X-Batch-Dispatch-Admitted" not in drift.headers
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batches_corpus_gap_maps_to_409_before_batch_admission(
  fake_batch_control: FakeBatchController,
) -> None:
  app = _make_app(mcp_client=ReadyCorpusMcpClient(ready=False))
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post(
      "/api/control/batches",
      headers=headers,
      json={
        "source": "explicit_ticker",
        "universe": ["PCTY"],
        "pipeline_template": "valuation-ready",
        "budget_usd": 25.0,
        "corpus_requirements": [{
          "ticker": "PCTY",
          "required_filings": ["2025-FY"],
          "required_transcripts": ["2026-Q2"],
        }],
      },
    )

  assert response.status_code == 409
  assert "X-Batch-Dispatch-Admitted" not in response.headers
  assert response.json()["detail"]["code"] == "corpus_not_ready"
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_control_batches_workflows_catalog(fake_batch_control: FakeBatchController) -> None:
  _ = fake_batch_control
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.get("/api/control/batches/workflows", headers=headers)

  assert response.status_code == 200, response.text
  workflows = response.json()["workflows"]
  assert workflows["earnings-review"]["source"] == "estimate_revisions"
  assert workflows["earnings-review"]["pipeline_template"] == "earnings-review"
  assert workflows["valuation-ready"]["pipeline_template"] == "valuation-ready"
  assert workflows["valuation-ready"]["source_pipeline_template"] == "compounder"
  assert workflows["valuation-ready"]["default_max_concurrency"] == 2
  # earnings-scenarios carries max_budget_usd: 6.0 (vs the $2 stage fallback),
  # so full-diligence reserves 59.0 and valuation-ready 30.0.
  assert workflows["full-diligence"]["reservation_budget_usd_per_name"] == 59.0
  assert workflows["full-diligence"]["suggested_budget_usd_per_name"] == 59.0
  assert workflows["valuation-ready"]["reservation_budget_usd_per_name"] == 30.0
  assert (
    workflows["full-diligence"]["budget_model"]["admission_formula"]
    == "spent_usd + in_flight_reserved_usd + next_stage_reserved_usd <= budget_usd"
  )
  assert workflows["full-diligence"]["budget_model"]["includes_goal_remediation_capacity"] is False


def test_control_batches_cancel_awaits_terminal_state(fake_batch_control: FakeBatchController) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    assert response.status_code == 200, response.text
    batch_id = int(response.json()["batch_id"])

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)
    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["batch"]["status"] == "cancelled", cancel_response.json()["batch"]


def test_terminal_batch_delete_is_read_only_and_does_not_republish(
  fake_batch_control: FakeBatchController,
) -> None:
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="test-host",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=None,
  )
  registry.set_status(batch_id, "completed")
  registry.close()

  app = _make_app()
  publish_calls = 0

  async def forbidden_publish(*_args: Any, **_kwargs: Any) -> bool:
    nonlocal publish_calls
    publish_calls += 1
    raise AssertionError("terminal DELETE must not republish")

  app.state.user_event_bus.publish_terminal_if_absent = forbidden_publish
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

  assert response.status_code == 200, response.text
  assert response.json()["batch"]["status"] == "completed"
  assert publish_calls == 0
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_duplicate_batch_cancel_during_reversible_preflight_is_not_attested(
  fake_batch_control: FakeBatchController,
) -> None:
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="test-host",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=None,
  )
  registry.close()

  app = _make_app()
  task_registry = batches_module.BatchTaskRegistry()
  app.state.batch_task_registry = task_registry
  task_registry.begin_batch_cancellation(
    owner_user_id="alice",
    batch_id=batch_id,
  )
  with TestClient(app) as client:
    try:
      headers = _headers(_control_session(client))
      response = client.delete(
        f"/api/control/batches/{batch_id}",
        headers=headers,
      )
    finally:
      task_registry.abort_batch_cancellation(
        owner_user_id="alice",
        batch_id=batch_id,
      )
    retry_fence = task_registry.begin_batch_cancellation(
      owner_user_id="alice",
      batch_id=batch_id,
    )
    task_registry.abort_batch_cancellation(
      owner_user_id="alice",
      batch_id=batch_id,
    )

  assert response.status_code == 409, response.text
  assert "X-Batch-Cancellation-Committed" not in response.headers
  assert response.json()["detail"] == "batch cancellation is already in progress"
  assert retry_fence.progress.boundary_crossed is False
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_duplicate_batch_cancel_after_boundary_is_marked_drainable(
  fake_batch_control: FakeBatchController,
) -> None:
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="test-host",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=None,
  )
  registry.close()

  app = _make_app()
  task_registry = batches_module.BatchTaskRegistry()
  app.state.batch_task_registry = task_registry
  cancellation_fence = task_registry.begin_batch_cancellation(
    owner_user_id="alice",
    batch_id=batch_id,
  )
  cancellation_fence.progress.boundary_crossed = True
  try:
    with TestClient(app) as client:
      headers = _headers(_control_session(client))
      response = client.delete(
        f"/api/control/batches/{batch_id}",
        headers=headers,
      )
  finally:
    task_registry.finish_batch_cancellation(
      owner_user_id="alice",
      batch_id=batch_id,
    )

  assert response.status_code == 409, response.text
  assert response.headers["X-Batch-Cancellation-Committed"] == "true"
  assert response.json()["detail"] == "batch cancellation is already in progress"
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_batch_cancel_terminal_publish_failure_is_marked_drainable(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()

  async def fail_terminal_publish(**_kwargs: Any) -> bool:
    raise RuntimeError("event bus unavailable")

  monkeypatch.setattr(
    app.state.batch_task_registry,
    "publish_terminal_event",
    fail_terminal_publish,
  )
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    dispatched = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )
    batch_id = int(dispatched.json()["batch_id"])
    response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )
    detail = client.get(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

  assert response.status_code == 503, response.text
  assert response.headers["X-Batch-Cancellation-Committed"] == "true"
  assert response.json()["detail"] == {
    "error": "Batch cancellation completion failed"
  }
  assert detail.status_code == 200, detail.text
  assert detail.json()["batch"]["status"] == "cancelled"


@pytest.mark.parametrize(
  ("fault_method", "expected_error"),
  [
    ("snapshot_fenced_batch", "Late batch approval quarantine failed"),
    ("close_batch", "Batch approval quarantine close failed"),
  ],
)
def test_batch_cancel_late_quarantine_fault_still_terminalizes_orphan(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
  fault_method: str,
  expected_error: str,
) -> None:
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=999_999,
  )
  registry.close()

  app = _make_app()
  projections = app.state.batch_task_registry.approval_projections
  original = getattr(projections, fault_method)
  fault_calls = 0

  def fail_once(*args: Any, **kwargs: Any):
    nonlocal fault_calls
    fault_calls += 1
    fail_on_call = 2 if fault_method == "snapshot_fenced_batch" else 1
    if fault_calls == fail_on_call:
      raise OSError(f"{fault_method} unavailable after cancellation boundary")
    return original(*args, **kwargs)

  monkeypatch.setattr(projections, fault_method, fail_once)
  published: list[dict[str, Any]] = []
  original_publish = app.state.batch_task_registry.publish_terminal_event

  async def capture_publish(**kwargs: Any) -> bool:
    published.append(dict(kwargs["event"]))
    return await original_publish(**kwargs)

  monkeypatch.setattr(
    app.state.batch_task_registry,
    "publish_terminal_event",
    capture_publish,
  )

  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )
    detail = client.get(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

  assert response.status_code == 503, response.text
  assert response.headers["X-Batch-Cancellation-Committed"] == "true"
  assert response.json()["detail"] == {"error": expected_error}
  assert detail.status_code == 200, detail.text
  assert detail.json()["batch"]["status"] == "cancelled"
  assert [event["status"] for event in published] == ["cancelled"]
  assert fault_calls >= 1
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_batch_cancel_preboundary_transition_failure_is_not_attested(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    dispatched = client.post(
      "/api/control/batches",
      headers=headers,
      json=_batch_spec(),
    )
    batch_id = int(dispatched.json()["batch_id"])

    def fail_transition(*_args: Any, **_kwargs: Any) -> None:
      raise OSError("registry unavailable before cancellation commit")

    monkeypatch.setattr(
      BatchRegistry,
      "transition_status_if_current",
      fail_transition,
    )
    response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

  assert response.status_code == 503, response.text
  assert "X-Batch-Cancellation-Committed" not in response.headers
  assert response.json()["detail"] == {
    "error": "Batch cancellation completion failed"
  }


def test_batch_cancel_orphan_transition_failure_is_not_attested(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  registry = batches_module._registry_for_user("alice")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=999_999,
  )
  registry.close()

  def fail_transition(*_args: Any, **_kwargs: Any) -> bool:
    raise OSError("registry unavailable before orphan cancellation commit")

  monkeypatch.setattr(
    BatchRegistry,
    "transition_status_if_current",
    fail_transition,
  )
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

  assert response.status_code == 503, response.text
  assert "X-Batch-Cancellation-Committed" not in response.headers
  assert response.json()["detail"] == {
    "error": "Batch cancellation completion failed"
  }


def test_http_batch_cancel_cancels_hanging_admitted_producer_before_drain(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class HangingPolicy:
    policy_id = "delete-hanging-admission"
    policy_version = "1"
    policy_bundle_hash = "delete-hanging-admission-bundle"

    def __init__(self) -> None:
      self.entered = threading.Event()
      self.finished = threading.Event()
      self.request: ApprovalRequest | None = None
      self.session: Any = None

    async def decide(self, *, payload, request, run_context):
      _ = payload, run_context
      self.request = request
      self.entered.set()
      await asyncio.Event().wait()
      raise AssertionError("hanging approval policy unexpectedly resumed")

    async def on_resolve(self, *, request):
      _ = request

    def role_authorized_for_class(
      self,
      *,
      decider_role: str | None,
      tool_class: str,
    ) -> bool:
      _ = decider_role, tool_class
      return True

  app = _make_app()
  policy = HangingPolicy()
  app.state.gateway_approval_policy = policy

  async def run_hanging_batch(
    batch_id: int,
    spec: dict[str, Any],
    *,
    registry: BatchRegistry,
    capability_execution_resolver: Any,
    session_driver_execution: Any,
    captured_run_admission_factory: Any,
    _identity: tuple[str, str | None] | None = None,
    _on_finalize=None,
  ) -> dict[str, Any]:
    _ = (
      spec,
      capability_execution_resolver,
      session_driver_execution,
      captured_run_admission_factory,
      _identity,
      _on_finalize,
    )
    scope = current_batch_approval_scope()
    assert scope is not None
    session = SimpleNamespace(
      session_id="delete-hanging-stage",
      user_id="alice",
      channel="tui",
      role="owner",
      batch_stage_run_seq=3,
      pending_tools={},
      approval_queues={},
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    policy.session = session
    dispatcher = ToolDispatcher(
      mcp_client=SimpleNamespace(),
      local_tool_handlers={},
      event_log=EventLog(),
      session=session,
      store=app.state.gateway_approval_store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id=f"batch_{batch_id}",
        run_id=f"batch_{batch_id}",
        session_id=session.session_id,
        profile="analyst",
        channel="tui",
        decider_role="owner",
      ),
    )
    try:
      await dispatcher._run_approval_lifecycle(
        tool_call_id="tool-delete-hanging",
        tool_name="memory_write",
        tool_input={},
        qualifier="",
        reason="delete hanging producer",
        allow_persistent=False,
      )
    finally:
      policy.finished.set()
    return {"batch_id": batch_id, "status": "completed"}

  monkeypatch.setattr(fake_batch_control, "run_acquired_batch", run_hanging_batch)
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    assert policy.entered.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
      cancel_response = executor.submit(
        client.delete,
        f"/api/control/batches/{batch_id}",
        headers=headers,
      ).result(timeout=3)

    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["batch"]["status"] == "cancelled"
    assert policy.finished.wait(timeout=1)
    assert policy.request is not None
    stored = _run(app.state.gateway_approval_store.get(policy.request.approval_id))
    assert stored is not None and stored.state == "denied"
    assert policy.session.pending_tools == {}
    assert policy.session.approval_queues == {}
    assert app.state.batch_task_registry.approval_projections.projections_for_batch(
      owner_user_id="alice",
      batch_id=batch_id,
    ) == []


def test_http_batch_cancel_authorizes_bound_durable_request_before_publication(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class InviteRestrictedPolicy:
    policy_id = "bound-request-preflight"
    policy_version = "1"
    policy_bundle_hash = "bound-request-preflight-bundle"

    def __init__(self) -> None:
      self.decision_entered = threading.Event()
      self.allow_decision = threading.Event()
      self.request: ApprovalRequest | None = None
      self.session: Any = None

    async def decide(self, *, payload, request, run_context):
      _ = payload, run_context
      self.request = request
      self.decision_entered.set()
      while not self.allow_decision.is_set():
        await asyncio.sleep(0.001)
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="owner authorization required",
      )

    async def on_resolve(self, *, request):
      _ = request

    def role_authorized_for_class(
      self,
      *,
      decider_role: str | None,
      tool_class: str,
    ) -> bool:
      return decider_role != "invite" or tool_class != "irreversible"

  app = _make_app()
  policy = InviteRestrictedPolicy()
  app.state.gateway_approval_policy = policy
  store = app.state.gateway_approval_store
  pending_committed = threading.Event()
  allow_pending_publication = threading.Event()
  original_enqueue = store.enqueue_pending_approval_notification

  async def paused_enqueue(request: ApprovalRequest) -> None:
    pending_committed.set()
    while not allow_pending_publication.is_set():
      await asyncio.sleep(0.001)
    await original_enqueue(request)

  monkeypatch.setattr(store, "enqueue_pending_approval_notification", paused_enqueue)

  async def run_waiting_batch(
    batch_id: int,
    spec: dict[str, Any],
    *,
    registry: BatchRegistry,
    capability_execution_resolver: Any,
    session_driver_execution: Any,
    captured_run_admission_factory: Any,
    _identity: tuple[str, str | None] | None = None,
    _on_finalize=None,
  ) -> dict[str, Any]:
    _ = (
      spec,
      capability_execution_resolver,
      session_driver_execution,
      captured_run_admission_factory,
      _identity,
      _on_finalize,
    )
    scope = current_batch_approval_scope()
    assert scope is not None
    session = SimpleNamespace(
      session_id="bound-request-preflight-stage",
      user_id="alice",
      channel="tui",
      role="invite",
      batch_stage_run_seq=3,
      pending_tools={},
      approval_queues={},
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    policy.session = session
    dispatcher = ToolDispatcher(
      mcp_client=SimpleNamespace(),
      local_tool_handlers={},
      event_log=EventLog(),
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id=f"batch_{batch_id}",
        run_id=f"batch_{batch_id}",
        session_id=session.session_id,
        profile="analyst",
        channel="tui",
        decider_role="invite",
      ),
    )
    await dispatcher._run_approval_lifecycle(
      tool_call_id="tool-bound-request-preflight",
      tool_name="execute_trade",
      tool_input={"ticker": "MSFT", "side": "buy", "quantity": 1},
      qualifier="",
      reason="bound request preflight race",
      allow_persistent=False,
    )
    return {"batch_id": batch_id, "status": "completed"}

  monkeypatch.setattr(fake_batch_control, "run_acquired_batch", run_waiting_batch)
  with TestClient(app) as client:
    control = _control_session(client)
    control_session = app.state.auth.session_store.get_session(control["session_id"])
    assert control_session is not None
    control_session.role = "invite"
    control["session_token"] = app.state.auth.issue_token(control_session)
    headers = _headers(control)
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    assert policy.decision_entered.wait(timeout=2)
    assert policy.request is not None
    task = app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    )
    assert task is not None and not task.done()
    assert policy.session.pending_tools == {}

    created_cancel_response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

    assert created_cancel_response.status_code == 403, created_cancel_response.text
    assert created_cancel_response.json()["detail"] == {
      "error": "Role is not authorized to cancel this approval",
      "approval_id": policy.request.approval_id,
      "tool_class": "irreversible",
    }
    stored = _run(store.get(policy.request.approval_id))
    assert stored is not None and stored.state == "created"
    assert stored.state_version == 0
    assert stored.decision is None
    assert stored.decision_reason is None
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is task
    assert not task.done()
    assert app.state.batch_task_registry.approval_projections.projections_for_batch(
      owner_user_id="alice",
      batch_id=batch_id,
    ) == []

    policy.allow_decision.set()
    assert pending_committed.wait(timeout=2)
    pending_cancel_response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )
    assert pending_cancel_response.status_code == 403, pending_cancel_response.text
    assert pending_cancel_response.json()["detail"]["approval_id"] == (
      policy.request.approval_id
    )
    stored = _run(store.get(policy.request.approval_id))
    assert stored is not None and stored.state == "pending_user"
    assert stored.state_version == 1
    assert stored.decision is None
    assert stored.decision_reason is None
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is task
    assert not task.done()
    assert policy.session.pending_tools == {}

    allow_pending_publication.set()
    deadline = time.monotonic() + 2
    while not policy.session.pending_tools:
      assert time.monotonic() < deadline
      time.sleep(0.005)
    published_cancel_response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )
    assert published_cancel_response.status_code == 403, published_cancel_response.text
    stored = _run(store.get(policy.request.approval_id))
    assert stored is not None and stored.state == "pending_user"
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is task
    assert not task.done()
    pending_entry = policy.session.pending_tools[policy.request.tool_call_id]
    assert pending_entry["status"] == "approval_pending"
    assert "_approval_cancel_requested" not in pending_entry

    control_session.role = "owner"
    control["session_token"] = app.state.auth.issue_token(control_session)
    cleanup_response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=_headers(control),
    )
    assert cleanup_response.status_code == 200, cleanup_response.text


def test_http_batch_cancel_retains_transient_projection_published_during_preflight(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class PausedPolicy:
    policy_id = "delete-transient-publication"
    policy_version = "1"
    policy_bundle_hash = "delete-transient-publication-bundle"

    def __init__(self) -> None:
      self.entered = threading.Event()
      self.allow_decision = threading.Event()
      self.finished = threading.Event()
      self.request: ApprovalRequest | None = None
      self.session: Any = None
      self.resolved: list[str] = []

    async def decide(self, *, payload, request, run_context):
      _ = payload, run_context
      self.request = request
      self.entered.set()
      while not self.allow_decision.is_set():
        await asyncio.sleep(0.001)
      return PolicyApprovalDecision(
        outcome="request_user_approval",
        reason="publish while cancellation preflight is paused",
      )

    async def on_resolve(self, *, request):
      self.resolved.append(request.approval_id)

    def role_authorized_for_class(
      self,
      *,
      decider_role: str | None,
      tool_class: str,
    ) -> bool:
      _ = decider_role, tool_class
      return True

  app = _make_app()
  policy = PausedPolicy()
  app.state.gateway_approval_policy = policy
  projections = app.state.batch_task_registry.approval_projections
  preflight_entered = threading.Event()
  publication_seen = threading.Event()
  initial_ids: list[str] = []
  drained_ids: list[str] = []

  original_preflight = batches_module._preflight_batch_approval_cancellation

  async def paused_preflight(*, projections, authenticated):
    initial_ids.extend(projection.approval_id for projection in projections)
    preflight_entered.set()
    deadline = time.monotonic() + 2
    while not publication_seen.is_set():
      assert time.monotonic() < deadline
      await asyncio.sleep(0.001)
    await original_preflight(
      projections=projections,
      authenticated=authenticated,
    )

  monkeypatch.setattr(
    batches_module,
    "_preflight_batch_approval_cancellation",
    paused_preflight,
  )
  original_publish = projections._publish_admitted_carrier

  def tracked_publish(admission):
    original_publish(admission)
    publication_seen.set()

  monkeypatch.setattr(projections, "_publish_admitted_carrier", tracked_publish)
  original_snapshot_after_drain = projections.snapshot_after_batch_drain

  async def tracked_snapshot_after_drain(**kwargs):
    snapshot = await original_snapshot_after_drain(**kwargs)
    drained_ids.extend(projection.approval_id for projection in snapshot)
    return snapshot

  monkeypatch.setattr(
    projections,
    "snapshot_after_batch_drain",
    tracked_snapshot_after_drain,
  )

  async def run_waiting_batch(
    batch_id: int,
    spec: dict[str, Any],
    *,
    registry: BatchRegistry,
    capability_execution_resolver: Any,
    session_driver_execution: Any,
    captured_run_admission_factory: Any,
    _identity: tuple[str, str | None] | None = None,
    _on_finalize=None,
  ) -> dict[str, Any]:
    _ = (
      spec,
      capability_execution_resolver,
      session_driver_execution,
      captured_run_admission_factory,
      _identity,
      _on_finalize,
    )
    scope = current_batch_approval_scope()
    assert scope is not None
    session = SimpleNamespace(
      session_id="delete-transient-publication-stage",
      user_id="alice",
      channel="tui",
      role="owner",
      batch_stage_run_seq=3,
      pending_tools={},
      approval_queues={},
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    policy.session = session
    dispatcher = ToolDispatcher(
      mcp_client=SimpleNamespace(),
      local_tool_handlers={},
      event_log=EventLog(),
      session=session,
      store=app.state.gateway_approval_store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id=f"batch_{batch_id}",
        run_id=f"batch_{batch_id}",
        session_id=session.session_id,
        profile="analyst",
        channel="tui",
        decider_role="owner",
      ),
    )
    try:
      await dispatcher._run_approval_lifecycle(
        tool_call_id="tool-delete-transient-publication",
        tool_name="memory_write",
        tool_input={},
        qualifier="",
        reason="transient publication race",
        allow_persistent=False,
      )
    finally:
      policy.finished.set()
    return {"batch_id": batch_id, "status": "completed"}

  monkeypatch.setattr(fake_batch_control, "run_acquired_batch", run_waiting_batch)
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    assert policy.entered.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=1) as executor:
      cancel_future = executor.submit(
        client.delete,
        f"/api/control/batches/{batch_id}",
        headers=headers,
      )
      assert preflight_entered.wait(timeout=2)
      policy.allow_decision.set()
      assert publication_seen.wait(timeout=2)
      cancel_response = cancel_future.result(timeout=3)

    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["batch"]["status"] == "cancelled"
    assert initial_ids == [policy.request.approval_id]
    assert policy.request is not None
    assert drained_ids == [policy.request.approval_id]
    stored = _run(app.state.gateway_approval_store.get(policy.request.approval_id))
    assert stored is not None and stored.state == "denied"
    assert stored.decision_reason == "batch_cancelled"
    assert policy.resolved == [policy.request.approval_id]
    assert policy.finished.wait(timeout=1)
    assert policy.session.pending_tools == {}
    assert policy.session.approval_queues == {}
    assert projections.projections_for_batch(
      owner_user_id="alice",
      batch_id=batch_id,
    ) == []


def test_control_batches_cancel_denies_pending_approval_before_task_teardown(
  fake_batch_control: FakeBatchController,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client)
    headers = _headers(control)
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, queue, _batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)
    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["batch"]["status"] == "cancelled"
    assert _run(app.state.gateway_approval_store.get(request_record.approval_id)).state == "denied"
    assert queue.get_nowait() == {
      "approved": False,
      "allow_tool_type": False,
      "approval_id": request_record.approval_id,
    }
    assert app.state.batch_task_registry.approval_projections.projections_for_owner(
      owner_user_id="alice",
      channel="tui",
    ) == []


def test_control_approval_list_exposes_exact_batch_stage_identity(
  fake_batch_control: FakeBatchController,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, _queue, _batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
      stage_run_seq=3,
    )

    listed = client.get("/api/control/approvals", headers=headers)

    assert listed.status_code == 200, listed.text
    approval = next(
      item
      for item in listed.json()["approvals"]
      if item["approval_id"] == request_record.approval_id
    )
    assert approval["batch_id"] == batch_id
    assert approval["run_id"] == f"batch_{batch_id}"
    assert approval["stage_run_seq"] == 3

    cancelled = client.delete(f"/api/control/batches/{batch_id}", headers=headers)
    assert cancelled.status_code == 200, cancelled.text


def test_control_batches_cancel_quarantines_projection_without_mutating_other_owner(
  fake_batch_control: FakeBatchController,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, queue, _batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
      durable_owner_user_id="bob",
    )
    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)
    assert cancel_response.status_code == 409
    assert cancel_response.json()["detail"]["error"] == (
      "Batch approval durable identity mismatch quarantined"
    )
    detail = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    assert detail.json()["batch"]["status"] == "cancelled"
    stored = _run(app.state.gateway_approval_store.get(request_record.approval_id))
    assert stored.user_id == "bob"
    assert stored.state == "pending_user"
    assert stored.state_version == 0
    assert stored.decider_id is None
    assert stored.decision_reason is None
    assert queue.get_nowait()["approved"] is False
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is None


def test_control_batches_cancel_uses_projection_authoritative_store_not_mutated_session_store(
  fake_batch_control: FakeBatchController,
  tmp_path: Path,
) -> None:
  from agent_gateway.approval_store import SQLiteApprovalStore

  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    alice_record, queue, batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )
    bob_store = SQLiteApprovalStore(tmp_path / "bob-approvals.sqlite3")
    bob_record = replace(alice_record, user_id="bob")
    _run(bob_store.create(bob_record))
    batch_session.approval_store = bob_store

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 200, cancel_response.text
    assert _run(app.state.gateway_approval_store.get(alice_record.approval_id)).state == "denied"
    preserved = _run(bob_store.get(bob_record.approval_id))
    assert preserved.user_id == "bob"
    assert preserved.state == "pending_user"
    assert preserved.state_version == 0
    assert queue.get_nowait()["approved"] is False


def test_control_batches_cancel_respects_policy_role_authorization(
  fake_batch_control: FakeBatchController,
) -> None:
  class _OwnerOnlyCancellationPolicy:
    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

    def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
      _ = tool_class
      return decider_role == "owner"

  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  app.state.gateway_approval_policy = _OwnerOnlyCancellationPolicy()
  with TestClient(app) as client:
    control = _control_session(client)
    control_session = app.state.auth.session_store.get_session(control["session_id"])
    assert control_session is not None
    control_session.role = "invite"
    control["session_token"] = app.state.auth.issue_token(control_session)
    headers = _headers(control)
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, queue, batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )
    task = app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    )
    assert task is not None

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 403
    assert cancel_response.json()["detail"]["error"] == (
      "Role is not authorized to cancel this approval"
    )
    stored = _run(app.state.gateway_approval_store.get(request_record.approval_id))
    assert stored.state == "pending_user"
    assert queue.empty()
    detail = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    assert detail.json()["batch"]["status"] == "running"
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is task
    assert not task.done()
    projections = app.state.batch_task_registry.approval_projections.projections_for_batch(
      owner_user_id="alice",
      batch_id=batch_id,
    )
    assert [projection.approval_id for projection in projections] == [
      request_record.approval_id
    ]
    pending_entry = batch_session.pending_tools[request_record.tool_call_id]
    assert pending_entry["status"] == "approval_pending"
    assert "_approval_cancel_requested" not in pending_entry


def test_http_batch_cancel_transient_preflight_read_failure_cannot_bypass_irreversible_role(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _InviteRestrictedPolicy:
    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

    def role_authorized_for_class(
      self,
      *,
      decider_role: str | None,
      tool_class: str,
    ) -> bool:
      return decider_role != "invite" or tool_class != "irreversible"

  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  app.state.gateway_approval_policy = _InviteRestrictedPolicy()
  with TestClient(app) as client:
    control = _control_session(client)
    control_session = app.state.auth.session_store.get_session(control["session_id"])
    assert control_session is not None
    control_session.role = "invite"
    control["session_token"] = app.state.auth.issue_token(control_session)
    headers = _headers(control)
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, queue, batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
      tool_class="irreversible",
    )
    task = app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    )
    assert task is not None and not task.done()
    store = app.state.gateway_approval_store
    original_get = store.get
    reads = 0

    async def fail_first_read(approval_id: str) -> Any:
      nonlocal reads
      reads += 1
      if reads == 1:
        raise OSError("transient approval store read failure")
      return await original_get(approval_id)

    monkeypatch.setattr(store, "get", fail_first_read)
    cancel_response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

    assert cancel_response.status_code == 503, cancel_response.text
    assert cancel_response.json()["detail"] == {
      "error": "Batch approval cancellation preflight failed"
    }
    stored = _run(original_get(request_record.approval_id))
    assert stored is not None and stored.state == "pending_user"
    assert stored.state_version == 0
    assert stored.decision is None
    assert stored.decision_reason is None
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is task
    assert not task.done()
    pending_entry = batch_session.pending_tools[request_record.tool_call_id]
    assert pending_entry["status"] == "approval_pending"
    assert "_approval_cancel_requested" not in pending_entry
    assert queue.empty()
    assert ("alice", batch_id) not in (
      app.state.batch_task_registry._cancel_admission_fences
    )
    assert ("alice", batch_id) not in (
      app.state.batch_task_registry.approval_projections._batch_admission_fences
    )


def test_control_batches_cancel_rejects_late_projection_after_frozen_preflight(
  fake_batch_control: FakeBatchController,
) -> None:
  class _MixedCancellationPolicy:
    def __init__(self) -> None:
      self.create_late_projection: Any = None

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      if request.tool_class == "state_write" and self.create_late_projection is not None:
        callback = self.create_late_projection
        self.create_late_projection = None
        await callback()

    def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
      return decider_role != "invite" or tool_class == "state_write"

  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  policy = _MixedCancellationPolicy()
  app.state.gateway_approval_policy = policy
  with TestClient(app) as client:
    control = _control_session(client)
    control_session = app.state.auth.session_store.get_session(control["session_id"])
    assert control_session is not None
    control_session.role = "invite"
    control["session_token"] = app.state.auth.issue_token(control_session)
    headers = _headers(control)
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    first_record, first_queue, first_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )
    task = app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    )
    assert task is not None
    late: dict[str, Any] = {}

    async def create_late_projection() -> None:
      suffix = uuid.uuid4().hex
      late_record = replace(
        first_record,
        approval_id=f"approval-late-unauthorized-{suffix}",
        tool_call_id=f"tool-late-unauthorized-{suffix}",
        approval_chain_id=f"approval-late-unauthorized-{suffix}",
        session_id=f"batch-stage-late-unauthorized-{suffix}",
        tool_class="irreversible",
        args_hash=f"late-unauthorized-hash-{suffix}",
      )
      late_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
      late_pending = {
        "approval_id": late_record.approval_id,
        "nonce": "late-unauthorized-nonce",
        "status": "approval_pending",
      }
      late_session = SimpleNamespace(
        session_id=late_record.session_id,
        user_id="alice",
        channel="tui",
        approval_store=app.state.gateway_approval_store,
        approval_policy=policy,
        pending_tools={late_record.tool_call_id: late_pending},
        approval_queues={late_record.tool_call_id: late_queue},
      )
      late.update(
        record=late_record,
        queue=late_queue,
        pending=late_pending,
      )
      try:
        app.state.batch_task_registry.approval_projections.register_session(
          batch_id=batch_id,
          owner_user_id="alice",
          channel="tui",
          session=late_session,
          store=app.state.gateway_approval_store,
          policy=policy,
        )
      except RuntimeError as exc:
        late["error"] = str(exc)
        return
      await app.state.gateway_approval_store.create(late_record)

    policy.create_late_projection = create_late_projection

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 200
    assert "admission is fenced" in late["error"]
    first_stored = _run(app.state.gateway_approval_store.get(first_record.approval_id))
    assert first_stored.state == "denied"
    assert first_queue.get_nowait()["approved"] is False
    assert first_session.pending_tools[first_record.tool_call_id]["status"] == (
      "approval_received"
    )
    late_stored = _run(app.state.gateway_approval_store.get(late["record"].approval_id))
    assert late_stored is None
    assert late["queue"].empty()
    assert late["pending"]["status"] == "approval_pending"
    assert "_approval_cancel_requested" not in late["pending"]
    detail = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    assert detail.json()["batch"]["status"] == "cancelled"
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is None
    assert task.done()
    projections = app.state.batch_task_registry.approval_projections.projections_for_batch(
      owner_user_id="alice",
      batch_id=batch_id,
    )
    assert projections == []


def test_control_batches_cancel_quarantines_late_row_on_preregistered_empty_carrier(
  fake_batch_control: FakeBatchController,
) -> None:
  class _LateApprovalPolicy:
    def __init__(self) -> None:
      self.create_late_approval: Any = None

    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      if self.create_late_approval is not None:
        callback = self.create_late_approval
        self.create_late_approval = None
        await callback(request)

    def role_authorized_for_class(
      self,
      *,
      decider_role: str | None,
      tool_class: str,
    ) -> bool:
      _ = decider_role, tool_class
      return True

  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  policy = _LateApprovalPolicy()
  app.state.gateway_approval_policy = policy
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    first_record, first_queue, _first_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )
    store = app.state.gateway_approval_store
    late_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    empty_session = SimpleNamespace(
      session_id="pre-registered-empty-stage",
      user_id="alice",
      channel="tui",
      approval_store=store,
      approval_policy=policy,
      pending_tools={},
      approval_queues={},
    )
    app.state.batch_task_registry.approval_projections.register_session(
      batch_id=batch_id,
      owner_user_id="alice",
      channel="tui",
      session=empty_session,
      store=store,
      policy=policy,
    )
    late_record = replace(
      first_record,
      approval_id=f"approval-late-existing-{uuid.uuid4().hex}",
      tool_call_id=f"tool-late-existing-{uuid.uuid4().hex}",
      approval_chain_id=f"approval-late-existing-{uuid.uuid4().hex}",
      session_id=empty_session.session_id,
      args_hash=f"late-existing-hash-{uuid.uuid4().hex}",
    )

    async def create_late_approval(_resolved: ApprovalRequest) -> None:
      await store.create(late_record)
      empty_session.pending_tools[late_record.tool_call_id] = {
        "approval_id": late_record.approval_id,
        "nonce": "late-existing-nonce",
        "status": "approval_pending",
      }
      empty_session.approval_queues[late_record.tool_call_id] = late_queue

    policy.create_late_approval = create_late_approval

    cancel_response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

    assert cancel_response.status_code == 200, cancel_response.text
    assert _run(store.get(first_record.approval_id)).state == "denied"
    late_stored = _run(store.get(late_record.approval_id))
    assert late_stored.state == "denied"
    assert late_stored.decision_reason == "batch_cancelled"
    assert first_queue.get_nowait()["approved"] is False
    assert late_queue.get_nowait()["approved"] is False
    assert empty_session.pending_tools[late_record.tool_call_id]["status"] == (
      "approval_received"
    )
    assert app.state.batch_task_registry.approval_projections.projections_for_batch(
      owner_user_id="alice",
      batch_id=batch_id,
    ) == []


def test_control_batches_unauthorized_cancel_with_late_registration_changes_nothing(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class _MixedCancellationPolicy:
    async def on_resolve(self, *, request: ApprovalRequest) -> None:
      _ = request

    def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
      return decider_role != "invite" or tool_class == "state_write"

  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  policy = _MixedCancellationPolicy()
  app.state.gateway_approval_policy = policy
  with TestClient(app) as client:
    control = _control_session(client)
    control_session = app.state.auth.session_store.get_session(control["session_id"])
    assert control_session is not None
    control_session.role = "invite"
    control["session_token"] = app.state.auth.issue_token(control_session)
    headers = _headers(control)
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    allowed, allowed_queue, allowed_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )
    denied, denied_queue, denied_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
      tool_class="irreversible",
    )
    task = app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    )
    assert task is not None
    projection_registry = app.state.batch_task_registry.approval_projections
    before_projection_ids = [
      projection.approval_id
      for projection in projection_registry.projections_for_batch(
        owner_user_id="alice",
        batch_id=batch_id,
      )
    ]
    before_carrier_keys = set(projection_registry._carriers)
    original_get = app.state.gateway_approval_store.get
    late: dict[str, Any] = {}

    async def get_with_late_registration(approval_id: str) -> Any:
      if not late:
        late_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        late_session = SimpleNamespace(
          session_id="late-rejected-session",
          user_id="alice",
          channel="tui",
          approval_store=app.state.gateway_approval_store,
          approval_policy=policy,
          pending_tools={
            "late-rejected-tool": {
              "approval_id": "late-rejected-approval",
              "nonce": "late-rejected-nonce",
              "status": "approval_pending",
            }
          },
          approval_queues={"late-rejected-tool": late_queue},
        )
        late.update(queue=late_queue, session=late_session)
        try:
          projection_registry.register_session(
            batch_id=batch_id,
            owner_user_id="alice",
            channel="tui",
            session=late_session,
            store=app.state.gateway_approval_store,
            policy=policy,
          )
        except RuntimeError as exc:
          late["error"] = str(exc)
      return await original_get(approval_id)

    monkeypatch.setattr(app.state.gateway_approval_store, "get", get_with_late_registration)
    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 403
    assert cancel_response.json()["detail"]["approval_id"] == denied.approval_id
    assert "admission is fenced" in late["error"]
    for request_record in (allowed, denied):
      stored = _run(original_get(request_record.approval_id))
      assert stored.state == "pending_user"
      assert stored.state_version == 0
    assert _run(original_get("late-rejected-approval")) is None
    assert allowed_queue.empty()
    assert denied_queue.empty()
    assert late["queue"].empty()
    for request_record, session in (
      (allowed, allowed_session),
      (denied, denied_session),
    ):
      pending = session.pending_tools[request_record.tool_call_id]
      assert pending["status"] == "approval_pending"
      assert "_approval_cancel_requested" not in pending
    detail = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    assert detail.json()["batch"]["status"] == "running"
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is task
    assert not task.done()
    after_projection_ids = [
      projection.approval_id
      for projection in projection_registry.projections_for_batch(
        owner_user_id="alice",
        batch_id=batch_id,
      )
    ]
    assert after_projection_ids == before_projection_ids
    assert set(projection_registry._carriers) == before_carrier_keys


def test_control_batches_cancel_terminalizes_pending_approval(
  fake_batch_control: FakeBatchController,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, queue, _batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 200, cancel_response.text
    stored = _run(app.state.gateway_approval_store.get(request_record.approval_id))
    assert stored.state == "denied"
    assert queue.get_nowait()["approved"] is False


def test_control_batches_cancel_missing_durable_row_still_settles_task(
  fake_batch_control: FakeBatchController,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session = SimpleNamespace(
      session_id="batch-stage-missing-endpoint",
      user_id="alice",
      channel="tui",
      approval_store=app.state.gateway_approval_store,
      approval_policy=app.state.gateway_approval_policy,
      pending_tools={
        "tool-missing-endpoint": {
          "approval_id": "approval-missing-endpoint",
          "nonce": "missing-endpoint-nonce",
          "status": "approval_pending",
        }
      },
      approval_queues={"tool-missing-endpoint": queue},
    )
    app.state.batch_task_registry.approval_projections.register_session(
      batch_id=batch_id,
      owner_user_id="alice",
      channel="tui",
      session=session,
    )

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 404
    assert cancel_response.json()["detail"]["error"] == "Approval request not found"
    detail = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    assert detail.json()["batch"]["status"] == "cancelled"
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is None
    assert queue.get_nowait()["approved"] is False


def test_control_batches_cancel_store_failure_after_preflight_still_settles_task(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, queue, batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )
    store = app.state.gateway_approval_store
    original_get = store.get
    reads = 0

    async def unavailable(approval_id: str) -> Any:
      nonlocal reads
      reads += 1
      if reads == 1:
        return await original_get(approval_id)
      raise OSError("approval store unavailable")

    monkeypatch.setattr(store, "get", unavailable)
    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 503
    assert cancel_response.headers["X-Batch-Cancellation-Committed"] == "true"
    assert cancel_response.json()["detail"]["error"] == (
      "Batch approval cancellation quarantine failed"
    )
    assert reads >= 2
    stored = _run(original_get(request_record.approval_id))
    assert stored.state == "pending_user"
    detail = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    assert detail.json()["batch"]["status"] == "cancelled"
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is None
    assert app.state.batch_task_registry.approval_projections.projections_for_batch(
      owner_user_id="alice",
      batch_id=batch_id,
    ) == []
    assert batch_session.pending_tools[request_record.tool_call_id]["status"] == (
      "approval_received"
    )
    assert queue.get_nowait()["approved"] is False


def test_control_batches_cancel_does_not_regress_completed_batch_after_approval_await(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])

    async def complete_while_cancel_waits(**kwargs: Any) -> None:
      registry = batches_module._registry_for_user(kwargs["owner_user_id"])
      try:
        registry.set_status(kwargs["batch_id"], "completed")
      finally:
        registry.close()
      await asyncio.sleep(0)

    monkeypatch.setattr(
      batches_module,
      "_deny_batch_pending_approvals_for_cancel",
      complete_while_cancel_waits,
    )

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["batch"]["status"] == "completed"
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is None


def test_batch_approval_cancel_interleaving_never_releases_approved_tool(
  tmp_path: Path,
) -> None:
  async def run_case() -> None:
    from agent_gateway.approval_store import SQLiteApprovalStore

    class _Policy:
      async def on_resolve(self, *, request: ApprovalRequest) -> None:
        _ = request

      def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
        _ = decider_role, tool_class
        return True

    store = SQLiteApprovalStore(tmp_path / "approval-race.sqlite3")
    policy = _Policy()
    request_record = ApprovalRequest(
      approval_id="approval-race",
      tool_call_id="tool-race",
      parent_approval_id=None,
      approval_chain_id="approval-race",
      request_id="batch_1",
      session_id="batch-stage-race",
      run_id="batch_1",
      user_id="alice",
      profile="analyst",
      channel="tui",
      tool_name="memory_write",
      tool_class="state_write",
      tool_args_redacted={},
      args_hash="race-hash",
      reason="requires approval",
      blast_radius_summary="state_write:memory_write",
      approval_constraint="standard",
      state="pending_user",
      requested_at=utc_now(),
    )
    await store.create(request_record)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    pending_entry = {
      "approval_id": request_record.approval_id,
      "nonce": "race-nonce",
      "status": "approval_pending",
    }
    session = SimpleNamespace(
      user_id="alice",
      approval_store=store,
      approval_policy=policy,
      pending_tools={request_record.tool_call_id: pending_entry},
      approval_queues={request_record.tool_call_id: queue},
    )
    durable_approved = asyncio.Event()
    release_vote = asyncio.Event()
    original_record_vote = store.record_vote

    async def blocked_record_vote(approval_id: str, vote: Any) -> ApprovalRequest:
      resolved = await original_record_vote(approval_id, vote)
      durable_approved.set()
      await release_vote.wait()
      return resolved

    store.record_vote = blocked_record_vote  # type: ignore[method-assign]
    approve_task = asyncio.create_task(
      approvals_module._record_vote_and_unblock(
        target_session=session,
        pending_entry=pending_entry,
        tool_call_id=request_record.tool_call_id,
        nonce="race-nonce",
        decider_id="alice",
        decider_role="owner",
        approved=True,
        allow_tool_type=False,
        reason="approved concurrently",
        app_state=SimpleNamespace(),
      )
    )
    await durable_approved.wait()
    cancel_task = asyncio.create_task(
      approvals_module._cancel_pending_approval_and_unblock(
        target_session=session,
        pending_entry=pending_entry,
        tool_call_id=request_record.tool_call_id,
        nonce="race-nonce",
        decider_id="alice",
        decider_role="owner",
        reason="batch_cancelled",
        app_state=SimpleNamespace(),
        authoritative_store=store,
        authoritative_policy=policy,
        expected_owner_user_id="alice",
        expected_request_id="batch_1",
        expected_run_id="batch_1",
        expected_session_id="batch-stage-race",
        expected_channel="tui",
      )
    )
    await asyncio.sleep(0)
    assert pending_entry["_approval_cancel_requested"] is True
    release_vote.set()
    approval_result, cancel_result = await asyncio.gather(approve_task, cancel_task)

    assert approval_result["status"] == "cancellation_pending"
    assert cancel_result["quarantined"] is True
    assert (await store.get(request_record.approval_id)).state == "approved"
    assert queue.get_nowait() == {
      "approved": False,
      "allow_tool_type": False,
      "approval_id": request_record.approval_id,
    }
    assert queue.empty()

  asyncio.run(run_case())


def test_http_approval_handler_racing_batch_cancel_never_releases_approved_tool(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, queue, batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )
    store = app.state.gateway_approval_store
    original_record_vote = store.record_vote
    vote_entered = threading.Event()
    release_vote = threading.Event()

    async def paused_record_vote(approval_id: str, vote: Any) -> Any:
      vote_entered.set()
      while not release_vote.is_set():
        await asyncio.sleep(0.001)
      return await original_record_vote(approval_id, vote)

    monkeypatch.setattr(store, "record_vote", paused_record_vote)

    with ThreadPoolExecutor(max_workers=2) as executor:
      approval_future = executor.submit(
        client.post,
        f"/api/control/runs/batch_{batch_id}/approvals/{request_record.approval_id}",
        headers=headers,
        json={"approved": True, "reason": "race approval"},
      )
      assert vote_entered.wait(timeout=2)
      cancel_future = executor.submit(
        client.delete,
        f"/api/control/batches/{batch_id}",
        headers=headers,
      )
      pending_entry = batch_session.pending_tools[request_record.tool_call_id]
      deadline = time.monotonic() + 2
      while pending_entry.get("_approval_cancel_requested") is not True:
        assert time.monotonic() < deadline
        time.sleep(0.005)
      release_vote.set()
      approval_response = approval_future.result(timeout=3)
      cancel_response = cancel_future.result(timeout=3)

    assert approval_response.status_code == 200, approval_response.text
    assert approval_response.json()["status"] == "cancellation_pending"
    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["batch"]["status"] == "cancelled"
    stored = _run(store.get(request_record.approval_id))
    assert stored.state == "approved"
    assert queue.get_nowait() == {
      "approved": False,
      "allow_tool_type": False,
      "approval_id": request_record.approval_id,
    }
    assert queue.empty()


def test_http_persistent_grant_creation_racing_batch_cancel_revokes_grant(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    scope_hint = f"state_write:memory_write:{uuid.uuid4().hex}"
    request_record, queue, batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
      persistent_grant_scope=scope_hint,
    )
    store = app.state.gateway_approval_store
    original_create_grant = store.create_persistent_grant
    original_fence_grants = store.fence_persistent_grants_for_cancellation
    grant_committed = threading.Event()
    fence_committed = threading.Event()
    release_grant_creator = threading.Event()
    post_fence_lookup: dict[str, Any] = {}

    async def paused_create_grant(grant: Any) -> Any:
      created = await original_create_grant(grant)
      grant_committed.set()
      while not release_grant_creator.is_set():
        await asyncio.sleep(0.001)
      return created

    async def tracked_fence_grants(*args: Any, **kwargs: Any) -> Any:
      result = await original_fence_grants(*args, **kwargs)
      post_fence_lookup["grant"] = await store.find_persistent_grant(
        user_id="alice",
        tool_name=request_record.tool_name,
        scope_hint=scope_hint,
      )
      fence_committed.set()
      return result

    monkeypatch.setattr(store, "create_persistent_grant", paused_create_grant)
    monkeypatch.setattr(
      store,
      "fence_persistent_grants_for_cancellation",
      tracked_fence_grants,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
      approval_future = executor.submit(
        client.post,
        f"/api/control/runs/batch_{batch_id}/approvals/{request_record.approval_id}",
        headers=headers,
        json={
          "approved": True,
          "allow_tool_type": True,
          "reason": "persistent race approval",
        },
      )
      assert grant_committed.wait(timeout=2)
      cancel_future = executor.submit(
        client.delete,
        f"/api/control/batches/{batch_id}",
        headers=headers,
      )
      pending_entry = batch_session.pending_tools[request_record.tool_call_id]
      deadline = time.monotonic() + 2
      while pending_entry.get("_approval_cancel_requested") is not True:
        assert time.monotonic() < deadline
        time.sleep(0.005)
      assert fence_committed.wait(timeout=2)
      assert post_fence_lookup["grant"] is None
      release_grant_creator.set()
      approval_response = approval_future.result(timeout=3)
      cancel_response = cancel_future.result(timeout=3)

    assert approval_response.status_code == 200, approval_response.text
    assert approval_response.json()["status"] == "cancellation_pending"
    assert cancel_response.status_code == 200, cancel_response.text
    assert queue.get_nowait()["approved"] is False
    assert _run(store.find_persistent_grant(
      user_id="alice",
      tool_name=request_record.tool_name,
      scope_hint=scope_hint,
    )) is None


def test_batch_cancel_survives_post_commit_audit_and_policy_cancelled_error(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake_batch_control.wait_for_cancel = True
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches", headers=headers, json=_batch_spec())
    batch_id = int(response.json()["batch_id"])
    request_record, queue, _batch_session = _install_batch_pending_approval(
      app=app,
      batch_id=batch_id,
    )
    store = app.state.gateway_approval_store
    original_emit = store._emit

    async def cancel_after_durable_commit(
      event_type: str,
      request: ApprovalRequest,
      **kwargs: Any,
    ) -> None:
      if event_type == "denied" and request.approval_id == request_record.approval_id:
        raise asyncio.CancelledError
      await original_emit(event_type, request, **kwargs)

    async def cancelled_policy_callback(*, request: ApprovalRequest) -> None:
      if request.approval_id == request_record.approval_id:
        raise asyncio.CancelledError

    monkeypatch.setattr(store, "_emit", cancel_after_durable_commit)
    monkeypatch.setattr(
      app.state.gateway_approval_policy,
      "on_resolve",
      cancelled_policy_callback,
    )

    cancel_response = client.delete(
      f"/api/control/batches/{batch_id}",
      headers=headers,
    )

    assert cancel_response.status_code == 200, cancel_response.text
    stored = _run(store.get(request_record.approval_id))
    assert stored.state == "denied"
    assert queue.get_nowait()["approved"] is False
    assert app.state.batch_task_registry.get(
      owner_user_id="alice",
      batch_id=batch_id,
    ) is None


def test_batch_task_registry_cancelled_task_finishes_cancelled(tmp_path: Path) -> None:
  async def run_case() -> None:
    registry = BatchRegistry(tmp_path / "cancelled-task.db")
    batch_id = _acquire_test_batch(registry,
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
    await task_registry._consume(("alice", batch_id), task, registry)

    digest = registry.get_batch_digest(batch_id)
    assert digest["status"] == "cancelled"
    assert digest["error"] is None
    registry.close()

  asyncio.run(run_case())


def test_batch_task_monitor_terminal_transition_cannot_overwrite_concurrent_completion(
  tmp_path: Path,
) -> None:
  registry = BatchRegistry(tmp_path / "monitor-status-race.db")
  batch_id = _acquire_test_batch(registry,
    user_id="alice",
    host="test-host",
    spec=_batch_spec(),
    budget_usd=1.0,
    pid=None,
  )
  original_transition = registry.transition_status_if_current

  def complete_before_compare_and_set(*args: Any, **kwargs: Any) -> bool:
    registry.set_status(batch_id, "completed")
    return original_transition(*args, **kwargs)

  registry.transition_status_if_current = complete_before_compare_and_set  # type: ignore[method-assign]

  batches_module._set_status_if_not_terminal(registry, batch_id, "cancelled")

  assert registry.get_batch_digest(batch_id)["status"] == "completed"
  registry.close()


def test_batch_task_registry_shutdown_terminalizes_pending_before_projection_clear(
  tmp_path: Path,
) -> None:
  async def run_case() -> None:
    from agent_gateway.approval_store import SQLiteApprovalStore

    class _Policy:
      def __init__(self) -> None:
        self.resolved: list[str] = []

      async def on_resolve(self, *, request: ApprovalRequest) -> None:
        self.resolved.append(request.approval_id)

    registry = BatchRegistry(tmp_path / "shutdown-batch.db")
    batch_id = _acquire_test_batch(registry,
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )
    store = SQLiteApprovalStore(tmp_path / "shutdown-approvals.sqlite3")
    policy = _Policy()
    request_record = ApprovalRequest(
      approval_id="approval-shutdown",
      tool_call_id="tool-shutdown",
      parent_approval_id=None,
      approval_chain_id="approval-shutdown",
      request_id=f"batch_{batch_id}",
      session_id="batch-stage-shutdown",
      run_id=f"batch_{batch_id}",
      user_id="alice",
      profile="analyst",
      channel="tui",
      tool_name="memory_write",
      tool_class="state_write",
      tool_args_redacted={},
      args_hash="shutdown-hash",
      reason="requires approval",
      blast_radius_summary="state_write:memory_write",
      approval_constraint="standard",
      state="pending_user",
      requested_at=utc_now(),
    )
    await store.create(request_record)
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session = SimpleNamespace(
      session_id=request_record.session_id,
      user_id="alice",
      channel="tui",
      approval_store=store,
      approval_policy=policy,
      pending_tools={
        request_record.tool_call_id: {
          "approval_id": request_record.approval_id,
          "nonce": "shutdown-nonce",
          "status": "approval_pending",
        }
      },
      approval_queues={request_record.tool_call_id: queue},
    )
    task_registry = batches_module.BatchTaskRegistry()
    task_registry.approval_projections.register_session(
      batch_id=batch_id,
      owner_user_id="alice",
      channel="tui",
      session=session,
    )
    never_finishes = asyncio.Event()
    task_registry.start(
      owner_user_id="alice",
      batch_id=batch_id,
      task=asyncio.create_task(never_finishes.wait()),
      registry=registry,
    )

    await task_registry.shutdown()

    stored = await store.get(request_record.approval_id)
    assert stored is not None
    assert stored.state == "denied"
    assert stored.decision_reason == "gateway_shutdown"
    assert policy.resolved == [request_record.approval_id]
    assert queue.get_nowait()["approved"] is False
    assert task_registry.approval_projections.batch_keys() == set()

  asyncio.run(run_case())


def test_batch_shutdown_quarantines_late_row_on_preregistered_empty_carrier(
  tmp_path: Path,
) -> None:
  async def run_case() -> None:
    class Policy:
      def __init__(self) -> None:
        self.create_late_approval: Any = None

      async def on_resolve(self, *, request: ApprovalRequest) -> None:
        if self.create_late_approval is not None:
          callback = self.create_late_approval
          self.create_late_approval = None
          await callback(request)

    registry = BatchRegistry(tmp_path / "shutdown-late-existing-batch.db")
    batch_id = _acquire_test_batch(registry,
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )
    store = SQLiteApprovalStore(tmp_path / "shutdown-late-existing-approvals.sqlite3")
    policy = Policy()
    task_registry = batches_module.BatchTaskRegistry()
    first_record = ApprovalRequest(
      approval_id="approval-shutdown-first-existing",
      tool_call_id="tool-shutdown-first-existing",
      parent_approval_id=None,
      approval_chain_id="approval-shutdown-first-existing",
      request_id=f"batch_{batch_id}",
      session_id="shutdown-first-stage",
      run_id=f"batch_{batch_id}",
      user_id="alice",
      profile="analyst",
      channel="tui",
      tool_name="memory_write",
      tool_class="state_write",
      tool_args_redacted={},
      args_hash="shutdown-first-existing-hash",
      reason="requires approval",
      blast_radius_summary="state_write:memory_write",
      approval_constraint="standard",
      state="pending_user",
      requested_at=utc_now(),
    )
    late_record = replace(
      first_record,
      approval_id="approval-shutdown-late-existing",
      tool_call_id="tool-shutdown-late-existing",
      approval_chain_id="approval-shutdown-late-existing",
      session_id="shutdown-empty-stage",
      args_hash="shutdown-late-existing-hash",
    )
    await store.create(first_record)
    first_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    late_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    first_session = SimpleNamespace(
      session_id=first_record.session_id,
      user_id="alice",
      channel="tui",
      pending_tools={
        first_record.tool_call_id: {
          "approval_id": first_record.approval_id,
          "nonce": "shutdown-first-existing-nonce",
          "status": "approval_pending",
        }
      },
      approval_queues={first_record.tool_call_id: first_queue},
    )
    empty_session = SimpleNamespace(
      session_id=late_record.session_id,
      user_id="alice",
      channel="tui",
      pending_tools={},
      approval_queues={},
    )
    for session in (first_session, empty_session):
      task_registry.approval_projections.register_session(
        batch_id=batch_id,
        owner_user_id="alice",
        channel="tui",
        session=session,
        store=store,
        policy=policy,
      )

    async def create_late_approval(_resolved: ApprovalRequest) -> None:
      await store.create(late_record)
      empty_session.pending_tools[late_record.tool_call_id] = {
        "approval_id": late_record.approval_id,
        "nonce": "shutdown-late-existing-nonce",
        "status": "approval_pending",
      }
      empty_session.approval_queues[late_record.tool_call_id] = late_queue

    policy.create_late_approval = create_late_approval
    task_registry.start(
      owner_user_id="alice",
      batch_id=batch_id,
      task=asyncio.create_task(asyncio.Event().wait()),
      registry=registry,
    )

    await task_registry.shutdown()

    stored_first = await store.get(first_record.approval_id)
    stored_late = await store.get(late_record.approval_id)
    assert stored_first is not None and stored_first.state == "denied"
    assert stored_late is not None and stored_late.state == "denied"
    assert stored_first.decision_reason == "gateway_shutdown"
    assert stored_late.decision_reason == "gateway_shutdown"
    assert first_queue.get_nowait()["approved"] is False
    assert late_queue.get_nowait()["approved"] is False
    assert task_registry.approval_projections.batch_keys() == set()

  asyncio.run(run_case())


def test_batch_task_registry_shutdown_rejects_projection_created_during_teardown(
  tmp_path: Path,
) -> None:
  async def run_case() -> None:
    from agent_gateway.approval_store import SQLiteApprovalStore

    class _YieldingPolicy:
      def __init__(self) -> None:
        self.resolve_calls = 0
        self.create_late_projection: Any = None

      async def on_resolve(self, *, request: ApprovalRequest) -> None:
        _ = request
        self.resolve_calls += 1
        if self.resolve_calls == 1:
          await self.create_late_projection()

    registry = BatchRegistry(tmp_path / "late-projection-batch.db")
    batch_id = _acquire_test_batch(registry,
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )
    store = SQLiteApprovalStore(tmp_path / "late-projection-approvals.sqlite3")
    policy = _YieldingPolicy()
    task_registry = batches_module.BatchTaskRegistry()
    records: list[ApprovalRequest] = []
    queues: list[asyncio.Queue] = []
    late_registration_errors: list[str] = []

    async def create_projection(index: int, *, persist_first: bool = True) -> None:
      approval_id = f"approval-late-{index}"
      tool_call_id = f"tool-late-{index}"
      session_id = f"batch-stage-late-{index}"
      request_record = ApprovalRequest(
        approval_id=approval_id,
        tool_call_id=tool_call_id,
        parent_approval_id=None,
        approval_chain_id=approval_id,
        request_id=f"batch_{batch_id}",
        session_id=session_id,
        run_id=f"batch_{batch_id}",
        user_id="alice",
        profile="analyst",
        channel="tui",
        tool_name="memory_write",
        tool_class="state_write",
        tool_args_redacted={},
        args_hash=f"late-hash-{index}",
        reason="requires approval",
        blast_radius_summary="state_write:memory_write",
        approval_constraint="standard",
        state="pending_user",
        requested_at=utc_now(),
      )
      queue: asyncio.Queue = asyncio.Queue(maxsize=1)
      session = SimpleNamespace(
        session_id=session_id,
        user_id="alice",
        channel="tui",
        approval_store=store,
        approval_policy=policy,
        pending_tools={
          tool_call_id: {
            "approval_id": approval_id,
            "nonce": f"late-nonce-{index}",
            "status": "approval_pending",
          }
        },
        approval_queues={tool_call_id: queue},
      )
      if persist_first:
        await store.create(request_record)
      task_registry.approval_projections.register_session(
        batch_id=batch_id,
        owner_user_id="alice",
        channel="tui",
        session=session,
        store=store,
        policy=policy,
      )
      if not persist_first:
        await store.create(request_record)
      records.append(request_record)
      queues.append(queue)

    async def create_late_projection() -> None:
      await asyncio.sleep(0)
      try:
        await create_projection(3, persist_first=False)
      except RuntimeError as exc:
        late_registration_errors.append(str(exc))

    policy.create_late_projection = create_late_projection
    await create_projection(1)
    await create_projection(2)
    task = asyncio.create_task(asyncio.Event().wait())
    task_registry.start(
      owner_user_id="alice",
      batch_id=batch_id,
      task=task,
      registry=registry,
    )

    await task_registry.shutdown()

    assert [record.approval_id for record in records] == [
      "approval-late-1",
      "approval-late-2",
    ]
    assert [(await store.get(record.approval_id)).state for record in records] == [
      "denied",
      "denied",
    ]
    assert await store.get("approval-late-3") is None
    assert late_registration_errors == [
      "batch approval projection registry is shutting down"
    ]
    assert [queue.get_nowait()["approved"] for queue in queues] == [False, False]
    assert task.done()
    assert task_registry.approval_projections.batch_keys() == set()

  asyncio.run(run_case())


def test_batch_shutdown_cancels_hanging_admitted_producer_before_drain(
  tmp_path: Path,
) -> None:
  async def run_case() -> None:
    class HangingPolicy:
      policy_id = "shutdown-hanging-test"
      policy_version = "1"
      policy_bundle_hash = "shutdown-hanging-bundle"

      def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.request: ApprovalRequest | None = None

      async def decide(self, *, payload, request, run_context):
        _ = payload, run_context
        self.request = request
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("hanging approval policy unexpectedly resumed")

      async def on_resolve(self, *, request):
        _ = request

    registry = BatchRegistry(tmp_path / "shutdown-hanging-batch.db")
    batch_id = _acquire_test_batch(registry,
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )
    store = SQLiteApprovalStore(tmp_path / "shutdown-hanging-approvals.sqlite3")
    policy = HangingPolicy()
    task_registry = batches_module.BatchTaskRegistry()
    session = SimpleNamespace(
      session_id="shutdown-hanging-stage",
      user_id="alice",
      channel="tui",
      role="owner",
      batch_stage_run_seq=3,
      pending_tools={},
      approval_queues={},
    )
    scope = BatchApprovalScope(
      batch_id=batch_id,
      owner_user_id="alice",
      channel="tui",
      store=store,
      policy=policy,
      registry=task_registry.approval_projections,
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    dispatcher = ToolDispatcher(
      mcp_client=SimpleNamespace(),
      local_tool_handlers={},
      event_log=EventLog(),
      session=session,
      store=store,
      policy=policy,
      run_context=RunContext(
        user_id="alice",
        request_id=f"batch_{batch_id}",
        run_id=f"batch_{batch_id}",
        session_id=session.session_id,
        profile="analyst",
        channel="tui",
        decider_role="owner",
      ),
    )
    lifecycle_task = asyncio.create_task(
      dispatcher._run_approval_lifecycle(
        tool_call_id="tool-shutdown-hanging",
        tool_name="memory_write",
        tool_input={},
        qualifier="",
        reason="shutdown hanging producer",
        allow_persistent=False,
      )
    )
    task_registry.start(
      owner_user_id="alice",
      batch_id=batch_id,
      task=lifecycle_task,
      registry=registry,
    )
    await policy.entered.wait()

    await asyncio.wait_for(task_registry.shutdown(), timeout=2)

    assert policy.request is not None
    stored = await store.get(policy.request.approval_id)
    assert stored is not None
    assert stored.state == "denied"
    assert lifecycle_task.done()
    assert session.pending_tools == {}
    assert session.approval_queues == {}
    assert task_registry.approval_projections.batch_keys() == set()

  asyncio.run(run_case())


def test_batch_shutdown_retains_projection_published_between_snapshots(
  tmp_path: Path,
) -> None:
  async def run_case() -> None:
    class Policy:
      def __init__(self) -> None:
        self.resolved: list[str] = []

      async def on_resolve(self, *, request):
        self.resolved.append(request.approval_id)

    registry = BatchRegistry(tmp_path / "shutdown-between-snapshots-batch.db")
    batch_id = _acquire_test_batch(registry,
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )
    store = SQLiteApprovalStore(
      tmp_path / "shutdown-between-snapshots-approvals.sqlite3"
    )
    policy = Policy()
    task_registry = batches_module.BatchTaskRegistry()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session = SimpleNamespace(
      session_id="shutdown-between-snapshots-stage",
      user_id="alice",
      channel="tui",
      pending_tools={},
      approval_queues={},
    )
    scope = BatchApprovalScope(
      batch_id=batch_id,
      owner_user_id="alice",
      channel="tui",
      store=store,
      policy=policy,
      registry=task_registry.approval_projections,
    )
    scope.register_session(session)
    session.batch_approval_scope = scope
    request_record = ApprovalRequest(
      approval_id="approval-between-shutdown-snapshots",
      tool_call_id="tool-between-shutdown-snapshots",
      parent_approval_id=None,
      approval_chain_id="approval-between-shutdown-snapshots",
      request_id=f"batch_{batch_id}",
      session_id=session.session_id,
      run_id=f"batch_{batch_id}",
      user_id="alice",
      profile="analyst",
      channel="tui",
      tool_name="memory_write",
      tool_class="state_write",
      tool_args_redacted={},
      args_hash="between-shutdown-snapshots-hash",
      reason="requires approval",
      blast_radius_summary="state_write:memory_write",
      approval_constraint="standard",
      state="pending_user",
      requested_at=utc_now(),
    )
    producer_started = asyncio.Event()

    async def publish_when_shutdown_cancels() -> None:
      admission = scope.acquire_admission(session)
      admission.bind_request(request=request_record, store=store)
      producer_started.set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        await store.create(request_record)
        session.pending_tools[request_record.tool_call_id] = {
          "approval_id": request_record.approval_id,
          "nonce": "between-shutdown-snapshots-nonce",
          "status": "approval_pending",
        }
        session.approval_queues[request_record.tool_call_id] = queue
        admission.publish_pending()
        session.pending_tools.pop(request_record.tool_call_id, None)
        session.approval_queues.pop(request_record.tool_call_id, None)
      finally:
        admission.release()

    producer_task = asyncio.create_task(publish_when_shutdown_cancels())
    task_registry.start(
      owner_user_id="alice",
      batch_id=batch_id,
      task=producer_task,
      registry=registry,
    )
    await producer_started.wait()

    await asyncio.wait_for(task_registry.shutdown(), timeout=2)

    stored = await store.get(request_record.approval_id)
    assert stored is not None
    assert stored.state == "denied"
    assert stored.decision_reason == "gateway_shutdown"
    assert policy.resolved == [request_record.approval_id]
    assert queue.empty()
    assert session.pending_tools == {}
    assert session.approval_queues == {}
    assert task_registry.approval_projections.batch_keys() == set()

  asyncio.run(run_case())


def test_batch_task_registry_shutdown_fences_post_empty_task_and_projection_admission(
  tmp_path: Path,
) -> None:
  async def run_case() -> None:
    task_registry = batches_module.BatchTaskRegistry()
    shutdown_task = asyncio.create_task(task_registry.shutdown())
    # Shutdown yields after each of two empty snapshots. Resume at its final
    # yield to prove admission cannot slip in after the last empty scan.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert not shutdown_task.done()

    registry = BatchRegistry(tmp_path / "post-empty-shutdown.db")
    batch_id = _acquire_test_batch(registry,
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )
    late_task = asyncio.create_task(asyncio.Event().wait())
    with pytest.raises(RuntimeError, match="admission is closed"):
      task_registry.start(
        owner_user_id="alice",
        batch_id=batch_id,
        task=late_task,
        registry=registry,
      )
    await asyncio.sleep(0)
    assert late_task.cancelled()

    late_session = SimpleNamespace(
      session_id="post-empty-stage",
      user_id="alice",
      channel="tui",
      pending_tools={},
      approval_queues={},
    )
    with pytest.raises(RuntimeError, match="shutting down"):
      task_registry.approval_projections.register_session(
        batch_id=batch_id,
        owner_user_id="alice",
        channel="tui",
        session=late_session,
      )

    await shutdown_task
    assert task_registry.live_count() == 0
    assert task_registry.approval_projections.batch_keys() == set()
    registry.close()

  asyncio.run(run_case())


def test_batch_task_registry_shutdown_missing_durable_row_still_settles_task(
  tmp_path: Path,
  caplog: pytest.LogCaptureFixture,
) -> None:
  async def run_case() -> None:
    from agent_gateway.approval_store import SQLiteApprovalStore

    class _Policy:
      async def on_resolve(self, *, request: ApprovalRequest) -> None:
        _ = request

    registry_path = tmp_path / "missing-row-batch.db"
    registry = BatchRegistry(registry_path)
    batch_id = _acquire_test_batch(registry,
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )
    store = SQLiteApprovalStore(tmp_path / "missing-row-approvals.sqlite3")
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    session = SimpleNamespace(
      session_id="batch-stage-missing",
      user_id="alice",
      channel="tui",
      approval_store=store,
      approval_policy=_Policy(),
      pending_tools={
        "tool-missing": {
          "approval_id": "approval-missing",
          "nonce": "missing-nonce",
          "status": "approval_pending",
        }
      },
      approval_queues={"tool-missing": queue},
    )
    task_registry = batches_module.BatchTaskRegistry()
    task_registry.approval_projections.register_session(
      batch_id=batch_id,
      owner_user_id="alice",
      channel="tui",
      session=session,
    )
    task = asyncio.create_task(asyncio.Event().wait())
    task_registry.start(
      owner_user_id="alice",
      batch_id=batch_id,
      task=task,
      registry=registry,
    )

    await task_registry.shutdown()

    assert task.done()
    assert task_registry.live_count() == 0
    assert task_registry.approval_projections.batch_keys() == set()
    assert queue.get_nowait()["approved"] is False
    assert session.pending_tools["tool-missing"]["status"] == "approval_received"
    reopened = BatchRegistry(registry_path)
    try:
      assert reopened.get_batch_digest(batch_id)["status"] == "cancelled"
    finally:
      reopened.close()

  caplog.set_level("ERROR", logger="agent_gateway.control_plane.batches")
  asyncio.run(run_case())
  assert "Approval request not found" in caplog.text


def test_batch_task_registry_does_not_collide_on_same_batch_id_across_users(tmp_path: Path) -> None:
  async def run_case() -> None:
    registries = []
    task_registry = batches_module.BatchTaskRegistry()
    gates = {"alice": asyncio.Event(), "bob": asyncio.Event()}
    for owner in gates:
      registry = BatchRegistry(tmp_path / f"{owner}.db")
      registries.append(registry)
      batch_id = _acquire_test_batch(registry,
        user_id=owner,
        host="test-host",
        spec=_batch_spec(),
        budget_usd=1.0,
        pid=None,
      )
      assert batch_id == 1

      async def wait_for_gate(user_id: str = owner) -> None:
        await gates[user_id].wait()

      task_registry.start(
        owner_user_id=owner,
        batch_id=batch_id,
        task=asyncio.create_task(wait_for_gate()),
        registry=registry,
      )

    assert task_registry.get(owner_user_id="alice", batch_id=1) is not None
    assert task_registry.get(owner_user_id="bob", batch_id=1) is not None
    assert task_registry.live_count() == 2
    assert await task_registry.cancel(owner_user_id="alice", batch_id=1)
    assert task_registry.get(owner_user_id="alice", batch_id=1) is None
    assert task_registry.get(owner_user_id="bob", batch_id=1) is not None
    await task_registry.shutdown()

  asyncio.run(run_case())


@pytest.mark.parametrize("terminal_status", ["budget_limited", "blocked"])
def test_control_batches_digest_allows_terminal_statuses(
  fake_batch_control: FakeBatchController,
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  terminal_status: str,
) -> None:
  registry_path = tmp_path / "alice.db"

  def registry_for_user(_user_id: str) -> BatchRegistry:
    return BatchRegistry(registry_path)

  monkeypatch.setattr(batches_module, "_registry_for_user", registry_for_user)
  source_registry = BatchRegistry(registry_path)
  batch_id = _acquire_test_batch(source_registry,
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
  source_registry.set_status(batch_id, terminal_status)
  source_registry.close()

  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    detail_response = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    list_response = client.get("/api/control/batches", headers=headers)

  assert detail_response.status_code == 200, detail_response.text
  detail = detail_response.json()
  assert detail["batch"]["status"] == terminal_status
  assert detail["batch"]["error"] is None
  assert detail["batch"]["counts_by_status"] == {"completed": 1, "skipped": 1}
  assert detail["failures"] == [{"ticker": "MSFT", "skill": "business-quality-assessment", "status": "skipped", "error": "budget"}]
  assert list_response.status_code == 200, list_response.text
  assert list_response.json()["batches"][0]["status"] == terminal_status


def test_batch_task_registry_does_not_overwrite_blocked_status(tmp_path: Path) -> None:
  async def run_case() -> None:
    registry = BatchRegistry(tmp_path / "blocked-task.db")
    batch_id = _acquire_test_batch(registry,
      user_id="alice",
      host="test-host",
      spec=_batch_spec(),
      budget_usd=1.0,
      pid=None,
    )
    registry.set_status(batch_id, "blocked", error="same blocker exhausted")

    async def failed_task() -> None:
      raise RuntimeError("late failure")

    task = asyncio.create_task(failed_task())
    task_registry = batches_module.BatchTaskRegistry()
    await task_registry._consume(("alice", batch_id), task, registry)

    digest = registry.get_batch_digest(batch_id)
    assert digest["status"] == "blocked"
    assert digest["error"] == "same blocker exhausted"
    registry.close()

  asyncio.run(run_case())


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
  source_batch_id = _acquire_test_batch(source_registry,
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
    replay = client.post(
      f"/api/control/batches/{source_batch_id}/retry-failed",
      headers=headers,
    )

  assert response.status_code == 200, response.text
  retry_batch_id = int(response.json()["batch_id"])
  assert replay.status_code == 200, replay.text
  assert int(replay.json()["batch_id"]) == retry_batch_id
  assert replay.json()["replayed"] is True
  assert retry_batch_id != source_batch_id
  assert fake_batch_control.acquire_calls[-1]["spec"]["universe"] == ["AAPL"]
  assert len(fake_batch_control.acquire_calls) == 1


def test_publish_batch_terminal_event_uses_batch_control_run_id() -> None:
  published: list[dict[str, Any]] = []

  class FakeBus:
    async def publish_terminal_if_absent(
      self,
      user_id: str,
      control_run_id: str,
      event: dict[str, Any],
    ) -> bool:
      published.append({"user_id": user_id, "control_run_id": control_run_id, "event": dict(event)})
      return True

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


def test_authorized_batch_cancellation_publishes_terminal_event() -> None:
  published: list[dict[str, Any]] = []
  terminalized: set[tuple[str, str]] = set()

  class FakeBus:
    async def publish_terminal_if_absent(
      self,
      user_id: str,
      control_run_id: str,
      event: dict[str, Any],
    ) -> bool:
      key = (user_id, control_run_id)
      if key in terminalized:
        return False
      terminalized.add(key)
      published.append({
        "user_id": user_id,
        "control_run_id": control_run_id,
        "event": dict(event),
      })
      return True

  class FakeRegistry:
    def __init__(self) -> None:
      self.digest = {
        "batch_id": 42,
        "status": "running",
        "user_id": "alice",
        "cost_usd": 0.0,
        "total_spent_usd": 0.0,
        "counts_by_status": {},
        "counts_by_gate_code": {},
        "counts_by_semantic_result": {},
        "counts_by_execution_termination": {},
        "error": None,
      }

    def transition_status_if_current(
      self,
      _batch_id: int,
      status: str,
      *,
      expected_statuses: set[str],
      error: str | None = None,
    ) -> None:
      if self.digest["status"] in expected_statuses:
        self.digest["status"] = status
        self.digest["error"] = error

    def set_status(
      self,
      _batch_id: int,
      status: str,
      *,
      error: str | None = None,
    ) -> None:
      self.digest["status"] = status
      self.digest["error"] = error

    def get_batch_digest(self, _batch_id: int) -> dict[str, Any]:
      return dict(self.digest)

    def close(self) -> None:
      return None

  async def case() -> None:
    registry = FakeRegistry()
    task_registry = batches_module.BatchTaskRegistry()
    task = asyncio.create_task(asyncio.Event().wait())
    task_registry.start(
      owner_user_id="alice",
      batch_id=42,
      task=task,
      registry=registry,
    )
    await asyncio.sleep(0)
    cancellation_fence = task_registry.begin_batch_cancellation(
      owner_user_id="alice",
      batch_id=42,
    )
    app_state = SimpleNamespace(user_event_bus=FakeBus())
    approval_error, digest = (
      await batches_module._complete_authorized_batch_cancellation(
        task_registry=task_registry,
        registry=registry,
        projections=cancellation_fence.projections,
        owner_user_id="alice",
        batch_id=42,
        authenticated=SimpleNamespace(role="owner"),
        app_state=app_state,
        progress=batches_module._BatchCancellationProgress(),
      )
    )
    task_registry.finish_batch_cancellation(
      owner_user_id="alice",
      batch_id=42,
    )

    assert approval_error is None
    assert digest["status"] == "cancelled"
    assert task.done()
    duplicate = batch_terminal_event_payload_from_digest(42, digest)
    assert await task_registry.publish_terminal_event(
      app_state=app_state,
      event=duplicate,
    ) is False

  from api.agent.batch.controller_finalization import (
    batch_terminal_event_payload_from_digest,
  )

  asyncio.run(case())

  assert len(published) == 1
  assert published[0]["user_id"] == "alice"
  assert published[0]["control_run_id"] == "batch_42"
  assert published[0]["event"]["state"] == "cancelled"


def test_batch_task_monitor_publishes_failure_before_controller_try() -> None:
  published: list[dict[str, Any]] = []

  class FakeBus:
    async def publish_terminal_if_absent(
      self,
      user_id: str,
      control_run_id: str,
      event: dict[str, Any],
    ) -> bool:
      published.append({
        "user_id": user_id,
        "control_run_id": control_run_id,
        "event": dict(event),
      })
      return True

  class FakeRegistry:
    def __init__(self) -> None:
      self.digest = {
        "batch_id": 43,
        "status": "running",
        "user_id": "alice",
        "cost_usd": 0.0,
        "total_spent_usd": 0.0,
        "counts_by_status": {},
        "counts_by_gate_code": {},
        "counts_by_semantic_result": {},
        "counts_by_execution_termination": {},
        "error": None,
      }

    def transition_status_if_current(
      self,
      _batch_id: int,
      status: str,
      *,
      expected_statuses: set[str],
      error: str | None = None,
    ) -> None:
      if self.digest["status"] in expected_statuses:
        self.digest["status"] = status
        self.digest["error"] = error

    def get_batch_digest(self, _batch_id: int) -> dict[str, Any]:
      return dict(self.digest)

    def close(self) -> None:
      return None

  async def fail_before_controller_try() -> None:
    raise RuntimeError("lease_acquisition_failed")

  async def case() -> None:
    registry = FakeRegistry()
    task_registry = batches_module.BatchTaskRegistry()
    task = asyncio.create_task(fail_before_controller_try())
    task_registry.start(
      owner_user_id="alice",
      batch_id=43,
      task=task,
      registry=registry,
      app_state=SimpleNamespace(user_event_bus=FakeBus()),
    )
    monitor = task_registry._monitors[("alice", 43)]
    await monitor

    assert registry.digest["status"] == "failed"
    assert registry.digest["error"] == "lease_acquisition_failed"
    assert task_registry.get(owner_user_id="alice", batch_id=43) is None

  asyncio.run(case())

  assert len(published) == 1
  assert published[0]["control_run_id"] == "batch_43"
  assert published[0]["event"]["state"] == "failed"
