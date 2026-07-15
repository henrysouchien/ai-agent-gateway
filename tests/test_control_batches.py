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
from agent_gateway.control_plane import batches as batches_module
from agent_gateway.control_plane.valuation_ready_tools import make_valuation_ready_skill_tool_bundle
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.skill_context import clear_current_skill, reset_current_skill, set_current_skill
from api.agent.batch.registry import BatchRegistry


API_KEY = "batch-control-key"


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


def _make_app(*, mcp_client: Any = None):
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
      mcp_client=mcp_client,
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


def _install_batch_pending_approval(
  *,
  app: Any,
  batch_id: int,
  owner_user_id: str = "alice",
  channel: str = "tui",
  required_decider_count: int = 1,
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
    required_decider_count=required_decider_count,
    eligible_decider_count=max(1, required_decider_count),
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
    assert detail["diligence_prs"] == []

    list_response = client.get("/api/control/batches", headers=headers)
    assert list_response.status_code == 200, list_response.text
    batches = list_response.json()["batches"]
    assert [item["batch_id"] for item in batches] == [batch_id]
    assert batches[0]["cost_usd"] == pytest.approx(0.125)
    assert fake_batch_control.run_calls[0]["identity"] == ("alice", "alice@example.com")


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
  batch_id = registry.acquire_batch(
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
  assert detail["diligence_prs"] == []
  registry.close()


def test_batch_detail_includes_diligence_prs(tmp_path: Path) -> None:
  registry = BatchRegistry(tmp_path / "batch.db")
  batch_id = registry.acquire_batch(
    user_id="alice",
    host="test-host",
    spec={"source": "quality_screen", "universe": ["MSFT"]},
    budget_usd=3.0,
    pid=None,
  )
  with registry._db_lock:
    conn = registry._ensure_db()
    conn.execute(
      """
      INSERT INTO diligence_prs (
        pr_id, batch_id, ticker, state, workspace_ref, workspace_id,
        model_workspace_json, proposal_ids_json, run_seqs_json,
        base_hashes_json, summary_json, created_at, updated_at
      ) VALUES (
        'dpr_batch_detail', ?, 'MSFT', 'open',
        'model_workspaces/batch_1/MSFT/workspace.json', 'batch_1_MSFT',
        '{}', '[]', '[1]',
        '{"schema_version":1,"thesis":{"hash":"sha256:abc"}}',
        '{}', 101.0, 101.0
      )
      """,
      (batch_id,),
    )
    conn.commit()

  detail = batches_module._batch_detail_payload(registry, batch_id, top_n=10)

  assert len(detail["diligence_prs"]) == 1
  pr = detail["diligence_prs"][0]
  assert pr["ticker"] == "MSFT"
  assert pr["state"] == "open"
  assert pr["workspace_id"] == "batch_1_MSFT"
  assert pr["base_hashes"]["thesis"]["hash"] == "sha256:abc"
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
      gateway_config=SimpleNamespace(mcp_client=mcp_client),
    )
    session = SimpleNamespace(
      user_id="alice",
      owner_user_id="1",
      risk_user_id=1,
      user_email="alice@example.com",
      channel="tui",
    )
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
      })
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
        }
      )
      assert error is None
      assert result["batch_id"] == 7
    finally:
      reset_current_skill(token)

  _run(run_case())
  assert captured["user_id"] == "1"
  assert captured["channel"] == "telegram"


def test_gateway_local_valuation_ready_dispatch_blocks_before_batch_admission_on_corpus_gap(
  fake_batch_control: FakeBatchController,
) -> None:
  async def run_case() -> None:
    mcp_client = ReadyCorpusMcpClient(ready=False)
    app_state = SimpleNamespace(
      user_event_bus=None,
      gateway_config=SimpleNamespace(mcp_client=mcp_client),
    )
    session = SimpleNamespace(
      user_id="alice",
      owner_user_id="1",
      risk_user_id=1,
      user_email="alice@example.com",
      channel="tui",
    )
    bundle = make_valuation_ready_skill_tool_bundle(app_state=app_state, session=session)

    token = set_current_skill("valuation-ready")
    try:
      result, error = await bundle["handlers"]["valuation_ready_batch_dispatch"]({
        "ticker": "PCTY",
        "required_filings": ["2025-FY"],
        "required_transcripts": ["2026-Q2"],
      })
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

    result, error = await dispatch({"ticker": "ADI"})

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
  assert "active batch already exists" in response.json()["detail"]


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
  assert workflows["full-diligence"]["reservation_budget_usd_per_name"] == 49.0
  assert workflows["full-diligence"]["suggested_budget_usd_per_name"] == 49.0
  assert workflows["valuation-ready"]["reservation_budget_usd_per_name"] == 25.0
  assert (
    workflows["full-diligence"]["budget_model"]["admission_formula"]
    == "spent_usd + in_flight_reserved_usd + next_stage_reserved_usd <= budget_usd"
  )
  assert workflows["full-diligence"]["budget_model"]["includes_goal_remediation_capacity"] is False


def test_fixture_batch_requires_dev_mode(fake_batch_control: FakeBatchController) -> None:
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches/fixture", headers=headers, json={"fixture": "mixed"})

  assert response.status_code == 403
  assert response.json()["detail"] == "fixture batch seed requires dev_mode=true"
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_fixture_batch_refuses_production_even_with_dev_flags(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("APP_ENV", "production")
  monkeypatch.setenv("AGENT_GATEWAY_ENV", "development")

  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches/fixture", headers=headers, json={"fixture": "mixed", "dev_mode": True})

  assert response.status_code == 403
  assert "fixture batch seed is dev-only" in response.json()["detail"]
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


def test_fixture_batch_seeds_registry_without_controller(
  fake_batch_control: FakeBatchController,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("APP_ENV", "test")
  app = _make_app()
  with TestClient(app) as client:
    headers = _headers(_control_session(client))
    response = client.post("/api/control/batches/fixture", headers=headers, json={"fixture": "mixed", "dev_mode": True})

    assert response.status_code == 200, response.text
    payload = response.json()
    batch_id = int(payload["batch"]["batch_id"])
    assert payload["batch"]["status"] == "completed"
    assert payload["batch"]["cost_usd"] == 0
    assert payload["batch"]["counts_by_status"] == {
      "completed": 1,
      "failed": 1,
      "rejected": 1,
    }
    assert payload["batch"]["counts_by_gate_code"] == {
      "INSUFFICIENT_DATA": 1,
      "PROCEED": 1,
      "STOP": 1,
    }
    assert [row["ticker"] for row in payload["verdict_matrix"]] == ["MSFT", "AAPL", "TSLA"]
    assert payload["candidates"][0]["proposal_id"] == "fixture-proposal-msft"
    assert payload["diligence_prs"] == []
    assert payload["failures"] == [
      {
        "ticker": "TSLA",
        "skill": "industry-landscape",
        "status": "failed",
        "error": "fixture failure for renderer QA",
      }
    ]

    detail_response = client.get(f"/api/control/batches/{batch_id}", headers=headers)
    list_response = client.get("/api/control/batches", headers=headers)

  assert detail_response.status_code == 200, detail_response.text
  assert detail_response.json()["batch"]["batch_id"] == batch_id
  assert list_response.status_code == 200, list_response.text
  assert [item["batch_id"] for item in list_response.json()["batches"]] == [batch_id]
  assert fake_batch_control.acquire_calls == []
  assert fake_batch_control.run_calls == []


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
    _identity: tuple[str, str | None] | None = None,
    _on_finalize=None,
  ) -> dict[str, Any]:
    _ = spec, _identity, _on_finalize
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
    _identity: tuple[str, str | None] | None = None,
    _on_finalize=None,
  ) -> dict[str, Any]:
    _ = spec, _identity, _on_finalize
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
    _identity: tuple[str, str | None] | None = None,
    _on_finalize=None,
  ) -> dict[str, Any]:
    _ = spec, _identity, _on_finalize
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
      "denied_by": None,
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
      required_decider_count=3,
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
    assert stored.votes_received_count == 0
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
      assert stored.votes_received_count == 0
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


def test_control_batches_cancel_terminalizes_multi_decider_approval_without_quorum(
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
      required_decider_count=2,
    )

    cancel_response = client.delete(f"/api/control/batches/{batch_id}", headers=headers)

    assert cancel_response.status_code == 200, cancel_response.text
    stored = _run(app.state.gateway_approval_store.get(request_record.approval_id))
    assert stored.state == "denied"
    assert stored.required_decider_count == 2
    assert stored.votes_received_count == 0
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
      "denied_by": None,
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
      "denied_by": None,
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
  batch_id = registry.acquire_batch(
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
    batch_id = registry.acquire_batch(
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
      required_decider_count=3,
      eligible_decider_count=3,
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
    batch_id = registry.acquire_batch(
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
    batch_id = registry.acquire_batch(
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
    batch_id = registry.acquire_batch(
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
    batch_id = registry.acquire_batch(
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
    batch_id = registry.acquire_batch(
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
    batch_id = registry.acquire_batch(
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
      batch_id = registry.acquire_batch(
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
    batch_id = registry.acquire_batch(
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
