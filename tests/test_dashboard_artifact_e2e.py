from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
API_DIR = ROOT / "api"
for path in (PKG_DIR, API_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent.shared.dashboard_artifact_tool import install_named_skill_emit_dashboard_artifact_handler
from agent_gateway import AgentRunner, EventLog, ToolDispatcher
from agent_gateway.fixture_gate import FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME, FIXTURE_MODEL_ID
from agent_gateway.providers.fixture import FixtureProvider
from schema.dashboard_payload import DashboardPayload

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from test_artifact_api import ArtifactApiFixture, USER_ID, _signed_headers, artifact_api


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_server_for_tool(self, _name: str) -> str | None:
    return None

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  async def call_tool(self, name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}


def test_fixture_provider_run_emits_dashboard_artifact_and_endpoints_serve(
  artifact_api: ArtifactApiFixture,
  monkeypatch,
) -> None:
  monkeypatch.setenv("APP_ENV", "test")
  monkeypatch.setattr("agent_gateway.providers.fixture._fixture_run_seconds", lambda: 0.0)
  fixture_payload = _fixture_payload("full")
  normalized_payload = DashboardPayload.model_validate(fixture_payload).model_dump(mode="json")
  workspace = _workspace(artifact_api)
  event_log = EventLog()
  local_handlers: dict[str, Any] = {}
  installed = install_named_skill_emit_dashboard_artifact_handler(
    local_handlers=local_handlers,
    skill_profile=SimpleNamespace(name=FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME, mutation_mode=None),
    skill_run_id="fixture-dashboard-run",
    context_ticker="PCTY",
    skill_scope="ticker",
    workspace_dir=workspace,
    excluded_tools=frozenset(),
    emit_event=event_log.append,
  )
  assert installed is True

  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers=local_handlers,
    event_log=event_log,
    session_id="fixture-dashboard-session",
  )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id="fixture-dashboard-session",
    provider=FixtureProvider(),
    auth_config={
      "auth_mode": "none",
      "api_key": "",
      "auth_token": "",
      "model": FIXTURE_MODEL_ID,
      "max_tokens": 1_024,
      "thinking": False,
    },
    get_tool_definitions=lambda: [
      {
        "name": "emit_dashboard_artifact",
        "input_schema": {
          "type": "object",
          "properties": {
            "payload": {"type": "object"},
            "summary": {"type": "string"},
            "profile": {"type": "string"},
          },
          "required": ["payload", "summary"],
        },
      }
    ],
    per_turn_timeout=5.0,
    stream_stall_timeout=5.0,
    user_id=USER_ID,
    billing_mode="byok",
    rate_table_version="unknown",
    channel="cli",
  )

  asyncio.run(
    runner.run(
      [{"role": "user", "content": "start fixture dashboard artifact"}],
      system_prompt=f"Execute the deterministic fixture skill {FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME}.",
      max_turns=2,
    )
  )

  events = [entry.event for entry in event_log.entries]
  ready_events = [event for event in events if event.get("type") == "artifact_ready"]
  assert len(ready_events) == 1
  ready = ready_events[0]
  artifact_id = ready["artifact_id"]
  assert ready["skill"] == "_dashboard"
  assert ready["contract_name"] == "DashboardArtifact"
  assert ready["artifact_path"] == f"artifacts/_dashboards/{artifact_id}.json"
  assert ready["binary_artifact_path"] == f"artifacts/_dashboards/{artifact_id}.payload.json"
  assert ready["data_source"] == "live"

  sidecar_path = workspace / "artifacts" / "_dashboards" / f"{artifact_id}.json"
  payload_path = workspace / "artifacts" / "_dashboards" / f"{artifact_id}.payload.json"
  assert sidecar_path.is_file()
  assert payload_path.is_file()
  assert json.loads(payload_path.read_text(encoding="utf-8")) == normalized_payload

  with TestClient(artifact_api.app) as client:
    list_response = client.get("/api/dashboard-artifacts", headers=_signed_headers())
    sidecar_response = client.get(f"/api/dashboard-artifacts/{artifact_id}", headers=_signed_headers())
    payload_response = client.get(f"/api/dashboard-artifacts/{artifact_id}/payload", headers=_signed_headers())

  assert list_response.status_code == 200
  assert [item["artifact_id"] for item in list_response.json()] == [artifact_id]
  assert sidecar_response.status_code == 200
  assert sidecar_response.json()["contract_name"] == "DashboardArtifact"
  assert sidecar_response.json()["source_skill"] == FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME
  assert payload_response.status_code == 200
  assert payload_response.json() == normalized_payload


def test_emit_dashboard_artifact_failing_payload_returns_error_without_write_or_event(
  artifact_api: ArtifactApiFixture,
) -> None:
  workspace = _workspace(artifact_api)
  event_log = EventLog()
  local_handlers: dict[str, Any] = {}
  install_named_skill_emit_dashboard_artifact_handler(
    local_handlers=local_handlers,
    skill_profile=SimpleNamespace(name=FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME, mutation_mode=None),
    skill_run_id="fixture-dashboard-run",
    context_ticker="PCTY",
    skill_scope="ticker",
    workspace_dir=workspace,
    excluded_tools=frozenset(),
    emit_event=event_log.append,
  )

  result, error = asyncio.run(
    local_handlers["emit_dashboard_artifact"](
      {
        "payload": _fixture_payload("failing"),
        "summary": "Intentional failing payload",
        "profile": "draft",
      },
      tool_ctx=SimpleNamespace(tool_call_id="tool-dashboard-failing"),
    )
  )

  assert error is None
  assert result is not None
  assert result["error"] == "dashboard_validation_failed"
  assert result["hard_failures"]
  assert event_log.entries == []
  assert not (workspace / "artifacts" / "_dashboards").exists()


def _fixture_payload(name: str) -> dict[str, Any]:
  return json.loads((ROOT / "tests" / "fixtures" / "dashboard" / f"{name}.payload.json").read_text(encoding="utf-8"))


def _workspace(fixture: ArtifactApiFixture) -> Path:
  return fixture.data_dir / "users" / USER_ID / "workspace"
