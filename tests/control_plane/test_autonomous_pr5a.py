from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "autonomous-pr5a-key"
HMAC_KEY = "autonomous-pr5a-hmac"


class _FakeAutonomousProcess:
  def __init__(self) -> None:
    self.returncode: int | None = None

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def terminate(self) -> None:
    if self.returncode is None:
      self.returncode = -15

  def kill(self) -> None:
    if self.returncode is None:
      self.returncode = -9


class _NoopRunner:
  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, model_override, max_turns


def _make_app(monkeypatch, tmp_path: Path):
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "autonomous-logs"))

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid: _runner_with_log(event_log),
    )

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="autonomous-pr5a-test-secret-0123456789",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _runner_with_log(_event_log: EventLog) -> _NoopRunner:
  return _NoopRunner()


def _control_session(client: TestClient, user_id: str, *, email: str | None = None) -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={
      "api_key": API_KEY,
      "user_id": user_id,
      "user_email": email or f"{user_id}@example.com",
      "context": {"channel": "tui"},
    },
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session['session_token']}"}


def _install_fake_spawn(monkeypatch) -> tuple[list[_FakeAutonomousProcess], list[dict[str, str]]]:
  from agent_gateway import autonomous_runner

  processes: list[_FakeAutonomousProcess] = []
  envs: list[dict[str, str]] = []

  async def fake_exec(*args, **kwargs):
    _ = args
    process = _FakeAutonomousProcess()
    processes.append(process)
    envs.append(dict(kwargs["env"]))
    return process

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  return processes, envs


def _to_httpx_response(response) -> httpx.Response:
  return httpx.Response(
    response.status_code,
    content=response.content,
    headers=dict(response.headers),
  )


def test_gateway_lifespan_creates_and_shuts_down_autonomous_registry(monkeypatch, tmp_path) -> None:
  app = _make_app(monkeypatch, tmp_path)
  calls: list[float] = []
  original_shutdown = app.state.subprocess_registry.shutdown

  async def spy_shutdown(*args, **kwargs):
    calls.append(time.time())
    return await original_shutdown(*args, **kwargs)

  app.state.subprocess_registry.shutdown = spy_shutdown

  with TestClient(app):
    assert app.state.subprocess_registry is not None

  assert len(calls) == 1


def test_autonomous_control_endpoints_spawn_read_logs_cancel_and_enforce_user_scope(monkeypatch, tmp_path) -> None:
  processes, envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")
    bob = _control_session(client, "bob", email="bob@example.com")

    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "earnings-review",
        "context": "Review AAPL earnings",
        "ticker": "AAPL",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    start_payload = start.json()
    assert start_payload["task_id"] == "bg_0"
    assert start_payload["run"]["run_id"] == "bg_0"
    assert start_payload["run"]["state"] == "running"

    assert envs[0]["AGENT_API_CLAIM_USER_ID"] == "alice"
    assert envs[0]["AGENT_API_CLAIM_USER_EMAIL"] == "alice@example.com"
    assert envs[0]["AGENT_API_USER_CLAIM_HMAC_KEY"] == HMAC_KEY
    assert Path(envs[0]["AGENT_AUTONOMOUS_EVENTS_PATH"]).name == "bg_0.events.jsonl"

    record = app.state.subprocess_registry._tasks["bg_0"]
    record.log_path.write_text("line 1\nline 2\n", encoding="utf-8")

    detail = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    assert detail.status_code == 200, detail.text
    assert detail.json()["user_id"] == "alice"

    logs = client.get("/api/control/runs/bg_0/logs?tail=1", headers=_headers(alice))
    assert logs.status_code == 200, logs.text
    assert logs.json()["log_lines"] == ["line 2"]
    assert logs.json()["more_available"] is True

    assert client.get("/api/control/runs/bg_0", headers=_headers(bob)).status_code == 404
    assert client.get("/api/control/runs/bg_0/logs", headers=_headers(bob)).status_code == 404
    assert client.delete("/api/control/runs/bg_0", headers=_headers(bob)).status_code == 404

    cancel = client.delete("/api/control/runs/bg_0", headers=_headers(alice))
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["state"] == "cancelled"
    assert processes[0].returncode == -15


def test_gateway_shutdown_terminates_inflight_autonomous_process(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    response = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert response.status_code == 200, response.text
    assert processes[0].returncode is None

  assert processes[0].returncode == -15


def test_agents_mcp_relay_round_trips_all_autonomous_tools(monkeypatch, tmp_path) -> None:
  _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  from mcp_servers.agents_mcp import gateway_client
  from mcp_servers.agents_mcp.config import AgentsMCPConfig

  gateway_client.reset_control_session()
  config = AgentsMCPConfig(
    repo_root=Path("."),
    api_dir=Path("."),
    gateway_url="https://testserver",
    api_key=API_KEY,
    user_id="alice",
    storage_user_id="alice",
    user_email="alice@example.com",
    python_executable=sys.executable,
    log_dir=tmp_path,
    tls_verify=False,
  )

  with TestClient(app) as client:
    async def fake_control_post(config, path, *, headers=None, json_body=None):
      return _to_httpx_response(client.post(f"/api{path}", headers=headers, json=json_body))

    async def fake_control_request(config, method, path, *, headers=None, json_body=None, params=None):
      return _to_httpx_response(
        client.request(method, f"/api{path}", headers=headers, json=json_body, params=params)
      )

    monkeypatch.setattr(gateway_client, "_control_post", fake_control_post)
    monkeypatch.setattr(gateway_client, "_control_request", fake_control_request)

    start = asyncio.run(
      gateway_client.autonomous_run_start(
        config,
        profile="analyst",
        mode="task",
        task="summarize",
        channel="tui",
      )
    )
    assert start["task_id"] == "bg_0"
    assert start["run"]["task_id"] == "bg_0"

    record = app.state.subprocess_registry._tasks["bg_0"]
    record.log_path.write_text("relay log\n", encoding="utf-8")

    status = asyncio.run(gateway_client.autonomous_run_status(config, "bg_0"))
    assert status["kind"] == "autonomous"
    assert status["task_id"] == "bg_0"

    logs = asyncio.run(gateway_client.autonomous_run_logs(config, "bg_0", tail=1))
    assert logs["log_lines"] == ["relay log"]

    cancel = asyncio.run(gateway_client.autonomous_run_cancel(config, "bg_0"))
    assert cancel["state"] == "cancelled"

    waited = asyncio.run(gateway_client.autonomous_run_wait(config, "bg_0", timeout_sec=30, poll_interval_sec=0))
    assert waited["state"] == "cancelled"
