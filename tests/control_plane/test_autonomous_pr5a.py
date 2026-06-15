from __future__ import annotations

import asyncio
import json
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


def _make_app(monkeypatch, tmp_path: Path, *, control_skills_dir: Path | None = None):
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
      control_skills_dir=control_skills_dir,
    )
  )


def _autonomous_log_dir(tmp_path: Path) -> Path:
  path = tmp_path / "autonomous-logs"
  path.mkdir(parents=True, exist_ok=True)
  return path


def _write_rehydrate_manifest(
  tmp_path: Path,
  task_id: str = "bg_0",
  *,
  user_id: str = "alice",
  user_email: str | None = "alice@example.com",
  control_run_id: str | None = None,
  state: str = "completed",
  skill: str | None = "earnings-review",
  mode: str = "skill",
  channel: str | None = "tui",
  completed_at: float | None = 125.0,
  exit_code: int | None = 0,
  error: str | None = None,
  context: str | None = "Original work packet",
) -> dict[str, Any]:
  log_dir = _autonomous_log_dir(tmp_path)
  manifest = {
    "manifest_version": 1,
    "task_id": task_id,
    "control_run_id": control_run_id or task_id,
    "user_id": user_id,
    "user_email": user_email,
    "profile": "analyst",
    "mode": mode,
    "task": None if mode == "skill" else "summarize",
    "skill": skill,
    "context": context,
    "ticker": "MSFT",
    "channel": channel,
    "dev_mode": False,
    "cmd": [sys.executable, "-m", "agent.autonomous", "--profile", "analyst"],
    "log_path": str(log_dir / f"{task_id}.log"),
    "events_path": str(log_dir / f"{task_id}.events.jsonl"),
    "operator_inbox_path": str(log_dir / f"{task_id}.operator-messages.jsonl"),
    "approval_decisions_path": str(log_dir / f"{task_id}.approval-decisions.jsonl"),
    "started_at": 100.0,
    "state": state,
    "exit_code": exit_code,
    "error": error,
    "completed_at": completed_at,
    "resumed_from": None,
    "resumed_as": [],
  }
  (log_dir / f"{task_id}.task.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
  return manifest


def _write_rehydrate_events(tmp_path: Path, task_id: str, events: list[dict[str, Any]]) -> None:
  log_dir = _autonomous_log_dir(tmp_path)
  (log_dir / f"{task_id}.events.jsonl").write_text(
    "".join(json.dumps(event) + "\n" for event in events),
    encoding="utf-8",
  )


def _runner_with_log(_event_log: EventLog) -> _NoopRunner:
  return _NoopRunner()


def _control_session(
  client: TestClient,
  user_id: str,
  *,
  email: str | None = None,
  channel: str | None = "tui",
) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "api_key": API_KEY,
    "user_id": user_id,
    "user_email": email or f"{user_id}@example.com",
    "context": {},
  }
  if channel is not None:
    payload["context"]["channel"] = channel
  response = client.post(
    "/api/control/session",
    json=payload,
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
    assert "--ticker" in start_payload["cmd"]
    assert start_payload["cmd"][start_payload["cmd"].index("--ticker") + 1] == "AAPL"

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


def test_autonomous_reaper_terminalizes_approval_pending_process_exit(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  async def run_case():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
      alice = await client.post(
        "/api/control/session",
        json={
          "api_key": API_KEY,
          "user_id": "alice",
          "user_email": "alice@example.com",
          "context": {"channel": "tui"},
        },
      )
      assert alice.status_code == 200, alice.text
      headers = _headers(alice.json())
      start = await client.post(
        "/api/control/runs",
        headers=headers,
        json={
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "task",
          "task": "approval pending exit",
          "channel": "tui",
        },
      )
      assert start.status_code == 200, start.text

      record = app.state.subprocess_registry._tasks["bg_0"]
      record.state = "approval_pending"
      processes[0].returncode = 1
      assert record.reaper_task is not None
      await asyncio.wait_for(asyncio.shield(record.reaper_task), timeout=2.0)

      detail = await client.get("/api/control/runs/bg_0", headers=headers)
      assert detail.status_code == 200, detail.text
      return record, detail.json()

  record, detail = asyncio.run(run_case())

  assert record.state == "failed"
  assert record.completed_at is not None
  assert record.error == "Process exited with code 1"
  assert detail["state"] == "failed"
  assert any(
    event.get("type") == "run_state_changed" and event.get("state") == "failed"
    for event in record.event_lines or ()
  )


def test_rehydrated_autonomous_run_is_owner_scoped_with_event_derived_fields(monkeypatch, tmp_path) -> None:
  _write_rehydrate_manifest(
    tmp_path,
    "bg_4",
    control_run_id="run-rehydrated",
    user_id="alice",
    user_email="alice@example.com",
    state="completed",
    completed_at=200.0,
  )
  _write_rehydrate_events(
    tmp_path,
    "bg_4",
    [
      {
        "type": "text_delta",
        "run_id": "run-rehydrated",
        "control_run_id": "run-rehydrated",
        "skill_run_id": "skill-run-1",
        "ts": 180,
      },
      {
        "type": "skill_result_captured",
        "run_id": "run-rehydrated",
        "control_run_id": "run-rehydrated",
        "skill_run_id": "skill-run-1",
        "skill": "monitoring-init",
        "verdict_echo": {
          "verdict_token": "monitor",
          "confidence": "medium",
          "one_line_summary": "Watch the setup",
        },
        "ts": 190,
      },
    ],
  )
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")
    bob = _control_session(client, "bob", email="bob@example.com")

    listed = client.get("/api/control/runs?kind=autonomous", headers=_headers(alice))
    detail = client.get("/api/control/runs/run-rehydrated", headers=_headers(alice))
    bob_listed = client.get("/api/control/runs?kind=autonomous", headers=_headers(bob))
    bob_detail = client.get("/api/control/runs/run-rehydrated", headers=_headers(bob))

  assert listed.status_code == 200, listed.text
  runs = listed.json()["runs"]
  assert [run["run_id"] for run in runs] == ["run-rehydrated"]
  assert runs[0]["state"] == "completed"
  assert runs[0]["skill_run_ids"] == ["skill-run-1"]
  assert runs[0]["current_verdict"] == {
    "verdict_token": "monitor",
    "confidence": "medium",
    "one_line_summary": "Watch the setup",
    "skill_run_id": "skill-run-1",
  }
  assert detail.status_code == 200, detail.text
  assert detail.json()["run_id"] == "run-rehydrated"
  assert bob_listed.status_code == 200
  assert bob_listed.json()["runs"] == []
  assert bob_detail.status_code == 404


def test_autonomous_state_passes_interrupted_through() -> None:
  from agent_gateway.control_plane.runs import _autonomous_state

  assert _autonomous_state("interrupted") == "interrupted"


def test_rehydrated_interrupted_resumable_skill_uses_existing_resume_flow(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "resumable-skill.md").write_text(
    """---
name: resumable-skill
description: Resumable test skill
agent_callable: true
resumable: true
catalog: false
---
Run the resumable test skill.
""",
    encoding="utf-8",
  )
  _write_rehydrate_manifest(
    tmp_path,
    "bg_0",
    state="running",
    completed_at=None,
    exit_code=None,
    skill="resumable-skill",
  )
  _write_rehydrate_events(
    tmp_path,
    "bg_0",
    [
      {
        "type": "text_delta",
        "text": "persisted event",
        "run_id": "bg_0",
        "control_run_id": "bg_0",
        "ts": 321,
      }
    ],
  )
  log_dir = _autonomous_log_dir(tmp_path)
  (log_dir / "bg_0.log").write_text("rehydrated log line\n", encoding="utf-8")
  (log_dir / "bg_0.operator-messages.jsonl").write_text(
    '{"message_id":"op-1","message":"operator context","sent_at":300}\n',
    encoding="utf-8",
  )
  app = _make_app(monkeypatch, tmp_path, control_skills_dir=skills_dir)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")

    detail = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    assert detail.status_code == 200, detail.text
    assert detail.json()["state"] == "interrupted"
    assert detail.json()["ended_at"] == "1970-01-01T00:05:21Z"
    assert detail.json()["resumable"] is True

    resumed = client.post(
      "/api/control/runs/bg_0/resume",
      headers=_headers(alice),
      json={"message": "resume safely", "request_id": "resume-rehydrated"},
    )

  assert resumed.status_code == 200, resumed.text
  payload = resumed.json()
  assert payload["resumed_from"] == "bg_0"
  assert payload["run"]["run_id"] == "bg_1"
  assert payload["run"]["resumed_from"] == "bg_0"
  assert payload["run"]["state"] == "running"
  assert len(processes) == 1
  original = app.state.subprocess_registry._tasks["bg_0"]
  assert original.state == "interrupted"
  assert original.error == "gateway restarted while run was active"
  assert original.completed_at == 321.0
  assert original.proc is None
  assert original.resumed_as == ["bg_1"]
  resumed_record = app.state.subprocess_registry._tasks["bg_1"]
  assert resumed_record.context is not None
  assert "state=interrupted" in resumed_record.context
  assert "persisted event" in resumed_record.context
  assert "operator context" in resumed_record.context
  assert "rehydrated log line" in resumed_record.context


def test_autonomous_resume_starts_linked_run_for_resumable_interrupted_skill(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "resumable-skill.md").write_text(
    """---
name: resumable-skill
description: Resumable test skill
agent_callable: true
resumable: true
catalog: false
---
Run the resumable test skill.
""",
    encoding="utf-8",
  )
  app = _make_app(monkeypatch, tmp_path, control_skills_dir=skills_dir)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "resumable-skill",
        "context": "Original work packet",
        "ticker": "MSFT",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text

    original = app.state.subprocess_registry._tasks["bg_0"]
    original.state = "failed"
    original.exit_code = 1
    original.completed_at = time.time()
    processes[0].returncode = 1
    original.log_path.write_text("prior log line\n", encoding="utf-8")
    original.operator_inbox_path.write_text(
      '{"message_id":"op-1","message":"tighten scope","sent_at":1}\n',
      encoding="utf-8",
    )

    detail = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    assert detail.status_code == 200, detail.text
    assert detail.json()["resumable"] is True

    resumed = client.post(
      "/api/control/runs/bg_0/resume",
      headers=_headers(alice),
      json={"message": "resume from the latest safe point", "request_id": "resume-1"},
    )
    assert resumed.status_code == 200, resumed.text
    payload = resumed.json()
    assert payload["resumed_from"] == "bg_0"
    assert payload["run"]["run_id"] == "bg_1"
    assert payload["run"]["resumed_from"] == "bg_0"
    assert payload["run"]["state"] == "running"

    assert original.resumed_as == ["bg_1"]
    resumed_record = app.state.subprocess_registry._tasks["bg_1"]
    assert resumed_record.context is not None
    assert "Original work packet" in resumed_record.context
    assert "resume from the latest safe point" in resumed_record.context
    assert "tighten scope" in resumed_record.context
    assert "prior log line" in resumed_record.context


def test_autonomous_resume_allows_only_one_active_replacement(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "3")
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "resumable-skill.md").write_text(
    """---
name: resumable-skill
description: Resumable test skill
agent_callable: true
resumable: true
---
Run the resumable test skill.
""",
    encoding="utf-8",
  )
  app = _make_app(monkeypatch, tmp_path, control_skills_dir=skills_dir)
  original_start = app.state.subprocess_registry.start

  async def delayed_start(*args, **kwargs):
    await asyncio.sleep(0.05)
    return await original_start(*args, **kwargs)

  app.state.subprocess_registry.start = delayed_start

  async def run_concurrent_resume():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
      alice = await client.post(
        "/api/control/session",
        json={
          "api_key": API_KEY,
          "user_id": "alice",
          "user_email": "alice@example.com",
          "context": {"channel": "tui"},
        },
      )
      assert alice.status_code == 200, alice.text
      headers = _headers(alice.json())
      start = await client.post(
        "/api/control/runs",
        headers=headers,
        json={
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "skill",
          "skill": "resumable-skill",
          "context": "Original work packet",
        },
      )
      assert start.status_code == 200, start.text

      original = app.state.subprocess_registry._tasks["bg_0"]
      original.state = "failed"
      original.exit_code = 1
      original.completed_at = time.time()
      processes[0].returncode = 1

      first, second = await asyncio.gather(
        client.post("/api/control/runs/bg_0/resume", headers=headers, json={"request_id": "resume-1"}),
        client.post("/api/control/runs/bg_0/resume", headers=headers, json={"request_id": "resume-2"}),
      )
      return original, first, second

  original, first, second = asyncio.run(run_concurrent_resume())

  statuses = sorted([first.status_code, second.status_code])
  assert statuses == [200, 409]
  assert original.resumed_as == ["bg_1"]
  assert set(app.state.subprocess_registry._tasks) == {"bg_0", "bg_1"}


def test_autonomous_resume_context_preserves_recovery_packet_when_original_context_is_large(
  monkeypatch,
  tmp_path,
) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "resumable-skill.md").write_text(
    """---
name: resumable-skill
description: Resumable test skill
agent_callable: true
resumable: true
---
Run the resumable test skill.
""",
    encoding="utf-8",
  )
  app = _make_app(monkeypatch, tmp_path, control_skills_dir=skills_dir)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "resumable-skill",
        "context": "ORIGINAL-" + ("x" * 20000),
      },
    )
    assert start.status_code == 200, start.text

    original = app.state.subprocess_registry._tasks["bg_0"]
    original.state = "failed"
    original.exit_code = 1
    original.completed_at = time.time()
    processes[0].returncode = 1
    original.log_path.write_text("prior log line\n", encoding="utf-8")
    original.operator_inbox_path.write_text(
      '{"message_id":"op-1","message":"tighten scope","sent_at":1}\n',
      encoding="utf-8",
    )

    resumed = client.post(
      "/api/control/runs/bg_0/resume",
      headers=_headers(alice),
      json={"message": "resume from the latest safe point"},
    )
    assert resumed.status_code == 200, resumed.text

  resumed_context = app.state.subprocess_registry._tasks["bg_1"].context or ""
  assert "resume from the latest safe point" in resumed_context
  assert "tighten scope" in resumed_context
  assert "prior log line" in resumed_context
  assert "Do not repeat durable writes" in resumed_context


def test_autonomous_resume_rejects_non_resumable_skill(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "one-shot.md").write_text(
    """---
name: one-shot
description: One shot test skill
agent_callable: true
resumable: false
---
Run once.
""",
    encoding="utf-8",
  )
  app = _make_app(monkeypatch, tmp_path, control_skills_dir=skills_dir)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "one-shot",
        "context": "Original work packet",
      },
    )
    assert start.status_code == 200, start.text

    original = app.state.subprocess_registry._tasks["bg_0"]
    original.state = "failed"
    original.exit_code = 1
    original.completed_at = time.time()
    processes[0].returncode = 1

    resumed = client.post("/api/control/runs/bg_0/resume", headers=_headers(alice), json={})
    assert resumed.status_code == 409
    assert resumed.json()["detail"] == "Autonomous run is not resumable"
    assert set(app.state.subprocess_registry._tasks) == {"bg_0"}


def test_autonomous_dispatch_requires_control_session(monkeypatch, tmp_path) -> None:
  _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    chat_session = app.state.auth.session_store.create_session(
      api_key_hash="test-hash",
      user_id="alice",
      user_email="alice@example.com",
      kind="chat",
    )
    chat_session.channel = "tui"
    chat_token = app.state.auth.issue_token(chat_session)

    response = client.post(
      "/api/control/runs",
      headers={"Authorization": f"Bearer {chat_token}"},
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )

    assert response.status_code == 401
    assert app.state.subprocess_registry._tasks == {}


def test_autonomous_dispatch_rejects_control_session_channel_override(monkeypatch, tmp_path) -> None:
  _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="tui")

    response = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "summarize",
        "channel": "excel",
      },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Channel mismatch"
    assert app.state.subprocess_registry._tasks == {}


def test_autonomous_dispatch_rejects_web_fixture_and_dev_mode_before_spawn(monkeypatch, tmp_path) -> None:
  processes, envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)
  cases = [
    {"profile": "_fixture", "skill": "earnings-review"},
    {"profile": "analyst", "skill": "fixture-sleep"},
    {"profile": "analyst", "skill": "earnings-review", "dev_mode": True},
    {"profile": "analyst", "skill": "earnings-review", "dev_mode": False},
  ]

  with TestClient(app) as client:
    web_session = _control_session(client, "alice", channel="web")
    payload_web_session = _control_session(client, "bob", channel=None)

    for session, payload_channel in (
      (web_session, None),
      (payload_web_session, "web"),
    ):
      for overrides in cases:
        payload = {
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "skill",
          "skill": "earnings-review",
          "context": "Exercise fixture guard.",
          "ticker": "AAPL",
          **overrides,
        }
        if payload_channel is not None:
          payload["channel"] = payload_channel
        response = client.post(
          "/api/control/runs",
          headers=_headers(session),
          json=payload,
        )

        assert response.status_code == 403
        assert response.json() == {
          "detail": {
            "error": "web_control_dev_dispatch_forbidden",
            "message": "Web Agent Control cannot launch fixture or dev-mode autonomous runs.",
          }
        }

  assert app.state.subprocess_registry._tasks == {}
  assert processes == []
  assert envs == []


def test_autonomous_dispatch_allows_scoped_web_fixture_qa_bridge(monkeypatch, tmp_path) -> None:
  _processes, envs = _install_fake_spawn(monkeypatch)
  monkeypatch.setenv("APP_ENV", "test")
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    web_session = _control_session(client, "alice", channel="web")

    response = client.post(
      "/api/control/runs",
      headers={
        **_headers(web_session),
        "X-Agent-Control-QA-Bridge": "fixture-approval-artifact",
      },
      json={
        "kind": "autonomous",
        "profile": "_fixture",
        "mode": "skill",
        "skill": "fixture-approval-html-artifact",
        "context": "Exercise paused approval evidence fixture.",
        "dev_mode": True,
      },
    )

  assert response.status_code == 200, response.text
  assert response.json()["run_id"] == "bg_0"
  record = app.state.subprocess_registry._tasks["bg_0"]
  assert record.channel == "web"
  assert record.profile == "_fixture"
  assert record.skill == "fixture-approval-html-artifact"
  assert record.dev_mode is True
  assert envs[-1]["_FIXTURE_DEV_MODE"] == "true"


def test_autonomous_dispatch_allows_scoped_web_terminal_failure_fixture_qa_bridge(monkeypatch, tmp_path) -> None:
  _processes, envs = _install_fake_spawn(monkeypatch)
  monkeypatch.setenv("APP_ENV", "test")
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    web_session = _control_session(client, "alice", channel="web")

    response = client.post(
      "/api/control/runs",
      headers={
        **_headers(web_session),
        "X-Agent-Control-QA-Bridge": "fixture-terminal-failure",
      },
      json={
        "kind": "autonomous",
        "profile": "_fixture",
        "mode": "skill",
        "skill": "fixture-terminal-failure",
        "context": "Exercise terminal failure presentation fixture.",
        "dev_mode": True,
      },
    )

  assert response.status_code == 200, response.text
  assert response.json()["run_id"] == "bg_0"
  record = app.state.subprocess_registry._tasks["bg_0"]
  assert record.channel == "web"
  assert record.profile == "_fixture"
  assert record.skill == "fixture-terminal-failure"
  assert record.dev_mode is True
  assert envs[-1]["_FIXTURE_DEV_MODE"] == "true"


def test_autonomous_dispatch_qa_bridge_does_not_allow_other_web_fixtures(monkeypatch, tmp_path) -> None:
  processes, envs = _install_fake_spawn(monkeypatch)
  monkeypatch.setenv("APP_ENV", "test")
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    web_session = _control_session(client, "alice", channel="web")

    response = client.post(
      "/api/control/runs",
      headers={
        **_headers(web_session),
        "X-Agent-Control-QA-Bridge": "fixture-approval-artifact",
      },
      json={
        "kind": "autonomous",
        "profile": "_fixture",
        "mode": "skill",
        "skill": "fixture-sleep",
        "context": "This fixture is not approval-evidence QA.",
        "dev_mode": True,
      },
    )

  assert response.status_code == 403
  assert response.json() == {
    "detail": {
      "error": "web_control_dev_dispatch_forbidden",
      "message": "Web Agent Control cannot launch fixture or dev-mode autonomous runs.",
    }
  }
  assert app.state.subprocess_registry._tasks == {}
  assert processes == []
  assert envs == []


def test_autonomous_dispatch_validation_error_returns_422_and_releases_slot(monkeypatch, tmp_path) -> None:
  _processes, envs = _install_fake_spawn(monkeypatch)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "1")
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")

    invalid = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "summarize",
        "context": "task mode should not pass separate context",
      },
    )
    assert invalid.status_code == 422
    assert invalid.json()["detail"] == "mode='task' only accepts the task parameter"

    valid = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert valid.status_code == 200, valid.text
    assert valid.json()["run_id"] == "bg_1"
    record = app.state.subprocess_registry._tasks["bg_1"]
    assert record.dev_mode is True
    assert record.cmd == [
      sys.executable,
      "-m",
      "agent.autonomous",
      "--profile",
      "analyst",
      "--task",
      "summarize",
    ]
    assert envs[-1]["ANALYST_DEV_MODE"] == "true"


def test_autonomous_dispatch_accepts_module_safe_profile_names(monkeypatch, tmp_path) -> None:
  _processes, _envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "custom_research_profile",
        "mode": "task",
        "task": "summarize",
      },
    )

    assert response.status_code == 200, response.text
    record = app.state.subprocess_registry._tasks["bg_0"]
    assert record.profile == "custom_research_profile"
    assert record.cmd[:5] == [
      sys.executable,
      "-m",
      "agent.autonomous",
      "--profile",
      "custom_research_profile",
    ]


def test_control_profiles_lists_backend_available_profiles(monkeypatch, tmp_path) -> None:
  _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")

    response = client.get("/api/control/profiles", headers=_headers(alice))

    assert response.status_code == 200, response.text
    profile_names = {entry["name"] for entry in response.json()["profiles"]}
    assert {"analyst", "advisor", "research_producer"} <= profile_names


def test_autonomous_run_response_preserves_waiting_and_queued_states(monkeypatch, tmp_path) -> None:
  _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert start.status_code == 200, start.text
    record = app.state.subprocess_registry._tasks["bg_0"]

    record.state = "queued"
    queued = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    assert queued.status_code == 200, queued.text
    assert queued.json()["state"] == "queued"

    record.state = "waiting"
    waiting = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    assert waiting.status_code == 200, waiting.text
    assert waiting.json()["state"] == "waiting"


def test_autonomous_waiting_run_accepts_operator_message(monkeypatch, tmp_path) -> None:
  _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert start.status_code == 200, start.text
    record = app.state.subprocess_registry._tasks["bg_0"]
    record.state = "waiting"

    response = client.post(
      "/api/control/runs/bg_0/messages",
      headers=_headers(alice),
      json={"message": "continue with read-only scope", "message_id": "op-waiting-1"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["delivery_status"] == "delivered"
    assert response.json()["run"]["state"] == "waiting"
    assert record.operator_inbox_path is not None
    assert "continue with read-only scope" in record.operator_inbox_path.read_text(encoding="utf-8")


def test_autonomous_dispatch_concurrency_limit_returns_429(monkeypatch, tmp_path) -> None:
  _install_fake_spawn(monkeypatch)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "1")
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")

    first = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert first.status_code == 200, first.text

    second = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize again"},
    )
    assert second.status_code == 429
    assert second.json()["detail"] == "Autonomous concurrency limit reached (1)"


def test_autonomous_run_reads_logs_and_cancel_are_channel_scoped(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    tui = _control_session(client, "alice", channel="tui")
    excel = _control_session(client, "alice", channel="excel")

    start = client.post(
      "/api/control/runs",
      headers=_headers(tui),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    record = app.state.subprocess_registry._tasks[run_id]
    record.log_path.write_text("scoped log\n", encoding="utf-8")

    wrong_list = client.get("/api/control/runs?kind=autonomous", headers=_headers(excel))
    assert wrong_list.status_code == 200
    assert wrong_list.json()["runs"] == []

    wrong_status = client.get(f"/api/control/runs/{run_id}", headers=_headers(excel))
    assert wrong_status.status_code == 404

    wrong_logs = client.get(f"/api/control/runs/{run_id}/logs", headers=_headers(excel))
    assert wrong_logs.status_code == 404

    wrong_cancel = client.delete(f"/api/control/runs/{run_id}", headers=_headers(excel))
    assert wrong_cancel.status_code == 404
    assert processes[0].returncode is None

    chat_session = app.state.auth.session_store.create_session(
      api_key_hash="test-hash",
      user_id="alice",
      user_email="alice@example.com",
      kind="chat",
    )
    chat_session.channel = "tui"
    chat_headers = {"Authorization": f"Bearer {app.state.auth.issue_token(chat_session)}"}

    chat_list = client.get("/api/control/runs", headers=chat_headers)
    assert chat_list.status_code == 401

    chat_status = client.get(f"/api/control/runs/{run_id}", headers=chat_headers)
    assert chat_status.status_code == 401

    chat_logs = client.get(f"/api/control/runs/{run_id}/logs", headers=chat_headers)
    assert chat_logs.status_code == 401

    chat_cancel = client.delete(f"/api/control/runs/{run_id}", headers=chat_headers)
    assert chat_cancel.status_code == 401
    assert processes[0].returncode is None

    right_list = client.get("/api/control/runs?kind=autonomous", headers=_headers(tui))
    assert right_list.status_code == 200
    assert [run["run_id"] for run in right_list.json()["runs"]] == [run_id]

    right_status = client.get(f"/api/control/runs/{run_id}", headers=_headers(tui))
    assert right_status.status_code == 200
    assert right_status.json()["run_id"] == run_id

    right_logs = client.get(f"/api/control/runs/{run_id}/logs", headers=_headers(tui))
    assert right_logs.status_code == 200
    assert right_logs.json()["log_lines"] == ["scoped log"]

    right_cancel = client.delete(f"/api/control/runs/{run_id}", headers=_headers(tui))
    assert right_cancel.status_code == 200
    assert right_cancel.json()["state"] == "cancelled"


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
        channel=None,
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
