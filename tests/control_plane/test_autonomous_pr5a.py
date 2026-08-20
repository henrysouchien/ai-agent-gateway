from __future__ import annotations

import asyncio
import hashlib
from itertools import count
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agent_gateway.autonomous_capability_handoff import (
  AutonomousCapabilityBinding,
)
from agent_gateway.autonomous_event_channel import (
  adopt_inherited_autonomous_event_channel,
)
from agent_gateway.autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
  verify_autonomous_launch_envelope,
)
from agent_gateway.capability_binding import (
  CapabilityBind,
  CredentialHandle,
)
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.control_plane import runs as runs_module
from agent_gateway.control_plane import runs_chat_helpers as chat_helpers_module
from agent_gateway.control_plane import runs_helpers as helpers_module
from agent_gateway.control_plane import runs_models as models_module
from agent_gateway.control_plane import runs_resume_helpers as resume_helpers_module
from agent_gateway.control_plane.runs import (
  _AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS,
  _AUTONOMOUS_RESUME_TOOL_RESULT_BLOCK_MAX_CHARS,
  _completed_tool_result_tail,
  _render_completed_tool_result_tail,
)
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.session import GatewaySession

from .manifest_helpers import TASK_MANIFEST_VERSION, write_v6_manifest


API_KEY = "autonomous-pr5a-key"
HMAC_KEY = "autonomous-pr5a-hmac-key-at-least-32-bytes"
API_DIR = Path(__file__).resolve().parents[4] / "api"
_MODEL_ENTRY = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:test-product:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="test-product",
  actor_id=None,
)
def _autonomous_capability_binding(request) -> AutonomousCapabilityBinding:
  return AutonomousCapabilityBinding(
    bind=request.required_bind
    or CapabilityBind(
      schema_version="1.0",
      capability_id="session.driver",
      model_key=_MODEL_ENTRY.key,
      provider=_MODEL_ENTRY.provider,
      upstream_model=_MODEL_ENTRY.upstream_model,
      adapter=_MODEL_ENTRY.adapter,
      protocol_profile=_MODEL_ENTRY.protocol_profile,
      route=_MODEL_ENTRY.route,
      effort="high",
      credential_principal="service",
      credential_ref=_SERVICE_HANDLE.handle_id,
      run_mode=request.run_mode,
      registry_revision=INITIAL_MODEL_REGISTRY.revision,
      policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
      selection_source="capability_default",
    ),
    materialized_credential=MaterializedCredential(
      handle=_SERVICE_HANDLE,
      auth_config={
        "provider": _SERVICE_HANDLE.provider,
        "auth_mode": "api",
        "api_key": "autonomous-pr5a-test-secret",
      },
    ),
  )


def test_control_plane_runs_parent_aliases_moved_helpers() -> None:
  assert runs_module.ChatRunResponse is helpers_module.ChatRunResponse
  assert helpers_module.ChatRunResponse is models_module.ChatRunResponse
  assert runs_module.AutonomousRunResponse is helpers_module.AutonomousRunResponse
  assert helpers_module.AutonomousRunResponse is models_module.AutonomousRunResponse
  assert helpers_module.ChatDispatchRequest is models_module.ChatDispatchRequest
  assert helpers_module.RunEnvelopeResponse is models_module.RunEnvelopeResponse
  assert runs_module._autonomous_state is helpers_module._autonomous_state
  assert runs_module._completed_tool_result_tail is resume_helpers_module._completed_tool_result_tail
  assert runs_module._render_completed_tool_result_tail is resume_helpers_module._render_completed_tool_result_tail
  assert runs_module._build_autonomous_resume_context is resume_helpers_module._build_autonomous_resume_context
  assert runs_module.cleanup_control_chat_tasks is chat_helpers_module.cleanup_control_chat_tasks
  assert runs_module._dispatch_control_chat_turn is chat_helpers_module._dispatch_control_chat_turn


def test_control_plane_run_status_vocab_normalizes_internal_states() -> None:
  assert helpers_module._autonomous_state("blocked") == "failed"
  assert helpers_module._autonomous_state("budget_limited") == "budget_limited"
  assert helpers_module._autonomous_state("remediating") == "running"
  assert helpers_module._TERMINAL_RUN_STATES == {
    "completed",
    "budget_limited",
    "failed",
    "interrupted",
    "cancelled",
  }


def test_chat_run_projection_ignores_child_terminal_events() -> None:
  session = GatewaySession(
    session_id="parent-chat",
    api_key_hash="hash",
    created_at=100,
    expires_at=200,
    user_id="alice",
  )
  session.event_history.append({
    "type": "error",
    "error": "child failed",
    "sub_agent_id": "sub0:spawned",
  })
  session.event_history.append({
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "sub_agent_id": "sub0:resumed",
  })
  session.event_history.append({
    "type": "stream_complete",
    "terminal_disposition": "interrupted",
  })

  run = helpers_module._chat_run_from_session(session)

  assert run.state == "interrupted"


_FAKE_PROCESS_PIDS = count(90_000)
_FAKE_PROCESSES: dict[int, "_FakeAutonomousProcess"] = {}


class _FakeStdin:
  def __init__(self) -> None:
    self.buffer = bytearray()

  def write(self, payload: bytes) -> None:
    self.buffer.extend(payload)

  async def drain(self) -> None:
    return None

  def close(self) -> None:
    return None

  async def wait_closed(self) -> None:
    return None


class _FakeAutonomousProcess:
  def __init__(self, inherited_event_fd: int, *, channel_id: str) -> None:
    self.pid = next(_FAKE_PROCESS_PIDS)
    self._returncode: int | None = None
    self._inherited_fds: list[int] = []
    self._event_channel = adopt_inherited_autonomous_event_channel(
      inherited_event_fd,
      channel_id=channel_id,
    )
    self._event_channel.start(timeout_seconds=2)
    self.stdin = _FakeStdin()
    _FAKE_PROCESSES[self.pid] = self

  @property
  def returncode(self) -> int | None:
    return self._returncode

  @returncode.setter
  def returncode(self, value: int | None) -> None:
    self._returncode = value
    if value is not None:
      self._close_inherited_fds()

  def _close_inherited_fds(self) -> None:
    self._event_channel.interrupt()
    for inherited_fd in self._inherited_fds:
      if inherited_fd >= 0:
        os.close(inherited_fd)
    self._inherited_fds.clear()

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    self._close_inherited_fds()
    return self.returncode

  def retain_inherited_fds(self, inherited_fds: tuple[int, ...]) -> None:
    self._inherited_fds.extend(os.dup(fd) for fd in inherited_fds)

  def terminate(self) -> None:
    if self.returncode is None:
      self.returncode = -15

  def kill(self) -> None:
    if self.returncode is None:
      self.returncode = -9


def _write_resumable_skill(skills_dir: Path) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / "resumable-skill.md").write_text(
    """---
name: resumable-skill
description: Resumable test skill
agent_callable: true
resumable: true
mutation_mode: read_only
required_context: []
requires_portfolio_context: false
catalog: false
semantic_metadata:
  contract_name: skill-metadata
  schema_version: '2'
  catalog_version: skill-catalog/2
  skill_id: resumable-skill
  tool_refs: []
  allowed_effects: [read]
  approval_constraints: [runtime_policy]
  output_contracts:
    - owner: platform
      contract_name: skill-result-envelope
      schema_version: '1'
  credential_requirements: []
  scheduling:
    eligibility: ineligible
    opt_in: not_required
  allowed_profiles: [analyst]
---
Run the resumable test skill.
""",
    encoding="utf-8",
  )


class _NoopRunner:
  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns


def _make_app(
  monkeypatch,
  tmp_path: Path,
  *,
  control_skills_dir: Path | None = None,
  dispatch_scope_validator: Any | None = None,
  claim_signing_authority_installed: bool = True,
  autonomous_api_dir: Path = API_DIR,
):
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "autonomous-logs"))

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid: _runner_with_log(event_log),
      capability_execution=request.capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="autonomous-pr5a-test-secret-0123456789",
      valid_api_keys={API_KEY},
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      build_chat_runtime=_build_chat_runtime,
      autonomous_capability_binding_resolver=_autonomous_capability_binding,
      autonomous_api_dir=autonomous_api_dir,
      control_skills_dir=control_skills_dir,
      dispatch_scope_validator=dispatch_scope_validator,
      claim_signing_authority=(
        GatewayClaimSigningAuthority(HMAC_KEY)
        if claim_signing_authority_installed
        else None
      ),
    )
  )


def _autonomous_log_dir(tmp_path: Path) -> Path:
  path = tmp_path / "autonomous-logs"
  path.mkdir(parents=True, exist_ok=True)
  return path


def test_gateway_app_without_autonomous_api_dir_retains_source_default(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway.server_artifact_helpers import _default_autonomous_api_dir

  app = _make_app(
    monkeypatch,
    tmp_path,
    autonomous_api_dir=None,
  )

  assert (
    app.state.subprocess_registry._api_dir.resolve()
    == _default_autonomous_api_dir().resolve()
  )


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
  capability_bind = CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key=_MODEL_ENTRY.key,
    provider=_MODEL_ENTRY.provider,
    upstream_model=_MODEL_ENTRY.upstream_model,
    adapter=_MODEL_ENTRY.adapter,
    protocol_profile=_MODEL_ENTRY.protocol_profile,
    route=_MODEL_ENTRY.route,
    effort="high",
    credential_principal="service",
    credential_ref=_SERVICE_HANDLE.handle_id,
    run_mode="autonomous",
    registry_revision=INITIAL_MODEL_REGISTRY.revision,
    policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
    selection_source="capability_default",
  )
  return write_v6_manifest(
    log_dir,
    task_id,
    control_run_id=control_run_id or task_id,
    user_id=user_id,
    user_email=user_email,
    mode=mode,
    task=None if mode == "skill" else "summarize",
    skill=skill,
    context=context,
    channel=channel,
    cmd=[sys.executable, "-m", "agent.autonomous", "--profile", "analyst"],
    state=state,
    exit_code=exit_code,
    error=error,
    completed_at=completed_at,
    capability_bind=capability_bind.receipt(),
  )


def _write_rehydrate_events(tmp_path: Path, task_id: str, events: list[dict[str, Any]]) -> None:
  log_dir = _autonomous_log_dir(tmp_path)
  (log_dir / f"{task_id}.events.jsonl").write_text(
    "".join(json.dumps(event) + "\n" for event in events),
    encoding="utf-8",
  )


def _readable_resource_event(
  *,
  resource_id: str = "rr:daily-note",
  control_run_id: str = "run-readable",
  skill_run_id: str = "skill-readable",
  content: str = "## Daily note\n\nCaptured markdown.\n",
  content_class: str = "human_readable",
) -> dict[str, Any]:
  content_bytes = content.encode("utf-8")
  digest = hashlib.sha256(content_bytes).hexdigest()
  return {
    "type": "readable_resource_ready",
    "resource_id": resource_id,
    "run_id": control_run_id,
    "control_run_id": control_run_id,
    "skill_run_id": skill_run_id,
    "contract_name": "MarkdownNote",
    "content_type": "text/markdown",
    "content_class": content_class,
    "content_snapshot_id": f"sha256:{digest}",
    "content_sha256": digest,
    "content_bytes": len(content_bytes),
    "content": content,
    "truncated": False,
    "title": "daily/2026-06-12.md",
    "source_path": "daily/2026-06-12.md",
    "byte_start": 0,
    "byte_end": len(content_bytes),
    "tool_name": "memory_write",
    "created_at": "2026-06-12T15:46:49Z",
    "ts": 190,
  }


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


@pytest.mark.parametrize(
  ("stored_role", "live_role"),
  [("invite", "owner"), ("owner", "invite")],
)
def test_autonomous_resume_uses_live_role_in_both_directions(
  monkeypatch,
  tmp_path,
  stored_role: str,
  live_role: str,
) -> None:
  processes, envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
  app = _make_app(monkeypatch, tmp_path, control_skills_dir=skills_dir)

  with TestClient(app) as client:
    session_payload = _control_session(client, "alice")
    session = app.state.auth.session_store.get_session(session_payload["session_id"])
    assert session is not None
    session.role = stored_role
    headers = {
      "Authorization": f"Bearer {app.state.auth.issue_token(session)}"
    }
    started = client.post(
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
    assert started.status_code == 200, started.text
    original = app.state.subprocess_registry._tasks["bg_0"]
    assert original.role == stored_role
    original.dev_mode = True  # Historical manifests may retain this retired field.
    original.state = "failed"
    original.exit_code = 1
    original.completed_at = time.time()
    processes[0].returncode = 1

    session.role = live_role
    headers = {
      "Authorization": f"Bearer {app.state.auth.issue_token(session)}"
    }
    resumed = client.post(
      "/api/control/runs/bg_0/resume",
      headers=headers,
      json={},
    )
    assert resumed.status_code == 200, resumed.text
    resumed_record = app.state.subprocess_registry._tasks["bg_1"]
    assert resumed_record.role == live_role
    assert resumed_record.dev_mode is False
    assert "--dev" not in resumed_record.cmd
    assert "ANALYST_DEV_MODE" not in envs[1]


def _install_fake_spawn(
  monkeypatch,
  *,
  invocations: list[dict[str, Any]] | None = None,
) -> tuple[list[_FakeAutonomousProcess], list[dict[str, str]]]:
  from agent_gateway import autonomous_runner

  processes: list[_FakeAutonomousProcess] = []
  envs: list[dict[str, str]] = []

  async def fake_exec(*args, **kwargs):
    _ = args
    envelope = json.loads(
      kwargs["env"][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV]
    )
    process = _FakeAutonomousProcess(
      os.dup(kwargs["pass_fds"][0]),
      channel_id=envelope["channel_id"],
    )
    process.retain_inherited_fds(tuple(kwargs["pass_fds"][1:]))
    processes.append(process)
    envs.append(dict(kwargs["env"]))
    if invocations is not None:
      invocations.append(dict(kwargs))
    return process

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setattr(autonomous_runner, "_get_process_group_id", lambda pid: pid)

  def signal_process_group(process_group_id: int, signal_number: int) -> None:
    _FAKE_PROCESSES[process_group_id].returncode = -signal_number

  monkeypatch.setattr(autonomous_runner, "_signal_process_group", signal_process_group)
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
        "max_budget_usd": 5.0,
      },
    )
    assert start.status_code == 200, start.text
    start_payload = start.json()
    assert start_payload["task_id"] == "bg_0"
    assert start_payload["run"]["run_id"] == "bg_0"
    assert start_payload["run"]["state"] == "running"
    assert "--ticker" in start_payload["cmd"]
    assert start_payload["cmd"][start_payload["cmd"].index("--ticker") + 1] == "AAPL"

    envelope = verify_autonomous_launch_envelope(
      HMAC_KEY,
      envs[0][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
    )
    authority = envelope.session_authority.ordinary_authority
    assert authority.user_id == "alice"
    assert authority.user_email == "alice@example.com"
    assert "AGENT_API_CLAIM_USER_EMAIL" not in envs[0]
    assert "AGENT_API_USER_CLAIM_HMAC_KEY" not in envs[0]
    assert "AGENT_AUTONOMOUS_EVENTS_PATH" not in envs[0]

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


def test_live_two_user_run_matrix_is_disjoint_and_cross_user_actions_are_404(
  monkeypatch,
  tmp_path,
) -> None:
  app = _make_app(monkeypatch, tmp_path)
  registry = app.state.subprocess_registry
  sequence = count()

  async def fake_start(**kwargs: Any) -> dict[str, str]:
    task_id = f"bg_live_{next(sequence)}"
    log_path = tmp_path / f"{task_id}.log"
    log_path.write_text("", encoding="utf-8")
    registry._tasks[task_id] = SimpleNamespace(
      task_id=task_id,
      control_run_id=task_id,
      session_id=task_id,
      profile=kwargs["profile"],
      mode=kwargs["mode"],
      skill=kwargs.get("skill"),
      task=kwargs.get("task"),
      ticker=kwargs.get("ticker"),
      channel=kwargs.get("channel"),
      user_id=kwargs["user_id"],
      owner_user_id=kwargs["owner_user_id"],
      raw_user_id=kwargs["user_id"],
      user_slug=kwargs.get("user_slug"),
      risk_user_id=kwargs.get("risk_user_id") or 0,
      user_email=kwargs.get("user_email"),
      user_aliases=list(kwargs.get("user_aliases") or []),
      identity_status=kwargs.get("identity_status") or "legacy_user_id_fallback",
      state="running",
      exit_code=None,
      error=None,
      proc=SimpleNamespace(returncode=None),
      operator_inbox_path=None,
      owner_lifeline_fd=None,
      approval_channel=None,
      event_channel=None,
      log_handle=None,
      started_at=time.time(),
      completed_at=None,
      event_lines=[],
      resumed_from=None,
      resumed_as=[],
      dispatch_scope=kwargs.get("dispatch_scope"),
      schedule_id=None,
      schedule_name=None,
      capability_bind=None,
      max_budget_usd=kwargs.get("max_budget_usd"),
      log_path=log_path,
      cmd=["recorded-fake-run"],
    )
    return {"task_id": task_id, "run_id": task_id}

  def fake_logs(task_id: str, *, tail: int) -> dict[str, Any]:
    lines = registry._tasks[task_id].log_path.read_text(encoding="utf-8").splitlines()
    return {"lines": lines[-tail:] if tail else [], "total_lines": len(lines)}

  async def fake_cancel(task_id: str) -> dict[str, Any]:
    record = registry._tasks[task_id]
    record.state = "killed"
    record.proc.returncode = -15
    record.completed_at = time.time()
    return {"task_id": task_id, "state": "killed"}

  monkeypatch.setattr(registry, "start", fake_start)
  monkeypatch.setattr(registry, "logs", fake_logs)
  monkeypatch.setattr(registry, "cancel", fake_cancel)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    bob = _control_session(client, "bob")
    starts: dict[str, dict[str, Any]] = {}
    for owner, session in (("alice", alice), ("bob", bob)):
      response = client.post(
        "/api/control/runs",
        headers=_headers(session),
        json={
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "task",
          "task": f"Review {owner}'s portfolio",
          "channel": "tui",
        },
      )
      assert response.status_code == 200, response.text
      starts[owner] = response.json()

    for owner, session in (("alice", alice), ("bob", bob)):
      listed = client.get(
        "/api/control/runs?kind=autonomous",
        headers=_headers(session),
      )
      assert listed.status_code == 200, listed.text
      assert [run["run_id"] for run in listed.json()["runs"]] == [
        starts[owner]["run_id"]
      ]

    for owner, session, other_session in (
      ("alice", alice, bob),
      ("bob", bob, alice),
    ):
      run_id = starts[owner]["run_id"]
      record = app.state.subprocess_registry._find_by_control_run_id(run_id)
      assert record is not None
      record.log_path.write_text(f"{owner} only\n", encoding="utf-8")

      detail = client.get(f"/api/control/runs/{run_id}", headers=_headers(session))
      logs = client.get(f"/api/control/runs/{run_id}/logs", headers=_headers(session))
      assert detail.status_code == 200, detail.text
      assert detail.json()["owner_user_id"] == owner
      assert logs.status_code == 200, logs.text
      assert logs.json()["log_lines"] == [f"{owner} only"]

      assert client.get(
        f"/api/control/runs/{run_id}", headers=_headers(other_session)
      ).status_code == 404
      assert client.get(
        f"/api/control/runs/{run_id}/logs", headers=_headers(other_session)
      ).status_code == 404
      assert client.delete(
        f"/api/control/runs/{run_id}", headers=_headers(other_session)
      ).status_code == 404
      assert record.proc is not None and record.proc.returncode is None

    for owner, session in (("alice", alice), ("bob", bob)):
      cancelled = client.delete(
        f"/api/control/runs/{starts[owner]['run_id']}",
        headers=_headers(session),
      )
      assert cancelled.status_code == 200, cancelled.text
      assert cancelled.json()["state"] == "cancelled"

def test_autonomous_control_route_requires_and_uses_installed_claim_authority(
  monkeypatch,
  tmp_path,
) -> None:
  calls: list[dict[str, Any]] = []
  payload = {
    "kind": "autonomous",
    "profile": "analyst",
    "mode": "task",
    "task": "summarize",
    "channel": "tui",
  }

  installed_app = _make_app(monkeypatch, tmp_path)
  record = SimpleNamespace(
    task_id="bg_stub",
    control_run_id="bg_stub",
    log_path=tmp_path / "bg_stub.log",
    started_at=1,
    cmd=["stub-runner"],
  )

  async def stub_start(**kwargs):
    assert type(
      installed_app.state.gateway_claim_signing_authority
    ) is GatewayClaimSigningAuthority
    calls.append(dict(kwargs))
    return {"task_id": record.task_id}

  installed_app.state.subprocess_registry.start = stub_start
  monkeypatch.setattr(
    runs_module,
    "_autonomous_task_for_user",
    lambda registry, task_id, owner_user_id: record,
  )
  monkeypatch.setattr(
    runs_module,
    "_autonomous_run_from_task",
    lambda _record, *, skills_dir=None: models_module.AutonomousRunResponse(
      kind="autonomous",
      run_id=record.control_run_id,
      task_id=record.task_id,
      agent="hank",
      profile="analyst",
      mode="task",
      skill=None,
      task="summarize",
      ticker=None,
      channel="tui",
      user_id="alice",
      state="running",
      started_at="1970-01-01T00:00:01+00:00",
      ended_at=None,
      cost_usd=None,
      skill_run_ids=[],
      current_verdict=None,
    ),
  )
  with TestClient(installed_app) as client:
    installed_session = _control_session(client, "alice")
    accepted = client.post(
      "/api/control/runs",
      headers=_headers(installed_session),
      json=payload,
    )

  assert accepted.status_code == 200, accepted.text
  assert accepted.json()["run_id"] == "bg_stub"
  assert len(calls) == 1

  absent_app = _make_app(
    monkeypatch,
    tmp_path,
    claim_signing_authority_installed=False,
  )
  with TestClient(absent_app) as client:
    absent_session = _control_session(client, "bob")
    rejected = client.post(
      "/api/control/runs",
      headers=_headers(absent_session),
      json=payload,
    )

  assert rejected.status_code == 409
  assert rejected.json() == {
    "detail": "autonomous dispatch requires installed claim-signing authority"
  }
  assert len(calls) == 1


def test_autonomous_runs_are_scoped_by_canonical_owner_alias(monkeypatch, tmp_path) -> None:
  invocations: list[dict[str, Any]] = []
  _processes, envs = _install_fake_spawn(
    monkeypatch,
    invocations=invocations,
  )
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([
      {
        "key": API_KEY,
        "channel": "mcp",
        "slug": "henry",
        "email": "henry@example.com",
        "risk_user_id": 1,
        "role": "owner",
      },
      {
        "key": "other-user-key",
        "channel": "mcp",
        "slug": "other",
        "email": "other@example.com",
        "risk_user_id": 2,
        "role": "owner",
      },
    ]),
  )
  application_api_dir = tmp_path / "installed-application-api"
  application_api_dir.mkdir()
  (application_api_dir / "user_identity.py").write_text(
    """
import json
import os

def get_mcp_user_key_entry(user_id, user_email=None):
  for entry in json.loads(os.environ.get("GATEWAY_USER_KEYS", "[]")):
    if entry.get("slug") == user_id or entry.get("email") == user_email:
      return dict(entry)
  return None
""".lstrip(),
    encoding="utf-8",
  )
  app = _make_app(
    monkeypatch,
    tmp_path,
    autonomous_api_dir=application_api_dir,
  )

  with TestClient(app) as client:
    henry_mcp = _control_session(client, "henry", email="henry@example.com")
    henry_web = _control_session(client, "1", email="henry@example.com")
    alice = _control_session(client, "alice", email="alice@example.com")

    assert henry_mcp["user_id"] == "1"
    assert henry_mcp["user_slug"] == "henry"
    assert henry_web["user_id"] == "1"
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "user_identity", raising=False)

    start = client.post(
      "/api/control/runs",
      headers=_headers(henry_mcp),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "earnings-review",
        "context": "Review AAPL earnings",
        "ticker": "AAPL",
        "channel": "tui",
        "max_budget_usd": 5.0,
      },
    )
    assert start.status_code == 200, start.text
    start_run = start.json()["run"]
    assert start_run["user_id"] == "1"
    assert start_run["owner_user_id"] == "1"
    assert start_run["raw_user_id"] == "henry"
    assert start_run["user_slug"] == "henry"
    assert start_run["risk_user_id"] == 1
    assert start_run["user_email"] == "henry@example.com"
    assert start_run["user_aliases"] == ["1", "henry", "henry@example.com"]
    assert start_run["identity_status"] == "gateway_user_key_mapping"
    assert start_run["max_budget_usd"] == 5.0
    assert start.json()["cmd"][start.json()["cmd"].index("--max-budget-usd") + 1] == "5.0"
    envelope = verify_autonomous_launch_envelope(
      HMAC_KEY,
      envs[0][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
    )
    authority = envelope.session_authority.ordinary_authority
    assert authority.user_id == "1"
    assert authority.raw_user_id == "henry"
    assert "AUTONOMOUS_USER_ID" not in envs[0]
    assert "AUTONOMOUS_RAW_USER_ID" not in envs[0]
    assert Path(invocations[0]["cwd"]).resolve() == application_api_dir.resolve()
    assert json.loads(envs[0]["GATEWAY_USER_KEYS"]) == [
      {
        "key": API_KEY,
        "slug": "henry",
        "email": "henry@example.com",
        "risk_user_id": 1,
        "channel": "mcp",
        "role": "owner",
      }
    ]

    manifest = json.loads((_autonomous_log_dir(tmp_path) / "bg_0.task.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == TASK_MANIFEST_VERSION
    assert manifest["owner_user_id"] == "1"
    assert manifest["user_id"] == "1"
    assert manifest["raw_user_id"] == "henry"
    assert manifest["user_slug"] == "henry"
    assert manifest["user_aliases"] == ["1", "henry", "henry@example.com"]
    assert manifest["max_budget_usd"] == 5.0

    detail = client.get("/api/control/runs/bg_0", headers=_headers(henry_web))
    assert detail.status_code == 200
    detail_run = detail.json()
    assert detail_run["user_id"] == "1"
    assert detail_run["owner_user_id"] == "1"
    assert detail_run["raw_user_id"] == "henry"
    assert detail_run["user_slug"] == "henry"
    assert detail_run["risk_user_id"] == 1
    assert detail_run["user_email"] == "henry@example.com"
    assert detail_run["user_aliases"] == ["1", "henry", "henry@example.com"]
    assert detail_run["identity_status"] == "gateway_user_key_mapping"
    assert detail_run["max_budget_usd"] == 5.0
    listed = client.get("/api/control/runs?kind=autonomous", headers=_headers(henry_web))
    assert listed.status_code == 200
    listed_run = listed.json()["runs"][0]
    assert listed_run["owner_user_id"] == "1"
    assert listed_run["raw_user_id"] == "henry"
    assert listed_run["identity_status"] == "gateway_user_key_mapping"
    assert listed_run["max_budget_usd"] == 5.0
    assert client.get("/api/control/runs/bg_0", headers=_headers(alice)).status_code == 404


def test_autonomous_dispatch_persists_redacted_dispatch_scope(monkeypatch, tmp_path) -> None:
  _processes, envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)
  dispatch_scope = {
    "kind": "portfolio",
    "source": "user_selected",
    "portfolio_name": "taxable_combined",
    "portfolio_id": None,
    "display_name": "Taxable Combined",
  }

  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web", email="alice@example.com")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "Review risk drift.",
        "dispatch_scope": dispatch_scope,
      },
    )

    assert start.status_code == 200, start.text
    start_payload = start.json()
    assert start_payload["run"]["dispatch_scope"] == dispatch_scope
    assert "--context" not in start_payload["cmd"]
    assert all("Portfolio:" not in part for part in start_payload["cmd"])
    envelope = verify_autonomous_launch_envelope(
      HMAC_KEY,
      envs[0][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
    )
    assert envelope.session_authority.dispatch_scope.receipt() == dispatch_scope

    manifest = json.loads((_autonomous_log_dir(tmp_path) / "bg_0.task.json").read_text(encoding="utf-8"))
    assert manifest["dispatch_scope"] == dispatch_scope

    detail = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    assert detail.status_code == 200, detail.text
    assert detail.json()["dispatch_scope"] == dispatch_scope
    listed = client.get("/api/control/runs?kind=autonomous", headers=_headers(alice))
    assert listed.status_code == 200, listed.text
    assert listed.json()["runs"][0]["dispatch_scope"] == dispatch_scope


def test_autonomous_dispatch_rejects_unredacted_dispatch_scope(monkeypatch, tmp_path) -> None:
  _processes, envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web")
    response = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "Review risk drift.",
        "dispatch_scope": {
          "kind": "portfolio",
          "source": "user_selected",
          "portfolio_name": "taxable_combined",
          "account_id": "acc-1",
        },
      },
    )

  assert response.status_code == 422, response.text
  assert envs == []


def test_autonomous_dispatch_rejects_dispatch_scope_mismatch_before_spawn(monkeypatch, tmp_path) -> None:
  _processes, envs = _install_fake_spawn(monkeypatch)

  async def validator(_session, scope: dict[str, Any]) -> dict[str, Any]:
    assert scope["portfolio_name"] == "taxable_combined"
    assert scope["portfolio_id"] == "portfolio-2"
    raise HTTPException(
      status_code=422,
      detail={
        "error": "dispatch_scope_portfolio_mismatch",
        "field": "dispatch_scope",
      },
    )

  app = _make_app(monkeypatch, tmp_path, dispatch_scope_validator=validator)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web")
    response = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "Review risk drift.",
        "dispatch_scope": {
          "kind": "portfolio",
          "source": "user_selected",
          "portfolio_name": "taxable_combined",
          "portfolio_id": "portfolio-2",
        },
      },
    )

  assert response.status_code == 422, response.text
  assert response.json()["detail"]["error"] == "dispatch_scope_portfolio_mismatch"
  assert envs == []


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
  assert record.error == (
    "Process exited with code 1; "
    "autonomous event channel failed: AutonomousEventChannelProtocolError: "
    "autonomous event channel closed before END"
  )
  assert detail["state"] == "failed"
  assert any(
    event.get("type") == "run_state_changed" and event.get("state") == "failed"
    for event in record.event_lines or ()
  )


def _rehydrated_owner_scope_responses(monkeypatch, tmp_path):
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
        "cost_usd": 0.12,
        "verdict_echo": {
          "verdict_token": "monitor",
          "confidence": "medium",
          "one_line_summary": "Watch the setup",
        },
        "fms_results": [
          {
            "proposal_id": "prop-monitoring-1",
            "status": "staged",
            "expires_at": 195.0,
            "subcommand": "propose_monitoring_init",
            "ticker": "STWD",
            "readback": {"research_file_id": 2042},
          }
        ],
        "ts": 190,
      },
      {
        "type": "stream_complete",
        "terminal_disposition": "completed",
        "run_id": "run-rehydrated",
        "control_run_id": "run-rehydrated",
        "ts": 199,
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

  return listed, detail, bob_listed, bob_detail


def test_rehydrated_autonomous_run_is_owner_scoped(monkeypatch, tmp_path) -> None:
  listed, detail, bob_listed, bob_detail = _rehydrated_owner_scope_responses(monkeypatch, tmp_path)
  assert listed.status_code == 200, listed.text
  runs = listed.json()["runs"]
  assert [run["run_id"] for run in runs] == ["run-rehydrated"]
  assert runs[0]["state"] == "completed"
  assert detail.status_code == 200, detail.text
  assert detail.json()["run_id"] == "run-rehydrated"
  assert bob_listed.status_code == 200
  assert bob_listed.json()["runs"] == []
  assert bob_detail.status_code == 404


def test_rehydrated_autonomous_run_event_derived_fields(monkeypatch, tmp_path) -> None:
  listed, detail, _bob_listed, _bob_detail = _rehydrated_owner_scope_responses(monkeypatch, tmp_path)
  runs = listed.json()["runs"]
  assert runs[0]["cost_usd"] == 0.12
  assert runs[0]["skill_run_ids"] == ["skill-run-1"]
  assert runs[0]["current_verdict"] == {
    "verdict_token": "monitor",
    "confidence": "medium",
    "one_line_summary": "Watch the setup",
    "skill_run_id": "skill-run-1",
  }
  assert runs[0]["staged_proposals"] == [
    {
      "proposal_id": "prop-monitoring-1",
      "status": "staged",
      "requires_apply": True,
      "expires_at": "1970-01-01T00:03:15Z",
      "subcommand": "propose_monitoring_init",
      "ticker": "STWD",
      "research_file_id": 2042,
      "skill_run_id": "skill-run-1",
    }
  ]
  detail_payload = detail.json()
  assert detail_payload["cost_usd"] == 0.12
  assert {
    "kind": "skill_run",
    "ref": "skill-run-1",
    "skill_run_id": "skill-run-1",
  } in detail_payload["terminal_receipt"]["result_refs"]


def _rehydrated_readable_resource_responses(monkeypatch, tmp_path):
  content = "## Daily note\n\nCaptured markdown.\n"
  _write_rehydrate_manifest(
    tmp_path,
    "bg_4",
    control_run_id="run-readable",
    user_id="alice",
    user_email="alice@example.com",
    state="completed",
    completed_at=200.0,
  )
  _write_rehydrate_events(
    tmp_path,
    "bg_4",
    [
      _readable_resource_event(content=content),
      _readable_resource_event(
        resource_id="rr:dev-only",
        control_run_id="run-readable",
        skill_run_id="skill-dev",
        content="{}",
        content_class="dev_only",
      ),
      _readable_resource_event(
        resource_id="rr:spoofed",
        control_run_id="other-run",
        skill_run_id="skill-spoofed",
        content="## Spoofed\n",
      ),
    ],
  )
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")
    bob = _control_session(client, "bob", email="bob@example.com")

    listed = client.get("/api/control/readable-resources?limit=24", headers=_headers(alice))
    filtered = client.get("/api/control/readable-resources?run_id=run-readable", headers=_headers(alice))
    detail = client.get("/api/control/readable-resources/rr:daily-note", headers=_headers(alice))
    spoofed = client.get("/api/control/readable-resources/rr:spoofed", headers=_headers(alice))
    bob_detail = client.get("/api/control/readable-resources/rr:daily-note", headers=_headers(bob))
    run_detail = client.get("/api/control/runs/run-readable", headers=_headers(alice))
    bob_run_detail = client.get("/api/control/runs/run-readable", headers=_headers(bob))

  return listed, filtered, detail, spoofed, bob_detail, run_detail, bob_run_detail


def test_readable_resources_rehydrated_run_remains_owner_scoped(monkeypatch, tmp_path) -> None:
  _listed, _filtered, _detail, _spoofed, _bob_detail, run_detail, bob_run_detail = (
    _rehydrated_readable_resource_responses(monkeypatch, tmp_path)
  )
  assert run_detail.status_code == 200, run_detail.text
  assert run_detail.json()["state"] == "completed"
  assert bob_run_detail.status_code == 404


def test_readable_resources_list_detail_visible_rehydrated_snapshots(monkeypatch, tmp_path) -> None:
  content = "## Daily note\n\nCaptured markdown.\n"
  listed, filtered, detail, spoofed, bob_detail, _run_detail, _bob_run_detail = (
    _rehydrated_readable_resource_responses(monkeypatch, tmp_path)
  )
  assert listed.status_code == 200, listed.text
  payload = listed.json()
  assert payload["next_cursor"] is None
  assert len(payload["readable_resources"]) == 1
  listed_resource = payload["readable_resources"][0]
  assert listed_resource["resource_id"] == "rr:daily-note"
  assert listed_resource["control_run_id"] == "run-readable"
  assert listed_resource["skill_run_id"] == "skill-readable"
  assert listed_resource["content_class"] == "human_readable"
  assert "content" not in listed_resource
  assert filtered.status_code == 200
  assert filtered.json()["readable_resources"] == [listed_resource]
  assert detail.status_code == 200, detail.text
  detail_payload = detail.json()
  assert detail_payload["content"] == content
  assert detail_payload["content_sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
  assert spoofed.status_code == 404
  assert bob_detail.status_code == 404


def _terminal_cost_detail(monkeypatch, tmp_path):
  _write_rehydrate_manifest(
    tmp_path,
    "bg_5",
    control_run_id="run-cost",
    user_id="alice",
    state="completed",
    completed_at=210.0,
  )
  _write_rehydrate_events(
    tmp_path,
    "bg_5",
    [
      {
        "type": "turn_complete",
        "run_id": "run-cost",
        "control_run_id": "run-cost",
        "turn": 1,
        "usage": {"input_tokens": 100, "output_tokens": 20, "estimated_cost": 0.03},
      },
      {
        "type": "turn_complete",
        "run_id": "run-cost",
        "control_run_id": "run-cost",
        "turn": 2,
        "usage": {"input_tokens": 200, "output_tokens": 40, "estimated_cost": 0.04},
      },
      {
        "type": "stream_complete",
        "run_id": "run-cost",
        "control_run_id": "run-cost",
        "usage": {"input_tokens": 300, "output_tokens": 60, "estimated_cost": 0.08},
      },
    ],
  )
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")
    detail = client.get("/api/control/runs/run-cost", headers=_headers(alice))

  return detail


def test_rehydrated_terminal_cost_run_is_visible(monkeypatch, tmp_path) -> None:
  detail = _terminal_cost_detail(monkeypatch, tmp_path)
  assert detail.status_code == 200, detail.text


def test_run_cost_prefers_terminal_stream_complete_total(monkeypatch, tmp_path) -> None:
  detail = _terminal_cost_detail(monkeypatch, tmp_path)
  assert detail.json()["cost_usd"] == 0.08


def _budget_exceeded_detail(monkeypatch, tmp_path):
  _write_rehydrate_manifest(
    tmp_path,
    "bg_8",
    control_run_id="run-budget",
    user_id="alice",
    state="completed",
    exit_code=0,
    error=None,
    completed_at=210.0,
  )
  _write_rehydrate_events(
    tmp_path,
    "bg_8",
    [{"type": "budget_exceeded", "run_id": "run-budget", "control_run_id": "run-budget", "total_cost": 1.25, "budget": 1.0, "ts": 209}],
  )
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")
    detail = client.get("/api/control/runs/run-budget", headers=_headers(alice))

  return detail


def test_rehydrated_budget_event_run_is_visible(monkeypatch, tmp_path) -> None:
  detail = _budget_exceeded_detail(monkeypatch, tmp_path)
  assert detail.status_code == 200, detail.text
  assert detail.json()["ended_at"] is not None


def test_autonomous_budget_exceeded_event_maps_to_budget_limited_run_state(
  monkeypatch,
  tmp_path,
) -> None:
  detail = _budget_exceeded_detail(monkeypatch, tmp_path)
  payload = detail.json()
  assert payload["state"] == "budget_limited"
  assert payload["resumable"] is False
  assert payload["terminal_receipt"]["disposition"] == "budget_limited"
  assert payload["terminal_receipt"]["error"] is None


@pytest.mark.parametrize(
  ("internal_state", "projected_state"),
  [
    ("budget_limited", "budget_limited"),
    ("budget_exceeded", "budget_limited"),
    ("blocked", "failed"),
  ],
)
def test_internal_autonomous_terminal_states_do_not_inherit_failed_resume_affordance(
  monkeypatch,
  tmp_path,
  internal_state: str,
  projected_state: str,
) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
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
      },
    )
    assert start.status_code == 200, start.text
    record = app.state.subprocess_registry._tasks["bg_0"]
    record.state = internal_state
    record.exit_code = 1
    record.completed_at = time.time()
    processes[0].returncode = 1

    detail = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    resumed = client.post("/api/control/runs/bg_0/resume", headers=_headers(alice), json={})

  assert detail.status_code == 200, detail.text
  assert detail.json()["state"] == projected_state
  assert detail.json()["resumable"] is False
  assert resumed.status_code == 409, resumed.text
  assert resumed.json()["detail"] == "Autonomous run is not resumable"


def _turn_estimate_cost_detail(monkeypatch, tmp_path):
  _write_rehydrate_manifest(
    tmp_path,
    "bg_6",
    control_run_id="run-live-cost",
    user_id="alice",
    state="completed",
    completed_at=220.0,
  )
  _write_rehydrate_events(
    tmp_path,
    "bg_6",
    [
      {
        "type": "turn_complete",
        "run_id": "run-live-cost",
        "control_run_id": "run-live-cost",
        "turn": 1,
        "usage": {"input_tokens": 100, "output_tokens": 20, "estimated_cost": 0.01},
      },
      {
        "type": "turn_complete",
        "run_id": "run-live-cost",
        "control_run_id": "run-live-cost",
        "turn": 2,
        "usage": {"input_tokens": 200, "output_tokens": 40, "estimated_cost": 0.02},
      },
    ],
  )
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")
    detail = client.get("/api/control/runs/run-live-cost", headers=_headers(alice))

  return detail


def test_rehydrated_turn_cost_run_is_visible(monkeypatch, tmp_path) -> None:
  detail = _turn_estimate_cost_detail(monkeypatch, tmp_path)
  assert detail.status_code == 200, detail.text


def test_run_cost_sums_turn_estimates_when_terminal_total_missing(monkeypatch, tmp_path) -> None:
  detail = _turn_estimate_cost_detail(monkeypatch, tmp_path)
  assert detail.json()["cost_usd"] == 0.03


def test_autonomous_state_passes_interrupted_through() -> None:
  from agent_gateway.control_plane.runs import _autonomous_state

  assert _autonomous_state("interrupted") == "interrupted"


def test_autonomous_state_maps_budget_aliases_to_budget_limited() -> None:
  from agent_gateway.control_plane.runs import _autonomous_state

  assert _autonomous_state("budget_limited") == "budget_limited"
  assert _autonomous_state("budget_exceeded") == "budget_limited"


def _interrupted_terminal_detail(
  monkeypatch,
  tmp_path,
) -> None:
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
  _write_rehydrate_manifest(
    tmp_path,
    "bg_9",
    control_run_id="run-interrupted-terminal",
    user_id="alice",
    state="completed",
    skill="resumable-skill",
    exit_code=0,
    error=None,
    completed_at=220.0,
  )
  _write_rehydrate_events(
    tmp_path,
    "bg_9",
    [
      {
        "type": "stream_complete",
        "terminal_disposition": "interrupted",
        "reason": "operator_pause",
        "run_id": "run-interrupted-terminal",
        "control_run_id": "run-interrupted-terminal",
        "ts": 219,
      }
    ],
  )
  app = _make_app(
    monkeypatch,
    tmp_path,
    control_skills_dir=skills_dir,
  )

  with TestClient(app) as client:
    alice = _control_session(client, "alice", email="alice@example.com")
    detail = client.get(
      "/api/control/runs/run-interrupted-terminal",
      headers=_headers(alice),
    )

  return detail


def test_rehydrated_terminal_run_is_visible(monkeypatch, tmp_path) -> None:
  detail = _interrupted_terminal_detail(monkeypatch, tmp_path)
  assert detail.status_code == 200, detail.text


def test_interrupted_terminal_projects_resumable_interrupted_control_run(
  monkeypatch,
  tmp_path,
) -> None:
  detail = _interrupted_terminal_detail(monkeypatch, tmp_path)
  payload = detail.json()
  assert payload["state"] == "interrupted"
  assert payload["resumable"] is True


@pytest.mark.parametrize(
  ("internal_state", "projected_state"),
  [
    ("starting", "starting"),
    ("queued", "queued"),
    ("remediating", "running"),
  ],
)
def test_autonomous_messageable_matches_operator_message_acceptance(
  monkeypatch,
  tmp_path,
  internal_state: str,
  projected_state: str,
) -> None:
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
    record.state = internal_state

    detail = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    message = client.post(
      "/api/control/runs/bg_0/messages",
      headers=_headers(alice),
      json={"message": "continue", "message_id": f"msg-{internal_state}"},
    )

  assert detail.status_code == 200, detail.text
  assert detail.json()["state"] == projected_state
  assert detail.json()["messageable"] is False
  assert message.status_code == 409, message.text
  assert message.json()["detail"] == "Autonomous run is not accepting messages"


def test_autonomous_child_stream_complete_keeps_parent_messageable(
  monkeypatch,
  tmp_path,
) -> None:
  _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "summarize",
      },
    )
    assert start.status_code == 200, start.text
    record = app.state.subprocess_registry._tasks["bg_0"]
    assert record.event_lines is not None
    record.event_lines.append({
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "sub_agent_id": "sub0:spawned",
    })

    detail = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    message = client.post(
      "/api/control/runs/bg_0/messages",
      headers=_headers(alice),
      json={
        "message": "continue parent work",
        "message_id": "msg-after-child-complete",
      },
    )

  assert detail.status_code == 200, detail.text
  assert detail.json()["state"] == "running"
  assert detail.json()["messageable"] is True
  assert message.status_code == 200, message.text
  assert message.json()["delivery_status"] == "delivered"


def _exercise_rehydrated_interrupted_resume(monkeypatch, tmp_path):
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
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
    resumed = client.post(
      "/api/control/runs/bg_0/resume",
      headers=_headers(alice),
      json={"message": "resume safely", "request_id": "resume-rehydrated"},
    )

  return detail, resumed, processes, app


def test_rehydrated_interrupted_resumable_skill_uses_existing_resume_flow(monkeypatch, tmp_path) -> None:
  detail, resumed, processes, app = _exercise_rehydrated_interrupted_resume(monkeypatch, tmp_path)
  assert detail.status_code == 200, detail.text
  assert detail.json()["state"] == "interrupted"
  assert detail.json()["resumable"] is True

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
  assert original.proc is None
  assert original.resumed_as == ["bg_1"]
  resumed_record = app.state.subprocess_registry._tasks["bg_1"]
  assert resumed_record.context is not None
  assert "state=interrupted" in resumed_record.context
  assert "operator context" in resumed_record.context
  assert "rehydrated log line" in resumed_record.context


def test_rehydrated_interrupted_resume_uses_durable_event_evidence(monkeypatch, tmp_path) -> None:
  detail, _resumed, _processes, app = _exercise_rehydrated_interrupted_resume(monkeypatch, tmp_path)
  assert detail.json()["ended_at"] == "1970-01-01T00:05:21Z"
  original = app.state.subprocess_registry._tasks["bg_0"]
  assert original.completed_at == 321.0
  resumed_record = app.state.subprocess_registry._tasks["bg_1"]
  assert resumed_record.context is not None
  assert "persisted event" in resumed_record.context


def test_autonomous_resume_starts_linked_run_for_resumable_interrupted_skill(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
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
        "max_budget_usd": 5.0,
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
    assert payload["run"]["max_budget_usd"] == 5.0
    assert payload["cmd"][payload["cmd"].index("--max-budget-usd") + 1] == "5.0"

    assert original.resumed_as == ["bg_1"]
    resumed_record = app.state.subprocess_registry._tasks["bg_1"]
    assert resumed_record.max_budget_usd == 5.0
    assert resumed_record.context is not None
    assert "Original work packet" in resumed_record.context
    assert "resume from the latest safe point" in resumed_record.context
    assert "tighten scope" in resumed_record.context
    assert "prior log line" in resumed_record.context


def test_autonomous_resume_allows_only_one_active_replacement(monkeypatch, tmp_path) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "3")
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
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
      replacement = app.state.subprocess_registry._tasks["bg_1"]
      processes[1].returncode = 0
      if replacement.reaper_task is not None:
        await replacement.reaper_task
      return original, first, second

  original, first, second = asyncio.run(run_concurrent_resume())

  statuses = sorted([first.status_code, second.status_code])
  assert statuses == [200, 409]
  assert original.resumed_as == ["bg_1"]
  assert set(app.state.subprocess_registry._tasks) == {"bg_0", "bg_1"}


@pytest.mark.parametrize("replacement_state", ["queued", "waiting"])
def test_autonomous_resume_blocks_existing_queued_or_waiting_replacement(
  monkeypatch,
  tmp_path,
  replacement_state: str,
) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_MAX_RUNNING", "3")
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
  app = _make_app(monkeypatch, tmp_path, control_skills_dir=skills_dir)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    first = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "resumable-skill",
        "context": "Original work packet",
      },
    )
    second = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "resumable-skill",
        "context": "Replacement already queued",
      },
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    original = app.state.subprocess_registry._tasks["bg_0"]
    replacement = app.state.subprocess_registry._tasks["bg_1"]
    original.state = "failed"
    original.exit_code = 1
    original.completed_at = time.time()
    original.resumed_as = ["bg_1"]
    replacement.state = replacement_state
    processes[0].returncode = 1

    resumed = client.post("/api/control/runs/bg_0/resume", headers=_headers(alice), json={})

  assert resumed.status_code == 409, resumed.text
  assert resumed.json()["detail"] == "Autonomous run already has an active resume"
  assert set(app.state.subprocess_registry._tasks) == {"bg_0", "bg_1"}


def test_autonomous_resume_context_preserves_recovery_packet_when_original_context_is_large(
  monkeypatch,
  tmp_path,
) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
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
    original.event_lines = [
      {
        "type": "tool_call_start",
        "tool_call_id": "tool-1",
        "tool_name": "fmp_fetch",
        "tool_input": {
          "endpoint": "income_statement",
          "symbol": "NVDA",
          "large": "x" * 2000,
        },
      },
      {
        "type": "tool_call_complete",
        "tool_call_id": "tool-1",
        "tool_name": "fmp_fetch",
        "result": {
          "status": "success",
          "rows": [{"period": "Q1 FY2027", "revenue": 44.06}],
          "large": "y" * 5000,
        },
        "duration_ms": 123,
        "server": "fmp-mcp",
        "is_error": False,
      },
      {
        "type": "tool_call_complete",
        "tool_call_id": "tool-2",
        "tool_name": "fms_report_earnings_review",
        "error": {"message": "judgment missing dashboard_headline"},
        "duration_ms": 42,
        "is_error": True,
      },
    ]
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
  assert "Prior completed tool results for recovery" in resumed_context
  assert "fmp_fetch" in resumed_context
  assert "income_statement" in resumed_context
  assert "Q1 FY2027" in resumed_context
  assert "fms_report_earnings_review" in resumed_context
  assert "judgment missing dashboard_headline" in resumed_context
  assert "prior log line" in resumed_context
  assert "Do not repeat durable writes" in resumed_context


def test_completed_tool_result_recovery_block_keeps_newest_entries_when_saturated() -> None:
  events: list[dict[str, Any]] = []
  for index in range(20):
    tool_call_id = f"tool-{index}"
    events.extend(
      [
        {
          "type": "tool_call_start",
          "tool_call_id": tool_call_id,
          "tool_name": f"saturated_tool_{index:02d}",
          "tool_input": {
            "symbol": "NVDA",
            "query_marker": f"input_marker_{index:02d}",
            "large": "x" * 5000,
          },
        },
        {
          "type": "tool_call_complete",
          "tool_call_id": tool_call_id,
          "tool_name": f"saturated_tool_{index:02d}",
          "result": {
            "result_marker": f"result_marker_{index:02d}",
            "large": "y" * 5000,
          },
          "duration_ms": index,
          "server": "fixture",
          "is_error": False,
        },
      ]
    )

  completed_tools = _completed_tool_result_tail(events)
  assert completed_tools[0]["tool_name"] == "saturated_tool_04"
  assert completed_tools[-1]["tool_name"] == "saturated_tool_19"

  rendered = _render_completed_tool_result_tail(completed_tools, max_chars=2200)
  decoded = json.loads(rendered)

  assert len(rendered) <= 2200
  assert decoded[0]["tool_name"] == "saturated_tool_19"
  assert decoded[1]["tool_name"] == "saturated_tool_18"
  assert "result_marker_19" in rendered
  assert "input_marker_19" in rendered
  assert "saturated_tool_04" not in rendered


def test_completed_tool_result_recovery_block_preserves_wide_object_tail_fields() -> None:
  long_key_prefix = "zz_late_marker_" + ("x" * 100)
  events = [
    {
      "type": "tool_call_complete",
      "tool_call_id": "tool-wide",
      "tool_name": "wide_object_tool",
      "result": {
        **{f"early_{index:02d}": f"value_{index:02d}" for index in range(40)},
        long_key_prefix + "_a": "late-field-survived-a",
        long_key_prefix + "_b": "late-field-survived-b",
      },
      "is_error": False,
    }
  ]

  rendered = _render_completed_tool_result_tail(_completed_tool_result_tail(events))
  decoded = json.loads(rendered)

  assert decoded[0]["tool_name"] == "wide_object_tool"
  assert "late-field-survived-a" in rendered
  assert "late-field-survived-b" in rendered
  assert "...#" in rendered
  assert "_truncated_keys" in rendered


def test_completed_tool_result_recovery_block_renders_single_oversized_summary_as_valid_json() -> None:
  rendered = _render_completed_tool_result_tail(
    [
      {
        "tool_name": "huge_tool",
        "tool_call_id": "tool-huge",
        "result": "z" * (_AUTONOMOUS_RESUME_TOOL_RESULT_BLOCK_MAX_CHARS * 2),
        "is_error": False,
      }
    ],
    max_chars=300,
  )
  decoded = json.loads(rendered)

  assert len(rendered) <= 300
  assert decoded[0]["tool_name"] == "huge_tool"


def test_autonomous_resume_context_reserves_tail_sections_with_saturated_recovery_packet(
  monkeypatch,
  tmp_path,
) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  _write_resumable_skill(skills_dir)
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
        "context": "ORIGINAL-SATURATED-" + ("x" * 20000),
      },
    )
    assert start.status_code == 200, start.text

    original = app.state.subprocess_registry._tasks["bg_0"]
    original.state = "failed"
    original.exit_code = 1
    original.completed_at = time.time()
    processes[0].returncode = 1
    event_lines: list[dict[str, Any]] = []
    for index in range(20):
      tool_call_id = f"tool-{index}"
      event_lines.extend(
        [
          {
            "type": "tool_call_start",
            "tool_call_id": tool_call_id,
            "tool_name": f"saturated_tool_{index:02d}",
            "tool_input": {
              "symbol": "NVDA",
              "query_marker": f"input_marker_{index:02d}",
              "large": "x" * 5000,
            },
          },
          {
            "type": "tool_call_complete",
            "tool_call_id": tool_call_id,
            "tool_name": f"saturated_tool_{index:02d}",
            "result": {
              "result_marker": f"result_marker_{index:02d}",
              "large": "y" * 5000,
            },
            "duration_ms": index,
            "server": "fixture",
            "is_error": False,
          },
        ]
      )
    event_lines.extend(
      {"type": "text_delta", "text": f"recent-event-{index:02d}-" + ("z" * 200)}
      for index in range(60)
    )
    original.event_lines = event_lines
    original.log_path.write_text(("prior saturated log line\n" * 80), encoding="utf-8")

    resumed = client.post(
      "/api/control/runs/bg_0/resume",
      headers=_headers(alice),
      json={"message": "resume saturated recovery"},
    )
    assert resumed.status_code == 200, resumed.text

  resumed_context = app.state.subprocess_registry._tasks["bg_1"].context or ""
  assert len(resumed_context) <= _AUTONOMOUS_RESUME_CONTEXT_MAX_CHARS
  assert "Prior completed tool results for recovery" in resumed_context
  assert "saturated_tool_19" in resumed_context
  assert "Recent control events" in resumed_context
  assert "recent-event-59" in resumed_context
  assert "recent-event-20" not in resumed_context
  assert "Recent log tail" in resumed_context
  assert "prior saturated log line" in resumed_context
  assert "Original context" in resumed_context
  assert "ORIGINAL-SATURATED" in resumed_context


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


def test_autonomous_resume_rejects_model_writer_skill_even_when_metadata_resumable(
  monkeypatch,
  tmp_path,
) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "forecast-assumptions.md").write_text(
    """---
name: forecast-assumptions
description: Forecast assumptions writer
agent_callable: true
resumable: true
mutation_mode: model_writer
required_context: []
requires_portfolio_context: false
catalog: false
semantic_metadata:
  contract_name: skill-metadata
  schema_version: '2'
  catalog_version: skill-catalog/2
  skill_id: forecast-assumptions
  tool_refs: []
  allowed_effects: [state_write]
  approval_constraints: [runtime_policy]
  output_contracts:
    - owner: platform
      contract_name: skill-result-envelope
      schema_version: '1'
  credential_requirements: []
  scheduling:
    eligibility: ineligible
    opt_in: not_required
  allowed_profiles: [analyst]
---
Update forecast assumptions.
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
        "skill": "forecast-assumptions",
        "context": "Original work packet",
        "ticker": "MSFT",
      },
    )
    assert start.status_code == 200, start.text

    original = app.state.subprocess_registry._tasks["bg_0"]
    original.state = "failed"
    original.exit_code = 1
    original.completed_at = time.time()
    processes[0].returncode = 1

    detail = client.get("/api/control/runs/bg_0", headers=_headers(alice))
    assert detail.status_code == 200, detail.text
    assert detail.json()["resumable"] is False

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


def test_autonomous_dispatch_rejects_retired_dev_mode_before_spawn(monkeypatch, tmp_path) -> None:
  processes, envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    web_session = _control_session(client, "alice", channel="web")
    cli_session = _control_session(client, "bob", channel="cli")

    for session in (web_session, cli_session):
      for retired_value in (True, False):
        payload = {
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "skill",
          "skill": "earnings-review",
          "context": "Retired dev authority must not be accepted.",
          "ticker": "AAPL",
          "dev_mode": retired_value,
        }
        response = client.post(
          "/api/control/runs",
          headers=_headers(session),
          json=payload,
        )

        assert response.status_code == 422
        assert "dev_mode" in response.text

  assert app.state.subprocess_registry._tasks == {}
  assert processes == []
  assert envs == []


def test_autonomous_dispatch_ignores_retired_qa_bridge_header(monkeypatch, tmp_path) -> None:
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
        "profile": "analyst",
        "mode": "skill",
        "skill": "earnings-review",
        "context": "A retired QA header must not bypass the web guard.",
        "dev_mode": True,
      },
    )

  assert response.status_code == 422
  assert "dev_mode" in response.text
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

    invalid_budget = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "summarize",
        "max_budget_usd": 5.0,
      },
    )
    assert invalid_budget.status_code == 422
    assert invalid_budget.json()["detail"] == "max_budget_usd requires mode='skill'"

    non_positive_budget = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "earnings-review",
        "max_budget_usd": 0,
      },
    )
    assert non_positive_budget.status_code == 422

    for coerced_budget in (True, "5"):
      rejected_coercion = client.post(
        "/api/control/runs",
        headers=_headers(alice),
        json={
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "skill",
          "skill": "earnings-review",
          "max_budget_usd": coerced_budget,
        },
      )
      assert rejected_coercion.status_code == 422

    valid = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert valid.status_code == 200, valid.text
    assert valid.json()["run_id"] == "bg_2"
    record = app.state.subprocess_registry._tasks["bg_2"]
    assert record.dev_mode is False
    assert record.cmd == [
      sys.executable,
      "-m",
      "agent.autonomous",
      "--profile",
      "analyst",
      "--task",
      "summarize",
    ]
    assert "ANALYST_DEV_MODE" not in envs[-1]


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


@pytest.mark.parametrize("state", ["queued", "waiting"])
def test_autonomous_cancel_terminalizes_queued_and_waiting_runs(monkeypatch, tmp_path, state: str) -> None:
  processes, _envs = _install_fake_spawn(monkeypatch)
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
    record.state = state

    cancel = client.delete("/api/control/runs/bg_0", headers=_headers(alice))

  assert cancel.status_code == 200, cancel.text
  assert cancel.json()["state"] == "cancelled"
  assert processes[0].returncode == -15
  assert record.state == "killed"


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
        mode="skill",
        skill="summarize",
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


def test_autonomous_dispatch_once_mode_rejects_task_skill_ticker_and_context(monkeypatch, tmp_path) -> None:
  processes, envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")

    for field, value in (
      ("skill", "earnings-review"),
      ("task", "summarize"),
      ("ticker", "AAPL"),
      ("context", "extra background"),
    ):
      response = client.post(
        "/api/control/runs",
        headers=_headers(alice),
        json={
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "once",
          field: value,
        },
      )

      assert response.status_code == 422, response.text
      assert "mode='once' does not accept skill, task, ticker, or context" in response.text

  assert app.state.subprocess_registry._tasks == {}
  assert processes == []
  assert envs == []


def test_autonomous_dispatch_task_and_skill_modes_are_mutually_exclusive(monkeypatch, tmp_path) -> None:
  processes, envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  cases = (
    (
      {"mode": "task", "task": "summarize", "skill": "earnings-review"},
      "mode='task' does not accept skill",
    ),
    ({"mode": "task"}, "mode='task' requires task"),
    ({"mode": "task", "task": "   "}, "mode='task' requires task"),
    (
      {"mode": "skill", "skill": "earnings-review", "task": "summarize"},
      "mode='skill' does not accept task",
    ),
    ({"mode": "skill"}, "mode='skill' requires skill"),
  )

  with TestClient(app) as client:
    alice = _control_session(client, "alice")

    for overrides, expected_detail in cases:
      response = client.post(
        "/api/control/runs",
        headers=_headers(alice),
        json={"kind": "autonomous", "profile": "analyst", **overrides},
      )

      assert response.status_code == 422, response.text
      assert expected_detail in response.text

  assert app.state.subprocess_registry._tasks == {}
  assert processes == []
  assert envs == []


def test_autonomous_dispatch_once_mode_accepts_the_mcp_serializer_explicit_nulls(monkeypatch, tmp_path) -> None:
  _processes, _envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "once",
        "skill": None,
        "context": None,
        "ticker": None,
        "channel": None,
        "max_budget_usd": None,
      },
    )

    assert response.status_code == 200, response.text
    record = app.state.subprocess_registry._tasks["bg_0"]
    assert record.cmd == [
      sys.executable,
      "-m",
      "agent.autonomous",
      "--profile",
      "analyst",
    ]


def test_autonomous_dispatch_skill_mode_keeps_ticker_and_context(monkeypatch, tmp_path) -> None:
  _processes, _envs = _install_fake_spawn(monkeypatch)
  app = _make_app(monkeypatch, tmp_path)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "earnings-review",
        "ticker": "AAPL",
        "context": "focus on guidance",
      },
    )

    assert response.status_code == 200, response.text
    record = app.state.subprocess_registry._tasks["bg_0"]
    assert "--skill" in record.cmd
    assert "--task" not in record.cmd


def test_agent_run_schedule_dispatch_rejects_the_opposite_mode_field() -> None:
  from pydantic import ValidationError

  from agent_gateway.control_plane.schedules import AgentRunScheduleDispatch

  assert AgentRunScheduleDispatch(
    kind="autonomous",
    profile="analyst",
    mode="task",
    task="summarize",
  ).task == "summarize"
  assert AgentRunScheduleDispatch(
    kind="autonomous",
    profile="analyst",
    mode="skill",
    skill="earnings-review",
    ticker="AAPL",
  ).skill == "earnings-review"

  with pytest.raises(ValidationError, match="task-mode schedule dispatch does not accept skill"):
    AgentRunScheduleDispatch(
      kind="autonomous",
      profile="analyst",
      mode="task",
      task="summarize",
      skill="earnings-review",
    )
  with pytest.raises(ValidationError, match="skill-mode schedule dispatch does not accept task"):
    AgentRunScheduleDispatch(
      kind="autonomous",
      profile="analyst",
      mode="skill",
      skill="earnings-review",
      task="summarize",
    )
