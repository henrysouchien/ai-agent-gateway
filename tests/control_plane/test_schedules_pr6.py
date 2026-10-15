from __future__ import annotations

import asyncio
import json
import types

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_gateway.approval_route import ControlPlaneApprovalRoute
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.single_user_policy import SingleUserApprovalPolicy
from agent_gateway.autonomous_runner import AutonomousAdmissionClosed, AutonomousRegistry
from agent_gateway.autonomous_runner_state import AutonomousTask
from model_authority.bind import CapabilityBind
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from agent_gateway.autonomous_launch_envelope import AutonomousControlAuthority
from agent_gateway.control_plane import schedules as schedules_module
from agent_gateway.server import (
  ChatRequest,
  ChatRuntime,
  GatewayServerConfig,
  create_gateway_app,
)
from agent_gateway.skill_limits import SkillExecutionLimits
from agent_gateway.session import AuthManager, GatewaySession

from .manifest_helpers import write_v6_manifest
from gateway_test_support.control_plane_identity import fake_identity_resolver, fake_mcp_user_key_lookup


API_KEY = "schedules-pr6-key"
LAUNCHD_PREFIX = "com.henrychien."
_MODEL_ENTRY = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")


def _make_app(
  operator_schedule_backend: schedules_module.OperatorScheduleBackend,
  *,
  dispatch_scope_validator: Any | None = None,
):
  def _unused_runner(*_args: object):
    raise AssertionError("schedule tests never run a chat turn")

  async def _build_chat_runtime(
    session: GatewaySession,
    request: ChatRequest,
    channel: str | None,
    auth_manager: AuthManager | None,
    *,
    storage_root: Path | None = None,
  ) -> ChatRuntime:
    _ = session, channel, auth_manager, storage_root
    capability_execution = request.capability_execution
    assert capability_execution is not None
    return ChatRuntime(
      system_prompt="test",
      build_runner=_unused_runner,
      capability_execution=capability_execution,
    )

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="schedules-pr6-test-secret-0123456789",
      valid_api_keys={API_KEY},
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      build_chat_runtime=_build_chat_runtime,
      dispatch_scope_validator=dispatch_scope_validator,
      operator_schedule_backend=operator_schedule_backend,
      identity_resolver=fake_identity_resolver,
      mcp_user_key_lookup=fake_mcp_user_key_lookup,
    )
  )


def _control_session(
  client: TestClient,
  user_id: str,
  *,
  channel: str = "tui",
  role: str = "invite",
) -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  payload = response.json()
  app = client.app
  assert isinstance(app, FastAPI)
  session = app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  session.role = role
  payload["role"] = role
  payload["session_token"] = app.state.auth.issue_token(session)
  return payload


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def test_agent_run_schedule_store_path_is_scoped_to_owner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))

  assert schedules_module.schedule_store_path_for("alice") == (
    tmp_path / "users" / "alice" / "agent-run-schedules.json"
  )
  with pytest.raises(ValueError, match="user_id is required for per-user store access"):
    schedules_module.schedule_store_for(None)


@pytest.fixture
def fake_schedule_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  (tmp_path / "gateway").mkdir(mode=0o700)
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  monkeypatch.delenv("AGENT_GATEWAY_AGENT_RUN_SCHEDULES_PATH", raising=False)
  monkeypatch.setenv("AGENT_GATEWAY_AGENT_RUN_SCHEDULE_POLL_SECONDS", "0")
  launchd_store: dict[str, dict[str, Any]] = {}
  jobs_store: dict[str, dict[str, Any]] = {}
  calls: dict[str, list[Any]] = {
    "launchd_create": [],
    "launchd_enable": [],
    "launchd_disable": [],
    "launchd_delete": [],
    "launchd_logs": [],
    "schedule_show": [],
    "jobs_create": [],
    "jobs_update": [],
    "jobs_delete": [],
  }

  def _strip_launchd_name(name: str) -> str:
    return name[len(LAUNCHD_PREFIX):] if name.startswith(LAUNCHD_PREFIX) else name

  def _launchd_show(name: str) -> dict[str, Any] | None:
    clean = _strip_launchd_name(name)
    schedule = launchd_store.get(clean)
    if schedule is None:
      return None
    return {
      "status": "ok",
      "source": "launchd",
      "label": f"{LAUNCHD_PREFIX}{clean}",
      "command": list(schedule["command"]),
      "working_directory": schedule["working_directory"],
      "schedule_description": schedule["schedule_description"],
      "enabled": schedule["enabled"],
      "last_exit_status": schedule["last_exit_status"],
      "recent_log_lines": list(schedule["recent_log_lines"]),
    }

  def _launchd_list_item(name: str, schedule: dict[str, Any]) -> dict[str, Any]:
    return {
      "name": name,
      "source": "launchd",
      "label": f"{LAUNCHD_PREFIX}{name}",
      "command": list(schedule["command"]),
      "working_directory": schedule["working_directory"],
      "schedule_description": schedule["schedule_description"],
      "enabled": schedule["enabled"],
      "last_exit_status": schedule["last_exit_status"],
    }

  def _jobs_show(name: str) -> dict[str, Any] | None:
    clean = str(name)
    schedule = jobs_store.get(clean)
    if schedule is None:
      schedule = next((item for item in jobs_store.values() if item["schedule_id"] == clean), None)
    if schedule is None:
      return None
    return {"status": "ok", "source": "jobs-mcp", **schedule}

  async def schedule_list(source: str | None = None) -> dict[str, Any]:
    if source not in (None, "launchd", "jobs-mcp"):
      return {"status": "error", "message": "source must be 'launchd', 'jobs-mcp', or None"}
    items: list[dict[str, Any]] = []
    if source in (None, "launchd"):
      items.extend(_launchd_list_item(name, schedule) for name, schedule in sorted(launchd_store.items()))
    if source in (None, "jobs-mcp"):
      for schedule in sorted(jobs_store.values(), key=lambda item: item["name"]):
        items.append(
          {
            "source": "jobs-mcp",
            **schedule,
          }
        )
    return {"status": "ok", "schedules": items}

  async def schedule_show(name: str, source: str | None = None) -> dict[str, Any]:
    calls["schedule_show"].append({"name": name, "source": source})
    if source in (None, "launchd"):
      launchd = _launchd_show(name)
      if launchd is not None:
        return launchd
      if source == "launchd":
        return {"status": "error", "code": "not_found", "message": f"Launchd schedule not found: {name}"}
    if source in (None, "jobs-mcp"):
      jobs = _jobs_show(name)
      if jobs is not None:
        return jobs
      if source == "jobs-mcp":
        return {"status": "error", "code": "not_found", "message": f"jobs-mcp schedule not found: {name}"}
    return {"status": "error", "code": "not_found", "message": f"Schedule not found: {name}"}

  def schedule_create(
    *,
    name: str,
    command: list[str],
    schedule: dict[str, Any] | list[dict[str, Any]],
    working_directory: str,
    log_file: str | None = None,
    comment: str = "",
  ) -> dict[str, Any]:
    calls["launchd_create"].append(
      {
        "name": name,
        "command": command,
        "schedule": schedule,
        "working_directory": working_directory,
        "log_file": log_file,
        "comment": comment,
      }
    )
    if name in launchd_store:
      return {"status": "error", "message": f"Schedule already exists: {name}"}
    launchd_store[name] = {
      "command": list(command),
      "working_directory": working_directory,
      "schedule": schedule,
      "schedule_description": "Daily at 9:30 AM",
      "enabled": True,
      "last_exit_status": None,
      "recent_log_lines": ["launchd line 1", "launchd line 2"],
    }
    return {"status": "ok", "label": f"{LAUNCHD_PREFIX}{name}"}

  async def schedule_enable(name: str) -> dict[str, Any]:
    calls["launchd_enable"].append(name)
    clean = _strip_launchd_name(name)
    if clean not in launchd_store:
      return {"status": "error", "code": "not_found", "message": f"Schedule not found: {name}"}
    launchd_store[clean]["enabled"] = True
    return {"status": "ok"}

  async def schedule_disable(name: str) -> dict[str, Any]:
    calls["launchd_disable"].append(name)
    clean = _strip_launchd_name(name)
    if clean not in launchd_store:
      return {"status": "error", "code": "not_found", "message": f"Schedule not found: {name}"}
    launchd_store[clean]["enabled"] = False
    return {"status": "ok"}

  async def schedule_delete(name: str, confirm: bool = False) -> dict[str, Any]:
    calls["launchd_delete"].append({"name": name, "confirm": confirm})
    clean = _strip_launchd_name(name)
    if not confirm:
      return {"status": "error", "message": "Destructive operation"}
    if clean not in launchd_store:
      return {"status": "error", "code": "not_found", "message": f"Schedule not found: {name}"}
    del launchd_store[clean]
    return {"status": "ok", "message": "Schedule deleted"}

  async def schedule_logs(name: str, lines: int = 50) -> dict[str, Any]:
    calls["launchd_logs"].append({"name": name, "lines": lines})
    clean = _strip_launchd_name(name)
    if clean in launchd_store:
      return {"status": "ok", "lines": launchd_store[clean]["recent_log_lines"][-lines:]}
    if name in jobs_store:
      return {"status": "error", "message": "Log reading not supported for jobs-mcp schedules. Check jobs-mcp directly."}
    return {"status": "error", "code": "not_found", "message": f"Schedule not found: {name}"}

  async def create_job_schedule(
    name: str,
    job_type: str,
    frequency: str,
    *,
    time_of_day: str | None = None,
    day_of_week: int | None = None,
    day_of_month: int | None = None,
    params: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    calls["jobs_create"].append(
      {
        "name": name,
        "job_type": job_type,
        "frequency": frequency,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "params": dict(params or {}),
      }
    )
    if name in jobs_store:
      return {"status": "error", "error": f"Schedule name already exists: {name}"}
    schedule_id = f"sched_{name}"
    jobs_store[name] = {
      "name": name,
      "schedule_id": schedule_id,
      "job_type": job_type,
      "frequency": frequency,
      "time_of_day": time_of_day,
      "day_of_week": day_of_week,
      "day_of_month": day_of_month,
      "params": dict(params or {}),
      "enabled": True,
      "schedule_description": "Mondays at 08:15",
      "last_run_at": None,
      "next_run_at": "2026-06-01T08:15:00Z",
    }
    return {"status": "success", "schedule_id": schedule_id}

  async def update_job_schedule(schedule_id: str, *, enabled: bool | None = None, **_kwargs: Any) -> dict[str, Any]:
    calls["jobs_update"].append({"schedule_id": schedule_id, "enabled": enabled})
    schedule = next((item for item in jobs_store.values() if item["schedule_id"] == schedule_id), None)
    if schedule is None:
      return {"status": "error", "code": "not_found", "error": f"Schedule not found: {schedule_id}"}
    if enabled is not None:
      schedule["enabled"] = enabled
    return {"status": "success", "schedule": dict(schedule)}

  async def delete_job_schedule(schedule_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    calls["jobs_delete"].append({"schedule_id": schedule_id, "dry_run": dry_run})
    for name, schedule in list(jobs_store.items()):
      if schedule["schedule_id"] == schedule_id:
        if not dry_run:
          del jobs_store[name]
        return {"status": "success", "deleted": name, "schedule": dict(schedule)}
    return {"status": "error", "code": "not_found", "error": f"Schedule not found: {schedule_id}"}

  scheduler_fake = types.SimpleNamespace(
    schedule_list=schedule_list,
    schedule_show=schedule_show,
    schedule_create=schedule_create,
    schedule_enable=schedule_enable,
    schedule_disable=schedule_disable,
    schedule_delete=schedule_delete,
    schedule_logs=schedule_logs,
  )
  jobs_fake = types.SimpleNamespace(
    create_schedule=create_job_schedule,
    update_schedule=update_job_schedule,
    delete_schedule=delete_job_schedule,
  )

  return {
    "backend": schedules_module.OperatorScheduleBackend(
      scheduler=scheduler_fake,
      jobs=jobs_fake,
      label_prefix=LAUNCHD_PREFIX,
    ),
    "launchd": launchd_store,
    "jobs": jobs_store,
    "calls": calls,
    "tmp_path": tmp_path,
  }


@pytest.mark.asyncio
async def test_jobs_schedule_list_passes_backend_frequency_through() -> None:
  detail_calls: list[tuple[str, str | None]] = []
  async def list_schedules(**_kwargs):
    return {
      "status": "ok",
      "schedules": [
        {
          "source": "jobs-mcp",
          "name": "novel",
          "schedule_id": "sched-novel",
          "job_type": "test",
          "enabled": True,
          "schedule_description": "Hourly",
          "frequency": "hourly",
        },
        {
          "source": "jobs-mcp",
          "name": "incomplete",
          "schedule_id": "sched-incomplete",
          "job_type": "test",
          "enabled": True,
          "schedule_description": "Incomplete",
        },
      ],
    }

  scheduler = types.SimpleNamespace(
    schedule_list=list_schedules,
    schedule_show=lambda name, source=None: detail_calls.append((name, source)),
  )
  backend = schedules_module.OperatorScheduleBackend(
    scheduler=scheduler,
    jobs=types.SimpleNamespace(),
  )

  schedules = await schedules_module._list_schedules("jobs-mcp", backend=backend)

  assert [schedule.name for schedule in schedules] == ["novel", "incomplete"]
  first, second = schedules
  assert isinstance(first, schedules_module.JobsMcpScheduleResponse)
  assert isinstance(second, schedules_module.JobsMcpScheduleResponse)
  assert first.frequency == "hourly"
  assert second.frequency == ""
  assert detail_calls == []


def test_existing_launchd_schedule_list_show_logs_toggle_delete(fake_schedule_backends) -> None:
  fake_schedule_backends["launchd"]["daily-close"] = {
    "command": ["/usr/bin/python3", "task.py"],
    "working_directory": "/tmp",
    "schedule": {"hour": 9, "minute": 30},
    "schedule_description": "Daily at 9:30 AM",
    "enabled": True,
    "last_exit_status": None,
    "recent_log_lines": ["launchd line 1", "launchd line 2"],
  }
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    session = _control_session(client, "alice", role="owner")
    headers = _headers(session)

    listed = client.get("/api/control/schedules?source=launchd", headers=headers)
    assert listed.status_code == 200, listed.text
    assert [item["name"] for item in listed.json()["schedules"]] == ["daily-close"]
    assert listed.json()["schedules"][0]["recent_log_lines"] == []

    detail = client.get("/api/control/schedules/daily-close", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["recent_log_lines"] == ["launchd line 1", "launchd line 2"]

    logs = client.get("/api/control/schedules/daily-close/logs?lines=1", headers=headers)
    assert logs.status_code == 200, logs.text
    assert logs.json() == {"name": "daily-close", "log_lines": ["launchd line 2"]}
    assert fake_schedule_backends["calls"]["launchd_logs"] == [{"name": "daily-close", "lines": 1}]

    disabled = client.put("/api/control/schedules/daily-close/enabled", headers=headers, json={"enabled": False})
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["schedule"]["enabled"] is False
    assert fake_schedule_backends["calls"]["launchd_disable"] == ["daily-close"]

    deleted = client.delete("/api/control/schedules/daily-close", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "name": "daily-close", "source": "launchd"}
    assert fake_schedule_backends["calls"]["launchd_delete"] == [{"name": "daily-close", "confirm": True}]


def test_jobs_mcp_schedule_create_list_show_toggle_delete(fake_schedule_backends) -> None:
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    session = _control_session(client, "alice", role="owner")
    headers = _headers(session)

    create = client.post(
      "/api/control/schedules",
      headers=headers,
      json={
        "source": "jobs-mcp",
        "name": "weekly-oi",
        "job_type": "oi_analysis",
        "frequency": "weekly",
        "time_of_day": "08:15",
        "day_of_week": 0,
        "params": {"symbols": ["AAPL"]},
      },
    )
    assert create.status_code == 201, create.text
    created_schedule = create.json()["schedule"]
    assert created_schedule["source"] == "jobs-mcp"
    assert created_schedule["schedule_id"] == "sched_weekly-oi"
    assert created_schedule["frequency"] == "weekly"
    assert created_schedule["day_of_week"] == 0
    assert created_schedule["day_of_month"] is None
    assert created_schedule["params"] == {"symbols": ["AAPL"]}
    assert fake_schedule_backends["calls"]["jobs_create"] == [
      {
        "name": "weekly-oi",
        "job_type": "oi_analysis",
        "frequency": "weekly",
        "time_of_day": "08:15",
        "day_of_week": 0,
        "day_of_month": None,
        "params": {"symbols": ["AAPL"]},
      }
    ]
    assert fake_schedule_backends["calls"]["schedule_show"] == [
      {"name": "weekly-oi", "source": "jobs-mcp"}
    ]

    listed = client.get("/api/control/schedules?source=jobs-mcp", headers=headers)
    assert listed.status_code == 200, listed.text
    listed_schedule = listed.json()["schedules"][0]
    assert listed_schedule["name"] == "weekly-oi"
    assert listed_schedule["frequency"] == "weekly"
    assert listed_schedule["time_of_day"] == "08:15"
    assert listed_schedule["params"] == {"symbols": ["AAPL"]}
    assert fake_schedule_backends["calls"]["schedule_show"] == [
      {"name": "weekly-oi", "source": "jobs-mcp"}
    ]

    detail = client.get("/api/control/schedules/weekly-oi", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["schedule_id"] == "sched_weekly-oi"
    assert fake_schedule_backends["calls"]["schedule_show"] == [
      {"name": "weekly-oi", "source": "jobs-mcp"},
      {"name": "weekly-oi", "source": None},
    ]

    logs = client.get("/api/control/schedules/weekly-oi/logs", headers=headers)
    assert logs.status_code == 400
    assert "not supported" in logs.json()["detail"]

    disabled = client.put("/api/control/schedules/weekly-oi/enabled", headers=headers, json={"enabled": False})
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["schedule"]["enabled"] is False
    assert fake_schedule_backends["calls"]["jobs_update"] == [
      {"schedule_id": "sched_weekly-oi", "enabled": False}
    ]

    deleted = client.delete("/api/control/schedules/weekly-oi", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "name": "weekly-oi", "source": "jobs-mcp"}
    assert fake_schedule_backends["calls"]["jobs_delete"] == [
      {"schedule_id": "sched_weekly-oi", "dry_run": False}
    ]


def test_operator_schedule_surfaces_are_owner_only(fake_schedule_backends) -> None:
  fake_schedule_backends["launchd"]["daily-close"] = {
    "command": ["/bin/echo", "launchd"],
    "working_directory": "/tmp",
    "schedule_description": "Daily at 9:30 AM",
    "enabled": True,
    "last_exit_status": 0,
    "recent_log_lines": [],
  }
  fake_schedule_backends["jobs"]["weekly-oi"] = {
    "name": "weekly-oi",
    "schedule_id": "sched_weekly-oi",
    "job_type": "oi_analysis",
    "frequency": "weekly",
    "time_of_day": "08:15",
    "day_of_week": 0,
    "day_of_month": None,
    "params": {"symbols": ["AAPL"]},
    "enabled": True,
    "schedule_description": "Mondays at 08:15",
    "last_run_at": None,
    "next_run_at": "2026-06-01T08:15:00Z",
  }

  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    alice = _control_session(client, "alice", role="owner")
    bob = _control_session(client, "bob", role="invite")

    all_for_alice = client.get("/api/control/schedules", headers=_headers(alice))
    all_for_bob = client.get("/api/control/schedules", headers=_headers(bob))
    launchd_only = client.get("/api/control/schedules?source=launchd", headers=_headers(alice))
    jobs_only = client.get("/api/control/schedules?source=jobs-mcp", headers=_headers(alice))
    invalid_source = client.get("/api/control/schedules?source=bad", headers=_headers(alice))

    assert all_for_alice.status_code == 200, all_for_alice.text
    assert all_for_bob.status_code == 200, all_for_bob.text
    assert {item["source"] for item in all_for_alice.json()["schedules"]} == {"launchd", "jobs-mcp"}
    assert all_for_bob.json()["schedules"] == []
    assert [item["source"] for item in launchd_only.json()["schedules"]] == ["launchd"]
    assert [item["source"] for item in jobs_only.json()["schedules"]] == ["jobs-mcp"]
    assert invalid_source.status_code == 422

    invite_headers = _headers(bob)
    direct_responses = (
      client.get("/api/control/schedules/daily-close", headers=invite_headers),
      client.get("/api/control/schedules/daily-close/logs", headers=invite_headers),
      client.post(
        "/api/control/schedules",
        headers=invite_headers,
        json={
          "source": "jobs-mcp",
          "name": "invite-job",
          "job_type": "oi_analysis",
          "frequency": "daily",
          "time_of_day": "08:15",
        },
      ),
      client.put(
        "/api/control/schedules/daily-close/enabled",
        headers=invite_headers,
        json={"enabled": False},
      ),
      client.put(
        "/api/control/schedules/weekly-oi/enabled",
        headers=invite_headers,
        json={"enabled": False},
      ),
      client.delete(
        "/api/control/schedules/daily-close",
        headers=invite_headers,
      ),
      client.delete(
        "/api/control/schedules/weekly-oi",
        headers=invite_headers,
      ),
    )
    assert all(response.status_code == 404 for response in direct_responses)
    assert fake_schedule_backends["calls"]["launchd_create"] == []
    assert fake_schedule_backends["calls"]["launchd_disable"] == []
    assert fake_schedule_backends["calls"]["launchd_delete"] == []
    assert fake_schedule_backends["calls"]["jobs_create"] == []
    assert fake_schedule_backends["calls"]["jobs_update"] == []
    assert fake_schedule_backends["calls"]["jobs_delete"] == []


def test_operator_schedule_routes_require_control_session_for_owner(fake_schedule_backends) -> None:
  """U5: operator-global schedule routes accept only an owner control session, not an owner chat bearer."""
  fake_schedule_backends["launchd"]["daily-close"] = {
    "command": ["/bin/echo", "launchd"],
    "working_directory": "/tmp",
    "schedule_description": "Daily at 9:30 AM",
    "enabled": True,
    "last_exit_status": 0,
    "recent_log_lines": ["line 1"],
  }
  fake_schedule_backends["jobs"]["weekly-oi"] = {
    "name": "weekly-oi",
    "schedule_id": "sched_weekly-oi",
    "job_type": "oi_analysis",
    "frequency": "weekly",
    "time_of_day": "08:15",
    "day_of_week": 0,
    "day_of_month": None,
    "params": {"symbols": ["AAPL"]},
    "enabled": True,
    "schedule_description": "Mondays at 08:15",
    "last_run_at": None,
    "next_run_at": "2026-06-01T08:15:00Z",
  }

  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    owner_control = _control_session(client, "alice", role="owner")

    # Create an owner chat session (kind="chat", role="owner")
    response = client.post(
      "/api/chat/init",
      json={"api_key": API_KEY, "user_id": "alice", "context": {"channel": "cli"}},
    )
    assert response.status_code == 200, response.text
    chat_payload = response.json()
    assert isinstance(client.app, FastAPI)
    chat_session = client.app.state.auth.session_store.get_session(chat_payload["session_id"])
    assert chat_session is not None
    assert chat_session.kind == "chat"
    chat_session.role = "owner"
    chat_headers = {"Authorization": f"Bearer {client.app.state.auth.issue_token(chat_session)}"}

    # 1. Listing with source=launchd / jobs-mcp
    chat_list = client.get("/api/control/schedules", headers=chat_headers)
    assert chat_list.status_code == 200
    # Chat session sees only owned schedules, not operator-global schedules
    assert {item["source"] for item in chat_list.json()["schedules"]} == set()

    control_list = client.get("/api/control/schedules", headers=_headers(owner_control))
    assert control_list.status_code == 200
    assert {item["source"] for item in control_list.json()["schedules"]} == {"launchd", "jobs-mcp"}

    # 2. Direct accesses with owner chat bearer return 404
    chat_direct_responses = (
      client.get("/api/control/schedules/daily-close", headers=chat_headers),
      client.get("/api/control/schedules/daily-close/logs", headers=chat_headers),
      client.post(
        "/api/control/schedules",
        headers=chat_headers,
        json={
          "source": "jobs-mcp",
          "name": "chat-job",
          "job_type": "oi_analysis",
          "frequency": "daily",
          "time_of_day": "08:15",
        },
      ),
      client.put(
        "/api/control/schedules/daily-close/enabled",
        headers=chat_headers,
        json={"enabled": False},
      ),
      client.delete(
        "/api/control/schedules/daily-close",
        headers=chat_headers,
      ),
    )
    assert all(r.status_code == 404 for r in chat_direct_responses)

    # 3. Direct accesses with owner control session succeed (200 / 201)
    assert client.get("/api/control/schedules/daily-close", headers=_headers(owner_control)).status_code == 200
    assert client.get("/api/control/schedules/daily-close/logs", headers=_headers(owner_control)).status_code == 200
    assert client.put(
      "/api/control/schedules/daily-close/enabled",
      headers=_headers(owner_control),
      json={"enabled": False},
    ).status_code == 200
    assert client.post(
      "/api/control/schedules",
      headers=_headers(owner_control),
      json={
        "source": "jobs-mcp",
        "name": "control-job",
        "job_type": "oi_analysis",
        "frequency": "daily",
        "time_of_day": "08:15",
      },
    ).status_code == 201
    assert client.delete(
      "/api/control/schedules/daily-close",
      headers=_headers(owner_control),
    ).status_code == 200

def test_web_schedule_reads_are_projected_and_raw_fields_redacted(fake_schedule_backends) -> None:
  fake_schedule_backends["launchd"]["daily-close"] = {
    "command": ["/bin/echo", "launchd"],
    "working_directory": "/tmp",
    "schedule_description": "Daily at 9:30 AM",
    "enabled": True,
    "last_exit_status": 0,
    "recent_log_lines": ["secret log"],
  }
  fake_schedule_backends["jobs"]["weekly-oi"] = {
    "name": "weekly-oi",
    "schedule_id": "sched_weekly-oi",
    "job_type": "oi_analysis",
    "frequency": "weekly",
    "time_of_day": "08:15",
    "day_of_week": 0,
    "day_of_month": None,
    "params": {"symbols": ["AAPL"]},
    "enabled": True,
    "schedule_description": "Mondays at 08:15",
    "last_run_at": None,
    "next_run_at": "2026-06-01T08:15:00Z",
  }

  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    session = _control_session(client, "alice", channel="web", role="owner")
    headers = _headers(session)

    listed = client.get("/api/control/schedules", headers=headers)
    detail = client.get("/api/control/schedules/daily-close", headers=headers)
    logs = client.get("/api/control/schedules/daily-close/logs", headers=headers)

    assert listed.status_code == 200, listed.text
    rows = {item["name"]: item for item in listed.json()["schedules"]}
    assert {key: value for key, value in rows["daily-close"].items() if value is not None} == {
      "source": "launchd",
      "name": "daily-close",
      "enabled": True,
      "schedule_description": "Daily at 9:30 AM",
      "last_exit_status": 0,
      "kind": "operator_schedule",
      "owned_by_current_user": False,
      "editable": False,
      "can_edit": False,
      "can_delete": False,
      "can_enable": False,
      "can_disable": False,
      "can_run_now": False,
    }
    assert {key: value for key, value in rows["weekly-oi"].items() if value is not None} == {
      "source": "jobs-mcp",
      "name": "weekly-oi",
      "schedule_id": "sched_weekly-oi",
      "enabled": True,
      "schedule_description": "Mondays at 08:15",
      "next_run_at": "2026-06-01T08:15:00Z",
      "kind": "operator_schedule",
      "owned_by_current_user": False,
      "editable": False,
      "can_edit": False,
      "can_delete": False,
      "can_enable": False,
      "can_disable": False,
      "can_run_now": False,
    }
    for row in rows.values():
      assert "command" not in row
      assert "working_directory" not in row
      assert "params" not in row
      assert "recent_log_lines" not in row

    assert detail.status_code == 200, detail.text
    assert detail.json()["kind"] == "operator_schedule"
    assert "command" not in detail.json()
    assert "working_directory" not in detail.json()
    assert "recent_log_lines" not in detail.json()
    assert logs.status_code == 200, logs.text
    assert logs.json() == {"name": "daily-close", "log_lines": ["secret log"]}


def test_web_owner_operator_schedule_writes_reach_backends(fake_schedule_backends) -> None:
  fake_schedule_backends["launchd"]["daily-close"] = {
    "command": ["/bin/echo", "launchd"],
    "working_directory": "/tmp",
    "schedule_description": "Daily at 9:30 AM",
    "enabled": True,
    "last_exit_status": 0,
    "recent_log_lines": [],
  }
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    session = _control_session(client, "alice", channel="web", role="owner")
    headers = _headers(session)

    create_jobs = client.post(
      "/api/control/schedules",
      headers=headers,
      json={
        "source": "jobs-mcp",
        "name": "weekly-oi",
        "job_type": "oi_analysis",
        "frequency": "weekly",
        "time_of_day": "08:15",
        "day_of_week": 0,
        "params": {"symbols": ["AAPL"]},
      },
    )
    toggle = client.put("/api/control/schedules/daily-close/enabled", headers=headers, json={"enabled": False})
    delete = client.delete("/api/control/schedules/daily-close", headers=headers)

    assert create_jobs.status_code == 201, create_jobs.text
    assert create_jobs.json()["schedule"]["kind"] == "operator_schedule"
    assert "params" not in create_jobs.json()["schedule"]
    assert toggle.status_code == 200, toggle.text
    assert toggle.json()["schedule"]["enabled"] is False
    assert "command" not in toggle.json()["schedule"]
    assert delete.status_code == 200, delete.text
    assert delete.json() == {"deleted": True, "name": "daily-close", "source": "launchd"}
    assert [call["name"] for call in fake_schedule_backends["calls"]["jobs_create"]] == ["weekly-oi"]
    assert fake_schedule_backends["calls"]["launchd_disable"] == ["daily-close"]
    assert fake_schedule_backends["calls"]["launchd_delete"] == [{"name": "daily-close", "confirm": True}]


def _agent_run_schedule_payload(**overrides: Any) -> dict[str, Any]:
  payload: dict[str, Any] = {
    "kind": "agent_run_schedule",
    "name": "weekday-nvda-earnings-watch",
    "enabled": True,
    "timezone": "America/New_York",
    "cadence": {
      "type": "weekly",
      "days_of_week": [1, 2, 3, 4, 5],
      "time_of_day": "08:30",
    },
    "dispatch": {
      "kind": "autonomous",
      "profile": "analyst",
      "mode": "skill",
      "skill": "earnings-review",
      "ticker": "NVDA",
      "context": "Review new earnings/news and report material changes.",
    },
    "request_id": "schedule-create-1",
  }
  payload.update(overrides)
  return payload


@pytest.mark.parametrize("role", ["owner", "invite"])
def test_schedule_create_stamps_exact_dispatch_role(
  tmp_path: Path,
  role: Literal["owner", "invite"],
) -> None:
  store = schedules_module.AgentRunScheduleStore(tmp_path / "schedules.json")
  session = GatewaySession(
    session_id="schedule-role",
    api_key_hash="hash",
    created_at=1,
    expires_at=4_000_000_000,
    user_id="alice",
    owner_user_id="alice",
    raw_user_id="alice",
    user_email=None,
    user_slug="alice",
    risk_user_id=0,
    user_aliases=("alice",),
    identity_status="active",
    channel="web",
    role=role,
  )
  request = schedules_module.CreateAgentRunScheduleRequest.model_validate(
    _agent_run_schedule_payload()
  )

  record = store.create(request, session)

  assert record["dispatch_role"] == role
  assert record["dispatch_text_lengths"] == {
    "profile": len("analyst"),
    "skill": len("earnings-review"),
    "context": len("Review new earnings/news and report material changes."),
  }
  assert record["dispatch_revision"] == 1
  assert record["dispatch_authored_by"] == "alice"


def test_schedule_create_rejects_mixed_case_role(tmp_path: Path) -> None:
  store = schedules_module.AgentRunScheduleStore(tmp_path / "schedules.json")
  session = GatewaySession(
    session_id="schedule-mixed-case-role",
    api_key_hash="hash",
    created_at=1,
    expires_at=4_000_000_000,
    user_id="alice",
    owner_user_id="alice",
    raw_user_id="alice",
    user_email=None,
    user_slug="alice",
    risk_user_id=0,
    user_aliases=("alice",),
    identity_status="active",
    channel="web",
    role="Owner",  # pyright: ignore[reportArgumentType]  # negative: mixed-case role rejection
  )
  request = schedules_module.CreateAgentRunScheduleRequest.model_validate(
    _agent_run_schedule_payload()
  )

  with pytest.raises(ValueError, match="role must be exactly"):
    store.create(request, session)


def test_schedule_dispatch_authority_tracks_dispatch_revision_and_enabling_role(
  fake_schedule_backends,
) -> None:
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    session_payload = _control_session(client, "alice", channel="web")
    headers = _headers(session_payload)
    live_session = app.state.auth.session_store.get_session(
      session_payload["session_id"]
    )
    assert live_session is not None
    live_session.role = "invite"

    created_response = client.post(
      "/api/control/schedules",
      headers=headers,
      json=_agent_run_schedule_payload(),
    )
    assert created_response.status_code == 201, created_response.text
    schedule_id = created_response.json()["schedule"]["schedule_id"]
    stored = app.state.agent_run_schedule_store_for("alice").get_for_owner("alice", schedule_id)
    assert stored is not None
    assert (
      stored["dispatch_role"],
      stored["dispatch_revision"],
      stored["dispatch_authored_by"],
    ) == ("invite", 1, "alice")
    invite_disabled = client.put(
      f"/api/control/schedules/{schedule_id}/enabled",
      headers=headers,
      json={"enabled": False},
    )
    assert invite_disabled.status_code == 200, invite_disabled.text

    live_session.role = "owner"
    headers = {
      "Authorization": f"Bearer {app.state.auth.issue_token(live_session)}"
    }
    metadata_response = client.patch(
      f"/api/control/schedules/{schedule_id}",
      headers=headers,
      json={"kind": "agent_run_schedule", "timezone": "UTC"},
    )
    assert metadata_response.status_code == 200, metadata_response.text
    stored = app.state.agent_run_schedule_store_for("alice").get_for_owner("alice", schedule_id)
    assert stored is not None
    assert (
      stored["dispatch_role"],
      stored["dispatch_revision"],
      stored["dispatch_authored_by"],
    ) == ("invite", 1, "alice")
    owner_enable_invite_dispatch = client.patch(
      f"/api/control/schedules/{schedule_id}",
      headers=headers,
      json={"kind": "agent_run_schedule", "enabled": True},
    )
    assert owner_enable_invite_dispatch.status_code == 200, owner_enable_invite_dispatch.text
    stored = app.state.agent_run_schedule_store_for("alice").get_for_owner("alice", schedule_id)
    assert stored is not None
    assert (stored["enabled"], stored["dispatch_role"], stored["dispatch_revision"]) == (
      True,
      "owner",
      1,
    )

    dispatch_response = client.patch(
      f"/api/control/schedules/{schedule_id}",
      headers=headers,
      json={
        "kind": "agent_run_schedule",
        "dispatch": {
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "skill",
          "skill": "earnings-review",
          "context": "Owner-authored revision",
        },
      },
    )
    assert dispatch_response.status_code == 200, dispatch_response.text
    disabled = client.put(
      f"/api/control/schedules/{schedule_id}/enabled",
      headers=headers,
      json={"enabled": False},
    )
    assert disabled.status_code == 200, disabled.text
    stored = app.state.agent_run_schedule_store_for("alice").get_for_owner("alice", schedule_id)
    assert stored is not None
    assert (stored["dispatch_role"], stored["dispatch_revision"]) == ("owner", 2)

    live_session.role = "invite"
    headers = {
      "Authorization": f"Bearer {app.state.auth.issue_token(live_session)}"
    }
    put_enable = client.put(
      f"/api/control/schedules/{schedule_id}/enabled",
      headers=headers,
      json={"enabled": True},
    )
    assert put_enable.status_code == 200, put_enable.text
    stored = app.state.agent_run_schedule_store_for("alice").get_for_owner("alice", schedule_id)
    assert stored is not None
    assert (stored["enabled"], stored["dispatch_role"], stored["dispatch_revision"]) == (
      True,
      "invite",
      2,
    )


def test_web_agent_run_schedule_crud_is_safe_and_owner_scoped(fake_schedule_backends) -> None:
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web")
    bob = _control_session(client, "bob", channel="web")
    alice_headers = _headers(alice)
    bob_headers = _headers(bob)

    create = client.post(
      "/api/control/schedules",
      headers=alice_headers,
      json=_agent_run_schedule_payload(),
    )
    assert create.status_code == 201, create.text
    created = create.json()["schedule"]
    schedule_id = created["schedule_id"]
    assert schedule_id.startswith("sched_")
    assert created["kind"] == "agent_run_schedule"
    assert created["source"] == "agent-gateway"
    assert created["owned_by_current_user"] is True
    assert created["can_edit"] is True
    assert created["can_disable"] is True
    assert created["can_enable"] is False
    assert created["cadence"] == {
      "type": "weekly",
      "days_of_week": [1, 2, 3, 4, 5],
      "time_of_day": "08:30",
    }
    assert created["dispatch"] == {
      "kind": "autonomous",
      "profile": "analyst",
      "mode": "skill",
      "skill": "earnings-review",
      "ticker": "NVDA",
      "context": "Review new earnings/news and report material changes.",
    }
    for raw_key in ("command", "working_directory", "params", "owner_user_id", "raw_user_id"):
      assert raw_key not in created

    retry = client.post(
      "/api/control/schedules",
      headers=alice_headers,
      json=_agent_run_schedule_payload(),
    )
    assert retry.status_code == 201, retry.text
    assert retry.json()["schedule"]["schedule_id"] == schedule_id

    conflict = client.post(
      "/api/control/schedules",
      headers=alice_headers,
      json=_agent_run_schedule_payload(timezone="UTC"),
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["error"] == "schedule_request_id_conflict"

    raw_field_create = client.post(
      "/api/control/schedules",
      headers=alice_headers,
      json={
        **_agent_run_schedule_payload(name="bad-raw-schedule", request_id="schedule-create-raw"),
        "command": ["python", "scripts/run_agent.py"],
        "working_directory": "/tmp",
      },
    )
    assert raw_field_create.status_code == 422

    listed = client.get("/api/control/schedules", headers=alice_headers)
    assert listed.status_code == 200, listed.text
    rows = {item["schedule_id"]: item for item in listed.json()["schedules"] if item.get("kind") == "agent_run_schedule"}
    assert set(rows) == {schedule_id}
    assert rows[schedule_id]["name"] == "weekday-nvda-earnings-watch"

    hidden_from_bob = client.get(f"/api/control/schedules/{schedule_id}", headers=bob_headers)
    assert hidden_from_bob.status_code == 404

    update = client.patch(
      f"/api/control/schedules/{schedule_id}",
      headers=alice_headers,
      json={
        "kind": "agent_run_schedule",
        "timezone": "UTC",
        "cadence": {"type": "daily", "time_of_day": "16:00"},
      },
    )
    assert update.status_code == 200, update.text
    updated = update.json()["schedule"]
    assert updated["timezone"] == "UTC"
    assert updated["cadence"] == {"type": "daily", "time_of_day": "16:00"}

    disabled = client.put(f"/api/control/schedules/{schedule_id}/enabled", headers=alice_headers, json={"enabled": False})
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["schedule"]["enabled"] is False
    assert disabled.json()["schedule"]["can_enable"] is True
    assert disabled.json()["schedule"]["can_disable"] is False
    next_run_before_run_now = disabled.json()["schedule"]["next_run_at"]

    class FakeRegistry(AutonomousRegistry):
      def __init__(self) -> None:
        super().__init__(
          api_dir=fake_schedule_backends["tmp_path"] / "run-now-registry-api",
          log_dir=fake_schedule_backends["tmp_path"] / "run-now-registry-logs",
        )
        self._tasks: dict[str, AutonomousTask] = {}
        self.starts: list[dict[str, Any]] = []
        self.user_event_bus = None

      def set_user_event_bus(self, user_event_bus: Any | None) -> None:
        self.user_event_bus = user_event_bus

      async def start(self, **kwargs: Any) -> dict[str, Any]:
        self.starts.append(kwargs)
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
          credential_ref="service:test-product:anthropic",
          run_mode="cron",
          registry_revision=INITIAL_MODEL_REGISTRY.revision,
          policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
          selection_source="capability_default",
        )
        manifest = write_v6_manifest(
          fake_schedule_backends["tmp_path"] / "fake-registry",
          "bg_run_now",
          mode=str(kwargs["mode"]),
          task=kwargs.get("task"),
          skill=kwargs.get("skill"),
          context=kwargs.get("context"),
          ticker=kwargs.get("ticker"),
          channel=kwargs.get("channel"),
          capability_bind=capability_bind.to_json(),
        )
        task = AutonomousTask(
          task_id="bg_run_now",
          control_run_id="bg_run_now",
          session_id="bg_run_now",
          channel_id=manifest["channel_id"],
          user_id=str(kwargs["user_id"]),
          user_email=kwargs.get("user_email"),
          role=str(kwargs["role"]),
          profile=str(kwargs["profile"]),
          mode=manifest["mode"],
          task=kwargs.get("task"),
          skill=kwargs.get("skill"),
          pack=None,
          deliver=True,
          context=kwargs.get("context"),
          ticker=kwargs.get("ticker"),
          channel=kwargs.get("channel"),
          dev_mode=bool(kwargs.get("dev_mode")),
          dispatch_scope=kwargs.get("dispatch_scope"),
          cmd=["python", "-m", "agent.autonomous"],
          log_path=Path(manifest["log_path"]),
          operator_inbox_path=Path(manifest["operator_inbox_path"]),
          approval_decisions_path=None,
          control_authority=AutonomousControlAuthority.from_receipt(
            manifest["control_authority"]
          ),
          owner_lease_path=Path(manifest["owner_lease_path"]),
          owner_lease_device=manifest["owner_lease_device"],
          owner_lease_inode=manifest["owner_lease_inode"],
          started_at=1_700_000_000.0,
          skill_resume_allowed=False,
          admitted_skill_execution_limits=(
            SkillExecutionLimits(None)
            if kwargs["mode"] == "skill"
            else None
          ),
          owner_user_id=kwargs.get("owner_user_id"),
          raw_user_id=str(kwargs["user_id"]),
          user_slug=kwargs.get("user_slug"),
          risk_user_id=int(kwargs.get("risk_user_id") or 0),
          user_aliases=list(kwargs.get("user_aliases") or []),
          identity_status=str(kwargs.get("identity_status") or "legacy_user_id_fallback"),
          schedule_id=kwargs.get("schedule_id"),
          schedule_name=kwargs.get("schedule_name"),
          capability_bind=capability_bind,
        )
        self._tasks[task.control_run_id] = task
        return {"run_id": task.control_run_id, "task_id": task.task_id}

    fake_registry = FakeRegistry()
    app.state.agent_run_schedule_runner.autonomous_registry = fake_registry
    run_now = client.post(f"/api/control/schedules/{schedule_id}/run-now", headers=alice_headers, json={})
    assert run_now.status_code == 200, run_now.text
    run_now_payload = run_now.json()
    assert run_now_payload["run_id"] == "bg_run_now"
    assert run_now_payload["run"]["run_id"] == "bg_run_now"
    assert run_now_payload["run"]["schedule_id"] == schedule_id
    assert run_now_payload["run"]["schedule_name"] == "weekday-nvda-earnings-watch"
    assert fake_registry.user_event_bus is not None
    assert fake_registry.starts[0] == {
      "parent": app.state.auth.session_store.get_session(alice["session_id"]).admitted_job_context,
      "role": "invite",
      "profile": "analyst",
      "mode": "skill",
      "task": None,
      "skill": "earnings-review",
      "context": "Review new earnings/news and report material changes.",
      "ticker": "NVDA",
      "channel": "web",
      "user_id": "alice",
      "user_email": None,
      "owner_user_id": "alice",
      "user_slug": "alice",
      "risk_user_id": 0,
      "user_aliases": ["alice"],
      "identity_status": "legacy_user_id_fallback",
      "dispatch_scope": None,
      "schedule_id": schedule_id,
      "schedule_name": "weekday-nvda-earnings-watch",
    }
    after_run_now = client.get(f"/api/control/schedules/{schedule_id}", headers=alice_headers)
    assert after_run_now.status_code == 200, after_run_now.text
    assert after_run_now.json()["last_run_id"] == "bg_run_now"
    assert after_run_now.json()["last_status"] == "started"
    assert after_run_now.json()["next_run_at"] == next_run_before_run_now

    deleted = client.delete(f"/api/control/schedules/{schedule_id}", headers=alice_headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {
      "deleted": True,
      "name": "weekday-nvda-earnings-watch",
      "source": "agent-gateway",
    }
    assert fake_schedule_backends["calls"]["launchd_create"] == []
    assert fake_schedule_backends["calls"]["jobs_create"] == []


def test_schedule_run_now_forwards_authenticated_parent(
  fake_schedule_backends,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web")
    headers = _headers(alice)
    session = app.state.auth.session_store.get_session(alice["session_id"])
    assert session is not None
    created = client.post(
      "/api/control/schedules",
      headers=headers,
      json=_agent_run_schedule_payload(),
    )
    assert created.status_code == 201, created.text
    schedule_id = created.json()["schedule"]["schedule_id"]
    starts: list[dict[str, Any]] = []

    async def capture_start(**kwargs: Any) -> dict[str, Any]:
      starts.append(kwargs)
      raise ValueError("launch captured before process creation")

    registry = AutonomousRegistry(
      api_dir=fake_schedule_backends["tmp_path"] / "parent-registry-api",
      log_dir=fake_schedule_backends["tmp_path"] / "parent-registry-logs",
    )
    monkeypatch.setattr(registry, "start", capture_start)
    app.state.agent_run_schedule_runner.autonomous_registry = registry
    response = client.post(
      f"/api/control/schedules/{schedule_id}/run-now",
      headers=headers,
      json={},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "launch captured before process creation"
    assert len(starts) == 1
    parent = starts[0]["parent"]
    assert parent is session.admitted_job_context
    assert parent.tenant is not None
    assert parent.tenant.tenant_id == session.tenant_id
    assert parent.tenant.storage_id == session.owner_user_id
    assert parent.tenant.role == session.role
    assert parent.job_id == session.session_id
    assert parent.research_binding.research_file_id is None
    assert parent.research_binding.ticker is None
    assert parent.withheld == frozenset()



def test_web_agent_run_schedule_create_validates_dispatch_scope(fake_schedule_backends) -> None:
  seen_scopes: list[dict[str, Any]] = []
  canonical_scope = {
    "kind": "portfolio",
    "source": "user_selected",
    "portfolio_name": "taxable_combined",
    "portfolio_id": "portfolio-taxable",
    "display_name": "Taxable Combined",
  }

  def validator(session: Any, scope: dict[str, Any]) -> dict[str, Any]:
    assert session.user_id == "alice"
    seen_scopes.append(dict(scope))
    return canonical_scope

  app = _make_app(fake_schedule_backends["backend"], dispatch_scope_validator=validator)
  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web")
    create = client.post(
      "/api/control/schedules",
      headers=_headers(alice),
      json=_agent_run_schedule_payload(
        name="taxable-risk-daily",
        request_id="taxable-risk-daily-1",
        dispatch={
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "skill",
          "skill": "portfolio_risk",
          "context": "Review daily risk.",
          "dispatch_scope": {
            "kind": "portfolio",
            "source": "user_selected",
            "portfolio_name": "taxable_combined",
            "portfolio_id": None,
            "display_name": "Taxable Combined",
          },
        },
      ),
    )

    assert create.status_code == 201, create.text
    assert seen_scopes == [
      {
        "kind": "portfolio",
        "source": "user_selected",
        "portfolio_name": "taxable_combined",
        "portfolio_id": None,
        "display_name": "Taxable Combined",
      }
    ]
    assert create.json()["schedule"]["dispatch"]["dispatch_scope"] == canonical_scope


def test_web_agent_run_schedule_accepts_task_mode_and_records_text_lengths(fake_schedule_backends) -> None:
  task = "Review the portfolio. " * 200
  context = "Prior notes. " * 1000
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    session = _control_session(client, "alice", channel="web")
    response = client.post(
      "/api/control/schedules",
      headers=_headers(session),
      json=_agent_run_schedule_payload(
        name="web-task-mode",
        request_id="web-task-mode-1",
        dispatch={
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "task",
          "task": task,
          "context": context,
        },
      ),
    )

    assert response.status_code == 201, response.text
    schedule = response.json()["schedule"]
    assert schedule["dispatch"]["mode"] == "task"
    assert schedule["dispatch_text_lengths"] == {
      "profile": len("analyst"),
      "task": len(task),
      "context": len(context),
    }


def test_owned_schedule_takes_precedence_over_operator_name_collision(fake_schedule_backends) -> None:
  fake_schedule_backends["launchd"]["daily-close"] = {
    "command": ["/usr/bin/python3", "task.py"],
    "working_directory": "/tmp",
    "schedule": {"hour": 9, "minute": 30},
    "schedule_description": "Daily at 9:30 AM",
    "enabled": True,
    "last_exit_status": None,
    "recent_log_lines": [],
  }
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    session = _control_session(client, "alice", role="owner")
    headers = _headers(session)

    safe_create = client.post(
      "/api/control/schedules",
      headers=headers,
      json=_agent_run_schedule_payload(name="daily-close", request_id="shadow-safe-1"),
    )
    assert safe_create.status_code == 201, safe_create.text
    safe_schedule_id = safe_create.json()["schedule"]["schedule_id"]

    detail = client.get("/api/control/schedules/daily-close", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["schedule_id"] == safe_schedule_id

    disabled = client.put("/api/control/schedules/daily-close/enabled", headers=headers, json={"enabled": False})
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["schedule"]["source"] == "agent-gateway"

    safe_detail = client.get(f"/api/control/schedules/{safe_schedule_id}", headers=headers)
    assert safe_detail.status_code == 200, safe_detail.text
    assert safe_detail.json()["source"] == "agent-gateway"
    assert safe_detail.json()["enabled"] is False

    deleted = client.delete("/api/control/schedules/daily-close", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "name": "daily-close", "source": "agent-gateway"}

    fallback_detail = client.get("/api/control/schedules/daily-close", headers=headers)
    assert fallback_detail.status_code == 200, fallback_detail.text
    assert fallback_detail.json()["source"] == "launchd"
    assert fallback_detail.json()["enabled"] is True


def test_agent_run_schedule_read_rows_project_without_raw_backend_fields() -> None:
  row = schedules_module._normalize_schedule(
    {
      "schedule_id": "schedule-1",
      "name": "weekday-nvda-earnings-watch",
      "kind": "agent_run_schedule",
      "enabled": True,
      "timezone": "UTC",
      "cadence": {"type": "daily", "time_of_day": "16:00"},
      "dispatch": {
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "Review the portfolio.",
      },
      "command": ["python", "scripts/run_agent.py"],
      "working_directory": "/tmp",
      "environment": {"SECRET_TOKEN": "do-not-forward"},
    }
  )

  payload = row.model_dump(mode="json", exclude_none=True)
  assert payload == {
    "schedule_id": "schedule-1",
    "name": "weekday-nvda-earnings-watch",
    "kind": "agent_run_schedule",
    "enabled": True,
    "timezone": "UTC",
    "cadence": {"type": "daily", "time_of_day": "16:00"},
    "dispatch": {
      "kind": "autonomous",
      "profile": "analyst",
      "mode": "task",
      "task": "Review the portfolio.",
    },
  }


def test_agent_run_schedule_store_claims_due_records_until_stale(tmp_path: Path) -> None:
  store_path = tmp_path / "agent-run-schedules.json"
  record = {
    "schedule_id": "sched_due",
    "name": "daily-review",
    "kind": "agent_run_schedule",
    "source": "agent-gateway",
    "enabled": True,
    "timezone": "UTC",
    "cadence": {"type": "daily", "time_of_day": "16:00"},
    "dispatch": {
      "kind": "autonomous",
      "profile": "analyst",
      "mode": "task",
      "task": "Review the portfolio.",
    },
    "owner_user_id": "alice-owner",
    "dispatch_role": "owner",
    "dispatch_revision": 1,
    "dispatch_authored_by": "alice-owner",
    "raw_user_id": "alice",
    "channel": "web",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "next_run_at": "2026-01-01T00:00:00Z",
  }
  store_path.write_text(json.dumps({"version": 1, "schedules": [record], "idempotency": {}}), encoding="utf-8")
  store = schedules_module.AgentRunScheduleStore(store_path)
  now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

  assert [item["schedule_id"] for item in store.due_records(now=now)] == ["sched_due"]
  claimed = store.claim_due_record(
    "sched_due", claim_id="claim_1", now=now
  )
  assert claimed is not None
  assert claimed["running_claim_id"] == "claim_1"
  not_claimed = store.claim_due_record(
    "sched_due", claim_id="claim_2", now=now
  )
  assert not_claimed is None
  assert store.due_records(now=now) == []

  stale_now = now + timedelta(minutes=16)
  reclaimed = store.claim_due_record(
    "sched_due", claim_id="claim_2", now=stale_now
  )
  assert reclaimed is not None
  assert reclaimed["running_claim_id"] == "claim_2"
  assert (
    store.record_fire_result(
      "sched_due",
      run_id="bg_stale",
      status_text="started",
      fired_at=stale_now,
      claim_id="claim_1",
    )
    is None
  )
  stored = store.record_fire_result(
    "sched_due",
    run_id="bg_2",
    status_text="started",
    fired_at=stale_now,
    claim_id="claim_2",
  )
  assert stored is not None
  assert stored["last_run_id"] == "bg_2"
  assert "running_claim_id" not in stored
  assert "running_claimed_at" not in stored


def test_per_user_schedule_files_are_isolated_and_one_tick_fires_both_owners(
  fake_schedule_backends: Any,
) -> None:
  app = _make_app(fake_schedule_backends["backend"])

  class FakeRegistry(AutonomousRegistry):
    def __init__(self) -> None:
      super().__init__(
        api_dir=fake_schedule_backends["tmp_path"] / "tick-registry-api",
        log_dir=fake_schedule_backends["tmp_path"] / "tick-registry-logs",
        tenant_id="test-product",
        approval_store=SQLiteApprovalStore(fake_schedule_backends["tmp_path"] / "tick-approvals.sqlite3"),
        approval_policy=SingleUserApprovalPolicy(),
      )
      self.starts: list[dict[str, Any]] = []

    def set_user_event_bus(self, user_event_bus: Any | None) -> None:
      _ = user_event_bus

    async def start(self, **kwargs: Any) -> dict[str, Any]:
      self.starts.append(kwargs)
      run_id = f"bg_{kwargs['owner_user_id']}"
      return {"run_id": run_id, "task_id": run_id}

  registry = FakeRegistry()
  app.state.agent_run_schedule_runner.autonomous_registry = registry

  with TestClient(app) as client:
    sessions = {
      owner: _control_session(client, owner, channel="web")
      for owner in ("alice", "bob")
    }
    created: dict[str, str] = {}
    for owner in ("alice", "bob"):
      response = client.post(
        "/api/control/schedules",
        headers=_headers(sessions[owner]),
        json=_agent_run_schedule_payload(
          name=f"{owner}-daily-review",
          request_id=f"create-{owner}-schedule",
        ),
      )
      assert response.status_code == 201, response.text
      created[owner] = response.json()["schedule"]["schedule_id"]

    for owner in ("alice", "bob"):
      listed = client.get(
        "/api/control/schedules",
        headers=_headers(sessions[owner]),
      )
      assert listed.status_code == 200, listed.text
      assert [
        row["schedule_id"]
        for row in listed.json()["schedules"]
        if row.get("kind") == "agent_run_schedule"
      ] == [created[owner]]

    for owner in ("alice", "bob"):
      path = (
        fake_schedule_backends["tmp_path"]
        / "users"
        / owner
        / "agent-run-schedules.json"
      )
      assert path.is_file()
      payload = json.loads(path.read_text(encoding="utf-8"))
      assert [row["owner_user_id"] for row in payload["schedules"]] == [owner]
      payload["schedules"][0]["next_run_at"] = "2026-01-01T00:00:00Z"
      path.write_text(json.dumps(payload), encoding="utf-8")

    results = asyncio.run(
      app.state.agent_run_schedule_runner.fire_due(
        now=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
      )
    )

  assert {result["schedule_id"] for result in results} == set(created.values())
  assert {start["owner_user_id"] for start in registry.starts} == {"alice", "bob"}
  assert {start["user_id"] for start in registry.starts} == {"alice", "bob"}


@pytest.mark.parametrize("previous_run_id", [None, "bg_previous"])
def test_native_schedule_restart_refusal_stays_due_and_retries_next_tick(
  fake_schedule_backends: Any,
  monkeypatch: pytest.MonkeyPatch,
  previous_run_id: str | None,
) -> None:
  app = _make_app(fake_schedule_backends["backend"])
  root = fake_schedule_backends["tmp_path"]
  registry = AutonomousRegistry(
    api_dir=root / "restart-due-api",
    log_dir=root / "restart-due-logs",
    tenant_id="test-product",
    approval_store=SQLiteApprovalStore(root / "restart-due-approvals.sqlite3"),
    approval_policy=SingleUserApprovalPolicy(),
  )
  attempts: list[dict[str, Any]] = []
  refusal = AutonomousAdmissionClosed()

  async def dispatch(**kwargs: Any) -> dict[str, Any]:
    attempts.append(kwargs)
    if len(attempts) == 1:
      raise refusal
    return {"run_id": "bg_after_restart", "task_id": "bg_after_restart"}

  monkeypatch.setattr(registry, "start", dispatch)
  runner = app.state.agent_run_schedule_runner
  runner.autonomous_registry = registry
  now = datetime(2026, 1, 1, tzinfo=timezone.utc)
  next_tick = now + timedelta(seconds=60)

  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web")
    created = client.post(
      "/api/control/schedules",
      headers=_headers(alice),
      json=_agent_run_schedule_payload(),
    )
    assert created.status_code == 201, created.text
    schedule_id = created.json()["schedule"]["schedule_id"]
    path = root / "users" / "alice" / "agent-run-schedules.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    original = payload["schedules"][0]
    original["next_run_at"] = "2026-01-01T00:00:00Z"
    if previous_run_id is not None:
      original["last_run_id"] = previous_run_id
      original["last_run_at"] = "2025-12-31T16:00:00Z"
      original["last_status"] = "started"
    path.write_text(json.dumps(payload), encoding="utf-8")

    refused = asyncio.run(runner.fire_due(now=now))
    assert refused == [{
      "schedule_id": schedule_id,
      "status": "deferred",
      "error": str(refusal),
      "error_status": 503,
    }]
    deferred = json.loads(path.read_text(encoding="utf-8"))["schedules"][0]
    assert deferred["enabled"] is True
    assert deferred["last_status"] == "deferred"
    assert deferred["next_run_at"] == original["next_run_at"]
    for field in ("last_run_id", "last_run_at"):
      assert (field in deferred) == (field in original)
      assert deferred.get(field) == original.get(field)
    assert "running_claim_id" not in deferred
    assert "running_claimed_at" not in deferred
    assert [
      row["schedule_id"]
      for row in runner.store_for("alice").due_records(now=next_tick)
    ] == [schedule_id]

    retried = asyncio.run(runner.fire_due(now=next_tick))
    assert retried == [{
      "schedule_id": schedule_id,
      "status": "started",
      "run_id": "bg_after_restart",
    }]
    started = json.loads(path.read_text(encoding="utf-8"))["schedules"][0]
    assert started["last_status"] == "started"
    assert started["last_run_id"] == "bg_after_restart"
    assert started["last_run_at"] == "2026-01-01T00:01:00Z"
    assert started["next_run_at"] > original["next_run_at"]
    assert "last_error" not in started
    assert "running_claim_id" not in started
    assert "running_claimed_at" not in started
    assert runner.store_for("alice").due_records(now=next_tick) == []

  assert len(attempts) == 2
  assert all(attempt["schedule_id"] == schedule_id for attempt in attempts)


def test_native_schedule_due_fire_forwards_control_plane_parent(
  fake_schedule_backends,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  app = _make_app(fake_schedule_backends["backend"])
  root = fake_schedule_backends["tmp_path"]
  approval_store = SQLiteApprovalStore(root / "due-parent-approvals.sqlite3")
  approval_policy = SingleUserApprovalPolicy()
  registry = AutonomousRegistry(
    api_dir=root / "due-parent-api",
    log_dir=root / "due-parent-logs",
    tenant_id="test-product",
    approval_store=approval_store,
    approval_policy=approval_policy,
  )
  starts: list[dict[str, Any]] = []

  async def capture_start(**kwargs: Any) -> dict[str, Any]:
    starts.append(kwargs)
    return {"run_id": "bg_due_parent", "task_id": "bg_due_parent"}

  monkeypatch.setattr(registry, "start", capture_start)
  app.state.agent_run_schedule_runner.autonomous_registry = registry
  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web")
    created = client.post(
      "/api/control/schedules",
      headers=_headers(alice),
      json=_agent_run_schedule_payload(),
    )
    assert created.status_code == 201, created.text
    schedule_id = created.json()["schedule"]["schedule_id"]
    path = root / "users" / "alice" / "agent-run-schedules.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schedules"][0]["next_run_at"] = "2026-01-01T00:00:00Z"
    path.write_text(json.dumps(payload), encoding="utf-8")
    results = asyncio.run(
      app.state.agent_run_schedule_runner.fire_due(
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
      )
    )
  assert results == [{
    "schedule_id": schedule_id,
    "status": "started",
    "run_id": "bg_due_parent",
  }]
  assert len(starts) == 1
  parent = starts[0]["parent"]
  assert parent.tenant is not None
  assert parent.tenant.tenant_id == "test-product"
  assert parent.tenant.storage_id == "alice"
  assert parent.tenant.role == "invite"
  assert parent.job_id.startswith("claim_")
  assert parent.tool_ids is None
  assert parent.withheld == frozenset()
  assert parent.research_binding.research_file_id is None
  assert parent.research_binding.ticker is None
  assert isinstance(parent.approval_route, ControlPlaneApprovalRoute)
  assert parent.approval_route.store is approval_store
  assert parent.approval_route.policy is approval_policy


def test_retired_global_schedule_file_is_never_read(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  global_path = tmp_path / "agent-run-schedules.json"
  global_path.write_text(
    json.dumps({
      "version": 1,
      "schedules": [{
        "kind": "agent_run_schedule",
        "schedule_id": "legacy-global",
        "owner_user_id": "alice",
      }],
      "idempotency": {},
    }),
    encoding="utf-8",
  )

  assert schedules_module.schedule_store_for("alice").list_for_owner("alice") == []
  registry = AutonomousRegistry(
    api_dir=tmp_path / "unused-registry-api",
    log_dir=tmp_path / "unused-registry-logs",
  )
  runner = schedules_module.AgentRunScheduleRunner(
    autonomous_registry=registry,
    poll_interval_seconds=0,
  )
  assert runner._existing_stores() == []


@pytest.mark.parametrize(
  ("roster_role", "fired_role"),
  [("owner", "owner"), ("invite", "invite")],
)
def test_agent_run_schedule_runner_starts_due_autonomous_run_with_owner_identity(
  tmp_path: Path,
  roster_role: str | None,
  fired_role: str,
) -> None:
  """A stored dispatch_role fires with no more authority than the owner's live role."""
  store_path = tmp_path / "users" / "alice-owner" / "agent-run-schedules.json"
  store_path.parent.mkdir(parents=True)
  dispatch_scope = {
    "kind": "portfolio",
    "source": "user_selected",
    "portfolio_name": "taxable_combined",
    "portfolio_id": "portfolio-taxable",
    "display_name": "Taxable Combined",
  }
  record = {
    "schedule_id": "sched_due",
    "name": "daily-review",
    "kind": "agent_run_schedule",
    "source": "agent-gateway",
    "enabled": True,
    "timezone": "UTC",
    "cadence": {"type": "daily", "time_of_day": "16:00"},
    "dispatch": {
      "kind": "autonomous",
      "profile": "analyst",
      "mode": "task",
      "task": "Review the portfolio.",
      "context": "Scheduled context.",
      "dispatch_scope": dispatch_scope,
    },
    "owner_user_id": "alice-owner",
    "raw_user_id": "alice",
    "user_email": "alice@example.com",
    "dispatch_role": "owner",
    "dispatch_revision": 1,
    "dispatch_authored_by": "alice-owner",
    "user_slug": "alice",
    "risk_user_id": 42,
    "user_aliases": ["alice-owner", "alice"],
    "identity_status": "risk_user_id_authoritative",
    "channel": "web",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "next_run_at": "2026-01-01T00:00:00Z",
  }
  store_path.write_text(json.dumps({"version": 1, "schedules": [record], "idempotency": {}}), encoding="utf-8")

  class FakeRegistry(AutonomousRegistry):
    def __init__(self) -> None:
      super().__init__(
        api_dir=tmp_path / "owner-registry-api",
        log_dir=tmp_path / "owner-registry-logs",
        tenant_id="test-product",
        approval_store=SQLiteApprovalStore(tmp_path / "owner-approvals.sqlite3"),
        approval_policy=SingleUserApprovalPolicy(),
        mcp_user_key_lookup=lambda user_id, user_email: (
          {"channel": "mcp", "role": roster_role}
          if (user_id, user_email) == ("alice", "alice@example.com")
          else None
        ),
      )
      self.user_event_bus = None
      self.starts: list[dict[str, Any]] = []

    def set_user_event_bus(self, user_event_bus: Any | None) -> None:
      self.user_event_bus = user_event_bus

    async def start(self, **kwargs: Any) -> dict[str, Any]:
      self.starts.append(kwargs)
      return {"run_id": "bg_1", "task_id": "bg_1"}

  registry = FakeRegistry()
  store = schedules_module.AgentRunScheduleStore(store_path)
  runner = schedules_module.AgentRunScheduleRunner(
    store_for=lambda _owner_user_id: store,
    users_root=tmp_path / "users",
    autonomous_registry=registry,
    user_event_bus_factory=lambda: "bus",
    poll_interval_seconds=0,
  )

  results = asyncio.run(runner.fire_due(now=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)))

  assert results == [{"schedule_id": "sched_due", "status": "started", "run_id": "bg_1"}]
  assert registry.user_event_bus == "bus"
  assert registry.starts == [
    {
      "parent": registry.starts[0]["parent"],
      "role": fired_role,
      "profile": "analyst",
      "mode": "task",
      "task": "Review the portfolio.",
      "skill": None,
      "context": "Scheduled context.",
      "ticker": None,
      "channel": "web",
      "user_id": "alice",
      "user_email": "alice@example.com",
      "owner_user_id": "alice-owner",
      "user_slug": "alice",
      "risk_user_id": 42,
      "user_aliases": ["alice-owner", "alice"],
      "identity_status": "risk_user_id_authoritative",
      "dispatch_scope": dispatch_scope,
      "schedule_id": "sched_due",
      "schedule_name": "daily-review",
    }
  ]
  stored = store.get_for_owner("alice-owner", "sched_due")
  assert stored is not None
  assert stored["last_run_id"] == "bg_1"
  assert stored["last_status"] == "started"
  if fired_role == "owner":
    assert "last_fire_warning" not in stored
  else:
    assert "exceeds the owner's live role" in stored["last_fire_warning"]
  assert stored["next_run_at"] > "2026-01-01T00:00:00Z"


@pytest.mark.parametrize("entry", ["run_now", "fire_due"])
def test_agent_run_schedule_runner_leaves_the_profile_class_to_the_runner(
  tmp_path: Path,
  entry: str,
) -> None:
  now = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
  store_path = tmp_path / "users" / "alice" / "agent-run-schedules.json"
  store_path.parent.mkdir(parents=True)
  record = {
    "schedule_id": "sched_community",
    "name": "community-task",
    "kind": "agent_run_schedule",
    "source": "agent-gateway",
    "enabled": True,
    "timezone": "UTC",
    "cadence": {"type": "daily", "time_of_day": "00:00"},
    "dispatch": {
      "kind": "autonomous",
      "profile": "community",
      "mode": "task",
      "task": "Summarize the portfolio.",
    },
    "owner_user_id": "alice",
    "raw_user_id": "alice",
    "dispatch_role": "owner",
    "channel": "web",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "next_run_at": "2026-01-01T00:00:00Z",
  }
  store_path.write_text(
    json.dumps({"version": 1, "schedules": [record], "idempotency": {}}),
    encoding="utf-8",
  )

  class FakeRegistry(AutonomousRegistry):
    def __init__(self) -> None:
      super().__init__(
        api_dir=tmp_path / "interactive-registry-api",
        log_dir=tmp_path / "interactive-registry-logs",
        tenant_id="test-product",
        approval_store=SQLiteApprovalStore(tmp_path / "interactive-approvals.sqlite3"),
        approval_policy=SingleUserApprovalPolicy(),
        mcp_user_key_lookup=lambda _user_id, _email: {"role": "owner"},
      )
      self.starts: list[dict[str, Any]] = []
      self.bus_calls = 0

    def set_user_event_bus(self, user_event_bus: Any | None) -> None:
      _ = user_event_bus
      self.bus_calls += 1

    async def start(self, **kwargs: Any) -> dict[str, Any]:
      self.starts.append(kwargs)
      return {"run_id": "bg_community", "task_id": "bg_community"}

  store = schedules_module.AgentRunScheduleStore(store_path)
  registry = FakeRegistry()
  runner = schedules_module.AgentRunScheduleRunner(
    store_for=lambda _owner_user_id: store,
    users_root=tmp_path / "users",
    autonomous_registry=registry,
    poll_interval_seconds=0,
  )

  if entry == "run_now":
    result = asyncio.run(
      runner.fire_record_now(record, live_role="owner", now=now)
    )
  else:
    result = asyncio.run(runner.fire_due(now=now))[0]

  assert result is not None
  assert result["status"] == "started"
  assert result["run_id"] == "bg_community"
  assert registry.bus_calls == 1
  assert [start["profile"] for start in registry.starts] == ["community"]


def test_jobs_mcp_create_contract_rejects_wrong_day_types_and_extra_fields(fake_schedule_backends) -> None:
  app = _make_app(fake_schedule_backends["backend"])
  with TestClient(app) as client:
    session = _control_session(client, "alice", role="owner")
    headers = _headers(session)

    string_day = client.post(
      "/api/control/schedules",
      headers=headers,
      json={
        "source": "jobs-mcp",
        "name": "bad-day",
        "job_type": "oi_analysis",
        "frequency": "weekly",
        "time_of_day": "08:15",
        "day_of_week": "0",
      },
    )
    create_enabled = client.post(
      "/api/control/schedules",
      headers=headers,
      json={
        "source": "jobs-mcp",
        "name": "enabled-not-accepted",
        "job_type": "oi_analysis",
        "frequency": "daily",
        "time_of_day": "08:15",
        "enabled": False,
      },
    )

    assert string_day.status_code == 422
    assert create_enabled.status_code == 422
    assert fake_schedule_backends["calls"]["jobs_create"] == []


def test_jobs_service_unavailable_is_not_an_empty_list_or_missing_schedule(fake_schedule_backends) -> None:
  async def unavailable(*args, **kwargs):
    return {"status": "error", "code": "service_unavailable", "error": "jobs-mcp unavailable"}

  backend = fake_schedule_backends["backend"]
  backend.scheduler.schedule_list = unavailable
  backend.scheduler.schedule_show = unavailable
  backend.jobs.create_schedule = unavailable
  app = _make_app(backend)
  with TestClient(app) as client:
    headers = _headers(_control_session(client, "alice", role="owner"))
    listing = client.get("/api/control/schedules?source=jobs-mcp", headers=headers)
    detail = client.get("/api/control/schedules/absent", headers=headers)
    created = client.post("/api/control/schedules", headers=headers, json={
      "source": "jobs-mcp", "name": "probe", "job_type": "oi_analysis",
      "frequency": "daily", "time_of_day": "08:15",
    })
    assert listing.status_code == detail.status_code == created.status_code == 503
    assert listing.json()["detail"] == "jobs-mcp unavailable"


def test_owner_agent_schedule_crud_survives_jobs_outage(fake_schedule_backends) -> None:
  async def unavailable(*args, **kwargs):
    return {"status": "error", "code": "service_unavailable", "error": "jobs-mcp unavailable"}

  backend = fake_schedule_backends["backend"]
  backend.scheduler.schedule_show = unavailable
  app = _make_app(backend)
  with TestClient(app) as client:
    headers = _headers(_control_session(client, "alice", role="owner"))
    created = client.post(
      "/api/control/schedules", headers=headers, json=_agent_run_schedule_payload(),
    )
    assert created.status_code == 201, created.text
    schedule = created.json()["schedule"]
    detail = client.get(f"/api/control/schedules/{schedule['schedule_id']}", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["source"] == "agent-gateway"
    disabled = client.put(
      f"/api/control/schedules/{schedule['name']}/enabled",
      headers=headers, json={"enabled": False},
    )
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["schedule"]["enabled"] is False
    deleted = client.delete(
      f"/api/control/schedules/{schedule['schedule_id']}", headers=headers,
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] is True
    assert schedules_module.schedule_store_for("alice").get_for_owner("alice", schedule["schedule_id"]) is None
