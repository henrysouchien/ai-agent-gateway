from __future__ import annotations

import asyncio
import json
import types

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_gateway.autonomous_runner_state import AutonomousTask
from agent_gateway.control_plane import schedules as schedules_module
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "schedules-pr6-key"
LAUNCHD_PREFIX = "com.henrychien."


def _make_app(*, dispatch_scope_validator: Any | None = None):
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(system_prompt="test", build_runner=lambda *_args: None)

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="schedules-pr6-test-secret-0123456789",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
      dispatch_scope_validator=dispatch_scope_validator,
    )
  )


def _control_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def test_agent_run_schedule_store_path_prefers_gateway_state_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  explicit = tmp_path / "explicit.json"
  user_data_dir = tmp_path / "state"
  gateway_log_dir = tmp_path / "gateway" / "logs"

  monkeypatch.setenv("AGENT_GATEWAY_AGENT_RUN_SCHEDULES_PATH", str(explicit))
  monkeypatch.setenv("USER_DATA_DIR", str(user_data_dir))
  monkeypatch.setenv("GATEWAY_LOG_DIR", str(gateway_log_dir))
  assert schedules_module.default_agent_run_schedule_store_path() == explicit

  monkeypatch.delenv("AGENT_GATEWAY_AGENT_RUN_SCHEDULES_PATH")
  assert schedules_module.default_agent_run_schedule_store_path() == user_data_dir / "agent-run-schedules.json"

  monkeypatch.delenv("USER_DATA_DIR")
  assert schedules_module.default_agent_run_schedule_store_path() == tmp_path / "gateway" / "agent-run-schedules.json"


@pytest.fixture
def fake_schedule_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
  monkeypatch.setenv(
    "AGENT_GATEWAY_AGENT_RUN_SCHEDULES_PATH",
    str(tmp_path / "agent-run-schedules.json"),
  )
  monkeypatch.setenv("AGENT_GATEWAY_AGENT_RUN_SCHEDULE_POLL_SECONDS", "0")
  launchd_store: dict[str, dict[str, Any]] = {}
  jobs_store: dict[str, dict[str, Any]] = {}
  calls: dict[str, list[Any]] = {
    "launchd_create": [],
    "launchd_enable": [],
    "launchd_disable": [],
    "launchd_delete": [],
    "launchd_logs": [],
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

  def schedule_list(source: str | None = None) -> dict[str, Any]:
    if source not in (None, "launchd", "jobs-mcp"):
      return {"status": "error", "message": "source must be 'launchd', 'jobs-mcp', or None"}
    items: list[dict[str, Any]] = []
    if source in (None, "launchd"):
      items.extend(_launchd_list_item(name, schedule) for name, schedule in sorted(launchd_store.items()))
    if source in (None, "jobs-mcp"):
      for schedule in sorted(jobs_store.values(), key=lambda item: item["name"]):
        items.append(
          {
            "name": schedule["name"],
            "source": "jobs-mcp",
            "schedule_id": schedule["schedule_id"],
            "job_type": schedule["job_type"],
            "schedule_description": schedule["schedule_description"],
            "enabled": schedule["enabled"],
            "last_run_at": schedule["last_run_at"],
            "next_run_at": schedule["next_run_at"],
          }
        )
    return {"status": "ok", "schedules": items}

  def schedule_show(name: str, source: str | None = None) -> dict[str, Any]:
    if source in (None, "launchd"):
      launchd = _launchd_show(name)
      if launchd is not None:
        return launchd
      if source == "launchd":
        return {"status": "error", "message": f"Launchd schedule not found: {name}"}
    if source in (None, "jobs-mcp"):
      jobs = _jobs_show(name)
      if jobs is not None:
        return jobs
      if source == "jobs-mcp":
        return {"status": "error", "message": f"jobs-mcp schedule not found: {name}"}
    return {"status": "error", "message": f"Schedule not found: {name}"}

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

  def schedule_enable(name: str) -> dict[str, Any]:
    calls["launchd_enable"].append(name)
    clean = _strip_launchd_name(name)
    if clean not in launchd_store:
      return {"status": "error", "message": f"Schedule not found: {name}"}
    launchd_store[clean]["enabled"] = True
    return {"status": "ok"}

  def schedule_disable(name: str) -> dict[str, Any]:
    calls["launchd_disable"].append(name)
    clean = _strip_launchd_name(name)
    if clean not in launchd_store:
      return {"status": "error", "message": f"Schedule not found: {name}"}
    launchd_store[clean]["enabled"] = False
    return {"status": "ok"}

  def schedule_delete(name: str, confirm: bool = False) -> dict[str, Any]:
    calls["launchd_delete"].append({"name": name, "confirm": confirm})
    clean = _strip_launchd_name(name)
    if not confirm:
      return {"status": "error", "message": "Destructive operation"}
    if clean not in launchd_store:
      return {"status": "error", "message": f"Schedule not found: {name}"}
    del launchd_store[clean]
    return {"status": "ok", "message": "Schedule deleted"}

  def schedule_logs(name: str, lines: int = 50) -> dict[str, Any]:
    calls["launchd_logs"].append({"name": name, "lines": lines})
    clean = _strip_launchd_name(name)
    if clean in launchd_store:
      return {"status": "ok", "lines": launchd_store[clean]["recent_log_lines"][-lines:]}
    if name in jobs_store:
      return {"status": "error", "message": "Log reading not supported for jobs-mcp schedules. Check jobs-mcp directly."}
    return {"status": "error", "message": f"Schedule not found: {name}"}

  def create_job_schedule(
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

  def update_job_schedule(schedule_id: str, *, enabled: bool | None = None, **_kwargs: Any) -> dict[str, Any]:
    calls["jobs_update"].append({"schedule_id": schedule_id, "enabled": enabled})
    schedule = next((item for item in jobs_store.values() if item["schedule_id"] == schedule_id), None)
    if schedule is None:
      return {"status": "error", "error": f"Schedule not found: {schedule_id}"}
    if enabled is not None:
      schedule["enabled"] = enabled
    return {"status": "success", "schedule": dict(schedule)}

  def delete_job_schedule(schedule_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    calls["jobs_delete"].append({"schedule_id": schedule_id, "dry_run": dry_run})
    for name, schedule in list(jobs_store.items()):
      if schedule["schedule_id"] == schedule_id:
        if not dry_run:
          del jobs_store[name]
        return {"status": "success", "deleted": name, "schedule": dict(schedule)}
    return {"status": "error", "error": f"Schedule not found: {schedule_id}"}

  scheduler_fake = types.SimpleNamespace(
    _PLIST_PREFIX=LAUNCHD_PREFIX,
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
  monkeypatch.setattr(schedules_module, "_scheduler_mcp", lambda: scheduler_fake)
  monkeypatch.setattr(schedules_module, "_jobs_api", lambda: jobs_fake)

  return {"launchd": launchd_store, "jobs": jobs_store, "calls": calls}


def test_launchd_schedule_create_list_show_logs_toggle_delete(fake_schedule_backends) -> None:
  app = _make_app()
  with TestClient(app) as client:
    session = _control_session(client, "alice")
    headers = _headers(session)

    create = client.post(
      "/api/control/schedules",
      headers=headers,
      json={
        "source": "launchd",
        "name": "daily-close",
        "command": ["/usr/bin/python3", "task.py"],
        "schedule": {"hour": 9, "minute": 30},
        "working_directory": "/tmp",
        "log_file": "/tmp/daily-close.log",
        "comment": "market close prep",
      },
    )
    assert create.status_code == 201, create.text
    schedule = create.json()["schedule"]
    assert schedule["source"] == "launchd"
    assert schedule["name"] == "daily-close"
    assert schedule["label"] == f"{LAUNCHD_PREFIX}daily-close"
    assert schedule["enabled"] is True
    assert schedule["recent_log_lines"] == ["launchd line 1", "launchd line 2"]
    assert fake_schedule_backends["calls"]["launchd_create"][0]["schedule"] == {"hour": 9, "minute": 30}

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

    missing_confirm = client.delete("/api/control/schedules/daily-close", headers=headers)
    assert missing_confirm.status_code == 400
    assert "confirm=true" in missing_confirm.json()["detail"]

    deleted = client.delete("/api/control/schedules/daily-close?confirm=true", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "name": "daily-close", "source": "launchd"}
    assert fake_schedule_backends["calls"]["launchd_delete"] == [{"name": "daily-close", "confirm": True}]


def test_jobs_mcp_schedule_create_list_show_toggle_delete(fake_schedule_backends) -> None:
  app = _make_app()
  with TestClient(app) as client:
    session = _control_session(client, "alice")
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

    listed = client.get("/api/control/schedules?source=jobs-mcp", headers=headers)
    assert listed.status_code == 200, listed.text
    listed_schedule = listed.json()["schedules"][0]
    assert listed_schedule["name"] == "weekly-oi"
    assert listed_schedule["frequency"] == "weekly"
    assert listed_schedule["time_of_day"] == "08:15"

    detail = client.get("/api/control/schedules/weekly-oi", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["schedule_id"] == "sched_weekly-oi"

    logs = client.get("/api/control/schedules/weekly-oi/logs", headers=headers)
    assert logs.status_code == 400
    assert "not supported" in logs.json()["detail"]

    disabled = client.put("/api/control/schedules/weekly-oi/enabled", headers=headers, json={"enabled": False})
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["schedule"]["enabled"] is False
    assert fake_schedule_backends["calls"]["jobs_update"] == [
      {"schedule_id": "sched_weekly-oi", "enabled": False}
    ]

    missing_confirm = client.delete("/api/control/schedules/weekly-oi", headers=headers)
    assert missing_confirm.status_code == 400

    deleted = client.delete("/api/control/schedules/weekly-oi?confirm=true", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "name": "weekly-oi", "source": "jobs-mcp"}
    assert fake_schedule_backends["calls"]["jobs_delete"] == [
      {"schedule_id": "sched_weekly-oi", "dry_run": False}
    ]


def test_schedule_list_source_filter_is_operator_global(fake_schedule_backends) -> None:
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

  app = _make_app()
  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    bob = _control_session(client, "bob")

    all_for_alice = client.get("/api/control/schedules", headers=_headers(alice))
    all_for_bob = client.get("/api/control/schedules", headers=_headers(bob))
    launchd_only = client.get("/api/control/schedules?source=launchd", headers=_headers(alice))
    jobs_only = client.get("/api/control/schedules?source=jobs-mcp", headers=_headers(alice))
    invalid_source = client.get("/api/control/schedules?source=bad", headers=_headers(alice))

    assert all_for_alice.status_code == 200, all_for_alice.text
    assert all_for_bob.status_code == 200, all_for_bob.text
    assert all_for_alice.json() == all_for_bob.json()
    assert {item["source"] for item in all_for_alice.json()["schedules"]} == {"launchd", "jobs-mcp"}
    assert [item["source"] for item in launchd_only.json()["schedules"]] == ["launchd"]
    assert [item["source"] for item in jobs_only.json()["schedules"]] == ["jobs-mcp"]
    assert invalid_source.status_code == 422


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

  app = _make_app()
  with TestClient(app) as client:
    session = _control_session(client, "alice", channel="web")
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
    assert logs.status_code == 403
    assert logs.json()["detail"]["error"] == "web_control_schedule_logs_forbidden"


def test_web_raw_schedule_writes_are_rejected_before_backends(fake_schedule_backends) -> None:
  app = _make_app()
  with TestClient(app) as client:
    session = _control_session(client, "alice", channel="web")
    headers = _headers(session)

    create_launchd = client.post(
      "/api/control/schedules",
      headers=headers,
      json={
        "source": "launchd",
        "name": "daily-close",
        "command": ["/usr/bin/python3", "task.py"],
        "schedule": {"hour": 9, "minute": 30},
        "working_directory": "/tmp",
      },
    )
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
      },
    )
    toggle = client.put("/api/control/schedules/daily-close/enabled", headers=headers, json={"enabled": False})
    delete = client.delete("/api/control/schedules/daily-close?confirm=true", headers=headers)

    for response in (create_launchd, create_jobs, toggle, delete):
      assert response.status_code == 403, response.text
      assert response.json()["detail"]["error"] == "web_control_raw_schedule_forbidden"
    assert fake_schedule_backends["calls"]["launchd_create"] == []
    assert fake_schedule_backends["calls"]["launchd_disable"] == []
    assert fake_schedule_backends["calls"]["launchd_delete"] == []
    assert fake_schedule_backends["calls"]["jobs_create"] == []


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


def test_web_agent_run_schedule_crud_is_safe_and_owner_scoped(fake_schedule_backends) -> None:
  app = _make_app()
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

    class FakeRegistry:
      def __init__(self) -> None:
        self._tasks: dict[str, AutonomousTask] = {}
        self.starts: list[dict[str, Any]] = []
        self.user_event_bus = None

      def set_user_event_bus(self, user_event_bus: Any | None) -> None:
        self.user_event_bus = user_event_bus

      async def start(self, **kwargs: Any) -> dict[str, Any]:
        self.starts.append(kwargs)
        task = AutonomousTask(
          task_id="bg_run_now",
          control_run_id="bg_run_now",
          user_id=str(kwargs["user_id"]),
          user_email=kwargs.get("user_email"),
          profile=str(kwargs["profile"]),
          mode=str(kwargs["mode"]),
          task=kwargs.get("task"),
          skill=kwargs.get("skill"),
          context=kwargs.get("context"),
          ticker=kwargs.get("ticker"),
          channel=kwargs.get("channel"),
          dev_mode=bool(kwargs.get("dev_mode")),
          dispatch_scope=kwargs.get("dispatch_scope"),
          cmd=["python", "-m", "agent.autonomous"],
          log_path=Path("bg_run_now.log"),
          events_path=None,
          operator_inbox_path=Path("bg_run_now.inbox"),
          approval_decisions_path=None,
          started_at=1_700_000_000.0,
          owner_user_id=kwargs.get("owner_user_id"),
          raw_user_id=str(kwargs["user_id"]),
          user_slug=kwargs.get("user_slug"),
          risk_user_id=int(kwargs.get("risk_user_id") or 0),
          user_aliases=list(kwargs.get("user_aliases") or []),
          identity_status=str(kwargs.get("identity_status") or "legacy_user_id_fallback"),
          schedule_id=kwargs.get("schedule_id"),
          schedule_name=kwargs.get("schedule_name"),
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
      "profile": "analyst",
      "mode": "skill",
      "task": None,
      "skill": "earnings-review",
      "context": "Review new earnings/news and report material changes.",
      "ticker": "NVDA",
      "channel": "web",
      "dev_mode": False,
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

  app = _make_app(dispatch_scope_validator=validator)
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


def test_web_agent_run_schedule_create_rejects_unknown_dispatch_scope(fake_schedule_backends) -> None:
  def validator(_session: Any, _scope: dict[str, Any]) -> None:
    raise ValueError("portfolio not visible")

  app = _make_app(dispatch_scope_validator=validator)
  with TestClient(app) as client:
    alice = _control_session(client, "alice", channel="web")
    response = client.post(
      "/api/control/schedules",
      headers=_headers(alice),
      json=_agent_run_schedule_payload(
        name="unknown-portfolio-risk",
        request_id="unknown-portfolio-risk-1",
        dispatch={
          "kind": "autonomous",
          "profile": "analyst",
          "mode": "skill",
          "skill": "portfolio_risk",
          "context": "Review daily risk.",
          "dispatch_scope": {
            "kind": "portfolio",
            "source": "user_selected",
            "portfolio_name": "unknown_book",
          },
        },
      ),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
      "error": "dispatch_scope_validation_failed",
      "message": "portfolio not visible",
    }


def test_web_agent_run_schedule_rejects_task_mode_dispatch(fake_schedule_backends) -> None:
  app = _make_app()
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
          "task": "Review the portfolio.",
        },
      ),
    )

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["error"] == "web_control_dev_dispatch_forbidden"
    assert fake_schedule_backends["calls"]["launchd_create"] == []
    assert fake_schedule_backends["calls"]["jobs_create"] == []


def test_operator_raw_schedule_routes_are_not_shadowed_by_owned_agent_schedule(fake_schedule_backends) -> None:
  app = _make_app()
  with TestClient(app) as client:
    session = _control_session(client, "alice")
    headers = _headers(session)

    raw_create = client.post(
      "/api/control/schedules",
      headers=headers,
      json={
        "source": "launchd",
        "name": "daily-close",
        "command": ["/usr/bin/python3", "task.py"],
        "schedule": {"hour": 9, "minute": 30},
        "working_directory": "/tmp",
      },
    )
    assert raw_create.status_code == 201, raw_create.text
    safe_create = client.post(
      "/api/control/schedules",
      headers=headers,
      json=_agent_run_schedule_payload(name="daily-close", request_id="shadow-safe-1"),
    )
    assert safe_create.status_code == 201, safe_create.text
    safe_schedule_id = safe_create.json()["schedule"]["schedule_id"]

    detail = client.get("/api/control/schedules/daily-close", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["source"] == "launchd"

    disabled = client.put("/api/control/schedules/daily-close/enabled", headers=headers, json={"enabled": False})
    assert disabled.status_code == 200, disabled.text
    assert disabled.json()["schedule"]["source"] == "launchd"
    assert fake_schedule_backends["calls"]["launchd_disable"] == ["daily-close"]

    safe_detail = client.get(f"/api/control/schedules/{safe_schedule_id}", headers=headers)
    assert safe_detail.status_code == 200, safe_detail.text
    assert safe_detail.json()["source"] == "agent-gateway"
    assert safe_detail.json()["enabled"] is True

    deleted = client.delete("/api/control/schedules/daily-close?confirm=true", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "name": "daily-close", "source": "launchd"}
    assert fake_schedule_backends["calls"]["launchd_delete"] == [{"name": "daily-close", "confirm": True}]

    fallback_detail = client.get("/api/control/schedules/daily-close", headers=headers)
    assert fallback_detail.status_code == 200, fallback_detail.text
    assert fallback_detail.json()["source"] == "agent-gateway"
    assert fallback_detail.json()["schedule_id"] == safe_schedule_id


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
  claimed = store.claim_due_record("sched_due", claim_id="claim_1", now=now)
  assert claimed is not None
  assert claimed["running_claim_id"] == "claim_1"
  assert store.claim_due_record("sched_due", claim_id="claim_2", now=now) is None
  assert store.due_records(now=now) == []

  stale_now = now + timedelta(minutes=16)
  reclaimed = store.claim_due_record("sched_due", claim_id="claim_2", now=stale_now)
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


def test_agent_run_schedule_runner_starts_due_autonomous_run_with_owner_identity(tmp_path: Path) -> None:
  store_path = tmp_path / "agent-run-schedules.json"
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

  class FakeRegistry:
    def __init__(self) -> None:
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
    store=store,
    autonomous_registry=registry,
    user_event_bus_factory=lambda: "bus",
    poll_interval_seconds=0,
  )

  results = asyncio.run(runner.fire_due(now=datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)))

  assert results == [{"schedule_id": "sched_due", "status": "started", "run_id": "bg_1"}]
  assert registry.user_event_bus == "bus"
  assert registry.starts == [
    {
      "profile": "analyst",
      "mode": "task",
      "task": "Review the portfolio.",
      "skill": None,
      "context": "Scheduled context.",
      "ticker": None,
      "channel": "web",
      "dev_mode": False,
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
  assert stored["next_run_at"] > "2026-01-01T00:00:00Z"


def test_jobs_mcp_create_contract_rejects_wrong_day_types_and_frequency(fake_schedule_backends) -> None:
  app = _make_app()
  with TestClient(app) as client:
    session = _control_session(client, "alice")
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
    bad_frequency = client.post(
      "/api/control/schedules",
      headers=headers,
      json={
        "source": "jobs-mcp",
        "name": "bad-frequency",
        "job_type": "oi_analysis",
        "frequency": "hourly",
        "time_of_day": "08:15",
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
    assert bad_frequency.status_code == 422
    assert create_enabled.status_code == 422
    assert fake_schedule_backends["calls"]["jobs_create"] == []
