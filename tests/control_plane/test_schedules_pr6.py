from __future__ import annotations

import types

from typing import Any

import pytest
from fastapi.testclient import TestClient

from agent_gateway.control_plane import schedules as schedules_module
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "schedules-pr6-key"
LAUNCHD_PREFIX = "com.henrychien."


def _make_app():
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
    )
  )


def _control_session(client: TestClient, user_id: str) -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


@pytest.fixture
def fake_schedule_backends(monkeypatch: pytest.MonkeyPatch):
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
