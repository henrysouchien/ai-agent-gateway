from __future__ import annotations

import asyncio
import json
import logging
import types

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi.testclient import TestClient

from agent_gateway.control_plane import schedules as schedules_module

from .test_schedules_pr6 import (
  _agent_run_schedule_payload,
  _control_session,
  _headers,
  _make_app,
  fake_schedule_backends,
)


STORE_LOGGER = "agent_gateway.control_plane.schedules"
NOW = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _record(
  *,
  schedule_id: str = "sched_alice",
  owner_user_id: str = "alice",
  name: str = "daily-review",
) -> dict[str, Any]:
  return {
    "schedule_id": schedule_id,
    "name": name,
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
    "owner_user_id": owner_user_id,
    "dispatch_role": "owner",
    "dispatch_revision": 1,
    "dispatch_authored_by": owner_user_id,
    "raw_user_id": owner_user_id,
    "channel": "web",
    "created_at": "2025-12-01T00:00:00Z",
    "updated_at": "2025-12-01T00:00:00Z",
    "next_run_at": "2025-12-31T00:00:00Z",
  }


def _payload(*records: dict[str, Any], **extra: Any) -> dict[str, Any]:
  return {
    "version": 1,
    "schedules": list(records),
    "idempotency": {},
    **extra,
  }


def _write_payload(path: Path, *records: dict[str, Any], **extra: Any) -> bytes:
  raw = (json.dumps(_payload(*records, **extra), separators=(",", ":")) + "\n").encode()
  path.write_bytes(raw)
  return raw


def _session(user_id: str = "alice") -> Any:
  return types.SimpleNamespace(
    user_id=user_id,
    raw_user_id=user_id,
    user_email=None,
    user_slug=user_id,
    risk_user_id=0,
    user_aliases=(user_id,),
    identity_status="legacy_user_id_fallback",
    channel="web",
    role="owner",
  )


def _create_request(*, request_id: str = "request-1") -> Any:
  return schedules_module.CreateAgentRunScheduleRequest.model_validate(
    _agent_run_schedule_payload(request_id=request_id)
  )


class _Registry:
  def __init__(self, on_start: Callable[[], None] | None = None) -> None:
    self.on_start = on_start
    self.starts: list[dict[str, Any]] = []

  def set_user_event_bus(self, _event_bus: Any | None) -> None:
    return None

  async def start(self, **kwargs: Any) -> dict[str, Any]:
    self.starts.append(kwargs)
    if self.on_start is not None:
      self.on_start()
    return {"run_id": "bg_1", "task_id": "bg_1"}


def _per_user_store_path(tmp_path: Path, owner_user_id: str = "alice") -> Path:
  path = tmp_path / "users" / owner_user_id / "agent-run-schedules.json"
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


def _runner_for_store(
  store: schedules_module.AgentRunScheduleStore,
  registry: Any,
  *,
  poll_interval_seconds: float = 0,
) -> schedules_module.AgentRunScheduleRunner:
  return schedules_module.AgentRunScheduleRunner(
    store_for=lambda _owner_user_id: store,
    users_root=store.path.parent.parent,
    autonomous_registry=registry,
    poll_interval_seconds=poll_interval_seconds,
  )


def test_incident_regression_create_never_saves_truncated_store(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  path = tmp_path / "schedules.json"
  alice = _record()
  bob = _record(schedule_id="sched_bob", owner_user_id="bob", name="weekly-review")
  original = _write_payload(path, alice, bob)
  store = schedules_module.AgentRunScheduleStore(path)
  save_calls: list[dict[str, Any]] = []
  monkeypatch.setattr(store, "_save", lambda payload: save_calls.append(payload))
  path.write_bytes(original[: len(original) // 2])
  corrupt = path.read_bytes()

  with pytest.raises(schedules_module.ScheduleStoreUnreadableError) as exc_info:
    store.create(_create_request(), _session("carol"))

  assert exc_info.value.path == path
  assert isinstance(exc_info.value.cause, ValueError)
  assert exc_info.value.__cause__ is exc_info.value.cause
  assert path.read_bytes() == corrupt
  assert save_calls == []

  path.write_bytes(original)
  assert [row["schedule_id"] for row in store.list_for_owner("alice")] == ["sched_alice"]
  assert [row["schedule_id"] for row in store.list_for_owner("bob")] == ["sched_bob"]


@pytest.mark.parametrize(
  "read",
  [
    lambda store: store.list_for_owner("alice"),
    lambda store: store._all_records(),
    lambda store: store.due_records(now=NOW),
  ],
  ids=["list_for_owner", "all_records", "due_records"],
)
def test_corrupt_store_reads_raise_instead_of_returning_empty(
  tmp_path: Path,
  read: Callable[[Any], Any],
) -> None:
  path = tmp_path / "schedules.json"
  path.write_bytes(b"{")
  with pytest.raises(schedules_module.ScheduleStoreUnreadableError):
    read(schedules_module.AgentRunScheduleStore(path))


@pytest.mark.parametrize(
  "mutate",
  [
    lambda store: store.update(
      "alice",
      "sched_alice",
      schedules_module.UpdateAgentRunScheduleRequest(kind="agent_run_schedule", name="renamed"),
      updated_by="alice",
      live_role="owner",
    ),
    lambda store: store.set_enabled(
      "alice",
      "sched_alice",
      enabled=False,
      updated_by="alice",
      live_role="owner",
    ),
    lambda store: store.delete("alice", "sched_alice"),
    lambda store: store.claim_due_record(
      "sched_alice", claim_id="claim_1", now=NOW
    ),
    lambda store: store.record_fire_result(
      "sched_alice",
      run_id="bg_1",
      status_text="started",
      fired_at=NOW,
    ),
  ],
  ids=["update", "set_enabled", "delete", "claim_due_record", "record_fire_result"],
)
def test_corrupt_store_mutations_raise_and_leave_bytes_untouched(
  tmp_path: Path,
  mutate: Callable[[Any], Any],
) -> None:
  path = tmp_path / "schedules.json"
  path.write_bytes(b"{not-json")
  before = path.read_bytes()
  with pytest.raises(schedules_module.ScheduleStoreUnreadableError):
    mutate(schedules_module.AgentRunScheduleStore(path))
  assert path.read_bytes() == before


INVALID_DOCUMENTS = [
  ("invalid_utf8", b"\xff\xfe"),
  ("array_payload", b"[]"),
  ("missing_schedules", b'{"version":1,"idempotency":{}}'),
  ("schedules_wrong_type", b'{"version":1,"schedules":{},"idempotency":{}}'),
  ("missing_idempotency", b'{"version":1,"schedules":[]}'),
  ("idempotency_wrong_type", b'{"version":1,"schedules":[],"idempotency":[]}'),
  ("missing_version", b'{"schedules":[],"idempotency":{}}'),
  ("version_non_int", b'{"version":"1","schedules":[],"idempotency":{}}'),
  ("version_two", b'{"version":2,"schedules":[],"idempotency":{}}'),
  ("version_bool", b'{"version":true,"schedules":[],"idempotency":{}}'),
  ("non_dict_schedule", b'{"version":1,"schedules":[7],"idempotency":{}}'),
  (
    "duplicate_top_level_schedules",
    b'{"version":1,"schedules":[{"schedule_id":"hidden"}],"schedules":[],"idempotency":{}}',
  ),
  (
    "duplicate_nested_record_key",
    b'{"version":1,"schedules":[{"schedule_id":"first","schedule_id":"second"}],"idempotency":{}}',
  ),
]


@pytest.mark.parametrize("_name,raw", INVALID_DOCUMENTS, ids=[item[0] for item in INVALID_DOCUMENTS])
def test_complete_invalid_document_matrix_fails_closed(
  tmp_path: Path,
  _name: str,
  raw: bytes,
) -> None:
  path = tmp_path / "schedules.json"
  path.write_bytes(raw)
  with pytest.raises(schedules_module.ScheduleStoreUnreadableError) as exc_info:
    schedules_module.AgentRunScheduleStore(path)._load()
  assert isinstance(exc_info.value.cause, ValueError)
  assert exc_info.value.__cause__ is exc_info.value.cause
  assert path.read_bytes() == raw


@pytest.mark.parametrize("read_error", [PermissionError("denied"), OSError("I/O failed")])
def test_os_read_failures_raise_with_cause(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  read_error: OSError,
) -> None:
  path = tmp_path / "schedules.json"
  original_read_text = Path.read_text

  def _read_text(self: Path, *args: Any, **kwargs: Any) -> str:
    if self == path:
      raise read_error
    return original_read_text(self, *args, **kwargs)

  monkeypatch.setattr(Path, "read_text", _read_text)
  with pytest.raises(schedules_module.ScheduleStoreUnreadableError) as exc_info:
    schedules_module.AgentRunScheduleStore(path)._load()
  assert exc_info.value.cause is read_error
  assert exc_info.value.__cause__ is read_error


def test_idempotency_replay_second_load_fails_closed(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  path = tmp_path / "schedules.json"
  request = _create_request(request_id="replayed")
  request_hash = schedules_module._request_body_hash(
    request.model_dump(mode="json", exclude_unset=True)
  )
  valid = _payload(_record())
  valid["idempotency"] = {
    "alice:replayed": {"schedule_id": "sched_alice", "body_hash": request_hash}
  }
  path.write_bytes(b"{")
  store = schedules_module.AgentRunScheduleStore(path)
  real_load = store._load
  loads = iter([valid, None])

  def _load() -> dict[str, Any]:
    first = next(loads)
    if first is not None:
      return first
    return real_load()

  monkeypatch.setattr(store, "_load", _load)
  with pytest.raises(schedules_module.ScheduleStoreUnreadableError):
    store.create(request, _session())
  assert path.read_bytes() == b"{"


def test_missing_file_create_round_trip_and_unknown_fields_are_preserved(tmp_path: Path) -> None:
  path = tmp_path / "schedules.json"
  store = schedules_module.AgentRunScheduleStore(path)
  assert store.list_for_owner("alice") == []
  created = store.create(_create_request(), _session())
  stored = json.loads(path.read_text())
  stored["future_top_level"] = {"kept": True}
  stored["schedules"][0]["future_record_field"] = "kept"
  path.write_text(json.dumps(stored), encoding="utf-8")

  loaded = store.get_for_owner("alice", created["schedule_id"])
  assert loaded is not None
  assert loaded["future_record_field"] == "kept"
  store.set_enabled(
    "alice",
    created["schedule_id"],
    enabled=False,
    updated_by="alice",
    live_role="owner",
  )
  rewritten = json.loads(path.read_text())
  assert rewritten["future_top_level"] == {"kept": True}
  assert rewritten["schedules"][0]["future_record_field"] == "kept"


@pytest.mark.parametrize(
  "method,path_suffix,json_body",
  [
    ("get", "", None),
    ("get", "/sched_alice", None),
    ("post", "", _agent_run_schedule_payload()),
    ("patch", "/sched_alice", {"kind": "agent_run_schedule", "name": "renamed"}),
    ("put", "/sched_alice/enabled", {"enabled": False}),
    ("delete", "/sched_alice", None),
    ("post", "/sched_alice/run-now", {}),
  ],
  ids=["list", "get_one", "create", "patch", "enabled", "delete", "run_now_pre_dispatch"],
)
def test_all_agent_schedule_routes_return_exact_top_level_503(
  fake_schedule_backends: Any,
  method: str,
  path_suffix: str,
  json_body: dict[str, Any] | None,
) -> None:
  store_path = _per_user_store_path(fake_schedule_backends["tmp_path"])
  store_path.write_bytes(b"{")
  app = _make_app()
  with TestClient(app) as client:
    session = _control_session(client, "alice", channel="web")
    response = client.request(
      method,
      f"/api/control/schedules{path_suffix}",
      headers=_headers(session),
      json=json_body,
    )

  assert response.status_code == 503
  assert response.json() == {
    "error": "schedule_store_unreadable",
    "message": (
      f"Schedule store at {store_path} is unreadable and was left untouched: "
      "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    ),
  }


def test_fire_due_with_corrupt_store_returns_empty_and_preserves_bytes(tmp_path: Path) -> None:
  path = _per_user_store_path(tmp_path)
  path.write_bytes(b"{")
  runner = _runner_for_store(
    schedules_module.AgentRunScheduleStore(path),
    _Registry(),
  )
  assert asyncio.run(runner.fire_due(now=NOW)) == []
  assert path.read_bytes() == b"{"


def test_fire_due_skips_record_when_corruption_arrives_before_claim(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  path = _per_user_store_path(tmp_path)
  record = _record()
  _write_payload(path, record)
  store = schedules_module.AgentRunScheduleStore(path)

  def _due_records(*, now: datetime | None = None) -> list[dict[str, Any]]:
    _ = now
    path.write_bytes(b"{")
    return [record]

  monkeypatch.setattr(store, "due_records", _due_records)
  registry = _Registry()
  runner = _runner_for_store(store, registry)
  assert asyncio.run(runner.fire_due(now=NOW)) == []
  assert registry.starts == []
  assert path.read_bytes() == b"{"


@pytest.mark.parametrize("mode", ["run_now", "fire_due"])
def test_started_run_survives_result_persistence_failure(
  tmp_path: Path,
  mode: str,
) -> None:
  path = _per_user_store_path(tmp_path)
  record = _record()
  _write_payload(path, record)
  registry = _Registry(on_start=lambda: path.write_bytes(b"{"))
  runner = _runner_for_store(
    schedules_module.AgentRunScheduleStore(path),
    registry,
  )

  if mode == "run_now":
    result = asyncio.run(
      runner.fire_record_now(record, live_role="owner", now=NOW)
    )
  else:
    result = asyncio.run(runner.fire_due(now=NOW))[0]

  assert result == {
    "schedule_id": "sched_alice",
    "status": "started",
    "run_id": "bg_1",
    "result_persisted": False,
  }
  assert len(registry.starts) == 1
  assert path.read_bytes() == b"{"


def test_run_now_uses_live_role_instead_of_persisted_dispatch_role(
  tmp_path: Path,
) -> None:
  record = _record()
  registry = _Registry()
  runner = _runner_for_store(
    schedules_module.AgentRunScheduleStore(_per_user_store_path(tmp_path)),
    registry,
  )

  result = asyncio.run(
    runner.fire_record_now(record, live_role="invite", now=NOW)
  )

  assert result is not None and result["status"] == "started"
  assert record["dispatch_role"] == "owner"
  assert registry.starts[0]["role"] == "invite"


def test_roleless_due_record_dispatches_as_owner_instead_of_retiring_the_work(
  tmp_path: Path,
) -> None:
  """A schedule predating the role plane must still run.

  Operator ruling 2026-08-03: process attestation may not refuse product work.
  The prior behavior failed the fire AND advanced next_run, silently retiring a
  real recurring job forever. Owner is the only role that can own a schedule
  store, so a roleless record dispatches as owner with a loud warning.
  """

  path = _per_user_store_path(tmp_path)
  record = _record()
  del record["dispatch_role"]
  _write_payload(path, record)
  registry = _Registry()
  store = schedules_module.AgentRunScheduleStore(path)
  runner = _runner_for_store(store, registry)

  results = asyncio.run(runner.fire_due(now=NOW))

  assert [row["status"] for row in results] == ["started"]
  assert len(registry.starts) == 1
  stored = store.get_for_owner("alice", "sched_alice")
  assert stored is not None
  assert stored.get("last_error") != "role_unattested"


@pytest.mark.parametrize(
  "record,registry",
  [
    ({**_record(), "dispatch": None}, _Registry()),
    (_record(), None),
  ],
  ids=["missing_dispatch", "registry_start_failure"],
)
def test_failed_fire_result_reports_unpersisted_for_both_failure_sites(
  tmp_path: Path,
  record: dict[str, Any],
  registry: _Registry | None,
) -> None:
  path = tmp_path / "schedules.json"
  path.write_bytes(b"{")

  class _FailingRegistry(_Registry):
    async def start(self, **kwargs: Any) -> dict[str, Any]:
      _ = kwargs
      raise RuntimeError("start failed")

  runner = _runner_for_store(
    schedules_module.AgentRunScheduleStore(path),
    registry or _FailingRegistry(),
  )
  result = asyncio.run(
    runner.fire_record_now(record, live_role="owner", now=NOW)
  )
  assert result is not None
  assert result["status"] == "failed"
  assert result["result_persisted"] is False


def test_run_now_route_returns_success_after_started_result_persistence_failure(
  monkeypatch: pytest.MonkeyPatch,
  fake_schedule_backends: Any,
) -> None:
  path = _per_user_store_path(fake_schedule_backends["tmp_path"])
  _write_payload(path, _record())
  app = _make_app()
  registry = _Registry(on_start=lambda: path.write_bytes(b"{"))
  app.state.agent_run_schedule_runner.autonomous_registry = registry
  task = types.SimpleNamespace(
    task_id="bg_1",
    control_run_id="bg_1",
    log_path=Path("/tmp/bg_1.log"),
    started_at=1_700_000_000,
    cmd=["python", "-m", "agent.autonomous"],
  )
  monkeypatch.setattr(schedules_module, "_autonomous_task_for_user", lambda *_args: task)
  monkeypatch.setattr(
    schedules_module,
    "_autonomous_run_from_task",
    lambda _task: {
      "kind": "autonomous",
      "run_id": "bg_1",
      "task_id": "bg_1",
      "agent": "hank",
      "profile": "analyst",
      "mode": "task",
      "skill": None,
      "task": "Review the portfolio.",
      "ticker": None,
      "channel": "web",
      "user_id": "alice",
      "owner_user_id": "alice",
      "state": "running",
      "started_at": "2026-01-01T00:00:00Z",
      "ended_at": None,
      "cost_usd": None,
      "skill_run_ids": [],
      "current_verdict": None,
      "schedule_id": "sched_alice",
      "schedule_name": "daily-review",
    },
  )

  with TestClient(app) as client:
    session = _control_session(client, "alice", channel="web")
    response = client.post(
      "/api/control/schedules/sched_alice/run-now",
      headers=_headers(session),
      json={},
    )

  assert response.status_code == 200, response.text
  assert response.json()["run_id"] == "bg_1"
  assert len(registry.starts) == 1
  assert path.read_bytes() == b"{"


def test_run_loop_survives_multiple_unreadable_ticks_and_retains_interval_wait(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  path = _per_user_store_path(tmp_path)
  path.write_bytes(b"{")
  runner = _runner_for_store(
    schedules_module.AgentRunScheduleStore(path),
    _Registry(),
    poll_interval_seconds=7.5,
  )
  waits: list[float | None] = []

  async def _wait_for(awaitable: Any, timeout: float | None = None) -> Any:
    waits.append(timeout)
    awaitable.close()
    if len(waits) < 3:
      raise asyncio.TimeoutError
    runner._stopped.set()
    return None

  monkeypatch.setattr(schedules_module.asyncio, "wait_for", _wait_for)
  asyncio.run(runner._run_loop())
  assert waits == [7.5, 7.5, 7.5]


def test_corrupt_owner_store_does_not_block_other_owner_due_run(
  caplog: pytest.LogCaptureFixture,
  tmp_path: Path,
) -> None:
  alice_path = _per_user_store_path(tmp_path, "alice")
  bob_path = _per_user_store_path(tmp_path, "bob")
  alice_path.write_bytes(b"{")
  _write_payload(
    bob_path,
    _record(schedule_id="sched_bob", owner_user_id="bob", name="bob-review"),
  )
  registry = _Registry()
  runner = schedules_module.AgentRunScheduleRunner(
    store_for=lambda owner: schedules_module.AgentRunScheduleStore(
      tmp_path / "users" / str(owner) / "agent-run-schedules.json"
    ),
    users_root=tmp_path / "users",
    autonomous_registry=registry,
    poll_interval_seconds=0,
  )
  caplog.set_level(logging.ERROR, logger=STORE_LOGGER)

  results = asyncio.run(runner.fire_due(now=NOW))

  assert results == [
    {"schedule_id": "sched_bob", "status": "started", "run_id": "bg_1"}
  ]
  assert [start["owner_user_id"] for start in registry.starts] == ["bob"]
  assert any(str(alice_path) in record.getMessage() for record in caplog.records)


def test_outage_transition_summary_boundary_and_recovery_logging(
  caplog: pytest.LogCaptureFixture,
  tmp_path: Path,
) -> None:
  path = tmp_path / "schedules.json"
  path.write_bytes(b"{")
  current = [100.0]
  store = schedules_module.AgentRunScheduleStore(path, clock=lambda: current[0])
  caplog.set_level(logging.INFO, logger=STORE_LOGGER)

  for elapsed in (0.0, 59.9, 60.0):
    current[0] = 100.0 + elapsed
    with pytest.raises(schedules_module.ScheduleStoreUnreadableError):
      store._load()
  _write_payload(path, _record())
  store._load()

  records = [record for record in caplog.records if record.name == STORE_LOGGER]
  assert [record.levelno for record in records] == [logging.ERROR, logging.WARNING, logging.INFO]
  assert "became unreadable" in records[0].getMessage()
  assert "remains unreadable" in records[1].getMessage()
  assert "recovered" in records[2].getMessage()


@pytest.mark.parametrize("recovery_first", ["route", "runner"])
def test_cross_path_outage_has_one_store_owned_transition_and_recovery(
  caplog: pytest.LogCaptureFixture,
  fake_schedule_backends: Any,
  recovery_first: str,
) -> None:
  path = _per_user_store_path(fake_schedule_backends["tmp_path"])
  path.write_bytes(b"{")
  app = _make_app()
  app.state.agent_run_schedule_runner.autonomous_registry = _Registry()
  caplog.set_level(logging.INFO, logger=STORE_LOGGER)

  with TestClient(app) as client:
    session = _control_session(client, "alice", channel="web")
    assert asyncio.run(app.state.agent_run_schedule_runner.fire_due(now=NOW)) == []
    response = client.get("/api/control/schedules", headers=_headers(session))
    assert response.status_code == 503
    _write_payload(path, _record())
    if recovery_first == "route":
      assert client.get("/api/control/schedules", headers=_headers(session)).status_code == 200
      assert asyncio.run(app.state.agent_run_schedule_runner.fire_due(now=NOW))[0]["status"] == "started"
    else:
      assert asyncio.run(app.state.agent_run_schedule_runner.fire_due(now=NOW))[0]["status"] == "started"
      assert client.get("/api/control/schedules", headers=_headers(session)).status_code == 200

  records = [record for record in caplog.records if record.name == STORE_LOGGER]
  assert len([record for record in records if record.levelno == logging.ERROR]) == 1
  assert len([record for record in records if record.levelno == logging.INFO]) == 1
  assert not [record for record in records if record.levelno == logging.WARNING]
