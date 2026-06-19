from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pytest

# The agent-claim verifier (settings + utils.agent_claim) lives in the separate
# risk_module repo. Default to the local dev checkout; allow override via
# RISK_MODULE_ROOT so CI / other machines can point at their own checkout. When
# risk_module is not importable (e.g. this package's CI runs in isolation), the
# 3 cross-repo contract tests below skip; the 7 gateway-side tests still run.
RISK_MODULE_ROOT = Path(
  os.environ.get("RISK_MODULE_ROOT", "/Users/henrychien/Documents/Jupyter/risk_module")
)
if str(RISK_MODULE_ROOT) not in sys.path:
  sys.path.append(str(RISK_MODULE_ROOT))

try:
  from settings import AGENT_API_CLAIM_MAX_TTL_SECONDS
  from utils.agent_claim import AGENT_API_CLAIM_HEADERS, verify

  _RISK_MODULE_AVAILABLE = True
except ModuleNotFoundError as exc:
  # Only treat the risk_module top-level modules being absent as "unavailable".
  # A present-but-broken risk_module (one of ITS transitive deps missing) raises
  # ModuleNotFoundError with a different name and must surface as a real failure.
  if (exc.name or "").split(".")[0] not in {"settings", "utils"}:
    raise
  AGENT_API_CLAIM_MAX_TTL_SECONDS = None  # type: ignore[assignment]
  AGENT_API_CLAIM_HEADERS = None  # type: ignore[assignment]
  verify = None  # type: ignore[assignment]
  _RISK_MODULE_AVAILABLE = False

_requires_risk_module = pytest.mark.skipif(
  not _RISK_MODULE_AVAILABLE,
  reason=(
    "risk_module verifier (settings.AGENT_API_CLAIM_MAX_TTL_SECONDS + "
    "utils.agent_claim) not importable; set RISK_MODULE_ROOT to run the "
    "cross-repo agent-claim contract tests"
  ),
)


HMAC_KEY = "test-hmac-key"
USER_ID = "1"
USER_EMAIL = "hc@henrychien.com"
CLAIM_ENV_KEYS = {
  "AGENT_API_CLAIM_AUDIENCE",
  "AGENT_API_CLAIM_ISSUED_AT",
  "AGENT_API_CLAIM_EXPIRY",
  "AGENT_API_CLAIM_USER_ID",
  "AGENT_API_CLAIM_USER_EMAIL",
  "AGENT_API_CLAIM_NONCE",
  "AGENT_API_CLAIM_SIGNATURE",
}


class FakeProcess:
  returncode: int | None = None

  async def wait(self) -> int:
    self.returncode = 0
    return 0

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class SlowFakeProcess:
  returncode: int | None = None

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class FailingFakeProcess:
  returncode: int | None = None

  async def wait(self) -> int:
    self.returncode = 1
    return 1

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class RaisingFakeProcess:
  returncode: int | None = None

  async def wait(self) -> int:
    raise RuntimeError("reap boom")

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class RecordingEventBus:
  def __init__(self) -> None:
    self.calls: list[tuple[str, dict]] = []

  async def seed_replay_buffer(
    self,
    user_id: str,
    control_run_id: str,
    events: list[dict],
    *,
    terminated: bool = False,
  ) -> int:
    self.calls.append(
      (
        "seed",
        {
          "user_id": user_id,
          "control_run_id": control_run_id,
          "events": list(events),
          "terminated": terminated,
        },
      )
    )
    return len(events)

  async def publish(self, user_id: str, control_run_id: str, event: dict) -> None:
    self.calls.append(
      (
        "publish",
        {
          "user_id": user_id,
          "control_run_id": control_run_id,
          "event": dict(event),
        },
      )
    )


def _registry(tmp_path: Path):
  from agent_gateway.autonomous_runner import AutonomousRegistry

  return AutonomousRegistry(
    api_dir=tmp_path,
    python_executable="python3",
    log_dir=tmp_path,
    max_running=1,
  )


def _manifest_path(tmp_path: Path, task_id: str = "bg_0") -> Path:
  return tmp_path / f"{task_id}.task.json"


def _read_manifest(tmp_path: Path, task_id: str = "bg_0") -> dict:
  return json.loads(_manifest_path(tmp_path, task_id).read_text(encoding="utf-8"))


def _write_manifest(tmp_path: Path, task_id: str = "bg_0", **overrides) -> dict:
  manifest = {
    "manifest_version": 1,
    "task_id": task_id,
    "control_run_id": task_id,
    "user_id": USER_ID,
    "user_email": USER_EMAIL,
    "profile": "analyst",
    "mode": "skill",
    "task": None,
    "skill": "earnings-review",
    "context": "Review current packet",
    "ticker": "AAPL",
    "channel": "tui",
    "dev_mode": False,
    "cmd": ["python3", "-m", "agent.autonomous", "--profile", "analyst"],
    "log_path": str(tmp_path / f"{task_id}.log"),
    "events_path": str(tmp_path / f"{task_id}.events.jsonl"),
    "operator_inbox_path": str(tmp_path / f"{task_id}.operator-messages.jsonl"),
    "approval_decisions_path": str(tmp_path / f"{task_id}.approval-decisions.jsonl"),
    "started_at": 100.0,
    "state": "completed",
    "exit_code": 0,
    "error": None,
    "completed_at": 125.0,
    "resumed_from": None,
    "resumed_as": [],
  }
  manifest.update(overrides)
  _manifest_path(tmp_path, task_id).write_text(json.dumps(manifest) + "\n", encoding="utf-8")
  return manifest


def _write_run_files(tmp_path: Path, task_id: str) -> None:
  for suffix in (".log", ".events.jsonl", ".operator-messages.jsonl", ".approval-decisions.jsonl"):
    (tmp_path / f"{task_id}{suffix}").write_text(f"{task_id}{suffix}\n", encoding="utf-8")


def _run_files_exist(tmp_path: Path, task_id: str) -> bool:
  return any(tmp_path.glob(f"{task_id}.*"))


def _claim_headers_from_env(env: dict[str, str]) -> dict[str, str]:
  return {
    AGENT_API_CLAIM_HEADERS["audience"]: env["AGENT_API_CLAIM_AUDIENCE"],
    AGENT_API_CLAIM_HEADERS["issued_at"]: env["AGENT_API_CLAIM_ISSUED_AT"],
    AGENT_API_CLAIM_HEADERS["expiry"]: env["AGENT_API_CLAIM_EXPIRY"],
    AGENT_API_CLAIM_HEADERS["user_id"]: env["AGENT_API_CLAIM_USER_ID"],
    AGENT_API_CLAIM_HEADERS["user_email"]: env["AGENT_API_CLAIM_USER_EMAIL"],
    AGENT_API_CLAIM_HEADERS["nonce"]: env["AGENT_API_CLAIM_NONCE"],
    AGENT_API_CLAIM_HEADERS["signature"]: env["AGENT_API_CLAIM_SIGNATURE"],
  }


async def _start_and_capture_env(
  monkeypatch,
  tmp_path: Path,
  *,
  user_id: str = USER_ID,
  user_email: str = USER_EMAIL,
  ttl_seconds: int | None = None,
) -> dict[str, str]:
  from agent_gateway import autonomous_runner

  captured: dict[str, str] = {}

  async def fake_exec(*args, **kwargs):
    captured.update(dict(kwargs["env"]))
    return FakeProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  if ttl_seconds is None:
    monkeypatch.delenv("AGENT_API_CLAIM_TTL_SECONDS", raising=False)
  else:
    monkeypatch.setenv("AGENT_API_CLAIM_TTL_SECONDS", str(ttl_seconds))

  registry = _registry(tmp_path)
  payload = await registry.start(
    profile="analyst",
    mode="task",
    task="summarize",
    user_id=user_id,
    user_email=user_email,
  )
  await registry.wait(payload["task_id"], timeout_sec=1)
  return captured


@_requires_risk_module
def test_autonomous_start_signs_claim(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert CLAIM_ENV_KEYS <= set(env)
  verified = verify(
    HMAC_KEY,
    _claim_headers_from_env(env),
    ttl_ceiling=AGENT_API_CLAIM_MAX_TTL_SECONDS,
    now=int(env["AGENT_API_CLAIM_ISSUED_AT"]),
  )
  assert verified is not None
  assert verified["user_id"] == USER_ID
  assert verified["user_email"] == USER_EMAIL


def test_autonomous_start_preserves_hmac_key_for_entry_capture(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert env["AGENT_API_USER_CLAIM_HMAC_KEY"] == HMAC_KEY


def test_autonomous_start_injects_events_path(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert env["AGENT_AUTONOMOUS_EVENTS_PATH"].endswith(".events.jsonl")
  assert Path(env["AGENT_AUTONOMOUS_EVENTS_PATH"]).exists()


def test_autonomous_start_injects_operator_inbox_path(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert env["AGENT_AUTONOMOUS_OPERATOR_INBOX_PATH"].endswith(".operator-messages.jsonl")
  assert Path(env["AGENT_AUTONOMOUS_OPERATOR_INBOX_PATH"]).exists()


def test_autonomous_start_injects_approval_bridge_paths(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert env["AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH"].endswith(".approval-decisions.jsonl")
  assert Path(env["AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH"]).exists()
  assert env["AGENT_AUTONOMOUS_CONTROL_RUN_ID"] == "bg_0"


def test_autonomous_start_resolves_approval_db_path_before_spawn(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway.autonomous_runner import AutonomousRegistry

  captured: dict[str, str] = {}

  async def fake_exec(*args, **kwargs):
    _ = args
    captured.update(dict(kwargs["env"]))
    return FakeProcess()

  server_dir = tmp_path / "server"
  api_dir = tmp_path / "api"
  server_dir.mkdir()
  api_dir.mkdir()
  monkeypatch.chdir(server_dir)
  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)

  registry = AutonomousRegistry(
    api_dir=api_dir,
    python_executable="python3",
    log_dir=tmp_path / "logs",
    max_running=1,
    approval_db_path=Path("data/gateway/approvals.sqlite3"),
  )

  payload = asyncio.run(
    registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
  )
  asyncio.run(registry.wait(payload["task_id"], timeout_sec=1))

  assert captured["AGENT_AUTONOMOUS_APPROVALS_DB_PATH"] == str(
    (server_dir / "data/gateway/approvals.sqlite3").resolve()
  )


def test_autonomous_start_injects_control_session_id(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  session_id = env["AGENT_AUTONOMOUS_GATEWAY_SESSION_ID"]
  assert session_id.startswith("agent-control:bg_0:")
  assert session_id.rsplit(":", 1)[1].isdigit()


def test_autonomous_start_does_not_clobber_existing_run_files_after_restart(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return FakeProcess()

  (tmp_path / "bg_0.log").write_text("stale log\n", encoding="utf-8")
  (tmp_path / "bg_0.events.jsonl").write_text('{"type":"stale"}\n', encoding="utf-8")
  (tmp_path / "bg_0.operator-messages.jsonl").write_text(
    '{"message_id":"stale","message":"old"}\n',
    encoding="utf-8",
  )
  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(tmp_path)

  payload = asyncio.run(
    registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
  )

  assert payload["task_id"] == "bg_1"
  assert (tmp_path / "bg_0.log").read_text(encoding="utf-8") == "stale log\n"
  assert (tmp_path / "bg_0.events.jsonl").read_text(encoding="utf-8") == '{"type":"stale"}\n'
  assert (tmp_path / "bg_0.operator-messages.jsonl").read_text(encoding="utf-8") == (
    '{"message_id":"stale","message":"old"}\n'
  )
  record = registry._tasks[payload["task_id"]]
  assert record.log_path.read_bytes() == b""
  assert record.events_path.read_text(encoding="utf-8") == ""
  assert record.operator_inbox_path is not None
  assert record.operator_inbox_path.read_text(encoding="utf-8") == ""


def test_autonomous_seq_continues_past_legacy_run_files(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return FakeProcess()

  (tmp_path / "bg_2.log").write_text("legacy log\n", encoding="utf-8")
  (tmp_path / "bg_7.events.jsonl").write_text("", encoding="utf-8")
  (tmp_path / "bg_9.operator-messages.jsonl").write_text("", encoding="utf-8")
  (tmp_path / "bg_11.task.json").write_text('{"manifest_version":1}\n', encoding="utf-8")
  (tmp_path / "bg_nope.log").write_text("", encoding="utf-8")
  (tmp_path / "not-bg_99.log").write_text("", encoding="utf-8")
  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(tmp_path)

  payload = asyncio.run(
    registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
  )

  assert payload["task_id"] == "bg_12"
  assert (tmp_path / "bg_9.operator-messages.jsonl").read_text(encoding="utf-8") == ""


def test_autonomous_registry_rehydrates_manifest_with_event_history(tmp_path) -> None:
  _write_manifest(
    tmp_path,
    "bg_3",
    control_run_id="run-3",
    resumed_from="run-2",
    resumed_as=["run-4"],
  )
  (tmp_path / "bg_3.events.jsonl").write_text(
    "\n".join(
      [
        '{"type":"text_delta","text":"hello","ts":101}',
        "not-json",
        '{"type":"skill_result_captured","skill_run_id":"skill-1","skill":"earnings-review","verdict_echo":{"verdict_token":"watch","confidence":"medium","one_line_summary":"Hold course"},"ts":102}',
      ]
    )
    + "\n",
    encoding="utf-8",
  )

  registry = _registry(tmp_path)

  assert registry._seq == 4
  record = registry._tasks["bg_3"]
  assert record.control_run_id == "run-3"
  assert record.user_id == USER_ID
  assert record.user_email == USER_EMAIL
  assert record.profile == "analyst"
  assert record.skill == "earnings-review"
  assert record.ticker == "AAPL"
  assert record.channel == "tui"
  assert record.state == "completed"
  assert record.exit_code == 0
  assert record.completed_at == 125.0
  assert record.proc is None
  assert record.reaper_task is None
  assert record.events_tail_task is None
  assert record.log_handle is None
  assert [event["type"] for event in record.event_lines or []] == ["text_delta", "skill_result_captured"]
  assert record.resumed_from == "run-2"
  assert list(record.resumed_as) == ["run-4"]

  record.resumed_as.append("run-5")
  assert _read_manifest(tmp_path, "bg_3")["resumed_as"] == ["run-4", "run-5"]


def test_autonomous_registry_rehydrates_budget_exceeded_as_budget_limited(tmp_path) -> None:
  _write_manifest(
    tmp_path,
    "bg_7",
    control_run_id="run-budget",
    state="completed",
    exit_code=2,
    error="Process exited with code 2",
    completed_at=200.0,
  )
  (tmp_path / "bg_7.events.jsonl").write_text(
    '{"type":"budget_exceeded","total_cost":1.25,"budget":1.0,"ts":199}\n',
    encoding="utf-8",
  )

  registry = _registry(tmp_path)

  record = registry._tasks["bg_7"]
  assert record.state == "budget_limited"
  assert record.error is None
  assert record.event_lines == [{"type": "budget_exceeded", "total_cost": 1.25, "budget": 1.0, "ts": 199}]
  assert registry._terminal_state_for_record(record) == "budget_limited"
  manifest = _read_manifest(tmp_path, "bg_7")
  assert manifest["state"] == "budget_limited"
  assert manifest["error"] is None


@pytest.mark.parametrize("state", ["running", "approval_pending", "queued", "waiting"])
def test_autonomous_registry_rehydrates_active_states_as_interrupted(tmp_path, state) -> None:
  _write_manifest(
    tmp_path,
    "bg_0",
    state=state,
    exit_code=None,
    error=None,
    completed_at=None,
  )
  (tmp_path / "bg_0.events.jsonl").write_text(
    '{"type":"text_delta","text":"before restart","ts":456}\n',
    encoding="utf-8",
  )

  registry = _registry(tmp_path)

  record = registry._tasks["bg_0"]
  assert record.state == "interrupted"
  assert record.exit_code is None
  assert record.error == "gateway restarted while run was active"
  assert record.completed_at == 456.0
  manifest = _read_manifest(tmp_path)
  assert manifest["state"] == "interrupted"
  assert manifest["completed_at"] == 456.0
  assert manifest["error"] == "gateway restarted while run was active"


def test_autonomous_registry_rehydrates_active_state_without_event_timestamp_uses_rehydrate_time(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  _write_manifest(
    tmp_path,
    "bg_0",
    state="running",
    exit_code=None,
    error=None,
    completed_at=None,
  )
  (tmp_path / "bg_0.events.jsonl").write_text(
    "\n".join(
      [
        '{"type":"text_delta","text":"before restart"}',
        '{"type":"tool_call_start","timestamp":"not-a-number"}',
      ]
    )
    + "\n",
    encoding="utf-8",
  )
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: 789.25)

  registry = _registry(tmp_path)

  record = registry._tasks["bg_0"]
  assert record.state == "interrupted"
  assert record.exit_code is None
  assert record.error == "gateway restarted while run was active"
  assert record.completed_at == 789.25
  manifest = _read_manifest(tmp_path)
  assert manifest["state"] == "interrupted"
  assert manifest["completed_at"] == 789.25
  assert manifest["error"] == "gateway restarted while run was active"


def test_autonomous_registry_over_cap_events_rehydrate_from_tail(monkeypatch, tmp_path, caplog) -> None:
  from agent_gateway import autonomous_runner

  _write_manifest(tmp_path, "bg_0")
  (tmp_path / "bg_0.events.jsonl").write_text(
    "".join(f'{{"type":"event","idx":{idx},"ts":{idx}}}\n' for idx in range(5)),
    encoding="utf-8",
  )
  monkeypatch.setattr(autonomous_runner, "_REHYDRATE_EVENTS_SIZE_CAP_BYTES", 10)
  monkeypatch.setattr(autonomous_runner, "_REHYDRATE_EVENTS_TAIL_LINES", 2)

  with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
    registry = _registry(tmp_path)

  record = registry._tasks["bg_0"]
  assert [event["idx"] for event in record.event_lines or []] == [3, 4]
  assert "Autonomous events file exceeds rehydrate cap; loading tail only" in caplog.text


def test_autonomous_registry_skips_corrupt_unknown_and_legacy_runs(tmp_path, caplog) -> None:
  (tmp_path / "bg_0.log").write_text("legacy only\n", encoding="utf-8")
  _manifest_path(tmp_path, "bg_1").write_text("{not-json\n", encoding="utf-8")
  _write_manifest(tmp_path, "bg_2", manifest_version=999)

  with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
    registry = _registry(tmp_path)

  assert registry._tasks == {}
  assert registry._seq == 3
  assert "Skipping unreadable autonomous task manifest" in caplog.text
  assert "Skipping autonomous manifest with unsupported version" in caplog.text


def test_autonomous_run_retention_prunes_only_old_completed_unlinked_run_groups(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  now = 1_000_000.0
  old = now - (8 * 86400)
  recent = now - (2 * 86400)
  monkeypatch.setenv("AGENT_AUTONOMOUS_RUN_RETENTION_DAYS", "7")
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: now)

  _write_manifest(tmp_path, "bg_1", state="completed", completed_at=old)
  _write_manifest(tmp_path, "bg_2", state="completed", completed_at=recent)
  _write_manifest(tmp_path, "bg_3", state="failed", completed_at=old)
  _write_manifest(tmp_path, "bg_4", state="running", completed_at=None, started_at=old)
  _write_manifest(tmp_path, "bg_5", state="completed", completed_at=old, resumed_as=["bg_6"])
  _write_manifest(tmp_path, "bg_99", state="completed", completed_at=old)
  for task_id in ("bg_1", "bg_2", "bg_3", "bg_4", "bg_5", "bg_99"):
    _write_run_files(tmp_path, task_id)

  registry = _registry(tmp_path)

  assert not _run_files_exist(tmp_path, "bg_1")
  assert not _run_files_exist(tmp_path, "bg_99")
  for task_id in ("bg_2", "bg_3", "bg_4", "bg_5"):
    assert _run_files_exist(tmp_path, task_id)
  assert set(registry._tasks) == {"bg_2", "bg_3", "bg_4", "bg_5"}
  assert registry._tasks["bg_4"].state == "interrupted"
  assert registry._seq == 100
  cursor = json.loads((tmp_path / ".autonomous-sequence.json").read_text(encoding="utf-8"))
  assert cursor == {"next_seq": 100}


def test_autonomous_run_retention_sequence_cursor_prevents_reuse_after_prune(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  now = 1_000_000.0
  old = now - (8 * 86400)
  monkeypatch.setenv("AGENT_AUTONOMOUS_RUN_RETENTION_DAYS", "7")
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: now)
  _write_manifest(tmp_path, "bg_42", state="completed", completed_at=old)
  _write_run_files(tmp_path, "bg_42")

  first = _registry(tmp_path)
  assert first._seq == 43
  assert not _run_files_exist(tmp_path, "bg_42")

  second = _registry(tmp_path)
  assert second._seq == 43


def test_autonomous_run_retention_skips_prune_when_sequence_cursor_cannot_persist(
  monkeypatch,
  tmp_path,
  caplog,
) -> None:
  from agent_gateway import autonomous_runner

  now = 1_000_000.0
  old = now - (8 * 86400)
  monkeypatch.setenv("AGENT_AUTONOMOUS_RUN_RETENTION_DAYS", "7")
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: now)

  def fail_replace(src, dst):
    _ = src, dst
    raise OSError("readonly-ish")

  _write_manifest(tmp_path, "bg_7", state="completed", completed_at=old)
  _write_run_files(tmp_path, "bg_7")
  monkeypatch.setattr(autonomous_runner.os, "replace", fail_replace)

  with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
    registry = _registry(tmp_path)

  assert _run_files_exist(tmp_path, "bg_7")
  assert set(registry._tasks) == {"bg_7"}
  assert registry._seq == 8
  assert "Failed to write autonomous run sequence cursor" in caplog.text


def test_autonomous_run_retention_ignores_malformed_manifest_task_id(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  now = 1_000_000.0
  old = now - (8 * 86400)
  monkeypatch.setenv("AGENT_AUTONOMOUS_RUN_RETENTION_DAYS", "7")
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: now)
  manifest = _write_manifest(tmp_path, "bg_8", state="completed", completed_at=old)
  manifest["task_id"] = "*"
  _manifest_path(tmp_path, "bg_8").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
  _write_run_files(tmp_path, "bg_8")

  registry = _registry(tmp_path)

  assert _run_files_exist(tmp_path, "bg_8")
  assert registry._tasks == {}
  assert registry._seq == 9


def test_record_and_publish_seeds_rehydrated_events_before_new_publish(tmp_path) -> None:
  from agent_gateway.autonomous_runner import AutonomousRegistry

  _write_manifest(tmp_path, "bg_11", control_run_id="run-11", state="interrupted")
  (tmp_path / "bg_11.events.jsonl").write_text(
    json.dumps({"type": "text_delta", "text": "old"}) + "\n",
    encoding="utf-8",
  )
  bus = RecordingEventBus()
  registry = AutonomousRegistry(api_dir=tmp_path, log_dir=tmp_path, user_event_bus=bus)
  record = registry._tasks["bg_11"]

  asyncio.run(
    registry._record_and_publish_event(
      record,
      {"type": "run_resumed", "resumed_run_id": "run-12"},
    )
  )

  assert [name for name, _payload in bus.calls] == ["seed", "publish"]
  seed_payload = bus.calls[0][1]
  assert seed_payload["user_id"] == USER_ID
  assert seed_payload["control_run_id"] == "run-11"
  assert seed_payload["terminated"] is True
  assert seed_payload["events"] == [
    {
      "type": "text_delta",
      "text": "old",
      "run_id": "run-11",
      "control_run_id": "run-11",
    }
  ]
  assert bus.calls[1][1]["event"] == {
    "type": "run_resumed",
    "resumed_run_id": "run-12",
    "run_id": "run-11",
    "control_run_id": "run-11",
  }


def test_autonomous_manifest_written_post_spawn_with_full_field_set(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      assert not _manifest_path(tmp_path).exists()
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      profile="analyst",
      mode="skill",
      skill="risk.scan",
      context=" inspect current book ",
      ticker="msft",
      channel="TUI",
      dev_mode=True,
      control_run_id="run-custom",
      user_id=USER_ID,
      user_email=None,
      resumed_from="prior-run",
    )
    try:
      manifest = _read_manifest(tmp_path)
      assert set(manifest) == {
        "manifest_version",
        "task_id",
        "control_run_id",
        "user_id",
        "user_email",
        "profile",
        "mode",
        "task",
        "skill",
        "context",
        "ticker",
        "channel",
        "dev_mode",
        "cmd",
        "log_path",
        "events_path",
        "operator_inbox_path",
        "approval_decisions_path",
        "started_at",
        "state",
        "exit_code",
        "error",
        "completed_at",
        "resumed_from",
        "resumed_as",
      }
      assert manifest["manifest_version"] == 1
      assert manifest["task_id"] == payload["task_id"] == "bg_0"
      assert manifest["control_run_id"] == "run-custom"
      assert manifest["user_id"] == USER_ID
      assert manifest["user_email"] is None
      assert manifest["profile"] == "analyst"
      assert manifest["mode"] == "skill"
      assert manifest["task"] is None
      assert manifest["skill"] == "risk.scan"
      assert manifest["context"] == "inspect current book"
      assert manifest["ticker"] == "MSFT"
      assert manifest["channel"] == "tui"
      assert manifest["dev_mode"] is True
      assert manifest["cmd"][:5] == ["python3", "-m", "agent.autonomous", "--profile", "analyst"]
      assert "--dev" in manifest["cmd"]
      assert manifest["log_path"] == str(tmp_path / "bg_0.log")
      assert manifest["events_path"] == str(tmp_path / "bg_0.events.jsonl")
      assert manifest["operator_inbox_path"] == str(tmp_path / "bg_0.operator-messages.jsonl")
      assert manifest["approval_decisions_path"] == str(tmp_path / "bg_0.approval-decisions.jsonl")
      assert isinstance(manifest["started_at"], float)
      assert manifest["state"] == "running"
      assert manifest["exit_code"] is None
      assert manifest["error"] is None
      assert manifest["completed_at"] is None
      assert manifest["resumed_from"] == "prior-run"
      assert manifest["resumed_as"] == []
      assert "proc" not in manifest
      assert "reaper_task" not in manifest
      assert "events_tail_task" not in manifest
      assert "log_handle" not in manifest
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_manifest_removed_when_spawn_fails(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    raise OSError("boom")

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(tmp_path)

  with pytest.raises(RuntimeError, match="spawn failed"):
    asyncio.run(
      registry.start(
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert registry._tasks == {}
  assert registry._reserved_slots == 0
  assert not _manifest_path(tmp_path).exists()
  assert list(tmp_path.glob("*.task.json")) == []


@pytest.mark.parametrize(
  ("process_factory", "expected_state", "expected_exit_code", "expected_error"),
  [
    (FakeProcess, "completed", 0, None),
    (FailingFakeProcess, "failed", 1, "Process exited with code 1"),
  ],
)
def test_autonomous_manifest_updates_on_terminal_reap(
  monkeypatch,
  tmp_path,
  process_factory,
  expected_state,
  expected_exit_code,
  expected_error,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return process_factory()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(tmp_path)
  payload = asyncio.run(
    registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
  )

  asyncio.run(registry.wait(payload["task_id"], timeout_sec=1))

  manifest = _read_manifest(tmp_path)
  assert manifest["state"] == expected_state
  assert manifest["exit_code"] == expected_exit_code
  assert manifest["error"] == expected_error
  assert isinstance(manifest["completed_at"], float)


def test_autonomous_manifest_updates_on_reaper_failure(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return RaisingFakeProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(tmp_path)
  payload = asyncio.run(
    registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
  )

  asyncio.run(registry.wait(payload["task_id"], timeout_sec=1))

  manifest = _read_manifest(tmp_path)
  assert manifest["state"] == "failed"
  assert manifest["exit_code"] is None
  assert manifest["error"] == "reaper failed: reap boom"
  assert isinstance(manifest["completed_at"], float)


def test_autonomous_manifest_updates_on_cancel(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )

    await registry.cancel(payload["task_id"])
    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "killed"
    assert manifest["error"] == "Process terminated by user"
    assert isinstance(manifest["completed_at"], float)

    await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_manifest_updates_on_shutdown(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )

    await registry.shutdown(grace_sec=0.1)

    manifest = _read_manifest(tmp_path, payload["task_id"])
    assert manifest["state"] == "killed"
    assert manifest["exit_code"] == -15
    assert manifest["error"] == "Process terminated during gateway shutdown"
    assert isinstance(manifest["completed_at"], float)

  asyncio.run(_case())


def test_autonomous_manifest_updates_on_resume_linkage_append(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    try:
      record = registry._tasks[payload["task_id"]]
      record.resumed_as.append("bg_1")

      manifest = _read_manifest(tmp_path)
      assert manifest["resumed_as"] == ["bg_1"]
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_manifest_write_failures_do_not_block_transitions(
  monkeypatch,
  tmp_path,
  caplog,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return SlowFakeProcess()

    def fail_replace(src, dst):
      _ = src, dst
      raise OSError("readonly-ish")

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(autonomous_runner.os, "replace", fail_replace)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
      payload = await registry.start(
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
      status = await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

    assert status["state"] == "killed"
    assert not _manifest_path(tmp_path).exists()
    assert "Failed to write autonomous task manifest for bg_0" in caplog.text

  asyncio.run(_case())


def test_autonomous_operator_message_idempotency_is_concurrent_safe(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
      channel="tui",
    )
    try:
      deliveries = await asyncio.gather(
        registry.send_operator_message(
          payload["run_id"],
          user_id=USER_ID,
          channel="tui",
          message="Check AWS exposure",
          message_id="msg-1",
        ),
        registry.send_operator_message(
          payload["run_id"],
          user_id=USER_ID,
          channel="tui",
          message="Check AWS exposure",
          message_id="msg-1",
        ),
      )
      assert sorted(delivery["delivery_status"] for delivery in deliveries) == ["delivered", "duplicate"]
      record = registry._tasks[payload["task_id"]]
      assert record.operator_inbox_path is not None
      lines = record.operator_inbox_path.read_text(encoding="utf-8").splitlines()
      assert len(lines) == 1
      assert json.loads(lines[0])["message_id"] == "msg-1"
      assert [event["type"] for event in record.event_lines].count("parent_message_sent") == 1
      assert record.events_path is not None
      persisted_events = [
        json.loads(line)
        for line in record.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
      ]
      persisted_parent_events = [
        event for event in persisted_events if event["type"] == "parent_message_sent"
      ]
      assert len(persisted_parent_events) == 1
      assert persisted_parent_events[0]["message_id"] == "msg-1"
      record.delivered_messages.clear()
      record.event_lines = [
        event for event in record.event_lines if event.get("type") != "parent_message_sent"
      ]
      record.events_path.write_text("", encoding="utf-8")
      repaired = await registry.send_operator_message(
        payload["run_id"],
        user_id=USER_ID,
        channel="tui",
        message="Different retry text should not replace the inbox payload",
        message_id="msg-1",
      )
      assert repaired["delivery_status"] == "duplicate"
      assert len(record.operator_inbox_path.read_text(encoding="utf-8").splitlines()) == 1
      repaired_events = [
        json.loads(line)
        for line in record.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
      ]
      repaired_parent_events = [
        event for event in repaired_events if event["type"] == "parent_message_sent"
      ]
      assert len(repaired_parent_events) == 1
      assert repaired_parent_events[0]["message"] == "Check AWS exposure"
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_operator_message_rejects_after_stream_complete(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
      channel="tui",
    )
    try:
      record = registry._tasks[payload["task_id"]]
      assert record.event_lines is not None
      record.event_lines.append({"type": "stream_complete"})

      with pytest.raises(RuntimeError, match="no longer accepting messages"):
        await registry.send_operator_message(
          payload["run_id"],
          user_id=USER_ID,
          channel="tui",
          message="too late",
        )
      assert record.operator_inbox_path is not None
      assert record.operator_inbox_path.read_text(encoding="utf-8") == ""
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_start_does_not_mutate_parent_environ(monkeypatch, tmp_path) -> None:
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  for key in CLAIM_ENV_KEYS:
    monkeypatch.delenv(key, raising=False)

  asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert os.environ.get("AGENT_API_USER_CLAIM_HMAC_KEY") == HMAC_KEY
  for key in CLAIM_ENV_KEYS:
    assert os.environ.get(key) is None


def test_autonomous_start_hmac_handoff_preserves_key_on_spawn_failure(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner

  captured_env: dict[str, str] = {}

  async def fake_exec(*args, **kwargs):
    captured_env.update(dict(kwargs["env"]))
    raise OSError("boom")

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(tmp_path)

  with pytest.raises(RuntimeError, match="spawn failed"):
    asyncio.run(
      registry.start(
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert captured_env["AGENT_API_USER_CLAIM_HMAC_KEY"] == HMAC_KEY
  assert os.environ.get("AGENT_API_USER_CLAIM_HMAC_KEY") == HMAC_KEY
  assert registry._tasks == {}
  assert registry._reserved_slots == 0
  assert not _manifest_path(tmp_path).exists()


def test_autonomous_start_aliases_autonomous_identity(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert env["AUTONOMOUS_USER_ID"] == USER_ID
  assert env["AUTONOMOUS_USER_EMAIL"] == USER_EMAIL
  assert env["AUTONOMOUS_USER_ID"] == env["AGENT_API_CLAIM_USER_ID"]
  assert env["AUTONOMOUS_USER_EMAIL"] == env["AGENT_API_CLAIM_USER_EMAIL"]


def test_autonomous_start_overrides_preexisting_autonomous_identity(monkeypatch, tmp_path) -> None:
  monkeypatch.setenv("AUTONOMOUS_USER_ID", "other")
  monkeypatch.setenv("AUTONOMOUS_USER_EMAIL", "other@example.com")

  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert env["AUTONOMOUS_USER_ID"] == USER_ID
  assert env["AUTONOMOUS_USER_EMAIL"] == USER_EMAIL
  assert env["AGENT_API_CLAIM_USER_ID"] == USER_ID
  assert env["AGENT_API_CLAIM_USER_EMAIL"] == USER_EMAIL


def test_autonomous_start_fails_without_hmac_key(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner

  called = False

  async def fake_exec(*args, **kwargs):
    nonlocal called
    called = True
    return FakeProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.delenv("AGENT_API_USER_CLAIM_HMAC_KEY", raising=False)
  registry = _registry(tmp_path)

  with pytest.raises(RuntimeError, match="AGENT_API_USER_CLAIM_HMAC_KEY required"):
    asyncio.run(
      registry.start(
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert called is False
  assert registry._tasks == {}
  assert registry._reserved_slots == 0
  assert not _manifest_path(tmp_path).exists()


@_requires_risk_module
def test_signed_claim_uses_configured_ttl_at_verifier_ceiling(monkeypatch, tmp_path) -> None:
  assert AGENT_API_CLAIM_MAX_TTL_SECONDS == 600

  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path, ttl_seconds=600))

  issued_at = int(env["AGENT_API_CLAIM_ISSUED_AT"])
  expiry = int(env["AGENT_API_CLAIM_EXPIRY"])
  assert expiry - issued_at == 600
  assert verify(
    HMAC_KEY,
    _claim_headers_from_env(env),
    ttl_ceiling=600,
    now=issued_at,
  ) is not None


@_requires_risk_module
def test_signed_claim_above_ceiling_rejected_by_verifier(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path, ttl_seconds=900))

  issued_at = int(env["AGENT_API_CLAIM_ISSUED_AT"])
  expiry = int(env["AGENT_API_CLAIM_EXPIRY"])
  assert expiry - issued_at == 900
  assert verify(
    HMAC_KEY,
    _claim_headers_from_env(env),
    ttl_ceiling=600,
    now=issued_at,
  ) is None
