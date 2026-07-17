from __future__ import annotations

import asyncio
import fcntl
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


def test_autonomous_runner_state_helper_preserves_parent_aliases() -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway import autonomous_runner_claims
  from agent_gateway import autonomous_runner_start
  from agent_gateway import autonomous_runner_state

  assert autonomous_runner.AutonomousTask is autonomous_runner_state.AutonomousTask
  assert autonomous_runner._ManifestTrackedList is autonomous_runner_state._ManifestTrackedList
  assert autonomous_runner.sign_user_claim is autonomous_runner_claims.sign_user_claim
  assert (
    autonomous_runner.get_agent_api_claim_ttl_seconds
    is autonomous_runner_claims.get_agent_api_claim_ttl_seconds
  )
  assert (
    autonomous_runner.AutonomousRegistry._task_from_manifest
    is autonomous_runner_state.AutonomousRegistryStateMixin._task_from_manifest
  )
  assert (
    autonomous_runner.AutonomousRegistry._write_task_manifest
    is autonomous_runner_state.AutonomousRegistryStateMixin._write_task_manifest
  )
  assert (
    autonomous_runner.AutonomousRegistry.rehydrate
    is autonomous_runner_state.AutonomousRegistryStateMixin.rehydrate
  )
  assert (
    autonomous_runner.AutonomousRegistry.start
    is autonomous_runner_start.AutonomousRegistryStartMixin.start
  )


def test_autonomous_runner_event_helper_preserves_parent_override_seams(tmp_path) -> None:
  from agent_gateway.autonomous_runner import AutonomousRegistry

  _write_manifest(tmp_path, "bg_0", control_run_id="run-0")
  registry = AutonomousRegistry(api_dir=tmp_path, log_dir=tmp_path)
  record = registry._tasks["bg_0"]
  record.event_lines = [{"type": "custom", "marker": "same"}]

  def duplicate_key(event: dict) -> tuple[str, str] | None:
    marker = event.get("marker")
    if not marker:
      return None
    return ("custom", str(marker))

  registry._event_duplicate_key = duplicate_key  # type: ignore[method-assign]
  assert registry._event_already_recorded(record, {"type": "other", "marker": "same"})

  registry._operator_inbox_record_for_message_id = (  # type: ignore[method-assign]
    lambda _record, _message_id: {
      "message": "from inbox",
      "sent_at": "from-inbox",
      "sender": {"user_id": "operator"},
    }
  )
  registry._event_for_record = (  # type: ignore[method-assign]
    lambda _record, event: {**event, "run_id": "patched-run", "control_run_id": "patched-run"}
  )

  parent_event = registry._parent_message_event(
    record,
    message_id="msg-1",
    text="fallback",
    user_id=USER_ID,
    sent_at=123.0,
  )

  assert parent_event["message"] == "from inbox"
  assert parent_event["sent_at"] == "from-inbox"
  assert parent_event["sender"] == {"user_id": "operator"}
  assert parent_event["run_id"] == "patched-run"


def test_autonomous_runner_status_helper_preserves_parent_tail_override(tmp_path) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway import autonomous_runner_status
  from agent_gateway.autonomous_runner import AutonomousRegistry

  assert autonomous_runner.AutonomousRegistry._status_payload.__module__ == autonomous_runner.__name__
  assert autonomous_runner_status.status_payload.__module__ == autonomous_runner_status.__name__

  _write_manifest(tmp_path, "bg_0", state="failed", exit_code=2, error="boom")
  registry = AutonomousRegistry(api_dir=tmp_path, log_dir=tmp_path)
  record = registry._tasks["bg_0"]

  def tail_lines(log_path: Path, line_count: int) -> tuple[list[str], int]:
    assert log_path == record.log_path
    assert line_count == autonomous_runner._STATUS_TAIL_LINES
    return ["patched line"], 1

  registry._tail_lines = tail_lines  # type: ignore[method-assign]

  assert registry._status_payload(record) == {
    "state": "failed",
    "elapsed_sec": record.elapsed_sec,
    "exit_code": 2,
    "error": "boom",
    "log_tail": "patched line",
  }


def test_autonomous_runner_status_tail_lines_counts_and_tails(tmp_path) -> None:
  from agent_gateway import autonomous_runner_status

  log_path = tmp_path / "run.log"
  log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

  assert autonomous_runner_status.tail_lines(log_path, 2) == (["two", "three"], 3)
  assert autonomous_runner_status.tail_lines(log_path, 0) == ([], 3)
  assert autonomous_runner_status.tail_lines(tmp_path / "missing.log", 10) == ([], 0)


def test_autonomous_runner_command_helper_preserves_parent_override_seams(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway import autonomous_runner_commands
  from agent_gateway.autonomous_runner import AutonomousRegistry

  assert autonomous_runner._AUTONOMOUS_PROFILE_NAME_RE is autonomous_runner_commands._AUTONOMOUS_PROFILE_NAME_RE
  assert autonomous_runner.normalize_autonomous_profile.__module__ == autonomous_runner.__name__
  assert (
    autonomous_runner_commands.normalize_autonomous_profile.__module__
    == autonomous_runner_commands.__name__
  )

  checks: list[str] = []

  def normalize_profile(profile: str) -> str:
    checks.append(f"profile:{profile}")
    return "patched_profile"

  def is_fixture_profile(profile: str) -> bool:
    checks.append(f"fixture-profile:{profile}")
    return False

  def is_fixture_skill(skill: str) -> bool:
    checks.append(f"fixture-skill:{skill}")
    return skill == "patched-fixture"

  def require_fixture_provider_available(reason: str, **kwargs) -> None:
    checks.append(f"guard:{reason}:{kwargs['error_type'].__name__}")

  monkeypatch.setattr(autonomous_runner, "normalize_autonomous_profile", normalize_profile)
  monkeypatch.setattr(autonomous_runner, "is_fixture_profile_name", is_fixture_profile)
  monkeypatch.setattr(autonomous_runner, "is_fixture_skill_name", is_fixture_skill)
  monkeypatch.setattr(
    autonomous_runner,
    "require_fixture_provider_available",
    require_fixture_provider_available,
  )

  registry = AutonomousRegistry(api_dir=tmp_path, python_executable="py", log_dir=tmp_path)
  cmd = registry._build_cmd(
    profile="Analyst",
    mode="skill",
    task=None,
    skill="patched-fixture",
    context=" context ",
    ticker="msft",
    dev_mode=True,
    max_budget_usd=5.0,
  )

  assert cmd == [
    "py",
    "-m",
    "agent.autonomous",
    "--profile",
    "patched_profile",
    "--dev",
    "--skill",
    "patched-fixture",
    "--max-budget-usd",
    "5.0",
    "--ticker",
    "MSFT",
    "--context",
    "context",
  ]
  assert checks == [
    "profile:Analyst",
    "fixture-profile:patched_profile",
    "fixture-skill:patched-fixture",
    "guard:fixture skill dispatch:ValueError",
  ]


@pytest.mark.parametrize("max_budget_usd", [True, 0, -1, float("inf"), float("nan"), "5"])
def test_autonomous_runner_rejects_invalid_max_budget(max_budget_usd, tmp_path) -> None:
  registry = _registry(tmp_path)

  with pytest.raises(ValueError, match="max_budget_usd must be a finite positive number"):
    registry._build_cmd(
      profile="analyst",
      mode="skill",
      task=None,
      skill="risk.scan",
      context=None,
      max_budget_usd=max_budget_usd,
    )


@pytest.mark.parametrize(
  ("mode", "task", "skill"),
  [("once", None, None), ("task", "summarize", None)],
)
def test_autonomous_runner_rejects_max_budget_outside_skill_mode(
  mode,
  task,
  skill,
  tmp_path,
) -> None:
  registry = _registry(tmp_path)

  with pytest.raises(ValueError, match="max_budget_usd requires mode='skill'"):
    registry._build_cmd(
      profile="analyst",
      mode=mode,
      task=task,
      skill=skill,
      context=None,
      max_budget_usd=5.0,
    )


def test_autonomous_runner_claim_helper_uses_parent_aliases(monkeypatch) -> None:
  from agent_gateway import autonomous_runner

  class _Clock:
    @staticmethod
    def time() -> float:
      return 1000.0

  patched_env_vars = {
    "audience": "PATCHED_AUDIENCE",
    "issued_at": "PATCHED_ISSUED_AT",
    "expiry": "PATCHED_EXPIRY",
    "user_id": "PATCHED_USER_ID",
    "user_email": "PATCHED_USER_EMAIL",
    "nonce": "PATCHED_NONCE",
    "signature": "PATCHED_SIGNATURE",
  }
  monkeypatch.delenv("AGENT_API_CLAIM_TTL_SECONDS", raising=False)
  monkeypatch.setattr(autonomous_runner, "_AGENT_API_CLAIM_TTL_SECONDS_DEFAULT", 123)
  monkeypatch.setattr(autonomous_runner, "_AGENT_API_CLAIM_AUDIENCE", "patched_audience")
  monkeypatch.setattr(autonomous_runner, "_AGENT_API_CLAIM_ENV_VARS", patched_env_vars)
  monkeypatch.setattr(autonomous_runner, "time", _Clock)

  claim = autonomous_runner.sign_user_claim(
    HMAC_KEY,
    user_id=USER_ID,
    user_email=None,
    ttl_seconds=5,
  )

  assert autonomous_runner.get_agent_api_claim_ttl_seconds() == 123
  assert claim["PATCHED_AUDIENCE"] == "patched_audience"
  assert claim["PATCHED_ISSUED_AT"] == "1000"
  assert claim["PATCHED_EXPIRY"] == "1005"
  assert claim["PATCHED_USER_ID"] == USER_ID
  assert claim["PATCHED_USER_EMAIL"] == ""
  assert claim["PATCHED_NONCE"]
  assert claim["PATCHED_SIGNATURE"]


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
  owner_user_id: str | None = None,
  user_slug: str | None = None,
  risk_user_id: int | None = None,
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
    owner_user_id=owner_user_id,
    user_slug=user_slug,
    risk_user_id=risk_user_id,
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
    max_budget_usd=4.5,
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
  assert record.max_budget_usd == 4.5
  assert record.proc is None
  assert record.reaper_task is None
  assert record.events_tail_task is None
  assert record.log_handle is None
  assert [event["type"] for event in record.event_lines or []] == ["text_delta", "skill_result_captured"]
  assert record.resumed_from == "run-2"
  assert list(record.resumed_as) == ["run-4"]

  record.resumed_as.append("run-5")
  assert _read_manifest(tmp_path, "bg_3")["resumed_as"] == ["run-4", "run-5"]


def test_autonomous_registry_rehydrates_v1_slug_manifest_to_canonical_owner(monkeypatch, tmp_path) -> None:
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([
      {
        "key": "mcp-key",
        "channel": "mcp",
        "slug": "henry",
        "email": "henry@example.com",
        "risk_user_id": 1,
        "role": "owner",
      }
    ]),
  )
  _write_manifest(
    tmp_path,
    "bg_4",
    control_run_id="run-4",
    user_id="henry",
    user_email="henry@example.com",
  )

  registry = _registry(tmp_path)

  record = registry._tasks["bg_4"]
  assert record.user_id == "1"
  assert record.owner_user_id == "1"
  assert record.raw_user_id == "henry"
  assert record.user_slug == "henry"
  assert record.risk_user_id == 1
  assert record.user_aliases == ["1", "henry", "henry@example.com"]
  assert record.identity_status == "gateway_user_key_mapping"
  assert record.max_budget_usd is None
  upgraded_manifest = _read_manifest(tmp_path, "bg_4")
  assert upgraded_manifest["manifest_version"] == 2
  assert upgraded_manifest["owner_user_id"] == "1"
  assert upgraded_manifest["user_id"] == "1"
  assert upgraded_manifest["raw_user_id"] == "henry"


@pytest.mark.parametrize("max_budget_usd", [True, 0, -1, float("inf"), float("nan"), "5"])
def test_autonomous_registry_drops_invalid_manifest_max_budget(tmp_path, max_budget_usd) -> None:
  _write_manifest(tmp_path, max_budget_usd=max_budget_usd)

  registry = _registry(tmp_path)

  assert registry._tasks["bg_0"].max_budget_usd is None


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


def test_autonomous_registry_rehydrates_blocked_as_terminal(tmp_path) -> None:
  _write_manifest(
    tmp_path,
    "bg_8",
    control_run_id="run-blocked",
    state="blocked",
    exit_code=0,
    error="same blocker exhausted",
    completed_at=200.0,
  )

  registry = _registry(tmp_path)

  record = registry._tasks["bg_8"]
  assert record.state == "blocked"
  assert record.error == "same blocker exhausted"
  assert record.completed_at == 200.0
  assert registry._terminal_state_for_record(record) == "blocked"


@pytest.mark.parametrize("state", ["running", "approval_pending", "queued", "waiting", "remediating"])
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


def test_legacy_autonomous_run_retention_env_no_longer_prunes_on_registry_boot(
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

  for task_id in ("bg_1", "bg_2", "bg_3", "bg_4", "bg_5", "bg_99"):
    assert _run_files_exist(tmp_path, task_id)
  assert set(registry._tasks) == {"bg_1", "bg_2", "bg_3", "bg_4", "bg_5", "bg_99"}
  assert registry._tasks["bg_4"].state == "interrupted"
  assert registry._seq == 100
  assert not (tmp_path / ".autonomous-sequence.json").exists()


def test_starting_manifest_cleanup_removes_dead_child_spill_before_rehydrate(tmp_path) -> None:
  spill_dir = tmp_path / "bg_12.tool_result_spill"
  spill_dir.mkdir()
  (spill_dir / "partial.index.json").write_text("partial", encoding="utf-8")
  _write_manifest(
    tmp_path,
    "bg_12",
    state="starting",
    exit_code=None,
    completed_at=None,
    tool_result_spill_dir=str(spill_dir),
  )

  registry = _registry(tmp_path)

  assert not spill_dir.exists()
  assert registry._tasks["bg_12"].state == "interrupted"
  assert _read_manifest(tmp_path, "bg_12")["state"] == "interrupted"


def test_starting_manifest_cleanup_skips_surviving_child_lease_then_retries(tmp_path) -> None:
  spill_dir = tmp_path / "bg_13.tool_result_spill"
  spill_dir.mkdir()
  _write_manifest(
    tmp_path,
    "bg_13",
    state="starting",
    exit_code=None,
    completed_at=None,
    tool_result_spill_dir=str(spill_dir),
  )
  lease_path = spill_dir / ".lease"
  fd = os.open(lease_path, os.O_RDWR | os.O_CREAT, 0o600)
  fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  try:
    while_live = _registry(tmp_path)
    assert spill_dir.is_dir()
    assert "bg_13" not in while_live._tasks
    assert _read_manifest(tmp_path, "bg_13")["state"] == "starting"
  finally:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)

  after_exit = _registry(tmp_path)
  assert not spill_dir.exists()
  assert after_exit._tasks["bg_13"].state == "interrupted"


def test_legacy_run_retention_env_leaves_registered_spill_for_central_sweeper(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  now = 1_000_000.0
  old = now - (8 * 86400)
  monkeypatch.setenv("AGENT_AUTONOMOUS_RUN_RETENTION_DAYS", "7")
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: now)
  spill_dir = tmp_path / "bg_14.tool_result_spill"
  spill_dir.mkdir()
  (spill_dir / "committed.json").write_text("{}", encoding="utf-8")
  _write_manifest(
    tmp_path,
    "bg_14",
    state="completed",
    completed_at=old,
    tool_result_spill_dir=str(spill_dir),
  )
  _write_run_files(tmp_path, "bg_14")

  registry = _registry(tmp_path)

  assert spill_dir.exists()
  assert _run_files_exist(tmp_path, "bg_14")
  assert "bg_14" in registry._tasks


def test_run_retention_never_traverses_symlinked_registered_spill(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  now = 1_000_000.0
  old = now - (8 * 86400)
  monkeypatch.setenv("AGENT_AUTONOMOUS_RUN_RETENTION_DAYS", "7")
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: now)
  outside = tmp_path / "outside"
  outside.mkdir()
  evidence = outside / "keep.txt"
  evidence.write_text("keep", encoding="utf-8")
  linked = tmp_path / "bg_15.tool_result_spill"
  linked.symlink_to(outside, target_is_directory=True)
  _write_manifest(
    tmp_path,
    "bg_15",
    state="completed",
    completed_at=old,
    tool_result_spill_dir=str(linked),
  )
  _write_run_files(tmp_path, "bg_15")

  _registry(tmp_path)

  assert evidence.read_text(encoding="utf-8") == "keep"
  assert linked.is_symlink()


def test_spill_cleanup_detects_directory_replacement_after_lease(
  tmp_path,
) -> None:
  registry = _registry(tmp_path)
  spill_dir = tmp_path / "bg_16.tool_result_spill"
  spill_dir.mkdir()
  outside = tmp_path / "outside-replacement"
  outside.mkdir()
  evidence = outside / "keep.txt"
  evidence.write_text("keep", encoding="utf-8")
  original_acquire = registry._acquire_tool_result_spill_cleanup_lease

  def replace_after_acquire(path: Path):
    lease = original_acquire(path)
    assert lease is not None
    path.rename(tmp_path / "moved.tool_result_spill")
    path.symlink_to(outside, target_is_directory=True)
    return lease

  registry._acquire_tool_result_spill_cleanup_lease = replace_after_acquire  # type: ignore[method-assign]

  assert registry._remove_registered_tool_result_spill_dir("bg_16", spill_dir) is False
  assert evidence.read_text(encoding="utf-8") == "keep"
  assert spill_dir.is_symlink()


def test_registry_sequence_scan_prevents_reuse_without_boot_retention(
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
  assert _run_files_exist(tmp_path, "bg_42")

  second = _registry(tmp_path)
  assert second._seq == 43


def test_retired_boot_retention_does_not_attempt_sequence_cursor_persistence(
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
  assert "Failed to write autonomous run sequence cursor" not in caplog.text


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


def test_autonomous_manifest_committed_before_spawn_with_full_field_set(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args
      starting = _read_manifest(tmp_path)
      assert starting["state"] == "starting"
      assert Path(starting["tool_result_spill_dir"]).is_dir()
      assert kwargs["env"]["AGENT_AUTONOMOUS_TOOL_RESULT_SPILL_DIR"] == starting["tool_result_spill_dir"]
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    monkeypatch.delenv("GATEWAY_USER_KEYS", raising=False)
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
      max_budget_usd=5.0,
    )
    try:
      manifest = _read_manifest(tmp_path)
      assert set(manifest) == {
        "manifest_version",
        "task_id",
        "control_run_id",
        "owner_user_id",
        "user_id",
        "raw_user_id",
        "user_slug",
        "risk_user_id",
        "user_email",
        "user_aliases",
        "identity_status",
        "profile",
        "mode",
        "task",
        "skill",
        "context",
        "ticker",
        "channel",
        "dev_mode",
        "max_budget_usd",
        "dispatch_scope",
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
        "schedule_id",
        "schedule_name",
        "tool_result_spill_dir",
      }
      assert manifest["manifest_version"] == 2
      assert manifest["task_id"] == payload["task_id"] == "bg_0"
      assert manifest["control_run_id"] == "run-custom"
      assert manifest["owner_user_id"] == USER_ID
      assert manifest["user_id"] == USER_ID
      assert manifest["raw_user_id"] == USER_ID
      assert manifest["user_slug"] is None
      assert manifest["risk_user_id"] == 1
      assert manifest["user_email"] is None
      assert manifest["user_aliases"] == [USER_ID]
      assert manifest["identity_status"] == "numeric_user_id"
      assert manifest["profile"] == "analyst"
      assert manifest["mode"] == "skill"
      assert manifest["task"] is None
      assert manifest["skill"] == "risk.scan"
      assert manifest["context"] == "inspect current book"
      assert manifest["ticker"] == "MSFT"
      assert manifest["channel"] == "tui"
      assert manifest["dev_mode"] is True
      assert manifest["max_budget_usd"] == 5.0
      assert manifest["dispatch_scope"] is None
      assert manifest["cmd"][:5] == ["python3", "-m", "agent.autonomous", "--profile", "analyst"]
      assert "--dev" in manifest["cmd"]
      assert manifest["cmd"][manifest["cmd"].index("--max-budget-usd") + 1] == "5.0"
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
      assert manifest["schedule_id"] is None
      assert manifest["schedule_name"] is None
      assert manifest["tool_result_spill_dir"] == str(tmp_path / "bg_0.tool_result_spill")
      assert "proc" not in manifest
      assert "reaper_task" not in manifest
      assert "events_tail_task" not in manifest
      assert "log_handle" not in manifest
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_start_rejects_unredacted_dispatch_scope_before_spawn(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      raise AssertionError("dispatch_scope validation must run before spawn")

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    registry = _registry(tmp_path)
    with pytest.raises(ValueError, match="dispatch_scope"):
      await registry.start(
        profile="analyst",
        mode="task",
        task="Review scope.",
        user_id=USER_ID,
        user_email=None,
        dispatch_scope={
          "kind": "portfolio",
          "source": "user_selected",
          "portfolio_name": "taxable_combined",
          "account_id": "acc-1",
        },
      )

    assert registry._tasks == {}
    assert not _manifest_path(tmp_path).exists()

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
  assert not (tmp_path / "bg_0.tool_result_spill").exists()


def test_autonomous_initial_manifest_failure_disables_spill_but_still_starts(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    captured_env: dict[str, str] = {}

    async def fake_exec(*args, **kwargs):
      _ = args
      captured_env.update(kwargs["env"])
      assert "AGENT_AUTONOMOUS_TOOL_RESULT_SPILL_DIR" not in kwargs["env"]
      assert not (tmp_path / "bg_0.tool_result_spill").exists()
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    original_write = registry._write_task_manifest
    calls = 0

    def fail_initial_once(record, *, checked: bool = False):
      nonlocal calls
      calls += 1
      if calls == 1:
        assert checked is True
        return False
      return original_write(record, checked=checked)

    registry._write_task_manifest = fail_initial_once  # type: ignore[method-assign]
    payload = await registry.start(
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    try:
      manifest = _read_manifest(tmp_path)
      assert manifest["state"] == "running"
      assert manifest["tool_result_spill_dir"] is None
      assert "AGENT_AUTONOMOUS_TOOL_RESULT_SPILL_DIR" not in captured_env
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_post_spawn_manifest_failure_terminates_child_and_cleans_spill(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    process = SlowFakeProcess()

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      assert (tmp_path / "bg_0.tool_result_spill").is_dir()
      return process

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    original_write = registry._write_task_manifest
    calls = 0

    def fail_running_commit(record, *, checked: bool = False):
      nonlocal calls
      calls += 1
      if calls == 2:
        assert checked is True
        return False
      return original_write(record, checked=checked)

    registry._write_task_manifest = fail_running_commit  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="commit running autonomous task manifest"):
      await registry.start(
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )

    assert process.returncode == -15
    assert not (tmp_path / "bg_0.tool_result_spill").exists()
    assert not _manifest_path(tmp_path).exists()
    assert registry._tasks == {}

  asyncio.run(_case())


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


@pytest.mark.parametrize("state", ["starting", "queued", "waiting"])
def test_autonomous_reaper_terminalizes_non_running_active_states(monkeypatch, tmp_path, state: str) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    processes: list[SlowFakeProcess] = []

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      process = SlowFakeProcess()
      processes.append(process)
      return process

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
    record = registry._tasks[payload["task_id"]]
    record.state = state
    processes[0].returncode = 0

    await registry.wait(payload["task_id"], timeout_sec=1)

    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "completed"
    assert manifest["exit_code"] == 0
    assert isinstance(manifest["completed_at"], float)

  asyncio.run(_case())


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


def test_autonomous_post_spawn_manifest_commit_failure_aborts_unowned_child(
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
      with pytest.raises(RuntimeError, match="commit running autonomous task manifest"):
        await registry.start(
          profile="analyst",
          mode="task",
          task="summarize",
          user_id=USER_ID,
          user_email=USER_EMAIL,
        )

    assert registry._tasks == {}
    assert registry._reserved_slots == 0
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
  assert env["AUTONOMOUS_RAW_USER_ID"] == USER_ID
  assert env["AUTONOMOUS_USER_SLUG"] == ""
  assert env["AUTONOMOUS_USER_EMAIL"] == USER_EMAIL
  assert env["AUTONOMOUS_USER_ID"] == env["AGENT_API_CLAIM_USER_ID"]
  assert env["AUTONOMOUS_USER_EMAIL"] == env["AGENT_API_CLAIM_USER_EMAIL"]


def test_autonomous_start_resolves_slug_metadata_to_canonical_owner(monkeypatch, tmp_path) -> None:
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([
      {
        "key": "mcp-key-for-henry",
        "channel": "mcp",
        "slug": "henry",
        "email": USER_EMAIL,
        "risk_user_id": 1,
        "role": "owner",
      }
    ]),
  )
  env = asyncio.run(
    _start_and_capture_env(
      monkeypatch,
      tmp_path,
      user_id="henry",
      user_email=USER_EMAIL,
    )
  )

  assert env["AUTONOMOUS_USER_ID"] == "1"
  assert env["AUTONOMOUS_RAW_USER_ID"] == "henry"
  assert env["AUTONOMOUS_USER_SLUG"] == "henry"
  assert env["AGENT_API_CLAIM_USER_ID"] == "1"


def test_autonomous_start_overrides_preexisting_autonomous_identity(monkeypatch, tmp_path) -> None:
  monkeypatch.setenv("AUTONOMOUS_USER_ID", "other")
  monkeypatch.setenv("AUTONOMOUS_USER_EMAIL", "other@example.com")

  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert env["AUTONOMOUS_USER_ID"] == USER_ID
  assert env["AUTONOMOUS_RAW_USER_ID"] == USER_ID
  assert env["AUTONOMOUS_USER_SLUG"] == ""
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
