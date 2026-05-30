from __future__ import annotations

import asyncio
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


def _registry(tmp_path: Path):
  from agent_gateway.autonomous_runner import AutonomousRegistry

  return AutonomousRegistry(
    api_dir=tmp_path,
    python_executable="python3",
    log_dir=tmp_path,
    max_running=1,
  )


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
