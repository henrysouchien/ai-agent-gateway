from __future__ import annotations

import ast
import asyncio
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from agent_gateway.autonomous_capability_handoff import (
  AutonomousCapabilityBinding,
)
from agent_gateway.autonomous_runner import AutonomousRegistry
from agent_gateway.autonomous_runner_state import (
  autonomous_owner_lease_is_released,
)
from agent_gateway.capability_binding import CapabilityBind, CredentialHandle
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.claim_signing_authority import (
  GatewayClaimSigningAuthority,
)
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)


_HMAC_KEY = "autonomous-runner-event-channel-test-key"
_TENANT_ID = "autonomous-event-channel-tests"


@pytest.fixture(autouse=True)
def _gateway_user_keys_for_spawn(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  # Autonomous spawn narrows GATEWAY_USER_KEYS to the admitted user's mcp
  # entry and refuses when none matches; every test here starts as owner-1.
  # risk_user_id deliberately omitted: canonical identity resolution skips
  # the entry (owner stays "owner-1"), while the spawn matcher admits it.
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([
      {
        "key": "event-channel-test-mcp-key",
        "channel": "mcp",
        "slug": "owner-1",
        "email": "owner@example.test",
        "role": "owner",
      }
    ]),
  )
  session_log_base = tmp_path / "session-logs"
  session_log_base.mkdir(mode=0o700)
  monkeypatch.setenv(
    "AGENT_SESSION_LOG_BASE_DIR",
    str(session_log_base),
  )
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Spawn-time GATEWAY_USER_KEYS narrowing imports user_identity from the
# registry's api_dir (origin-validated), so the harness registry must point at
# the real api tree — fake children only write to explicit argv paths.
_REPO_API_DIR = Path(__file__).resolve().parents[3] / "api"
_CHILD_IMPORT_PREFIX = (
  "import sys;"
  f"sys.path.insert(0, {str(_PACKAGE_ROOT)!r});"
)


def _binding(request) -> AutonomousCapabilityBinding:
  handle = _service_credential_handle()
  entry = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
  return AutonomousCapabilityBinding(
    bind=request.required_bind or CapabilityBind(
      schema_version="1.0",
      capability_id="session.driver",
      model_key=entry.key,
      provider=entry.provider,
      upstream_model=entry.upstream_model,
      adapter=entry.adapter,
      protocol_profile=entry.protocol_profile,
      route=entry.route,
      effort="high",
      credential_principal="service",
      credential_ref=handle.handle_id,
      run_mode=request.run_mode,
      registry_revision=INITIAL_MODEL_REGISTRY.revision,
      policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
      selection_source="capability_default",
    ),
    materialized_credential=MaterializedCredential(
      handle=handle,
      auth_config={
        "provider": handle.provider,
        "auth_mode": "api",
        "api_key": "event-channel-test-secret",
      },
    ),
  )


def _service_credential_handle() -> CredentialHandle:
  return CredentialHandle(
    handle_id="service:autonomous-event-channel-tests:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id=_TENANT_ID,
    actor_id=None,
  )


def _registry(
  tmp_path: Path,
  *,
  child_source: str,
  user_event_bus: Any | None = None,
) -> AutonomousRegistry:
  registry = AutonomousRegistry(
    api_dir=_REPO_API_DIR,
    tenant_id=_TENANT_ID,
    python_executable=sys.executable,
    log_dir=tmp_path,
    max_running=1,
    user_event_bus=user_event_bus,
    service_provider_handles={
      "anthropic": _service_credential_handle(),
    },
    autonomous_capability_binding_resolver=_binding,
    claim_signing_authority=GatewayClaimSigningAuthority(_HMAC_KEY),
  )
  registry._build_cmd = lambda **_kwargs: [  # type: ignore[method-assign]
    sys.executable,
    "-c",
    _CHILD_IMPORT_PREFIX + child_source,
  ]
  return registry


async def _start(registry: AutonomousRegistry) -> dict[str, Any]:
  return await registry.start(
    role="owner",
    profile="analyst",
    mode="task",
    task="event channel integration",
    user_id="owner-1",
    user_email="owner@example.test",
  )


_SUCCESS_CHILD = """
import json,os
from agent_gateway.autonomous_event_channel import (
  AUTONOMOUS_EVENT_CHANNEL_FD_ENV,
  adopt_inherited_autonomous_event_channel,
)
envelope=json.loads(os.environ["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"])
session_log_authority=envelope["workload"]["session_log_authority"]
if session_log_authority["layout"] == "v2":
    for env_name, device_name, inode_name in (
        ("AGENT_SESSION_LOG_ROOT_FD", "root_device", "root_inode"),
        ("AGENT_SESSION_LOG_ACTIVE_FD", "active_device", "active_inode"),
        ("AGENT_SESSION_LOG_META_FD", "meta_device", "meta_inode"),
    ):
        pinned=os.fstat(int(os.environ[env_name]))
        assert (pinned.st_dev, pinned.st_ino) == (
            session_log_authority[device_name],
            session_log_authority[inode_name],
        )
channel=adopt_inherited_autonomous_event_channel(
  int(os.environ[AUTONOMOUS_EVENT_CHANNEL_FD_ENV]),
  channel_id=envelope["channel_id"],
)
channel.start(timeout_seconds=20)
channel.send_event({"type":"text_delta","text":"hello"}, timeout_seconds=20)
channel.complete(
  {"type":"stream_complete","terminal_disposition":"completed"},
  timeout_seconds=20,
)
"""


_SESSION_RECAP_CHILD = """
import json,os
from agent_gateway.autonomous_event_channel import (
  AUTONOMOUS_EVENT_CHANNEL_FD_ENV,
  adopt_inherited_autonomous_event_channel,
)
from agent_gateway.session_recap import emit_recap_then_terminal
envelope=json.loads(os.environ["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"])
event_fd=int(os.environ[AUTONOMOUS_EVENT_CHANNEL_FD_ENV])
from agent.autonomous.child_event_delivery import AutonomousChannelEventOwner
channel=adopt_inherited_autonomous_event_channel(
  event_fd,
  channel_id=envelope["channel_id"],
)
channel.start(timeout_seconds=20)
owner=AutonomousChannelEventOwner(channel)
event_log=owner.event_log(session_id="captured-recap-run")
event_log.append({"type":"turn_complete","turn":1,"usage":{}})
emit_recap_then_terminal(
  event_log,
  {
    "type":"stream_complete",
    "terminal_disposition":"completed",
    "usage":{},
  },
  session_id="captured-recap-run",
  started_at=1.0,
)
owner.complete(0)
"""


_BUDGET_LIMITED_CHILD = """
import json,os,sys
from agent_gateway.autonomous_event_channel import (
  AUTONOMOUS_EVENT_CHANNEL_FD_ENV,
  adopt_inherited_autonomous_event_channel,
)
from agent_gateway.session_recap import emit_recap_then_terminal
envelope=json.loads(os.environ["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"])
event_fd=int(os.environ[AUTONOMOUS_EVENT_CHANNEL_FD_ENV])
from agent.autonomous.child_event_delivery import AutonomousChannelEventOwner
channel=adopt_inherited_autonomous_event_channel(
  event_fd,
  channel_id=envelope["channel_id"],
)
channel.start(timeout_seconds=20)
owner=AutonomousChannelEventOwner(channel)
event_log=owner.event_log(session_id="captured-budget-run")
event_log.append({
  "type":"budget_exceeded",
  "total_cost":5.5,
  "budget":5.0,
})
emit_recap_then_terminal(
  event_log,
  {
    "type":"stream_complete",
    "terminal_disposition":"interrupted",
    "reason":"budget_exceeded",
    "usage":{},
  },
  session_id="captured-budget-run",
  started_at=1.0,
)
owner.complete(2)
raise SystemExit(2)
"""


def test_autonomous_start_uses_private_event_lifeline_and_lease_fds(
  monkeypatch,
  tmp_path: Path,
) -> None:
  from agent_gateway import autonomous_runner

  async def case() -> None:
    captured: dict[str, Any] = {}
    real_spawn = autonomous_runner.asyncio.create_subprocess_exec

    async def recording_spawn(*args, **kwargs):
      captured.update(kwargs)
      captured["passed_fd_stats"] = {
        fd: os.fstat(fd)
        for fd in kwargs["pass_fds"]
      }
      captured["passed_fd_inheritable"] = {
        fd: os.get_inheritable(fd)
        for fd in kwargs["pass_fds"]
      }
      return await real_spawn(*args, **kwargs)

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      recording_spawn,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _HMAC_KEY)
    monkeypatch.setenv("AGENT_SESSION_LOG_LAYOUT", "v2")
    registry = _registry(tmp_path, child_source=_SUCCESS_CHILD)

    payload = await _start(registry)
    status = await registry.wait(payload["task_id"], timeout_sec=20)
    record = registry._tasks[payload["task_id"]]

    assert status["state"] == "completed"
    assert captured["start_new_session"] is True
    assert len(captured["pass_fds"]) == 7
    passed_fd = int(
      captured["env"]["AGENT_AUTONOMOUS_EVENT_CHANNEL_FD"]
    )
    assert passed_fd in captured["pass_fds"]
    assert captured["env"]["AGENT_AUTONOMOUS_EVENT_CHANNEL_FD"] == str(
      passed_fd
    )
    envelope = json.loads(
      captured["env"]["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"]
    )
    log_authority = envelope["workload"]["session_log_authority"]
    assert envelope["version"] == 5
    assert log_authority["layout"] == "v2"
    assert log_authority["base_path"] == captured["env"][
      "AGENT_SESSION_LOG_BASE_DIR"
    ]
    assert tuple(
      int(captured["env"][name])
      for name in (
        "AGENT_SESSION_LOG_ROOT_FD",
        "AGENT_SESSION_LOG_ACTIVE_FD",
        "AGENT_SESSION_LOG_META_FD",
      )
    ) == captured["pass_fds"][2:5]
    assert all(
      inheritable is False
      for inheritable in captured["passed_fd_inheritable"].values()
    )
    assert "AGENT_AUTONOMOUS_EVENTS_PATH" not in captured["env"]
    assert record.process_group_pid == record.proc.pid
    assert record.process_group_id == record.proc.pid
    assert [
      event["type"]
      for event in record.event_channel_projected_events
    ] == ["text_delta", "stream_complete"]
    assert len(record.event_channel_records) == 2
    assert record.event_channel_stream is not None
    assert all(
      stream_record is delivered_record
      for stream_record, delivered_record in zip(
        record.event_channel_stream.records,
        record.event_channel_records,
        strict=True,
      )
    )
    assert record.event_channel_acknowledgement is not None
    assert (
      record.event_channel_acknowledgement.event_digest
      == record.event_channel_stream.event_digest
    )
    assert record.event_channel is not None
    assert record.event_channel.fileno() == -1
    events_path = tmp_path / f"{payload['task_id']}.events.jsonl"
    assert events_path.exists()
    assert [
      json.loads(line)
      for line in events_path.read_text(encoding="utf-8").splitlines()
    ] == record.event_lines
    manifest = json.loads(
      (tmp_path / f"{payload['task_id']}.task.json").read_text(
        encoding="utf-8"
      )
    )
    assert manifest["events_path"] == str(events_path)
    lease_identity = captured["passed_fd_stats"][
      captured["pass_fds"][6]
    ]
    assert manifest["owner_lease_device"] == lease_identity.st_dev
    assert manifest["owner_lease_inode"] == lease_identity.st_ino
    assert record.owner_lifeline_fd is None

  asyncio.run(case())


def test_captured_run_recap_emits_end_and_settles_completed(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    registry = _registry(tmp_path, child_source=_SESSION_RECAP_CHILD)

    payload = await _start(registry)
    status = await registry.wait(payload["task_id"], timeout_sec=20)
    record = registry._tasks[payload["task_id"]]

    assert status["state"] == "completed"
    assert status.get("error") is None
    assert record.event_channel_stream is not None
    assert record.event_channel_acknowledgement is not None
    assert [
      event["type"]
      for event in record.event_channel_projected_events
    ] == ["turn_complete", "session_recap", "stream_complete"]
    recap = record.event_channel_projected_events[1]
    assert recap["seq_range"] == [1, 1]
    manifest = json.loads(
      (tmp_path / f"{payload['task_id']}.task.json").read_text(
        encoding="utf-8"
      )
    )
    assert manifest["state"] == "completed"
    assert manifest.get("error") is None

  asyncio.run(case())


def test_budget_stop_emits_end_and_settles_budget_limited(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    registry = _registry(
      tmp_path,
      child_source=_BUDGET_LIMITED_CHILD,
    )

    payload = await _start(registry)
    status = await registry.wait(payload["task_id"], timeout_sec=20)
    record = registry._tasks[payload["task_id"]]

    assert status["state"] == "budget_limited"
    assert status.get("error") is None
    assert record.exit_code == 2
    assert record.event_channel_stream is not None
    assert record.event_channel_acknowledgement is not None
    assert [
      event["type"]
      for event in record.event_channel_projected_events
    ] == ["budget_exceeded", "session_recap", "stream_complete"]
    assert record.event_channel_projected_events[-1] == {
      "type": "stream_complete",
      "terminal_disposition": "interrupted",
      "reason": "budget_exceeded",
      "usage": {},
      "terminal_reason": None,
      "run_id": payload["task_id"],
      "control_run_id": payload["task_id"],
    }
    manifest = json.loads(
      (tmp_path / f"{payload['task_id']}.task.json").read_text(
        encoding="utf-8"
      )
    )
    assert manifest["state"] == "budget_limited"
    assert manifest["exit_code"] == 2
    assert manifest.get("error") is None

  asyncio.run(case())


def test_owner_sigkill_reaps_target_grandchild_before_releasing_lease(
  tmp_path: Path,
) -> None:
  from agent_gateway.autonomous_runner_start import (
    _OWNED_PROCESS_SENTINEL_SOURCE,
    _create_owner_lifeline,
  )

  report_path = tmp_path / "owned-processes.json"
  lease_path = tmp_path / "owner.lease"
  lease_fd = os.open(
    lease_path,
    os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
    0o600,
  )
  os.fchmod(lease_fd, 0o600)
  lease_stat = os.fstat(lease_fd)
  fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  owner_read_fd, owner_write_fd = _create_owner_lifeline()
  owner_read_stat = os.fstat(owner_read_fd)
  event_read_fd, event_write_fd = os.pipe()
  claim_broker_read_fd, claim_broker_write_fd = os.pipe()
  owner_process = subprocess.Popen(
    [
      sys.executable,
      "-c",
      "import signal; signal.pause()",
    ],
    close_fds=True,
    pass_fds=(owner_write_fd,),
  )
  target_source = r"""
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

report_path = pathlib.Path(sys.argv[1])
lifeline_fd = int(sys.argv[2])
lifeline_device = int(sys.argv[3])
lifeline_inode = int(sys.argv[4])
lease_fd = int(sys.argv[5])
lease_device = int(sys.argv[6])
lease_inode = int(sys.argv[7])


def inherited_exact_identity(fd, device, inode):
    try:
        info = os.fstat(fd)
    except OSError:
        return False
    return (info.st_dev, info.st_ino) == (device, inode)


signal.signal(signal.SIGTERM, signal.SIG_IGN)
grandchild = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal,time;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "time.sleep(60)",
    ],
    close_fds=True,
)
report_path.write_text(
    json.dumps({
        "target_pid": os.getpid(),
        "grandchild_pid": grandchild.pid,
        "lifeline_inherited": inherited_exact_identity(
            lifeline_fd,
            lifeline_device,
            lifeline_inode,
        ),
        "lease_inherited": inherited_exact_identity(
            lease_fd,
            lease_device,
            lease_inode,
        ),
    }),
    encoding="utf-8",
)
time.sleep(60)
"""
  env = dict(os.environ)
  env.pop("AGENT_AUTONOMOUS_APPROVAL_CHANNEL_FD", None)
  env["AGENT_AUTONOMOUS_EVENT_CHANNEL_FD"] = str(event_write_fd)
  env["AGENT_AUTONOMOUS_CLAIM_BROKER_FD"] = str(
    claim_broker_read_fd
  )
  env["AGENT_AUTONOMOUS_CREDENTIAL_HANDOFF"] = "stdin-json-v1"
  sentinel = subprocess.Popen(
    [
      sys.executable,
      "-c",
      _OWNED_PROCESS_SENTINEL_SOURCE,
      "1024",
      str(owner_read_fd),
      str(lease_fd),
      "1.0",
      sys.executable,
      "-c",
      target_source,
      str(report_path),
      str(owner_read_fd),
      str(owner_read_stat.st_dev),
      str(owner_read_stat.st_ino),
      str(lease_fd),
      str(lease_stat.st_dev),
      str(lease_stat.st_ino),
    ],
    close_fds=True,
    env=env,
    pass_fds=(
      event_write_fd,
      claim_broker_read_fd,
      owner_read_fd,
      lease_fd,
    ),
    start_new_session=True,
    stdin=subprocess.PIPE,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
  )
  assert sentinel.stdin is not None
  sentinel.stdin.write(b'{"auth_config":{}}\n')
  sentinel.stdin.close()
  os.close(event_write_fd)
  event_write_fd = -1
  os.close(claim_broker_read_fd)
  claim_broker_read_fd = -1
  os.close(claim_broker_write_fd)
  claim_broker_write_fd = -1
  os.close(owner_read_fd)
  owner_read_fd = -1
  os.close(lease_fd)
  lease_fd = -1
  os.close(owner_write_fd)
  owner_write_fd = -1
  try:
    deadline = time.monotonic() + 10
    while not report_path.exists() and time.monotonic() < deadline:
      time.sleep(0.01)
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["lifeline_inherited"] is False
    assert report["lease_inherited"] is False
    assert autonomous_owner_lease_is_released(
      lease_path,
      expected_device=lease_stat.st_dev,
      expected_inode=lease_stat.st_ino,
    ) is False

    owner_process.kill()
    assert owner_process.wait(timeout=10) == -signal.SIGKILL
    assert sentinel.poll() is None
    assert autonomous_owner_lease_is_released(
      lease_path,
      expected_device=lease_stat.st_dev,
      expected_inode=lease_stat.st_ino,
    ) is False
    assert sentinel.wait(timeout=10) == -signal.SIGKILL
    assert autonomous_owner_lease_is_released(
      lease_path,
      expected_device=lease_stat.st_dev,
      expected_inode=lease_stat.st_ino,
    ) is True

    for pid_field in ("target_pid", "grandchild_pid"):
      pid = int(report[pid_field])
      deadline = time.monotonic() + 10
      while time.monotonic() < deadline:
        try:
          os.kill(pid, 0)
        except ProcessLookupError:
          break
        subprocess.run(
          ["/usr/bin/true"],
          check=True,
          close_fds=True,
        )
        time.sleep(0.01)
      else:
        pytest.fail(f"{pid_field} survived owner-lifeline cleanup")
  finally:
    for fd in (
      event_read_fd,
      event_write_fd,
      claim_broker_read_fd,
      claim_broker_write_fd,
      owner_read_fd,
      owner_write_fd,
      lease_fd,
    ):
      if fd >= 0:
        try:
          os.close(fd)
        except OSError:
          pass
    if sentinel.poll() is None:
      os.killpg(sentinel.pid, signal.SIGKILL)
      sentinel.wait(timeout=10)
    if owner_process.poll() is None:
      owner_process.kill()
      owner_process.wait(timeout=10)


class _BlockingTerminalBus:
  def __init__(self, *, fail_terminal: bool = False) -> None:
    self.fail_terminal = fail_terminal
    self.terminal_started = asyncio.Event()
    self.release_terminal = asyncio.Event()
    self.events: list[dict[str, Any]] = []

  async def seed_replay_buffer(
    self,
    _user_id: str,
    _control_run_id: str,
    _events: list[dict[str, Any]],
    *,
    terminated: bool = False,
  ) -> int:
    _ = terminated
    return 0

  async def publish(
    self,
    *,
    user_id: str,
    control_run_id: str,
    event: dict[str, Any],
  ) -> None:
    _ = user_id, control_run_id
    self.events.append(dict(event))
    if event.get("type") != "stream_complete":
      return
    self.terminal_started.set()
    if self.fail_terminal:
      raise RuntimeError("durable delivery refused")
    await self.release_terminal.wait()

  async def cleanup_run(self, _user_id: str, _control_run_id: str) -> None:
    return None


def test_terminal_is_delivered_before_ack(monkeypatch, tmp_path: Path) -> None:
  async def case() -> None:
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _HMAC_KEY)
    bus = _BlockingTerminalBus()
    registry = _registry(
      tmp_path,
      child_source=_SUCCESS_CHILD,
      user_event_bus=bus,
    )

    payload = await _start(registry)
    record = registry._tasks[payload["task_id"]]
    await asyncio.wait_for(bus.terminal_started.wait(), timeout=10)

    assert record.event_channel_acknowledgement is None
    assert record.proc.returncode is None

    bus.release_terminal.set()
    status = await registry.wait(payload["task_id"], timeout_sec=20)

    assert status["state"] == "completed"
    assert record.event_channel_acknowledgement is not None

  asyncio.run(case())


def test_publish_failure_never_acknowledges_child(
  monkeypatch,
  tmp_path: Path,
) -> None:
  async def case() -> None:
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _HMAC_KEY)
    bus = _BlockingTerminalBus(fail_terminal=True)
    registry = _registry(
      tmp_path,
      child_source=_SUCCESS_CHILD,
      user_event_bus=bus,
    )

    payload = await _start(registry)
    status = await registry.wait(payload["task_id"], timeout_sec=20)
    record = registry._tasks[payload["task_id"]]

    assert status["state"] == "failed"
    assert "event channel failed" in (status["error"] or "")
    assert record.event_channel_acknowledgement is None
    assert record.event_channel_stream is None
    assert len(record.event_channel_records) == 2
    assert [
      event["type"]
      for event in record.event_channel_projected_events
    ] == ["text_delta"]

  asyncio.run(case())


def _invalid_child(mode: str) -> str:
  return f"""
import json,os,socket,struct
from agent_gateway.autonomous_event_channel import AUTONOMOUS_EVENT_CHANNEL_FD_ENV
envelope=json.loads(os.environ["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"])
channel_id=envelope["channel_id"]
sock=socket.socket(fileno=int(os.environ[AUTONOMOUS_EVENT_CHANNEL_FD_ENV]))
os.set_inheritable(sock.fileno(),False)
def frame(value):
  body=json.dumps(value,sort_keys=True,separators=(",",":")).encode()
  return struct.pack(">I",len(body))+body
sock.sendall(frame({{"kind":"HELLO","version":1,"channel_id":channel_id}}))
mode={mode!r}
if mode=="malformed":
  body=b"{{not-json"
  sock.sendall(struct.pack(">I",len(body))+body)
elif mode=="gap":
  sock.sendall(frame({{
    "kind":"EVENT","version":1,"channel_id":channel_id,"seq":1,
    "event":{{"type":"text_delta","text":"gap"}},
  }}))
else:
  sock.sendall(frame({{
    "kind":"EVENT","version":1,"channel_id":channel_id,"seq":0,
    "event":{{"type":"text_delta","text":"no terminal"}},
  }}))
sock.shutdown(socket.SHUT_WR)
try:
  sock.recv(1)
except OSError:
  pass
"""


@pytest.mark.parametrize("mode", ["malformed", "gap", "terminal_loss"])
def test_invalid_stream_never_receives_ack(
  monkeypatch,
  tmp_path: Path,
  mode: str,
) -> None:
  async def case() -> None:
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _HMAC_KEY)
    registry = _registry(tmp_path, child_source=_invalid_child(mode))

    payload = await _start(registry)
    status = await registry.wait(payload["task_id"], timeout_sec=20)
    record = registry._tasks[payload["task_id"]]

    assert status["state"] == "failed"
    assert record.event_channel_acknowledgement is None
    assert record.event_channel_stream is None

  asyncio.run(case())


class _ReadyBus:
  def __init__(self) -> None:
    self.ready = asyncio.Event()

  async def seed_replay_buffer(
    self,
    _user_id: str,
    _control_run_id: str,
    _events: list[dict[str, Any]],
    *,
    terminated: bool = False,
  ) -> int:
    _ = terminated
    return 0

  async def publish(
    self,
    *,
    user_id: str,
    control_run_id: str,
    event: dict[str, Any],
  ) -> None:
    _ = user_id, control_run_id
    if event.get("type") == "ready":
      self.ready.set()

  async def cleanup_run(self, _user_id: str, _control_run_id: str) -> None:
    return None


def test_cancel_kills_owned_child_and_grandchild(
  monkeypatch,
  tmp_path: Path,
) -> None:
  async def case() -> None:
    pid_path = tmp_path / "grandchild.pid"
    child_source = f"""
import json,os,pathlib,signal,subprocess,sys
from agent_gateway.autonomous_event_channel import (
  AUTONOMOUS_EVENT_CHANNEL_FD_ENV,
  adopt_inherited_autonomous_event_channel,
)
envelope=json.loads(os.environ["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"])
channel=adopt_inherited_autonomous_event_channel(
  int(os.environ[AUTONOMOUS_EVENT_CHANNEL_FD_ENV]),
  channel_id=envelope["channel_id"],
)
channel.start(timeout_seconds=20)
grandchild=subprocess.Popen(
  [sys.executable,"-c","import signal; signal.pause()"],
  close_fds=True,
  stdout=subprocess.DEVNULL,
  stderr=subprocess.DEVNULL,
)
pathlib.Path({str(pid_path)!r}).write_text(str(grandchild.pid),encoding="utf-8")
channel.send_event({{"type":"ready"}}, timeout_seconds=20)
signal.pause()
"""
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _HMAC_KEY)
    bus = _ReadyBus()
    registry = _registry(
      tmp_path,
      child_source=child_source,
      user_event_bus=bus,
    )

    payload = await _start(registry)
    await asyncio.wait_for(bus.ready.wait(), timeout=10)
    grandchild_pid = int(pid_path.read_text(encoding="utf-8"))

    status = await registry.cancel(payload["task_id"])
    record = registry._tasks[payload["task_id"]]

    assert status["state"] == "killed"
    assert record.proc.returncode is not None
    assert record.slot_reserved is False
    assert registry._reserved_slots == 0
    assert record.event_channel_acknowledgement is None
    with pytest.raises(ProcessLookupError):
      os.kill(grandchild_pid, 0)

  asyncio.run(case())


def test_sentinel_pins_group_through_child_exit_pid_churn_and_final_cleanup(
  monkeypatch,
  tmp_path: Path,
) -> None:
  async def case() -> None:
    pid_path = tmp_path / "orphaned-grandchild.pid"
    actual_exit_path = tmp_path / "actual-child-exited"
    child_source = f"""
import json,os,pathlib,subprocess,sys
from agent_gateway.autonomous_event_channel import (
  AUTONOMOUS_EVENT_CHANNEL_FD_ENV,
  adopt_inherited_autonomous_event_channel,
)
envelope=json.loads(os.environ["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"])
channel=adopt_inherited_autonomous_event_channel(
  int(os.environ[AUTONOMOUS_EVENT_CHANNEL_FD_ENV]),
  channel_id=envelope["channel_id"],
)
channel.start(timeout_seconds=20)
grandchild=subprocess.Popen(
  [sys.executable,"-c","import signal; signal.pause()"],
  close_fds=True,
  stdout=subprocess.DEVNULL,
  stderr=subprocess.DEVNULL,
)
pathlib.Path({str(pid_path)!r}).write_text(str(grandchild.pid),encoding="utf-8")
channel.complete(
  {{"type":"stream_complete","terminal_disposition":"completed"}},
  timeout_seconds=20,
)
pathlib.Path({str(actual_exit_path)!r}).write_text("done",encoding="utf-8")
"""
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _HMAC_KEY)
    registry = _registry(tmp_path, child_source=child_source)
    signal_observations: list[tuple[int, int | None]] = []

    def tracked_group_signal(process_group_id: int, signal_number: int) -> None:
      record = registry._tasks["bg_0"]
      signal_observations.append((signal_number, record.proc.returncode))
      assert record.proc.returncode is None
      if signal_number == signal.SIGKILL:
        assert actual_exit_path.read_text(encoding="utf-8") == "done"
        for _ in range(128):
          subprocess.run(
            ["/usr/bin/true"],
            check=True,
            close_fds=True,
          )
        assert record.proc.returncode is None
      os.killpg(process_group_id, signal_number)

    monkeypatch.setattr(
      "agent_gateway.autonomous_runner._signal_process_group",
      tracked_group_signal,
    )

    payload = await _start(registry)
    status = await registry.wait(payload["task_id"], timeout_sec=20)
    record = registry._tasks[payload["task_id"]]
    grandchild_pid = int(pid_path.read_text(encoding="utf-8"))

    assert signal_observations == [(signal.SIGKILL, None)]
    assert record.proc.returncode == -signal.SIGKILL
    assert record.exit_code == 0
    assert status["state"] == "completed"
    assert record.event_channel_acknowledgement is not None
    assert record.slot_reserved is False
    assert registry._reserved_slots == 0
    # Generous bound: real-process teardown under a CPU-saturated shard can
    # exceed short fixed windows; the loop exits early on success.
    for _ in range(6000):
      try:
        os.kill(grandchild_pid, 0)
      except ProcessLookupError:
        break
      await asyncio.sleep(0.01)
    else:
      pytest.fail("owned grandchild survived final sentinel group cleanup")

  asyncio.run(case())


def test_cancel_after_ack_commit_waits_for_completed_process(
  monkeypatch,
  tmp_path: Path,
) -> None:
  async def case() -> None:
    child_source = """
import json,os,time
from agent_gateway.autonomous_event_channel import (
  AUTONOMOUS_EVENT_CHANNEL_FD_ENV,
  adopt_inherited_autonomous_event_channel,
)
envelope=json.loads(os.environ["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"])
channel=adopt_inherited_autonomous_event_channel(
  int(os.environ[AUTONOMOUS_EVENT_CHANNEL_FD_ENV]),
  channel_id=envelope["channel_id"],
)
channel.start(timeout_seconds=20)
channel.complete(
  {"type":"stream_complete","terminal_disposition":"completed"},
  timeout_seconds=20,
)
time.sleep(1.5)
"""
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", _HMAC_KEY)
    registry = _registry(tmp_path, child_source=child_source)

    payload = await _start(registry)
    record = registry._tasks[payload["task_id"]]
    # Generous bound: real child-interpreter startup under a CPU-saturated
    # shard can exceed 2s; the loop exits early on acknowledgement.
    for _ in range(6000):
      if record.event_channel_acknowledgement is not None:
        break
      await asyncio.sleep(0.01)
    else:
      pytest.fail("child stream was not acknowledged")

    status = await registry.cancel(payload["task_id"])

    assert status["state"] == "completed"
    assert record.cancellation_requested is False
    assert record.proc.returncode == -signal.SIGKILL
    assert record.exit_code == 0
    assert record.slot_reserved is False

  asyncio.run(case())


def test_event_file_polling_surface_is_structurally_absent() -> None:
  package_dir = _PACKAGE_ROOT / "agent_gateway"
  sources = {
    filename: (package_dir / filename).read_text(encoding="utf-8")
    for filename in (
      "autonomous_runner.py",
      "autonomous_runner_start.py",
      "autonomous_runner_state.py",
      "autonomous_runner_events.py",
    )
  }
  source = "\n".join(sources.values())

  for retired_token in (
    "_tail_events_file",
    "_finish_events_tail",
    "events_tail_task",
    "asyncio.sleep(",
  ):
    assert retired_token not in source
  start_source = sources["autonomous_runner_start.py"]
  assert start_source.count("AGENT_AUTONOMOUS_EVENTS_PATH") == 1
  assert (
    "for retired_name in _RETIRED_AUTONOMOUS_CHILD_ENV_NAMES:"
    in start_source
  )
  assert "env.pop(retired_name, None)" in start_source
  assert "owner_lifeline_read_fd" in source
  assert "owner_lease_fd" in source
  assert "pass_fds=(" in source
  assert "start_new_session=True" in source
  assert "os.killpg(" in source
  assert "endpoint.receive_next" in source
  assert "endpoint.acknowledge" in source


def test_subprocess_drain_is_unbounded_after_hello_and_interruptible() -> None:
  package_dir = _PACKAGE_ROOT / "agent_gateway"
  runner_source = (package_dir / "autonomous_runner.py").read_text(
    encoding="utf-8"
  )
  runner_tree = ast.parse(runner_source)
  drain = next(
    node
    for node in ast.walk(runner_tree)
    if isinstance(node, ast.AsyncFunctionDef)
    and node.name == "_drain_event_channel"
  )
  to_thread_calls = [
    node
    for node in ast.walk(drain)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "to_thread"
  ]
  receive_calls = [
    node
    for node in to_thread_calls
    if node.args
    and isinstance(node.args[0], ast.Attribute)
    and node.args[0].attr == "receive_next"
  ]
  acknowledge_calls = [
    node
    for node in to_thread_calls
    if node.args
    and isinstance(node.args[0], ast.Attribute)
    and node.args[0].attr == "acknowledge"
  ]

  assert len(receive_calls) == 2
  first_receive_keywords = {
    keyword.arg: ast.literal_eval(keyword.value)
    for keyword in receive_calls[0].keywords
  }
  assert first_receive_keywords == {"unbounded_stream": True}
  assert receive_calls[1].keywords == []
  assert len(acknowledge_calls) == 1
  assert "_AUTONOMOUS_RUN_MAX_" + "SECONDS" not in runner_source
  assert "return_when=asyncio.FIRST_COMPLETED" in runner_source
  assert "interrupt_event_channel()" in runner_source

  protocol_source = (
    package_dir / "autonomous_event_channel.py"
  ).read_text(encoding="utf-8")
  protocol_tree = ast.parse(protocol_source)
  parent_class = next(
    node
    for node in protocol_tree.body
    if isinstance(node, ast.ClassDef)
    and node.name == "AutonomousEventChannelParent"
  )
  methods = {
    node.name: ast.get_source_segment(protocol_source, node) or ""
    for node in parent_class.body
    if isinstance(node, ast.FunctionDef)
  }

  assert "deadline, unbounded = self._receive_mode()" in methods["_advance_receive"]
  assert "unbounded_stream=" in methods["receive_next"]
  assert "self._advance_receive()" in methods["receive_next"]
  assert (
    "time.monotonic() + self._timeout(timeout_seconds)"
    in methods["acknowledge"]
  )

  start_source = (package_dir / "autonomous_runner_start.py").read_text(
    encoding="utf-8"
  )
  assert (
    "io_timeout_seconds=AUTONOMOUS_EVENT_CHANNEL_MAX_IO_TIMEOUT_SECONDS"
    in start_source
  )
