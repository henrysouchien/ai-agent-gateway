from __future__ import annotations

import ast
import asyncio
from contextlib import nullcontext
import fcntl
import hashlib
import json
import logging
import os
import signal
import stat
import subprocess
import sys
import threading
from itertools import count
from pathlib import Path
from types import SimpleNamespace

import pytest

from .control_plane.manifest_helpers import write_v6_manifest

# The agent-claim verifier (settings + utils.agent_claim) lives in the separate
# risk_module repo. The root conftest resolves RISK_MODULE_CHECKOUT first and
# retains sibling inference for local checkouts. When risk_module is unavailable,
# the cross-repo contract tests below skip; gateway-side tests still run.
RISK_MODULE_ROOT = Path(os.environ["RISK_MODULE_CHECKOUT"]).expanduser().resolve() \
  if os.environ.get("RISK_MODULE_CHECKOUT") else Path(__file__).resolve().parents[3].parent / "risk_module"
_risk_root_text = str(RISK_MODULE_ROOT)
_risk_root_original_index = (
  sys.path.index(_risk_root_text) if _risk_root_text in sys.path else None
)
if _risk_root_original_index is not None:
  sys.path.pop(_risk_root_original_index)
if RISK_MODULE_ROOT.is_dir():
  sys.path.insert(0, _risk_root_text)
_prior_config_module = sys.modules.pop("config", None)

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
finally:
  if sys.path and sys.path[0] == _risk_root_text:
    sys.path.pop(0)
  if _risk_root_original_index is not None:
    sys.path.insert(_risk_root_original_index, _risk_root_text)
  sys.modules.pop("config", None)
  if _prior_config_module is not None:
    sys.modules["config"] = _prior_config_module

_requires_risk_module = pytest.mark.skipif(
  not _RISK_MODULE_AVAILABLE,
  reason=(
    "risk_module verifier (settings.AGENT_API_CLAIM_MAX_TTL_SECONDS + "
    "utils.agent_claim) not importable; set RISK_MODULE_CHECKOUT to run the "
    "cross-repo agent-claim contract tests"
  ),
)


HMAC_KEY = "test-hmac-key-at-least-32-bytes-long"
USER_ID = "1"
USER_EMAIL = "hc@henrychien.com"
TENANT_ID = "autonomous-runner-tests"
API_DIR = Path(__file__).resolve().parents[3] / "api"
CLAIM_ENV_KEYS = {
  "AGENT_API_CLAIM_AUDIENCE",
  "AGENT_API_CLAIM_ISSUED_AT",
  "AGENT_API_CLAIM_EXPIRY",
  "AGENT_API_CLAIM_USER_ID",
  "AGENT_API_CLAIM_USER_EMAIL",
  "AGENT_API_CLAIM_NONCE",
  "AGENT_API_CLAIM_SIGNATURE",
}


def _test_capability_bind(run_mode: str):
  from agent_gateway.capability_binding import CapabilityBind
  from agent_gateway.model_registry import (
    INITIAL_MODEL_REGISTRY,
    INITIAL_MODEL_SELECTION_POLICY,
  )

  entry = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
  handle = _test_service_credential_handle()

  return CapabilityBind(
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
    run_mode=run_mode,
    registry_revision=INITIAL_MODEL_REGISTRY.revision,
    policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
    selection_source="capability_default",
  )


def _test_service_credential_handle():
  from agent_gateway.capability_binding import CredentialHandle

  return CredentialHandle(
    handle_id="service:autonomous-runner-tests:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id=TENANT_ID,
    actor_id=None,
  )


def _test_autonomous_capability_binding(request):
  from agent_gateway.autonomous_capability_handoff import (
    AutonomousCapabilityBinding,
  )
  from agent_gateway.capability_execution import (
    MaterializedCredential,
  )

  bind = request.required_bind or _test_capability_bind(request.run_mode)
  handle = _test_service_credential_handle()
  return AutonomousCapabilityBinding(
    bind=bind,
    materialized_credential=MaterializedCredential(
      handle=handle,
      auth_config={
        "provider": handle.provider,
        "auth_mode": "api",
        "api_key": "test-service-secret",
      },
    ),
  )


_FAKE_PROCESS_PIDS = count(90_000)
_FAKE_PROCESSES: dict[int, "_FakeProcessIdentity"] = {}
_LAST_FAKE_EVENT_CHANNEL = None


class _FakeStdin:
  def __init__(self) -> None:
    self.buffer = bytearray()
    self.closed = False

  def write(self, payload: bytes) -> None:
    self.buffer.extend(payload)

  async def drain(self) -> None:
    return None

  def close(self) -> None:
    self.closed = True

  async def wait_closed(self) -> None:
    return None


class _FakeProcessIdentity:
  def __init__(self) -> None:
    self.pid = next(_FAKE_PROCESS_PIDS)
    self._returncode: int | None = None
    self.stdin = _FakeStdin()
    _FAKE_PROCESSES[self.pid] = self
    if _LAST_FAKE_EVENT_CHANNEL is not None:
      _LAST_FAKE_EVENT_CHANNEL.attach_process(self)

  @property
  def returncode(self) -> int | None:
    return self._returncode

  @returncode.setter
  def returncode(self, value: int | None) -> None:
    self._returncode = value
    if _LAST_FAKE_EVENT_CHANNEL is not None:
      _LAST_FAKE_EVENT_CHANNEL.notify_process_exit(self, value)


class FakeProcess(_FakeProcessIdentity):

  async def wait(self) -> int:
    self.returncode = 0
    return 0

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class SlowFakeProcess(_FakeProcessIdentity):

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class FailingFakeProcess(_FakeProcessIdentity):

  async def wait(self) -> int:
    self.returncode = 1
    return 1

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class RaisingFakeProcess(_FakeProcessIdentity):

  async def wait(self) -> int:
    raise RuntimeError("reap boom")

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class _FakeEventChannelChild:
  def __init__(self) -> None:
    self._read_fd, self._write_fd = os.pipe()

  def fileno(self) -> int:
    return self._write_fd

  def close(self) -> None:
    if self._write_fd < 0:
      return
    os.close(self._write_fd)
    self._write_fd = -1

  def close_all(self) -> None:
    self.close()
    if self._read_fd >= 0:
      os.close(self._read_fd)
      self._read_fd = -1


class _FakeEventChannelParent:
  channel_id = "a" * 64

  def __init__(self, child: _FakeEventChannelChild) -> None:
    self._child = child
    self._process: _FakeProcessIdentity | None = None
    self._exited = threading.Event()
    self._closed = threading.Event()
    self._record = None
    self._stream = None
    self._delivered_record = False

  def attach_process(self, process: _FakeProcessIdentity) -> None:
    self._process = process

  def notify_process_exit(
    self,
    process: _FakeProcessIdentity,
    returncode: int | None,
  ) -> None:
    if process is self._process and returncode is not None:
      self._exited.set()

  def receive_next(
    self,
    *,
    timeout_seconds: float | None = None,
    unbounded_stream: bool | None = None,
  ):
    from agent_gateway.autonomous_event_channel import (
      AutonomousEventRecord,
      ReceivedAutonomousEventStream,
    )

    _ = timeout_seconds, unbounded_stream
    while not self._exited.wait(0.01):
      if self._closed.is_set():
        raise RuntimeError("fake autonomous event channel closed")
    if not self._delivered_record:
      assert self._process is not None
      if self._process.returncode == 0:
        event = {
          "type": "stream_complete",
          "terminal_disposition": "completed",
        }
      else:
        event = {
          "type": "error",
          "error": f"Process exited with code {self._process.returncode}",
        }
      line = (
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
      )
      self._record = AutonomousEventRecord(
        seq=0,
        frame_bytes=b"fake-frame",
        event_line_bytes=line,
      )
      self._stream = ReceivedAutonomousEventStream(
        channel_id=self.channel_id,
        records=(self._record,),
        event_count=1,
        event_digest=hashlib.sha256(b"fake-frame").hexdigest(),
        exact_event_frame_bytes=len(b"fake-frame"),
      )
      self._delivered_record = True
      return self._record
    return self._stream

  def acknowledge(self, stream, *, timeout_seconds: float | None = None):
    from agent_gateway.autonomous_event_channel import (
      AutonomousEventAcknowledgement,
    )

    _ = timeout_seconds
    assert stream is self._stream
    return AutonomousEventAcknowledgement(
      channel_id=stream.channel_id,
      event_count=stream.event_count,
      event_digest=stream.event_digest,
    )

  def close(self) -> None:
    self._closed.set()
    self._child.close_all()

  def interrupt(self) -> None:
    self.close()


class _FakeEventChannelPair:
  def __init__(self) -> None:
    self.child = _FakeEventChannelChild()
    self.parent = _FakeEventChannelParent(self.child)


@pytest.fixture(autouse=True)
def _fake_subprocess_ownership_and_event_channel(monkeypatch):
  from agent_gateway import autonomous_runner
  from agent_gateway import autonomous_runner_start

  global _LAST_FAKE_EVENT_CHANNEL
  _LAST_FAKE_EVENT_CHANNEL = None
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([{
      "key": "test-mcp-key",
      "slug": "",
      "email": USER_EMAIL,
      "risk_user_id": 1,
      "channel": "mcp",
      "role": "owner",
    }]),
  )

  def fake_channel_factory(**_kwargs):
    global _LAST_FAKE_EVENT_CHANNEL
    pair = _FakeEventChannelPair()
    _LAST_FAKE_EVENT_CHANNEL = pair.parent
    return pair

  def fake_getpgid(pid: int) -> int:
    if pid in _FAKE_PROCESSES:
      if _LAST_FAKE_EVENT_CHANNEL is not None:
        _LAST_FAKE_EVENT_CHANNEL.attach_process(_FAKE_PROCESSES[pid])
      return pid
    return os.getpgid(pid)

  def fake_killpg(pid: int, signal_number: int) -> None:
    process = _FAKE_PROCESSES.get(pid)
    if process is None:
      os.killpg(pid, signal_number)
      return
    if process.returncode is not None:
      return
    if signal_number == signal.SIGKILL:
      process.kill()
    else:
      process.terminate()

  monkeypatch.setattr(
    autonomous_runner_start,
    "create_autonomous_event_channel",
    fake_channel_factory,
  )
  monkeypatch.setattr(autonomous_runner, "_get_process_group_id", fake_getpgid)
  monkeypatch.setattr(autonomous_runner, "_signal_process_group", fake_killpg)
  yield
  _LAST_FAKE_EVENT_CHANNEL = None


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


class ManifestObservingEventBus(RecordingEventBus):
  def __init__(self, manifest_path: Path) -> None:
    super().__init__()
    self.manifest_path = manifest_path
    self.terminal_manifest_states: list[str | None] = []
    self.cleanup_manifest_states: list[str | None] = []

  def _visible_manifest_state(self) -> str | None:
    try:
      payload = json.loads(
        self.manifest_path.read_text(encoding="utf-8")
      )
    except FileNotFoundError:
      return None
    return (
      payload.get("state")
      if isinstance(payload, dict)
      and isinstance(payload.get("state"), str)
      else None
    )

  async def publish(
    self,
    user_id: str,
    control_run_id: str,
    event: dict,
  ) -> None:
    if (
      event.get("type") == "run_state_changed"
      and event.get("state") in {"completed", "failed", "cancelled"}
    ):
      self.terminal_manifest_states.append(
        self._visible_manifest_state()
      )
    await super().publish(user_id, control_run_id, event)

  async def cleanup_run(
    self,
    user_id: str,
    control_run_id: str,
  ) -> None:
    self.cleanup_manifest_states.append(
      self._visible_manifest_state()
    )
    self.calls.append((
      "cleanup",
      {
        "user_id": user_id,
        "control_run_id": control_run_id,
      },
    ))


def _registry(
  tmp_path: Path,
  *,
  approval_store=None,
  skill_resume_allowed_resolver=lambda _skill: False,
):
  from agent_gateway.autonomous_runner import AutonomousRegistry
  from agent_gateway.claim_signing_authority import (
    GatewayClaimSigningAuthority,
  )

  return AutonomousRegistry(
    api_dir=API_DIR,
    tenant_id=TENANT_ID,
    python_executable="python3",
    log_dir=tmp_path,
    max_running=1,
    service_provider_handles={
      "anthropic": _test_service_credential_handle(),
    },
    autonomous_capability_binding_resolver=_test_autonomous_capability_binding,
    skill_resume_allowed_resolver=skill_resume_allowed_resolver,
    claim_signing_authority=GatewayClaimSigningAuthority(HMAC_KEY),
    approval_store=approval_store,
  )


def test_owner_capacity_counts_only_live_processes_for_exact_owner(
  tmp_path: Path,
) -> None:
  from agent_gateway.autonomous_runner import (
    resolve_autonomous_owner_run_limit,
  )

  registry = _registry(tmp_path)
  registry._tasks = {
    "alice-live-1": SimpleNamespace(
      owner_user_id="alice",
      user_id="alice-raw",
      proc=SimpleNamespace(returncode=None),
    ),
    "alice-live-2": SimpleNamespace(
      owner_user_id="alice",
      user_id="alice-raw",
      proc=SimpleNamespace(returncode=None),
    ),
    "alice-finished": SimpleNamespace(
      owner_user_id="alice",
      user_id="alice-raw",
      proc=SimpleNamespace(returncode=0),
    ),
    "bob-live": SimpleNamespace(
      owner_user_id="bob",
      user_id="bob-raw",
      proc=SimpleNamespace(returncode=None),
    ),
    "unowned-live": SimpleNamespace(
      owner_user_id=None,
      user_id="alice",
      proc=SimpleNamespace(returncode=None),
    ),
  }
  resolver_calls: list[tuple[str, int]] = []
  registry._owner_run_limit_resolver = (
    lambda owner, count: resolver_calls.append((owner, count)) or None
  )

  assert registry.live_process_count_for_owner("alice") == 2
  assert registry.live_process_count_for_owner("bob") == 1
  assert registry.owner_capacity(" alice ") == {
    "owner_user_id": "alice",
    "in_flight_count": 2,
    "limit": None,
  }
  assert resolver_calls == [("alice", 2)]
  assert resolve_autonomous_owner_run_limit("alice", 2) is None
  with pytest.raises(ValueError, match="owner_user_id is required"):
    registry.live_process_count_for_owner("")


def test_autonomous_approval_relay_is_exact_and_idempotent(
  tmp_path: Path,
) -> None:
  async def case() -> None:
    from agent_gateway.autonomous_approval_channel import (
      AutonomousApprovalChannelAuthority,
      create_autonomous_approval_channel,
    )

    _write_manifest(
      tmp_path,
      state="completed",
      control_run_id="run-approval-relay",
      channel="tui",
    )
    registry = _registry(tmp_path)
    record = registry._tasks["bg_0"]
    record.state = "running"
    record.proc = SlowFakeProcess()
    record.launch_nonce = "a" * 32
    pair = create_autonomous_approval_channel(
      authority=AutonomousApprovalChannelAuthority(
        launch_nonce=record.launch_nonce,
        task_id=record.task_id,
        control_run_id=record.control_run_id,
        session_id=record.session_id,
        channel_id=record.channel_id,
      ),
    )
    record.approval_channel = pair.parent
    try:
      first = await registry.send_approval_decision(
        "run-approval-relay",
        user_id=USER_ID,
        channel="tui",
        approval_id="approval-irreversible",
        tool_call_id="tool-irreversible",
        nonce="nonce-irreversible",
        approved=True,
        decided_at_ns=1_784_980_800_000_000_000,
        delivery_sequence=1,
        publication_transaction=lambda: nullcontext(),
        sent_reconciliation=lambda: nullcontext(),
      )
      received = pair.child.receive(timeout_seconds=0.1)
      duplicate = await registry.send_approval_decision(
        "run-approval-relay",
        user_id=USER_ID,
        channel="tui",
        approval_id="approval-irreversible",
        tool_call_id="tool-irreversible",
        nonce="nonce-irreversible",
        approved=True,
        decided_at_ns=1_784_980_800_000_000_000,
        delivery_sequence=1,
        publication_transaction=lambda: nullcontext(),
        sent_reconciliation=lambda: nullcontext(),
      )
      replay = pair.child.receive(timeout_seconds=0.1)

      assert received.duplicate is False
      assert replay.duplicate is True
      assert replay.decision == received.decision
      assert first["delivery_status"] == "delivered"
      assert duplicate["delivery_status"] == "duplicate"
      relayed_events = [
        event
        for event in record.event_lines
        if event.get("type") == "approval_decision_sent"
      ]
      assert relayed_events[-1]["allow_tool_type"] is False
    finally:
      pair.close()

  asyncio.run(case())


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


def _run_real_owned_sentinel_output_canary(
  tmp_path: Path,
  *,
  oversized: bool,
) -> tuple[dict, bytes]:
  from agent_gateway.autonomous_credential_handoff import (
    AUTONOMOUS_CREDENTIAL_HANDOFF_ENV,
    AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES,
    AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN,
    encode_autonomous_credential_handoff,
  )
  from agent_gateway.autonomous_runner_start import (
    _OWNED_PROCESS_SENTINEL_SOURCE,
  )
  from agent_gateway.capability_binding import CredentialHandle
  from agent_gateway.capability_execution import MaterializedCredential

  secret = "CUSTOM-ACTIVE-CREDENTIAL-autonomous-output-8f21d7"
  handle = CredentialHandle(
    handle_id="service:tenant-1:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="tenant-1",
    actor_id=None,
  )
  handoff = encode_autonomous_credential_handoff(MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "billing_mode": "byok",
      "api_key": secret,
    },
  ))
  target_source = """
import sys
from agent_gateway.autonomous_credential_handoff import read_autonomous_credential_handoff
material = read_autonomous_credential_handoff(
    expected_handle_id="service:tenant-1:anthropic",
    expected_provider="anthropic",
    expected_principal="service",
    expected_tenant_id="tenant-1",
    expected_actor_id=None,
)
secret = material.auth_config["api_key"]
if sys.argv[1] == "oversized":
    sys.stdout.write("x" * 131073 + secret)
else:
    sys.stdout.write("ordinary stdout " + secret + "\\n")
    sys.stderr.write("ordinary stderr " + secret + "\\n")
sys.stdout.flush()
sys.stderr.flush()
"""
  event_read, event_write = os.pipe()
  claim_read, claim_write = os.pipe()
  owner_read, owner_write = os.pipe()
  lease_path = tmp_path / ("oversized.lease" if oversized else "normal.lease")
  lease_fd = os.open(lease_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
  log_path = tmp_path / ("oversized.log" if oversized else "normal.log")
  log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  env = dict(os.environ)
  env[AUTONOMOUS_CREDENTIAL_HANDOFF_ENV] = AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN
  env["AGENT_AUTONOMOUS_EVENT_CHANNEL_FD"] = str(event_write)
  env["AGENT_AUTONOMOUS_CLAIM_BROKER_FD"] = str(claim_write)
  package_dir = Path(__file__).resolve().parents[1]
  env["PYTHONPATH"] = os.pathsep.join(
    [str(package_dir), env.get("PYTHONPATH", "")]
  ).rstrip(os.pathsep)
  proc = None
  try:
    with os.fdopen(log_fd, "wb") as log_handle:
      proc = subprocess.Popen(
        [
          sys.executable,
          "-c",
          _OWNED_PROCESS_SENTINEL_SOURCE,
          str(AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES),
          str(owner_read),
          str(lease_fd),
          "0.01",
          sys.executable,
          "-c",
          target_source,
          "oversized" if oversized else "normal",
        ],
        stdin=subprocess.PIPE,
        stdout=log_handle,
        stderr=subprocess.PIPE,
        env=env,
        pass_fds=(event_write, claim_write, owner_read, lease_fd),
        start_new_session=True,
      )
      assert proc.stdin is not None
      proc.stdin.write(handoff)
      proc.stdin.close()
      assert proc.stderr is not None
      status = json.loads(proc.stderr.readline())
  finally:
    os.close(owner_write)
    if proc is not None:
      proc.wait(timeout=5)
    for fd in (event_read, event_write, claim_read, claim_write, owner_read, lease_fd):
      try:
        os.close(fd)
      except OSError:
        pass
  return status, log_path.read_bytes()


def test_real_owned_sentinel_redacts_child_stdout_and_stderr_from_run_log(
  tmp_path: Path,
) -> None:
  status, persisted = _run_real_owned_sentinel_output_canary(
    tmp_path,
    oversized=False,
  )

  assert status == {"kind": "exited", "returncode": 0, "version": 1}
  assert b"CUSTOM-ACTIVE-CREDENTIAL-autonomous-output-8f21d7" not in persisted
  assert persisted.count(b"<redacted-secret>") == 2
  assert b"ordinary stdout" in persisted
  assert b"ordinary stderr" in persisted


def test_real_owned_sentinel_output_projection_failure_is_value_free(
  tmp_path: Path,
) -> None:
  from agent_gateway.autonomous_runner import AutonomousRegistry
  from agent_gateway.ui_blocks_metrics import snapshot as metrics_snapshot

  status, persisted = _run_real_owned_sentinel_output_canary(
    tmp_path,
    oversized=True,
  )

  assert status == {
    "error": "child_output_projection_failed",
    "kind": "error",
    "version": 1,
  }
  assert persisted == b"<secret-sanitization-failed>\n"
  assert b"CUSTOM-ACTIVE-CREDENTIAL-autonomous-output-8f21d7" not in persisted

  class AsyncStatus:
    async def readline(self) -> bytes:
      return json.dumps(status).encode("utf-8") + b"\n"

  before = metrics_snapshot().get("secret_boundary_sanitization_failed", 0)
  registry = object.__new__(AutonomousRegistry)
  record = SimpleNamespace(proc=SimpleNamespace(stderr=AsyncStatus()))
  with pytest.raises(RuntimeError, match="child_output_projection_failed"):
    asyncio.run(registry._read_owned_process_sentinel_status(record))
  assert metrics_snapshot()["secret_boundary_sanitization_failed"] == before + 1


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
        "text": "from inbox",
        "sent_at_ns": 123_000_000_000,
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
  assert parent_event["sent_at"] == 123.0
  assert parent_event["sender"] == {"user_id": USER_ID}
  assert parent_event["run_id"] == "patched-run"


def test_autonomous_event_identity_overwrites_payload_claims(
  tmp_path: Path,
) -> None:
  from agent_gateway import autonomous_runner_events

  _write_manifest(tmp_path, "bg_0", control_run_id="run-canonical")
  record = _registry(tmp_path)._tasks["bg_0"]
  source = {
    "type": "text_delta",
    "text": "producer output",
    "run_id": "run-forged",
    "control_run_id": "run-forged",
  }

  projected = autonomous_runner_events.event_for_record(record, source)

  assert projected == {
    "type": "text_delta",
    "text": "producer output",
    "run_id": "run-canonical",
    "control_run_id": "run-canonical",
  }
  assert source["run_id"] == "run-forged"
  assert source["control_run_id"] == "run-forged"


def test_autonomous_runner_status_tail_lines_counts_and_tails(tmp_path) -> None:
  from agent_gateway import autonomous_runner_status

  log_path = tmp_path / "run.log"
  log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

  assert autonomous_runner_status.tail_lines(log_path, 2) == (["two", "three"], 3)
  assert autonomous_runner_status.tail_lines(log_path, 0) == ([], 3)
  assert autonomous_runner_status.tail_lines(tmp_path / "missing.log", 10) == ([], 0)


def test_autonomous_runner_command_helper_preserves_profile_normalization_seam(monkeypatch, tmp_path) -> None:
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

  monkeypatch.setattr(autonomous_runner, "normalize_autonomous_profile", normalize_profile)

  registry = AutonomousRegistry(api_dir=tmp_path, python_executable="py", log_dir=tmp_path)
  cmd = registry._build_cmd(
    profile="Analyst",
    mode="skill",
    task=None,
    skill="patched-skill",
    context=" context ",
    ticker="msft",
    max_budget_usd=5.0,
  )

  assert cmd == [
    "py",
    "-m",
    "agent.autonomous",
    "--profile",
    "patched_profile",
    "--skill",
    "patched-skill",
    "--max-budget-usd",
    "5.0",
    "--ticker",
    "MSFT",
    "--context",
    "context",
  ]
  assert checks == ["profile:Analyst"]


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
  manifest_overrides = {
    "user_id": USER_ID,
    "user_email": USER_EMAIL,
    "context": "Review current packet",
    "ticker": "AAPL",
    "capability_bind": _test_capability_bind("autonomous").receipt(),
  }
  manifest_overrides.update(overrides)
  return write_v6_manifest(tmp_path, task_id, **manifest_overrides)


def _replace_manifest_payload(
  tmp_path: Path,
  manifest: dict,
  task_id: str = "bg_0",
) -> bytes:
  encoded = (json.dumps(manifest, sort_keys=True) + "\n").encode("utf-8")
  _manifest_path(tmp_path, task_id).write_bytes(encoded)
  return encoded


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
  dispatch_scope: dict | None = None,
  start_overrides: dict | None = None,
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
  start_kwargs = {
    "role": "owner",
    "profile": "analyst",
    "mode": "task",
    "task": "summarize",
    "user_id": user_id,
    "user_email": user_email,
    "owner_user_id": owner_user_id,
    "user_slug": user_slug,
    "risk_user_id": risk_user_id,
    "dispatch_scope": dispatch_scope,
  }
  if start_overrides is not None:
    start_kwargs.update(start_overrides)
  payload = await registry.start(**start_kwargs)
  await registry.wait(payload["task_id"], timeout_sec=1)
  return captured


def test_autonomous_manifest_fixture_omits_unconfigured_approval_bridge(
  tmp_path: Path,
) -> None:
  manifest = _write_manifest(tmp_path)

  assert manifest["approval_decisions_path"] is None
  assert manifest["control_authority"]["approval_decisions_path"] is None
  assert not (tmp_path / "bg_0.approval-decisions.jsonl").exists()


def test_autonomous_start_uses_private_claim_broker(monkeypatch, tmp_path) -> None:
  from agent_gateway.autonomous_claim_broker import (
    AUTONOMOUS_CLAIM_BROKER_FD_ENV,
  )

  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert CLAIM_ENV_KEYS.isdisjoint(env)
  assert env[AUTONOMOUS_CLAIM_BROKER_FD_ENV].isdigit()


def test_autonomous_start_never_forwards_global_hmac_key(monkeypatch, tmp_path) -> None:
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert "AGENT_API_USER_CLAIM_HMAC_KEY" not in env
  assert CLAIM_ENV_KEYS.isdisjoint(env)


def test_autonomous_start_injects_event_channel_fd_only(monkeypatch, tmp_path) -> None:
  monkeypatch.setenv("AGENT_AUTONOMOUS_EVENTS_PATH", "/tmp/retired.events.jsonl")
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert env["AGENT_AUTONOMOUS_EVENT_CHANNEL_FD"].isdigit()
  assert "AGENT_AUTONOMOUS_EVENTS_PATH" not in env


def test_autonomous_start_signs_operator_inbox_without_env_fallback(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway.autonomous_launch_envelope import (
    AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
    verify_autonomous_launch_envelope,
  )

  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert "AGENT_AUTONOMOUS_OPERATOR_INBOX_PATH" not in env
  authority = verify_autonomous_launch_envelope(
    HMAC_KEY,
    env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  ).control_authority
  assert authority.operator_inbox_path is not None
  assert authority.operator_inbox_path.endswith(
    ".operator-messages.jsonl"
  )
  assert Path(authority.operator_inbox_path).exists()


def test_autonomous_start_omits_unconfigured_approval_bridge(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway.autonomous_launch_envelope import (
    AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
    verify_autonomous_launch_envelope,
  )

  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert "AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH" not in env
  assert "AGENT_AUTONOMOUS_APPROVALS_DB_PATH" not in env
  assert "AGENT_AUTONOMOUS_CONTROL_RUN_ID" not in env
  authority = verify_autonomous_launch_envelope(
    HMAC_KEY,
    env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  ).control_authority
  assert authority.approval_decisions_path is None
  assert authority.approval_store_path is None


def test_autonomous_start_keeps_approval_store_parent_only_and_inherits_channel(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway.approval_store import SQLiteApprovalStore
  from agent_gateway.autonomous_approval_channel import (
    AUTONOMOUS_APPROVAL_CHANNEL_FD_ENV,
  )
  from agent_gateway.autonomous_launch_envelope import (
    AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
    verify_autonomous_launch_envelope,
  )
  from agent_gateway.autonomous_runner import AutonomousRegistry

  captured: dict[str, object] = {}

  async def fake_exec(*args, **kwargs):
    _ = args
    captured["env"] = dict(kwargs["env"])
    captured["pass_fds"] = tuple(kwargs["pass_fds"])
    return FakeProcess()

  server_dir = tmp_path / "server"
  server_dir.mkdir()
  monkeypatch.chdir(server_dir)
  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  approval_store_path = (server_dir / "approvals.sqlite3").resolve()
  approval_store_path.parent.chmod(0o700)
  approval_store = SQLiteApprovalStore(approval_store_path)
  from agent_gateway.claim_signing_authority import (
    GatewayClaimSigningAuthority,
  )

  registry = AutonomousRegistry(
    api_dir=API_DIR,
    tenant_id=TENANT_ID,
    python_executable="python3",
    log_dir=tmp_path / "logs",
    max_running=1,
    approval_store=approval_store,
    service_provider_handles={
      "anthropic": _test_service_credential_handle(),
    },
    autonomous_capability_binding_resolver=_test_autonomous_capability_binding,
    claim_signing_authority=GatewayClaimSigningAuthority(HMAC_KEY),
  )

  async def start_and_wait() -> None:
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    await registry.wait(payload["task_id"], timeout_sec=1)

  asyncio.run(start_and_wait())

  env = captured["env"]
  assert isinstance(env, dict)
  assert "AGENT_AUTONOMOUS_APPROVALS_DB_PATH" not in env
  assert "AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH" not in env
  approval_channel_fd = int(env[AUTONOMOUS_APPROVAL_CHANNEL_FD_ENV])
  assert approval_channel_fd in captured["pass_fds"]
  authority = verify_autonomous_launch_envelope(
    HMAC_KEY,
    env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  ).control_authority
  assert authority.approval_store_path is None
  assert authority.approval_decisions_path is None
  assert str(approval_store_path) not in json.dumps(env, sort_keys=True)


def test_autonomous_start_signs_exact_session_authority_only(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway.autonomous_launch_envelope import (
    AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
    verify_autonomous_launch_envelope,
  )

  monkeypatch.setenv(
    "AGENT_AUTONOMOUS_GATEWAY_SESSION_ID",
    "ambient-session",
  )
  monkeypatch.setenv(
    "AGENT_AUTONOMOUS_DISPATCH_SCOPE_JSON",
    '{"kind":"portfolio"}',
  )
  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert "AGENT_AUTONOMOUS_GATEWAY_SESSION_ID" not in env
  assert "AGENT_AUTONOMOUS_DISPATCH_SCOPE_JSON" not in env
  envelope = verify_autonomous_launch_envelope(
    HMAC_KEY,
    env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  )
  authority = envelope.session_authority.ordinary_authority
  assert authority is not None
  assert authority.session_id == "bg_0"
  assert authority.tenant_id == TENANT_ID
  assert authority.user_id == USER_ID
  assert authority.owner_user_id == USER_ID
  assert authority.channel == "cli"
  assert authority.expires_at - authority.created_at == 24 * 60 * 60
  assert authority.credential_handle == _test_service_credential_handle()


@pytest.mark.parametrize(
  ("start_overrides", "expected_workload"),
  [
    (
      {"mode": "once", "task": None},
      {
        "profile": "analyst",
        "mode": "run_once",
        "task": None,
        "skill": None,
        "pack": None,
        "context": None,
        "ticker": None,
        "dev_mode": False,
        "max_budget_usd": None,
        "deliver": True,
      },
    ),
    (
      {"mode": "task", "task": "  investigate variance  "},
      {
        "profile": "analyst",
        "mode": "task",
        "task": "investigate variance",
        "skill": None,
        "pack": None,
        "context": None,
        "ticker": None,
        "dev_mode": False,
        "max_budget_usd": None,
        "deliver": True,
      },
    ),
    (
      {"mode": "pack", "task": None, "pack": "  daily-risk  "},
      {
        "profile": "analyst",
        "mode": "pack",
        "task": None,
        "skill": None,
        "pack": "daily-risk",
        "context": None,
        "ticker": None,
        "dev_mode": False,
        "max_budget_usd": None,
        "deliver": True,
      },
    ),
    (
      {
        "mode": "skill",
        "task": None,
        "skill": "  earnings-review  ",
        "context": "  compare guidance  ",
        "ticker": " msft ",
        "max_budget_usd": 12.5,
        "deliver": False,
      },
      {
        "profile": "analyst",
        "mode": "skill",
        "task": None,
        "skill": "earnings-review",
        "pack": None,
        "context": "compare guidance",
        "ticker": "MSFT",
        "dev_mode": False,
        "max_budget_usd": 12.5,
        "deliver": False,
      },
    ),
  ],
)
def test_autonomous_start_signs_exact_executable_workload(
  monkeypatch,
  tmp_path,
  start_overrides,
  expected_workload,
) -> None:
  from agent_gateway.autonomous_launch_envelope import (
    AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
    verify_autonomous_launch_envelope,
  )

  env = asyncio.run(
    _start_and_capture_env(
      monkeypatch,
      tmp_path,
      start_overrides=start_overrides,
    )
  )
  envelope = verify_autonomous_launch_envelope(
    HMAC_KEY,
    env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  )

  workload_receipt = envelope.workload.receipt()
  session_log_authority = workload_receipt.pop(
    "session_log_authority"
  )
  assert workload_receipt == expected_workload
  assert session_log_authority["layout"] == "v1"
  assert session_log_authority["base_path"] == env[
    "AGENT_SESSION_LOG_BASE_DIR"
  ]


def test_autonomous_start_signs_dispatch_scope_without_unsigned_copy(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway.autonomous_launch_envelope import (
    AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
    verify_autonomous_launch_envelope,
  )

  dispatch_scope = {
    "kind": "portfolio",
    "source": "user_selected",
    "portfolio_name": "Core",
    "portfolio_id": "portfolio-1",
    "display_name": "Core Portfolio",
  }
  env = asyncio.run(
    _start_and_capture_env(
      monkeypatch,
      tmp_path,
      dispatch_scope=dispatch_scope,
    )
  )

  assert "AGENT_AUTONOMOUS_DISPATCH_SCOPE_JSON" not in env
  envelope = verify_autonomous_launch_envelope(
    HMAC_KEY,
    env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  )
  assert envelope.session_authority.dispatch_scope is not None
  assert envelope.session_authority.dispatch_scope.receipt() == dispatch_scope


@pytest.mark.parametrize(
  (
    "tenant_id",
    "service_provider_handles",
    "binding_resolver",
    "expected_error",
  ),
  [
    (
      None,
      {"anthropic": _test_service_credential_handle()},
      _test_autonomous_capability_binding,
      "tenant_id is required",
    ),
    (
      TENANT_ID,
      {},
      None,
      "capability binding resolver is required",
    ),
  ],
)
def test_autonomous_start_refuses_missing_session_authority_source(
  monkeypatch,
  tmp_path,
  tenant_id,
  service_provider_handles,
  binding_resolver,
  expected_error,
) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway.autonomous_runner import AutonomousRegistry
  from agent_gateway.claim_signing_authority import (
    GatewayClaimSigningAuthority,
  )

  async def fake_exec(*_args, **_kwargs):
    raise AssertionError("authority validation must precede spawn")

  monkeypatch.setattr(
    autonomous_runner.asyncio,
    "create_subprocess_exec",
    fake_exec,
  )
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = AutonomousRegistry(
    api_dir=tmp_path,
    tenant_id=tenant_id,
    python_executable="python3",
    log_dir=tmp_path,
    max_running=1,
    service_provider_handles=service_provider_handles,
    autonomous_capability_binding_resolver=binding_resolver,
    claim_signing_authority=GatewayClaimSigningAuthority(HMAC_KEY),
  )

  with pytest.raises(RuntimeError, match=expected_error):
    asyncio.run(
      registry.start(
        role="owner",
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert registry._tasks == {}
  assert registry._reserved_slots == 0
  assert not list(tmp_path.glob("*.task.json"))


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
      role="owner",
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
  assert record.events_path == tmp_path / f"{payload['task_id']}.events.jsonl"
  assert record.events_path.exists()
  assert stat.S_IMODE(record.events_path.stat().st_mode) == 0o600
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
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
  )

  assert payload["task_id"] == "bg_12"
  assert (tmp_path / "bg_9.operator-messages.jsonl").read_text(encoding="utf-8") == ""


def test_autonomous_registry_rehydrates_manifest_without_event_file_compatibility(
  tmp_path,
) -> None:
  _write_manifest(
    tmp_path,
    "bg_3",
    control_run_id="run-3",
    resumed_from="run-2",
    resumed_as=["run-4"],
    max_budget_usd=4.5,
    error="stale process error",
  )
  (tmp_path / "bg_3.events.jsonl").write_text(
    "\n".join(
      [
        '{"type":"text_delta","text":"hello","ts":101}',
        "not-json",
        '{"type":"skill_result_captured","skill_run_id":"skill-1","skill":"earnings-review","verdict_echo":{"verdict_token":"watch","confidence":"medium","one_line_summary":"Hold course"},"ts":102}',
        '{"type":"stream_complete","terminal_disposition":"completed","ts":103}',
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
  assert record.role == "owner"
  assert record.profile == "analyst"
  assert record.skill == "earnings-review"
  assert record.pack is None
  assert record.deliver is True
  assert record.ticker == "AAPL"
  assert record.channel == "tui"
  assert record.state == "completed"
  assert record.exit_code == 0
  assert record.error is None
  assert record.completed_at == 125.0
  assert record.max_budget_usd == 4.5
  assert record.proc is None
  assert record.reaper_task is None
  assert record.event_channel is None
  assert record.event_channel_task is None
  assert record.log_handle is None
  assert record.event_lines == [
    {"type": "text_delta", "text": "hello", "ts": 101},
    {
      "type": "skill_result_captured",
      "skill_run_id": "skill-1",
      "skill": "earnings-review",
      "verdict_echo": {
        "verdict_token": "watch",
        "confidence": "medium",
        "one_line_summary": "Hold course",
      },
      "ts": 102,
    },
    {"type": "stream_complete", "terminal_disposition": "completed", "ts": 103},
  ]
  assert record.events_evidence_status == "partial_malformed"
  assert record.event_channel_records == []
  assert record.event_channel_stream is None
  assert record.events_path == tmp_path / "bg_3.events.jsonl"
  assert record.resumed_from == "run-2"
  assert list(record.resumed_as) == ["run-4"]

  record.resumed_as.append("run-5")
  rehydrated_manifest = _read_manifest(tmp_path, "bg_3")
  assert rehydrated_manifest["error"] is None
  assert rehydrated_manifest["resumed_as"] == ["run-4", "run-5"]
  assert rehydrated_manifest["events_path"] == str(tmp_path / "bg_3.events.jsonl")


@pytest.mark.parametrize(
  (
    "platform_name",
    "expected_backend",
    "expected_degraded",
  ),
  (
    ("linux", "landlock-v6", False),
    ("darwin", "darwin-degraded", True),
  ),
)
def test_autonomous_manifest_names_platform_containment_expectation(
  monkeypatch,
  tmp_path,
  platform_name,
  expected_backend,
  expected_degraded,
) -> None:
  from agent_gateway import autonomous_runner_state

  _write_manifest(tmp_path)
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]
  monkeypatch.setattr(
    autonomous_runner_state.sys,
    "platform",
    platform_name,
  )

  manifest = registry._manifest_payload(record)

  assert manifest["containment_expectation"] == {
    "platform": platform_name,
    "expected_backend": expected_backend,
    "expected_degraded": expected_degraded,
  }
  assert manifest["manifest_version"] == 7


def test_autonomous_v7_manifest_without_containment_expectation_loads(
  tmp_path,
) -> None:
  manifest = _write_manifest(tmp_path)
  assert "containment_expectation" not in manifest

  registry = _registry(tmp_path)

  assert registry._tasks["bg_0"].manifest_version == 7


@pytest.mark.parametrize("resolved", [False, True])
def test_missing_v7_skill_resume_fact_is_resolved_once_and_frozen(
  tmp_path,
  resolved,
) -> None:
  manifest = _write_manifest(tmp_path, skill="legacy-skill")
  manifest.pop("skill_resume_allowed")
  _replace_manifest_payload(tmp_path, manifest)
  calls: list[str] = []

  def resolver(skill: str) -> bool:
    calls.append(skill)
    return resolved

  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=resolver,
  )

  assert calls == ["legacy-skill"]
  assert registry._tasks["bg_0"].skill_resume_allowed is resolved
  assert _read_manifest(tmp_path)["skill_resume_allowed"] is resolved

  second = _registry(
    tmp_path,
    skill_resume_allowed_resolver=lambda _skill: (_ for _ in ()).throw(
      AssertionError("frozen manifest must not resolve source again")
    ),
  )
  assert second._tasks["bg_0"].skill_resume_allowed is resolved


def test_noncanonical_manifest_filename_cannot_resolve_or_overwrite_task(
  tmp_path,
) -> None:
  _write_manifest(tmp_path, "bg_0", skill_resume_allowed=False)
  original_bytes = _manifest_path(tmp_path, "bg_0").read_bytes()
  misplaced = _write_manifest(
    tmp_path,
    "bg_1",
    skill="legacy-skill",
  )
  misplaced["task_id"] = "bg_0"
  misplaced["session_id"] = "bg_0"
  misplaced.pop("skill_resume_allowed")
  _replace_manifest_payload(tmp_path, misplaced, "bg_1")
  calls: list[str] = []

  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=lambda skill: calls.append(skill) or True,
  )

  assert calls == []
  assert set(registry._tasks) == {"bg_0"}
  assert _manifest_path(tmp_path, "bg_0").read_bytes() == original_bytes
  assert "skill_resume_allowed" not in _read_manifest(tmp_path, "bg_1")


def test_missing_v7_non_skill_resume_fact_freezes_false_without_resolver(
  tmp_path,
) -> None:
  manifest = _write_manifest(
    tmp_path,
    mode="task",
    task="summarize",
    skill=None,
  )
  manifest.pop("skill_resume_allowed")
  _replace_manifest_payload(tmp_path, manifest)

  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=lambda _skill: (_ for _ in ()).throw(
      AssertionError("non-skill migration must not resolve source")
    ),
  )

  assert registry._tasks["bg_0"].skill_resume_allowed is False
  assert _read_manifest(tmp_path)["skill_resume_allowed"] is False


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_present_v7_skill_resume_fact_requires_exact_bool(
  tmp_path,
  value,
) -> None:
  _write_manifest(tmp_path, skill_resume_allowed=value)

  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=lambda _skill: (_ for _ in ()).throw(
      AssertionError("present malformed fact must not be rederived")
    ),
  )

  assert registry._tasks == {}


def test_present_v7_non_skill_true_resume_fact_is_rejected(tmp_path) -> None:
  _write_manifest(
    tmp_path,
    mode="task",
    task="summarize",
    skill=None,
    skill_resume_allowed=True,
  )

  assert _registry(tmp_path)._tasks == {}


def test_missing_v7_resume_freeze_pre_replace_failure_preserves_old_bytes(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner_state

  manifest = _write_manifest(tmp_path, skill="legacy-skill")
  manifest.pop("skill_resume_allowed")
  old_bytes = _replace_manifest_payload(tmp_path, manifest)

  def fail_replace(_source: Path, _destination: Path) -> None:
    raise OSError("injected pre-replace failure")

  monkeypatch.setattr(autonomous_runner_state, "_os_replace", fail_replace)

  with pytest.raises(
    RuntimeError,
    match="failed to persist autonomous skill resume compatibility fact",
  ):
    _registry(tmp_path, skill_resume_allowed_resolver=lambda _skill: True)

  assert _manifest_path(tmp_path).read_bytes() == old_bytes


def test_missing_v7_resume_resolver_failure_preserves_old_bytes(
  tmp_path,
) -> None:
  manifest = _write_manifest(tmp_path, skill="legacy-skill")
  manifest.pop("skill_resume_allowed")
  old_bytes = _replace_manifest_payload(tmp_path, manifest)

  with pytest.raises(RuntimeError, match="unexpected resolver failure"):
    _registry(
      tmp_path,
      skill_resume_allowed_resolver=lambda _skill: (_ for _ in ()).throw(
        RuntimeError("unexpected resolver failure")
      ),
    )

  assert _manifest_path(tmp_path).read_bytes() == old_bytes


def test_missing_v7_resume_freeze_post_replace_failure_aborts_then_rehydrates(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner_state

  manifest = _write_manifest(tmp_path, skill="legacy-skill")
  manifest.pop("skill_resume_allowed")
  _replace_manifest_payload(tmp_path, manifest)
  original_fsync = autonomous_runner_state.os.fsync

  def fail_directory_fsync(fd: int) -> None:
    if stat.S_ISDIR(os.fstat(fd).st_mode):
      raise OSError("injected post-replace directory fsync failure")
    original_fsync(fd)

  monkeypatch.setattr(
    autonomous_runner_state.os,
    "fsync",
    fail_directory_fsync,
  )
  with pytest.raises(
    RuntimeError,
    match="failed to persist autonomous skill resume compatibility fact",
  ):
    _registry(tmp_path, skill_resume_allowed_resolver=lambda _skill: True)

  assert _read_manifest(tmp_path)["skill_resume_allowed"] is True
  monkeypatch.setattr(autonomous_runner_state.os, "fsync", original_fsync)
  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=lambda _skill: (_ for _ in ()).throw(
      AssertionError("complete replacement must be idempotent")
    ),
  )
  assert registry._tasks["bg_0"].skill_resume_allowed is True


@pytest.mark.parametrize("state", ["completed", "running"])
def test_missing_v7_resume_freeze_refuses_active_prior_owner(
  tmp_path,
  state,
) -> None:
  manifest = _write_manifest(
    tmp_path,
    skill="legacy-skill",
    state=state,
    exit_code=None if state == "running" else 0,
    completed_at=None if state == "running" else 125.0,
  )
  manifest.pop("skill_resume_allowed")
  old_bytes = _replace_manifest_payload(tmp_path, manifest)
  lease_fd = os.open(manifest["owner_lease_path"], os.O_RDONLY)
  fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  calls: list[str] = []
  try:
    with pytest.raises(RuntimeError, match="prior owner is active"):
      _registry(
        tmp_path,
        skill_resume_allowed_resolver=lambda skill: calls.append(skill) or True,
      )
  finally:
    os.close(lease_fd)

  assert calls == []
  assert _manifest_path(tmp_path).read_bytes() == old_bytes


def test_gateway_generic_legacy_resume_resolver_preserves_source_semantics(
  tmp_path,
) -> None:
  from agent_gateway.server import _generic_skill_resume_allowed_resolver

  skills_dir = tmp_path / "skills"
  skills_dir.mkdir()
  (skills_dir / "read-only.md").write_text(
    """---
name: read-only
description: Read-only resumable work
resumable: true
mutation_mode: read_only
catalog: false
---
Resume the read-only work.
""",
    encoding="utf-8",
  )
  (skills_dir / "writer.md").write_text(
    """---
name: writer
description: Writer work
resumable: true
mutation_mode: model_writer
---
Do writer work.
""",
    encoding="utf-8",
  )
  (skills_dir / "malformed.md").write_text(
    "---\nname: malformed\nresumable: []\n---\nMalformed policy.\n",
    encoding="utf-8",
  )
  resolver = _generic_skill_resume_allowed_resolver(skills_dir)

  assert resolver("read-only") is True
  assert resolver("writer") is False
  assert resolver("missing") is False
  assert resolver("malformed") is False


def test_effective_resume_projection_preserves_dynamic_narrowing_order(
  tmp_path,
) -> None:
  from agent_gateway.control_plane.runs_helpers import (
    _autonomous_task_resumable,
  )

  _write_manifest(
    tmp_path,
    state="interrupted",
    skill="frozen-skill",
    skill_resume_allowed=True,
  )
  record = _registry(tmp_path)._tasks["bg_0"]
  capability_bind = record.capability_bind
  assert _autonomous_task_resumable(record) is True

  record.capability_bind = None
  assert _autonomous_task_resumable(record) is False
  record.capability_bind = capability_bind
  record.mode = "task"
  assert _autonomous_task_resumable(record) is False
  record.mode = "skill"
  record.state = "running"
  assert _autonomous_task_resumable(record) is False
  record.state = "interrupted"
  record.skill_resume_allowed = False
  assert _autonomous_task_resumable(record) is False


@pytest.mark.parametrize("schedule_id", [None, "schedule-1"])
def test_initial_direct_and_scheduled_skill_starts_freeze_resolver_fact(
  monkeypatch,
  tmp_path,
  schedule_id,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*_args, **_kwargs):
    return FakeProcess()

  calls: list[str] = []
  monkeypatch.setattr(
    autonomous_runner.asyncio,
    "create_subprocess_exec",
    fake_exec,
  )
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=lambda skill: calls.append(skill) or True,
  )

  async def run_case() -> None:
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="skill",
      skill="frozen-skill",
      user_id=USER_ID,
      user_email=USER_EMAIL,
      schedule_id=schedule_id,
    )
    try:
      record = registry._tasks[payload["task_id"]]
      assert record.skill_resume_allowed is True
      assert _read_manifest(tmp_path, record.task_id)[
        "skill_resume_allowed"
      ] is True
      assert record.capability_bind is not None
      assert record.capability_bind.run_mode == (
        "cron" if schedule_id is not None else "autonomous"
      )
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(run_case())
  assert calls == ["frozen-skill"]


def test_invalid_skill_launch_shape_fails_before_resume_policy_resolution(
  tmp_path,
) -> None:
  calls: list[str] = []
  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=lambda skill: calls.append(skill) or True,
  )

  with pytest.raises(ValueError, match="mode='skill' does not accept task"):
    asyncio.run(
      registry.start(
        role="owner",
        profile="analyst",
        mode="skill",
        skill="frozen-skill",
        task="invalid extra task",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert calls == []


def test_resumed_start_inherits_frozen_resume_fact_without_source_resolution(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*_args, **_kwargs):
    return FakeProcess()

  _write_manifest(
    tmp_path,
    control_run_id="original-run",
    state="interrupted",
    skill="frozen-skill",
    skill_resume_allowed=True,
  )
  monkeypatch.setattr(
    autonomous_runner.asyncio,
    "create_subprocess_exec",
    fake_exec,
  )
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=lambda _skill: (_ for _ in ()).throw(
      AssertionError("resume must inherit rather than resolve current source")
    ),
  )

  async def run_case() -> None:
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="skill",
      skill="frozen-skill",
      user_id=USER_ID,
      user_email=USER_EMAIL,
      resumed_from="original-run",
    )
    try:
      resumed = registry._tasks[payload["task_id"]]
      assert resumed.skill_resume_allowed is True
      assert _read_manifest(tmp_path, resumed.task_id)[
        "skill_resume_allowed"
      ] is True
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(run_case())


@pytest.mark.parametrize(
  "resume_overrides",
  [
    {"mode": "skill", "skill": "different-skill"},
    {"mode": "task", "skill": None, "task": "different task"},
  ],
)
def test_resumed_start_refuses_changed_mode_or_skill_before_side_effects(
  monkeypatch,
  tmp_path,
  resume_overrides,
) -> None:
  from agent_gateway import autonomous_runner

  resolver_calls: list[str] = []
  spawn_calls: list[bool] = []

  async def fail_spawn(*_args, **_kwargs):
    spawn_calls.append(True)
    raise AssertionError("mismatched resume must not spawn")

  _write_manifest(
    tmp_path,
    control_run_id="original-run",
    state="interrupted",
    skill="frozen-skill",
    skill_resume_allowed=True,
  )
  monkeypatch.setattr(
    autonomous_runner.asyncio,
    "create_subprocess_exec",
    fail_spawn,
  )
  registry = _registry(
    tmp_path,
    skill_resume_allowed_resolver=(
      lambda skill: resolver_calls.append(skill) or True
    ),
  )

  with pytest.raises(
    ValueError,
    match="mode and skill must match the origin",
  ):
    asyncio.run(
      registry.start(
        role="owner",
        profile="analyst",
        mode=resume_overrides["mode"],
        skill=resume_overrides.get("skill"),
        task=resume_overrides.get("task"),
        user_id=USER_ID,
        user_email=USER_EMAIL,
        resumed_from="original-run",
      )
    )

  assert resolver_calls == []
  assert spawn_calls == []
  assert set(registry._tasks) == {"bg_0"}
  assert not _manifest_path(tmp_path, "bg_1").exists()


def test_autonomous_v7_manifest_round_trips_invite_role(tmp_path) -> None:
  _write_manifest(tmp_path, role="invite")

  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]

  assert record.role == "invite"
  assert registry._manifest_payload(record)["role"] == "invite"


@pytest.mark.parametrize("role", ["Owner", " owner ", "", None, True])
def test_autonomous_v7_manifest_rejects_malformed_role(tmp_path, role) -> None:
  _write_manifest(tmp_path, role=role)
  assert "bg_0" not in _registry(tmp_path)._tasks


def test_autonomous_v7_manifest_rejects_missing_role(tmp_path) -> None:
  manifest = _write_manifest(tmp_path)
  del manifest["role"]
  _manifest_path(tmp_path).write_text(json.dumps(manifest) + "\n", encoding="utf-8")

  assert "bg_0" not in _registry(tmp_path)._tasks


@pytest.mark.parametrize(
  ("field_name", "field_value"),
  [
    ("pack", "unexpected-pack"),
    ("deliver", "true"),
  ],
)
def test_autonomous_registry_rejects_corrupt_v5_pack_deliver_contract(
  tmp_path,
  field_name,
  field_value,
) -> None:
  _write_manifest(tmp_path, **{field_name: field_value})

  registry = _registry(tmp_path)

  assert "bg_0" not in registry._tasks


@pytest.mark.parametrize("missing_field", ["pack", "deliver"])
def test_autonomous_registry_rejects_v5_manifest_missing_exact_launch_field(
  tmp_path,
  missing_field,
) -> None:
  manifest = _write_manifest(tmp_path)
  del manifest[missing_field]
  _manifest_path(tmp_path).write_text(
    json.dumps(manifest) + "\n",
    encoding="utf-8",
  )

  registry = _registry(tmp_path)

  assert "bg_0" not in registry._tasks


def test_autonomous_registry_rejects_v4_manifest_without_pack_deliver_contract(
  tmp_path,
) -> None:
  _write_manifest(tmp_path, manifest_version=4)

  registry = _registry(tmp_path)

  assert "bg_0" not in registry._tasks


def test_autonomous_registry_rejects_v1_slug_manifest(
  monkeypatch,
  tmp_path,
) -> None:
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
    manifest_version=1,
  )

  registry = _registry(tmp_path)

  assert "bg_4" not in registry._tasks
  assert _read_manifest(tmp_path, "bg_4")["manifest_version"] == 1


@pytest.mark.parametrize("max_budget_usd", [True, 0, -1, float("inf"), float("nan"), "5"])
def test_autonomous_registry_drops_invalid_manifest_max_budget(tmp_path, max_budget_usd) -> None:
  _write_manifest(tmp_path, max_budget_usd=max_budget_usd)

  registry = _registry(tmp_path)

  assert registry._tasks["bg_0"].max_budget_usd is None


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
def test_autonomous_registry_rehydrates_active_states_as_interrupted(
  monkeypatch,
  tmp_path,
  state,
) -> None:
  from agent_gateway import autonomous_runner

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
  monkeypatch.setattr(autonomous_runner.time, "time", lambda: 789.25)

  registry = _registry(tmp_path)

  record = registry._tasks["bg_0"]
  assert record.state == "interrupted"
  assert record.exit_code is None
  assert record.error == "gateway restarted while run was active"
  assert record.completed_at == 456.0
  assert record.event_lines == [
    {"type": "text_delta", "text": "before restart", "ts": 456}
  ]
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


def test_restart_resume_waits_for_prior_sentinel_owner_lease(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  manifest = _write_manifest(
    tmp_path,
    state="running",
    exit_code=None,
    error=None,
    completed_at=None,
  )
  lease_fd = os.open(manifest["owner_lease_path"], os.O_RDWR)
  fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]

  assert record.state == "interrupted"
  assert _read_manifest(tmp_path)["state"] == "running"

  async def case() -> None:
    async def fake_exec(*_args, **_kwargs):
      return SlowFakeProcess()

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)

    with pytest.raises(
      RuntimeError,
      match="prior autonomous run owner cleanup is still active",
    ):
      await registry.start(
        role=record.role,
        profile=record.profile,
        mode=record.mode,
        skill=record.skill,
        context=record.context,
        ticker=record.ticker,
        channel=record.channel,
        user_id=record.raw_user_id or record.user_id,
        user_email=record.user_email,
        owner_user_id=record.owner_user_id,
        resumed_from=record.control_run_id,
      )

    fcntl.flock(lease_fd, fcntl.LOCK_UN)
    os.close(lease_fd)
    payload = await registry.start(
      role=record.role,
      profile=record.profile,
      mode=record.mode,
      skill=record.skill,
      context=record.context,
      ticker=record.ticker,
      channel=record.channel,
      user_id=record.raw_user_id or record.user_id,
      user_email=record.user_email,
      owner_user_id=record.owner_user_id,
      resumed_from=record.control_run_id,
    )
    await registry.cancel(payload["task_id"])

  try:
    asyncio.run(case())
  finally:
    try:
      os.close(lease_fd)
    except OSError:
      pass


def test_autonomous_registry_reads_conventional_event_file(tmp_path, caplog) -> None:
  _write_manifest(tmp_path, "bg_0")
  (tmp_path / "bg_0.events.jsonl").write_text(
    "".join(f'{{"type":"event","idx":{idx},"ts":{idx}}}\n' for idx in range(5)),
    encoding="utf-8",
  )

  with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
    registry = _registry(tmp_path)

  record = registry._tasks["bg_0"]
  assert record.event_lines == [
    {"type": "event", "idx": idx, "ts": idx}
    for idx in range(5)
  ]
  assert record.events_evidence_status == "complete"
  assert "events file" not in caplog.text.lower()


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


def test_autonomous_registry_warn_once_is_process_and_disk_scoped(tmp_path, caplog) -> None:
  from agent_gateway import autonomous_runner_state

  _write_manifest(tmp_path, "bg_1", manifest_version=999)
  skip_path = tmp_path / autonomous_runner_state._SKIP_WARNED_FILE

  with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
    first = _registry(tmp_path)
    second = _registry(tmp_path)

  assert first._tasks == {}
  assert second._tasks == {}
  assert caplog.text.count("Skipping autonomous manifest with unsupported version") == 1
  assert skip_path.is_file()
  persisted = json.loads(skip_path.read_text(encoding="utf-8"))
  assert any(str(_manifest_path(tmp_path, "bg_1")) in item for item in persisted)

  saved_warned = set(autonomous_runner_state._SKIP_WARNED)
  saved_dirs = set(autonomous_runner_state._SKIP_WARNED_LOADED_DIRS)
  autonomous_runner_state._SKIP_WARNED.clear()
  autonomous_runner_state._SKIP_WARNED_LOADED_DIRS.clear()
  caplog.clear()
  try:
    with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
      restarted = _registry(tmp_path)

    assert restarted._tasks == {}
    assert "Skipping autonomous manifest with unsupported version" not in caplog.text

    manifest_path = _manifest_path(tmp_path, "bg_1")
    os.utime(manifest_path, (0, manifest_path.stat().st_mtime + 5))
    autonomous_runner_state._SKIP_WARNED.clear()
    autonomous_runner_state._SKIP_WARNED_LOADED_DIRS.clear()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
      rewritten = _registry(tmp_path)
    assert rewritten._tasks == {}
    assert caplog.text.count("Skipping autonomous manifest with unsupported version") == 1
  finally:
    autonomous_runner_state._SKIP_WARNED.clear()
    autonomous_runner_state._SKIP_WARNED.update(saved_warned)
    autonomous_runner_state._SKIP_WARNED_LOADED_DIRS.clear()
    autonomous_runner_state._SKIP_WARNED_LOADED_DIRS.update(saved_dirs)


def test_autonomous_registry_skips_manifest_with_invalid_task_invariants(
  tmp_path,
  caplog,
) -> None:
  _write_manifest(
    tmp_path,
    state="failed",
    exit_code=1,
    error="not a successful terminal",
    terminal_reason="writer_lease_already_held",
  )

  with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
    registry = _registry(tmp_path)

  assert registry._tasks == {}
  assert "invalid task invariants" in caplog.text


def test_autonomous_manifest_write_fsyncs_file_and_directory(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner_state

  _write_manifest(tmp_path)
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]
  original_fsync = os.fsync
  fsync_targets: list[str] = []

  def recording_fsync(fd: int) -> None:
    fd_stat = os.fstat(fd)
    fsync_targets.append("directory" if stat.S_ISDIR(fd_stat.st_mode) else "file")
    original_fsync(fd)

  monkeypatch.setattr(autonomous_runner_state.os, "fsync", recording_fsync)

  assert registry._write_task_manifest(record, checked=True) is True
  assert fsync_targets == ["file", "directory"]
  assert stat.S_IMODE(_manifest_path(tmp_path).stat().st_mode) == 0o600


def test_autonomous_manifest_delete_fsyncs_parent_and_reports_result(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner_state

  _write_manifest(tmp_path)
  registry = _registry(tmp_path)
  original_fsync = os.fsync
  directory_fsyncs = 0

  def recording_fsync(fd: int) -> None:
    nonlocal directory_fsyncs
    if stat.S_ISDIR(os.fstat(fd).st_mode):
      directory_fsyncs += 1
    original_fsync(fd)

  monkeypatch.setattr(
    autonomous_runner_state.os,
    "fsync",
    recording_fsync,
  )

  assert registry._delete_task_manifest("bg_0") is True
  assert directory_fsyncs == 1
  assert not _manifest_path(tmp_path).exists()


def test_autonomous_manifest_delete_retry_fsyncs_prior_unlink(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner_state

  _write_manifest(tmp_path)
  registry = _registry(tmp_path)
  original_fsync = os.fsync
  fail_once = True

  def fail_first_directory_fsync(fd: int) -> None:
    nonlocal fail_once
    if fail_once and stat.S_ISDIR(os.fstat(fd).st_mode):
      fail_once = False
      raise OSError("injected directory fsync failure")
    original_fsync(fd)

  monkeypatch.setattr(
    autonomous_runner_state.os,
    "fsync",
    fail_first_directory_fsync,
  )

  assert registry._delete_task_manifest("bg_0") is False
  assert not _manifest_path(tmp_path).exists()
  assert registry._delete_task_manifest("bg_0") is True


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


def test_record_and_publish_seeds_rehydrated_event_file(tmp_path) -> None:
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
  assert seed_payload["events"] == [{
    "type": "text_delta",
    "text": "old",
    "run_id": "run-11",
    "control_run_id": "run-11",
  }]
  assert bus.calls[1][1]["event"] == {
    "type": "run_resumed",
    "resumed_run_id": "run-12",
    "run_id": "run-11",
    "control_run_id": "run-11",
  }


def test_autonomous_manifest_committed_before_spawn_with_full_field_set(monkeypatch, tmp_path) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner
    from agent_gateway.autonomous_launch_envelope import (
      AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
      verify_autonomous_launch_envelope,
    )

    async def fake_exec(*args, **kwargs):
      _ = args
      starting = _read_manifest(tmp_path)
      assert starting["state"] == "starting"
      assert Path(starting["tool_result_spill_dir"]).is_dir()
      assert kwargs["env"]["AGENT_AUTONOMOUS_TOOL_RESULT_SPILL_DIR"] == starting["tool_result_spill_dir"]
      envelope = verify_autonomous_launch_envelope(
        HMAC_KEY,
        kwargs["env"][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
      )
      assert envelope.control_run_id == starting["control_run_id"]
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    monkeypatch.setenv(
      "GATEWAY_USER_KEYS",
      json.dumps([{
        "key": "test-mcp-key",
        "slug": "",
        "email": "",
        "risk_user_id": 1,
        "channel": "mcp",
        "role": "owner",
      }]),
    )
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="skill",
      skill="risk.scan",
      context=" inspect current book ",
      ticker="msft",
      channel="TUI",
      control_run_id="  run-custom  ",
      user_id=USER_ID,
      user_email=None,
      max_budget_usd=5.0,
    )
    try:
      manifest = _read_manifest(tmp_path)
      assert set(manifest) == {
        "manifest_version",
        "task_id",
        "control_run_id",
        "session_id",
        "channel_id",
        "owner_user_id",
        "user_id",
        "raw_user_id",
        "user_slug",
        "risk_user_id",
        "user_email",
        "user_aliases",
        "identity_status",
        "role",
        "profile",
        "mode",
        "task",
        "skill",
        "pack",
          "deliver",
          "skill_resume_allowed",
        "context",
        "ticker",
        "channel",
        "dev_mode",
        "max_budget_usd",
        "dispatch_scope",
        "containment_expectation",
        "cmd",
          "log_path",
          "events_path",
        "operator_inbox_path",
        "approval_decisions_path",
        "owner_lease_path",
        "owner_lease_device",
        "owner_lease_inode",
        "control_authority",
        "started_at",
        "state",
        "exit_code",
        "error",
        "terminal_reason",
        "completed_at",
        "resumed_from",
        "resumed_as",
        "schedule_id",
        "schedule_name",
        "capability_bind",
        "tool_result_spill_dir",
      }
      assert manifest["manifest_version"] == 7
      assert manifest["task_id"] == payload["task_id"] == "bg_0"
      assert manifest["control_run_id"] == "run-custom"
      assert payload["run_id"] == "run-custom"
      assert manifest["session_id"] == "bg_0"
      assert len(manifest["channel_id"]) == 64
      assert manifest["owner_user_id"] == USER_ID
      assert manifest["user_id"] == USER_ID
      assert manifest["raw_user_id"] == USER_ID
      assert manifest["user_slug"] is None
      assert manifest["risk_user_id"] == 1
      assert manifest["user_email"] is None
      assert manifest["user_aliases"] == [USER_ID]
      assert manifest["identity_status"] == "numeric_user_id"
      assert manifest["role"] == "owner"
      assert manifest["profile"] == "analyst"
      assert manifest["mode"] == "skill"
      assert manifest["task"] is None
      assert manifest["skill"] == "risk.scan"
      assert manifest["pack"] is None
      assert manifest["deliver"] is True
      assert manifest["skill_resume_allowed"] is False
      assert manifest["context"] == "inspect current book"
      assert manifest["ticker"] == "MSFT"
      assert manifest["channel"] == "tui"
      assert manifest["dev_mode"] is False
      assert manifest["max_budget_usd"] == 5.0
      assert manifest["dispatch_scope"] is None
      assert manifest["containment_expectation"] == {
        "platform": sys.platform,
        "expected_backend": (
          "landlock-v6"
          if sys.platform == "linux"
          else "darwin-degraded"
        ),
        "expected_degraded": sys.platform == "darwin",
      }
      assert manifest["cmd"][:5] == ["python3", "-m", "agent.autonomous", "--profile", "analyst"]
      assert "--dev" not in manifest["cmd"]
      assert manifest["cmd"][manifest["cmd"].index("--max-budget-usd") + 1] == "5.0"
      assert manifest["log_path"] == str(tmp_path / "bg_0.log")
      assert manifest["events_path"] == str(tmp_path / "bg_0.events.jsonl")
      assert manifest["operator_inbox_path"] == str(tmp_path / "bg_0.operator-messages.jsonl")
      assert manifest["approval_decisions_path"] is None
      assert manifest["owner_lease_path"] == str(
        tmp_path / "bg_0.owner-lease"
      )
      assert isinstance(manifest["owner_lease_device"], int)
      assert isinstance(manifest["owner_lease_inode"], int)
      assert manifest["control_authority"]["control_mode"] == "file"
      assert (
        manifest["control_authority"]["operator_inbox_path"]
        == manifest["operator_inbox_path"]
      )
      assert manifest["control_authority"]["approval_decisions_path"] is None
      assert manifest["control_authority"]["approval_store_path"] is None
      assert isinstance(manifest["started_at"], float)
      assert manifest["state"] == "running"
      assert manifest["exit_code"] is None
      assert manifest["error"] is None
      assert manifest["terminal_reason"] is None
      assert manifest["completed_at"] is None
      assert manifest["resumed_from"] is None
      assert manifest["resumed_as"] == []
      assert manifest["schedule_id"] is None
      assert manifest["schedule_name"] is None
      assert manifest["capability_bind"] == _test_capability_bind(
        "autonomous"
      ).receipt()
      assert manifest["tool_result_spill_dir"] == str(tmp_path / "bg_0.tool_result_spill")
      assert "proc" not in manifest
      assert "reaper_task" not in manifest
      assert "event_channel" not in manifest
      assert "event_channel_task" not in manifest
      assert "event_channel_records" not in manifest
      assert "event_channel_projected_events" not in manifest
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
        role="owner",
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
        role="owner",
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
  assert list(tmp_path.glob("*.events.jsonl")) == []
  assert not (tmp_path / "bg_0.tool_result_spill").exists()


def test_autonomous_initial_manifest_failure_refuses_before_spawn(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    spawn_calls = 0

    async def fake_exec(*args, **kwargs):
      nonlocal spawn_calls
      _ = args, kwargs
      spawn_calls += 1
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
    with pytest.raises(
      RuntimeError,
      match="failed to persist starting autonomous task manifest",
    ):
      await registry.start(
        role="owner",
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )

    assert spawn_calls == 0
    assert registry._tasks == {}
    assert registry._reserved_slots == 0
    assert not _manifest_path(tmp_path).exists()
    assert not (tmp_path / "bg_0.tool_result_spill").exists()

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
      assert _LAST_FAKE_EVENT_CHANNEL is not None
      _LAST_FAKE_EVENT_CHANNEL.attach_process(process)
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
        role="owner",
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
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process_factory()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    await registry.wait(payload["task_id"], timeout_sec=1)

    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == expected_state
    assert manifest["exit_code"] == expected_exit_code
    assert manifest["error"] == expected_error
    assert isinstance(manifest["completed_at"], float)

  asyncio.run(_case())


def test_autonomous_terminal_replace_failure_commits_fenced_failure_before_publish(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return FakeProcess()

    manifest_path = _manifest_path(tmp_path)
    bus = ManifestObservingEventBus(manifest_path)
    original_replace = os.replace
    failed_terminal_replace = False

    def fail_first_completed_replace(src, dst, *args, **kwargs):
      nonlocal failed_terminal_replace
      source_path = Path(src)
      destination_path = Path(dst)
      if (
        not failed_terminal_replace
        and destination_path == manifest_path
        and source_path.exists()
      ):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if payload.get("state") == "completed":
          failed_terminal_replace = True
          raise OSError("injected terminal replace failure")
      return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setattr(
      autonomous_runner.os,
      "replace",
      fail_first_completed_replace,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    registry.set_user_event_bus(bus)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )

    status = await registry.wait(payload["task_id"], timeout_sec=1)
    record = registry._tasks[payload["task_id"]]
    manifest = _read_manifest(tmp_path)

    assert failed_terminal_replace is True
    assert status["state"] == "failed"
    assert record.terminal_manifest_committed is True
    assert manifest["state"] == "failed"
    assert "run fenced from completed outcome" in manifest["error"]
    assert bus.terminal_manifest_states == ["failed"]
    assert bus.cleanup_manifest_states == ["failed"]
    assert not any(
      name == "publish"
      and payload["event"].get("type") == "run_state_changed"
      and payload["event"].get("state") == "completed"
      for name, payload in bus.calls
    )

  asyncio.run(_case())


@pytest.mark.parametrize(
  "failure_target",
  ["file", "directory"],
)
def test_autonomous_terminal_fsync_failure_commits_fenced_failure_before_publish(
  monkeypatch,
  tmp_path,
  failure_target: str,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner
    from agent_gateway import autonomous_runner_state

    process = SlowFakeProcess()

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    manifest_path = _manifest_path(tmp_path)
    bus = ManifestObservingEventBus(manifest_path)
    registry = _registry(tmp_path)
    registry.set_user_event_bus(bus)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )

    original_fsync = os.fsync
    record = registry._tasks[payload["task_id"]]
    fail_once = True

    def fail_terminal_fsync(fd: int) -> None:
      nonlocal fail_once
      fd_is_directory = stat.S_ISDIR(os.fstat(fd).st_mode)
      target_matches = (
        fd_is_directory
        if failure_target == "directory"
        else (
          not fd_is_directory
          and (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
          != (record.events_device, record.events_inode)
        )
      )
      if fail_once and target_matches:
        fail_once = False
        raise OSError(
          f"injected terminal {failure_target} fsync failure"
        )
      original_fsync(fd)

    monkeypatch.setattr(
      autonomous_runner_state.os,
      "fsync",
      fail_terminal_fsync,
    )
    process.returncode = 0

    status = await registry.wait(payload["task_id"], timeout_sec=1)
    manifest = _read_manifest(tmp_path)

    assert fail_once is False
    assert status["state"] == "failed"
    assert record.terminal_manifest_committed is True
    assert manifest["state"] == "failed"
    assert "run fenced from completed outcome" in manifest["error"]
    assert bus.terminal_manifest_states == ["failed"]
    assert bus.cleanup_manifest_states == ["failed"]

  asyncio.run(_case())


def test_autonomous_double_terminal_manifest_failure_removes_stale_manifest_without_publish(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    process = SlowFakeProcess()

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    manifest_path = _manifest_path(tmp_path)
    bus = ManifestObservingEventBus(manifest_path)
    registry = _registry(tmp_path)
    registry.set_user_event_bus(bus)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    record = registry._tasks[payload["task_id"]]
    original_replace = os.replace

    def fail_terminal_replaces(src, dst, *args, **kwargs):
      if Path(dst) == manifest_path:
        raise OSError("injected persistent terminal replace failure")
      return original_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(
      autonomous_runner.os,
      "replace",
      fail_terminal_replaces,
    )
    process.returncode = 0

    status = await registry.wait(payload["task_id"], timeout_sec=1)

    assert process.returncode == 0
    assert status["state"] == "failed"
    assert "could not be durably persisted" in (status["error"] or "")
    assert record.terminal_manifest_committed is False
    assert not manifest_path.exists()
    assert bus.terminal_manifest_states == []
    assert bus.cleanup_manifest_states == []
    assert not any(
      name == "publish"
      and call_payload["event"].get("type") == "run_state_changed"
      and call_payload["event"].get("state")
      in {"completed", "failed", "cancelled"}
      for name, call_payload in bus.calls
    )

    restarted = _registry(tmp_path)
    assert payload["task_id"] not in restarted._tasks

  asyncio.run(_case())


def test_autonomous_reaper_is_event_driven_without_elapsed_reap() -> None:
  from agent_gateway import autonomous_runner

  source = Path(autonomous_runner.__file__).read_text(encoding="utf-8")
  tree = ast.parse(source)
  reaper = next(
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.AsyncFunctionDef)
    and node.name == "_reap_owned_process"
  )
  waits = [
    node
    for node in ast.walk(reaper)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr == "wait"
  ]

  assert "_AUTONOMOUS_RUN_MAX_" + "SECONDS" not in source
  assert "runtime " + "limit" not in source
  assert any(
    any(
      keyword.arg == "return_when"
      and isinstance(keyword.value, ast.Attribute)
      and keyword.value.attr == "FIRST_COMPLETED"
      for keyword in call.keywords
    )
    for call in waits
  )
  assert all(
    all(keyword.arg != "timeout" for keyword in call.keywords)
    for call in waits
    if any(keyword.arg == "return_when" for keyword in call.keywords)
  )


def test_process_exit_wakes_reaper_fences_group_then_settles_channel(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner
    from agent_gateway import autonomous_runner_start
    from agent_gateway.autonomous_event_channel import (
      AutonomousEventRecord,
      ReceivedAutonomousEventStream,
    )

    class StatusPipe:
      def __init__(self) -> None:
        self.lines: asyncio.Queue[bytes] = asyncio.Queue()

      async def readline(self) -> bytes:
        return await self.lines.get()

    class SentinelProcess(SlowFakeProcess):
      def __init__(self) -> None:
        super().__init__()
        self.stderr = StatusPipe()

      def report_direct_exit(self, returncode: int) -> None:
        self.stderr.lines.put_nowait(json.dumps({
          "kind": "exited",
          "returncode": returncode,
          "version": 1,
        }).encode("utf-8") + b"\n")

    process = SentinelProcess()
    pair = _FakeEventChannelPair()
    fenced = threading.Event()
    delivered = False
    terminal_record = AutonomousEventRecord(
      seq=0,
      frame_bytes=b"buffered-terminal-frame",
      event_line_bytes=(
        b'{"terminal_disposition":"completed","type":"stream_complete"}\n'
      ),
    )
    terminal_stream = ReceivedAutonomousEventStream(
      channel_id=pair.parent.channel_id,
      records=(terminal_record,),
      event_count=1,
      event_digest=hashlib.sha256(b"buffered-terminal-frame").hexdigest(),
      exact_event_frame_bytes=len(b"buffered-terminal-frame"),
    )
    pair.parent._record = terminal_record
    pair.parent._stream = terminal_stream

    def held_receive_next(**kwargs):
      nonlocal delivered
      assert kwargs in ({"unbounded_stream": True}, {})
      assert fenced.wait(2)
      if not delivered:
        delivered = True
        return terminal_record
      return terminal_stream

    pair.parent.receive_next = held_receive_next

    def channel_factory(**_kwargs):
      global _LAST_FAKE_EVENT_CHANNEL
      _LAST_FAKE_EVENT_CHANNEL = pair.parent
      return pair

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    signal_observations: list[tuple[int, int | None]] = []

    def tracked_group_signal(_process_group_id: int, signal_number: int) -> None:
      signal_observations.append((signal_number, process.returncode))
      fenced.set()
      process.kill()

    monkeypatch.setattr(
      autonomous_runner_start,
      "create_autonomous_event_channel",
      channel_factory,
    )
    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setattr(
      autonomous_runner,
      "_signal_process_group",
      tracked_group_signal,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )

    process.report_direct_exit(0)
    status = await registry.wait(payload["task_id"], timeout_sec=2)
    record = registry._tasks[payload["task_id"]]

    assert signal_observations == [(signal.SIGKILL, None)]
    assert status["state"] == "completed"
    assert record.exit_code == 0
    assert record.event_channel_stream is terminal_stream
    assert record.event_channel_acknowledgement is not None
    assert record.slot_reserved is False

  asyncio.run(_case())


def test_pre_ack_cancellation_interrupts_signals_and_settles(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner
    from agent_gateway import autonomous_runner_start

    process = SlowFakeProcess()
    pair = _FakeEventChannelPair()
    observations: list[str] = []
    original_interrupt = pair.parent.interrupt

    def tracked_interrupt() -> None:
      observations.append("interrupt")
      original_interrupt()

    pair.parent.interrupt = tracked_interrupt

    def channel_factory(**_kwargs):
      global _LAST_FAKE_EVENT_CHANNEL
      _LAST_FAKE_EVENT_CHANNEL = pair.parent
      return pair

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    def tracked_group_signal(_process_group_id: int, signal_number: int) -> None:
      observations.append(signal.Signals(signal_number).name)
      if signal_number == signal.SIGKILL:
        process.kill()
      else:
        process.terminate()

    monkeypatch.setattr(
      autonomous_runner_start,
      "create_autonomous_event_channel",
      channel_factory,
    )
    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setattr(
      autonomous_runner,
      "_signal_process_group",
      tracked_group_signal,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )

    status = await registry.cancel(payload["task_id"])
    record = registry._tasks[payload["task_id"]]

    assert observations[:2] == ["interrupt", "SIGTERM"]
    assert status["state"] == "killed"
    assert record.event_channel_task.done()
    assert record.slot_reserved is False

  asyncio.run(_case())


def test_registry_local_interrupted_terminal_cannot_settle_channel_run(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    process = SlowFakeProcess()

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    record = registry._tasks[payload["task_id"]]
    await registry._record_and_publish_event(
      record,
      {
        "type": "stream_complete",
        "terminal_disposition": "interrupted",
        "reason": "operator_pause",
      },
    )
    process.returncode = 0

    await registry.wait(payload["task_id"], timeout_sec=1)

    assert record.exit_code == 0
    assert record.state == "completed"
    assert record.error is None
    assert registry._terminal_state_for_record(record) == "completed"
    assert any(
      event.get("type") == "run_state_changed"
      and event.get("state") == "completed"
      for event in record.event_lines or []
    )
    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "completed"
    assert manifest["error"] is None

  asyncio.run(_case())


@pytest.mark.parametrize(
  "terminal_event",
  [
    {"type": "stream_complete"},
    {
      "type": "stream_complete",
      "terminal_disposition": "future",
    },
  ],
  ids=["missing-disposition", "invalid-disposition"],
)
def test_registry_local_invalid_terminal_cannot_settle_channel_run(
  monkeypatch,
  tmp_path,
  terminal_event: dict,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    process = SlowFakeProcess()

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    record = registry._tasks[payload["task_id"]]
    await registry._record_and_publish_event(record, terminal_event)
    process.returncode = 0

    await registry.wait(payload["task_id"], timeout_sec=1)

    assert record.exit_code == 0
    assert record.state == "completed"
    assert record.error is None
    assert registry._terminal_state_for_record(record) == "completed"
    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "completed"
    assert manifest["error"] is None

  asyncio.run(_case())


def test_writer_lease_terminal_reason_persists_and_rehydrates(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> str:
    from agent_gateway import autonomous_runner

    process = SlowFakeProcess()

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    record = registry._tasks[payload["task_id"]]
    terminal_event = {
      "type": "stream_complete",
      "terminal_disposition": "completed",
      "terminal_reason": "writer_lease_already_held",
    }
    projected = await registry._record_and_publish_event(
      record,
      terminal_event,
      strict=True,
    )
    assert projected is not None
    record.event_channel_projected_events.append(projected)
    process.returncode = 0

    await registry.wait(payload["task_id"], timeout_sec=1)

    assert record.state == "completed"
    assert record.exit_code == 0
    assert record.error is None
    assert record.terminal_reason == "writer_lease_already_held"
    return payload["task_id"]

  task_id = asyncio.run(_case())
  manifest = _read_manifest(tmp_path, task_id)
  assert manifest["terminal_reason"] == "writer_lease_already_held"

  rehydrated = _registry(tmp_path)
  assert (
    rehydrated._tasks[task_id].terminal_reason
    == "writer_lease_already_held"
  )


def test_autonomous_manifest_updates_on_reaper_failure(monkeypatch, tmp_path) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return RaisingFakeProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  registry = _registry(tmp_path)
  async def start_and_wait() -> None:
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    await registry.wait(payload["task_id"], timeout_sec=1)

  asyncio.run(start_and_wait())

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
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    record = registry._tasks[payload["task_id"]]
    record.state = state
    await registry._record_and_publish_event(
      record,
      {
        "type": "stream_complete",
        "terminal_disposition": "completed",
      },
    )
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
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )

    await registry.cancel(payload["task_id"])
    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "killed"
    assert manifest["exit_code"] == -15
    assert manifest["error"] == "Process terminated by user"
    assert isinstance(manifest["completed_at"], float)
    assert registry._tasks[payload["task_id"]].reaper_task.done()

    await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_cancel_manifest_failure_does_not_suppress_process_kill(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    process = SlowFakeProcess()

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    original_write = registry._write_task_manifest
    failed_request_write = False

    def raise_on_cancellation_request(record, *, checked: bool = False):
      nonlocal failed_request_write
      if record.cancellation_requested and not failed_request_write:
        failed_request_write = True
        raise OSError("injected cancellation manifest failure")
      return original_write(record, checked=checked)

    registry._write_task_manifest = raise_on_cancellation_request  # type: ignore[method-assign]

    status = await registry.cancel(payload["task_id"])

    assert failed_request_write is True
    assert process.returncode == -15
    assert status["state"] == "killed"
    assert registry._tasks[payload["task_id"]].terminal_manifest_committed
    assert _read_manifest(tmp_path)["state"] == "killed"

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
      role="owner",
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


def test_autonomous_shutdown_manifest_failure_does_not_suppress_process_kill(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    process = SlowFakeProcess()

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return process

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
      profile="analyst",
      mode="task",
      task="summarize",
      user_id=USER_ID,
      user_email=USER_EMAIL,
    )
    original_write = registry._write_task_manifest
    failed_request_write = False

    def raise_on_shutdown_request(record, *, checked: bool = False):
      nonlocal failed_request_write
      if record.cancellation_requested and not failed_request_write:
        failed_request_write = True
        raise OSError("injected shutdown manifest failure")
      return original_write(record, checked=checked)

    registry._write_task_manifest = raise_on_shutdown_request  # type: ignore[method-assign]

    await registry.shutdown(grace_sec=0.1)

    assert failed_request_write is True
    assert process.returncode == -15
    record = registry._tasks[payload["task_id"]]
    assert record.state == "killed"
    assert record.terminal_manifest_committed
    assert _read_manifest(tmp_path)["state"] == "killed"

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
      role="owner",
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

    original_replace = autonomous_runner.os.replace

    def fail_replace(src, dst):
      if Path(dst) == _manifest_path(tmp_path) and Path(dst).exists():
        raise OSError("readonly-ish")
      return original_replace(src, dst)

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(autonomous_runner.os, "replace", fail_replace)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    with caplog.at_level(logging.WARNING, logger="agent_gateway.autonomous_runner"):
      with pytest.raises(RuntimeError, match="commit running autonomous task manifest"):
        await registry.start(
          role="owner",
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
      role="owner",
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
      record.delivered_messages.clear()
      record.event_lines = [
        event for event in record.event_lines if event.get("type") != "parent_message_sent"
      ]
      with pytest.raises(
        RuntimeError,
        match="reused with different content",
      ):
        await registry.send_operator_message(
          payload["run_id"],
          user_id=USER_ID,
          channel="tui",
          message="Different retry text should not replace the inbox payload",
          message_id="msg-1",
        )
      assert len(record.operator_inbox_path.read_text(encoding="utf-8").splitlines()) == 1
      repaired_parent_events = [
        event
        for event in record.event_lines
        if event["type"] == "parent_message_sent"
      ]
      assert repaired_parent_events == []
    finally:
      await registry.cancel(payload["task_id"])
      await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())


def test_autonomous_parent_operator_message_quota_rejects_before_append(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    from agent_gateway.autonomous_control_contract import (
      AUTONOMOUS_OPERATOR_MESSAGE_LIMIT,
    )

    _write_manifest(
      tmp_path,
      state="completed",
      channel="tui",
    )
    registry = _registry(tmp_path)
    record = registry._tasks["bg_0"]
    record.state = "running"
    record.proc = SlowFakeProcess()

    for index in range(AUTONOMOUS_OPERATOR_MESSAGE_LIMIT):
      await registry.send_operator_message(
        record.control_run_id,
        user_id=USER_ID,
        channel="tui",
        message=f"message-{index}",
        message_id=f"message-{index}",
      )

    with pytest.raises(RuntimeError, match="message quota"):
      await registry.send_operator_message(
        record.control_run_id,
        user_id=USER_ID,
        channel="tui",
        message="one too many",
        message_id="message-overflow",
      )

    assert record.operator_inbox_path is not None
    assert len(
      record.operator_inbox_path.read_text(encoding="utf-8").splitlines()
    ) == AUTONOMOUS_OPERATOR_MESSAGE_LIMIT

  asyncio.run(_case())


def test_autonomous_parent_operator_byte_quota_rejects_before_append(
  monkeypatch,
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    _write_manifest(
      tmp_path,
      state="completed",
      channel="tui",
    )
    registry = _registry(tmp_path)
    record = registry._tasks["bg_0"]
    record.state = "running"
    record.proc = SlowFakeProcess()
    monkeypatch.setattr(
      autonomous_runner,
      "AUTONOMOUS_OPERATOR_AGGREGATE_BYTES_LIMIT",
      1,
    )

    with pytest.raises(RuntimeError, match="aggregate byte quota"):
      await registry.send_operator_message(
        record.control_run_id,
        user_id=USER_ID,
        channel="tui",
        message="bounded",
        message_id="message-1",
      )

    assert record.operator_inbox_path is not None
    assert record.operator_inbox_path.read_bytes() == b""

  asyncio.run(_case())


def test_autonomous_parent_approval_quota_rejects_before_append(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    from agent_gateway.autonomous_approval_channel import (
      AutonomousApprovalChannelAuthority,
      create_autonomous_approval_channel,
    )
    from agent_gateway.autonomous_control_contract import (
      AUTONOMOUS_APPROVAL_DECISION_LIMIT,
    )

    _write_manifest(
      tmp_path,
      state="completed",
      channel="tui",
    )
    registry = _registry(tmp_path)
    record = registry._tasks["bg_0"]
    record.state = "running"
    record.proc = SlowFakeProcess()
    record.launch_nonce = "b" * 32
    pair = create_autonomous_approval_channel(
      authority=AutonomousApprovalChannelAuthority(
        launch_nonce=record.launch_nonce,
        task_id=record.task_id,
        control_run_id=record.control_run_id,
        session_id=record.session_id,
        channel_id=record.channel_id,
      ),
    )
    record.approval_channel = pair.parent
    try:
      for index in range(AUTONOMOUS_APPROVAL_DECISION_LIMIT):
        await registry.send_approval_decision(
          record.control_run_id,
          user_id=USER_ID,
          channel="tui",
          approval_id=f"approval-{index}",
          tool_call_id=f"tool-{index}",
          nonce=f"nonce-{index}",
          approved=bool(index % 2),
          decided_at_ns=index + 1,
          delivery_sequence=index + 1,
          publication_transaction=lambda: nullcontext(),
          sent_reconciliation=lambda: nullcontext(),
        )
        received = pair.child.receive(timeout_seconds=0.1)
        assert received.decision.delivery_sequence == index + 1

      with pytest.raises(RuntimeError, match="over quota"):
        await registry.send_approval_decision(
          record.control_run_id,
          user_id=USER_ID,
          channel="tui",
          approval_id="approval-overflow",
          tool_call_id="tool-overflow",
          nonce="nonce-overflow",
          approved=False,
          decided_at_ns=AUTONOMOUS_APPROVAL_DECISION_LIMIT + 1,
          delivery_sequence=AUTONOMOUS_APPROVAL_DECISION_LIMIT + 1,
          publication_transaction=lambda: nullcontext(),
          sent_reconciliation=lambda: nullcontext(),
        )
    finally:
      pair.close()

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
      role="owner",
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


def test_autonomous_operator_message_accepts_after_child_stream_complete(
  monkeypatch,
  tmp_path,
) -> None:
  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return SlowFakeProcess()

    monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      role="owner",
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
      record.event_lines.append({
        "type": "stream_complete",
        "terminal_disposition": "completed",
        "sub_agent_id": "sub0:spawned",
      })

      response = await registry.send_operator_message(
        payload["run_id"],
        user_id=USER_ID,
        channel="tui",
        message="parent is still running",
        message_id="msg-after-child-complete",
      )

      assert response["delivery_status"] == "delivered"
      assert record.operator_inbox_path is not None
      assert "parent is still running" in record.operator_inbox_path.read_text(
        encoding="utf-8",
      )
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


def test_autonomous_start_scrubs_key_on_spawn_failure(monkeypatch, tmp_path) -> None:
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
        role="owner",
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert "AGENT_API_USER_CLAIM_HMAC_KEY" not in captured_env
  assert CLAIM_ENV_KEYS.isdisjoint(captured_env)
  assert os.environ.get("AGENT_API_USER_CLAIM_HMAC_KEY") == HMAC_KEY
  assert registry._tasks == {}
  assert registry._reserved_slots == 0
  assert not _manifest_path(tmp_path).exists()


def test_autonomous_start_signs_identity_without_unsigned_aliases(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway.autonomous_launch_envelope import (
    AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
    verify_autonomous_launch_envelope,
  )

  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  for key in (
    "AUTONOMOUS_USER_ID",
    "AUTONOMOUS_RAW_USER_ID",
    "AUTONOMOUS_USER_SLUG",
    "AUTONOMOUS_USER_EMAIL",
  ):
    assert key not in env
  authority = verify_autonomous_launch_envelope(
    HMAC_KEY,
    env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  ).session_authority.ordinary_authority
  assert authority is not None
  assert authority.user_id == USER_ID
  assert authority.raw_user_id == USER_ID
  assert authority.user_slug is None
  assert authority.user_email == USER_EMAIL


def test_autonomous_start_resolves_slug_metadata_to_canonical_owner(monkeypatch, tmp_path) -> None:
  from agent_gateway.autonomous_launch_envelope import (
    AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
    verify_autonomous_launch_envelope,
  )

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
        "extraneous_secret": "must-not-reach-child",
      },
      {
        "key": "excel-key-for-henry",
        "channel": "excel",
        "slug": "henry",
        "email": USER_EMAIL,
        "risk_user_id": 1,
        "role": "owner",
      },
      {
        "key": "mcp-key-for-other-user",
        "channel": "mcp",
        "slug": "other",
        "email": "other@example.com",
        "risk_user_id": 2,
        "role": "invite",
      },
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

  authority = verify_autonomous_launch_envelope(
    HMAC_KEY,
    env[AUTONOMOUS_CAPABILITY_ENVELOPE_ENV],
  ).session_authority.ordinary_authority
  assert authority is not None
  assert authority.user_id == "1"
  assert authority.raw_user_id == "henry"
  assert authority.user_slug == "henry"
  assert CLAIM_ENV_KEYS.isdisjoint(env)
  assert json.loads(env["GATEWAY_USER_KEYS"]) == [{
    "key": "mcp-key-for-henry",
    "slug": "henry",
    "email": USER_EMAIL,
    "risk_user_id": 1,
    "channel": "mcp",
    "role": "owner",
  }]
  assert "must-not-reach-child" not in env["GATEWAY_USER_KEYS"]
  assert "mcp-key-for-other-user" not in env["GATEWAY_USER_KEYS"]
  assert "excel-key-for-henry" not in env["GATEWAY_USER_KEYS"]


def test_autonomous_start_narrows_mcp_key_by_envelope_session_slug(
  monkeypatch,
  tmp_path,
) -> None:
  monkeypatch.setenv(
    "GATEWAY_USER_KEYS",
    json.dumps([{
      "key": "mcp-key-for-henry",
      "channel": "mcp",
      "slug": "henry",
      "email": USER_EMAIL,
      "risk_user_id": 1,
      "role": "owner",
    }]),
  )

  env = asyncio.run(
    _start_and_capture_env(
      monkeypatch,
      tmp_path,
      user_id="henry",
      user_email=USER_EMAIL,
      owner_user_id="henry",
      user_slug="henry",
    )
  )

  assert json.loads(env["GATEWAY_USER_KEYS"])[0]["slug"] == "henry"


@pytest.mark.parametrize(
  "gateway_user_keys",
  ("{not-json", json.dumps({"channel": "mcp"})),
)
def test_autonomous_start_contains_malformed_gateway_user_keys_system_exit(
  monkeypatch,
  tmp_path,
  gateway_user_keys,
) -> None:
  from agent_gateway import autonomous_runner

  spawned = False

  async def fake_exec(*_args, **_kwargs):
    nonlocal spawned
    spawned = True
    return FakeProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("GATEWAY_USER_KEYS", gateway_user_keys)
  registry = _registry(tmp_path)

  with pytest.raises(
    RuntimeError,
    match="autonomous spawn refused: GATEWAY_USER_KEYS is malformed",
  ):
    asyncio.run(
      registry.start(
        role="owner",
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert spawned is False
  assert registry._tasks == {}
  assert registry._reserved_slots == 0
  assert not list(tmp_path.glob("*.task.json"))


@pytest.mark.parametrize(
  ("entry", "error"),
  (
    (
      {
        "key": "",
        "slug": "henry",
        "email": USER_EMAIL,
        "risk_user_id": 1,
        "channel": "mcp",
        "role": "owner",
      },
      "entry key is empty",
    ),
    (
      {
        "key": "secret-for-someone-else",
        "slug": "other",
        "email": "other@example.com",
        "risk_user_id": 2,
        "channel": "mcp",
        "role": "invite",
      },
      "no GATEWAY_USER_KEYS channel='mcp' entry for user '1'",
    ),
  ),
)
def test_autonomous_start_refuses_unusable_mcp_entry_without_spawning(
  monkeypatch,
  tmp_path,
  entry,
  error,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*_args, **_kwargs):
    raise AssertionError("MCP key refusal must precede spawn")

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setenv("GATEWAY_USER_KEYS", json.dumps([entry]))
  registry = _registry(tmp_path)

  with pytest.raises(RuntimeError, match=error) as exc_info:
    asyncio.run(
      registry.start(
        role="owner",
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  if entry["key"]:
    assert entry["key"] not in str(exc_info.value)
  assert registry._tasks == {}
  assert registry._reserved_slots == 0


def test_autonomous_start_refuses_absent_api_identity_module_even_when_cached(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway.autonomous_runner import AutonomousRegistry
  from agent_gateway.autonomous_runner_state import _user_identity_api
  from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority

  cached_api = _user_identity_api()
  assert cached_api is not None
  assert Path(cached_api.__file__).resolve() == (API_DIR / "user_identity.py").resolve()

  async def fake_exec(*_args, **_kwargs):
    raise AssertionError("absent identity module must precede spawn")

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  registry = AutonomousRegistry(
    api_dir=tmp_path,
    tenant_id=TENANT_ID,
    python_executable="python3",
    log_dir=tmp_path,
    service_provider_handles={"anthropic": _test_service_credential_handle()},
    autonomous_capability_binding_resolver=_test_autonomous_capability_binding,
    claim_signing_authority=GatewayClaimSigningAuthority(HMAC_KEY),
  )

  with pytest.raises(RuntimeError, match="user identity API is unavailable"):
    asyncio.run(
      registry.start(
        role="owner",
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert registry._tasks == {}
  assert registry._reserved_slots == 0


def test_autonomous_start_refuses_identity_module_import_failure(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway import autonomous_runner_start

  def fail_import(*, api_dir=None):
    _ = api_dir
    raise ImportError("sensitive import detail")

  async def fake_exec(*_args, **_kwargs):
    raise AssertionError("identity import failure must precede spawn")

  monkeypatch.setattr(autonomous_runner_start, "_user_identity_api", fail_import)
  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  registry = _registry(tmp_path)

  with pytest.raises(RuntimeError, match="user identity API import failed") as exc_info:
    asyncio.run(
      registry.start(
        role="owner",
        profile="analyst",
        mode="task",
        task="summarize",
        user_id=USER_ID,
        user_email=USER_EMAIL,
      )
    )

  assert "sensitive import detail" not in str(exc_info.value)
  assert registry._tasks == {}
  assert registry._reserved_slots == 0


def test_autonomous_start_scrubs_preexisting_unsigned_identity(
  monkeypatch,
  tmp_path,
) -> None:
  monkeypatch.setenv("AUTONOMOUS_USER_ID", "other")
  monkeypatch.setenv("AUTONOMOUS_USER_EMAIL", "other@example.com")

  env = asyncio.run(_start_and_capture_env(monkeypatch, tmp_path))

  assert "AUTONOMOUS_USER_ID" not in env
  assert "AUTONOMOUS_RAW_USER_ID" not in env
  assert "AUTONOMOUS_USER_SLUG" not in env
  assert "AUTONOMOUS_USER_EMAIL" not in env
  assert CLAIM_ENV_KEYS.isdisjoint(env)


def test_autonomous_start_scrubs_ambient_credential_and_profile_dev_authority(
  monkeypatch,
  tmp_path,
) -> None:
  authority_env_names = (
    "AGENT_AUTONOMOUS_USER_CREDENTIAL_HANDOFF",
    "ANALYST_DEV_MODE",
    "ADVISOR_DEV_MODE",
    "RESEARCH_PRODUCER_DEV_MODE",
  )
  dev_tuning_env_names = (
    "ANALYST_DEV_MAX_BUDGET_USD",
    "ANALYST_DEV_MAX_TURNS",
    "ANALYST_DEV_STREAM_STALL_TIMEOUT",
    "ANALYST_DEV_TIMEOUT",
  )
  for env_name in (*authority_env_names, *dev_tuning_env_names):
    monkeypatch.setenv(env_name, "hostile-ambient-value")

  non_dev_env = asyncio.run(
    _start_and_capture_env(
      monkeypatch,
      tmp_path / "non-dev",
      start_overrides={
        "mode": "skill",
        "task": None,
        "skill": "earnings-review",
      },
    )
  )
  assert all(
    env_name not in non_dev_env
    for env_name in (*authority_env_names, *dev_tuning_env_names)
  )

  task_env = asyncio.run(
    _start_and_capture_env(
      monkeypatch,
      tmp_path / "task",
      start_overrides={
        "mode": "task",
        "task": "complete a product task",
        "skill": None,
      },
    )
  )
  assert all(
    env_name not in task_env
    for env_name in (*authority_env_names, *dev_tuning_env_names)
  )

def test_autonomous_start_fails_without_installed_claim_authority(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway.autonomous_runner import AutonomousRegistry

  called = False

  async def fake_exec(*args, **kwargs):
    nonlocal called
    called = True
    return FakeProcess()

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  registry = AutonomousRegistry(
    api_dir=tmp_path,
    tenant_id=TENANT_ID,
    python_executable="python3",
    log_dir=tmp_path,
    max_running=1,
    service_provider_handles={
      "anthropic": _test_service_credential_handle(),
    },
    autonomous_capability_binding_resolver=(
      _test_autonomous_capability_binding
    ),
  )

  with pytest.raises(
    RuntimeError,
    match="requires installed claim-signing authority",
  ):
    asyncio.run(
      registry.start(
        role="owner",
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
def test_authority_claim_uses_verifier_ttl_ceiling() -> None:
  from agent_gateway.claim_signing_authority import (
    GatewayClaimSigningAuthority,
  )

  assert AGENT_API_CLAIM_MAX_TTL_SECONDS == 600

  env = GatewayClaimSigningAuthority(HMAC_KEY).sign_user_claim(
    user_id=USER_ID,
    user_email=USER_EMAIL,
    ttl_seconds=600,
  )

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
def test_authority_claim_above_verifier_ceiling_is_rejected() -> None:
  from agent_gateway.claim_signing_authority import (
    GatewayClaimSigningAuthority,
  )

  env = GatewayClaimSigningAuthority(HMAC_KEY).sign_user_claim(
    user_id=USER_ID,
    user_email=USER_EMAIL,
    ttl_seconds=900,
  )

  issued_at = int(env["AGENT_API_CLAIM_ISSUED_AT"])
  expiry = int(env["AGENT_API_CLAIM_EXPIRY"])
  assert expiry - issued_at == 900
  assert verify(
    HMAC_KEY,
    _claim_headers_from_env(env),
    ttl_ceiling=600,
    now=issued_at,
  ) is None


def test_event_evidence_concurrent_callers_are_durable_before_publish(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  _write_manifest(tmp_path, state="interrupted")
  bus = RecordingEventBus()
  registry = _registry(tmp_path)
  registry.set_user_event_bus(bus)
  record = registry._tasks["bg_0"]
  original_append = autonomous_runner.append_open_json_record
  first_started = threading.Event()
  release_first = threading.Event()
  append_order: list[int] = []

  def ordered_append(path, *, expected_device, expected_inode, payload):
    append_order.append(payload["idx"])
    if payload["idx"] == 1:
      first_started.set()
      assert release_first.wait(timeout=2)
    original_append(
      path,
      expected_device=expected_device,
      expected_inode=expected_inode,
      payload=payload,
    )

  monkeypatch.setattr(autonomous_runner, "append_open_json_record", ordered_append)

  async def case() -> None:
    first = asyncio.create_task(
      registry._record_and_publish_event(record, {"type": "ordered", "idx": 1})
    )
    assert await asyncio.to_thread(first_started.wait, 2)
    second = asyncio.create_task(
      registry._record_and_publish_event(record, {"type": "ordered", "idx": 2})
    )
    await asyncio.sleep(0)
    assert append_order == [1]
    release_first.set()
    await asyncio.gather(first, second)

  asyncio.run(case())

  assert append_order == [1, 2]
  assert [payload["event"]["idx"] for name, payload in bus.calls if name == "publish"] == [1, 2]
  assert [json.loads(line)["idx"] for line in record.events_path.read_text().splitlines()] == [1, 2]


def test_event_evidence_burst_append_does_not_starve_event_loop(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  _write_manifest(tmp_path, state="interrupted")
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]
  original_append = autonomous_runner.append_open_json_record

  def slow_append(*args, **kwargs):
    threading.Event().wait(0.01)
    return original_append(*args, **kwargs)

  monkeypatch.setattr(autonomous_runner, "append_open_json_record", slow_append)

  async def case() -> int:
    running = True
    heartbeats = 0

    async def heartbeat() -> None:
      nonlocal heartbeats
      while running:
        heartbeats += 1
        await asyncio.sleep(0)

    pulse = asyncio.create_task(heartbeat())
    await asyncio.gather(*(
      registry._record_and_publish_event(
        record,
        {"type": "burst", "idx": idx},
      )
      for idx in range(12)
    ))
    running = False
    await pulse
    return heartbeats

  assert asyncio.run(case()) > 100


def test_event_evidence_cancellation_waits_for_worker_before_unlock(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  _write_manifest(tmp_path, state="interrupted")
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]
  original_append = autonomous_runner.append_open_json_record
  first_started = threading.Event()
  release_first = threading.Event()
  append_order: list[int] = []

  def blocking_append(path, *, expected_device, expected_inode, payload):
    append_order.append(payload["idx"])
    if payload["idx"] == 1:
      first_started.set()
      assert release_first.wait(timeout=2)
    original_append(
      path,
      expected_device=expected_device,
      expected_inode=expected_inode,
      payload=payload,
    )

  monkeypatch.setattr(autonomous_runner, "append_open_json_record", blocking_append)

  async def case() -> None:
    first = asyncio.create_task(
      registry._record_and_publish_event(record, {"type": "cancel", "idx": 1})
    )
    assert await asyncio.to_thread(first_started.wait, 2)
    first.cancel()
    second = asyncio.create_task(
      registry._record_and_publish_event(record, {"type": "cancel", "idx": 2})
    )
    await asyncio.sleep(0)
    assert append_order == [1]
    release_first.set()
    with pytest.raises(asyncio.CancelledError):
      await first
    await second

  asyncio.run(case())

  assert append_order == [1, 2]
  assert [json.loads(line)["idx"] for line in record.events_path.read_text().splitlines()] == [1, 2]
  assert [event["idx"] for event in record.event_lines if event.get("type") == "cancel"] == [2]


def test_event_evidence_cancelled_worker_terminates_wait(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  _write_manifest(tmp_path, state="interrupted")
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]
  append_started = threading.Event()
  release_append = threading.Event()
  original_create_task = asyncio.create_task
  worker_tasks: list[asyncio.Task] = []

  def blocking_append(_record, _event):
    append_started.set()
    assert release_append.wait(timeout=2)

  def capture_worker(coro):
    worker = original_create_task(coro)
    worker_tasks.append(worker)
    return worker

  monkeypatch.setattr(registry, "_append_event_evidence_sync", blocking_append)
  monkeypatch.setattr(autonomous_runner.asyncio, "create_task", capture_worker)

  async def case() -> None:
    caller = original_create_task(
      registry._record_and_publish_event(record, {"type": "worker-cancel"})
    )
    assert await asyncio.to_thread(append_started.wait, 2)
    assert len(worker_tasks) == 1
    worker_tasks[0].cancel()
    try:
      with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(caller, timeout=0.5)
    finally:
      release_append.set()

  asyncio.run(case())


@pytest.mark.parametrize("stream_recovered", [True, False])
def test_event_evidence_any_live_append_failure_fences_strict_false(
  monkeypatch,
  tmp_path,
  stream_recovered,
) -> None:
  from agent_gateway.autonomous_control_files import AutonomousControlAppendError

  _write_manifest(tmp_path, state="interrupted")
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]
  record.proc = SlowFakeProcess()
  signals: list[int] = []

  def fail_append(_record, _event):
    raise AutonomousControlAppendError(
      "injected evidence failure",
      stream_recovered=stream_recovered,
    )

  monkeypatch.setattr(registry, "_append_event_evidence_sync", fail_append)
  monkeypatch.setattr(
    registry,
    "_signal_owned_process_group",
    lambda _record, sig: signals.append(sig) or True,
  )

  with pytest.raises(AutonomousControlAppendError):
    asyncio.run(
      registry._record_and_publish_event(record, {"type": "fence"})
    )

  assert record.cancellation_requested is True
  assert "Autonomous event evidence append failure" in record.error
  assert signals == [signal.SIGTERM]
  assert not any(event.get("type") == "fence" for event in record.event_lines)


def test_event_evidence_processless_append_failure_degrades_without_fence(
  monkeypatch,
  tmp_path,
) -> None:
  _write_manifest(tmp_path, state="interrupted")
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]

  def fail_append(_record, _event):
    raise OSError("injected adoption failure")

  monkeypatch.setattr(registry, "_append_event_evidence_sync", fail_append)
  result = asyncio.run(
    registry._record_and_publish_event(record, {"type": "run_resumed"})
  )

  assert result is None
  assert record.events_evidence_status == "unreadable"
  assert record.cancellation_requested is False


def test_torn_event_tail_is_repaired_on_adoption_and_next_restart_recovers_append(
  tmp_path,
) -> None:
  _write_manifest(tmp_path, state="interrupted")
  events_path = tmp_path / "bg_0.events.jsonl"
  events_path.write_bytes(b'{"type":"before","ts":1}\n{"type":"torn"')
  first_registry = _registry(tmp_path)
  first_record = first_registry._tasks["bg_0"]

  assert first_record.event_lines == [{"type": "before", "ts": 1}]
  assert first_record.events_evidence_status == "partial_malformed"
  asyncio.run(
    first_registry._record_and_publish_event(
      first_record,
      {"type": "after", "ts": 2},
    )
  )

  second_record = _registry(tmp_path)._tasks["bg_0"]
  assert [event["type"] for event in second_record.event_lines] == ["before", "after"]


def test_missing_event_file_adoption_fsyncs_directory_before_append_and_rehydrates(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  _write_manifest(tmp_path, state="interrupted")
  registry = _registry(tmp_path)
  record = registry._tasks["bg_0"]
  operations: list[str] = []
  real_dir_fsync = autonomous_runner.fsync_owned_file_directory
  real_append = autonomous_runner.append_open_json_record

  def observed_dir_fsync(path):
    operations.append("directory_fsync")
    real_dir_fsync(path)

  def observed_append(*args, **kwargs):
    operations.append("append")
    real_append(*args, **kwargs)

  monkeypatch.setattr(autonomous_runner, "fsync_owned_file_directory", observed_dir_fsync)
  monkeypatch.setattr(autonomous_runner, "append_open_json_record", observed_append)
  asyncio.run(
    registry._record_and_publish_event(record, {"type": "adopted", "ts": 3})
  )

  assert operations == ["directory_fsync", "append"]
  assert _registry(tmp_path)._tasks["bg_0"].event_lines == [
    {
      "type": "adopted",
      "ts": 3,
      "run_id": "bg_0",
      "control_run_id": "bg_0",
    }
  ]


def test_oversized_event_evidence_tail_load_is_structured(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner_state

  _write_manifest(tmp_path, state="completed")
  (tmp_path / "bg_0.events.jsonl").write_text(
    "".join(json.dumps({"type": "tail", "idx": idx}) + "\n" for idx in range(8)),
    encoding="utf-8",
  )
  monkeypatch.setattr(autonomous_runner_state, "_REHYDRATE_EVENTS_SIZE_CAP_BYTES", 1)
  monkeypatch.setattr(autonomous_runner_state, "_REHYDRATE_EVENTS_TAIL_LINES", 3)

  record = _registry(tmp_path)._tasks["bg_0"]
  assert record.events_evidence_status == "tail_truncated"
  assert [event["idx"] for event in record.event_lines] == [5, 6, 7]


@pytest.mark.parametrize(
  ("manifest_state", "manifest_error"),
  [
    ("completed", None),
    ("failed", "recorded failure"),
    ("killed", "cancelled"),
    ("interrupted", "restart"),
    ("budget_limited", "limited"),
    ("budget_exceeded", "exceeded"),
    ("blocked", "blocked"),
  ],
)
def test_rehydrated_terminal_state_matrix_preserves_recorded_precedence(
  tmp_path,
  manifest_state,
  manifest_error,
) -> None:
  _write_manifest(
    tmp_path,
    state=manifest_state,
    error=manifest_error,
    terminal_reason=None,
  )
  (tmp_path / "bg_0.events.jsonl").write_text(
    json.dumps({"type": "stream_complete", "terminal_disposition": "interrupted", "ts": 7}) + "\n",
    encoding="utf-8",
  )

  record = _registry(tmp_path)._tasks["bg_0"]
  if manifest_state == "completed":
    assert record.state == "interrupted"
    assert record.error is None
  elif manifest_state in {"budget_limited", "budget_exceeded"}:
    assert record.state == "budget_limited"
    assert record.error is None
  else:
    assert record.state == manifest_state
    assert record.error == manifest_error


def test_finished_manifest_without_terminal_event_rehydrates_interrupted(tmp_path) -> None:
  _write_manifest(tmp_path, state="finished", error=None, completed_at=None)
  (tmp_path / "bg_0.events.jsonl").write_text(
    '{"type":"text_delta","ts":8}\n',
    encoding="utf-8",
  )

  record = _registry(tmp_path)._tasks["bg_0"]
  assert record.state == "interrupted"
  assert record.error == "gateway restarted while run was active"
  assert record.completed_at == 8.0


@pytest.mark.parametrize("active_state", ["running", "approval_pending", "queued", "waiting", "remediating"])
def test_rehydrated_active_state_matrix_uses_event_timestamp(
  tmp_path,
  active_state,
) -> None:
  _write_manifest(tmp_path, state=active_state, error=None, completed_at=None)
  (tmp_path / "bg_0.events.jsonl").write_text(
    '{"type":"text_delta","ts":9}\n',
    encoding="utf-8",
  )

  record = _registry(tmp_path)._tasks["bg_0"]
  assert record.state == "interrupted"
  assert record.completed_at == 9.0


@pytest.mark.parametrize("shape", ["escape", "cross_task", "symlink", "hard_link"])
def test_event_evidence_path_and_identity_containment(tmp_path, shape) -> None:
  manifest = _write_manifest(tmp_path, state="completed")
  canonical = tmp_path / "bg_0.events.jsonl"
  outside = tmp_path / "outside.events.jsonl"
  outside.write_text('{"type":"outside"}\n', encoding="utf-8")
  if shape == "escape":
    manifest["events_path"] = str(tmp_path.parent / "escaped.events.jsonl")
  elif shape == "cross_task":
    manifest["events_path"] = str(tmp_path / "bg_1.events.jsonl")
  elif shape == "symlink":
    canonical.symlink_to(outside)
  else:
    os.link(outside, canonical)
  _manifest_path(tmp_path).write_text(json.dumps(manifest) + "\n", encoding="utf-8")

  record = _registry(tmp_path)._tasks["bg_0"]
  assert record.event_lines == []
  assert record.events_evidence_status == (
    "path_mismatch" if shape in {"escape", "cross_task"} else "unreadable"
  )


# --- F-4 zombie-run regression pins (blocker-hunt live battery 2026-08-11) ---
#
# Live incident bg_150: the autonomous child crashed pre-admission (before
# sending any event), the parked sentinel reported the exit, and the reaper
# then died on an unhandled PermissionError from os.killpg (Darwin EPERM
# against a mid-exit process group). Nothing else flips the record, so the
# run stayed state="running" forever and cancel replayed the stored
# PermissionError as an HTTP 500. These tests pin: (1) a child that dies
# without sending events still reaches a terminal failed state carrying its
# real error, (2) killpg EPERM is absorbed as a cleanup failure, (3) even an
# unexpected reaper crash forces a terminal disposition, and (4) cancel
# survives signalling failures instead of erroring.


class _SilentEventChannelParent:
  """Event channel whose child never sends any event before dying."""

  channel_id = "b" * 64

  def __init__(self, child: _FakeEventChannelChild) -> None:
    self._child = child
    self._process: _FakeProcessIdentity | None = None
    self._exited = threading.Event()
    self._closed = threading.Event()

  def attach_process(self, process: _FakeProcessIdentity) -> None:
    self._process = process

  def notify_process_exit(
    self,
    process: _FakeProcessIdentity,
    returncode: int | None,
  ) -> None:
    if process is self._process and returncode is not None:
      self._exited.set()

  def receive_next(
    self,
    *,
    timeout_seconds: float | None = None,
    unbounded_stream: bool | None = None,
  ):
    _ = timeout_seconds, unbounded_stream
    while not self._exited.wait(0.01):
      if self._closed.is_set():
        break
    raise RuntimeError(
      "autonomous event channel peer closed before any event"
    )

  def acknowledge(self, stream, *, timeout_seconds: float | None = None):
    raise AssertionError("silent event channel has no events to acknowledge")

  def close(self) -> None:
    self._closed.set()
    self._child.close_all()

  def interrupt(self) -> None:
    self.close()


class _SilentEventChannelPair:
  def __init__(self) -> None:
    self.child = _FakeEventChannelChild()
    self.parent = _SilentEventChannelParent(self.child)


def _install_silent_event_channel(monkeypatch) -> None:
  from agent_gateway import autonomous_runner_start

  def factory(**_kwargs):
    global _LAST_FAKE_EVENT_CHANNEL
    pair = _SilentEventChannelPair()
    _LAST_FAKE_EVENT_CHANNEL = pair.parent
    return pair

  monkeypatch.setattr(
    autonomous_runner_start,
    "create_autonomous_event_channel",
    factory,
  )


class _ParkedSentinelStderr:
  """Sentinel status pipe: reports the child's exit once, then stays open."""

  def __init__(self, status_line: bytes) -> None:
    self._status_line = status_line
    self._delivered = False
    self._never = asyncio.Event()

  async def readline(self) -> bytes:
    if not self._delivered:
      self._delivered = True
      return self._status_line
    await self._never.wait()
    return b""


class ParkedSentinelFakeProcess(_FakeProcessIdentity):
  """Models the real sentinel: reports the child's exit on its stderr status
  pipe but parks forever afterwards (it ignores SIGTERM by design)."""

  def __init__(self, child_returncode: int = 1) -> None:
    super().__init__()
    self.stderr = _ParkedSentinelStderr(
      json.dumps(
        {"kind": "exited", "returncode": child_returncode, "version": 1},
        sort_keys=True,
        separators=(",", ":"),
      ).encode("utf-8")
      + b"\n"
    )

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


_START_KWARGS = dict(
  role="owner",
  profile="analyst",
  mode="task",
  task="summarize",
)


def test_autonomous_child_death_before_any_events_reaches_failed_terminal_state(
  monkeypatch,
  tmp_path,
) -> None:
  """A child that dies pre-admission, without sending a single event, must
  still drive the run record to a terminal failed state that carries the
  child's real error (its exit code), and cancel afterwards must reconcile
  instead of raising."""

  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return FailingFakeProcess()

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    _install_silent_event_channel(monkeypatch)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      user_id=USER_ID,
      user_email=USER_EMAIL,
      **_START_KWARGS,
    )

    status = await registry.wait(payload["task_id"], timeout_sec=2)

    record = registry._tasks[payload["task_id"]]
    # wait() returns as soon as the record is terminal; the reaper may still
    # be finishing its terminal publish. It must complete without raising.
    await asyncio.wait_for(asyncio.shield(record.reaper_task), timeout=5)
    assert status["state"] == "failed"
    assert record.reaper_task.done()
    assert record.reaper_task.exception() is None
    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "failed"
    assert manifest["exit_code"] == 1
    assert "Process exited with code 1" in manifest["error"]
    assert isinstance(manifest["completed_at"], float)

    cancel_status = await registry.cancel(payload["task_id"])
    assert cancel_status["state"] == "failed"

  asyncio.run(_case())


def test_autonomous_reaper_absorbs_killpg_eperm_and_fails_run(
  monkeypatch,
  tmp_path,
) -> None:
  """bg_150 pin: killpg raising EPERM (Darwin, mid-exit group) during reap
  cleanup must be recorded as a cleanup failure, not crash the reaper and
  strand the run in state="running"."""

  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return ParkedSentinelFakeProcess(child_returncode=1)

    def eperm_killpg(process_group_id: int, signal_number: int) -> None:
      _ = process_group_id, signal_number
      raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setattr(autonomous_runner, "_signal_process_group", eperm_killpg)
    monkeypatch.setattr(autonomous_runner, "_POST_EXIT_SETTLE_SECONDS", 0.05)
    monkeypatch.setattr(autonomous_runner, "_SPAWN_CLEANUP_GRACE_SEC", 0.1)
    _install_silent_event_channel(monkeypatch)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      user_id=USER_ID,
      user_email=USER_EMAIL,
      **_START_KWARGS,
    )

    status = await registry.wait(payload["task_id"], timeout_sec=5)

    record = registry._tasks[payload["task_id"]]
    await asyncio.wait_for(asyncio.shield(record.reaper_task), timeout=5)
    assert status["state"] == "failed"
    assert record.reaper_task.done()
    assert record.reaper_task.exception() is None
    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "failed"
    assert manifest["exit_code"] == 1
    assert "Process exited with code 1" in manifest["error"]
    assert "Operation not permitted" in manifest["error"]

    cancel_status = await registry.cancel(payload["task_id"])
    assert cancel_status["state"] == "failed"

  asyncio.run(_case())


def test_autonomous_reaper_crash_still_commits_terminal_failure(
  monkeypatch,
  tmp_path,
) -> None:
  """Even an unexpected exception escaping the reap flow itself must force
  the record to a terminal failed state instead of leaving a zombie."""

  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return ParkedSentinelFakeProcess(child_returncode=1)

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setattr(autonomous_runner, "_POST_EXIT_SETTLE_SECONDS", 0.05)
    monkeypatch.setattr(autonomous_runner, "_SPAWN_CLEANUP_GRACE_SEC", 0.1)
    _install_silent_event_channel(monkeypatch)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)

    def raw_permission_error(record, signal_number):
      _ = record, signal_number
      raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(
      registry,
      "_signal_owned_process_group",
      raw_permission_error,
    )
    payload = await registry.start(
      user_id=USER_ID,
      user_email=USER_EMAIL,
      **_START_KWARGS,
    )

    status = await registry.wait(payload["task_id"], timeout_sec=5)

    record = registry._tasks[payload["task_id"]]
    await asyncio.wait_for(asyncio.shield(record.reaper_task), timeout=5)
    assert status["state"] == "failed"
    assert record.reaper_task.done()
    assert record.reaper_task.exception() is None
    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "failed"
    assert "reaper crashed" in manifest["error"]
    assert "PermissionError" in manifest["error"]

    cancel_status = await registry.cancel(payload["task_id"])
    assert cancel_status["state"] == "failed"

  asyncio.run(_case())


def test_autonomous_cancel_survives_process_group_signal_failure(
  monkeypatch,
  tmp_path,
) -> None:
  """Cancel must not surface a signalling failure (bg_150's HTTP 500 shape):
  a first EPERM from killpg is folded into the record error and the run
  still settles to its cancelled terminal state."""

  async def _case() -> None:
    from agent_gateway import autonomous_runner

    async def fake_exec(*args, **kwargs):
      _ = args, kwargs
      return SlowFakeProcess()

    calls = {"count": 0}

    def eperm_once_killpg(process_group_id: int, signal_number: int) -> None:
      calls["count"] += 1
      if calls["count"] == 1:
        raise PermissionError(1, "Operation not permitted")
      process = _FAKE_PROCESSES.get(process_group_id)
      if process is None:
        raise ProcessLookupError(process_group_id)
      if process.returncode is not None:
        return
      if signal_number == signal.SIGKILL:
        process.kill()
      else:
        process.terminate()

    monkeypatch.setattr(
      autonomous_runner.asyncio,
      "create_subprocess_exec",
      fake_exec,
    )
    monkeypatch.setattr(
      autonomous_runner,
      "_signal_process_group",
      eperm_once_killpg,
    )
    _install_silent_event_channel(monkeypatch)
    monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
    registry = _registry(tmp_path)
    payload = await registry.start(
      user_id=USER_ID,
      user_email=USER_EMAIL,
      **_START_KWARGS,
    )

    status = await registry.cancel(payload["task_id"])

    record = registry._tasks[payload["task_id"]]
    await asyncio.wait_for(asyncio.shield(record.reaper_task), timeout=5)
    assert status["state"] == "killed"
    assert "Process terminated by user" in (record.error or "")
    assert "process-group terminate failed" in (record.error or "")
    assert record.reaper_task.done()
    assert record.reaper_task.exception() is None
    manifest = _read_manifest(tmp_path)
    assert manifest["state"] == "killed"

    await registry.shutdown(grace_sec=0.1)

  asyncio.run(_case())
