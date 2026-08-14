from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time

import pytest

import agent_gateway.autonomous_claim_broker as broker_module
from agent_gateway.autonomous_claim_broker import (
  AUTONOMOUS_CLAIM_BROKER_FD_ENV,
  AutonomousClaimBroker,
  AutonomousClaimBrokerError,
  AutonomousClaimSigner,
)
from agent_gateway.claim_signing_authority import (
  GatewayClaimSigningAuthority,
)
from agent_gateway.server_artifact_helpers import (
  _verify_agent_claim_headers,
)
from tests.autonomous_exact_test_support import (
  build_exact_autonomous_test_runtime,
)


_SECRET = "exact-autonomous-test-secret-at-least-32-bytes"
_ADVERSARIAL_IO_TIMEOUT_SECONDS = 0.2
_CLAIM_ENV_TO_FIELD = {
  "AGENT_API_CLAIM_AUDIENCE": "audience",
  "AGENT_API_CLAIM_ISSUED_AT": "issued_at",
  "AGENT_API_CLAIM_EXPIRY": "expiry",
  "AGENT_API_CLAIM_USER_ID": "user_id",
  "AGENT_API_CLAIM_USER_EMAIL": "user_email",
  "AGENT_API_CLAIM_NONCE": "nonce",
  "AGENT_API_CLAIM_SIGNATURE": "signature",
}


def _claim_headers(claim: dict[str, str]) -> dict[str, str]:
  return {
    field: claim[env_name]
    for env_name, field in _CLAIM_ENV_TO_FIELD.items()
  }


def test_broker_signer_is_fixed_to_admitted_identity() -> None:
  runtime = build_exact_autonomous_test_runtime()
  try:
    signer = runtime.claim_signer
    assert signer.user_id == "42"
    assert signer.user_email == "owner@example.test"

    claim = signer.sign_claim(ttl_seconds=45)

    verified = _verify_agent_claim_headers(
      _SECRET,
      _claim_headers(claim),
      ttl_ceiling=600,
    )
    assert verified is not None
    assert verified["user_id"] == "42"
    assert verified["user_email"] == "owner@example.test"
  finally:
    runtime.close()


def test_broker_enforces_ttl_and_closes_after_rejection() -> None:
  runtime = build_exact_autonomous_test_runtime(
    broker_max_ttl_seconds=30,
  )
  try:
    with pytest.raises(AutonomousClaimBrokerError):
      runtime.claim_signer.sign_claim(ttl_seconds=31)
    with pytest.raises(AutonomousClaimBrokerError, match="closed"):
      runtime.claim_signer.sign_claim(ttl_seconds=30)
  finally:
    runtime.close()


def test_broker_enforces_request_count_and_closes() -> None:
  runtime = build_exact_autonomous_test_runtime(
    broker_max_requests=1,
  )
  try:
    runtime.claim_signer.sign_claim(ttl_seconds=30)
    with pytest.raises(AutonomousClaimBrokerError):
      runtime.claim_signer.sign_claim(ttl_seconds=30)
    with pytest.raises(AutonomousClaimBrokerError, match="closed"):
      runtime.claim_signer.sign_claim(ttl_seconds=30)
  finally:
    runtime.close()


def test_broker_rejects_any_envelope_other_than_exact_launch() -> None:
  runtime = build_exact_autonomous_test_runtime()
  second_broker = AutonomousClaimBroker(
    GatewayClaimSigningAuthority(_SECRET),
    runtime.envelope_json,
  )
  child_fd = second_broker.take_child_fd()
  payload = json.loads(runtime.envelope_json)
  payload["task_id"] = "different-task"
  changed = json.dumps(
    payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  )
  try:
    with pytest.raises(AutonomousClaimBrokerError):
      AutonomousClaimSigner(
        child_fd,
        envelope_json=changed,
      )
  finally:
    second_broker.close()
    runtime.close()


def test_broker_admission_waits_for_slow_child_boot() -> None:
  runtime = build_exact_autonomous_test_runtime()
  broker = AutonomousClaimBroker(
    GatewayClaimSigningAuthority(_SECRET),
    runtime.envelope_json,
    io_timeout_seconds=_ADVERSARIAL_IO_TIMEOUT_SECONDS,
  )
  child_fd = broker.take_child_fd()
  child_socket = socket.socket(fileno=child_fd)
  try:
    # A child that is still booting the runtime sends nothing at all. The
    # admission channel must stay open well past the per-frame I/O timeout;
    # only the signed session authority bounds the wait.
    time.sleep(_ADVERSARIAL_IO_TIMEOUT_SECONDS * 5)
    assert broker._thread.is_alive()
  finally:
    child_socket.close()
    broker.close()
    runtime.close()


def test_broker_parent_closure_fails_child_signing() -> None:
  runtime = build_exact_autonomous_test_runtime()
  try:
    runtime.claim_broker.close()
    with pytest.raises(AutonomousClaimBrokerError):
      runtime.claim_signer.sign_claim(ttl_seconds=30)
  finally:
    runtime.close()


def test_broker_rejects_unbounded_request_configuration() -> None:
  runtime = build_exact_autonomous_test_runtime()
  try:
    with pytest.raises(ValueError, match="max_requests"):
      AutonomousClaimBroker(
        GatewayClaimSigningAuthority(_SECRET),
        runtime.envelope_json,
        max_requests=(
          broker_module.AUTONOMOUS_CLAIM_BROKER_MAX_REQUESTS
          + 1
        ),
      )
  finally:
    runtime.close()


def _broker_with_controlled_clock(
  monkeypatch: pytest.MonkeyPatch,
) -> tuple[
  object,
  dict[str, float],
  AutonomousClaimBroker,
  AutonomousClaimSigner,
]:
  clock = {
    "wall": broker_module.time.time() + 500.0,
    "monotonic": 500.0,
  }
  monkeypatch.setattr(
    broker_module.time,
    "time",
    lambda: clock["wall"],
  )
  monkeypatch.setattr(
    broker_module.time,
    "monotonic",
    lambda: clock["monotonic"],
  )
  runtime = build_exact_autonomous_test_runtime()
  return (
    runtime,
    clock,
    runtime.claim_broker,
    runtime.claim_signer,
  )


def test_broker_rejects_wall_clock_rollback(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runtime, clock, broker, signer = _broker_with_controlled_clock(
    monkeypatch
  )
  try:
    clock["wall"] -= 1
    clock["monotonic"] += 1
    with pytest.raises(AutonomousClaimBrokerError):
      signer.sign_claim(ttl_seconds=30)
  finally:
    signer.close()
    broker.close()
    runtime.close()


def test_broker_monotonic_deadline_survives_frozen_wall_clock(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runtime, clock, broker, signer = _broker_with_controlled_clock(
    monkeypatch
  )
  try:
    clock["monotonic"] += 101
    with pytest.raises(AutonomousClaimBrokerError):
      signer.sign_claim(ttl_seconds=1)
  finally:
    signer.close()
    broker.close()
    runtime.close()


@pytest.mark.parametrize(
  "fragment",
  (
    b"\x00\x00",
    struct.pack("!I", 32) + b"{",
  ),
  ids=("partial-header", "partial-body"),
)
def test_broker_partial_admission_frame_times_out_and_closes(
  fragment: bytes,
) -> None:
  runtime = build_exact_autonomous_test_runtime()
  broker = AutonomousClaimBroker(
    GatewayClaimSigningAuthority(_SECRET),
    runtime.envelope_json,
    io_timeout_seconds=_ADVERSARIAL_IO_TIMEOUT_SECONDS,
  )
  child_fd = broker.take_child_fd()
  child_socket = socket.socket(fileno=child_fd)
  started = time.monotonic()
  try:
    child_socket.sendall(fragment)
    broker._thread.join(timeout=2.0)
    assert not broker._thread.is_alive()
    assert time.monotonic() - started < 1.5
  finally:
    child_socket.close()
    broker.close()
    runtime.close()


@pytest.mark.parametrize(
  "fragment",
  (
    b"\x00\x00",
    struct.pack("!I", 32) + b"{",
  ),
  ids=("partial-header", "partial-body"),
)
def test_signer_partial_admission_response_times_out_and_closes(
  fragment: bytes,
) -> None:
  server_socket, child_socket = socket.socketpair()
  child_fd = child_socket.detach()
  peer_errors: list[BaseException] = []

  def partial_peer() -> None:
    try:
      broker_module._recv_frame(
        server_socket,
        deadline=time.monotonic() + 1.0,
      )
      server_socket.sendall(fragment)
      time.sleep(_ADVERSARIAL_IO_TIMEOUT_SECONDS * 2)
    except BaseException as exc:
      peer_errors.append(exc)
    finally:
      server_socket.close()

  peer = threading.Thread(target=partial_peer, daemon=True)
  peer.start()
  started = time.monotonic()
  with pytest.raises(
    AutonomousClaimBrokerError,
    match="deadline",
  ):
    AutonomousClaimSigner(
      child_fd,
      envelope_json="{}",
      io_timeout_seconds=_ADVERSARIAL_IO_TIMEOUT_SECONDS,
    )
  assert time.monotonic() - started < 1.5
  with pytest.raises(OSError):
    os.fstat(child_fd)
  peer.join(timeout=2.0)
  assert not peer.is_alive()
  assert peer_errors == []


def test_frame_send_times_out_when_peer_does_not_read() -> None:
  sender, non_reader = socket.socketpair()
  sender.setblocking(False)
  sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
  started = time.monotonic()
  try:
    with pytest.raises(
      AutonomousClaimBrokerError,
      match="deadline",
    ):
      broker_module._send_frame(
        sender,
        {"payload": "x" * (256 * 1024)},
        deadline=(
          time.monotonic()
          + _ADVERSARIAL_IO_TIMEOUT_SECONDS
        ),
      )
    assert time.monotonic() - started < 1.5
  finally:
    sender.close()
    non_reader.close()


def test_take_child_fd_closes_detached_fd_when_hardening_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runtime = build_exact_autonomous_test_runtime()
  broker = AutonomousClaimBroker(
    GatewayClaimSigningAuthority(_SECRET),
    runtime.envelope_json,
  )
  detached_fds: list[int] = []

  def fail_set_inheritable(fd: int, _inheritable: bool) -> None:
    detached_fds.append(fd)
    raise OSError("hardening failed")

  monkeypatch.setattr(
    broker_module.os,
    "set_inheritable",
    fail_set_inheritable,
  )
  try:
    with pytest.raises(OSError, match="hardening failed"):
      broker.take_child_fd()
    assert len(detached_fds) == 1
    with pytest.raises(OSError):
      os.fstat(detached_fds[0])
    assert broker._child_socket is None
  finally:
    broker.close()
    runtime.close()


def test_broker_constructor_closes_both_sockets_on_setup_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runtime = build_exact_autonomous_test_runtime()

  class SetupSocket:
    def __init__(self, *, fail: bool) -> None:
      self.closed = False
      self.fail = fail

    def set_inheritable(self, _inheritable: bool) -> None:
      if self.fail:
        raise OSError("socket setup failed")

    def setblocking(self, _blocking: bool) -> None:
      pass

    def close(self) -> None:
      self.closed = True

  server_socket = SetupSocket(fail=False)
  child_socket = SetupSocket(fail=True)
  monkeypatch.setattr(
    broker_module.socket,
    "socketpair",
    lambda *_args: (server_socket, child_socket),
  )
  try:
    with pytest.raises(OSError, match="socket setup failed"):
      AutonomousClaimBroker(
        GatewayClaimSigningAuthority(_SECRET),
        runtime.envelope_json,
      )
    assert server_socket.closed is True
    assert child_socket.closed is True
  finally:
    runtime.close()


def test_broker_constructor_closes_all_sockets_when_thread_start_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  runtime = build_exact_autonomous_test_runtime()
  original_socketpair = socket.socketpair
  created_sockets: list[socket.socket] = []

  def tracked_socketpair(*args):
    pair = original_socketpair(*args)
    created_sockets.extend(pair)
    return pair

  def fail_thread_start(_thread: threading.Thread) -> None:
    raise RuntimeError("thread start failed")

  monkeypatch.setattr(
    broker_module.socket,
    "socketpair",
    tracked_socketpair,
  )
  monkeypatch.setattr(
    broker_module.threading.Thread,
    "start",
    fail_thread_start,
  )
  try:
    with pytest.raises(RuntimeError, match="thread start failed"):
      AutonomousClaimBroker(
        GatewayClaimSigningAuthority(_SECRET),
        runtime.envelope_json,
      )
    assert len(created_sockets) == 2
    assert all(sock.fileno() == -1 for sock in created_sockets)
  finally:
    runtime.close()


def test_signer_constructor_closes_owned_fd_on_socket_setup_failure(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  server_socket, child_socket = socket.socketpair()
  child_fd = child_socket.detach()

  class FailingSocket:
    def __init__(self, *, fileno: int) -> None:
      assert fileno == child_fd
      self.fd = fileno

    def setblocking(self, _blocking: bool) -> None:
      raise RuntimeError("signer socket setup failed")

    def close(self) -> None:
      os.close(self.fd)

  monkeypatch.setattr(
    broker_module.socket,
    "socket",
    FailingSocket,
  )
  try:
    with pytest.raises(
      AutonomousClaimBrokerError,
      match="descriptor is invalid",
    ):
      AutonomousClaimSigner(
        child_fd,
        envelope_json="{}",
      )
    with pytest.raises(OSError):
      os.fstat(child_fd)
  finally:
    server_socket.close()


@pytest.mark.skipif(
  not Path("/proc/self/environ").exists(),
  reason="Linux /proc regression",
)
def test_exec_child_never_receives_global_key_in_proc_environ() -> None:
  runtime = build_exact_autonomous_test_runtime()
  broker = AutonomousClaimBroker(
    GatewayClaimSigningAuthority(_SECRET),
    runtime.envelope_json,
  )
  child_fd = broker.take_child_fd()
  child_env = dict(os.environ)
  child_env.pop("AGENT_API_USER_CLAIM_HMAC_KEY", None)
  child_env[AUTONOMOUS_CLAIM_BROKER_FD_ENV] = str(child_fd)
  child_env["AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE"] = (
    runtime.envelope_json
  )
  repo_root = Path(__file__).resolve().parents[3]
  child_env["PYTHONPATH"] = os.pathsep.join((
    str(repo_root / "packages" / "agent-gateway"),
    str(repo_root / "api"),
    str(repo_root),
  ))
  source = r"""
import json
import os
from agent_gateway.autonomous_claim_broker import AutonomousClaimSigner

fd = int(os.environ.pop("AGENT_AUTONOMOUS_CLAIM_BROKER_FD"))
envelope = os.environ.pop("AGENT_AUTONOMOUS_CAPABILITY_ENVELOPE")
signer = AutonomousClaimSigner(fd, envelope_json=envelope)
raw = open("/proc/self/environ", "rb").read()
claim = signer.sign_claim(ttl_seconds=30)
print(json.dumps({
  "key_name_present": b"AGENT_API_USER_CLAIM_HMAC_KEY=" in raw,
  "key_value_present": b"exact-autonomous-test-secret-at-least-32-bytes" in raw,
  "broker_fd_inheritable": os.get_inheritable(signer._socket.fileno()),
  "user_id": claim["AGENT_API_CLAIM_USER_ID"],
}, sort_keys=True))
signer.close()
"""
  try:
    completed = subprocess.run(
      [sys.executable, "-c", source],
      cwd=repo_root,
      env=child_env,
      pass_fds=(child_fd,),
      check=True,
      capture_output=True,
      text=True,
      timeout=15,
    )
  finally:
    os.close(child_fd)
    broker.close()
    runtime.close()
  result = json.loads(completed.stdout)
  assert result == {
    "broker_fd_inheritable": False,
    "key_name_present": False,
    "key_value_present": False,
    "user_id": "42",
  }
