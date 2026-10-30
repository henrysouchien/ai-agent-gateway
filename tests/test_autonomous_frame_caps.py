from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import agent_gateway.autonomous_claim_broker as broker_module
import agent_gateway.autonomous_event_channel as event_module
from agent_gateway.autonomous_claim_broker import (
  AutonomousClaimBroker,
  AutonomousClaimBrokerError,
  AutonomousClaimBrokerFrameCapError,
  AutonomousClaimSigner,
)
from agent_gateway.autonomous_event_channel import (
  AutonomousEventChannelBoundsError,
  AutonomousEventChannelFrameCapError,
  AutonomousEventChannelProtocolError,
  create_autonomous_event_channel,
)
from agent_gateway.autonomous_frame_metrics import FrameCapFinding
from agent_gateway.autonomous_launch_envelope import (
  AutonomousControlAuthority,
  AutonomousSessionAuthority,
  sign_autonomous_launch_envelope,
)
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.control_plane.health import build_health_router
from agent_gateway.launch_nonce_store import prepare_launch_nonce_store
from agent_gateway.server_artifact_helpers import _verify_agent_claim_headers
from gateway_test_support.autonomous_launch_test_support import (
  _bind,
  _ordinary_session_authority,
  _workload,
)


_SECRET = "frame-cap-test-signing-secret-at-least-32-bytes"
_TWO_POINT_ONE_MIB = 21 * 1024 * 1024 // 10


def _health_frame_caps(client: TestClient) -> dict[str, Any]:
  response = client.get("/api/control/health")
  assert response.status_code == 200
  return response.json()["frame_caps"]


@pytest.fixture
def health_client():
  app = FastAPI()
  app.include_router(
    build_health_router(route_prefix="/api"),
    prefix="/api/control",
  )
  with TestClient(app) as client:
    yield client


def _signed_envelope(tmp_path: Path, *, task: str) -> str:
  now_ns = time.time_ns()
  now_seconds = now_ns // 1_000_000_000
  ordinary = _ordinary_session_authority().ordinary_authority
  session_authority = AutonomousSessionAuthority.ordinary(
    replace(
      ordinary,
      created_at=now_seconds - 1,
      expires_at=now_seconds + 600,
    ),
  )
  inbox = tmp_path / "operator-messages.jsonl"
  inbox.touch()
  inbox_stat = inbox.stat()
  ledger = prepare_launch_nonce_store(tmp_path / "admissions.sqlite3")
  return sign_autonomous_launch_envelope(
    _SECRET,
    task_id="bg_7",
    control_run_id="run-7",
    owner_user_id="42",
    channel_id="ab" * 32,
    bind=_bind(),
    workload=_workload(mode="task", task=task),
    control_authority=AutonomousControlAuthority(
      control_mode="file",
      admission_ledger_path=ledger.path,
      admission_ledger_device=ledger.device,
      admission_ledger_inode=ledger.inode,
      operator_inbox_path=str(inbox),
      operator_inbox_device=inbox_stat.st_dev,
      operator_inbox_inode=inbox_stat.st_ino,
    ),
    session_authority=session_authority,
    now_ns=now_ns,
  )


def test_signed_launch_above_event_cap_is_admitted_and_signs_claim(
  tmp_path: Path,
) -> None:
  envelope_json = _signed_envelope(tmp_path, task="x" * _TWO_POINT_ONE_MIB)
  admission_body = broker_module._canonical_json({
    "envelope": envelope_json,
    "op": "admit",
    "version": broker_module.AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION,
  })
  assert len(admission_body) >= _TWO_POINT_ONE_MIB
  assert len(admission_body) > event_module.AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES
  assert len(admission_body) < broker_module.AUTONOMOUS_CLAIM_BROKER_MAX_FRAME_BYTES

  broker = AutonomousClaimBroker(
    GatewayClaimSigningAuthority(_SECRET),
    envelope_json,
  )
  try:
    signer = AutonomousClaimSigner(
      broker.take_child_fd(),
      envelope_json=envelope_json,
    )
    try:
      claim = signer.sign_claim(ttl_seconds=45)
      verified = _verify_agent_claim_headers(_SECRET, {
        "audience": claim["AGENT_API_CLAIM_AUDIENCE"],
        "issued_at": claim["AGENT_API_CLAIM_ISSUED_AT"],
        "expiry": claim["AGENT_API_CLAIM_EXPIRY"],
        "user_id": claim["AGENT_API_CLAIM_USER_ID"],
        "user_email": claim["AGENT_API_CLAIM_USER_EMAIL"],
        "nonce": claim["AGENT_API_CLAIM_NONCE"],
        "signature": claim["AGENT_API_CLAIM_SIGNATURE"],
      })
      assert verified is not None
      assert verified["user_id"] == "42"
      assert verified["user_email"] == "owner@example.test"
    finally:
      signer.close()
  finally:
    broker.close()


def test_declared_event_frame_cap_precedes_body_read_and_reaches_health(
  monkeypatch: pytest.MonkeyPatch,
  health_client: TestClient,
) -> None:
  before = event_module.frame_cap_metrics.snapshot()
  health_before = _health_frame_caps(health_client)["event_channel"]["count"]
  assert health_before == before.count
  read_sizes: list[int] = []
  recv_exact = event_module._recv_exact

  def record_read(sock: socket.socket, byte_count: int, **kwargs: Any):
    read_sizes.append(byte_count)
    return recv_exact(sock, byte_count, **kwargs)

  monkeypatch.setattr(event_module, "_recv_exact", record_read)
  with create_autonomous_event_channel() as pair:
    pair.child.start(timeout_seconds=1)
    with socket.socket(fileno=os.dup(pair.child.fileno())) as raw_child:
      # HELLO is real; the next frame is the EVENT slot. Keep the peer open
      # without sending its body so a body read would fail on the deadline.
      raw_child.sendall(event_module._FRAME_HEADER.pack(_TWO_POINT_ONE_MIB))
      with pytest.raises(AutonomousEventChannelFrameCapError) as caught:
        pair.parent.receive(timeout_seconds=1)

  assert isinstance(caught.value, AutonomousEventChannelBoundsError)
  finding = caught.value.finding
  assert type(finding) is FrameCapFinding
  assert finding.channel == "event_channel"
  assert finding.direction == "receive"
  assert finding.size_basis == "declared"
  assert finding.frame_size_bytes == event_module._FRAME_HEADER.size + _TWO_POINT_ONE_MIB
  assert finding.cap_bytes == event_module.AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES
  assert len(read_sizes) == 3  # HELLO header/body, then only the EVENT header.
  assert read_sizes[0] == read_sizes[-1] == event_module._FRAME_HEADER.size
  assert _TWO_POINT_ONE_MIB not in read_sizes
  after = event_module.frame_cap_metrics.snapshot()
  assert after.count == before.count + 1
  assert after.last_finding == finding
  health_after = _health_frame_caps(health_client)["event_channel"]
  assert health_after["count"] == health_before + 1
  assert health_after["last_finding"] == finding.model_dump()


def test_declared_claim_frame_cap_precedes_body_read_and_reaches_health(
  monkeypatch: pytest.MonkeyPatch,
  health_client: TestClient,
) -> None:
  before = broker_module.frame_cap_metrics.snapshot()
  health_before = _health_frame_caps(health_client)["claim_broker"]["count"]
  assert health_before == before.count
  declared_body_bytes = 7 * 1024 * 1024
  read_sizes: list[int] = []
  recv_exact = broker_module._recv_exact

  def record_read(sock: socket.socket, length: int, **kwargs: Any):
    read_sizes.append(length)
    return recv_exact(sock, length, **kwargs)

  monkeypatch.setattr(broker_module, "_recv_exact", record_read)
  parent, child = socket.socketpair()
  try:
    # Exercise the broker's real receiver on a private socket with no body.
    # Keeping the peer open distinguishes cap refusal from EOF or timeout.
    child.sendall(struct.pack("!I", declared_body_bytes))
    with pytest.raises(AutonomousClaimBrokerFrameCapError) as caught:
      broker_module._recv_frame(parent, deadline=time.monotonic() + 1)
  finally:
    parent.close()
    child.close()

  assert isinstance(caught.value, AutonomousClaimBrokerError)
  finding = caught.value.finding
  assert type(finding) is FrameCapFinding
  assert finding.channel == "claim_broker"
  assert finding.direction == "receive"
  assert finding.size_basis == "declared"
  assert finding.frame_size_bytes == declared_body_bytes
  assert finding.cap_bytes == broker_module.AUTONOMOUS_CLAIM_BROKER_MAX_FRAME_BYTES
  assert read_sizes == [4]
  after = broker_module.frame_cap_metrics.snapshot()
  assert after.count == before.count + 1
  assert after.last_finding == finding
  health_after = _health_frame_caps(health_client)["claim_broker"]
  assert health_after["count"] == health_before + 1
  assert health_after["last_finding"] == finding.model_dump()


@pytest.mark.parametrize("failure_site", ["pack", "write"])
def test_oversized_claim_send_preserves_typed_error_on_prefix_failure(
  monkeypatch: pytest.MonkeyPatch,
  failure_site: str,
) -> None:
  failure = OSError("prefix unavailable")
  writes: list[bytes] = []

  def fail_pack(*args: Any) -> bytes:
    raise failure

  def fail_write(sock: socket.socket, payload: bytes, **kwargs: Any) -> None:
    writes.append(payload)
    raise failure

  monkeypatch.setattr(broker_module, "_send_all", fail_write)
  if failure_site == "pack":
    monkeypatch.setattr(
      broker_module, "struct", SimpleNamespace(pack=fail_pack, unpack=struct.unpack),
    )
  assert struct.pack("!I", 1) == b"\x00\x00\x00\x01"
  payload = "x" * (broker_module.AUTONOMOUS_CLAIM_BROKER_MAX_FRAME_BYTES + 1)
  actual_body_bytes = len(broker_module._canonical_json(payload))
  parent, child = socket.socketpair()
  try:
    with pytest.raises(AutonomousClaimBrokerFrameCapError) as caught:
      broker_module._send_frame(
        child, payload, deadline=time.monotonic() + 1,
      )
  finally:
    parent.close()
    child.close()

  assert caught.value.__cause__ is failure
  assert caught.value.finding.frame_size_bytes == actual_body_bytes
  assert caught.value.finding.size_basis == "actual"
  if failure_site == "pack":
    assert writes == []
  else:
    assert writes == [actual_body_bytes.to_bytes(4, "big")]


def test_oversized_child_admission_reaches_parent_health_without_body(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  health_client: TestClient,
) -> None:
  cap = broker_module.AUTONOMOUS_CLAIM_BROKER_MAX_FRAME_BYTES
  envelope_json = _signed_envelope(tmp_path, task="x" * (cap + 1))
  admission_body = broker_module._canonical_json({
    "envelope": envelope_json,
    "op": "admit",
    "version": broker_module.AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION,
  })
  actual_body_bytes = len(admission_body)
  assert actual_body_bytes > cap
  envelope_path = tmp_path / "launch-envelope.json"
  envelope_path.write_text(envelope_json, encoding="utf-8")
  health_before = _health_frame_caps(health_client)["claim_broker"]["count"]
  read_sizes: list[int] = []
  recv_exact = broker_module._recv_exact

  def record_read(sock: socket.socket, length: int, **kwargs: Any):
    read_sizes.append(length)
    return recv_exact(sock, length, **kwargs)

  monkeypatch.setattr(broker_module, "_recv_exact", record_read)
  broker = AutonomousClaimBroker(
    GatewayClaimSigningAuthority(_SECRET),
    envelope_json,
  )
  child_fd = broker.take_child_fd()
  try:
    child = subprocess.run(
      [
        sys.executable,
        "-c",
        """
import json
import os
from pathlib import Path
import sys
import socket
import time
from agent_gateway.autonomous_claim_broker import (
  AutonomousClaimBrokerFrameCapError,
  AutonomousClaimSigner,
  _recv_frame,
)
peer = socket.socket(fileno=os.dup(int(sys.argv[2])))
try:
  signer = AutonomousClaimSigner(
    int(sys.argv[2]),
    envelope_json=Path(sys.argv[1]).read_text(encoding="utf-8"),
  )
except AutonomousClaimBrokerFrameCapError as exc:
  try:
    rejection = _recv_frame(peer, deadline=time.monotonic() + 5)
    peer.settimeout(5)
    eof = peer.recv(1) == b""
  finally:
    peer.close()
  print(json.dumps({
    "frame_size_bytes": exc.finding.frame_size_bytes,
    "cap_bytes": exc.finding.cap_bytes,
    "rejection_code": rejection["error"],
    "eof": eof,
  }))
else:
  signer.close()
  peer.close()
  raise SystemExit("oversized admission unexpectedly succeeded")
""",
        str(envelope_path),
        str(child_fd),
      ],
      pass_fds=(child_fd,),
      capture_output=True,
      text=True,
      timeout=30,
      cwd=Path(__file__).resolve().parents[1],
      check=True,
    )
  finally:
    os.close(child_fd)
    broker.close()

  assert child.stderr == ""
  assert json.loads(child.stdout) == {
    "frame_size_bytes": actual_body_bytes,
    "cap_bytes": cap,
    "rejection_code": "broker_rejected_request",
    "eof": True,
  }
  assert sum(read_sizes) == struct.calcsize("!I")
  assert broker.claims_served == 0
  health_after = _health_frame_caps(health_client)["claim_broker"]
  assert health_after["count"] == health_before + 1
  assert health_after["last_finding"] == {
    "channel": "claim_broker",
    "direction": "receive",
    "size_basis": "declared",
    "frame_size_bytes": actual_body_bytes,
    "cap_bytes": cap,
  }


def _minimum_event_frame_size(event: dict[str, Any]) -> int:
  body = json.dumps(
    {
      "kind": "EVENT",
      "version": 1,
      "channel_id": "ab" * 32,
      "seq": 0,
      "event": event,
    },
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")
  return 4 + len(body)


@pytest.mark.parametrize("event", [
  {"type": "text_delta", "text": "x" * 1024},
  {
    "type": "tool_call_complete",
    "result": {"prefix": [9, -99, True, None, 1.25], "text": "é" * 1024},
  },
  {
    "type": "tool_call_complete",
    "result": [{"prefix": [9]}, {"x" * 1024: None}],
  },
])
def test_event_byte_preflight_reports_uncounted_typed_lower_bound(
  monkeypatch: pytest.MonkeyPatch,
  event: dict[str, Any],
) -> None:
  monkeypatch.setattr(event_module, "AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES", 512)
  monkeypatch.setattr(event_module, "AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES", 512)
  actual_minimum_frame_size = _minimum_event_frame_size(event)
  before = event_module.frame_cap_metrics.snapshot()

  def reject_render(*args: Any, **kwargs: Any) -> str:
    raise AssertionError("byte preflight must not render the rejected event")

  monkeypatch.setattr(event_module.json, "dumps", reject_render)
  with pytest.raises(AutonomousEventChannelFrameCapError) as caught:
    event_module.snapshot_autonomous_event(event)

  finding = caught.value.finding
  assert finding.channel == "event_channel"
  assert finding.direction == "send"
  assert finding.size_basis == "lower_bound"
  assert finding.cap_bytes == 512
  assert finding.cap_bytes < finding.frame_size_bytes <= actual_minimum_frame_size
  assert event_module.frame_cap_metrics.snapshot() == before


def test_post_snapshot_fit_refusal_reports_minimum_seq_envelope(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(event_module, "AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES", 512)
  monkeypatch.setattr(event_module, "AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES", 512)
  event = {"type": "text_delta", "text": "x" * 400}
  before = event_module.frame_cap_metrics.snapshot()

  with pytest.raises(AutonomousEventChannelFrameCapError) as caught:
    event_module.snapshot_autonomous_event(event)

  finding = caught.value.finding
  assert finding.model_dump() == {
    "channel": "event_channel",
    "direction": "send",
    "size_basis": "lower_bound",
    "frame_size_bytes": _minimum_event_frame_size(event),
    "cap_bytes": 512,
  }
  assert event_module.frame_cap_metrics.snapshot() == before


@pytest.mark.parametrize("text_size", [400, 1024])
def test_event_fit_probe_does_not_record_frame_cap_findings(
  monkeypatch: pytest.MonkeyPatch,
  text_size: int,
) -> None:
  monkeypatch.setattr(event_module, "AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES", 512)
  monkeypatch.setattr(event_module, "AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES", 512)
  before = event_module.frame_cap_metrics.snapshot()

  assert not event_module.autonomous_event_fits_frame({
    "type": "text_delta",
    "text": "x" * text_size,
  })
  assert event_module.frame_cap_metrics.snapshot() == before


def test_event_node_count_bound_does_not_produce_frame_cap_finding(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(event_module, "_MAX_JSON_VALUE_NODES", 4)
  before = event_module.frame_cap_metrics.snapshot()

  with pytest.raises(AutonomousEventChannelBoundsError) as caught:
    event_module.snapshot_autonomous_event({
      "type": "text_delta",
      "text": [1, 2, 3],
    })

  assert not isinstance(caught.value, AutonomousEventChannelFrameCapError)
  assert event_module.frame_cap_metrics.snapshot() == before


@pytest.mark.parametrize("value", [9, -9, 99, -99, 10**100 - 1])
def test_integer_byte_refusal_does_not_report_digit_upper_estimate(
  value: int,
) -> None:
  actual_size = len(str(value))
  with pytest.raises(event_module._CanonicalJsonByteBoundsError) as caught:
    event_module._canonical_json_bytes(value, max_bytes=actual_size)

  assert 0 < caught.value.minimum_size_bytes <= actual_size


def test_wire_frame_byte_preflight_constructs_uncounted_typed_finding(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(event_module, "AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES", 512)
  payload = {"kind": "PRIVATE_EXECUTABLE_CAPTURE", "payload": ["x" * 1024]}
  actual_size = 4 + len(json.dumps(
    payload,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8"))
  before = event_module.frame_cap_metrics.snapshot()

  with pytest.raises(AutonomousEventChannelFrameCapError) as caught:
    event_module._encode_wire_frame(payload)

  finding = caught.value.finding
  assert finding.direction == "send"
  assert finding.size_basis == "lower_bound"
  assert finding.cap_bytes == 512
  assert finding.cap_bytes < finding.frame_size_bytes <= actual_size
  assert event_module.frame_cap_metrics.snapshot() == before


def test_small_frame_cap_finding_event_roundtrip_records_once_and_completes(
  health_client: TestClient,
) -> None:
  before = event_module.frame_cap_metrics.snapshot()
  health_before = _health_frame_caps(health_client)["event_channel"]["count"]
  assert health_before == before.count
  with pytest.raises(AutonomousEventChannelFrameCapError) as caught:
    event_module.snapshot_autonomous_event({
      "type": "text_delta",
      "text": "x" * _TWO_POINT_ONE_MIB,
    })
  finding = caught.value.finding
  finding_event = {"type": "frame_cap_finding", "finding": finding.model_dump()}
  assert _minimum_event_frame_size(finding_event) < 1024
  assert event_module.frame_cap_metrics.snapshot() == before
  child_result: queue.Queue[object] = queue.Queue()

  with create_autonomous_event_channel(io_timeout_seconds=2) as pair:
    def finish_child() -> None:
      try:
        pair.child.start(timeout_seconds=2)
        pair.child.send_event(finding_event, timeout_seconds=2)
        pair.child.send_event(
          {"type": "text_delta", "text": "after dropped event"},
          timeout_seconds=2,
        )
        child_result.put(pair.child.complete(
          {"type": "stream_complete"},
          timeout_seconds=2,
        ))
      except BaseException as exc:
        child_result.put(exc)

    worker = threading.Thread(target=finish_child, daemon=True)
    worker.start()
    stream = pair.parent.receive(timeout_seconds=2)
    parent_ack = pair.parent.acknowledge(stream, timeout_seconds=2)
    worker.join(timeout=2)
    assert not worker.is_alive()
    child_ack = child_result.get_nowait()
    assert not isinstance(child_ack, BaseException)
    assert child_ack == parent_ack

  assert stream.events == (
    finding_event,
    {"type": "text_delta", "text": "after dropped event"},
    {"type": "stream_complete"},
  )
  assert stream.event_count == 3
  after = event_module.frame_cap_metrics.snapshot()
  assert after.count == before.count + 1
  assert after.last_finding == finding
  health_after = _health_frame_caps(health_client)["event_channel"]
  assert health_after["count"] == health_before + 1
  assert health_after["last_finding"] == finding.model_dump()


@pytest.mark.parametrize("event", [
  {"type": "frame_cap_finding", "finding": {}},
  {
    "type": "frame_cap_finding",
    "finding": {
      "channel": "event_channel",
      "direction": "send",
      "size_basis": "lower_bound",
      "frame_size_bytes": "2097153",
      "cap_bytes": 2097152,
    },
  },
  {
    "type": "frame_cap_finding",
    "finding": {
      "channel": "event_channel",
      "direction": "send",
      "size_basis": "lower_bound",
      "frame_size_bytes": 2097153,
      "cap_bytes": 2097152,
      "unexpected": True,
    },
  },
])
def test_parent_rejects_invalid_frame_cap_finding_without_counting(
  event: dict[str, Any],
) -> None:
  before = event_module.frame_cap_metrics.snapshot()
  with create_autonomous_event_channel(io_timeout_seconds=1) as pair:
    pair.child.start(timeout_seconds=1)
    pair.child.send_event(event, timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelProtocolError, match="Finding event"):
      pair.parent.receive(timeout_seconds=1)

  assert event_module.frame_cap_metrics.snapshot() == before


@pytest.mark.parametrize("value", [
  [9, ""],
  {"prefix": [9, ""]},
  [9, None],
  [9, False],
  [9, 1.25],
  [9, []],
  [9, {"": None}],
])
def test_byte_refusal_prefix_uses_minimum_not_estimated_integer_digits(
  value: Any,
) -> None:
  actual_size = len(json.dumps(
    value,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8"))
  with pytest.raises(event_module._CanonicalJsonByteBoundsError) as caught:
    event_module._canonical_json_bytes(value, max_bytes=actual_size)

  assert caught.value.minimum_size_bytes == actual_size


def test_private_capture_byte_drop_forwards_finding_before_abort(
  monkeypatch: pytest.MonkeyPatch,
  health_client: TestClient,
) -> None:
  before = event_module.frame_cap_metrics.snapshot()
  health_before = _health_frame_caps(health_client)["event_channel"]["count"]
  assert health_before == before.count
  payload = {"tool_name": "record_note", "input": {"text": "x" * _TWO_POINT_ONE_MIB}}
  render = event_module.json.dumps

  def reject_capture_render(value: Any, *args: Any, **kwargs: Any) -> str:
    if type(value) is dict and value.get("kind") == "PRIVATE_EXECUTABLE_CAPTURE":
      raise AssertionError("rejected capture must not be rendered to measure it")
    return render(value, *args, **kwargs)

  monkeypatch.setattr(event_module.json, "dumps", reject_capture_render)
  with create_autonomous_event_channel(io_timeout_seconds=1) as pair:
    pair.child.start(timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelFrameCapError) as caught:
      pair.child.send_private_executable_capture(payload, timeout_seconds=1)

    finding = caught.value.finding
    assert finding.channel == "event_channel"
    assert finding.direction == "send"
    assert finding.size_basis == "lower_bound"
    assert finding.frame_size_bytes > finding.cap_bytes
    assert pair.child.fileno() == -1
    assert event_module.frame_cap_metrics.snapshot() == before
    record = pair.parent.receive_next(timeout_seconds=1, unbounded_stream=False)
    assert isinstance(record, event_module.AutonomousEventRecord)
    assert record.event == {
      "type": "frame_cap_finding",
      "finding": finding.model_dump(),
    }
    after_finding = event_module.frame_cap_metrics.snapshot()
    assert after_finding.count == before.count + 1
    assert after_finding.last_finding == finding
    health_after = _health_frame_caps(health_client)["event_channel"]
    assert health_after["count"] == health_before + 1
    assert health_after["last_finding"] == finding.model_dump()

    with pytest.raises(AutonomousEventChannelProtocolError, match="closed before END"):
      pair.parent.receive_next()

  assert event_module.frame_cap_metrics.snapshot() == after_finding


def test_private_capture_node_bound_aborts_without_finding(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(event_module, "_MAX_JSON_VALUE_NODES", 4)
  before = event_module.frame_cap_metrics.snapshot()
  with create_autonomous_event_channel(io_timeout_seconds=1) as pair:
    pair.child.start(timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelBoundsError) as caught:
      pair.child.send_private_executable_capture({"input": [1, 2, 3]}, timeout_seconds=1)

    assert not isinstance(caught.value, AutonomousEventChannelFrameCapError)
    assert pair.child.fileno() == -1
    with pytest.raises(AutonomousEventChannelProtocolError, match="closed before END"):
      pair.parent.receive_next(timeout_seconds=1, unbounded_stream=False)

  assert event_module.frame_cap_metrics.snapshot() == before


def test_private_capture_preserves_original_cap_error_if_finding_send_fails(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  diagnostic_error = event_module.AutonomousEventChannelTransportError(
    "Finding transport failed"
  )

  def fail_finding(*args: Any, **kwargs: Any) -> None:
    raise diagnostic_error

  monkeypatch.setattr(event_module.AutonomousEventChannelChild, "_send_event", fail_finding)
  before = event_module.frame_cap_metrics.snapshot()
  with create_autonomous_event_channel(io_timeout_seconds=1) as pair:
    pair.child.start(timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelFrameCapError) as caught:
      pair.child.send_private_executable_capture({
        "input": {"text": "x" * _TWO_POINT_ONE_MIB},
      }, timeout_seconds=1)

    assert caught.value.__cause__ is diagnostic_error
    assert caught.value.finding.size_basis == "lower_bound"
    assert caught.value.finding.frame_size_bytes > caught.value.finding.cap_bytes
    assert pair.child.fileno() == -1

  assert event_module.frame_cap_metrics.snapshot() == before


@pytest.mark.parametrize("refusal", ["widest_seq", "integer_estimate"])
def test_raw_event_byte_preflight_refusal_counts_receive_finding_once(
  monkeypatch: pytest.MonkeyPatch,
  health_client: TestClient,
  refusal: str,
) -> None:
  cap = event_module.AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES
  event: dict[str, Any]
  if refusal == "widest_seq":
    event = {"type": "text_delta", "text": ""}
    event["text"] = "x" * (cap - 1 - _minimum_event_frame_size(event))
  else:
    # 9 is one canonical byte, but the existing estimator budgets two digits.
    # Fill the actual frame to the declared cap without changing that refusal.
    event = {"type": "tool_call_complete", "result": [9, ""]}
    event["result"][1] = "x" * (cap - _minimum_event_frame_size(event))
  body = json.dumps(
    {
      "kind": "EVENT",
      "version": 1,
      "channel_id": "ab" * 32,
      "seq": 0,
      "event": event,
    },
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")
  frame = struct.pack(">I", len(body)) + body
  assert len(frame) <= cap
  before = event_module.frame_cap_metrics.snapshot()
  health_before = _health_frame_caps(health_client)["event_channel"]["count"]
  sender_result: queue.Queue[object] = queue.Queue()

  if refusal == "integer_estimate":
    render = event_module.json.dumps

    def reject_event_frame_render(value: Any, *args: Any, **kwargs: Any) -> str:
      if type(value) is dict and value.get("kind") == "EVENT":
        raise AssertionError("rejected raw EVENT must fail before rendering")
      return render(value, *args, **kwargs)

    monkeypatch.setattr(event_module.json, "dumps", reject_event_frame_render)

  with create_autonomous_event_channel(
    channel_id="ab" * 32,
    io_timeout_seconds=2,
  ) as pair:
    pair.child.start(timeout_seconds=2)
    with socket.socket(fileno=os.dup(pair.child.fileno())) as raw_child:
      raw_child.settimeout(2)

      def send_raw_event() -> None:
        try:
          raw_child.sendall(frame)
          sender_result.put(None)
        except BaseException as exc:
          sender_result.put(exc)

      worker = threading.Thread(target=send_raw_event, daemon=True)
      worker.start()
      try:
        with pytest.raises(AutonomousEventChannelFrameCapError) as caught:
          pair.parent.receive(timeout_seconds=2)
      finally:
        worker.join(timeout=2)
      assert not worker.is_alive()
      assert sender_result.get_nowait() is None

  finding = caught.value.finding
  assert finding.model_dump() == {
    "channel": "event_channel",
    "direction": "receive",
    "size_basis": "declared",
    "frame_size_bytes": len(frame),
    "cap_bytes": cap,
  }
  after = event_module.frame_cap_metrics.snapshot()
  assert after.count == before.count + 1
  assert after.last_finding is finding
  health_after = _health_frame_caps(health_client)["event_channel"]
  assert health_after["count"] == health_before + 1
  assert health_after["last_finding"] == finding.model_dump()


def test_raw_event_node_preflight_refusal_is_not_counted_as_frame_cap(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(event_module, "_MAX_JSON_VALUE_NODES", 12)
  body = json.dumps(
    {
      "kind": "EVENT",
      "version": 1,
      "channel_id": "ab" * 32,
      "seq": 0,
      "event": {"type": "text_delta", "text": [0] * 10},
    },
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")
  before = event_module.frame_cap_metrics.snapshot()
  with create_autonomous_event_channel(channel_id="ab" * 32) as pair:
    pair.child.start(timeout_seconds=1)
    with socket.socket(fileno=os.dup(pair.child.fileno())) as raw_child:
      raw_child.sendall(struct.pack(">I", len(body)) + body)
      with pytest.raises(AutonomousEventChannelBoundsError) as caught:
        pair.parent.receive(timeout_seconds=1)

  assert not isinstance(caught.value, AutonomousEventChannelFrameCapError)
  assert event_module.frame_cap_metrics.snapshot() == before
