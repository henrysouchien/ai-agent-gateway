from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import queue
import socket
import struct
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import agent_gateway.autonomous_event_channel as channel
from agent_gateway.autonomous_event_channel import (
  AUTONOMOUS_EVENT_CHANNEL_DIGEST_DOMAIN,
  AUTONOMOUS_EVENT_CHANNEL_FD_ENV,
  AutonomousEventChannelAcknowledgementError,
  AutonomousEventChannelBoundsError,
  AutonomousEventChannelError,
  AutonomousEventChannelProtocolError,
  AutonomousEventChannelStateError,
  AutonomousEventChannelTimeout,
  AutonomousEventChannelTransportError,
  ReceivedAutonomousEventStream,
  adopt_inherited_autonomous_event_channel,
  create_autonomous_event_channel,
  snapshot_autonomous_event,
)


_CHANNEL_ID = "ab" * 32
_OTHER_CHANNEL_ID = "cd" * 32
_HEADER = struct.Struct(">I")


def test_fd_environment_variable_name_is_the_exported_single_source() -> None:
  assert AUTONOMOUS_EVENT_CHANNEL_FD_ENV == "AGENT_AUTONOMOUS_EVENT_CHANNEL_FD"
  assert "AUTONOMOUS_EVENT_CHANNEL_FD_ENV" in channel.__all__


def _canonical_frame(payload: dict[str, Any]) -> bytes:
  body = json.dumps(
    payload,
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")
  return _HEADER.pack(len(body)) + body


def _hello(*, channel_id: str = _CHANNEL_ID) -> dict[str, Any]:
  return {
    "kind": "HELLO",
    "version": 1,
    "channel_id": channel_id,
  }


def _event(
  seq: int,
  event: dict[str, Any],
  *,
  channel_id: str = _CHANNEL_ID,
) -> dict[str, Any]:
  return {
    "kind": "EVENT",
    "version": 1,
    "channel_id": channel_id,
    "seq": seq,
    "event": event,
  }


def _event_digest(*frames: bytes) -> str:
  digest = hashlib.sha256()
  digest.update(AUTONOMOUS_EVENT_CHANNEL_DIGEST_DOMAIN)
  for frame in frames:
    digest.update(frame)
  return digest.hexdigest()


def _end(
  event_count: int,
  event_digest: str,
  *,
  channel_id: str = _CHANNEL_ID,
) -> dict[str, Any]:
  return {
    "kind": "END",
    "version": 1,
    "channel_id": channel_id,
    "event_count": event_count,
    "event_digest": event_digest,
  }


def _ack(
  event_count: int,
  event_digest: str,
  *,
  channel_id: str = _CHANNEL_ID,
) -> dict[str, Any]:
  return {
    "kind": "ACK",
    "version": 1,
    "channel_id": channel_id,
    "event_count": event_count,
    "event_digest": event_digest,
  }


def _duplicate_endpoint(endpoint) -> socket.socket:
  duplicated_fd = os.dup(endpoint.fileno())
  raw = socket.socket(fileno=duplicated_fd)
  endpoint.close()
  return raw


def _raw_recv_frame(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
  header = b""
  while len(header) < _HEADER.size:
    chunk = sock.recv(_HEADER.size - len(header))
    if not chunk:
      raise AssertionError("unexpected EOF before frame header")
    header += chunk
  (body_length,) = _HEADER.unpack(header)
  body = b""
  while len(body) < body_length:
    chunk = sock.recv(body_length - len(body))
    if not chunk:
      raise AssertionError("unexpected EOF before frame body")
    body += chunk
  return json.loads(body), header + body


def _run_parent_ack(
  parent,
  result: queue.Queue[object],
  *,
  acknowledge: bool = True,
) -> None:
  try:
    stream = parent.receive(timeout_seconds=2)
    if acknowledge:
      ack = parent.acknowledge(stream, timeout_seconds=2)
      result.put((stream, ack))
    else:
      result.put(stream)
  except BaseException as exc:
    result.put(exc)


def _happy_stream(
  nonterminal_events: list[dict[str, Any]],
  terminal_event: dict[str, Any],
) -> tuple[ReceivedAutonomousEventStream, object, object]:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=2,
  )
  result: queue.Queue[object] = queue.Queue()
  worker = threading.Thread(
    target=_run_parent_ack,
    args=(pair.parent, result),
    daemon=True,
  )
  worker.start()
  try:
    pair.child.start(timeout_seconds=2)
    for event in nonterminal_events:
      pair.child.send_event(event, timeout_seconds=2)
    child_ack = pair.child.complete(terminal_event, timeout_seconds=2)
    worker.join(timeout=2)
    assert not worker.is_alive()
    parent_result = result.get_nowait()
    if isinstance(parent_result, BaseException):
      raise parent_result
    stream, parent_ack = parent_result
    return stream, parent_ack, child_ack
  finally:
    pair.close()


def _raw_parent_pair() -> tuple[object, socket.socket]:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=1,
  )
  raw_child = _duplicate_endpoint(pair.child)
  return pair.parent, raw_child


def _send_raw_stream(
  raw_child: socket.socket,
  frames: list[bytes],
) -> None:
  raw_child.sendall(b"".join(frames))
  raw_child.shutdown(socket.SHUT_WR)


def test_round_trip_preserves_exact_frames_and_binds_ack() -> None:
  stream, parent_ack, child_ack = _happy_stream(
    [
      {"type": "text_delta", "text": "héllo"},
      {"type": "tool_call_complete", "tool_name": "file_read"},
    ],
    {"type": "stream_complete", "usage": {"input_tokens": 7}},
  )

  assert stream.event_count == 3
  assert [record.seq for record in stream.records] == [0, 1, 2]
  assert [event["type"] for event in stream.events] == [
    "text_delta",
    "tool_call_complete",
    "stream_complete",
  ]
  assert stream.exact_event_frame_bytes == sum(
    len(frame) for frame in stream.raw_event_frames
  )
  assert stream.event_digest == _event_digest(*stream.raw_event_frames)
  assert parent_ack == child_ack
  assert child_ack.event_count == 3
  assert child_ack.event_digest == stream.event_digest
  for record in stream.records:
    assert record.event_line_bytes.endswith(b"\n")
    assert json.loads(record.event_line_bytes) == record.event


def test_error_is_an_allowed_final_terminal_event() -> None:
  stream, _, _ = _happy_stream(
    [{"type": "text_delta", "text": "partial"}],
    {"type": "error", "message": "provider failed"},
  )
  assert stream.events[-1]["type"] == "error"


def test_receive_does_not_ack_before_explicit_delivery_boundary() -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=2,
  )
  completed = threading.Event()
  child_result: queue.Queue[object] = queue.Queue()

  def finish_child() -> None:
    try:
      child_result.put(pair.child.complete(
        {"type": "stream_complete"},
        timeout_seconds=2,
      ))
    except BaseException as exc:
      child_result.put(exc)
    finally:
      completed.set()

  try:
    pair.child.start(timeout_seconds=2)
    worker = threading.Thread(target=finish_child, daemon=True)
    worker.start()
    stream = pair.parent.receive(timeout_seconds=2)
    assert completed.is_set() is False
    parent_ack = pair.parent.acknowledge(stream, timeout_seconds=2)
    worker.join(timeout=2)
    assert not worker.is_alive()
    child_ack = child_result.get_nowait()
    assert not isinstance(child_ack, BaseException)
    assert child_ack == parent_ack
  finally:
    pair.close()


def test_incremental_receive_returns_live_records_in_order_before_final_stream() -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=2,
  )
  child_result: queue.Queue[object] = queue.Queue()
  child_completed = threading.Event()

  def child_work() -> None:
    try:
      pair.child.start(timeout_seconds=2)
      pair.child.send_event(
        {"type": "text_delta", "text": "first"},
        timeout_seconds=2,
      )
      pair.child.send_event(
        {"type": "tool_call_complete", "tool_name": "file_read"},
        timeout_seconds=2,
      )
      child_result.put(pair.child.complete(
        {"type": "stream_complete"},
        timeout_seconds=2,
      ))
    except BaseException as exc:
      child_result.put(exc)
    finally:
      child_completed.set()

  child_worker = threading.Thread(target=child_work, daemon=True)
  child_worker.start()
  try:
    records = [
      pair.parent.receive_next(
        timeout_seconds=2,
        unbounded_stream=False,
      ),
      pair.parent.receive_next(),
      pair.parent.receive_next(),
    ]
    assert all(isinstance(record, channel.AutonomousEventRecord) for record in records)
    assert [record.event["type"] for record in records] == [
      "text_delta",
      "tool_call_complete",
      "stream_complete",
    ]
    assert child_completed.is_set() is False

    stream = pair.parent.receive_next()
    assert isinstance(stream, ReceivedAutonomousEventStream)
    assert all(
      stream.records[index] is record
      for index, record in enumerate(records)
    )
    parent_ack = pair.parent.acknowledge(stream, timeout_seconds=2)
    child_worker.join(timeout=2)
    assert not child_worker.is_alive()
    child_ack = child_result.get_nowait()
    assert not isinstance(child_ack, BaseException)
    assert child_ack == parent_ack
  finally:
    pair.close()


def test_incremental_delivery_pause_consumes_original_absolute_deadline(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parent, raw_child = _raw_parent_pair()
  first = _canonical_frame(_event(0, {"type": "text_delta", "text": "first"}))
  terminal = _canonical_frame(_event(1, {"type": "stream_complete"}))
  _send_raw_stream(raw_child, [
    _canonical_frame(_hello()),
    first,
    terminal,
    _canonical_frame(_end(2, _event_digest(first, terminal))),
  ])
  clock = [0.0]
  monkeypatch.setattr(channel.time, "monotonic", lambda: clock[0])
  try:
    record = parent.receive_next(
      timeout_seconds=0.5,
      unbounded_stream=False,
    )
    assert isinstance(record, channel.AutonomousEventRecord)
    clock[0] = 0.6
    with pytest.raises(AutonomousEventChannelTimeout):
      parent.receive_next()
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_unbounded_incremental_receive_discards_hello_deadline(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  _send_raw_stream(raw_child, [
    _canonical_frame(_hello()),
    terminal,
    _canonical_frame(_end(1, _event_digest(terminal))),
  ])
  calls: list[tuple[float | None, bool]] = []
  original_recv = channel._recv_wire_frame

  def recording_recv(*args, **kwargs):
    calls.append((kwargs["deadline"], kwargs.get("unbounded", False)))
    return original_recv(*args, **kwargs)

  monkeypatch.setattr(channel, "_recv_wire_frame", recording_recv)
  monkeypatch.setattr(channel.time, "monotonic", lambda: 10.0)
  try:
    record = parent.receive_next(unbounded_stream=True)
    stream = parent.receive_next()
    assert isinstance(record, channel.AutonomousEventRecord)
    assert isinstance(stream, ReceivedAutonomousEventStream)
    assert calls[0] == (
      10.0 + channel.HELLO_HANDSHAKE_TIMEOUT_SECONDS,
      False,
    )
    assert all(
      deadline is None and unbounded
      for deadline, unbounded in calls[1:]
    )
    assert parent._receive_deadline is None
    assert parent._socket is not None
    assert parent._socket.gettimeout() is None
  finally:
    raw_child.close()
    parent.close()


def test_incremental_receive_mode_is_fixed_by_first_call() -> None:
  parent, raw_child = _raw_parent_pair()
  first = _canonical_frame(
    _event(0, {"type": "text_delta", "text": "one"})
  )
  terminal = _canonical_frame(_event(1, {"type": "stream_complete"}))
  _send_raw_stream(raw_child, [
    _canonical_frame(_hello()),
    first,
    terminal,
    _canonical_frame(_end(2, _event_digest(first, terminal))),
  ])
  try:
    assert isinstance(
      parent.receive_next(unbounded_stream=True),
      channel.AutonomousEventRecord,
    )
    with pytest.raises(
      AutonomousEventChannelStateError,
      match="mode cannot change",
    ):
      parent.receive_next(unbounded_stream=False)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_unbounded_incremental_receive_hello_expiry_is_mechanical(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parent, raw_child = _raw_parent_pair()
  observed: list[tuple[float | None, bool]] = []

  def expire_hello(*_args, **kwargs):
    observed.append((kwargs["deadline"], kwargs.get("unbounded", False)))
    raise AutonomousEventChannelTimeout(
      "autonomous event channel read timed out"
    )

  monkeypatch.setattr(channel, "_recv_wire_frame", expire_hello)
  monkeypatch.setattr(channel.time, "monotonic", lambda: 25.0)
  try:
    with pytest.raises(AutonomousEventChannelTimeout, match="read timed out"):
      parent.receive_next(unbounded_stream=True)
    assert observed == [(
      25.0 + channel.HELLO_HANDSHAKE_TIMEOUT_SECONDS,
      False,
    )]
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_incremental_concurrent_next_calls_abort_both_owners(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parent, raw_child = _raw_parent_pair()
  raw_child.sendall(_canonical_frame(_hello()))
  entered_event_read = threading.Event()
  original_recv = channel._recv_wire_frame
  call_count = 0

  def marked_recv(*args, **kwargs):
    nonlocal call_count
    call_count += 1
    if call_count == 2:
      entered_event_read.set()
    return original_recv(*args, **kwargs)

  monkeypatch.setattr(channel, "_recv_wire_frame", marked_recv)
  first_result: queue.Queue[object] = queue.Queue()

  def first_next() -> None:
    try:
      first_result.put(parent.receive_next(
        timeout_seconds=0.5,
        unbounded_stream=False,
      ))
    except BaseException as exc:
      first_result.put(exc)

  worker = threading.Thread(target=first_next, daemon=True)
  worker.start()
  assert entered_event_read.wait(timeout=1)
  try:
    with pytest.raises(AutonomousEventChannelStateError, match="concurrent"):
      parent.receive_next()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert isinstance(first_result.get_nowait(), AutonomousEventChannelError)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_incremental_concurrent_call_cannot_escape_after_terminal_validation(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  _send_raw_stream(raw_child, [
    _canonical_frame(_hello()),
    terminal,
    _canonical_frame(_end(1, _event_digest(terminal))),
  ])
  suffix_validated = threading.Event()
  release_active_call = threading.Event()
  original_suffix = channel.AutonomousEventChannelParent._receive_terminal_suffix

  def held_suffix(self, *, deadline, unbounded):
    original_suffix(self, deadline=deadline, unbounded=unbounded)
    suffix_validated.set()
    assert release_active_call.wait(timeout=1)

  monkeypatch.setattr(
    channel.AutonomousEventChannelParent,
    "_receive_terminal_suffix",
    held_suffix,
  )
  active_result: queue.Queue[object] = queue.Queue()

  def receive_terminal() -> None:
    try:
      active_result.put(parent.receive_next(
        timeout_seconds=1,
        unbounded_stream=False,
      ))
    except BaseException as exc:
      active_result.put(exc)

  worker = threading.Thread(target=receive_terminal, daemon=True)
  worker.start()
  assert suffix_validated.wait(timeout=1)
  try:
    with pytest.raises(AutonomousEventChannelStateError, match="concurrent"):
      parent.receive_next()
    release_active_call.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert isinstance(
      active_result.get_nowait(),
      AutonomousEventChannelStateError,
    )
    with pytest.raises(
      AutonomousEventChannelStateError,
      match="closed or failed",
    ):
      parent.receive_next()
    assert parent.fileno() == -1
  finally:
    release_active_call.set()
    worker.join(timeout=2)
    raw_child.close()
    parent.close()


def test_incremental_rejects_eof_after_terminal_before_end() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      _canonical_frame(_event(0, {"type": "stream_complete"})),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="before END"):
      parent.receive_next(timeout_seconds=1, unbounded_stream=False)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_incremental_rejects_malformed_event_sequence() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      _canonical_frame(_event(1, {"type": "stream_complete"})),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="contiguous"):
      parent.receive_next(timeout_seconds=1, unbounded_stream=False)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_incremental_rejects_event_after_terminal() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      _canonical_frame(_event(0, {"type": "stream_complete"})),
      _canonical_frame(_event(1, {"type": "text_delta", "text": "late"})),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="final EVENT"):
      parent.receive_next(timeout_seconds=1, unbounded_stream=False)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_incremental_rejects_end_digest_mismatch() -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
      _canonical_frame(_end(1, "0" * 64)),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="does not bind"):
      parent.receive_next(timeout_seconds=1, unbounded_stream=False)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_incremental_rejects_bytes_after_end_before_exposing_terminal() -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
      _canonical_frame(_end(1, _event_digest(terminal))),
      _canonical_frame(_ack(1, _event_digest(terminal))),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="after END"):
      parent.receive_next(timeout_seconds=1, unbounded_stream=False)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_receive_and_incremental_receive_have_identical_streams() -> None:
  first = _canonical_frame(_event(0, {
    "type": "text_delta",
    "text": "same",
  }))
  terminal = _canonical_frame(_event(1, {"type": "stream_complete"}))
  frames = [
    _canonical_frame(_hello()),
    first,
    terminal,
    _canonical_frame(_end(2, _event_digest(first, terminal))),
  ]

  full_parent, full_raw_child = _raw_parent_pair()
  incremental_parent, incremental_raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(full_raw_child, frames)
    _send_raw_stream(incremental_raw_child, frames)
    full_stream = full_parent.receive(timeout_seconds=1)

    first_record = incremental_parent.receive_next(
      timeout_seconds=1,
      unbounded_stream=False,
    )
    terminal_record = incremental_parent.receive_next()
    incremental_stream = incremental_parent.receive_next()
    assert isinstance(first_record, channel.AutonomousEventRecord)
    assert isinstance(terminal_record, channel.AutonomousEventRecord)
    assert isinstance(incremental_stream, ReceivedAutonomousEventStream)
    assert incremental_stream == full_stream
  finally:
    full_raw_child.close()
    incremental_raw_child.close()
    full_parent.close()
    incremental_parent.close()


def test_incremental_receive_cannot_reset_deadline_or_mix_with_receive() -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
      _canonical_frame(_end(1, _event_digest(terminal))),
    ])
    record = parent.receive_next(
      timeout_seconds=1,
      unbounded_stream=False,
    )
    assert isinstance(record, channel.AutonomousEventRecord)
    with pytest.raises(AutonomousEventChannelStateError, match="cannot be reset"):
      parent.receive_next(timeout_seconds=1, unbounded_stream=False)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()

  mixed_parent, mixed_raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(mixed_raw_child, [
      _canonical_frame(_hello()),
      terminal,
      _canonical_frame(_end(1, _event_digest(terminal))),
    ])
    mixed_parent.receive_next(
      timeout_seconds=1,
      unbounded_stream=False,
    )
    with pytest.raises(AutonomousEventChannelStateError, match="begin exactly once"):
      mixed_parent.receive()
    assert mixed_parent.fileno() == -1
  finally:
    mixed_raw_child.close()
    mixed_parent.close()


def test_incremental_receive_reuse_after_final_stream_fails_closed() -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
      _canonical_frame(_end(1, _event_digest(terminal))),
    ])
    parent.receive_next(timeout_seconds=1, unbounded_stream=False)
    stream = parent.receive_next()
    assert isinstance(stream, ReceivedAutonomousEventStream)
    with pytest.raises(AutonomousEventChannelStateError, match="not active"):
      parent.receive_next()
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_incremental_receive_refuses_ack_before_final_stream() -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
      _canonical_frame(_end(1, _event_digest(terminal))),
    ])
    record = parent.receive_next(
      timeout_seconds=1,
      unbounded_stream=False,
    )
    assert isinstance(record, channel.AutonomousEventRecord)
    forged = ReceivedAutonomousEventStream(
      channel_id=_CHANNEL_ID,
      records=(record,),
      event_count=1,
      event_digest=_event_digest(terminal),
      exact_event_frame_bytes=len(terminal),
    )
    with pytest.raises(AutonomousEventChannelStateError, match="exact received"):
      parent.acknowledge(forged, timeout_seconds=1)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_received_event_property_does_not_mutate_retained_event() -> None:
  stream, _, _ = _happy_stream(
    [{"type": "text_delta", "nested": {"value": 1}}],
    {"type": "stream_complete"},
  )
  event = stream.records[0].event
  event["nested"]["value"] = 99
  assert stream.records[0].event["nested"]["value"] == 1


def test_socketpair_is_unix_stream_and_immediately_cloexec() -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  try:
    for endpoint in (pair.parent, pair.child):
      fd = endpoint.fileno()
      assert fd >= 0
      assert os.get_inheritable(fd) is False
      probe = socket.socket(fileno=os.dup(fd))
      try:
        assert probe.family == socket.AF_UNIX
        assert probe.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM
      finally:
        probe.close()
  finally:
    pair.close()


def test_adopt_inherited_descriptor_takes_ownership_and_sets_cloexec() -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  inherited_fd = os.dup(pair.child.fileno())
  pair.child.close()
  os.set_inheritable(inherited_fd, True)
  adopted = adopt_inherited_autonomous_event_channel(
    inherited_fd,
    channel_id=_CHANNEL_ID,
  )
  try:
    assert adopted.fileno() == inherited_fd
    assert os.get_inheritable(inherited_fd) is False
  finally:
    adopted.close()
    pair.parent.close()
  with pytest.raises(OSError):
    os.fstat(inherited_fd)


def test_adopt_rejects_non_socket_and_closes_transferred_descriptor() -> None:
  read_fd, write_fd = os.pipe()
  try:
    with pytest.raises(AutonomousEventChannelTransportError):
      adopt_inherited_autonomous_event_channel(
        read_fd,
        channel_id=_CHANNEL_ID,
      )
    with pytest.raises(OSError):
      os.fstat(read_fd)
  finally:
    os.close(write_fd)


@pytest.mark.parametrize(
  ("channel_id", "timeout"),
  [
    ("not-a-digest", 1),
    (_CHANNEL_ID, 0),
  ],
)
def test_adopt_closes_transferred_descriptor_on_configuration_failure(
  channel_id: str,
  timeout: float,
) -> None:
  left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
  transferred_fd = left.detach()
  os.set_inheritable(transferred_fd, True)
  try:
    with pytest.raises((TypeError, ValueError)):
      adopt_inherited_autonomous_event_channel(
        transferred_fd,
        channel_id=channel_id,
        io_timeout_seconds=timeout,
      )
    with pytest.raises(OSError):
      os.fstat(transferred_fd)
  finally:
    right.close()


@pytest.mark.parametrize(
  "payload",
  [
    b'{"channel_id":"' + _CHANNEL_ID.encode() + b'","kind":"HELLO","kind":"HELLO","version":1}',
    b'{ "channel_id":"' + _CHANNEL_ID.encode() + b'","kind":"HELLO","version":1}',
    b'{"channel_id":"' + _CHANNEL_ID.encode() + b'","kind":"HELLO","version":1,"extra":null}',
    b'{"channel_id":"' + _CHANNEL_ID.encode() + b'","kind":"HELLO","version":true}',
    b'{"channel_id":"' + _CHANNEL_ID.encode() + b'","kind":"HELLO","version":1}\n',
  ],
  ids=[
    "duplicate-key",
    "noncanonical-whitespace",
    "unknown-field",
    "boolean-version",
    "trailing-newline",
  ],
)
def test_parent_rejects_noncanonical_or_open_hello(payload: bytes) -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    raw_child.sendall(_HEADER.pack(len(payload)) + payload)
    raw_child.shutdown(socket.SHUT_WR)
    with pytest.raises(AutonomousEventChannelProtocolError):
      parent.receive(timeout_seconds=1)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_hello_bound_to_other_channel() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(raw_child, [_canonical_frame(_hello(channel_id=_OTHER_CHANNEL_ID))])
    with pytest.raises(AutonomousEventChannelProtocolError, match="different channel"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_duplicate_key_inside_event_payload() -> None:
  parent, raw_child = _raw_parent_pair()
  body = (
    b'{"channel_id":"'
    + _CHANNEL_ID.encode()
    + b'","event":{"type":"stream_complete","value":1,"value":2},'
    + b'"kind":"EVENT","seq":0,"version":1}'
  )
  try:
    raw_child.sendall(
      _canonical_frame(_hello())
      + _HEADER.pack(len(body))
      + body
    )
    raw_child.shutdown(socket.SHUT_WR)
    with pytest.raises(AutonomousEventChannelProtocolError, match="duplicate key"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_sequence_gap() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      _canonical_frame(_event(1, {"type": "stream_complete"})),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="contiguous"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_stream_error() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      _canonical_frame(_event(0, {"type": "stream_error"})),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="stream_error"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_event_after_terminal() -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  extra = _canonical_frame(_event(1, {"type": "text_delta", "text": "late"}))
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
      extra,
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="final EVENT"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_end_without_terminal_event() -> None:
  parent, raw_child = _raw_parent_pair()
  nonterminal = _canonical_frame(_event(0, {"type": "text_delta", "text": "x"}))
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      nonterminal,
      _canonical_frame(_end(1, _event_digest(nonterminal))),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="terminal EVENT"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


@pytest.mark.parametrize("wrong_field", ["count", "digest"])
def test_parent_rejects_end_not_bound_to_exact_event_frames(wrong_field: str) -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  count = 2 if wrong_field == "count" else 1
  digest = "0" * 64 if wrong_field == "digest" else _event_digest(terminal)
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
      _canonical_frame(_end(count, digest)),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="does not bind"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_bytes_after_end() -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  digest = _event_digest(terminal)
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
      _canonical_frame(_end(1, digest)),
      _canonical_frame(_ack(1, digest)),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="after END"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_eof_before_end() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      _canonical_frame(_event(0, {"type": "stream_complete"})),
    ])
    with pytest.raises(AutonomousEventChannelProtocolError, match="before END"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_partial_frame_at_eof() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    raw_child.sendall(_canonical_frame(_hello()) + b"\x00\x00")
    raw_child.shutdown(socket.SHUT_WR)
    with pytest.raises(AutonomousEventChannelProtocolError, match="premature EOF"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_rejects_declared_oversize_frame_without_reading_body() -> None:
  parent, raw_child = _raw_parent_pair()
  try:
    raw_child.sendall(_HEADER.pack(channel.AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES))
    with pytest.raises(AutonomousEventChannelBoundsError, match="2 MiB"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_enforces_line_bound(monkeypatch: pytest.MonkeyPatch) -> None:
  parent, raw_child = _raw_parent_pair()
  monkeypatch.setattr(channel, "AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES", 32)
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      _canonical_frame(_event(0, {
        "type": "stream_complete",
        "text": "x" * 64,
      })),
    ])
    with pytest.raises(AutonomousEventChannelBoundsError, match="line"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_enforces_aggregate_exact_frame_bound(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  monkeypatch.setattr(
    channel,
    "AUTONOMOUS_EVENT_CHANNEL_MAX_EVENT_BYTES",
    len(terminal) - 1,
  )
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      terminal,
    ])
    with pytest.raises(AutonomousEventChannelBoundsError, match="aggregate"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


def test_parent_enforces_event_count_bound(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parent, raw_child = _raw_parent_pair()
  monkeypatch.setattr(channel, "AUTONOMOUS_EVENT_CHANNEL_MAX_EVENTS", 1)
  try:
    _send_raw_stream(raw_child, [
      _canonical_frame(_hello()),
      _canonical_frame(_event(0, {"type": "text_delta", "text": "x"})),
      _canonical_frame(_event(1, {"type": "stream_complete"})),
    ])
    with pytest.raises(AutonomousEventChannelBoundsError, match="count"):
      parent.receive(timeout_seconds=1)
  finally:
    raw_child.close()
    parent.close()


@pytest.mark.parametrize(
  "bad_event",
  [
    {"type": "text_delta", "value": float("nan")},
    {"type": "text_delta", "value": ("tuple",)},
    {1: "non-string-key", "type": "text_delta"},
    {"type": " text_delta"},
    {"type": ""},
  ],
  ids=[
    "nan",
    "tuple",
    "non-string-key",
    "noncanonical-type",
    "empty-type",
  ],
)
def test_child_rejects_noncanonical_event_and_aborts(
  bad_event: dict[str, Any],
) -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  try:
    pair.child.start(timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelProtocolError):
      pair.child.send_event(bad_event, timeout_seconds=1)
    assert pair.child.fileno() == -1
  finally:
    pair.close()


def test_child_rejects_circular_json_and_aborts() -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  event: dict[str, Any] = {"type": "text_delta"}
  event["self"] = event
  try:
    pair.child.start(timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelProtocolError, match="circular"):
      pair.child.send_event(event, timeout_seconds=1)
    assert pair.child.fileno() == -1
  finally:
    pair.close()


def test_canonical_preflight_rejects_22_node_shared_list_dag_in_bounded_work(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  shared: list[Any] = []
  for _ in range(22):
    shared = [shared, shared]
  event = {"type": "text_delta", "value": shared}
  dumps_called = False

  def unexpected_dump(*args, **kwargs):
    nonlocal dumps_called
    dumps_called = True
    raise AssertionError("oversized DAG reached json.dumps")

  monkeypatch.setattr(channel.json, "dumps", unexpected_dump)
  started = time.monotonic()
  with pytest.raises(
    AutonomousEventChannelProtocolError,
    match="reuse container identities",
  ):
    snapshot_autonomous_event(event)
  assert time.monotonic() - started < 0.5
  assert dumps_called is False


@pytest.mark.parametrize("value_kind", ["huge-string", "huge-int"])
def test_canonical_preflight_rejects_huge_scalar_before_json_dumps(
  monkeypatch: pytest.MonkeyPatch,
  value_kind: str,
) -> None:
  if value_kind == "huge-string":
    value: Any = "x" * (
      channel.AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES * 4
    )
  else:
    value = 1 << (
      channel.AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES * 16
    )
  dumps_called = False

  def unexpected_dump(*args, **kwargs):
    nonlocal dumps_called
    dumps_called = True
    raise AssertionError("oversized scalar reached json.dumps")

  monkeypatch.setattr(channel.json, "dumps", unexpected_dump)
  started = time.monotonic()
  with pytest.raises(AutonomousEventChannelBoundsError, match="line"):
    snapshot_autonomous_event({"type": "text_delta", "value": value})
  assert time.monotonic() - started < 1.0
  assert dumps_called is False


def test_public_event_snapshot_is_canonical_bounded_and_immutable() -> None:
  caller_event = {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "nested": {"marker": "before"},
  }

  snapshot = snapshot_autonomous_event(caller_event)
  caller_event["nested"]["marker"] = "after"
  decoded = snapshot.event
  decoded["nested"]["marker"] = "decoded-mutation"

  assert snapshot.event_type == "stream_complete"
  assert snapshot.event_line_bytes.endswith(b"\n")
  assert snapshot.event == {
    "nested": {"marker": "before"},
    "terminal_disposition": "completed",
    "type": "stream_complete",
  }


def test_child_reserves_terminal_events_for_complete() -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  try:
    pair.child.start(timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelProtocolError, match="complete"):
      pair.child.send_event({"type": "stream_complete"}, timeout_seconds=1)
    assert pair.child.fileno() == -1
  finally:
    pair.close()


@pytest.mark.parametrize(
  "terminal_event",
  [
    {"type": "text_delta"},
    {"type": "stream_error"},
  ],
  ids=["nonterminal", "legacy-stream-error"],
)
def test_child_complete_requires_supported_terminal_and_aborts(
  terminal_event: dict[str, Any],
) -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  try:
    pair.child.start(timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelProtocolError):
      pair.child.complete(terminal_event, timeout_seconds=1)
    assert pair.child.fileno() == -1
  finally:
    pair.close()


def test_child_enforces_frame_bound(monkeypatch: pytest.MonkeyPatch) -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  try:
    pair.child.start(timeout_seconds=1)
    monkeypatch.setattr(channel, "AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES", 512)
    with pytest.raises(AutonomousEventChannelBoundsError, match="frame"):
      pair.child.send_event(
        {"type": "text_delta", "text": "x" * 1024},
        timeout_seconds=1,
      )
  finally:
    pair.close()


def test_child_enforces_line_bound(monkeypatch: pytest.MonkeyPatch) -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  try:
    pair.child.start(timeout_seconds=1)
    monkeypatch.setattr(channel, "AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES", 32)
    with pytest.raises(AutonomousEventChannelBoundsError, match="line"):
      pair.child.send_event(
        {"type": "text_delta", "text": "x" * 64},
        timeout_seconds=1,
      )
  finally:
    pair.close()


def test_child_enforces_count_bound(monkeypatch: pytest.MonkeyPatch) -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  try:
    pair.child.start(timeout_seconds=1)
    monkeypatch.setattr(channel, "AUTONOMOUS_EVENT_CHANNEL_MAX_EVENTS", 1)
    pair.child.send_event({"type": "text_delta"}, timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelBoundsError, match="count"):
      pair.child.complete({"type": "stream_complete"}, timeout_seconds=1)
  finally:
    pair.close()


def test_child_enforces_aggregate_bound_over_exact_frames(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)
  first = _canonical_frame(_event(0, {"type": "text_delta"}))
  try:
    pair.child.start(timeout_seconds=1)
    monkeypatch.setattr(
      channel,
      "AUTONOMOUS_EVENT_CHANNEL_MAX_EVENT_BYTES",
      len(first),
    )
    pair.child.send_event({"type": "text_delta"}, timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelBoundsError, match="aggregate"):
      pair.child.complete({"type": "stream_complete"}, timeout_seconds=1)
  finally:
    pair.close()


def _run_child_completion_against_raw_parent(
  *,
  ack_builder,
) -> BaseException | object:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=1,
  )
  raw = _duplicate_endpoint(pair.parent)
  result: queue.Queue[object] = queue.Queue()

  def child_work() -> None:
    try:
      pair.child.start(timeout_seconds=1)
      result.put(pair.child.complete(
        {"type": "stream_complete"},
        timeout_seconds=1,
      ))
    except BaseException as exc:
      result.put(exc)

  worker = threading.Thread(target=child_work, daemon=True)
  worker.start()
  try:
    hello, _ = _raw_recv_frame(raw)
    event, event_frame = _raw_recv_frame(raw)
    end, _ = _raw_recv_frame(raw)
    assert hello["kind"] == "HELLO"
    assert event["kind"] == "EVENT"
    assert end["kind"] == "END"
    assert raw.recv(1) == b""
    response = ack_builder(end, event_frame)
    if response is not None:
      raw.sendall(response)
      raw.shutdown(socket.SHUT_WR)
    raw.close()
    worker.join(timeout=2)
    assert not worker.is_alive()
    return result.get_nowait()
  finally:
    raw.close()
    pair.close()


@pytest.mark.parametrize("wrong_field", ["count", "digest", "channel"])
def test_child_requires_ack_bound_to_exact_completion(wrong_field: str) -> None:
  def ack_builder(end: dict[str, Any], _event_frame: bytes) -> bytes:
    count = end["event_count"] + 1 if wrong_field == "count" else end["event_count"]
    digest = "0" * 64 if wrong_field == "digest" else end["event_digest"]
    channel_id = _OTHER_CHANNEL_ID if wrong_field == "channel" else _CHANNEL_ID
    return _canonical_frame(_ack(count, digest, channel_id=channel_id))

  outcome = _run_child_completion_against_raw_parent(
    ack_builder=ack_builder,
  )
  assert isinstance(outcome, AutonomousEventChannelProtocolError)


def test_child_rejects_eof_without_ack() -> None:
  outcome = _run_child_completion_against_raw_parent(
    ack_builder=lambda _end, _event_frame: None,
  )
  assert isinstance(outcome, AutonomousEventChannelAcknowledgementError)


def test_child_rejects_bytes_after_ack() -> None:
  def ack_builder(end: dict[str, Any], _event_frame: bytes) -> bytes:
    ack = _canonical_frame(_ack(end["event_count"], end["event_digest"]))
    return ack + ack

  outcome = _run_child_completion_against_raw_parent(
    ack_builder=ack_builder,
  )
  assert isinstance(outcome, AutonomousEventChannelAcknowledgementError)


def test_parent_refuses_forged_copy_of_received_stream() -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=1,
  )
  child_result: queue.Queue[object] = queue.Queue()

  def finish_child() -> None:
    try:
      pair.child.start(timeout_seconds=1)
      child_result.put(pair.child.complete(
        {"type": "stream_complete"},
        timeout_seconds=1,
      ))
    except BaseException as exc:
      child_result.put(exc)

  worker = threading.Thread(target=finish_child, daemon=True)
  worker.start()
  try:
    stream = pair.parent.receive(timeout_seconds=1)
    with pytest.raises(AutonomousEventChannelStateError, match="exact"):
      pair.parent.acknowledge(replace(stream), timeout_seconds=1)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert isinstance(child_result.get_nowait(), AutonomousEventChannelProtocolError)
  finally:
    pair.close()


def test_receive_has_a_finite_idle_timeout_and_closes_descriptor() -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=0.02,
  )
  try:
    with pytest.raises(AutonomousEventChannelTimeout):
      pair.parent.receive()
    assert pair.parent.fileno() == -1
  finally:
    pair.close()


def test_parent_receive_uses_one_absolute_deadline_through_end_and_eof(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parent, raw_child = _raw_parent_pair()
  terminal = _canonical_frame(_event(0, {"type": "stream_complete"}))
  _send_raw_stream(raw_child, [
    _canonical_frame(_hello()),
    terminal,
    _canonical_frame(_end(1, _event_digest(terminal))),
  ])
  clock = [0.0]
  original_recv = channel._recv_wire_frame

  def slow_drip_recv(*args, **kwargs):
    frame = original_recv(*args, **kwargs)
    clock[0] += 0.2
    return frame

  monkeypatch.setattr(channel.time, "monotonic", lambda: clock[0])
  monkeypatch.setattr(channel, "_recv_wire_frame", slow_drip_recv)
  try:
    with pytest.raises(AutonomousEventChannelTimeout):
      parent.receive(timeout_seconds=0.5)
    assert parent.fileno() == -1
  finally:
    raw_child.close()
    parent.close()


def test_child_complete_uses_one_absolute_deadline_through_ack_and_eof(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=1,
  )
  pair.child.start(timeout_seconds=1)
  raw_parent = _duplicate_endpoint(pair.parent)
  peer_result: queue.Queue[object] = queue.Queue()

  def raw_peer() -> None:
    try:
      _raw_recv_frame(raw_parent)
      _, event_frame = _raw_recv_frame(raw_parent)
      end, _ = _raw_recv_frame(raw_parent)
      assert raw_parent.recv(1) == b""
      assert end["event_digest"] == _event_digest(event_frame)
      raw_parent.sendall(_canonical_frame(_ack(
        end["event_count"],
        end["event_digest"],
      )))
      raw_parent.shutdown(socket.SHUT_WR)
      peer_result.put(None)
    except BaseException as exc:
      peer_result.put(exc)

  peer = threading.Thread(target=raw_peer, daemon=True)
  peer.start()
  clock = [0.0]
  original_send = channel._send_wire_frame

  def slow_complete_send(sock, frame, *, deadline):
    original_send(sock, frame, deadline=deadline)
    kind = json.loads(frame.body)["kind"]
    if kind in {"EVENT", "END"}:
      clock[0] += 0.3

  monkeypatch.setattr(channel.time, "monotonic", lambda: clock[0])
  monkeypatch.setattr(channel, "_send_wire_frame", slow_complete_send)
  try:
    with pytest.raises(AutonomousEventChannelTimeout):
      pair.child.complete(
        {"type": "stream_complete"},
        timeout_seconds=0.5,
      )
    peer.join(timeout=2)
    assert not peer.is_alive()
    peer_outcome = peer_result.get_nowait()
    assert peer_outcome is None or isinstance(peer_outcome, BrokenPipeError)
    assert pair.child.fileno() == -1
  finally:
    raw_parent.close()
    pair.close()


def test_child_event_snapshot_is_immune_to_caller_mutation_during_encoding(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=2,
  )
  terminal_event = {
    "type": "stream_complete",
    "marker": "before",
  }
  parent_result: queue.Queue[object] = queue.Queue()
  parent_worker = threading.Thread(
    target=_run_parent_ack,
    args=(pair.parent, parent_result),
    daemon=True,
  )
  dump_entered = threading.Event()
  mutation_done = threading.Event()
  original_dumps = channel.json.dumps
  triggered = False

  def mutate_caller() -> None:
    assert dump_entered.wait(timeout=1)
    terminal_event.clear()
    terminal_event.update({
      "type": "error",
      "marker": "after",
    })
    mutation_done.set()

  def hooked_dumps(value, *args, **kwargs):
    nonlocal triggered
    if (
      not triggered
      and type(value) is dict
      and value.get("type") == "stream_complete"
    ):
      triggered = True
      dump_entered.set()
      assert mutation_done.wait(timeout=1)
    return original_dumps(value, *args, **kwargs)

  mutator = threading.Thread(target=mutate_caller, daemon=True)
  try:
    pair.child.start(timeout_seconds=2)
    monkeypatch.setattr(channel.json, "dumps", hooked_dumps)
    parent_worker.start()
    mutator.start()
    child_ack = pair.child.complete(terminal_event, timeout_seconds=2)
    parent_worker.join(timeout=2)
    mutator.join(timeout=2)
    assert not parent_worker.is_alive()
    assert not mutator.is_alive()
    outcome = parent_result.get_nowait()
    if isinstance(outcome, BaseException):
      raise outcome
    stream, parent_ack = outcome
    assert parent_ack == child_ack
    assert stream.events[-1] == {
      "type": "stream_complete",
      "marker": "before",
    }
    assert terminal_event == {
      "type": "error",
      "marker": "after",
    }
    assert triggered is True
  finally:
    pair.close()


def test_child_ack_wait_has_a_finite_timeout() -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=0.02,
  )
  raw_parent = _duplicate_endpoint(pair.parent)
  result: queue.Queue[object] = queue.Queue()

  def child_work() -> None:
    try:
      pair.child.start()
      result.put(pair.child.complete({"type": "stream_complete"}))
    except BaseException as exc:
      result.put(exc)

  worker = threading.Thread(target=child_work, daemon=True)
  worker.start()
  try:
    _raw_recv_frame(raw_parent)
    _raw_recv_frame(raw_parent)
    _raw_recv_frame(raw_parent)
    assert raw_parent.recv(1) == b""
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert isinstance(result.get_nowait(), AutonomousEventChannelTimeout)
    assert pair.child.fileno() == -1
  finally:
    raw_parent.close()
    pair.close()


def test_child_write_backpressure_has_a_finite_timeout() -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=0.02,
  )
  try:
    assert pair.child._socket is not None
    pair.child._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    pair.child.start()
    with pytest.raises(AutonomousEventChannelTimeout):
      pair.child.send_event({
        "type": "text_delta",
        "text": "x" * (512 * 1024),
      })
    assert pair.child.fileno() == -1
  finally:
    pair.close()


def test_base_exception_during_write_aborts_descriptor(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pair = create_autonomous_event_channel(channel_id=_CHANNEL_ID)

  def interrupt(*args, **kwargs) -> None:
    raise KeyboardInterrupt

  monkeypatch.setattr(channel, "_send_wire_frame", interrupt)
  try:
    with pytest.raises(KeyboardInterrupt):
      pair.child.start(timeout_seconds=1)
    assert pair.child.fileno() == -1
  finally:
    pair.close()


def test_concurrent_endpoint_owner_is_rejected_without_waiting(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  pair = create_autonomous_event_channel(
    channel_id=_CHANNEL_ID,
    io_timeout_seconds=1,
  )
  entered = threading.Event()
  original = channel._recv_wire_frame

  def marked_recv(*args, **kwargs):
    entered.set()
    return original(*args, **kwargs)

  monkeypatch.setattr(channel, "_recv_wire_frame", marked_recv)
  first_result: queue.Queue[object] = queue.Queue()

  def first_receive() -> None:
    try:
      first_result.put(pair.parent.receive(timeout_seconds=1))
    except BaseException as exc:
      first_result.put(exc)

  worker = threading.Thread(target=first_receive, daemon=True)
  worker.start()
  assert entered.wait(timeout=1)
  try:
    with pytest.raises(AutonomousEventChannelStateError, match="concurrent"):
      pair.parent.receive(timeout_seconds=1)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert isinstance(first_result.get_nowait(), AutonomousEventChannelError)
  finally:
    pair.close()


@pytest.mark.parametrize(
  "timeout",
  [0, -1, float("inf"), float("nan"), True, "1"],
)
def test_timeout_configuration_is_finite_and_bounded(timeout: object) -> None:
  with pytest.raises((TypeError, ValueError)):
    create_autonomous_event_channel(
      channel_id=_CHANNEL_ID,
      io_timeout_seconds=timeout,  # type: ignore[arg-type]
    )


def test_protocol_source_contains_no_polling_sleep_or_file_io() -> None:
  source_path = Path(inspect.getsourcefile(channel) or "")
  tree = ast.parse(source_path.read_text(encoding="utf-8"))
  forbidden_calls: list[str] = []
  for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
      continue
    function = node.func
    if isinstance(function, ast.Name) and function.id in {"open", "sleep"}:
      forbidden_calls.append(function.id)
    if (
      isinstance(function, ast.Attribute)
      and function.attr in {
        "Thread",
        "open",
        "read_text",
        "write_text",
        "sleep",
      }
    ):
      forbidden_calls.append(function.attr)
  assert forbidden_calls == []
