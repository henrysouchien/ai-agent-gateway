from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import socket
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from .autonomous_frame_metrics import (
  FrameCapFinding,
  FrameCapFindingEvent,
  FrameCapMetrics,
)


AUTONOMOUS_EVENT_CHANNEL_VERSION = 1
# Hang bound on the private gateway↔child frame channel: a runaway child or
# corrupt length prefix must not make the gateway allocate or read without limit.
# FRAME-CAPS-ARE-CONTAINMENT (Henry, 723db4743e); precedent e9d4dc138c:
# a 2.49 MB tool result killed bg_353. Keep the durable record at full fidelity,
# stream its compacted projection, and warn/drop an undeliverable non-terminal.
# This channel ceiling includes the four-byte length header.
AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES = 2 * 1024 * 1024
AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES = 2 * 1024 * 1024
AUTONOMOUS_EVENT_CHANNEL_MAX_EVENTS = 100_000
AUTONOMOUS_EVENT_CHANNEL_MAX_EVENT_BYTES = 64 * 1024 * 1024
# Resource containment: these are the only hang bounds on the child-process
# peer sockets (no tool or request clock reaches either socket); the approval
# channel imports both. DEFAULT_IO_TIMEOUT is the one default for each
# blocking send/recv (event create/adopt, approval send, runner start);
# MAX_IO_TIMEOUT is the largest value a caller may configure. HELLO bounds
# only the wait for the peer's first frame (the incremental stream is
# unbounded afterward). They refuse no event.
AUTONOMOUS_EVENT_CHANNEL_DEFAULT_IO_TIMEOUT_SECONDS = 300.0
AUTONOMOUS_EVENT_CHANNEL_MAX_IO_TIMEOUT_SECONDS = 24 * 60 * 60.0
HELLO_HANDSHAKE_TIMEOUT_SECONDS = 120.0
AUTONOMOUS_EVENT_CHANNEL_FD_ENV = "AGENT_AUTONOMOUS_EVENT_CHANNEL_FD"
AUTONOMOUS_EVENT_CHANNEL_DIGEST_DOMAIN = (
  b"agent-gateway.autonomous-event-channel/event-frames/v1\x00"
)

_FRAME_HEADER = struct.Struct(">I")
# Fixed cost of one EVENT frame around its canonical event, without seq digits:
# the length header, sibling keys, and the 64-hex channel id.
_EVENT_FRAME_FIXED_ENVELOPE_BYTES = (
  _FRAME_HEADER.size
  + len(b'{"channel_id":"","event":,"kind":"EVENT","seq":,"version":1}')
  + 64
)
_EVENT_FRAME_ENVELOPE_BYTES = (
  _EVENT_FRAME_FIXED_ENVELOPE_BYTES
  + len(str(AUTONOMOUS_EVENT_CHANNEL_MAX_EVENTS))
)
# A Finding reports only bytes known to exist, including the minimum seq 0.
_EVENT_FRAME_MINIMUM_ENVELOPE_BYTES = _EVENT_FRAME_FIXED_ENVELOPE_BYTES + 1
_CHANNEL_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_JSON_NESTING = 128
_MAX_JSON_VALUE_NODES = 100_000
_TERMINAL_EVENT_TYPES = frozenset({"stream_complete", "error"})
_FORBIDDEN_EVENT_TYPES = frozenset({"stream_error"})
_PROTOCOL_KINDS = frozenset({
  "HELLO", "EVENT", "END", "ACK", "PRIVATE_EXECUTABLE_CAPTURE",
})
_HELLO_FIELDS = frozenset({"kind", "version", "channel_id"})
_EVENT_FIELDS = frozenset({"kind", "version", "channel_id", "seq", "event"})
_PRIVATE_CAPTURE_FIELDS = frozenset({
  "kind", "version", "channel_id", "private_executable_capture",
})
_SUMMARY_FIELDS = frozenset({
  "kind",
  "version",
  "channel_id",
  "event_count",
  "event_digest",
})

frame_cap_metrics = FrameCapMetrics("event_channel")


class AutonomousEventChannelError(RuntimeError):
  """Base failure for the autonomous inherited event channel."""


class AutonomousEventChannelStateError(AutonomousEventChannelError):
  """An endpoint was used outside its single-owner protocol state."""


class AutonomousEventChannelProtocolError(AutonomousEventChannelError):
  """The peer sent bytes that violate the closed event-channel protocol."""


class AutonomousEventChannelBoundsError(AutonomousEventChannelProtocolError):
  """A declared event-channel resource bound was exceeded."""


class _CanonicalJsonByteBoundsError(AutonomousEventChannelBoundsError):
  """A byte preflight refusal with the canonical bytes already observed."""

  def __init__(self, minimum_size_bytes: int) -> None:
    self.minimum_size_bytes = minimum_size_bytes
    super().__init__(
      "autonomous event channel canonical JSON exceeds its byte bound"
    )


class AutonomousEventChannelFrameCapError(AutonomousEventChannelBoundsError):
  """A frame exceeded the channel's hang bound, with its typed Finding."""

  def __init__(self, finding: FrameCapFinding) -> None:
    self.finding = finding
    super().__init__(
      "autonomous event channel frame exceeds the 2 MiB bound: "
      f"size_basis={finding.size_basis}, "
      f"frame_size_bytes={finding.frame_size_bytes}, cap_bytes={finding.cap_bytes}"
    )


class AutonomousEventChannelTimeout(AutonomousEventChannelError):
  """A bounded socket operation did not complete before its deadline."""


class AutonomousEventChannelTransportError(AutonomousEventChannelError):
  """The inherited socket transport failed."""


class AutonomousEventChannelAcknowledgementError(
  AutonomousEventChannelProtocolError
):
  """The peer acknowledgement does not bind the completed event stream."""


@dataclass(frozen=True, slots=True)
class AutonomousEventRecord:
  """One validated event and its exact canonical wire representation."""

  seq: int
  frame_bytes: bytes
  event_line_bytes: bytes

  @property
  def event(self) -> dict[str, Any]:
    return _decode_canonical_json_object(self.event_line_bytes[:-1])


@dataclass(frozen=True, slots=True)
class AutonomousEventSnapshot:
  """One immutable, bounded canonical event before transport ownership."""

  event_type: str
  event_line_bytes: bytes

  @property
  def event(self) -> dict[str, Any]:
    return _decode_canonical_json_object(self.event_line_bytes[:-1])


@dataclass(frozen=True, slots=True, repr=False)
class AutonomousPrivateExecutableCapture:
  """An ephemeral control operand, never an event or retained stream record."""

  channel_id: str
  payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReceivedAutonomousEventStream:
  """A terminal, EOF-validated stream that has not necessarily been ACKed."""

  channel_id: str
  records: tuple[AutonomousEventRecord, ...]
  event_count: int
  event_digest: str
  exact_event_frame_bytes: int

  @property
  def events(self) -> tuple[dict[str, Any], ...]:
    return tuple(record.event for record in self.records)

  @property
  def raw_event_frames(self) -> tuple[bytes, ...]:
    return tuple(record.frame_bytes for record in self.records)


@dataclass(frozen=True, slots=True)
class AutonomousEventAcknowledgement:
  """The exact count and digest accepted by both channel endpoints."""

  channel_id: str
  event_count: int
  event_digest: str


@dataclass(frozen=True, slots=True)
class _WireFrame:
  body: bytes
  frame_bytes: bytes


@dataclass(frozen=True, slots=True)
class _CanonicalJsonSnapshot:
  value: Any
  encoded: bytes


@dataclass(frozen=True, slots=True)
class _EventSnapshot:
  event_type: str
  encoded: bytes
  line_bytes: bytes


@dataclass(slots=True)
class _JsonSnapshotState:
  seen_containers: dict[int, object]
  value_nodes: int = 0


def _validate_channel_id(value: object) -> str:
  if type(value) is not str or _CHANNEL_ID_RE.fullmatch(value) is None:
    raise ValueError(
      "autonomous event channel_id must be 64 lowercase hexadecimal characters"
    )
  return value


def _validate_timeout(value: object, *, field_name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise TypeError(f"{field_name} must be a finite number of seconds")
  timeout = float(value)
  if (
    not math.isfinite(timeout)
    or timeout <= 0
    or timeout > AUTONOMOUS_EVENT_CHANNEL_MAX_IO_TIMEOUT_SECONDS
  ):
    raise ValueError(
      f"{field_name} must be greater than zero and at most "
      f"{AUTONOMOUS_EVENT_CHANNEL_MAX_IO_TIMEOUT_SECONDS:g} seconds"
    )
  return timeout


def _json_string_size(value: str, *, byte_limit: int) -> int:
  size = 2
  if size > byte_limit:
    raise _CanonicalJsonByteBoundsError(size)
  for character in value:
    codepoint = ord(character)
    if character in {'"', "\\"} or character in {
      "\b",
      "\f",
      "\n",
      "\r",
      "\t",
    }:
      increment = 2
    elif codepoint <= 0x1F:
      increment = 6
    elif 0xD800 <= codepoint <= 0xDFFF:
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel JSON strings must not contain surrogates"
      )
    elif codepoint <= 0x7F:
      increment = 1
    elif codepoint <= 0x7FF:
      increment = 2
    elif codepoint <= 0xFFFF:
      increment = 3
    else:
      increment = 4
    size += increment
    if size > byte_limit:
      raise _CanonicalJsonByteBoundsError(size)
  return size


def _json_int_minimum_size(value: int) -> int:
  # 1233 / 4096 is below log10(2); unlike the refusal estimator, this
  # cannot overstate the digits, even at a power-of-ten boundary.
  return ((max(value.bit_length() - 1, 0) * 1233) >> 12) + 1 + (value < 0)


def _json_int_size(value: int, *, byte_limit: int) -> int:
  if value == 0:
    return 1
  bit_length = value.bit_length()
  decimal_digit_upper_bound = ((bit_length * 1234) >> 12) + 1
  size = decimal_digit_upper_bound + (1 if value < 0 else 0)
  if size > byte_limit:
    raise _CanonicalJsonByteBoundsError(_json_int_minimum_size(value))
  return size


def _snapshot_json_value(
  value: Any,
  *,
  depth: int,
  byte_limit: int,
  state: _JsonSnapshotState,
) -> tuple[Any, int, int]:
  state.value_nodes += 1
  if state.value_nodes > _MAX_JSON_VALUE_NODES:
    raise AutonomousEventChannelBoundsError(
      "autonomous event channel JSON exceeds the value-node bound"
    )
  if depth > _MAX_JSON_NESTING:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel JSON exceeds the nesting bound"
    )
  if value is None:
    if byte_limit < 4:
      raise _CanonicalJsonByteBoundsError(4)
    return None, 4, 4
  if type(value) is bool:
    size = 4 if value else 5
    if size > byte_limit:
      raise _CanonicalJsonByteBoundsError(size)
    return value, size, size
  if type(value) is int:
    size = _json_int_size(value, byte_limit=byte_limit)
    return value, size, _json_int_minimum_size(value)
  if type(value) is str:
    size = _json_string_size(value, byte_limit=byte_limit)
    return value, size, size
  if type(value) is float:
    if not math.isfinite(value):
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel JSON numbers must be finite"
      )
    size = len(repr(value))
    if size > byte_limit:
      raise _CanonicalJsonByteBoundsError(size)
    return value, size, size
  if type(value) not in {dict, list}:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel payloads must contain only exact JSON values"
    )

  container_id = id(value)
  if container_id in state.seen_containers:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel JSON must not contain circular references"
    )
  state.seen_containers[container_id] = value
  if byte_limit < 2:
    raise _CanonicalJsonByteBoundsError(2)

  used = 2
  minimum_used = 2
  try:
    if type(value) is dict:
      snapshot: dict[str, Any] = {}
      first = True
      for key, item in value.items():
        if type(key) is not str:
          raise AutonomousEventChannelProtocolError(
            "autonomous event channel JSON object keys must be strings"
          )
        separator_size = 0 if first else 1
        key_budget = byte_limit - used - separator_size - 1
        minimum_used += separator_size + 1
        key_size = _json_string_size(key, byte_limit=key_budget)
        used += separator_size + key_size + 1
        minimum_used += key_size
        item_snapshot, item_size, item_minimum_size = _snapshot_json_value(
          item,
          depth=depth + 1,
          byte_limit=byte_limit - used,
          state=state,
        )
        snapshot[key] = item_snapshot
        used += item_size
        minimum_used += item_minimum_size
        first = False
      return snapshot, used, minimum_used
    else:
      snapshot_list: list[Any] = []
      first = True
      for item in value:
        separator_size = 0 if first else 1
        used += separator_size
        minimum_used += separator_size
        item_snapshot, item_size, item_minimum_size = _snapshot_json_value(
          item,
          depth=depth + 1,
          byte_limit=byte_limit - used,
          state=state,
        )
        snapshot_list.append(item_snapshot)
        used += item_size
        minimum_used += item_minimum_size
        first = False
      return snapshot_list, used, minimum_used
  except _CanonicalJsonByteBoundsError as exc:
    raise _CanonicalJsonByteBoundsError(
      minimum_used + exc.minimum_size_bytes
    ) from exc
  except AutonomousEventChannelError:
    raise
  except RuntimeError as exc:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel JSON mutated during snapshot"
    ) from exc
  finally:
    del state.seen_containers[container_id]


def _canonical_json_snapshot(
  value: Any,
  *,
  max_bytes: int,
) -> _CanonicalJsonSnapshot:
  if type(max_bytes) is not int or max_bytes <= 0:
    raise ValueError("autonomous event channel canonical JSON bound is invalid")
  snapshot, estimated_size, _ = _snapshot_json_value(
    value,
    depth=0,
    byte_limit=max_bytes,
    state=_JsonSnapshotState(seen_containers={}),
  )
  try:
    rendered = json.dumps(
      snapshot,
      allow_nan=False,
      ensure_ascii=False,
      separators=(",", ":"),
      sort_keys=True,
    )
    encoded = rendered.encode("utf-8")
  except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel payload is not canonical-JSON encodable"
    ) from exc
  if len(encoded) > max_bytes:
    raise _CanonicalJsonByteBoundsError(len(encoded))
  if len(encoded) > estimated_size:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel canonical JSON preflight underestimated output"
    )
  return _CanonicalJsonSnapshot(value=snapshot, encoded=encoded)


def _canonical_json_bytes(value: Any, *, max_bytes: int) -> bytes:
  return _canonical_json_snapshot(value, max_bytes=max_bytes).encoded


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  value: dict[str, Any] = {}
  for key, item in pairs:
    if key in value:
      raise AutonomousEventChannelProtocolError(
        f"autonomous event channel JSON contains duplicate key: {key}"
      )
    value[key] = item
  return value


def _reject_json_constant(value: str) -> None:
  raise AutonomousEventChannelProtocolError(
    f"autonomous event channel JSON contains non-finite number: {value}"
  )


def _decode_canonical_json_object(body: bytes) -> dict[str, Any]:
  try:
    text = body.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel frame is not valid UTF-8"
    ) from exc
  try:
    value = json.loads(
      text,
      object_pairs_hook=_unique_json_object,
      parse_constant=_reject_json_constant,
    )
  except AutonomousEventChannelProtocolError:
    raise
  except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel frame is not valid JSON"
    ) from exc
  if type(value) is not dict:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel frame must be a JSON object"
    )
  canonical = _canonical_json_bytes(
    value,
    max_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES - _FRAME_HEADER.size,
  )
  if not hmac.compare_digest(canonical, body):
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel frame is not canonical JSON"
    )
  return value


def _decode_received_wire_frame(wire: _WireFrame) -> dict[str, Any]:
  """Decode a received frame, recording byte refusals only at wire entry."""
  try:
    return _decode_canonical_json_object(wire.body)
  except _CanonicalJsonByteBoundsError as exc:
    raise AutonomousEventChannelFrameCapError(frame_cap_metrics.record(
      frame_size_bytes=len(wire.frame_bytes),
      cap_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES,
      direction="receive",
      size_basis="declared",
    )) from exc


def _require_exact_fields(
  value: dict[str, Any],
  fields: frozenset[str],
  *,
  kind: str,
) -> None:
  actual = set(value)
  if actual != fields:
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    raise AutonomousEventChannelProtocolError(
      f"autonomous event channel {kind} fields differ "
      f"(missing={missing}, unknown={unknown})"
    )


def _require_kind(value: dict[str, Any], expected: str) -> None:
  kind = value.get("kind")
  if type(kind) is not str or kind not in _PROTOCOL_KINDS:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel frame kind is invalid"
    )
  if kind != expected:
    raise AutonomousEventChannelProtocolError(
      f"autonomous event channel expected {expected}, received {kind}"
    )


def _require_version(value: dict[str, Any]) -> None:
  version = value.get("version")
  if type(version) is not int or version != AUTONOMOUS_EVENT_CHANNEL_VERSION:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel version is unsupported"
    )


def _require_frame_channel(value: dict[str, Any], expected: str) -> None:
  try:
    channel_id = _validate_channel_id(value.get("channel_id"))
  except (TypeError, ValueError) as exc:
    raise AutonomousEventChannelProtocolError(str(exc)) from exc
  if not hmac.compare_digest(channel_id, expected):
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel frame is bound to a different channel"
    )


def _require_nonnegative_int(value: object, *, field_name: str) -> int:
  if type(value) is not int or value < 0:
    raise AutonomousEventChannelProtocolError(
      f"autonomous event channel {field_name} must be a nonnegative integer"
    )
  return value


def _require_digest(value: object, *, field_name: str) -> str:
  if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
    raise AutonomousEventChannelProtocolError(
      f"autonomous event channel {field_name} must be a lowercase SHA-256 digest"
    )
  return value


def _event_type(event: dict[str, Any]) -> str:
  event_type = event.get("type")
  if (
    type(event_type) is not str
    or not event_type
    or event_type != event_type.strip()
  ):
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel event.type must be a non-empty canonical string"
    )
  if event_type in _FORBIDDEN_EVENT_TYPES:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel rejects the legacy stream_error terminal"
    )
  return event_type


def _snapshot_event(event: dict[str, Any]) -> _EventSnapshot:
  if type(event) is not dict:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel events must be JSON objects"
    )
  try:
    snapshot = _canonical_json_snapshot(
      event,
      max_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES - 1,
    )
  except _CanonicalJsonByteBoundsError as exc:
    raise AutonomousEventChannelFrameCapError(FrameCapFinding(
      channel="event_channel",
      direction="send",
      size_basis="lower_bound",
      frame_size_bytes=(
        _EVENT_FRAME_MINIMUM_ENVELOPE_BYTES + exc.minimum_size_bytes
      ),
      cap_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES,
    )) from exc
  except AutonomousEventChannelBoundsError as exc:
    raise AutonomousEventChannelBoundsError(
      "autonomous event channel event line exceeds the 2 MiB bound"
    ) from exc
  # A snapshot is the pre-flight an owner runs before it touches the socket,
  # and a bounds failure inside `send_event` aborts the endpoint. So the EVENT
  # framing budget is checked here too: every event this function returns can
  # actually be framed and sent.
  if (
    len(snapshot.encoded)
    > AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES - _EVENT_FRAME_ENVELOPE_BYTES
  ):
    raise AutonomousEventChannelFrameCapError(FrameCapFinding(
      channel="event_channel",
      direction="send",
      size_basis="lower_bound",
      frame_size_bytes=(
        _EVENT_FRAME_MINIMUM_ENVELOPE_BYTES + len(snapshot.encoded)
      ),
      cap_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES,
    ))
  if type(snapshot.value) is not dict:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel events must be JSON objects"
    )
  event_type = _event_type(snapshot.value)
  return _EventSnapshot(
    event_type=event_type,
    encoded=snapshot.encoded,
    line_bytes=snapshot.encoded + b"\n",
  )


def snapshot_autonomous_event(
  event: dict[str, Any],
) -> AutonomousEventSnapshot:
  """Return an immutable event snapshot under the channel's exact bounds."""

  snapshot = _snapshot_event(event)
  return AutonomousEventSnapshot(
    event_type=snapshot.event_type,
    event_line_bytes=snapshot.line_bytes,
  )


def autonomous_event_fits_frame(event: dict[str, Any]) -> bool:
  """Return whether ``event`` fits one event frame on this channel.

  The channel owns its own size verdict, measured on the event a producer
  would actually send.  A producer deciding whether to degrade an event
  before streaming it asks here instead of estimating the payload it can
  afford; only the size question is answered.  An event the channel would
  refuse for shape — non-exact JSON values, a circular reference, a missing
  ``type`` — is not a size failure and is reported as fitting, so that
  refusal still surfaces where the event is sent.
  """

  try:
    _snapshot_event(event)
  except AutonomousEventChannelBoundsError:
    return False
  except AutonomousEventChannelProtocolError:
    return True
  return True


def _encode_wire_frame(value: dict[str, Any]) -> _WireFrame:
  try:
    body = _canonical_json_bytes(
      value,
      max_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES - _FRAME_HEADER.size,
    )
  except _CanonicalJsonByteBoundsError as exc:
    # Preflight constructs evidence only; a forwarded Finding EVENT is counted
    # by the parent receiver, not once here and again on receipt.
    raise AutonomousEventChannelFrameCapError(FrameCapFinding(
      channel="event_channel",
      frame_size_bytes=_FRAME_HEADER.size + exc.minimum_size_bytes,
      cap_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES,
      direction="send",
      size_basis="lower_bound",
    )) from exc
  frame_length = _FRAME_HEADER.size + len(body)
  if frame_length > AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES:
    raise AutonomousEventChannelFrameCapError(FrameCapFinding(
      channel="event_channel",
      frame_size_bytes=frame_length,
      cap_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES,
      direction="send",
      size_basis="actual",
    ))
  frame_bytes = _FRAME_HEADER.pack(len(body)) + body
  return _WireFrame(body=body, frame_bytes=frame_bytes)


def _encode_event_wire_frame(
  snapshot: _EventSnapshot,
  *,
  channel_id: str,
  seq: int,
) -> _WireFrame:
  body = (
    b'{"channel_id":"'
    + channel_id.encode("ascii")
    + b'","event":'
    + snapshot.encoded
    + b',"kind":"EVENT","seq":'
    + str(seq).encode("ascii")
    + b',"version":1}'
  )
  if _FRAME_HEADER.size + len(body) > AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES:
    raise AutonomousEventChannelFrameCapError(frame_cap_metrics.record(
      frame_size_bytes=_FRAME_HEADER.size + len(body),
      cap_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES,
      direction="send",
      size_basis="actual",
    ))
  return _WireFrame(
    body=body,
    frame_bytes=_FRAME_HEADER.pack(len(body)) + body,
  )


def _new_event_digest() -> Any:
  digest = hashlib.sha256()
  digest.update(AUTONOMOUS_EVENT_CHANNEL_DIGEST_DOMAIN)
  return digest


def _socket_timeout(
  sock: socket.socket,
  *,
  deadline: float | None,
  operation: str,
  unbounded: bool = False,
) -> None:
  if unbounded:
    try:
      sock.settimeout(None)
    except OSError as exc:
      raise AutonomousEventChannelTransportError(
        f"autonomous event channel could not clear {operation} deadline"
      ) from exc
    return
  if deadline is None:
    raise AutonomousEventChannelStateError(
      f"autonomous event channel {operation} deadline is missing"
    )
  remaining = deadline - time.monotonic()
  if remaining <= 0:
    raise AutonomousEventChannelTimeout(
      f"autonomous event channel {operation} timed out"
    )
  try:
    sock.settimeout(remaining)
  except OSError as exc:
    raise AutonomousEventChannelTransportError(
      f"autonomous event channel could not set {operation} deadline"
    ) from exc


def _send_wire_frame(
  sock: socket.socket,
  frame: _WireFrame,
  *,
  deadline: float,
) -> None:
  _socket_timeout(sock, deadline=deadline, operation="write")
  try:
    sock.sendall(frame.frame_bytes)
  except socket.timeout as exc:
    raise AutonomousEventChannelTimeout(
      "autonomous event channel write timed out"
    ) from exc
  except OSError as exc:
    raise AutonomousEventChannelTransportError(
      "autonomous event channel write failed"
    ) from exc


def _recv_exact(
  sock: socket.socket,
  byte_count: int,
  *,
  deadline: float | None,
  allow_clean_eof: bool,
  unbounded: bool = False,
) -> bytes | None:
  payload = bytearray(byte_count)
  view = memoryview(payload)
  received = 0
  while received < byte_count:
    _socket_timeout(
      sock,
      deadline=deadline,
      operation="read",
      unbounded=unbounded,
    )
    try:
      chunk_size = sock.recv_into(view[received:], byte_count - received)
    except socket.timeout as exc:
      raise AutonomousEventChannelTimeout(
        "autonomous event channel read timed out"
      ) from exc
    except OSError as exc:
      raise AutonomousEventChannelTransportError(
        "autonomous event channel read failed"
      ) from exc
    if chunk_size == 0:
      if received == 0 and allow_clean_eof:
        return None
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel encountered premature EOF"
      )
    received += chunk_size
  return bytes(payload)


def _recv_wire_frame(
  sock: socket.socket,
  *,
  deadline: float | None,
  allow_clean_eof: bool,
  unbounded: bool = False,
) -> _WireFrame | None:
  header = _recv_exact(
    sock,
    _FRAME_HEADER.size,
    deadline=deadline,
    allow_clean_eof=allow_clean_eof,
    unbounded=unbounded,
  )
  if header is None:
    return None
  (body_length,) = _FRAME_HEADER.unpack(header)
  if body_length == 0:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel rejects empty frames"
    )
  if _FRAME_HEADER.size + body_length > AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES:
    raise AutonomousEventChannelFrameCapError(frame_cap_metrics.record(
      frame_size_bytes=_FRAME_HEADER.size + body_length,
      cap_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES,
      direction="receive",
      size_basis="declared",
    ))
  body = _recv_exact(
    sock,
    body_length,
    deadline=deadline,
    allow_clean_eof=False,
    unbounded=unbounded,
  )
  if body is None:
    raise AutonomousEventChannelProtocolError(
      "autonomous event channel encountered premature EOF"
    )
  return _WireFrame(body=body, frame_bytes=header + body)


def _shutdown_write(sock: socket.socket) -> None:
  try:
    sock.shutdown(socket.SHUT_WR)
  except OSError as exc:
    raise AutonomousEventChannelTransportError(
      "autonomous event channel could not close its write direction"
    ) from exc


def _validate_endpoint_socket(sock: socket.socket) -> None:
  try:
    socket_type = sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
  except OSError as exc:
    raise AutonomousEventChannelTransportError(
      "autonomous event channel descriptor is not a socket"
    ) from exc
  if sock.family != socket.AF_UNIX or socket_type != socket.SOCK_STREAM:
    raise AutonomousEventChannelTransportError(
      "autonomous event channel requires an AF_UNIX SOCK_STREAM descriptor"
    )
  try:
    sock.set_inheritable(False)
  except OSError as exc:
    raise AutonomousEventChannelTransportError(
      "autonomous event channel could not set CLOEXEC"
    ) from exc
  if sock.get_inheritable():
    raise AutonomousEventChannelTransportError(
      "autonomous event channel descriptor remained inheritable"
    )


class _OwnedEventChannelEndpoint:
  def __init__(
    self,
    sock: socket.socket,
    *,
    channel_id: str,
    io_timeout_seconds: float,
  ) -> None:
    _validate_endpoint_socket(sock)
    self._socket: socket.socket | None = sock
    self._channel_id = _validate_channel_id(channel_id)
    self._io_timeout_seconds = _validate_timeout(
      io_timeout_seconds,
      field_name="io_timeout_seconds",
    )
    self._operation_state_lock = threading.Lock()
    self._operation_active = False
    self._operation_epoch = 0
    self._descriptor_lock = threading.Lock()
    self._closed = False
    self._failed = False

  @property
  def channel_id(self) -> str:
    return self._channel_id

  def fileno(self) -> int:
    with self._descriptor_lock:
      if self._socket is None:
        return -1
      return self._socket.fileno()

  def _timeout(self, timeout_seconds: float | None) -> float:
    if timeout_seconds is None:
      return self._io_timeout_seconds
    return _validate_timeout(timeout_seconds, field_name="timeout_seconds")

  def _require_socket(self) -> socket.socket:
    with self._descriptor_lock:
      sock = self._socket
    if sock is None:
      raise AutonomousEventChannelStateError(
        "autonomous event channel endpoint is closed"
      )
    return sock

  @contextmanager
  def _exclusive_operation(self) -> Iterator[None]:
    concurrent_owner = False
    unavailable = False
    with self._operation_state_lock:
      if self._operation_active:
        self._failed = True
        self._closed = True
        self._operation_epoch += 1
        concurrent_owner = True
        operation_epoch = -1
      elif self._failed or self._closed:
        unavailable = True
        operation_epoch = -1
      else:
        self._operation_active = True
        operation_epoch = self._operation_epoch

    if concurrent_owner:
      self._abort()
      raise AutonomousEventChannelStateError(
        "autonomous event channel endpoint has concurrent owners"
      )
    if unavailable:
      raise AutonomousEventChannelStateError(
        "autonomous event channel endpoint is closed or failed"
      )

    body_failed = False
    try:
      yield
    except BaseException:
      body_failed = True
      raise
    finally:
      with self._operation_state_lock:
        operation_aborted = (
          self._failed or self._operation_epoch != operation_epoch
        )
        self._operation_active = False
      if operation_aborted and not body_failed:
        self._abort()
        raise AutonomousEventChannelStateError(
          "autonomous event channel operation was aborted by a concurrent owner"
        )

  def _detach_socket(self) -> socket.socket | None:
    with self._descriptor_lock:
      sock = self._socket
      self._socket = None
    return sock

  def _abort(self) -> None:
    with self._operation_state_lock:
      self._failed = True
      self._closed = True
      self._operation_epoch += 1
    sock = self._detach_socket()
    if sock is None:
      return
    try:
      sock.shutdown(socket.SHUT_RDWR)
    except OSError:
      pass
    try:
      sock.close()
    except OSError:
      pass

  def _close_transport(self) -> None:
    sock = self._detach_socket()
    if sock is None:
      return
    try:
      sock.close()
    except OSError as exc:
      raise AutonomousEventChannelTransportError(
        "autonomous event channel descriptor close failed"
      ) from exc

  def _close_after_success(self) -> None:
    with self._operation_state_lock:
      self._closed = True
    self._close_transport()

  def close(self) -> None:
    with self._operation_state_lock:
      if self._operation_active:
        self._failed = True
        self._operation_epoch += 1
      self._closed = True
    self._close_transport()

  def interrupt(self) -> None:
    """Synchronously abort this endpoint and release a blocking operation."""
    self._abort()

  def __enter__(self):
    self._require_socket()
    return self

  def __exit__(self, exc_type, exc, traceback) -> None:
    if exc_type is None:
      self.close()
    else:
      self._abort()


class AutonomousEventChannelChild(_OwnedEventChannelEndpoint):
  """Single-owner child writer. Completion is impossible without a bound ACK."""

  def __init__(
    self,
    sock: socket.socket,
    *,
    channel_id: str,
    io_timeout_seconds: float,
  ) -> None:
    super().__init__(
      sock,
      channel_id=channel_id,
      io_timeout_seconds=io_timeout_seconds,
    )
    self._state = "new"
    self._next_seq = 0
    self._event_count = 0
    self._event_bytes = 0
    self._event_digest = _new_event_digest()

  def _send_event(
    self,
    event: dict[str, Any],
    *,
    terminal_required: bool,
    deadline: float,
  ) -> None:
    snapshot = _snapshot_event(event)
    is_terminal = snapshot.event_type in _TERMINAL_EVENT_TYPES
    if terminal_required != is_terminal:
      if terminal_required:
        raise AutonomousEventChannelProtocolError(
          "autonomous event channel completion requires stream_complete or error"
        )
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel terminal events must be sent through complete()"
      )
    if self._event_count >= AUTONOMOUS_EVENT_CHANNEL_MAX_EVENTS:
      raise AutonomousEventChannelBoundsError(
        "autonomous event channel event count exceeds 100,000"
      )
    frame = _encode_event_wire_frame(
      snapshot,
      channel_id=self._channel_id,
      seq=self._next_seq,
    )
    next_event_bytes = self._event_bytes + len(frame.frame_bytes)
    if next_event_bytes > AUTONOMOUS_EVENT_CHANNEL_MAX_EVENT_BYTES:
      raise AutonomousEventChannelBoundsError(
        "autonomous event channel aggregate EVENT frames exceed 64 MiB"
      )
    _send_wire_frame(
      self._require_socket(),
      frame,
      deadline=deadline,
    )
    self._event_digest.update(frame.frame_bytes)
    self._event_bytes = next_event_bytes
    self._event_count += 1
    self._next_seq += 1

  def start(self, *, timeout_seconds: float | None = None) -> None:
    with self._exclusive_operation():
      try:
        if self._state != "new":
          raise AutonomousEventChannelStateError(
            "autonomous event channel HELLO may be sent exactly once"
          )
        deadline = time.monotonic() + self._timeout(timeout_seconds)
        frame = _encode_wire_frame({
          "kind": "HELLO",
          "version": AUTONOMOUS_EVENT_CHANNEL_VERSION,
          "channel_id": self._channel_id,
        })
        _send_wire_frame(
          self._require_socket(),
          frame,
          deadline=deadline,
        )
        self._state = "started"
      except BaseException:
        self._abort()
        self._state = "failed"
        raise

  def send_event(
    self,
    event: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
  ) -> None:
    with self._exclusive_operation():
      try:
        if self._state != "started":
          raise AutonomousEventChannelStateError(
            "autonomous event channel is not accepting events"
          )
        deadline = time.monotonic() + self._timeout(timeout_seconds)
        self._send_event(
          event,
          terminal_required=False,
          deadline=deadline,
        )
      except BaseException:
        self._abort()
        self._state = "failed"
        raise

  def send_private_executable_capture(
    self,
    payload: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
  ) -> None:
    """Send original operands on the inherited lifeline, outside EVENT custody."""

    with self._exclusive_operation():
      try:
        if self._state != "started":
          raise AutonomousEventChannelStateError(
            "autonomous event channel is not accepting private captures"
          )
        if type(payload) is not dict:
          raise AutonomousEventChannelProtocolError(
            "autonomous private executable capture must be a JSON object"
          )
        try:
          frame = _encode_wire_frame({
            "kind": "PRIVATE_EXECUTABLE_CAPTURE",
            "version": AUTONOMOUS_EVENT_CHANNEL_VERSION,
            "channel_id": self._channel_id,
            "private_executable_capture": payload,
          })
        except AutonomousEventChannelFrameCapError as exc:
          try:
            self._send_event(
              {"type": "frame_cap_finding", "finding": exc.finding.model_dump()},
              terminal_required=False,
              deadline=time.monotonic() + self._timeout(timeout_seconds),
            )
          except Exception as finding_exc:
            # Diagnostic transport cannot replace the original capture failure.
            raise exc from finding_exc
          raise
        _send_wire_frame(
          self._require_socket(),
          frame,
          deadline=time.monotonic() + self._timeout(timeout_seconds),
        )
      except BaseException:
        self._abort()
        self._state = "failed"
        raise

  def complete(
    self,
    terminal_event: dict[str, Any],
    *,
    timeout_seconds: float | None = None,
  ) -> AutonomousEventAcknowledgement:
    """Send the sole final terminal event and return only after a matching ACK."""

    with self._exclusive_operation():
      try:
        if self._state != "started":
          raise AutonomousEventChannelStateError(
            "autonomous event channel is not eligible for completion"
          )
        deadline = time.monotonic() + self._timeout(timeout_seconds)
        self._send_event(
          terminal_event,
          terminal_required=True,
          deadline=deadline,
        )
        event_digest = self._event_digest.hexdigest()
        end_frame = _encode_wire_frame({
          "kind": "END",
          "version": AUTONOMOUS_EVENT_CHANNEL_VERSION,
          "channel_id": self._channel_id,
          "event_count": self._event_count,
          "event_digest": event_digest,
        })
        _send_wire_frame(
          self._require_socket(),
          end_frame,
          deadline=deadline,
        )
        _shutdown_write(self._require_socket())
        self._state = "awaiting_ack"

        ack_wire = _recv_wire_frame(
          self._require_socket(),
          deadline=deadline,
          allow_clean_eof=True,
        )
        if ack_wire is None:
          raise AutonomousEventChannelAcknowledgementError(
            "autonomous event channel closed before ACK"
          )
        ack = _decode_received_wire_frame(ack_wire)
        _require_exact_fields(ack, _SUMMARY_FIELDS, kind="ACK")
        _require_kind(ack, "ACK")
        _require_version(ack)
        _require_frame_channel(ack, self._channel_id)
        ack_count = _require_nonnegative_int(
          ack["event_count"],
          field_name="ACK event_count",
        )
        ack_digest = _require_digest(
          ack["event_digest"],
          field_name="ACK event_digest",
        )
        if (
          ack_count != self._event_count
          or not hmac.compare_digest(ack_digest, event_digest)
        ):
          raise AutonomousEventChannelAcknowledgementError(
            "autonomous event channel ACK does not bind the completed stream"
          )

        trailing = _recv_wire_frame(
          self._require_socket(),
          deadline=deadline,
          allow_clean_eof=True,
        )
        if trailing is not None:
          raise AutonomousEventChannelAcknowledgementError(
            "autonomous event channel received bytes after ACK"
          )
        acknowledgement = AutonomousEventAcknowledgement(
          channel_id=self._channel_id,
          event_count=ack_count,
          event_digest=ack_digest,
        )
        self._close_after_success()
        self._state = "completed"
        return acknowledgement
      except BaseException:
        self._abort()
        self._state = "failed"
        raise


class AutonomousEventChannelParent(_OwnedEventChannelEndpoint):
  """Parent reader that ACKs only an identity-exact validated stream."""

  def __init__(
    self,
    sock: socket.socket,
    *,
    channel_id: str,
    io_timeout_seconds: float,
  ) -> None:
    super().__init__(
      sock,
      channel_id=channel_id,
      io_timeout_seconds=io_timeout_seconds,
    )
    self._state = "new"
    self._received: ReceivedAutonomousEventStream | None = None
    self._pending_received: ReceivedAutonomousEventStream | None = None
    self._receive_deadline: float | None = None
    self._receive_unbounded: bool | None = None
    self._receive_records: list[AutonomousEventRecord] = []
    self._receive_digest = _new_event_digest()
    self._receive_event_bytes = 0
    self._receive_expected_seq = 0
    self._receive_terminal_seen = False

  def _begin_receive(
    self,
    timeout_seconds: float | None,
    *,
    unbounded_stream: bool = False,
  ) -> None:
    if self._state != "new":
      raise AutonomousEventChannelStateError(
        "autonomous event channel receive may begin exactly once"
      )
    if unbounded_stream and timeout_seconds is not None:
      raise AutonomousEventChannelStateError(
        "unbounded incremental receive uses the fixed HELLO timeout"
      )
    deadline = time.monotonic() + (
      HELLO_HANDSHAKE_TIMEOUT_SECONDS
      if unbounded_stream
      else self._timeout(timeout_seconds)
    )
    hello_wire = _recv_wire_frame(
      self._require_socket(),
      deadline=deadline,
      allow_clean_eof=True,
    )
    if hello_wire is None:
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel closed before HELLO"
      )
    hello = _decode_received_wire_frame(hello_wire)
    _require_exact_fields(hello, _HELLO_FIELDS, kind="HELLO")
    _require_kind(hello, "HELLO")
    _require_version(hello)
    _require_frame_channel(hello, self._channel_id)
    self._receive_deadline = None if unbounded_stream else deadline
    self._receive_unbounded = unbounded_stream
    self._state = "receiving"

  def _receive_mode(self) -> tuple[float | None, bool]:
    deadline = self._receive_deadline
    unbounded = self._receive_unbounded
    if self._state != "receiving" or unbounded is None:
      raise AutonomousEventChannelStateError(
        "autonomous event channel receive is not active"
      )
    if not unbounded and deadline is None:
      raise AutonomousEventChannelStateError(
        "autonomous event channel receive deadline is missing"
      )
    return deadline, unbounded

  def _receive_event_record(
    self,
    value: dict[str, Any],
    wire: _WireFrame,
  ) -> AutonomousEventRecord:
    _require_exact_fields(value, _EVENT_FIELDS, kind="EVENT")
    _require_kind(value, "EVENT")
    _require_version(value)
    _require_frame_channel(value, self._channel_id)
    seq = _require_nonnegative_int(value["seq"], field_name="EVENT seq")
    if seq != self._receive_expected_seq:
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel EVENT sequence is not contiguous"
      )
    try:
      event_snapshot = _snapshot_event(value["event"])
    except AutonomousEventChannelFrameCapError as exc:
      raise AutonomousEventChannelFrameCapError(frame_cap_metrics.record(
        frame_size_bytes=len(wire.frame_bytes),
        cap_bytes=AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES,
        direction="receive",
        size_basis="declared",
      )) from exc
    finding_event = None
    if event_snapshot.event_type == "frame_cap_finding":
      try:
        finding_event = FrameCapFindingEvent.model_validate(value["event"])
      except ValidationError as exc:
        raise AutonomousEventChannelProtocolError(
          "autonomous event channel frame-cap Finding event is invalid"
        ) from exc
    if len(self._receive_records) >= AUTONOMOUS_EVENT_CHANNEL_MAX_EVENTS:
      raise AutonomousEventChannelBoundsError(
        "autonomous event channel event count exceeds 100,000"
      )
    next_event_bytes = self._receive_event_bytes + len(wire.frame_bytes)
    if next_event_bytes > AUTONOMOUS_EVENT_CHANNEL_MAX_EVENT_BYTES:
      raise AutonomousEventChannelBoundsError(
        "autonomous event channel aggregate EVENT frames exceed 64 MiB"
      )
    self._receive_digest.update(wire.frame_bytes)
    self._receive_event_bytes = next_event_bytes
    record = AutonomousEventRecord(
      seq=seq,
      frame_bytes=wire.frame_bytes,
      event_line_bytes=event_snapshot.line_bytes,
    )
    self._receive_records.append(record)
    self._receive_expected_seq += 1
    self._receive_terminal_seen = (
      event_snapshot.event_type in _TERMINAL_EVENT_TYPES
    )
    if finding_event is not None:
      frame_cap_metrics.record_finding(finding_event.finding)
    return record

  def _receive_final_stream(
    self,
    value: dict[str, Any],
    *,
    deadline: float | None,
    unbounded: bool,
  ) -> None:
    _require_exact_fields(value, _SUMMARY_FIELDS, kind="END")
    _require_kind(value, "END")
    _require_version(value)
    _require_frame_channel(value, self._channel_id)
    if not self._receive_terminal_seen:
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel END requires a final terminal EVENT"
      )
    claimed_count = _require_nonnegative_int(
      value["event_count"],
      field_name="END event_count",
    )
    claimed_digest = _require_digest(
      value["event_digest"],
      field_name="END event_digest",
    )
    actual_digest = self._receive_digest.hexdigest()
    if (
      claimed_count != len(self._receive_records)
      or not hmac.compare_digest(claimed_digest, actual_digest)
    ):
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel END does not bind the received stream"
      )
    trailing = _recv_wire_frame(
      self._require_socket(),
      deadline=deadline,
      allow_clean_eof=True,
      unbounded=unbounded,
    )
    if trailing is not None:
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel received bytes after END"
      )
    stream = ReceivedAutonomousEventStream(
      channel_id=self._channel_id,
      records=tuple(self._receive_records),
      event_count=len(self._receive_records),
      event_digest=actual_digest,
      exact_event_frame_bytes=self._receive_event_bytes,
    )
    self._pending_received = stream
    self._state = "final_pending"

  def _receive_terminal_suffix(
    self,
    *,
    deadline: float | None,
    unbounded: bool,
  ) -> None:
    end_wire = _recv_wire_frame(
      self._require_socket(),
      deadline=deadline,
      allow_clean_eof=True,
      unbounded=unbounded,
    )
    if end_wire is None:
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel closed before END"
      )
    end = _decode_received_wire_frame(end_wire)
    if end.get("kind") != "END":
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel terminal event was not the final EVENT"
      )
    self._receive_final_stream(
      end,
      deadline=deadline,
      unbounded=unbounded,
    )

  def _expose_final_stream(self) -> ReceivedAutonomousEventStream:
    stream = self._pending_received
    if self._state != "final_pending" or stream is None:
      raise AutonomousEventChannelStateError(
        "autonomous event channel final stream is not pending"
      )
    self._pending_received = None
    self._received = stream
    self._state = "received"
    return stream

  def _advance_receive(
    self,
  ) -> AutonomousEventRecord | AutonomousPrivateExecutableCapture:
    deadline, unbounded = self._receive_mode()
    wire = _recv_wire_frame(
      self._require_socket(),
      deadline=deadline,
      allow_clean_eof=True,
      unbounded=unbounded,
    )
    if wire is None:
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel closed before END"
      )
    value = _decode_received_wire_frame(wire)
    kind = value.get("kind")
    if self._receive_terminal_seen and kind != "END":
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel terminal event was not the final EVENT"
      )
    if kind == "PRIVATE_EXECUTABLE_CAPTURE":
      _require_exact_fields(value, _PRIVATE_CAPTURE_FIELDS, kind=kind)
      _require_version(value)
      _require_frame_channel(value, self._channel_id)
      payload = value["private_executable_capture"]
      if type(payload) is not dict:
        raise AutonomousEventChannelProtocolError(
          "autonomous private executable capture must be a JSON object"
        )
      return AutonomousPrivateExecutableCapture(
        channel_id=self._channel_id,
        payload=payload,
      )
    if kind == "EVENT":
      record = self._receive_event_record(value, wire)
      if self._receive_terminal_seen:
        self._receive_terminal_suffix(
          deadline=deadline,
          unbounded=unbounded,
        )
      return record
    if kind == "END":
      self._receive_final_stream(
        value,
        deadline=deadline,
        unbounded=unbounded,
      )
      raise AutonomousEventChannelStateError(
        "autonomous event channel END bypassed terminal delivery"
      )
    if type(kind) is not str or kind not in _PROTOCOL_KINDS:
      raise AutonomousEventChannelProtocolError(
        "autonomous event channel frame kind is invalid"
      )
    raise AutonomousEventChannelProtocolError(
      f"autonomous event channel received unexpected {kind}"
    )

  def receive_next(
    self,
    *,
    timeout_seconds: float | None = None,
    unbounded_stream: bool | None = None,
  ) -> (
    AutonomousEventRecord
    | AutonomousPrivateExecutableCapture
    | ReceivedAutonomousEventStream
  ):
    """Return a private control operand or event; retain only EVENT records."""

    if unbounded_stream is not None and type(unbounded_stream) is not bool:
      raise TypeError("unbounded_stream must be an exact boolean or None")

    with self._exclusive_operation():
      try:
        if self._state == "new":
          self._begin_receive(
            timeout_seconds,
            unbounded_stream=(
              False if unbounded_stream is None else unbounded_stream
            ),
          )
        elif self._state == "receiving":
          if timeout_seconds is not None:
            raise AutonomousEventChannelStateError(
              "autonomous event channel receive deadline cannot be reset"
            )
          if (
            unbounded_stream is not None
            and unbounded_stream is not self._receive_unbounded
          ):
            raise AutonomousEventChannelStateError(
              "autonomous event channel receive mode cannot change"
            )
        elif self._state == "final_pending":
          if timeout_seconds is not None:
            raise AutonomousEventChannelStateError(
              "autonomous event channel receive deadline cannot be reset"
            )
          if (
            unbounded_stream is not None
            and unbounded_stream is not self._receive_unbounded
          ):
            raise AutonomousEventChannelStateError(
              "autonomous event channel receive mode cannot change"
            )
          return self._expose_final_stream()
        else:
          raise AutonomousEventChannelStateError(
            "autonomous event channel incremental receive is not active"
          )
        return self._advance_receive()
      except BaseException:
        self._abort()
        self._state = "failed"
        raise

  def receive(
    self,
    *,
    timeout_seconds: float | None = None,
  ) -> ReceivedAutonomousEventStream:
    """Receive through END and required child-write EOF without sending ACK."""

    with self._exclusive_operation():
      try:
        self._begin_receive(timeout_seconds)
        while True:
          result = self._advance_receive()
          if isinstance(result, AutonomousPrivateExecutableCapture):
            raise AutonomousEventChannelStateError(
              "private executable capture requires incremental owner ingestion"
            )
          if self._state == "final_pending":
            return self._expose_final_stream()
      except BaseException:
        self._abort()
        self._state = "failed"
        raise

  def acknowledge(
    self,
    stream: ReceivedAutonomousEventStream,
    *,
    timeout_seconds: float | None = None,
  ) -> AutonomousEventAcknowledgement:
    """ACK only the exact stream object returned by this endpoint."""

    with self._exclusive_operation():
      try:
        if self._state != "received" or stream is not self._received:
          raise AutonomousEventChannelStateError(
            "autonomous event channel may ACK only its exact received stream"
          )
        deadline = time.monotonic() + self._timeout(timeout_seconds)
        ack = AutonomousEventAcknowledgement(
          channel_id=self._channel_id,
          event_count=stream.event_count,
          event_digest=stream.event_digest,
        )
        frame = _encode_wire_frame({
          "kind": "ACK",
          "version": AUTONOMOUS_EVENT_CHANNEL_VERSION,
          "channel_id": ack.channel_id,
          "event_count": ack.event_count,
          "event_digest": ack.event_digest,
        })
        _send_wire_frame(
          self._require_socket(),
          frame,
          deadline=deadline,
        )
        _shutdown_write(self._require_socket())
        self._close_after_success()
        self._state = "acknowledged"
        return ack
      except BaseException:
        self._abort()
        self._state = "failed"
        raise


@dataclass(slots=True)
class AutonomousEventChannelPair:
  """Unique parent/child owners for one freshly-created socketpair."""

  parent: AutonomousEventChannelParent
  child: AutonomousEventChannelChild

  def close(self) -> None:
    first_error: BaseException | None = None
    for endpoint in (self.parent, self.child):
      try:
        endpoint.close()
      except BaseException as exc:
        if first_error is None:
          first_error = exc
    if first_error is not None:
      raise first_error

  def __enter__(self) -> AutonomousEventChannelPair:
    return self

  def __exit__(self, exc_type, exc, traceback) -> None:
    if exc_type is None:
      self.close()
      return
    self.parent._abort()
    self.child._abort()


def create_autonomous_event_channel(
  *,
  channel_id: str | None = None,
  io_timeout_seconds: float = AUTONOMOUS_EVENT_CHANNEL_DEFAULT_IO_TIMEOUT_SECONDS,
) -> AutonomousEventChannelPair:
  """Create a private AF_UNIX stream pair with CLOEXEC set on both endpoints."""

  resolved_channel_id = (
    os.urandom(32).hex()
    if channel_id is None
    else _validate_channel_id(channel_id)
  )
  timeout = _validate_timeout(
    io_timeout_seconds,
    field_name="io_timeout_seconds",
  )
  parent_sock: socket.socket | None = None
  child_sock: socket.socket | None = None
  parent: AutonomousEventChannelParent | None = None
  try:
    parent_sock, child_sock = socket.socketpair(
      socket.AF_UNIX,
      socket.SOCK_STREAM,
    )
    _validate_endpoint_socket(parent_sock)
    _validate_endpoint_socket(child_sock)
    parent = AutonomousEventChannelParent(
      parent_sock,
      channel_id=resolved_channel_id,
      io_timeout_seconds=timeout,
    )
    parent_sock = None
    child = AutonomousEventChannelChild(
      child_sock,
      channel_id=resolved_channel_id,
      io_timeout_seconds=timeout,
    )
    child_sock = None
    return AutonomousEventChannelPair(parent=parent, child=child)
  except BaseException:
    if parent is not None:
      try:
        parent.close()
      except AutonomousEventChannelError:
        pass
    if parent_sock is not None:
      try:
        parent_sock.close()
      except OSError:
        pass
    if child_sock is not None:
      try:
        child_sock.close()
      except OSError:
        pass
    raise


def adopt_inherited_autonomous_event_channel(
  fd: int,
  *,
  channel_id: str,
  io_timeout_seconds: float = AUTONOMOUS_EVENT_CHANNEL_DEFAULT_IO_TIMEOUT_SECONDS,
) -> AutonomousEventChannelChild:
  """Take sole ownership of an explicitly inherited child socket descriptor."""

  if type(fd) is not int or fd < 0:
    raise ValueError(
      "autonomous event channel inherited descriptor must be a nonnegative integer"
    )
  try:
    os.set_inheritable(fd, False)
  except OSError as exc:
    try:
      os.close(fd)
    except OSError:
      pass
    raise AutonomousEventChannelTransportError(
      "autonomous event channel could not set inherited descriptor CLOEXEC"
    ) from exc
  try:
    resolved_channel_id = _validate_channel_id(channel_id)
    timeout = _validate_timeout(
      io_timeout_seconds,
      field_name="io_timeout_seconds",
    )
  except BaseException:
    try:
      os.close(fd)
    except OSError:
      pass
    raise
  try:
    sock = socket.socket(fileno=fd)
  except OSError as exc:
    try:
      os.close(fd)
    except OSError:
      pass
    raise AutonomousEventChannelTransportError(
      "autonomous event channel inherited descriptor is not a socket"
    ) from exc
  try:
    _validate_endpoint_socket(sock)
    child = AutonomousEventChannelChild(
      sock,
      channel_id=resolved_channel_id,
      io_timeout_seconds=timeout,
    )
    sock = None
    return child
  except BaseException:
    if sock is not None:
      sock.close()
    raise


__all__ = [
  "AUTONOMOUS_EVENT_CHANNEL_DEFAULT_IO_TIMEOUT_SECONDS",
  "AUTONOMOUS_EVENT_CHANNEL_DIGEST_DOMAIN",
  "AUTONOMOUS_EVENT_CHANNEL_FD_ENV",
  "AUTONOMOUS_EVENT_CHANNEL_MAX_EVENTS",
  "AUTONOMOUS_EVENT_CHANNEL_MAX_EVENT_BYTES",
  "AUTONOMOUS_EVENT_CHANNEL_MAX_FRAME_BYTES",
  "AUTONOMOUS_EVENT_CHANNEL_MAX_IO_TIMEOUT_SECONDS",
  "AUTONOMOUS_EVENT_CHANNEL_MAX_LINE_BYTES",
  "AUTONOMOUS_EVENT_CHANNEL_VERSION",
  "HELLO_HANDSHAKE_TIMEOUT_SECONDS",
  "AutonomousEventAcknowledgement",
  "AutonomousEventChannelAcknowledgementError",
  "AutonomousEventChannelBoundsError",
  "AutonomousEventChannelChild",
  "AutonomousEventChannelError",
  "AutonomousEventChannelFrameCapError",
  "AutonomousEventChannelPair",
  "AutonomousEventChannelParent",
  "AutonomousEventChannelProtocolError",
  "AutonomousEventChannelStateError",
  "AutonomousEventChannelTimeout",
  "AutonomousEventChannelTransportError",
  "AutonomousEventRecord",
  "AutonomousEventSnapshot",
  "AutonomousPrivateExecutableCapture",
  "ReceivedAutonomousEventStream",
  "adopt_inherited_autonomous_event_channel",
  "autonomous_event_fits_frame",
  "create_autonomous_event_channel",
  "frame_cap_metrics",
  "snapshot_autonomous_event",
]
