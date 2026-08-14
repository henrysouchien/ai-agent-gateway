from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import select
import socket
import struct
import threading
import time
from typing import Any, Iterator, Mapping

from .autonomous_control_contract import (
  AUTONOMOUS_APPROVAL_DECISION_LIMIT,
)


AUTONOMOUS_APPROVAL_CHANNEL_FD_ENV = (
  "AGENT_AUTONOMOUS_APPROVAL_CHANNEL_FD"
)
AUTONOMOUS_APPROVAL_CHANNEL_VERSION = 1
AUTONOMOUS_APPROVAL_CHANNEL_MAX_FRAME_BYTES = 16 * 1024
AUTONOMOUS_APPROVAL_CHANNEL_DEFAULT_SEND_TIMEOUT_SECONDS = 5.0
AUTONOMOUS_APPROVAL_CHANNEL_DEFAULT_RECEIVE_TIMEOUT_SECONDS = (
  24 * 60 * 60.0
)
AUTONOMOUS_APPROVAL_CHANNEL_MAX_TIMEOUT_SECONDS = 24 * 60 * 60.0

_FRAME_HEADER = struct.Struct("!I")
_DECISION_FIELDS = frozenset({
  "kind",
  "version",
  "launch_nonce",
  "task_id",
  "control_run_id",
  "session_id",
  "channel_id",
  "delivery_sequence",
  "approval_id",
  "tool_call_id",
  "nonce",
  "approved",
  "allow_tool_type",
  "decided_at_ns",
})
_DECISION_KIND = "APPROVAL_DECISION"


class AutonomousApprovalChannelError(RuntimeError):
  """Base failure for the inherited autonomous approval channel."""


class AutonomousApprovalChannelStateError(
  AutonomousApprovalChannelError
):
  """The endpoint is closed, failed, or has concurrent owners."""


class AutonomousApprovalChannelProtocolError(
  AutonomousApprovalChannelError
):
  """A frame violated the closed approval-delivery protocol."""


class AutonomousApprovalChannelTimeout(
  AutonomousApprovalChannelError
):
  """A bounded channel operation did not complete before its deadline."""


class AutonomousApprovalChannelTransportError(
  AutonomousApprovalChannelError
):
  """The private socket transport failed."""


def _canonical_text(
  value: object,
  *,
  field_name: str,
  max_length: int = 512,
) -> str:
  if (
    type(value) is not str
    or not value
    or value != value.strip()
    or len(value) > max_length
    or any(ord(character) < 0x20 for character in value)
  ):
    raise ValueError(
      f"autonomous approval channel {field_name} is invalid"
    )
  return value


def _hex_text(
  value: object,
  *,
  field_name: str,
  length: int,
) -> str:
  normalized = _canonical_text(
    value,
    field_name=field_name,
    max_length=length,
  )
  if (
    len(normalized) != length
    or any(character not in "0123456789abcdef" for character in normalized)
  ):
    raise ValueError(
      f"autonomous approval channel {field_name} is invalid"
    )
  return normalized


def _positive_int(value: object, *, field_name: str) -> int:
  if type(value) is not int or value < 1:
    raise ValueError(
      f"autonomous approval channel {field_name} is invalid"
    )
  return value


def _timeout(value: object, *, field_name: str) -> float:
  if (
    isinstance(value, bool)
    or not isinstance(value, (int, float))
    or not 0 < float(value) <= AUTONOMOUS_APPROVAL_CHANNEL_MAX_TIMEOUT_SECONDS
  ):
    raise ValueError(
      f"autonomous approval channel {field_name} is invalid"
    )
  return float(value)


@dataclass(frozen=True, slots=True)
class AutonomousApprovalChannelAuthority:
  launch_nonce: str
  task_id: str
  control_run_id: str
  session_id: str
  channel_id: str

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "launch_nonce",
      _hex_text(
        self.launch_nonce,
        field_name="launch_nonce",
        length=32,
      ),
    )
    for field_name in ("task_id", "control_run_id", "session_id"):
      object.__setattr__(
        self,
        field_name,
        _canonical_text(
          getattr(self, field_name),
          field_name=field_name,
        ),
      )
    object.__setattr__(
      self,
      "channel_id",
      _hex_text(
        self.channel_id,
        field_name="channel_id",
        length=64,
      ),
    )

  def receipt(self) -> dict[str, str]:
    return {
      "launch_nonce": self.launch_nonce,
      "task_id": self.task_id,
      "control_run_id": self.control_run_id,
      "session_id": self.session_id,
      "channel_id": self.channel_id,
    }


@dataclass(frozen=True, slots=True)
class AutonomousApprovalDecision:
  authority: AutonomousApprovalChannelAuthority
  delivery_sequence: int
  approval_id: str
  tool_call_id: str
  nonce: str
  approved: bool
  decided_at_ns: int

  def __post_init__(self) -> None:
    if type(self.authority) is not AutonomousApprovalChannelAuthority:
      raise TypeError(
        "autonomous approval decision requires exact channel authority"
      )
    object.__setattr__(
      self,
      "delivery_sequence",
      _positive_int(
        self.delivery_sequence,
        field_name="delivery_sequence",
      ),
    )
    for field_name in ("approval_id", "tool_call_id", "nonce"):
      object.__setattr__(
        self,
        field_name,
        _canonical_text(
          getattr(self, field_name),
          field_name=field_name,
        ),
      )
    if type(self.approved) is not bool:
      raise ValueError(
        "autonomous approval channel approved is invalid"
      )
    object.__setattr__(
      self,
      "decided_at_ns",
      _positive_int(
        self.decided_at_ns,
        field_name="decided_at_ns",
      ),
    )

  @property
  def delivery_id(self) -> tuple[str, str, str]:
    return (self.approval_id, self.tool_call_id, self.nonce)

  def frame(self) -> dict[str, Any]:
    return {
      "kind": _DECISION_KIND,
      "version": AUTONOMOUS_APPROVAL_CHANNEL_VERSION,
      **self.authority.receipt(),
      "delivery_sequence": self.delivery_sequence,
      "approval_id": self.approval_id,
      "tool_call_id": self.tool_call_id,
      "nonce": self.nonce,
      "approved": self.approved,
      "allow_tool_type": False,
      "decided_at_ns": self.decided_at_ns,
    }


@dataclass(frozen=True, slots=True)
class ReceivedAutonomousApprovalDecision:
  decision: AutonomousApprovalDecision
  duplicate: bool

  def __post_init__(self) -> None:
    if type(self.decision) is not AutonomousApprovalDecision:
      raise TypeError(
        "received autonomous approval requires exact decision"
      )
    if type(self.duplicate) is not bool:
      raise TypeError(
        "received autonomous approval duplicate flag is invalid"
      )


def _reject_duplicate_fields(
  pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(
        f"autonomous approval frame repeats field {key!r}"
      )
    result[key] = value
  return result


def _reject_non_json_constant(value: str) -> Any:
  raise ValueError(
    f"autonomous approval frame contains non-JSON constant {value}"
  )


def _canonical_body(value: Mapping[str, Any]) -> bytes:
  try:
    body = json.dumps(
      value,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
      allow_nan=False,
    ).encode("utf-8")
  except (TypeError, ValueError) as exc:
    raise AutonomousApprovalChannelProtocolError(
      "autonomous approval frame is not canonical JSON"
    ) from exc
  if not body or len(body) > AUTONOMOUS_APPROVAL_CHANNEL_MAX_FRAME_BYTES:
    raise AutonomousApprovalChannelProtocolError(
      "autonomous approval frame exceeds its byte bound"
    )
  return body


def _encode_decision(
  decision: AutonomousApprovalDecision,
) -> tuple[bytes, bytes]:
  body = _canonical_body(decision.frame())
  return _FRAME_HEADER.pack(len(body)) + body, body


def _decode_decision(
  body: bytes,
  *,
  authority: AutonomousApprovalChannelAuthority,
) -> AutonomousApprovalDecision:
  try:
    value = json.loads(
      body.decode("utf-8", errors="strict"),
      object_pairs_hook=_reject_duplicate_fields,
      parse_constant=_reject_non_json_constant,
    )
  except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise AutonomousApprovalChannelProtocolError(
      "autonomous approval frame is malformed"
    ) from exc
  if (
    not isinstance(value, dict)
    or set(value) != _DECISION_FIELDS
    or value.get("kind") != _DECISION_KIND
    or type(value.get("version")) is not int
    or value.get("version") != AUTONOMOUS_APPROVAL_CHANNEL_VERSION
    or value.get("allow_tool_type") is not False
  ):
    raise AutonomousApprovalChannelProtocolError(
      "autonomous approval frame violates its closed contract"
    )
  if _canonical_body(value) != body:
    raise AutonomousApprovalChannelProtocolError(
      "autonomous approval frame is not canonically encoded"
    )
  try:
    frame_authority = AutonomousApprovalChannelAuthority(
      launch_nonce=value["launch_nonce"],
      task_id=value["task_id"],
      control_run_id=value["control_run_id"],
      session_id=value["session_id"],
      channel_id=value["channel_id"],
    )
    decision = AutonomousApprovalDecision(
      authority=frame_authority,
      delivery_sequence=value["delivery_sequence"],
      approval_id=value["approval_id"],
      tool_call_id=value["tool_call_id"],
      nonce=value["nonce"],
      approved=value["approved"],
      decided_at_ns=value["decided_at_ns"],
    )
  except (TypeError, ValueError) as exc:
    raise AutonomousApprovalChannelProtocolError(
      "autonomous approval frame fields are invalid"
    ) from exc
  if decision.authority != authority:
    raise AutonomousApprovalChannelProtocolError(
      "autonomous approval frame changed admitted authority"
    )
  return decision


def _validate_socket(sock: socket.socket) -> None:
  try:
    socket_type = sock.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)
  except OSError as exc:
    raise AutonomousApprovalChannelTransportError(
      "autonomous approval descriptor is not a socket"
    ) from exc
  if sock.family != socket.AF_UNIX or socket_type != socket.SOCK_STREAM:
    raise AutonomousApprovalChannelTransportError(
      "autonomous approval channel requires an AF_UNIX stream socket"
    )
  try:
    sock.getpeername()
  except OSError as exc:
    raise AutonomousApprovalChannelTransportError(
      "autonomous approval channel socket is not connected"
    ) from exc
  try:
    sock.setblocking(False)
    sock.set_inheritable(False)
  except OSError as exc:
    raise AutonomousApprovalChannelTransportError(
      "autonomous approval channel could not secure its descriptor"
    ) from exc
  if sock.get_inheritable():
    raise AutonomousApprovalChannelTransportError(
      "autonomous approval channel descriptor remained inheritable"
    )


def _wait_for(
  sock: socket.socket,
  *,
  readable: bool,
  deadline: float,
) -> None:
  while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      raise AutonomousApprovalChannelTimeout(
        "autonomous approval channel operation timed out"
      )
    try:
      ready_read, ready_write, exceptional = select.select(
        [sock] if readable else [],
        [] if readable else [sock],
        [sock],
        remaining,
      )
    except InterruptedError:
      continue
    except (OSError, ValueError) as exc:
      raise AutonomousApprovalChannelTransportError(
        "autonomous approval channel readiness check failed"
      ) from exc
    if exceptional:
      raise AutonomousApprovalChannelTransportError(
        "autonomous approval channel entered an exceptional state"
      )
    if ready_read or ready_write:
      return


def _send_all(
  sock: socket.socket,
  payload: bytes,
  *,
  deadline: float,
) -> None:
  view = memoryview(payload)
  offset = 0
  while offset < len(view):
    _wait_for(sock, readable=False, deadline=deadline)
    try:
      sent = sock.send(view[offset:])
    except BlockingIOError:
      continue
    except OSError as exc:
      raise AutonomousApprovalChannelTransportError(
        "autonomous approval channel send failed"
      ) from exc
    if sent <= 0:
      raise AutonomousApprovalChannelTransportError(
        "autonomous approval channel closed during send"
      )
    offset += sent


def _receive_exact(
  sock: socket.socket,
  size: int,
  *,
  deadline: float,
  field_name: str,
) -> bytes:
  chunks: list[bytes] = []
  remaining = size
  while remaining:
    _wait_for(sock, readable=True, deadline=deadline)
    try:
      chunk = sock.recv(remaining)
    except BlockingIOError:
      continue
    except OSError as exc:
      raise AutonomousApprovalChannelTransportError(
        f"autonomous approval channel failed while reading {field_name}"
      ) from exc
    if not chunk:
      raise AutonomousApprovalChannelTransportError(
        f"autonomous approval channel closed during {field_name}"
      )
    chunks.append(chunk)
    remaining -= len(chunk)
  return b"".join(chunks)


class _OwnedApprovalEndpoint:
  def __init__(
    self,
    sock: socket.socket,
    *,
    authority: AutonomousApprovalChannelAuthority,
    timeout_seconds: float,
  ) -> None:
    if type(authority) is not AutonomousApprovalChannelAuthority:
      raise TypeError(
        "autonomous approval endpoint requires exact authority"
      )
    _validate_socket(sock)
    self._socket: socket.socket | None = sock
    self._authority = authority
    self._timeout_seconds = timeout_seconds
    self._descriptor_lock = threading.Lock()
    self._operation_lock = threading.Lock()
    self._state_lock = threading.Lock()
    self._closed = False
    self._failed = False

  @property
  def authority(self) -> AutonomousApprovalChannelAuthority:
    return self._authority

  def fileno(self) -> int:
    with self._descriptor_lock:
      return -1 if self._socket is None else self._socket.fileno()

  def _require_socket(self) -> socket.socket:
    with self._descriptor_lock:
      sock = self._socket
    if sock is None:
      raise AutonomousApprovalChannelStateError(
        "autonomous approval channel endpoint is closed"
      )
    return sock

  @contextmanager
  def _operation(self) -> Iterator[None]:
    if not self._operation_lock.acquire(blocking=False):
      self._abort()
      raise AutonomousApprovalChannelStateError(
        "autonomous approval channel endpoint has concurrent owners"
      )
    try:
      with self._state_lock:
        if self._closed or self._failed:
          raise AutonomousApprovalChannelStateError(
            "autonomous approval channel endpoint is unavailable"
          )
      yield
    except BaseException:
      self._abort()
      raise
    finally:
      self._operation_lock.release()

  def _detach_socket(self) -> socket.socket | None:
    with self._descriptor_lock:
      sock = self._socket
      self._socket = None
    return sock

  def _abort(self) -> None:
    with self._state_lock:
      self._failed = True
      self._closed = True
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

  def close(self) -> None:
    with self._state_lock:
      self._closed = True
    sock = self._detach_socket()
    if sock is None:
      return
    try:
      sock.shutdown(socket.SHUT_RDWR)
    except OSError:
      pass
    try:
      sock.close()
    except OSError as exc:
      raise AutonomousApprovalChannelTransportError(
        "autonomous approval channel close failed"
      ) from exc


class AutonomousApprovalChannelParent(_OwnedApprovalEndpoint):
  """The sole bounded writer for one admitted child session."""

  def __init__(
    self,
    sock: socket.socket,
    *,
    authority: AutonomousApprovalChannelAuthority,
    timeout_seconds: float,
  ) -> None:
    super().__init__(
      sock,
      authority=authority,
      timeout_seconds=timeout_seconds,
    )
    self._sent_by_sequence: dict[int, bytes] = {}
    self._sequence_by_delivery: dict[tuple[str, str, str], int] = {}
    self._last_sequence = 0

  def send(
    self,
    decision: AutonomousApprovalDecision,
    *,
    timeout_seconds: float | None = None,
  ) -> AutonomousApprovalDecision:
    with self._operation():
      if type(decision) is not AutonomousApprovalDecision:
        raise TypeError(
          "autonomous approval channel send requires exact decision"
        )
      if decision.authority != self._authority:
        raise AutonomousApprovalChannelProtocolError(
          "autonomous approval send changed admitted authority"
        )
      payload, body = _encode_decision(decision)
      prior_body = self._sent_by_sequence.get(
        decision.delivery_sequence
      )
      prior_sequence = self._sequence_by_delivery.get(
        decision.delivery_id
      )
      if prior_body is not None:
        if prior_body != body or prior_sequence != decision.delivery_sequence:
          raise AutonomousApprovalChannelProtocolError(
            "autonomous approval sequence was reused with different content"
          )
      elif (
        prior_sequence is not None
        or decision.delivery_sequence <= self._last_sequence
        or len(self._sent_by_sequence)
        >= AUTONOMOUS_APPROVAL_DECISION_LIMIT
      ):
        raise AutonomousApprovalChannelProtocolError(
          "autonomous approval delivery is stale, conflicting, or over quota"
        )
      resolved_timeout = (
        self._timeout_seconds
        if timeout_seconds is None
        else _timeout(timeout_seconds, field_name="timeout_seconds")
      )
      _send_all(
        self._require_socket(),
        payload,
        deadline=time.monotonic() + resolved_timeout,
      )
      if prior_body is None:
        self._sent_by_sequence[decision.delivery_sequence] = body
        self._sequence_by_delivery[
          decision.delivery_id
        ] = decision.delivery_sequence
        self._last_sequence = decision.delivery_sequence
      return decision

  def require_sent(
    self,
    decision: AutonomousApprovalDecision,
  ) -> None:
    if type(decision) is not AutonomousApprovalDecision:
      raise TypeError(
        "autonomous approval sent lookup requires exact decision"
      )
    _payload, body = _encode_decision(decision)
    with self._state_lock:
      if (
        self._sent_by_sequence.get(decision.delivery_sequence) != body
        or self._sequence_by_delivery.get(decision.delivery_id)
        != decision.delivery_sequence
      ):
        raise AutonomousApprovalChannelProtocolError(
          "autonomous approval ACK has no exact live-session send"
        )


class AutonomousApprovalChannelChild(_OwnedApprovalEndpoint):
  """The sole reader for decisions addressed to one admitted child."""

  def __init__(
    self,
    sock: socket.socket,
    *,
    authority: AutonomousApprovalChannelAuthority,
    timeout_seconds: float,
  ) -> None:
    super().__init__(
      sock,
      authority=authority,
      timeout_seconds=timeout_seconds,
    )
    self._received_by_sequence: dict[
      int,
      tuple[bytes, AutonomousApprovalDecision],
    ] = {}
    self._sequence_by_delivery: dict[tuple[str, str, str], int] = {}
    self._last_sequence = 0

  def take_inherited_fd(self) -> int:
    """Transfer this endpoint without shutting down the inherited socket."""

    with self._operation():
      sock = self._detach_socket()
      if sock is None:
        raise AutonomousApprovalChannelStateError(
          "autonomous approval child descriptor is unavailable"
        )
      with self._state_lock:
        self._closed = True
      try:
        inherited_fd = sock.detach()
      except OSError as exc:
        try:
          sock.close()
        except OSError:
          pass
        raise AutonomousApprovalChannelTransportError(
          "autonomous approval child descriptor transfer failed"
        ) from exc
      if inherited_fd < 0:
        raise AutonomousApprovalChannelTransportError(
          "autonomous approval child descriptor transfer failed"
        )
      return inherited_fd

  def receive(
    self,
    *,
    timeout_seconds: float | None = None,
  ) -> ReceivedAutonomousApprovalDecision:
    with self._operation():
      resolved_timeout = (
        self._timeout_seconds
        if timeout_seconds is None
        else _timeout(timeout_seconds, field_name="timeout_seconds")
      )
      deadline = time.monotonic() + resolved_timeout
      header = _receive_exact(
        self._require_socket(),
        _FRAME_HEADER.size,
        deadline=deadline,
        field_name="frame header",
      )
      (body_length,) = _FRAME_HEADER.unpack(header)
      if (
        body_length < 1
        or body_length > AUTONOMOUS_APPROVAL_CHANNEL_MAX_FRAME_BYTES
      ):
        raise AutonomousApprovalChannelProtocolError(
          "autonomous approval frame length is invalid"
        )
      body = _receive_exact(
        self._require_socket(),
        body_length,
        deadline=deadline,
        field_name="frame body",
      )
      decision = _decode_decision(
        body,
        authority=self._authority,
      )
      prior_delivery = self._received_by_sequence.get(
        decision.delivery_sequence
      )
      prior_sequence = self._sequence_by_delivery.get(
        decision.delivery_id
      )
      if prior_delivery is not None:
        prior_body, prior_decision = prior_delivery
        if prior_body != body or prior_sequence != decision.delivery_sequence:
          raise AutonomousApprovalChannelProtocolError(
            "autonomous approval sequence was reused with different content"
          )
        return ReceivedAutonomousApprovalDecision(
          decision=prior_decision,
          duplicate=True,
        )
      if (
        prior_sequence is not None
        or decision.delivery_sequence <= self._last_sequence
        or len(self._received_by_sequence)
        >= AUTONOMOUS_APPROVAL_DECISION_LIMIT
      ):
        raise AutonomousApprovalChannelProtocolError(
          "autonomous approval delivery is stale, conflicting, or over quota"
        )
      self._received_by_sequence[decision.delivery_sequence] = (
        body,
        decision,
      )
      self._sequence_by_delivery[
        decision.delivery_id
      ] = decision.delivery_sequence
      self._last_sequence = decision.delivery_sequence
      return ReceivedAutonomousApprovalDecision(
        decision=decision,
        duplicate=False,
      )


@dataclass(slots=True)
class AutonomousApprovalChannelPair:
  parent: AutonomousApprovalChannelParent
  child: AutonomousApprovalChannelChild

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


def create_autonomous_approval_channel(
  *,
  authority: AutonomousApprovalChannelAuthority,
  send_timeout_seconds: float = (
    AUTONOMOUS_APPROVAL_CHANNEL_DEFAULT_SEND_TIMEOUT_SECONDS
  ),
  receive_timeout_seconds: float = (
    AUTONOMOUS_APPROVAL_CHANNEL_DEFAULT_RECEIVE_TIMEOUT_SECONDS
  ),
) -> AutonomousApprovalChannelPair:
  if type(authority) is not AutonomousApprovalChannelAuthority:
    raise TypeError(
      "autonomous approval channel requires exact authority"
    )
  send_timeout = _timeout(
    send_timeout_seconds,
    field_name="send_timeout_seconds",
  )
  receive_timeout = _timeout(
    receive_timeout_seconds,
    field_name="receive_timeout_seconds",
  )
  parent_sock: socket.socket | None = None
  child_sock: socket.socket | None = None
  try:
    parent_sock, child_sock = socket.socketpair(
      socket.AF_UNIX,
      socket.SOCK_STREAM,
    )
    _validate_socket(parent_sock)
    _validate_socket(child_sock)
    parent_sock.shutdown(socket.SHUT_RD)
    child_sock.shutdown(socket.SHUT_WR)
    parent = AutonomousApprovalChannelParent(
      parent_sock,
      authority=authority,
      timeout_seconds=send_timeout,
    )
    parent_sock = None
    child = AutonomousApprovalChannelChild(
      child_sock,
      authority=authority,
      timeout_seconds=receive_timeout,
    )
    child_sock = None
    return AutonomousApprovalChannelPair(
      parent=parent,
      child=child,
    )
  except BaseException:
    for sock in (parent_sock, child_sock):
      if sock is not None:
        try:
          sock.close()
        except OSError:
          pass
    raise


def adopt_inherited_autonomous_approval_channel(
  fd: int,
  *,
  authority: AutonomousApprovalChannelAuthority,
  receive_timeout_seconds: float = (
    AUTONOMOUS_APPROVAL_CHANNEL_DEFAULT_RECEIVE_TIMEOUT_SECONDS
  ),
) -> AutonomousApprovalChannelChild:
  if type(fd) is not int or fd < 0:
    raise ValueError(
      "autonomous approval inherited descriptor must be nonnegative"
    )
  try:
    os.set_inheritable(fd, False)
  except OSError as exc:
    try:
      os.close(fd)
    except OSError:
      pass
    raise AutonomousApprovalChannelTransportError(
      "autonomous approval descriptor could not be made non-inheritable"
    ) from exc
  sock: socket.socket | None = None
  try:
    sock = socket.socket(fileno=fd)
    _validate_socket(sock)
    child = AutonomousApprovalChannelChild(
      sock,
      authority=authority,
      timeout_seconds=_timeout(
        receive_timeout_seconds,
        field_name="receive_timeout_seconds",
      ),
    )
    sock = None
    return child
  except BaseException:
    if sock is not None:
      sock.close()
    else:
      try:
        os.close(fd)
      except OSError:
        pass
    raise


__all__ = [
  "AUTONOMOUS_APPROVAL_CHANNEL_DEFAULT_RECEIVE_TIMEOUT_SECONDS",
  "AUTONOMOUS_APPROVAL_CHANNEL_DEFAULT_SEND_TIMEOUT_SECONDS",
  "AUTONOMOUS_APPROVAL_CHANNEL_FD_ENV",
  "AUTONOMOUS_APPROVAL_CHANNEL_MAX_FRAME_BYTES",
  "AUTONOMOUS_APPROVAL_CHANNEL_MAX_TIMEOUT_SECONDS",
  "AUTONOMOUS_APPROVAL_CHANNEL_VERSION",
  "AutonomousApprovalChannelAuthority",
  "AutonomousApprovalChannelChild",
  "AutonomousApprovalChannelError",
  "AutonomousApprovalChannelPair",
  "AutonomousApprovalChannelParent",
  "AutonomousApprovalChannelProtocolError",
  "AutonomousApprovalChannelStateError",
  "AutonomousApprovalChannelTimeout",
  "AutonomousApprovalChannelTransportError",
  "AutonomousApprovalDecision",
  "ReceivedAutonomousApprovalDecision",
  "adopt_inherited_autonomous_approval_channel",
  "create_autonomous_approval_channel",
]
