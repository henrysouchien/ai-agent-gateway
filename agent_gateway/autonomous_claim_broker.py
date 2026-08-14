from __future__ import annotations

import json
import math
import os
import re
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping

from .autonomous_admission_ledger import (
  AutonomousAdmissionRecord,
  AutonomousAdmissionLedgerError,
  AutonomousAdmissionLedgerIdentity,
  OrdinaryAutonomousAdmissionReceipt,
  consume_ordinary_autonomous_launch_once,
)
from .autonomous_launch_envelope import (
  AutonomousLaunchEnvelope,
  _decode_broker_verified_autonomous_launch_envelope,
)
from .claim_signing_authority import GatewayClaimSigningAuthority


AUTONOMOUS_CLAIM_BROKER_FD_ENV = (
  "AGENT_AUTONOMOUS_CLAIM_BROKER_FD"
)
AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION = 2
AUTONOMOUS_CLAIM_BROKER_MAX_FRAME_BYTES = 6 * 1024 * 1024
AUTONOMOUS_CLAIM_BROKER_MAX_REQUESTS = 256
AUTONOMOUS_CLAIM_BROKER_MAX_TTL_SECONDS = 600
AUTONOMOUS_CLAIM_BROKER_IO_TIMEOUT_SECONDS = 5.0
AUTONOMOUS_CLAIM_BROKER_MAX_IO_TIMEOUT_SECONDS = 60.0

_CLAIM_FIELDS = frozenset({
  "AGENT_API_CLAIM_AUDIENCE",
  "AGENT_API_CLAIM_ISSUED_AT",
  "AGENT_API_CLAIM_EXPIRY",
  "AGENT_API_CLAIM_USER_ID",
  "AGENT_API_CLAIM_USER_EMAIL",
  "AGENT_API_CLAIM_NONCE",
  "AGENT_API_CLAIM_SIGNATURE",
})
_REQUEST_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{64}$")


class AutonomousClaimBrokerError(RuntimeError):
  """The private autonomous signing channel failed closed."""


def _closed_object(
  pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
  result: dict[str, Any] = {}
  for key, value in pairs:
    if key in result:
      raise ValueError(
        f"autonomous claim broker duplicate field: {key}"
      )
    result[key] = value
  return result


def _canonical_json(value: object) -> bytes:
  return json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
  ).encode("utf-8")


def _io_timeout_seconds(value: object) -> float:
  if (
    isinstance(value, bool)
    or not isinstance(value, (int, float))
  ):
    raise ValueError(
      "autonomous claim broker I/O timeout must be numeric"
    )
  timeout = float(value)
  if (
    not math.isfinite(timeout)
    or not 0 < timeout
    <= AUTONOMOUS_CLAIM_BROKER_MAX_IO_TIMEOUT_SECONDS
  ):
    raise ValueError(
      "autonomous claim broker I/O timeout is invalid"
    )
  return timeout


def _io_deadline(timeout_seconds: float) -> float:
  return time.monotonic() + timeout_seconds


def _wait_for_socket(
  sock: socket.socket,
  *,
  readable: bool,
  deadline: float,
) -> None:
  while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
      raise AutonomousClaimBrokerError(
        "autonomous claim broker I/O deadline exceeded"
      )
    try:
      readers, writers, _ = select.select(
        (sock,) if readable else (),
        () if readable else (sock,),
        (),
        remaining,
      )
    except InterruptedError:
      continue
    except (OSError, ValueError) as exc:
      raise AutonomousClaimBrokerError(
        "autonomous claim broker channel is unavailable"
      ) from exc
    if readers or writers:
      return
    raise AutonomousClaimBrokerError(
      "autonomous claim broker I/O deadline exceeded"
    )


def _send_all(
  sock: socket.socket,
  payload: bytes,
  *,
  deadline: float,
) -> None:
  remaining = memoryview(payload)
  while remaining:
    _wait_for_socket(
      sock,
      readable=False,
      deadline=deadline,
    )
    try:
      sent = sock.send(remaining)
    except (BlockingIOError, InterruptedError):
      continue
    if sent <= 0:
      raise AutonomousClaimBrokerError(
        "autonomous claim broker channel closed"
      )
    remaining = remaining[sent:]


def _send_frame(
  sock: socket.socket,
  payload: object,
  *,
  deadline: float,
) -> None:
  encoded = _canonical_json(payload)
  if len(encoded) > AUTONOMOUS_CLAIM_BROKER_MAX_FRAME_BYTES:
    raise AutonomousClaimBrokerError(
      "autonomous claim broker frame exceeds its byte bound"
    )
  _send_all(
    sock,
    struct.pack("!I", len(encoded)) + encoded,
    deadline=deadline,
  )


def _recv_exact(
  sock: socket.socket,
  length: int,
  *,
  deadline: float,
) -> bytes:
  chunks: list[bytes] = []
  remaining = length
  while remaining:
    _wait_for_socket(
      sock,
      readable=True,
      deadline=deadline,
    )
    try:
      chunk = sock.recv(remaining)
    except (BlockingIOError, InterruptedError):
      continue
    if not chunk:
      raise AutonomousClaimBrokerError(
        "autonomous claim broker channel closed"
      )
    chunks.append(chunk)
    remaining -= len(chunk)
  return b"".join(chunks)


def _recv_frame(
  sock: socket.socket,
  *,
  deadline: float,
  progress_timeout_seconds: float | None = None,
) -> dict[str, Any]:
  if progress_timeout_seconds is None:
    header = _recv_exact(sock, 4, deadline=deadline)
    progress_deadline = deadline
  else:
    first_header_byte = _recv_exact(
      sock,
      1,
      deadline=deadline,
    )
    progress_deadline = _io_deadline(
      progress_timeout_seconds
    )
    header = first_header_byte + _recv_exact(
      sock,
      3,
      deadline=progress_deadline,
    )
  (length,) = struct.unpack("!I", header)
  if (
    length < 2
    or length > AUTONOMOUS_CLAIM_BROKER_MAX_FRAME_BYTES
  ):
    raise AutonomousClaimBrokerError(
      "autonomous claim broker frame length is invalid"
    )
  raw = _recv_exact(
    sock,
    length,
    deadline=progress_deadline,
  )
  try:
    text = raw.decode("utf-8")
    payload = json.loads(
      text,
      object_pairs_hook=_closed_object,
      parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"invalid JSON constant: {value}")
      ),
    )
  except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
    raise AutonomousClaimBrokerError(
      "autonomous claim broker frame is invalid JSON"
    ) from exc
  if type(payload) is not dict or _canonical_json(payload) != raw:
    raise AutonomousClaimBrokerError(
      "autonomous claim broker frame must be a canonical object"
    )
  return payload


def _exact_fields(
  payload: Mapping[str, Any],
  fields: frozenset[str],
  *,
  message: str,
) -> None:
  if set(payload) != fields:
    raise AutonomousClaimBrokerError(message)


@dataclass(frozen=True, slots=True)
class _BrokerBinding:
  envelope_json: str
  envelope: AutonomousLaunchEnvelope
  user_id: str
  user_email: str | None
  session_expires_at: int
  monotonic_expires_at: float


class AutonomousClaimBroker:
  """Parent-owned, bounded signer for one admitted autonomous launch."""

  __slots__ = (
    "_authority",
    "_binding",
    "_child_socket",
    "_closed",
    "_lock",
    "_last_wall_time",
    "_max_requests",
    "_max_ttl_seconds",
    "_io_timeout_seconds",
    "_server_socket",
    "_thread",
  )

  def __init__(
    self,
    authority: GatewayClaimSigningAuthority,
    envelope_json: str,
    *,
    max_requests: int = AUTONOMOUS_CLAIM_BROKER_MAX_REQUESTS,
    max_ttl_seconds: int = (
      AUTONOMOUS_CLAIM_BROKER_MAX_TTL_SECONDS
    ),
    io_timeout_seconds: float = (
      AUTONOMOUS_CLAIM_BROKER_IO_TIMEOUT_SECONDS
    ),
  ) -> None:
    if type(authority) is not GatewayClaimSigningAuthority:
      raise TypeError(
        "autonomous claim broker requires exact signing authority"
      )
    if (
      isinstance(max_requests, bool)
      or not isinstance(max_requests, int)
      or not 1
      <= max_requests
      <= AUTONOMOUS_CLAIM_BROKER_MAX_REQUESTS
    ):
      raise ValueError(
        "autonomous claim broker max_requests is invalid"
      )
    if (
      isinstance(max_ttl_seconds, bool)
      or not isinstance(max_ttl_seconds, int)
      or not 1
      <= max_ttl_seconds
      <= AUTONOMOUS_CLAIM_BROKER_MAX_TTL_SECONDS
    ):
      raise ValueError(
        "autonomous claim broker max_ttl_seconds is invalid"
      )
    normalized_io_timeout_seconds = _io_timeout_seconds(
      io_timeout_seconds
    )
    envelope = authority.verify_autonomous_launch_envelope(
      envelope_json
    )
    if type(envelope) is not AutonomousLaunchEnvelope:
      raise TypeError(
        "autonomous claim broker verifier returned wrong type"
      )
    session = envelope.session_authority.to_gateway_session()
    if session.user_id != envelope.owner_user_id:
      raise RuntimeError(
        "autonomous claim broker launch identity is inconsistent"
      )
    initial_wall_time = time.time()
    remaining_session_seconds = (
      float(session.expires_at) - initial_wall_time
    )
    if remaining_session_seconds <= 0:
      raise RuntimeError(
        "autonomous claim broker session authority has expired"
      )
    monotonic_expires_at = (
      time.monotonic() + remaining_session_seconds
    )
    server_socket: socket.socket | None = None
    child_socket: socket.socket | None = None
    initialized = False
    try:
      server_socket, child_socket = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_STREAM,
      )
      server_socket.set_inheritable(False)
      child_socket.set_inheritable(False)
      server_socket.setblocking(False)
      child_socket.setblocking(False)
      self._authority = authority
      self._binding = _BrokerBinding(
        envelope_json=envelope_json,
        envelope=envelope,
        user_id=session.user_id,
        user_email=session.user_email,
        session_expires_at=session.expires_at,
        monotonic_expires_at=monotonic_expires_at,
      )
      self._server_socket = server_socket
      self._child_socket = child_socket
      self._max_requests = max_requests
      self._max_ttl_seconds = max_ttl_seconds
      self._io_timeout_seconds = normalized_io_timeout_seconds
      self._last_wall_time = initial_wall_time
      self._closed = False
      self._lock = threading.RLock()
      self._thread = threading.Thread(
        target=self._serve,
        name=f"autonomous-claim-broker-{envelope.task_id}",
        daemon=True,
      )
      initialized = True
      self._thread.start()
    except BaseException:
      if initialized:
        self.close()
      else:
        if child_socket is not None:
          child_socket.close()
        if server_socket is not None:
          server_socket.close()
      raise

  @property
  def envelope(self) -> AutonomousLaunchEnvelope:
    return self._binding.envelope

  def take_child_fd(self) -> int:
    with self._lock:
      if self._closed or self._child_socket is None:
        raise RuntimeError(
          "autonomous claim broker child descriptor is unavailable"
        )
      child_socket = self._child_socket
      self._child_socket = None
      fd = child_socket.detach()
    try:
      os.set_inheritable(fd, False)
      if os.get_inheritable(fd):
        raise OSError("descriptor remained inheritable")
    except BaseException:
      try:
        os.close(fd)
      except OSError:
        pass
      raise
    return fd

  def close(self) -> None:
    child_socket: socket.socket | None
    with self._lock:
      if self._closed:
        return
      self._closed = True
      child_socket = self._child_socket
      self._child_socket = None
    if child_socket is not None:
      child_socket.close()
    try:
      self._server_socket.shutdown(socket.SHUT_RDWR)
    except OSError:
      pass
    self._server_socket.close()
    if (
      self._thread is not threading.current_thread()
      and self._thread.is_alive()
    ):
      self._thread.join()

  def _reject(self, code: str) -> None:
    try:
      _send_frame(
        self._server_socket,
        {
          "error": code,
          "ok": False,
          "version": AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION,
        },
        deadline=_io_deadline(self._io_timeout_seconds),
      )
    except (AutonomousClaimBrokerError, OSError):
      pass

  def _serve(self) -> None:
    try:
      # The child must boot the full runtime (interpreter start plus package
      # imports) before it can send its admission frame, and boot time varies
      # with cache state and concurrent launches. Wait for the first byte
      # until the signed session authority expires — the actual security
      # boundary — and only then hold the frame to the tight I/O timeout.
      admission = _recv_frame(
        self._server_socket,
        deadline=self._binding.monotonic_expires_at,
        progress_timeout_seconds=self._io_timeout_seconds,
      )
      _exact_fields(
        admission,
        frozenset({"envelope", "op", "version"}),
        message="autonomous claim broker admission contract is invalid",
      )
      if (
        admission["op"] != "admit"
        or admission["version"]
        != AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION
        or admission["envelope"] != self._binding.envelope_json
      ):
        raise AutonomousClaimBrokerError(
          "autonomous claim broker admission binding is invalid"
        )
      envelope = self._binding.envelope
      ordinary_admission = consume_ordinary_autonomous_launch_once(
        AutonomousAdmissionLedgerIdentity.from_verified_envelope(
          envelope
        ),
        OrdinaryAutonomousAdmissionReceipt.from_verified_envelope(
          envelope
        ),
      )
      if type(ordinary_admission) is not AutonomousAdmissionRecord:
        raise AutonomousClaimBrokerError(
          "autonomous claim broker admission ledger returned "
          "invalid authority"
        )
      _send_frame(
        self._server_socket,
        {
          "channel_id": envelope.channel_id,
          "control_run_id": envelope.control_run_id,
          "envelope": self._binding.envelope_json,
          "ok": True,
          "op": "admit",
          "ordinary_admission": (
            ordinary_admission.authority_receipt()
          ),
          "task_id": envelope.task_id,
          "user_email": self._binding.user_email,
          "user_id": self._binding.user_id,
          "version": AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION,
        },
        deadline=_io_deadline(self._io_timeout_seconds),
      )
      request_count = 0
      while request_count < self._max_requests:
        request = _recv_frame(
          self._server_socket,
          deadline=self._binding.monotonic_expires_at,
          progress_timeout_seconds=self._io_timeout_seconds,
        )
        _exact_fields(
          request,
          frozenset({
            "op",
            "request_id",
            "ttl_seconds",
            "version",
          }),
          message="autonomous claim broker sign contract is invalid",
        )
        request_id = request["request_id"]
        ttl_seconds = request["ttl_seconds"]
        if (
          request["op"] != "sign"
          or request["version"]
          != AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION
          or type(request_id) is not str
          or _REQUEST_ID_RE.fullmatch(request_id) is None
          or isinstance(ttl_seconds, bool)
          or not isinstance(ttl_seconds, int)
          or not 1 <= ttl_seconds <= self._max_ttl_seconds
        ):
          raise AutonomousClaimBrokerError(
            "autonomous claim broker sign request is invalid"
          )
        current_wall_time = time.time()
        current_monotonic_time = time.monotonic()
        if current_wall_time < self._last_wall_time:
          raise AutonomousClaimBrokerError(
            "autonomous claim broker detected wall-clock rollback"
          )
        self._last_wall_time = current_wall_time
        if (
          current_wall_time + ttl_seconds
          > self._binding.session_expires_at
          or current_monotonic_time + ttl_seconds
          > self._binding.monotonic_expires_at
        ):
          raise AutonomousClaimBrokerError(
            "autonomous claim broker session authority has expired"
          )
        claim = self._authority.sign_user_claim(
          user_id=self._binding.user_id,
          user_email=self._binding.user_email,
          ttl_seconds=ttl_seconds,
        )
        _send_frame(
          self._server_socket,
          {
            "claim": claim,
            "ok": True,
            "op": "sign",
            "request_id": request_id,
            "version": AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION,
          },
          deadline=_io_deadline(self._io_timeout_seconds),
        )
        request_count += 1
      self._reject("request_limit_exhausted")
    except (AutonomousClaimBrokerError, AutonomousAdmissionLedgerError):
      self._reject("broker_rejected_request")
    except (OSError, ValueError):
      self._reject("broker_failure")
    finally:
      self.close()


class AutonomousClaimSigner:
  """Child-side client bound to one broker-admitted user and launch."""

  __slots__ = (
    "_channel_id",
    "_closed",
    "_control_run_id",
    "_envelope",
    "_io_timeout_seconds",
    "_lock",
    "_ordinary_admission",
    "_socket",
    "_task_id",
    "_user_email",
    "_user_id",
  )

  def __init__(
    self,
    inherited_fd: int,
    *,
    envelope_json: str,
    io_timeout_seconds: float = (
      AUTONOMOUS_CLAIM_BROKER_IO_TIMEOUT_SECONDS
    ),
  ) -> None:
    if type(inherited_fd) is not int or inherited_fd < 0:
      raise ValueError(
        "autonomous claim broker descriptor must be nonnegative"
      )
    normalized_io_timeout_seconds = _io_timeout_seconds(
      io_timeout_seconds
    )
    broker_socket: socket.socket | None = None
    try:
      os.set_inheritable(inherited_fd, False)
      if os.get_inheritable(inherited_fd):
        raise OSError("descriptor remained inheritable")
      broker_socket = socket.socket(fileno=inherited_fd)
      broker_socket.setblocking(False)
    except BaseException as exc:
      if broker_socket is not None:
        broker_socket.close()
      else:
        try:
          os.close(inherited_fd)
        except OSError:
          pass
      if not isinstance(exc, Exception):
        raise
      raise AutonomousClaimBrokerError(
        "autonomous claim broker descriptor is invalid"
      ) from exc
    self._socket = broker_socket
    self._io_timeout_seconds = normalized_io_timeout_seconds
    self._lock = threading.RLock()
    self._closed = False
    try:
      deadline = _io_deadline(self._io_timeout_seconds)
      _send_frame(
        broker_socket,
        {
          "envelope": envelope_json,
          "op": "admit",
          "version": AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION,
        },
        deadline=deadline,
      )
      response = _recv_frame(
        broker_socket,
        deadline=deadline,
      )
      _exact_fields(
        response,
        frozenset({
          "channel_id",
          "control_run_id",
          "envelope",
          "ok",
          "op",
          "ordinary_admission",
          "task_id",
          "user_email",
          "user_id",
          "version",
        }),
        message="autonomous claim broker admission response is invalid",
      )
      if (
        response["ok"] is not True
        or response["op"] != "admit"
        or response["version"]
        != AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION
        or response["envelope"] != envelope_json
      ):
        raise AutonomousClaimBrokerError(
          "autonomous claim broker rejected admission"
        )
      envelope = (
        _decode_broker_verified_autonomous_launch_envelope(
          envelope_json
        )
      )
      session = envelope.session_authority.to_gateway_session()
      if (
        response["task_id"] != envelope.task_id
        or response["control_run_id"] != envelope.control_run_id
        or response["channel_id"] != envelope.channel_id
        or response["user_id"] != session.user_id
        or response["user_email"] != session.user_email
      ):
        raise AutonomousClaimBrokerError(
          "autonomous claim broker admission identity is invalid"
        )
      ordinary_admission_payload = response["ordinary_admission"]
      try:
        ordinary_admission = (
          AutonomousAdmissionRecord.from_authority_receipt(
            ordinary_admission_payload
          )
        )
        expected_receipt = (
          OrdinaryAutonomousAdmissionReceipt.from_verified_envelope(
            envelope
          )
        )
      except (TypeError, ValueError) as exc:
        raise AutonomousClaimBrokerError(
          "autonomous claim broker ordinary admission is invalid"
        ) from exc
      if ordinary_admission.receipt != expected_receipt:
        raise AutonomousClaimBrokerError(
          "autonomous claim broker ordinary admission changed "
          "launch authority"
        )
      self._envelope = envelope
      self._ordinary_admission = ordinary_admission
      self._task_id = envelope.task_id
      self._control_run_id = envelope.control_run_id
      self._channel_id = envelope.channel_id
      self._user_id = session.user_id
      self._user_email = session.user_email
    except BaseException:
      broker_socket.close()
      self._closed = True
      raise

  @property
  def envelope(self) -> AutonomousLaunchEnvelope:
    return self._envelope

  @property
  def consumed_ordinary_admission(
    self,
  ) -> AutonomousAdmissionRecord:
    return self._ordinary_admission

  @property
  def user_id(self) -> str:
    return self._user_id

  @property
  def user_email(self) -> str | None:
    return self._user_email

  def sign_claim(self, *, ttl_seconds: int) -> dict[str, str]:
    if (
      isinstance(ttl_seconds, bool)
      or not isinstance(ttl_seconds, int)
      or not 1
      <= ttl_seconds
      <= AUTONOMOUS_CLAIM_BROKER_MAX_TTL_SECONDS
    ):
      raise ValueError(
        "autonomous claim signer ttl_seconds is invalid"
      )
    request_id = os.urandom(16).hex()
    with self._lock:
      if self._closed:
        raise AutonomousClaimBrokerError(
          "autonomous claim signer is closed"
        )
      try:
        deadline = _io_deadline(self._io_timeout_seconds)
        _send_frame(
          self._socket,
          {
            "op": "sign",
            "request_id": request_id,
            "ttl_seconds": ttl_seconds,
            "version": AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION,
          },
          deadline=deadline,
        )
        response = _recv_frame(
          self._socket,
          deadline=deadline,
        )
      except (AutonomousClaimBrokerError, OSError) as exc:
        self.close()
        raise AutonomousClaimBrokerError(
          "autonomous claim broker signing failed"
        ) from exc
    try:
      _exact_fields(
        response,
        frozenset({
          "claim",
          "ok",
          "op",
          "request_id",
          "version",
        }),
        message="autonomous claim broker sign response is invalid",
      )
    except AutonomousClaimBrokerError:
      self.close()
      raise
    if (
      response["ok"] is not True
      or response["op"] != "sign"
      or response["request_id"] != request_id
      or response["version"]
      != AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION
      or type(response["claim"]) is not dict
    ):
      self.close()
      raise AutonomousClaimBrokerError(
        "autonomous claim broker sign response is invalid"
      )
    claim = response["claim"]
    try:
      _exact_fields(
        claim,
        _CLAIM_FIELDS,
        message="autonomous claim broker claim is invalid",
      )
    except AutonomousClaimBrokerError:
      self.close()
      raise
    try:
      issued_at = int(claim["AGENT_API_CLAIM_ISSUED_AT"])
      expiry = int(claim["AGENT_API_CLAIM_EXPIRY"])
    except (TypeError, ValueError) as exc:
      self.close()
      raise AutonomousClaimBrokerError(
        "autonomous claim broker claim lifetime is invalid"
      ) from exc
    if (
      claim["AGENT_API_CLAIM_AUDIENCE"] != "agent_api_v1"
      or claim["AGENT_API_CLAIM_USER_ID"] != self._user_id
      or claim["AGENT_API_CLAIM_USER_EMAIL"]
      != (self._user_email or "")
      or expiry - issued_at != ttl_seconds
      or issued_at > int(time.time()) + 5
      or expiry <= int(time.time())
      or type(claim["AGENT_API_CLAIM_NONCE"]) is not str
      or _NONCE_RE.fullmatch(
        claim["AGENT_API_CLAIM_NONCE"]
      ) is None
      or type(claim["AGENT_API_CLAIM_SIGNATURE"]) is not str
      or _SIGNATURE_RE.fullmatch(
        claim["AGENT_API_CLAIM_SIGNATURE"]
      ) is None
    ):
      self.close()
      raise AutonomousClaimBrokerError(
        "autonomous claim broker claim binding is invalid"
      )
    return dict(claim)

  def close(self) -> None:
    with self._lock:
      if self._closed:
        return
      self._closed = True
      self._socket.close()


def adopt_inherited_autonomous_claim_signer(
  inherited_fd: int,
  *,
  envelope_json: str,
  io_timeout_seconds: float = (
    AUTONOMOUS_CLAIM_BROKER_IO_TIMEOUT_SECONDS
  ),
) -> AutonomousClaimSigner:
  return AutonomousClaimSigner(
    inherited_fd,
    envelope_json=envelope_json,
    io_timeout_seconds=io_timeout_seconds,
  )


__all__ = [
  "AUTONOMOUS_CLAIM_BROKER_FD_ENV",
  "AUTONOMOUS_CLAIM_BROKER_MAX_FRAME_BYTES",
  "AUTONOMOUS_CLAIM_BROKER_IO_TIMEOUT_SECONDS",
  "AUTONOMOUS_CLAIM_BROKER_MAX_IO_TIMEOUT_SECONDS",
  "AUTONOMOUS_CLAIM_BROKER_MAX_REQUESTS",
  "AUTONOMOUS_CLAIM_BROKER_MAX_TTL_SECONDS",
  "AUTONOMOUS_CLAIM_BROKER_PROTOCOL_VERSION",
  "AutonomousClaimBroker",
  "AutonomousClaimBrokerError",
  "AutonomousClaimSigner",
  "adopt_inherited_autonomous_claim_signer",
]
