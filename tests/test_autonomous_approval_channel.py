from __future__ import annotations

import json
import os
import socket
import struct

import pytest

from agent_gateway.autonomous_approval_channel import (
  AUTONOMOUS_APPROVAL_CHANNEL_MAX_FRAME_BYTES,
  AutonomousApprovalChannelAuthority,
  AutonomousApprovalChannelProtocolError,
  AutonomousApprovalChannelStateError,
  AutonomousApprovalChannelTimeout,
  AutonomousApprovalChannelTransportError,
  AutonomousApprovalDecision,
  adopt_inherited_autonomous_approval_channel,
  create_autonomous_approval_channel,
)


_HEADER = struct.Struct("!I")


def _authority(
  *,
  task_id: str = "bg_7",
) -> AutonomousApprovalChannelAuthority:
  return AutonomousApprovalChannelAuthority(
    launch_nonce="a" * 32,
    task_id=task_id,
    control_run_id="run-7",
    session_id="bg_7",
    channel_id="b" * 64,
  )


def _decision(
  authority: AutonomousApprovalChannelAuthority,
  *,
  sequence: int = 1,
  approval_id: str = "approval-1",
  approved: bool = True,
) -> AutonomousApprovalDecision:
  return AutonomousApprovalDecision(
    authority=authority,
    delivery_sequence=sequence,
    approval_id=approval_id,
    tool_call_id=f"tool-{approval_id}",
    nonce=f"nonce-{approval_id}",
    approved=approved,
    decided_at_ns=1_900_000_000_000_000_000,
  )


def _canonical_frame(
  value: dict[str, object],
) -> bytes:
  body = json.dumps(
    value,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")
  return _HEADER.pack(len(body)) + body


def _raw_child(
  authority: AutonomousApprovalChannelAuthority,
):
  parent_sock, child_sock = socket.socketpair(
    socket.AF_UNIX,
    socket.SOCK_STREAM,
  )
  parent_sock.shutdown(socket.SHUT_RD)
  child_sock.shutdown(socket.SHUT_WR)
  child_fd = child_sock.detach()
  child = adopt_inherited_autonomous_approval_channel(
    child_fd,
    authority=authority,
    receive_timeout_seconds=0.2,
  )
  return parent_sock, child


def test_channel_delivers_exact_decision_and_exact_duplicate() -> None:
  authority = _authority()
  pair = create_autonomous_approval_channel(
    authority=authority,
  )
  decision = _decision(authority)
  try:
    assert pair.parent.send(decision) is decision
    received = pair.child.receive()
    assert received.decision == decision
    assert received.duplicate is False

    assert pair.parent.send(decision) is decision
    duplicate = pair.child.receive()
    assert duplicate.decision is received.decision
    assert duplicate.duplicate is True
    pair.parent.require_sent(decision)
  finally:
    pair.close()


def test_channel_descriptors_are_noninheritable_and_one_way() -> None:
  pair = create_autonomous_approval_channel(
    authority=_authority(),
  )
  try:
    assert os.get_inheritable(pair.parent.fileno()) is False
    assert os.get_inheritable(pair.child.fileno()) is False
    duplicate_fd = os.dup(pair.child.fileno())
    child_probe = socket.socket(fileno=duplicate_fd)
    try:
      with pytest.raises(OSError):
        child_probe.send(b"forged-parent-frame")
    finally:
      child_probe.close()
  finally:
    pair.close()


def test_child_descriptor_transfer_preserves_inherited_connection() -> None:
  authority = _authority()
  pair = create_autonomous_approval_channel(
    authority=authority,
  )
  inherited_child = None
  inherited_fd = pair.child.take_inherited_fd()
  try:
    assert pair.child.fileno() == -1
    inherited_child = adopt_inherited_autonomous_approval_channel(
      inherited_fd,
      authority=authority,
    )
    decision = _decision(authority)

    pair.parent.send(decision)

    assert inherited_child.receive().decision == decision
    with pytest.raises(
      AutonomousApprovalChannelStateError,
      match="unavailable",
    ):
      pair.child.take_inherited_fd()
  finally:
    if inherited_child is not None:
      inherited_child.close()
    else:
      os.close(inherited_fd)
    pair.close()


def test_parent_rejects_conflicting_reuse_and_fails_closed() -> None:
  authority = _authority()
  pair = create_autonomous_approval_channel(
    authority=authority,
  )
  first = _decision(authority)
  conflict = AutonomousApprovalDecision(
    authority=authority,
    delivery_sequence=first.delivery_sequence,
    approval_id=first.approval_id,
    tool_call_id=first.tool_call_id,
    nonce=first.nonce,
    approved=False,
    decided_at_ns=first.decided_at_ns,
  )
  try:
    pair.parent.send(first)
    assert pair.child.receive().decision == first
    with pytest.raises(
      AutonomousApprovalChannelProtocolError,
      match="reused",
    ):
      pair.parent.send(conflict)
    assert pair.parent.fileno() == -1
  finally:
    pair.close()


def test_child_rejects_cross_run_frame() -> None:
  authority = _authority()
  parent_sock, child = _raw_child(authority)
  try:
    forged = _decision(_authority(task_id="bg_8")).frame()
    parent_sock.sendall(_canonical_frame(forged))
    with pytest.raises(
      AutonomousApprovalChannelProtocolError,
      match="changed admitted authority",
    ):
      child.receive()
    assert child.fileno() == -1
  finally:
    parent_sock.close()
    child.close()


def test_child_rejects_noncanonical_and_duplicate_key_json() -> None:
  authority = _authority()
  parent_sock, child = _raw_child(authority)
  try:
    canonical_body = json.dumps(
      _decision(authority).frame(),
      sort_keys=True,
      separators=(",", ":"),
    )
    body = (
      canonical_body[:-1]
      + ',"approved":true}'
    ).encode("utf-8")
    parent_sock.sendall(_HEADER.pack(len(body)) + body)
    with pytest.raises(
      AutonomousApprovalChannelProtocolError,
      match="malformed",
    ):
      child.receive()
  finally:
    parent_sock.close()
    child.close()


@pytest.mark.parametrize("body_length", [0, AUTONOMOUS_APPROVAL_CHANNEL_MAX_FRAME_BYTES + 1])
def test_child_rejects_invalid_frame_length(
  body_length: int,
) -> None:
  authority = _authority()
  parent_sock, child = _raw_child(authority)
  try:
    parent_sock.sendall(_HEADER.pack(body_length))
    with pytest.raises(
      AutonomousApprovalChannelProtocolError,
      match="length",
    ):
      child.receive()
  finally:
    parent_sock.close()
    child.close()


def test_child_rejects_partial_frame_eof() -> None:
  authority = _authority()
  parent_sock, child = _raw_child(authority)
  parent_sock.sendall(_HEADER.pack(100) + b"{")
  parent_sock.close()
  try:
    with pytest.raises(
      AutonomousApprovalChannelTransportError,
      match="closed during frame body",
    ):
      child.receive()
  finally:
    child.close()


def test_child_receive_is_bounded_without_polling() -> None:
  authority = _authority()
  pair = create_autonomous_approval_channel(
    authority=authority,
    receive_timeout_seconds=0.01,
  )
  try:
    with pytest.raises(AutonomousApprovalChannelTimeout):
      pair.child.receive()
    assert pair.child.fileno() == -1
  finally:
    pair.close()


def test_parent_send_fails_closed_on_child_backpressure() -> None:
  authority = _authority()
  pair = create_autonomous_approval_channel(
    authority=authority,
    send_timeout_seconds=0.01,
  )
  probe = socket.socket(fileno=os.dup(pair.parent.fileno()))
  probe.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
  probe.close()
  timed_out = False
  try:
    for sequence in range(1, 257):
      decision = _decision(
        authority,
        sequence=sequence,
        approval_id=f"approval-{sequence}-" + ("x" * 400),
      )
      try:
        pair.parent.send(decision)
      except AutonomousApprovalChannelTimeout:
        timed_out = True
        break
    assert timed_out is True
    assert pair.parent.fileno() == -1
  finally:
    pair.close()
