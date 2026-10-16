from __future__ import annotations

import asyncio
import copy
import logging
import multiprocessing
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import json
import os
import socket
import struct

import pytest

from agent_gateway.autonomous_approval_channel import (
  AutonomousApprovalChannelAuthority,
  AutonomousApprovalChannelProtocolError,
  AutonomousApprovalChannelStateError,
  AutonomousApprovalChannelTimeout,
  AutonomousApprovalChannelTransportError,
  AutonomousApprovalDecision,
  adopt_inherited_autonomous_approval_channel,
  create_autonomous_approval_channel,
)
from agent_gateway import AgentRunner, EventLog, SessionContextBuilder, ToolDispatcher
from agent_gateway.admission import capture_job_context, root_context
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.approval_policy import ApprovalVote, RunContext
from agent_gateway.approval_route import ParentDelegatedApprovalRoute
from agent_gateway.approval_store import ApprovalExecutableBindingError, SQLiteApprovalStore
from agent_gateway.autonomous_claim_broker import AutonomousClaimBroker
from agent_gateway.autonomous_launch_envelope import (
  sign_autonomous_launch_envelope,
  verify_autonomous_launch_envelope,
)
from agent_gateway.autonomous_runner import AutonomousRegistry
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.control_run_lifecycle import canonical_control_run_state
from agent_gateway.control_plane.autonomous_approval_delivery import (
  admit_delegated_executable_capture,
  autonomous_decision_unavailable,
  autonomous_approval_delivery_context,
  create_delegated_autonomous_approval,
  claim_continued_executable_attempt,
  decided_approval_continuation,
  deliver_autonomous_approval_outbox,
  record_continued_executable_result,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.sub_agent import ApprovedCallRef
from agent_gateway.tool_dispatcher_approval_lifecycle import (
  ApprovalPark,
  ExecutableCapture,
  await_user_approval_via_pending_tools,
  run_approval_lifecycle,
)
from agent_gateway.tool_policy_registry import PreparedToolCall
from gateway_test_support.autonomous_launch_test_support import (
  _ordinary_session_authority,
  _workload,
)
from gateway_test_support.capability_execution_test_support import (
  stub_bound_capability_execution,
)
from gateway_test_support.host_policy import owner_session_host_policy  # noqa: F401

import sys

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
  sys.path.insert(0, str(TESTS_DIR))

from control_plane.manifest_helpers import write_v6_manifest  # noqa: E402


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


def test_channel_delivers_past_the_former_decision_quota_and_reports_it() -> None:
  authority = _authority()
  pair = create_autonomous_approval_channel(
    authority=authority,
  )
  try:
    last = None
    for sequence in range(1, 258):
      last = _decision(
        authority,
        sequence=sequence,
        approval_id=f"approval-{sequence}",
      )
      pair.parent.send(last)
      assert pair.child.receive().decision == last
    assert last is not None
    assert pair.parent.decisions_sent == 257
    last_body = json.dumps(
      last.frame(),
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
    ).encode("utf-8")
    assert pair.parent.sent_frame_bytes(257) == len(last_body)
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


@pytest.mark.parametrize("body_length", [0])
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


_CONTINUATION_SECRET = "detached-approval-tests-signing-key-at-least-32-bytes"


class _ContinuationProvider(ModelProvider):
  name = "anthropic"

  def __init__(self) -> None:
    self.requests: list[dict[str, Any]] = []

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    return None

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)

  async def stream(self, client: Any, params: dict[str, Any]):
    self.requests.append(json.loads(json.dumps(params, default=str)))
    restored_note = any(
      block.get("type") == "tool_result" and block.get("tool_use_id") == "call-before"
      for message in params.get("messages", [])
      if isinstance(message.get("content"), list)
      for block in message["content"]
    )
    if not restored_note:
      # Without the restored transcript the model re-requests the completed
      # effect, which the dispatcher test double refuses to run twice.
      yield StreamEvent(type="tool_use_start", tool_id="call-replayed", tool_name="record_note")
      yield StreamEvent(
        type="tool_use_end", tool_id="call-replayed", tool_name="record_note", tool_input={},
        raw_block={"type": "tool_use", "id": "call-replayed", "name": "record_note", "input": {}},
      )
      yield StreamEvent(type="message_end", stop_reason="tool_use")
      return
    yield StreamEvent(type="text_delta", text="Original order decision handled.")
    yield StreamEvent(
      type="text_end",
      raw_block={"type": "text", "text": "Original order decision handled."},
    )
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def _park_approval_process(connection: Any, session_authority: Any, log_path: str, root: str) -> None:
  """Run the real delegated lifecycle in a disposable OS process, without a voter."""
  async def scenario() -> None:
    session = session_authority.to_gateway_session()
    journal = AgentSessionLog(path=Path(log_path))
    session.agent_session_log = journal
    # A completed effect before the money-boundary call must never be replayed.
    Path(root, "preceding-effect.txt").write_text("completed once\n", encoding="utf-8")
    await journal.append({"type": "user_message", "content": "Record the note, then place the order."})
    await journal.append({
      "type": "assistant_message",
      "content_blocks": [{
        "type": "tool_use", "id": "call-before", "name": "record_note", "input": {},
      }],
      "stop_reason": "tool_use",
    })
    await journal.append({
      "type": "tool_call_complete", "tool_call_id": "call-before",
      "tool_name": "record_note", "result": {"note": "completed once"},
      "error": None, "is_error": False,
      "final_tool_result_blocks": [{
        "type": "tool_result", "tool_use_id": "call-before",
        "content": '{"note":"completed once"}',
      }],
    })
    await journal.append({
      "type": "assistant_message",
      "content_blocks": [{
        "type": "tool_use", "id": "call-original", "name": "native_write",
        # The journal is a public projection, not an executable operand source.
        "input": {"units": 999, "account": "[redacted]"},
      }],
      "stop_reason": "tool_use",
    })
    await journal.append({
      "type": "tool_call_start", "tool_call_id": "call-original",
      "tool_name": "native_write", "tool_input": {"units": 999, "account": "[redacted]"},
    })

    async def capture(payload: dict[str, Any]) -> None:
      connection.send(("capture", payload))
      assert connection.recv() == "captured"

    def publish(event: dict[str, Any]) -> None:
      connection.send(("ask", event))
      assert connection.recv() == "persisted"

    async def ask(request: Any, decision: Any, **kwargs: Any) -> dict[str, Any]:
      return await await_user_approval_via_pending_tools(
        session=session, append_event_fn=publish,
        request=request, decision=decision, log=logging.getLogger(__name__), **kwargs,
      )

    pair = create_autonomous_approval_channel(authority=_authority())
    try:
      route = ParentDelegatedApprovalRoute(pair.child, session, private_capture_writer=capture)
      original = PreparedToolCall(
        {"units": 7, "account": "private-original-account"},
        exact_backend="original-order-backend",
      )
      try:
        await run_approval_lifecycle(
          route=route, session=session, producer_session=session,
          executable_capture=ExecutableCapture(
            prepared_call=original,
            dispatch_context={
              "call_index": 7,
              "advertised_tool_names": None,
              "skill_run_id": None,
              "step_id": None,
              "workspace_dir": None,
              "batch_id": None,
              "capture_readable_resource_snapshot": False,
              "allow_uncertain_mcp_replay": False,
            },
            job_id="bg_7", tenant_storage_id="42",
            credential_generation_ref=None, root_turn=None,
          ),
          tool_call_id="call-original", tool_name="native_write",
          tool_input=original.materialize_input(), qualifier="", reason="Place the order",
          allow_persistent=False,
          resolve_run_context_fn=lambda: RunContext(
            user_id="42", request_id="run-7", session_id="bg_7",
            profile="analyst", channel="cli", decider_role="owner",
          ),
          current_skill_admission_fn=lambda: None,
          redact_for_approval_request_fn=lambda *_args: (
            {"units": 7, "account": "[redacted]"}, "original-args-hash",
          ),
          resolve_tool_class_fn=lambda _name: "irreversible",
          effective_trade_approval_decision_fn=lambda _name, _args, decision: decision,
          await_user_approval_via_pending_tools_fn=ask,
        )
      except ApprovalPark as park:
        await journal.append({
          "type": "stream_complete", "terminal_disposition": "awaiting_approval",
          "job_id": park.job_id, "approval_id": park.approval_id,
          "original_tool_call_id": park.original_tool_call_id,
        })
        connection.send(("parked", {
          "job_id": park.job_id, "tool_call_id": park.original_tool_call_id,
          "approval_id": park.approval_id, "nonce": park.nonce,
          "pending_tools": session.pending_tools, "approval_queues": session.approval_queues,
        }))
      else:
        raise AssertionError("The human ask returned without parking or a decision")
    finally:
      pair.close()
      connection.close()

  asyncio.run(scenario())


def _receive_process_message(connection: Any) -> tuple[str, dict[str, Any]]:
  assert connection.poll(10), "Delegated approval process did not publish its next lifecycle event"
  return connection.recv()


@pytest.fixture
def parked_detached_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  """Durable request and original operands left behind by an exited process."""
  import sys
  tests_dir = str(Path(__file__).resolve().parent)
  pkg_dir = str(Path(__file__).resolve().parents[1])
  monkeypatch.syspath_prepend(pkg_dir)
  monkeypatch.syspath_prepend(tests_dir)
  monkeypatch.setitem(sys.modules, "test_autonomous_approval_channel", sys.modules[__name__])
  monkeypatch.setattr(_park_approval_process, "__module__", "test_autonomous_approval_channel")
  provider = _ContinuationProvider()
  execution = stub_bound_capability_execution(
    provider=provider, model="claude-test", effort="none",
    run_mode="autonomous", credential_ref="autonomous-service:test-handle",
  )
  authority = _ordinary_session_authority(bind=execution.bind)
  authority = replace(authority, ordinary_authority=replace(
    authority.ordinary_authority,
    created_at=int(time.time()) - 1, expires_at=int(time.time()) + 600,
  ))
  workload = _workload()
  assert workload.session_log_authority is not None
  workload = replace(workload, session_log_authority=replace(
    workload.session_log_authority, base_path=str(tmp_path / "sessions"),
  ))
  assert workload.session_log_authority is not None
  # Use the journal path the authenticated launch's own layout resolves.
  from agent_gateway.agent_session_log_records import AgentSessionRef, resolve_agent_session_id
  from agent_gateway.openai_history_fence import scope_provider_session_id
  agent_id = scope_provider_session_id(
    workload.profile, provider=execution.bind.provider, durable=True, openai_epoch=None,
  )
  journal_path = AgentSessionLog.path_for_session(
    workload.session_log_authority.base_path,
    AgentSessionRef(
      user_id="42", agent_id=agent_id,
      agent_session_id=resolve_agent_session_id("42", agent_id),
    ),
  )
  write_v6_manifest(
    tmp_path, "bg_7", control_run_id="run-7", user_id="42", owner_user_id="42",
    mode="once", skill=None, channel="cli", capability_bind=execution.bind.to_json(),
  )
  registry = AutonomousRegistry(
    api_dir=Path(__file__).resolve().parents[3] / "api",
    log_dir=tmp_path, approval_store=SQLiteApprovalStore(tmp_path / "approvals.sqlite3"),
  )
  source = registry._tasks["bg_7"]
  raw = sign_autonomous_launch_envelope(
    _CONTINUATION_SECRET, task_id=source.task_id, control_run_id=source.control_run_id,
    owner_user_id="42", channel_id=source.channel_id, bind=execution.bind,
    workload=workload, session_authority=authority, control_authority=source.control_authority,
  )
  envelope = verify_autonomous_launch_envelope(_CONTINUATION_SECRET, raw)
  broker = AutonomousClaimBroker(GatewayClaimSigningAuthority(_CONTINUATION_SECRET), raw)
  pair = create_autonomous_approval_channel(authority=AutonomousApprovalChannelAuthority(
    launch_nonce=envelope.nonce, task_id=source.task_id,
    control_run_id=source.control_run_id, session_id=source.session_id,
    channel_id=source.channel_id,
  ))
  source.claim_broker = broker
  source.approval_channel = pair.parent
  setattr(source, "event_channel", SimpleNamespace(channel_id=source.channel_id))
  source.launch_nonce = envelope.nonce
  context = multiprocessing.get_context("spawn")
  parent_connection, child_connection = context.Pipe()
  process = context.Process(
    target=_park_approval_process,
    args=(child_connection, authority, str(journal_path), str(tmp_path)),
  )
  process.start()
  child_connection.close()

  async def persist_ask() -> dict[str, Any]:
    kind, private = await asyncio.to_thread(_receive_process_message, parent_connection)
    assert kind == "capture"
    capture = admit_delegated_executable_capture(source, private)
    parent_connection.send("captured")
    kind, event = await asyncio.to_thread(_receive_process_message, parent_connection)
    assert kind == "ask"
    assert "private-original-account" not in json.dumps(event)
    request = await create_delegated_autonomous_approval(
      store=registry._approval_store, policy=None, record=source,
      event=event, private_executable_capture=capture,
    )
    assert request.state == "pending_user"
    parent_connection.send("persisted")
    kind, parked = await asyncio.to_thread(_receive_process_message, parent_connection)
    assert kind == "parked"
    return parked

  try:
    parked = asyncio.run(persist_ask())
    process.join(timeout=10)
    assert process.exitcode == 0
    assert not process.is_alive()
    # Closed transports are intentionally retained: a continuation may not use them.
    pair.close()
    source.event_channel = None
    source.state = "completed"
    source.exit_code = process.exitcode
    source.event_channel_projected_events = [{
      "type": "stream_complete", "terminal_disposition": "awaiting_approval",
    }]
    assert registry._apply_terminal_event_state(source)
    yield SimpleNamespace(
      registry=registry, source=source, store=registry._approval_store,
      parked=parked, execution=execution, provider=provider,
      envelope=envelope, journal_path=journal_path, root=tmp_path,
      process_exitcode=process.exitcode,
    )
  finally:
    if process.is_alive():
      process.terminate()
      process.join(timeout=10)
    process.close()
    parent_connection.close()
    pair.close()
    broker.close()


def test_human_ask_parks_with_durable_binding_and_releases_original_process(
  parked_detached_approval: Any,
) -> None:
  parked = parked_detached_approval

  async def scenario() -> None:
    request = await parked.store.get(parked.parked["approval_id"])
    assert request is not None and request.state == "pending_user"
    binding, _ = await SQLiteApprovalStore(parked.store.path)._read_executable_binding_for_executor(
      job_id="bg_7", tool_call_id="call-original", nonce=parked.parked["nonce"],
      tenant_storage_id="42",
    )
    assert binding.prepared_input == {"units": 7, "account": "private-original-account"}
    assert binding.exact_backend == "original-order-backend"
    assert binding.dispatch_context == {
      "call_index": 7,
      "advertised_tool_names": None,
      "skill_run_id": None,
      "step_id": None,
      "workspace_dir": None,
      "batch_id": None,
      "capture_readable_resource_snapshot": False,
      "allow_uncertain_mcp_replay": False,
    }
    assert parked.parked["pending_tools"] == parked.parked["approval_queues"] == {}
    assert parked.process_exitcode == 0
    assert parked.source.approval_channel.fileno() == -1
    assert parked.source.state == "awaiting_approval"
    assert parked.source.completed_at is None
    assert not parked.registry._is_active_process_state(parked.source)
    assert autonomous_decision_unavailable(parked.source) is None
    # The ledger's location is never handed to the contained child.
    assert parked.source.approval_continuation == {
      "approval_id": request.approval_id, "job_id": "bg_7",
      "tool_call_id": "call-original", "nonce": parked.parked["nonce"],
      "tenant_storage_id": "42",
      "session_log_path": str(parked.journal_path), "pack_cursor": None,
    }
    # The control plane represents a parked run as the approval wait.
    assert canonical_control_run_state(parked.source.state) == "approval_pending"
    assert parked.registry._write_task_manifest(parked.source, checked=True)
    persisted = json.loads((parked.root / "bg_7.task.json").read_text(encoding="utf-8"))
    assert persisted["state"] == "awaiting_approval"
    assert persisted["completed_at"] is None
    assert persisted["approval_continuation"] == parked.source.approval_continuation
    recovered = AutonomousRegistry(
      api_dir=parked.registry._api_dir, log_dir=parked.root,
      approval_store=SQLiteApprovalStore(parked.store.path),
    )._tasks["bg_7"]
    assert recovered.state == "awaiting_approval"
    assert recovered.approval_continuation == parked.source.approval_continuation
    assert recovered.approval_channel is None
    assert not parked.registry._is_active_process_state(recovered)

  asyncio.run(scenario())


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "deny"])
def test_detached_decision_resumes_captured_call_without_replaying_preceding_effects(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  approved: bool,
) -> None:
  parked = parked_detached_approval

  async def scenario() -> None:
    request = await parked.store.get(parked.parked["approval_id"])
    assert request is not None
    await parked.store.transition_state(
      request.approval_id, "approved" if approved else "denied",
      expected_state_version=request.state_version, decider_id="42", decider_role="owner",
    )
    # The parent alone reads its ledger and signs the decided call into the launch.
    binding, decision = await parked.store._read_executable_binding_for_executor(
      job_id="bg_7", tool_call_id="call-original", nonce=parked.parked["nonce"],
      tenant_storage_id="42",
    )
    signed = decided_approval_continuation(
      parked=parked.source.approval_continuation,
      binding=binding, decision=decision,
    )
    assert isinstance(signed, dict)
    assert signed["decision"] == ("approved" if approved else "denied")
    assert signed["prepared_input"] == {"units": 7, "account": "private-original-account"}
    assert signed["exact_backend"] == "original-order-backend"
    assert signed["session_log_path"] == str(parked.journal_path)
    if approved:
      # Signing claims nothing: the launch commits the one attempt itself, and a
      # second claim is then refused.
      assert await parked.store._read_completed_executable_result_for_executor(
        binding=binding,
      ) is None
      await claim_continued_executable_attempt(store=parked.store, continuation=signed)
      with pytest.raises(ApprovalExecutableBindingError) as refused:
        await claim_continued_executable_attempt(store=parked.store, continuation=signed)
      assert refused.value.code == "execution_outcome_unknown"
    else:
      assert signed["claim_token"] is None
    resumed_session = parked.envelope.session_authority.to_gateway_session()
    resumed_channel = create_autonomous_approval_channel(authority=_authority())
    returned: list[dict[str, Any]] = []

    async def return_result(payload: dict[str, Any]) -> None:
      returned.append(copy.deepcopy(payload))

    # The contained successor holds no ledger: only its private capture writer.
    resumed_session.approval_route = ParentDelegatedApprovalRoute(
      resumed_channel.child, resumed_session, private_capture_writer=return_result,
    )
    resumed_channel.close()
    capture_job_context(resumed_session, replace(
      root_context(resumed_session, withheld_tools=frozenset(), research_file_id=None),
      job_id="bg_7",
    ))
    executions: list[dict[str, Any]] = []
    replays: list[dict[str, Any]] = []

    async def execute_original(tool_input: dict[str, Any], **_kwargs: Any):
      executions.append(dict(tool_input))
      units = tool_input["units"]
      (parked.root / "order-effect.txt").write_text(str(units), encoding="utf-8")
      return {"executed_units": units}, None

    async def reject_preceding_replay(tool_input: dict[str, Any], **_kwargs: Any):
      replays.append(dict(tool_input))
      raise AssertionError("The completed preceding effect was replayed")

    events = EventLog()
    dispatcher = ToolDispatcher(
      capability_bind=parked.execution.bind, mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={"native_write": execute_original, "record_note": reject_preceding_replay},
      needs_approval=lambda *_args: True, session=resumed_session, role="owner",
      approval_route=resumed_session.approval_route,
      run_context=RunContext(user_id="42", request_id="run-successor"),
    )
    assert dispatcher._approval_store is None
    runner = AgentRunner(
      capability_execution=parked.execution, dispatcher=dispatcher, event_log=events,
      session_id="bg_successor", user_id="42", billing_mode="byok", rate_table_version="unknown",
    )
    runner.bind_resume_approved_call(
      ApprovedCallRef("bg_7", "call-original", parked.parked["nonce"]),
      dispatcher.parent_decided_executor(signed),
    )
    messages = await SessionContextBuilder(
      agent_session_log=AgentSessionLog(path=parked.journal_path),
    ).build()
    try:
      await runner.run(messages=messages)
      # A lost transcript would make the model re-request the completed record_note.
      assert replays == []
      assert len(parked.provider.requests) == 1
      sent = parked.provider.requests[0]["messages"]
      results = [
        block for message in sent if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "tool_result" and block.get("tool_use_id") == "call-original"
      ]
      assert len(results) == 1
      if approved:
        assert executions == [{"units": 7, "account": "private-original-account"}]
        assert (parked.root / "order-effect.txt").read_text(encoding="utf-8") == "7"
        assert "executed_units" in json.dumps(results[0])
        assert not results[0].get("is_error", False)
        assert len(returned) == 1
        completed = returned[0]["completed_executable_result"]
        assert completed["approval_id"] == request.approval_id
        successor = SimpleNamespace(launch_approval_continuation=signed)
        forged = {**completed, "nonce": "another-nonce"}
        with pytest.raises(ApprovalExecutableBindingError):
          await record_continued_executable_result(
            store=parked.store, record=successor, completed=forged,  # type: ignore[arg-type]
          )
        await record_continued_executable_result(
          store=parked.store, record=successor, completed=completed,  # type: ignore[arg-type]
        )
        # The parent's ledger now holds the exact result, so a delivery retry
        # reports it rather than relaunching or executing twice.
        assert await parked.store._read_completed_executable_result_for_executor(
          binding=binding,
        ) == ({"executed_units": 7}, None)
        duplicate = await parked.registry.continue_approval(
          approval_id=request.approval_id, tenant_storage_id="42",
        )
        assert duplicate is not None and duplicate.status == "already_delivered"
        assert len(executions) == 1
      else:
        assert executions == []
        assert returned == []
        assert not (parked.root / "order-effect.txt").exists()
        assert results[0]["is_error"] is True
        assert "user_denied" in json.dumps(results[0])
      assert (parked.root / "preceding-effect.txt").read_text(encoding="utf-8") == "completed once\n"
      assert any(
        entry.event.get("type") == "stream_complete"
        and entry.event.get("terminal_disposition") == "completed"
        for entry in events.entries
      )
      assert parked.source.approval_channel.fileno() == -1
    finally:
      resumed_channel.close()

  asyncio.run(scenario())


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "deny"])
def test_signed_continuation_round_trip_authenticates_decided_call_and_journal(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  approved: bool,
) -> None:
  parked = parked_detached_approval
  envelope = parked.envelope

  async def decided() -> dict[str, Any]:
    await _decide_original(parked, approved=approved)
    binding, decision = await parked.store._read_executable_binding_for_executor(
      job_id="bg_7", tool_call_id="call-original", nonce=parked.parked["nonce"],
      tenant_storage_id="42",
    )
    signed = decided_approval_continuation(
      parked=parked.source.approval_continuation,
      binding=binding, decision=decision,
    )
    assert isinstance(signed, dict)
    return signed

  continuation = asyncio.run(decided())
  raw = sign_autonomous_launch_envelope(
    _CONTINUATION_SECRET, task_id="bg_successor", control_run_id="run-successor",
    owner_user_id="42", channel_id=envelope.channel_id, bind=envelope.bind,
    workload=replace(envelope.workload, approval_continuation=continuation),
    session_authority=replace(envelope.session_authority, ordinary_authority=replace(
      envelope.session_authority.ordinary_authority, session_id="bg_successor",
    )),
    control_authority=envelope.control_authority,
  )
  verified = verify_autonomous_launch_envelope(_CONTINUATION_SECRET, raw)
  assert verified.workload.approval_continuation == continuation
  assert "approval_store_path" not in continuation
  for field, forged in (
    ("job_id", "another-job"), ("tool_call_id", "another-call"),
    ("decision", "denied" if approved else "approved"),
    ("prepared_input", {"units": 9999, "account": "forged"}),
    ("session_log_path", str(parked.root / "another-session.jsonl")),
  ):
    payload = json.loads(raw)
    payload["workload"]["approval_continuation"][field] = forged
    with pytest.raises(ValueError, match="signature is invalid"):
      verify_autonomous_launch_envelope(
        _CONTINUATION_SECRET,
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
      )


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "deny"])
def test_outbox_continuation_waits_for_original_reaper_and_launches_once(
  parked_detached_approval: Any,
  monkeypatch: pytest.MonkeyPatch,
  approved: bool,
) -> None:
  parked = parked_detached_approval

  async def scenario() -> None:
    audited: list[str] = []

    class _AuditEmitter:
      async def emit_audit_for_lifecycle_event(self, **kwargs: Any) -> None:
        audited.append(kwargs["event_type"])

    store = SQLiteApprovalStore(
      parked.store.path,
      audit_emitter=_AuditEmitter(),
    )
    parked.registry._approval_store = store
    request = await store.record_vote_with_autonomous_delivery(
      parked.parked["approval_id"],
      ApprovalVote(
        vote_id="owner-decision", approval_id=parked.parked["approval_id"],
        decider_id="42", decider_role="owner",
        decision="approved" if approved else "denied", decision_reason="Reviewed original call",
        decided_at=datetime.now(UTC),
      ),
      delivery=autonomous_approval_delivery_context(
        parked.source, tool_call_id="call-original", nonce=parked.parked["nonce"],
      ),
    )
    delivery = await store.get_autonomous_approval_delivery(
      request.approval_id, tool_call_id="call-original", nonce=parked.parked["nonce"],
    )
    assert delivery is not None and delivery["state"] == "pending"
    reaper_started, release_reaper = asyncio.Event(), asyncio.Event()
    binding_checked = asyncio.Event()
    read_completed = store._read_completed_executable_result_for_executor

    async def observe_binding_read(**kwargs: Any):
      result = await read_completed(**kwargs)
      binding_checked.set()
      return result

    monkeypatch.setattr(
      store, "_read_completed_executable_result_for_executor", observe_binding_read,
    )

    async def finish_reaping() -> None:
      reaper_started.set()
      await release_reaper.wait()

    parked.source.reaper_task = asyncio.create_task(finish_reaping())
    launches: list[dict[str, Any]] = []

    async def launch_process(**kwargs: Any) -> dict[str, str]:
      assert parked.source.reaper_task.done()
      assert parked.source.approval_channel.fileno() == -1
      launches.append(kwargs)
      successor = replace(
        parked.source, task_id="bg_successor", session_id="bg_successor",
        control_run_id="run-successor", state="running",
        resumed_from=kwargs["resumed_from"], reaper_task=None,
        approval_continuation=None,
        launch_approval_continuation=kwargs["approval_continuation"],
        resumed_approval_id=kwargs["approval_continuation"]["approval_id"],
      )
      parked.registry._tasks[successor.task_id] = successor
      return {"run_id": successor.control_run_id, "task_id": successor.task_id}

    async def reject_closed_socket_delivery(*_args: Any, **_kwargs: Any) -> None:
      raise AssertionError("Parked approval delivery attempted the original closed socket")

    async def deliver() -> None:
      await deliver_autonomous_approval_outbox(
        registry=parked.registry, store=store, record=parked.source,
        request_record=request, delivery=delivery, approval_id=request.approval_id,
        tool_call_id="call-original", nonce=parked.parked["nonce"],
        approved=approved, user_id="42", channel="cli",
      )

    monkeypatch.setattr(parked.registry, "start", launch_process)
    monkeypatch.setattr(parked.registry, "send_approval_decision", reject_closed_socket_delivery)
    first = asyncio.create_task(deliver())
    await reaper_started.wait()
    await asyncio.wait_for(binding_checked.wait(), timeout=10)
    assert not first.done()
    assert launches == []
    release_reaper.set()
    await first
    await deliver()
    published = await store.get_autonomous_approval_delivery(
      request.approval_id, tool_call_id="call-original", nonce=parked.parked["nonce"],
    )
    assert published is not None and published["state"] == "published"
    assert published["attempt_count"] == 1
    assert ("approved" if approved else "denied") in audited
    assert parked.source.state == "continued"
    assert not parked.registry._is_active_process_state(parked.source)
    duplicate = await parked.registry.continue_approval(
      approval_id=request.approval_id, tenant_storage_id="42",
    )
    assert duplicate is not None and duplicate.status == "already_delivered"
    assert len(launches) == 1
    assert launches[0]["resumed_from"] == "run-7"
    assert launches[0]["approval_continuation"]["approval_id"] == request.approval_id
    assert launches[0]["approval_continuation"]["decision"] == (
      "approved" if approved else "denied"
    )
    assert launches[0]["approval_continuation"]["decision_reason"] == "Reviewed original call"
    assert launches[0]["approval_continuation"]["prepared_input"] == {
      "units": 7, "account": "private-original-account",
    }
    assert launches[0]["approval_continuation"]["session_log_path"] == str(parked.journal_path)
    assert launches[0]["owner_user_id"] == "42"
    assert launches[0]["profile"] == parked.source.profile
    assert launches[0]["channel"] == parked.source.channel
    assert parked.source.resumed_as == ["run-successor"]
    assert await parked.registry.continue_approval(
      approval_id="not-owned", tenant_storage_id="42",
    ) is None

  asyncio.run(scenario())


def _decide_original(parked: Any, *, approved: bool):
  async def decide() -> Any:
    request = await parked.store.get(parked.parked["approval_id"])
    assert request is not None
    await parked.store.transition_state(
      request.approval_id, "approved" if approved else "denied",
      expected_state_version=request.state_version, decider_id="42", decider_role="owner",
    )
    return request
  return decide()


@pytest.mark.parametrize("approved", [True, False], ids=["approve", "deny"])
@pytest.mark.parametrize("later", ["parked-again", "cancelled"])
def test_redelivered_decision_never_launches_a_second_successor(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
  approved: bool,
  later: str,
) -> None:
  parked = parked_detached_approval

  async def scenario() -> None:
    request = await _decide_original(parked, approved=approved)
    launches: list[dict[str, Any]] = []

    async def launch_process(**kwargs: Any) -> dict[str, str]:
      launches.append(kwargs)
      continuation = kwargs["approval_continuation"]
      successor = replace(
        parked.source, task_id="bg_successor", session_id="bg_successor",
        control_run_id="run-successor", state="running",
        resumed_from=kwargs["resumed_from"], reaper_task=None,
        approval_continuation=None, launch_approval_continuation=continuation,
        resumed_approval_id=continuation["approval_id"],
      )
      parked.registry._tasks[successor.task_id] = successor
      return {"run_id": successor.control_run_id, "task_id": successor.task_id}

    monkeypatch.setattr(parked.registry, "start", launch_process)
    first = await parked.registry.continue_approval(
      approval_id=request.approval_id, tenant_storage_id="42",
    )
    assert first is not None and first.status == "delivered"
    assert len(launches) == 1
    signed = launches[0]["approval_continuation"]
    assert signed["decision"] == ("approved" if approved else "denied")
    assert signed["decision_reason"] is None
    assert signed["prepared_input"] == {"units": 7, "account": "private-original-account"}
    assert (signed["claim_token"] is not None) is approved
    successor = parked.registry._tasks["bg_successor"]
    if later == "parked-again":
      # The successor reached a second ask, overwriting its own continuation.
      successor.state = "awaiting_approval"
      successor.approval_continuation = {
        **parked.source.approval_continuation, "approval_id": "second-approval",
      }
    else:
      successor.state = "killed"
    assert canonical_control_run_state("continued") == "completed"
    # The delivery row was not marked published before the process died: a
    # re-delivery finds the one successor, whatever it has since become.
    again = await parked.registry.continue_approval(
      approval_id=request.approval_id, tenant_storage_id="42",
    )
    assert again is not None and again.status == "already_delivered"
    assert len(launches) == 1
    assert parked.source.resumed_as == ["run-successor"]

  asyncio.run(scenario())


async def _resume_signed_call(
  parked: Any,
  signed: dict[str, Any],
  handler: Any,
) -> SimpleNamespace:
  """Run the contained successor's resumed call; return what it sent and saw."""

  resumed_session = parked.envelope.session_authority.to_gateway_session()
  resumed_channel = create_autonomous_approval_channel(authority=_authority())
  returned: list[dict[str, Any]] = []

  async def return_result(payload: dict[str, Any]) -> None:
    returned.append(copy.deepcopy(payload))

  resumed_session.approval_route = ParentDelegatedApprovalRoute(
    resumed_channel.child, resumed_session, private_capture_writer=return_result,
  )
  resumed_channel.close()
  capture_job_context(resumed_session, replace(
    root_context(resumed_session, withheld_tools=frozenset(), research_file_id=None),
    job_id="bg_7",
  ))

  async def reject_replay(_tool_input: dict[str, Any], **_kwargs: Any):
    raise AssertionError("The completed preceding effect was replayed")

  dispatcher = ToolDispatcher(
    capability_bind=parked.execution.bind, mcp_client=McpClientManager(config_path=None),
    local_tool_handlers={"native_write": handler, "record_note": reject_replay},
    needs_approval=lambda *_args: True, session=resumed_session, role="owner",
    approval_route=resumed_session.approval_route,
    run_context=RunContext(user_id="42", request_id="run-successor"),
  )
  runner = AgentRunner(
    capability_execution=parked.execution, dispatcher=dispatcher, event_log=EventLog(),
    session_id="bg_successor", user_id="42", billing_mode="byok", rate_table_version="unknown",
  )
  runner.bind_resume_approved_call(
    ApprovedCallRef("bg_7", "call-original", parked.parked["nonce"]),
    dispatcher.parent_decided_executor(signed),
  )
  messages = await SessionContextBuilder(
    agent_session_log=AgentSessionLog(path=parked.journal_path),
  ).build()
  try:
    await runner.run(messages=messages)
  finally:
    resumed_channel.close()
  results = [
    block for message in parked.provider.requests[-1]["messages"]
    if isinstance(message.get("content"), list)
    for block in message["content"]
    if block.get("type") == "tool_result" and block.get("tool_use_id") == "call-original"
  ]
  assert len(results) == 1
  return SimpleNamespace(returned=returned, result=results[0])


async def _approved_claimed_continuation(parked: Any) -> tuple[Any, dict[str, Any]]:
  await _decide_original(parked, approved=True)
  binding, decision = await parked.store._read_executable_binding_for_executor(
    job_id="bg_7", tool_call_id="call-original", nonce=parked.parked["nonce"],
    tenant_storage_id="42",
  )
  signed = decided_approval_continuation(
    parked=parked.source.approval_continuation, binding=binding, decision=decision,
  )
  assert isinstance(signed, dict)
  # The successor's launch commits this claim before the call can run.
  await claim_continued_executable_attempt(store=parked.store, continuation=signed)
  return binding, signed


@pytest.mark.parametrize("refusal", ["handler_raised_after_effect", "never_executed"])
def test_successor_outcome_that_is_not_a_known_result_is_never_recorded_as_completed(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  refusal: str,
) -> None:
  parked = parked_detached_approval

  async def scenario() -> None:
    binding, signed = await _approved_claimed_continuation(parked)
    executions: list[dict[str, Any]] = []

    async def effect_then_raise(tool_input: dict[str, Any], **_kwargs: Any):
      executions.append(dict(tool_input))
      (parked.root / "order-effect.txt").write_text("7", encoding="utf-8")
      raise RuntimeError("connection lost after the order was sent")

    if refusal == "never_executed":
      # The successor holds another credential generation than the one the
      # call was authorized under: refused before the claim frontier, so the
      # handler never runs.
      signed = {**signed, "credential_generation_ref": "anthropic-generation-other"}
    seen = await _resume_signed_call(parked, signed, effect_then_raise)
    # Only a call that returned an outcome sends the parent a frame to record.
    assert seen.returned == []
    assert seen.result["is_error"] is True
    text = json.dumps(seen.result)
    if refusal == "handler_raised_after_effect":
      assert executions == [{"units": 7, "account": "private-original-account"}]
      assert "execution_outcome" in text and "unknown" in text
      assert "do not retry or reauthorize" in text
      # No invitation to repeat the same operands.
      assert "arguments" not in text and "private-original-account" not in text
    else:
      assert executions == []
      assert "execution_outcome" not in text
    # The parent's claim stands with no result: the ledger reads outcome-unknown.
    with pytest.raises(ApprovalExecutableBindingError) as unknown:
      await parked.store._read_completed_executable_result_for_executor(binding=binding)
    assert unknown.value.code == "execution_outcome_unknown"

  asyncio.run(scenario())


def test_failed_successor_launch_leaves_the_approved_call_redeliverable(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parked = parked_detached_approval

  async def scenario() -> None:
    request = await _decide_original(parked, approved=True)

    async def failing_launch(**_kwargs: Any) -> dict[str, str]:
      raise RuntimeError("successor launch failed before it could run the call")

    monkeypatch.setattr(parked.registry, "start", failing_launch)
    with pytest.raises(RuntimeError, match="successor launch failed"):
      await parked.registry.continue_approval(
        approval_id=request.approval_id, tenant_storage_id="42",
      )
    binding, _ = await parked.store._read_executable_binding_for_executor(
      job_id="bg_7", tool_call_id="call-original", nonce=parked.parked["nonce"],
      tenant_storage_id="42",
    )
    # Nothing ran, so nothing is claimed: the decision stays deliverable.
    assert await parked.store._read_completed_executable_result_for_executor(
      binding=binding,
    ) is None
    assert parked.source.state == "awaiting_approval"
    launches: list[dict[str, Any]] = []

    async def launch_process(**kwargs: Any) -> dict[str, str]:
      launches.append(kwargs)
      continuation = kwargs["approval_continuation"]
      await claim_continued_executable_attempt(store=parked.store, continuation=continuation)
      successor = replace(
        parked.source, task_id="bg_successor", session_id="bg_successor",
        control_run_id="run-successor", state="running",
        resumed_from=kwargs["resumed_from"], reaper_task=None,
        approval_continuation=None, launch_approval_continuation=continuation,
        resumed_approval_id=continuation["approval_id"],
      )
      parked.registry._tasks[successor.task_id] = successor
      return {"run_id": successor.control_run_id, "task_id": successor.task_id}

    monkeypatch.setattr(parked.registry, "start", launch_process)
    redelivered = await parked.registry.continue_approval(
      approval_id=request.approval_id, tenant_storage_id="42",
    )
    assert redelivered is not None and redelivered.status == "delivered"
    assert len(launches) == 1
    with pytest.raises(ApprovalExecutableBindingError) as unknown:
      await parked.store._read_completed_executable_result_for_executor(binding=binding)
    assert unknown.value.code == "execution_outcome_unknown"

  asyncio.run(scenario())


def test_launch_whose_claim_finds_the_call_attempted_reports_outcome_unknown(
  parked_detached_approval: Any,
  owner_session_host_policy: Any,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  parked = parked_detached_approval

  async def scenario() -> None:
    request = await _decide_original(parked, approved=True)

    async def refused_launch(**_kwargs: Any) -> dict[str, str]:
      raise ApprovalExecutableBindingError("execution_outcome_unknown")

    monkeypatch.setattr(parked.registry, "start", refused_launch)
    outcome = await parked.registry.continue_approval(
      approval_id=request.approval_id, tenant_storage_id="42",
    )
    assert outcome is not None and outcome.status == "outcome_unknown"
    assert parked.source.state == "awaiting_approval"

  asyncio.run(scenario())


def test_parked_and_continued_runs_have_control_run_states() -> None:
  assert canonical_control_run_state("awaiting_approval") == "approval_pending"
  assert canonical_control_run_state("continued") == "completed"
