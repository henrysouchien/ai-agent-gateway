"""Consumer-visible approval remote transport, not a wiring copy."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import MappingProxyType
from pathlib import Path
from typing import Any, Callable, Literal, get_args

import pytest
from hank_contracts.agent_api import (
  AGENT_API_CLAIM_ENV_VARS,
  AGENT_API_CLAIM_HEADERS,
)
from pydantic import TypeAdapter, ValidationError

from agent_gateway import approval_store as approval_store_module
from agent_gateway.approval_policy import ApprovalRequest, DelegationGrant
from agent_gateway.approval_remote import (
  APPROVAL_REMOTE_PROTOCOL_VERSION,
  ApprovalRemoteIdentityMismatch,
  ApprovalRemoteProtocolError,
  ApprovalRemoteReservationMismatch,
  ApprovalRemoteStaleReservation,
  AsyncApprovalsEgressTransport,
  AutonomousPublicationReservation,
  BlockingApprovalsEgressTransport,
  BoundApprovalRemoteClient,
  PreparedBusinessModelChangeWire,
  PublishAbortResult,
  PublishCommitResult,
  PublishReserveResult,
  StandardBase64,
  decode_remote_error,
  encode_operation_result,
  encode_remote_error,
  parse_remote_request,
)
from agent_gateway.approval_store import (
  ApprovalExecutableBindingError,
  ExecutableApprovalBinding,
  PolicyGeneration,
  RootTurnExecutableRef,
  RootTurnRecord,
  RootTurnRecordError,
  SQLiteApprovalStore,
  _executable_result_json,
)
from agent_gateway.prepared_business_model_store import (
  PreparedBusinessModelChange,
  PreparedBusinessModelLifecycle,
)


def _request() -> ApprovalRequest:
  return ApprovalRequest(
    approval_id="approval-deadline",
    tool_call_id="tool-deadline",
    parent_approval_id=None,
    approval_chain_id="approval-deadline",
    request_id="run-deadline",
    session_id="session-deadline",
    run_id="run-deadline",
    user_id="owner",
    profile="analyst",
    channel="tui",
    tool_name="memory_write",
    tool_class="state_write",
    tool_args_redacted={"a": 1},
    args_hash="deadline-hash",
    reason="deadline test",
    blast_radius_summary="state_write:memory_write",
    state="pending_user",
    requested_at=datetime(2034, 12, 31, tzinfo=UTC),
    approval_constraint="standard",
  )


def _binding() -> ExecutableApprovalBinding:
  return ExecutableApprovalBinding(
    approval_id="a",
    job_id="j",
    tool_call_id="c",
    nonce="n",
    tenant_storage_id="u",
    tool_name="memory_write",
    prepared_input={"x": 1, "nested": {"k": True}},
    dispatch_context={"y": [2], "registry_scope": {"scope_type": "none"}},
    exact_backend="kernel",
    credential_generation_ref=None,
    operand_digest="sha256:" + "ab" * 32,
  )


def _canonical(value: object) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
  ).encode()


def _prepared_record() -> PreparedBusinessModelChange:
  body = _canonical({"change_set_id": "2" * 64})
  return PreparedBusinessModelChange(
    schema_version="prepared-business-model-change.v1",
    caller_kind="research_http",
    user_scope="tenant-1:user-7",
    idempotency_locator="request-1",
    intent_digest="1" * 64,
    prepared_payload=body,
    change_set_id="2" * 64,
    change_hash="3" * 64,
    base_vector_hash="4" * 64,
    prepared_payload_digest=hashlib.sha256(body).hexdigest(),
    lifecycle=PreparedBusinessModelLifecycle.PENDING,
    created_at="2034-01-01T00:00:00+00:00",
    approval_id="approval-1",
    approval_chain_id="chain-1",
  )


class _MemoryTransport:
  def __init__(self) -> None:
    self.calls: list[tuple[str, bytes]] = []
    self.handler: Callable[[str, bytes], tuple[int, bytes]] = self._echo

  def post(self, operation: str, body: bytes) -> tuple[int, bytes]:
    self.calls.append((operation, body))
    return self.handler(operation, body)

  def _echo(self, operation: str, body: bytes) -> tuple[int, bytes]:
    _op, arguments = parse_remote_request(body, url_operation=operation)
    if operation == "request.create":
      return 200, encode_operation_result(operation, arguments["request"])
    if operation == "request.create_raw_patch":
      return 200, encode_operation_result(operation, arguments["request"])
    if operation == "grant.create_delegation":
      return 200, encode_operation_result(operation, arguments["grant"])
    if operation == "request.get":
      return 200, encode_operation_result(operation, None)
    if operation == "executable.complete":
      return 200, encode_operation_result(operation, None)
    if operation == "executable.read_completed":
      return 200, encode_operation_result(operation, ({"ok": True}, None))
    if operation == "prepared_bm.insert_or_verify":
      return 200, encode_operation_result(operation, (arguments["record"], True))
    if operation == "prepared_bm.transition":
      return 200, encode_operation_result(operation, _prepared_record())
    if operation == "policy.record":
      return 200, encode_operation_result(operation, PolicyGeneration(
        generation_ref="gen-1",
        policy_id=arguments["policy_id"],
        policy_version=arguments["policy_version"],
        policy_bundle_hash=arguments["policy_bundle_hash"],
      ))
    if operation == "policy.read_root_turn":
      return 200, encode_operation_result(operation, RootTurnRecord(
        job_id="j",
        turn_id="t",
        tenant_storage_id="u",
        capability_bind_json="{}",
        credential_generation_ref=None,
        effective_grant_json="{}",
        grant_digest="sha256:" + "a" * 64,
        research_binding_json="{}",
        selection_snapshot_json="{}",
        selection_provenance_json="null",
        approval_route_kind="parked",
        policy_generation_ref="gen-1",
        created_at=datetime(2034, 1, 1, tzinfo=UTC),
        executable_refs=(RootTurnExecutableRef(tool_call_id="c", nonce="n"),),
      ))
    raise AssertionError(operation)


def _client(
  transport: _MemoryTransport | None = None,
) -> tuple[BoundApprovalRemoteClient, _MemoryTransport]:
  sync = transport or _MemoryTransport()
  return BoundApprovalRemoteClient(sync_transport=sync), sync


def test_module_import_does_not_load_approval_store() -> None:
  package = Path(__file__).resolve().parents[1]
  script = (
    "import sys\n"
    "import agent_gateway.approval_remote\n"
    "assert 'agent_gateway.approval_store' not in sys.modules\n"
  )
  completed = subprocess.run(
    [sys.executable, "-c", script],
    check=False,
    capture_output=True,
    text=True,
    env={**os.environ, "PYTHONPATH": str(package)},
  )
  assert completed.returncode == 0, completed.stderr




def test_create_round_trip_sends_no_tenant_selector(monkeypatch: pytest.MonkeyPatch) -> None:
  client, transport = _client()

  def refuse(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("sync transport must not be submitted to the caller loop")

  async def scenario() -> ApprovalRequest:
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", refuse)
    return await client.create(_request())

  restored = asyncio.run(scenario())
  assert restored == _request()
  operation, body = transport.calls[0]
  envelope = json.loads(body)
  assert operation == "request.create" == envelope["operation"]
  assert set(envelope) == {"version", "operation", "payload"}
  assert envelope["version"] == APPROVAL_REMOTE_PROTOCOL_VERSION
  assert "storage_id" not in envelope
  assert "placement_generation" not in envelope
  assert set(envelope["payload"]) == {"request"}
  assert envelope["payload"]["request"]["user_id"] == "owner"


def test_pending_deliveries_payload_has_no_client_limit() -> None:
  client, transport = _client()

  def accept(operation: str, body: bytes) -> tuple[int, bytes]:
    assert operation == "outbox.list_pending"
    return 200, encode_operation_result(operation, [])

  transport.handler = accept
  assert asyncio.run(client.list_pending_autonomous_approval_deliveries()) == []
  payload = json.loads(transport.calls[0][1])["payload"]
  assert payload == {"after_sequence": 0, "through_sequence": None}



def test_non_utf8_bytes_use_padded_standard_base64() -> None:
  raw = b"\xff\xfe\x00not-text"
  encoded = TypeAdapter(StandardBase64).dump_python(raw, mode="json")
  assert isinstance(encoded, str)
  assert base64.b64decode(encoded, validate=True) == raw
  with pytest.raises(ValidationError):
    TypeAdapter(StandardBase64).validate_python(encoded.rstrip("="))

  client, transport = _client()
  asyncio.run(client.create_raw_patch_authorization(
    _request(), prepared_payload=raw,
  ))
  payload = json.loads(transport.calls[0][1])["payload"]
  assert base64.b64decode(payload["prepared_payload"], validate=True) == raw

  record = _prepared_record()
  wire = PreparedBusinessModelChangeWire(
    schema_version=record.schema_version,
    caller_kind=record.caller_kind,
    user_scope=record.user_scope,
    idempotency_locator=record.idempotency_locator,
    intent_digest=record.intent_digest,
    prepared_payload=record.prepared_payload,
    change_set_id=record.change_set_id,
    change_hash=record.change_hash,
    base_vector_hash=record.base_vector_hash,
    prepared_payload_digest=record.prepared_payload_digest,
    lifecycle="PENDING",
    created_at=record.created_at,
    approval_id=record.approval_id,
    approval_chain_id=record.approval_chain_id,
  )
  dumped = wire.model_dump(mode="json")
  assert dumped["prepared_payload"] != record.prepared_payload.decode()
  assert base64.b64decode(dumped["prepared_payload"], validate=True) == (
    record.prepared_payload
  )

  client, transport = _client()
  restored, created = asyncio.run(
    client.insert_or_verify_prepared_business_model_change(record)
  )
  assert created is True
  assert restored.prepared_payload == record.prepared_payload
  sent = json.loads(transport.calls[0][1])["payload"]["record"]["prepared_payload"]
  assert base64.b64decode(sent, validate=True) == record.prepared_payload

  client, transport = _client()
  asyncio.run(client.transition_prepared_business_model_change(
    caller_kind="research_http",
    user_scope="tenant-1:user-7",
    idempotency_locator="request-1",
    expected=PreparedBusinessModelLifecycle.PENDING,
    target=PreparedBusinessModelLifecycle.AUTHORIZED,
    execution_receipt=raw,
  ))
  receipt = json.loads(transport.calls[0][1])["payload"]["execution_receipt"]
  assert base64.b64decode(receipt, validate=True) == raw


def test_set_serializes_as_json_array_and_restores_as_set() -> None:
  client, transport = _client()
  asyncio.run(client.get("approval-deadline", owner_user_id={"b", "a"}))
  owners = json.loads(transport.calls[0][1])["payload"]["owner_user_id"]
  assert isinstance(owners, list)
  assert set(owners) == {"a", "b"}
  _operation, arguments = parse_remote_request(
    transport.calls[0][1], url_operation=transport.calls[0][0],
  )
  assert type(arguments["owner_user_id"]) is set
  assert arguments["owner_user_id"] == {"a", "b"}
  asyncio.run(client.get("approval-deadline", owner_user_id=None))
  _operation, omitted = parse_remote_request(
    transport.calls[1][1], url_operation=transport.calls[1][0],
  )
  assert omitted["owner_user_id"] is None

  grant = DelegationGrant(
    delegation_id="d1",
    delegator_user_id="u",
    delegator_run_id=None,
    delegator_session_id=None,
    delegator_profile="analyst",
    delegator_channel="tui",
    bound_excel_session_id="s",
    bound_relay_request_id="r",
    bound_workbook=None,
    tool_class_ceiling=frozenset({"read", "state_write"}),
    args_predicate=None,
    window_seconds=10,
    created_at=datetime(2034, 1, 1, tzinfo=UTC),
  )
  client, _transport = _client()
  restored = asyncio.run(client.create_delegation_grant(grant))
  assert type(restored.tool_class_ceiling) is frozenset
  assert restored.tool_class_ceiling == grant.tool_class_ceiling


def test_tool_result_and_binding_use_authoritative_serializers() -> None:
  binding = _binding()
  result = ({"ok": True, "n": 1}, {"message": "kept"})
  client, transport = _client()
  asyncio.run(client._complete_executable_binding_for_executor(
    binding=binding,
    decision_state_version=3,
    claim_token="tok",
    tool_result=result,
  ))
  payload = json.loads(transport.calls[0][1])["payload"]
  assert payload["tool_result"] == _executable_result_json(result)
  assert '"kind":"json"' in payload["tool_result"]
  _operation, arguments = parse_remote_request(
    transport.calls[0][1], url_operation=transport.calls[0][0],
  )
  assert arguments["binding"] == binding
  assert payload["binding"]["dispatch_context"] == {"y": [2], "registry_scope": {"scope_type": "none"}}
  assert type(payload["binding"]["prepared_input"]) is dict

  client, _transport = _client()
  restored = asyncio.run(
    client._read_completed_executable_result_for_executor(binding=binding)
  )
  assert restored == ({"ok": True}, None)


def test_policy_and_root_turn_round_trip() -> None:
  client, _transport = _client()
  generation = asyncio.run(client.record_policy_generation(
    policy_id="single-user",
    policy_version="1",
    policy_bundle_hash="unknown",
  ))
  assert generation == PolicyGeneration(
    generation_ref="gen-1",
    policy_id="single-user",
    policy_version="1",
    policy_bundle_hash="unknown",
  )
  record = asyncio.run(client._read_root_turn_for_executor(
    job_id="j", turn_id="t", tenant_storage_id="u",
  ))
  assert record is not None
  assert record.created_at == datetime(2034, 1, 1, tzinfo=UTC)
  assert record.executable_refs == (
    RootTurnExecutableRef(tool_call_id="c", nonce="n"),
  )


def test_wrong_version_operation_tenant_and_malformed_replies() -> None:
  client, transport = _client()

  def identity(_operation: str, _body: bytes) -> tuple[int, bytes]:
    _status, body = encode_remote_error(ApprovalRemoteIdentityMismatch("closure"))
    return 500, body

  transport.handler = identity
  with pytest.raises(ApprovalRemoteIdentityMismatch) as identity_error:
    asyncio.run(client.create(_request()))
  assert identity_error.value.code == "runtime_identity_mismatch"
  assert identity_error.value.status == 403

  def version_mismatch(_operation: str, _body: bytes) -> tuple[int, bytes]:
    return 200, json.dumps({
      "version": "hank.approval-remote.v0",
      "result": None,
    }).encode()

  transport.handler = version_mismatch
  with pytest.raises(ApprovalRemoteProtocolError) as version_error:
    asyncio.run(client.get("approval-deadline"))
  assert version_error.value.code == "version_mismatch"

  def extra(_operation: str, _body: bytes) -> tuple[int, bytes]:
    return 200, json.dumps({
      "version": APPROVAL_REMOTE_PROTOCOL_VERSION,
      "result": None,
      "storage_id": "other",
    }).encode()

  transport.handler = extra
  with pytest.raises(ApprovalRemoteProtocolError) as extra_error:
    asyncio.run(client.get("approval-deadline"))
  assert extra_error.value.code == "malformed_envelope"

  def garbage(_operation: str, _body: bytes) -> tuple[int, bytes]:
    return 200, b"{"

  transport.handler = garbage
  with pytest.raises(ApprovalRemoteProtocolError) as garbage_error:
    asyncio.run(client.get("approval-deadline"))
  assert garbage_error.value.code == "malformed_envelope"

  body = json.dumps({
    "version": APPROVAL_REMOTE_PROTOCOL_VERSION,
    "operation": "request.get",
    "payload": {"approval_id": "a", "owner_user_id": None},
  }).encode()
  with pytest.raises(ApprovalRemoteProtocolError) as mismatch:
    parse_remote_request(body, url_operation="request.create")
  assert mismatch.value.code == "operation_mismatch"

  planted = json.loads(body)
  planted["payload"]["storage_id"] = "other-tenant"
  with pytest.raises(ApprovalRemoteProtocolError) as planted_error:
    parse_remote_request(json.dumps(planted).encode(), url_operation="request.get")
  assert planted_error.value.code == "malformed_envelope"

  unknown = {
    "version": APPROVAL_REMOTE_PROTOCOL_VERSION,
    "exception": "OSError",
    "code": "store_exception",
    "message": "disk",
    "retryable": False,
    "domain_code": None,
  }
  with pytest.raises(ApprovalRemoteProtocolError) as unknown_error:
    raise decode_remote_error(json.dumps(unknown).encode())
  assert unknown_error.value.code == "malformed_envelope"


def test_invalid_signed_claim_has_a_versioned_typed_wire_error() -> None:
  status, body = encode_remote_error(ApprovalRemoteProtocolError(
    "invalid signed claim", code="invalid_signed_claim",
  ))
  assert status == 401
  assert json.loads(body)["version"] == APPROVAL_REMOTE_PROTOCOL_VERSION
  error = decode_remote_error(body)
  assert isinstance(error, ApprovalRemoteProtocolError)
  assert error.code == "invalid_signed_claim"


def test_domain_code_is_preserved_and_wire_message_is_ignored() -> None:
  body = {
    "version": APPROVAL_REMOTE_PROTOCOL_VERSION,
    "exception": "ApprovalExecutableBindingError",
    "code": "nonce_mismatch",
    "domain_code": "nonce_mismatch",
    "message": "secret operand detail",
    "retryable": False,
  }
  client, transport = _client()

  def reply(_operation: str, _request: bytes) -> tuple[int, bytes]:
    return 400, json.dumps(body).encode()

  transport.handler = reply
  with pytest.raises(ApprovalExecutableBindingError) as caught:
    asyncio.run(client.get("approval-deadline"))
  assert caught.value.code == "nonce_mismatch"
  assert "secret" not in str(caught.value)

  mismatched = dict(body)
  mismatched["code"] = "tenant_mismatch"
  with pytest.raises(ApprovalRemoteProtocolError):
    decode_remote_error(json.dumps(mismatched).encode())

  _value_status, value_body = encode_remote_error(ValueError("decision changed"))

  def wrong_status(_operation: str, _request: bytes) -> tuple[int, bytes]:
    return 404, value_body

  transport.handler = wrong_status
  with pytest.raises(ValueError, match="decision changed"):
    asyncio.run(client.get("approval-deadline"))


class _PublicationPeer:
  def __init__(self) -> None:
    self.rows: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    self.generation = 4
    self.fail_next_commit = False
    self.paths: list[str] = []
    self.bodies: list[bytes] = []

  def handle(self, path: str, body: bytes) -> tuple[int, bytes]:
    self.paths.append(path)
    self.bodies.append(body)
    operation = path.rstrip("/").rsplit("/", 1)[-1]
    try:
      parsed, arguments = parse_remote_request(body, url_operation=operation)
    except ApprovalRemoteProtocolError as exc:
      return encode_remote_error(exc)
    if parsed == "publish.reserve":
      return self._reserve(arguments)
    if parsed == "publish.commit":
      return self._commit(arguments["reservation"])
    if parsed == "publish.abort":
      return self._abort(arguments["reservation"])
    if parsed == "publish.reconcile_duplicate":
      return 200, encode_operation_result(parsed, {
        "state": "published", "attempt_count": 1,
      })
    raise AssertionError(parsed)

  def _reserve(self, arguments: dict[str, Any]) -> tuple[int, bytes]:
    decision_id = arguments["decision_id"]
    tool_call_id = arguments["tool_call_id"]
    nonce = arguments["nonce"]
    key = (decision_id, tool_call_id, nonce, self.generation)
    if self._other_generation(decision_id, tool_call_id, nonce, self.generation) is not None:
      return encode_remote_error(ApprovalRemoteStaleReservation("generation"))
    existing = self.rows.get(key)
    if existing is not None and existing["approved"] is not arguments["approved"]:
      return encode_remote_error(ValueError(
        "autonomous approval delivery decision changed"
      ))
    if existing is None:
      existing = {
        "decision_id": decision_id,
        "tool_call_id": tool_call_id,
        "nonce": nonce,
        "approved": arguments["approved"],
        "placement_generation": self.generation,
        "state": "active",
        "attempt_count": 0,
      }
      self.rows[key] = existing
    reservation = AutonomousPublicationReservation(
      decision_id=existing["decision_id"],
      tool_call_id=existing["tool_call_id"],
      nonce=existing["nonce"],
      approved=existing["approved"],
      placement_generation=existing["placement_generation"],
    )
    return 200, encode_operation_result(
      "publish.reserve", PublishReserveResult(reservation=reservation),
    )

  def _other_generation(
    self,
    decision_id: str,
    tool_call_id: str,
    nonce: str,
    generation: int,
  ) -> dict[str, Any] | None:
    for key, row in self.rows.items():
      if key[:3] == (decision_id, tool_call_id, nonce) and key[3] != generation:
        return row
    return None

  def _commit(
    self, reservation: AutonomousPublicationReservation,
  ) -> tuple[int, bytes]:
    key = (
      reservation.decision_id,
      reservation.tool_call_id,
      reservation.nonce,
      reservation.placement_generation,
    )
    row = self.rows.get(key)
    if row is None:
      if self._other_generation(*key) is not None:
        return encode_remote_error(ApprovalRemoteStaleReservation("generation"))
      return encode_remote_error(ApprovalRemoteReservationMismatch("gone"))
    if row["placement_generation"] != self.generation:
      return encode_remote_error(ApprovalRemoteStaleReservation("generation"))
    if row["approved"] is not reservation.approved:
      return encode_remote_error(ApprovalRemoteReservationMismatch("field"))
    if self.fail_next_commit and row["state"] == "active":
      self.fail_next_commit = False
      return encode_remote_error(RuntimeError("publication compare-and-set failed"))
    if row["state"] == "active":
      row["state"] = "committed"
      row["attempt_count"] += 1
    return 200, encode_operation_result(
      "publish.commit", PublishCommitResult(state="published"),
    )

  def _abort(
    self, reservation: AutonomousPublicationReservation,
  ) -> tuple[int, bytes]:
    key = (
      reservation.decision_id,
      reservation.tool_call_id,
      reservation.nonce,
      reservation.placement_generation,
    )
    row = self.rows.get(key)
    if row is None:
      if self._other_generation(*key) is not None:
        return encode_remote_error(ApprovalRemoteStaleReservation("generation"))
      return 200, encode_operation_result(
        "publish.abort", PublishAbortResult(released=False),
      )
    if row["placement_generation"] != self.generation:
      return encode_remote_error(ApprovalRemoteStaleReservation("generation"))
    if row["state"] == "committed":
      return 200, encode_operation_result(
        "publish.abort", PublishAbortResult(released=False),
      )
    if row["approved"] is not reservation.approved:
      return encode_remote_error(ApprovalRemoteReservationMismatch("field"))
    del self.rows[key]
    return 200, encode_operation_result(
      "publish.abort", PublishAbortResult(released=True),
    )


def _serve(peer: _PublicationPeer) -> tuple[ThreadingHTTPServer, str]:
  class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
      length = int(self.headers.get("content-length", "0"))
      body = self.rfile.read(length)
      status, response = peer.handle(self.path, body)
      self.send_response(status)
      self.send_header("content-type", "application/json")
      self.send_header("content-length", str(len(response)))
      self.end_headers()
      self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
      return

  server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  threading.Thread(target=server.serve_forever, daemon=True).start()
  host, port = server.server_address[:2]
  return server, f"http://{host}:{port}/egress/approvals"


def test_idempotent_publication_over_blocking_transport() -> None:
  peer = _PublicationPeer()
  server, base = _serve(peer)
  try:
    client = BoundApprovalRemoteClient(
      sync_transport=BlockingApprovalsEgressTransport(base),
    )
    first = client.reserve_autonomous_approval_publication(
      "approval-1", tool_call_id="tool-1", nonce="nonce-1", approved=True,
    )
    second = client.reserve_autonomous_approval_publication(
      "approval-1", tool_call_id="tool-1", nonce="nonce-1", approved=True,
    )
    assert first == second
    assert first.reservation.placement_generation == 4
    reserve_body = json.loads(peer.bodies[0])
    assert "placement_generation" not in reserve_body
    assert "storage_id" not in reserve_body
    assert reserve_body["payload"]["decision_id"] == "approval-1"
    assert "approval_id" not in reserve_body["payload"]
    assert peer.paths[0].endswith("/publish.reserve")
    assert reserve_body["operation"] == "publish.reserve"
    with pytest.raises(ValueError, match="autonomous approval delivery decision changed"):
      client.reserve_autonomous_approval_publication(
        "approval-1", tool_call_id="tool-1", nonce="nonce-1", approved=False,
      )
    committed = client.commit_autonomous_approval_publication(first.reservation)
    again = client.commit_autonomous_approval_publication(first.reservation)
    assert committed.state == again.state == "published"
    key = ("approval-1", "tool-1", "nonce-1", 4)
    assert peer.rows[key]["attempt_count"] == 1
    aborted = client.abort_autonomous_approval_publication(first.reservation)
    assert aborted.released is False
    assert peer.rows[key]["state"] == "committed"

    with client.autonomous_approval_delivery_append_transaction(
      "approval-2", tool_call_id="tool-2", nonce="nonce-2", approved=False,
    ):
      assert ("approval-2", "tool-2", "nonce-2", 4) in peer.rows
    assert peer.rows[("approval-2", "tool-2", "nonce-2", 4)]["attempt_count"] == 1

    class Stop(BaseException):
      pass

    try:
      with client.autonomous_approval_delivery_append_transaction(
        "approval-3", tool_call_id="tool-3", nonce="nonce-3", approved=True,
      ):
        raise Stop()
    except Stop:
      pass
    assert ("approval-3", "tool-3", "nonce-3", 4) not in peer.rows

    peer.fail_next_commit = True
    with pytest.raises(RuntimeError, match="compare-and-set"):
      with client.autonomous_approval_delivery_append_transaction(
        "approval-4", tool_call_id="tool-4", nonce="nonce-4", approved=True,
      ):
        pass
    assert ("approval-4", "tool-4", "nonce-4", 4) not in peer.rows

    before = len(peer.paths)
    with client.autonomous_approval_delivery_duplicate_transaction(
      "approval-5", tool_call_id="tool-5", nonce="nonce-5", approved=True,
    ):
      assert peer.paths[-1].endswith("/publish.reconcile_duplicate")
    duplicate_paths = peer.paths[before:]
    assert [path.rsplit("/", 1)[-1] for path in duplicate_paths] == [
      "publish.reconcile_duplicate",
    ]
  finally:
    server.shutdown()
    server.server_close()


def test_blocking_transport_refreshes_claim_headers() -> None:
  seen: list[str | None] = []

  counter = {"n": 0}

  def mint() -> dict[str, str]:
    counter["n"] += 1
    return {
      AGENT_API_CLAIM_ENV_VARS[field]: f"{field}-{counter['n']}"
      for field in AGENT_API_CLAIM_HEADERS
    }

  class ClaimHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
      seen.append(self.headers.get(AGENT_API_CLAIM_HEADERS["nonce"]))
      length = int(self.headers.get("content-length", "0"))
      body = self.rfile.read(length)
      envelope = json.loads(body)
      assert self.path.endswith("/" + envelope["operation"])
      if envelope["operation"] == "policy.current":
        response = json.dumps({
          "version": APPROVAL_REMOTE_PROTOCOL_VERSION,
          "result": None,
        }).encode()
      else:
        response = json.dumps({
          "version": APPROVAL_REMOTE_PROTOCOL_VERSION,
          "result": None,
        }).encode()
      self.send_response(200)
      self.send_header("content-length", str(len(response)))
      self.end_headers()
      self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
      return

  claim_server = ThreadingHTTPServer(("127.0.0.1", 0), ClaimHandler)
  threading.Thread(target=claim_server.serve_forever, daemon=True).start()
  claim_host, claim_port = claim_server.server_address[:2]
  try:
    base = f"http://{claim_host}:{claim_port}/egress/approvals"
    sync = BlockingApprovalsEgressTransport(
      base, mint_claim=mint,
    )
    client = BoundApprovalRemoteClient(sync_transport=sync)
    assert asyncio.run(client.get("approval-1")) is None
    assert asyncio.run(client.get("approval-1")) is None
    assert seen == ["nonce-1", "nonce-2"]
    async_client = BoundApprovalRemoteClient(
      sync_transport=BlockingApprovalsEgressTransport(base),
      async_transport=AsyncApprovalsEgressTransport(base),
    )
    assert asyncio.run(async_client.current_policy_generation()) is None
    assert seen[-1] is None
  finally:
    claim_server.shutdown()
    claim_server.server_close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_egress_transport_ignores_ambient_proxy(
  monkeypatch: pytest.MonkeyPatch, asynchronous: bool,
) -> None:
  direct_requests: list[dict[str, str]] = []
  proxy_requests: list[dict[str, str]] = []

  class DirectHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
      direct_requests.append(dict(self.headers.items()))
      self.rfile.read(int(self.headers["content-length"]))
      self.send_response(200)
      self.send_header("content-length", "2")
      self.end_headers()
      self.wfile.write(b"{}")

    def log_message(self, format: str, *args: Any) -> None:
      return

  class ProxyHandler(DirectHandler):
    def do_POST(self) -> None:  # noqa: N802
      proxy_requests.append(dict(self.headers.items()))
      self.send_response(502)
      self.send_header("content-length", "0")
      self.end_headers()

  direct = ThreadingHTTPServer(("127.0.0.1", 0), DirectHandler)
  proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
  for server in (direct, proxy):
    threading.Thread(target=server.serve_forever, daemon=True).start()
  try:
    proxy_url = f"http://127.0.0.1:{proxy.server_port}"
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
      monkeypatch.setenv(name, proxy_url)
      monkeypatch.setenv(name.lower(), proxy_url)
    for name in ("NO_PROXY", "no_proxy"):
      monkeypatch.setenv(name, "")
    base = f"http://127.0.0.1:{direct.server_port}/egress/approvals"
    claim = {
      AGENT_API_CLAIM_ENV_VARS[field]: f"secret-{field}"
      for field in AGENT_API_CLAIM_HEADERS
    }
    if asynchronous:
      result = asyncio.run(
        AsyncApprovalsEgressTransport(base).post("policy.current", b"{}"),
      )
    else:
      result = BlockingApprovalsEgressTransport(
        base, mint_claim=lambda: claim,
      ).post("policy.current", b"{}")
    assert result == (200, b"{}")
    assert len(direct_requests) == 1
    assert proxy_requests == []
    received = {key.lower(): value for key, value in direct_requests[0].items()}
    for field, header in AGENT_API_CLAIM_HEADERS.items():
      assert received.get(header.lower()) == (
        None if asynchronous else claim[AGENT_API_CLAIM_ENV_VARS[field]]
      )
  finally:
    for server in (direct, proxy):
      server.shutdown()
      server.server_close()


def test_mappingproxy_arguments_round_trip_as_plain_dicts() -> None:
  nested = MappingProxyType({"tool": MappingProxyType({"name": "memory_write"})})
  transport = _MemoryTransport()

  def accept(operation: str, body: bytes) -> tuple[int, bytes]:
    return 200, encode_operation_result(operation, None)

  transport.handler = accept
  client = BoundApprovalRemoteClient(sync_transport=transport)
  asyncio.run(client.record_root_turn(
    job_id="j",
    turn_id="t",
    tenant_storage_id="u",
    capability_bind=nested,
    credential_generation_ref=None,
    effective_grant=nested,
    research_binding=None,
    selection_snapshot=nested,
    approval_route_kind="parked",
    policy_generation_ref="gen",
  ))
  payload = json.loads(transport.calls[0][1])["payload"]
  assert payload["effective_grant"] == {"tool": {"name": "memory_write"}}
  assert type(payload["effective_grant"]["tool"]) is dict
  _operation, arguments = parse_remote_request(
    transport.calls[0][1], url_operation=transport.calls[0][0],
  )
  assert arguments["effective_grant"] == {"tool": {"name": "memory_write"}}
  assert type(arguments["capability_bind"]) is dict


def test_server_decode_rejects_coercion_and_extra_domain_keys() -> None:
  reserve = {
    "version": APPROVAL_REMOTE_PROTOCOL_VERSION,
    "operation": "publish.reserve",
    "payload": {
      "decision_id": "a",
      "tool_call_id": "t",
      "nonce": "n",
      "approved": 1,
    },
  }
  with pytest.raises(ApprovalRemoteProtocolError):
    parse_remote_request(json.dumps(reserve).encode(), url_operation="publish.reserve")
  client, transport = _client()
  asyncio.run(client.create(_request()))
  raw = json.loads(transport.calls[0][1])
  raw["payload"]["request"]["not_a_field"] = "x"
  with pytest.raises(ApprovalRemoteProtocolError):
    parse_remote_request(json.dumps(raw).encode(), url_operation="request.create")


def test_non_json_tool_result_is_result_mismatch() -> None:
  client, _transport = _client()
  with pytest.raises(ApprovalExecutableBindingError) as caught:
    asyncio.run(client._complete_executable_binding_for_executor(
      binding=_binding(),
      decision_state_version=1,
      claim_token="tok",
      tool_result=({"n": float("nan")}, None),
    ))
  assert caught.value.code == "result_mismatch"
  with pytest.raises(ApprovalExecutableBindingError) as shape:
    asyncio.run(client._complete_executable_binding_for_executor(
      binding=_binding(),
      decision_state_version=1,
      claim_token="tok",
      tool_result=("only-one",),  # type: ignore[arg-type]
    ))
  assert shape.value.code == "result_mismatch"


def test_async_method_runs_blocking_post_off_the_loop() -> None:
  loop_ids: list[int] = []
  post_ids: list[int] = []

  class OffloadTransport:
    def post(self, operation: str, body: bytes) -> tuple[int, bytes]:
      post_ids.append(threading.get_ident())
      return 200, encode_operation_result(operation, None)

  client = BoundApprovalRemoteClient(sync_transport=OffloadTransport())

  async def scenario() -> None:
    loop_ids.append(threading.get_ident())
    assert await client.get("approval-1") is None

  asyncio.run(scenario())
  assert post_ids[0] != loop_ids[0]


@pytest.mark.parametrize(
  ("alias_name", "exception_type"),
  [
    ("ApprovalExecutableBindingErrorCode", ApprovalExecutableBindingError),
    ("RootTurnRecordErrorCode", RootTurnRecordError),
  ],
)
def test_domain_error_codes_follow_store_producers(
  monkeypatch: pytest.MonkeyPatch,
  alias_name: str,
  exception_type: Callable[[str], ApprovalExecutableBindingError | RootTurnRecordError],
) -> None:
  for code in get_args(getattr(approval_store_module, alias_name)):
    status, body = encode_remote_error(exception_type(code))
    assert status == 409
    restored = decode_remote_error(body)
    assert isinstance(restored, (ApprovalExecutableBindingError, RootTurnRecordError))
    assert type(restored) is exception_type
    assert restored.code == code
  # A producer revision is accepted without updating a second literal list.
  monkeypatch.setattr(approval_store_module, alias_name, Literal["producer_revision"])
  _status, body = encode_remote_error(exception_type("producer_revision"))
  restored = decode_remote_error(body)
  assert isinstance(restored, (ApprovalExecutableBindingError, RootTurnRecordError))
  assert restored.code == "producer_revision"
  with pytest.raises(ValidationError):
    encode_remote_error(exception_type("not_a_producer_code"))


def test_prepared_wire_follows_dataclass_fields_and_lifecycle_producer() -> None:
  record = _prepared_record()
  for lifecycle in PreparedBusinessModelLifecycle:
    values = {**asdict(record), "lifecycle": lifecycle}
    wire = PreparedBusinessModelChangeWire.model_validate(values)
    raw = wire.model_dump(mode="json")
    assert raw["lifecycle"] == lifecycle.value
    assert PreparedBusinessModelChangeWire.model_validate_json(
      wire.model_dump_json(),
    ).model_dump()["lifecycle"] is lifecycle
  with pytest.raises(ValidationError):
    PreparedBusinessModelChangeWire.model_validate({
      **asdict(record), "lifecycle": "UNDECLARED",
    })


def test_encode_refuses_unknown_prepared_and_binding_producer_fields() -> None:
  @dataclass(frozen=True)
  class EvolvedPrepared(PreparedBusinessModelChange):
    producer_added: str = "must not disappear"

  @dataclass(frozen=True)
  class EvolvedBinding(ExecutableApprovalBinding):
    producer_added: str = "must not disappear"

  client, transport = _client()
  with pytest.raises(ValidationError, match="producer_added"):
    asyncio.run(client.insert_or_verify_prepared_business_model_change(
      EvolvedPrepared(**asdict(_prepared_record())),
    ))
  binding = _binding()
  with pytest.raises(ValidationError, match="producer_added"):
    asyncio.run(client._complete_executable_binding_for_executor(
      binding=EvolvedBinding(**{
        field.name: getattr(binding, field.name) for field in fields(binding)
      }),
      decision_state_version=1,
      claim_token="tok",
      tool_result=({"ok": True}, None),
    ))
  assert transport.calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_egress_reconcile_can_wait_beyond_httpx_default_timeout(
  asynchronous: bool,
) -> None:
  class ReconcileHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
      assert self.path.endswith("/publish.reconcile_duplicate")
      self.rfile.read(int(self.headers["content-length"]))
      time.sleep(5.1)
      self.send_response(200)
      self.send_header("content-length", "2")
      self.end_headers()
      self.wfile.write(b"{}")

    def log_message(self, format: str, *args: Any) -> None:
      return

  server = ThreadingHTTPServer(("127.0.0.1", 0), ReconcileHandler)
  threading.Thread(target=server.serve_forever, daemon=True).start()
  try:
    base = f"http://127.0.0.1:{server.server_port}/egress/approvals"
    if asynchronous:
      result = asyncio.run(AsyncApprovalsEgressTransport(base).post(
        "publish.reconcile_duplicate", b"{}",
      ))
    else:
      result = BlockingApprovalsEgressTransport(base).post(
        "publish.reconcile_duplicate", b"{}",
      )
    assert result == (200, b"{}")
  finally:
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_connection_refusal_is_unavailable(asynchronous: bool) -> None:
  server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
  base = f"http://127.0.0.1:{server.server_port}/egress/approvals"
  server.server_close()
  with pytest.raises(ApprovalRemoteProtocolError) as caught:
    if asynchronous:
      asyncio.run(AsyncApprovalsEgressTransport(base).post("policy.current", b"{}"))
    else:
      BlockingApprovalsEgressTransport(base).post("policy.current", b"{}")
  assert caught.value.code == "unavailable"
  status, _body = encode_remote_error(caught.value)
  assert status == 503
