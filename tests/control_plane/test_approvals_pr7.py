from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway_test_support.control_plane_identity import fake_identity_resolver, fake_mcp_user_key_lookup
from agent_gateway import AgentRunner
from agent_gateway.autonomous_capability_handoff import AutonomousCapabilityBinding
from agent_gateway.autonomous_event_channel import (
  adopt_inherited_autonomous_event_channel,
)
from agent_gateway.autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
  AUTONOMOUS_RUNTIME_SESSION_PURPOSE,
)
from agent_gateway.approval_audit import ApprovalAuditEmitter
from agent_gateway.approval_notifications import ApprovalNotificationDestination
from agent_gateway.approval_continuation import (
  ChildContinueOutcome,
  ChildContinueRequest,
  RootContinueOutcome,
  RootContinueRequest,
)
from agent_gateway.approval_policy import (
  ApprovalRequest,
  ApprovalRequestPayload,
  DelegationGrant,
  RunContext,
  utc_now,
  sha256_args,
)
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.audit_writer import JSONLAuditWriter
from model_authority.bind import CapabilityBind
from model_authority.binding import CredentialHandle
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.control_plane import approvals as approvals_module
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.tool_policy_registry import PreparedToolCall


API_KEY = "approvals-pr7-key"
HMAC_KEY = "approvals-pr7-test-hmac-key-at-least-32-bytes"
_MODEL_ENTRY = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:approvals-pr7:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="test-product",
  actor_id=None,
)


def _autonomous_capability_binding(request) -> AutonomousCapabilityBinding:
  handle = _SERVICE_HANDLE
  return AutonomousCapabilityBinding(
    bind=request.required_bind
    or CapabilityBind(
      schema_version="1.0",
      capability_id="session.driver",
      model_key=_MODEL_ENTRY.key,
      provider=_MODEL_ENTRY.provider,
      upstream_model=_MODEL_ENTRY.upstream_model,
      adapter=_MODEL_ENTRY.adapter,
      protocol_profile=_MODEL_ENTRY.protocol_profile,
      route=_MODEL_ENTRY.route,
      effort="high",
      credential_principal="service",
      credential_ref=handle.handle_id,
      run_mode=request.run_mode,
      registry_revision=INITIAL_MODEL_REGISTRY.revision,
      policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
      selection_source="capability_default",
    ),
    materialized_credential=MaterializedCredential(
      handle=handle,
      auth_config={
        "provider": handle.provider,
        "auth_mode": "api",
        "api_key": "approvals-pr7-test-secret",
      },
    ),
  )


_FAKE_PROCESS_PIDS = count(90_000)
_FAKE_PROCESSES: dict[int, "_FakeAutonomousProcess"] = {}


class _FakeStdin:
  def __init__(self) -> None:
    self.buffer = bytearray()

  def write(self, payload: bytes) -> None:
    self.buffer.extend(payload)

  async def drain(self) -> None:
    return None

  def close(self) -> None:
    return None

  async def wait_closed(self) -> None:
    return None


class _FakeAutonomousProcess:
  def __init__(self, inherited_event_fd: int) -> None:
    self.pid = next(_FAKE_PROCESS_PIDS)
    self._returncode: int | None = None
    self._inherited_fds = [inherited_event_fd]
    self._event_channel = None
    self.stdin = _FakeStdin()
    _FAKE_PROCESSES[self.pid] = self

  @property
  def returncode(self) -> int | None:
    return self._returncode

  @returncode.setter
  def returncode(self, value: int | None) -> None:
    self._returncode = value
    if value is not None:
      self._close_child_endpoints()

  def start_event_channel(self, channel_id: str) -> None:
    inherited_fd = self._inherited_fds.pop(0)
    self._event_channel = adopt_inherited_autonomous_event_channel(
      inherited_fd,
      channel_id=channel_id,
    )
    self._event_channel.start(timeout_seconds=1)

  def _close_child_endpoints(self) -> None:
    if self._event_channel is not None:
      self._event_channel.close()
      self._event_channel = None
    for inherited_fd in self._inherited_fds:
      if inherited_fd >= 0:
        os.close(inherited_fd)
    self._inherited_fds.clear()

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def retain_inherited_fds(self, inherited_fds: tuple[int, ...]) -> None:
    self._inherited_fds.extend(os.dup(fd) for fd in inherited_fds)

  def terminate(self) -> None:
    if self.returncode is None:
      self.returncode = -15

  def kill(self) -> None:
    if self.returncode is None:
      self.returncode = -9


def _patch_fake_autonomous_spawn(
  monkeypatch,
  autonomous_runner,
  fake_exec,
) -> None:
  async def retaining_fake_exec(*args: Any, **kwargs: Any):
    process = await fake_exec(*args, **kwargs)
    envelope = json.loads(
      kwargs["env"][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV]
    )
    process.start_event_channel(envelope["channel_id"])
    process.retain_inherited_fds(tuple(kwargs["pass_fds"][1:]))
    return process

  monkeypatch.setattr(
    autonomous_runner.asyncio,
    "create_subprocess_exec",
    retaining_fake_exec,
  )
  monkeypatch.setattr(
    autonomous_runner,
    "_get_process_group_id",
    lambda pid: pid,
  )

  def signal_process_group(process_group_id: int, signal_number: int) -> None:
    process = _FAKE_PROCESSES[process_group_id]
    process.returncode = -signal_number

  monkeypatch.setattr(
    autonomous_runner,
    "_signal_process_group",
    signal_process_group,
  )


def _run(coro):
  return asyncio.run(coro)


class _NoopPolicy:
  policy_bundle_hash = "test-policy-bundle"

  def __init__(self, *, owner_only_classes: set[str] | None = None) -> None:
    self.owner_only_classes = set(owner_only_classes or set())
    self.resolved: list[str] = []
    self.authorization_checks: list[tuple[str | None, str]] = []

  async def decide(
    self,
    *,
    payload: ApprovalRequestPayload,
    request: ApprovalRequest,
    run_context: RunContext,
  ):
    raise AssertionError("decide is not used by PR7 endpoint tests")

  async def on_resolve(self, *, request: ApprovalRequest) -> None:
    self.resolved.append(request.approval_id)

  async def revoke_persistent_grant(self, *, grant_id: str, reason: str) -> None:
    _ = grant_id, reason

  def role_authorized_for_class(self, *, decider_role: str | None, tool_class: str) -> bool:
    self.authorization_checks.append((decider_role, tool_class))
    if tool_class in self.owner_only_classes:
      return decider_role == "owner"
    return True


class _NoopRunner(AgentRunner):
  def __init__(self) -> None:
    pass

  def bind_selected_content(self, bindings) -> None:
    _ = bindings

  async def run(
    self,
    messages,
    system_prompt=None,
    *,
    resume_initial_messages=None,
  ) -> None:
    _ = messages, system_prompt, resume_initial_messages




def _make_app(tmp_path, policy: _NoopPolicy | None = None):
  async def _build_chat_runtime(session, request, channel, auth_manager, *, storage_root: Path | None = None):
    _ = session, channel, auth_manager
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda _event_log, _sid, _started_at: _NoopRunner(),
      capability_execution=request.capability_execution,
    )

  app = create_gateway_app(
    GatewayServerConfig(
      identity_resolver=fake_identity_resolver,
      mcp_user_key_lookup=fake_mcp_user_key_lookup,
      jwt_secret="approvals-pr7-test-secret-0123456789",
      valid_api_keys={API_KEY},
      tenant_id="test-product",
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      service_provider_handles={"anthropic": _SERVICE_HANDLE},
      build_chat_runtime=_build_chat_runtime,
      autonomous_capability_binding_resolver=_autonomous_capability_binding,
      claim_signing_authority=GatewayClaimSigningAuthority(HMAC_KEY),
    )
  )
  writer = JSONLAuditWriter(tmp_path / "audit")
  emitter = ApprovalAuditEmitter(
    writer=writer,
    deployment_secret=b"approval-test-secret",
    key_id="test-key",
  )
  store = SQLiteApprovalStore(tmp_path / "approvals.sqlite3", audit_emitter=emitter)
  resolved_policy = policy or _NoopPolicy()
  app.state.gateway_approval_audit_writer = writer
  app.state.gateway_approval_audit_emitter = emitter
  app.state.gateway_approval_store = store
  app.state.gateway_approval_policy = resolved_policy
  return app, store, resolved_policy, writer


def _control_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  return _with_model_entitlements(client, response.json())


def _chat_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/chat/init",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  return _with_model_entitlements(client, response.json())


def _with_model_entitlements(
  client: TestClient,
  session_payload: dict[str, Any],
) -> dict[str, Any]:
  app = client.app
  assert isinstance(app, FastAPI)
  session = app.state.auth.session_store.get_session(
    session_payload["session_id"]
  )
  assert session is not None
  session.model_entitled_capabilities = CAPABILITY_IDS
  session.model_entitled_keys = frozenset(INITIAL_MODEL_REGISTRY.models)
  payload = dict(session_payload)
  payload["session_token"] = app.state.auth.issue_token(session)
  return payload


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def _session(app, session_payload: dict[str, Any]):
  session = app.state.auth.session_store.get_session(session_payload["session_id"])
  assert session is not None
  return session


def _with_role(app, session_payload: dict[str, Any], role: str) -> dict[str, Any]:
  session = _session(app, session_payload)
  session.role = role
  updated = dict(session_payload)
  updated["session_token"] = app.state.auth.issue_token(session)
  return updated


def _approval_record(
  *,
  chat_payload: dict[str, Any],
  tool_call_id: str,
  approval_id: str,
  user_id: str,
  delegation_id: str | None = None,
  channel: str = "tui",
  tool_name: str = "execute_trade",
  tool_class: str = "external_write",
  state: str = "pending_user",
  persistent_grant_scope: str | None = None,
  approval_constraint: str = "standard",
  required_owner_user_id: str | None = None,
) -> ApprovalRequest:
  now = utc_now()
  return ApprovalRequest(
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    parent_approval_id=None,
    approval_chain_id=approval_id,
    delegation_id=delegation_id,
    request_id=f"req-{approval_id}",
    session_id=chat_payload["session_id"],
    run_id=f"skill-run-{approval_id}",
    user_id=user_id,
    profile="chat",
    channel=channel,
    tool_name=tool_name,
    tool_class=tool_class,  # type: ignore[arg-type]
    tool_args_redacted={"ticker": "AAPL"},
    args_hash=f"hash-{approval_id}",
    reason="needs approval",
    blast_radius_summary="test approval",
    state=state,  # type: ignore[arg-type]
    requested_at=now,
    decided_at=now if state in {"approved", "denied", "expired"} else None,
    decider_id="prior" if state in {"approved", "denied"} else None,
    decision=state if state in {"approved", "denied", "expired"} else None,  # type: ignore[arg-type]
    approval_constraint=approval_constraint,  # type: ignore[arg-type]
    required_owner_user_id=required_owner_user_id,
    persistent_grant_scope=persistent_grant_scope,
    policy_id="test-policy",
    policy_version="1",
    policy_bundle_hash="test-policy-bundle",
  )


def _install_pending_approval(
  *,
  app,
  store: SQLiteApprovalStore,
  chat_payload: dict[str, Any],
  user_id: str,
  delegation_id: str | None = None,
  channel: str = "tui",
  tool_name: str = "execute_trade",
  tool_class: str = "external_write",
  state: str = "pending_user",
  persistent_grant_scope: str | None = None,
  approval_constraint: str = "standard",
  required_owner_user_id: str | None = None,
) -> tuple[ApprovalRequest, asyncio.Queue]:
  tool_call_id = f"tool-{uuid.uuid4().hex}"
  approval_id = f"appr-{uuid.uuid4().hex}"
  request_record = _approval_record(
    chat_payload=chat_payload,
    tool_call_id=tool_call_id,
    approval_id=approval_id,
    user_id=user_id,
    delegation_id=delegation_id,
    channel=channel,
    tool_name=tool_name,
    tool_class=tool_class,
    state=state,
    persistent_grant_scope=persistent_grant_scope,
    approval_constraint=approval_constraint,
    required_owner_user_id=required_owner_user_id,
  )
  _run(store.create(request_record))

  session = _session(app, chat_payload)
  queue: asyncio.Queue = asyncio.Queue(maxsize=1)
  session.pending_tools[tool_call_id] = {
    "approval_id": approval_id,
    "nonce": f"nonce-{approval_id}",
    "requested_at": int(time.time()),
    "status": "approval_pending",
    "tool_name": request_record.tool_name,
    "tool_input": dict(request_record.tool_args_redacted),
    "resolved_qualifier": f"{tool_class}:{tool_name}",
    "reason": request_record.reason,
    "allow_persistent_approval": persistent_grant_scope is not None,
  }
  session.approval_queues[tool_call_id] = queue
  return request_record, queue


def _install_captured_approval(
  *,
  store: SQLiteApprovalStore,
  chat_payload: dict[str, Any],
  user_id: str,
) -> tuple[ApprovalRequest, str]:
  approval_id = f"appr-{uuid.uuid4().hex}"
  prepared = PreparedToolCall(
    {"ticker": "AAPL", "token": "private-original-operand"},
    exact_backend="remote",
  )
  approval = replace(
    _approval_record(
      chat_payload=chat_payload,
      approval_id=approval_id,
      tool_call_id=f"tool-{uuid.uuid4().hex}",
      user_id=user_id,
    ),
    args_hash=sha256_args(prepared.materialize_input()),
  )
  nonce = f"nonce-{approval_id}"
  _run(store.create(approval))
  _run(store.record_executable_binding(
    approval_id=approval.approval_id,
    job_id=f"child-{approval_id}",
    tool_call_id=approval.tool_call_id,
    nonce=nonce,
    tenant_storage_id=user_id,
    tool_name=approval.tool_name,
    prepared_call=prepared,
    dispatch_context={"call_index": 0},
    credential_generation_ref="private-original-generation",
  ))
  return approval, nonce


def _link_sdk_root_approval(store: SQLiteApprovalStore, approval: ApprovalRequest) -> None:
  session_id = approval.session_id
  assert session_id is not None
  generation = _run(store.record_policy_generation(
    policy_id=approval.policy_id,
    policy_version=approval.policy_version,
    policy_bundle_hash=approval.policy_bundle_hash,
  ))
  _run(store.record_root_turn(
    job_id=session_id,
    turn_id=f"turn-{approval.approval_id}",
    tenant_storage_id=approval.user_id,
    capability_bind={
      "model_key": _MODEL_ENTRY.key,
      "provider": _MODEL_ENTRY.provider,
      "credential_ref": _SERVICE_HANDLE.handle_id,
    },
    credential_generation_ref=None,
    effective_grant={"tool_ids": [approval.tool_name], "withheld_tools": []},
    research_binding=None,
    selection_snapshot={"role": "owner", "effort": "high"},
    approval_route_kind="DurableLocalApprovalRoute",
    policy_generation_ref=generation.generation_ref,
  ))
  _run(store.link_root_turn_executable_binding(
    job_id=session_id,
    turn_id=f"turn-{approval.approval_id}",
    tool_call_id=approval.tool_call_id,
    nonce=f"nonce-{approval.approval_id}",
    tenant_storage_id=approval.user_id,
  ))


def _create_delegation_grant(
  store: SQLiteApprovalStore,
  *,
  delegation_id: str,
  delegator_user_id: str,
  excel_session_id: str,
  delegator_channel: str = "web",
) -> DelegationGrant:
  grant = DelegationGrant(
    delegation_id=delegation_id,
    delegator_user_id=delegator_user_id,
    delegator_run_id="delegator-run",
    delegator_session_id="delegator-session",
    delegator_profile="orchestrator",
    delegator_channel=delegator_channel,
    bound_excel_session_id=excel_session_id,
    bound_relay_request_id=f"relay-{delegation_id}",
    bound_workbook="Book1.xlsx",
    tool_class_ceiling=frozenset({"external_write"}),
    args_predicate=None,
    window_seconds=300,
    created_at=utc_now(),
  )
  return _run(store.create_delegation_grant(grant))


def _set_session_channel(app, session_id: str, channel: str) -> None:
  # Test sessions don't bind context.channel to session.channel (it stays None),
  # which would make _channel_matches always true and bypass the cross-channel
  # delegation path. Set distinct channels so the delegation gating is genuinely
  # exercised (mirrors the real excel-channel taskpane vs orchestrator-channel split).
  sess = app.state.auth.session_store.get_session(session_id)
  assert sess is not None, f"session not found: {session_id}"
  sess.channel = channel


def _install_autonomous_pending_approval(
  *,
  app,
  store: SQLiteApprovalStore,
  run_id: str,
  user_id: str,
  channel: str = "tui",
  tool_class: str = "state_write",
) -> ApprovalRequest:
  record = app.state.subprocess_registry._find_by_control_run_id(run_id)
  assert record is not None
  tool_call_id = f"tool-{uuid.uuid4().hex}"
  approval_id = f"appr-{uuid.uuid4().hex}"
  request_record = ApprovalRequest(
    approval_id=approval_id,
    tool_call_id=tool_call_id,
    parent_approval_id=None,
    approval_chain_id=approval_id,
    request_id=run_id,
    session_id=record.session_id,
    run_id=run_id,
    user_id=user_id,
    profile="analyst",
    channel=channel,
    tool_name="memory_write",
    tool_class=tool_class,  # type: ignore[arg-type]
    tool_args_redacted={"file": "notes/approval.md", "content": "<hash:test>"},
    args_hash=f"hash-{approval_id}",
    reason="Tool requires user approval",
    blast_radius_summary="state_write:memory_write",
    state="pending_user",
    requested_at=utc_now(),
    approval_constraint="standard",
    persistent_grant_scope="state_write:memory_write",
    policy_id="test-policy",
    policy_version="1",
    policy_bundle_hash="test-policy-bundle",
  )
  _run(store.create(request_record))
  event = {
    "type": "tool_approval_request",
    "approval_id": approval_id,
    "tool_call_id": tool_call_id,
    "nonce": f"nonce-{approval_id}",
    "tool_name": "memory_write",
    "tool_input": dict(request_record.tool_args_redacted),
    "resolved_qualifier": "",
    "reason": request_record.reason,
    "allow_persistent_approval": True,
    "ts": time.time(),
  }
  record.event_lines.append(event)
  return request_record


def _install_batch_pending_approval(
  *,
  app,
  store: SQLiteApprovalStore,
  batch_id: int,
  owner_user_id: str,
  channel: str = "tui",
  request_id_override: str | None = None,
  expires_at=None,
) -> tuple[ApprovalRequest, asyncio.Queue]:
  run_id = f"batch_{batch_id}"
  session_id = f"batch-stage-{batch_id}-{owner_user_id}"
  tool_call_id = f"tool-batch-{batch_id}-{owner_user_id}"
  approval_id = f"approval-batch-{batch_id}-{owner_user_id}"
  request_record = replace(
    _approval_record(
      chat_payload={"session_id": session_id},
      tool_call_id=tool_call_id,
      approval_id=approval_id,
      user_id=owner_user_id,
      channel=channel,
    ),
    request_id=request_id_override or run_id,
    run_id=run_id,
    profile="analyst",
    expires_at=expires_at,
  )
  _run(store.create(request_record))
  queue: asyncio.Queue = asyncio.Queue(maxsize=1)
  batch_session = SimpleNamespace(
    session_id=session_id,
    user_id=owner_user_id,
    user_email=None,
    channel=channel,
    role="owner",
    approval_store=store,
    approval_policy=app.state.gateway_approval_policy,
    pending_tools={
      tool_call_id: {
        "approval_id": approval_id,
        "nonce": f"nonce-{approval_id}",
        "status": "approval_pending",
        "tool_name": request_record.tool_name,
        "tool_input": dict(request_record.tool_args_redacted),
        "resolved_qualifier": "external_write:execute_trade:AAPL",
        "reason": request_record.reason,
        "allow_persistent_approval": False,
      }
    },
    approval_queues={tool_call_id: queue},
  )
  app.state.batch_task_registry.approval_projections.register_session(
    batch_id=batch_id,
    owner_user_id=owner_user_id,
    channel=channel,
    session=batch_session,
  )
  return request_record, queue


def _install_pinned_batch_pending_approval(
  *,
  app,
  durable_store: SQLiteApprovalStore,
  projection_policy: _NoopPolicy,
  batch_id: int,
  approval_id: str,
  session_store: SQLiteApprovalStore | None = None,
  session_policy: _NoopPolicy | None = None,
  reason: str = "projection-only approval",
) -> tuple[ApprovalRequest, asyncio.Queue]:
  run_id = f"batch_{batch_id}"
  session_id = f"batch-stage-{approval_id}"
  tool_call_id = f"tool-{approval_id}"
  request_record = replace(
    _approval_record(
      chat_payload={"session_id": session_id},
      tool_call_id=tool_call_id,
      approval_id=approval_id,
      user_id="alice",
      channel="tui",
      tool_class="state_write",
    ),
    request_id=run_id,
    run_id=run_id,
    profile="analyst",
    reason=reason,
  )
  _run(durable_store.create(request_record))
  queue: asyncio.Queue = asyncio.Queue(maxsize=1)
  batch_session = SimpleNamespace(
    session_id=session_id,
    user_id="alice",
    channel="tui",
    approval_store=session_store or durable_store,
    approval_policy=session_policy or projection_policy,
    pending_tools={
      tool_call_id: {
        "approval_id": approval_id,
        "nonce": f"nonce-{approval_id}",
        "status": "approval_pending",
      }
    },
    approval_queues={tool_call_id: queue},
  )
  app.state.batch_task_registry.approval_projections.register_session(
    batch_id=batch_id,
    owner_user_id="alice",
    channel="tui",
    session=batch_session,
    store=durable_store,
    policy=projection_policy,
  )
  return request_record, queue


def test_control_approval_list_resolve_unblocks_queue_persists_grant_and_audits(tmp_path) -> None:
  app, store, policy, writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _with_role(app, _control_session(client, "alice"), "owner")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
      tool_name="memory_write",
      tool_class="state_write",
      persistent_grant_scope="state_write:memory_write",
    )

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    approvals = listed.json()["approvals"]
    assert [item["approval_id"] for item in approvals] == [approval.approval_id]
    assert approvals[0]["state"] == "pending_user"

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": True, "reason": "looks safe"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["approval"]["approval_id"] == approval.approval_id
    assert body["approval"]["state"] == "approved"
    assert body["approval"]["decider_id"] == "alice"

    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": True,
      "approval_id": approval.approval_id,
    }

    grant = _run(
      store.find_persistent_grant(
        user_id="alice",
        tool_name="memory_write",
        scope_hint="state_write:memory_write",
      )
    )
    assert grant is not None
    entries, _cursor = _run(writer.query(owner_user_id="alice", approval_id=approval.approval_id, event_type="vote_recorded"))
    assert len(entries) == 1
    assert entries[0].decider_id == "alice"
    assert entries[0].decider_role == "owner"
    assert entries[0].decision_reason == "looks safe"


def test_control_approval_uses_canonical_owner_for_fresh_owner_vote(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice-control-alias")
    chat = _chat_session(client, "alice-chat-alias")
    outsider = _control_session(client, "bob")
    control_session = _session(app, control)
    chat_session = _session(app, chat)
    control_session.owner_user_id = "101"
    control_session.risk_user_id = 101
    control_session.role = "owner"
    chat_session.owner_user_id = "101"
    chat_session.risk_user_id = 101
    control["session_token"] = app.state.auth.issue_token(control_session)
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="101",
      approval_constraint="fresh_human_owner",
      required_owner_user_id="101",
    )

    owner_list = client.get("/api/control/approvals", headers=_headers(control))
    outsider_list = client.get("/api/control/approvals", headers=_headers(outsider))
    cross_user = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(outsider),
      json={"approved": True},
    )
    assert owner_list.status_code == 200, owner_list.text
    assert [item["approval_id"] for item in owner_list.json()["approvals"]] == [
      approval.approval_id
    ]
    assert outsider_list.status_code == 200, outsider_list.text
    assert outsider_list.json()["approvals"] == []
    assert cross_user.status_code == 404
    assert queue.empty()

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={
        "approved": True,
        "allow_tool_type": False,
        "reason": "canonical owner confirms",
      },
    )

    assert response.status_code == 200, response.text
    resolved = response.json()["approval"]
    assert resolved["state"] == "approved"
    assert resolved["decider_id"] == "101"
    assert resolved["required_owner_user_id"] == "101"
    assert queue.get_nowait()["approved"] is True


def test_fresh_owner_money_request_does_not_run_on_another_owner_vote(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(
    tmp_path,
    _NoopPolicy(owner_only_classes={"irreversible"}),
  )
  with TestClient(app) as client:
    control = _control_session(client, "101")
    chat = _chat_session(client, "101")
    control_session = _session(app, control)
    control_session.role = "owner"
    control["session_token"] = app.state.auth.issue_token(control_session)
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="101",
      tool_class="irreversible",
      approval_constraint="fresh_human_owner",
      required_owner_user_id="202",
    )

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )

    assert response.status_code == 200, response.text
    assert response.json()["approval"]["decider_id"] == "101"
    assert queue.get_nowait()["approved"] is False


def test_control_approval_endpoints_reject_chat_session_tokens(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
    )

    listed = client.get("/api/control/approvals", headers=_headers(chat))
    assert listed.status_code == 401
    assert listed.json()["detail"] == "Control session required"

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(chat),
      json={"approved": True},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Control session required"
    retry_response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}/notifications/retry",
      headers=_headers(chat),
    )
    assert retry_response.status_code == 401
    assert retry_response.json()["detail"] == "Control session required"
    assert queue.empty()


def test_control_approval_notification_retry_requeues_failed_outbox_with_redacted_response(tmp_path) -> None:
  async def failing_sender(_row: dict[str, Any]) -> None:
    raise RuntimeError("telegram down")

  app, store, _policy, _writer = _make_app(tmp_path)
  store._notification_destination_resolver = lambda _request: [
    ApprovalNotificationDestination(channel="telegram", destination="chat-private")
  ]
  store._notification_sender = failing_sender
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
    )
    _run(store.enqueue_pending_approval_notification(approval))
    assert _run(store.deliver_pending_approval_notifications()) == 1
    failed_rows = _run(store.list_approval_notification_outbox(approval.approval_id))
    assert failed_rows[0]["state"] == "failed_retryable"
    assert failed_rows[0]["destination"] == "chat-private"
    store._notification_sender = None

    response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}/notifications/retry",
      headers=_headers(control),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
      "status": "queued",
      "approval_id": approval.approval_id,
      "requeued": 1,
      "delivery_scheduled": False,
      "notification": {"state": "pending", "channels": ["telegram"]},
    }
    assert "chat-private" not in response.text
    assert "message" not in body
    requeued_rows = _run(store.list_approval_notification_outbox(approval.approval_id))
    assert requeued_rows[0]["state"] == "pending"
    assert requeued_rows[0]["last_error"] is None

    repeat = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}/notifications/retry",
      headers=_headers(control),
    )

    assert repeat.status_code == 200, repeat.text
    assert repeat.json() == {
      "status": "queued",
      "approval_id": approval.approval_id,
      "requeued": 0,
      "delivery_scheduled": False,
      "notification": {"state": "pending", "channels": ["telegram"]},
    }
    assert queue.empty()


def test_control_approval_list_resolve_routes_approval_pending_autonomous_decision_to_child_inbox(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args
    return _FakeAutonomousProcess(os.dup(kwargs["pass_fds"][0]))

  _patch_fake_autonomous_spawn(monkeypatch, autonomous_runner, fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    autonomous_session = app.state.auth.session_store.get_session(run_id)
    assert autonomous_session is not None
    assert autonomous_session.kind == "chat"
    assert autonomous_session.purpose == AUTONOMOUS_RUNTIME_SESSION_PURPOSE
    assert autonomous_session.pending_tools == {}
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    approvals = listed.json()["approvals"]
    assert [item["approval_id"] for item in approvals] == [approval.approval_id]
    assert approvals[0]["session_id"] == run_id
    assert approvals[0]["run_id"] == run_id

    response = client.post(
      f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": False, "reason": "reviewed diff"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert policy.resolved == [approval.approval_id]

    delivery = _run(
      store.get_autonomous_approval_delivery(
        approval.approval_id,
        tool_call_id=approval.tool_call_id,
        nonce=f"nonce-{approval.approval_id}",
      )
    )
    assert delivery is not None
    assert delivery["approved"] is True
    assert delivery["allow_tool_type"] is False
    assert record.approval_decisions_path is None
    assert record.approval_channel is not None


def test_batch_approval_list_and_vote_use_canonical_owner_and_exact_durable_join(
  tmp_path,
  monkeypatch,
) -> None:
  monkeypatch.setattr(
    approvals_module,
    "_session_owner_user_id",
    lambda _session: "1",
  )
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control_payload = _control_session(client, "henry")
    approval, queue = _install_batch_pending_approval(
      app=app,
      store=store,
      batch_id=59,
      owner_user_id="1",
    )

    listed = client.get("/api/control/approvals", headers=_headers(control_payload))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [
      approval.approval_id
    ]
    assert listed.json()["approvals"][0]["run_id"] == "batch_59"

    response = client.post(
      f"/api/control/runs/batch_59/approvals/{approval.approval_id}",
      headers=_headers(control_payload),
      json={"approved": True, "reason": "reviewed batch write"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
    }


def test_batch_approval_endpoints_pin_projection_store_and_policy_across_same_id_abc(
  tmp_path,
) -> None:
  app_policy = _NoopPolicy()
  session_policy = _NoopPolicy()
  projection_policy = _NoopPolicy()
  app, app_store, _, _writer = _make_app(tmp_path, app_policy)
  session_store = SQLiteApprovalStore(tmp_path / "session-approvals.sqlite3")
  projection_store = SQLiteApprovalStore(tmp_path / "projection-approvals.sqlite3")
  with TestClient(app) as client:
    control = _with_role(app, _control_session(client, "alice"), "owner")
    approval, queue = _install_pinned_batch_pending_approval(
      app=app,
      durable_store=projection_store,
      projection_policy=projection_policy,
      batch_id=71,
      approval_id="approval-same-id-abc",
      session_store=session_store,
      session_policy=session_policy,
      reason="projection-C",
    )
    _run(app_store.create(replace(approval, reason="application-A")))
    _run(session_store.create(replace(approval, reason="session-B")))

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [
      approval.approval_id
    ]
    assert listed.json()["approvals"][0]["reason"] == "projection-C"
    response = client.post(
      f"/api/control/runs/batch_71/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "reason": "pin C"},
    )

    assert response.status_code == 200, response.text
    app_record = _run(app_store.get(approval.approval_id))
    session_record = _run(session_store.get(approval.approval_id))
    projection_record = _run(projection_store.get(approval.approval_id))
    assert app_record is not None
    assert session_record is not None
    assert projection_record is not None
    assert app_record.state == "pending_user"
    assert session_record.state == "pending_user"
    assert projection_record.state == "approved"
    assert app_policy.resolved == []
    assert session_policy.resolved == []
    assert projection_policy.resolved == [approval.approval_id]
    assert app_policy.authorization_checks == []
    assert session_policy.authorization_checks == []
    assert projection_policy.authorization_checks == [("owner", "state_write")]
    assert queue.get_nowait()["approved"] is True


def test_batch_approval_projection_only_store_is_listed_and_resolved(tmp_path) -> None:
  app, app_store, app_policy, _writer = _make_app(tmp_path)
  projection_store = SQLiteApprovalStore(tmp_path / "projection-only.sqlite3")
  projection_policy = _NoopPolicy()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    approval, queue = _install_pinned_batch_pending_approval(
      app=app,
      durable_store=projection_store,
      projection_policy=projection_policy,
      batch_id=72,
      approval_id="approval-projection-only",
      session_store=app_store,
      session_policy=app_policy,
    )

    assert _run(app_store.get(approval.approval_id)) is None
    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [
      approval.approval_id
    ]
    response = client.post(
      f"/api/control/runs/batch_72/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": False, "reason": "projection only"},
    )
    assert response.status_code == 200, response.text
    projection_record = _run(projection_store.get(approval.approval_id))
    assert projection_record is not None
    assert projection_record.state == "denied"
    assert _run(app_store.get(approval.approval_id)) is None
    assert projection_policy.resolved == [approval.approval_id]
    assert app_policy.resolved == []
    assert queue.get_nowait()["approved"] is False


def test_batch_approval_rejects_cross_user_channel_run_and_durable_identity_mismatch(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    bob = _control_session(client, "bob")
    web_alice = _control_session(client, "alice", channel="web")
    approval, queue = _install_batch_pending_approval(
      app=app,
      store=store,
      batch_id=60,
      owner_user_id="alice",
    )

    for session_payload, run_id in (
      (bob, "batch_60"),
      (web_alice, "batch_60"),
      (alice, "batch_61"),
    ):
      response = client.post(
        f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
        headers=_headers(session_payload),
        json={"approved": True},
      )
      assert response.status_code == 404
    assert queue.empty()

    mismatched, mismatch_queue = _install_batch_pending_approval(
      app=app,
      store=store,
      batch_id=62,
      owner_user_id="alice",
      request_id_override="batch_999",
    )
    assert [item["approval_id"] for item in client.get(
      "/api/control/approvals",
      headers=_headers(alice),
    ).json()["approvals"]] == [approval.approval_id]
    response = client.post(
      f"/api/control/runs/batch_62/approvals/{mismatched.approval_id}",
      headers=_headers(alice),
      json={"approved": True},
    )
    assert response.status_code == 404
    assert queue.empty()
    assert mismatch_queue.empty()


def test_batch_approval_delivers_late_authenticated_vote_after_old_expiry(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    approval, queue = _install_batch_pending_approval(
      app=app,
      store=store,
      batch_id=61,
      owner_user_id="alice",
      expires_at=datetime(2000, 1, 1, tzinfo=UTC),
    )
    pending_record = _run(store.get(approval.approval_id))
    assert pending_record is not None
    assert pending_record.state == "pending_user"
    response = client.post(
      f"/api/control/runs/batch_61/approvals/{approval.approval_id}",
      headers=_headers(alice),
      json={"approved": True},
    )
    assert response.status_code == 200, response.text
    approved_record = _run(store.get(approval.approval_id))
    assert approved_record is not None
    assert approved_record.state == "approved"
    assert approved_record.decider_id == "alice"
    delivered = queue.get_nowait()
    assert delivered["approved"] is True
    assert queue.empty()


def test_cancel_autonomous_run_denies_pending_approval_and_clears_queue(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args
    return _FakeAutonomousProcess(os.dup(kwargs["pass_fds"][0]))

  _patch_fake_autonomous_spawn(monkeypatch, autonomous_runner, fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval cancel bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [approval.approval_id]

    cancelled = client.delete(f"/api/control/runs/{run_id}", headers=_headers(control))
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["state"] == "cancelled"

    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "denied"
    assert stored.decider_id == "alice"
    assert stored.decision == "denied"
    assert stored.decision_reason == "run_cancelled"
    assert policy.resolved == [approval.approval_id]

    after_cancel = client.get("/api/control/approvals", headers=_headers(control))
    assert after_cancel.status_code == 200, after_cancel.text
    assert after_cancel.json()["approvals"] == []

    assert record is not None
    delivery = _run(
      store.get_autonomous_approval_delivery(
        approval.approval_id,
        tool_call_id=approval.tool_call_id,
        nonce=f"nonce-{approval.approval_id}",
      )
    )
    assert delivery is not None
    assert delivery["approved"] is False
    assert delivery["allow_tool_type"] is False
    assert record.approval_decisions_path is None
    assert record.approval_channel is None


def test_autonomous_cancel_after_approval_commit_fences_grant_and_sends_no_denial(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args
    return _FakeAutonomousProcess(os.dup(kwargs["pass_fds"][0]))

  _patch_fake_autonomous_spawn(monkeypatch, autonomous_runner, fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval cancellation race",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    grant_committed = threading.Event()
    cancellation_fence_committed = threading.Event()
    release_approval = threading.Event()
    release_cancellation = threading.Event()
    original_create_grant = store.create_persistent_grant
    original_fence = store.fence_persistent_grants_for_cancellation

    async def pause_after_grant_commit(*args: Any, **kwargs: Any) -> Any:
      created = await original_create_grant(*args, **kwargs)
      grant_committed.set()
      while not release_approval.is_set():
        await asyncio.sleep(0.001)
      return created

    async def track_cancellation_fence(*args: Any, **kwargs: Any) -> Any:
      result = await original_fence(*args, **kwargs)
      cancellation_fence_committed.set()
      while not release_cancellation.is_set():
        await asyncio.sleep(0.001)
      return result

    monkeypatch.setattr(
      store,
      "create_persistent_grant",
      pause_after_grant_commit,
    )
    monkeypatch.setattr(
      store,
      "fence_persistent_grants_for_cancellation",
      track_cancellation_fence,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
      approve_future = executor.submit(
        client.post,
        f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
        headers=_headers(control),
        json={
          "approved": True,
          "allow_tool_type": True,
          "reason": "approve before cancellation",
        },
      )
      assert grant_committed.wait(timeout=2)
      cancel_future = executor.submit(
        client.delete,
        f"/api/control/runs/{run_id}",
        headers=_headers(control),
      )
      assert cancellation_fence_committed.wait(timeout=2)
      assert record.proc is not None
      assert record.proc.returncode is None
      release_approval.set()
      approve_response = approve_future.result(timeout=3)
      assert record.proc.returncode is None
      release_cancellation.set()
      cancel_response = cancel_future.result(timeout=3)

    assert approve_response.status_code == 409, approve_response.text
    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["state"] == "cancelled"
    assert record.proc is not None
    assert record.proc.returncode is not None
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "approved"
    delivery = _run(
      store.get_autonomous_approval_delivery(
        approval.approval_id,
        tool_call_id=approval.tool_call_id,
        nonce=f"nonce-{approval.approval_id}",
      )
    )
    assert delivery is not None
    assert delivery["approved"] is True
    assert record.approval_decisions_path is None
    assert record.approval_channel is None
    assert _run(
      store.find_persistent_grant(
        user_id="alice",
        tool_name=approval.tool_name,
        scope_hint=str(approval.persistent_grant_scope),
      )
    ) is None


def test_autonomous_cancel_terminalization_wins_and_delivers_only_denial(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args
    return _FakeAutonomousProcess(os.dup(kwargs["pass_fds"][0]))

  _patch_fake_autonomous_spawn(monkeypatch, autonomous_runner, fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "cancellation wins approval race",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    terminalized = threading.Event()
    release_cancellation = threading.Event()
    original_terminalize = store.terminalize_pending_for_cancellation

    async def pause_after_terminalization(
      *args: Any,
      **kwargs: Any,
    ) -> Any:
      result = await original_terminalize(*args, **kwargs)
      if result[1]:
        terminalized.set()
        while not release_cancellation.is_set():
          await asyncio.sleep(0.001)
      return result

    monkeypatch.setattr(
      store,
      "terminalize_pending_for_cancellation",
      pause_after_terminalization,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
      cancel_future = executor.submit(
        client.delete,
        f"/api/control/runs/{run_id}",
        headers=_headers(control),
      )
      assert terminalized.wait(timeout=2)
      approve_future = executor.submit(
        client.post,
        f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
        headers=_headers(control),
        json={
          "approved": True,
          "allow_tool_type": True,
          "reason": "late approval",
        },
      )
      approve_response = approve_future.result(timeout=3)
      release_cancellation.set()
      cancel_response = cancel_future.result(timeout=3)

    assert approve_response.status_code == 409, approve_response.text
    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["state"] == "cancelled"
    assert record.proc is not None
    assert record.proc.returncode is not None
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "denied"
    delivery = _run(
      store.get_autonomous_approval_delivery(
        approval.approval_id,
        tool_call_id=approval.tool_call_id,
        nonce=f"nonce-{approval.approval_id}",
      )
    )
    assert delivery is not None
    assert delivery["approved"] is False
    assert record.approval_decisions_path is None
    assert record.approval_channel is None
    assert _run(
      store.find_persistent_grant(
        user_id="alice",
        tool_name=approval.tool_name,
        scope_hint=str(approval.persistent_grant_scope),
      )
    ) is None


def test_terminal_autonomous_run_hides_stale_pending_approval(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args
    return _FakeAutonomousProcess(os.dup(kwargs["pass_fds"][0]))

  _patch_fake_autonomous_spawn(monkeypatch, autonomous_runner, fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval timeout bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [approval.approval_id]

    record.state = "failed"
    record.exit_code = 1
    record.completed_at = time.time()
    assert record.proc is not None
    record.proc.returncode = 1

    after_failure = client.get("/api/control/approvals", headers=_headers(control))
    assert after_failure.status_code == 200, after_failure.text
    assert after_failure.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": False, "reason": "too late"},
    )

    assert response.status_code == 409, response.text
    assert response.json()["error"] == "Autonomous run is not running"
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert policy.resolved == []


def test_exited_autonomous_process_hides_pending_approval_even_before_state_update(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args
    return _FakeAutonomousProcess(os.dup(kwargs["pass_fds"][0]))

  _patch_fake_autonomous_spawn(monkeypatch, autonomous_runner, fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval process exit bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    record.state = "approval_pending"

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [item["approval_id"] for item in listed.json()["approvals"]] == [approval.approval_id]

    assert record.proc is not None
    record.proc.returncode = 1

    after_process_exit = client.get("/api/control/approvals", headers=_headers(control))
    assert after_process_exit.status_code == 200, after_process_exit.text
    assert after_process_exit.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": False, "reason": "process already exited"},
    )

    assert response.status_code == 409, response.text
    assert response.json()["error"] == "Autonomous run is not running"
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert policy.resolved == []


def test_autonomous_approval_delivery_failure_preserves_durable_retry(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner

  async def fake_exec(*args, **kwargs):
    _ = args
    return _FakeAutonomousProcess(os.dup(kwargs["pass_fds"][0]))

  _patch_fake_autonomous_spawn(monkeypatch, autonomous_runner, fake_exec)
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "approval bridge test",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    approval = _install_autonomous_pending_approval(
      app=app,
      store=store,
      run_id=run_id,
      user_id="alice",
    )
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    assert record is not None
    assert record.approval_channel is not None
    record.approval_channel.close()

    response = client.post(
      f"/api/control/runs/{run_id}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "allow_tool_type": False, "reason": "reviewed diff"},
    )

    assert response.status_code == 409, response.text
    assert "approval channel endpoint is unavailable" in response.json()["error"]
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "approved"
    delivery = _run(
      store.get_autonomous_approval_delivery(
        approval.approval_id,
        tool_call_id=approval.tool_call_id,
        nonce=f"nonce-{approval.approval_id}",
      )
    )
    assert delivery is not None
    assert delivery["state"] == "pending"
    assert delivery["approved"] is True
    assert delivery["last_error"] is not None
    assert policy.resolved == [approval.approval_id]


def test_role_class_precheck_blocks_approval_but_allows_denial_for_invite(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path, policy=_NoopPolicy(owner_only_classes={"external_write"}))
  with TestClient(app) as client:
    control = _with_role(app, _control_session(client, "alice"), "invite")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(app=app, store=store, chat_payload=chat, user_id="alice")

    blocked = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )
    assert blocked.status_code == 403, blocked.text
    after_block = _run(store.get(approval.approval_id))
    assert after_block is not None
    assert after_block.state == "pending_user"
    assert policy.authorization_checks == [("invite", "external_write")]
    assert queue.empty()

    denied = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": False, "reason": "cancel"},
    )
    assert denied.status_code == 200, denied.text
    assert denied.json()["approval"]["state"] == "denied"
    assert queue.get_nowait() == {
      "approved": False,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
    }
    assert policy.authorization_checks == [("invite", "external_write")]


def test_control_approval_wrong_run_and_cross_user_are_404(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    bob_chat = _chat_session(client, "bob")
    bob_approval, _queue = _install_pending_approval(app=app, store=store, chat_payload=bob_chat, user_id="bob")

    wrong_run = client.post(
      f"/api/control/runs/not-a-run/approvals/{bob_approval.approval_id}",
      headers=_headers(alice_control),
      json={"approved": True},
    )
    cross_user = client.post(
      f"/api/control/runs/{bob_chat['session_id']}/approvals/{bob_approval.approval_id}",
      headers=_headers(alice_control),
      json={"approved": True},
    )

    assert wrong_run.status_code == 404
    assert cross_user.status_code == 404


def test_control_approval_chat_endpoints_are_channel_scoped(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    web_control = _control_session(client, "alice", channel="web")
    excel_chat = _chat_session(client, "alice", channel="excel")
    _session(app, excel_chat).channel = "excel"
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=excel_chat,
      user_id="alice",
      channel="excel",
    )

    listed = client.get("/api/control/approvals", headers=_headers(web_control))
    assert listed.status_code == 200, listed.text
    assert listed.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{excel_chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(web_control),
      json={"approved": False, "reason": "wrong channel"},
    )

    assert response.status_code == 404, response.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert queue.empty()


def test_delegated_chat_approval_visible_and_resolvable_cross_channel(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    web_control = _control_session(client, "alice", channel="web")
    excel_chat = _chat_session(client, "alice", channel="excel")
    _set_session_channel(app, web_control["session_id"], "web")
    _set_session_channel(app, excel_chat["session_id"], "excel")
    delegation_id = f"deleg-{uuid.uuid4().hex}"
    grant = _create_delegation_grant(
      store,
      delegation_id=delegation_id,
      delegator_user_id="alice",
      excel_session_id=excel_chat["session_id"],
    )
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=excel_chat,
      user_id="alice",
      delegation_id=grant.delegation_id,
      channel="excel",
    )

    listed = client.get("/api/control/approvals", headers=_headers(web_control))
    assert listed.status_code == 200, listed.text
    approvals = listed.json()["approvals"]
    assert [item["approval_id"] for item in approvals] == [approval.approval_id]
    assert approvals[0]["delegation_id"] == delegation_id
    assert approvals[0]["channel"] == "excel"

    response = client.post(
      f"/api/control/runs/{excel_chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(web_control),
      json={"approved": True, "allow_tool_type": False, "reason": "delegated review"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert response.json()["approval"]["decider_id"] == "alice"
    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
    }


def test_chat_tool_approval_endpoint_records_ordinary_user_denial(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(app=app, store=store, chat_payload=chat, user_id="alice")
    session = _session(app, chat)
    pending = next(iter(session.pending_tools.values()))

    response = client.post(
      "/api/chat/tool-approval",
      headers=_headers(chat),
      json={
        "tool_call_id": approval.tool_call_id,
        "nonce": pending["nonce"],
        "approved": False,
        "allow_tool_type": True,
      },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["approval"]["state"] == "denied"
    assert payload["approval"]["decision_reason"] is None
    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": False,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
    }
    assert pending["allow_tool_type"] is False
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "denied"
    assert stored.decision_reason is None


def test_delegated_chat_approval_hidden_from_non_delegator_cross_channel(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    bob_control = _control_session(client, "bob", channel="web")
    excel_chat = _chat_session(client, "alice", channel="excel")
    delegation_id = f"deleg-{uuid.uuid4().hex}"
    grant = _create_delegation_grant(
      store,
      delegation_id=delegation_id,
      delegator_user_id="alice",
      excel_session_id=excel_chat["session_id"],
    )
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=excel_chat,
      user_id="alice",
      delegation_id=grant.delegation_id,
      channel="excel",
    )

    listed = client.get("/api/control/approvals", headers=_headers(bob_control))
    assert listed.status_code == 200, listed.text
    assert listed.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{excel_chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(bob_control),
      json={"approved": True, "reason": "not mine"},
    )

    assert response.status_code == 404, response.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert queue.empty()


def test_delegated_chat_approval_hidden_when_grant_delegator_differs(tmp_path) -> None:
  # Same-user floor passes (alice owns the excel session, authed as alice), but the
  # grant's delegator is a DIFFERENT user, so the delegation visibility check must
  # still reject. Pins _approval_visible_via_delegation independently of the same-user
  # floor (the non-delegator test above is short-circuited by that floor first).
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice", channel="web")
    excel_chat = _chat_session(client, "alice", channel="excel")
    _set_session_channel(app, alice_control["session_id"], "web")
    _set_session_channel(app, excel_chat["session_id"], "excel")
    delegation_id = f"deleg-{uuid.uuid4().hex}"
    grant = _create_delegation_grant(
      store,
      delegation_id=delegation_id,
      delegator_user_id="carol",
      excel_session_id=excel_chat["session_id"],
    )
    approval, queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=excel_chat,
      user_id="alice",
      delegation_id=grant.delegation_id,
      channel="excel",
    )

    listed = client.get("/api/control/approvals", headers=_headers(alice_control))
    assert listed.status_code == 200, listed.text
    assert listed.json()["approvals"] == []

    response = client.post(
      f"/api/control/runs/{excel_chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(alice_control),
      json={"approved": True, "reason": "wrong delegator"},
    )
    assert response.status_code == 404, response.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert stored.state == "pending_user"
    assert queue.empty()


def test_terminal_and_expired_approval_states_are_rejected(tmp_path) -> None:
  app, store, _policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    terminal, _terminal_queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
      state="approved",
    )
    expired, _expired_queue = _install_pending_approval(
      app=app,
      store=store,
      chat_payload=chat,
      user_id="alice",
      state="expired",
    )

    terminal_response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{terminal.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )
    expired_response = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{expired.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )

    assert terminal_response.status_code == 409
    assert expired_response.status_code == 410

    terminal_retry = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{terminal.approval_id}/notifications/retry",
      headers=_headers(control),
    )
    expired_retry = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{expired.approval_id}/notifications/retry",
      headers=_headers(control),
    )

    assert terminal_retry.status_code == 409
    assert terminal_retry.json()["error"] == "Approval request already resolved"
    assert expired_retry.status_code == 409
    assert expired_retry.json()["error"] == "Approval request already resolved"


def test_chat_tool_approval_endpoint_uses_shared_helper(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(app=app, store=store, chat_payload=chat, user_id="alice")
    session = _session(app, chat)
    pending = next(iter(session.pending_tools.values()))

    response = client.post(
      "/api/chat/tool-approval",
      headers=_headers(chat),
      json={
        "tool_call_id": approval.tool_call_id,
        "nonce": pending["nonce"],
        "approved": True,
        "allow_tool_type": False,
      },
    )
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert policy.resolved == [approval.approval_id]
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
    }


def test_control_captured_approval_votes_after_cache_clear_and_stale_address(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice-control-alias")
    chat = _chat_session(client, "alice-chat-alias")
    for payload in (control, chat):
      owner_session = _session(app, payload)
      owner_session.owner_user_id = "101"
      owner_session.risk_user_id = 101
    control = _with_role(app, control, "owner")
    approval, nonce = _install_captured_approval(
      store=store,
      chat_payload=chat,
      user_id="101",
    )
    session = _session(app, chat)
    session.pending_tools[approval.tool_call_id] = {
      "approval_id": approval.approval_id,
      "nonce": nonce,
      "status": "approval_pending",
    }
    session.approval_queues[approval.tool_call_id] = asyncio.Queue(maxsize=1)
    session.pending_tools.clear()
    session.approval_queues.clear()

    response = client.post(
      f"/api/control/runs/stale-receiving-address/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "reason": "current owner decision"},
    )
    assert response.status_code == 200, response.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert (stored.state, stored.decider_id, stored.decider_role, stored.decision_reason) == (
      "approved", "101", "owner", "current owner decision",
    )
    assert response.json()["continuation"] == {
      "available": False,
      "unavailable_owner": "root_continue_hook",
      "job_id": f"child-{approval.approval_id}",
      "executable_binding_state": "captured",
    }
    for private_value in ("private-original-operand", "private-original-generation", nonce):
      assert private_value not in response.text
    assert session.pending_tools == {}
    assert session.approval_queues == {}

    replay = client.post(
      f"/api/control/runs/another-address/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": False, "reason": "cannot replace durable decision"},
    )
    assert replay.status_code == 200, replay.text
    assert _run(store.get(approval.approval_id)) == stored
    assert policy.resolved == [approval.approval_id]


def test_control_lists_parked_approval_without_live_session_entry(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    chat = _with_role(app, _chat_session(client, "alice"), "owner")
    control = _with_role(app, _control_session(client, "alice"), "owner")
    web_control = _control_session(client, "alice", channel="web")
    bob_control = _control_session(client, "bob")
    approval, nonce = _install_captured_approval(
      store=store,
      chat_payload=chat,
      user_id="alice",
    )
    session = _session(app, chat)
    assert approval.tool_call_id not in session.pending_tools

    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    approvals = listed.json()["approvals"]
    assert [item["approval_id"] for item in approvals] == [approval.approval_id]
    assert approvals[0]["state"] == "pending_user"
    assert approvals[0]["tool_call_id"] == approval.tool_call_id
    assert "nonce" not in approvals[0]
    for private_value in ("private-original-operand", "private-original-generation", nonce):
      assert private_value not in listed.text

    session.pending_tools[approval.tool_call_id] = {
      "approval_id": approval.approval_id,
      "nonce": nonce,
      "status": "approval_pending",
    }
    live_and_parked = client.get("/api/control/approvals", headers=_headers(control))
    assert [item["approval_id"] for item in live_and_parked.json()["approvals"]] == [
      approval.approval_id
    ]
    session.pending_tools.clear()

    for outsider in (bob_control, web_control):
      hidden = client.get("/api/control/approvals", headers=_headers(outsider))
      assert hidden.status_code == 200, hidden.text
      assert hidden.json()["approvals"] == []

    decided = client.post(
      f"/api/control/runs/{chat['session_id']}/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True, "reason": "found it parked"},
    )
    assert decided.status_code == 200, decided.text
    assert policy.resolved == [approval.approval_id]
    after_decision = client.get("/api/control/approvals", headers=_headers(control))
    assert after_decision.status_code == 200, after_decision.text
    assert after_decision.json()["approvals"] == []


def test_captured_approval_keeps_auth_tenant_channel_and_current_role_policy(tmp_path) -> None:
  app, store, policy, _writer = _make_app(
    tmp_path,
    policy=_NoopPolicy(owner_only_classes={"external_write"}),
  )
  with TestClient(app) as client:
    old_chat = _with_role(app, _chat_session(client, "alice"), "owner")
    current_chat = _with_role(app, _chat_session(client, "alice"), "invite")
    bob_chat = _chat_session(client, "bob")
    web_control = _control_session(client, "alice", channel="web")
    approval, nonce = _install_captured_approval(
      store=store,
      chat_payload=old_chat,
      user_id="alice",
    )
    continuations: list[Any] = []

    async def unexpected_continuation(request):
      continuations.append(request)
      raise AssertionError("An unauthorized vote must not reach continuation")

    app.state.root_continue_hook = unexpected_continuation
    payload = {
      "tool_call_id": approval.tool_call_id,
      "nonce": nonce,
      "approved": True,
      "role": "owner",
      "user_id": "alice",
    }
    unauthenticated = client.post("/api/chat/tool-approval", json=payload)
    assert unauthenticated.status_code == 401
    cross_tenant = client.post(
      "/api/chat/tool-approval",
      headers=_headers(bob_chat),
      json=payload,
    )
    assert cross_tenant.status_code == 404
    assert approval.approval_id not in cross_tenant.text
    assert nonce not in cross_tenant.text
    channel_denied = client.post(
      f"/api/control/runs/old-address/approvals/{approval.approval_id}",
      headers=_headers(web_control),
      json={"approved": True},
    )
    assert channel_denied.status_code == 404
    wrong_nonce = client.post(
      "/api/chat/tool-approval",
      headers=_headers(current_chat),
      json={**payload, "nonce": "not-the-original-nonce"},
    )
    assert wrong_nonce.status_code == 404
    role_denied = client.post(
      "/api/chat/tool-approval",
      headers=_headers(current_chat),
      json=payload,
    )
    for private_value in ("private-original-operand", "private-original-generation", nonce):
      assert private_value not in role_denied.text
    pending = _run(store.get(approval.approval_id))
    assert pending == approval
    assert policy.resolved == []
    assert continuations == []
    binding, canonical = _run(store._read_executable_binding_for_executor(
      job_id=f"child-{approval.approval_id}",
      tool_call_id=approval.tool_call_id,
      nonce=nonce,
      tenant_storage_id="alice",
    ))
    assert canonical == approval
    assert _run(store._read_completed_executable_result_for_executor(binding=binding)) is None


def test_current_invite_control_can_deny_captured_approval_without_executing_it(tmp_path) -> None:
  app, store, policy, _writer = _make_app(
    tmp_path,
    policy=_NoopPolicy(owner_only_classes={"external_write"}),
  )
  with TestClient(app) as client:
    chat = _with_role(app, _chat_session(client, "alice"), "owner")
    control = _with_role(app, _control_session(client, "alice"), "invite")
    approval, nonce = _install_captured_approval(
      store=store,
      chat_payload=chat,
      user_id="alice",
    )
    denied = client.post(
      f"/api/control/runs/stale-address/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": False},
    )
    assert denied.status_code == 200, denied.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None
    assert (stored.state, stored.decider_id, stored.decider_role) == ("denied", "alice", "invite")
    assert policy.resolved == [approval.approval_id]
    binding, canonical = _run(store._read_executable_binding_for_executor(
      job_id=f"child-{approval.approval_id}",
      tool_call_id=approval.tool_call_id,
      nonce=nonce,
      tenant_storage_id="alice",
    ))
    assert canonical == stored
    assert _run(store._read_completed_executable_result_for_executor(binding=binding)) is None


def test_linked_live_sdk_approval_uses_original_queue_once_without_native_restore(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app, store=store, chat_payload=chat, user_id="alice",
    )
    _link_sdk_root_approval(store, approval)
    response = client.post(
      f"/api/control/runs/stale-address/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )
    assert response.status_code == 200, response.text
    assert "continuation" not in response.json()
    assert queue.get_nowait() == {
      "approved": True,
      "allow_tool_type": False,
      "approval_id": approval.approval_id,
    }
    replay = client.post(
      f"/api/control/runs/stale-address/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": False},
    )
    assert replay.status_code == 409
    assert queue.empty()
    assert policy.resolved == [approval.approval_id]
    locator = _run(store.locate_request(
      approval_id=approval.approval_id, tenant_storage_id="alice",
    ))
    assert locator is not None
    assert locator.executable_binding_state == "not_restorable"


def test_ended_sdk_approval_accepts_durable_vote_without_fabricating_native_call(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    approval, queue = _install_pending_approval(
      app=app, store=store, chat_payload=chat, user_id="alice",
    )
    _link_sdk_root_approval(store, approval)
    session = _session(app, chat)
    session.pending_tools.clear()
    session.approval_queues.clear()
    response = client.post(
      f"/api/control/runs/no-longer-live/approvals/{approval.approval_id}",
      headers=_headers(control),
      json={"approved": True},
    )
    assert response.status_code == 200, response.text
    stored = _run(store.get(approval.approval_id))
    assert stored is not None and stored.state == "approved"
    assert response.json()["continuation"] == {
      "available": False,
      "unavailable_owner": "root_continue_hook",
      "job_id": chat["session_id"],
      "executable_binding_state": "not_restorable",
    }
    assert queue.empty()
    assert policy.resolved == [approval.approval_id]
    locator = _run(store.locate_request(
      approval_id=approval.approval_id, tenant_storage_id="alice",
    ))
    assert locator is not None
    assert locator.job_id == chat["session_id"]
    assert locator.executable_binding_state == "not_restorable"


def test_child_hook_consumes_committed_vote_and_reports_failure_without_retry(tmp_path) -> None:
  app, store, policy, _writer = _make_app(tmp_path)
  child_requests: list[ChildContinueRequest] = []
  failed_approval_ids: set[str] = set()

  async def classify_root(request: RootContinueRequest) -> RootContinueOutcome:
    root = await store._locate_root_turn_for_approval(
      approval_id=request.approval_id,
      tenant_storage_id=request.tenant_storage_id,
    )
    assert root is None
    return RootContinueOutcome(status="not_root")

  async def consume_child_vote(request: ChildContinueRequest) -> ChildContinueOutcome:
    located = await store._locate_request_for_decision(
      approval_id=request.approval_id,
      tenant_storage_id=request.tenant_storage_id,
    )
    assert located is not None
    decision, locator, _nonce = located
    # The real durable decision must be visible to the installed consumer,
    # not merely queued or projected as an approved HTTP response.
    assert decision.state == "approved"
    assert decision.decider_id == request.tenant_storage_id
    assert decision.decided_at is not None
    child_requests.append(request)
    if request.approval_id in failed_approval_ids:
      raise RuntimeError("private-child-adapter-fault")
    # This fixture has no restored runtime: do not claim an execution effect.
    return ChildContinueOutcome(
      status="not_restorable",
      job_id=locator.job_id,
      tool_call_id=locator.tool_call_id,
    )

  with TestClient(app) as client:
    app.state.root_continue_hook = classify_root
    app.state.child_continue_hook = consume_child_vote
    control = _with_role(app, _control_session(client, "alice"), "owner")
    chat = _chat_session(client, "alice")
    approvals: list[ApprovalRequest] = []
    for fail_delivery in (False, True):
      approval, _nonce = _install_captured_approval(
        store=store,
        chat_payload=chat,
        user_id="alice",
      )
      approvals.append(approval)
      if fail_delivery:
        failed_approval_ids.add(approval.approval_id)
      response = client.post(
        f"/api/control/runs/not-the-owning-address/approvals/{approval.approval_id}",
        headers=_headers(control),
        json={
          "approved": True,
          "tenant_storage_id": "bob",
          "parked_task_id": "invented-task",
        },
      )
      assert response.status_code == 200, response.text
      stored = _run(store.get(approval.approval_id))
      assert stored is not None and stored.state == "approved"
      if fail_delivery:
        assert response.json()["continuation"] == {
          "status": "outcome_unknown",
          "failed_owner": "child_continue_hook",
          "job_id": f"child-{approval.approval_id}",
        }
        assert "private-child-adapter-fault" not in response.text
      else:
        assert response.json()["continuation"] == {
          "status": "not_restorable",
          "job_id": f"child-{approval.approval_id}",
          "tool_call_id": approval.tool_call_id,
        }
    assert child_requests == [
      ChildContinueRequest(approval.approval_id, "alice")
      for approval in approvals
    ]
    assert policy.resolved == [approval.approval_id for approval in approvals]


def test_unrestricted_autonomous_launch_row_is_listed_and_votable_from_web(
  monkeypatch,
  tmp_path,
) -> None:
  from agent_gateway import autonomous_runner
  from agent_gateway.approval_policy import approval_request_projection
  from agent_gateway.control_plane.autonomous_approval_delivery import (
    create_delegated_autonomous_approval,
  )

  async def fake_exec(*args, **kwargs):
    return _FakeAutonomousProcess(os.dup(kwargs["pass_fds"][0]))

  _patch_fake_autonomous_spawn(monkeypatch, autonomous_runner, fake_exec)
  app, store, policy, _writer = _make_app(tmp_path)
  with TestClient(app) as client:
    control = _control_session(client, "alice", channel="web")
    registry = app.state.subprocess_registry
    assert client.portal is not None
    started = client.portal.call(lambda: registry.start(
      role="owner", profile="analyst", mode="task", task="unrestricted approval",
      user_id="alice", user_email="alice@example.test", channel=None,
    ))
    record = registry._find_by_control_run_id(started["run_id"])
    assert record is not None
    assert record.channel is None
    request = _approval_record(
      chat_payload={"session_id": record.session_id},
      tool_call_id="unrestricted-tool-call",
      approval_id="unrestricted-approval",
      user_id="alice",
      channel="child-projection-is-not-authority",
    )
    nonce = "unrestricted-original-nonce"
    event = {
      "type": "tool_approval_request",
      "approval_id": request.approval_id,
      "tool_call_id": request.tool_call_id,
      "nonce": nonce,
      "tool_name": request.tool_name,
      "tool_input": dict(request.tool_args_redacted),
      "durable_request": approval_request_projection(request),
    }
    stored = client.portal.call(lambda: create_delegated_autonomous_approval(
      store=store, policy=policy, record=record, event=event,
    ))
    assert stored.channel == "cli"
    assert record.channel is None
    record.event_lines.append(event)
    record.state = "approval_pending"
    listed = client.get("/api/control/approvals", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [row["approval_id"] for row in listed.json()["approvals"]] == [
      request.approval_id,
    ]
    response = client.post(
      f"/api/control/runs/{record.control_run_id}/approvals/{request.approval_id}",
      headers=_headers(control),
      json={"approved": True, "reason": "authorized web vote"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["approval"]["state"] == "approved"
    assert response.json()["approval"]["channel"] == "cli"
    assert record.channel is None
    delivery = _run(store.get_autonomous_approval_delivery(
      request.approval_id, tool_call_id=request.tool_call_id, nonce=nonce,
    ))
    assert delivery is not None
    assert delivery["approved"] is True
