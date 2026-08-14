from __future__ import annotations

import asyncio
from itertools import count
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from agent_gateway.autonomous_capability_handoff import AutonomousCapabilityBinding
from agent_gateway.autonomous_event_channel import (
  adopt_inherited_autonomous_event_channel,
)
from agent_gateway.autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_ENV,
)
from agent_gateway.capability_binding import (
  CapabilityBind,
  CredentialHandle,
)
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.control_plane import events as events_module
from agent_gateway.control_plane.events import (
  _projected_control_event_chunks,
  _shielded_aclose,
)
from agent_gateway.server import (
  ChatRuntime,
  GatewayServerConfig,
  MaterializedCredential,
  create_gateway_app,
)


API_KEY = "events-pr5b-key"
HMAC_KEY = "events-pr5b-hmac-key-at-least-32-bytes"
_MODEL_ENTRY = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:events-pr5b:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="events-pr5b",
  actor_id=None,
)
_SERVICE_MATERIAL = MaterializedCredential(
  handle=_SERVICE_HANDLE,
  auth_config={
    "provider": "anthropic",
    "api_key": "test-key",
    "billing_mode": "byok",
    "rate_table_version": "test",
  },
)


def _materialize_service_credential(
  handle: CredentialHandle,
) -> MaterializedCredential:
  if handle is not _SERVICE_HANDLE:
    raise RuntimeError("unknown test credential handle")
  return _SERVICE_MATERIAL


def _autonomous_capability_binding(request) -> AutonomousCapabilityBinding:
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
      credential_ref=_SERVICE_HANDLE.handle_id,
      run_mode=request.run_mode,
      registry_revision=INITIAL_MODEL_REGISTRY.revision,
      policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
      selection_source="capability_default",
    ),
    materialized_credential=_SERVICE_MATERIAL,
  )


_FAKE_PROCESS_PIDS = count(95_000)
_FAKE_PROCESSES: dict[int, "_CompletedProcess"] = {}


class _FakeStdin:
  def write(self, payload: bytes) -> None:
    _ = payload

  async def drain(self) -> None:
    return None

  def close(self) -> None:
    return None

  async def wait_closed(self) -> None:
    return None


class _CompletedProcess:
  def __init__(self, inherited_event_fd: int) -> None:
    self.pid = next(_FAKE_PROCESS_PIDS)
    self._returncode: int | None = None
    self.stdin = _FakeStdin()
    self._inherited_fds = [inherited_event_fd]
    self._event_channel = None
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
      os.close(inherited_fd)
    self._inherited_fds.clear()

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def retain_inherited_fds(self, inherited_fds: tuple[int, ...]) -> None:
    self._inherited_fds.extend(os.dup(fd) for fd in inherited_fds)

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


class _NoopRunner:
  def __init__(self, capability_execution: Any) -> None:
    self.capability_execution = capability_execution

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, max_turns


def _make_app(monkeypatch, tmp_path: Path, events: list[dict[str, Any]]):
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "logs"))

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda *_args: _NoopRunner(
        request.capability_execution
      ),
      capability_execution=request.capability_execution,
    )

  async def fake_exec(*args, **kwargs):
    _ = args
    process = _CompletedProcess(os.dup(kwargs["pass_fds"][0]))
    envelope = json.loads(
      kwargs["env"][AUTONOMOUS_CAPABILITY_ENVELOPE_ENV]
    )
    process.start_event_channel(envelope["channel_id"])
    process.retain_inherited_fds(tuple(kwargs["pass_fds"][1:]))
    return process

  from agent_gateway import autonomous_runner

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  monkeypatch.setattr(autonomous_runner, "_get_process_group_id", lambda pid: pid)
  monkeypatch.setattr(
    autonomous_runner,
    "_signal_process_group",
    lambda process_group_id, signal_number: setattr(
      _FAKE_PROCESSES[process_group_id], "returncode", -signal_number
    ),
  )
  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="events-pr5b-test-secret-0123456789",
      valid_api_keys={API_KEY},
      tenant_id="events-pr5b",
      allow_service_credentials_for_interactive=True,
      model_registry=INITIAL_MODEL_REGISTRY,
      model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      service_provider_handles={"anthropic": _SERVICE_HANDLE},
      service_auth_config_resolver=_materialize_service_credential,
      build_chat_runtime=_build_chat_runtime,
      autonomous_capability_binding_resolver=_autonomous_capability_binding,
      claim_signing_authority=GatewayClaimSigningAuthority(HMAC_KEY),
    )
  )


def _control_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  payload = response.json()
  session = client.app.state.auth.session_store.get_session(payload["session_id"])
  assert session is not None
  session.model_entitled_capabilities = CAPABILITY_IDS
  session.model_entitled_keys = frozenset(INITIAL_MODEL_REGISTRY.models)
  payload["session_token"] = client.app.state.auth.issue_token(session)
  return payload


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def _decode_sse_chunk(chunk: bytes) -> dict[str, Any]:
  line = chunk.decode("utf-8").strip()
  assert line.startswith("data: ")
  return json.loads(line.removeprefix("data: "))


def test_projected_control_event_adaptation_failure_logs_traceback(caplog) -> None:
  caplog.set_level(
    logging.WARNING,
    logger="agent_gateway.control_plane.events",
  )

  async def subscription():
    yield SimpleNamespace(
      event={"type": "stream_complete", "usage": {}},
      seq=1,
      control_run_id="run-adaptation-failure",
    )

  async def case() -> list[dict[str, Any]]:
    chunks = _projected_control_event_chunks(
      subscription=subscription(),
      auth=None,  # type: ignore[arg-type]
      app_state=SimpleNamespace(),
      authenticated=None,  # type: ignore[arg-type]
      schema_version=99,
      enforce_visibility=False,
    )
    return [_decode_sse_chunk(chunk) async for chunk in chunks]

  events = asyncio.run(case())

  assert events == [{
    "run_id": "run-adaptation-failure",
    "seq": 1,
    "event": {
      "type": "stream_error",
      "error": "No control adapter for schema_version=99",
    },
  }]
  records = [
    record
    for record in caplog.records
    if record.name == "agent_gateway.control_plane.events"
    and "control event adaptation failed" in record.getMessage()
  ]
  assert len(records) == 1
  assert records[0].exc_info
  assert records[0].exc_info[0] is ValueError


def test_control_event_serialization_failure_logs_traceback(
  monkeypatch,
  tmp_path: Path,
  caplog,
) -> None:
  class _RaisingStr:
    def __str__(self) -> str:
      raise RuntimeError("string conversion failed")

  class _SerializationFailureBus:
    def subscribe(self, _user_id: str, *, control_run_id: str | None = None):
      _ = control_run_id

      async def subscription():
        yield {"type": "custom", "value": _RaisingStr()}

      return subscription()

    async def shutdown(self) -> None:
      return None

  caplog.set_level(
    logging.WARNING,
    logger="agent_gateway.control_plane.events",
  )
  app = _make_app(monkeypatch, tmp_path, [])

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    app.state.user_event_bus = _SerializationFailureBus()

    async def case() -> dict[str, Any]:
      route = next(
        route
        for route in app.routes
        if getattr(route, "path", None) == "/api/control/events"
      )
      request = Request(
        {
          "type": "http",
          "method": "GET",
          "path": "/api/control/events",
          "headers": [(
            b"authorization",
            f"Bearer {control['session_token']}".encode("utf-8"),
          )],
          "query_string": b"",
          "app": app,
        }
      )
      response = await route.endpoint(
        request,
        control_run_id=None,
        run_id=None,
        schema_version=None,
        after_seq=0,
      )
      try:
        chunk = await response.body_iterator.__anext__()
        return _decode_sse_chunk(chunk)
      finally:
        await _shielded_aclose(response.body_iterator)
        if response.background is not None:
          await response.background()

    event = asyncio.run(case())

  assert event == {
    "type": "stream_error",
    "error": "SSE serialization failed: string conversion failed",
  }
  records = [
    record
    for record in caplog.records
    if record.name == "agent_gateway.control_plane.events"
    and "SSE serialization failed" in record.getMessage()
  ]
  assert len(records) == 1
  assert records[0].exc_info
  assert records[0].exc_info[0] is RuntimeError


def test_control_events_cancelled_aclose_does_not_leave_pending_close_task() -> None:
  class _SlowClose:
    def __init__(self) -> None:
      self.started = asyncio.Event()
      self.release = asyncio.Event()
      self.finished = asyncio.Event()

    async def aclose(self) -> None:
      self.started.set()
      await self.release.wait()
      self.finished.set()

  async def case() -> None:
    iterator = _SlowClose()
    before = set(asyncio.all_tasks())
    close_task = asyncio.create_task(_shielded_aclose(iterator))
    await asyncio.wait_for(iterator.started.wait(), timeout=0.5)

    close_task.cancel()
    await asyncio.sleep(0)
    close_task.cancel()
    await asyncio.sleep(0)
    assert close_task.done() is False

    iterator.release.set()
    with pytest.raises(asyncio.CancelledError):
      await close_task
    assert iterator.finished.is_set()

    leaked = [
      task
      for task in asyncio.all_tasks() - before
      if task is not asyncio.current_task() and not task.done()
    ]
    assert leaked == []

  asyncio.run(case())


async def _collect_control_events(app, token: str, *, control_run_id: str, count: int) -> list[dict[str, Any]]:
  route = next(route for route in app.routes if getattr(route, "path", None) == "/api/control/events")
  request = Request(
    {
      "type": "http",
      "method": "GET",
      "path": "/api/control/events",
      "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
      "query_string": f"control_run_id={control_run_id}".encode("utf-8"),
      "app": app,
    }
  )
  response = await route.endpoint(request, control_run_id=control_run_id, run_id=None)
  events: list[dict[str, Any]] = []
  try:
    iterator = response.body_iterator
    for _ in range(count):
      chunk = await asyncio.wait_for(iterator.__anext__(), timeout=0.5)
      events.append(_decode_sse_chunk(chunk))
  finally:
    close = getattr(response.body_iterator, "aclose", None)
    if callable(close):
      await close()
    if response.background is not None:
      await response.background()
  return events


def test_control_events_replays_autonomous_buffered_events(monkeypatch, tmp_path: Path) -> None:
  typed_events = [
    {"type": "skill_run_started", "skill_run_id": "skill-1", "skill": "earnings-review"},
    {
      "type": "skill_result_captured",
      "skill_run_id": "skill-1",
      "skill": "earnings-review",
      "verdict_echo": {"verdict_token": "BUY", "one_line_summary": "ok"},
    },
    {"type": "artifact_ready", "skill_run_id": "skill-1", "artifact_id": "artifact-1"},
    {"type": "artifact_failed", "skill_run_id": "skill-1", "artifact_id": "artifact-2"},
  ]
  app = _make_app(monkeypatch, tmp_path, typed_events)

  with TestClient(app) as client:
    bus = app.state.user_event_bus
    app.state.user_event_bus = None
    alice = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "skill",
        "skill": "earnings-review",
        "ticker": "AAPL",
        "channel": "tui",
      },
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    registry = app.state.subprocess_registry
    record = registry._tasks[start.json()["task_id"]]
    record.proc.returncode = 0

    async def finish_and_inject() -> None:
      if record.reaper_task is not None:
        await record.reaper_task
      for event in typed_events:
        await registry._record_and_publish_event(record, event)

    client.portal.call(finish_and_inject)
    app.state.user_event_bus = bus
    registry.set_user_event_bus(bus)

    async def collect() -> list[dict[str, Any]]:
      return await _collect_control_events(
        app, alice["session_token"], control_run_id=run_id, count=6
      )

    received = client.portal.call(collect)

    received_types = [event["type"] for event in received]
    for expected in ["skill_run_started", "skill_result_captured", "artifact_ready", "artifact_failed"]:
      assert expected in received_types
    assert "run_state_changed" in received_types
    assert all(event["control_run_id"] == run_id for event in received)


def test_control_events_fast_run_race_replays_late_subscriber(monkeypatch, tmp_path: Path) -> None:
  fast_events = [{"type": "fast_event", "seq": index} for index in range(3)]
  app = _make_app(monkeypatch, tmp_path, fast_events)

  with TestClient(app) as client:
    bus = app.state.user_event_bus
    app.state.user_event_bus = None
    alice = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "quick"},
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    registry = app.state.subprocess_registry
    record = registry._tasks[start.json()["task_id"]]
    record.proc.returncode = 0

    async def finish_and_inject() -> None:
      if record.reaper_task is not None:
        await record.reaper_task
      for event in fast_events:
        await registry._record_and_publish_event(record, event)

    client.portal.call(finish_and_inject)
    app.state.user_event_bus = bus
    registry.set_user_event_bus(bus)

    async def collect() -> list[dict[str, Any]]:
      return await _collect_control_events(
        app, alice["session_token"], control_run_id=run_id, count=5
      )

    received = client.portal.call(collect)

    seqs = [event.get("seq") for event in received if event.get("type") == "fast_event"]
    assert seqs == [0, 1, 2]


def test_control_events_scopes_chat_tokens_to_their_own_run(monkeypatch, tmp_path: Path) -> None:
  app = _make_app(monkeypatch, tmp_path, [])
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    first = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={"kind": "chat", "message": "first", "channel": "tui"},
    )
    second = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={"kind": "chat", "message": "second", "channel": "tui"},
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    denied = client.get(
      f"/api/control/events?control_run_id={second.json()['chat_session_id']}",
      headers={"Authorization": f"Bearer {first.json()['chat_session_token']}"},
    )
    assert denied.status_code == 401


def test_control_events_require_exact_autonomous_owner_and_known_run(
  monkeypatch,
  tmp_path: Path,
) -> None:
  app = _make_app(monkeypatch, tmp_path, [])
  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    bob = _control_session(client, "bob")
    started = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={
        "kind": "autonomous",
        "profile": "analyst",
        "mode": "task",
        "task": "owner visibility",
      },
    )
    assert started.status_code == 200, started.text
    run_id = started.json()["run_id"]
    record = app.state.subprocess_registry._find_by_control_run_id(run_id)
    alice_session = app.state.auth.session_store.get_session(alice["session_id"])
    bob_session = app.state.auth.session_store.get_session(bob["session_id"])
    assert record is not None
    assert alice_session is not None
    assert bob_session is not None
    record.user_id = "raw-alice-alias"
    record.owner_user_id = "alice"

    assert events_module._run_visible_to_session(
      auth=app.state.auth,
      app_state=app.state,
      authenticated=alice_session,
      run_id=run_id,
    )
    assert not events_module._run_visible_to_session(
      auth=app.state.auth,
      app_state=app.state,
      authenticated=bob_session,
      run_id=run_id,
    )
    assert client.get(
      f"/api/control/events?control_run_id={run_id}",
      headers=_headers(bob),
    ).status_code == 404
    assert client.get(
      "/api/control/events?control_run_id=bg_unknown",
      headers=_headers(alice),
    ).status_code == 404


def test_control_events_rejects_wrong_channel_control_token_for_chat_run(monkeypatch, tmp_path: Path) -> None:
  app = _make_app(monkeypatch, tmp_path, [])
  with TestClient(app) as client:
    control = _control_session(client, "alice", channel="tui")
    chat = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={"kind": "chat", "message": "first", "channel": "tui"},
    )
    assert chat.status_code == 200, chat.text

    wrong_channel = _control_session(client, "alice", channel="excel")
    denied = client.get(
      f"/api/control/events?control_run_id={chat.json()['chat_session_id']}",
      headers=_headers(wrong_channel),
    )

    assert denied.status_code == 404
