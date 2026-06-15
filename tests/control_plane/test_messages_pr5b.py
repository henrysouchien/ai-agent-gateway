from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "messages-pr5b-key"
HMAC_KEY = "messages-pr5b-hmac"


class _EchoRunner:
  def __init__(self, event_log: EventLog, turns: list[list[dict[str, Any]]]) -> None:
    self._event_log = event_log
    self._turns = turns

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = system_prompt, model_override, max_turns
    self._turns.append(list(messages))
    last_user = next((message["content"] for message in reversed(messages) if message.get("role") == "user"), "")
    self._event_log.append({"type": "text_delta", "text": f"echo:{last_user}"})
    self._event_log.append({"type": "stream_complete", "usage": {}})


class _FakeAutonomousProcess:
  returncode: int | None = None

  async def wait(self) -> int:
    while self.returncode is None:
      await asyncio.sleep(0.01)
    return self.returncode

  def terminate(self) -> None:
    self.returncode = -15

  def kill(self) -> None:
    self.returncode = -9


def _make_app(turns: list[list[dict[str, Any]]] | None = None):
  turns = turns if turns is not None else []

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid: _EchoRunner(event_log, turns),
    )

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="messages-pr5b-test-secret-0123456789",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _control_session(client: TestClient, user_id: str, *, channel: str | None = "tui") -> dict[str, Any]:
  payload: dict[str, Any] = {"api_key": API_KEY, "user_id": user_id, "context": {}}
  if channel is not None:
    payload["context"]["channel"] = channel
  response = client.post(
    "/api/control/session",
    json=payload,
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session_payload: dict[str, Any], *, token_key: str = "session_token") -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload[token_key]}"}


def _dispatch_chat(client: TestClient, control: dict[str, Any], message: str = "first") -> dict[str, Any]:
  response = client.post(
    "/api/control/runs",
    headers=_headers(control),
    json={"kind": "chat", "message": message, "channel": "tui"},
  )
  assert response.status_code == 200, response.text
  return response.json()


async def _collect_available_user_events(app: Any, user_id: str, control_run_id: str) -> list[dict[str, Any]]:
  subscription = app.state.user_event_bus.subscribe(user_id, control_run_id=control_run_id)
  events: list[dict[str, Any]] = []
  try:
    while True:
      try:
        events.append(await asyncio.wait_for(subscription.__anext__(), timeout=0.1))
      except asyncio.TimeoutError:
        return events
  finally:
    close = getattr(subscription, "aclose", None)
    if callable(close):
      await close()
  return events


def test_messages_openapi_keeps_typed_request_union() -> None:
  app = _make_app()

  schema = app.openapi()
  request_schema = schema["paths"]["/api/control/runs/{control_run_id}/messages"]["post"]["requestBody"]["content"][
    "application/json"
  ]["schema"]

  encoded = json.dumps(request_schema)
  assert "ChatContinuationRequest" in encoded
  assert "AutonomousRunMessageRequest" in encoded


def test_chat_messages_continue_with_chat_session_token_and_full_transcript() -> None:
  turns: list[list[dict[str, Any]]] = []
  app = _make_app(turns)
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    dispatched = _dispatch_chat(client, control, "first")

    response = client.post(
      f"/api/control/runs/{dispatched['chat_session_id']}/messages",
      headers={"Authorization": f"Bearer {dispatched['chat_session_token']}"},
      json={
        "messages": [
          {"role": "user", "content": "first"},
          {"role": "assistant", "content": "echo:first"},
          {"role": "user", "content": "second"},
        ],
        "context": {"channel": "tui"},
        "request_id": "continue-second",
      },
    )

    assert response.status_code == 200, response.text
    response_body = response.json()
    assert response_body["run"]["state"] == "completed"
    assert response_body["message_id"] == "continue-second"
    assert response_body["delivery_status"] == "delivered"
    assert turns == [
      [{"role": "user", "content": "first"}],
      [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "echo:first"},
        {"role": "user", "content": "second"},
      ],
    ]

    session = app.state.auth.session_store.get_session(dispatched["chat_session_id"])
    assert session is not None
    retained = session.event_history.snapshot()
    retained_text = [event.get("text") for event in retained if event.get("type") == "text_delta"]
    assert retained_text == ["echo:first", "echo:second"]

    parent_events = [event for event in retained if event.get("type") == "parent_message_sent"]
    assert parent_events == [
      {
        "type": "parent_message_sent",
        "run_id": dispatched["chat_session_id"],
        "control_run_id": dispatched["chat_session_id"],
        "session_id": dispatched["chat_session_id"],
        "message_id": "continue-second",
        "request_id": "continue-second",
        "message": "second",
        "channel": "tui",
        "sender": {
          "session_id": dispatched["chat_session_id"],
          "user_id": "alice",
        },
        "sent_at": parent_events[0]["sent_at"],
        "ts": parent_events[0]["ts"],
      }
    ]

    logs = client.get(
      f"/api/control/runs/{dispatched['chat_session_id']}/logs?tail=40",
      headers=_headers(control),
    )
    assert logs.status_code == 200, logs.text
    log_events = [json.loads(line) for line in logs.json()["log_lines"]]
    assert any(event == parent_events[0] for event in log_events)

    replayed = asyncio.run(_collect_available_user_events(app, "alice", dispatched["chat_session_id"]))
    assert any(event == parent_events[0] for event in replayed)

    duplicate = client.post(
      f"/api/control/runs/{dispatched['chat_session_id']}/messages",
      headers={"Authorization": f"Bearer {dispatched['chat_session_token']}"},
      json={
        "messages": [
          {"role": "user", "content": "first"},
          {"role": "assistant", "content": "echo:first"},
          {"role": "user", "content": "different duplicate text"},
        ],
        "context": {"channel": "tui"},
        "request_id": "continue-second",
      },
    )
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["delivery_status"] == "duplicate"
    assert duplicate.json()["message_id"] == "continue-second"
    assert turns == [
      [{"role": "user", "content": "first"}],
      [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "echo:first"},
        {"role": "user", "content": "second"},
      ],
    ]
    retained_after_duplicate = session.event_history.snapshot()
    assert [
      event for event in retained_after_duplicate if event.get("type") == "parent_message_sent"
    ] == parent_events


def test_chat_messages_accept_matching_control_session_and_reject_wrong_tokens() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    first = _dispatch_chat(client, control, "first")
    second = _dispatch_chat(client, control, "other")

    wrong_chat_token = client.post(
      f"/api/control/runs/{first['chat_session_id']}/messages",
      headers={"Authorization": f"Bearer {second['chat_session_token']}"},
      json={"messages": [{"role": "user", "content": "bad"}]},
    )
    matching_control_token = client.post(
      f"/api/control/runs/{first['chat_session_id']}/messages",
      headers=_headers(control),
      json={"messages": [{"role": "user", "content": "second via control"}]},
    )
    wrong_user_control = _control_session(client, "bob")
    wrong_user_response = client.post(
      f"/api/control/runs/{first['chat_session_id']}/messages",
      headers=_headers(wrong_user_control),
      json={"messages": [{"role": "user", "content": "bad user"}]},
    )
    wrong_channel_control = _control_session(client, "alice", channel="excel")
    wrong_channel_response = client.post(
      f"/api/control/runs/{first['chat_session_id']}/messages",
      headers=_headers(wrong_channel_control),
      json={"messages": [{"role": "user", "content": "bad channel"}]},
    )

    assert wrong_chat_token.status_code == 401
    assert matching_control_token.status_code == 200
    assert matching_control_token.json()["run"]["state"] == "completed"
    assert wrong_user_response.status_code == 401
    assert wrong_channel_response.status_code == 404


def test_autonomous_messages_deliver_to_operator_inbox(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "logs"))
  app = _make_app()

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeAutonomousProcess()

  from agent_gateway import autonomous_runner

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert start.status_code == 200, start.text
    assert start.json()["run"]["messageable"] is True

    response = client.post(
      f"/api/control/runs/{start.json()['run_id']}/messages",
      headers=_headers(control),
      json={"message": "Focus on AWS exposure next.", "message_id": "msg-1"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["message_id"] == "msg-1"
    assert body["delivery_status"] == "delivered"
    assert body["run"]["kind"] == "autonomous"
    assert body["run"]["messageable"] is True

    record = app.state.subprocess_registry._tasks[start.json()["task_id"]]
    assert record.operator_inbox_path is not None
    inbox_lines = record.operator_inbox_path.read_text(encoding="utf-8").splitlines()
    assert len(inbox_lines) == 1
    inbox_record = json.loads(inbox_lines[0])
    assert inbox_record["message_id"] == "msg-1"
    assert inbox_record["text"] == "Focus on AWS exposure next."
    assert inbox_record["sender"]["user_id"] == "alice"

    event = next(event for event in record.event_lines if event["type"] == "parent_message_sent")
    assert event["message_id"] == "msg-1"
    assert event["task_id"] == start.json()["task_id"]
    assert event["task_type"] == "autonomous"
    assert record.events_path is not None
    persisted_events = [
      json.loads(line)
      for line in record.events_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
    ]
    persisted_parent_events = [
      event for event in persisted_events if event["type"] == "parent_message_sent"
    ]
    assert len(persisted_parent_events) == 1
    assert persisted_parent_events[0]["message_id"] == "msg-1"
    assert persisted_parent_events[0]["run_id"] == start.json()["run_id"]
    assert persisted_parent_events[0]["control_run_id"] == start.json()["run_id"]
    record.event_lines = [
      event for event in record.event_lines if event.get("type") != "parent_message_sent"
    ]
    record.events_path.write_text("", encoding="utf-8")

    duplicate = client.post(
      f"/api/control/runs/{start.json()['run_id']}/messages",
      headers=_headers(control),
      json={"message": "Different duplicate text should not replace inbox text.", "message_id": "msg-1"},
    )
    assert duplicate.status_code == 200, duplicate.text
    assert duplicate.json()["delivery_status"] == "duplicate"
    assert len(record.operator_inbox_path.read_text(encoding="utf-8").splitlines()) == 1
    persisted_events_after_duplicate = [
      json.loads(line)
      for line in record.events_path.read_text(encoding="utf-8").splitlines()
      if line.strip()
    ]
    parent_events_after_duplicate = [
      event for event in persisted_events_after_duplicate if event["type"] == "parent_message_sent"
    ]
    assert len(parent_events_after_duplicate) == 1
    assert parent_events_after_duplicate[0]["message"] == "Focus on AWS exposure next."

    other_control = _control_session(client, "bob")
    wrong_user = client.post(
      f"/api/control/runs/{start.json()['run_id']}/messages",
      headers=_headers(other_control),
      json={"message": "take over", "message_id": "msg-2"},
    )
    assert wrong_user.status_code == 404

    wrong_channel = _control_session(client, "alice", channel="excel")
    wrong_channel_response = client.post(
      f"/api/control/runs/{start.json()['run_id']}/messages",
      headers=_headers(wrong_channel),
      json={"message": "wrong channel", "message_id": "msg-3"},
    )
    assert wrong_channel_response.status_code == 404

    no_channel = _control_session(client, "alice", channel=None)
    no_channel_response = client.post(
      f"/api/control/runs/{start.json()['run_id']}/messages",
      headers=_headers(no_channel),
      json={"message": "missing channel", "message_id": "msg-4"},
    )
    assert no_channel_response.status_code == 404

    chat = _dispatch_chat(client, control, "hello")
    chat_token_response = client.post(
      f"/api/control/runs/{start.json()['run_id']}/messages",
      headers={"Authorization": f"Bearer {chat['chat_session_token']}"},
      json={"message": "chat token", "message_id": "msg-5"},
    )
    assert chat_token_response.status_code == 401


def test_autonomous_messages_reject_terminal_run_with_409(monkeypatch, tmp_path: Path) -> None:
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "logs"))
  app = _make_app()

  async def fake_exec(*args, **kwargs):
    _ = args, kwargs
    return _FakeAutonomousProcess()

  from agent_gateway import autonomous_runner

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)

  with TestClient(app) as client:
    control = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "summarize"},
    )
    assert start.status_code == 200, start.text

    record = app.state.subprocess_registry._tasks[start.json()["task_id"]]
    record.state = "completed"
    record.completed_at = time.time()

    run = client.get(
      f"/api/control/runs/{start.json()['run_id']}",
      headers=_headers(control),
    )
    assert run.status_code == 200, run.text
    assert run.json()["messageable"] is False

    response = client.post(
      f"/api/control/runs/{start.json()['run_id']}/messages",
      headers=_headers(control),
      json={"message": "too late", "message_id": "msg-terminal"},
    )

    assert response.status_code == 409
