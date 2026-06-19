from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

from fastapi import Request
from fastapi.testclient import TestClient

from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatMessage, ChatRequest, ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.session import AuthManager


API_KEY = "runs-test-key"


class _EchoTurnRunner:
  def __init__(self, event_log: EventLog) -> None:
    self._event_log = event_log

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = system_prompt, model_override, max_turns
    last_user = next((message["content"] for message in reversed(messages) if message.get("role") == "user"), "")
    self._event_log.append({"type": "text_delta", "text": f"echo:{last_user}"})
    self._event_log.append({"type": "stream_complete", "usage": {}})


class _FailingTurnRunner:
  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, model_override, max_turns
    raise RuntimeError("provider stream failed")


class _HangingTurnRunner:
  def __init__(self) -> None:
    self.started = threading.Event()
    self.cancelled = threading.Event()
    self.disconnected = threading.Event()

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, model_override, max_turns
    self.started.set()
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError:
      self.cancelled.set()
      raise

  async def on_disconnect(self) -> None:
    self.disconnected.set()
    await asyncio.sleep(0)


class _StreamingHangingTurnRunner(_HangingTurnRunner):
  def __init__(self, event_log: EventLog) -> None:
    super().__init__()
    self._event_log = event_log

  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, model_override, max_turns
    self.started.set()
    self._event_log.append({"type": "text_delta", "text": "started"})
    try:
      await asyncio.Event().wait()
    except asyncio.CancelledError:
      self.cancelled.set()
      raise


def _make_app():
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda event_log, _sid: _EchoTurnRunner(event_log),
    )

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _make_failing_app():
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda _event_log, _sid: _FailingTurnRunner(),
    )

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _make_setup_failing_app():
  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    raise RuntimeError("runtime setup failed")

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _make_hanging_app() -> tuple[Any, _HangingTurnRunner]:
  runner = _HangingTurnRunner()

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(
      system_prompt="system",
      build_runner=lambda _event_log, _sid: runner,
      disconnect_handler=runner.on_disconnect,
    )

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return app, runner


def _make_streaming_hanging_app() -> tuple[Any, dict[str, _StreamingHangingTurnRunner]]:
  captured: dict[str, _StreamingHangingTurnRunner] = {}

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager

    def _build_runner(event_log: EventLog, _sid: str) -> _StreamingHangingTurnRunner:
      runner = _StreamingHangingTurnRunner(event_log)
      captured["runner"] = runner
      return runner

    return ChatRuntime(
      system_prompt="system",
      build_runner=_build_runner,
      disconnect_handler=lambda: captured["runner"].on_disconnect(),
    )

  app = create_gateway_app(
    GatewayServerConfig(
      jwt_secret="runs-pr4-test-secret-0123456789x",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )
  return app, captured


def _control_session(client: TestClient, user_id: str) -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": "tui"}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _chat_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/chat/init",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session['session_token']}"}


def _wait_until(predicate, *, timeout: float = 1.0) -> bool:
  deadline = time.time() + timeout
  while time.time() < deadline:
    if predicate():
      return True
    time.sleep(0.01)
  return bool(predicate())


def _collect_chat_turn_events(client: TestClient, token: str, message: str) -> list[dict[str, Any]]:
  events: list[dict[str, Any]] = []
  with client.stream(
    "POST",
    "/api/chat",
    headers={"Authorization": f"Bearer {token}"},
    json={"messages": [{"role": "user", "content": message}], "context": {"channel": "tui"}},
  ) as response:
    assert response.status_code == 200, response.text
    for line in response.iter_lines():
      if not line:
        continue
      text = line.decode("utf-8") if isinstance(line, bytes) else line
      if text.startswith("data:"):
        text = text[5:].strip()
      events.append(_unwrap_sse_payload(json.loads(text)))
  return events


def _unwrap_sse_payload(payload: dict[str, Any]) -> dict[str, Any]:
  candidate = payload.get("event")
  if isinstance(payload.get("seq"), int) and isinstance(candidate, dict) and isinstance(candidate.get("type"), str):
    return candidate
  return payload


def _consume_chat_turn(client: TestClient, token: str, message: str) -> None:
  _collect_chat_turn_events(client, token, message)


def test_chat_stream_body_iterator_close_keeps_dispatch_task_until_session_expiry() -> None:
  async def case() -> None:
    app, captured = _make_streaming_hanging_app()
    auth = app.state.auth
    session = auth.session_store.create_session(
      AuthManager.hash_api_key(API_KEY),
      user_id="alice",
      kind="chat",
    )
    token = auth.issue_token(session)
    route = next(route for route in app.routes if getattr(route, "path", None) == "/api/chat")
    request = Request(
      {
        "type": "http",
        "method": "POST",
        "path": "/api/chat",
        "headers": [(b"authorization", f"Bearer {token}".encode("utf-8"))],
        "query_string": b"",
        "app": app,
      }
    )
    response = await route.endpoint(
      request,
      body=ChatRequest(
        messages=[ChatMessage(role="user", content="stream then disconnect")],
        context={"channel": "tui"},
      ),
    )

    chunk = await asyncio.wait_for(response.body_iterator.__anext__(), timeout=0.5)
    assert b"text_delta" in chunk
    close = getattr(response.body_iterator, "aclose", None)
    assert callable(close)
    await close()

    await asyncio.sleep(0)
    running = [
      task
      for task in asyncio.all_tasks()
      if task is not asyncio.current_task()
      and not task.done()
      and "_dispatch_chat_turn" in getattr(task.get_coro(), "__qualname__", repr(task.get_coro()))
    ]
    try:
      assert running
      runner = captured["runner"]
      assert not runner.disconnected.is_set()
      assert not runner.cancelled.is_set()
      assert session.stream_active is True
      assert session.active_turn is not None
      assert session.active_turn.is_running
      assert session.active_turn.subscribers == {}

      await auth.session_store.expire_session_async(session.session_id)
      await asyncio.sleep(0)
      leaked = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and "_dispatch_chat_turn" in getattr(task.get_coro(), "__qualname__", repr(task.get_coro()))
      ]
      assert leaked == []
      assert runner.disconnected.is_set()
      assert runner.cancelled.is_set()
      assert session.stream_active is False
      assert session.active_turn is None
    finally:
      for task in running:
        task.cancel()
      if running:
        await asyncio.gather(*running, return_exceptions=True)
      await app.state.user_event_bus.shutdown()

  asyncio.run(case())


def test_list_runs_returns_user_chat_sessions_and_excludes_control_sessions() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_empty_chat = _chat_session(client, "alice")
    alice_chat = _chat_session(client, "alice")
    bob_chat = _chat_session(client, "bob")

    alice_session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    alice_empty_session = app.state.auth.session_store.get_session(alice_empty_chat["session_id"])
    bob_session = app.state.auth.session_store.get_session(bob_chat["session_id"])
    assert alice_session is not None
    assert alice_empty_session is not None
    assert bob_session is not None
    alice_session.initial_message = "hello alice"
    alice_session.event_history.append({"type": "stream_complete", "usage": {}})
    bob_session.event_history.append({"type": "stream_complete", "usage": {}})

    response = client.get("/api/control/runs?kind=chat", headers=_headers(alice_control))
    assert response.status_code == 200, response.text
    runs = response.json()["runs"]

    assert [run["run_id"] for run in runs] == [alice_chat["session_id"]]
    assert runs[0]["kind"] == "chat"
    assert runs[0]["session_id"] == alice_chat["session_id"]
    assert runs[0]["user_id"] == "alice"
    assert runs[0]["state"] == "completed"
    assert runs[0]["initial_message"] == "hello alice"

    control_filtered = client.get("/api/control/runs?kind=control", headers=_headers(alice_control))
    assert control_filtered.status_code == 200
    assert control_filtered.json() == {"runs": []}

    state_filtered = client.get("/api/control/runs?state=running", headers=_headers(alice_control))
    assert state_filtered.status_code == 200
    assert state_filtered.json() == {"runs": []}

    empty_detail = client.get(f"/api/control/runs/{alice_empty_chat['session_id']}", headers=_headers(alice_control))
    empty_logs = client.get(
      f"/api/control/runs/{alice_empty_chat['session_id']}/logs",
      headers=_headers(alice_control),
    )
    empty_message = client.post(
      f"/api/control/runs/{alice_empty_chat['session_id']}/messages",
      headers=_headers(alice_control),
      json={"messages": [{"role": "user", "content": "continue blank session"}]},
    )

    assert empty_detail.status_code == 404
    assert empty_logs.status_code == 404
    assert empty_message.status_code == 404


def test_get_run_returns_chat_run_shape_and_404s_unknown_or_cross_user() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")
    bob_chat = _chat_session(client, "bob")

    session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    assert session is not None
    session.channel = "tui"
    session.initial_message = "first"
    session.event_history.append(
      {
        "type": "skill_result_captured",
        "skill_run_id": "skill-1",
        "skill": "model-review",
        "cost_usd": 0.09,
        "verdict_echo": {
          "verdict_token": "PRICE_TARGET_SET",
          "confidence": "HIGH",
          "one_line_summary": "summary",
        },
      }
    )

    response = client.get(f"/api/control/runs/{alice_chat['session_id']}", headers=_headers(alice_control))
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload == {
      "kind": "chat",
      "run_id": alice_chat["session_id"],
      "session_id": alice_chat["session_id"],
      "agent": "hank",
      "channel": "tui",
      "user_id": "alice",
      "state": "starting",
      "started_at": payload["started_at"],
      "ended_at": None,
      "cost_usd": 0.09,
      "initial_message": "first",
      "skill_run_ids": ["skill-1"],
      "current_verdict": {
        "verdict_token": "PRICE_TARGET_SET",
        "confidence": "HIGH",
        "one_line_summary": "summary",
        "skill_run_id": "skill-1",
      },
      "pending_approval": None,
    }

    unknown = client.get("/api/control/runs/not-a-run", headers=_headers(alice_control))
    cross_user = client.get(f"/api/control/runs/{bob_chat['session_id']}", headers=_headers(alice_control))

    assert unknown.status_code == 404
    assert cross_user.status_code == 404


def test_chat_run_cost_accumulates_completed_turns_and_live_partial() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")

    session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    assert session is not None
    session.channel = "tui"
    session.initial_message = "first"
    session.event_history.append({"type": "stream_complete", "usage": {"estimated_cost": 0.10}})
    session.event_history.append({"type": "stream_complete", "usage": {"estimated_cost": 0.20}})
    session.stream_active = True
    session.event_history.append({"type": "turn_complete", "turn": 1, "usage": {"estimated_cost": 0.03}})

    response = client.get(f"/api/control/runs/{alice_chat['session_id']}", headers=_headers(alice_control))

  assert response.status_code == 200, response.text
  payload = response.json()
  assert payload["state"] == "running"
  assert payload["cost_usd"] == 0.33


def test_chat_run_cost_accumulates_child_and_parent_stream_totals() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")

    session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    assert session is not None
    session.channel = "tui"
    session.initial_message = "parent with sub-agent"
    session.event_history.append(
      {
        "type": "stream_complete",
        "sub_agent_id": "sub0:child",
        "usage": {"estimated_cost": 0.20},
      }
    )
    session.event_history.append(
      {
        "type": "stream_complete",
        "usage": {"estimated_cost": 0.03},
      }
    )

    response = client.get(f"/api/control/runs/{alice_chat['session_id']}", headers=_headers(alice_control))

  assert response.status_code == 200, response.text
  assert response.json()["cost_usd"] == 0.23


def test_get_run_logs_reads_session_event_history_tail_and_enforces_user_scope() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")
    bob_chat = _chat_session(client, "bob")

    alice_session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    bob_session = app.state.auth.session_store.get_session(bob_chat["session_id"])
    assert alice_session is not None
    assert bob_session is not None
    for index in range(3):
      alice_session.event_history.append({"type": "event", "index": index})
    bob_session.event_history.append({"type": "event", "index": "bob"})

    response = client.get(
      f"/api/control/runs/{alice_chat['session_id']}/logs?tail=2",
      headers=_headers(alice_control),
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["run_id"] == alice_chat["session_id"]
    assert payload["more_available"] is True
    assert [json.loads(line)["index"] for line in payload["log_lines"]] == [1, 2]

    cross_user = client.get(
      f"/api/control/runs/{bob_chat['session_id']}/logs",
      headers=_headers(alice_control),
    )
    assert cross_user.status_code == 404


def test_chat_session_event_history_retains_events_across_three_turns() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")

    streamed_event_types: list[str | None] = []
    for index in range(3):
      streamed_event_types.extend(
        event.get("type") for event in _collect_chat_turn_events(client, chat["session_token"], f"turn-{index}")
      )

    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    retained = session.event_history.snapshot()
    retained_text = [event.get("text") for event in retained if event.get("type") == "text_delta"]
    retained_states = [event.get("state") for event in retained if event.get("type") == "run_state_changed"]

    assert retained_text == ["echo:turn-0", "echo:turn-1", "echo:turn-2"]
    assert retained_states == ["running", "completed", "running", "completed", "running", "completed"]
    assert [event.get("type") for event in retained].count("stream_complete") == 3
    assert "run_state_changed" not in streamed_event_types

    run = client.get(f"/api/control/runs/{chat['session_id']}", headers=_headers(control))
    assert run.status_code == 200, run.text
    assert run.json()["state"] == "completed"
    assert run.json()["ended_at"] is not None

    response = client.get(
      f"/api/control/runs/{chat['session_id']}/logs?tail=20",
      headers=_headers(control),
    )
    assert response.status_code == 200, response.text
    log_text = "\n".join(response.json()["log_lines"])
    assert "echo:turn-0" in log_text
    assert "echo:turn-1" in log_text
    assert "echo:turn-2" in log_text
    assert '"state": "completed"' in log_text


def test_failed_normal_chat_turn_records_terminal_control_lifecycle() -> None:
  app = _make_failing_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")

    _consume_chat_turn(client, chat["session_token"], "fail")

    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    retained = session.event_history.snapshot()
    retained_states = [event.get("state") for event in retained if event.get("type") == "run_state_changed"]

    assert retained_states == ["running", "failed"]
    assert retained[-2]["type"] == "error"
    assert retained[-1]["state"] == "failed"

    run = client.get(f"/api/control/runs/{chat['session_id']}", headers=_headers(control))
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["state"] == "failed"
    assert payload["ended_at"] is not None


def test_setup_failure_records_terminal_control_lifecycle() -> None:
  app = _make_setup_failing_app()
  with TestClient(app, raise_server_exceptions=False) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")

    response = client.post(
      "/api/chat",
      headers=_headers(chat),
      json={"messages": [{"role": "user", "content": "fail during setup"}], "context": {"channel": "tui"}},
    )

    assert response.status_code == 200
    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    retained = session.event_history.snapshot()
    retained_states = [event.get("state") for event in retained if event.get("type") == "run_state_changed"]

    assert retained_states == ["running", "failed"]
    assert any(event.get("type") == "error" and "runtime setup failed" in event.get("error", "") for event in retained)

    run = client.get(f"/api/control/runs/{chat['session_id']}", headers=_headers(control))
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["state"] == "failed"
    assert payload["ended_at"] is not None


def test_active_later_chat_turn_overrides_prior_terminal_lifecycle() -> None:
  app = _make_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")
    chat = _chat_session(client, "alice")
    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    session.event_history.append({"type": "run_state_changed", "state": "running", "ts": int(time.time()) - 3})
    session.event_history.append({"type": "stream_complete", "usage": {}})
    session.event_history.append({"type": "run_state_changed", "state": "completed", "ts": int(time.time()) - 2})
    session.event_history.append({"type": "run_state_changed", "state": "running", "ts": int(time.time()) - 1})
    session.stream_active = True

    run = client.get(f"/api/control/runs/{chat['session_id']}", headers=_headers(control))
    assert run.status_code == 200, run.text
    payload = run.json()
    assert payload["state"] == "running"
    assert payload["ended_at"] is None


def test_delete_running_control_chat_cancels_background_task_and_disconnects() -> None:
  app, runner = _make_hanging_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "keep running",
        "channel": "tui",
        "deadline_sec": 1,
      },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["run"]["state"] == "running"
    assert runner.started.is_set()

    chat_session_id = payload["chat_session_id"]
    session = app.state.auth.session_store.get_session(chat_session_id)
    assert session is not None
    assert session.stream_active is True
    assert any(key.startswith("control_chat_turn:") for key in session.control_chat_tasks)
    assert not any(key.startswith("control_chat_turn:") for key in session.background_tasks)

    deleted = client.delete(f"/api/control/runs/{chat_session_id}", headers=_headers(control))
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["state"] == "cancelled"
    retained_states = [event.get("state") for event in session.event_history.snapshot() if event.get("type") == "run_state_changed"]
    assert retained_states == ["running", "cancelled"]

    assert runner.disconnected.wait(timeout=1.0)
    assert runner.cancelled.wait(timeout=1.0)
    assert _wait_until(lambda: session.stream_active is False)
    assert _wait_until(lambda: not any(key.startswith("control_chat_turn:") for key in session.control_chat_tasks))
    assert app.state.auth.session_store.get_session(chat_session_id) is None


def test_expiring_running_control_chat_cancels_background_task_and_disconnects() -> None:
  app, runner = _make_hanging_app()
  with TestClient(app) as client:
    control = _control_session(client, "alice")

    response = client.post(
      "/api/control/runs",
      headers=_headers(control),
      json={
        "kind": "chat",
        "message": "expire while running",
        "channel": "tui",
        "deadline_sec": 1,
      },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["run"]["state"] == "running"
    assert runner.started.is_set()

    chat_session_id = payload["chat_session_id"]
    session = app.state.auth.session_store.get_session(chat_session_id)
    assert session is not None
    assert session.stream_active is True
    assert any(key.startswith("control_chat_turn:") for key in session.control_chat_tasks)

    session.expires_at = 0
    listed = client.get("/api/control/runs?kind=chat", headers=_headers(control))
    assert listed.status_code == 200, listed.text
    assert [run["run_id"] for run in listed.json()["runs"]] == []

    assert runner.disconnected.wait(timeout=1.0)
    assert runner.cancelled.wait(timeout=1.0)
    assert _wait_until(lambda: session.stream_active is False)
    assert _wait_until(lambda: not any(key.startswith("control_chat_turn:") for key in session.control_chat_tasks))
    assert app.state.auth.session_store.get_session(chat_session_id) is None
