from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from agent_gateway.event_log import EventLog
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


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


def _consume_chat_turn(client: TestClient, token: str, message: str) -> None:
  with client.stream(
    "POST",
    "/api/chat",
    headers={"Authorization": f"Bearer {token}"},
    json={"messages": [{"role": "user", "content": message}], "context": {"channel": "tui"}},
  ) as response:
    assert response.status_code == 200, response.text
    list(response.iter_lines())


def test_list_runs_returns_user_chat_sessions_and_excludes_control_sessions() -> None:
  app = _make_app()
  with TestClient(app) as client:
    alice_control = _control_session(client, "alice")
    alice_chat = _chat_session(client, "alice")
    bob_chat = _chat_session(client, "bob")

    alice_session = app.state.auth.session_store.get_session(alice_chat["session_id"])
    bob_session = app.state.auth.session_store.get_session(bob_chat["session_id"])
    assert alice_session is not None
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
        "type": "verdict_emitted",
        "skill_run_id": "skill-1",
        "verdict_token": "PRICE_TARGET_SET",
        "confidence": "HIGH",
        "one_line_summary": "summary",
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
      "cost_usd": None,
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

    for index in range(3):
      _consume_chat_turn(client, chat["session_token"], f"turn-{index}")

    session = app.state.auth.session_store.get_session(chat["session_id"])
    assert session is not None
    retained = session.event_history.snapshot()
    retained_text = [event.get("text") for event in retained if event.get("type") == "text_delta"]

    assert retained_text == ["echo:turn-0", "echo:turn-1", "echo:turn-2"]
    assert [event.get("type") for event in retained].count("stream_complete") == 3

    response = client.get(
      f"/api/control/runs/{chat['session_id']}/logs?tail=20",
      headers=_headers(control),
    )
    assert response.status_code == 200, response.text
    log_text = "\n".join(response.json()["log_lines"])
    assert "echo:turn-0" in log_text
    assert "echo:turn-1" in log_text
    assert "echo:turn-2" in log_text
