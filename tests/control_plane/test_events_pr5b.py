from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from starlette.requests import Request

from agent_gateway.control_plane.events import _shielded_aclose
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app


API_KEY = "events-pr5b-key"
HMAC_KEY = "events-pr5b-hmac"


class _CompletedProcess:
  returncode = 0

  async def wait(self) -> int:
    return 0

  def terminate(self) -> None:
    pass

  def kill(self) -> None:
    pass


class _NoopRunner:
  async def run(
    self,
    *,
    messages: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_override: str | None = None,
    max_turns: int | None = None,
  ) -> None:
    _ = messages, system_prompt, model_override, max_turns


def _make_app(monkeypatch, tmp_path: Path, events: list[dict[str, Any]]):
  monkeypatch.setenv("AGENT_API_USER_CLAIM_HMAC_KEY", HMAC_KEY)
  monkeypatch.setenv("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", str(tmp_path / "logs"))

  async def _build_chat_runtime(*, session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(system_prompt="system", build_runner=lambda *_args: _NoopRunner())

  async def fake_exec(*args, **kwargs):
    _ = args
    events_path = Path(kwargs["env"]["AGENT_AUTONOMOUS_EVENTS_PATH"])
    with events_path.open("a", encoding="utf-8") as handle:
      for event in events:
        handle.write(json.dumps(event) + "\n")
    await asyncio.sleep(0)
    return _CompletedProcess()

  from agent_gateway import autonomous_runner

  monkeypatch.setattr(autonomous_runner.asyncio, "create_subprocess_exec", fake_exec)
  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="events-pr5b-test-secret-0123456789",
      valid_api_keys={API_KEY},
      auth_config={"model": "test-model"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
    )
  )


def _control_session(client: TestClient, user_id: str, *, channel: str = "tui") -> dict[str, Any]:
  response = client.post(
    "/api/control/session",
    json={"api_key": API_KEY, "user_id": user_id, "context": {"channel": channel}},
  )
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def _decode_sse_chunk(chunk: bytes) -> dict[str, Any]:
  line = chunk.decode("utf-8").strip()
  assert line.startswith("data: ")
  return json.loads(line.removeprefix("data: "))


def test_control_events_cancelled_aclose_does_not_leave_pending_close_task() -> None:
  class _SlowClose:
    def __init__(self) -> None:
      self.started = asyncio.Event()
      self.cancelled = asyncio.Event()

    async def aclose(self) -> None:
      self.started.set()
      try:
        await asyncio.Event().wait()
      except asyncio.CancelledError:
        self.cancelled.set()
        raise

  async def case() -> None:
    iterator = _SlowClose()
    before = set(asyncio.all_tasks())
    close_task = asyncio.create_task(_shielded_aclose(iterator))
    await asyncio.wait_for(iterator.started.wait(), timeout=0.5)

    close_task.cancel()
    try:
      await close_task
    except asyncio.CancelledError:
      pass

    leaked = [
      task
      for task in asyncio.all_tasks() - before
      if task is not asyncio.current_task() and not task.done()
    ]
    try:
      assert leaked == []
      assert iterator.cancelled.is_set()
    finally:
      for task in leaked:
        task.cancel()
      if leaked:
        await asyncio.gather(*leaked, return_exceptions=True)

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


def test_control_events_replays_autonomous_jsonl_bridge_events(monkeypatch, tmp_path: Path) -> None:
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
    time.sleep(0.3)

    received = asyncio.run(
      _collect_control_events(app, alice["session_token"], control_run_id=run_id, count=6)
    )

    received_types = [event["type"] for event in received]
    for expected in ["skill_run_started", "skill_result_captured", "artifact_ready", "artifact_failed"]:
      assert expected in received_types
    assert "run_state_changed" in received_types
    assert all(event["control_run_id"] == run_id for event in received)


def test_control_events_fast_run_race_replays_late_subscriber(monkeypatch, tmp_path: Path) -> None:
  fast_events = [{"type": "fast_event", "seq": index} for index in range(3)]
  app = _make_app(monkeypatch, tmp_path, fast_events)

  with TestClient(app) as client:
    alice = _control_session(client, "alice")
    start = client.post(
      "/api/control/runs",
      headers=_headers(alice),
      json={"kind": "autonomous", "profile": "analyst", "mode": "task", "task": "quick"},
    )
    assert start.status_code == 200, start.text
    run_id = start.json()["run_id"]
    time.sleep(0.5)

    received = asyncio.run(
      _collect_control_events(app, alice["session_token"], control_run_id=run_id, count=5)
    )

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


def test_autonomous_runner_event_log_writes_events_jsonl(tmp_path: Path) -> None:
  api_dir = Path(__file__).resolve().parents[4] / "api"
  if str(api_dir) not in sys.path:
    sys.path.insert(0, str(api_dir))

  from agent.autonomous.runner import _event_log_with_jsonl

  events_path = tmp_path / "events.jsonl"
  event_log = _event_log_with_jsonl(events_path)
  event_log.append({"type": "skill_run_started", "skill_run_id": "skill-1"})
  event_log.append({"type": "stream_complete", "usage": {}})

  lines = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
  assert lines == [
    {"type": "skill_run_started", "skill_run_id": "skill-1"},
    {"type": "stream_complete", "usage": {}},
  ]
