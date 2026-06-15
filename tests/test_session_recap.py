from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.easy import create_agent
from agent_gateway.event_log import EventLog
from agent_gateway.events import RecapFailure, SessionRecapEvent, event_from_dict
from agent_gateway.multi_user.billing import SessionUsageSummary
from agent_gateway.providers import ModelProvider, StreamEvent
from agent_gateway.providers.base import ModelInfo
from agent_gateway.runner import AgentRunner
from agent_gateway.sdk_runner import AgentSDKRunner
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.session import SessionStream
from agent_gateway.session_recap import compute_recap, emit_recap_then_terminal


def test_compute_recap_buckets_renderer_visible_events() -> None:
  event_log = EventLog(session_id="session-1")
  event_log.append(
    {
      "type": "artifact_ready",
      "artifact_id": "artifact-1",
      "skill": "earnings-scenarios",
      "contract_name": "EarningsScenarios",
      "ticker": "PCTY",
      "artifact_path": "artifacts/research/PCTY/earnings-scenarios.json",
      "ts": 2.0,
    }
  )
  event_log.append(
    {
      "type": "skill_result_captured",
      "skill_run_id": "run-1",
      "skill": "earnings-scenarios",
      "ticker": "PCTY",
      "verdict_echo": {
        "verdict_token": "SCENARIOS_BUILT",
        "confidence": "HIGH",
        "materiality_cushion": 3.15,
        "one_line_summary": "FY28 base case clears materiality",
      },
      "ts": 3.0,
    }
  )
  event_log.append(
    {
      "type": "tool_approval_decided",
      "tool_call_id": "toolu_1",
      "tool_name": "run_bash",
      "outcome": "approved",
      "decision_source": "user_approved",
      "allow_tool_type_applied": True,
      "ts": 4.0,
    }
  )
  event_log.append(
    {
      "type": "tool_call_start",
      "tool_call_id": "toolu_1",
      "tool_name": "mcp__excel__read_range",
      "server": "excel",
      "ts": 5.0,
    }
  )
  event_log.append(
    {
      "type": "tool_call_complete",
      "tool_call_id": "toolu_1",
      "tool_name": "mcp__excel__read_range",
      "server": "excel",
      "is_error": False,
      "ts": 6.0,
    }
  )
  event_log.append({"type": "artifact_failed", "error_detail": "artifact validation failed", "ts": 7.0})
  event_log.append({"type": "budget_exceeded", "message": "budget hit", "ts": 8.0})
  event_log.append({"type": "max_turns_reached", "turn_count": 5, "max_turns": 5, "ts": 9.0})

  recap = compute_recap(
    event_log,
    session_id="session-1",
    started_at=1.0,
    trigger="turn_end",
  )

  assert recap.session_id == "session-1"
  assert recap.seq_range == (1, 8)
  assert [artifact.artifact_id for artifact in recap.artifacts] == ["artifact-1"]
  assert [verdict.verdict_token for verdict in recap.verdicts] == ["SCENARIOS_BUILT"]
  assert [approval.decision_source for approval in recap.approvals] == ["user_approved"]
  assert recap.tool_calls_summary.total_calls == 1
  assert recap.tool_calls_summary.successes == 1
  assert recap.tool_calls_summary.errors == 0
  assert recap.tool_calls_summary.by_tool_name == {"mcp__excel__read_range": 1}
  assert recap.tool_calls_summary.by_server == {"excel": 1}
  assert [failure.failure_type for failure in recap.failures] == [
    "artifact_failed",
    "budget_exceeded",
    "max_turns_reached",
  ]


def test_emit_recap_then_terminal_orders_recap_before_stream_complete() -> None:
  event_log = EventLog(session_id="session-1")
  event_log.append(
    {
      "type": "artifact_ready",
      "artifact_id": "artifact-1",
      "skill": "html-artifact",
      "contract_name": "HtmlArtifact",
      "ticker": None,
      "artifact_path": "artifacts/research/html-artifact.json",
      "ts": 2.0,
    }
  )

  emit_recap_then_terminal(
    event_log,
    {"type": "stream_complete", "usage": {}},
    session_id="session-1",
    started_at=1.0,
  )

  entries = event_log.entries
  assert [entry.event["type"] for entry in entries] == ["artifact_ready", "session_recap", "stream_complete"]
  assert event_log.closed is True
  recap = event_from_dict(entries[1].event)
  assert isinstance(recap, SessionRecapEvent)
  assert recap.artifacts[0].artifact_id == "artifact-1"
  assert recap.seq_range == (1, 1)


def test_emit_recap_then_terminal_captures_terminal_error_as_pending_failure() -> None:
  event_log = EventLog(session_id="session-1")

  emit_recap_then_terminal(
    event_log,
    {"type": "error", "error": "provider failed", "ts": 4.0},
    session_id="session-1",
    started_at=1.0,
  )

  entries = event_log.entries
  assert [entry.event["type"] for entry in entries] == ["session_recap", "error"]
  recap = event_from_dict(entries[0].event)
  assert isinstance(recap, SessionRecapEvent)
  assert recap.failures[0].failure_type == "terminal_error"
  assert recap.failures[0].detail == "provider failed"
  assert recap.failures[0].emitted_at_seq == 2
  assert recap.seq_range == (1, 2)


def test_emit_recap_then_terminal_can_skip_recap_for_sub_agent_logs() -> None:
  event_log = EventLog(session_id="sub0:session-1")

  emit_recap_then_terminal(
    event_log,
    {"type": "stream_complete", "usage": {}},
    session_id="sub0:session-1",
    started_at=1.0,
    emit_recap=False,
  )

  assert [entry.event["type"] for entry in event_log.entries] == ["stream_complete"]


def test_emit_recap_then_terminal_respects_explicit_pending_failure_seq() -> None:
  event_log = EventLog(session_id="session-1")
  failure = RecapFailure(
    failure_type="budget_exceeded",
    detail="budget hit",
    emitted_at_seq=42,
    ts=5.0,
  )

  emit_recap_then_terminal(
    event_log,
    {"type": "stream_complete", "usage": {}},
    session_id="session-1",
    started_at=1.0,
    pending_failure=failure,
  )

  recap = event_from_dict(event_log.entries[0].event)
  assert isinstance(recap, SessionRecapEvent)
  assert recap.failures[0].emitted_at_seq == 42
  assert recap.seq_range == (1, 42)


def test_agent_runner_append_wraps_terminal_events_with_recap() -> None:
  event_log = EventLog(session_id="session-1")
  runner = AgentRunner.__new__(AgentRunner)
  runner._log = event_log
  runner._full_session_id = "session-1"
  runner._session_started_at = 1.0
  runner._emit_session_recap = True

  AgentRunner._append(runner, {"type": "stream_complete", "usage": {}})

  assert [entry.event["type"] for entry in event_log.entries] == ["session_recap", "stream_complete"]


def test_agent_runner_append_skips_recap_when_disabled() -> None:
  event_log = EventLog(session_id="sub0:session-1")
  runner = AgentRunner.__new__(AgentRunner)
  runner._log = event_log
  runner._full_session_id = "sub0:session-1"
  runner._session_started_at = 1.0
  runner._emit_session_recap = False

  AgentRunner._append(runner, {"type": "stream_complete", "usage": {}})

  assert [entry.event["type"] for entry in event_log.entries] == ["stream_complete"]


def test_agent_runner_error_event_calls_terminal_aware_hook() -> None:
  class _FakeAgentSessionLog:
    def __init__(self) -> None:
      self.events: list[dict[str, Any]] = []

    async def append(self, event: dict[str, Any]):
      self.events.append(dict(event))
      return type("Entry", (), {"seq": len(self.events)})()

  event_log = EventLog(session_id="session-1")
  agent_session_log = _FakeAgentSessionLog()
  runner = AgentRunner.__new__(AgentRunner)
  runner._log = event_log
  runner._full_session_id = "session-1"
  runner._session_started_at = 1.0
  runner._emit_session_recap = False
  runner._agent_session_log = agent_session_log
  runner._runner_id = "runner-1"
  runner._role = "writer"
  runner._sub_agent_id = None
  runner._last_durable_seq = 0
  captured: dict[str, Any] = {}

  async def _hook(active_event_log: EventLog, terminal_event: dict[str, Any]) -> None:
    captured["entries_before_terminal"] = [entry.event["type"] for entry in active_event_log.entries]
    captured["terminal_event"] = dict(terminal_event)
    event = {"type": "skill_result_captured", "outcome": "error"}
    active_event_log.append(event)
    await AgentRunner._append_durable_event(runner, event)

  runner._on_before_stream_complete = _hook

  asyncio.run(AgentRunner._emit_error_event(runner, "failed"))

  assert captured["entries_before_terminal"] == []
  assert captured["terminal_event"] == {"type": "error", "error": "failed"}
  assert [entry.event["type"] for entry in event_log.entries] == ["skill_result_captured", "error"]
  assert [event["type"] for event in agent_session_log.events] == ["skill_result_captured", "error"]
  assert agent_session_log.events[0]["runner_id"] == "runner-1"
  assert agent_session_log.events[1]["runner_id"] == "runner-1"


def test_sdk_runner_append_wraps_terminal_events_with_recap() -> None:
  event_log = EventLog(session_id="session-1")
  runner = AgentSDKRunner.__new__(AgentSDKRunner)
  runner._log = event_log
  runner._session_id = "session-1"
  runner._session_started_at = 1.0
  runner._emit_session_recap = True

  AgentSDKRunner._append(runner, {"type": "error", "error": "sdk failed"})

  assert [entry.event["type"] for entry in event_log.entries] == ["session_recap", "error"]


def _run(coro):
  return asyncio.run(coro)


def _make_recap_app(transcript_dir: Path | None = None):
  async def _build_chat_runtime(session, request, channel, auth_manager):
    _ = session, request, channel, auth_manager
    return ChatRuntime(system_prompt="test", build_runner=lambda _event_log, _session_id, _started_at=None: None)

  return create_gateway_app(
    GatewayServerConfig(
      jwt_secret="session-recap-test-secret-0123456789",
      auth_config={"api_key": "test-key", "model": "test-model", "max_tokens": 256},
      valid_api_keys={"gateway-key"},
      allowed_models=set(),
      build_chat_runtime=_build_chat_runtime,
      transcript_dir=transcript_dir,
    )
  )


async def _init_session(client: httpx.AsyncClient, *, user_id: str = "alice") -> dict[str, Any]:
  response = await client.post("/api/chat/init", json={"api_key": "gateway-key", "user_id": user_id})
  assert response.status_code == 200, response.text
  return response.json()


def _headers(session_info: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_info['session_token']}"}


def _attach_active_turn(session, *, closed: bool = False) -> SessionStream:
  event_log = EventLog(session_id=session.session_id)
  event_log.append(
    {
      "type": "artifact_ready",
      "artifact_id": "artifact-1",
      "skill": "html-artifact",
      "contract_name": "HtmlArtifact",
      "ticker": "PCTY",
      "artifact_path": "artifacts/research/PCTY/html-artifact.json",
      "ts": 2.0,
    }
  )
  if closed:
    event_log.append({"type": "stream_complete", "usage": {}, "ts": 3.0})
  active_turn = SessionStream(event_log=event_log, runner_task=None)
  session.active_turn = active_turn
  return active_turn


def _transcript_events(transcript_dir: Path, session_id: str) -> list[dict[str, Any]]:
  path = transcript_dir / f"{session_id}.jsonl"
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _usage_summary(session_id: str, *, request_id: str = "req-1") -> SessionUsageSummary:
  return SessionUsageSummary(
    user_id="alice",
    session_id=session_id,
    request_id=request_id,
    input_tokens=11,
    output_tokens=7,
    cache_read_tokens=0,
    cache_creation_tokens=0,
    cost=0.0123,
    turns=1,
    channel="excel",
    started_at=1.0,
    ended_at=2.0,
  )


def test_post_chat_recap_appends_to_open_event_log(tmp_path: Path) -> None:
  async def case() -> None:
    app = _make_recap_app(tmp_path)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None
      active_turn = _attach_active_turn(session)
      session.cached_usage = _usage_summary(session.session_id)

      response = await client.post(
        "/api/chat/recap",
        headers=_headers(session_info),
        json={"session_id": session.session_id, "scope": "active_turn"},
      )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["type"] == "session_recap"
    assert payload["trigger"] == "explicit"
    assert payload["usage"]["session_id"] == session.session_id
    assert [entry.event["type"] for entry in active_turn.event_log.entries] == [
      "artifact_ready",
      "session_recap",
    ]

  _run(case())


def test_post_chat_recap_closed_log_writes_transcript_only(tmp_path: Path) -> None:
  async def case() -> None:
    app = _make_recap_app(tmp_path)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None
      active_turn = _attach_active_turn(session, closed=True)

      response = await client.post(
        "/api/chat/recap",
        headers=_headers(session_info),
        json={"session_id": session.session_id, "scope": "active_turn"},
      )

    assert response.status_code == 200, response.text
    assert [entry.event["type"] for entry in active_turn.event_log.entries] == ["artifact_ready", "stream_complete"]
    transcript_events = _transcript_events(tmp_path, session.session_id)
    assert [event["type"] for event in transcript_events] == ["session_recap"]
    assert transcript_events[0]["trigger"] == "explicit"

  _run(case())


def test_post_chat_recap_auth_scope_and_missing_active_turn_errors(tmp_path: Path) -> None:
  async def case() -> None:
    app = _make_recap_app(tmp_path)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      alice = await _init_session(client, user_id="alice")
      bob = await _init_session(client, user_id="bob")

      missing_auth = await client.post("/api/chat/recap", json={"session_id": alice["session_id"]})
      assert missing_auth.status_code == 401

      wrong_session = await client.post(
        "/api/chat/recap",
        headers=_headers(bob),
        json={"session_id": alice["session_id"], "scope": "active_turn"},
      )
      assert wrong_session.status_code == 403

      no_active = await client.post(
        "/api/chat/recap",
        headers=_headers(alice),
        json={"session_id": alice["session_id"], "scope": "active_turn"},
      )
      assert no_active.status_code == 404

      session = app.state.auth.session_store.get_session(alice["session_id"])
      assert session is not None
      _attach_active_turn(session)

      cumulative = await client.post(
        "/api/chat/recap",
        headers=_headers(alice),
        json={"session_id": alice["session_id"], "scope": "session_cumulative"},
      )
      assert cumulative.status_code == 501

      invalid_scope = await client.post(
        "/api/chat/recap",
        headers=_headers(alice),
        json={"session_id": alice["session_id"], "scope": "unknown"},
      )
      assert invalid_scope.status_code == 400

  _run(case())


def test_session_gc_writes_recap_transcript(tmp_path: Path) -> None:
  async def case() -> None:
    app = _make_recap_app(tmp_path)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None
      _attach_active_turn(session)
      session.cached_usage = _usage_summary(session.session_id)

      await app.state.auth.session_store.expire_session_async(session.session_id)

    transcript_events = _transcript_events(tmp_path, session.session_id)
    assert [event["type"] for event in transcript_events] == ["session_recap"]
    assert transcript_events[0]["trigger"] == "session_gc"
    assert transcript_events[0]["usage"]["session_id"] == session.session_id
    assert session.active_turn is None

  _run(case())


def test_session_gc_without_active_turn_is_noop(tmp_path: Path) -> None:
  async def case() -> None:
    app = _make_recap_app(tmp_path)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None

      await app.state.auth.session_store.expire_session_async(session.session_id)

    assert _transcript_events(tmp_path, session.session_id) == []

  _run(case())


class _UsageClient:
  pass


class _UsageProvider(ModelProvider):
  name = "usage-test"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return _UsageClient()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=3)
    yield StreamEvent(type="text_delta", text="done")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "done"})
    yield StreamEvent(type="usage_update", output_tokens=5)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def test_create_agent_usage_summary_populates_cached_usage_without_observer() -> None:
  async def case() -> None:
    app = create_agent(
      "test",
      provider=_UsageProvider(),
      model="usage-model",
      valid_api_keys={"gateway-key"},
      jwt_secret="session-recap-create-agent-secret-0123456789",
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
      session_info = await _init_session(client)
      async with client.stream(
        "POST",
        "/api/chat",
        headers=_headers(session_info),
        json={"messages": [{"role": "user", "content": "hello"}], "context": {}},
      ) as response:
        assert response.status_code == 200, response.text
        async for _line in response.aiter_lines():
          pass

      session = app.state.auth.session_store.get_session(session_info["session_id"])
      assert session is not None
      assert session.cached_usage is not None
      assert session.cached_usage.session_id == session.session_id

      recap_response = await client.post(
        "/api/chat/recap",
        headers=_headers(session_info),
        json={"session_id": session.session_id, "scope": "active_turn"},
      )

    assert recap_response.status_code == 200, recap_response.text
    assert recap_response.json()["usage"]["session_id"] == session_info["session_id"]

  _run(case())
