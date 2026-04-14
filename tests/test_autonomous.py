import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.autonomous as autonomous
from agent_gateway import EventLog
from agent_gateway.providers import ModelInfo, ModelProvider


def _run(coro):
  return asyncio.run(coro)


class _MockResponse:
  def raise_for_status(self) -> None:
    return None


class _RecordingAsyncClient:
  def __init__(self, calls: list[dict[str, Any]], *args: Any, **kwargs: Any) -> None:
    _ = args, kwargs
    self._calls = calls

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb) -> None:
    _ = exc_type, exc, tb

  async def post(self, url: str, json: dict[str, Any] | None = None, headers: dict[str, str] | None = None):
    self._calls.append({"url": url, "json": json, "headers": headers})
    return _MockResponse()


class _StubProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if False:
      yield


def test_collect_run_output() -> None:
  event_log = EventLog()
  event_log.append({"type": "text_delta", "text": "stale"})
  event_log.append({"type": "tool_call_start", "tool_name": "old_tool"})
  event_log.append({"type": "stream_retry"})
  event_log.append({"type": "text_delta", "text": "Hello"})
  event_log.append({"type": "text_delta", "text": " world"})
  event_log.append({"type": "tool_call_start", "tool_name": "fresh_tool"})
  event_log.append({"type": "budget_exceeded"})
  event_log.append({"type": "max_turns_reached"})
  event_log.append({"type": "stream_complete", "usage": {"input_tokens": 10, "output_tokens": 20}})

  output = autonomous.collect_run_output(event_log, timed_out=False)

  assert output.response == "Hello world"
  assert output.tools_used == ["fresh_tool"]
  assert output.usage == {"input_tokens": 10, "output_tokens": 20}
  assert output.error is None
  assert output.timed_out is False
  assert output.budget_exceeded is True
  assert output.max_turns_reached is True


def test_run_session_timeout() -> None:
  class _SlowRunner:
    async def run(self, **kwargs: Any) -> None:
      _ = kwargs
      await asyncio.sleep(0.2)

  output = _run(
    autonomous.run_session(
      _SlowRunner(),  # type: ignore[arg-type]
      EventLog(),
      model="claude-sonnet-4-6",
      max_turns=3,
      timeout_seconds=0.01,
      initial_message="hello",
      system_prompt="You are helpful.",
    )
  )

  assert output.timed_out is True
  assert output.response == ""
  assert output.error is None


@pytest.mark.parametrize("timeout_seconds", [0, None, -1])
def test_run_session_no_wall_clock(
  monkeypatch: pytest.MonkeyPatch,
  timeout_seconds: float | None,
) -> None:
  event_log = EventLog()

  class _Runner:
    async def run(self, **kwargs: Any) -> None:
      _ = kwargs
      event_log.append({"type": "text_delta", "text": "completed"})
      event_log.append({"type": "stream_complete", "usage": {}})

  async def _unexpected_wait_for(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("asyncio.wait_for should not wrap non-positive timeouts")

  monkeypatch.setattr(autonomous.asyncio, "wait_for", _unexpected_wait_for)

  output = _run(
    autonomous.run_session(
      _Runner(),  # type: ignore[arg-type]
      event_log,
      model="claude-sonnet-4-6",
      max_turns=3,
      timeout_seconds=timeout_seconds,
      initial_message="hello",
      system_prompt="You are helpful.",
    )
  )

  assert output.timed_out is False
  assert output.error is None
  assert output.response == "completed"


@pytest.mark.parametrize(
  ("output", "expected"),
  [
    (autonomous.RunOutput("ok", [], {}, None, False), 0),
    (autonomous.RunOutput("ok", [], {}, "boom", False), 1),
    (autonomous.RunOutput("ok", [], {}, None, False, budget_exceeded=True), 2),
    (autonomous.RunOutput("ok", [], {}, None, False, max_turns_reached=True), 3),
    (autonomous.RunOutput("ok", [], {}, None, True), 124),
  ],
)
def test_run_output_exit_code(output: autonomous.RunOutput, expected: int) -> None:
  assert autonomous.run_output_exit_code(output) == expected


@pytest.mark.parametrize(
  ("output", "expected"),
  [
    (autonomous.RunOutput("ok", [], {}, None, False), "success"),
    (autonomous.RunOutput("ok", [], {}, None, True), "timeout"),
    (autonomous.RunOutput("ok", [], {}, "boom", False), "error"),
    (autonomous.RunOutput("ok", [], {}, None, False, budget_exceeded=True), "budget_exceeded"),
    (autonomous.RunOutput("ok", [], {}, None, False, max_turns_reached=True), "max_turns"),
  ],
)
def test_run_output_outcome(output: autonomous.RunOutput, expected: str) -> None:
  assert autonomous.run_output_outcome(output) == expected


def test_extract_state_update() -> None:
  text = """
Summary text.

## STATE_UPDATE_JSON
```json
{"alerts": ["Check filings"], "next_session": ["Review guidance"]}
```
"""

  assert autonomous.extract_state_update(text) == {
    "alerts": ["Check filings"],
    "next_session": ["Review guidance"],
  }


def test_load_save_state(tmp_path: Path) -> None:
  payload = {"skill": "monitor", "count": 2}

  autonomous.save_state(tmp_path, payload)

  assert autonomous.load_state(tmp_path) == payload


def test_build_state_payload() -> None:
  output = autonomous.RunOutput(
    response="Long summary text",
    tools_used=["file_read", "file_read", "web_fetch"],
    usage={"input_tokens": 5},
    error=None,
    timed_out=False,
  )

  payload = autonomous.build_state_payload(
    previous_state={"existing": "value", "alerts": ["keep"], "next_session": [None, "next"]},
    model_state={"alerts": ["fresh"], "new_key": 1},
    run_output=output,
    model_name="claude-opus-4-6",
    briefing_file="briefings/daily.md",
    connected_servers={"prices", "filings"},
    active_servers={"prices"},
  )

  assert payload["existing"] == "value"
  assert payload["new_key"] == 1
  assert payload["model"] == "claude-opus-4-6"
  assert payload["briefing_file"] == "briefings/daily.md"
  assert payload["connected_servers"] == ["filings", "prices"]
  assert payload["active_servers"] == ["prices"]
  assert payload["tools_used"] == ["file_read", "web_fetch"]
  assert payload["usage"] == {"input_tokens": 5}
  assert payload["last_summary"] == "Long summary text"
  assert payload["alerts"] == ["fresh"]
  assert payload["next_session"] == ["next"]
  assert "last_run" in payload


def test_format_run_summary() -> None:
  output = autonomous.RunOutput(
    response="This is a compact summary.",
    tools_used=["tool_a", "tool_b"],
    usage={"input_tokens": 11, "output_tokens": 22, "estimated_cost": 0.12},
    error=None,
    timed_out=False,
  )

  message = autonomous.format_run_summary(
    output,
    label="Nightly analyst run",
    state={"briefing_file": "notes/briefing.md", "ideas": 3},
    format_state_fn=lambda state: f"Ideas: {state['ideas']}",
  )

  assert "Nightly analyst run" in message
  assert "Status: completed" in message
  assert "Briefing: notes/briefing.md" in message
  assert "Ideas: 3" in message
  assert "Summary:" in message


def test_deliver_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
  calls: list[dict[str, Any]] = []
  monkeypatch.setattr(autonomous.httpx, "AsyncClient", lambda *args, **kwargs: _RecordingAsyncClient(calls, *args, **kwargs))

  output = autonomous.RunOutput("Summary", ["tool"], {}, None, False)
  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(
        telegram_bot_token="bot-token",
        telegram_chat_id="chat-id",
        telegram_label="Run label",
      ),
      output,
      {"briefing_file": "notes/briefing.md"},
    )
  )

  assert len(calls) == 1
  assert calls[0]["url"] == "https://api.telegram.org/botbot-token/sendMessage"
  assert "Run label" in str(calls[0]["json"]["text"])


def test_deliver_telegram_briefing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  calls: list[dict[str, Any]] = []
  monkeypatch.setattr(autonomous.httpx, "AsyncClient", lambda *args, **kwargs: _RecordingAsyncClient(calls, *args, **kwargs))

  briefing_path = tmp_path / "briefing.md"
  content = "A" * 4100
  briefing_path.write_text(content, encoding="utf-8")
  output = autonomous.RunOutput("Summary", [], {}, None, False)

  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(
        telegram_bot_token="bot-token",
        telegram_chat_id="chat-id",
        briefing_file=briefing_path,
      ),
      output,
      None,
    )
  )

  assert len(calls) == 1 + len(autonomous.split_messages(content))


def test_deliver_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
  calls: list[dict[str, Any]] = []
  monkeypatch.setattr(autonomous.httpx, "AsyncClient", lambda *args, **kwargs: _RecordingAsyncClient(calls, *args, **kwargs))

  output = autonomous.RunOutput("Summary", ["tool"], {"input_tokens": 1}, None, False)
  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(webhook_url="https://example.com/hook"),
      output,
      {"briefing_file": "notes/briefing.md"},
    )
  )

  assert len(calls) == 1
  assert calls[0]["url"] == "https://example.com/hook"
  assert calls[0]["json"]["outcome"] == "success"
  assert calls[0]["json"]["run_output"]["response"] == "Summary"
  assert calls[0]["json"]["state"]["briefing_file"] == "notes/briefing.md"


def test_deliver_callback() -> None:
  seen: dict[str, Any] = {}

  async def _on_complete(run_output: autonomous.RunOutput, state: dict[str, Any] | None) -> None:
    seen["response"] = run_output.response
    seen["state"] = dict(state or {})

  _run(
    autonomous.deliver(
      autonomous.DeliveryConfig(on_complete=_on_complete),
      autonomous.RunOutput("Summary", [], {}, None, False),
      {"briefing_file": "notes/briefing.md"},
    )
  )

  assert seen == {
    "response": "Summary",
    "state": {"briefing_file": "notes/briefing.md"},
  }


def test_run_autonomous_simple(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  captured: dict[str, Any] = {}
  delivered: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      captured["dispatcher_args"] = args
      captured["dispatcher_kwargs"] = kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      captured["runner_args"] = args
      captured["runner_kwargs"] = kwargs

  async def _fake_run_session(
    runner: Any,
    event_log: EventLog,
    *,
    model: str,
    max_turns: int,
    timeout_seconds: float,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    captured["run_session_runner"] = runner
    captured["run_session_event_log"] = event_log
    captured["run_session_kwargs"] = {
      "model": model,
      "max_turns": max_turns,
      "timeout_seconds": timeout_seconds,
      "initial_message": initial_message,
      "system_prompt": system_prompt,
    }
    return autonomous.RunOutput(
      response=(
        "Completed.\n\n"
        "## STATE_UPDATE_JSON\n"
        "```json\n"
        '{"alerts":["Watch earnings"],"next_session":["Review transcript"]}\n'
        "```"
      ),
      tools_used=["file_read"],
      usage={"input_tokens": 1, "output_tokens": 2},
      error=None,
      timed_out=False,
    )

  async def _on_complete(run_output: autonomous.RunOutput, state: dict[str, Any] | None) -> None:
    delivered["response"] = run_output.response
    delivered["state"] = dict(state or {})

  state_dir = tmp_path / "state"
  autonomous.save_state(state_dir, {"existing": "value"})
  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  provider = _StubProvider()
  result = _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      provider=provider,
      model="stub-model",
      tool_handlers={"local_tool": lambda *_args, **_kwargs: None},
      tool_definitions=[{"name": "local_tool", "description": "Local tool", "input_schema": {"type": "object"}}],
      max_turns=7,
      timeout_seconds=12,
      state_dir=state_dir,
      delivery=autonomous.DeliveryConfig(
        on_complete=_on_complete,
        briefing_file="notes/briefing.md",
      ),
      session_id="session-123",
    )
  )

  assert result.response.startswith("Completed.")
  assert captured["runner_kwargs"]["provider"] is provider
  assert captured["runner_kwargs"]["auth_config"]["model"] == "stub-model"
  assert captured["runner_kwargs"]["session_id"] == "session-123"
  assert captured["run_session_kwargs"]["initial_message"] == "Run the task."
  assert captured["run_session_kwargs"]["system_prompt"] == "You are helpful."

  saved_state = autonomous.load_state(state_dir)
  assert saved_state["existing"] == "value"
  assert saved_state["alerts"] == ["Watch earnings"]
  assert saved_state["next_session"] == ["Review transcript"]
  assert saved_state["briefing_file"] == "notes/briefing.md"
  assert delivered["state"]["existing"] == "value"


def test_run_autonomous_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
  captured: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args
      captured["runner_kwargs"] = kwargs

  async def _fake_run_session(
    runner: Any,
    event_log: EventLog,
    *,
    model: str,
    max_turns: int,
    timeout_seconds: float | None,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = runner, event_log, model, initial_message, system_prompt
    captured["run_session_kwargs"] = {
      "max_turns": max_turns,
      "timeout_seconds": timeout_seconds,
    }
    return autonomous.RunOutput(
      response="Completed.",
      tools_used=[],
      usage={},
      error=None,
      timed_out=False,
    )

  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      provider=_StubProvider(),
      model="stub-model",
    )
  )

  assert captured["run_session_kwargs"]["max_turns"] == 80
  assert captured["run_session_kwargs"]["timeout_seconds"] is None
  assert captured["runner_kwargs"]["per_turn_timeout"] == 300.0


def test_run_autonomous_forwards_outputs_dir(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  async def _fake_run_session(
    runner: Any,
    event_log: EventLog,
    *,
    model: str,
    max_turns: int,
    timeout_seconds: float,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = runner, event_log, model, max_turns, timeout_seconds, initial_message, system_prompt
    return autonomous.RunOutput(
      response="Completed.",
      tools_used=[],
      usage={},
      error=None,
      timed_out=False,
    )

  async def _fake_run_agent(_tool_input, **_kwargs):
    return {"response": "ok"}, None

  def _fake_make_run_agent_handler(*args, **kwargs):
    _ = args
    captured["kwargs"] = kwargs
    return _fake_run_agent

  skills_dir = tmp_path / "skills"
  outputs_dir = tmp_path / "outputs"
  skills_dir.mkdir(parents=True, exist_ok=True)

  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)
  monkeypatch.setattr(autonomous, "make_run_agent_handler", _fake_make_run_agent_handler)

  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      provider=_StubProvider(),
      model="stub-model",
      skills_dir=skills_dir,
      outputs_dir=outputs_dir,
    )
  )

  assert captured["kwargs"]["outputs_dir"] == outputs_dir


def test_run_autonomous_skills_dir_registers_send_message_tool_and_builtin_name(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  captured: dict[str, Any] = {}
  skills_dir = tmp_path / "skills"
  skills_dir.mkdir(parents=True, exist_ok=True)

  class _StubMcpClientManager:
    def __init__(
      self,
      *,
      inline_servers: dict[str, dict[str, Any]] | None = None,
      config_path: str | Path | None = None,
      builtin_tool_names: set[str] | None = None,
      timeout_overrides: dict[str, int] | None = None,
    ) -> None:
      _ = inline_servers, config_path, timeout_overrides
      self._builtin_tool_names = set(builtin_tool_names or set())

    async def startup(self) -> None:
      return None

    async def shutdown(self) -> None:
      return None

    def get_server_names(self) -> list[str]:
      return []

    def get_tool_definitions(self) -> list[dict[str, Any]]:
      return []

  async def _fake_run_session(
    runner: Any,
    event_log: EventLog,
    *,
    model: str,
    max_turns: int,
    timeout_seconds: float,
    initial_message: str,
    system_prompt: str | list[tuple[str, bool]],
  ) -> autonomous.RunOutput:
    _ = event_log, model, max_turns, timeout_seconds, initial_message, system_prompt
    captured["local_handlers"] = set(runner._dispatcher._local)
    captured["tool_defs"] = {tool["name"] for tool in runner._get_tool_definitions()}
    captured["builtin_tool_names"] = set(runner._mcp_client._builtin_tool_names)
    return autonomous.RunOutput(
      response="Completed.",
      tools_used=[],
      usage={},
      error=None,
      timed_out=False,
    )

  monkeypatch.setattr(autonomous, "McpClientManager", _StubMcpClientManager)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      provider=_StubProvider(),
      model="stub-model",
      skills_dir=skills_dir,
      mcp_servers={"filesystem": {"command": "npx", "args": ["-y", "server"]}},
    )
  )

  assert captured["local_handlers"] >= {"run_agent", "get_background_result", "send_message"}
  assert captured["tool_defs"] >= {"run_agent", "get_background_result", "send_message"}
  assert captured["builtin_tool_names"] >= {"run_agent", "get_background_result", "send_message"}


def test_extract_state_empty_text() -> None:
  assert autonomous.extract_state_update("") == {}


def test_extract_state_non_dict_json() -> None:
  text = """
## STATE_UPDATE_JSON
```json
[1, 2, 3]
```
"""

  assert autonomous.extract_state_update(text) == {}


def test_extract_state_malformed_json() -> None:
  text = """
## STATE_UPDATE_JSON
```json
{"alerts": [}
```
"""

  assert autonomous.extract_state_update(text) == {}


def test_split_messages_empty() -> None:
  assert autonomous.split_messages("") == []


def test_split_messages_force_split_long_line() -> None:
  assert autonomous.split_messages("ABCDEFGHIJ", max_len=4) == ["ABCD", "EFGH", "IJ"]
