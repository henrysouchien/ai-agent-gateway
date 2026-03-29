import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.autonomous as autonomous
from agent_gateway import EventLog
from agent_gateway.providers import ModelInfo, ModelProvider


def _run(coro):
  return asyncio.run(coro)


class _StubProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
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


class _FakeMcpClientManager:
  instances: list["_FakeMcpClientManager"] = []

  def __init__(self, **kwargs: Any) -> None:
    self.kwargs = kwargs
    self.inline_servers = dict(kwargs.get("inline_servers") or {})
    type(self).instances.append(self)

  async def startup(self) -> None:
    return None

  async def shutdown(self) -> None:
    return None

  def get_server_names(self) -> set[str]:
    return set(self.inline_servers.keys())

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
  _ = runner, event_log, model, max_turns, timeout_seconds, initial_message, system_prompt
  return autonomous.RunOutput(
    response="Completed.",
    tools_used=[],
    usage={},
    error=None,
    timed_out=False,
  )


def test_run_autonomous_forwards_mcp_session_inject_servers_to_dispatcher(
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args
      captured["dispatcher_kwargs"] = kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  _FakeMcpClientManager.instances.clear()
  monkeypatch.setattr(autonomous, "McpClientManager", _FakeMcpClientManager)
  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      provider=_StubProvider(),
      model="stub-model",
      mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
      mcp_session_inject_servers={"browser"},
    )
  )

  assert captured["dispatcher_kwargs"]["mcp_session_inject_servers"] == {"browser"}


def test_run_autonomous_forwards_mcp_timeout_overrides_to_manager(
  monkeypatch,
) -> None:
  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  _FakeMcpClientManager.instances.clear()
  monkeypatch.setattr(autonomous, "McpClientManager", _FakeMcpClientManager)
  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      provider=_StubProvider(),
      model="stub-model",
      mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
      mcp_timeout_overrides={"browser": 90},
    )
  )

  assert _FakeMcpClientManager.instances[0].kwargs["timeout_overrides"] == {"browser": 90}


def test_run_autonomous_defaults_to_no_injection_and_no_timeout_overrides(
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args
      captured["dispatcher_kwargs"] = kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  _FakeMcpClientManager.instances.clear()
  monkeypatch.setattr(autonomous, "McpClientManager", _FakeMcpClientManager)
  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)

  _run(
    autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      provider=_StubProvider(),
      model="stub-model",
      mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
    )
  )

  assert captured["dispatcher_kwargs"]["mcp_session_inject_servers"] is None
  assert _FakeMcpClientManager.instances[0].kwargs["timeout_overrides"] is None


def test_run_autonomous_forwards_mcp_session_inject_servers_to_sub_agents(
  monkeypatch,
  tmp_path: Path,
) -> None:
  captured: dict[str, Any] = {}

  class _StubDispatcher:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  class _StubRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      _ = args, kwargs

  async def _fake_run_agent(_tool_input, **_kwargs):
    return {"response": "ok"}, None

  def _fake_make_run_agent_handler(*args, **kwargs):
    _ = args
    captured["run_agent_kwargs"] = kwargs
    return _fake_run_agent

  skills_dir = tmp_path / "skills"
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / "browser-worker.md").write_text("Use the browser tools.", encoding="utf-8")

  _FakeMcpClientManager.instances.clear()
  monkeypatch.setattr(autonomous, "McpClientManager", _FakeMcpClientManager)
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
      mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
      skills_dir=skills_dir,
      mcp_session_inject_servers={"browser"},
    )
  )

  assert captured["run_agent_kwargs"]["mcp_session_inject_servers"] == {"browser"}
