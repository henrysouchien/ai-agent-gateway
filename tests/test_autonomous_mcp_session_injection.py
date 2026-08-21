import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.autonomous as autonomous  # noqa: E402
from agent_gateway import BoundCapabilityExecution, EventLog  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from tests.capability_execution_test_support import (  # noqa: E402
  stub_capability_execution_resolver,
)


def _run(coro):
  return asyncio.run(coro)


class _EmptyOperationCatalog:
  def resolve_operation(self, _selector):
    raise FileNotFoundError("no operations")

  def list_callable_operations_with_descriptions(self):
    return []


def _bound_execution() -> dict[str, Any]:
  resolver = stub_capability_execution_resolver(run_mode="autonomous")
  resolved = resolver.resolve("session.driver")
  execution = BoundCapabilityExecution(
    bind=resolved.bind,
    registry=resolved.registry,
    adapter=resolved.adapter,
    auth_config={
      **resolved.auth_config,
      "max_tokens": 16_000,
    },
  )
  return {
    "capability_execution": execution,
    "capability_execution_resolver": resolver,
    "session": GatewaySession(
      session_id="autonomous-mcp-test",
      api_key_hash="test",
      created_at=1,
      expires_at=2,
      user_id="alice",
      role="owner",
    ),
  }


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
  max_turns: int,
  timeout_seconds: float,
  initial_message: str,
  system_prompt: str | list[tuple[str, bool]],
) -> autonomous.RunOutput:
  _ = runner, event_log, max_turns, timeout_seconds, initial_message, system_prompt
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
      **_bound_execution(),
      mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
      trusted_mcp_allowed_servers={"browser"},
      mcp_session_inject_servers={"browser"},
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert captured["dispatcher_kwargs"]["mcp_session_inject_servers"] == {"browser"}


def test_run_autonomous_rejects_skills_dir_with_operation_catalog() -> None:
  async def _invoke() -> None:
    await autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      **_bound_execution(),
      skills_dir="skills",
      operation_catalog=_EmptyOperationCatalog(),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )

  try:
    _run(_invoke())
  except ValueError as exc:
    assert "either skills_dir or operation_catalog" in str(exc)
  else:
    raise AssertionError("expected mutually exclusive operation sources")


def test_run_autonomous_forwards_injected_operation_catalog(
  monkeypatch,
  tmp_path: Path,
) -> None:
  catalog = _EmptyOperationCatalog()
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
    captured.update(kwargs)
    return _fake_run_agent

  monkeypatch.setattr(autonomous, "ToolDispatcher", _StubDispatcher)
  monkeypatch.setattr(autonomous, "AgentRunner", _StubRunner)
  monkeypatch.setattr(autonomous, "run_session", _fake_run_session)
  monkeypatch.setattr(
    autonomous,
    "make_run_agent_handler",
    _fake_make_run_agent_handler,
  )

  _run(autonomous.run_autonomous(
    "You are helpful.",
    "Run the task.",
    **_bound_execution(),
    operation_catalog=catalog,
    session_log_base_dir=tmp_path,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  ))

  assert captured["operation_catalog"] is catalog
  assert captured["skill_loader"] is None


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
      **_bound_execution(),
      mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
      trusted_mcp_allowed_servers={"browser"},
      mcp_timeout_overrides={"browser": 90},
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
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
      **_bound_execution(),
      mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
      trusted_mcp_allowed_servers={"browser"},
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
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
      **_bound_execution(),
      mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
      trusted_mcp_allowed_servers={"browser"},
      skills_dir=skills_dir,
      mcp_session_inject_servers={"browser"},
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert captured["run_agent_kwargs"]["mcp_session_inject_servers"] == {"browser"}
