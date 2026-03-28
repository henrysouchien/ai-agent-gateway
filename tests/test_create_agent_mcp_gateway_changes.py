import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.easy as easy_module
import agent_gateway.sub_agent as sub_agent_module
from agent_gateway import EventLog, McpClientManager, create_agent
from agent_gateway.server import ChatRequest


def _run(coro):
  return asyncio.run(coro)


def _build_runtime(app):
  session = app.state.auth.session_store.create_session(api_key_hash="hash")
  runtime = _run(
    app.state.gateway_config.build_chat_runtime(
      session=session,
      request=ChatRequest(messages=[{"role": "user", "content": "hello"}], context={}),
      channel=None,
      auth_manager=app.state.auth,
    )
  )
  return session, runtime


def _write_skill(skills_dir: Path, name: str, body: str) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


def test_create_agent_forwards_new_mcp_options_to_manager_and_dispatcher() -> None:
  app = create_agent(
    "test",
    mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
    mcp_timeout_overrides={"browser": 90},
    mcp_default_tool_timeout=45,
    mcp_session_inject_servers={"browser"},
    mcp_strip_input_fields={"_session_id"},
  )

  mcp_client = app.state.gateway_config.mcp_client
  assert isinstance(mcp_client, McpClientManager)
  assert mcp_client._timeout_overrides == {"browser": 90}
  assert mcp_client._default_tool_timeout == 45
  assert mcp_client._strip_input_fields == {"_session_id"}

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)

  assert runner._dispatcher._mcp_session_inject_servers == {"browser"}


def test_create_agent_forwards_mcp_session_inject_servers_to_run_agent(
  monkeypatch,
  tmp_path: Path,
) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "browser-worker", "Use the browser tools.")
  captured: dict[str, Any] = {}

  async def _fake_run_agent(_tool_input, **_kwargs):
    return {"response": "ok"}, None

  def _fake_make_run_agent_handler(*args, **kwargs):
    _ = args
    captured["kwargs"] = kwargs
    return _fake_run_agent

  monkeypatch.setattr(sub_agent_module, "make_run_agent_handler", _fake_make_run_agent_handler)

  app = create_agent(
    "test",
    skills_dir=skills_dir,
    mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
    mcp_session_inject_servers={"browser"},
  )

  _build_runtime(app)

  assert captured["kwargs"]["mcp_session_inject_servers"] == {"browser"}


def test_create_agent_expiry_cleanup_chains_code_execution_and_mcp_session_close(
  monkeypatch,
  tmp_path: Path,
) -> None:
  class _FakeMcpClientManager:
    def __init__(self, **kwargs: Any) -> None:
      self.kwargs = kwargs
      self.calls: list[tuple[str, dict[str, Any]]] = []
      self.path_checks: list[bool] = []
      self.cleanup_path: Path | None = None

    async def startup(self) -> None:
      return None

    async def shutdown(self) -> None:
      return None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
      return []

    def is_mcp_tool(self, name: str) -> bool:
      return name == "browser_close_session"

    async def call_tool(self, name: str, tool_input: dict[str, Any]):
      if self.cleanup_path is not None:
        self.path_checks.append(self.cleanup_path.exists())
      self.calls.append((name, dict(tool_input)))
      return {"ok": True}, None

  monkeypatch.setattr(easy_module, "McpClientManager", _FakeMcpClientManager)

  app = create_agent(
    "test",
    code_execution=True,
    mcp_servers={"browser": {"command": "python3", "args": ["run_server.py"]}},
    mcp_session_inject_servers={"browser"},
  )

  mcp_client = app.state.gateway_config.mcp_client
  session = app.state.auth.session_store.create_session(api_key_hash="hash")
  work_dir = tmp_path / "expiry-cleanup"
  work_dir.mkdir()
  session.code_execution_work_dir = str(work_dir)
  mcp_client.cleanup_path = work_dir

  _run(app.state.auth.session_store.expire_session_async(session.session_id))

  assert not work_dir.exists()
  assert mcp_client.path_checks == [False]
  assert mcp_client.calls == [("browser_close_session", {"_session_id": session.session_id})]
