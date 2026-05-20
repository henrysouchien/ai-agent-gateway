import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from api.credentials import get_default_model as get_canonical_default_model
import agent_gateway.mcp_client as mcp_client_module
import agent_gateway.sub_agent as sub_agent_module
from agent_gateway import EventLog, McpClientManager, ToolResultContext, create_agent
from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway._provider_utils import _resolve_provider
from agent_gateway.providers import AnthropicProvider, CodexProvider, OpenAIProvider
from agent_gateway.server import ChatRequest

DEFAULT_ANTHROPIC_MODEL = get_canonical_default_model("anthropic")


def _run(coro):
  return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_credential_env(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_MODE", raising=False)
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _build_runtime(app, *, request_model: str | None = None):
  session = app.state.auth.session_store.create_session(api_key_hash="hash", user_id="alice")
  runtime = _run(
    app.state.gateway_config.build_chat_runtime(
      session=session,
      request=ChatRequest(messages=[{"role": "user", "content": "hello"}], context={}, model=request_model),
      channel=None,
      auth_manager=app.state.auth,
    )
  )
  return session, runtime


def _build_runtime_with_request(app, *, request: ChatRequest, channel: str | None):
  session = app.state.auth.session_store.create_session(api_key_hash="hash", user_id="alice")
  runtime = _run(
    app.state.gateway_config.build_chat_runtime(
      session=session,
      request=request,
      channel=channel,
      auth_manager=app.state.auth,
    )
  )
  return session, runtime


def _collect_sse_events(response) -> list[dict]:
  events = []
  for line in response.iter_lines():
    if line.startswith("data: "):
      events.append(json.loads(line[6:]))
  return events


def _write_skill(skills_dir: Path, name: str, body: str) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


def test_create_agent_exposes_default_routes_and_open_defaults() -> None:
  app = create_agent("test")

  route_paths = {route.path for route in app.routes}
  assert "/api/chat/init" in route_paths
  assert "/api/chat" in route_paths
  assert "/api/health" in route_paths

  config = app.state.gateway_config
  assert config.auth_config["auth_mode"] == "api"
  assert config.auth_config["api_key"] == ""
  assert config.auth_config["auth_token"] == ""
  assert config.auth_config["model"] == DEFAULT_ANTHROPIC_MODEL
  assert config.cors_origins == ["*"]
  assert config.allowed_models == set()
  assert len(app.state.auth._secret) == 64


def test_create_agent_uses_explicit_api_key() -> None:
  app = create_agent("test", api_key="sk-123")
  assert app.state.gateway_config.auth_config == {
    "auth_mode": "api",
    "api_key": "sk-123",
    "auth_token": "",
    "model": DEFAULT_ANTHROPIC_MODEL,
    "max_tokens": 16000,
  }


@pytest.mark.parametrize("value", ["", "   "])
def test_create_agent_blank_api_key_falls_back_to_env(value: str, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  app = create_agent("test", api_key=value)
  assert app.state.gateway_config.auth_config["auth_mode"] == "api"
  assert app.state.gateway_config.auth_config["api_key"] == "env-key"


def test_create_agent_uses_explicit_auth_token() -> None:
  app = create_agent("test", auth_token="tok-123")
  assert app.state.gateway_config.auth_config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok-123",
    "model": DEFAULT_ANTHROPIC_MODEL,
    "max_tokens": 16000,
  }


def test_create_agent_with_credentials_resolver_skips_env_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
  async def _resolver(_api_key: str, _init_request):
    return ResolverResult(
      user_id="alice",
      channel="excel",
      auth_config=AuthConfig.from_dict(
        {
          "provider": "anthropic",
          "billing_mode": "byok",
          "api_key": "resolver-key",
          "model": DEFAULT_ANTHROPIC_MODEL,
          "max_tokens": 16000,
        }
      ),
      risk_user_id=101,
      role="owner",
    )

  monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
  app = create_agent("test", credentials_resolver=_resolver)

  assert app.state.gateway_config.credentials_resolver is _resolver
  assert app.state.gateway_config.auth_config == {
    "auth_mode": "api",
    "api_key": "",
    "auth_token": "",
    "model": DEFAULT_ANTHROPIC_MODEL,
    "max_tokens": 16000,
  }


def test_create_agent_tolerates_model_free_resolved_auth_config(monkeypatch: pytest.MonkeyPatch) -> None:
  def _fake_resolve_provider(*args, **kwargs):
    _ = args, kwargs
    return AnthropicProvider(), "anthropic", {
      "auth_mode": "api",
      "api_key": "resolver-key",
      "max_tokens": 16000,
    }

  monkeypatch.setattr("agent_gateway.easy._resolve_provider", _fake_resolve_provider)

  app = create_agent("test")
  _session, runtime = _build_runtime(app)

  assert runtime.model_override == DEFAULT_ANTHROPIC_MODEL


def test_create_agent_threads_request_metadata_to_runner() -> None:
  app = create_agent("test")
  request = ChatRequest(
    messages=[{"role": "user", "content": "hello"}],
    request_id="req-123",
    context={"channel": "web"},
  )
  session = app.state.auth.session_store.create_session(api_key_hash="hash", user_id="alice")
  session.user_id = "alice"
  session.auth_config = {"api_key": "k", "model": DEFAULT_ANTHROPIC_MODEL, "billing_mode": "metered"}
  runtime = _run(
    app.state.gateway_config.build_chat_runtime(
      session=session,
      request=request,
      channel="web",
      auth_manager=app.state.auth,
    )
  )

  runner = runtime.build_runner(EventLog(), session.session_id)

  assert runner._request_id == "req-123"
  assert runner._usage_user_id == "alice"
  assert runner._billing_mode == "metered"
  assert runner._channel == "web"


def test_create_agent_openai_provider_string_uses_openai_defaults() -> None:
  app = create_agent("test", provider="openai")

  config = app.state.gateway_config
  assert isinstance(config.default_provider, OpenAIProvider)
  assert config.auth_config == {
    "model": "gpt-4o",
    "max_tokens": 16000,
  }
  assert config.allowed_models == set()


def test_create_agent_codex_provider_string_uses_codex_defaults() -> None:
  app = create_agent("test", provider="codex")

  config = app.state.gateway_config
  assert isinstance(config.default_provider, CodexProvider)
  assert config.auth_config == {
    "model": "gpt-5.4",
    "max_tokens": 16000,
  }
  assert config.allowed_models == set()


def test_create_agent_accepts_provider_instance_when_model_is_explicit() -> None:
  provider = OpenAIProvider()
  app = create_agent("test", provider=provider, model="gpt-4o-mini")

  config = app.state.gateway_config
  assert config.default_provider is provider
  assert config.auth_config == {
    "model": "gpt-4o-mini",
    "max_tokens": 16000,
  }
  assert config.allowed_models == set()


def test_create_agent_provider_instance_requires_model() -> None:
  with pytest.raises(ValueError, match="model is required when passing a ModelProvider instance"):
    create_agent("test", provider=OpenAIProvider())


def test_create_agent_unknown_provider_raises_value_error() -> None:
  with pytest.raises(ValueError, match="Unknown provider: unknown"):
    create_agent("test", provider="unknown")


def test_create_agent_invalid_provider_type_raises_type_error() -> None:
  with pytest.raises(TypeError, match="provider must be a string"):
    create_agent("test", provider=123)  # type: ignore[arg-type]


def test_create_agent_openai_does_not_eagerly_read_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("OPENAI_API_KEY", "env-openai-key")
  app = create_agent("test", provider="openai")
  assert app.state.gateway_config.auth_config == {
    "model": "gpt-4o",
    "max_tokens": 16000,
  }


def test_create_agent_openai_uses_explicit_api_key_and_provider_config() -> None:
  app = create_agent(
    "test",
    provider="openai",
    api_key="sk-openai",
    auth_token="ignored",
    provider_config={
      "base_url": "https://custom.example/v1",
      "compat": {"streaming": True},
    },
  )
  assert app.state.gateway_config.auth_config == {
    "auth_mode": "api",
    "api_key": "sk-openai",
    "auth_token": "",
    "model": "gpt-4o",
    "max_tokens": 16000,
    "base_url": "https://custom.example/v1",
    "compat": {"streaming": True},
  }


def test_create_agent_openai_uses_explicit_auth_token() -> None:
  app = create_agent(
    "test",
    provider="openai",
    auth_token="oat-xxx",
  )

  assert app.state.gateway_config.auth_config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "oat-xxx",
    "model": "gpt-4o",
    "max_tokens": 16000,
  }


def test_create_agent_codex_uses_explicit_auth_token() -> None:
  app = create_agent(
    "test",
    provider="codex",
    auth_token="oat-codex",
  )

  assert app.state.gateway_config.auth_config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "oat-codex",
    "model": "gpt-5.4",
    "max_tokens": 16000,
  }


def test_resolve_provider_openai_auth_config_with_oauth() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "openai",
    None,
    None,
    None,
    None,
    auth_config={"auth_mode": "oauth", "auth_token": "tok"},
  )

  assert config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
    "model": "gpt-4o",
    "max_tokens": 16000,
  }


def test_resolve_provider_openai_auth_config_infers_mode() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "openai",
    None,
    None,
    None,
    None,
    auth_config={"auth_token": "tok"},
  )

  assert config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
    "model": "gpt-4o",
    "max_tokens": 16000,
  }


def test_resolve_provider_openai_auth_config_plain_passthrough() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "openai",
    None,
    None,
    None,
    None,
    auth_config={"api_key": "sk-x"},
  )

  assert config == {
    "auth_mode": "api",
    "api_key": "sk-x",
    "auth_token": "",
    "model": "gpt-4o",
    "max_tokens": 16000,
  }


def test_resolve_provider_codex_auth_config_with_oauth() -> None:
  _provider, _provider_name, config = _resolve_provider(
    "codex",
    None,
    None,
    None,
    None,
    auth_config={"auth_mode": "oauth", "auth_token": "tok"},
  )

  assert config == {
    "auth_mode": "oauth",
    "api_key": "",
    "auth_token": "tok",
    "model": "gpt-5.4",
    "max_tokens": 16000,
  }


def test_create_agent_blank_auth_token_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "env-token")
  app = create_agent("test", auth_token="   ")
  assert app.state.gateway_config.auth_config["auth_mode"] == "oauth"
  assert app.state.gateway_config.auth_config["auth_token"] == "env-token"


def test_create_agent_no_credentials_constructs_and_streams_stub_response() -> None:
  app = create_agent("test")

  with TestClient(app) as client:
    init_response = client.post("/api/chat/init", json={"api_key": "any-key", "user_id": "test-user"})
    assert init_response.status_code == 200
    token = init_response.json()["session_token"]

    with client.stream(
      "POST",
      "/api/chat",
      headers={"Authorization": f"Bearer {token}"},
      json={
        "messages": [{"role": "user", "content": "Hello gateway"}],
        "context": {},
      },
    ) as response:
      assert response.status_code == 200
      events = _collect_sse_events(response)

  text = "".join(event.get("text", "") for event in events if event.get("type") == "text_delta")
  event_types = {event.get("type") for event in events}
  assert "stream_complete" in event_types
  assert "Stub response (no Anthropic credential configured)." in text
  assert "You asked: Hello gateway" in text


def test_create_agent_valid_api_keys_and_jwt_secret() -> None:
  app = create_agent("test", valid_api_keys={"k1"}, jwt_secret="s")
  assert app.state.auth._secret == "s"

  with TestClient(app) as client:
    assert client.post("/api/chat/init", json={"api_key": "bad"}).status_code == 401
    assert client.post("/api/chat/init", json={"api_key": "k1", "user_id": "test-user"}).status_code == 200


def test_create_agent_inline_mcp_uses_inline_only_config_and_builtin_filtering() -> None:
  async def _tool(_tool_input, **_kwargs):
    return {"ok": True}, None

  app = create_agent(
    "test",
    mcp_servers={"filesystem": {"command": "npx", "args": ["-y", "server"]}},
    tool_handlers={"local_tool": _tool},
    code_execution=True,
  )

  mcp_client = app.state.gateway_config.mcp_client
  assert isinstance(mcp_client, McpClientManager)
  assert mcp_client._config_path is None
  assert mcp_client._inline_servers == {"filesystem": {"command": "npx", "args": ["-y", "server"]}}
  assert mcp_client._builtin_tool_names >= {"local_tool", "code_execute", "code_execute_status"}


def test_create_agent_skills_dir_registers_run_agent_handler_tool_def_and_builtin_name(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")

  app = create_agent(
    "test",
    skills_dir=skills_dir,
    mcp_servers={"filesystem": {"command": "npx", "args": ["-y", "server"]}},
  )

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)
  tool_defs = runtime.get_tool_definitions()

  assert "run_agent" in runner._dispatcher._local
  assert "get_background_result" in runner._dispatcher._local
  assert "send_message" in runner._dispatcher._local
  assert {tool["name"] for tool in tool_defs} >= {"run_agent", "get_background_result", "send_message"}
  assert app.state.gateway_config.mcp_client._builtin_tool_names >= {"run_agent", "get_background_result", "send_message"}


def test_create_agent_forwards_outputs_dir(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  skills_dir = tmp_path / "skills"
  outputs_dir = tmp_path / "outputs"
  skills_dir.mkdir(parents=True, exist_ok=True)
  captured: dict[str, object] = {}

  async def _fake_run_agent(_tool_input, **_kwargs):
    return {"response": "ok"}, None

  def _fake_make_run_agent_handler(*args, **kwargs):
    _ = args
    captured["kwargs"] = kwargs
    return _fake_run_agent

  monkeypatch.setattr(sub_agent_module, "make_run_agent_handler", _fake_make_run_agent_handler)

  app = create_agent("test", skills_dir=skills_dir, outputs_dir=outputs_dir)

  _build_runtime(app)

  assert captured["kwargs"]["outputs_dir"] == outputs_dir


def test_create_agent_forwards_needs_approval_and_cache_denylist() -> None:
  async def _tool(_tool_input, **_kwargs):
    return {"ok": True}, None

  app = create_agent(
    "test",
    tool_handlers={"execute_trade": _tool},
    needs_approval=lambda name, _tool_input=None, _qualifier="": name == "execute_trade",
    session_cache_denied_tools=frozenset({"execute_trade"}),
  )

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)
  dispatcher = runner._dispatcher

  assert dispatcher._needs_approval("execute_trade", {}, "") is True
  assert dispatcher._session_cache_denied == frozenset({"execute_trade"})


def test_create_agent_skills_dir_does_not_duplicate_user_run_agent_tool_definition(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")
  tool_def = {
    "name": "run_agent",
    "description": "custom",
    "input_schema": {"type": "object", "properties": {}},
  }

  app = create_agent("test", skills_dir=skills_dir, tool_definitions=[tool_def])

  _session, runtime = _build_runtime(app)

  tool_defs = runtime.get_tool_definitions()
  assert [tool for tool in tool_defs if tool["name"] == "run_agent"] == [tool_def]
  assert any(tool["name"] == "get_background_result" for tool in tool_defs)
  assert any(tool["name"] == "send_message" for tool in tool_defs)


def test_create_agent_skills_dir_respects_custom_run_agent_handler_override(tmp_path: Path) -> None:
  async def _custom_run_agent(_tool_input, **_kwargs):
    return {"override": True}, None

  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")
  app = create_agent("test", skills_dir=skills_dir, tool_handlers={"run_agent": _custom_run_agent})

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)

  assert runner._dispatcher._local["run_agent"] is _custom_run_agent
  assert runtime.get_tool_definitions() == []


def test_create_agent_skills_dir_propagates_model_to_sub_agent_default(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")
  app = create_agent("test", model="claude-opus-4-6", skills_dir=skills_dir)

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)
  captured: dict[str, object] = {}

  async def _fake_spawn_sub_agent(task: str, **kwargs):
    captured["task"] = task
    captured.update(kwargs)
    return {"response": "ok"}, None

  runner.spawn_sub_agent = _fake_spawn_sub_agent  # type: ignore[method-assign]

  result, error = _run(runner._dispatcher._local["run_agent"]({"task": "Collect"}))

  assert error is None
  assert result == {"response": "ok"}
  assert captured["task"] == "Collect"
  assert captured["model"] == "claude-opus-4-6"


def test_create_agent_request_model_override_does_not_change_sub_agent_default(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "deep-research", "Research deeply.")
  app = create_agent("test", model="claude-haiku-4-5", skills_dir=skills_dir)

  session, runtime = _build_runtime(app, request_model="claude-opus-4-6")
  assert runtime.model_override == "claude-opus-4-6"

  runner = runtime.build_runner(EventLog(), session.session_id)
  captured: dict[str, object] = {}

  async def _fake_spawn_sub_agent(task: str, **kwargs):
    captured["task"] = task
    captured.update(kwargs)
    return {"response": "ok"}, None

  runner.spawn_sub_agent = _fake_spawn_sub_agent  # type: ignore[method-assign]

  result, error = _run(runner._dispatcher._local["run_agent"]({"task": "Collect"}))

  assert error is None
  assert result == {"response": "ok"}
  assert captured["task"] == "Collect"
  assert captured["model"] == "claude-haiku-4-5"


def test_create_agent_mcp_config_path_expands_and_merges_inline_servers(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  monkeypatch.setenv("HOME", str(tmp_path))
  config_path = tmp_path / ".claude.json"
  config_path.write_text(
    json.dumps(
      {
        "mcpServers": {
          "from_config": {"command": "cfg"},
          "shared": {"command": "cfg-shared"},
        }
      }
    ),
    encoding="utf-8",
  )

  app = create_agent(
    "test",
    mcp_servers={
      "shared": {"command": "inline"},
      "from_inline": {"command": "inline-only"},
    },
    mcp_config_path="~/.claude.json",
  )
  mcp_client = app.state.gateway_config.mcp_client
  assert isinstance(mcp_client, McpClientManager)
  assert mcp_client._config_path == config_path

  seen: dict[str, dict] = {}

  async def _fake_connect_or_warn(self, name: str, config: dict):
    seen[name] = dict(config)
    return None

  monkeypatch.setattr(mcp_client_module, "MCP_IMPORT_ERROR", None)
  monkeypatch.setattr(McpClientManager, "_connect_or_warn", _fake_connect_or_warn)

  _run(mcp_client.startup())

  assert set(seen) == {"from_config", "shared", "from_inline"}
  assert seen["shared"]["command"] == "inline"


def test_create_agent_registers_local_tool_handlers_and_definitions() -> None:
  async def _tool(_tool_input, **_kwargs):
    return {"ok": True}, None

  tool_def = {
    "name": "t",
    "description": "test tool",
    "input_schema": {"type": "object", "properties": {}},
  }
  app = create_agent("test", tool_handlers={"t": _tool}, tool_definitions=[tool_def])

  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)

  assert runner._dispatcher._local["t"] is _tool
  assert runtime.get_tool_definitions() == [tool_def]
  assert runner._dispatcher._request_approval is not None


def test_create_agent_code_execution_wires_hooks_approval_and_expiry_cleanup(tmp_path: Path) -> None:
  async def _user_on_tool_result(_ctx):
    return [{"type": "text", "text": "extra"}]

  app = create_agent("test", code_execution=True, on_tool_result=_user_on_tool_result)
  session, runtime = _build_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)

  assert {"code_execute", "code_execute_status"} <= set(runner._dispatcher._local)
  assert {tool["name"] for tool in runtime.get_tool_definitions()} == {
    "code_execute",
    "code_execute_status",
  }
  assert runner._dispatcher._request_approval is not None
  assert runner._dispatcher._approved_tool_types is session.approved_tool_types
  assert runner._dispatcher._should_request_approval("code_execute", {"host": "subprocess"}, "subprocess") is True

  session.approved_tool_types.add("code_execute:subprocess")
  assert runner._dispatcher._should_request_approval("code_execute", {"host": "subprocess"}, "subprocess") is False

  ctx = ToolResultContext(
    tool_name="code_execute",
    tool_input={},
    result={"images": [{"filename": "plot.png", "data_base64": "abc"}]},
    error=None,
    duration_ms=5,
    tool_call_id="tool_1",
    session_id=session.session_id,
    server=None,
    result_entry={"content": json.dumps({"images": [{"filename": "plot.png", "data_base64": "abc"}]})},
  )
  extra_blocks = _run(runner._on_tool_result(ctx))
  payload = json.loads(ctx.result_entry["content"])

  assert extra_blocks == [{"type": "text", "text": "extra"}]
  assert payload["images"][0]["data_base64"] == "[image: plot.png]"

  work_dir = tmp_path / "expiry-cleanup"
  work_dir.mkdir()
  session.code_execution_work_dir = str(work_dir)
  _run(app.state.auth.session_store.expire_session_async(session.session_id))
  assert not work_dir.exists()


def test_create_agent_code_execution_cleans_up_active_sessions_on_shutdown(tmp_path: Path) -> None:
  app = create_agent("test", code_execution=True)
  work_dir = tmp_path / "shutdown-cleanup"

  with TestClient(app):
    session = app.state.auth.session_store.create_session(api_key_hash="hash", user_id="alice")
    work_dir.mkdir()
    session.code_execution_work_dir = str(work_dir)

  assert not work_dir.exists()


def test_create_agent_model_and_cors_configuration() -> None:
  app = create_agent("test", model="claude-haiku-4-5", cors_origins=["https://x.com"])
  assert app.state.gateway_config.allowed_models == set()
  assert app.state.gateway_config.cors_origins == ["https://x.com"]

  empty_cors_app = create_agent("test", cors_origins=[])
  assert empty_cors_app.state.gateway_config.cors_origins == []
