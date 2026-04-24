import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.event_log import EventLog
from agent_gateway.providers.agent_sdk import AgentSDKConfig
from agent_gateway.runner import AgentRunner
from agent_gateway.sdk_runner import AgentSDKRunner
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.tool_dispatcher import ToolDispatcher


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  def get_server_for_tool(self, _name: str) -> str | None:
    return None


@dataclass
class _TestAppState:
  runtime: ChatRuntime | None = None
  runner: Any = None
  event_log: EventLog | None = None
  session: Any = None
  disconnect_hook_calls: int = 0


@pytest.fixture
def auth_config_model_free() -> dict[str, Any]:
  return {"api_key": "test-key", "max_tokens": 256}


@pytest.fixture
def make_test_app():
  def _make_test_app(
    *,
    provider: Any = None,
    dispatcher: ToolDispatcher | Callable[[EventLog, str, Any], ToolDispatcher] | None = None,
    runner_class: type[AgentRunner] | type[AgentSDKRunner] = AgentRunner,
    runner_kwargs: dict[str, Any] | None = None,
    tool_definitions: list[dict[str, Any]] | None = None,
    auth_config: dict[str, Any] | None = None,
    on_disconnect: Callable[[_TestAppState], Any] | None = None,
    attach_disconnect_handler: bool = True,
    tool_call_timeout: float | None = 120.0,
    per_turn_timeout: float | None = 300.0,
    stream_stall_timeout: float | None = 60.0,
    sdk_config: AgentSDKConfig | None = None,
    allow_model_free_auth_config: bool = False,
  ):
    state = _TestAppState()
    resolved_auth_config = dict(
      auth_config if auth_config is not None else {"api_key": "test-key", "model": "claude-sonnet-4-6", "max_tokens": 256}
    )
    resolved_tool_definitions = list(tool_definitions or [])
    fallback_model = "" if allow_model_free_auth_config else "claude-sonnet-4-6"
    resolved_sdk_config = sdk_config or AgentSDKConfig(
      api_key="test-key",
      model=str(resolved_auth_config.get("model") or fallback_model),
    )

    async def _build_chat_runtime(session, request, channel, auth_manager):
      _ = channel, auth_manager
      state.session = session

      def _build_runner(event_log: EventLog, session_id: str):
        state.event_log = event_log
        kwargs = dict(runner_kwargs or {})

        if issubclass(runner_class, AgentSDKRunner):
          runner = runner_class(
            event_log=event_log,
            session_id=session_id,
            sdk_config=resolved_sdk_config,
            system_prompt="test",
            on_usage=None,
            on_tool_result=None,
            on_tool_timing=None,
            **kwargs,
          )
        else:
          if provider is None:
            raise ValueError("provider is required when runner_class is AgentRunner")

          if dispatcher is None:
            dispatcher_obj = ToolDispatcher(
              mcp_client=_NullMcpClient(),
              local_tool_handlers={},
              event_log=event_log,
              session_id=session_id,
            )
          elif callable(dispatcher):
            dispatcher_obj = dispatcher(event_log, session_id, session)
          else:
            dispatcher_obj = dispatcher

          runner = runner_class(
            event_log=event_log,
            dispatcher=dispatcher_obj,
            session_id=session_id,
            provider=provider,
            auth_config=session.auth_config or resolved_auth_config,
            get_tool_definitions=lambda: list(resolved_tool_definitions),
            per_turn_timeout=per_turn_timeout,
            stream_stall_timeout=stream_stall_timeout,
            tool_call_timeout=tool_call_timeout,
            **kwargs,
          )

        state.runner = runner
        return runner

      async def _on_disconnect() -> None:
        state.disconnect_hook_calls += 1
        if on_disconnect is not None:
          result = on_disconnect(state)
          if inspect.isawaitable(result):
            await result
        runner = state.runner
        if runner is None:
          return
        await runner.on_disconnect()

      runtime_model_override = request.model or str(resolved_auth_config.get("model") or "")
      if not runtime_model_override and not allow_model_free_auth_config:
        runtime_model_override = "claude-sonnet-4-6"

      runtime_kwargs: dict[str, Any] = {
        "system_prompt": "test",
        "build_runner": _build_runner,
        "get_tool_definitions": lambda: list(resolved_tool_definitions),
        "provider": provider,
        "model_override": runtime_model_override or None,
      }
      if attach_disconnect_handler:
        runtime_kwargs["disconnect_handler"] = _on_disconnect

      runtime = ChatRuntime(
        **runtime_kwargs,
      )
      state.runtime = runtime
      return runtime

    app = create_gateway_app(
      GatewayServerConfig(
        jwt_secret="stream-lifecycle-test-secret-0123456789",
        auth_config=resolved_auth_config,
        valid_api_keys={"gateway-key"},
        allowed_models=set(),
        build_chat_runtime=_build_chat_runtime,
      )
    )
    app.state.test_state = state
    return app

  return _make_test_app
