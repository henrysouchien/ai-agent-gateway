import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog, McpClientManager, ModelInfo, ModelProvider, ToolDispatcher


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}


class _CaptureProvider(ModelProvider):
  name = "capture"

  def __init__(self) -> None:
    self.captured_config: dict[str, Any] | None = None

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    self.captured_config = dict(config)
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    raise ValueError(f"stop after create_client: {model}")

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
    if False:
      yield None


def test_runner_preserves_extra_auth_config_keys_for_provider_create_client() -> None:
  provider = _CaptureProvider()
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=EventLog(),
    session_id="sess_passthrough",
  )
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=dispatcher,
    session_id="sess_passthrough",
    provider=provider,
    auth_config={
      "api_key": "k",
      "base_url": "https://custom.example/v1",
      "compat": {"streaming": True},
    },
    mcp_client=McpClientManager(config_path=None),
    get_tool_definitions=lambda: [],
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert provider.captured_config == {
    "auth_mode": "api",
    "api_key": "k",
    "auth_token": "",
    "model": "claude-sonnet-4-6",
    "max_tokens": 16000,
    "thinking": True,
    "base_url": "https://custom.example/v1",
    "compat": {"streaming": True},
  }
