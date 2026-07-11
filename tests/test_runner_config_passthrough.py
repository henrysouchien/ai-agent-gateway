import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog, McpClientManager, ModelInfo, ModelProvider, ToolDispatcher  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner import STREAM_STALL_TIMEOUT, STREAM_THINKING_STALL_TIMEOUT  # noqa: E402


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


def _make_runner(*, stream_stall_timeout: float | None = None) -> AgentRunner:
  event_log = EventLog()
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log,
    session_id="sess_watchdog",
  )
  return AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id="sess_watchdog",
    provider=_CaptureProvider(),
    auth_config={"api_key": "k"},
    get_tool_definitions=lambda: [],
    stream_stall_timeout=stream_stall_timeout,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_default_stream_stall_timeout_extends_for_thinking_requests() -> None:
  runner = _make_runner()

  assert runner._effective_stream_stall_timeout(
    config={"thinking": True},
    model_info=ModelInfo(id="claude-opus-4-7", provider="anthropic", supports_thinking=True),
    max_tokens=4096,
  ) == STREAM_THINKING_STALL_TIMEOUT


def test_default_stream_stall_timeout_stays_short_without_thinking() -> None:
  runner = _make_runner()

  assert runner._effective_stream_stall_timeout(
    config={"thinking": False},
    model_info=ModelInfo(id="claude-opus-4-7", provider="anthropic", supports_thinking=True),
    max_tokens=4096,
  ) == STREAM_STALL_TIMEOUT
  assert runner._effective_stream_stall_timeout(
    config={"thinking": True},
    model_info=ModelInfo(id="gpt-test", provider="openai", supports_thinking=False),
    max_tokens=4096,
  ) == STREAM_STALL_TIMEOUT
  assert runner._effective_stream_stall_timeout(
    config={"thinking": True},
    model_info=ModelInfo(id="claude-opus-4-7", provider="anthropic", supports_thinking=True),
    max_tokens=1024,
  ) == STREAM_STALL_TIMEOUT


def test_explicit_stream_stall_timeout_overrides_thinking_default() -> None:
  runner = _make_runner(stream_stall_timeout=42)

  assert runner._effective_stream_stall_timeout(
    config={"thinking": True},
    model_info=ModelInfo(id="claude-opus-4-7", provider="anthropic", supports_thinking=True),
    max_tokens=4096,
  ) == 42


def test_observed_thinking_history_extends_later_turns_when_request_flag_is_false() -> None:
  runner = _make_runner()
  model_info = ModelInfo(id="claude-sonnet-5", provider="anthropic", supports_thinking=True)
  current_messages = [
    {
      "role": "assistant",
      "provider": "anthropic",
      "model": "claude-sonnet-5",
      "content": [{"type": "thinking", "thinking": "", "signature": "sig"}],
    }
  ]

  assert runner._effective_stream_stall_timeout(
    config={"thinking": False},
    model_info=model_info,
    max_tokens=16384,
    current_messages=current_messages,
  ) == STREAM_THINKING_STALL_TIMEOUT


def test_runner_stall_timeout_delegate_uses_runner_module_constants(monkeypatch) -> None:
  runner = _make_runner()
  monkeypatch.setattr(gateway_runner, "STREAM_STALL_TIMEOUT", 7)
  monkeypatch.setattr(gateway_runner, "STREAM_THINKING_STALL_TIMEOUT", 11)

  assert runner._effective_stream_stall_timeout(
    config={"thinking": True},
    model_info=ModelInfo(id="claude-opus-4-7", provider="anthropic", supports_thinking=True),
    max_tokens=4096,
  ) == 11
  assert runner._effective_stream_stall_timeout(
    config={"thinking": False},
    model_info=ModelInfo(id="claude-opus-4-7", provider="anthropic", supports_thinking=True),
    max_tokens=4096,
  ) == 7


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
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
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
