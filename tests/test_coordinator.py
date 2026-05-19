import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (
  AgentRunner,
  COORDINATOR_DEFAULT_PREAMBLE,
  CoordinatorConfig,
  EventLog,
  ModelInfo,
  ModelProvider,
  ProviderResolver,
  ResolvedProvider,
  ToolDispatcher,
  make_run_agent_handler,
)
from agent_gateway.providers import StreamEvent
from agent_gateway.sub_agent import _DEFAULT_EXCLUDED_TOOLS


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _PromptCaptureProvider(ModelProvider):
  name = "capture"

  def __init__(self) -> None:
    self.system_prompts: list[str | list[tuple[str, bool]] | None] = []

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

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
    _ = model, messages, tools, max_tokens, kwargs
    self.system_prompts.append(system_prompt)
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(type="message_start", input_tokens=1)
    yield StreamEvent(type="text_delta", text="ok")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "ok"})
    yield StreamEvent(type="usage_update", output_tokens=1)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _ResolvedProviderModel(ModelProvider):
  def __init__(self, name: str) -> None:
    self.name = name

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

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
    if False:
      yield None


class _StubRunner:
  def __init__(self) -> None:
    self._full_session_id = "session-coordinator"
    self.calls: list[dict[str, Any]] = []

  async def spawn_sub_agent(self, task: str, **kwargs: Any):
    self.calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None


async def _dummy_tool(_tool_input, **_kwargs):
  return {"ok": True}, None


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-coordinator",
  )


def test_coordinator_config_defaults() -> None:
  config = CoordinatorConfig()

  assert config.enabled is False
  assert config.auto_notify is True
  assert config.max_workers == 3


def test_coordinator_preamble_injected_into_string_system_prompt() -> None:
  provider = _PromptCaptureProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-coordinator",
    provider=provider,
    auth_config={"api_key": "k", "model": "stub-model"},
    coordinator=CoordinatorConfig(enabled=True),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}], system_prompt="Base prompt"))

  assert provider.system_prompts == [f"{COORDINATOR_DEFAULT_PREAMBLE}\n\nBase prompt"]


def test_coordinator_custom_preamble_overrides_default_for_prompt_blocks() -> None:
  provider = _PromptCaptureProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-coordinator",
    provider=provider,
    auth_config={"api_key": "k", "model": "stub-model"},
    coordinator=CoordinatorConfig(enabled=True, preamble="Custom coordinator preamble"),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(
    runner.run(
      messages=[{"role": "user", "content": "hello"}],
      system_prompt=[("Base prompt", True)],
    )
  )

  assert provider.system_prompts == [[("Custom coordinator preamble", False), ("Base prompt", True)]]


def test_coordinator_disabled_does_not_inject_preamble() -> None:
  provider = _PromptCaptureProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-coordinator",
    provider=provider,
    auth_config={"api_key": "k", "model": "stub-model"},
    coordinator=CoordinatorConfig(enabled=False, preamble="Should not appear"),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}], system_prompt="Base prompt"))

  assert provider.system_prompts == ["Base prompt"]


def test_make_run_agent_handler_merges_worker_excluded_tools() -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={
      "keep_tool": _dummy_tool,
      "drop_tool": _dummy_tool,
      "coord_tool": _dummy_tool,
    },
    excluded_tools={"drop_tool"},
    coordinator_config=CoordinatorConfig(enabled=True, worker_excluded_tools={"coord_tool"}),
  )

  result, error = _run(handler({"task": "Collect"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["excluded_tools"] == _DEFAULT_EXCLUDED_TOOLS | {"drop_tool", "coord_tool"}
  assert runner.calls[0]["dispatcher"]._local == {"keep_tool": _dummy_tool}


def test_coordinator_max_workers_overrides_max_background_tasks() -> None:
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-coordinator",
    provider=_PromptCaptureProvider(),
    auth_config={"api_key": "k", "model": "stub-model"},
    max_concurrent_sub_agents=1,
    coordinator=CoordinatorConfig(enabled=True, max_workers=5),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  assert runner._max_background_tasks == 5
  assert runner._task_registry._max_inflight == 5


def test_make_run_agent_handler_uses_default_worker_provider_fallback() -> None:
  runner = _StubRunner()
  resolved_provider = _ResolvedProviderModel("openai")

  def _resolve_provider(name: str) -> ResolvedProvider:
    assert name == "openai"
    return ResolvedProvider(
      provider=resolved_provider,
      auth_config={"api_key": "openai-key"},
      allowed_models={"gpt-4o-mini"},
      default_model="gpt-4o-mini",
    )

  resolver: ProviderResolver = _resolve_provider
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    provider_resolver=resolver,
    coordinator_config=CoordinatorConfig(enabled=True, default_worker_provider="openai"),
  )

  result, error = _run(handler({"task": "Collect"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["provider"] is resolved_provider
  assert runner.calls[0]["auth_config"] == {"api_key": "openai-key"}


def test_make_run_agent_handler_uses_default_worker_model_fallback() -> None:
  runner = _StubRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    default_model="claude-sonnet-4-6",
    allowed_models={"gpt-4o-mini"},
    coordinator_config=CoordinatorConfig(enabled=True, default_worker_model="gpt-4o-mini"),
  )

  result, error = _run(handler({"task": "Collect"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["model"] == "gpt-4o-mini"


def test_make_run_agent_handler_uses_coordinator_provider_resolver_fallback() -> None:
  runner = _StubRunner()
  resolved_provider = _ResolvedProviderModel("openai")

  def _resolve_provider(name: str) -> ResolvedProvider:
    assert name == "openai"
    return ResolvedProvider(
      provider=resolved_provider,
      auth_config={"api_key": "openai-key"},
      allowed_models={"gpt-4o-mini"},
      default_model="gpt-4o-mini",
    )

  resolver: ProviderResolver = _resolve_provider
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    coordinator_config=CoordinatorConfig(
      enabled=True,
      provider_resolver=resolver,
    ),
  )

  result, error = _run(handler({"task": "Collect", "provider": "openai"}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["provider"] is resolved_provider
  assert runner.calls[0]["auth_config"] == {"api_key": "openai-key"}
