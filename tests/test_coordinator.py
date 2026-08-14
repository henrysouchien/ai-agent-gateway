# ruff: noqa: E402

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
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
  ToolDispatcher,
  make_run_agent_handler,
)
from agent_gateway.capability_binding import (
  CapabilityBind,
)
from agent_gateway.model_registry import INITIAL_MODEL_REGISTRY
from agent_gateway.providers import StreamEvent
from agent_gateway.sub_agent import _DEFAULT_EXCLUDED_TOOLS
from tests.capability_execution_test_support import (
  stub_runner_capability_execution,
)
from tests.tool_catalog_test_support import OWNER_GATEWAY_SESSION


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


class _StubRunner:
  def __init__(self) -> None:
    self._full_session_id = "session-coordinator"
    self.calls: list[dict[str, Any]] = []

  async def spawn_sub_agent(self, task: str, **kwargs: Any):
    self.calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None

  def _get_tool_definitions(self) -> list[dict[str, str]]:
    return [
      {"name": "file_glob"},
      {"name": "file_read"},
      {"name": "memory_read"},
    ]


class _CapabilityResolver:
  def __init__(self) -> None:
    self.registry = INITIAL_MODEL_REGISTRY
    self.calls: list[dict[str, Any]] = []

  def resolve(self, capability_id: str, **kwargs: Any) -> SimpleNamespace:
    self.calls.append({"capability_id": capability_id, **kwargs})
    entry = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
    bind = CapabilityBind(
      schema_version="1.0",
      capability_id=capability_id,
      model_key=entry.key,
      provider=entry.provider,
      upstream_model=entry.upstream_model,
      adapter=entry.adapter,
      protocol_profile=entry.protocol_profile,
      route=entry.route,
      effort="high",
      credential_principal="user",
      credential_ref="test-user:anthropic",
      run_mode="interactive",
      registry_revision="test.1",
      policy_revision="test.1",
      selection_source="internal_policy",
    )
    return SimpleNamespace(
      bind=bind,
      provider=SimpleNamespace(name=entry.provider),
      auth_config={
        "provider": entry.provider,
        "api_key": "coordinator-test-key",
      },
    )


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
  assert "do not poll running workers" in COORDINATOR_DEFAULT_PREAMBLE
  assert "automatic completion notifications" in COORDINATOR_DEFAULT_PREAMBLE


def test_coordinator_preamble_injected_into_string_system_prompt() -> None:
  provider = _PromptCaptureProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-coordinator",
    capability_execution=stub_runner_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
    ),
    coordinator=CoordinatorConfig(enabled=True),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}], system_prompt="Base prompt"))

  reminder = runner._build_notification_reminder()
  assert provider.system_prompts == [
    f"{COORDINATOR_DEFAULT_PREAMBLE}\n\nBase prompt\n\n{reminder}"
  ]


def test_coordinator_custom_preamble_overrides_default_for_prompt_blocks() -> None:
  provider = _PromptCaptureProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-coordinator",
    capability_execution=stub_runner_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
    ),
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

  reminder = runner._build_notification_reminder()
  assert provider.system_prompts == [[
    ("Custom coordinator preamble", False),
    ("Base prompt", True),
    (reminder, False),
  ]]


def test_coordinator_disabled_does_not_inject_preamble() -> None:
  provider = _PromptCaptureProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-coordinator",
    capability_execution=stub_runner_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
    ),
    coordinator=CoordinatorConfig(enabled=False, preamble="Should not appear"),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}], system_prompt="Base prompt"))

  reminder = runner._build_notification_reminder()
  assert provider.system_prompts == [f"Base prompt\n\n{reminder}"]


def test_make_run_agent_handler_merges_worker_excluded_tools() -> None:
  runner = _StubRunner()
  resolver = _CapabilityResolver()
  handler = make_run_agent_handler(
    [runner],
    parent_session=OWNER_GATEWAY_SESSION,
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={
      "file_read": _dummy_tool,
      "memory_read": _dummy_tool,
      "file_glob": _dummy_tool,
    },
    excluded_tools={"memory_read"},
    coordinator_config=CoordinatorConfig(
      enabled=True,
      worker_excluded_tools={"file_glob"},
    ),
    capability_execution_resolver=resolver,
  )

  result, error = _run(handler({"objective": "Collect", "background": False}))

  assert error is None
  assert result == {"response": "ok"}
  assert _DEFAULT_EXCLUDED_TOOLS | {
    "file_glob",
    "memory_read",
  } <= runner.calls[0]["excluded_tools"]
  worker_tools = runner.calls[0]["dispatcher"]._local
  assert set(worker_tools) == {"file_read"}
  assert worker_tools["file_read"] is _dummy_tool
  assert "report_door" not in runner.calls[0]


def test_coordinator_max_workers_overrides_max_background_tasks() -> None:
  provider = _PromptCaptureProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-coordinator",
    capability_execution=stub_runner_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
    ),
    max_concurrent_sub_agents=1,
    coordinator=CoordinatorConfig(enabled=True, max_workers=5),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  assert runner._max_background_tasks == 5
  assert runner._task_registry._max_inflight == 5


def test_make_run_agent_handler_uses_authenticated_worker_bind() -> None:
  runner = _StubRunner()
  resolver = _CapabilityResolver()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={"file_read": _dummy_tool},
    coordinator_config=CoordinatorConfig(enabled=True),
    capability_execution_resolver=resolver,
  )

  result, error = _run(handler({"objective": "Collect", "background": False}))

  assert error is None
  assert result == {"response": "ok"}
  assert runner.calls[0]["capability_execution"].bind.provider == "anthropic"
  assert runner.calls[0]["capability_execution"].bind.upstream_model == "claude-opus-5"


def test_make_run_agent_handler_rejects_provider_selection_input() -> None:
  runner = _StubRunner()
  resolver = _CapabilityResolver()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    coordinator_config=CoordinatorConfig(enabled=True),
    capability_execution_resolver=resolver,
  )

  result, error = _run(handler({"objective": "Collect", "provider": "openai"}))

  assert result is None
  assert error["code"] == "invalid_input"
  assert error["message"] == "unknown run_agent input fields: provider"
  assert resolver.calls == []
  assert runner.calls == []


def test_make_run_agent_handler_rejects_model_selection_input() -> None:
  runner = _StubRunner()
  resolver = _CapabilityResolver()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    coordinator_config=CoordinatorConfig(enabled=True),
    capability_execution_resolver=resolver,
  )

  result, error = _run(handler({"objective": "Collect", "model": "gpt-4o-mini"}))

  assert result is None
  assert error["code"] == "invalid_input"
  assert error["message"] == "unknown run_agent input fields: model"
  assert resolver.calls == []
  assert runner.calls == []
