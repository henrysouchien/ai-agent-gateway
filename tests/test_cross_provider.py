# ruff: noqa: E402

import asyncio
import datetime
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (
  AgentRunner,
  EventLog,
  ModelInfo,
  ModelProvider,
  SkillLoader,
  ToolDispatcher,
  make_run_agent_handler,
  make_run_agent_tool_def,
  parse_skill_file,
)
from agent_gateway.capability_binding import (
  CapabilityResolutionError,
)
from agent_gateway.capability_execution import BoundCapabilityExecution
from agent_gateway.providers import StreamEvent
from tests.tool_catalog_test_support import OWNER_GATEWAY_SESSION
from tests.capability_execution_test_support import stub_bound_capability_execution


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _RecordingProvider(ModelProvider):
  def __init__(self, name: str) -> None:
    self.name = name
    self.created_configs: list[dict[str, Any]] = []
    self.request_tools: list[list[dict[str, Any]]] = []

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = timeout
    self.created_configs.append(dict(config))
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      supports_thinking=True,
      input_cost_per_mtok=0.0,
      output_cost_per_mtok=0.0,
      cache_read_cost_per_mtok=0.0,
      cache_write_cost_per_mtok=0.0,
    )

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
    _ = model, messages, system_prompt, max_tokens, kwargs
    self.request_tools.append(list(tools))
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    text = f"{self.name} response"
    yield StreamEvent(type="message_start", input_tokens=1)
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="usage_update", output_tokens=1)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def _execution(
  provider: _RecordingProvider,
  model: str,
  *,
  capability_id: str = "node.implement",
) -> BoundCapabilityExecution:
  return stub_bound_capability_execution(
    provider=provider,
    model=model,
    effort="high",
    capability_id=capability_id,
    credential_principal="user",
    auth_config={
      "api_key": f"{provider.name}-key",
      "max_tokens": 16_000,
    },
  )


class _ExactResolver:
  def __init__(self) -> None:
    self.providers = {
      "anthropic": _RecordingProvider("anthropic"),
      "openai": _RecordingProvider("openai"),
    }
    self.calls: list[dict[str, Any]] = []

  def resolve(self, capability_id: str, **kwargs: Any) -> BoundCapabilityExecution:
    self.calls.append({"capability_id": capability_id, **kwargs})
    return _execution(
      self.providers["anthropic"],
      "claude-sonnet-4-6",
      capability_id=capability_id,
    )


class _StubRunner:
  def __init__(self) -> None:
    self._full_session_id = "session-cross-provider"
    self._agent_session_log = object()
    self.calls: list[dict[str, Any]] = []
    self.durable_events: list[dict[str, Any]] = []

  async def _append_durable_event(
    self,
    event: dict[str, Any],
  ) -> object:
    self.durable_events.append(dict(event))
    return object()

  async def _confirm_durable_skill_event(
    self,
    event: dict[str, Any],
  ) -> dict[str, Any] | None:
    return next(
      (
        dict(durable_event)
        for durable_event in self.durable_events
        if durable_event == event
      ),
      None,
    )

  async def spawn_sub_agent(self, task: str, **kwargs: Any):
    self.calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None


def _make_dispatcher(
  event_log: EventLog | None = None,
  *,
  get_tool_definitions: Any | None = None,
) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-parent",
    get_tool_definitions=get_tool_definitions,
  )


def _make_runner(provider: _RecordingProvider) -> AgentRunner:
  return AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_execution(
      provider,
      "claude-sonnet-4-6",
      capability_id="session.driver",
    ),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def _write_skill(skills_dir: Path, name: str, body: str) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


def test_spawn_sub_agent_requires_bound_execution_instead_of_parent_fallback() -> None:
  runner = _make_runner(_RecordingProvider("anthropic"))

  with pytest.raises(TypeError, match="capability_execution"):
    _run(
      runner.spawn_sub_agent(  # type: ignore[call-arg]
        "Collect context",
        dispatcher=_make_dispatcher(),
        max_turns=1,
        timeout=5.0,
      )
    )


def test_bound_execution_rejects_provider_family_mismatch() -> None:
  execution = _execution(_RecordingProvider("openai"), "gpt-4o-mini")
  with pytest.raises(CapabilityResolutionError, match="does not match"):
    BoundCapabilityExecution(
      bind=execution.bind,
      registry=execution.registry,
      adapter=_RecordingProvider("anthropic"),
      auth_config=execution.auth_config,
    )


def test_run_agent_rejects_raw_provider_and_model_selection_before_spawn() -> None:
  runner = _StubRunner()
  resolver = _ExactResolver()
  handler = make_run_agent_handler(
    [runner],
    parent_session=OWNER_GATEWAY_SESSION,
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    capability_execution_resolver=resolver,
  )

  result, error = _run(handler({
    "background": False,
    "task": "Collect",
    "provider": "openai",
    "model": "gpt-4o-mini",
  }))

  assert result is None
  assert error["code"] == "invalid_input"
  assert resolver.calls == []
  assert runner.calls == []


def test_run_agent_rejects_provider_only_before_resolver_or_spawn() -> None:
  runner = _StubRunner()
  resolver = _ExactResolver()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    capability_execution_resolver=resolver,
  )

  result, error = _run(handler({"task": "Collect", "provider": "openai"}))

  assert result is None
  assert error["code"] == "invalid_input"
  assert resolver.calls == []
  assert runner.calls == []


def test_run_agent_rejects_bare_upstream_model() -> None:
  runner = _StubRunner()
  resolver = _ExactResolver()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    capability_execution_resolver=resolver,
  )

  result, error = _run(handler({
    "background": False,
    "task": "Collect",
    "model": "gpt-4o-mini",
  }))

  assert result is None
  assert error["code"] == "invalid_input"
  assert resolver.calls == []
  assert runner.calls == []


def test_skill_profile_parses_provider_frontmatter(tmp_path: Path) -> None:
  skill_path = tmp_path / "worker.md"
  skill_path.write_text(
    "---\nprovider: openai\n---\nUse the OpenAI worker.\n",
    encoding="utf-8",
  )

  profile = parse_skill_file(skill_path)

  assert profile.provider == "openai"


def test_run_agent_rejects_legacy_skill_and_task_selectors(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(
    skills_dir,
    "openai-worker",
    (
      "---\nagent_callable: true\nagent_description: Uses the OpenAI worker.\n"
      "provider: openai\nmodel: gpt-4o-mini\nmutation_mode: read_only\n"
      "---\nUse the OpenAI worker."
    ),
  )
  runner = _StubRunner()
  resolver = _ExactResolver()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    capability_execution_resolver=resolver,
  )

  result, error = _run(handler({
    "agent": "openai-worker",
    "task": "Collect",
    "background": False,
  }))

  assert result is None
  assert error["code"] == "invalid_input"
  assert resolver.calls == []
  assert runner.calls == []


def test_run_agent_tool_schema_omits_model_selection_authority() -> None:
  schema = make_run_agent_tool_def()["input_schema"]

  assert not {"provider", "model", "model_key", "effort"} & set(
    schema["properties"]
  )
  assert schema["required"] == ["objective"]
  assert schema["additionalProperties"] is False
