import asyncio
import datetime
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (
  AgentRunner,
  EventLog,
  GatewaySession,
  ModelInfo,
  ModelProvider,
  ProviderResolver,
  ResolvedProvider,
  SkillLoader,
  ToolDispatcher,
  make_run_agent_handler,
  make_run_agent_tool_def,
  parse_skill_file,
)
from agent_gateway.providers import StreamEvent


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

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

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
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    text = f"{self.name} response"
    yield StreamEvent(type="message_start", input_tokens=1)
    yield StreamEvent(type="text_delta", text=text)
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": text})
    yield StreamEvent(type="usage_update", output_tokens=1)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _StubRunner:
  def __init__(self) -> None:
    self._full_session_id = "session-cross-provider"
    self.calls: list[dict[str, Any]] = []

  async def spawn_sub_agent(self, task: str, **kwargs: Any):
    self.calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-parent",
  )


def _make_runner(provider: ModelProvider, *, auth_config: dict[str, Any] | None = None) -> AgentRunner:
  return AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    provider=provider,
    auth_config=auth_config or {"api_key": "parent-key", "model": "claude-sonnet-4-6"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def _write_skill(skills_dir: Path, name: str, body: str) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  (skills_dir / f"{name}.md").write_text(body, encoding="utf-8")


def test_spawn_sub_agent_provider_override_uses_provided_provider_and_auth_config() -> None:
  parent_provider = _RecordingProvider("anthropic")
  child_provider = _RecordingProvider("openai")
  runner = _make_runner(parent_provider)

  result, error = _run(
    runner.spawn_sub_agent(
      "Collect context",
      provider=child_provider,
      auth_config={"api_key": "child-key", "model": "gpt-4o-mini"},
      dispatcher=_make_dispatcher(),
      max_turns=1,
      timeout=5.0,
    )
  )

  assert error is None
  assert result is not None
  assert result["response"] == "openai response"
  assert parent_provider.created_configs == []
  assert child_provider.created_configs == [
    {
      "auth_mode": "api",
      "api_key": "child-key",
      "auth_token": "",
      "model": "gpt-4o-mini",
      "max_tokens": 16000,
      "thinking": True,
    }
  ]


def test_spawn_sub_agent_provider_override_requires_auth_config() -> None:
  runner = _make_runner(_RecordingProvider("anthropic"))

  result, error = _run(
    runner.spawn_sub_agent(
      "Collect context",
      provider=_RecordingProvider("openai"),
      dispatcher=_make_dispatcher(),
      max_turns=1,
      timeout=5.0,
    )
  )

  assert result is None
  assert error == {"code": "invalid_input", "message": "auth_config required when overriding provider"}


def test_spawn_sub_agent_without_provider_uses_parent_provider_and_existing_auth_chain() -> None:
  parent_provider = _RecordingProvider("anthropic")
  runner = _make_runner(parent_provider)
  sub_session = GatewaySession(
    session_id="sub0:sess-parent",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
    auth_config={"api_key": "sub-key", "model": "claude-opus-4-6"},
  )

  result, error = _run(
    runner.spawn_sub_agent(
      "Collect context",
      dispatcher=_make_dispatcher(),
      sub_session=sub_session,
      max_turns=1,
      timeout=5.0,
    )
  )

  assert error is None
  assert result is not None
  assert result["response"] == "anthropic response"
  assert parent_provider.created_configs == [
    {
      "auth_mode": "api",
      "api_key": "sub-key",
      "auth_token": "",
      "model": "claude-opus-4-6",
      "max_tokens": 16000,
      "thinking": True,
    }
  ]


def test_make_run_agent_handler_resolves_provider_before_model_validation() -> None:
  runner = _StubRunner()
  resolved_provider = _RecordingProvider("openai")

  def _resolve_provider(name: str) -> ResolvedProvider:
    assert name == "openai"
    return ResolvedProvider(
      provider=resolved_provider,
      auth_config={"api_key": "openai-key"},
      allowed_models={"gpt-4o-mini"},
    )

  resolver: ProviderResolver = _resolve_provider
  handler = make_run_agent_handler(
    [runner],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    allowed_models={"claude-sonnet-4-6"},
    provider_resolver=resolver,
  )

  result, error = _run(handler({"task": "Collect", "provider": "openai", "model": "gpt-4o-mini"}))

  assert error is None
  assert result == {"response": "ok"}
  assert len(runner.calls) == 1
  assert runner.calls[0]["task"] == "Collect"
  assert runner.calls[0]["provider"] is resolved_provider
  assert runner.calls[0]["auth_config"] == {"api_key": "openai-key"}
  assert runner.calls[0]["model"] == "gpt-4o-mini"
  assert runner.calls[0]["system_prompt"].endswith(f"Today's date: {datetime.date.today().isoformat()}")
  assert runner.calls[0]["excluded_tools"] == {"run_agent", "get_background_result", "send_message"}


def test_make_run_agent_handler_without_provider_resolver_rejects_requested_provider() -> None:
  handler = make_run_agent_handler(
    [_StubRunner()],
    skill_loader=None,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
  )

  result, error = _run(handler({"task": "Collect", "provider": "openai"}))

  assert result is None
  assert error == {
    "code": "provider_not_supported",
    "message": "Provider 'openai' requested but no provider_resolver configured",
  }


def test_skill_profile_parses_provider_frontmatter(tmp_path: Path) -> None:
  skill_path = tmp_path / "worker.md"
  skill_path.write_text("---\nprovider: openai\n---\nUse the OpenAI worker.\n", encoding="utf-8")

  profile = parse_skill_file(skill_path)

  assert profile.provider == "openai"


def test_make_run_agent_handler_uses_skill_provider_and_resolved_default_model(tmp_path: Path) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(
    skills_dir,
    "openai-worker",
    "---\nagent_callable: true\nagent_description: Uses the OpenAI worker.\nprovider: openai\n---\nUse the OpenAI worker.",
  )
  runner = _StubRunner()
  resolved_provider = _RecordingProvider("openai")

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
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    default_model="claude-sonnet-4-6",
    allowed_models={"claude-sonnet-4-6"},
    provider_resolver=resolver,
  )

  result, error = _run(handler({"agent": "openai-worker", "task": "Collect"}))

  assert error is None
  assert result == {"response": "ok"}
  assert len(runner.calls) == 1
  assert runner.calls[0]["provider"] is resolved_provider
  assert runner.calls[0]["auth_config"] == {"api_key": "openai-key"}
  assert runner.calls[0]["model"] == "gpt-4o-mini"


def test_make_run_agent_tool_def_includes_provider_field() -> None:
  tool_def = make_run_agent_tool_def()

  assert "provider" in tool_def["input_schema"]["properties"]
