import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog, ToolDispatcher  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_run_loop import RunnerRunLoopMixin  # noqa: E402


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}


class _NoCredentialProvider:
  name = "patched-provider"

  def __init__(self) -> None:
    self.seen_config: dict[str, Any] | None = None

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    self.seen_config = dict(config)
    return False


def _make_no_credential_runner(provider: _NoCredentialProvider) -> AgentRunner:
  event_log = EventLog()
  dispatcher = ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log,
    session_id="sess_run_loop",
  )
  return AgentRunner(
    event_log=event_log,
    dispatcher=dispatcher,
    session_id="sess_run_loop",
    provider=provider,
    auth_config={"api_key": "k"},
    get_tool_definitions=lambda: [],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_runner_run_loop_method_is_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerRunLoopMixin)
  assert gateway_runner.RunnerRunLoopMixin is RunnerRunLoopMixin
  assert AgentRunner.run is RunnerRunLoopMixin.run


def test_runner_still_reexports_run_loop_constants() -> None:
  assert gateway_runner._MAX_NOTIFICATIONS_PER_TURN == 5
  assert gateway_runner._MAX_TOKENS_CONTINUATIONS == 3
  assert "tool-first response" in gateway_runner._MAX_TOKENS_NUDGE


def test_run_loop_resolves_compatibility_aliases_from_runner_module(monkeypatch) -> None:
  calls: dict[str, Any] = {}

  def fake_default_model(provider_name: str | None) -> str:
    calls["provider_name"] = provider_name
    return "parent-default-model"

  def fake_normalized_run_config(
    auth_config: dict[str, Any],
    *,
    default_model: str,
    model_override: str | None,
  ) -> dict[str, Any]:
    calls["auth_config"] = dict(auth_config)
    calls["default_model"] = default_model
    calls["model_override"] = model_override
    return {"model": "parent-normalized-model", "thinking": False}

  provider = _NoCredentialProvider()
  runner = _make_no_credential_runner(provider)

  async def fake_stub_response(messages: list[dict[str, Any]]) -> None:
    calls["stub_messages"] = list(messages)

  monkeypatch.setattr(gateway_runner, "_get_default_model_for_provider", fake_default_model)
  monkeypatch.setattr(gateway_runner, "_normalized_run_config", fake_normalized_run_config)
  monkeypatch.setattr(runner, "_emit_stub_response", fake_stub_response)

  asyncio.run(
    runner.run(
      messages=[{"role": "user", "content": "hello"}],
      model_override="request-model",
    )
  )

  assert calls["provider_name"] == "patched-provider"
  assert calls["auth_config"] == {"api_key": "k"}
  assert calls["default_model"] == "parent-default-model"
  assert calls["model_override"] == "request-model"
  assert provider.seen_config == {"model": "parent-normalized-model", "thinking": False}
  assert calls["stub_messages"] == [{"role": "user", "content": "hello"}]
