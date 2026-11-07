"""A root turn ends only on event_log.TERMINAL_EVENT_TYPES.

stream_error is an advisory event the gateway substitutes for one it cannot
adapt. It does not end the turn. Root-run settlement derives that set from
event_log. The autonomous channel still refuses stream_error: a child has no
producer for it.
"""
from __future__ import annotations

import asyncio
from typing import Any

from agent_gateway import AgentRunner, EventLog, ToolDispatcher
from agent_gateway.autonomous_runner_state import (
  AutonomousRegistryStateMixin,
  is_root_terminal_event,
)
from agent_gateway.control_plane.runs_helpers import _state_from_session
from agent_gateway.event_log import TERMINAL_EVENT_TYPES
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.registry_scope import RegistryScope
from agent_gateway.runner_state import StreamTurnFailure
from agent_gateway.session import GatewaySession
from gateway_test_support.capability_execution_test_support import (
  stub_bound_capability_execution,
)
from gateway_test_support.run_context_test_support import run_context_for_test
from hank_model_transport.providers import CostEstimate, ModelInfo, ModelProvider


class _FailingStreamProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    return None

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(max_output_tokens=16_384, id=model, provider=self.name)

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    return CostEstimate()


def _session() -> GatewaySession:
  return GatewaySession(
    session_id="sess",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
  )


def test_stream_error_does_not_end_a_root_turn() -> None:
  assert "stream_error" not in TERMINAL_EVENT_TYPES
  assert TERMINAL_EVENT_TYPES == {"stream_complete", "error"}

  log = EventLog()
  log.append({"type": "stream_error", "error": "advisory"})
  assert log.has_terminal is False
  assert log.closed is False
  log.append({"type": "error", "error": "provider stream failed"})
  assert log.has_terminal is True

  advisory = {"type": "stream_error", "error": "advisory"}
  assert is_root_terminal_event(advisory) is False
  assert is_root_terminal_event({"type": "error", "error": "failed"}) is True

  lone_outcome, _, _ = AutonomousRegistryStateMixin._terminal_event_outcome([advisory])
  assert lone_outcome is None

  outcome, error, _ = AutonomousRegistryStateMixin._terminal_event_outcome([
    advisory,
    {"type": "error", "error": "provider stream failed"},
  ])
  assert outcome == "failed"
  assert error == "provider stream failed"

  assert _state_from_session(_session(), [
    advisory,
    {"type": "stream_complete", "terminal_disposition": "interrupted"},
  ]) == "interrupted"
  assert _state_from_session(_session(), [
    advisory,
    {"type": "error", "error": "provider stream failed"},
  ]) == "failed"


def test_provider_stream_failure_settles_failed_through_error_terminal() -> None:
  """A failed provider stream still ends failed on `error`, not on stream_error."""

  async def case() -> None:
    event_log = EventLog()
    provider = _FailingStreamProvider()
    execution = stub_bound_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
      auth_config={"api_key": "k"},
    )
    dispatcher = ToolDispatcher(
      capability_bind=execution.bind,
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess_run_loop",
      run_context=run_context_for_test(
        registry_scope=RegistryScope("run", "test-run", "/tmp/test-store"),
      ),
    )
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id="sess_run_loop",
      capability_execution=execution,
      allow_stub_response=True,
      get_tool_definitions=lambda: [],
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )

    async def stream_turn(**_kwargs: Any) -> StreamTurnFailure:
      return StreamTurnFailure(
        error=RuntimeError("provider stream failed"),
        formatted_error="provider stream failed",
      )

    runner._stream_turn = stream_turn
    await runner.run(messages=[{"role": "user", "content": "finish"}])
    events = [entry.event for entry in runner._log.entries]
    terminals = [
      event for event in events
      if event.get("type") in TERMINAL_EVENT_TYPES
    ]
    assert terminals
    assert terminals[0]["type"] == "error"
    assert "provider stream failed" in str(terminals[0].get("error"))
    assert not any(event.get("type") == "stream_error" for event in events)
    assert not any(
      event.get("type") == "stream_complete"
      and event.get("terminal_disposition") == "completed"
      for event in events
    )
    outcome, _, _ = AutonomousRegistryStateMixin._terminal_event_outcome(events)
    assert outcome == "failed"
    session = _session()
    assert session.stream_active is False
    assert _state_from_session(session, events) == "failed"

  asyncio.run(case())
