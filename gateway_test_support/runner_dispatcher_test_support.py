from copy import deepcopy
from typing import Any, Mapping

from agent_gateway.approval_policy import RunContext
from agent_gateway.registry_scope import NoRegistryScope
from agent_gateway.runner_tool_audit import redact_tool_input_for_event
from agent_gateway.tool_dispatch_classification import (
  ToolResultSettlement,
  settle_catalogless_tool_result,
)
from agent_gateway.tool_dispatcher import _SpawningTurn
from agent_gateway.tool_policy_registry import PreparedToolCall
from gateway_test_support.run_context_test_support import run_context_for_test


class CataloglessDispatcherTestSupport:
  """``AgentRunnerDispatcher`` surface for catalogless doubles.

  The runner reads ``run_context.registry_scope`` for every tool call; a
  catalogless double declares no citation registry.
  """

  _run_context: RunContext | None = None

  @property
  def run_context(self) -> RunContext:
    if self._run_context is None:
      self._run_context = run_context_for_test(registry_scope=NoRegistryScope())
    return self._run_context

  @run_context.setter
  def run_context(self, run_context: RunContext) -> None:
    self._run_context = run_context

  def rebind_run_context(self, run_context: RunContext) -> None:
    self._run_context = run_context

  def bind_research_file_activity_owner(self, owner: Any) -> None:
    self.research_file_activity_owner = owner

  @staticmethod
  def redact_raw_tool_input_for_history(
    tool_name: str,
    tool_input: Mapping[str, Any],
  ) -> dict[str, Any]:
    return redact_tool_input_for_event(
      tool_name,
      deepcopy(dict(tool_input)),
    )

  @staticmethod
  def redact_prepared_tool_input(
    tool_name: str,
    prepared_call: PreparedToolCall,
  ) -> dict[str, Any]:
    return redact_tool_input_for_event(
      tool_name,
      prepared_call.materialize_input(),
    )

  @staticmethod
  def settle_tool_result(
    _tool_name: str,
    dispatch_entry: Any,
    result: Any,
    error: Any,
    semantic_error: Any = None,
    *,
    prepared_call: PreparedToolCall,
  ) -> ToolResultSettlement:
    _ = prepared_call
    return settle_catalogless_tool_result(
      entry=dispatch_entry,
      result=result,
      error=error,
      semantic_error=semantic_error,
    )

  async def prepare_tool_call_async(
    self,
    tool_name: str,
    tool_input: Mapping[str, Any],
  ) -> PreparedToolCall:
    _ = tool_name
    return PreparedToolCall(dict(tool_input))

  async def dispatch_prepared(
    self,
    tool_call_id: str,
    tool_name: str,
    prepared_call: PreparedToolCall,
    **kwargs: Any,
  ) -> Any:
    return await self.dispatch(  # type: ignore[attr-defined]
      tool_call_id,
      tool_name,
      prepared_call.materialize_input(),
      **kwargs,
    )

  def capture_spawning_turn(self) -> _SpawningTurn:
    """No admitted turn: catalogless doubles are not a spawning runner."""
    return _SpawningTurn(context=None, root_turn=None)
