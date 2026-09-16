from copy import deepcopy
from typing import Any, Mapping

from agent_gateway.runner_tool_audit import redact_tool_input_for_event
from agent_gateway.tool_dispatch_classification import (
  ToolResultSettlement,
  settle_catalogless_tool_result,
)
from agent_gateway.tool_policy_registry import PreparedToolCall


class CataloglessDispatcherTestSupport:
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
