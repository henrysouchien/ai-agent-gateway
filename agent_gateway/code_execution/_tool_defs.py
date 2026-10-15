from __future__ import annotations

from typing import Any, Dict, Sequence

from ._config import CodeExecutionConfig
from ._frame_reader import BRIDGE_OPERATIONS


def _bridge_operation_names() -> str:
  """`hank.<name>(<args>)` for every registered bridge op, in registration order."""
  return ", ".join(
    f"hank.{row['name']}({', '.join(row['arguments'])})" for row in BRIDGE_OPERATIONS
  )

def _model_call_operation_names() -> str:
  """Registered leaf model-call operations."""
  from ..capability_execution import list_model_call_operations

  return ", ".join(op.name for op in list_model_call_operations())


def make_code_execute_tool_def(
  config: CodeExecutionConfig,
  available_hosts: Sequence[str],
) -> Dict[str, Any]:
  """Build the public tool schema for `code_execute`."""
  description_suffix = f" {config.tool_description_suffix.strip()}" if config.tool_description_suffix.strip() else ""
  return {
    "name": "code_execute",
    "description": (
      "Execute Python for computation and composable data analysis; return compact output and plots. "
      "Use for computation, data transformation, visualization, and analysis. "
      "Pre-installed: numpy, pandas, matplotlib, scipy. "
      "Docker/kernel hosts retain locals and imports across cells; explicit display(value) emits "
      "rich results, while a final expression alone does not. Subprocess starts fresh each call. "
      "The working directory persists until session teardown. Results include a bounded stdout tail "
      "and spill paths for complete output. For an oversized tool-result preview, use tool_result_read "
      "with its opaque spill_ref.\n\n"
      "Discovery: use code_execute when you need arithmetic verification, data wrangling, chart "
      "generation, or repeatable Python analysis. For long-running work, set background=true: the "
      "cell runs as a background job and its result is delivered like any background task "
      "result (task notification, or get_background_result(task_id=...) when notifications are off).\n\n"
      "Sibling tools: use run_bash for non-Python shell commands and file tools for simple file "
      "inspection.\n\n"
      "For multi-step filing, market or portfolio analysis, use the preloaded hank client inside "
      "the cell: hank.search(query), hank.describe(operation), hank.call(operation, **arguments); "
      f"and to delegate from a cell, {_bridge_operation_names()}. "
      f"Registered leaf model-call operations for completion and judge: {_model_call_operation_names()}. "
      "Use discovered names/schemas; retain fetched data in variables and display derived summaries. "
      "Granted domain reads run through the gateway, which holds credentials and returns source "
      "identities. Docker has no direct network access; do not copy secrets into Python. "
      "Direct MCP tools remain useful for simple lookups."
      f"{description_suffix}"
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "code": {
          "type": "string",
          "description": "Python code to execute.",
        },
        "timeout_ms": {
          "type": "integer",
          "minimum": 1000,
          "description": (
            "Compute timeout in milliseconds. On docker/kernel hosts, time the cell's own thread spends "
            "blocked on hank calls does not count; the result reports it as host_wait_ms beside compute_ms. "
            f"Subprocess bounds total wall time. Default: {config.default_timeout_ms}."
          ),
        },
        "host_wait_ms": {
          "type": "integer",
          "description": (
            "Expected time blocked on hank calls, in milliseconds. Recorded and echoed on the result "
            "as requested_host_wait_ms beside the measured host_wait_ms; not a timeout, a blocked "
            "call is never interrupted for it."
            + (
              f" Default: {config.default_host_wait_ms}."
              if config.default_host_wait_ms is not None
              else ""
            )
          ),
        },
        "background": {
          "type": "boolean",
          "description": "Run in background. Default: false.",
        },
        "host": {
          "type": "string",
          "enum": list(available_hosts),
          "description": "Execution backend. Default: auto.",
        },
      },
      "required": ["code"],
    },
  }
