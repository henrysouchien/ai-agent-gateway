from __future__ import annotations

from typing import Any, Dict, Sequence

from ._config import CodeExecutionConfig
CODE_EXECUTE_TOOL_NAME = "code_execute"
CODE_EXECUTION_TOOL_NAMES: frozenset[str] = frozenset({CODE_EXECUTE_TOOL_NAME})

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
    "name": CODE_EXECUTE_TOOL_NAME,
    "description": (
      "Execute Python for computation and composable data analysis; return compact output and plots. "
      "Use for computation, data transformation, visualization, and analysis. "
      "Pre-installed: numpy, pandas, matplotlib, scipy. "
      "Docker/kernel hosts are session-sticky: locals and imports persist across cells and turns. "
      "print(...) is the primary output channel and a cell's final expression returns its repr; "
      "use display(value) only for images and rich output. Subprocess starts fresh each call. "
      "The working directory persists until session teardown. Results include a bounded stdout tail "
      "and spill paths for complete output. For an oversized tool-result preview, use tool_result_read "
      "with its opaque spill_ref.\n\n"
      "Use code_execute when you need arithmetic verification, data wrangling, chart "
      "generation, or repeatable Python analysis. Make several tool calls per cell instead of "
      "one call per turn. For long-running work, set background=true: the "
      "cell runs as a background job and its result is delivered like any background task "
      "result (task notification, or get_background_result(task_id=...) when notifications are off).\n\n"
      "Sibling tools: use run_bash for non-Python shell commands and file tools for simple file "
      "inspection.\n\n"
      "Inside a cell, call any tool in this session's active catalog directly — local doors and "
      "MCP tools, reads and writes — as hank.<tool_name>(**kwargs), or hank.call(name, **kwargs) "
      "when the name is computed or collides with a hank method; no hank.search / hank.describe "
      "step is required first (they remain optional discovery when a name or schema is genuinely "
      "unknown). A cell cannot recurse into code_execute itself, and money/irreversible trade "
      "execution still goes through the human-approval door. "
      f"To delegate from a cell, {_bridge_operation_names()}. "
      f"Registered leaf model-call operations for completion and judge: {_model_call_operation_names()}. "
      "Full parsed dict/list results arrive in kernel memory untruncated: retain fetched data "
      "in Python locals across cells, filter and compute in Python, and print only compact "
      "summaries so only the printed summary enters the transcript. "
      "To publish a computed figure with its lineage, select the producer values it came from with "
      "hank.input(container, key) (a producer result, or an object/list kept from one; kernel and Docker "
      "cells keep them across cells) and call hank.derive(name, value, expression=..., inputs=[...]); the result's `derived` rows "
      "carry the expression and those inputs' source rows. Arithmetic outside hank.input/hank.derive carries "
      "no lineage. "
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
