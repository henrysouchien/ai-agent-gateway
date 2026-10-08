from __future__ import annotations

from typing import Any, Dict, Sequence

from ._config import CodeExecutionConfig


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
      "generation, or repeatable Python analysis. For long-running work, set background=true and "
      "poll with code_execute_status.\n\n"
      "Sibling tools: use code_execute_status for background progress or cancellation. Use run_bash "
      "for non-Python shell commands and file tools for simple file inspection.\n\n"
      "For multi-step filing, market or portfolio analysis, use the preloaded hank client inside "
      "the cell: hank.search(query), hank.describe(operation), hank.call(operation, **arguments). "
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
          "description": f"Timeout in milliseconds. Default: {config.default_timeout_ms}.",
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


def make_code_execute_status_tool_def() -> Dict[str, Any]:
  """Build the public tool schema for `code_execute_status`."""
  return {
    "name": "code_execute_status",
    "description": "Check or cancel a background code_execute task.",
    "input_schema": {
      "type": "object",
      "properties": {
        "task_id": {
          "type": "string",
          "description": "Background task id returned by code_execute(background=true).",
        },
        "cancel": {
          "type": "boolean",
          "description": "Set true to cancel the task.",
        },
      },
      "required": ["task_id"],
    },
  }
