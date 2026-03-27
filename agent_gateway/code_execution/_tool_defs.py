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
      "Execute Python code and return stdout, stderr, exit code, and any generated plots/images. "
      "Use for computation, data transformation, visualization, and analysis. "
      "Pre-installed: numpy, pandas, matplotlib, scipy. "
      "The working directory persists across calls within a session."
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
          "maximum": config.max_timeout_ms,
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
