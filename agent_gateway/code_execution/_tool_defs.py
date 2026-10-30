from __future__ import annotations

import re
from typing import Any, Dict

from ._config import CodeExecutionConfig
CODE_EXECUTE_TOOL_NAME = "code_execute"
CODE_EXECUTION_TOOL_NAMES: frozenset[str] = frozenset({CODE_EXECUTE_TOOL_NAME})

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}
_ANY = {}
_NUM = {"type": "number"}

# The bridge operations the cell client generates as `hank.<name>(...)`: one
# registration, JSON-schema shaped like any tool definition. An op lowers to
# the same authority the direct door uses; it is never in a grant or an MCP
# policy set. `properties` order is the positional/argument order.
CODE_EXECUTION_DELEGATION_TOOL_DEFS: tuple[Dict[str, Any], ...] = (
  {
    "name": "agent",
    "description": (
      "Submit a delegated job through the same admission as run_agent; "
      "returns its task_id at once. Wait on it with hank.wait."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "objective": _STR,
        "operation": _STR,
        "handles": _STR_LIST,
        "role": _STR,
        "effort": _STR,
        "ticker": _STR,
        "research_file_id": _STR,
      },
      "required": ["objective"],
    },
  },
  {
    "name": "wait",
    "description": (
      "Block until the named jobs settle or timeout (seconds) elapses; "
      "returns each job's state and, once completed, its error and "
      "task_result_id (for agent jobs). A result reaches you as its background task "
      "notification after the cell returns, or by get_background_result(task_id=...) "
      "when notifications are off. Jobs keep running when the wait is interrupted."
    ),
    "input_schema": {
      "type": "object",
      "properties": {"task_ids": _STR_LIST, "timeout": _NUM},
      "required": ["task_ids"],
    },
  },
  {
    "name": "completion",
    "description": (
      "Submit a leaf model completion under a registered model-call operation; "
      "returns its task_id at once. Wait on it with hank.wait."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "operation": _STR,
        "material": _STR,
        "role": _STR,
        "effort": _STR,
        "evidence_handles": _STR_LIST,
      },
      "required": ["operation"],
    },
  },
  {
    "name": "judge",
    "description": (
      "Submit a leaf judgment under a registered judgment operation and questions; "
      "returns its task_id at once. Wait on it with hank.wait."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "operation": _STR,
        "state": _ANY,
        "questions": _ANY,
        "role": _STR,
        "effort": _STR,
        "evidence_handles": _STR_LIST,
      },
      "required": ["operation", "state", "questions"],
    },
  },
  {
    "name": "workpool",
    "description": (
      "Submit a batch of items as one P4 controller job drawing from the per-runner six-slot queue; "
      "returns the controller task_id, submitted item keys, and item task_ids. Wait on it with hank.wait."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "items": _ANY,
        "operation": _STR,
        "mode": _STR,
        "role": _STR,
        "effort": _STR,
        "context_handles": _STR_LIST,
        "timeout": _NUM,
      },
      "required": ["items"],
    },
  },
  {
    "name": "judge_batch",
    "description": (
      "Apply questions across keyed states under a registered judgment operation; "
      "returns keyed judgments under the registered model-call operation. Wait on it with hank.wait when wait=false."
    ),
    "input_schema": {
      "type": "object",
      "properties": {
        "operation": _STR,
        "states": _ANY,
        "questions": _ANY,
        "role": _STR,
        "effort": _STR,
        "evidence_handles": _STR_LIST,
        "wait": {"type": "boolean"},
        "timeout": _NUM,
      },
      "required": ["operation", "states", "questions"],
    },
  },
)


def _bridge_operation_names() -> str:
  """`hank.<name>(<args>)` for every registered bridge op, in registration order."""
  return ", ".join(
    f"hank.{row['name']}({', '.join(row['input_schema']['properties'])})"
    for row in CODE_EXECUTION_DELEGATION_TOOL_DEFS
  )

def _model_call_operation_names() -> str:
  """Registered leaf model-call operations."""
  from hank_model_transport.capability_execution import list_model_call_operations

  return ", ".join(op.name for op in list_model_call_operations())


def _placement_description(config: CodeExecutionConfig) -> str:
  """Where the session's one kernel runs and what that placement exposes.

  A host kernel always imports `portfolio_math` (it carries the serving
  interpreter's import roots); `_risk` exists only where the preamble binds it.
  """
  if config.placement == "docker":
    return (
      "The kernel runs in a Docker container with no direct network access; the user's "
      "artifact files are readable there, read-only, at their host paths. "
    )
  if re.search(r"^_risk\s*=", config.preamble_suffix, re.MULTILINE):
    return "The kernel runs on the host: _risk and portfolio_math are available in it. "
  return (
    "The kernel runs on the host: portfolio_math is importable in it; _risk is not "
    "bound, so reach portfolio and risk data through hank.<tool_name>(...). "
  )


def make_code_execute_tool_def(config: CodeExecutionConfig) -> Dict[str, Any]:
  """Build the public tool schema for `code_execute`."""
  description_suffix = f" {config.tool_description_suffix.strip()}" if config.tool_description_suffix.strip() else ""
  return {
    "name": CODE_EXECUTE_TOOL_NAME,
    "description": (
      "Execute Python for computation and composable data analysis; return compact output and plots. "
      "Use for computation, data transformation, visualization, and analysis. "
      "Pre-installed: numpy, pandas, matplotlib, scipy. "
      "One Python kernel per session: locals and imports persist across cells and turns. "
      "print(...) is the primary output channel and a cell's final expression returns its repr; "
      "use display(value) only for images and rich output. "
      "The working directory persists until session teardown. Results include a bounded stdout tail "
      "and spill paths for complete output. For an oversized tool-result preview, use tool_result_read "
      "with its opaque spill_ref.\n\n"
      "Use code_execute when combining multiple tool calls in a programmatic loop, performing "
      "cross-statement reconciliation math, or transforming/visualizing tables in Python. Single or "
      "vectorized door reads and stage writes (model_document, check_model, corpus_state, fms_persist_*) "
      "belong on the direct outer tool surface, which projects deltas/readbacks in one turn and runs "
      "parallel tool calls in a single turn; do not wrap a single door read in a code_execute cell. "
      "Make several tool calls per cell instead of one call per turn when a cell genuinely loops or "
      "computes over them. Whenever a cell stores a result in a persistent kernel local variable, "
      "print(...) both the compact summary and a one-line structural summary (top-level keys, row "
      "count, period axis) in the same cell, so you never spend a follow-up turn probing type(x) or "
      "x.keys(). For long-running work, set background=true: the "
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
      "Tool dict/list results retain their producer reference, including nested objects, arrays "
      "and slices kept across cells. Select each actual operand with hank.input(container, key), "
      "then publish a computed figure with hank.derive(name, value, expression=..., inputs=[...]). "
      "The result's derived rows retain the expression, selected observations and their original "
      "source rows; literals or unissued references publish input_unsourced findings rather than "
      "refusing the computation. Derived values are not primary sources. Ordinary arithmetic "
      "does not propagate lineage automatically. "
      "For figures stored in a model, keep model lineage on the model door as well: derived lines "
      "use model_document formulas; historical inputs use sourced historical foundation or "
      "correspondence; off-sheet judgments use an analyst note naming the method and inputs. "
      "Granted domain reads run through the gateway, which holds credentials and returns source "
      "identities; do not copy secrets into Python. "
      f"{_placement_description(config)}"
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
            "Compute timeout in milliseconds. Time the cell's own thread spends "
            "blocked on hank calls does not count; the result reports it as host_wait_ms beside compute_ms. "
            "A cell queued behind another cell on the sticky kernel waits at most this long, then returns "
            "error.code kernel_busy naming the holder in error.held_by. "
            f"Default: {config.default_timeout_ms}."
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
      },
      "required": ["code"],
    },
  }
