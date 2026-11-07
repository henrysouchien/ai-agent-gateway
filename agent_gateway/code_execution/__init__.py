"""Code execution: tool definitions eagerly, the runtime on first use.

The tool names and definitions (`_tool_defs`, `_config`) are dependency-free and
are what policy tables read at import time. The runtime members (backends,
handlers, frame reader, hooks, cleanup) import the runner, session and provider
stack, so they resolve on first attribute access: importing this package for a
tool name no longer costs that whole closure (design
docs/design/testing-compute-and-time-2026-11-06.md, P2.2).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ._config import CodeExecutionConfig
from ._tool_defs import (
  CODE_EXECUTE_TOOL_NAME,
  CODE_EXECUTION_TOOL_NAMES,
  make_code_execute_tool_def,
)

if TYPE_CHECKING:
  from ._backends import DockerBackend, ExecutionBackend, ExecutionHandle, OnOutputChunk
  from ._cleanup import cleanup_code_execution
  from ._frame_reader import CellChannel, FrameReader
  from ._handlers import (
    CodeExecutionBundle,
    build_code_execution,
    clear_docker_availability_cache,
    docker_backend_unavailable_reason,
  )
  from ._hooks import strip_code_execute_base64_hook

_LAZY_EXPORTS = {
  "DockerBackend": "._backends",
  "ExecutionBackend": "._backends",
  "ExecutionHandle": "._backends",
  "OnOutputChunk": "._backends",
  "cleanup_code_execution": "._cleanup",
  "CellChannel": "._frame_reader",
  "FrameReader": "._frame_reader",
  "CodeExecutionBundle": "._handlers",
  "build_code_execution": "._handlers",
  "clear_docker_availability_cache": "._handlers",
  "docker_backend_unavailable_reason": "._handlers",
  "strip_code_execute_base64_hook": "._hooks",
}


def __getattr__(name: str) -> Any:
  try:
    module_name = _LAZY_EXPORTS[name]
  except KeyError as exc:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
  value = getattr(importlib.import_module(module_name, __name__), name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:
  return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
  "CODE_EXECUTE_TOOL_NAME",
  "CODE_EXECUTION_TOOL_NAMES",
  "CellChannel",
  "CodeExecutionBundle",
  "CodeExecutionConfig",
  "DockerBackend",
  "ExecutionBackend",
  "ExecutionHandle",
  "FrameReader",
  "OnOutputChunk",
  "build_code_execution",
  "cleanup_code_execution",
  "clear_docker_availability_cache",
  "docker_backend_unavailable_reason",
  "make_code_execute_tool_def",
  "strip_code_execute_base64_hook",
]
