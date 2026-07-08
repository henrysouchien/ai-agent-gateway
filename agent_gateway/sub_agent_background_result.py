from __future__ import annotations

from typing import Any


def make_get_background_result_handler(runner_ref: list[Any]):
  """Build the local handler used by the ``get_background_result`` tool."""

  async def _handle_get_background_result(tool_input: dict[str, Any], **kwargs: Any):
    _ = kwargs
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
    return await runner.get_background_result(tool_input)

  return _handle_get_background_result


__all__ = ["make_get_background_result_handler"]
