from __future__ import annotations

import inspect
from typing import Callable

from .tool_dispatcher_helpers import NeedsApprovalCallback


def normalize_needs_approval(
  needs_approval: Callable[..., bool] | None,
) -> NeedsApprovalCallback:
  if needs_approval is None:
    return lambda _name, _tool_input, _qualifier: False

  try:
    arg_count = len(inspect.signature(needs_approval).parameters)
  except (TypeError, ValueError):
    return needs_approval

  if arg_count == 1:
    return lambda name, _tool_input, _qualifier: needs_approval(name)
  if arg_count == 2:
    return lambda name, tool_input, _qualifier: needs_approval(name, tool_input)
  return needs_approval
