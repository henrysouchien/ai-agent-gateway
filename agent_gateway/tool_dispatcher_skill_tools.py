from __future__ import annotations

from typing import Any, Callable, MutableMapping


def ensure_gateway_local_tool_handler(
  tool_name: str,
  *,
  local_handlers: MutableMapping[str, Any],
  session: Any | None,
  current_skill_fn: Callable[[], str | None],
) -> bool:
  normalized_tool = str(tool_name or "").strip()
  if not normalized_tool:
    return False
  if normalized_tool in local_handlers:
    return True
  active_skill = current_skill_fn()
  if not active_skill or session is None:
    return False
  bundles = getattr(session, "gateway_local_skill_tools", None)
  if isinstance(bundles, dict):
    bundles = [bundles]
  if not isinstance(bundles, (list, tuple)):
    return False
  for bundle in bundles:
    if not isinstance(bundle, dict):
      continue
    if str(bundle.get("skill_name") or "").strip() != active_skill:
      continue
    handlers = bundle.get("handlers")
    if not isinstance(handlers, dict):
      continue
    handler = handlers.get(normalized_tool)
    if callable(handler):
      local_handlers[normalized_tool] = handler
      return True
  return False


__all__ = ["ensure_gateway_local_tool_handler"]
