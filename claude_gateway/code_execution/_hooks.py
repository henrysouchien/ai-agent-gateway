from __future__ import annotations

import json
from typing import Any, Dict

from ..runner import ToolResultContext


def _load_result_payload(ctx: ToolResultContext) -> Dict[str, Any] | None:
  if ctx.result_entry is None:
    return None
  content = ctx.result_entry.get("content")
  if not isinstance(content, str):
    return None
  try:
    parsed = json.loads(content)
  except Exception:
    return None
  return parsed if isinstance(parsed, dict) else None


def strip_code_execute_base64_hook(ctx: ToolResultContext) -> None:
  """Replace inline image base64 payloads with filename markers.

  Use this in `on_tool_result` pipelines when you want tool results to stay
  readable in logs or downstream model messages.
  """
  if ctx.tool_name != "code_execute" or ctx.result_entry is None or ctx.error is not None:
    return

  result = _load_result_payload(ctx)
  if result is None:
    return

  images = result.get("images")
  if not isinstance(images, list):
    return

  updated = False
  sanitized_images = []
  for image in images:
    if not isinstance(image, dict):
      sanitized_images.append(image)
      continue
    next_image = dict(image)
    if "data_base64" in next_image:
      filename = str(next_image.get("filename") or "image")
      next_image["data_base64"] = f"[image: {filename}]"
      updated = True
    sanitized_images.append(next_image)

  if not updated:
    return

  result["images"] = sanitized_images
  ctx.result_entry["content"] = json.dumps(result, default=str)
