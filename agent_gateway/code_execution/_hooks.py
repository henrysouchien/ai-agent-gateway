from __future__ import annotations

import json
from typing import Any, Dict

from ..runner import ToolResultContext


def _load_result_payload(content: Any) -> Dict[str, Any] | None:
  if not isinstance(content, str):
    return None
  try:
    parsed = json.loads(content)
  except Exception:
    return None
  return parsed if isinstance(parsed, dict) else None


def strip_code_execute_base64_content(content: str) -> str:
  """The model-visible form of one serialized `code_execute` result.

  Image bytes become `[image: <name>]` placeholders and native `computations`
  are dropped; content that is not a result object is returned unchanged.
  A foreground cell's result entry and a background cell's job result both
  pass through here.
  """
  result = _load_result_payload(content)
  if result is None:
    return content

  updated = False
  for key in ("computations",):
    if key in result:
      result.pop(key, None)
      updated = True
  displays = result.get("displays")
  if isinstance(displays, list):
    for index, bundle in enumerate(displays):
      if not isinstance(bundle, dict):
        continue
      for mime in tuple(bundle):
        if mime.startswith("image/") and mime != "image/svg+xml":
          bundle[mime] = f"[image: display_{index + 1}]"
          updated = True

  images = result.get("images")
  if isinstance(images, list):
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
    result["images"] = sanitized_images

  if not updated:
    return content
  return json.dumps(result, default=str)


def strip_code_execute_base64_hook(ctx: ToolResultContext) -> None:
  """Sanitize model-visible code-execution results.

  Use this in `on_tool_result` pipelines when you want tool results to stay
  readable in logs or downstream model messages.
  """
  if ctx.tool_name != "code_execute" or ctx.result_entry is None or ctx.error is not None:
    return
  content = ctx.result_entry.get("content")
  if not isinstance(content, str):
    return
  stripped = strip_code_execute_base64_content(content)
  if stripped is not content:
    ctx.result_entry["content"] = stripped
