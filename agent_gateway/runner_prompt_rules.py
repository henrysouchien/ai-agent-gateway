from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union


def prepend_system_prompt_preamble(
  system_prompt: Optional[Union[str, List[Tuple[str, bool]]]],
  preamble: str,
) -> Union[str, List[Tuple[str, bool]]]:
  if isinstance(system_prompt, list):
    return [(preamble, False)] + list(system_prompt)
  if system_prompt:
    return f"{preamble}\n\n{system_prompt}"
  return preamble


def system_prompt_text(system_prompt: Optional[Union[str, List[Tuple[str, bool]]]]) -> str:
  if isinstance(system_prompt, str):
    return system_prompt
  if isinstance(system_prompt, list):
    return "\n".join(str(part[0]) for part in system_prompt if part)
  return ""


def system_prompt_requires_tool_only_turns(system_prompt: Optional[Union[str, List[Tuple[str, bool]]]]) -> bool:
  text = system_prompt_text(system_prompt).lower()
  return (
    "tool-call messages are tool-only" in text
    or "every assistant message that contains any tool call must contain zero visible text" in text
  )


def message_content_text(content: Any) -> str:
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts: list[str] = []
    for block in content:
      if isinstance(block, dict):
        text = block.get("text")
        if isinstance(text, str):
          parts.append(text)
      elif isinstance(block, str):
        parts.append(block)
    return "\n".join(parts)
  return ""


def last_user_message(request_messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
  for msg in reversed(request_messages):
    if msg.get("role") == "user":
      return dict(msg)
  return None


def messages_require_tool_only_turns(messages: List[Dict[str, Any]]) -> bool:
  for message in messages:
    if not isinstance(message, dict):
      continue
    text = message_content_text(message.get("content")).lower()
    if (
      "tool-call messages are tool-only" in text
      or "every assistant message that contains any tool call must contain zero visible text" in text
    ):
      return True
  return False
