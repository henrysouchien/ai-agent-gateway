import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_prompt_rules import (  # noqa: E402
  last_user_message,
  message_content_text,
  messages_require_tool_only_turns,
  prepend_system_prompt_preamble,
  system_prompt_requires_tool_only_turns,
  system_prompt_text,
)


def test_runner_preserves_prompt_rule_helper_aliases() -> None:
  assert gateway_runner._system_prompt_text is system_prompt_text
  assert gateway_runner._system_prompt_requires_tool_only_turns is system_prompt_requires_tool_only_turns
  assert gateway_runner._message_content_text is message_content_text
  assert gateway_runner._messages_require_tool_only_turns is messages_require_tool_only_turns
  assert gateway_runner._prepend_system_prompt_preamble is prepend_system_prompt_preamble
  assert gateway_runner._last_user_message is last_user_message


def test_system_prompt_text_accepts_string_list_and_missing_values() -> None:
  assert system_prompt_text("plain") == "plain"
  assert system_prompt_text([("first", True), ("second", False)]) == "first\nsecond"
  assert system_prompt_text(None) == ""


def test_prepend_system_prompt_preamble_handles_string_list_and_missing_values() -> None:
  blocks = [("static", True), ("dynamic", False)]

  assert prepend_system_prompt_preamble("base", "coord") == "coord\n\nbase"
  assert prepend_system_prompt_preamble("", "coord") == "coord"
  assert prepend_system_prompt_preamble(None, "coord") == "coord"
  assert prepend_system_prompt_preamble(blocks, "coord") == [
    ("coord", False),
    ("static", True),
    ("dynamic", False),
  ]
  assert blocks == [("static", True), ("dynamic", False)]


def test_system_prompt_detects_tool_only_turn_instruction_case_insensitively() -> None:
  assert system_prompt_requires_tool_only_turns("Tool-call messages are tool-only.") is True
  assert (
    system_prompt_requires_tool_only_turns(
      "Every assistant message that contains any tool call must contain zero visible text."
    )
    is True
  )
  assert (
    system_prompt_requires_tool_only_turns(
      "Every assistant message before the FMS call must be tool-only (`text=0`)."
    )
    is True
  )
  assert (
    system_prompt_requires_tool_only_turns(
      "No visible pre-FMS prose. All pre-FMS tool turns must be tool-only with text=0."
    )
    is True
  )
  assert system_prompt_requires_tool_only_turns("normal prompt") is False


def test_message_content_text_flattens_text_blocks_and_ignores_other_content() -> None:
  assert message_content_text("hello") == "hello"
  assert (
    message_content_text(
      [
        {"type": "text", "text": "alpha"},
        {"type": "image", "source": "ignored"},
        "beta",
        {"type": "text", "text": 3},
      ]
    )
    == "alpha\nbeta"
  )
  assert message_content_text({"not": "text"}) == ""


def test_last_user_message_returns_copy_of_latest_user_message() -> None:
  latest_user = {"role": "user", "content": "latest", "metadata": {"id": "u2"}}
  messages = [
    {"role": "system", "content": "setup"},
    {"role": "user", "content": "earlier"},
    {"role": "assistant", "content": "reply"},
    latest_user,
  ]

  result = last_user_message(messages)

  assert result == latest_user
  assert result is not latest_user
  runner = gateway_runner.AgentRunner.__new__(gateway_runner.AgentRunner)
  assert runner._extract_last_user_message(messages) == latest_user

  assert result is not None
  result["content"] = "changed"
  assert latest_user["content"] == "latest"


def test_last_user_message_returns_none_without_user_message() -> None:
  messages = [
    {"role": "system", "content": "setup"},
    {"role": "assistant", "content": "reply"},
  ]

  assert last_user_message(messages) is None
  runner = gateway_runner.AgentRunner.__new__(gateway_runner.AgentRunner)
  assert runner._extract_last_user_message(messages) is None


def test_messages_detect_tool_only_turn_instruction_in_any_message_content() -> None:
  messages = [
    {"role": "user", "content": "ordinary"},
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Every assistant message that contains any tool call must contain zero visible text."}
      ],
    },
  ]

  assert messages_require_tool_only_turns(messages) is True
  assert (
    messages_require_tool_only_turns(
      [
        {"role": "user", "content": "Every assistant message before the FMS call must be tool-only (`text=0`)."}
      ]
    )
    is True
  )
  assert messages_require_tool_only_turns([{"role": "user", "content": "ordinary"}, "bad"]) is False
