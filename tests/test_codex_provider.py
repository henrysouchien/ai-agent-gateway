# ruff: noqa: E402

import base64
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.providers import CodexProvider
from agent_gateway.providers.base import ThinkingLevel
import agent_gateway.providers.codex as codex_provider_module
import agent_gateway.providers.codex_helpers as codex_helpers
import agent_gateway.providers.codex_model_info as codex_model_info
from agent_gateway.providers.codex import (
  JWT_CLAIM_PATH,
  _ResponsesStreamState,
  _convert_messages,
  _convert_tools,
  _encode_text_signature_v1,
  _extract_account_id,
  _map_event,
  _parse_sse,
  _resolve_codex_url,
)


def test_codex_provider_helper_exports_are_parent_aliases() -> None:
  helper_names = (
    "DEFAULT_CODEX_BASE_URL",
    "JWT_CLAIM_PATH",
    "DEFAULT_INSTRUCTIONS",
    "_BETA_HEADER",
    "_RETRYABLE_STATUSES",
    "_RETRYABLE_RE",
    "_CODEX_RESPONSE_STATUSES",
    "_TOOL_ID_RE",
    "_SURROGATE_RE",
    "_MODEL_INFO_BY_TAG",
    "_ResponsesStreamState",
    "_config_base_url",
    "_model_matches_tag",
    "_credential_token",
    "_json_dumps_compact",
    "_encode_text_signature_v1",
    "_parse_text_signature",
    "_short_hash",
    "_sanitize_surrogates",
    "_system_prompt_text",
    "_resolve_codex_url",
    "_normalize_codex_status",
    "_map_reasoning_effort",
    "_clamp_reasoning_effort",
    "_same_model_message",
    "_synthetic_tool_result",
    "_normalize_responses_tool_call_id",
    "_assistant_text_block",
    "_tool_result_output",
    "_parse_streaming_json",
    "_extract_account_id",
    "_build_headers",
    "_convert_tools",
    "_convert_messages",
    "_parse_sse",
    "_map_stop_reason",
    "_map_event",
    "_parse_error_response",
  )

  for name in helper_names:
    assert getattr(codex_provider_module, name) is getattr(codex_helpers, name)

  model_info_helper_names = (
    "_MODEL_INFO_BY_TAG",
    "_model_matches_tag",
    "_map_reasoning_effort",
    "_clamp_reasoning_effort",
  )

  for name in model_info_helper_names:
    assert getattr(codex_helpers, name) is getattr(codex_model_info, name)


def _fake_jwt(account_id: str) -> str:
  def _segment(value: dict) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

  return ".".join(
    [
      _segment({"alg": "none", "typ": "JWT"}),
      _segment({JWT_CLAIM_PATH: {"chatgpt_account_id": account_id}}),
      "signature",
    ]
  )


def test_extract_account_id_reads_chatgpt_claim() -> None:
  token = _fake_jwt("acct_123")

  assert _extract_account_id(token) == "acct_123"


def test_extract_account_id_rejects_invalid_token() -> None:
  with pytest.raises(ValueError, match="Failed to extract accountId"):
    _extract_account_id("not-a-jwt")


def test_codex_has_active_credential_accepts_auth_token() -> None:
  provider = CodexProvider()

  assert provider.has_active_credential({"auth_mode": "oauth", "auth_token": _fake_jwt("acct_123")}) is True
  assert provider.has_active_credential({"auth_mode": "oauth", "auth_token": "   "}) is False


def test_codex_gpt55_uses_gpt5_family_metadata() -> None:
  provider = CodexProvider()

  model_info = provider.get_model_info("gpt-5.5")

  assert model_info.id == "gpt-5.5"
  assert model_info.provider == "codex"
  assert model_info.context_window == 1_050_000
  assert model_info.input_cost_per_mtok == 5.00
  assert model_info.output_cost_per_mtok == 30.00
  assert model_info.supports_thinking is True
  assert model_info.supports_vision is True


def test_codex_gpt55_cost_estimation_is_non_zero() -> None:
  provider = CodexProvider()

  estimate = provider.estimate_cost("gpt-5.5", 1_000, 500, cache_read_tokens=100)

  assert estimate.total > 0
  assert estimate.input_cost > 0
  assert estimate.output_cost > 0
  assert estimate.cache_read_cost > 0


def test_build_request_params_supplies_default_instructions_when_system_prompt_missing() -> None:
  provider = CodexProvider()

  params = provider.build_request_params(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "hello"}],
    system_prompt=None,
    tools=[],
    max_tokens=1024,
  )

  assert params["instructions"] == "Follow the user's instructions."


def test_build_request_params_omits_unsupported_temperature() -> None:
  provider = CodexProvider()

  params = provider.build_request_params(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "hello"}],
    system_prompt="system",
    tools=[],
    max_tokens=1024,
    temperature=0.0,
  )

  assert "temperature" not in params


def test_build_request_params_preserves_reasoning_effort_clamps() -> None:
  provider = CodexProvider()

  gpt55_params = provider.build_request_params(
    model="gpt-5.5",
    messages=[{"role": "user", "content": "hello"}],
    system_prompt="system",
    tools=[],
    max_tokens=1024,
    thinking_level=ThinkingLevel.MINIMAL,
  )
  mini_params = provider.build_request_params(
    model="gpt-5.1-codex-mini",
    messages=[{"role": "user", "content": "hello"}],
    system_prompt="system",
    tools=[],
    max_tokens=1024,
    thinking_level=ThinkingLevel.LOW,
  )

  assert gpt55_params["reasoning"]["effort"] == "low"
  assert mini_params["reasoning"]["effort"] == "medium"


def test_resolve_codex_url_normalizes_backend_api_variants() -> None:
  assert _resolve_codex_url("https://chatgpt.com/backend-api") == "https://chatgpt.com/backend-api/codex/responses"
  assert _resolve_codex_url("https://chatgpt.com/backend-api/codex") == "https://chatgpt.com/backend-api/codex/responses"
  assert _resolve_codex_url("https://chatgpt.com/backend-api/codex/responses") == "https://chatgpt.com/backend-api/codex/responses"


def test_convert_tools_uses_gateway_input_schema_and_null_strict() -> None:
  converted = _convert_tools(
    [
      {
        "name": "read_file",
        "description": "Read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
      }
    ],
    strict=None,
  )

  assert converted == [
    {
      "type": "function",
      "name": "read_file",
      "description": "Read a file",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
      "strict": None,
    }
  ]


def test_convert_messages_translates_same_model_assistant_text_and_tool_history() -> None:
  provider = CodexProvider()
  model_info = provider.get_model_info("gpt-5.1-codex-mini")
  messages = [
    {
      "role": "assistant",
      "provider": "codex",
      "model": "gpt-5.1-codex-mini",
      "content": [
        {
          "type": "thinking",
          "thinking": "",
          "thinkingSignature": json.dumps({"type": "reasoning", "id": "rs_1", "summary": []}),
        },
        {
          "type": "text",
          "text": "Plan complete.",
          "textSignature": _encode_text_signature_v1("msg_text", "final_answer"),
        },
        {
          "type": "tool_use",
          "id": "call_1|fc_item_1",
          "name": "read_file",
          "input": {"path": "README.md"},
        },
      ],
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "call_1|fc_item_1",
          "content": "{\"ok\": true}",
        }
      ],
    },
  ]

  converted = _convert_messages(messages, model_info)

  assert converted[0] == {"type": "reasoning", "id": "rs_1", "summary": []}
  assert converted[1] == {
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "Plan complete.", "annotations": []}],
    "status": "completed",
    "id": "msg_text",
    "phase": "final_answer",
  }
  assert converted[2] == {
    "type": "function_call",
    "call_id": "call_1",
    "id": "fc_item_1",
    "name": "read_file",
    "arguments": "{\"path\": \"README.md\"}",
  }
  assert converted[3] == {
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "{\"ok\": true}",
  }


def test_normalize_messages_normalizes_cross_model_tool_ids() -> None:
  provider = CodexProvider()
  model_info = provider.get_model_info("gpt-5.4")
  messages = [
    {
      "role": "assistant",
      "provider": "codex",
      "model": "gpt-5.1-codex-mini",
      "content": [
        {
          "type": "tool_use",
          "id": "call 1|bad item__",
          "name": "read_file",
          "input": {"path": "README.md"},
        }
      ],
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "call 1|bad item__",
          "content": "{\"ok\": true}",
        }
      ],
    },
  ]

  normalized = provider.normalize_messages(messages, model_info)
  assistant_block = normalized[0]["content"][0]
  tool_result_block = normalized[1]["content"][0]

  assert assistant_block["id"] == "call_1|fc_bad_item"
  assert tool_result_block["tool_use_id"] == "call_1|fc_bad_item"

  converted = _convert_messages(normalized, model_info)
  assert converted[0] == {
    "type": "function_call",
    "call_id": "call_1",
    "name": "read_file",
    "arguments": "{\"path\": \"README.md\"}",
  }
  assert converted[1] == {
    "type": "function_call_output",
    "call_id": "call_1",
    "output": "{\"ok\": true}",
  }


def test_parse_sse_and_map_event_translate_responses_stream() -> None:
  state = _ResponsesStreamState()
  payload = (
    'data: {"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_1","summary":[]}}\n\n'
    'data: {"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":""}}\n\n'
    'data: {"type":"response.reasoning_summary_text.delta","delta":"thinking"}\n\n'
    'data: {"type":"response.reasoning_summary_part.done"}\n\n'
    'data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking\\n\\n"}]}}\n\n'
    'data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}\n\n'
    'data: {"type":"response.content_part.added","part":{"type":"output_text","text":""}}\n\n'
    'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
    'data: {"type":"response.output_item.done","item":{"type":"message","id":"msg_1","phase":"final_answer","content":[{"type":"output_text","text":"hello"}]}}\n\n'
    'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_item_1","call_id":"call_1","name":"lookup","arguments":""}}\n\n'
    'data: {"type":"response.function_call_arguments.delta","delta":"{\\"path\\":\\""}\n\n'
    'data: {"type":"response.function_call_arguments.delta","delta":"README.md\\"}"}\n\n'
    'data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_item_1","call_id":"call_1","name":"lookup","arguments":"{\\"path\\":\\"README.md\\"}"}}\n\n'
    'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":2},"output_tokens":7}}}\n\n'
  )

  parsed, rest = _parse_sse(payload)
  assert rest == ""

  events = []
  for item in parsed:
    events.extend(_map_event(item, state))

  assert [event.type for event in events] == [
    "thinking_delta",
    "thinking_delta",
    "thinking_end",
    "text_delta",
    "text_end",
    "tool_use_start",
    "tool_use_delta",
    "tool_use_delta",
    "tool_use_end",
    "message_start",
    "usage_update",
    "message_end",
  ]

  thinking_end = events[2]
  assert thinking_end.thinking_text == "thinking\n\n"
  assert thinking_end.raw_block["thinkingSignature"] == json.dumps(
    {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking\n\n"}]},
    separators=(",", ":"),
  )

  text_end = events[4]
  assert text_end.text == "hello"
  assert text_end.raw_block["textSignature"] == _encode_text_signature_v1("msg_1", "final_answer")

  tool_use_end = events[8]
  assert tool_use_end.tool_id == "call_1|fc_item_1"
  assert tool_use_end.tool_name == "lookup"
  assert tool_use_end.tool_input == {"path": "README.md"}
  assert tool_use_end.tool_input_json == "{\"path\":\"README.md\"}"

  message_start = events[9]
  assert message_start.input_tokens == 8
  assert message_start.cache_read_tokens == 2

  usage_update = events[10]
  assert usage_update.output_tokens == 7

  message_end = events[11]
  assert message_end.stop_reason == "tool_use"


def test_normalize_messages_converts_compaction_to_text_and_truncates() -> None:
  provider = CodexProvider()
  model_info = provider.get_model_info("gpt-5.5")
  messages = [
    {"role": "user", "content": "old history"},
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "summary"},
        {"type": "text", "text": "answer"},
      ],
    },
    {"role": "user", "content": "next"},
  ]

  normalized = provider.normalize_messages(messages, model_info)

  assert len(normalized) == 2
  first_block = normalized[0]["content"][0]
  assert first_block["type"] == "text"
  assert "summary" in first_block["text"]
  assert not any(
    isinstance(b, dict) and b.get("type") == "compaction"
    for m in normalized
    for b in (m.get("content") if isinstance(m.get("content"), list) else [])
  )
