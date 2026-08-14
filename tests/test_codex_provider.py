# ruff: noqa: E402

import base64
import json
import sys
from pathlib import Path
from typing import Any

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
    "_CODEX_ORIGINATOR",
    "_CODEX_USER_AGENT",
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


def test_build_headers_uses_current_codex_subscription_identity() -> None:
  headers = codex_helpers._build_headers(None, None, "acct_123", "token")

  assert headers["originator"] == "codex_cli_rs"
  assert headers["User-Agent"] == "codex_cli_rs/0.144.0"
  assert headers["chatgpt-account-id"] == "acct_123"
  assert headers["OpenAI-Beta"] == "responses=experimental"


def test_build_headers_allows_explicit_identity_override() -> None:
  headers = codex_helpers._build_headers(
    {"originator": "configured", "User-Agent": "configured/1"},
    {"originator": "request", "User-Agent": "request/2"},
    "acct_123",
    "token",
  )

  assert headers["originator"] == "request"
  assert headers["User-Agent"] == "request/2"


def test_codex_has_active_credential_accepts_auth_token() -> None:
  provider = CodexProvider()

  assert provider.has_active_credential({"auth_mode": "oauth", "auth_token": _fake_jwt("acct_123")}) is True
  assert provider.has_active_credential({"auth_mode": "oauth", "auth_token": "   "}) is False


def test_codex_gpt55_uses_gpt5_family_metadata() -> None:
  provider = CodexProvider()

  model_info = provider.get_model_info("gpt-5.5")

  assert model_info.id == "gpt-5.5"
  assert model_info.provider == "codex"
  assert model_info.context_window == 400_000
  assert model_info.input_cost_per_mtok == 5.00
  assert model_info.output_cost_per_mtok == 30.00
  assert model_info.supports_thinking is True
  assert model_info.supports_vision is True


@pytest.mark.parametrize("model_id", ["gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6", "gpt-5.5", "gpt-5.1"])
def test_codex_gpt5x_family_window_matches_chatgpt_backend(model_id: str) -> None:
  # ChatGPT backend enforces ~370-385k input (probed live 2026-07-21); the
  # registry pins 400k total so the proactive compaction trigger (80% of
  # window) fires before the backend's real wall instead of never.
  provider = CodexProvider()

  assert provider.get_model_info(model_id).context_window == 400_000


def test_codex_terra_effective_compaction_trigger_is_reachable() -> None:
  from agent_gateway.runner_limits import effective_compaction_trigger

  provider = CodexProvider()

  trigger = effective_compaction_trigger(160_000, provider.get_model_info("gpt-5.6-terra"))

  assert trigger == 320_000


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
    # Field set and shapes come from a real captured stream: part.added always carries an
    # empty text, part.done echoes the completed part, and the server's summary text never
    # carries a trailing separator -- which is why the durable rebuild uses "\n\n".join().
    'data: {"type":"response.reasoning_summary_part.added","item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""},"sequence_number":2}\n\n'
    'data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_1","output_index":0,"summary_index":0,"delta":"thinking","sequence_number":3}\n\n'
    'data: {"type":"response.reasoning_summary_part.done","item_id":"rs_1","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":"thinking"},"sequence_number":4}\n\n'
    'data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"thinking"}]}}\n\n'
    'data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}\n\n'
    'data: {"type":"response.content_part.added","part":{"type":"output_text","text":""}}\n\n'
    'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
    'data: {"type":"response.output_item.done","item":{"type":"message","id":"msg_1","phase":"final_answer","content":[{"type":"output_text","text":"hello"}]}}\n\n'
    'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_item_1","call_id":"call_1","name":"lookup","arguments":""}}\n\n'
    'data: {"type":"response.function_call_arguments.delta","delta":"{\\"path\\":\\""}\n\n'
    'data: {"type":"response.function_call_arguments.delta","delta":"README.md\\"}"}\n\n'
    'data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_item_1","call_id":"call_1","name":"lookup","arguments":"{\\"path\\":\\"README.md\\"}"}}\n\n'
    'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":2},"output_tokens":7,"output_tokens_details":{"reasoning_tokens":3}}}}\n\n'
  )

  parsed, rest = _parse_sse(payload)
  assert rest == ""

  events = []
  for item in parsed:
    events.extend(_map_event(item, state))

  # A single summary part yields ONE thinking_delta. The mapper used to emit a second
  # "\n\n" delta on part.done, which no "\n\n".join() rebuild ever produces.
  assert [event.type for event in events] == [
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

  # Select by type rather than index: a change in how many deltas a block emits should
  # not silently re-point every assertion below it at the wrong event.
  def _only(event_type: str):
    matches = [event for event in events if event.type == event_type]
    assert len(matches) == 1, f"expected exactly one {event_type}, got {len(matches)}"
    return matches[0]

  thinking_end = _only("thinking_end")
  assert thinking_end.thinking_text == "thinking"
  assert thinking_end.raw_block["thinkingSignature"] == json.dumps(
    {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
    separators=(",", ":"),
  )

  text_end = _only("text_end")
  assert text_end.text == "hello"
  assert text_end.raw_block["textSignature"] == _encode_text_signature_v1("msg_1", "final_answer")

  tool_use_end = _only("tool_use_end")
  assert tool_use_end.tool_id == "call_1|fc_item_1"
  assert tool_use_end.tool_name == "lookup"
  assert tool_use_end.tool_input == {"path": "README.md"}
  assert tool_use_end.tool_input_json == "{\"path\":\"README.md\"}"

  message_start = _only("message_start")
  assert message_start.input_tokens == 8
  assert message_start.cache_read_tokens == 2

  usage_update = _only("usage_update")
  assert usage_update.output_tokens == 7
  assert usage_update.reasoning_tokens == 3

  message_end = _only("message_end")
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


def _reasoning_events(part_texts: list[str], *, item_id: str = "rs_1", final_summary: Any = ...) -> list[dict[str, Any]]:
  """Build a protocol-shaped reasoning stream.

  Shapes follow a real captured stream: part.added carries an empty text, deltas carry the
  content, and the server's summary parts never end with a separator. Every part gets a
  .done -- INCLUDING the last one. Omitting the final .done is exactly what hid this defect
  in the openai twin of this mapper through a full review round, because the stray separator
  only lands after the final part.
  """
  events: list[dict[str, Any]] = [
    {"type": "response.output_item.added",
     "item": {"type": "reasoning", "id": item_id, "summary": [], "encrypted_content": "ENC"}},
  ]
  for index, text in enumerate(part_texts):
    events.append({"type": "response.reasoning_summary_part.added", "item_id": item_id,
                   "summary_index": index, "part": {"type": "summary_text", "text": ""}})
    events.append({"type": "response.reasoning_summary_text.delta", "item_id": item_id,
                   "summary_index": index, "delta": text})
    events.append({"type": "response.reasoning_summary_part.done", "item_id": item_id,
                   "summary_index": index, "part": {"type": "summary_text", "text": text}})
  done_item: dict[str, Any] = {"type": "reasoning", "id": item_id, "encrypted_content": "ENC"}
  if final_summary is ...:
    done_item["summary"] = [{"type": "summary_text", "text": text} for text in part_texts]
  elif final_summary is not None:
    done_item["summary"] = final_summary
  events.append({"type": "response.output_item.done", "item": done_item})
  return events


def _drive_thinking(events: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
  """Return one (streamed, durable, signature) triple per reasoning block.

  `streamed` is the concatenation of every thinking_delta -- exactly what a streaming client
  accumulates. `durable` is the thinking_end block a transcript persists. They are built by
  different code paths and must agree.
  """
  state = _ResponsesStreamState()
  blocks: list[tuple[str, str, str]] = []
  streamed: list[str] = []
  for raw in events:
    for mapped in _map_event(raw, state):
      if mapped.type == "thinking_delta":
        streamed.append(mapped.thinking_text or "")
      elif mapped.type == "thinking_end":
        blocks.append(("".join(streamed), mapped.thinking_text or "", mapped.signature or ""))
        streamed = []
  return blocks


def test_single_part_reasoning_summary_has_no_trailing_separator() -> None:
  (streamed, durable, _), = _drive_thinking(_reasoning_events(["only"]))
  assert streamed == durable == "only"


def test_multi_part_reasoning_summary_streams_what_it_persists() -> None:
  (streamed, durable, _), = _drive_thinking(_reasoning_events(["first", "second", "third"]))
  assert durable == "first\n\nsecond\n\nthird"
  assert streamed == durable


def test_consecutive_reasoning_items_start_without_a_leading_separator() -> None:
  """One response can carry several reasoning items -- a live capture showed six.

  Each output_item.added opens a fresh block, so the second item must not inherit a
  separator from the first.
  """
  events = _reasoning_events(["a1", "a2"], item_id="rs_1") + _reasoning_events(["b1", "b2"], item_id="rs_2")
  blocks = _drive_thinking(events)
  assert [durable for _, durable, _ in blocks] == ["a1\n\na2", "b1\n\nb2"]
  assert [streamed for streamed, _, _ in blocks] == [durable for _, durable, _ in blocks]


def test_reasoning_signature_replays_the_server_item_verbatim() -> None:
  """The signature is what same-model replay sends back to the model.

  It is serialized from the event's item, so the mapper's own text bookkeeping must never
  appear in it -- live traffic carries ~4KB of encrypted_content per reasoning item that has
  to survive byte for byte.
  """
  (_, _, signature), = _drive_thinking(_reasoning_events(["first", "second"]))
  replayed = json.loads(signature)
  assert [part["text"] for part in replayed["summary"]] == ["first", "second"]
  assert replayed["encrypted_content"] == "ENC"


def test_reasoning_without_a_final_summary_keeps_the_streamed_text() -> None:
  """The one path where this fix changes the DURABLE text, not just the stream.

  When the final item omits a summary list, finalization cannot rebuild and keeps whatever
  was accumulated. That text reaches the model again on cross-model replay, where
  normalize_messages converts a thinking block to plain assistant text -- so a stray trailing
  separator here would become model input, not just a display artifact.
  """
  (streamed, durable, _), = _drive_thinking(_reasoning_events(["first", "second"], final_summary=None))
  assert durable == "first\n\nsecond"
  assert streamed == durable


def _drive_tool_call(
  seeded_arguments: str,
  deltas: list[str],
  *,
  done_arguments: str | None = None,
) -> tuple[str, str]:
  """Return (streamed, durable) argument JSON for one function call.

  `done_arguments=None` omits the field from output_item.done, which forces finalization to
  fall back on the accumulator (`current_tool_json`) -- the only way to observe what the
  delta handler actually built.
  """
  state = _ResponsesStreamState()
  events: list[dict[str, Any]] = [
    {"type": "response.output_item.added",
     "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
              "name": "lookup", "arguments": seeded_arguments}},
  ]
  events.extend({"type": "response.function_call_arguments.delta", "delta": d} for d in deltas)
  done_item: dict[str, Any] = {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                               "name": "lookup"}
  if done_arguments is not None:
    done_item["arguments"] = done_arguments
  events.append({"type": "response.output_item.done", "item": done_item})

  streamed: list[str] = []
  durable = ""
  for raw in events:
    for mapped in _map_event(raw, state):
      if mapped.type == "tool_use_delta":
        streamed.append(mapped.tool_input_json or "")
      elif mapped.type == "tool_use_end":
        durable = mapped.tool_input_json or ""
  return "".join(streamed), durable


def test_seeded_tool_arguments_are_replaced_by_authoritative_deltas() -> None:
  """A backend that seeds arguments AND repeats them as deltas must not double them.

  Pre-fix the seed stayed in the accumulator and the deltas appended to it, yielding
  '{"path":"README.md"}{"path":"README.md"}' -- which no longer parses as one object.
  Matches the guard openai_responses_helpers.py already carries.
  """
  streamed, durable = _drive_tool_call('{"path":"README.md"}', ['{"path":"', 'README.md"}'])
  assert durable == '{"path":"README.md"}'
  assert streamed == durable


def test_tool_arguments_without_deltas_keep_the_seeded_snapshot() -> None:
  """The guard must not clear the seed when no deltas ever arrive.

  This is the xAI shape -- complete call in output_item.added, no separate delta events --
  so a fix that reset the accumulator unconditionally would erase the arguments entirely.
  """
  _, durable = _drive_tool_call('{"path":"README.md"}', [])
  assert durable == '{"path":"README.md"}'


def test_authoritative_done_arguments_still_win_over_the_accumulator() -> None:
  """function_call_arguments.done overwrites whatever the deltas built."""
  state = _ResponsesStreamState()
  for raw in [
    {"type": "response.output_item.added",
     "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
              "name": "lookup", "arguments": ""}},
    {"type": "response.function_call_arguments.delta", "delta": '{"path":"tru'},
    {"type": "response.function_call_arguments.done", "arguments": '{"path":"truncated.md"}'},
    {"type": "response.output_item.done",
     "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup"}},
  ]:
    events = _map_event(raw, state)
  assert [e.tool_input_json for e in events if e.type == "tool_use_end"] == [
    '{"path":"truncated.md"}'
  ]


def test_consecutive_tool_calls_each_get_their_own_arguments() -> None:
  """Two calls on ONE state must not contaminate each other.

  Both calls carry a seed AND non-empty deltas -- that combination is what makes this test
  bite. What isolates the calls is the reset at call 2's output_item.added, NOT anything at
  finalization, which is why no reset lives there. Without that reset, call 1's flag survives,
  call 2 skips the replace branch, and its deltas concatenate onto its seed as
  '{"b":2}{"b":2}'. Reviews r1 and r2 both flagged weaker versions of this test: r1 for
  claiming to guard a finalization reset it never exercised, r2 for giving call 2 no deltas,
  which left the flag unread and the assertion vacuous.
  """
  state = _ResponsesStreamState()
  durable: list[str] = []
  for raw in [
    {"type": "response.output_item.added",
     "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
              "name": "lookup", "arguments": '{"a":1}'}},
    {"type": "response.function_call_arguments.delta", "delta": '{"a":'},
    {"type": "response.function_call_arguments.delta", "delta": '1}'},
    {"type": "response.output_item.done",
     "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup"}},
    {"type": "response.output_item.added",
     "item": {"type": "function_call", "id": "fc_2", "call_id": "call_2",
              "name": "lookup", "arguments": '{"b":2}'}},
    # Call 2 carries BOTH a seed and real deltas: that is what exercises the seed-site reset.
    # With call 1's flag still set, the replace branch is skipped and these concatenate onto
    # the seed as '{"b":2}{"b":2}'. (r2 caught the earlier version, which gave call 2 no
    # deltas and so never read the flag at all.)
    {"type": "response.function_call_arguments.delta", "delta": '{"b":'},
    {"type": "response.function_call_arguments.delta", "delta": '2}'},
    {"type": "response.output_item.done",
     "item": {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "lookup"}},
  ]:
    durable.extend(e.tool_input_json or "" for e in _map_event(raw, state) if e.type == "tool_use_end")
  assert durable == ['{"a":1}', '{"b":2}']


def test_empty_argument_delta_does_not_discard_the_seeded_snapshot() -> None:
  """Over-correction guard: a zero-length delta must not wipe a valid seed.

  The replace-on-first-delta branch is gated on a non-empty delta. Without that gate an
  empty delta clears the snapshot, and a finalization that omits arguments then degrades the
  whole call to "{}" -- a regression the fix itself would have introduced. Review r1 caught
  this; the same gate was applied to the OpenAI twin, which had it too.
  """
  _, durable = _drive_tool_call('{"path":"README.md"}', [""])
  assert durable == '{"path":"README.md"}'


def test_finalized_arguments_survive_a_stale_completed_item() -> None:
  """The accumulator outranks output_item.done.arguments, and must keep doing so.

  function_call_arguments.done is the streaming contract's finalization event and writes
  through to the accumulator. Review r1 proposed reordering this to match the OpenAI mapper;
  review r2 refuted it with this case -- a stale item snapshot would silently replace
  explicitly finalized arguments. The mappers disagree here on purpose.
  """
  state = _ResponsesStreamState()
  durable: list[str] = []
  for raw in [
    {"type": "response.output_item.added",
     "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
              "name": "lookup", "arguments": ""}},
    {"type": "response.function_call_arguments.delta", "delta": '{"path":"README.md"}'},
    {"type": "response.function_call_arguments.done", "arguments": '{"path":"README.md"}'},
    {"type": "response.output_item.done",
     "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
              "name": "lookup", "arguments": '{"path":"READ"}'}},
  ]:
    durable.extend(e.tool_input_json or "" for e in _map_event(raw, state) if e.type == "tool_use_end")
  assert durable == ['{"path":"README.md"}']


def test_accumulator_reaches_finalization_when_the_item_omits_arguments() -> None:
  """Fallback path: nothing on the completed item -> use what the deltas built."""
  _, durable = _drive_tool_call("", ['{"path":"', 'README.md"}'])
  assert durable == '{"path":"README.md"}'
