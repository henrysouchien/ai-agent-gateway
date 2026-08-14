# ruff: noqa: E402

import asyncio
import json
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, EventLog
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.openai_history_fence import REASONING_SIGNATURE_MARKER, TEXT_SIGNATURE_MARKER
from agent_gateway.provider_summarize import provider_summarize
from agent_gateway.providers import OpenAIProvider, ThinkingLevel
from agent_gateway.providers.openai import OpenAIConfigurationError
from agent_gateway.providers.openai_responses_helpers import _ResponsesStreamState, map_event
from tests.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)


class _EventStream:
  def __init__(self, events: list[Any]):
    self.events = events

  def __aiter__(self):
    self._iterator = iter(self.events)
    return self

  async def __anext__(self):
    try:
      return next(self._iterator)
    except StopIteration as exc:
      raise StopAsyncIteration from exc


class _Responses:
  def __init__(self, events: list[Any]):
    self.events = events
    self.params: dict[str, Any] | None = None

  async def create(self, **params: Any):
    self.params = params
    return _EventStream(self.events)


class _StreamingClient:
  def __init__(self, events: list[Any]):
    self.responses = _Responses(events)


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch: pytest.MonkeyPatch):
  monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch):
  module = types.ModuleType("openai")

  class FakeAsyncOpenAI:
    def __init__(self, **kwargs: Any):
      self.kwargs = kwargs
      self.responses = SimpleNamespace(create=lambda **_kwargs: None)

  module.AsyncOpenAI = FakeAsyncOpenAI
  monkeypatch.setitem(sys.modules, "openai", module)
  return FakeAsyncOpenAI


def test_provider_does_not_fall_back_to_process_api_key(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("OPENAI_API_KEY", "process-service-secret")
  _install_fake_openai(monkeypatch)
  provider = OpenAIProvider()

  assert provider.has_active_credential({"auth_mode": "api", "api_key": ""}) is False
  with pytest.raises(RuntimeError, match="No OpenAI api credential configured"):
    provider.create_client({"auth_mode": "api", "api_key": ""})


def test_client_ignores_ambient_route_organization_and_project(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.invalid/v1")
  monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
  monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
  fake = _install_fake_openai(monkeypatch)

  client = OpenAIProvider().create_client({
    "auth_mode": "api",
    "api_key": "bound-api-key",
  })

  assert isinstance(client, fake)
  assert client.kwargs["api_key"] == "bound-api-key"
  assert client.kwargs["base_url"] == "https://api.openai.com/v1"
  assert client.kwargs["organization"] == ""
  assert client.kwargs["project"] == ""


def test_client_passes_bound_route_organization_and_project(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("OPENAI_BASE_URL", "https://ambient.invalid/v1")
  monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
  monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
  fake = _install_fake_openai(monkeypatch)

  client = OpenAIProvider().create_client({
    "auth_mode": "oauth",
    "auth_token": "bound-oauth-token",
    "baseURL": "https://api.openai.com",
    "organization": "bound-org",
    "project": "bound-project",
  })

  assert isinstance(client, fake)
  assert client.kwargs["api_key"] == "bound-oauth-token"
  assert client.kwargs["base_url"] == "https://api.openai.com/v1"
  assert client.kwargs["organization"] == "bound-org"
  assert client.kwargs["project"] == "bound-project"


@pytest.mark.parametrize("key", ["base_url", "baseURL", "api_base_url", "api_base"])
@pytest.mark.parametrize("url", ["https://api.openai.com", "https://api.openai.com/v1", "https://API.OPENAI.COM/v1/"])
def test_official_base_url_aliases_canonicalize(monkeypatch: pytest.MonkeyPatch, key: str, url: str) -> None:
  fake = _install_fake_openai(monkeypatch)
  client = OpenAIProvider().create_client({"api_key": "sk-test", key: url})
  assert isinstance(client, fake)
  assert client.kwargs["base_url"] == "https://api.openai.com/v1"


@pytest.mark.parametrize("url", [
  "http://api.openai.com/v1",
  "https://api.openai.com.evil.test/v1",
  "https://user:pass@api.openai.com/v1",
  "https://api.openai.com/v1?proxy=yes",
  "https://api.openai.com/v1/responses",
  "https://api.chutes.ai/v1",
])
def test_non_official_or_unsafe_base_urls_fail(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
  _install_fake_openai(monkeypatch)
  with pytest.raises(OpenAIConfigurationError, match="Responses-only|base_url"):
    OpenAIProvider().create_client({"api_key": "sk-test", "base_url": url})


def test_nonempty_compat_fails(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_fake_openai(monkeypatch)
  with pytest.raises(OpenAIConfigurationError, match="compatibility overrides"):
    OpenAIProvider().create_client({"api_key": "sk-test", "compat": {"streaming": True}})


def test_supported_sdk_contract_is_present() -> None:
  from openai import AsyncOpenAI
  assert hasattr(AsyncOpenAI, "responses")


def test_request_contract_reasoning_tools_and_local_history() -> None:
  provider = OpenAIProvider()
  params = provider.build_request_params(
    model="gpt-5.6-terra",
    messages=[
      {"role": "user", "content": [{"type": "text", "text": "look up x"}]},
      {
        "role": "assistant",
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "content": [{"type": "tool_use", "id": "call_1|fc_1", "name": "lookup", "input": {"q": "x"}}],
      },
      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1|fc_1", "content": "ok"}]},
    ],
    system_prompt=[("first", False), ("second", True)],
    tools=[{"name": "lookup", "description": "Lookup", "input_schema": {"type": "object"}}],
    max_tokens=2048,
    thinking_level=ThinkingLevel.LOW,
  )
  assert params["stream"] is True
  assert params["store"] is False
  assert params["include"] == ["reasoning.encrypted_content"]
  assert params["instructions"] == "first\n\nsecond"
  assert params["max_output_tokens"] == 2048
  assert params["reasoning"] == {"effort": "low", "summary": "auto"}
  assert params["tools"] == [{
    "type": "function", "name": "lookup", "description": "Lookup",
    "parameters": {"type": "object"}, "strict": False,
  }]
  assert params["tool_choice"] == "auto"
  assert params["parallel_tool_calls"] is True
  assert {item["type"] for item in params["input"] if "type" in item} >= {"function_call", "function_call_output"}
  for removed in ("messages", "max_completion_tokens", "max_tokens", "stream_options", "reasoning_effort", "previous_response_id"):
    assert removed not in params


def test_reasoning_matrix_none_and_effort_are_nested() -> None:
  provider = OpenAIProvider()
  none_params = provider.build_request_params(
    model="gpt-5.6", messages=[], system_prompt=None, tools=[], max_tokens=100,
    thinking_level=ThinkingLevel.NONE,
  )
  assert none_params["reasoning"] == {"effort": "none"}
  high_params = provider.build_request_params(
    model="gpt-5.4", messages=[], system_prompt=None, tools=[], max_tokens=100,
    thinking_level=ThinkingLevel.HIGH,
  )
  # Responses nests reasoning; a top-level reasoning_effort is what gpt-5.6 rejects
  # alongside tools on chat/completions.
  assert high_params["reasoning"]["effort"] == "high"
  assert "reasoning_effort" not in high_params


# Every shipped OpenAI model now supports reasoning and function tools (the gpt-4o /
# o1 / o3-mini / o1-mini rows were dropped once we settled on the gpt-5.x family).
# The non-reasoning and text-only guards in build_request_params are still live code
# with no shipped model that reaches them, so exercise them with synthetic capability
# rows rather than leaving the branches uncovered.
def _synthetic_model(monkeypatch, provider: OpenAIProvider, **overrides) -> None:
  base = provider.get_model_info("gpt-5.6")
  from dataclasses import replace as _replace

  monkeypatch.setattr(provider, "get_model_info", lambda _m: _replace(base, **overrides))


def test_non_reasoning_model_omits_reasoning(monkeypatch) -> None:
  provider = OpenAIProvider()
  # ModelInfo.__post_init__ forces supports_thinking back to True whenever
  # thinking_mode != "none", so both must be set to build a non-reasoning row.
  _synthetic_model(monkeypatch, provider, supports_thinking=False, thinking_mode="none")
  params = provider.build_request_params(
    model="synthetic-nonreasoning", messages=[], system_prompt=None, tools=[], max_tokens=100,
    thinking_level=ThinkingLevel.HIGH,
  )
  assert "reasoning" not in params


def test_text_only_model_rejects_required_tools(monkeypatch) -> None:
  provider = OpenAIProvider()
  compat = dict(provider.get_model_info("gpt-5.6").compat or {})
  compat["supportsResponsesFunctionTools"] = False
  _synthetic_model(monkeypatch, provider, supports_tool_use=False, compat=compat)
  params = provider.build_request_params(
    model="synthetic-textonly", messages=[{"role": "user", "content": "hello"}],
    system_prompt=None, tools=[], max_tokens=100,
  )
  assert "tools" not in params
  with pytest.raises(ValueError, match="does not support Responses function tools"):
    provider.build_request_params(
      model="synthetic-textonly", messages=[], system_prompt=None,
      tools=[{"name": "x", "input_schema": {"type": "object"}}], max_tokens=100,
    )


def test_unverified_model_is_rejected() -> None:
  with pytest.raises(ValueError, match="no verified Responses capability row"):
    OpenAIProvider().get_model_info("gpt-5.2")


def test_legacy_and_native_history_conversion() -> None:
  provider = OpenAIProvider()
  native_text_signature = json.dumps({
    "v": TEXT_SIGNATURE_MARKER, "item_id": "msg_real", "status": "completed", "phase": "final_answer",
  })
  native_reasoning_signature = json.dumps({
    "v": REASONING_SIGNATURE_MARKER,
    "item": {"type": "reasoning", "id": "rs_1", "status": "completed", "encrypted_content": "FAKE_CIPHERTEXT"},
  })
  params = provider.build_request_params(
    model="gpt-5.6",
    messages=[{
      "role": "assistant", "provider": "openai", "model": "gpt-5.6",
      "content": [
        {"type": "thinking", "thinking": "", "signature": native_reasoning_signature},
        {"type": "text", "text": "signed", "textSignature": native_text_signature},
        {"type": "text", "text": "legacy", "signature": "reasoning_content"},
      ],
    }],
    system_prompt=None, tools=[], max_tokens=100,
  )
  assert params["input"][0]["type"] == "reasoning"
  assert params["input"][0]["encrypted_content"] == "FAKE_CIPHERTEXT"
  assert params["input"][1]["id"] == "msg_real"
  assert params["input"][1]["phase"] == "final_answer"
  assert params["input"][2] == {"role": "assistant", "content": "legacy"}


def test_malformed_signatures_are_not_forwarded_as_opaque_items() -> None:
  params = OpenAIProvider().build_request_params(
    model="gpt-5.6",
    messages=[{
      "role": "assistant", "provider": "openai", "model": "gpt-5.6",
      "content": [
        {"type": "thinking", "thinking": "hidden", "signature": '{"v":"wrong","encrypted_content":"leak"}'},
        {"type": "text", "text": "safe", "textSignature": '{"v":"wrong","id":"fake"}'},
      ],
    }],
    system_prompt=None, tools=[], max_tokens=100,
  )
  assert params["input"] == [{"role": "assistant", "content": "safe"}]
  assert "leak" not in json.dumps(params)


def test_stream_maps_sdk_objects_and_terminal_usage_once() -> None:
  events = [
    SimpleNamespace(type="response.output_item.added", item=SimpleNamespace(type="message", id="msg_1", status="in_progress", content=[])),
    {"type": "response.content_part.added", "part": {"type": "output_text", "text": "", "annotations": []}},
    {"type": "response.output_text.delta", "delta": "hello"},
    {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1", "status": "completed", "content": [{"type": "output_text", "text": "hello", "annotations": []}]}},
    {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 25}, "output_tokens": 40, "output_tokens_details": {"reasoning_tokens": 10}}}},
  ]
  client = _StreamingClient(events)
  collected = asyncio.run(_collect(OpenAIProvider(), client))
  assert [event.type for event in collected] == ["text_delta", "text_end", "message_start", "usage_update", "message_end"]
  assert collected[1].raw_block["textSignature"].startswith('{"v":"openai.responses.text.v1"')
  assert (collected[2].input_tokens, collected[2].cache_read_tokens) == (75, 25)
  assert (collected[3].output_tokens, collected[3].reasoning_tokens) == (40, 10)


def test_failed_response_emits_billable_usage_and_terminal_event_before_error() -> None:
  client = _StreamingClient([{
    "type": "response.failed",
    "response": {
      "status": "failed",
      "error": {"message": "request failed after inference"},
      "usage": {
        "input_tokens": 18,
        "input_tokens_details": {"cached_tokens": 3},
        "output_tokens": 7,
        "output_tokens_details": {"reasoning_tokens": 2},
      },
    },
  }])

  async def collect_failed() -> list[Any]:
    emitted = []
    with pytest.raises(RuntimeError, match="request failed after inference"):
      async for event in OpenAIProvider().stream(client, {"model": "gpt-5.6"}):
        emitted.append(event)
    return emitted

  collected = asyncio.run(collect_failed())
  assert [event.type for event in collected] == ["message_start", "usage_update", "message_end"]
  assert (collected[0].input_tokens, collected[0].cache_read_tokens) == (15, 3)
  assert (collected[1].output_tokens, collected[1].reasoning_tokens) == (7, 2)
  assert collected[2].stop_reason == "error"


async def _collect(provider: OpenAIProvider, client: _StreamingClient):
  return [event async for event in provider.stream(client, {"model": "gpt-4o"})]


def test_complete_function_call_added_and_done_is_not_duplicated() -> None:
  state = _ResponsesStreamState()
  added = map_event({
    "type": "response.output_item.added",
    "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": '{"q":"x"}'},
  }, state)
  done = map_event({
    "type": "response.output_item.done",
    "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": '{"q":"x"}'},
  }, state)
  assert [event.type for event in added] == ["tool_use_start"]
  assert [event.type for event in done] == ["tool_use_delta", "tool_use_end"]
  assert done[1].tool_input == {"q": "x"}


def test_empty_argument_delta_does_not_discard_the_seeded_snapshot() -> None:
  """A zero-length delta must not clear a seeded arguments snapshot.

  The replace-on-first-delta branch is gated on a non-empty delta. Ungated, an empty delta
  wipes the seed and finalization degrades the call to "{}". This mirrors the codex-side
  guard; review r3 flagged that the one-line OpenAI gate was otherwise mutation-unprotected,
  since the existing lifecycle tests send no delta events at all.
  """
  state = _ResponsesStreamState()
  for raw in [
    {"type": "response.output_item.added",
     "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1",
              "name": "lookup", "arguments": '{"q":"x"}'}},
    {"type": "response.function_call_arguments.delta", "delta": ""},
  ]:
    map_event(raw, state)
  assert state.current_tool_json == '{"q":"x"}'
  assert state.saw_argument_delta is False

  done = map_event({
    "type": "response.output_item.done",
    "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup"},
  }, state)
  assert [event.tool_input for event in done if event.type == "tool_use_end"] == [{"q": "x"}]


def test_complete_function_call_done_only_has_full_lifecycle() -> None:
  events = map_event({
    "type": "response.output_item.done",
    "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": '{"q":"x"}'},
  }, _ResponsesStreamState())
  assert [event.type for event in events] == ["tool_use_start", "tool_use_delta", "tool_use_end"]
  assert events[-1].tool_input == {"q": "x"}


def test_context_length_responses_body_is_classified() -> None:
  class ResponsesError(Exception):
    body = {"error": {"type": "invalid_request_error", "code": "context_length_exceeded", "message": "input too long"}}
  assert OpenAIProvider().is_context_length_error(ResponsesError("bad request")) is True


@pytest.mark.parametrize("message", [
  "Your request exceeds the model's context window.",
  "input-too-long",
])
def test_context_length_responses_message_shapes_are_classified(message: str) -> None:
  class ResponsesError(Exception):
    body = {"error": {"type": "invalid_request_error", "message": message}}

  assert OpenAIProvider().is_context_length_error(ResponsesError("bad request")) is True


def test_output_token_parameter_error_is_not_classified_as_context_length() -> None:
  class ResponsesError(Exception):
    body = {
      "error": {
        "code": "invalid_value",
        "message": "max_output_tokens is above this model's token limit",
      }
    }

  assert OpenAIProvider().is_context_length_error(ResponsesError("bad request")) is False


def test_two_part_reasoning_summary_emits_client_separator() -> None:
  state = _ResponsesStreamState()
  events: list[Any] = []
  for event in (
    {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}},
    {"type": "response.reasoning_summary_part.added", "part": {"type": "summary_text", "text": ""}},
    {"type": "response.reasoning_summary_text.delta", "delta": "first"},
    {"type": "response.reasoning_summary_part.done"},
    {"type": "response.reasoning_summary_part.added", "part": {"type": "summary_text", "text": ""}},
    {"type": "response.reasoning_summary_text.delta", "delta": "second"},
    # A real stream emits part.done for the FINAL part too. Omitting it here hid a
    # trailing-separator regression: the client stream ended "second\n\n" while the
    # durable block ended "second".
    {"type": "response.reasoning_summary_part.done"},
    {
      "type": "response.output_item.done",
      "item": {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [
          {"type": "summary_text", "text": "first"},
          {"type": "summary_text", "text": "second"},
        ],
        "encrypted_content": "FAKE_CIPHERTEXT",
      },
    },
  ):
    events.extend(map_event(event, state))

  client_thinking = "".join(
    event.thinking_text for event in events if event.type == "thinking_delta"
  )
  durable_thinking = next(event.thinking_text for event in events if event.type == "thinking_end")
  assert client_thinking == durable_thinking == "first\n\nsecond"


def test_single_part_reasoning_summary_has_no_trailing_separator() -> None:
  state = _ResponsesStreamState()
  events: list[Any] = []
  for event in (
    {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}},
    {"type": "response.reasoning_summary_part.added", "part": {"type": "summary_text", "text": ""}},
    {"type": "response.reasoning_summary_text.delta", "delta": "only"},
    {"type": "response.reasoning_summary_part.done"},
    {
      "type": "response.output_item.done",
      "item": {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "only"}],
        "encrypted_content": "FAKE_CIPHERTEXT",
      },
    },
  ):
    events.extend(map_event(event, state))

  client_thinking = "".join(
    event.thinking_text for event in events if event.type == "thinking_delta"
  )
  durable_thinking = next(event.thinking_text for event in events if event.type == "thinking_end")
  assert client_thinking == durable_thinking == "only"


def test_private_durable_replay_storage_and_marker(tmp_path: Path) -> None:
  path = tmp_path / "sessions" / "run.jsonl"
  log = AgentSessionLog(path=path)
  path.chmod(0o644)
  signature = json.dumps({
    "v": REASONING_SIGNATURE_MARKER,
    "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": "FAKE_CIPHERTEXT"},
  })
  asyncio.run(log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "thinking", "thinking": "", "signature": signature}],
  }))
  assert stat.S_IMODE(path.stat().st_mode) == 0o600
  assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
  assert "FAKE_CIPHERTEXT" in path.read_text()
  assert '"openai_history_version":"responses-v1"' in path.read_text()


def test_runner_executes_responses_tool_loop_and_replays_function_output() -> None:
  response_batches = [
    [
      {"type": "response.output_item.added", "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": '{"q":"x"}'}},
      {"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup", "arguments": '{"q":"x"}'}},
      {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 5, "output_tokens": 2}}},
    ],
    [
      {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_2", "status": "in_progress", "content": []}},
      {"type": "response.content_part.added", "part": {"type": "output_text", "text": "", "annotations": []}},
      {"type": "response.output_text.delta", "delta": "final"},
      {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_2", "status": "completed", "content": [{"type": "output_text", "text": "final", "annotations": []}]}},
      {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 9, "output_tokens": 3}}},
    ],
  ]

  class Responses:
    def __init__(self):
      self.requests: list[dict[str, Any]] = []

    async def create(self, **params: Any):
      self.requests.append(params)
      return _EventStream(response_batches[len(self.requests) - 1])

  responses = Responses()
  client = SimpleNamespace(responses=responses)

  class Provider(OpenAIProvider):
    def create_client(self, config: dict[str, Any], *, timeout: float | None = None):
      return client

    async def close_client(self, client: Any, timeout: float = 2.0) -> None:
      return None

  class Dispatcher:
    async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
      assert (tool_name, tool_input) == ("lookup", {"q": "x"})
      return {"ok": True}, None

    def requires_approval(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
      return False

  event_log = EventLog(session_id="responses-tool-loop")
  provider = Provider()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=Dispatcher(),  # type: ignore[arg-type]
    session_id="responses-tool-loop",
    capability_execution=stub_runner_capability_execution(
      provider=provider,
      model="gpt-5.6-terra",
      effort="low",
      auth_config={"api_key": "sk-test", "max_tokens": 512},
    ),
    get_tool_definitions=lambda: [{"name": "lookup", "description": "lookup", "input_schema": {"type": "object"}}],
    user_id="alice",
    billing_mode="byok",
    rate_table_version="test",
    emit_session_recap=False,
  )
  asyncio.run(runner.run([{"role": "user", "content": "look up x"}], max_turns=3))
  assert len(responses.requests) == 2
  second_input = responses.requests[1]["input"]
  assert any(item.get("type") == "function_call" for item in second_input)
  assert any(item.get("type") == "function_call_output" and '"ok": true' in item["output"] for item in second_input)
  assert any(entry.event.get("type") == "stream_complete" for entry in event_log.entries)


def test_openai_compaction_summary_uses_responses_with_effort_none() -> None:
  events = [
    {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_summary", "status": "in_progress", "content": []}},
    {"type": "response.content_part.added", "part": {"type": "output_text", "text": "", "annotations": []}},
    {"type": "response.output_text.delta", "delta": "compact summary"},
    {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_summary", "status": "completed", "content": [{"type": "output_text", "text": "compact summary", "annotations": []}]}},
    {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 8, "output_tokens": 2}}},
  ]
  client = _StreamingClient(events)

  class Provider(OpenAIProvider):
    def create_client(self, config: dict[str, Any], *, timeout: float | None = None):
      return client

    async def close_client(self, client: Any, timeout: float = 2.0) -> None:
      return None

  result = asyncio.run(provider_summarize(
    capability_execution=stub_runner_capability_execution(
      provider=Provider(),
      model="gpt-5.6",
      effort="none",
      auth_config={"api_key": "sk-test"},
    ),
    messages=[{"role": "user", "content": "summarize"}],
    system_prompt="summarize",
    max_tokens=128,
  ))
  assert result.text == "compact summary"
  assert client.responses.params is not None
  assert client.responses.params["reasoning"] == {"effort": "none"}


def test_declared_adapter_support_matches_responses_only_implementation() -> None:
  from agent_gateway.model_registry import INITIAL_MODEL_REGISTRY

  declaration = OpenAIProvider.adapter_route_support()

  assert declaration is not None
  assert declaration.adapter == "openai.responses"
  assert declaration.provider == "openai"
  assert declaration.protocol_profiles == frozenset({"responses.reasoning"})
  assert declaration.routes == frozenset({"openai.public"})

  # The declaration admits the packaged Responses entry and refuses the
  # Risk-local chat-completions execution identity.
  assert declaration.supports(INITIAL_MODEL_REGISTRY.require("openai.gpt-5-6"))
  assert not declaration.supports(
    INITIAL_MODEL_REGISTRY.require("openai.gpt-5-4-mini-sdk")
  )

  # Behavior matches the declaration: requests are Responses-shaped, never
  # Chat Completions-shaped.
  params = OpenAIProvider().build_request_params(
    model="gpt-5.6",
    messages=[{"role": "user", "content": "hello"}],
    system_prompt="system",
    tools=[],
    max_tokens=128,
  )
  assert "input" in params
  assert "instructions" in params
  assert "max_output_tokens" in params
  assert "messages" not in params
