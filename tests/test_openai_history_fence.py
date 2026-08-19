"""Tests for the Phase 0 OpenAI Responses rollback-fence primitives.

Spec: docs/design/gateway-openai-responses-migration-plan.md section 10.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_gateway.autonomous as autonomous_module
from agent_gateway import AgentSessionLog, SessionContextBuilder
from agent_gateway.event_log import EventLog
from agent_gateway.openai_history_fence import (
  DURABLE_HISTORY_VERSION_KEY,
  OpenAISessionEpochError,
  REASONING_SIGNATURE_MARKER,
  RESPONSES_HISTORY_VERSION,
  TEXT_SIGNATURE_MARKER,
  contains_openai_responses_history,
  scope_provider_session_id,
)
from agent_gateway.providers import OpenAIProvider
from agent_gateway.runner_introspection import derive_sub_agent_id
from agent_gateway.server_chat_transcripts import (
  _compute_session_recap_payload,
  _read_session_transcript_events,
)
from agent_gateway.session import GatewaySession, SessionStream
from tests.capability_execution_test_support import stub_bound_capability_execution


class _BoundOpenAIProvider(OpenAIProvider):
  def create_client(self, config, *, timeout=None):
    _ = config, timeout
    raise AssertionError("provider client must not be created by these seam tests")


def _openai_bound_execution(model: str = "gpt-5.6") -> dict:
  return {
    "capability_execution": stub_bound_capability_execution(
      provider=_BoundOpenAIProvider(),
      model=model,
      effort="none",
      run_mode="autonomous",
      auth_config={
        "max_tokens": 16_000,
        "auth_mode": "api",
        "api_key": "test-openai-key",
      },
    ),
    "session": _gateway_session(),
  }


# --- contains_openai_responses_history ------------------------------------


def test_empty_and_none_payloads_are_not_responses_history():
  assert contains_openai_responses_history(None) is False
  assert contains_openai_responses_history([]) is False
  assert contains_openai_responses_history({}) is False
  assert contains_openai_responses_history("") is False


def test_plain_chat_history_is_not_flagged():
  messages = [
    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "hi"},
        {"type": "tool_use", "id": "call_1", "name": "read", "input": {}},
      ],
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "ok"}]},
  ]
  assert contains_openai_responses_history(messages) is False


def test_reasoning_signature_marker_is_detected():
  messages = [
    {
      "role": "assistant",
      "content": [{"type": "thinking", "thinking": "...", "signature": REASONING_SIGNATURE_MARKER}],
    }
  ]
  assert contains_openai_responses_history(messages) is True


def test_text_signature_marker_is_detected():
  messages = [
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "hi", "signature": TEXT_SIGNATURE_MARKER}],
    }
  ]
  assert contains_openai_responses_history(messages) is True


def test_marker_embedded_in_a_larger_signature_payload_is_detected():
  # Signatures are serialized envelopes; the marker appears inside them.
  signature = f'{{"v":"{TEXT_SIGNATURE_MARKER}","item_id":"msg_123"}}'
  messages = [{"role": "assistant", "content": [{"type": "text", "signature": signature}]}]
  assert contains_openai_responses_history(messages) is True


def test_durable_history_version_is_detected():
  record = {"session": "s1", DURABLE_HISTORY_VERSION_KEY: RESPONSES_HISTORY_VERSION}
  assert contains_openai_responses_history(record) is True


def test_future_responses_history_version_is_detected():
  record = {DURABLE_HISTORY_VERSION_KEY: "responses-v2"}
  assert contains_openai_responses_history(record) is True


def test_non_responses_history_version_is_not_flagged():
  record = {DURABLE_HISTORY_VERSION_KEY: "chat-prep-v1"}
  assert contains_openai_responses_history(record) is False


def test_marker_nested_deep_in_events_is_detected():
  events = {"events": [{"payload": {"blocks": [{"signature": REASONING_SIGNATURE_MARKER}]}}]}
  assert contains_openai_responses_history(events) is True


def test_single_message_not_wrapped_in_a_list_is_supported():
  message = {"role": "assistant", "content": [{"signature": TEXT_SIGNATURE_MARKER}]}
  assert contains_openai_responses_history(message) is True


def test_deeply_nested_payload_does_not_recurse_without_bound():
  payload: dict = {}
  cursor = payload
  for _ in range(500):
    child: dict = {}
    cursor["next"] = child
    cursor = child
  # Must terminate (depth-bounded) rather than raising RecursionError.
  assert contains_openai_responses_history(payload) is False


# --- scope_provider_session_id --------------------------------------------


@pytest.mark.parametrize("provider", ["anthropic", "codex", "xai", "fixture", "Anthropic"])
def test_non_openai_providers_are_returned_byte_for_byte(provider):
  base = "agentsess_research_producer_1"
  assert scope_provider_session_id(base, provider=provider, durable=True, openai_epoch="chat-prep-v1") is base


def test_non_openai_provider_does_not_require_an_epoch():
  base = "agentsess_analyst_1"
  assert scope_provider_session_id(base, provider="codex", durable=True, openai_epoch=None) == base


def test_non_durable_openai_session_keeps_its_identifier():
  base = "chatsess_abc123"
  assert scope_provider_session_id(base, provider="openai", durable=False, openai_epoch=None) == base


def test_durable_openai_session_is_scoped_by_epoch():
  assert (
    scope_provider_session_id("sess_1", provider="openai", durable=True, openai_epoch="chat-prep-v1")
    == "sess_1--openai-chat-prep-v1"
  )


def test_scoping_is_idempotent_for_the_same_epoch():
  once = scope_provider_session_id("sess_1", provider="openai", durable=True, openai_epoch="responses-v1")
  twice = scope_provider_session_id(once, provider="openai", durable=True, openai_epoch="responses-v1")
  assert once == twice == "sess_1--openai-responses-v1"


def test_rescoping_to_a_different_epoch_is_rejected():
  scoped = scope_provider_session_id("sess_1", provider="openai", durable=True, openai_epoch="chat-prep-v1")
  with pytest.raises(OpenAISessionEpochError):
    scope_provider_session_id(scoped, provider="openai", durable=True, openai_epoch="responses-v1")


@pytest.mark.parametrize("epoch", [None, "", "   "])
def test_durable_openai_fails_closed_without_an_epoch(epoch):
  with pytest.raises(OpenAISessionEpochError):
    scope_provider_session_id("sess_1", provider="openai", durable=True, openai_epoch=epoch)


@pytest.mark.parametrize(
  "epoch",
  [
    "Chat-Prep-v1",  # uppercase
    "-leading-hyphen",
    ".leading-dot",
    "_leading-underscore",
    "has space",
    "has/slash",
    "a" * 33,  # one over the 32-char limit
    "épocher",
  ],
)
def test_invalid_epoch_values_fail_closed(epoch):
  with pytest.raises(OpenAISessionEpochError):
    scope_provider_session_id("sess_1", provider="openai", durable=True, openai_epoch=epoch)


@pytest.mark.parametrize("epoch", ["a", "0", "chat-prep-v1", "responses-v1", "rollback-2026.07.22", "a" * 32])
def test_valid_epoch_values_are_accepted(epoch):
  assert scope_provider_session_id("sess_1", provider="openai", durable=True, openai_epoch=epoch) == (
    f"sess_1--openai-{epoch}"
  )


def test_epoch_is_whitespace_trimmed_before_validation():
  assert (
    scope_provider_session_id("sess_1", provider="openai", durable=True, openai_epoch="  chat-prep-v1  ")
    == "sess_1--openai-chat-prep-v1"
  )


def test_session_epoch_error_is_not_a_new_session_signal():
  # Distinct from the history fence: a missing epoch is an operator
  # misconfiguration, not a client-visible "start a new session" condition.
  assert OpenAISessionEpochError().requires_new_session is False


# --- Post-cutover history routing -----------------------------------------


def _gateway_session() -> GatewaySession:
  return GatewaySession(
    session_id="sess_interactive",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
  )


def test_openai_responses_provider_accepts_native_history_before_request_build():
  provider = OpenAIProvider()
  signature = json.dumps({
    "v": TEXT_SIGNATURE_MARKER,
    "item_id": "msg_1",
    "status": "completed",
  })
  messages = [
    {
      "role": "assistant", "provider": "openai", "model": "gpt-5.6",
      "content": [{"type": "text", "text": "prior", "textSignature": signature}],
    }
  ]
  params = provider.build_request_params(
    model="gpt-5.6", messages=messages, system_prompt=None, tools=[], max_tokens=100,
  )
  assert params["input"][0]["id"] == "msg_1"


def test_gateway_transcript_recap_reconstruction_sanitizes_replay_markers(tmp_path):
  transcript_dir = tmp_path / "chat_logs"
  transcript_dir.mkdir()
  (transcript_dir / "sess_1.jsonl").write_text(
    json.dumps(
      {
        "type": "chat_request",
        "messages": [{"role": "assistant", "content": [{"signature": TEXT_SIGNATURE_MARKER}]}],
      }
    )
    + "\n",
    encoding="utf-8",
  )

  events = _read_session_transcript_events(transcript_dir, "sess_1")
  assert len(events) == 1
  assert TEXT_SIGNATURE_MARKER not in json.dumps(events)


def test_active_turn_recap_context_accepts_responses_markers():
  event_log = EventLog()
  event_log.append(
    {"type": "assistant_message", "content_blocks": [{"signature": REASONING_SIGNATURE_MARKER}]}
  )

  recap = _compute_session_recap_payload(
    _gateway_session(),
    SessionStream(event_log=event_log, runner_task=None),
    trigger="disconnect",
  )
  assert recap["type"] == "session_recap"


def test_agent_session_log_remains_readable_and_reconstructable_after_cutover(tmp_path):
  log = AgentSessionLog(path=tmp_path / "sessions" / "marked.jsonl")
  asyncio.run(
    log.append(
      {
        "type": "assistant_message",
        "content_blocks": [{"type": "thinking", "signature": REASONING_SIGNATURE_MARKER}],
      }
    )
  )

  entries, _ = asyncio.run(log.query(order="asc"))
  assert len(entries) == 1
  assert entries[0].event[DURABLE_HISTORY_VERSION_KEY] == RESPONSES_HISTORY_VERSION
  assert isinstance(asyncio.run(SessionContextBuilder(agent_session_log=log).build()), list)


def test_package_autonomous_missing_epoch_fails_before_client_construction(
  monkeypatch: pytest.MonkeyPatch,
):
  monkeypatch.delenv("OPENAI_SESSION_EPOCH", raising=False)

  with pytest.raises(OpenAISessionEpochError):
    asyncio.run(
      autonomous_module.run_autonomous(
        "system",
        "hello",
        **_openai_bound_execution(),
        session_id="caller-durable-id",
        user_id="alice",
        billing_mode="byok",
        rate_table_version="test",
      )
    )


def test_package_autonomous_state_dir_missing_epoch_fails_before_client_construction(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
):
  monkeypatch.delenv("OPENAI_SESSION_EPOCH", raising=False)

  with pytest.raises(OpenAISessionEpochError):
    asyncio.run(
      autonomous_module.run_autonomous(
        "system",
        "hello",
        **_openai_bound_execution(),
        state_dir=tmp_path,
        user_id="alice",
        billing_mode="byok",
        rate_table_version="test",
      )
    )


def test_package_autonomous_ephemeral_openai_run_does_not_require_epoch(
  monkeypatch: pytest.MonkeyPatch,
):
  captured: dict[str, str] = {}

  class _Dispatcher:
    def __init__(self, **kwargs):
      captured["dispatcher"] = kwargs["session_id"]

  class _Runner:
    def __init__(self, **kwargs):
      captured["runner"] = kwargs["session_id"]

  async def _run_session(*_args, **_kwargs):
    return SimpleNamespace(response="ok")

  monkeypatch.delenv("OPENAI_SESSION_EPOCH", raising=False)
  monkeypatch.delenv("EXCEL_ORCHESTRATION_DEV", raising=False)
  monkeypatch.setattr(autonomous_module, "ToolDispatcher", _Dispatcher)
  monkeypatch.setattr(autonomous_module, "AgentRunner", _Runner)
  monkeypatch.setattr(autonomous_module, "run_session", _run_session)

  result = asyncio.run(
    autonomous_module.run_autonomous(
      "system",
      "hello",
      **_openai_bound_execution(),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="test",
    )
  )

  assert result.response == "ok"
  assert captured["dispatcher"] == captured["runner"]
  assert captured["runner"].startswith("autonomous-")
  assert "--openai-" not in captured["runner"]


def test_package_autonomous_scopes_caller_id_before_dispatcher_and_runner(
  monkeypatch: pytest.MonkeyPatch,
):
  captured: dict[str, str] = {}

  class _Dispatcher:
    def __init__(self, **kwargs):
      captured["dispatcher"] = kwargs["session_id"]

  class _Runner:
    def __init__(self, **kwargs):
      captured["runner"] = kwargs["session_id"]

  async def _run_session(*_args, **_kwargs):
    return SimpleNamespace(response="ok")

  monkeypatch.setenv("OPENAI_SESSION_EPOCH", "chat-prep-v1")
  monkeypatch.delenv("EXCEL_ORCHESTRATION_DEV", raising=False)
  monkeypatch.setattr(autonomous_module, "ToolDispatcher", _Dispatcher)
  monkeypatch.setattr(autonomous_module, "AgentRunner", _Runner)
  monkeypatch.setattr(autonomous_module, "run_session", _run_session)

  result = asyncio.run(
    autonomous_module.run_autonomous(
      "system",
      "hello",
      **_openai_bound_execution(),
      session_id="caller-durable-id",
      user_id="alice",
      billing_mode="byok",
      rate_table_version="test",
    )
  )

  assert result.response == "ok"
  assert captured == {
    "dispatcher": "caller-durable-id--openai-chat-prep-v1",
    "runner": "caller-durable-id--openai-chat-prep-v1",
  }


def test_sub_agent_id_inherits_already_scoped_parent_without_rescoping():
  parent = "autonomous-parent--openai-chat-prep-v1"
  assert derive_sub_agent_id(parent, 4) == f"sub4:{parent}"


def test_transcript_write_strips_encrypted_reasoning(tmp_path):
  """D17: replay-only OpenAI state must never be persisted to GATEWAY_LOG_DIR.

  Regression for the review finding that _write_transcript copied the entry
  verbatim; read-path sanitization is too late because the ciphertext is
  already on disk.
  """
  from agent_gateway.server_chat_transcripts import _write_transcript

  entry = {
    "type": "chat_request",
    "messages": [
      {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "visible", "textSignature": "sig-should-not-persist"},
          {"type": "thinking", "encrypted_content": "CIPHERTEXT_MUST_NOT_PERSIST"},
        ],
      }
    ],
    "openai_history_version": "responses-v1",
  }
  _write_transcript(tmp_path, "sess_d17", entry)

  written = (tmp_path / "sess_d17.jsonl").read_text(encoding="utf-8")
  assert "CIPHERTEXT_MUST_NOT_PERSIST" not in written
  assert "sig-should-not-persist" not in written
  assert "responses-v1" not in written
  assert "visible" in written  # non-replay content is retained
