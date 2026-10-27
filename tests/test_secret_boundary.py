from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.providers.anthropic import AnthropicProvider  # noqa: E402
from agent_gateway.providers.base import ModelInfo  # noqa: E402
from agent_gateway.secret_boundary import (  # noqa: E402
  REDACTED_SECRET,
  SANITIZATION_FAILED,
  UNSUPPORTED_VALUE,
  SecretBoundary,
  sanitize_boundary_value,
  sanitize_tool_event,
)
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)


def _execution(secret: str):
  return stub_runner_capability_execution(
    provider=SimpleNamespace(name="stub"),
    model="stub-model",
    effort="none",
    auth_config={"api_key": secret},
  )


def test_registered_credential_is_exact_lifecycle_local_and_not_serializable() -> None:
  secret_a = "CUSTOM-ACTIVE-CREDENTIAL-aaaaaaaa"
  secret_b = "CUSTOM-ACTIVE-CREDENTIAL-bbbbbbbb"
  boundary_a = SecretBoundary.from_capability_execution(_execution(secret_a))
  boundary_b = SecretBoundary.from_capability_execution(_execution(secret_b))

  assert boundary_a.sanitize({"value": secret_a}, sink="model") == {
    "value": REDACTED_SECRET
  }
  assert boundary_a.sanitize({"value": secret_b}, sink="model") == {
    "value": secret_b
  }
  assert boundary_b.sanitize({"value": secret_a}, sink="model") == {
    "value": secret_a
  }
  with pytest.raises(TypeError):
    pickle.dumps(boundary_a)
  with pytest.raises(TypeError):
    json.dumps(boundary_a)

  short_boundary = SecretBoundary.from_capability_execution(_execution("k"))
  assert short_boundary.sanitize("k", sink="model") == REDACTED_SECRET
  assert short_boundary.sanitize("ordinary lookup", sink="model") == "ordinary lookup"
  metadata = {"api_key_set": True, "token_count": 1}
  assert short_boundary.sanitize(metadata, sink="model") == metadata


def test_auth_config_registration_uses_exact_material_without_global_retention() -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-auth-config-8f21d7"
  boundary = SecretBoundary.from_auth_config({
    "provider": "anthropic",
    "api_key": secret,
    "api_key_set": True,
  })

  assert boundary.sanitize(
    {"value": secret, "api_key_set": True},
    sink="autonomous_log",
  ) == {"value": REDACTED_SECRET, "api_key_set": True}
  assert SecretBoundary().sanitize(secret, sink="other_lifecycle") == secret


def test_unknown_auth_config_keys_register_as_secrets_and_config_fields_do_not() -> None:
  novel_secret = "CUSTOM-NOVEL-PROVIDER-CREDENTIAL-8f21d7"
  boundary = SecretBoundary.from_auth_config({
    "provider": "anthropic",
    "auth_mode": "oauth",
    "base_url": "https://api.example.test",
    "max_tokens": 16_000,
    "session_credential": novel_secret,
  })

  assert boundary.sanitize(
    {"value": novel_secret},
    sink="autonomous_log",
  ) == {"value": REDACTED_SECRET}
  prose = "provider anthropic via https://api.example.test in oauth mode"
  assert boundary.sanitize(prose, sink="autonomous_log") == prose


def test_high_confidence_material_is_removed_without_scanning_prose_or_key_names() -> None:
  canary = "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"
  value = {
    "discussion": "An api_key is configured; sk-example is illustrative.",
    "api_key_set": True,
    "credential_status": "ready",
    "token_count": 42,
    "hash": "hmac-sha256-v1:key-1:" + ("a" * 64),
    "hint": "Use the credential ref, never inline it.",
    "ref": "credential://tenant/provider",
    "path": "/Users/alice/Documents/report.xlsx",
    "secret_value": canary,
  }

  sanitized = sanitize_boundary_value(value, sink="durable")

  assert sanitized["discussion"] == value["discussion"]
  assert sanitized["api_key_set"] is True
  assert sanitized["credential_status"] == "ready"
  assert sanitized["token_count"] == 42
  assert sanitized["hash"] == value["hash"]
  assert sanitized["hint"] == value["hint"]
  assert sanitized["ref"] == value["ref"]
  assert sanitized["path"] == value["path"]
  assert sanitized["secret_value"] == REDACTED_SECRET


def test_typed_event_policy_projects_authored_text_and_leaves_results_alone() -> None:
  canary = "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"
  ordinary = {
    "type": "user_message",
    "content": "Discuss paths and api_key examples without treating prose as authority.",
  }
  authored = {
    "type": "assistant_message",
    "content_blocks": [
      {
        "type": "tool_use",
        "id": "tool-1",
        "name": "run_bash",
        "input": {"command": f"curl -H 'x-key: {canary}' https://example.test"},
      }
    ],
  }
  returned = {
    "type": "user_message",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "tool-1",
        "content": json.dumps({"stdout": "ok"}),
      }
    ],
  }

  assert sanitize_tool_event(ordinary, sink="replay") == ordinary
  projected = sanitize_tool_event(authored, sink="replay")
  assert canary not in json.dumps(projected)
  assert REDACTED_SECRET in json.dumps(projected)
  # A returned payload is projected by the tool that produced it, not again here.
  assert sanitize_tool_event(returned, sink="replay") == returned


@pytest.mark.parametrize("sink", ["durable_event", "stream_event"])
def test_planned_calls_redact_credentials_without_mutating_execution_input(sink: str) -> None:
  secret = "CUSTOM-ACTIVE-PLANNED-CREDENTIAL-8f21d7"
  tool_id = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
  boundary = SecretBoundary.from_auth_config({"api_key": secret})
  event = {
    "type": "tool_calls_planned",
    "turn": 1,
    "calls": [
      {
        "tool_call_id": tool_id,
        "tool_name": "search",
        "tool_input": {"query": f"find {secret}"},
        "display": {"label": "Searching", "detail": secret},
      },
      {
        "tool_call_id": "quote-2",
        "tool_name": "get_quote",
        "tool_input": {"ticker": "NVDA"},
        "display": {"label": "Get quote", "detail": "NVDA"},
      },
    ],
  }

  projected = sanitize_tool_event(event, sink=sink, boundary=boundary)

  assert projected == {
    **event,
    "calls": [
      {
        **event["calls"][0],
        "tool_input": {"query": f"find {REDACTED_SECRET}"},
        "display": {"label": "Searching", "detail": REDACTED_SECRET},
      },
      event["calls"][1],
    ],
  }
  started = sanitize_tool_event(
    {"type": "tool_call_start", **event["calls"][0]},
    sink=sink,
    boundary=boundary,
  )
  assert projected["calls"][0]["tool_call_id"] == started["tool_call_id"] == tool_id
  assert event["calls"][0]["tool_input"]["query"] == f"find {secret}"
  assert event["calls"][0]["display"]["detail"] == secret


def test_sanitizer_failure_returns_fixed_tombstone(monkeypatch: pytest.MonkeyPatch) -> None:
  def _raise(self, value, *, sink):
    del self, value, sink
    raise RuntimeError("canary must not be returned")

  monkeypatch.setattr(SecretBoundary, "sanitize", _raise)
  assert sanitize_boundary_value(
    {"value": "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"},
    sink="durable",
  ) == SANITIZATION_FAILED


def test_settled_tool_result_crosses_the_boundary_unprojected(
  caplog: pytest.LogCaptureFixture,
) -> None:
  # fetch_financials emitted 42,606 value records in the buyer live cohort, and
  # walking them blanked the whole result. Nothing walks a settled result now.
  value = {
    "result_key": "income",
    "row_index": 0,
    "field": "revenue",
    "value_kind": "number",
    "canonical_value": "123",
    "concept": "Revenue",
    "period": "2025",
  }
  event = {
    "type": "tool_call_complete",
    "tool_name": "fetch_financials",
    "result": {
      "lineage_descriptor": {"values": [dict(value) for _ in range(42_606)]},
      "hint": "Financial data is available by reference.",
    },
    "error": None,
    "dispatch": {"outcome": "ok", "sources": []},
  }

  projected = sanitize_tool_event(event, sink="tool_complete")

  assert projected == event
  assert projected["result"] is event["result"]
  assert not [
    record for record in caplog.records
    if record.name == "agent_gateway.secret_boundary"
  ]


def test_over_deep_model_input_drops_only_the_unreadable_subtree() -> None:
  deep: object = "ordinary"
  for _ in range(33):
    deep = {"nested": deep}
  assistant = sanitize_tool_event(
    {
      "type": "assistant_message",
      "content_blocks": [
        {
          "type": "tool_use",
          "id": "tool-lookup",
          "name": "lookup",
          "input": {"value": deep, "flag": "keep", "rows": ["ordinary"] * 100_000},
        },
      ],
    },
    sink="model_history",
  )

  projected_input = assistant["content_blocks"][0]["input"]
  assert assistant["content_blocks"][0]["name"] == "lookup"
  assert projected_input["flag"] == "keep"
  # Breadth is not a budget any more; only unreadable depth is dropped, and the
  # call keeps the shape the model wrote around it.
  assert projected_input["rows"] == ["ordinary"] * 100_000
  assert UNSUPPORTED_VALUE in json.dumps(projected_input["value"])


def test_replayed_tool_result_blocks_keep_their_call_identity() -> None:
  from agent_gateway.transcript import _tool_result_blocks_from_event

  event = {
    "type": "tool_call_complete",
    "tool_call_id": "tool-lookup",
    "tool_name": "lookup",
    "result": {"answer": 42},
    "final_tool_result_blocks": [
      {"type": "tool_result", "tool_use_id": "tool-lookup", "content": '{"answer": 42}'},
      {"type": "text", "text": "cited [S1]"},
    ],
  }
  projected = sanitize_tool_event(event, sink="durable_event", boundary=SecretBoundary())
  replay = _tool_result_blocks_from_event(projected)
  normalized = AnthropicProvider().normalize_messages(
    [
      {"role": "assistant", "content": [
        {"type": "tool_use", "id": "tool-lookup", "name": "lookup", "input": {}},
      ]},
      {"role": "user", "content": replay},
    ],
    ModelInfo(id="test-model", provider="anthropic"),
  )

  assert projected["tool_call_id"] == "tool-lookup"
  assert normalized[-1]["content"] == [
    {"type": "tool_result", "tool_use_id": "tool-lookup", "content": '{"answer": 42}'},
    {"type": "text", "text": "cited [S1]"},
  ]


def test_typed_tool_call_failure_is_structurally_valid(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-TYPED-FAILURE-8f21d7"

  def _fail(self, _value, *, sink):
    _ = self, sink
    return SANITIZATION_FAILED

  boundary = SecretBoundary((secret,))
  monkeypatch.setattr(SecretBoundary, "sanitize", _fail)
  projected = sanitize_tool_event(
    {
      "type": "assistant_message",
      "content_blocks": [
        {
          "type": "tool_use",
          "id": secret,
          "name": secret,
          "input": {"credential": secret},
        },
      ],
    },
    sink="model_history",
    boundary=boundary,
  )

  (call,) = projected["content_blocks"]
  assert call["type"] == "tool_use"
  assert isinstance(call["id"], str) and call["id"]
  assert isinstance(call["name"], str) and call["name"]
  assert isinstance(call["input"], dict)
  assert secret not in json.dumps(projected)


def test_dispatch_record_is_sanitized_on_tool_call_complete() -> None:
  """D-B1-3: `dispatch.sources` carries URLs and document ids."""

  secret = "CUSTOM-ACTIVE-CREDENTIAL-DISPATCH-1f9c02"
  boundary = SecretBoundary((secret,))

  projected = sanitize_tool_event(
    {
      "type": "tool_call_complete",
      "tool_call_id": "toolu_1",
      "tool_name": "web_fetch",
      "result": None,
      "error": None,
      "duration_ms": 3,
      "server": None,
      "is_error": False,
      "dispatch": {
        "outcome": "ok",
        "attempts": 1,
        "route_id": "local/web_fetch",
        "sources": [
          {
            "document_id": "web:deadbeef",
            "source_kind": "web",
            "source_url": f"https://example.test/a?token={secret}",
          }
        ],
      },
    },
    sink="replay",
    boundary=boundary,
  )

  serialized = json.dumps(projected["dispatch"])
  assert secret not in serialized
  assert REDACTED_SECRET in serialized
  assert projected["dispatch"]["outcome"] == "ok"


def test_runtime_guard_message_is_boundary_sanitized() -> None:
  """Guard messages carry the delivery nudges' objective echoes (CUR-E2E-08
  observability) — free prose that must cross the boundary like every other
  durable copy of the dispatch objective."""
  secret = "CUSTOM-ACTIVE-CREDENTIAL-cccccccc"
  boundary = SecretBoundary.from_capability_execution(_execution(secret))
  event = {
    "type": "runtime_guard",
    "guard": "unread_result_handle_nudge",
    "message": f"task bg-1 (fetch filings with key {secret}): read it",
  }
  projected = sanitize_tool_event(
    event,
    sink="durable_event",
    boundary=boundary,
  )
  assert secret not in json.dumps(projected)
  assert projected["guard"] == "unread_result_handle_nudge"
  assert "task bg-1" in projected["message"]


def test_payg_bound_auth_config_preserves_public_credential_ref_in_skill_result_projection() -> None:
  """A PAYG/pool-backed auth_config carries deployment_api_key_source = 'anthropic-generation-<sha256>'.

  SecretBoundary must redact the raw api_key while preserving the public
  generation ref and classification stamps so ToolDispatcher._append_event
  does not redact CapabilityBind.credential_ref and break
  SkillRunEventEmitter.project_result_captured's exact-match check.
  """
  from agent_gateway.event_log import EventLog
  from model_authority.bind import CapabilityBind
  from agent_gateway.providers.anthropic_oauth import (
    ANTHROPIC_DEPLOYMENT_API_KEY_FIELD,
    ANTHROPIC_USER_SCOPE_FIELD,
    CREDENTIAL_AFFINITY_KEY_FIELD,
    CREDENTIAL_PREPARED_FIELD,
  )
  from agent_gateway.sub_agent_skill_events import SkillRunEventEmitter
  from agent_gateway.tool_dispatcher import ToolDispatcher

  raw_api_key = "sk-ant-api03-TEST-SECRET-MATERIAL-735ddab23cdd9fb8"
  public_gen_ref = "anthropic-generation-ee842ddf563c4624dceabf5fb78192d9d1c7dea06d3297969951728a3694ee08"
  affinity_digest = "a" * 64
  auth_config = {
    "provider": "anthropic",
    "placement": "local",
    ANTHROPIC_DEPLOYMENT_API_KEY_FIELD: public_gen_ref,
    "auth_mode": "api",
    "auth_mechanism": "api_key",
    "entitlement_kind": "org_commercial",
    "ownership": "deployment:org_commercial",
    "api_key": raw_api_key,
    "auth_token": "",
    "max_tokens": 16000,
    ANTHROPIC_USER_SCOPE_FIELD: "13",
    CREDENTIAL_AFFINITY_KEY_FIELD: affinity_digest,
    CREDENTIAL_PREPARED_FIELD: True,
    "workspace_id": "wrks_01TestWorkspace",
    "account_identity": "api-key-735ddab23cdd9fb8",
    "credential_source": {
      "account_identity": "api-key-735ddab23cdd9fb8",
      "generation_ref": public_gen_ref,
      "automatic": True,
    },
  }
  boundary = SecretBoundary.from_auth_config(auth_config)
  assert boundary.sanitize(raw_api_key, sink="dispatcher_event") == REDACTED_SECRET
  assert boundary.sanitize(public_gen_ref, sink="dispatcher_event") == public_gen_ref
  assert boundary.sanitize("local", sink="dispatcher_event") == "local"
  assert boundary.sanitize("api_key", sink="dispatcher_event") == "api_key"
  assert boundary.sanitize("org_commercial", sink="dispatcher_event") == "org_commercial"
  assert boundary.sanitize("wrks_01TestWorkspace", sink="dispatcher_event") == "wrks_01TestWorkspace"

  bind_dict = {
    "schema_version": "1.0",
    "capability_id": "node.explore",
    "model_key": "anthropic.claude-haiku-5-5",
    "provider": "anthropic",
    "upstream_model": "claude-haiku-5-5",
    "adapter": "anthropic.messages",
    "protocol_profile": "messages.adaptive",
    "route": "anthropic.public",
    "effort": "low",
    "credential_principal": "service",
    "credential_ref": public_gen_ref,
    "run_mode": "interactive",
    "registry_revision": "2026-10-15.5",
    "policy_revision": "2026-10-15.5",
    "selection_source": "internal_policy",
  }
  raw_log = EventLog()
  dispatcher = ToolDispatcher(
    cast(Any, None),
    event_log=raw_log,
    capability_bind=CapabilityBind.from_json(bind_dict),
  )
  dispatcher.bind_secret_boundary(boundary)
  tool_ctx = SimpleNamespace(
    event_log=dispatcher._boundary_event_log,
    emit=None,
  )

  async def _noop_append(_ev):
    return None

  async def _noop_confirm(_ev):
    return None

  emitter = SkillRunEventEmitter(
    skill_run_id="0ce3a19eb6e5578241b99b160c5e72b5",
    skill_name="explore",
    semantic_scope="portfolio",
    context_ticker=None,
    portfolio_id=None,
    event_log_getter=lambda: tool_ctx.event_log,
    tool_ctx=tool_ctx,
    durable_appender=_noop_append,
    durable_confirmer=_noop_confirm,
  )
  captured_event = {
    "type": "skill_result_captured",
    "skill_run_id": "0ce3a19eb6e5578241b99b160c5e72b5",
    "skill": "explore",
    "scope": "portfolio",
    "ticker": None,
    "portfolio_id": None,
    "exit_code": 0,
    "outcome": "success",
    "status": "complete",
    "capability_bind": bind_dict,
  }

  assert emitter.project_result_captured(captured_event) is True
  assert len(raw_log.entries) == 1
  assert raw_log.entries[0].event["capability_bind"]["credential_ref"] == public_gen_ref
