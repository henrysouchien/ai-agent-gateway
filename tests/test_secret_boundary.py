from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.secret_boundary import (  # noqa: E402
  REDACTED_SECRET,
  SANITIZATION_FAILED,
  SecretBoundary,
  sanitize_boundary_value,
  sanitize_tool_event,
)
from agent_gateway.ui_blocks_metrics import snapshot as metrics_snapshot  # noqa: E402
from tests.capability_execution_test_support import (  # noqa: E402
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


def test_typed_event_policy_leaves_ordinary_chat_prose_and_sanitizes_tool_blocks() -> None:
  canary = "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"
  ordinary = {
    "type": "user_message",
    "content": "Discuss paths and api_key examples without treating prose as authority.",
  }
  tool_message = {
    "type": "user_message",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "tool-1",
        "content": json.dumps({"credential": canary}),
      }
    ],
  }

  assert sanitize_tool_event(ordinary, sink="replay") == ordinary
  serialized = json.dumps(sanitize_tool_event(tool_message, sink="replay"))
  assert canary not in serialized
  assert REDACTED_SECRET in serialized


def test_sanitizer_failure_returns_fixed_tombstone(monkeypatch: pytest.MonkeyPatch) -> None:
  def _raise(self, value, *, sink):
    del self, value, sink
    raise RuntimeError("canary must not be returned")

  monkeypatch.setattr(SecretBoundary, "sanitize", _raise)
  assert sanitize_boundary_value(
    {"value": "sk-ant-api03-CODEX-WAVE0-CANARY-DO-NOT-USE-8f21d7"},
    sink="durable",
  ) == SANITIZATION_FAILED


def test_typed_tool_block_failure_is_structurally_valid_and_observable(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-TYPED-FAILURE-8f21d7"
  before = metrics_snapshot().get(
    "secret_boundary_sanitization_failed",
    0,
  )

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
        {
          "type": "tool_result",
          "tool_use_id": secret,
          "content": secret,
        },
      ],
    },
    sink="model_history",
    boundary=boundary,
  )

  assert projected["content_blocks"] == [
    {
      "type": "tool_use",
      "id": "boundary-sanitization-failed",
      "name": "boundary_sanitization_failed",
      "input": {"_boundary_error": SANITIZATION_FAILED},
    },
    {
      "type": "tool_result",
      "tool_use_id": "boundary-sanitization-failed",
      "content": SANITIZATION_FAILED,
      "is_error": True,
    },
  ]
  assert secret not in json.dumps(projected)
  assert metrics_snapshot()["secret_boundary_sanitization_failed"] >= before + 2


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
