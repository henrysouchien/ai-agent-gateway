# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSessionLog
from agent_gateway import agent_session_log_sidecars as sidecar_helpers


def test_agent_session_log_sidecar_wrappers_preserve_parent_override_seams(tmp_path: Path) -> None:
  class CustomLog(AgentSessionLog):
    def _logical_stream_id(self) -> str:
      return "logical-custom"

    def _stream_hash(self) -> str:
      return "hash-custom"

    def _telemetry_source_id(self, role: str, suffix: str) -> str:
      return f"telemetry:{role}:{suffix}"

  log = CustomLog(tmp_path / "agentsess_analyst_alice.jsonl")

  assert log._stream_hash() == "hash-custom"
  assert log._telemetry_source_id("active", "000002") == "telemetry:active:000002"
  assert log._active_sidecar_payload({"agent_session_id": "sess-1"}, active_generation=2) == {
    "agent_session_id": "sess-1",
    "schema_version": 2,
    "file_role": "active",
    "logical_stream_id": "logical-custom",
    "telemetry_source_id": "telemetry:active:000002",
    "active_generation": 2,
  }
  assert log._segment_sidecar_payload(
    {"agent_session_id": "sess-1", "created_at": "created"},
    segment_id="000000000001-000000000003-g000002",
    first_seq=1,
    last_seq=3,
    active_generation=2,
    rotated_from_file_identity={"size": 10},
  ) == {
    "schema_version": 2,
    "agent_session_id": "sess-1",
    "agent_id": None,
    "user_id": None,
    "product_id": None,
    "file_kind": "canonical",
    "channel": None,
    "profile": None,
    "created_at": "created",
    "file_role": "segment",
    "logical_stream_id": "logical-custom",
    "telemetry_source_id": "telemetry:segment:000000000001-000000000003-g000002",
    "active_generation": 2,
    "segment_id": "000000000001-000000000003-g000002",
    "first_seq": 1,
    "last_seq": 3,
    "rotated_from_source_id": "telemetry:active:000002",
    "rotated_from_path": str(log.path),
    "rotated_from_file_identity": {"size": 10},
  }


def test_sidecar_repair_base_prefers_active_then_segment_then_fallback() -> None:
  fallback = {"agent_session_id": "fallback"}
  segment = {"agent_session_id": "segment", "ignored": "value"}
  active = {"agent_session_id": "active"}

  assert sidecar_helpers.sidecar_base_for_repair(
    [segment],
    load_sidecar_payload_fn=lambda: active,
    sidecar_base_from_segment_meta_fn=sidecar_helpers.sidecar_base_from_segment_meta,
    fallback_sidecar_base_fn=lambda: fallback,
  ) == active

  assert sidecar_helpers.sidecar_base_for_repair(
    [segment],
    load_sidecar_payload_fn=lambda: None,
    sidecar_base_from_segment_meta_fn=sidecar_helpers.sidecar_base_from_segment_meta,
    fallback_sidecar_base_fn=lambda: fallback,
  ) == {"agent_session_id": "segment"}

  assert sidecar_helpers.sidecar_base_for_repair(
    [{}],
    load_sidecar_payload_fn=lambda: None,
    sidecar_base_from_segment_meta_fn=sidecar_helpers.sidecar_base_from_segment_meta,
    fallback_sidecar_base_fn=lambda: fallback,
  ) == fallback


def test_v2_storage_identity_survives_segment_and_repair_projection(
  tmp_path: Path,
) -> None:
  path = tmp_path / "s-digest" / "agentsess_s-digest_alice.jsonl"
  storage = {
    "storage_layout": 2,
    "tenant_id": "hank",
    "workload_profile": "analyst",
    "provider": "openai",
    "provider_session_epoch": "responses-v1",
    "storage_identity_digest": "a" * 64,
  }
  segment = sidecar_helpers.segment_sidecar_payload(
    path,
    {
      "agent_session_id": path.stem,
      "agent_id": "analyst",
      "user_id": "alice",
      "product_id": "hank",
      "file_kind": "canonical",
      "channel": None,
      "profile": "analyst",
      "created_at": "created",
      **storage,
    },
    segment_id="000000000001-000000000003-g000002",
    first_seq=1,
    last_seq=3,
    active_generation=2,
    rotated_from_file_identity={"size": 10},
    logical_stream_id_fn=lambda: str(path),
    telemetry_source_id_fn=(
      lambda role, suffix: f"telemetry:{role}:{suffix}"
    ),
  )

  assert {
    key: segment[key]
    for key in sidecar_helpers.V2_STORAGE_IDENTITY_FIELDS
  } == storage
  repaired = sidecar_helpers.sidecar_base_from_segment_meta(segment)
  assert repaired is not None
  assert {
    key: repaired[key]
    for key in sidecar_helpers.V2_STORAGE_IDENTITY_FIELDS
  } == storage


def test_fallback_sidecar_base_derives_user_from_canonical_session_name(tmp_path: Path) -> None:
  path = tmp_path / "agent" / "agentsess_analyst_henry.jsonl"

  assert sidecar_helpers.fallback_sidecar_base(path, now_iso_fn=lambda: "now") == {
    "agent_session_id": "agentsess_analyst_henry",
    "agent_id": "agent",
    "user_id": "henry",
    "product_id": None,
    "file_kind": "canonical",
    "channel": None,
    "profile": None,
    "created_at": "now",
  }
