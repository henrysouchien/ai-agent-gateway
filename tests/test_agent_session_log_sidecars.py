# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import agent_session_log_sidecars as sidecar_helpers


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
