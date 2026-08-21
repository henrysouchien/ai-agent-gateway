from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_gateway.agent_session_log import (
  AgentSessionLog,
  AgentSessionRef,
)
from agent_gateway.agent_session_log_inventory import (
  SessionLogInventoryError,
  enumerate_selected_agent_session_logs,
  read_session_log_physical_range,
)
from agent_gateway.agent_session_log_layout import prepare_autonomous_session_log
from agent_gateway.agent_session_log_records import resolve_agent_session_id


_ALL_KINDS = frozenset({
  "canonical", "batch", "pipeline", "ephemeral",
})


def _prepare_v2(base: Path):
  prepared = prepare_autonomous_session_log(
    base_dir=base,
    layout="v2",
    tenant="hank",
    owner="henry",
    workload_profile="analyst",
    provider="openai",
    provider_session_epoch="responses-v1",
    now_iso=lambda: "2026-08-20T00:00:00+00:00",
  )
  prepared.close()
  assert prepared.authority.meta_path is not None
  assert prepared.authority.active_path is not None
  return Path(prepared.authority.active_path), Path(prepared.authority.meta_path)


def _prepare_flat_batch(base: Path, *, product_id: str) -> AgentSessionLog:
  agent_id = "b1_2"
  session_ref = AgentSessionRef(
    user_id="henry",
    agent_id=agent_id,
    agent_session_id=resolve_agent_session_id("henry", agent_id),
  )
  log = AgentSessionLog(session_ref=session_ref, base_dir=base)
  log.path.with_suffix(".meta.json").write_text(
    json.dumps({
      "schema_version": 1,
      "agent_session_id": session_ref.agent_session_id,
      "agent_id": agent_id,
      "user_id": "henry",
      "product_id": product_id,
      "tenant_id": product_id,
      "file_kind": "canonical",
      "run_kind": "batch",
      "run_seq": 2,
      "batch_id": "batch-1",
      "ticker": "PCTY",
      "stage": "analysis",
      "skill": "test",
      "channel": None,
      "profile": "analyst",
      "created_at": "2026-08-20T00:00:00+00:00",
    }),
    encoding="utf-8",
  )
  return log


def _enumerate_v2(base: Path):
  return enumerate_selected_agent_session_logs(
    base,
    layout="v2",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank"}),
    allowed_stream_kinds=_ALL_KINDS,
  )


def test_v1_legacy_sidecarless_stream_fails_classification(
  tmp_path: Path,
) -> None:
  active = tmp_path / "sessions" / "analyst" / "session.jsonl"
  active.parent.mkdir(parents=True)
  active.write_text("", encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match="cannot be classified"):
    enumerate_selected_agent_session_logs(
      tmp_path / "sessions",
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank"}),
      allowed_stream_kinds=_ALL_KINDS,
    )


@pytest.mark.parametrize("field", ["tenant_id", "product_id"])
def test_v2_rejects_wrong_trusted_tenant(tmp_path: Path, field: str) -> None:
  _active, meta = _prepare_v2(tmp_path)
  payload = json.loads(meta.read_text(encoding="utf-8"))
  payload[field] = "other"
  meta.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match="contradicts storage"):
    _enumerate_v2(tmp_path)


def test_v2_rejects_duplicate_sidecar_fields(tmp_path: Path) -> None:
  _active, meta = _prepare_v2(tmp_path)
  raw = meta.read_text(encoding="utf-8")
  meta.write_text(raw[:-1] + ',"tenant_id":"other"}', encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match="requires valid metadata"):
    _enumerate_v2(tmp_path)


def test_v2_rejects_group_writable_selected_parent(tmp_path: Path) -> None:
  active, _meta = _prepare_v2(tmp_path)
  active.parent.chmod(0o770)

  with pytest.raises(SessionLogInventoryError, match="parent is unsafe"):
    _enumerate_v2(tmp_path)


def test_v2_product_policy_excludes_coherent_archive_but_archiver_admits_it(
  tmp_path: Path,
) -> None:
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")

  current = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v2",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank"}),
    allowed_stream_kinds=_ALL_KINDS,
  )
  assert current == ()

  archived = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v2",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=_ALL_KINDS,
  )
  assert [item.path for item in archived] == [log.path]
  assert archived[0].stream_kind == "batch"


def test_v1_archive_policy_preserves_coherent_historical_stream(
  tmp_path: Path,
) -> None:
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  selected = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v1",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=_ALL_KINDS,
  )
  assert [item.path for item in selected] == [log.path]


def test_bound_physical_range_rejects_named_file_replacement(
  tmp_path: Path,
) -> None:
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "bound"})
  selected = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v1",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=_ALL_KINDS,
  )
  physical = next(
    item for item in selected[0].files if item.role == "active"
  )
  assert read_session_log_physical_range(
    physical,
    offset_lo=0,
    offset_hi=physical.file_identity.size,
  ) == log.path.read_bytes()

  log.path.unlink()
  log.path.write_bytes(b"replacement\n")
  with pytest.raises(SessionLogInventoryError, match="identity changed"):
    read_session_log_physical_range(
      physical,
      offset_lo=0,
      offset_hi=physical.file_identity.size,
    )


def test_flat_rotated_segment_requires_coherent_metadata(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  selected = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v2",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=_ALL_KINDS,
  )
  segment = next(
    physical
    for physical in selected[0].files
    if physical.role == "segment"
  )
  assert segment.classification_payload is not None
  assert segment.classification_payload["run_kind"] == "batch"

  segment.sidecar_path.unlink()
  with pytest.raises(SessionLogInventoryError, match="segment requires valid"):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v2",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank", "hank-dev"}),
      allowed_stream_kinds=_ALL_KINDS,
    )


def test_flat_ephemeral_sidecar_must_derive_exact_active_stem(
  tmp_path: Path,
) -> None:
  active = tmp_path / "cli_analyst" / "agentsess_gateway_1_henry.jsonl"
  active.parent.mkdir(parents=True)
  AgentSessionLog(path=active)
  active.with_suffix(".meta.json").write_text(
    json.dumps({
      "schema_version": 1,
      "agent_session_id": "other-session",
      "agent_id": "cli:analyst",
      "user_id": "henry",
      "product_id": "hank",
      "file_kind": "ephemeral",
      "channel": "cli",
      "profile": "analyst",
      "created_at": "2026-08-20T00:00:00+00:00",
    }),
    encoding="utf-8",
  )
  with pytest.raises(SessionLogInventoryError, match="active path"):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank"}),
      allowed_stream_kinds=_ALL_KINDS,
    )


def test_v2_rejects_unknown_active_sidecar_field(tmp_path: Path) -> None:
  _active, meta = _prepare_v2(tmp_path)
  payload = json.loads(meta.read_text(encoding="utf-8"))
  payload["unexpected"] = "field"
  meta.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match="closed field contract"):
    _enumerate_v2(tmp_path)
