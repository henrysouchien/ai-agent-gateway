from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from agent_gateway.agent_session_log import (
  AgentSessionLog,
  AgentSessionRef,
)
from agent_gateway.agent_session_log_inventory import (
  SUPPORTED_SESSION_LOG_STREAM_KINDS,
  SessionLogInventoryError,
  enumerate_selected_agent_session_logs,
)
from agent_gateway.agent_session_log_layout import prepare_autonomous_session_log
from agent_gateway.agent_session_log_records import resolve_agent_session_id


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


def _source_id(logical_stream_id: str, role: str, suffix: str) -> str:
  digest = hashlib.sha1(logical_stream_id.encode("utf-8")).hexdigest()[:16]
  return f"agent_session_log:{digest}:{role}:{suffix}"


def _rewrite_flat_lineage(
  log: AgentSessionLog,
  logical_stream_id: str,
  *,
  manifest_uses_physical_path: bool = False,
  relocate_segment_identity: bool = False,
) -> None:
  active_meta_path = log.path.with_suffix(".meta.json")
  active_meta = json.loads(active_meta_path.read_text(encoding="utf-8"))
  active_generation = int(active_meta.get("active_generation", 0))
  active_meta.update({
    "schema_version": 2,
    "file_role": "active",
    "logical_stream_id": logical_stream_id,
    "active_generation": active_generation,
    "telemetry_source_id": _source_id(
      logical_stream_id,
      "active",
      f"{active_generation:06d}",
    ),
  })
  active_meta_path.write_text(json.dumps(active_meta), encoding="utf-8")

  if not log.segments_dir.exists():
    return
  segment_payloads: dict[str, dict[str, object]] = {}
  for segment_meta_path in sorted(log.segments_dir.glob("*.meta.json")):
    segment_meta = json.loads(
      segment_meta_path.read_text(encoding="utf-8")
    )
    segment_id = str(segment_meta["segment_id"])
    generation = int(segment_meta["active_generation"])
    segment_meta.update({
      "logical_stream_id": logical_stream_id,
      "rotated_from_path": logical_stream_id,
      "telemetry_source_id": _source_id(
        logical_stream_id,
        "segment",
        segment_id,
      ),
      "rotated_from_source_id": _source_id(
        logical_stream_id,
        "active",
        f"{generation:06d}",
      ),
    })
    if relocate_segment_identity:
      rotated_identity = dict(segment_meta["rotated_from_file_identity"])
      rotated_identity["st_dev"] = int(rotated_identity["st_dev"]) + 1
      rotated_identity["st_ino"] = int(rotated_identity["st_ino"]) + 1
      segment_meta["rotated_from_file_identity"] = rotated_identity
    segment_meta_path.write_text(
      json.dumps(segment_meta),
      encoding="utf-8",
    )
    segment_payloads[segment_id] = segment_meta

  manifest_path = log.segments_dir / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest_logical_stream_id = (
    str(log.path)
    if manifest_uses_physical_path
    else logical_stream_id
  )
  manifest["logical_stream_id"] = manifest_logical_stream_id
  manifest["active_telemetry_source_id"] = _source_id(
    manifest_logical_stream_id,
    "active",
    f"{active_generation:06d}",
  )
  for descriptor in manifest["segments"]:
    segment = segment_payloads[str(descriptor["segment_id"])]
    descriptor["telemetry_source_id"] = segment["telemetry_source_id"]
    descriptor["rotated_from_source_id"] = segment[
      "rotated_from_source_id"
    ]
    descriptor["rotated_from_path"] = logical_stream_id
    descriptor["rotated_from_file_identity"] = segment[
      "rotated_from_file_identity"
    ]
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _enumerate_v2(base: Path):
  return enumerate_selected_agent_session_logs(
    base,
    layout="v2",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank"}),
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
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
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
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
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
  )
  assert current == ()

  archived = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v2",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
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
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
  )
  assert [item.path for item in selected] == [log.path]


@pytest.mark.parametrize(
  "historical_root",
  [
    "/historical/dev-checkout/api/sessions",
    "/historical/prod-release/api/sessions",
  ],
)
def test_v1_accepts_self_consistent_historical_logical_lineage(
  tmp_path: Path,
  historical_root: str,
) -> None:
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  logical_stream_id = str(
    Path(historical_root) / log.path.parent.name / log.path.name
  )
  _rewrite_flat_lineage(log, logical_stream_id)

  selected = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v1",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
  )

  assert [item.path for item in selected] == [log.path]
  assert selected[0].sidecar_payload is not None
  assert (
    selected[0].sidecar_payload["logical_stream_id"]
    == logical_stream_id
  )


def test_v1_accepts_current_physical_logical_lineage(tmp_path: Path) -> None:
  log = _prepare_flat_batch(tmp_path, product_id="hank")
  _rewrite_flat_lineage(log, str(log.path))

  selected = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v1",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank"}),
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
  )

  assert [item.path for item in selected] == [log.path]


def test_v1_rejects_historical_lineage_for_another_family(
  tmp_path: Path,
) -> None:
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  _rewrite_flat_lineage(
    log,
    f"/historical/sessions/other-family/{log.path.name}",
  )

  with pytest.raises(SessionLogInventoryError, match="historical lineage"):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank", "hank-dev"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
    )


@pytest.mark.parametrize(
  "logical_stream_id",
  [
    "relative/sessions/b1_2/agentsess_b1_2_henry.jsonl",
    "/historical/sessions/../sessions/b1_2/agentsess_b1_2_henry.jsonl",
    "/historical/sessions/b1_2/agentsess_other_henry.jsonl",
    "/historical\x00/sessions/b1_2/agentsess_b1_2_henry.jsonl",
  ],
)
def test_v1_rejects_noncanonical_or_wrong_basename_lineage(
  tmp_path: Path,
  logical_stream_id: str,
) -> None:
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  _rewrite_flat_lineage(log, logical_stream_id)

  with pytest.raises(SessionLogInventoryError, match="historical lineage"):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank", "hank-dev"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
    )


def test_v1_rejects_historical_lineage_source_id_mismatch(
  tmp_path: Path,
) -> None:
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  logical_stream_id = (
    f"/historical/sessions/{log.path.parent.name}/{log.path.name}"
  )
  _rewrite_flat_lineage(log, logical_stream_id)
  meta_path = log.path.with_suffix(".meta.json")
  payload = json.loads(meta_path.read_text(encoding="utf-8"))
  payload["telemetry_source_id"] = "agent_session_log:forged:active:000000"
  meta_path.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match="historical lineage"):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank", "hank-dev"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
    )


def test_v1_accepts_relocated_rotated_lineage_with_repaired_manifest(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  historical = (
    "/historical/prod-release/api/sessions/"
    f"{log.path.parent.name}/{log.path.name}"
  )
  _rewrite_flat_lineage(
    log,
    historical,
    manifest_uses_physical_path=True,
    relocate_segment_identity=True,
  )

  selected = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v1",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
  )

  assert [item.path for item in selected] == [log.path]
  assert selected[0].manifest_payload is not None
  assert selected[0].manifest_payload["logical_stream_id"] == str(log.path)
  segment = next(
    item for item in selected[0].files if item.role == "segment"
  )
  assert segment.sidecar_payload is not None
  assert segment.sidecar_payload["logical_stream_id"] == historical


@pytest.mark.parametrize("lineage_change", ["root", "namespace", "basename"])
def test_v1_repaired_manifest_preserves_historical_lineage(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  lineage_change: str,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  historical = (
    f"/historical/AI-excel-addin/api/sessions/{log.path.parent.name}/{log.path.name}"
  )
  _rewrite_flat_lineage(log, historical, relocate_segment_identity=True)
  manifest_lineage = Path(historical.replace("AI-excel-addin", "ai-excel-addin"))
  if lineage_change == "namespace":
    manifest_lineage = manifest_lineage.parent.with_name("other-family") / log.path.name
  elif lineage_change == "basename":
    manifest_lineage = manifest_lineage.with_name("agentsess_other_henry.jsonl")
  manifest_path = log.segments_dir / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["logical_stream_id"] = str(manifest_lineage)
  manifest["active_telemetry_source_id"] = _source_id(
    str(manifest_lineage), "active", f"{manifest['active_generation']:06d}",
  )
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  def enumerate_relocated():
    return enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank-dev",
      allowed_product_ids=frozenset({"hank-dev"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
      strict_rotated=True,
    )

  if lineage_change != "root":
    with pytest.raises(SessionLogInventoryError):
      enumerate_relocated()
    return

  selected = enumerate_relocated()
  assert [item.path for item in selected] == [log.path]
  assert selected[0].manifest_payload["logical_stream_id"] == str(manifest_lineage)
  retained = next(item for item in selected[0].files if item.role == "segment")
  assert retained.sidecar_payload["logical_stream_id"] == historical


@pytest.mark.parametrize("layout", ["v1", "v2"])
@pytest.mark.parametrize(
  ("mtime_delta_ns", "extra_bytes", "accepted"),
  [
    pytest.param(52, b"", True, id="relocation-rounding"),
    pytest.param(238, b"", True, id="positive-precision-bound"),
    pytest.param(-238, b"", True, id="negative-precision-bound"),
    pytest.param(239, b"", False, id="above-positive-precision-bound"),
    pytest.param(-239, b"", False, id="below-negative-precision-bound"),
    pytest.param(1_000_000_000, b"", False, id="modified-same-size"),
    pytest.param(52, b" ", False, id="changed-size"),
  ],
)
def test_relocated_segment_preserves_timestamp_precision(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  layout: str,
  mtime_delta_ns: int,
  extra_bytes: bytes,
  accepted: bool,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  original = tmp_path / "original"
  original.mkdir(mode=0o700)
  relocated = tmp_path / "relocated"
  product_id = "hank" if layout == "v2" else "hank-dev"
  if layout == "v2":
    active, _meta = _prepare_v2(original)
    log = AgentSessionLog(active)
  else:
    log = _prepare_flat_batch(original, product_id=product_id)
  log.append_sync({"type": "user_message", "body": "first"})
  # At this fixed epoch the float-second ULP is 238.4185791015625 ns.
  recorded_mtime_ns = 1784202247994842239
  os.utime(log.path, ns=(recorded_mtime_ns, recorded_mtime_ns))
  log.append_sync({"type": "user_message", "body": "second"})
  log.repair_manifest()
  active_relative = log.path.relative_to(original)
  segment_relative = next(log.segments_dir.glob("*.jsonl")).relative_to(original)
  original.rename(relocated)
  segment_path = relocated / segment_relative
  original_info = segment_path.stat()
  original_bytes = segment_path.read_bytes()
  replacement = segment_path.with_suffix(".replacement")
  replacement.write_bytes(original_bytes + extra_bytes)
  os.utime(
    replacement,
    ns=(original_info.st_atime_ns, original_info.st_mtime_ns + mtime_delta_ns),
  )
  replacement.replace(segment_path)

  def enumerate_relocated():
    return enumerate_selected_agent_session_logs(
      relocated,
      layout=layout,
      trusted_product_id=product_id,
      allowed_product_ids=frozenset({product_id}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
      strict_rotated=True,
    )

  if not accepted:
    with pytest.raises(SessionLogInventoryError, match="physical"):
      enumerate_relocated()
    return

  selected = enumerate_relocated()
  assert [item.path for item in selected] == [relocated / active_relative]
  retained = next(item for item in selected[0].files if item.role == "segment")
  assert retained.path.read_bytes() == original_bytes
  assert retained.sidecar_payload is not None
  assert retained.sidecar_payload["logical_stream_id"] == str(original / active_relative)


@pytest.mark.parametrize("layout", ["v1", "v2"])
@pytest.mark.parametrize("storage", ["empty", "invalid-manifest", "missing-manifest"])
def test_manifest_requirement_follows_rotated_storage(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  layout: str,
  storage: str,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  if layout == "v2":
    active, _meta = _prepare_v2(tmp_path)
    log = AgentSessionLog(active)
  else:
    log = _prepare_flat_batch(tmp_path, product_id="hank")
  log.segments_dir.mkdir(mode=0o700, exist_ok=True)
  if storage == "invalid-manifest":
    log.manifest_path.write_text("{", encoding="utf-8")
  elif storage == "missing-manifest":
    log.append_sync({"type": "user_message", "body": "first"})
    log.append_sync({"type": "user_message", "body": "second"})
    log.manifest_path.unlink()

  def enumerate_storage():
    return enumerate_selected_agent_session_logs(
      tmp_path,
      layout=layout,
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
      strict_rotated=True,
    )

  if storage != "empty":
    with pytest.raises(SessionLogInventoryError, match="requires a valid manifest"):
      enumerate_storage()
    return

  selected = enumerate_storage()
  assert [item.path for item in selected] == [log.path]
  assert all(physical.role == "active" for physical in selected[0].files)


def test_foreign_jsonl_in_segments_dir_does_not_refuse_inventory(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  (log.segments_dir / "not-a-segment.jsonl").write_text(
    '{"stray": true}\n', encoding="utf-8"
  )

  selected = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v1",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
  )

  assert [item.path for item in selected] == [log.path]
  segment_names = {
    item.path.name for item in selected[0].files if item.role == "segment"
  }
  assert "not-a-segment.jsonl" not in segment_names
  assert segment_names


def test_v1_rejects_historical_rotated_size_drift(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  historical = (
    "/historical/prod-release/api/sessions/"
    f"{log.path.parent.name}/{log.path.name}"
  )
  _rewrite_flat_lineage(
    log,
    historical,
    relocate_segment_identity=True,
  )
  segment_meta_path = next(log.segments_dir.glob("*.meta.json"))
  segment = json.loads(segment_meta_path.read_text(encoding="utf-8"))
  segment["rotated_from_file_identity"]["size"] += 1
  segment_meta_path.write_text(json.dumps(segment), encoding="utf-8")
  manifest_path = log.segments_dir / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["segments"][0]["rotated_from_file_identity"] = segment[
    "rotated_from_file_identity"
  ]
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match="physical storage"):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank", "hank-dev"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
    )


def test_v1_rejects_historical_sidecar_manifest_identity_disagreement(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  historical = (
    "/historical/prod-release/api/sessions/"
    f"{log.path.parent.name}/{log.path.name}"
  )
  _rewrite_flat_lineage(
    log,
    historical,
    relocate_segment_identity=True,
  )
  manifest_path = log.segments_dir / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["segments"][0]["rotated_from_file_identity"]["st_ino"] += 1
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  with pytest.raises(
    SessionLogInventoryError,
    match="manifest contradicts physical",
  ):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank", "hank-dev"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
    )


def test_v1_current_lineage_rejects_rotated_inode_drift(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  segment_meta_path = next(log.segments_dir.glob("*.meta.json"))
  segment = json.loads(segment_meta_path.read_text(encoding="utf-8"))
  segment["rotated_from_file_identity"]["st_ino"] += 1
  segment_meta_path.write_text(json.dumps(segment), encoding="utf-8")
  manifest_path = log.segments_dir / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  manifest["segments"][0]["rotated_from_file_identity"] = segment[
    "rotated_from_file_identity"
  ]
  manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match="physical storage"):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank", "hank-dev"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
    )


def test_v1_preserves_schema1_rotated_active_compatibility(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  active_meta_path = log.path.with_suffix(".meta.json")
  active_meta = json.loads(active_meta_path.read_text(encoding="utf-8"))
  for field in (
    "file_role",
    "logical_stream_id",
    "telemetry_source_id",
    "active_generation",
  ):
    active_meta.pop(field)
  active_meta["schema_version"] = 1
  active_meta_path.write_text(json.dumps(active_meta), encoding="utf-8")

  selected = enumerate_selected_agent_session_logs(
    tmp_path,
    layout="v1",
    trusted_product_id="hank",
    allowed_product_ids=frozenset({"hank", "hank-dev"}),
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
  )

  assert [item.path for item in selected] == [log.path]


@pytest.mark.parametrize(
  ("target", "field", "value", "message"),
  [
    ("segment", "logical_stream_id", "/other/family/log.jsonl", "segment metadata"),
    (
      "segment",
      "rotated_from_source_id",
      "agent_session_log:forged:active:000000",
      "physical storage",
    ),
    (
      "manifest",
      "active_telemetry_source_id",
      "agent_session_log:forged:active:000001",
      "manifest contradicts",
    ),
    (
      "descriptor",
      "rotated_from_path",
      "/other/family/log.jsonl",
      "manifest contradicts physical",
    ),
  ],
)
def test_v1_rejects_rotated_historical_lineage_contradiction(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  target: str,
  field: str,
  value: str,
  message: str,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  log = _prepare_flat_batch(tmp_path, product_id="hank-dev")
  log.append_sync({"type": "user_message", "body": "first"})
  log.append_sync({"type": "user_message", "body": "second"})
  historical = (
    "/historical/dev-checkout/api/sessions/"
    f"{log.path.parent.name}/{log.path.name}"
  )
  _rewrite_flat_lineage(log, historical)
  segment_meta_path = next(log.segments_dir.glob("*.meta.json"))
  manifest_path = log.segments_dir / "manifest.json"
  if target == "segment":
    payload = json.loads(segment_meta_path.read_text(encoding="utf-8"))
    payload[field] = value
    segment_meta_path.write_text(json.dumps(payload), encoding="utf-8")
  else:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if target == "manifest":
      manifest[field] = value
    else:
      manifest["segments"][0][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match=message):
    enumerate_selected_agent_session_logs(
      tmp_path,
      layout="v1",
      trusted_product_id="hank",
      allowed_product_ids=frozenset({"hank", "hank-dev"}),
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
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
    allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
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
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
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
      allowed_stream_kinds=SUPPORTED_SESSION_LOG_STREAM_KINDS,
    )


def test_v2_rejects_unknown_active_sidecar_field(tmp_path: Path) -> None:
  _active, meta = _prepare_v2(tmp_path)
  payload = json.loads(meta.read_text(encoding="utf-8"))
  payload["unexpected"] = "field"
  meta.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError, match="closed field contract"):
    _enumerate_v2(tmp_path)


@pytest.mark.asyncio
async def test_v2_preserves_moved_stream_read_rotation_recovery_and_retirement(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  old, new = tmp_path / "OLD", tmp_path / "NEW"
  old.mkdir(mode=0o700)
  active, meta = _prepare_v2(old)
  log = AgentSessionLog(active)
  for ordinal in (1, 2):
    assert log.append_sync({"type": "user_message", "ordinal": ordinal}).seq == ordinal
  log.repair_manifest()
  before = _enumerate_v2(old)[0]
  lineage = str(active)
  retained = {
    physical.path.relative_to(old): (
      physical.path.read_bytes(), physical.sidecar_path.read_bytes(),
    )
    for physical in before.files
    if physical.role == "segment"
  }
  active_meta_bytes = meta.read_bytes()
  del log
  old.rename(new)

  moved_active, moved_meta = _prepare_v2(new)
  assert moved_active == new / active.relative_to(old)
  assert moved_meta.read_bytes() == active_meta_bytes
  moved = _enumerate_v2(new)[0]
  log = AgentSessionLog(moved.location)
  for ordinal in (3, 4):
    assert log.append_sync({"type": "user_message", "ordinal": ordinal}).seq == ordinal
  after = _enumerate_v2(new)[0]
  for relative, (body, sidecar) in retained.items():
    assert (new / relative).read_bytes() == body
    assert (new / relative).with_suffix(".meta.json").read_bytes() == sidecar
  assert not old.exists()
  assert after.sidecar_payload["logical_stream_id"] == lineage
  for physical in after.files:
    payload = physical.sidecar_payload
    assert payload["logical_stream_id"] == lineage
    suffix = (
      physical.path.stem if physical.role == "segment"
      else f'{payload["active_generation"]:06d}'
    )
    assert payload["telemetry_source_id"] == _source_id(lineage, physical.role, suffix)
  entries, _ = await log.query_current_strict()
  assert [entry.seq for entry in entries] == [1, 2, 3, 4]
  assert [entry.event["ordinal"] for entry in entries] == [1, 2, 3, 4]

  # Rebuild from retained segments after losing only the manifest, then retire
  # the prefix in order, including segments rotated on either side of the move.
  log.manifest_path.unlink()
  log = AgentSessionLog(moved_active)
  recovered = _enumerate_v2(new)[0]
  assert recovered.manifest_payload["logical_stream_id"] == lineage
  assert [
    row["telemetry_source_id"] for row in recovered.manifest_payload["segments"]
  ] == [
    row["telemetry_source_id"] for row in after.manifest_payload["segments"]
  ]
  def identity(value):
    return (value.device, value.inode, value.size, value.mtime_ns)

  remaining = recovered
  while remaining.manifest_payload["segments"]:
    descriptor = remaining.manifest_payload["segments"][0]
    segment = next(
      physical for physical in remaining.files
      if physical.path.name == descriptor["path"]
    )

    retired = AgentSessionLog.retire_segment(
      remaining.location,
      descriptor,
      expected_manifest=remaining.manifest_payload,
      expected_manifest_identity=identity(remaining.manifest_identity),
      expected_segment_identity=identity(segment.file_identity),
      expected_sidecar_identity=identity(segment.sidecar_identity),
    )
    assert retired is not None
    assert not segment.path.exists()
    assert not segment.sidecar_path.exists()
    entries, _ = await AgentSessionLog(moved_active).query_current_strict()
    assert [entry.seq for entry in entries] == list(range(descriptor["last_seq"] + 1, 5))
    remaining = _enumerate_v2(new)[0]
    assert remaining.manifest_payload["logical_stream_id"] == lineage


@pytest.mark.parametrize("drift", ["relative", "traversal", "nul", "namespace", "source_id"])
def test_v2_rejects_invalid_historical_lineage(tmp_path: Path, drift: str) -> None:
  active, meta = _prepare_v2(tmp_path)
  payload = json.loads(meta.read_text(encoding="utf-8"))
  historical = f"/historical/sessions/{active.parent.name}/{active.name}"
  if drift == "relative":
    historical = historical.lstrip("/")
  elif drift == "traversal":
    historical = historical.replace("/sessions/", "/discard/../sessions/")
  elif drift == "nul":
    historical = historical.replace("/sessions/", "/sessions\x00/")
  elif drift == "namespace":
    historical = historical.replace(active.parent.name, "another-stream", 1)
  payload["logical_stream_id"] = historical
  if drift != "source_id":
    payload["telemetry_source_id"] = _source_id(historical, "active", "000000")
  meta.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError):
    _enumerate_v2(tmp_path)


@pytest.mark.parametrize("target", ["segment", "manifest", "descriptor"])
def test_v2_rejects_rotated_lineage_disagreement(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  target: str,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "1")
  active, _meta = _prepare_v2(tmp_path)
  log = AgentSessionLog(active)
  log.append_sync({"type": "user_message", "ordinal": 1})
  log.append_sync({"type": "user_message", "ordinal": 2})
  log.repair_manifest()
  selected = _enumerate_v2(tmp_path)[0]
  if target == "segment":
    path = next(physical.sidecar_path for physical in selected.files if physical.role == "segment")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["logical_stream_id"] = "/other/stream.jsonl"
  else:
    path = log.manifest_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if target == "manifest":
      payload["logical_stream_id"] = "/other/stream.jsonl"
    else:
      payload["segments"][0]["rotated_from_path"] = "/other/stream.jsonl"
  path.write_text(json.dumps(payload), encoding="utf-8")

  with pytest.raises(SessionLogInventoryError):
    _enumerate_v2(tmp_path)
