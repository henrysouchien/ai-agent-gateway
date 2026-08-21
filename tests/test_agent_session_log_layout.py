from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path

import pytest

from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.agent_session_log_layout import (
  AgentSessionLogLayoutError,
  SESSION_LOG_LAYOUT_ENV,
  derive_v2_agent_session_log_paths,
  prepare_autonomous_session_log,
  resolve_agent_session_log_layout,
  verify_autonomous_session_log,
)


def _now_iso() -> str:
  return datetime(2026, 8, 20, tzinfo=UTC).isoformat()


def _prepare(base: Path):
  return prepare_autonomous_session_log(
    base_dir=base,
    layout="v2",
    tenant="hank",
    owner="henry",
    workload_profile="analyst",
    provider="openai",
    provider_session_epoch="responses-v1",
    now_iso=_now_iso,
  )


def test_layout_choice_is_explicit_and_bounded() -> None:
  assert resolve_agent_session_log_layout({}) == "v1"
  assert resolve_agent_session_log_layout({SESSION_LOG_LAYOUT_ENV: "v2"}) == "v2"
  with pytest.raises(AgentSessionLogLayoutError, match="must be v1 or v2"):
    resolve_agent_session_log_layout({SESSION_LOG_LAYOUT_ENV: "proof"})


def test_v2_path_is_collapsed_and_full_digest_detects_prefix_aliases(
  tmp_path: Path,
) -> None:
  root, active, meta, digest = derive_v2_agent_session_log_paths(
    tmp_path,
    tenant="hank",
    owner="henry",
    workload_profile="analyst",
    provider="anthropic",
    provider_session_epoch=None,
  )
  assert root.parent == tmp_path / ".session-log-v2"
  assert root.name == f"s-{digest[:52]}"
  assert active.parent == root
  assert active.name == f"agentsess_s-{digest[:52]}_henry.jsonl"
  assert meta == active.with_suffix(".meta.json")
  assert len(digest) == 64
  with pytest.raises(ValueError, match="must be null"):
    derive_v2_agent_session_log_paths(
      tmp_path,
      tenant="hank",
      owner="henry",
      workload_profile="analyst",
      provider="anthropic",
      provider_session_epoch="ignored",
    )


def test_prepare_and_verify_bind_exact_root_active_and_sidecar(
  tmp_path: Path,
) -> None:
  prepared = _prepare(tmp_path)
  verified = None
  child_fds = tuple(os.dup(fd) for fd in prepared.pass_fds)
  try:
    authority = prepared.authority
    assert authority.root_path is not None
    assert authority.active_path is not None
    assert authority.meta_path is not None
    assert Path(authority.active_path).parent == Path(authority.root_path)
    payload = json.loads(Path(authority.meta_path).read_text(encoding="utf-8"))
    assert payload["storage_layout"] == 2
    assert payload["tenant_id"] == "hank"
    assert payload["agent_id"] == "analyst--openai-responses-v1"
    assert payload["storage_identity_digest"] == authority.storage_identity_digest
    verified = verify_autonomous_session_log(
      authority,
      root_fd=child_fds[0],
      active_fd=child_fds[1],
      meta_fd=child_fds[2],
      tenant="hank",
      owner="henry",
      workload_profile="analyst",
      provider="openai",
      projected_base_path=str(tmp_path),
    )
    assert verified.location is not None
    assert verified.location.path == Path(authority.active_path)
    assert verified.location.active_identity == (
      authority.active_device,
      authority.active_inode,
    )
  finally:
    if verified is not None:
      verified.close()
    else:
      for fd in child_fds:
        os.close(fd)
    prepared.close()


def test_verify_rejects_same_owner_named_replacement(tmp_path: Path) -> None:
  prepared = _prepare(tmp_path)
  authority = prepared.authority
  child_fds = tuple(os.dup(fd) for fd in prepared.pass_fds)
  assert authority.active_path is not None
  active = Path(authority.active_path)
  displaced = active.with_name("displaced.jsonl")
  os.replace(active, displaced)
  active.write_bytes(b"")
  os.chmod(active, 0o600)
  try:
    with pytest.raises(AgentSessionLogLayoutError, match="active path was displaced"):
      verify_autonomous_session_log(
        authority,
        root_fd=child_fds[0],
        active_fd=child_fds[1],
        meta_fd=child_fds[2],
        tenant="hank",
        owner="henry",
        workload_profile="analyst",
        provider="openai",
        projected_base_path=str(tmp_path),
      )
  finally:
    for fd in child_fds:
      os.close(fd)
    prepared.close()


def test_v1_authority_inherits_no_storage_descriptors(tmp_path: Path) -> None:
  prepared = prepare_autonomous_session_log(
    base_dir=tmp_path,
    layout="v1",
    tenant="hank",
    owner="henry",
    workload_profile="analyst",
    provider="anthropic",
    provider_session_epoch=None,
    now_iso=_now_iso,
  )
  assert prepared.pass_fds == ()
  verified = verify_autonomous_session_log(
    prepared.authority,
    root_fd=None,
    active_fd=None,
    meta_fd=None,
    tenant="hank",
    owner="henry",
    workload_profile="analyst",
    provider="anthropic",
    projected_base_path=str(tmp_path),
  )
  assert verified.location is None


def test_prepare_rejects_writable_managed_base(tmp_path: Path) -> None:
  tmp_path.chmod(0o777)
  with pytest.raises(AgentSessionLogLayoutError, match="session-log base is unsafe"):
    _prepare(tmp_path)


def test_prepare_rejects_existing_sidecar_identity_drift(tmp_path: Path) -> None:
  prepared = _prepare(tmp_path)
  authority = prepared.authority
  prepared.close()
  assert authority.meta_path is not None
  payload = json.loads(Path(authority.meta_path).read_text(encoding="utf-8"))
  payload["tenant_id"] = "peer"
  Path(authority.meta_path).write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
  )
  os.chmod(authority.meta_path, 0o600)
  with pytest.raises(
    AgentSessionLogLayoutError,
    match="tenant_id does not match storage authority",
  ):
    _prepare(tmp_path)


@pytest.mark.parametrize("incomplete_bytes", [b"", b'{"schema_version":'])
def test_prepare_recovers_interrupted_sidecar_initialization(
  tmp_path: Path,
  incomplete_bytes: bytes,
) -> None:
  prepared = _prepare(tmp_path)
  authority = prepared.authority
  prepared.close()
  assert authority.active_path is not None
  assert authority.meta_path is not None
  active = Path(authority.active_path)
  meta = Path(authority.meta_path)
  assert active.stat().st_size == 0
  meta.write_bytes(incomplete_bytes)
  os.chmod(meta, 0o600)

  recovered = _prepare(tmp_path)
  try:
    payload = json.loads(meta.read_text(encoding="utf-8"))
    assert payload["storage_identity_digest"] == (
      recovered.authority.storage_identity_digest
    )
    assert payload["agent_id"] == "analyst--openai-responses-v1"
  finally:
    recovered.close()


def test_prepare_recovers_active_created_before_sidecar_publication(
  tmp_path: Path,
) -> None:
  prepared = _prepare(tmp_path)
  authority = prepared.authority
  prepared.close()
  assert authority.active_path is not None
  assert authority.meta_path is not None
  assert Path(authority.active_path).stat().st_size == 0
  Path(authority.meta_path).unlink()

  recovered = _prepare(tmp_path)
  try:
    payload = json.loads(
      Path(authority.meta_path).read_text(encoding="utf-8")
    )
    assert payload["storage_identity_digest"] == (
      recovered.authority.storage_identity_digest
    )
  finally:
    recovered.close()


def test_prepare_rejects_incomplete_sidecar_after_active_append(
  tmp_path: Path,
) -> None:
  prepared = _prepare(tmp_path)
  authority = prepared.authority
  prepared.close()
  assert authority.active_path is not None
  assert authority.meta_path is not None
  Path(authority.active_path).write_bytes(b'{"seq":1}\n')
  Path(authority.meta_path).write_bytes(b'{"schema_version":')

  with pytest.raises(AgentSessionLogLayoutError, match="sidecar is invalid"):
    _prepare(tmp_path)


def test_prepare_rejects_duplicate_or_unknown_sidecar_fields(
  tmp_path: Path,
) -> None:
  prepared = _prepare(tmp_path)
  authority = prepared.authority
  prepared.close()
  assert authority.meta_path is not None
  meta = Path(authority.meta_path)
  payload = json.loads(meta.read_text(encoding="utf-8"))
  meta.write_text(
    '{"schema_version":2,"schema_version":2}\n',
    encoding="utf-8",
  )
  os.chmod(meta, 0o600)
  with pytest.raises(AgentSessionLogLayoutError, match="duplicate fields"):
    _prepare(tmp_path)

  payload["unexpected_identity"] = "peer"
  meta.write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
  )
  os.chmod(meta, 0o600)
  with pytest.raises(AgentSessionLogLayoutError, match="closed field contract"):
    _prepare(tmp_path)


@pytest.mark.asyncio
async def test_v2_location_append_rotation_manifest_and_read_lifecycle(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("AGENT_SESSION_LOG_MAX_ACTIVE_BYTES", "256")
  prepared = _prepare(tmp_path)
  child_fds = tuple(os.dup(fd) for fd in prepared.pass_fds)
  verified = verify_autonomous_session_log(
    prepared.authority,
    root_fd=child_fds[0],
    active_fd=child_fds[1],
    meta_fd=child_fds[2],
    tenant="hank",
    owner="henry",
    workload_profile="analyst",
    provider="openai",
    projected_base_path=str(tmp_path),
  )
  try:
    assert verified.location is not None
    log = AgentSessionLog(verified.location)
    for ordinal in range(4):
      log.append_sync({
        "type": "foundation_canary",
        "ordinal": ordinal,
        "body": "x" * 384,
      })
    entries, _cursor = await log.query_current_strict(
      event_types={"foundation_canary"},
    )
    assert [entry.event["ordinal"] for entry in entries] == [0, 1, 2, 3]
    manifest = json.loads(log.manifest_path.read_text(encoding="utf-8"))
    active_meta = json.loads(
      log.path.with_suffix(".meta.json").read_text(encoding="utf-8")
    )
    assert manifest["segments"]
    for field in (
      "storage_layout",
      "tenant_id",
      "workload_profile",
      "provider",
      "provider_session_epoch",
      "storage_identity_digest",
    ):
      assert manifest[field] == active_meta[field]
    for segment in log.segments_dir.glob("*.jsonl"):
      segment_meta = json.loads(
        segment.with_suffix(".meta.json").read_text(encoding="utf-8")
      )
      assert segment_meta["logical_stream_id"] == str(log.path)
      assert segment_meta["storage_identity_digest"] == (
        prepared.authority.storage_identity_digest
      )
  finally:
    verified.close()
    prepared.close()
