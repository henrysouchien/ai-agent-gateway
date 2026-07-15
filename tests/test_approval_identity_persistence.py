from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from agent_gateway.approval_audit import AuditBuildError, build_audit_entry
from agent_gateway.approval_policy import ApprovalRequest
from agent_gateway.approval_store import SQLiteApprovalStore
from agent_gateway.audit_writer import JSONLAuditWriter


_FMS_DIGEST_A = "a" * 64
_FMS_DIGEST_B = "b" * 64
_FMS_DIGEST_C = "c" * 64
_REVIEWED_DIGEST_A = f"sha256:{'a' * 64}"
_REVIEWED_DIGEST_B = f"sha256:{'b' * 64}"
_REVIEWED_DIGEST_C = f"sha256:{'c' * 64}"
_REVIEWED_DIGEST_D = f"sha256:{'d' * 64}"
_REVIEWED_DIGEST_E = f"sha256:{'e' * 64}"
_REVIEWED_CHANGE_ID = f"reviewed-change:v1:{'d' * 64}"
_REVIEWED_CHANGE_HASH = (
  "sha256:80189bae1d552eba5327d496d393cb6fb4abe0cb48995b07d8808505428b46b1"
)
_ALTERNATE_REVIEWED_CHANGE_ID = f"reviewed-change:v1:{'a' * 64}"
_ALTERNATE_REVIEWED_CHANGE_HASH = (
  "sha256:81489a8d791ea698ccae88a3d22459c0c417f9884989c105f1627e04ada418ce"
)

_IDENTITY_FIELDS = (
  "identity_source",
  "change_set_id",
  "change_hash",
  "base_vector_hash",
  "reviewed_change_binding_digest",
  "review_reference",
  "execution_semantics_digest",
)


def _review_reference(
  *,
  authority: str,
  review_domain: str,
  snapshot_digest: str = _REVIEWED_DIGEST_A,
  review_binding_id: str | None = None,
  bundle_manifest_digest: str | None = None,
) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "authority": authority,
    "review_domain": review_domain,
    "snapshot_digest": snapshot_digest,
    "review_binding_id": review_binding_id,
    "bundle_manifest_digest": bundle_manifest_digest,
  }


_CANONICAL_REVIEW_REFERENCE_JSON = json.dumps(
  _review_reference(
    authority="advisory",
    review_domain="platform.diligence_pr.review_snapshot.v1",
    snapshot_digest=_REVIEWED_DIGEST_A,
    review_binding_id="diligence-review-417",
    bundle_manifest_digest=_REVIEWED_DIGEST_B,
  ),
  sort_keys=True,
  separators=(",", ":"),
  ensure_ascii=True,
  allow_nan=False,
)


def _run(awaitable: Any) -> Any:
  return asyncio.run(awaitable)


def _request(
  *,
  approval_id: str,
  identity_source: str | None = None,
) -> ApprovalRequest:
  identity: dict[str, Any]
  if identity_source == "change_set":
    identity = {
      "identity_source": "change_set",
      "change_set_id": _FMS_DIGEST_A,
      "change_hash": _FMS_DIGEST_B,
      "base_vector_hash": _FMS_DIGEST_C,
      "review_reference": _review_reference(
        authority="human",
        review_domain="platform.github.pull_request.v1",
        review_binding_id="example/research#417",
      ),
    }
  elif identity_source == "reviewed_change_binding":
    identity = {
      "identity_source": "reviewed_change_binding",
      "change_set_id": _REVIEWED_CHANGE_ID,
      "change_hash": _REVIEWED_CHANGE_HASH,
      "base_vector_hash": _REVIEWED_DIGEST_C,
      "reviewed_change_binding_digest": _REVIEWED_DIGEST_D,
      "review_reference": _review_reference(
        authority="advisory",
        review_domain="platform.diligence_pr.review_snapshot.v1",
        snapshot_digest=_REVIEWED_DIGEST_A,
        review_binding_id="diligence-review-417",
        bundle_manifest_digest=_REVIEWED_DIGEST_B,
      ),
      "execution_semantics_digest": _REVIEWED_DIGEST_E,
    }
  elif identity_source is None:
    identity = {}
  else:
    raise AssertionError(f"unsupported test identity source: {identity_source}")
  return ApprovalRequest(
    approval_id=approval_id,
    tool_call_id=f"tool-{approval_id}",
    parent_approval_id=None,
    approval_chain_id=f"chain-{approval_id}",
    request_id=f"request-{approval_id}",
    session_id="session-1",
    run_id="run-1",
    user_id="user-1",
    profile="chat",
    channel="web",
    tool_name="apply_research_change",
    tool_class="external_write",
    tool_args_redacted={"proposal_id": "proposal-1"},
    args_hash="legacy-args-hash",
    reason="approval required",
    blast_radius_summary="one research file",
    state="pending_user",
    requested_at=datetime(2026, 7, 14, 12, 0, tzinfo=UTC),
    policy_id="test-policy",
    policy_version="1",
    policy_bundle_hash="test-policy-bundle",
    **identity,
  )


def _identity_snapshot(request: ApprovalRequest) -> dict[str, Any]:
  return {field: getattr(request, field) for field in _IDENTITY_FIELDS}


@pytest.mark.parametrize("identity_source", ["change_set", "reviewed_change_binding"])
def test_identity_round_trips_and_survives_store_reopen(
  tmp_path: Path,
  identity_source: str,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  request = _request(approval_id=f"approval-{identity_source}", identity_source=identity_source)
  expected = _identity_snapshot(request)

  first_store = SQLiteApprovalStore(path)
  _run(first_store.create(request))
  loaded = _run(first_store.get(request.approval_id))
  assert loaded is not None
  assert _identity_snapshot(loaded) == expected
  assert _identity_snapshot(_run(first_store.get_by_tool_call_id(request.tool_call_id))) == expected
  with sqlite3.connect(path) as conn:
    stored_reference = conn.execute(
      "SELECT review_reference_json FROM approval_requests WHERE approval_id = ?",
      (request.approval_id,),
    ).fetchone()[0]
  assert stored_reference == json.dumps(
    expected["review_reference"],
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
  )

  reopened_store = SQLiteApprovalStore(path)
  transitioned = _run(
    reopened_store.transition_state(
      request.approval_id,
      "approved",
      expected_state_version=0,
      decider_id="reviewer-1",
      decision="approved",
      decision_reason="reviewed",
    )
  )
  assert transitioned.state == "approved"
  assert _identity_snapshot(transitioned) == expected

  reopened_again = SQLiteApprovalStore(path)
  after_restart = _run(reopened_again.get(request.approval_id))
  assert after_restart is not None
  assert _identity_snapshot(after_restart) == expected


def test_tool_call_get_or_create_is_atomic_across_store_instances(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  first_store = SQLiteApprovalStore(path)
  second_store = SQLiteApprovalStore(path)
  first = _request(
    approval_id="approval-race-a",
    identity_source="reviewed_change_binding",
  )
  second = replace(
    _request(
      approval_id="approval-race-b",
      identity_source="reviewed_change_binding",
    ),
    tool_call_id=first.tool_call_id,
  )

  async def create_both():
    return await asyncio.gather(
      first_store.create_or_get_by_tool_call_id(first),
      second_store.create_or_get_by_tool_call_id(second),
    )

  results = _run(create_both())
  approvals = [result[0] for result in results]
  created = [result[1] for result in results]

  assert created.count(True) == 1
  assert created.count(False) == 1
  assert len({approval.approval_id for approval in approvals}) == 1
  with sqlite3.connect(path) as conn:
    count = conn.execute(
      "SELECT COUNT(*) FROM approval_requests WHERE tool_call_id = ?",
      (first.tool_call_id,),
    ).fetchone()[0]
  assert count == 1


def test_old_schema_migrates_existing_row_with_null_identity(tmp_path: Path) -> None:
  path = tmp_path / "legacy.sqlite3"
  _create_legacy_approval_table(path)
  _insert_with_legacy_sql(path, approval_id="legacy-before-migration")

  store = SQLiteApprovalStore(path)
  columns = _approval_columns(path)
  assert {
    "identity_source",
    "change_set_id",
    "change_hash",
    "base_vector_hash",
    "reviewed_change_binding_digest",
    "review_reference_json",
    "execution_semantics_digest",
  } <= columns

  loaded = _run(store.get("legacy-before-migration"))
  assert loaded is not None
  assert _identity_snapshot(loaded) == dict.fromkeys(_IDENTITY_FIELDS)


def test_legacy_insert_and_update_sql_preserve_new_identity_columns(tmp_path: Path) -> None:
  path = tmp_path / "downgrade.sqlite3"
  store = SQLiteApprovalStore(path)
  bound = _request(
    approval_id="bound-before-downgrade",
    identity_source="reviewed_change_binding",
  )
  expected = _identity_snapshot(bound)
  _run(store.create(bound))

  with sqlite3.connect(path) as conn:
    conn.execute(
      """
      UPDATE approval_requests
      SET reason = ?, blast_radius_summary = ?, policy_version = ?
      WHERE approval_id = ?
      """,
      ("updated by prior binary", "legacy update", "0", bound.approval_id),
    )
  _insert_with_legacy_sql(path, approval_id="inserted-by-prior-binary")

  reopened = SQLiteApprovalStore(path)
  preserved = _run(reopened.get(bound.approval_id))
  assert preserved is not None
  assert preserved.reason == "updated by prior binary"
  assert _identity_snapshot(preserved) == expected
  legacy_insert = _run(reopened.get("inserted-by-prior-binary"))
  assert legacy_insert is not None
  assert _identity_snapshot(legacy_insert) == dict.fromkeys(_IDENTITY_FIELDS)


@pytest.mark.parametrize(
  ("updates", "error_match"),
  [
    ({"identity_source": None}, "require identity_source"),
    ({"change_hash": None}, "change_hash must be"),
    ({"change_set_id": f"reviewed-change:v1:{'0' * 64}"}, "canonically linked"),
    ({"change_hash": _REVIEWED_DIGEST_A}, "canonically linked"),
    ({"base_vector_hash": _REVIEWED_DIGEST_A}, "canonically linked"),
    ({"reviewed_change_binding_digest": None}, "reviewed_change_binding_digest must be"),
    ({"reviewed_change_binding_digest": _REVIEWED_DIGEST_A}, "canonically linked"),
    ({"execution_semantics_digest": "sha256:NOT-CANONICAL"}, "execution_semantics_digest must be"),
    ({"execution_semantics_digest": _REVIEWED_DIGEST_A}, "canonically linked"),
    ({"review_reference_json": "[]"}, "review_reference must be an object"),
    ({"review_reference_json": '{}'}, "typed review contract"),
    ({"review_reference_json": '{"non_finite":NaN}'}, "strict JSON values"),
  ],
)
def test_corrupt_or_partial_persisted_identity_fails_closed(
  tmp_path: Path,
  updates: dict[str, Any],
  error_match: str,
) -> None:
  path = tmp_path / "corrupt.sqlite3"
  store = SQLiteApprovalStore(path)
  request = _request(
    approval_id="approval-corrupt",
    identity_source="reviewed_change_binding",
  )
  _run(store.create(request))

  assignments = ", ".join(f"{column} = ?" for column in updates)
  with sqlite3.connect(path) as conn:
    conn.execute(
      f"UPDATE approval_requests SET {assignments} WHERE approval_id = ?",
      (*updates.values(), request.approval_id),
    )

  with pytest.raises(ValueError, match=error_match):
    _run(store.get(request.approval_id))


def test_malformed_review_reference_json_fails_closed(tmp_path: Path) -> None:
  path = tmp_path / "malformed-json.sqlite3"
  store = SQLiteApprovalStore(path)
  request = _request(
    approval_id="approval-malformed-json",
    identity_source="reviewed_change_binding",
  )
  _run(store.create(request))
  with sqlite3.connect(path) as conn:
    conn.execute(
      "UPDATE approval_requests SET review_reference_json = ? WHERE approval_id = ?",
      ("{not-json", request.approval_id),
    )

  with pytest.raises(json.JSONDecodeError):
    _run(store.get(request.approval_id))


@pytest.mark.parametrize(
  ("corrupt_json", "error_match"),
  [
    (
      _CANONICAL_REVIEW_REFERENCE_JSON.replace(
        '"authority":"advisory"',
        '"authority":"human","authority":"advisory"',
        1,
      ),
      "duplicate object key",
    ),
    (
      json.dumps(
        json.loads(_CANONICAL_REVIEW_REFERENCE_JSON),
        sort_keys=True,
      ),
      "canonical identity JSON encoding",
    ),
  ],
  ids=("duplicate-key", "noncanonical-whitespace"),
)
def test_persisted_review_reference_requires_unique_canonical_json(
  tmp_path: Path,
  corrupt_json: str,
  error_match: str,
) -> None:
  path = tmp_path / "noncanonical-reference.sqlite3"
  store = SQLiteApprovalStore(path)
  request = _request(
    approval_id="approval-noncanonical-reference",
    identity_source="reviewed_change_binding",
  )
  _run(store.create(request))
  with sqlite3.connect(path) as conn:
    conn.execute(
      "UPDATE approval_requests SET review_reference_json = ? WHERE approval_id = ?",
      (corrupt_json, request.approval_id),
    )

  with pytest.raises(ValueError, match=error_match):
    _run(store.get(request.approval_id))


@pytest.mark.parametrize(
  "review_reference",
  [
    {1: "integer keys would reopen as strings"},
    {"tuple": ("would", "reopen", "as", "a", "list")},
  ],
)
def test_review_reference_rejects_lossy_non_json_shapes(
  review_reference: dict[Any, Any],
) -> None:
  request = _request(
    approval_id="approval-lossy-reference",
    identity_source="reviewed_change_binding",
  )
  with pytest.raises(ValueError, match="strict JSON values"):
    replace(request, review_reference=review_reference)  # type: ignore[arg-type]


def test_review_reference_rejects_untyped_plain_json() -> None:
  request = _request(
    approval_id="approval-untyped-reference",
    identity_source="reviewed_change_binding",
  )
  with pytest.raises(ValueError, match="typed review contract"):
    replace(request, review_reference={"kind": "diligence_pr"})


def test_reviewed_binding_constructor_rejects_each_cross_field_mismatch() -> None:
  request = _request(
    approval_id="approval-cross-field",
    identity_source="reviewed_change_binding",
  )
  for field_name, new_value in (
    ("change_set_id", f"reviewed-change:v1:{'0' * 64}"),
    ("change_hash", _REVIEWED_DIGEST_A),
    ("base_vector_hash", _REVIEWED_DIGEST_A),
    ("reviewed_change_binding_digest", _REVIEWED_DIGEST_A),
    ("execution_semantics_digest", _REVIEWED_DIGEST_A),
  ):
    with pytest.raises(ValueError, match="canonically linked"):
      replace(request, **{field_name: new_value})


def test_update_request_rejects_coherent_identity_replacement(
  tmp_path: Path,
) -> None:
  path = tmp_path / "immutable.sqlite3"
  store = SQLiteApprovalStore(path)
  request = _request(
    approval_id="approval-immutable",
    identity_source="reviewed_change_binding",
  )
  expected = _identity_snapshot(request)
  _run(store.create(request))

  changed = replace(
    request,
    change_set_id=_ALTERNATE_REVIEWED_CHANGE_ID,
    change_hash=_ALTERNATE_REVIEWED_CHANGE_HASH,
    reviewed_change_binding_digest=_REVIEWED_DIGEST_A,
    review_reference=_review_reference(
      authority="advisory",
      review_domain="platform.diligence_pr.review_snapshot.v1",
      snapshot_digest=_REVIEWED_DIGEST_B,
      review_binding_id="different-review",
      bundle_manifest_digest=_REVIEWED_DIGEST_C,
    ),
  )
  with pytest.raises(ValueError, match="identity is immutable"):
    _run(store.update_request(changed))

  loaded = _run(SQLiteApprovalStore(path).get(request.approval_id))
  assert loaded is not None
  assert _identity_snapshot(loaded) == expected


def test_update_request_allows_nonidentity_changes_and_preserves_identity(
  tmp_path: Path,
) -> None:
  path = tmp_path / "mutable-metadata.sqlite3"
  store = SQLiteApprovalStore(path)
  request = _request(
    approval_id="approval-mutable-metadata",
    identity_source="reviewed_change_binding",
  )
  expected = _identity_snapshot(request)
  _run(store.create(request))

  allowed = replace(request, reason="updated without changing identity")
  updated = _run(store.update_request(allowed))
  assert updated.reason == "updated without changing identity"
  loaded = _run(SQLiteApprovalStore(path).get(request.approval_id))
  assert loaded is not None
  assert loaded.reason == "updated without changing identity"
  assert _identity_snapshot(loaded) == expected


def test_create_revalidates_mutated_request_before_persistence(tmp_path: Path) -> None:
  path = tmp_path / "mutated-before-create.sqlite3"
  store = SQLiteApprovalStore(path)
  request = _request(
    approval_id="approval-mutated-before-create",
    identity_source="reviewed_change_binding",
  )
  request.change_hash = _REVIEWED_DIGEST_A

  with pytest.raises(ValueError, match="canonically linked"):
    _run(store.create(request))

  assert _run(store.get(request.approval_id)) is None


def test_audit_revalidates_mutated_request_before_emission() -> None:
  request = _request(
    approval_id="approval-mutated-before-audit",
    identity_source="reviewed_change_binding",
  )
  assert request.review_reference is not None
  request.review_reference["snapshot_digest"] = "not-a-digest"

  with pytest.raises(AuditBuildError, match="failed"):
    build_audit_entry(
      raw_tool_args={},
      deployment_secret=b"test-secret",
      key_id="test-key",
      event_type="request_created",
      request=request,
    )


@pytest.mark.parametrize("identity_source", ["change_set", "reviewed_change_binding"])
def test_audit_entry_echoes_approval_identity_without_aliasing(
  tmp_path: Path,
  identity_source: str,
) -> None:
  request = _request(approval_id=f"approval-audit-{identity_source}", identity_source=identity_source)
  expected = _identity_snapshot(request)
  expected_review_reference = copy.deepcopy(request.review_reference)

  entry = build_audit_entry(
    raw_tool_args={},
    deployment_secret=b"test-secret",
    key_id="test-key",
    event_type="approval_requested",
    request=request,
  )

  assert {field: getattr(entry, field) for field in _IDENTITY_FIELDS} == expected
  assert entry.review_reference is not request.review_reference
  assert request.review_reference is not None
  request.review_reference["mutated_after_audit"] = True
  assert entry.review_reference == expected_review_reference

  audit_root = tmp_path / "audit"
  _run(JSONLAuditWriter(audit_root).write(entry))
  persisted, cursor = _run(
    JSONLAuditWriter(audit_root).query(approval_id=request.approval_id)
  )
  assert cursor is None
  assert len(persisted) == 1
  assert {
    field: getattr(persisted[0], field)
    for field in _IDENTITY_FIELDS
  } == {
    **expected,
    "review_reference": expected_review_reference,
  }


def _approval_columns(path: Path) -> set[str]:
  with sqlite3.connect(path) as conn:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(approval_requests)")}


def _insert_with_legacy_sql(path: Path, *, approval_id: str) -> None:
  with sqlite3.connect(path) as conn:
    conn.execute(
      """
      INSERT INTO approval_requests (
        approval_id, tool_call_id, parent_approval_id, approval_chain_id,
        request_id, session_id, run_id, user_id, profile, channel, tool_name,
        tool_class, tool_args_redacted, args_hash, reason,
        blast_radius_summary, state, requested_at, policy_id, policy_version,
        policy_bundle_hash
      ) VALUES (?, ?, NULL, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
      """,
      (
        approval_id,
        f"tool-{approval_id}",
        f"chain-{approval_id}",
        f"request-{approval_id}",
        "legacy-user",
        "chat",
        "web",
        "legacy_tool",
        "external_write",
        "{}",
        "legacy-hash",
        "legacy row",
        "pending_user",
        "2026-07-14T12:00:00+00:00",
        "legacy-policy",
        "1",
        "legacy-bundle",
      ),
    )


def _create_legacy_approval_table(path: Path) -> None:
  with sqlite3.connect(path) as conn:
    conn.execute(
      """
      CREATE TABLE approval_requests (
        approval_id TEXT PRIMARY KEY,
        tool_call_id TEXT NOT NULL,
        parent_approval_id TEXT,
        approval_chain_id TEXT NOT NULL,
        delegation_id TEXT,
        request_id TEXT NOT NULL,
        session_id TEXT,
        run_id TEXT,
        user_id TEXT NOT NULL,
        profile TEXT NOT NULL,
        channel TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        tool_class TEXT NOT NULL,
        tool_args_redacted TEXT NOT NULL,
        args_hash TEXT NOT NULL,
        reason TEXT,
        blast_radius_summary TEXT NOT NULL,
        state TEXT NOT NULL,
        state_version INTEGER NOT NULL DEFAULT 0,
        requested_at TEXT NOT NULL,
        decided_at TEXT,
        expires_at TEXT,
        decider_id TEXT,
        decider_role TEXT,
        decision TEXT,
        decision_reason TEXT,
        required_decider_count INTEGER NOT NULL DEFAULT 1,
        eligible_decider_count INTEGER NOT NULL DEFAULT 1,
        votes_received_count INTEGER NOT NULL DEFAULT 0,
        args_predicate TEXT,
        chain_trust_window_seconds INTEGER,
        route_target TEXT,
        route_target_type TEXT,
        external_callback_id TEXT,
        policy_id TEXT NOT NULL,
        policy_version TEXT NOT NULL,
        policy_bundle_hash TEXT NOT NULL,
        persistent_grant_scope TEXT,
        tenant_id TEXT,
        model_id TEXT,
        model_version TEXT,
        system_prompt_hash TEXT,
        tool_schema_version TEXT,
        mcp_server_version TEXT,
        skill TEXT,
        notification_policy TEXT NOT NULL DEFAULT 'auto'
      )
      """
    )
