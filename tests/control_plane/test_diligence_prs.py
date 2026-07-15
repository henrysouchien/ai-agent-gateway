from __future__ import annotations

import asyncio
from dataclasses import replace
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import uuid

from fastapi.testclient import TestClient
import pytest

from agent_gateway.control_plane import batches as batches_module
from agent_gateway.control_plane import diligence_prs as diligence_prs_module
from research.reviewed_change_binding import (
  BindingKind,
  CanonicalProjection,
  DILIGENCE_THESIS_STORE_BASE_DOMAIN,
  PROPOSAL_EXECUTABLE_DOMAIN,
  ReviewAuthority,
  ReviewReference,
  ReviewedChangeBinding,
  ReviewedChangeLeaf,
  build_diligence_business_model_accept_thesis_store_base_v1,
  build_diligence_business_model_accept_transition_v1,
  project_diligence_business_model_accept_post_thesis_v1,
)


def _planned_merge_handler(
  captured: dict[str, Any],
  *,
  execution_result: dict[str, Any] | None = None,
  execution_error: dict[str, Any] | None = None,
):
  test_nonce = uuid.uuid4().hex

  async def handler(_tool_input: dict[str, Any]):
    raise AssertionError("owner route must not call the untrusted direct handler")

  async def plan_owner_merge(
    tool_input: dict[str, Any],
    **trusted_inputs: Any,
  ):
    captured["tool_input"] = tool_input
    captured["trusted_inputs"] = dict(trusted_inputs)
    proposal_ids = list(tool_input["expected_proposal_ids"])
    pre_thesis = {
      "thesis_id": f"thesis-{test_nonce}",
      "user_id": "tui-user",
      "ticker": tool_input["expected_ticker"],
      "label": "default",
      "version": 1,
      "created_at": "2026-01-01T00:00:00Z",
      "updated_at": "2026-01-01T00:00:00Z",
      "markdown_path": None,
      "business_model_ref": None,
      "decisions_log": [],
    }
    transition = build_diligence_business_model_accept_transition_v1(
      pre_thesis,
      pr_id=tool_input["pr_id"],
      research_file_id=tool_input["expected_research_file_id"],
      ticker=tool_input["expected_ticker"],
      workspace_business_model={
        "revision": f"revision-{test_nonce}",
        "payload_hash": "sha256:" + "1" * 64,
        "file_hash": "sha256:" + "2" * 64,
      },
      decision_entry_id=str(uuid.uuid5(uuid.NAMESPACE_URL, test_nonce)),
      canonical_business_model_path=f"/tmp/{test_nonce}.md",
    )
    post_thesis = project_diligence_business_model_accept_post_thesis_v1(
      pre_thesis,
      transition,
    )
    leaves = tuple(
      ReviewedChangeLeaf.create(
        binding_kind=BindingKind.DILIGENCE_PR_MERGE,
        ordinal=ordinal,
        reviewed_executable=CanonicalProjection.from_value(
          PROPOSAL_EXECUTABLE_DOMAIN,
          {
            "proposal_id": proposal_id,
            "research_file_id": tool_input["expected_research_file_id"],
            "thesis_id": pre_thesis["thesis_id"],
          },
        ),
        execution_constraints={},
        store_base=build_diligence_business_model_accept_thesis_store_base_v1(
          post_thesis,
          research_file_id=tool_input["expected_research_file_id"],
        ),
        store_base_domain=DILIGENCE_THESIS_STORE_BASE_DOMAIN,
        planned_result={},
      )
      for ordinal, proposal_id in enumerate(proposal_ids)
    )
    reference = ReviewReference.create(
      authority=ReviewAuthority.ADVISORY,
      review_domain="platform.diligence_pr.review_snapshot.v1",
      snapshot_digest="sha256:" + "a" * 64,
      review_binding_id=f"dra_{test_nonce}",
      bundle_manifest_digest="sha256:" + "b" * 64,
    )
    review_identity = {
      "pr_id": tool_input["pr_id"],
      "owner_user_id": "tui-user",
      "ticker": tool_input["expected_ticker"],
      "workspace_id": tool_input["expected_workspace_id"],
      "review_bundle_ref": f"diligence_review_bundles/{test_nonce}",
      "review_bundle_manifest_digest": reference.bundle_manifest_digest,
      "review_snapshot_digest": reference.snapshot_digest,
      "review_aggregate_id": reference.review_binding_id,
      "proposal_leaf_digests": [
        leaf.reviewed_executable.digest for leaf in leaves
      ],
      "frozen_base_hashes": {"test_nonce": test_nonce},
      "frozen_workspace_hash": "sha256:" + "c" * 64,
      "frozen_workspace_leaf_digests": ["sha256:" + "d" * 64],
      "workspace_descriptor_digest": "sha256:" + "e" * 64,
      "frozen_at": 1000.0,
    }
    binding = ReviewedChangeBinding.create(
      binding_kind=BindingKind.DILIGENCE_PR_MERGE,
      base_vector={
        "ordered_store_base_digests": [
          leaf.store_base.digest for leaf in leaves
        ],
        "diligence_review": review_identity,
        "promotion_thesis_transition": transition,
      },
      operation_semantics={
        "mode": "diligence_pr_merge",
        "pr_id": tool_input["pr_id"],
        "owner_user_id": "tui-user",
        "ordered_proposal_ids": proposal_ids,
        "apply_effects_via_outbox": True,
      },
      process_outbox=tool_input["process_outbox"],
      normalized_tool_semantics=tool_input,
      leaves=leaves,
      review_reference=reference,
    )
    prepared = SimpleNamespace(binding=binding)
    return binding, prepared

  async def execute_owner_merge(
    supplied_prepared: Any,
    *,
    authorized_identity: Any,
    approval_id: str,
    approval_chain_id: str,
  ):
    captured["prepared"] = supplied_prepared
    captured["authorized_identity"] = authorized_identity
    captured["approval_id"] = approval_id
    captured["approval_chain_id"] = approval_chain_id
    return execution_result, execution_error

  setattr(handler, "plan_owner_merge", plan_owner_merge)
  setattr(handler, "execute_owner_merge", execute_owner_merge)
  return handler


def _headers(session_payload: dict[str, Any]) -> dict[str, str]:
  return {"Authorization": f"Bearer {session_payload['session_token']}"}


def _merge_payload(**overrides: Any) -> dict[str, Any]:
  payload = {
    "confirm_merge": True,
    "expected_ticker": "msft",
    "expected_workspace_id": "batch_7_MSFT",
    "expected_proposal_ids": ["proposal-1", "proposal-2"],
    "expected_research_file_id": 42,
    "expected_handoff_id": 9,
  }
  payload.update(overrides)
  return payload


def test_owner_merge_planner_never_falls_back_to_write_registry() -> None:
  from agent.shared.tool_handlers.apply_proposals import (
    make_merge_diligence_pr_handler,
  )
  from agent_gateway.tool_dispatcher_helpers import PlannedWritePlanningRejected

  def fail_write_factory(*_args: Any, **_kwargs: Any):
    raise AssertionError("planner opened the write-mode registry")

  handler = make_merge_diligence_pr_handler(
    session=SimpleNamespace(kind="control", role="owner", user_id="tui-user"),
    batch_registry_factory=fail_write_factory,
  )
  planner = handler.plan_owner_merge

  with pytest.raises(PlannedWritePlanningRejected) as exc_info:
    asyncio.run(planner({"pr_id": "dpr-1", **_merge_payload()}))

  result, error = exc_info.value.tool_result()
  assert result is None
  assert error is not None
  assert error["code"] == "diligence_pr_read_only_planning_unavailable"


def _approval_payload(**overrides: Any) -> dict[str, Any]:
  payload = {
    "approval_request_id": "approval-request-1",
    "expected_review_snapshot_digest": "sha256:" + "a" * 64,
    "expected_review_aggregate_id": "dra_" + "b" * 64,
    "expected_review_content_version": 3,
    "expected_state_version": 7,
  }
  payload.update(overrides)
  return payload


def test_owner_merge_planned_business_model_path_never_seeds_workspace(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory
  from api.agent.shared.tool_handlers import apply_proposals

  workspace = tmp_path / "not-yet-created" / "workspace"

  def fail_seed(*_args: Any, **_kwargs: Any):
    raise AssertionError("owner planning attempted to seed workspace assets")

  monkeypatch.setattr(memory, "get_workspace_path", lambda _user_id: workspace)
  monkeypatch.setattr(memory, "get_workspace_dir", fail_seed)
  monkeypatch.setattr(memory, "_seed_workspace_assets", fail_seed)

  planned = apply_proposals._planned_business_model_artifact_path(
    user_id="tui-user",
    ticker="msft",
  )

  assert planned == workspace / "theses" / "MSFT_business_model.md"
  assert not workspace.exists()


def test_owner_merge_planning_registry_is_read_only_and_never_seeds_workspace(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory
  from agent.batch.registry import BatchRegistry

  workspace = tmp_path / "user" / "workspace"
  db_path = workspace / "batch_registry.db"
  writer = BatchRegistry(db_path)
  try:
    expected = writer.get_diligence_pr_protocol_state()
  finally:
    writer.close()
  before = {
    str(path.relative_to(tmp_path)): path.read_bytes()
    for path in sorted(tmp_path.rglob("*"))
    if path.is_file()
  }

  def fail_seed(*_args: Any, **_kwargs: Any):
    raise AssertionError("owner planning attempted to seed workspace assets")

  monkeypatch.setattr(memory, "get_workspace_path", lambda _user_id: workspace)
  monkeypatch.setattr(memory, "get_workspace_dir", fail_seed)
  monkeypatch.setattr(memory, "_seed_workspace_assets", fail_seed)

  reader = batches_module._read_only_registry_for_user("tui-user")
  try:
    assert reader.get_diligence_pr_protocol_state() == expected
  finally:
    reader.close()

  after = {
    str(path.relative_to(tmp_path)): path.read_bytes()
    for path in sorted(tmp_path.rglob("*"))
    if path.is_file()
  }
  assert after == before


def _refreeze_payload(**overrides: Any) -> dict[str, Any]:
  payload = {
    "expected_predecessor_state": "approved",
    "expected_predecessor_state_version": 3,
    "refreeze_request_id": "refreeze-request-1",
  }
  payload.update(overrides)
  return payload


def _model_workspace_descriptor(workspace: Path) -> dict[str, Any]:
  root = workspace / "model_workspaces" / "batch_1" / "MSFT"
  return {
    "ticker": "MSFT",
    "root_dir": str(root),
    "overrides_dir": str(root / "TickerOverrides"),
    "mbc_path": str(root / "model_build_context.json"),
    "current_model_ref_path": str(root / "current_model_ref.json"),
    "workbook_path": str(root / "MSFT_preview_model.xlsx"),
    "business_model_path": str(root / "business_model.md"),
    "workspace_id": "batch_1_MSFT",
    "batch_id": 1,
    "pipeline_id": None,
  }


def test_approve_diligence_pr_route_delegates_exact_owner_request(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}

  async def approve(session: Any, approval_input: dict[str, Any]):
    captured["session"] = session
    captured["approval_input"] = approval_input
    return {
      "status": "approved",
      "idempotent": False,
      "pr": {"pr_id": approval_input["pr_id"], "state": "approved"},
    }, None

  monkeypatch.setattr(
    diligence_prs_module,
    "_approve_diligence_pr_for_session",
    approve,
  )
  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/approve",
    headers=_headers(test_control_session),
    json=_approval_payload(),
  )

  assert response.status_code == 200, response.text
  assert response.json()["pr"]["state"] == "approved"
  assert captured["session"].user_id == "tui-user"
  assert captured["approval_input"] == {
    "pr_id": "dpr_7_MSFT_abc",
    **_approval_payload(),
  }


def test_refreeze_diligence_pr_route_delegates_exact_owner_request(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}

  async def refreeze(session: Any, refreeze_input: dict[str, Any]):
    captured["session"] = session
    captured["refreeze_input"] = refreeze_input
    return {
      "status": "forked",
      "idempotent": False,
      "pr": {"pr_id": "successor-1", "state": "open"},
    }, None

  monkeypatch.setattr(
    diligence_prs_module,
    "_refreeze_diligence_pr_for_session",
    refreeze,
  )
  response = client.post(
    "/api/control/diligence-prs/dpr_legacy/refreeze",
    headers=_headers(test_control_session),
    json=_refreeze_payload(),
  )
  assert response.status_code == 200, response.text
  assert response.json()["pr"]["state"] == "open"
  assert captured["session"].user_id == "tui-user"
  assert captured["refreeze_input"] == {
    "pr_id": "dpr_legacy",
    **_refreeze_payload(),
  }


@pytest.mark.parametrize(
  ("code", "status_code"),
  (
    ("diligence_pr_protocol_incompatible", 503),
    ("diligence_pr_not_found", 404),
    ("diligence_pr_fork_conflict", 409),
    ("diligence_pr_promotion_in_progress", 409),
    ("diligence_pr_refreeze_required", 409),
    ("diligence_pr_refreeze_unavailable", 500),
  ),
)
def test_refreeze_diligence_pr_route_pins_error_taxonomy(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
  code: str,
  status_code: int,
) -> None:
  async def refreeze(_session: Any, refreeze_input: dict[str, Any]):
    return None, {
      "code": code,
      "message": "synthetic refreeze result",
      "pr_id": refreeze_input["pr_id"],
    }

  monkeypatch.setattr(
    diligence_prs_module,
    "_refreeze_diligence_pr_for_session",
    refreeze,
  )
  response = client.post(
    "/api/control/diligence-prs/dpr_legacy/refreeze",
    headers=_headers(test_control_session),
    json=_refreeze_payload(),
  )
  assert response.status_code == status_code, response.text
  assert response.json()["code"] == code


def test_refreeze_route_rejects_nonexact_or_ill_typed_input_before_execution(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  calls = 0

  async def refreeze(_session: Any, _refreeze_input: dict[str, Any]):
    nonlocal calls
    calls += 1
    raise AssertionError("invalid requests must not reach refreeze execution")

  monkeypatch.setattr(
    diligence_prs_module,
    "_refreeze_diligence_pr_for_session",
    refreeze,
  )
  url = "/api/control/diligence-prs/dpr_legacy/refreeze"
  headers = _headers(test_control_session)
  responses = (
    client.post(
      url,
      headers=headers,
      json={key: value for key, value in _refreeze_payload().items() if key != "refreeze_request_id"},
    ),
    client.post(
      url,
      headers=headers,
      json={**_refreeze_payload(), "refrozen_by_user_id": "forged"},
    ),
    client.post(
      url,
      headers=headers,
      json=_refreeze_payload(expected_predecessor_state_version=True),
    ),
    client.post(
      url,
      headers=headers,
      json=_refreeze_payload(expected_predecessor_state="reviewing"),
    ),
  )
  assert [response.status_code for response in responses] == [422, 422, 422, 422]
  assert calls == 0


def test_refreeze_route_requires_owner_role(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  session = control_plane_app.state.auth.session_store.get_session(
    test_control_session["session_id"]
  )
  assert session is not None
  session.role = "invite"
  invite_payload = dict(test_control_session)
  invite_payload["session_token"] = control_plane_app.state.auth.issue_token(session)

  async def refreeze(_session: Any, _refreeze_input: dict[str, Any]):
    raise AssertionError("invite sessions must not reach refreeze execution")

  monkeypatch.setattr(
    diligence_prs_module,
    "_refreeze_diligence_pr_for_session",
    refreeze,
  )
  response = client.post(
    "/api/control/diligence-prs/dpr_legacy/refreeze",
    headers=_headers(invite_payload),
    json=_refreeze_payload(),
  )
  assert response.status_code == 403, response.text
  assert response.json()["detail"] == (
    "Owner control session required to refreeze a diligence PR"
  )


def test_refreeze_response_loss_replay_skips_live_capture(monkeypatch) -> None:
  observed: dict[str, Any] = {"closed": False}

  class FakeRegistry:
    def get_diligence_pr_refreeze_replay_v2(self, pr_id: str, **kwargs: Any):
      observed["replay"] = {"pr_id": pr_id, **kwargs}
      return {
        "status": "forked",
        "idempotent": True,
        "pr": {"pr_id": "successor-1", "state": "open"},
      }

    def get_diligence_pr(self, _pr_id: str):
      raise AssertionError("replay must not re-read mutable predecessor content")

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(
    diligence_prs_module,
    "_registry_for_user",
    lambda _user_id: FakeRegistry(),
  )
  session = type("Session", (), {"user_id": "owner-1"})()
  result, error = asyncio.run(
    diligence_prs_module._refreeze_diligence_pr_for_session(
      session,
      {"pr_id": "legacy-1", **_refreeze_payload()},
    )
  )
  assert error is None
  assert result is not None and result["idempotent"] is True
  assert observed["replay"] == {
    "pr_id": "legacy-1",
    "refreeze_request_id": "refreeze-request-1",
    "refrozen_by_user_id": "owner-1",
  }
  assert observed["closed"] is True


def test_refreeze_execution_captures_legacy_content_and_uses_session_actor(
  monkeypatch,
  tmp_path: Path,
) -> None:
  fms_state = importlib.import_module("fms.core.state")
  memory = importlib.import_module("memory")
  dpr_module = importlib.import_module("agent.batch.diligence_prs")
  observed: dict[str, Any] = {"closed": False}
  active_provider_registry = object()

  proposal = {
    "proposal_id": "proposal-1",
    "research_file_id": 42,
    "thesis_id": "thesis-1",
    "base_thesis_id": "thesis-base",
    "base_version": 7,
    "canonical_ops": {"operations": []},
    "source_id_remap": {},
    "gate_metadata": {},
    "op_hash": "sha256:ops",
    "result_hash": "sha256:result",
    "created_at": 1.0,
    "expires_at": 1000.0,
    "applied_at": None,
  }

  class FakeRepo:
    def get_patch_proposal(self, proposal_id: str):
      observed.setdefault("proposal_reads", []).append(proposal_id)
      return dict(proposal) if proposal_id == "proposal-1" else None

    def get_file_by_ticker_label(self, _ticker: str, _label: str = ""):
      return None

  class FakeRegistry:
    provider_registry = active_provider_registry

    def get_diligence_pr_refreeze_replay_v2(self, *_args: Any, **_kwargs: Any):
      return None

    def get_diligence_pr(self, pr_id: str):
      return {
        "pr_id": pr_id,
        "state": "approved",
        "state_version": 3,
        "ticker": "MSFT",
        "proposal_ids": ["proposal-1"],
        "workspace_ref": "model_workspaces/batch_1/MSFT/workspace.json",
        "workspace_id": "batch_1_MSFT",
        "model_workspace": _model_workspace_descriptor(tmp_path),
      }

    def refreeze_legacy_diligence_pr_v2(self, pr_id: str, **kwargs: Any):
      observed["refreeze"] = {"pr_id": pr_id, **kwargs}
      return {
        "status": "forked",
        "idempotent": False,
        "pr": {"pr_id": "successor-1", "state": "open"},
      }

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(
    diligence_prs_module,
    "_registry_for_user",
    lambda _user_id: FakeRegistry(),
  )
  monkeypatch.setattr(fms_state, "resolve_repo", lambda user_id: FakeRepo())
  monkeypatch.setattr(memory, "get_workspace_dir", lambda _user_id: tmp_path)

  def base_hashes(*, repo, ticker, provider_registry=None, **_kwargs):
    _ = repo
    observed["base_capture"] = {
      "ticker": ticker,
      "provider_registry": provider_registry,
    }
    return {"schema_version": 1, "ticker": ticker}

  monkeypatch.setattr(dpr_module, "base_hashes_for_ticker", base_hashes)
  session = type("Session", (), {"user_id": "owner-1"})()
  result, error = asyncio.run(
    diligence_prs_module._refreeze_diligence_pr_for_session(
      session,
      {"pr_id": "legacy-1", **_refreeze_payload()},
    )
  )
  assert error is None
  assert result is not None and result["status"] == "forked"
  assert observed["proposal_reads"] == ["proposal-1"]
  assert observed["refreeze"]["expected_predecessor_state_version"] == 3
  assert observed["refreeze"]["refreeze_request_id"] == "refreeze-request-1"
  assert observed["refreeze"]["refrozen_by_user_id"] == "owner-1"
  assert observed["refreeze"]["proposal_ids"] == ["proposal-1"]
  assert len(observed["refreeze"]["proposal_leaf_digests"]) == 1
  assert len(
    observed["refreeze"]["workspace_payload"]["workspace_content_hashes"]
  ) == 6
  assert observed["base_capture"] == {
    "ticker": "MSFT",
    "provider_registry": active_provider_registry,
  }
  assert observed["closed"] is True


@pytest.mark.parametrize(
  "persisted_ids",
  [
    [7],
    [" proposal-1 "],
    ["proposal-1", "proposal-1"],
  ],
)
def test_refreeze_rejects_noncanonical_persisted_proposal_identity_before_capture(
  monkeypatch,
  persisted_ids: list[object],
) -> None:
  fms_state = importlib.import_module("fms.core.state")
  observed: dict[str, Any] = {"closed": False, "proposal_reads": 0}

  class FakeRepo:
    def get_patch_proposal(self, _proposal_id: str):
      observed["proposal_reads"] += 1
      raise AssertionError("invalid persisted identity must fail before capture")

  class FakeRegistry:
    def get_diligence_pr_refreeze_replay_v2(self, *_args: Any, **_kwargs: Any):
      return None

    def get_diligence_pr(self, pr_id: str):
      return {
        "pr_id": pr_id,
        "state": "approved",
        "state_version": 3,
        "ticker": "MSFT",
        "proposal_ids": persisted_ids,
      }

    def refreeze_legacy_diligence_pr_v2(self, *_args: Any, **_kwargs: Any):
      raise AssertionError("invalid persisted identity must not reach the writer")

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(
    diligence_prs_module,
    "_registry_for_user",
    lambda _user_id: FakeRegistry(),
  )
  monkeypatch.setattr(fms_state, "resolve_repo", lambda _user_id: FakeRepo())
  session = type("Session", (), {"user_id": "owner-1"})()
  result, error = asyncio.run(
    diligence_prs_module._refreeze_diligence_pr_for_session(
      session,
      {"pr_id": "legacy-1", **_refreeze_payload()},
    )
  )

  assert result is None
  assert error == {
    "code": "diligence_pr_fork_conflict",
    "message": "legacy predecessor proposal identity is not canonical",
    "pr_id": "legacy-1",
  }
  assert observed == {"closed": True, "proposal_reads": 0}


def test_refreeze_missing_live_proposal_fails_closed_before_registry_mutation(
  monkeypatch,
) -> None:
  fms_state = importlib.import_module("fms.core.state")
  observed: dict[str, Any] = {"closed": False}

  class FakeRepo:
    def get_patch_proposal(self, proposal_id: str):
      observed.setdefault("proposal_reads", []).append(proposal_id)
      return None

  class FakeRegistry:
    def get_diligence_pr_refreeze_replay_v2(self, *_args: Any, **_kwargs: Any):
      return None

    def get_diligence_pr(self, pr_id: str):
      return {
        "pr_id": pr_id,
        "state": "approved",
        "state_version": 3,
        "ticker": "MSFT",
        "proposal_ids": ["proposal-missing"],
        "workspace_id": "batch_1_MSFT",
        "model_workspace": {"ticker": "MSFT"},
      }

    def refreeze_legacy_diligence_pr_v2(self, *_args: Any, **_kwargs: Any):
      raise AssertionError("missing live content must not reach the registry writer")

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(
    diligence_prs_module,
    "_registry_for_user",
    lambda _user_id: FakeRegistry(),
  )
  monkeypatch.setattr(fms_state, "resolve_repo", lambda _user_id: FakeRepo())
  session = type("Session", (), {"user_id": "owner-1"})()
  result, error = asyncio.run(
    diligence_prs_module._refreeze_diligence_pr_for_session(
      session,
      {"pr_id": "legacy-1", **_refreeze_payload()},
    )
  )

  assert result is None
  assert error == {
    "code": "diligence_pr_fork_conflict",
    "message": "legacy proposal is unavailable: proposal-missing",
    "pr_id": "legacy-1",
  }
  assert observed["proposal_reads"] == ["proposal-missing"]
  assert observed["closed"] is True


def test_refreeze_missing_workspace_descriptor_fails_closed_before_registry_mutation(
  monkeypatch,
  tmp_path: Path,
) -> None:
  fms_state = importlib.import_module("fms.core.state")
  memory = importlib.import_module("memory")
  observed: dict[str, Any] = {"closed": False}

  proposal = {
    "proposal_id": "proposal-1",
    "research_file_id": 42,
    "thesis_id": "thesis-1",
    "base_thesis_id": "thesis-base",
    "base_version": 7,
    "canonical_ops": {"operations": []},
    "source_id_remap": {},
    "gate_metadata": {},
    "op_hash": "sha256:ops",
    "result_hash": "sha256:result",
    "created_at": 1.0,
    "expires_at": 1000.0,
    "applied_at": None,
  }

  class FakeRepo:
    def get_patch_proposal(self, proposal_id: str):
      observed.setdefault("proposal_reads", []).append(proposal_id)
      return dict(proposal)

  class FakeRegistry:
    def get_diligence_pr_refreeze_replay_v2(self, *_args: Any, **_kwargs: Any):
      return None

    def get_diligence_pr(self, pr_id: str):
      return {
        "pr_id": pr_id,
        "state": "approved",
        "state_version": 3,
        "ticker": "MSFT",
        "proposal_ids": ["proposal-1"],
        "workspace_ref": "model_workspaces/batch_1/MSFT/workspace.json",
        "workspace_id": "batch_1_MSFT",
        "model_workspace": {"ticker": "MSFT"},
      }

    def refreeze_legacy_diligence_pr_v2(self, *_args: Any, **_kwargs: Any):
      raise AssertionError("missing workspace content must not reach the registry writer")

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(
    diligence_prs_module,
    "_registry_for_user",
    lambda _user_id: FakeRegistry(),
  )
  monkeypatch.setattr(fms_state, "resolve_repo", lambda _user_id: FakeRepo())
  monkeypatch.setattr(memory, "get_workspace_dir", lambda _user_id: tmp_path)
  session = type("Session", (), {"user_id": "owner-1"})()
  result, error = asyncio.run(
    diligence_prs_module._refreeze_diligence_pr_for_session(
      session,
      {"pr_id": "legacy-1", **_refreeze_payload()},
    )
  )

  assert result is None
  assert error is not None
  assert error["code"] == "diligence_pr_fork_conflict"
  assert error["pr_id"] == "legacy-1"
  assert "model_workspace.root_dir is required" in error["message"]
  assert observed["proposal_reads"] == ["proposal-1"]
  assert observed["closed"] is True


@pytest.mark.parametrize(
  ("code", "status_code"),
  (
    ("diligence_pr_protocol_incompatible", 503),
    ("diligence_pr_not_found", 404),
    ("diligence_pr_approval_state_conflict", 409),
    ("diligence_pr_approval_version_conflict", 409),
    ("diligence_pr_review_identity_mismatch", 409),
    ("diligence_pr_review_stale", 409),
    ("diligence_pr_review_bundle_invalid", 409),
    ("diligence_pr_approval_unavailable", 500),
  ),
)
def test_approve_diligence_pr_route_pins_error_taxonomy(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
  code: str,
  status_code: int,
) -> None:
  async def approve(_session: Any, approval_input: dict[str, Any]):
    return None, {
      "code": code,
      "message": "synthetic approval result",
      "pr_id": approval_input["pr_id"],
    }

  monkeypatch.setattr(
    diligence_prs_module,
    "_approve_diligence_pr_for_session",
    approve,
  )
  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/approve",
    headers=_headers(test_control_session),
    json=_approval_payload(),
  )

  assert response.status_code == status_code, response.text
  assert response.json()["code"] == code


def test_approve_diligence_pr_route_rejects_nonexact_or_ill_typed_input(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  calls = 0

  async def approve(_session: Any, _approval_input: dict[str, Any]):
    nonlocal calls
    calls += 1
    raise AssertionError("invalid requests must not reach approval execution")

  monkeypatch.setattr(
    diligence_prs_module,
    "_approve_diligence_pr_for_session",
    approve,
  )
  url = "/api/control/diligence-prs/dpr_7_MSFT_abc/approve"
  headers = _headers(test_control_session)

  missing = client.post(
    url,
    headers=headers,
    json={key: value for key, value in _approval_payload().items() if key != "approval_request_id"},
  )
  unexpected = client.post(
    url,
    headers=headers,
    json={**_approval_payload(), "approved_by_user_id": "forged"},
  )
  bool_version = client.post(
    url,
    headers=headers,
    json=_approval_payload(expected_state_version=True),
  )
  blank_identity = client.post(
    url,
    headers=headers,
    json=_approval_payload(expected_review_snapshot_digest=" "),
  )

  assert [response.status_code for response in (
    missing,
    unexpected,
    bool_version,
    blank_identity,
  )] == [422, 422, 422, 422]
  assert calls == 0


def test_approve_diligence_pr_route_requires_owner_role(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  session = control_plane_app.state.auth.session_store.get_session(
    test_control_session["session_id"]
  )
  assert session is not None
  session.role = "invite"
  invite_payload = dict(test_control_session)
  invite_payload["session_token"] = control_plane_app.state.auth.issue_token(session)

  async def approve(_session: Any, _approval_input: dict[str, Any]):
    raise AssertionError("invite sessions must not reach approval execution")

  monkeypatch.setattr(
    diligence_prs_module,
    "_approve_diligence_pr_for_session",
    approve,
  )
  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/approve",
    headers=_headers(invite_payload),
    json=_approval_payload(),
  )

  assert response.status_code == 403, response.text
  assert response.json()["detail"] == (
    "Owner control session required to approve a diligence PR"
  )


def test_reviewing_approval_threads_exact_provider_registry_to_live_check(
  monkeypatch,
) -> None:
  bundle_module = importlib.import_module(
    "agent.batch.diligence_pr_review_bundle"
  )
  capture_module = importlib.import_module(
    "agent.batch.diligence_pr_review_capture"
  )
  fms_state = importlib.import_module("fms.core.state")
  memory_module = importlib.import_module("memory")
  verified_bundle = object()
  repo = object()
  active_provider_registry = object()
  observed: dict[str, Any] = {}

  class FakeRegistry:
    provider_registry = active_provider_registry

    def get_diligence_pr(self, pr_id: str):
      return {
        "pr_id": pr_id,
        "state": "reviewing",
        "review_bundle_ref": "diligence_review_bundles/" + "a" * 64,
        "review_bundle_manifest_digest": "sha256:" + "a" * 64,
      }

    def approve_diligence_pr_v2(self, pr_id: str, **kwargs):
      observed["approval"] = {"pr_id": pr_id, **kwargs}
      return {"status": "approved", "idempotent": False, "pr": {"pr_id": pr_id}}

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(
    diligence_prs_module,
    "_registry_for_user",
    lambda _user_id: FakeRegistry(),
  )
  monkeypatch.setattr(memory_module, "get_workspace_dir", lambda _user_id: "/workspace")
  monkeypatch.setattr(fms_state, "resolve_repo", lambda _user_id: repo)
  monkeypatch.setattr(
    bundle_module,
    "load_verified_review_bundle",
    lambda *_args, **_kwargs: verified_bundle,
  )

  def verify_live(**kwargs):
    observed["live_check"] = kwargs

  monkeypatch.setattr(
    capture_module,
    "verify_live_review_bundle_equivalence",
    verify_live,
  )
  session = type("Session", (), {"user_id": "owner-user"})()
  result, error = asyncio.run(
    diligence_prs_module._approve_diligence_pr_for_session(
      session,
      {"pr_id": "dpr-reviewing", **_approval_payload()},
    )
  )

  assert error is None
  assert result is not None and result["status"] == "approved"
  assert observed["live_check"] == {
    "repo": repo,
    "user_workspace": "/workspace",
    "verified_bundle": verified_bundle,
    "provider_registry": active_provider_registry,
  }
  assert observed["approval"]["verified_bundle"] is verified_bundle
  assert observed["closed"] is True


def test_approved_response_loss_replay_skips_mutable_live_recheck(
  monkeypatch,
) -> None:
  bundle_module = importlib.import_module(
    "agent.batch.diligence_pr_review_bundle"
  )
  capture_module = importlib.import_module(
    "agent.batch.diligence_pr_review_capture"
  )
  memory_module = importlib.import_module("memory")
  verified_bundle = object()
  observed: dict[str, Any] = {}

  class FakeRegistry:
    def get_diligence_pr(self, pr_id: str):
      assert pr_id == "dpr-approved"
      return {
        "pr_id": pr_id,
        "state": "approved",
        "review_bundle_ref": "diligence_review_bundles/" + "a" * 64,
        "review_bundle_manifest_digest": "sha256:" + "a" * 64,
      }

    def approve_diligence_pr_v2(self, pr_id: str, **kwargs):
      observed["approval"] = {"pr_id": pr_id, **kwargs}
      return {"status": "approved", "idempotent": True, "pr": {"pr_id": pr_id}}

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(
    diligence_prs_module,
    "_registry_for_user",
    lambda user_id: FakeRegistry(),
  )
  monkeypatch.setattr(memory_module, "get_workspace_dir", lambda user_id: "/workspace")
  monkeypatch.setattr(
    bundle_module,
    "load_verified_review_bundle",
    lambda *args, **kwargs: verified_bundle,
  )

  def reject_live_recheck(**_kwargs):
    raise AssertionError("approved replay must not consult mutable live sources")

  monkeypatch.setattr(
    capture_module,
    "verify_live_review_bundle_equivalence",
    reject_live_recheck,
  )
  session = type("Session", (), {"user_id": "owner-user"})()
  approval_input = {
    "pr_id": "dpr-approved",
    **_approval_payload(),
  }

  result, error = asyncio.run(
    diligence_prs_module._approve_diligence_pr_for_session(
      session,
      approval_input,
    )
  )

  assert error is None
  assert result == {
    "status": "approved",
    "idempotent": True,
    "pr": {"pr_id": "dpr-approved"},
  }
  assert observed["approval"]["approved_by_user_id"] == "owner-user"
  assert observed["approval"]["verified_bundle"] is verified_bundle
  assert observed["closed"] is True


def test_nonreviewing_approval_conflicts_before_bundle_or_live_access(
  monkeypatch,
) -> None:
  observed: dict[str, Any] = {}

  class FakeRegistry:
    def get_diligence_pr(self, pr_id: str):
      return {"pr_id": pr_id, "state": "changes_requested"}

    def approve_diligence_pr_v2(self, *_args, **_kwargs):
      raise AssertionError("non-reviewing rows must not reach approval CAS")

    def close(self):
      observed["closed"] = True

  monkeypatch.setattr(
    diligence_prs_module,
    "_registry_for_user",
    lambda _user_id: FakeRegistry(),
  )
  session = type("Session", (), {"user_id": "owner-user"})()
  result, error = asyncio.run(
    diligence_prs_module._approve_diligence_pr_for_session(
      session,
      {"pr_id": "dpr-not-reviewing", **_approval_payload()},
    )
  )

  assert result is None
  assert error == {
    "code": "diligence_pr_approval_state_conflict",
    "message": "diligence PR is not awaiting owner approval",
    "pr_id": "dpr-not-reviewing",
  }
  assert observed["closed"] is True


def test_merge_diligence_pr_route_delegates_to_s4m_handler(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}

  def handler_factory(session: Any):
    captured["session"] = session
    return _planned_merge_handler(
      captured,
      execution_result={
        "tool": "merge_diligence_pr",
        "mutation_mode": "apply",
        "status": "success",
        "pr_id": "dpr_7_MSFT_abc",
        "pr": {"state": "merged"},
      },
    )

  monkeypatch.setattr(diligence_prs_module, "_merge_handler_for_session", handler_factory)

  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 200, response.text
  assert response.json()["pr"]["state"] == "merged"
  assert captured["session"].user_id == "tui-user"
  assert captured["tool_input"] == {
    "pr_id": "dpr_7_MSFT_abc",
    "confirm_merge": True,
    "expected_ticker": "MSFT",
    "expected_workspace_id": "batch_7_MSFT",
    "expected_proposal_ids": ["proposal-1", "proposal-2"],
    "expected_research_file_id": 42,
    "expected_handoff_id": 9,
    "process_outbox": True,
  }
  assert captured["trusted_inputs"] == {}
  assert "planned_acceptance_at" not in captured["tool_input"]
  assert "decision_entry_id" not in captured["tool_input"]
  assert captured["approval_id"] == captured["approval_chain_id"]
  assert captured["prepared"].binding is captured["authorized_identity"]
  approval = asyncio.run(
    control_plane_app.state.gateway_approval_store.get(captured["approval_id"])
  )
  assert approval is not None
  assert approval.state == "approved"
  assert approval.decision == "approved"
  assert approval.decider_id == "tui-user"
  assert approval.decider_role == "owner"
  assert approval.votes_received_count == 1
  assert approval.identity_source == "reviewed_change_binding"
  assert approval.parent_approval_id is None


def test_merge_diligence_pr_route_fails_closed_without_durable_approval_store(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}
  handler = _planned_merge_handler(
    captured,
    execution_result={"status": "success", "pr": {"state": "merged"}},
  )
  monkeypatch.setattr(
    diligence_prs_module,
    "_merge_handler_for_session",
    lambda _session: handler,
  )
  monkeypatch.setattr(
    control_plane_app.state,
    "gateway_approval_store",
    None,
  )

  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 503, response.text
  assert response.json()["code"] == "owner_approval_store_unavailable"
  assert "prepared" not in captured


def test_merge_diligence_pr_route_rejects_automatic_approval_before_execution(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}
  handler = _planned_merge_handler(
    captured,
    execution_result={"status": "success", "pr": {"state": "merged"}},
  )
  monkeypatch.setattr(
    diligence_prs_module,
    "_merge_handler_for_session",
    lambda _session: handler,
  )
  store = control_plane_app.state.gateway_approval_store
  record_vote = store.record_vote

  async def return_automatic_outcome(approval_id: str, vote: Any):
    approved = await record_vote(approval_id, vote)
    return replace(approved, state="auto_approved")

  monkeypatch.setattr(store, "record_vote", return_automatic_outcome)

  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 409, response.text
  assert response.json()["code"] == "owner_approval_identity_mismatch"
  assert "prepared" not in captured


def test_merge_diligence_pr_route_reuses_exact_manual_approval_and_vote(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}
  handler = _planned_merge_handler(
    captured,
    execution_result={"status": "success", "pr": {"state": "merged"}},
  )
  monkeypatch.setattr(
    diligence_prs_module,
    "_merge_handler_for_session",
    lambda _session: handler,
  )
  url = "/api/control/diligence-prs/dpr_7_MSFT_abc/merge"
  headers = _headers(test_control_session)

  first = client.post(url, headers=headers, json=_merge_payload())
  assert first.status_code == 200, first.text
  first_approval_id = captured["approval_id"]
  second = client.post(url, headers=headers, json=_merge_payload())
  assert second.status_code == 200, second.text
  assert captured["approval_id"] == first_approval_id

  approval = asyncio.run(
    control_plane_app.state.gateway_approval_store.get(first_approval_id)
  )
  assert approval is not None
  assert approval.state == "approved"
  assert approval.votes_received_count == 1


def test_merge_diligence_pr_route_rejects_tampered_durable_identity(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}
  handler = _planned_merge_handler(
    captured,
    execution_result={"status": "success", "pr": {"state": "merged"}},
  )
  monkeypatch.setattr(
    diligence_prs_module,
    "_merge_handler_for_session",
    lambda _session: handler,
  )
  url = "/api/control/diligence-prs/dpr_7_MSFT_abc/merge"
  headers = _headers(test_control_session)

  first = client.post(url, headers=headers, json=_merge_payload())
  assert first.status_code == 200, first.text
  approval_id = captured["approval_id"]
  store = control_plane_app.state.gateway_approval_store
  with store._connection() as conn:
    conn.execute(
      """
      UPDATE approval_requests
      SET reviewed_change_binding_digest = ?
      WHERE approval_id = ?
      """,
      ("sha256:" + "f" * 64, approval_id),
    )
  captured.pop("prepared")

  second = client.post(url, headers=headers, json=_merge_payload())

  assert second.status_code == 409, second.text
  assert second.json()["code"] == "owner_approval_identity_mismatch"
  assert "prepared" not in captured


def test_merge_planning_failure_creates_no_approval_row(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  store = control_plane_app.state.gateway_approval_store
  with store._connection() as conn:
    before = int(
      conn.execute("SELECT COUNT(*) FROM approval_requests").fetchone()[0]
    )

  async def handler(_tool_input: dict[str, Any]):
    raise AssertionError("the untrusted handler must not run")

  async def fail_planning(_tool_input: dict[str, Any], **_trusted_inputs: Any):
    raise RuntimeError("synthetic planning failure")

  async def fail_execution(*_args: Any, **_kwargs: Any):
    raise AssertionError("planning failure must not reach execution")

  setattr(handler, "plan_owner_merge", fail_planning)
  setattr(handler, "execute_owner_merge", fail_execution)
  monkeypatch.setattr(
    diligence_prs_module,
    "_merge_handler_for_session",
    lambda _session: handler,
  )

  with pytest.raises(RuntimeError, match="synthetic planning failure"):
    client.post(
      "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
      headers=_headers(test_control_session),
      json=_merge_payload(),
    )

  with store._connection() as conn:
    after = int(
      conn.execute("SELECT COUNT(*) FROM approval_requests").fetchone()[0]
    )
  assert after == before


def test_merge_diligence_pr_route_returns_conflict_for_blocked_merge(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}
  monkeypatch.setattr(
    diligence_prs_module,
    "_merge_handler_for_session",
    lambda _session: _planned_merge_handler(
      captured,
      execution_result={
        "tool": "merge_diligence_pr",
        "mutation_mode": "apply",
        "status": "blocked",
        "pr_id": "dpr_7_MSFT_abc",
        "blockers": [{"code": "base_hash_stale"}],
      },
    ),
  )

  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 409, response.text
  assert response.json()["blockers"] == [{"code": "base_hash_stale"}]


def test_merge_diligence_pr_route_maps_missing_pr_to_not_found(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}
  monkeypatch.setattr(
    diligence_prs_module,
    "_merge_handler_for_session",
    lambda _session: _planned_merge_handler(
      captured,
      execution_error={
        "code": "diligence_pr_not_found",
        "message": "diligence PR not found",
        "pr_id": "dpr_missing",
      },
    ),
  )

  response = client.post(
    "/api/control/diligence-prs/dpr_missing/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 404, response.text
  assert response.json()["code"] == "diligence_pr_not_found"


def test_merge_diligence_pr_route_maps_invalid_review_bundle_to_conflict(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  captured: dict[str, Any] = {}
  monkeypatch.setattr(
    diligence_prs_module,
    "_merge_handler_for_session",
    lambda _session: _planned_merge_handler(
      captured,
      execution_error={
        "code": "diligence_pr_review_bundle_invalid",
        "message": "bundle digest mismatch",
        "pr_id": "dpr_7_MSFT_abc",
      },
    ),
  )
  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(test_control_session),
    json=_merge_payload(),
  )

  assert response.status_code == 409, response.text
  assert response.json()["code"] == "diligence_pr_review_bundle_invalid"


def test_merge_diligence_pr_route_requires_confirmation_and_expected_metadata(
  client: TestClient,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  calls = 0

  def handler_factory(_session: Any):
    nonlocal calls
    calls += 1
    raise AssertionError("invalid requests must not reach the merge handler")

  monkeypatch.setattr(diligence_prs_module, "_merge_handler_for_session", handler_factory)
  url = "/api/control/diligence-prs/dpr_7_MSFT_abc/merge"
  headers = _headers(test_control_session)

  confirmation = client.post(url, headers=headers, json=_merge_payload(confirm_merge=False))
  missing_workspace = client.post(url, headers=headers, json=_merge_payload(expected_workspace_id=""))
  duplicate_proposals = client.post(
    url,
    headers=headers,
    json=_merge_payload(expected_proposal_ids=["proposal-1", "proposal-1"]),
  )
  padded_proposal = client.post(
    url,
    headers=headers,
    json=_merge_payload(expected_proposal_ids=[" proposal-1"]),
  )
  unexpected = client.post(url, headers=headers, json={**_merge_payload(), "receipt": {}})

  assert confirmation.status_code == 400
  assert missing_workspace.status_code == 422
  assert duplicate_proposals.status_code == 422
  assert padded_proposal.status_code == 422
  assert unexpected.status_code == 422
  assert calls == 0


def test_merge_diligence_pr_route_requires_owner_role(
  client: TestClient,
  control_plane_app,
  test_control_session: dict[str, Any],
  monkeypatch,
) -> None:
  session = control_plane_app.state.auth.session_store.get_session(test_control_session["session_id"])
  assert session is not None
  session.role = "invite"
  invite_payload = dict(test_control_session)
  invite_payload["session_token"] = control_plane_app.state.auth.issue_token(session)

  def handler_factory(_session: Any):
    raise AssertionError("invite sessions must not reach the merge handler")

  monkeypatch.setattr(diligence_prs_module, "_merge_handler_for_session", handler_factory)

  response = client.post(
    "/api/control/diligence-prs/dpr_7_MSFT_abc/merge",
    headers=_headers(invite_payload),
    json=_merge_payload(),
  )

  assert response.status_code == 403, response.text
  assert response.json()["detail"] == "Owner control session required to merge a diligence PR"
