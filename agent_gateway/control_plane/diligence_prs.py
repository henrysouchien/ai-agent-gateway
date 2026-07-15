from __future__ import annotations

import uuid
from dataclasses import replace
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_gateway.session import AuthManager

from .batches import _read_only_registry_for_user, _registry_for_user
from .runs import _require_bearer_session, _require_control_session
from .runs_helpers import _session_owner_user_id


_ALLOWED_MERGE_FIELDS = frozenset({
  "confirm_merge",
  "expected_handoff_id",
  "expected_proposal_ids",
  "expected_research_file_id",
  "expected_ticker",
  "expected_workspace_id",
  "process_outbox",
})
_ALLOWED_APPROVAL_FIELDS = frozenset({
  "approval_request_id",
  "expected_review_aggregate_id",
  "expected_review_content_version",
  "expected_review_snapshot_digest",
  "expected_state_version",
})
_ALLOWED_REFREEZE_FIELDS = frozenset({
  "expected_predecessor_state",
  "expected_predecessor_state_version",
  "refreeze_request_id",
})
_INVALID_INPUT_CODES = frozenset({
  "diligence_pr_merge_invalid_input",
  "invalid_tool_input",
  "proposal_series_invalid_input",
})


def build_diligence_prs_router(*, auth: AuthManager) -> APIRouter:
  router = APIRouter(prefix="/diligence-prs")

  @router.post("/{pr_id}/approve", response_model=None)
  async def approve_diligence_pr(
    request: Request,
    pr_id: str,
    payload: dict[str, Any] = Body(...),
  ) -> dict[str, Any] | JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    if str(getattr(authenticated, "role", "") or "") != "owner":
      raise HTTPException(
        status_code=403,
        detail="Owner control session required to approve a diligence PR",
      )

    approval_input = _validated_approval_input(pr_id, payload)
    result, error = await _approve_diligence_pr_for_session(
      authenticated,
      approval_input,
    )
    if error is not None:
      return JSONResponse(error, status_code=_approval_error_status_code(error))
    if not isinstance(result, dict):
      raise HTTPException(
        status_code=500,
        detail="Diligence PR approval returned an invalid response",
      )
    status = str(result.get("status") or "").strip().lower()
    if status == "conflict":
      return JSONResponse(result, status_code=409)
    if status != "approved":
      raise HTTPException(
        status_code=500,
        detail="Diligence PR approval returned an unknown status",
      )
    return result

  @router.post("/{pr_id}/refreeze", response_model=None)
  async def refreeze_diligence_pr(
    request: Request,
    pr_id: str,
    payload: dict[str, Any] = Body(...),
  ) -> dict[str, Any] | JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    if str(getattr(authenticated, "role", "") or "") != "owner":
      raise HTTPException(
        status_code=403,
        detail="Owner control session required to refreeze a diligence PR",
      )
    refreeze_input = _validated_refreeze_input(pr_id, payload)
    result, error = await _refreeze_diligence_pr_for_session(
      authenticated,
      refreeze_input,
    )
    if error is not None:
      return JSONResponse(error, status_code=_refreeze_error_status_code(error))
    if not isinstance(result, dict):
      raise HTTPException(
        status_code=500,
        detail="Diligence PR refreeze returned an invalid response",
      )
    if str(result.get("status") or "") == "conflict":
      return JSONResponse(result, status_code=409)
    if str(result.get("status") or "") != "forked":
      raise HTTPException(
        status_code=500,
        detail="Diligence PR refreeze returned an unknown status",
      )
    return result

  @router.post("/{pr_id}/merge", response_model=None)
  async def merge_diligence_pr(
    request: Request,
    pr_id: str,
    payload: dict[str, Any] = Body(...),
  ) -> dict[str, Any] | JSONResponse:
    authenticated = _require_bearer_session(request, auth)
    _require_control_session(authenticated)
    if str(getattr(authenticated, "role", "") or "") != "owner":
      raise HTTPException(status_code=403, detail="Owner control session required to merge a diligence PR")

    merge_input = _validated_merge_input(pr_id, payload)
    handler = _merge_handler_for_session(authenticated)
    planner = getattr(handler, "plan_owner_merge", None)
    executor = getattr(handler, "execute_owner_merge", None)
    if not callable(planner) or not callable(executor):
      raise HTTPException(
        status_code=503,
        detail="Exact diligence merge planning is unavailable",
      )
    from agent_gateway.tool_dispatcher_helpers import (
      PlannedWritePlanningRejected,
      TrustedToolPlan,
    )

    try:
      identity, prepared = await planner(merge_input)
    except PlannedWritePlanningRejected as exc:
      result, error = exc.tool_result()
      if error is not None:
        return JSONResponse(error, status_code=_merge_error_status_code(error))
      if isinstance(result, dict):
        return JSONResponse(result, status_code=409)
      raise

    trusted_plan = TrustedToolPlan.create(
      identity_source="reviewed_change_binding",
      identity=identity,
      prepared=prepared,
    )
    approval, approval_error = await _authorize_owner_merge_plan(
      request,
      authenticated,
      merge_input=merge_input,
      trusted_plan=trusted_plan,
    )
    if approval_error is not None:
      return JSONResponse(
        approval_error,
        status_code=_merge_error_status_code(approval_error),
      )
    if approval is None:
      raise HTTPException(
        status_code=503,
        detail="Owner merge approval was not persisted",
      )
    try:
      trusted_plan.verify_approval_request(approval)
    except Exception:
      approval_error = {
        "code": "owner_approval_identity_mismatch",
        "message": "Owner approval changed before promotion execution",
      }
      return JSONResponse(
        approval_error,
        status_code=_merge_error_status_code(approval_error),
      )
    result, error = await executor(
      trusted_plan.prepared,
      authorized_identity=trusted_plan.identity,
      approval_id=approval.approval_id,
      approval_chain_id=approval.approval_chain_id,
    )
    if error is not None:
      return JSONResponse(error, status_code=_merge_error_status_code(error))
    if not isinstance(result, dict):
      raise HTTPException(status_code=500, detail="Diligence PR merge returned an invalid response")
    if str(result.get("status") or "").strip().lower() == "blocked":
      return JSONResponse(result, status_code=409)
    if str(result.get("status") or "").strip().lower() != "success":
      raise HTTPException(status_code=500, detail="Diligence PR merge returned an unknown status")
    return result

  return router


def _validated_approval_input(pr_id: str, payload: dict[str, Any]) -> dict[str, Any]:
  unexpected = sorted(set(payload) - _ALLOWED_APPROVAL_FIELDS)
  if unexpected:
    raise HTTPException(
      status_code=422,
      detail=f"unexpected approval fields: {', '.join(unexpected)}",
    )
  missing = sorted(_ALLOWED_APPROVAL_FIELDS - set(payload))
  if missing:
    raise HTTPException(
      status_code=422,
      detail=f"missing approval fields: {', '.join(missing)}",
    )
  return {
    "pr_id": str(pr_id or "").strip(),
    "approval_request_id": _required_text(payload, "approval_request_id"),
    "expected_review_snapshot_digest": _required_text(
      payload,
      "expected_review_snapshot_digest",
    ),
    "expected_review_aggregate_id": _required_text(
      payload,
      "expected_review_aggregate_id",
    ),
    "expected_review_content_version": _required_nonnegative_int(
      payload,
      "expected_review_content_version",
    ),
    "expected_state_version": _required_nonnegative_int(
      payload,
      "expected_state_version",
    ),
  }


def _validated_refreeze_input(pr_id: str, payload: dict[str, Any]) -> dict[str, Any]:
  unexpected = sorted(set(payload) - _ALLOWED_REFREEZE_FIELDS)
  if unexpected:
    raise HTTPException(
      status_code=422,
      detail=f"unexpected refreeze fields: {', '.join(unexpected)}",
    )
  missing = sorted(_ALLOWED_REFREEZE_FIELDS - set(payload))
  if missing:
    raise HTTPException(
      status_code=422,
      detail=f"missing refreeze fields: {', '.join(missing)}",
    )
  expected_state = _required_text(payload, "expected_predecessor_state")
  if expected_state != "approved":
    raise HTTPException(
      status_code=422,
      detail="expected_predecessor_state must be 'approved'",
    )
  return {
    "pr_id": str(pr_id or "").strip(),
    "expected_predecessor_state": expected_state,
    "expected_predecessor_state_version": _required_nonnegative_int(
      payload,
      "expected_predecessor_state_version",
    ),
    "refreeze_request_id": _required_text(payload, "refreeze_request_id"),
  }


async def _refreeze_diligence_pr_for_session(
  session: Any,
  refreeze_input: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  user_id = str(getattr(session, "user_id", "") or "").strip()
  pr_id = str(refreeze_input.get("pr_id") or "").strip()
  registry = None
  try:
    registry = _registry_for_user(user_id)
    replay = registry.get_diligence_pr_refreeze_replay_v2(
      pr_id,
      refreeze_request_id=refreeze_input["refreeze_request_id"],
      refrozen_by_user_id=user_id,
    )
    if replay is not None:
      if replay.get("status") == "forked":
        return replay, None
      return None, replay
    pr = registry.get_diligence_pr(pr_id)
    if (
      str(pr.get("state") or "") != refreeze_input["expected_predecessor_state"]
      or int(pr.get("state_version") or 0)
      != refreeze_input["expected_predecessor_state_version"]
    ):
      return None, {
        "code": "diligence_pr_fork_conflict",
        "message": "legacy predecessor state/version changed before refreeze",
        "pr_id": pr_id,
      }
    try:
      from fms.core import state as fms_state
    except ModuleNotFoundError as exc:
      if exc.name not in {"fms", "fms.core", "fms.core.state"}:
        raise
      from api.fms.core import state as fms_state
    try:
      from agent.batch import diligence_prs as dpr
      from agent.batch.diligence_pr_review_bundle import build_proposal_review_leaf
    except ModuleNotFoundError as exc:
      if exc.name not in {
        "agent",
        "agent.batch",
        "agent.batch.diligence_prs",
        "agent.batch.diligence_pr_review_bundle",
      }:
        raise
      from api.agent.batch import diligence_prs as dpr
      from api.agent.batch.diligence_pr_review_bundle import build_proposal_review_leaf
    try:
      from memory import get_workspace_dir
    except ModuleNotFoundError as exc:
      if exc.name != "memory":
        raise
      from api.memory import get_workspace_dir
    repo = fms_state.resolve_repo(user_id)
    try:
      proposal_ids = dpr.normalize_proposal_ids(
        pr.get("proposal_ids"),
        field="persisted_proposal_ids",
        require_ordered_array=True,
      )
    except ValueError:
      return None, {
        "code": "diligence_pr_fork_conflict",
        "message": "legacy predecessor proposal identity is not canonical",
        "pr_id": pr_id,
      }
    if not proposal_ids:
      return None, {
        "code": "diligence_pr_fork_conflict",
        "message": "legacy predecessor has no proposals to refreeze",
        "pr_id": pr_id,
      }
    proposal_leaf_digests: list[str] = []
    for ordinal, proposal_id in enumerate(proposal_ids):
      proposal = repo.get_patch_proposal(proposal_id)
      if proposal is None:
        return None, {
          "code": "diligence_pr_fork_conflict",
          "message": f"legacy proposal is unavailable: {proposal_id}",
          "pr_id": pr_id,
        }
      proposal_leaf_digests.append(
        build_proposal_review_leaf(proposal, ordinal=ordinal).executable.digest
      )
    try:
      workspace_content_hashes = dpr.capture_workspace_content_hashes(
        pr.get("model_workspace"),
        workspace_root=get_workspace_dir(user_id),
      )
    except (OSError, ValueError) as exc:
      return None, {
        "code": "diligence_pr_fork_conflict",
        "message": f"legacy workspace content is unavailable: {exc}",
        "pr_id": pr_id,
      }
    result = registry.refreeze_legacy_diligence_pr_v2(
      pr_id,
      expected_predecessor_state_version=refreeze_input[
        "expected_predecessor_state_version"
      ],
      refreeze_request_id=refreeze_input["refreeze_request_id"],
      refrozen_by_user_id=user_id,
      proposal_ids=proposal_ids,
      proposal_leaf_digests=proposal_leaf_digests,
      workspace_payload={
        "workspace_ref": pr.get("workspace_ref"),
        "workspace_id": pr.get("workspace_id"),
        "model_workspace": pr.get("model_workspace"),
        "workspace_content_hashes": workspace_content_hashes,
      },
      base_hashes=dpr.base_hashes_for_ticker(
        repo=repo,
        ticker=str(pr.get("ticker") or ""),
        **(
          {"provider_registry": registry.provider_registry}
          if getattr(registry, "provider_registry", None) is not None
          else {}
        ),
      ),
    )
    if result.get("status") == "conflict":
      return None, result
    return result, None
  except KeyError:
    return None, {
      "code": "diligence_pr_not_found",
      "message": "diligence PR not found",
      "pr_id": pr_id,
    }
  except Exception as exc:
    return None, {
      "code": "diligence_pr_refreeze_unavailable",
      "message": str(exc),
      "pr_id": pr_id,
    }
  finally:
    if registry is not None:
      registry.close()


async def _approve_diligence_pr_for_session(
  session: Any,
  approval_input: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
  user_id = str(getattr(session, "user_id", "") or "").strip()
  pr_id = str(approval_input.get("pr_id") or "").strip()
  registry = None
  try:
    registry = _registry_for_user(user_id)
    pr = registry.get_diligence_pr(pr_id)
    pr_state = str(pr.get("state") or "").strip()
    if pr_state not in {"approved", "reviewing"}:
      return None, {
        "code": "diligence_pr_approval_state_conflict",
        "message": "diligence PR is not awaiting owner approval",
        "pr_id": pr_id,
      }
    try:
      from memory import get_workspace_dir
    except ModuleNotFoundError as exc:
      if exc.name != "memory":
        raise
      from api.memory import get_workspace_dir
    try:
      from agent.batch.diligence_pr_review_bundle import (
        ReviewBundleError,
        load_verified_review_bundle,
      )
      from agent.batch.diligence_pr_review_capture import (
        ReviewBundleLiveMismatch,
        verify_live_review_bundle_equivalence,
      )
    except ModuleNotFoundError as exc:
      if exc.name not in {
        "agent",
        "agent.batch",
        "agent.batch.diligence_pr_review_bundle",
        "agent.batch.diligence_pr_review_capture",
      }:
        raise
      from api.agent.batch.diligence_pr_review_bundle import (
        ReviewBundleError,
        load_verified_review_bundle,
      )
      from api.agent.batch.diligence_pr_review_capture import (
        ReviewBundleLiveMismatch,
        verify_live_review_bundle_equivalence,
      )
    workspace = get_workspace_dir(user_id)
    try:
      verified_bundle = load_verified_review_bundle(
        workspace,
        bundle_ref=str(pr.get("review_bundle_ref") or ""),
        expected_manifest_digest=str(pr.get("review_bundle_manifest_digest") or ""),
      )
      if pr_state != "approved":
        try:
          from fms.core import state as fms_state
        except ModuleNotFoundError as exc:
          if exc.name not in {"fms", "fms.core", "fms.core.state"}:
            raise
          from api.fms.core import state as fms_state
        repo = fms_state.resolve_repo(user_id)
        verify_live_review_bundle_equivalence(
          repo=repo,
          user_workspace=workspace,
          verified_bundle=verified_bundle,
          provider_registry=getattr(registry, "provider_registry", None),
        )
    except ReviewBundleLiveMismatch as exc:
      return None, {
        "code": "diligence_pr_review_stale",
        "message": str(exc),
        "pr_id": pr_id,
      }
    except (ReviewBundleError, OSError) as exc:
      return None, {
        "code": "diligence_pr_review_bundle_invalid",
        "message": str(exc),
        "pr_id": pr_id,
      }
    result = registry.approve_diligence_pr_v2(
      pr_id,
      expected_review_snapshot_digest=approval_input[
        "expected_review_snapshot_digest"
      ],
      expected_review_aggregate_id=approval_input["expected_review_aggregate_id"],
      expected_review_content_version=approval_input[
        "expected_review_content_version"
      ],
      expected_state_version=approval_input["expected_state_version"],
      approval_request_id=approval_input["approval_request_id"],
      approved_by_user_id=user_id,
      verified_bundle=verified_bundle,
    )
    return result, None
  except KeyError:
    return None, {
      "code": "diligence_pr_not_found",
      "message": "diligence PR not found",
      "pr_id": pr_id,
    }
  except Exception as exc:
    return None, {
      "code": "diligence_pr_approval_unavailable",
      "message": str(exc),
      "pr_id": pr_id,
    }
  finally:
    if registry is not None:
      registry.close()


def _validated_merge_input(pr_id: str, payload: dict[str, Any]) -> dict[str, Any]:
  unexpected = sorted(set(payload) - _ALLOWED_MERGE_FIELDS)
  if unexpected:
    raise HTTPException(status_code=422, detail=f"unexpected merge fields: {', '.join(unexpected)}")
  if payload.get("confirm_merge") is not True:
    raise HTTPException(status_code=400, detail="confirm_merge=true is required to merge a diligence PR")

  expected_ticker = _required_text(payload, "expected_ticker").upper()
  expected_workspace_id = _required_text(payload, "expected_workspace_id")
  expected_proposal_ids = _required_proposal_ids(payload)
  process_outbox = payload.get("process_outbox", True)
  if not isinstance(process_outbox, bool):
    raise HTTPException(status_code=422, detail="process_outbox must be a boolean")

  merge_input: dict[str, Any] = {
    "pr_id": str(pr_id or "").strip(),
    "confirm_merge": True,
    "expected_ticker": expected_ticker,
    "expected_workspace_id": expected_workspace_id,
    "expected_proposal_ids": expected_proposal_ids,
    "process_outbox": process_outbox,
  }
  for field in ("expected_research_file_id", "expected_handoff_id"):
    value = _optional_positive_int(payload, field)
    if value is not None:
      merge_input[field] = value
  return merge_input


def _required_text(payload: dict[str, Any], field: str) -> str:
  value = payload.get(field)
  if not isinstance(value, str) or not value.strip():
    raise HTTPException(status_code=422, detail=f"{field} must be a non-empty string")
  return value.strip()


def _required_proposal_ids(payload: dict[str, Any]) -> list[str]:
  raw_ids = payload.get("expected_proposal_ids")
  if not isinstance(raw_ids, list) or not raw_ids:
    raise HTTPException(status_code=422, detail="expected_proposal_ids must be a non-empty array")
  proposal_ids: list[str] = []
  seen: set[str] = set()
  for index, raw_id in enumerate(raw_ids):
    if (
      not isinstance(raw_id, str)
      or not raw_id
      or raw_id != raw_id.strip()
    ):
      raise HTTPException(
        status_code=422,
        detail=(
          f"expected_proposal_ids[{index}] must be a non-empty exact string"
        ),
      )
    proposal_id = raw_id
    if proposal_id in seen:
      raise HTTPException(status_code=422, detail="expected_proposal_ids must not contain duplicates")
    seen.add(proposal_id)
    proposal_ids.append(proposal_id)
  return proposal_ids


def _required_nonnegative_int(payload: dict[str, Any], field: str) -> int:
  value = payload.get(field)
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise HTTPException(
      status_code=422,
      detail=f"{field} must be a nonnegative integer",
    )
  return value


def _optional_positive_int(payload: dict[str, Any], field: str) -> int | None:
  if field not in payload or payload.get(field) is None:
    return None
  value = payload.get(field)
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise HTTPException(status_code=422, detail=f"{field} must be a positive integer")
  return value


def _merge_handler_for_session(session: Any):
  try:
    from agent.shared.tool_handlers.apply_proposals import make_merge_diligence_pr_handler
  except ModuleNotFoundError as exc:
    if exc.name not in {
      "agent",
      "agent.shared",
      "agent.shared.tool_handlers",
      "agent.shared.tool_handlers.apply_proposals",
    }:
      raise
    from api.agent.shared.tool_handlers.apply_proposals import make_merge_diligence_pr_handler

  return make_merge_diligence_pr_handler(
    session=session,
    batch_registry_factory=_registry_for_user,
    planning_batch_registry_factory=_read_only_registry_for_user,
  )


async def _authorize_owner_merge_plan(
  request: Request,
  session: Any,
  *,
  merge_input: dict[str, Any],
  trusted_plan: Any,
) -> tuple[Any | None, dict[str, Any] | None]:
  """Persist one explicit owner vote without consulting auto-approval policy."""

  store = getattr(request.app.state, "gateway_approval_store", None)
  required_store_methods = (
    "create_or_get_by_tool_call_id",
    "record_vote",
  )
  if store is None or any(
    not callable(getattr(store, method, None))
    for method in required_store_methods
  ):
    return None, {
      "code": "owner_approval_store_unavailable",
      "message": "Durable owner approval storage is required before promotion",
    }

  from agent_gateway.approval_policy import (
    ApprovalVote,
    RunContext,
    build_approval_request,
    utc_now,
  )
  from agent_gateway.tool_dispatcher_approval_lifecycle import (
    redact_for_approval_request,
  )
  from agent_gateway.tool_dispatcher_helpers import TrustedToolPlanError
  from research.reviewed_change_binding import (
    BindingKind,
    canonical_json_bytes,
    verify_reviewed_change_structure_v1,
  )

  user_id = _session_owner_user_id(session)
  required_owner_user_id: str | None = None
  try:
    trusted_plan.verify_integrity()
    binding = trusted_plan.identity
    if binding.binding_kind is not BindingKind.DILIGENCE_PR_MERGE:
      raise ValueError("owner merge requires a diligence execution binding")
    verify_reviewed_change_structure_v1(binding)
    bound_semantics = binding.execution_semantics.value().get(
      "normalized_tool_semantics"
    )
    review_identity = binding.base_vector.value().get("diligence_review")
    if isinstance(review_identity, dict):
      frozen_owner = review_identity.get("owner_user_id")
      if isinstance(frozen_owner, str) and frozen_owner == frozen_owner.strip():
        required_owner_user_id = frozen_owner or None
    if (
      required_owner_user_id is None
      or required_owner_user_id != user_id
      or canonical_json_bytes(bound_semantics)
      != canonical_json_bytes(merge_input)
    ):
      raise ValueError("owner merge binding does not match this request")
  except (AttributeError, KeyError, TypeError, ValueError):
    return None, {
      "code": "owner_approval_identity_mismatch",
      "message": "Planned owner merge does not match this request",
    }
  binding_digest = str(trusted_plan.reviewed_change_binding_digest or "")
  tool_call_id = f"diligence-pr-merge:{binding_digest}"
  redacted, args_hash = redact_for_approval_request(
    "merge_diligence_pr",
    merge_input,
    event_log=None,
  )

  try:
    run_context = RunContext(
      user_id=user_id,
      request_id=(
        str(request.headers.get("x-request-id") or "").strip()
        or tool_call_id
      ),
      session_id=str(getattr(session, "session_id", "") or "") or None,
      profile="control",
      channel=str(getattr(session, "channel", "") or "control"),
      decider_role="owner",
      parent_approval_id=None,
    )
    candidate = build_approval_request(
      tool_call_id=tool_call_id,
      tool_name="merge_diligence_pr",
      tool_class="state_write",
      tool_args_redacted=redacted,
      args_hash=args_hash,
      run_context=run_context,
      reason="Authenticated owner confirmed exact diligence PR promotion",
      approval_identity=trusted_plan.approval_identity(),
      approval_constraint="fresh_human_owner",
      required_owner_user_id=required_owner_user_id,
    )
    candidate = replace(
      candidate,
      authorization_mode="OWNER_CONTROL_PLANE",
    )
    approval, _created = await store.create_or_get_by_tool_call_id(candidate)

    trusted_plan.verify_approval_request(approval)
    invariant_mismatch = (
      approval.tool_name != "merge_diligence_pr"
      or approval.tool_call_id != tool_call_id
      or approval.user_id != user_id
      or approval.args_hash != args_hash
      or approval.parent_approval_id is not None
      or approval.approval_constraint != "fresh_human_owner"
      or approval.required_owner_user_id != required_owner_user_id
      or approval.authorization_mode != "OWNER_CONTROL_PLANE"
    )
    if invariant_mismatch:
      return None, {
        "code": "owner_approval_identity_mismatch",
        "message": "Existing owner approval does not match the exact merge",
      }
    if approval.state == "created":
      approval = await store.record_vote(
        approval.approval_id,
        ApprovalVote(
          vote_id=str(uuid.uuid4()),
          approval_id=approval.approval_id,
          decider_id=user_id,
          decider_role="owner",
          decision="approved",
          decision_reason=(
            "Authenticated owner control request confirmed exact promotion"
          ),
          decided_at=utc_now(),
        ),
      )
    if (
      approval.state != "approved"
      or approval.decision != "approved"
      or approval.decider_id != user_id
      or approval.decider_role != "owner"
    ):
      return None, {
        "code": "owner_approval_identity_mismatch",
        "message": "Promotion requires a non-automatic approval by this owner",
      }
    trusted_plan.verify_approval_request(approval)
    return approval, None
  except (TrustedToolPlanError, ValueError):
    return None, {
      "code": "owner_approval_identity_mismatch",
      "message": "Existing owner approval does not match the exact merge",
    }
  except Exception as exc:
    return None, {
      "code": "owner_approval_persistence_failed",
      "message": f"Owner approval could not be persisted: {type(exc).__name__}",
    }


def _merge_error_status_code(error: dict[str, Any]) -> int:
  code = str(error.get("code") or "").strip()
  if code == "merge_confirmation_required":
    return 400
  if code == "diligence_pr_not_found":
    return 404
  if code in _INVALID_INPUT_CODES:
    return 422
  if code in {
    "diligence_pr_protocol_incompatible",
    "merge_diligence_pr_unavailable",
    "owner_approval_store_unavailable",
    "owner_approval_persistence_failed",
  }:
    return 503
  if code in {
    "diligence_pr_review_bundle_invalid",
    "diligence_pr_review_identity_mismatch",
    "owner_approval_identity_mismatch",
    "diligence_pr_execution_identity_mismatch",
  }:
    return 409
  if code == "ValueError":
    return 409
  return 500


def _approval_error_status_code(error: dict[str, Any]) -> int:
  code = str(error.get("code") or "").strip()
  if code == "diligence_pr_not_found":
    return 404
  if code == "diligence_pr_protocol_incompatible":
    return 503
  if code in {
    "diligence_pr_approval_state_conflict",
    "diligence_pr_approval_version_conflict",
    "diligence_pr_review_bundle_invalid",
    "diligence_pr_review_identity_mismatch",
    "diligence_pr_review_stale",
  }:
    return 409
  return 500


def _refreeze_error_status_code(error: dict[str, Any]) -> int:
  code = str(error.get("code") or "").strip()
  if code == "diligence_pr_not_found":
    return 404
  if code == "diligence_pr_protocol_incompatible":
    return 503
  if code in {
    "diligence_pr_fork_conflict",
    "diligence_pr_promotion_in_progress",
    "diligence_pr_refreeze_required",
  }:
    return 409
  return 500


__all__ = ["build_diligence_prs_router"]
