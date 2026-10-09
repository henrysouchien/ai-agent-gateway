from __future__ import annotations

import asyncio
import copy
import datetime
import inspect
import json
import logging
from dataclasses import dataclass
from functools import partial
import secrets
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Awaitable, Callable, Mapping, cast
from collections.abc import Collection, Sequence, Set as AbstractSet

from pydantic import JsonValue, ValidationError

from agent_workflow_contracts import (
  AgentCompletionEnvelope,
  AdmittedDataRef,
  AdmittedInputBinding,
  AdmittedTask,
  AdmittedToolRoute,
  AttemptRef,
  ContentHandle,
  ContentReadGrant,
  ContractRef,
  ContextSourceRef,
  ExecuteTaskDisposition,
  InlineExactContextView,
  InvocationArgumentSelector,
  LiteralSelector,
  OperationUnavailable,
  OrdinaryDelegationTaskRef,
  OutcomePolicy,
  OutcomeRequirement,
  OutcomeRoute,
  OwnerBinding,
  ProviderToolDefinition,
  RequestedDataRef,
  ResultRequirement,
  ResolvedAuthority,
  TaskResult,
  TaskResultProvenance,
  UsageObservation,
  WorkspaceGrant,
  WorkflowNodeTaskRef,
  canonical_json_bytes,
  sha256_digest,
  terminal_task_result,
)
from agent_workflow_contracts.context_refs import declared_ref_lines, read_back_refs
from agent_workflow_contracts.models import AdmittedResearchBinding
from agent_workflow_contracts.research_file_contract import (
  context_text_with_run_binding,
)
from agent_workflow_contracts.ticker_contract import (
  TICKER_INPUT_CONTRACT,
  require_canonical_contract_ticker,
)

from .admission import (
  AdmissionOwners,
  AdmittedContext,
  ContinuationOwners,
  ContinueRequest,
  OperationRegistration,
  ParentContext,
  Refusal,
  SpawnRequest,
  admit,
  capture_job_context,
  context_for_child,
  root_context,
  current_parent_context,
)
from .autonomous_output import extract_state_update
from .approval_policy import RunContext
from .agent_result_content import make_get_agent_result_content_handler
from .job_caller import CallerContext, JobScope, JobScopedHandler, rebind_job_handlers
from .tool_result_spill import (
  SPILL_REF_PREFIX,
  SpillReadError,
  job_spill_sink,
  job_spill_output_handles,
)
from model_authority.bind import CapabilityBind
from model_authority.binding import CapabilityResolutionError
from .capability_execution import CapabilityExecutionResolver
from .code_execution import (
  CodeExecutionBundle,
  CodeExecutionConfig,
  cleanup_code_execution,
  make_code_execute_status_tool_def,
  make_code_execute_tool_def,
)
from .commercial_work_start import (
  CommercialWorkStartError,
  require_commercial_child_provider,
)
from .event_log import EventLog
from .multi_user.billing import _UsageAggregator
from .execution_snapshot import (
  build_agent_execution_snapshot,
  provider_tool_definitions_for_grant,
  render_result_instructions,
  resume_agent_execution_snapshot,
)
from .policy_imports import require_inherited_role
from .mcp_activation import RunAgentMcpActivationError
from .operation_catalog import (
  AgentOperationCatalog,
  OperationRuntimePolicy,
  ResolvedOperationRuntime,
)
from .runner import _derive_sub_agent_id
from .research_file_activity_lock import try_acquire_research_file_activity
from .research_file_current_projection import (
  ResearchFileCurrentProjectionUnavailable,
  task_registration_is_current,
)
from .runner_background_tasks import (
  agent_completion_message_id,
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
  task_completed_event_payload,
  task_correlation_payload,
  task_registered_event_payload,
)
from .directory_control_skill_catalog import PORTFOLIO_CONTEXT_NAMES
from .runner_session_events import build_agent_completion_event
from .session import GatewaySession, child_gateway_session
from .skill_limits import SkillExecutionLimits
from .skill_lifecycle import drain_owned_lifecycle_task
from .skills import (
  ResolvedAgentOperation,
  SkillLoader,
  SkillProfile,
  SkillStateStore,
  compile_agent_operation,
  generic_explore_profile,
  operation_tool_ids,
)
from .sub_agent_cost_observation import (
  CostObservationThresholdError,
  DEFAULT_COST_OBSERVATION_THRESHOLD_USD,
  resolve_cost_observation_threshold_usd,
)
from .sub_agent_result_evidence import (
  SubAgentResultEvidence,
  collect_sub_agent_result_evidence,
  merge_sub_agent_result_evidence,
)
from .sub_agent_skill_events import (
  DurableSkillEventPersistenceError,
  SkillRunEventEmitter,
)
from .tool_definition import validate_live_tool_route_binding
from .capability_resolution import (
  admitted_tool_routes,
  derive_dispatcher_allowlist,
  granted_tool_ids,
)
from .execution_identity import dispatch_identity, execution_identity_from_session
from .sub_agent_scope_receipt import (
  ADMITTED_TASK_METADATA_KEY,
  OperationToolAdmissionError,
  admit_operation_tools,
  dispatcher_scopes_from_admitted_routes,
  reissue_tool_grant,
)
from .sub_agent_helpers import (
  ExcludedToolsResolver as ExcludedToolsResolver,
  MutationModeExclusionsApplier as MutationModeExclusionsApplier,
  NeedsApprovalResolver as NeedsApprovalResolver,
  _ARTIFACT_EMIT_TOOLS as _ARTIFACT_EMIT_TOOLS,
  _DEFAULT_SYSTEM_PROMPT_TEMPLATE as _DEFAULT_SYSTEM_PROMPT_TEMPLATE,
  _extract_ticker_from_task as _extract_ticker_from_task,
  _optional_research_file_id as _optional_research_file_id,
  _resolve_context_ticker as _resolve_context_ticker,
  make_get_background_result_tool_def as make_get_background_result_tool_def,
  make_resume_tool_def as make_resume_tool_def,
  make_run_agent_tool_def as make_run_agent_tool_def,
  make_send_message_tool_def as make_send_message_tool_def,
)
from .sub_agent_background_result import make_get_background_result_handler
from .sub_agent_messages import make_send_message_handler as _make_send_message_handler
from .sub_agent_narrative_result import read_task_result_terminal_narrative
from . import sub_agent_skill_state as _sub_agent_skill_state
from . import sub_agent_tool_definitions as _sub_agent_tool_definitions
from .task_registry import (
  ResumeSuccessorConflictError,
  TaskEntry,
  TaskState,
  task_state_for_result,
  is_canonical_handle_ref,
  job_output_handles,
)
from .tool_dispatcher import LocalToolHandler, ToolDispatcher, ToolInterceptor
from .transcript import (
  ChildRunSegment,
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
  place_resume_messages,
  reconstruct_child_run_lineage,
  reconstruct_messages_for_task,
  reconstruct_parent_messages,
)


_RESEARCH_FILE_ID_CONTRACT_MANIFEST = {
  "namespace": "workflow",
  "name": "research-file-id",
  "version": "1.0",
  "canonical_encoding": "utf-8-json-integer",
  "persisted_rule": "positive-signed-64-bit-integer/v1",
}
_RESEARCH_FILE_ID_INPUT_CONTRACT = ContractRef(
  namespace=_RESEARCH_FILE_ID_CONTRACT_MANIFEST["namespace"],
  name=_RESEARCH_FILE_ID_CONTRACT_MANIFEST["name"],
  version=_RESEARCH_FILE_ID_CONTRACT_MANIFEST["version"],
  digest=sha256_digest(_RESEARCH_FILE_ID_CONTRACT_MANIFEST),
)
_REQUIRED_CONTEXT_CONTRACT_MANIFEST = {
  "namespace": "workflow",
  "name": "required-context",
  "version": "1.0",
  "canonical_encoding": "utf-8-json",
}
_REQUIRED_CONTEXT_INPUT_CONTRACT = ContractRef(
  namespace=_REQUIRED_CONTEXT_CONTRACT_MANIFEST["namespace"],
  name=_REQUIRED_CONTEXT_CONTRACT_MANIFEST["name"],
  version=_REQUIRED_CONTEXT_CONTRACT_MANIFEST["version"],
  digest=sha256_digest(_REQUIRED_CONTEXT_CONTRACT_MANIFEST),
)
# Declared keys naming the child's subject, not an exact workflow ref: ticker and
# research_file_id bind below, portfolio context rides the parent's dispatch
# scope, and the industry or artifact subject is the objective's.
_SUBJECT_CONTEXT_KEYS = frozenset({
  "ticker",
  "research_file_id",
  "industry",
  "artifact",
}) | PORTFOLIO_CONTEXT_NAMES


def _child_run_context(
  *,
  parent_session: GatewaySession | None,
  tool_ctx: Any,
  skill_run_id: str | None,
  skill_name: str | None,
  admitted_skill_execution_limits: SkillExecutionLimits | None,
  research_file_id: int | None,
  user_id: str | None,
  session_id: str,
  approval_policy: Any | None,
) -> RunContext:
  """Bind lifecycle tools to the exact server-owned child run identity."""

  return RunContext(
    user_id=str(user_id or getattr(parent_session, "user_id", None) or "unknown"),
    request_id=str(
      getattr(tool_ctx, "request_id", None)
      or getattr(parent_session, "request_id", None)
      or session_id
      or "request-unknown"
    ),
    session_id=session_id or getattr(parent_session, "session_id", None),
    run_id=skill_run_id,
    # A child runs inside its parent's profile partition: the authority owner
    # (server_policies.authority_denies_tool) reads this as a profile name.
    # The skill name rides `skill`.
    profile=str(getattr(tool_ctx, "profile", None) or skill_name or "sub_agent"),
    channel=str(getattr(parent_session, "channel", None) or "web"),
    skill=skill_name,
    admitted_skill_execution_limits=admitted_skill_execution_limits,
    research_file_id=research_file_id,
    decider_role=require_inherited_role(parent_session),
    policy_bundle_hash=str(
      getattr(approval_policy, "policy_bundle_hash", "unknown")
    ),
  )


DurableSkillEventAppender = Callable[
  [dict[str, Any]],
  Awaitable[Any | None],
]
DurableSkillEventConfirmer = Callable[
  [dict[str, Any]],
  Awaitable[dict[str, Any] | None],
]


def _durable_skill_event_transport(
  runner: Any,
) -> tuple[
  DurableSkillEventAppender,
  DurableSkillEventConfirmer,
] | None:
  """Resolve the required runner protocol for named-skill lifecycle events."""

  appender: DurableSkillEventAppender | None = getattr(
    runner,
    "_append_durable_event",
    None,
  )
  confirmer: DurableSkillEventConfirmer | None = getattr(
    runner,
    "_confirm_durable_skill_event",
    None,
  )
  if not callable(appender) or not callable(confirmer):
    return None
  return appender, confirmer


def _effective_mcp_session_inject_servers(
  *,
  admitted_mcp_scope: Mapping[str, set[str] | frozenset[str]],
  configured_servers: AbstractSet[str] | None,
) -> set[str] | None:
  """Bound configured session injection to the exact admitted MCP routes.

  ``configured_servers`` may be a live view onto the parent's activation fold
  (the autonomous and interactive child handlers hand one through), so it is
  read — and materialized — here, at spawn time: a server the parent loaded
  after its handlers were installed is still an injection target.
  """

  effective = set(admitted_mcp_scope)
  if configured_servers is not None:
    effective &= set(configured_servers)
  return effective


def _durable_skill_event_persistence_error() -> dict[str, str]:
  return {
    "code": "durable_skill_event_persistence_failed",
    "message": (
      "The named-agent lifecycle event was not durably confirmed; "
      "the child result was not accepted"
    ),
  }


log = logging.getLogger("agent_gateway.sub_agent")

_CODE_EXECUTION_TOOL_NAMES = frozenset(
  definition["name"]
  for definition in (
    make_code_execute_tool_def(CodeExecutionConfig(), ()),
    make_code_execute_status_tool_def(),
  )
)


def _child_tool_definitions_getter(
  *,
  runner: Any,
  mcp_client: Any,
  excluded_tools: set[str],
  extra_tool_definitions: list[dict[str, Any]] | None = None,
  local_tool_handlers: Mapping[str, Any] | None = None,
  granted_tools: frozenset[str] | None = None,
) -> Callable[[], list[dict[str, Any]]] | None:
  getter = _sub_agent_tool_definitions.child_tool_definitions_getter(
    runner=runner,
    mcp_client=mcp_client,
    excluded_tools=excluded_tools,
    extra_tool_definitions=extra_tool_definitions,
    local_tool_handlers=local_tool_handlers,
  )
  if getter is None or granted_tools is None:
    return getter

  def _declared_tool_definitions() -> list[dict[str, Any]]:
    return [
      definition
      for definition in getter()
      if str(definition.get("name") or "") in granted_tools
    ]

  return _declared_tool_definitions


def _admitted_tool_definitions_getter(
  definitions: Sequence[ProviderToolDefinition],
  *,
  unavailable_local_tools: Collection[str] = (),
) -> Callable[[], list[dict[str, Any]]]:
  admitted = tuple(
    copy.deepcopy(definition.definition)
    for definition in definitions
    if definition.definition.get("name") not in unavailable_local_tools
  )

  def _tool_definitions() -> list[dict[str, Any]]:
    return [copy.deepcopy(definition) for definition in admitted]

  return _tool_definitions


def _operation_private_mcp_tool_definitions(
  *,
  profile: SkillProfile | None = None,
  mcp_tools_by_server: Mapping[str, Collection[str]] | None = None,
  mcp_client: Any,
  exact_tool_ids: frozenset[str],
  tool_definition_projector: (
    Callable[[dict[str, Any]], dict[str, Any] | None] | None
  ) = None,
) -> list[dict[str, Any]]:
  """Read connected schemas for one operation without exposing them to its parent."""

  declared_tools = (
    mcp_tools_by_server
    if mcp_tools_by_server is not None
    else (profile.mcp_tools if profile is not None else {})
  )

  get_server_definitions: (
    Callable[[set[str]], Sequence[dict[str, Any]]] | None
  ) = getattr(
    mcp_client,
    "get_server_tool_definitions",
    None,
  )
  if not callable(get_server_definitions):
    return []
  declared_servers = set(declared_tools or {})
  projected: list[dict[str, Any]] = []
  for definition in get_server_definitions(declared_servers):
    if (
      not isinstance(definition, dict)
      or str(definition.get("name") or "") not in exact_tool_ids
    ):
      continue
    candidate = (
      tool_definition_projector(definition)
      if tool_definition_projector is not None
      else definition
    )
    if candidate is not None:
      if not isinstance(candidate, dict):
        raise TypeError("tool definition projector must return an object or None")
      projected.append(candidate)
  return projected


def _operation_private_mcp_tool_ids(
  mcp_tools_by_server: Mapping[str, Collection[str]],
  *,
  exact_tool_ids: frozenset[str],
) -> frozenset[str]:
  return frozenset(
    tool_id
    for tool_ids in mcp_tools_by_server.values()
    for tool_id in tool_ids
    if tool_id in exact_tool_ids
  )


def _operation_has_investment_claim_routes(
  mcp_tools_by_server: Mapping[str, Collection[str]],
  *,
  exact_tool_ids: frozenset[str] | None,
) -> bool:
  investment_tools = mcp_tools_by_server.get("idea-workbench-mcp", ())
  return (
    "start_quant_research" in investment_tools
    and (
      exact_tool_ids is None
      or "start_quant_research" in exact_tool_ids
    )
  )


def _runtime_policy_dispatch_projection(
  policy: OperationRuntimePolicy,
  *,
  mcp_client: Any,
  live_local_tool_ids: AbstractSet[str] = frozenset(),
) -> tuple[frozenset[str], Mapping[str, str]]:
  """Resolve canonical policy identities onto exact live dispatcher names."""

  local_tool_ids = {
    tool_id
    for tool_id in policy.exact_tool_ids
    if not tool_id.startswith("mcp__")
  }
  canonical_mcp_ids = {
    f"mcp__{server_id}__{tool_id}": (server_id, tool_id)
    for server_id, tool_ids in policy.mcp_tools_by_server.items()
    for tool_id in tool_ids
  }
  live_routes_by_identity: dict[tuple[str, str], str] = {}
  if canonical_mcp_ids:
    get_route_bindings: (
      Callable[[set[str]], Sequence[object]] | None
    ) = getattr(
      mcp_client,
      "get_server_tool_route_bindings",
      None,
    )
    if not callable(get_route_bindings):
      raise ValueError(
        "operation MCP identities require exact live route resolution"
      )
    first_canonical_id = sorted(canonical_mcp_ids)[0]
    try:
      route_bindings = get_route_bindings({
        server_id
        for server_id, _tool_id in canonical_mcp_ids.values()
      })
      for raw_binding in route_bindings:
        binding = validate_live_tool_route_binding(raw_binding)
        identity = (
          binding.logical_server_id,
          binding.logical_name,
        )
        live_routes_by_identity[identity] = binding.exposed_name
    except Exception as exc:
      raise ValueError(
        f"canonical MCP identity {first_canonical_id!r} "
        "live route resolution failed"
      ) from exc

  canonical_to_exposed: dict[str, str] = {}
  canonical_by_exposed: dict[str, str] = {}
  for canonical_id, (server_id, original_tool_id) in sorted(
    canonical_mcp_ids.items()
  ):
    exposed_tool_id = live_routes_by_identity.get((
      server_id,
      original_tool_id,
    ))
    if (
      type(exposed_tool_id) is not str
      or not exposed_tool_id
      or exposed_tool_id != exposed_tool_id.strip()
    ):
      raise ValueError(
        f"canonical MCP identity {canonical_id!r} has no exact live route"
      )
    previous = canonical_by_exposed.get(exposed_tool_id)
    if previous is not None and previous != canonical_id:
      raise ValueError(
        "canonical MCP identities collide on exposed dispatcher name "
        f"{exposed_tool_id!r}: {previous!r}, {canonical_id!r}"
      )
    canonical_to_exposed[canonical_id] = exposed_tool_id
    canonical_by_exposed[exposed_tool_id] = canonical_id

  exact_mcp_tool_ids = set(canonical_to_exposed.values())
  local_mcp_collisions = sorted(local_tool_ids & exact_mcp_tool_ids)
  if local_mcp_collisions:
    raise ValueError(
      "local exact tools collide with exposed MCP dispatcher names: "
      + ", ".join(local_mcp_collisions)
    )
  live_local_mcp_collisions = sorted(
    set(live_local_tool_ids) & exact_mcp_tool_ids
  )
  if live_local_mcp_collisions:
    raise ValueError(
      "live local handlers collide with exposed MCP dispatcher names: "
      + ", ".join(live_local_mcp_collisions)
    )
  return (
    frozenset(local_tool_ids | exact_mcp_tool_ids),
    MappingProxyType(dict(sorted(canonical_to_exposed.items()))),
  )


def _tool_grant_has_investment_claim_routes(
  tool_grant: Any,
) -> bool:
  return any(
    entry.tool_id == "start_quant_research"
    for entry in tool_grant.tools
  )




def _result_response_text(result: Any | None) -> str:
  return _sub_agent_skill_state.result_response_text(result)


def _skill_state_prompt(skill_name: str, previous_state: dict[str, Any]) -> str:
  return _sub_agent_skill_state.skill_state_prompt(skill_name, previous_state)


def _capability_resolution_error(exc: CapabilityResolutionError) -> dict[str, Any]:
  return {
    "code": exc.code,
    "message": str(exc),
    "details": exc.to_error(),
  }


def _ordinary_outcome_policy() -> OutcomePolicy:
  return OutcomePolicy(routes=tuple(
    OutcomeRoute(disposition=disposition, action="settle")
    for disposition in (
      "complete",
      "partial",
      "insufficient_evidence",
      "blocked",
      "not_assessed",
    )
  ))


def _foreground_completion_transport_ready(runner: Any) -> bool:
  """Return whether one runner can publish a readable foreground result.

  A few unit-level handler doubles intentionally exercise only admission and
  canonical TaskResult creation. Real runners always provide this durable
  identity/storage surface because narrative child execution already requires
  the same session log and workspace.
  """

  return (
    callable(getattr(runner, "_append_durable_event", None))
    and getattr(runner, "_agent_session_log", None) is not None
    and bool(str(getattr(runner, "_gateway_session_id", "") or ""))
    and bool(str(getattr(runner, "_runner_id", "") or ""))
    and getattr(runner, "_workspace_dir", None) is not None
  )


async def _publish_foreground_completion(
  *,
  runner: Any,
  result: TaskResult,
  admitted_task: AdmittedTask,
  agent_name: str,
  capability_bind_receipt: Mapping[str, str],
  parent_turn_id: str | None,
  call_index: int,
) -> AgentCompletionEnvelope:
  """Durably publish and normalize one direct foreground delegation result."""

  runner_id = str(getattr(runner, "_runner_id", "") or "")
  gateway_session_id = str(
    getattr(runner, "_gateway_session_id", "") or ""
  )
  role = str(getattr(runner, "_role", "") or "writer")
  workspace_dir = getattr(runner, "_workspace_dir", None)
  if not runner_id or not gateway_session_id or workspace_dir is None:
    raise RuntimeError(
      "foreground completion requires durable parent session identity"
    )
  if result.attempt.physical_task_id != admitted_task.attempt.physical_task_id:
    raise RuntimeError(
      "foreground completion result does not settle the admitted task"
    )

  policy = ordinary_parent_result_policy(result)

  def _read_terminal(canonical: TaskResult) -> str:
    return read_task_result_terminal_narrative(
      canonical,
      workspace_dir=workspace_dir,
    )

  def _read_grant(source: ContentHandle) -> ContentReadGrant:
    identity = sha256_digest({
      "task_id": result.attempt.physical_task_id,
      "content_id": source.content_id,
      "scope": "direct_parent",
      "principal_id": runner_id,
    }).removeprefix("sha256:")
    return ContentReadGrant(
      grant_id=f"content-read:{identity}",
      content_id=source.content_id,
      scope="direct_parent",
      principal_id=runner_id,
    )

  envelope = build_agent_completion_envelope(
    result,
    policy=policy,
    terminal_narrative_reader=_read_terminal,
    read_grant_factory=_read_grant,
    message_id=agent_completion_message_id(result),
  )
  now = datetime.datetime.now(datetime.UTC).timestamp()
  result_payload = result.model_dump(mode="json")
  entry = TaskEntry(
    task_id=result.attempt.physical_task_id,
    task_type="foreground_agent",
    agent_name=agent_name,
    state=task_state_for_result(result),
    started_at=now,
    completed_at=now,
    result=result_payload,
    metadata={
      "task_type": "foreground",
      "owner_runner_id": runner_id,
      "owner_role": role,
      "parent_turn_id": parent_turn_id,
      "call_index": call_index,
      "admitted_task": admitted_task.model_dump(mode="json"),
      "parent_result_policy": policy.model_dump(mode="json"),
    },
    capability_bind_receipt=dict(capability_bind_receipt),
    admitted_task=admitted_task,
    task_result=result,
    parent_result_policy=policy,
    completion_envelope=envelope,
  )
  correlation = task_correlation_payload(
    entry,
    runner_id=runner_id,
    role=role,
  )
  append = runner._append_durable_event
  await append(task_registered_event_payload(
    entry,
    correlation_payload=correlation,
    agent_name=agent_name,
    parent_session_id=gateway_session_id,
  ))
  await append(task_completed_event_payload(
    entry,
    entry.state,
    correlation_payload=correlation,
    completed_at=now,
  ))
  await append(build_agent_completion_event(
    task_id=entry.task_id,
    envelope=envelope,
    ts=now,
  ))
  return envelope


def seal_admitted_task_payload(payload: dict[str, Any]) -> dict[str, Any]:
  """Stamp the canonical ``admitted_task_digest`` on an admission payload.

  This is the single definition of ``admitted_task_digest``: the SHA-256
  canonical digest of the complete admission payload — every other field,
  component digests included — computed last. Callers must not supply a
  precomputed digest or reimplement the rule over a field subset.
  """
  if "admitted_task_digest" in payload:
    raise ValueError("admitted_task_digest is derived from the payload, never supplied")
  payload["admitted_task_digest"] = sha256_digest(payload)
  return payload


# Stage refs a door mints when it persists that stage. A read also names them
# (``get_current_model`` names the sets its version compiled), but that is a
# fact of an earlier write, so it never displaces a set a door persisted in
# this turn; a later door result still replaces an earlier one.
_DOOR_FRESH_REF_KEYS = frozenset({"forecast_assumption_set_ref", "scenario_set_ref"})
# The refs the parent's own ``get_current_model`` read names for a research
# file's current version (``_current_model_read_payload``); any other declared
# ``<skill>_ref`` is minted by the parent's ``get_skill_artifact`` read of that
# skill (``read_back_refs``).
_CURRENT_MODEL_REF_KEYS = frozenset({
  "model_version_ref",
  "source_bundle_id",
  "forecast_assumption_set_ref",
  "scenario_set_ref",
  "valuation_policy_ref",
})


def _required_context_next_call(
  key: str,
  *,
  research_file_id: int | None,
  ticker: str | None,
) -> dict[str, Any] | None:
  """The parent's in-turn read that returns ``key``, or None without a subject."""
  if key in _CURRENT_MODEL_REF_KEYS:
    if research_file_id is None:
      return None
    return {
      "tool_name": "get_current_model",
      "arguments": {"research_file_id": research_file_id},
    }
  if ticker is None or not key.endswith("_ref"):
    return None
  return {
    "tool_name": "get_skill_artifact",
    "arguments": {"ticker": ticker, "skill": key.removesuffix("_ref").replace("_", "-")},
  }


def _ticker_refusal(
  code: str, message: str, tool_input: Mapping[str, Any], *, required: bool,
) -> dict[str, Any]:
  """A ticker refusal whose repair follows the operation's declared context."""
  refusal: dict[str, Any] = {
    "code": code,
    "message": message,
    "next_call": {
      "tool_name": "run_agent",
      "arguments": {key: value for key, value in tool_input.items() if key != "ticker"},
    },
  }
  if required:
    refusal["missing_keys"] = ["ticker"]
    refusal["fix"] = "add ticker: the one issuer this operation is about"
  else:
    refusal["fix"] = "this operation declares no ticker; repeat the call without one"
  return refusal


def _result_refs(result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
  """Name the exact refs one successful tool result returns.

  A result carries refs at its top level, in a context read's ``header`` (the
  door's echo of the refs it validated) and in a door's ``readback`` (minted by
  the shared ``read_back_refs``). Results that delegate carry their doors' refs
  at ``stage_refs``: a pipeline result its completed stages' bag, and a
  ``run_agent`` completion envelope its child's terminal-door refs, inline even
  when the child's result overflowed to a handle. Returns every ref, and the
  subset a door minted.
  """
  header = result.get("header")
  stage_refs = result.get("stage_refs")
  door_refs = {
    **(stage_refs if isinstance(stage_refs, Mapping) else {}),
    **read_back_refs(result),
  }
  refs = {
    **result,
    **(header if isinstance(header, Mapping) else {}),
    **door_refs,
  }
  return refs, door_refs


def _turn_required_refs(
  runner: Any,
  required_keys: tuple[str, ...],
) -> dict[str, Any]:
  """Read exact declared inputs from this runner turn, never session history.

  Every successful tool call contributes the refs its result returns and the
  exact ref objects its input carried: the tool that accepted them validated
  them. A failed call contributes nothing; a later call overrides an earlier
  one, except that a stage ref a door minted (``_DOOR_FRESH_REF_KEYS``) is
  replaced only by a later door's.
  """
  refs: dict[str, Any] = {}
  if not required_keys:
    return refs
  door_minted: set[str] = set()
  turn_log = getattr(runner, "_log", None)
  tool_inputs: dict[Any, Mapping[str, Any]] = {}
  for entry in getattr(turn_log, "entries", ()):
    event = entry.event
    tool_call_id = event.get("tool_call_id")
    if event.get("type") == "tool_call_start":
      tool_input = event.get("tool_input")
      if tool_call_id is not None and isinstance(tool_input, Mapping):
        tool_inputs[tool_call_id] = tool_input
      continue
    if (
      event.get("type") != "tool_call_complete"
      or event.get("error") is not None
      or event.get("is_error")
    ):
      continue
    result = event.get("result")
    result_refs, door_refs = (
      _result_refs(result) if isinstance(result, Mapping) else ({}, {})
    )
    for returned, minted in (
      (tool_inputs.get(tool_call_id, {}), {}),
      (result_refs, door_refs),
    ):
      for key in required_keys:
        if key in _DOOR_FRESH_REF_KEYS and minted.get(key) is not None:
          refs[key] = minted[key]
          door_minted.add(key)
        elif returned.get(key) is not None and not (
          key in _DOOR_FRESH_REF_KEYS and key in door_minted
        ):
          refs[key] = returned[key]
  return refs


def _ref_admitted_input(
  name: str,
  value: Any,
  *,
  invocation_id: str,
  parent_session: GatewaySession | None,
) -> AdmittedInputBinding:
  """Bind an exact server-returned JSON value, not a model-filled argument."""
  encoded = canonical_json_bytes(value)
  content_id = sha256_digest(value)
  logical_source_id = f"turn-tool-result:{name}"
  contract = _REQUIRED_CONTEXT_INPUT_CONTRACT
  source_ref = ContextSourceRef(logical_source_id=logical_source_id, content_id=content_id)
  return AdmittedInputBinding(
    name=name,
    source=AdmittedDataRef(
      request=RequestedDataRef(
        name=name,
        selector=LiteralSelector(value=value),
        expected_contract=contract,
      ),
      source_kind="literal",
      logical_source_id=logical_source_id,
      owner=OwnerBinding(
        tenant_id=str(getattr(parent_session, "tenant_id", "") or ""),
        session_id=getattr(parent_session, "session_id", None),
        invocation_id=invocation_id,
      ),
      actual_contract=contract,
      content=ContentHandle(
        content_id=content_id,
        content_sha256=content_id.removeprefix("sha256:"),
        content_bytes=len(encoded),
        content_chars=len(encoded.decode("utf-8")),
        contract=contract,
        media_type="application/json",
        encoding="utf-8",
        retention="session",
      ),
    ),
    context=InlineExactContextView(
      source=source_ref,
      content=value,
      content_bytes=len(encoded),
    ),
  )


def _ticker_admitted_input(
  ticker: str,
  *,
  invocation_id: str,
  parent_session: Any | None,
) -> AdmittedInputBinding:
  canonical = require_canonical_contract_ticker(ticker)
  tenant_id = str(getattr(parent_session, "tenant_id", "") or "").strip()
  session_id = str(getattr(parent_session, "session_id", "") or "").strip()
  if not tenant_id or not session_id:
    raise ValueError(
      "typed ticker admission requires exact tenant and session ownership"
    )
  encoded = canonical_json_bytes(canonical)
  content_id = sha256_digest(canonical)
  logical_source_id = "invocation-argument:ticker"
  source_ref = ContextSourceRef(
    logical_source_id=logical_source_id,
    content_id=content_id,
  )
  requested = RequestedDataRef(
    name="ticker",
    selector=InvocationArgumentSelector(argument_name="ticker"),
    expected_contract=TICKER_INPUT_CONTRACT,
  )
  content = ContentHandle(
    content_id=content_id,
    content_sha256=content_id.removeprefix("sha256:"),
    content_bytes=len(encoded),
    content_chars=len(encoded.decode("utf-8")),
    contract=TICKER_INPUT_CONTRACT,
    media_type="application/json",
    encoding="utf-8",
    retention="session",
  )
  return AdmittedInputBinding(
    name="ticker",
    source=AdmittedDataRef(
      request=requested,
      source_kind="invocation_argument",
      logical_source_id=logical_source_id,
      owner=OwnerBinding(
        tenant_id=tenant_id,
        session_id=session_id,
        invocation_id=invocation_id,
      ),
      actual_contract=TICKER_INPUT_CONTRACT,
      content=content,
    ),
    context=InlineExactContextView(
      source=source_ref,
      content=canonical,
      content_bytes=len(encoded),
    ),
  )


def _require_research_file_id(value: object) -> int:
  if (
    isinstance(value, bool)
    or not isinstance(value, int)
    or not 1 <= value < (1 << 63)
  ):
    raise ValueError(
      "research_file_id must be a positive signed 64-bit integer"
    )
  return value


def _research_file_id_admitted_input(
  research_file_id: int,
  *,
  invocation_id: str,
  parent_session: Any | None,
) -> AdmittedInputBinding:
  canonical = _require_research_file_id(research_file_id)
  tenant_id = str(getattr(parent_session, "tenant_id", "") or "").strip()
  session_id = str(getattr(parent_session, "session_id", "") or "").strip()
  if not tenant_id or not session_id:
    raise ValueError(
      "typed research-file admission requires exact tenant and session ownership"
    )
  encoded = canonical_json_bytes(canonical)
  content_id = sha256_digest(canonical)
  logical_source_id = "research-turn:research_file_id"
  source_ref = ContextSourceRef(
    logical_source_id=logical_source_id,
    content_id=content_id,
  )
  requested = RequestedDataRef(
    name="research_file_id",
    selector=LiteralSelector(value=canonical),
    expected_contract=_RESEARCH_FILE_ID_INPUT_CONTRACT,
  )
  content = ContentHandle(
    content_id=content_id,
    content_sha256=content_id.removeprefix("sha256:"),
    content_bytes=len(encoded),
    content_chars=len(encoded.decode("utf-8")),
    contract=_RESEARCH_FILE_ID_INPUT_CONTRACT,
    media_type="application/json",
    encoding="utf-8",
    retention="session",
  )
  return AdmittedInputBinding(
    name="research_file_id",
    source=AdmittedDataRef(
      request=requested,
      source_kind="literal",
      logical_source_id=logical_source_id,
      owner=OwnerBinding(
        tenant_id=tenant_id,
        session_id=session_id,
        invocation_id=invocation_id,
      ),
      actual_contract=_RESEARCH_FILE_ID_INPUT_CONTRACT,
      content=content,
    ),
    context=InlineExactContextView(
      source=source_ref,
      content=canonical,
      content_bytes=len(encoded),
    ),
  )


def _research_file_id_from_admitted_inputs(
  inputs: tuple[AdmittedInputBinding, ...],
  *,
  required: bool,
  owner_invocation_id: str | None = None,
) -> int | None:
  matches = tuple(
    binding for binding in inputs if binding.name == "research_file_id"
  )
  if not matches:
    if required:
      raise ValueError(
        "research-file-required admission has no typed identity binding"
      )
    return None
  if len(matches) != 1:
    raise ValueError(
      "admission must contain exactly one typed research-file binding"
    )
  binding = matches[0]
  source = binding.source
  request = source.request
  selector = request.selector
  context = binding.context
  selector_research_file_id = (
    _require_research_file_id(selector.value)
    if isinstance(selector, LiteralSelector)
    else None
  )
  if (
    not isinstance(selector, LiteralSelector)
    or not isinstance(context, InlineExactContextView)
    or request.name != "research_file_id"
    or selector_research_file_id != context.content
    or request.context_policy is not None
    or request.expected_contract != _RESEARCH_FILE_ID_INPUT_CONTRACT
    or source.source_kind != "literal"
    or source.logical_source_id != "research-turn:research_file_id"
    or source.actual_contract != _RESEARCH_FILE_ID_INPUT_CONTRACT
    or source.content.contract != _RESEARCH_FILE_ID_INPUT_CONTRACT
    or source.read_grant is not None
  ):
    raise ValueError(
      "admission contains a non-canonical typed research-file binding"
    )
  if (
    not source.owner.tenant_id
    or not source.owner.session_id
    or not source.owner.invocation_id
    or (
      owner_invocation_id is not None
      and source.owner.invocation_id != owner_invocation_id
    )
  ):
    raise ValueError(
      "admission research-file binding has invalid owner identity"
    )
  research_file_id = _require_research_file_id(context.content)
  encoded = canonical_json_bytes(research_file_id)
  content_id = sha256_digest(research_file_id)
  if (
    context.source.logical_source_id != source.logical_source_id
    or context.source.content_id != content_id
    or context.content_bytes != len(encoded)
    or source.content.content_id != content_id
    or source.content.content_sha256 != content_id.removeprefix("sha256:")
    or source.content.content_bytes != len(encoded)
    or source.content.content_chars != len(encoded.decode("utf-8"))
    or source.content.media_type != "application/json"
    or source.content.encoding != "utf-8"
    or source.content.retention != "session"
  ):
    raise ValueError(
      "admission research-file content does not match its exact binding"
    )
  return research_file_id


def _ticker_from_admitted_inputs(
  inputs: tuple[AdmittedInputBinding, ...],
  *,
  required: bool,
  owner_invocation_id: str | None = None,
) -> str | None:
  matches = tuple(binding for binding in inputs if binding.name == "ticker")
  if not matches:
    if required:
      raise ValueError("ticker-required admission has no typed ticker binding")
    return None
  if len(matches) != 1:
    raise ValueError("admission must contain exactly one typed ticker binding")
  binding = matches[0]
  source = binding.source
  request = source.request
  selector = request.selector
  context = binding.context
  if not isinstance(selector, InvocationArgumentSelector):
    raise ValueError("ticker binding must use the invocation argument selector")
  if (
    request.name != "ticker"
    or selector.argument_name != "ticker"
    or request.context_policy is not None
    or request.expected_contract != TICKER_INPUT_CONTRACT
    or source.source_kind != "invocation_argument"
    or source.logical_source_id != "invocation-argument:ticker"
    or source.actual_contract != TICKER_INPUT_CONTRACT
    or source.content.contract != TICKER_INPUT_CONTRACT
    or source.read_grant is not None
    or not isinstance(context, InlineExactContextView)
  ):
    raise ValueError("admission contains a non-canonical typed ticker binding")
  if (
    not source.owner.tenant_id
    or not source.owner.session_id
    or not source.owner.invocation_id
    or (
      owner_invocation_id is not None
      and source.owner.invocation_id != owner_invocation_id
    )
  ):
    raise ValueError("admission ticker binding has invalid owner identity")
  ticker = require_canonical_contract_ticker(context.content)
  encoded = canonical_json_bytes(ticker)
  content_id = sha256_digest(ticker)
  if (
    context.source.logical_source_id != source.logical_source_id
    or context.source.content_id != content_id
    or context.content_bytes != len(encoded)
    or source.content.content_id != content_id
    or source.content.content_sha256 != content_id.removeprefix("sha256:")
    or source.content.content_bytes != len(encoded)
    or source.content.content_chars != len(encoded.decode("utf-8"))
    or source.content.media_type != "application/json"
    or source.content.encoding != "utf-8"
    or source.content.retention != "session"
  ):
    raise ValueError("admission ticker content does not match its exact binding")
  return ticker


def _ordinary_logical_task(
  admitted_task: AdmittedTask,
) -> OrdinaryDelegationTaskRef:
  logical_task = admitted_task.logical_task
  if not isinstance(logical_task, OrdinaryDelegationTaskRef):
    raise ValueError("ordinary ticker admission requires a delegation identity")
  delegation_id = str(logical_task.delegation_id or "").strip()
  if not delegation_id:
    raise ValueError("ordinary ticker admission has no logical invocation owner")
  return logical_task


def _ordinary_logical_invocation_owner(
  logical_task: OrdinaryDelegationTaskRef,
) -> str:
  return str(logical_task.delegation_id or "").strip()


def _ordinary_admitted_task_factory(
  *,
  operation: Any,
  execution_snapshot: Any,
  capability_bindings: tuple[Any, ...],
  tool_grant: Any,
  tool_routes: tuple[AdmittedToolRoute, ...],
  model_bind: CapabilityBind,
  result_requirement: ResultRequirement,
  objective: JsonValue,
  parent_session: Any | None,
  inputs: tuple[AdmittedInputBinding, ...] = (),
  attempt_number: int = 1,
  resume_of_task_id: str | None = None,
  logical_task_override: OrdinaryDelegationTaskRef | None = None,
) -> Callable[[Any], AdmittedTask]:
  model_bind_digest = sha256_digest(model_bind)
  capability_binding_digest = sha256_digest([
    binding.model_dump(mode="json")
    for binding in capability_bindings
  ])

  def _factory(entry: Any) -> AdmittedTask:
    task_id = str(getattr(entry, "task_id", "") or "").strip()
    if not task_id:
      raise ValueError("ordinary task admission requires a reserved task ID")
    attempt = AttemptRef(
      attempt_number=attempt_number,
      attempt_id=f"{task_id}:attempt:{attempt_number}",
      physical_task_id=task_id,
      resume_of_task_id=resume_of_task_id,
    )
    logical_task = logical_task_override or OrdinaryDelegationTaskRef(
      delegation_id=task_id,
      operation=operation.operation,
    )
    workspace = WorkspaceGrant(
      workspace_id=(
        f"session:{getattr(parent_session, 'session_id', '')}"
        if str(getattr(parent_session, "session_id", "") or "").strip()
        else f"delegation:{task_id}"
      ),
      scope=operation.workspace_scope,
    )
    payload = {
      "schema_version": "1.2",
      "admitted_task_id": f"admitted:{task_id}",
      "logical_task": logical_task.model_dump(mode="json"),
      "attempt": attempt.model_dump(mode="json"),
      "objective": objective,
      "workflow_identity": None,
      "execution_disposition": ExecuteTaskDisposition().model_dump(
        mode="json"
      ),
      "execution_snapshot": execution_snapshot.model_dump(mode="json"),
      "operation": operation.model_dump(mode="json"),
      "inputs": [binding.model_dump(mode="json") for binding in inputs],
      "capability_bindings": [
        binding.model_dump(mode="json")
        for binding in capability_bindings
      ],
      "tool_grant": tool_grant.model_dump(mode="json"),
      "tool_routes": [
        route.model_dump(mode="json") for route in tool_routes
      ],
      "content_read_grants": [],
      "workspace_grant": workspace.model_dump(mode="json"),
      "model_bind": model_bind.model_dump(mode="json"),
      "result_requirement": result_requirement.model_dump(mode="json"),
      "outcome_policy": _ordinary_outcome_policy().model_dump(mode="json"),
      "admitted_plan_digest": None,
      "model_bind_digest": model_bind_digest,
      "capability_binding_digest": capability_binding_digest,
      "tool_grant_digest": tool_grant.digest,
    }
    seal_admitted_task_payload(payload)
    return AdmittedTask.model_validate(payload)

  return _factory


def _prior_result_evidence(
  lineage: list[ChildRunSegment],
) -> SubAgentResultEvidence:
  evidence_parts: list[SubAgentResultEvidence] = []
  for segment in lineage:
    evidence = collect_sub_agent_result_evidence(
      segment.entries,
      durable=True,
    )
    has_interrupted_event = any(
      log_entry.event.get("type") == "interrupted"
      for log_entry in segment.entries
    )
    warning_parts = evidence.warning_parts
    if segment.completion is None and not has_interrupted_event:
      warning_parts = (
        *warning_parts,
        f"Prior child task {segment.task_id} was interrupted",
      )
    if warning_parts != evidence.warning_parts:
      evidence = SubAgentResultEvidence(
        usage=evidence.usage,
        tools_used=evidence.tools_used,
        fms_results=evidence.fms_results,
        artifact_events=evidence.artifact_events,
        warning_parts=warning_parts,
        gap=evidence.gap,
        observed_sources=evidence.observed_sources,
      )
    evidence_parts.append(evidence)
  return merge_sub_agent_result_evidence(*evidence_parts)


async def _finalize_resume_abandoned(
  *,
  runner: Any,
  entry: Any,
  code: str,
  message: str,
  evidence: SubAgentResultEvidence | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
  retained = evidence or SubAgentResultEvidence.empty()
  admitted = getattr(entry, "admitted_task", None)
  if not isinstance(admitted, AdmittedTask):
    return None, {
      "code": "invalid_task_metadata",
      "message": (
        f"Task {entry.task_id} cannot settle resume abandonment without its "
        "exact admitted task"
      ),
    }
  task_result = terminal_task_result(
    admitted,
    status="failed",
    reason=f"resume_abandoned:{code}: {message}",
    tools_used=retained.tools_used,
    usage=UsageObservation(**retained.usage),
  )
  result_payload = task_result.model_dump(mode="json")

  async def _owned_finalizer() -> tuple[Any | None, dict[str, Any] | None]:
    async def _reconcile_durable_completion() -> tuple[Any | None, dict[str, Any] | None] | None:
      durable_entry = await runner._lookup_task_in_log(entry.task_id)
      if durable_entry is None or durable_entry.state == TaskState.INTERRUPTED:
        return None
      durable_result = (
        dict(durable_entry.result)
        if isinstance(durable_entry.result, dict)
        else None
      )
      durable_error = (
        dict(durable_entry.error)
        if isinstance(durable_entry.error, dict)
        else None
      )
      entry.completion_persistence_state = "committed"
      entry.completion_persistence_error = None
      runner._task_registry.finalize_interrupted(
        entry.task_id,
        durable_entry.state,
        result=durable_result,
        error=durable_error,
      )
      try:
        canonical = TaskResult.model_validate(durable_result)
      except Exception:
        canonical = None
      if canonical == task_result:
        entry.task_result = canonical
        return canonical, None
      return None, {
        "code": "resume_state_changed",
        "message": (
          f"Task {entry.task_id} was durably finalized before resume "
          "abandonment could be committed"
        ),
      }

    async with entry.finalization_lock:
      if entry.state != TaskState.INTERRUPTED:
        existing = getattr(entry, "task_result", None)
        if existing == task_result:
          return existing, None
        return None, {
          "code": "resume_state_changed",
          "message": (
            f"Task {entry.task_id} changed state before resume abandonment "
            "could be finalized"
          ),
        }

      if entry.completion_persistence_state == "uncertain":
        reconciled = await _reconcile_durable_completion()
        if reconciled is not None:
          return reconciled

      entry.result = result_payload
      entry.task_result = task_result
      entry.error = None
      entry.completion_persistence_state = "in_flight"
      try:
        await runner._append_task_completed_event(
          entry,
          TaskState.FAILED,
          result=result_payload,
          error=None,
        )
      except BaseException as exc:
        entry.completion_persistence_state = "uncertain"
        entry.completion_persistence_error = (
          f"{type(exc).__name__}: {str(exc).strip() or type(exc).__name__}"
        )
        try:
          reconciled = await _reconcile_durable_completion()
        except asyncio.CancelledError:
          raise
        except BaseException:
          runner._writer_lease_poisoned = True
          raise exc
        if reconciled is not None:
          return reconciled
        raise
      entry.completion_persistence_state = "committed"
      entry.completion_persistence_error = None
      runner._task_registry.finalize_interrupted(
        entry.task_id,
        TaskState.FAILED,
        result=result_payload,
      )
    return task_result, None

  finalizer_task = asyncio.create_task(
    _owned_finalizer(),
    name=f"{entry.task_id}:resume-abandoned-finalize",
  )
  track_finalizer = getattr(
    runner,
    "_track_background_initialization",
    None,
  )
  if callable(track_finalizer):
    track_finalizer(finalizer_task)
  try:
    return await asyncio.shield(finalizer_task)
  except asyncio.CancelledError:
    timeout_getter: Callable[[], float] | None = getattr(
      runner,
      "_background_completion_persist_timeout",
      None,
    )
    timeout = (
      float(timeout_getter())
      if callable(timeout_getter)
      else 5.0
    )
    try:
      done, _pending = await asyncio.wait(
        {finalizer_task},
        timeout=max(0.0, timeout),
      )
    except asyncio.CancelledError:
      done = set()
    if finalizer_task in done and not finalizer_task.cancelled():
      try:
        finalizer_task.result()
      except BaseException:
        log.exception(
          "Resume-abandoned finalization failed after caller cancellation "
          "for task %s",
          entry.task_id,
        )
    elif not finalizer_task.done():
      entry.completion_persistence_state = "uncertain"
      entry.completion_persistence_error = (
        "resume-abandoned finalization continued after caller cancellation"
      )
    raise


def _install_child_code_execution(
  handlers: dict[str, Any],
  *,
  admitted_local_tools: Collection[str],
  session: GatewaySession | None,
  factory: Callable[[GatewaySession], CodeExecutionBundle | None] | None,
) -> CodeExecutionBundle | None:
  if factory is None:
    return None
  # The factory owns these routes even when this placement has no bundle.
  # Inherited session closures are not an alternative child interpreter.
  for name in _CODE_EXECUTION_TOOL_NAMES:
    handlers.pop(name, None)
  bundle = factory(session) if session is not None else None
  if bundle is not None:
    handlers.update({
      name: handler
      for name, handler in bundle.handlers.items()
      if name in admitted_local_tools
    })
  return bundle


async def _dispose_child_code_execution(session: GatewaySession | None) -> None:
  if session is None:
    return
  cleanup_task = asyncio.create_task(
    cleanup_code_execution(session),
    name=f"sub-agent-session:{session.session_id}:cleanup",
  )
  try:
    await asyncio.shield(cleanup_task)
  except asyncio.CancelledError:
    await drain_owned_lifecycle_task(cleanup_task)
    raise


def make_run_agent_handler(
  runner_ref: list[Any],
  *,
  parent_session: GatewaySession | None = None,
  skill_loader: SkillLoader | None = None,
  operation_catalog: AgentOperationCatalog | None = None,
  mcp_client: Any,
  needs_approval: Callable[..., bool] | None = None,
  interceptors: Sequence[ToolInterceptor] | None = None,
  mcp_session_inject_servers: AbstractSet[str] | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  user_id: str | None = None,
  user_email: str | None = None,
  trusted_research_file_id: int | None = None,
  parent_user_id: str | None = None,
  parent_user_email: str | None = None,
  credentials_resolver_active: bool = False,
  local_tool_handlers: dict[str, Any] | None = None,
  artifact_tools_installer: Callable[..., list[dict[str, Any]]] | None = None,
  declared_local_tool_definitions: (
    Callable[[frozenset[str]], Sequence[Mapping[str, Any]]] | None
  ) = None,
  fms_rebinder: Callable[[dict[str, Any], int], None] | None = None,
  workflow_handle_membership: (
    Callable[[ParentContext, frozenset[str]], Awaitable[frozenset[str]]] | None
  ) = None,
  excluded_tools: set[str] | None = None,
  default_max_tokens: int = 64000,
  default_cost_observation_threshold_usd: float | None = (
    DEFAULT_COST_OBSERVATION_THRESHOLD_USD
  ),
  anonymous_system_prompt_template: str = _DEFAULT_SYSTEM_PROMPT_TEMPLATE,
  on_sub_event: Callable[[dict[str, Any], str, int], None] | None = None,
  on_before_background: Callable[[str | None], None] | None = None,
  on_background_complete: Callable[[Any], Awaitable[None]] | None = None,
  capability_execution_resolver: CapabilityExecutionResolver,
  outputs_dir: Path | None = None,
  skill_state_store: SkillStateStore | None = None,
  approval_key_qualifier: Callable[[str, dict[str, Any]], str] | None = None,
  tool_registration_catalog: Any | None = None,
  tool_policy_implementations: Any | None = None,
  redaction_context_factory: Callable[..., Any] | None = None,
  code_execution_factory: (
    Callable[[GatewaySession], CodeExecutionBundle | None] | None
  ) = None,
  code_execution_input_preparation_factory: (
    Callable[[CodeExecutionBundle | None], Callable[..., Any] | None] | None
  ) = None,
  code_execution_binder: (
    Callable[[CodeExecutionBundle, ToolDispatcher], None] | None
  ) = None,
  commercial_work_start: Any | None = None,
  commercial_irreversible_recheck: Callable[[Any], None] | None = None,
  commercial_mcp_servers: frozenset[str] | None = None,
  operation_mcp_activator: (
    Callable[
      [SkillProfile | ResolvedOperationRuntime],
      RunAgentMcpActivationError | None,
    ]
    | None
  ) = None,
  tool_definition_projector: (
    Callable[[dict[str, Any]], dict[str, Any] | None] | None
  ) = None,
) -> JobScopedHandler:
  """Build the local handler used by the `run_agent` tool.

  The returned handler resolves a full operation identity (or the registered
  generic ``explore`` operation), admits its exact model/tool authority, and
  delegates execution to ``AgentRunner.spawn_sub_agent()``.
  """
  if skill_loader is not None and operation_catalog is not None:
    raise ValueError(
      "make_run_agent_handler accepts either skill_loader or "
      "operation_catalog, not both"
    )
  if local_tool_handlers is not None:
    local_tool_handlers.setdefault(
      "get_agent_result_content",
      make_get_agent_result_content_handler(
        runner_ref, caller=lambda: current_parent_context(parent_session),
      ),
    )
  if trusted_research_file_id is not None:
    _require_research_file_id(trusted_research_file_id)
  skill_state_lock = asyncio.Lock()

  async def _handle_run_agent(
    tool_input: dict[str, Any], *, scope: JobScope, **kwargs: Any,
  ) -> tuple[Any | None, dict[str, Any] | None]:
    runner = scope.runner()
    parent_session = scope.session
    parent_dispatch_scope = (
      parent_session.dispatch_scope if parent_session is not None else None
    )
    parent_portfolio_id = (
      parent_dispatch_scope.get("portfolio_id")
      if parent_dispatch_scope is not None else None
    )
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
    root = scope.caller()
    # run_agent's arguments are model-authored external input (rule 1 of
    # docs/engineering/fail-closed-local-guards.md; rule 6's "malformed input
    # shape"): a malformed argument must not become a paid child. Each arm names
    # the field; only the unknown-field arm can derive the corrected call, the
    # other values are the agent's own to supply.
    if not isinstance(tool_input, dict):
      return None, {
        "code": "invalid_input",
        "message": "input must be an object",
        "next_call": None,
      }
    allowed_input_keys = frozenset({
      "operation",
      "objective",
      "ticker",
      "research_file_id",
      "cost_observation_threshold_usd",
      "background",
      "handles",
      "role",
      "effort",
    })
    unknown_input_keys = sorted(set(tool_input) - allowed_input_keys)
    if unknown_input_keys:
      return None, {
        "code": "invalid_input",
        "message": (
          "unknown run_agent input fields: "
          + ", ".join(unknown_input_keys)
        ),
        "next_call": {
          "tool_name": "run_agent",
          "arguments": {
            key: value
            for key, value in tool_input.items()
            if key in allowed_input_keys
          },
        },
      }
    tool_ctx = kwargs.get("tool_ctx")
    parent_turn_id = getattr(tool_ctx, "tool_call_id", None)

    task = tool_input.get("objective", "")
    if not task or not isinstance(task, str):
      return None, {
        "code": "invalid_input",
        "message": "objective is required",
        "next_call": None,
      }
    raw_research_file_id = tool_input.get("research_file_id")
    if raw_research_file_id is not None:
      try:
        raw_research_file_id = _optional_research_file_id(raw_research_file_id)
      except ValueError as exc:
        return None, {
          "code": "invalid_input",
          "message": str(exc),
          "next_call": None,
        }
      if raw_research_file_id is None or raw_research_file_id <= 0:
        return None, {
          "code": "invalid_input",
          "message": "research_file_id must be a positive integer",
          "next_call": None,
        }
    asserted_research_file_id = raw_research_file_id
    raw_context_ticker = tool_input.get("ticker")
    raw_operation = tool_input.get("operation")
    named_operation = raw_operation is not None
    role = tool_input.get("role")
    effort = tool_input.get("effort")
    for name, value in (("role", role), ("effort", effort)):
      if value is not None and not isinstance(value, str):
        return None, {
          "code": "invalid_input",
          "message": f"{name} must be a string",
          "next_call": None,
        }
    if effort is not None and effort not in {
      "none", "minimal", "low", "medium", "high", "xhigh", "max",
    }:
      return None, {
        "code": "invalid_input",
        "message": "effort must be an exact model reasoning effort",
        "next_call": None,
      }

    background = tool_input.get("background", True)
    if not isinstance(background, bool):
      return None, {
        "code": "invalid_input",
        "message": "background must be a boolean",
        "next_call": None,
      }

    raw_handles = tool_input.get("handles")
    if raw_handles is None:
      handles = ()
    elif not isinstance(raw_handles, list) or any(
      not is_canonical_handle_ref(handle) for handle in raw_handles
    ):
      return None, {
        "code": "invalid_input",
        "message": "'handles' must be a list of handle strings",
        "next_call": None,
      }
    else:
      handles = tuple(dict.fromkeys(raw_handles))

    call_index = int(kwargs.get("call_index", 0) or 0)
    ordinary_task_id = secrets.token_hex(16)

    profile: SkillProfile | None
    runtime_policy: OperationRuntimePolicy | None
    operation_activation_target: SkillProfile | ResolvedOperationRuntime
    mcp_tools_by_server: Mapping[str, Collection[str]]
    if operation_catalog is not None:
      try:
        canonical_operation = operation_catalog.resolve_operation(raw_operation)
        if type(canonical_operation) is not ResolvedOperationRuntime:
          raise TypeError(
            "operation catalog must return an exact ResolvedOperationRuntime"
          )
        operation = canonical_operation.snapshot
        runtime_policy = canonical_operation.policy
        operation_activation_target = canonical_operation
      except FileNotFoundError as exc:
        return None, {"code": "not_found", "message": str(exc)}
      except Exception as exc:
        return None, {"code": "invalid_operation", "message": str(exc)}
      profile = None
      operation_dispatch_tool_ids = frozenset()
      canonical_to_exposed_mcp_tool_ids = MappingProxyType({})
      mcp_tools_by_server = runtime_policy.mcp_tools_by_server
      operation_scope = runtime_policy.semantic_scope
      operation_persist_state = runtime_policy.persist_state
      operation_version = operation.operation.version
      configured_max_tokens = runtime_policy.max_tokens
    elif skill_loader is not None:
      try:
        resolved_operation = skill_loader.resolve_operation(raw_operation)
      except FileNotFoundError as exc:
        return None, {"code": "not_found", "message": str(exc)}
      except Exception as exc:
        return None, {"code": "invalid_operation", "message": str(exc)}
      operation = resolved_operation.snapshot
      profile = resolved_operation.methodology_profile
      runtime_policy = None
      operation_activation_target = profile
      operation_dispatch_tool_ids = operation_tool_ids(profile)
      canonical_to_exposed_mcp_tool_ids = MappingProxyType({})
      mcp_tools_by_server = profile.mcp_tools or {}
      operation_scope = profile.scope
      operation_persist_state = profile.persist_state
      operation_version = profile.version
      configured_max_tokens = profile.max_tokens
    elif raw_operation is not None:
      return None, {
        "code": "not_available",
        "message": "Registered operations are not available",
      }
    else:
      generic_profile = generic_explore_profile()
      resolved_operation = ResolvedAgentOperation(
        snapshot=compile_agent_operation(generic_profile),
        methodology_profile=generic_profile,
      )
      operation = resolved_operation.snapshot
      profile = resolved_operation.methodology_profile
      runtime_policy = None
      operation_activation_target = profile
      operation_dispatch_tool_ids = operation_tool_ids(profile)
      canonical_to_exposed_mcp_tool_ids = MappingProxyType({})
      mcp_tools_by_server = profile.mcp_tools or {}
      operation_scope = profile.scope
      operation_persist_state = profile.persist_state
      operation_version = profile.version
      configured_max_tokens = profile.max_tokens
    registered_runtime = runtime_policy is not None or named_operation
    agent_name = operation.operation.name
    ticker_required = "ticker" in operation.required_context
    research_file_required = (
      "research_file_id" in operation.required_context
    )
    trusted_research_origin_required = _operation_has_investment_claim_routes(
      mcp_tools_by_server,
      exact_tool_ids=(
        None
        if runtime_policy is not None
        else operation_dispatch_tool_ids
      ),
    )
    # Data integrity, not completeness: an operation whose routes write
    # investment claims writes them into the turn's verified research file
    # (``fms_rebinder`` rebinds the child's doors to it). With no verified file
    # there is no file the child may write, and an asserted id is not one.
    if (
      root.research_binding.research_file_id is None
      and trusted_research_origin_required
    ):
      return None, {
        "code": "required_context_missing",
        "message": (
          "operation requires a verified research_file_id from the active "
          "research turn"
        ),
        "next_call": None,
      }
    try:
      resolved_context_ticker = _resolve_context_ticker(
        typed_ticker=raw_context_ticker,
        verified_server_ticker=None,
        prose_fallback=(
          (lambda: _extract_ticker_from_task(task))
          if ticker_required
          else None
        ),
      )
    except ValueError as exc:
      return None, _ticker_refusal(
        "invalid_input", str(exc), tool_input, required=ticker_required,
      )
    # The operation's subject is its input: an absent or ambiguous ticker is a
    # malformed input shape (rule 6), and the child cannot choose one.
    if ticker_required and resolved_context_ticker is None:
      return None, _ticker_refusal(
        "required_context_missing",
        "operation requires one unambiguous typed ticker subject",
        tool_input,
        required=True,
      )
    # The grant is admitted over the operation's declared ceiling (canonical
    # route ids) and projected onto live dispatcher names after the input
    # checks, as the admitted bind is materialized after them.
    declared_route_ceiling = (
      runtime_policy.exact_tool_ids
      if runtime_policy is not None
      else frozenset(operation_dispatch_tool_ids)
    )
    declared_mcp_route_tool_names = MappingProxyType(
      {
        f"mcp__{server_id}__{tool_id}": tool_id
        for server_id, tool_ids in runtime_policy.mcp_tools_by_server.items()
        for tool_id in tool_ids
      }
      if runtime_policy is not None
      else {}
    )
    declared_private_mcp_routes = (
      (
        frozenset(declared_mcp_route_tool_names) & runtime_policy.exact_tool_ids
        if runtime_policy is not None
        else _operation_private_mcp_tool_ids(
          mcp_tools_by_server,
          exact_tool_ids=operation_dispatch_tool_ids,
        )
      )
      if registered_runtime
      else frozenset()
    )
    mcp_is_mcp_tool = getattr(mcp_client, "is_mcp_tool", None)

    def _route_registered(tool_id: str) -> bool:
      if (
        tool_id in (local_tool_handlers or {})
        or tool_id in _ARTIFACT_EMIT_TOOLS
        or tool_id in declared_private_mcp_routes
      ):
        return True
      try:
        return bool(callable(mcp_is_mcp_tool) and mcp_is_mcp_tool(tool_id))
      except Exception:
        return False

    context_research_file_id = root.research_binding.research_file_id
    required_ref_keys = tuple(
      key for key in operation.required_context
      if key not in _SUBJECT_CONTEXT_KEYS
    )
    context_refs = _turn_required_refs(runner, required_ref_keys)
    missing_keys = [key for key in required_ref_keys if key not in context_refs]
    if missing_keys:
      # Henry's ruling, 2026-09-29: the bag is this turn's prior results only;
      # no latest lookup. The refs are the child's operands, the exact artifact
      # versions it builds on (rule 9: a computation needs its operands); a
      # child spawned without them would bind whatever is latest. The next call
      # is the parent's own read of the first missing ref; a retry names
      # whatever is still missing.
      return None, {
        "code": "required_context_missing",
        "message": "operation requires prior tool results in this turn for: " + ", ".join(missing_keys),
        "missing_keys": missing_keys,
        "next_call": _required_context_next_call(
          missing_keys[0],
          research_file_id=context_research_file_id,
          ticker=resolved_context_ticker,
        ),
      }
    # Each binding is built under its input name, so a construction failure
    # names the clause that failed. The values were already coerced above (the
    # ticker normalized, the research file verified, each ref taken from this
    # turn's results); what remains is the session's ownership or a ref value
    # that is not canonical JSON, neither of which a repeat call changes.
    admission_plan: list[tuple[str, Callable[[], AdmittedInputBinding]]] = []
    if resolved_context_ticker is not None:
      admission_plan.append((
        "ticker",
        partial(
          _ticker_admitted_input,
          resolved_context_ticker,
          invocation_id=ordinary_task_id,
          parent_session=parent_session,
        ),
      ))
    if (
      (research_file_required or trusted_research_origin_required)
      and root.research_binding.research_file_id is not None
      and context_research_file_id is not None
    ):
      admission_plan.append((
        "research_file_id",
        partial(
          _research_file_id_admitted_input,
          context_research_file_id,
          invocation_id=ordinary_task_id,
          parent_session=parent_session,
        ),
      ))
    admission_plan.extend(
      (
        key,
        partial(
          _ref_admitted_input,
          key,
          context_refs[key],
          invocation_id=ordinary_task_id,
          parent_session=parent_session,
        ),
      )
      for key in required_ref_keys
    )
    admitted_bindings: list[AdmittedInputBinding] = []
    for field_name, build_binding in admission_plan:
      try:
        admitted_bindings.append(build_binding())
      except ValueError as exc:
        return None, {
          "code": "invalid_context",
          "message": f"{field_name}: {exc}",
          "details": {"field": field_name, "reason": str(exc)},
          "next_call": None,
        }
    admitted_inputs = tuple(admitted_bindings)
    skill_run_id: str | None = None
    durable_skill_event_transport: tuple[
      DurableSkillEventAppender,
      DurableSkillEventConfirmer,
    ] | None = None
    if registered_runtime:
      if getattr(runner, "_agent_session_log", None) is None:
        return None, {
          "code": "durable_session_log_required",
          "message": (
            "Named operations require a durable session log; "
            "no child was registered or started"
          ),
        }
      durable_skill_event_transport = (
        _durable_skill_event_transport(runner)
      )
      if durable_skill_event_transport is None:
        return None, {
          "code": "durable_skill_event_transport_unavailable",
          "message": (
            "Named operations require durable methodology lifecycle append and "
            "confirmation support; no child was registered or started"
          ),
        }
      skill_run_id = secrets.token_hex(16)

    try:
      cost_observation_threshold_usd = resolve_cost_observation_threshold_usd(
        call_threshold_usd=tool_input.get("cost_observation_threshold_usd"),
        configured_default_threshold_usd=(
          default_cost_observation_threshold_usd
        ),
      )
    except CostObservationThresholdError as exc:
      return None, {
        "code": exc.code,
        "message": str(exc),
        "details": exc.receipt(),
      }

    admitted_result_requirement = ResultRequirement(
      outcome=OutcomeRequirement(),
    )


    if registered_runtime and operation_mcp_activator is not None:
      try:
        activation_error = operation_mcp_activator(
          operation_activation_target
        )
      except Exception as exc:
        log.exception(
          "Declared MCP activation failed for named operation %s",
          agent_name,
        )
        return None, {
          "code": "mcp_activation_failed",
          "message": (
            f"Declared MCP activation failed for named operation "
            f"'{agent_name}': {exc}"
          ),
        }
      if activation_error is not None:
        if not isinstance(activation_error, dict):
          return None, {
            "code": "mcp_activation_failed",
            "message": (
              f"Declared MCP activation returned an invalid result for "
              f"named operation '{agent_name}'"
            ),
          }
        return None, dict(activation_error)

    # Integrity and resource: the projection raises when an MCP alias would
    # impersonate a live local handler or the live route registry fails; either
    # way there is no route the child could run on.
    if runtime_policy is not None:
      try:
        (
          operation_dispatch_tool_ids,
          canonical_to_exposed_mcp_tool_ids,
        ) = _runtime_policy_dispatch_projection(
          runtime_policy,
          mcp_client=mcp_client,
          live_local_tool_ids=(
            set(local_tool_handlers or {}) | set(_ARTIFACT_EMIT_TOOLS)
          ),
        )
      except ValueError as exc:
        return None, {
          "code": "operation_unavailable",
          "message": str(exc),
          "next_call": None,
        }

    previous_state: dict[str, Any] | None = None
    methodology_state_instructions: str | None = None
    if operation_persist_state and skill_state_store is not None:
      try:
        previous_state = skill_state_store.get(agent_name)
      except Exception:
        log.warning(
          "Failed to load persisted state for skill %s",
          agent_name,
          exc_info=True,
        )
        previous_state = {}
      methodology_state_instructions = _skill_state_prompt(
        agent_name,
        previous_state,
      )
    effective_max_tokens = (
      configured_max_tokens
      if configured_max_tokens is not None
      else default_max_tokens
    )
    operation_private_mcp_tool_ids = (
      (
        frozenset(
          exposed_tool_id
          for canonical_id, exposed_tool_id in (
            canonical_to_exposed_mcp_tool_ids.items()
          )
          if canonical_id in runtime_policy.exact_tool_ids
        )
        if runtime_policy is not None
        else _operation_private_mcp_tool_ids(
          mcp_tools_by_server,
          exact_tool_ids=operation_dispatch_tool_ids,
        )
      )
      if registered_runtime
      else frozenset()
    )

    admitted_task_ref: list[AdmittedTask | None] = [None]

    def _admitted_context_ticker() -> str | None:
      admitted_task = admitted_task_ref[0]
      if admitted_task is None:
        raise RuntimeError("admitted task is unavailable before dispatch")
      return _ticker_from_admitted_inputs(
        admitted_task.inputs,
        required=ticker_required,
        owner_invocation_id=_ordinary_logical_invocation_owner(
          _ordinary_logical_task(admitted_task)
        ),
      )

    def _admitted_context_research_file_id() -> int | None:
      admitted_task = admitted_task_ref[0]
      if admitted_task is None:
        raise RuntimeError("admitted task is unavailable before dispatch")
      return _research_file_id_from_admitted_inputs(
        admitted_task.inputs,
        required=trusted_research_origin_required,
        owner_invocation_id=_ordinary_logical_invocation_owner(
          _ordinary_logical_task(admitted_task)
        ),
      ) or context_research_file_id

    candidate_tools = frozenset(operation_dispatch_tool_ids)
    sub_local = {
      name: handler
      for name, handler in (local_tool_handlers or {}).items()
      if name in candidate_tools
    }
    effective_parent_user_id = (
      parent_user_id
      or user_id
      or getattr(parent_session, "user_id", None)
    )
    effective_parent_user_email = (
      parent_user_email
      or user_email
      or getattr(parent_session, "user_email", None)
    )

    skill_event_emitter: SkillRunEventEmitter | None = None

    def _emit_parent_event(event: dict[str, Any]) -> None:
      if skill_event_emitter is not None:
        skill_event_emitter.emit_parent_event(event)

    async def _emit_skill_run_started() -> None:
      if skill_event_emitter is not None:
        await skill_event_emitter.emit_started()

    async def _emit_skill_result_captured(
      result: Any | None,
      error: dict[str, Any] | None,
    ) -> None:
      if skill_event_emitter is not None:
        await skill_event_emitter.emit_result_captured(result, error)

    async def _persist_skill_state(result: Any | None, error: dict[str, Any] | None) -> None:
      await _sub_agent_skill_state.persist_skill_state(
        result,
        error,
        agent_name=agent_name,
        profile=profile,
        persist_state=operation_persist_state,
        operation_version=operation_version,
        skill_state_store=skill_state_store,
        skill_state_lock=skill_state_lock,
        effective_model=effective_model,
        extract_state_update_fn=extract_state_update,
        result_response_text_fn=_result_response_text,
        logger=log,
      )

    extra_tool_definitions: list[dict[str, Any]] = []
    if skill_run_id and agent_name and artifact_tools_installer is not None:
      extra_tool_definitions = artifact_tools_installer(
        excluded_tools=set(_ARTIFACT_EMIT_TOOLS - candidate_tools),
        sub_local=sub_local,
        profile=profile,
        skill_name=agent_name,
        semantic_scope=operation_scope,
        skill_run_id=skill_run_id,
        context_ticker=_admitted_context_ticker,
        context_research_file_id=_admitted_context_research_file_id,
        parent_session=parent_session,
        fallback_user_id=effective_parent_user_id,
        emit_parent_event=_emit_parent_event,
      )

    if registered_runtime:
      extra_tool_definitions.extend(_operation_private_mcp_tool_definitions(
        profile=profile,
        mcp_tools_by_server=mcp_tools_by_server,
        mcp_client=mcp_client,
        exact_tool_ids=operation_private_mcp_tool_ids,
        tool_definition_projector=tool_definition_projector,
      ))
    if declared_local_tool_definitions is not None:
      already = {
        str(definition.get("name") or "")
        for definition in extra_tool_definitions
      }
      requested = frozenset(
        tool_id
        for tool_id in operation_dispatch_tool_ids
        if tool_id in sub_local and tool_id not in already
      )
      if requested:
        seen = set(already)
        for definition in declared_local_tool_definitions(requested):
          if not isinstance(definition, dict):
            raise TypeError(
              "declared local tool definitions must be objects"
            )
          name = str(definition.get("name") or "")
          if name not in requested or name in seen:
            continue
          seen.add(name)
          extra_tool_definitions.append(definition)

    candidate_definitions_getter = _child_tool_definitions_getter(
      runner=runner,
      mcp_client=mcp_client,
      excluded_tools=set(),
      extra_tool_definitions=extra_tool_definitions,
      local_tool_handlers=sub_local,
      granted_tools=candidate_tools,
    )
    candidate_definitions = (
      candidate_definitions_getter()
      if candidate_definitions_getter is not None
      else []
    )

    def _resolve_tool_authority(
      canonical_tool_ids: frozenset[str],
    ) -> ResolvedAuthority | OperationUnavailable:
      return admit_operation_tools(
        operation,
        grant_id=f"grant:{ordinary_task_id}",
        operation_tool_ids=(
          canonical_to_exposed_mcp_tool_ids.get(tool_id, tool_id)
          for tool_id in canonical_tool_ids
        ),
        definitions=candidate_definitions,
        local_tool_handlers=sub_local,
        mcp_client=mcp_client,
        identity=execution_identity_from_session(parent_session),
      )

    durable_log = getattr(runner, "_agent_session_log", None)
    outputs: frozenset[str] = frozenset()
    if handles and durable_log is not None:
      assert root.job_id is not None, "the executing job must have an identity"
      outputs = await job_output_handles(durable_log, job_id=root.job_id)
    spill_sink = getattr(runner, "_spill_dir_provider", None)
    spill_outputs: set[str] = set()
    for handle in handles:
      if not handle.startswith(SPILL_REF_PREFIX):
        continue
      try:
        assert root.job_id is not None, "the executing job must have an identity"
        spill_outputs.update(await job_spill_output_handles(
          spill_sink, handle, job_id=root.job_id, durable_log=durable_log,
        ))
      except SpillReadError:
        continue
    outputs = outputs | spill_outputs
    if handles and workflow_handle_membership is not None:
      outputs = outputs | await workflow_handle_membership(root, frozenset(handles))

    spawn_resolver = cast(
      CapabilityExecutionResolver, runner.capability_execution.resolver,
    )
    try:
      admission = admit(
        SpawnRequest(
          operation=agent_name,
          task=task,
          handles=handles,
          role=role,
          effort=effort,
          binding=AdmittedResearchBinding(
            research_file_id=asserted_research_file_id,
            ticker=resolved_context_ticker,
          ),
        ),
        root,
        owners=AdmissionOwners(
          operation=OperationRegistration(
            name=agent_name,
            execution_class=operation.execution_class,
            route_ceiling=declared_route_ceiling,
            operation_routes=(
              declared_private_mcp_routes
              | (declared_route_ceiling & _ARTIFACT_EMIT_TOOLS)
              if registered_runtime
              else frozenset()
            ),
            route_tool_names=declared_mcp_route_tool_names,
            children=operation.children,
            role=operation.role,
            effort=operation.effort,
          ),
          capability_execution_resolver=spawn_resolver,
          session=parent_session,
          route_registered=_route_registered,
          resolve_tool_authority=_resolve_tool_authority,
          # The spawn core's executor is this process's runner.
          placement="in_process",
          outputs=outputs,
        ),
      )
    except CapabilityResolutionError as exc:
      return None, _capability_resolution_error(exc)
    # The existing semantic owner returns its own unavailable authority.
    # Admission neither wraps that verdict nor publishes a wider grant.
    if isinstance(admission, OperationUnavailable):
      return None, {
        "code": "operation_unavailable",
        "message": admission.detail,
        "next_call": None,
      }
    if isinstance(admission, Refusal):
      # Data integrity: the child is bound to the turn's verified file, so a
      # different asserted id means writes to a file the child would not
      # touch; the next call carries the verified id.
      if admission.reason == "binding_not_granted":
        return None, {
          "code": "context_research_file_id_mismatch",
          "message": (
            "research_file_id must match the verified active research turn "
            f"({root.research_binding.research_file_id})"
          ),
          "next_call": {
            "tool_name": "run_agent",
            "arguments": {
              **tool_input,
              "research_file_id": root.research_binding.research_file_id,
            },
          },
        }
      error = {
        "code": admission.reason,
        "message": admission.detail,
        "owner": admission.owner,
        "next_call": admission.next_call,
      }
      if admission.classification is not None:
        error["classification"] = admission.classification
      return None, error
    admitted_context = admission.wire
    operation_authority = admission.tool_authority
    context_research_file_id = admitted_context.research_binding.research_file_id
    try:
      execution = spawn_resolver.materialize_bind(
        admitted_context.capability_bind
      )
    except CapabilityResolutionError as exc:
      return None, _capability_resolution_error(exc)
    try:
      require_commercial_child_provider(
        commercial_work_start,
        execution.bind.provider,
      )
    except CommercialWorkStartError as exc:
      return None, {"code": exc.code, "message": str(exc)}
    if (
      fms_rebinder is not None
      and context_research_file_id is not None
      and context_research_file_id > 0
    ):
      fms_rebinder(sub_local, context_research_file_id)
    effective_model = execution.bind.upstream_model
    admitted_dispatch_tools = granted_tool_ids(operation_authority)
    admitted_mcp_scope = derive_dispatcher_allowlist(operation_authority)
    candidate_local_names = set(sub_local)
    sub_local = {
      name: handler
      for name, handler in sub_local.items()
      if name in admitted_dispatch_tools
    }
    # The child runner's vocabulary is exclusion: every name it could meet
    # that the grant does not hold.
    child_excluded = (
      set(local_tool_handlers or {})
      | candidate_local_names
      | {str(definition.get("name") or "") for definition in candidate_definitions}
    ) - admitted_dispatch_tools
    admitted_provider_definitions = provider_tool_definitions_for_grant(
      candidate_definitions,
      grant=operation_authority.grant,
    )
    execution_snapshot = build_agent_execution_snapshot(
      operation=operation,
      result_instructions=render_result_instructions(
        admitted_result_requirement
      ),
      persisted_methodology_state=previous_state,
      methodology_state_instructions=methodology_state_instructions,
      max_tokens=effective_max_tokens,
      cost_observation_threshold_usd=cost_observation_threshold_usd,
      provider_tool_definitions=admitted_provider_definitions,
      max_resume_chain_depth=getattr(runner, "_max_resume_chain_depth", 3),
      admitted_context=admitted_context,
    )
    system_prompt = execution_snapshot.system_prompt
    admitted_task_factory = _ordinary_admitted_task_factory(
      operation=operation,
      execution_snapshot=execution_snapshot,
      capability_bindings=operation_authority.bindings,
      tool_grant=operation_authority.grant,
      tool_routes=admitted_tool_routes(operation_authority),
      model_bind=execution.bind,
      result_requirement=admitted_result_requirement,
      objective=task,
      parent_session=parent_session,
      inputs=admitted_inputs,
    )
    admitted_task = admitted_task_factory(
      SimpleNamespace(task_id=ordinary_task_id)
    )
    admitted_task_ref[0] = admitted_task
    context_ticker = _admitted_context_ticker()
    child_research_file_id = _admitted_context_research_file_id()
    # Both paths render the declared refs from admitted server-held values.
    child_refs = {
      binding.name: binding.context.content
      for binding in admitted_task.inputs
      if binding.name in required_ref_keys
      and isinstance(binding.context, InlineExactContextView)
    }
    child_message = context_text_with_run_binding(
      "\n".join((task, *declared_ref_lines(operation, child_refs))),
      ticker=context_ticker,
      research_file_id=child_research_file_id,
    )
    result_provenance = TaskResultProvenance(
      admitted_task_digest=admitted_task.admitted_task_digest,
      model_bind_digest=admitted_task.model_bind_digest,
      capability_binding_digest=admitted_task.capability_binding_digest,
      tool_grant_digest=admitted_task.tool_grant_digest,
    )

    sub_log = EventLog()
    skill_event_emitter = (
      SkillRunEventEmitter(
        skill_run_id=skill_run_id,
        profile=profile,
        skill_name=agent_name,
        semantic_scope=(
          None if operation_scope == "global" else operation_scope
        ),
        context_ticker=context_ticker,
        portfolio_id=parent_portfolio_id,
        event_log_getter=lambda: sub_log,
        tool_ctx=tool_ctx,
        durable_appender=durable_skill_event_transport[0],
        durable_confirmer=durable_skill_event_transport[1],
      )
      if (
        skill_run_id
        and agent_name
        and durable_skill_event_transport is not None
      )
      else None
    )
    effective_session_inject_servers = (
      _effective_mcp_session_inject_servers(
        admitted_mcp_scope=admitted_mcp_scope,
        configured_servers=(
          (
            runtime_policy.session_inject_servers
            if mcp_session_inject_servers is None
            else (
              runtime_policy.session_inject_servers
              & frozenset(mcp_session_inject_servers)
            )
          )
          if runtime_policy is not None
          else mcp_session_inject_servers
        ),
      )
    )
    async def _dispatch_sub_agent(_background_input: dict[str, Any], **background_kwargs: Any):
      background_call_index = int(background_kwargs.get("call_index", call_index) or 0)
      task_entry = background_kwargs.get("task_entry")

      def _record_sub_event(event: dict[str, Any], session_id: str) -> None:
        sub_log.append(event)
        if on_sub_event is None:
          return
        try:
          on_sub_event(event, session_id, background_call_index)
        except Exception:
          log.warning("External sub-agent event callback failed", exc_info=True)

      child_context = context_for_child(
        admission, job_id=admitted_task.attempt.physical_task_id,
      )
      sub_session = None
      if parent_session is not None:
        sub_session = child_gateway_session(
          parent_session,
          session_id=_derive_sub_agent_id(parent_session, background_call_index),
          tenant=admitted_context.tenant,
          auth_config=execution.auth_config,
          user_id=effective_parent_user_id or parent_session.user_id,
          user_email=effective_parent_user_email,
        )
        capture_job_context(sub_session, child_context)
      try:
        # Without a factory, callers retain their own handler semantics.
        child_local = dict(sub_local)
        child_spill_sink = (
          job_spill_sink(
            runner._spill_dir_provider, admitted_task.attempt.physical_task_id,
          )
          if runner._spill_dir_provider is not None
          else None
        )
        child_runner_ref: list[Any] = []
        rebind_job_handlers(child_local, JobScope(
          caller=lambda: current_parent_context(sub_session),
          spill_sink=lambda: child_spill_sink,
          runner=lambda: child_runner_ref[0],
          session=sub_session,
        ))
        code_execution = _install_child_code_execution(
          child_local,
          admitted_local_tools=sub_local,
          session=sub_session,
          factory=code_execution_factory,
        )
        child_input_preparation = (
          code_execution_input_preparation_factory(code_execution)
          if code_execution_input_preparation_factory is not None
          else None
        )
        sub_dispatcher = ToolDispatcher(
          capability_bind=execution.bind,
          mcp_client=mcp_client,
          local_tool_handlers=child_local,
          plan_validator=getattr(getattr(runner, "_dispatcher", None), "plan_validator", None),
          plan_review_renderer=getattr(getattr(runner, "_dispatcher", None), "plan_review_renderer", None),
          needs_approval=needs_approval,
          event_log=sub_log,
          interceptors=interceptors,
          session_id=getattr(runner, "_full_session_id", ""),
          should_avoid_permission_prompts=background,
          approval_key_qualifier=(
            code_execution.approval_qualifier
            if code_execution is not None
            else approval_key_qualifier
          ),
          tool_registration_catalog=tool_registration_catalog,
          tool_policy_implementations=tool_policy_implementations,
          input_preparation_context_factory=child_input_preparation,
          redaction_context_factory=redaction_context_factory,
          mcp_session_inject_servers=effective_session_inject_servers,
          mcp_meta_inject_servers=mcp_meta_inject_servers,
          # The admitted execution identity and exact approval route remain
          # host-owned; only the interpreter's session is child-local.
          identity=dispatch_identity(
            session=parent_session,
            execution=operation_authority.identity,
            context=child_context,
            attempt_lineage=None,
            user_id=effective_parent_user_id,
            credentials_resolver_active=credentials_resolver_active,
          ),
          role=require_inherited_role(parent_session),
          approval_route=admission.approval_route,
          run_context=_child_run_context(
            parent_session=parent_session,
            tool_ctx=tool_ctx,
            skill_run_id=skill_run_id,
            skill_name=agent_name,
            admitted_skill_execution_limits=SkillExecutionLimits(
              max_tokens=execution_snapshot.max_tokens,
            ),
            research_file_id=child_research_file_id,
            user_id=effective_parent_user_id,
            session_id=getattr(runner, "_full_session_id", ""),
            approval_policy=getattr(parent_session, "approval_policy", None),
          ),
          allowed_mcp_tools_by_server=admitted_mcp_scope,
          get_tool_definitions=_admitted_tool_definitions_getter(
            admitted_provider_definitions,
            unavailable_local_tools=sub_local.keys() - child_local.keys(),
          ),
          commercial_work_start=commercial_work_start,
          commercial_irreversible_recheck=commercial_irreversible_recheck,
          commercial_mcp_servers=commercial_mcp_servers,
        )
        if code_execution is not None and code_execution_binder is not None:
          code_execution_binder(code_execution, sub_dispatcher)
        try:
          await _emit_skill_run_started()
        except DurableSkillEventPersistenceError:
          return None, _durable_skill_event_persistence_error()
        return await runner.spawn_sub_agent(
          child_message,
          capability_execution=execution,
          skill_name=agent_name,
          logical_task=admitted_task.logical_task,
          attempt=admitted_task.attempt,
          result_requirement=admitted_result_requirement,
          result_provenance=result_provenance,
          system_prompt=system_prompt,
          dispatcher=sub_dispatcher,
          sub_session=sub_session,
          bind_job_runner=child_runner_ref.append,
          excluded_tools=child_excluded,
          max_tokens=execution_snapshot.max_tokens,
          cost_observation_threshold_usd=(
            execution_snapshot.cost_observation_threshold_usd
          ),
          skill_run_id=skill_run_id,
          call_index=background_call_index,
          parent_turn_id=parent_turn_id,
          task_entry=task_entry,
          on_sub_event=_record_sub_event,
          admitted_task=admitted_task,
        )
      finally:
        # Runner teardown owns a running child; this also covers construction
        # and durable-start failures before it can take ownership.
        await _dispose_child_code_execution(sub_session)

    if background:
      if outputs_dir is not None and agent_name:
        today = datetime.date.today().isoformat()
        stale_path = outputs_dir / agent_name / f"{today}.md"
        try:
          stale_path.unlink(missing_ok=True)
        except OSError as exc:
          return None, {
            "code": "file_cleanup_failed",
            "message": f"Failed to clean stale output {stale_path}: {exc}",
          }
      enriched_tool_input = dict(tool_input)
      if "resumable" in enriched_tool_input:
        raw_resumable = enriched_tool_input["resumable"]
        if not isinstance(raw_resumable, bool):
          return None, {
            "code": "invalid_input",
            "message": f"resumable must be a bool, got {type(raw_resumable).__name__}: {raw_resumable!r}",
          }
      else:
        enriched_tool_input["resumable"] = operation.resumable
      enriched_tool_input["result_requirement"] = (
        admitted_result_requirement.model_dump(mode="json")
      )
      enriched_tool_input["cost_observation_threshold_usd"] = (
        cost_observation_threshold_usd
      )
      enriched_tool_input["operation"] = operation.operation.model_dump(
        mode="json"
      )

      async def _on_background_complete(bg_task: Any) -> None:
        if on_background_complete is not None:
          await on_background_complete(bg_task)
        await _persist_skill_state(bg_task.result, bg_task.error)

      def _build_background_skill_result(
        _bg_task: Any,
        result: dict[str, Any] | None,
        error: dict[str, Any] | None,
      ) -> dict[str, Any] | None:
        if skill_event_emitter is None:
          raise RuntimeError("background skill lifecycle is not configured")
        return skill_event_emitter.build_result_captured_event(
          result,
          error,
        )

      result, error = await runner._register_background_task(
        tool_input=enriched_tool_input,
        handler=_dispatch_sub_agent,
        agent_name=agent_name,
        parent_turn_id=parent_turn_id,
        capability_bind_receipt=execution.bind.to_json(),
        admitted_task=admitted_task,
        task_id_override=ordinary_task_id,
        on_before_start=(
          partial(on_before_background, agent_name)
          if on_before_background is not None
          else None
        ),
        on_complete=_on_background_complete if (skill_run_id or on_background_complete is not None) else None,
        required_skill_lifecycle=(
          skill_event_emitter.required_lifecycle_metadata()
          if skill_event_emitter is not None
          else None
        ),
        required_skill_result_event_factory=(
          _build_background_skill_result
          if skill_event_emitter is not None
          else None
        ),
        required_skill_result_projector=(
          skill_event_emitter.project_result_captured
          if skill_event_emitter is not None
          else None
        ),
      )
      if error is None and isinstance(result, dict):
        result = {
          **result,
          "granted_tools": list(sorted(admitted_dispatch_tools)),
        }
      return result, error
    result, error = await _dispatch_sub_agent(tool_input, call_index=call_index)
    if (
      isinstance(error, dict)
      and error.get("code")
      == "durable_skill_event_persistence_failed"
    ):
      return result, error
    canonical_result: TaskResult | None = None
    if error is None:
      try:
        canonical_result = TaskResult.model_validate(result)
      except ValidationError as exc:
        # Our runner returned no TaskResult: a runner defect, raised to the
        # dispatcher (local_tool_failed). The started run still settles; its
        # terminal event records the defect as a failed child.
        defect = {
          "code": "invalid_task_result",
          "message": (
            "Sub-agent runner returned an invalid result: "
            f"{exc.errors()[0]['msg']}"
          ),
          "next_call": None,
        }
        await _emit_skill_result_captured(None, defect)
        await _persist_skill_state(None, defect)
        raise
    try:
      await _emit_skill_result_captured(result, error)
    except DurableSkillEventPersistenceError:
      return None, _durable_skill_event_persistence_error()
    await _persist_skill_state(result, error)
    if (
      canonical_result is not None
      and _foreground_completion_transport_ready(runner)
    ):
      try:
        result = await _publish_foreground_completion(
          runner=runner,
          result=canonical_result,
          admitted_task=admitted_task,
          agent_name=agent_name,
          capability_bind_receipt=execution.bind.to_json(),
          parent_turn_id=parent_turn_id,
          call_index=call_index,
        )
      except Exception as exc:
        log.exception("Foreground agent completion publication failed")
        status = canonical_result.execution.status
        reason = canonical_result.execution.terminal_reason
        if status != "succeeded":
          happened = f"{status} ({reason})" if reason else status
        else:
          happened = "completed"
        return None, {
          "code": "agent_completion_materialization_failed",
          "message": (
            f"Foreground agent {happened} but its exact parent-readable result "
            f"could not be published: {type(exc).__name__}"
          ),
        }
    return result, error

  def _build(scope: JobScope):
    async def _scoped_run_agent(tool_input: dict[str, Any], **kwargs: Any):
      return await _handle_run_agent(tool_input, scope=scope, **kwargs)
    return _scoped_run_agent

  return JobScopedHandler(
    _build,
    JobScope(
      caller=lambda: _spawn_root_context(
        parent_session,
        withheld_tools=frozenset(excluded_tools or ()),
        research_file_id=trusted_research_file_id,
      ),
      spill_sink=lambda: None,
      runner=lambda: runner_ref[0],
      session=parent_session,
    ),
  )


def _spawn_root_context(
  parent_session: GatewaySession | None,
  *,
  withheld_tools: frozenset[str],
  research_file_id: int | None,
) -> ParentContext:
  """Re-mint the spawning turn's root with this handler's tool scope.

  The root keeps the selection snapshot the turn edge captured on the session:
  the snapshot is taken once per turn, so this re-mint names it explicitly
  rather than dropping it. An admitted child's context is returned unchanged
  by ``root_context``."""

  captured: ParentContext | None = getattr(
    parent_session, "admitted_job_context", None
  )
  return root_context(
    parent_session,
    withheld_tools=withheld_tools,
    research_file_id=research_file_id,
    selection_snapshot=(
      captured.selection_snapshot if captured is not None else None
    ),
  )


def _admitted_resume_source_error(
  entry: TaskEntry | None,
  task_id: str,
) -> dict[str, Any] | None:
  if entry is None:
    return {"code": "not_found", "message": f"Unknown background task: {task_id}"}
  if (
    entry.completion_finalizer_detached
    or entry.completion_persistence_state in {"in_flight", "uncertain"}
  ):
    return {
      "code": "completion_pending",
      "message": f"Task {task_id} has an unresolved completion",
    }
  if entry.state not in {TaskState.AWAITING_APPROVAL, TaskState.INTERRUPTED}:
    return {"code": "not_resumable", "message": f"Task {task_id} is not parked"}
  if not isinstance(entry.admitted_task, AdmittedTask):
    return {
      "code": "invalid_task_metadata",
      "message": f"Task {task_id} has no canonical admitted task",
    }
  return None


async def _admitted_resume_source(runner: Any, task_id: str) -> TaskEntry | None:
  entry = runner._task_registry.get(task_id)
  if entry is None:
    entry = await runner._lookup_task_in_log(task_id)
    if entry is not None:
      entry = runner._task_registry.adopt_reconstructed(entry)
  return entry


async def reserve_admitted_successor(
  runner,
  *,
  parked_task_id: str,
) -> tuple[str | None, dict | None]:
  """Reserve physical lineage only; the workflow owner supplies admission."""
  entry = await _admitted_resume_source(runner, parked_task_id)
  error = _admitted_resume_source_error(entry, parked_task_id)
  if error is not None:
    return None, error
  depth = await runner._resume_chain_depth(parked_task_id)
  root_id = await runner._resume_root_task_id(parked_task_id)
  error = _admitted_resume_source_error(
    runner._task_registry.get(parked_task_id) or entry, parked_task_id,
  )
  if error is not None:
    return None, error
  successor_id = f"{root_id}_r{depth + 1}"
  assert entry is not None
  try:
    runner._task_registry.claim_resume_successor(
      "background",
      task_id=successor_id,
      original_task_id=parked_task_id,
      agent_name=entry.agent_name,
    )
  except ResumeSuccessorConflictError as exc:
    return None, {"code": "resume_successor_conflict", "message": str(exc)}
  return successor_id, None


class _ResumeHandler(JobScopedHandler):
  """Carry a host-only entry beside, never through, the model handler."""

  def __init__(
    self, build, scope, admitted_entry, approved_entry, operation_resume_bindings,
  ) -> None:
    super().__init__(build, scope)
    self._scope = scope
    self._admitted_entry = admitted_entry
    self._approved_entry = approved_entry
    self._operation_resume_bindings = operation_resume_bindings

  def for_job(self, scope: JobScope) -> "_ResumeHandler":
    return _ResumeHandler(
      self._build, scope, self._admitted_entry, self._approved_entry,
      self._operation_resume_bindings,
    )


@dataclass(frozen=True)
class ApprovedCallRef:
  """Exact durable original call selected by a trusted continuation host."""

  job_id: str
  tool_call_id: str
  nonce: str


async def resume_admitted_successor(
  runner,
  *,
  parked_task_id: str,
  successor: AdmittedTask,
  approved_call: ApprovedCallRef | None,
  on_complete: Callable[[TaskEntry], Awaitable[None] | None],
  on_park: Callable[[TaskEntry], Awaitable[None]] | None,
) -> tuple[TaskEntry | None, dict | None]:
  """Execute a trusted workflow-owner admission using configured resume wiring.

  Return ``(entry, None)`` whenever the successor started or already exists;
  ``(None, error)`` is returned only before any worker exists. ``CancelledError``
  leaves successor state unknown: initialization continues, and re-calling with
  the same successor returns its live entry. Other exceptions mean the successor
  did not start or was compensated to a terminal state.
  """
  handler = next(
    (
      value
      for value in runner._dispatcher.local_tool_handlers().values()
      if isinstance(value, _ResumeHandler)
    ),
    None,
  )
  if handler is None:
    return None, {
      "code": "not_available",
      "message": "Host successor entry requires configured resume wiring",
    }
  return await handler._admitted_entry(
    runner,
    parked_task_id=parked_task_id,
    successor=successor,
    on_complete=on_complete,
    on_park=on_park,
    approved_call=approved_call,
    caller=handler._scope.caller,
  )


async def continue_approved_child(
  runner,
  *,
  parked_task_id: str,
  approved_call: ApprovedCallRef,
  on_complete: Callable[[TaskEntry], Awaitable[None] | None],
  on_park: Callable[[TaskEntry], Awaitable[None]] | None,
) -> tuple[TaskEntry | None, dict | None]:
  """Continue a parked native call through host wiring, never model arguments.

  Return and cancellation semantics are those of ``resume_admitted_successor``.
  """
  handler = next((
    value for value in runner._dispatcher.local_tool_handlers().values()
    if isinstance(value, _ResumeHandler)
  ), None)
  if handler is None:
    return None, {
      "code": "not_available",
      "message": "Host successor entry requires configured resume wiring",
    }
  return await handler._approved_entry(
    runner, parked_task_id=parked_task_id, approved_call=approved_call,
    on_complete=on_complete,
    on_park=on_park, caller=handler._scope.caller,
  )


@dataclass(frozen=True)
class OperationResumeBinding:
  """Runtime-only operation wiring selected from the parked admission.

  A ``None`` usage aggregator gives the successor independent accounting.
  Settlement receives its canonical cumulative result after activity release.
  """

  local_handler_scopes: Mapping[
    str, Callable[[LocalToolHandler], LocalToolHandler]
  ]
  usage_aggregator: _UsageAggregator | None
  on_settled: Callable[[TaskEntry], Awaitable[None]] | None


def make_resume_handler(
  runner_ref: list[Any],
  *,
  operation_resume_bindings: Callable[[AdmittedTask], OperationResumeBinding | None] | None,
  parent_session: GatewaySession | None = None,
  caller: CallerContext | None = None,
  mcp_client: Any,
  needs_approval: Callable[..., bool] | None = None,
  needs_approval_resolver: NeedsApprovalResolver | None = None,
  interceptors: Sequence[ToolInterceptor] | None = None,
  mcp_session_inject_servers: AbstractSet[str] | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  user_id: str | None = None,
  credentials_resolver_active: bool = False,
  local_tool_handlers: dict[str, Any] | None = None,
  artifact_tools_installer: Callable[..., list[dict[str, Any]]] | None = None,
  fms_rebinder: Callable[[dict[str, Any], int], None] | None = None,
  excluded_tools_resolver: ExcludedToolsResolver | None = None,
  default_max_tokens: int = 64000,
  capability_execution_resolver: CapabilityExecutionResolver,
  tool_registration_catalog: Any | None = None,
  tool_policy_implementations: Any | None = None,
  redaction_context_factory: Callable[..., Any] | None = None,
  code_execution_factory: (
    Callable[[GatewaySession], CodeExecutionBundle | None] | None
  ) = None,
  code_execution_input_preparation_factory: (
    Callable[[CodeExecutionBundle | None], Callable[..., Any] | None] | None
  ) = None,
  code_execution_binder: (
    Callable[[CodeExecutionBundle, ToolDispatcher], None] | None
  ) = None,
  commercial_work_start: Any | None = None,
  commercial_irreversible_recheck: Callable[[Any], None] | None = None,
  commercial_mcp_servers: frozenset[str] | None = None,
  tool_definition_projector: (
    Callable[[dict[str, Any]], dict[str, Any] | None] | None
  ) = None,
) -> JobScopedHandler:
  if excluded_tools_resolver is None:
    raise ValueError("make_resume_handler requires excluded_tools_resolver")

  parent_dispatch_scope = (
    parent_session.dispatch_scope
    if parent_session is not None
    else None
  )
  parent_portfolio_id = (
    parent_dispatch_scope.get("portfolio_id")
    if parent_dispatch_scope is not None
    else None
  )

  def _current_caller() -> ParentContext:
    if parent_session is not None:
      return current_parent_context(parent_session)
    return root_context(
      None,
      withheld_tools=frozenset(excluded_tools_resolver()),
      research_file_id=None,
    )

  async def _execute_resume(
    runner: Any,
    tool_input: dict[str, Any],
    *,
    caller: CallerContext,
    supplied_successor: AdmittedTask | None = None,
    supplied_on_complete: Callable[[TaskEntry], Awaitable[None] | None] | None = None,
    supplied_on_park: Callable[[TaskEntry], Awaitable[None]] | None,
    approved_call: ApprovedCallRef | None = None,
    dispatch_kwargs: dict[str, Any],
  ):
    kwargs = dispatch_kwargs
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
    if supplied_successor is not None:
      existing = runner._task_registry.get(supplied_successor.attempt.physical_task_id)
      if (
        existing is not None
        and existing.original_task_id == tool_input.get("task_id")
        and existing.admitted_task == supplied_successor
      ):
        started = await runner._started_background_task(existing.task_id)
        if started is not None:
          return started, None
    root = caller()
    durable_log = getattr(runner, "_agent_session_log", None)
    if durable_log is None:
      return None, {"code": "not_available", "message": "Durable session log not configured"}

    raw_task_id = tool_input.get("task_id")
    if not isinstance(raw_task_id, str) or not raw_task_id.strip():
      return None, {"code": "invalid_input", "message": "task_id is required"}
    task_id = raw_task_id.strip()
    additional_context = tool_input.get("additional_context")
    if additional_context is not None and not isinstance(additional_context, str):
      return None, {"code": "invalid_input", "message": "additional_context must be a string"}
    forbidden_overrides = sorted(
      field_name
      for field_name in ("provider", "model", "effort")
      if field_name in tool_input
    )
    if forbidden_overrides:
      return None, {
        "code": "invalid_input",
        "message": (
          "resume uses the original capability bind; overrides are not allowed: "
          + ", ".join(forbidden_overrides)
        ),
      }

    live_entry = runner._task_registry.get(task_id)
    if live_entry is not None and (
      live_entry.completion_finalizer_detached
      or live_entry.completion_persistence_state
      in {"in_flight", "uncertain"}
    ):
      return None, {
        "code": "completion_pending",
        "message": (
          f"Task {task_id} has an unresolved completion; retry after its "
          "durable outcome settles"
        ),
      }

    try:
      source_task_is_current = await task_registration_is_current(
        durable_log,
        task_id,
      )
    except ResearchFileCurrentProjectionUnavailable:
      return None, {
        "code": "current_projection_unavailable",
        "message": "Current background task state is unavailable",
      }
    if not source_task_is_current:
      return None, {
        "code": "current_projection_unavailable",
        "message": f"Task {task_id} is not available in current history",
      }

    if supplied_successor is not None:
      entry = await _admitted_resume_source(runner, task_id)
    else:
      await runner._rebuild_task_registry_from_log()
      entry = runner._task_registry.get(task_id)
      if entry is None:
        reconstructed_entry = await runner._lookup_task_in_log(task_id)
        if reconstructed_entry is not None:
          if reconstructed_entry.state != TaskState.INTERRUPTED:
            return None, {
              "code": "not_interrupted",
              "message": f"Task {task_id} is not interrupted",
            }
          entry = runner._task_registry.adopt_reconstructed(
            reconstructed_entry
          )
    if entry is None:
      return None, {"code": "not_found", "message": f"Unknown background task: {task_id}"}

    def _source_state_error() -> dict[str, Any] | None:
      current = runner._task_registry.get(task_id) or entry
      if supplied_successor is not None:
        return _admitted_resume_source_error(current, task_id)
      assert current is not None
      if (
        current.completion_finalizer_detached
        or current.completion_persistence_state
        in {"in_flight", "uncertain"}
      ):
        return {
          "code": "completion_pending",
          "message": (
            f"Task {task_id} has an unresolved completion; retry after its "
            "durable outcome settles"
          ),
        }
      if current.state != TaskState.INTERRUPTED:
        return {
          "code": "not_interrupted",
          "message": f"Task {task_id} is not interrupted",
        }
      return None

    source_state_error = _source_state_error()
    if source_state_error is not None:
      return None, source_state_error
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    prior_evidence = SubAgentResultEvidence.empty()

    async def _abandon(
      code: str,
      message: str,
    ) -> tuple[Any | None, dict[str, Any] | None]:
      if supplied_successor is not None:
        return None, {"code": code, "message": message}
      return await _finalize_resume_abandoned(
        runner=runner,
        entry=entry,
        code=code,
        message=message,
        evidence=prior_evidence,
      )

    try:
      canonical_admission = getattr(entry, "admitted_task", None)
      if isinstance(canonical_admission, AdmittedTask):
        original_admitted_task = canonical_admission
      else:
        original_admitted_task = AdmittedTask.model_validate(
          metadata.get(ADMITTED_TASK_METADATA_KEY),
        )
    except Exception as exc:
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has no canonical admitted task: {exc}",
      )
    original_execution_snapshot = original_admitted_task.execution_snapshot
    if original_execution_snapshot is None:
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has no immutable execution snapshot",
      )
    original_bind = original_admitted_task.model_bind
    # AdmittedTask's validator requires a bind for every execution snapshot.
    assert original_bind is not None
    if (
      original_admitted_task.attempt.physical_task_id != task_id
      or original_admitted_task.operation.execution_class
      != original_bind.capability_id
    ):
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} admission does not match its physical/model bind",
      )
    operation_resume_binding = (
      operation_resume_bindings(original_admitted_task)
      if operation_resume_bindings is not None
      else None
    )
    successor_context = None
    reserved: TaskEntry | None = None
    if supplied_successor is not None:
      source_context = original_execution_snapshot.admitted_context
      successor_snapshot = supplied_successor.execution_snapshot
      successor_context = (
        successor_snapshot.admitted_context if successor_snapshot is not None else None
      )
      reserved = runner._task_registry.get(
        supplied_successor.attempt.physical_task_id,
      )
      if supplied_successor.logical_task != original_admitted_task.logical_task:
        return None, {
          "code": "logical_task_mismatch",
          "message": "Successor must retain the parked logical task",
        }
      if (
        root.tenant is None
        or source_context is None
        or successor_context is None
        or source_context.tenant != root.tenant
        or successor_context.tenant != root.tenant
      ):
        return None, {
          "code": "tenant_mismatch",
          "message": "Successor must retain the authenticated parked tenant",
        }
      if supplied_successor.attempt.resume_of_task_id != task_id:
        return None, {
          "code": "resume_source_mismatch",
          "message": "Successor must name the parked attempt as its resume source",
        }
      if (
        reserved is None
        or reserved.original_task_id != task_id
        or supplied_successor.admitted_task_id != reserved.task_id
      ):
        return None, {
          "code": "resume_successor_conflict",
          "message": "Successor must use its reserved physical and admitted task id",
        }
      if supplied_successor.tool_grant != original_admitted_task.tool_grant:
        return None, {
          "code": "tool_grant_mismatch",
          "message": "Successor must retain the parked attempt's exact tool grant",
        }
      if supplied_successor.model_bind != original_bind:
        return None, {
          "code": "capability_bind_mismatch",
          "message": "Successor must retain the parked capability bind",
        }
    depth = await runner._resume_chain_depth(task_id)
    source_state_error = _source_state_error()
    if source_state_error is not None:
      return None, source_state_error

    operation = original_admitted_task.operation
    ticker_required = "ticker" in operation.required_context
    trusted_research_origin_required = _tool_grant_has_investment_claim_routes(
      original_admitted_task.tool_grant,
    )
    artifact_writer_requires_fenced_identity = any(
      entry.tool_id in _ARTIFACT_EMIT_TOOLS
      or (
        entry.tool_id.startswith("fms_")
        and entry.effect in {"artifact_write", "state_write"}
      )
      for entry in original_admitted_task.tool_grant.tools
    )
    try:
      if (
        supplied_successor is not None
        and isinstance(original_admitted_task.logical_task, WorkflowNodeTaskRef)
      ):
        assert successor_context is not None
        logical_invocation_owner = None
        admitted_ticker = successor_context.research_binding.ticker
        admitted_research_file_id = successor_context.research_binding.research_file_id
      else:
        original_logical_task = _ordinary_logical_task(original_admitted_task)
        logical_invocation_owner = _ordinary_logical_invocation_owner(
          original_logical_task
        )
        admitted_ticker = _ticker_from_admitted_inputs(
          original_admitted_task.inputs,
          required=ticker_required,
          owner_invocation_id=logical_invocation_owner,
        )
        admitted_research_file_id = _research_file_id_from_admitted_inputs(
          original_admitted_task.inputs,
          required=(
            trusted_research_origin_required
            or artifact_writer_requires_fenced_identity
          ),
          owner_invocation_id=logical_invocation_owner,
        )
    except ValueError as exc:
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has invalid admitted typed context: {exc}",
      )
    persisted_tool_routes = original_admitted_task.tool_routes
    if persisted_tool_routes is None:
      if original_admitted_task.tool_grant.tools:
        return await _abandon(
          "durable_tool_routes_unavailable",
          f"Task {task_id} predates durable routes for its tool grant",
        )
      persisted_tool_routes = ()
    if (
      original_execution_snapshot.provider_tool_definitions is None
      and original_admitted_task.tool_grant.tools
    ):
      return await _abandon(
        "durable_tool_definitions_unavailable",
        f"Task {task_id} predates durable definitions for its tool grant",
      )
    agent_name = operation.methodology.name
    # Resume executes the immutable admitted operation. Methodology files may
    # have changed or disappeared since the original attempt and are never
    # reloaded as execution authority.
    operation_semantic_scope = (
      "ticker" if "ticker" in operation.required_context else None
    )
    result_requirement = original_admitted_task.result_requirement
    source_state_error = _source_state_error()
    if source_state_error is not None:
      return None, source_state_error

    if supplied_successor is not None:
      from .approval_route import child_approval_route

      assert successor_context is not None
      continuation = AdmittedContext(
        successor_context,
        supplied_successor,
        child_approval_route(root.approval_route),
      )
    else:
      continuation = admit(
        ContinueRequest(
          job_id=task_id,
          cause="revive",
          message=additional_context,
        ),
        root,
        owners=ContinuationOwners(
          task=original_admitted_task,
          research_binding=AdmittedResearchBinding(
            research_file_id=admitted_research_file_id,
            ticker=admitted_ticker,
          ),
          placement="in_process",
        ),
      )
      if isinstance(continuation, Refusal):
        return await _abandon(continuation.reason, continuation.detail)
      original_admitted_task = continuation.tool_authority
    resumer_job_id = root.job_id

    try:
      resume_resolver = capability_execution_resolver
      from .providers.anthropic_oauth import (
        AnthropicSourceError,
        anthropic_generation_auth_config,
        is_anthropic_generation_ref,
      )

      if (
        original_bind.provider == "anthropic"
        and is_anthropic_generation_ref(original_bind.credential_ref)
      ):
        try:
          exact_config = anthropic_generation_auth_config(
            original_bind.credential_ref,
            user_scope=getattr(parent_session, "risk_user_id", None),
            base_config=getattr(parent_session, "auth_config", None) or {},
          )
        except AnthropicSourceError as exc:
          raise CapabilityResolutionError(
            "credential_unavailable",
            "resumed Anthropic credential generation is unavailable",
            capability_id=original_bind.capability_id,
          ) from exc
        resume_resolver = resume_resolver.with_frozen_credential(
          original_bind, exact_config,
        )
      execution = resume_resolver.materialize_bind(original_bind)
    except CapabilityResolutionError as exc:
      resolution_error = _capability_resolution_error(exc)
      return await _abandon(
        str(resolution_error["code"]),
        str(resolution_error["message"]),
      )
    try:
      require_commercial_child_provider(
        commercial_work_start,
        execution.bind.provider,
      )
    except CommercialWorkStartError as exc:
      return None, {"code": exc.code, "message": str(exc)}

    outer_activity_lease_ref: list[Any | None] = [None]
    child_activity_lease_ref: list[Any | None] = [None]

    def _release_activity_ref(ref: list[Any | None]) -> None:
      lease, ref[0] = ref[0], None
      if lease is not None:
        lease.release()

    if admitted_research_file_id is not None:
      workspace_dir = getattr(runner, "_workspace_dir", None)
      if workspace_dir is None:
        return None, {
          "code": "research_file_activity_unavailable",
          "message": "Resumed research work has no exact workspace activity boundary",
        }
      try:
        outer_activity_lease_ref[0] = try_acquire_research_file_activity(
          Path(workspace_dir),
          research_file_id=admitted_research_file_id,
          exclusive=False,
        )
        if outer_activity_lease_ref[0] is not None:
          child_activity_lease_ref[0] = try_acquire_research_file_activity(
            Path(workspace_dir),
            research_file_id=admitted_research_file_id,
            exclusive=False,
          )
      except (OSError, ValueError):
        _release_activity_ref(outer_activity_lease_ref)
        _release_activity_ref(child_activity_lease_ref)
        return None, {
          "code": "research_file_activity_unavailable",
          "message": "Resumed research file activity is unavailable",
        }
      if (
        outer_activity_lease_ref[0] is None
        or child_activity_lease_ref[0] is None
      ):
        _release_activity_ref(outer_activity_lease_ref)
        _release_activity_ref(child_activity_lease_ref)
        return None, {
          "code": "research_file_activity_unavailable",
          "message": "Resumed research file activity is unavailable",
        }

    try:
      source_task_is_current = await task_registration_is_current(
        durable_log,
        task_id,
      )
      current_entry = (
        await _admitted_resume_source(runner, task_id)
        if supplied_successor is not None
        else await runner._lookup_task_in_log(task_id)
      )
    except ResearchFileCurrentProjectionUnavailable:
      _release_activity_ref(outer_activity_lease_ref)
      _release_activity_ref(child_activity_lease_ref)
      return None, {
        "code": "current_projection_unavailable",
        "message": "Current background task state is unavailable",
      }
    if (
      not source_task_is_current
      or current_entry is None
      or (
        _admitted_resume_source_error(current_entry, task_id) is not None
        if supplied_successor is not None
        else current_entry.state != TaskState.INTERRUPTED
      )
      or current_entry.admitted_task != original_admitted_task
    ):
      _release_activity_ref(outer_activity_lease_ref)
      _release_activity_ref(child_activity_lease_ref)
      return None, {
        "code": "current_projection_unavailable",
        "message": f"Task {task_id} is not available in current history",
      }
    entry = current_entry

    successor_task_id_ref: list[str | None] = [None]

    def _release_untransferred_activity_after_entry() -> None:
      successor_task_id = successor_task_id_ref[0]
      successor_entry = (
        runner._task_registry.get(successor_task_id)
        if successor_task_id is not None
        else None
      )
      if successor_entry is None:
        _release_activity_ref(outer_activity_lease_ref)
        _release_activity_ref(child_activity_lease_ref)
        return
      initialization_task = getattr(
        successor_entry,
        "initialization_task",
        None,
      )
      if (
        initialization_task is not None
        and callable(getattr(initialization_task, "done", None))
        and not initialization_task.done()
      ):
        initialization_task.add_done_callback(
          lambda _task: _release_untransferred_activity_after_entry()
        )
        return
      if (
        getattr(successor_entry, "completion_callback", None)
        is not completion_callback
      ):
        _release_activity_ref(outer_activity_lease_ref)
      worker_task = getattr(successor_entry, "asyncio_task", None)
      if (
        child_activity_lease_ref[0] is not None
        and worker_task is not None
        and callable(getattr(worker_task, "done", None))
        and not worker_task.done()
      ):
        worker_task.add_done_callback(
          lambda _task: _release_activity_ref(child_activity_lease_ref)
        )
      elif child_activity_lease_ref[0] is not None:
        _release_activity_ref(child_activity_lease_ref)

    def _release_outer_activity(_entry: Any) -> None:
      _release_activity_ref(outer_activity_lease_ref)

    async def _fanout_completion(completed: TaskEntry) -> None:
      _release_outer_activity(completed)
      _release_activity_ref(child_activity_lease_ref)
      if (
        operation_resume_binding is not None
        and operation_resume_binding.on_settled is not None
      ):
        await operation_resume_binding.on_settled(completed)
      if supplied_on_complete is not None:
        returned = supplied_on_complete(completed)
        if inspect.isawaitable(returned):
          await returned

    completion_callback = (
      _fanout_completion
      if supplied_successor is not None or operation_resume_binding is not None
      else _release_outer_activity
    )

    current_task = asyncio.current_task()
    if current_task is not None:
      current_task.add_done_callback(
        lambda _task: _release_untransferred_activity_after_entry()
      )

    lineage = await reconstruct_child_run_lineage(durable_log, task_id)
    source_state_error = _source_state_error()
    if source_state_error is not None:
      _release_activity_ref(outer_activity_lease_ref)
      _release_activity_ref(child_activity_lease_ref)
      return None, source_state_error
    prior_evidence = (
      _prior_result_evidence(lineage)
      if lineage
      else SubAgentResultEvidence.empty()
    )
    prior_terminal_tool_result = (
      _sub_agent_skill_state.latest_successful_declared_terminal_tool_result(
        (
          log_entry
          for segment in lineage
          for log_entry in segment.entries
        ),
        declared_terminal_doors=(
          _sub_agent_skill_state.terminal_result_doors(original_admitted_task)
        ),
      )
    )
    if not lineage or lineage[-1].task_id != task_id:
      result = await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has no exact durable child-run lineage",
      )
      _release_activity_ref(outer_activity_lease_ref)
      _release_activity_ref(child_activity_lease_ref)
      return result

    transcript = await reconstruct_messages_for_task(durable_log, task_id)
    orphan_ids = detect_orphan_tool_uses(transcript)
    synthetic_results = build_synthetic_tool_results(orphan_ids)
    parent_messages = await reconstruct_parent_messages(
      durable_log,
      task_id,
      datetime.datetime.now(datetime.UTC).timestamp(),
    )
    if supplied_successor is not None:
      # The workflow owner's frozen snapshot records what the model is told.
      assert supplied_successor.execution_snapshot is not None
      successor_execution_snapshot = supplied_successor.execution_snapshot
      exact_resume_instruction = successor_execution_snapshot.resume_instruction
      assert exact_resume_instruction is not None
    else:
      exact_resume_instruction = (
        additional_context.strip()
        if additional_context is not None and additional_context.strip()
        else "Resume the admitted task from its durable transcript."
      )
      successor_execution_snapshot = resume_agent_execution_snapshot(
        original_execution_snapshot,
        resume_instruction=exact_resume_instruction,
      ).model_copy(update={"admitted_context": continuation.wire})
    reconstructed_messages = place_resume_messages(
      transcript,
      synthetic_results,
      parent_messages,
      exact_resume_instruction,
    )
    if successor_execution_snapshot.provider_tool_definitions is None:
      successor_execution_snapshot = successor_execution_snapshot.model_copy(
        update={"provider_tool_definitions": ()}
      )
    successor_provider_definitions = (
      successor_execution_snapshot.provider_tool_definitions
    )
    assert successor_provider_definitions is not None
    system_prompt = successor_execution_snapshot.system_prompt
    try:
      (
        admitted_dispatch_tools,
        admitted_local_tool_ids,
        admitted_mcp_scope,
      ) = dispatcher_scopes_from_admitted_routes(
        original_admitted_task.tool_grant,
        persisted_tool_routes,
      )
    except OperationToolAdmissionError as exc:
      return await _abandon("admitted_tool_route_unavailable", str(exc))
    child_excluded = set(local_tool_handlers or {}) - admitted_dispatch_tools
    child_excluded |= _ARTIFACT_EMIT_TOOLS - admitted_dispatch_tools
    sub_local = dict(local_tool_handlers or {})
    child_needs_approval = (
      needs_approval_resolver(frozenset(child_excluded))
      if needs_approval_resolver is not None
      else needs_approval
    )
    tool_ctx = kwargs.get("tool_ctx")
    parent_turn_id = getattr(tool_ctx, "tool_call_id", None)
    call_index = int(kwargs.get("call_index", 0) or 0)
    skill_run_id = secrets.token_hex(16)
    context_research_file_id = admitted_research_file_id
    if context_research_file_id == 0:
      context_research_file_id = None
    if fms_rebinder is not None and context_research_file_id is not None and context_research_file_id > 0:
      fms_rebinder(sub_local, context_research_file_id)
    resume_root_id = await runner._resume_root_task_id(task_id)
    successor_task_id = f"{resume_root_id}_r{depth + 1}"
    successor_task_id_ref[0] = successor_task_id
    if supplied_successor is not None:
      if supplied_successor.attempt.physical_task_id != successor_task_id:
        _release_activity_ref(outer_activity_lease_ref)
        _release_activity_ref(child_activity_lease_ref)
        return None, {
          "code": "resume_successor_conflict",
          "message": "Successor does not match the deterministic reserved lineage",
        }
      successor_attempt = supplied_successor.attempt
    else:
      successor_attempt = AttemptRef(
        attempt_number=original_admitted_task.attempt.attempt_number + 1,
        attempt_id=(
          f"{successor_task_id}:attempt:"
          f"{original_admitted_task.attempt.attempt_number + 1}"
        ),
        physical_task_id=successor_task_id,
        resume_of_task_id=task_id,
        resumer_job_id=resumer_job_id,
      )
    successor_admitted_task_ref: list[AdmittedTask | None] = [None]

    def _successor_context_ticker() -> str | None:
      successor_admitted_task = successor_admitted_task_ref[0]
      if successor_admitted_task is None:
        raise RuntimeError("successor admission is unavailable before dispatch")
      if supplied_successor is not None:
        return admitted_ticker
      return _ticker_from_admitted_inputs(
        successor_admitted_task.inputs,
        required=ticker_required,
        owner_invocation_id=logical_invocation_owner,
      )

    skill_event_emitter: SkillRunEventEmitter | None = None

    def _emit_parent_event(event: dict[str, Any]) -> None:
      if skill_event_emitter is not None:
        skill_event_emitter.emit_parent_event(event)

    async def _emit_skill_run_started() -> None:
      if skill_event_emitter is not None:
        await skill_event_emitter.emit_started()

    if artifact_tools_installer is not None:
      artifact_tools_installer(
        excluded_tools=child_excluded,
        sub_local=sub_local,
        skill_name=agent_name,
        semantic_scope=operation_semantic_scope,
        skill_run_id=skill_run_id,
        context_ticker=_successor_context_ticker,
        context_research_file_id=context_research_file_id,
        parent_session=parent_session,
        fallback_user_id=user_id or getattr(parent_session, "user_id", None),
        emit_parent_event=_emit_parent_event,
      )
    candidate_local_names = set(sub_local)
    sub_local = {
      name: handler
      for name, handler in sub_local.items()
      if name in admitted_local_tool_ids
    }
    # A current local handler may share the ID of a persisted MCP route. Drop
    # that handler, but keep the ID dispatchable through its admitted server.
    child_excluded |= candidate_local_names - admitted_dispatch_tools
    if supplied_successor is not None:
      successor_admitted_task = supplied_successor
    else:
      successor_grant = reissue_tool_grant(
        original_admitted_task.tool_grant,
        grant_id=f"grant:{successor_task_id}",
      )
      successor_payload = original_admitted_task.model_dump(
        mode="json", exclude={"admitted_task_digest"},
      )
      successor_payload.update({
        "schema_version": "1.2",
        "admitted_task_id": f"admitted:{successor_task_id}",
        "attempt": successor_attempt.model_dump(mode="json"),
        "objective": f"Resume interrupted delegation {task_id}",
        "execution_snapshot": successor_execution_snapshot.model_dump(mode="json"),
        "tool_grant": successor_grant.model_dump(mode="json"),
        "tool_grant_digest": successor_grant.digest,
        "tool_routes": [
          route.model_dump(mode="json") for route in persisted_tool_routes
        ],
      })
      seal_admitted_task_payload(successor_payload)
      successor_admitted_task = AdmittedTask.model_validate(successor_payload)
    successor_admitted_task_ref[0] = successor_admitted_task
    context_ticker = _successor_context_ticker()
    successor_provenance = TaskResultProvenance(
      admitted_task_digest=successor_admitted_task.admitted_task_digest,
      model_bind_digest=successor_admitted_task.model_bind_digest,
      capability_binding_digest=(
        successor_admitted_task.capability_binding_digest
      ),
      tool_grant_digest=successor_admitted_task.tool_grant_digest,
    )
    sub_log = EventLog()
    skill_event_emitter = SkillRunEventEmitter(
      skill_run_id=skill_run_id,
      skill_name=agent_name,
      semantic_scope=operation_semantic_scope,
      context_ticker=context_ticker,
      portfolio_id=parent_portfolio_id,
      event_log_getter=lambda: sub_log,
      tool_ctx=tool_ctx,
      durable_appender=runner._append_durable_event,
      durable_confirmer=runner._confirm_durable_skill_event,
    )
    effective_session_inject_servers = (
      _effective_mcp_session_inject_servers(
        admitted_mcp_scope=admitted_mcp_scope,
        configured_servers=mcp_session_inject_servers,
      )
    )
    async def _attempt_lineage(original_task_id: str) -> bool:
      return await runner._is_resume_successor(
        successor_admitted_task,
        original_task_id,
      )

    async def _dispatch_resume(_background_input: dict[str, Any], **background_kwargs: Any):
      background_call_index = int(background_kwargs.get("call_index", call_index) or 0)
      task_entry = background_kwargs.get("task_entry")
      child_context = context_for_child(
        continuation,
        job_id=successor_admitted_task.attempt.physical_task_id,
      )
      sub_session = None
      if parent_session is not None:
        sub_session = child_gateway_session(
          parent_session,
          session_id=_derive_sub_agent_id(parent_session, background_call_index),
          tenant=continuation.wire.tenant,
          auth_config=execution.auth_config,
          user_id=parent_session.user_id,
          user_email=parent_session.user_email,
        )
        capture_job_context(sub_session, child_context)
      try:
        child_local: dict[str, LocalToolHandler] = dict(sub_local)
        child_spill_sink = (
          job_spill_sink(
            runner._spill_dir_provider,
            successor_admitted_task.attempt.physical_task_id,
          )
          if runner._spill_dir_provider is not None
          else None
        )
        child_runner_ref: list[Any] = []
        rebind_job_handlers(child_local, JobScope(
          caller=lambda: current_parent_context(sub_session),
          spill_sink=lambda: child_spill_sink,
          runner=lambda: child_runner_ref[0],
          session=sub_session,
        ))
        code_execution = _install_child_code_execution(
          child_local,
          admitted_local_tools=admitted_local_tool_ids,
          session=sub_session,
          factory=code_execution_factory,
        )
        child_input_preparation = (
          code_execution_input_preparation_factory(code_execution)
          if code_execution_input_preparation_factory is not None
          else None
        )
        if operation_resume_binding is not None:
          for tool_name, scope in operation_resume_binding.local_handler_scopes.items():
            stock = child_local.get(tool_name)
            if not callable(stock):
              raise KeyError(tool_name)
            child_local[tool_name] = scope(stock)
        sub_dispatcher = ToolDispatcher(
          capability_bind=original_bind,
          mcp_client=mcp_client,
          local_tool_handlers=child_local,
          plan_validator=getattr(getattr(runner, "_dispatcher", None), "plan_validator", None),
          plan_review_renderer=getattr(getattr(runner, "_dispatcher", None), "plan_review_renderer", None),
          needs_approval=child_needs_approval,
          event_log=sub_log,
          interceptors=interceptors,
          session_id=getattr(runner, "_full_session_id", ""),
          should_avoid_permission_prompts=True,
          approval_key_qualifier=(
            code_execution.approval_qualifier
            if code_execution is not None
            else None
          ),
          tool_registration_catalog=tool_registration_catalog,
          tool_policy_implementations=tool_policy_implementations,
          input_preparation_context_factory=child_input_preparation,
          redaction_context_factory=redaction_context_factory,
          mcp_session_inject_servers=effective_session_inject_servers,
          mcp_meta_inject_servers=mcp_meta_inject_servers,
          # Resume keeps its persisted routes and grant; fresh interpreter
          # state is not new dispatch authority.
          identity=dispatch_identity(
            session=parent_session,
            execution=execution_identity_from_session(parent_session),
            context=child_context,
            attempt_lineage=_attempt_lineage,
            user_id=user_id or getattr(parent_session, "user_id", None),
            credentials_resolver_active=credentials_resolver_active,
          ),
          role=require_inherited_role(parent_session),
          approval_route=continuation.approval_route,
          run_context=_child_run_context(
            parent_session=parent_session,
            tool_ctx=tool_ctx,
            skill_run_id=skill_run_id,
            skill_name=agent_name,
            admitted_skill_execution_limits=SkillExecutionLimits(
              max_tokens=successor_execution_snapshot.max_tokens,
            ),
            research_file_id=context_research_file_id,
            user_id=user_id or getattr(parent_session, "user_id", None),
            session_id=getattr(runner, "_full_session_id", ""),
            approval_policy=getattr(parent_session, "approval_policy", None),
          ),
          allowed_mcp_tools_by_server=admitted_mcp_scope,
          get_tool_definitions=_admitted_tool_definitions_getter(
            successor_provider_definitions,
            unavailable_local_tools=sub_local.keys() - child_local.keys(),
          ),
          commercial_work_start=commercial_work_start,
          commercial_irreversible_recheck=commercial_irreversible_recheck,
          commercial_mcp_servers=commercial_mcp_servers,
        )
        if code_execution is not None and code_execution_binder is not None:
          code_execution_binder(code_execution, sub_dispatcher)
        await _emit_skill_run_started()
        def _bind_resumed_runner(sub_runner: Any) -> None:
          child_runner_ref.append(sub_runner)
        return await runner.resume_sub_agent(
          original_task_id=task_id,
          reconstructed_messages=reconstructed_messages,
          parent_messages=parent_messages,
          capability_execution=execution,
          skill_name=agent_name,
          logical_task=successor_admitted_task.logical_task,
          attempt=successor_admitted_task.attempt,
          result_requirement=result_requirement,
          result_provenance=successor_provenance,
          prior_evidence=prior_evidence,
          prior_terminal_tool_result=prior_terminal_tool_result,
          system_prompt=system_prompt,
          dispatcher=sub_dispatcher,
          sub_session=sub_session,
          bind_job_runner=_bind_resumed_runner,
          approved_call=approved_call,
          excluded_tools=child_excluded,
          max_tokens=successor_execution_snapshot.max_tokens,
          skill_run_id=skill_run_id,
          call_index=background_call_index,
          parent_turn_id=parent_turn_id,
          task_entry=task_entry,
          cost_observation_threshold_usd=(
            successor_execution_snapshot.cost_observation_threshold_usd
          ),
          on_sub_event=lambda event, _sid: sub_log.append(event),
          bind_research_file_activity_lease_func=(
            _bind_child_research_file_activity
            if child_activity_lease_ref[0] is not None
            else None
          ),
          admitted_task=successor_admitted_task,
          usage_aggregator=(
            operation_resume_binding.usage_aggregator
            if operation_resume_binding is not None
            else runner._aggregator
          ),
        )
      finally:
        await _dispose_child_code_execution(sub_session)

    def _bind_child_research_file_activity(sub_runner: Any) -> None:
      lease = child_activity_lease_ref[0]
      if lease is None:
        return
      bind_activity = getattr(
        sub_runner,
        "bind_research_file_activity_lease",
        None,
      )
      if not callable(bind_activity):
        raise RuntimeError(
          "resumed runner cannot own research file activity"
        )
      setattr(
        sub_runner,
        "_context_research_file_id",
        admitted_research_file_id,
      )
      bind_activity(lease)
      child_activity_lease_ref[0] = None

    # Resume depth is a measurement on the task record, never a stop.
    resume_tool_input = {
      "task_id": task_id,
      "operation": operation.operation.model_dump(mode="json"),
      "resumable": True,
      "result_requirement": result_requirement.model_dump(mode="json"),
      "resume_chain_depth": depth + 1,
    }
    if successor_execution_snapshot.cost_observation_threshold_usd is not None:
      resume_tool_input["cost_observation_threshold_usd"] = (
        successor_execution_snapshot.cost_observation_threshold_usd
      )
    def _build_resumed_skill_result(
      _bg_task: Any,
      result: dict[str, Any] | None,
      error: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
      return skill_event_emitter.build_result_captured_event(
        result,
        error,
      )

    source_lock: asyncio.Lock | None = (
      entry.finalization_lock if supplied_successor is not None else None
    )
    if source_lock is not None:
      # Parked STOP and successor installation share the source's lock.
      # Registration and completion only finalize the successor entry.
      await source_lock.acquire()
    try:
      if supplied_successor is not None:
        source_state_error = _source_state_error()
        if source_state_error is not None:
          current_source = runner._task_registry.get(task_id) or entry
          if current_source.state not in {
            TaskState.AWAITING_APPROVAL, TaskState.INTERRUPTED,
          }:
            source_state_error = {
              "code": "source_terminated",
              "message": f"Task {task_id} is no longer parked",
            }
          assert reserved is not None
          runner._task_registry.discard_unstarted(
            successor_task_id, expected=reserved,
          )
          _release_activity_ref(outer_activity_lease_ref)
          _release_activity_ref(child_activity_lease_ref)
          return None, source_state_error
      result, error = await runner._register_background_task(
        tool_input=resume_tool_input,
        handler=_dispatch_resume,
        agent_name=agent_name,
        parent_turn_id=parent_turn_id,
        on_complete=completion_callback,
        on_park=supplied_on_park,
        capability_bind_receipt=execution.bind.to_json(),
        admitted_task=successor_admitted_task,
        task_id_override=successor_task_id,
        required_skill_lifecycle=(
          skill_event_emitter.required_lifecycle_metadata()
        ),
        required_skill_result_event_factory=_build_resumed_skill_result,
        required_skill_result_projector=(
          skill_event_emitter.project_result_captured
        ),
        original_task_id=task_id,
        validate_resume_source=supplied_successor is None,
        resume_source_entry=entry,
      )
    except BaseException:
      _release_untransferred_activity_after_entry()
      raise
    finally:
      if source_lock is not None:
        successor_entry = runner._task_registry.get(successor_task_id)
        initialization_task = (
          successor_entry.initialization_task
          if successor_entry is not None else None
        )
        if initialization_task is not None and not initialization_task.done():
          # Cancellation leaves shielded initialization running. Keep STOP
          # excluded until that registration has durably started or settled.
          initialization_task.add_done_callback(
            lambda _task, lock=source_lock: lock.release(),
          )
        else:
          source_lock.release()
    _release_untransferred_activity_after_entry()
    if error is not None:
      return None, error
    if supplied_successor is not None:
      return runner._task_registry.get(successor_task_id), None
    return result, None

  async def _admitted_entry(
    runner: Any,
    *,
    parked_task_id: str,
    successor: AdmittedTask,
    approved_call: ApprovedCallRef | None,
    on_complete: Callable[[TaskEntry], Awaitable[None] | None],
    on_park: Callable[[TaskEntry], Awaitable[None]] | None,
    caller: CallerContext,
  ) -> tuple[TaskEntry | None, dict | None]:
    return await _execute_resume(
      runner,
      {"task_id": parked_task_id},
      caller=caller,
      supplied_successor=successor,
      supplied_on_complete=on_complete,
      supplied_on_park=on_park,
      approved_call=approved_call,
      dispatch_kwargs={},
    )

  async def _approved_entry(
    runner: Any,
    *,
    parked_task_id: str,
    approved_call: ApprovedCallRef,
    on_complete: Callable[[TaskEntry], Awaitable[None] | None],
    on_park: Callable[[TaskEntry], Awaitable[None]] | None,
    caller: CallerContext,
  ) -> tuple[TaskEntry | None, dict | None]:
    successor_id, error = await reserve_admitted_successor(
      runner, parked_task_id=parked_task_id,
    )
    if error is not None:
      return None, error
    assert successor_id is not None
    existing = runner._task_registry.get(successor_id)
    if existing is not None:
      started = await runner._started_background_task(successor_id)
      if started is not None:
        return started, None
    source = await _admitted_resume_source(runner, parked_task_id)
    assert source is not None and isinstance(source.admitted_task, AdmittedTask)
    original = source.admitted_task
    snapshot = original.execution_snapshot
    assert snapshot is not None
    attempt_number = original.attempt.attempt_number + 1
    payload = original.model_dump(mode="json", exclude={"admitted_task_digest"})
    payload.update({
      "admitted_task_id": successor_id,
      "attempt": AttemptRef(
        attempt_number=attempt_number,
        attempt_id=f"{successor_id}:attempt:{attempt_number}",
        physical_task_id=successor_id,
        resume_of_task_id=parked_task_id,
        resumer_job_id=approved_call.job_id,
      ).model_dump(mode="json"),
      "execution_snapshot": resume_agent_execution_snapshot(
        snapshot,
        resume_instruction="Resume the admitted task from its durable transcript.",
      ).model_dump(mode="json"),
    })
    seal_admitted_task_payload(payload)
    successor = AdmittedTask.model_validate(payload).model_copy(
      update={"tool_grant": original.tool_grant},
    )
    return await _execute_resume(
      runner, {"task_id": parked_task_id}, caller=caller,
      supplied_successor=successor, supplied_on_complete=on_complete,
      supplied_on_park=on_park,
      approved_call=approved_call, dispatch_kwargs={},
    )

  def _build(scope: JobScope):
    async def _scoped_resume(tool_input: dict[str, Any], **kwargs: Any):
      # Dispatch kwargs remain data: they cannot select the trusted host path.
      return await _execute_resume(
        runner_ref[0], tool_input, caller=scope.caller,
        supplied_on_park=None, dispatch_kwargs=kwargs,
      )
    return _scoped_resume

  return _ResumeHandler(
    _build,
    JobScope(
      caller=caller if caller is not None else _current_caller,
      spill_sink=lambda: None,
    ),
    _admitted_entry,
    _approved_entry,
    operation_resume_bindings,
  )


def make_send_message_handler(runner_ref: list[Any], *, caller):
  return _make_send_message_handler(runner_ref, caller=caller)


async def resume_binding_for(
  runner: Any,
  task_id: str,
) -> tuple[
  TaskEntry, SubAgentResultEvidence | None, OperationResumeBinding | None,
] | None:
  """Attach an admitted task and recover its own durable resume inputs."""
  entry = await runner.attach_background_task(
    task_id, on_complete=None, on_park=None,
  )
  if entry is None or not isinstance(entry.admitted_task, AdmittedTask):
    return None
  durable_log = runner._agent_session_log
  lineage = (
    await reconstruct_child_run_lineage(durable_log, task_id)
    if durable_log is not None
    else []
  )
  evidence = _prior_result_evidence(lineage) if lineage else None
  handler = next((
    value for value in runner._dispatcher.local_tool_handlers().values()
    if isinstance(value, _ResumeHandler)
  ), None)
  binding = (
    handler._operation_resume_bindings(entry.admitted_task)
    if handler is not None and handler._operation_resume_bindings is not None
    else None
  )
  return entry, evidence, binding


__all__ = [
  "OperationResumeBinding",
  "seal_admitted_task_payload",
  "reserve_admitted_successor",
  "resume_admitted_successor",
  "resume_binding_for",
  "ApprovedCallRef",
  "continue_approved_child",
  "make_get_background_result_handler",
  "make_get_background_result_tool_def",
  "make_send_message_handler",
  "make_send_message_tool_def",
  "make_resume_handler",
  "make_resume_tool_def",
  "make_run_agent_handler",
  "make_run_agent_tool_def",
]
