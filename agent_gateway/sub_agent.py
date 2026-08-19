from __future__ import annotations

import asyncio
import datetime
import logging
import secrets
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Mapping
from collections.abc import Set as AbstractSet

from agent_workflow_contracts import (
  AgentCompletionEnvelope,
  AdmittedDataRef,
  AdmittedInputBinding,
  AdmittedTask,
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
  ParentResultPolicy,
  RequestedDataRef,
  ResultRequirement,
  TaskResult,
  TaskResultProvenance,
  WorkspaceGrant,
  canonical_json_bytes,
  sha256_digest,
  terminal_task_result,
)
from agent_workflow_contracts.ticker_contract import (
  TICKER_INPUT_CONTRACT,
  require_canonical_contract_ticker,
)

from .autonomous_output import extract_state_update
from .approval_policy import RunContext
from .agent_result_content import make_get_agent_result_content_handler
from .capability_binding import (
  CapabilityBind,
  CapabilityResolutionError,
)
from .capability_execution import CapabilityExecutionResolver
from .commercial_work_start import (
  CommercialWorkStartError,
  require_commercial_child_provider,
)
from .event_log import EventLog
from .execution_snapshot import (
  build_agent_execution_snapshot,
  render_result_instructions,
  resume_agent_execution_snapshot,
)
from .policy_imports import require_inherited_role, role_denied_tools_for_session
from .runner import _derive_sub_agent_id
from .runner_background_tasks import (
  agent_completion_message_id,
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
  task_completed_event_payload,
  task_correlation_payload,
  task_registered_event_payload,
)
from .runner_session_events import build_agent_completion_event
from .session import GatewaySession
from .skills import (
  GENERIC_EXPLORE_OPERATION_NAME,
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
from .capability_resolution import (
  derive_dispatcher_allowlist,
  granted_tool_ids,
)
from .execution_identity import dispatch_identity, execution_identity_from_session
from .sub_agent_scope_receipt import (
  ADMITTED_TASK_METADATA_KEY,
  OperationToolAdmissionError,
  admit_operation_tools,
  reissue_tool_grant,
  scopes_from_tool_grant,
)
from .sub_agent_helpers import (
  DEFAULT_SUB_AGENT_TIMEOUT_SECONDS as DEFAULT_SUB_AGENT_TIMEOUT_SECONDS,
  ExcludedToolsResolver as ExcludedToolsResolver,
  MutationModeExclusionsApplier as MutationModeExclusionsApplier,
  NeedsApprovalResolver as NeedsApprovalResolver,
  _ARTIFACT_EMIT_TOOLS as _ARTIFACT_EMIT_TOOLS,
  _DEFAULT_SYSTEM_PROMPT_TEMPLATE as _DEFAULT_SYSTEM_PROMPT_TEMPLATE,
  _DEFAULT_EXCLUDED_TOOLS as _DEFAULT_EXCLUDED_TOOLS,
  _RESEARCH_FILE_ID_RE as _RESEARCH_FILE_ID_RE,
  _RESUME_AGENT_DESCRIPTION as _RESUME_AGENT_DESCRIPTION,
  _RUN_AGENT_DESCRIPTION as _RUN_AGENT_DESCRIPTION,
  _SKILL_SYSTEM_PROMPT_TEMPLATE as _SKILL_SYSTEM_PROMPT_TEMPLATE,
  _TICKER_STOPWORDS as _TICKER_STOPWORDS,
  _artifact_storage_user_id as _artifact_storage_user_id,
  _dashboard_artifact_scope as _dashboard_artifact_scope,
  _dashboard_artifact_ticker as _dashboard_artifact_ticker,
  _extract_research_file_id_from_resume_messages as _extract_research_file_id_from_resume_messages,
  _extract_research_file_id_from_task as _extract_research_file_id_from_task,
  _extract_ticker_from_task as _extract_ticker_from_task,
  _artifact_scope as _artifact_scope,
  _artifact_ticker as _artifact_ticker,
  _install_emit_dashboard_artifact_handler as _install_emit_dashboard_artifact_handler,
  _install_emit_canvas_artifact_handler as _install_emit_canvas_artifact_handler,
  _message_content_text as _message_content_text,
  _optional_research_file_id as _optional_research_file_id,
  _render_agent_param_description as _render_agent_param_description,
  _resolve_context_ticker as _resolve_context_ticker,
  _skill_extra_excluded_tool_names as _skill_extra_excluded_tool_names,
  _skill_artifact_excluded_tools as _skill_artifact_excluded_tools,
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
  CoordinatorConfig,
  TaskEntry,
  TaskState,
  task_state_for_result,
)
from .tool_dispatcher import ToolDispatcher
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


def _child_run_context(
  *,
  parent_session: GatewaySession | None,
  tool_ctx: Any,
  skill_run_id: str,
  skill_name: str | None,
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
    profile=str(skill_name or "sub_agent"),
    channel=str(getattr(parent_session, "channel", None) or "web"),
    skill=skill_name,
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

  appender = getattr(runner, "_append_durable_event", None)
  confirmer = getattr(runner, "_confirm_durable_skill_event", None)
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


def _operation_private_mcp_tool_definitions(
  *,
  profile: SkillProfile,
  mcp_client: Any,
  exact_tool_ids: frozenset[str],
) -> list[dict[str, Any]]:
  """Read connected schemas for one operation without exposing them to its parent."""

  get_server_definitions = getattr(
    mcp_client,
    "get_server_tool_definitions",
    None,
  )
  if not callable(get_server_definitions):
    return []
  declared_servers = set(profile.mcp_tools or {})
  return [
    definition
    for definition in get_server_definitions(declared_servers)
    if isinstance(definition, dict)
    and str(definition.get("name") or "") in exact_tool_ids
  ]


def _operation_private_mcp_tool_ids(
  profile: SkillProfile,
  *,
  exact_tool_ids: frozenset[str],
) -> frozenset[str]:
  return frozenset(
    tool_id
    for tool_ids in (profile.mcp_tools or {}).values()
    for tool_id in tool_ids
    if tool_id in exact_tool_ids
  )


def _operation_has_investment_claim_routes(
  profile: SkillProfile,
  *,
  exact_tool_ids: frozenset[str],
) -> bool:
  declared = profile.mcp_tools or {}
  investment_tools = declared.get("idea-workbench-mcp", ())
  return "start_quant_research" in (
    exact_tool_ids & frozenset(investment_tools)
  )


def _tool_grant_has_investment_claim_routes(
  tool_grant: Any,
) -> bool:
  return any(
    entry.tool_id == "start_quant_research"
    for entry in tool_grant.tools
  )


def _artifact_emit_tool_definitions(installed_tool_names: set[str]) -> list[dict[str, Any]]:
  return _sub_agent_tool_definitions.artifact_emit_tool_definitions(
    installed_tool_names,
    artifact_emit_tools=_ARTIFACT_EMIT_TOOLS,
  )


def _result_response_text(result: Any | None) -> str:
  return _sub_agent_skill_state.result_response_text(result)


def _skill_state_prompt(skill_name: str, previous_state: dict[str, Any]) -> str:
  return _sub_agent_skill_state.skill_state_prompt(skill_name, previous_state)


def _capability_resolution_error(exc: CapabilityResolutionError) -> dict[str, Any]:
  return {
    "code": exc.code,
    "message": str(exc),
    "details": exc.receipt(),
  }


def _canonical_result_requirement(
  *,
  operation: Any,
) -> ResultRequirement:
  if len(operation.result_modes) != 1:
    raise ValueError(
      "ordinary delegation requires exactly one admitted result mode"
    )
  if operation.result_modes != ("narrative",):
    raise ValueError(
      "agent operations must use the terminal-message result contract"
    )
  if operation.projection_contracts:
    raise ValueError(
      "agent operations cannot require a model-authored result projection"
    )
  return ResultRequirement(
    mode="narrative",
    projection=None,
    terminal_narrative="required",
    outcome=OutcomeRequirement(required=False, source="none"),
  )


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


def _ordinary_parent_result_policy(
  requirement: ResultRequirement,
) -> ParentResultPolicy:
  return ParentResultPolicy(
    preferred=(
      "terminal_narrative_inline_exact"
      if requirement.terminal_narrative == "required"
      else "projection_inline"
    ),
    max_inline_bytes=20_000,
    on_overflow="result_handle",
  )


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
    or request.name != "research_file_id"
    or selector_research_file_id != context.content
    or request.context_policy is not None
    or request.expected_contract != _RESEARCH_FILE_ID_INPUT_CONTRACT
    or source.source_kind != "literal"
    or source.logical_source_id != "research-turn:research_file_id"
    or source.actual_contract != _RESEARCH_FILE_ID_INPUT_CONTRACT
    or source.content.contract != _RESEARCH_FILE_ID_INPUT_CONTRACT
    or source.read_grant is not None
    or not isinstance(context, InlineExactContextView)
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


def _ordinary_logical_invocation_owner(admitted_task: AdmittedTask) -> str:
  logical_task = admitted_task.logical_task
  if not isinstance(logical_task, OrdinaryDelegationTaskRef):
    raise ValueError("ordinary ticker admission requires a delegation identity")
  delegation_id = str(logical_task.delegation_id or "").strip()
  if not delegation_id:
    raise ValueError("ordinary ticker admission has no logical invocation owner")
  return delegation_id


def _ordinary_admitted_task_factory(
  *,
  operation: Any,
  execution_snapshot: Any,
  capability_bindings: tuple[Any, ...],
  tool_grant: Any,
  model_bind: CapabilityBind,
  result_requirement: ResultRequirement,
  objective: str,
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
      "schema_version": "1.0",
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
        admission_rejected=evidence.admission_rejected,
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
    usage=retained.usage,
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
    timeout_getter = getattr(
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


def make_run_agent_handler(
  runner_ref: list[Any],
  *,
  parent_session: GatewaySession | None = None,
  skill_loader: SkillLoader | None = None,
  mcp_client: Any,
  needs_approval: Callable[..., bool] | None = None,
  mcp_session_inject_servers: AbstractSet[str] | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  user_id: str | None = None,
  user_email: str | None = None,
  trusted_research_file_id: int | None = None,
  parent_user_id: str | None = None,
  parent_user_email: str | None = None,
  credentials_resolver_active: bool = False,
  local_tool_handlers: dict[str, Any] | None = None,
  fms_rebinder: Callable[[dict[str, Any], int], None] | None = None,
  excluded_tools: set[str] | None = None,
  default_max_turns: int = 15,
  default_timeout: float | None = DEFAULT_SUB_AGENT_TIMEOUT_SECONDS,
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
  coordinator_config: CoordinatorConfig | None = None,
  approval_key_qualifier: Callable[[str, dict[str, Any]], str] | None = None,
  commercial_work_start: Any | None = None,
  commercial_irreversible_recheck: Callable[[Any], None] | None = None,
  commercial_mcp_servers: frozenset[str] | None = None,
  operation_mcp_activator: (
    Callable[[Any], dict[str, Any] | None] | None
  ) = None,
):
  """Build the local handler used by the `run_agent` tool.

  The returned handler resolves a full operation identity (or the registered
  generic ``explore`` operation), admits its exact model/tool authority, and
  delegates execution to ``AgentRunner.spawn_sub_agent()``.
  """
  if local_tool_handlers is not None:
    local_tool_handlers.setdefault(
      "get_agent_result_content",
      make_get_agent_result_content_handler(runner_ref),
    )
  effective_coordinator = coordinator_config if coordinator_config is not None and coordinator_config.enabled else None
  if trusted_research_file_id is not None:
    _require_research_file_id(trusted_research_file_id)
  skill_state_lock = asyncio.Lock()
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

  async def _handle_run_agent(tool_input: dict[str, Any], **kwargs: Any):
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
    if not isinstance(tool_input, dict):
      return None, {"code": "invalid_input", "message": "input must be an object"}
    allowed_input_keys = frozenset({
      "operation",
      "objective",
      "ticker",
      "research_file_id",
      "cost_observation_threshold_usd",
      "background",
    })
    unknown_input_keys = sorted(set(tool_input) - allowed_input_keys)
    if unknown_input_keys:
      return None, {
        "code": "invalid_input",
        "message": (
          "unknown run_agent input fields: "
          + ", ".join(unknown_input_keys)
        ),
      }
    tool_ctx = kwargs.get("tool_ctx")
    parent_turn_id = getattr(tool_ctx, "tool_call_id", None)

    task = tool_input.get("objective", "")
    if not task or not isinstance(task, str):
      return None, {"code": "invalid_input", "message": "objective is required"}
    derived_research_file_id = _extract_research_file_id_from_task(task)
    raw_research_file_id = tool_input.get("research_file_id")
    if raw_research_file_id is not None:
      try:
        raw_research_file_id = _optional_research_file_id(raw_research_file_id)
      except ValueError as exc:
        return None, {"code": "invalid_input", "message": str(exc)}
      if raw_research_file_id is None or raw_research_file_id <= 0:
        return None, {
          "code": "invalid_input",
          "message": "research_file_id must be a positive integer",
        }
      if (
        derived_research_file_id is not None
        and raw_research_file_id != derived_research_file_id
      ):
        return None, {
          "code": "context_research_file_id_mismatch",
          "message": (
            "research_file_id must match the exact ID stated in objective"
          ),
        }
    asserted_research_file_id = (
      raw_research_file_id or derived_research_file_id
    )
    raw_context_ticker = tool_input.get("ticker")
    raw_operation = tool_input.get("operation")
    named_operation = raw_operation is not None

    background = tool_input.get("background", True)
    if not isinstance(background, bool):
      return None, {"code": "invalid_input", "message": "background must be a boolean"}

    call_index = int(kwargs.get("call_index", 0) or 0)
    ordinary_task_id = secrets.token_hex(16)

    if skill_loader is not None:
      try:
        resolved_operation = skill_loader.resolve_operation(raw_operation)
      except FileNotFoundError as exc:
        return None, {"code": "not_found", "message": str(exc)}
      except Exception as exc:
        return None, {"code": "invalid_operation", "message": str(exc)}
    elif raw_operation is not None:
      return None, {
        "code": "not_available",
        "message": "Registered operations are not available",
      }
    else:
      generic_profile = generic_explore_profile()
      resolved_operation = ResolvedAgentOperation(
        snapshot=compile_agent_operation(
          generic_profile,
          execution_class="node.explore",
        ),
        methodology_profile=generic_profile,
      )
    operation = resolved_operation.snapshot
    profile = resolved_operation.methodology_profile
    agent_name = operation.operation.name
    exact_operation_tool_ids = operation_tool_ids(profile)
    ticker_required = "ticker" in operation.required_context
    research_file_required = (
      "research_file_id" in operation.required_context
    )
    trusted_research_origin_required = _operation_has_investment_claim_routes(
      profile,
      exact_tool_ids=exact_operation_tool_ids,
    )
    if (
      trusted_research_file_id is None
      and trusted_research_origin_required
    ):
      return None, {
        "code": "required_context_missing",
        "message": (
          "operation requires a verified research_file_id from the active "
          "research turn"
        ),
      }
    if (
      trusted_research_file_id is not None
      and asserted_research_file_id is not None
      and asserted_research_file_id != trusted_research_file_id
    ):
      return None, {
        "code": "context_research_file_id_mismatch",
        "message": (
          "research_file_id must match the verified active research turn"
        ),
      }
    context_research_file_id = (
      trusted_research_file_id
      if trusted_research_file_id is not None
      else asserted_research_file_id
    )
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
      return None, {
        "code": "invalid_input",
        "message": str(exc),
      }
    if ticker_required and resolved_context_ticker is None:
      return None, {
        "code": "required_context_missing",
        "message": "operation requires one unambiguous typed ticker subject",
      }
    try:
      admitted_inputs = tuple(
        [
          _ticker_admitted_input(
            resolved_context_ticker,
            invocation_id=ordinary_task_id,
            parent_session=parent_session,
          )
        ]
        if resolved_context_ticker is not None
        else []
      ) + tuple(
        [
          _research_file_id_admitted_input(
            context_research_file_id,
            invocation_id=ordinary_task_id,
            parent_session=parent_session,
          )
        ]
        if (
          (research_file_required or trusted_research_origin_required)
          and trusted_research_file_id is not None
          and context_research_file_id is not None
        )
        else []
      )
    except ValueError as exc:
      return None, {"code": "invalid_context", "message": str(exc)}
    admitted_task_ref: list[AdmittedTask | None] = [None]

    def _admitted_context_ticker() -> str | None:
      admitted_task = admitted_task_ref[0]
      if admitted_task is None:
        raise RuntimeError("admitted task is unavailable before dispatch")
      return _ticker_from_admitted_inputs(
        admitted_task.inputs,
        required=ticker_required,
        owner_invocation_id=_ordinary_logical_invocation_owner(admitted_task),
      )

    def _admitted_context_research_file_id() -> int | None:
      admitted_task = admitted_task_ref[0]
      if admitted_task is None:
        raise RuntimeError("admitted task is unavailable before dispatch")
      return _research_file_id_from_admitted_inputs(
        admitted_task.inputs,
        required=trusted_research_origin_required,
        owner_invocation_id=_ordinary_logical_invocation_owner(admitted_task),
      ) or context_research_file_id

    skill_run_id: str | None = None
    durable_skill_event_transport: tuple[
      DurableSkillEventAppender,
      DurableSkillEventConfirmer,
    ] | None = None
    if named_operation:
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

    try:
      admitted_result_requirement = _canonical_result_requirement(
        operation=operation,
      )
    except ValueError as exc:
      return None, {"code": "invalid_operation", "message": str(exc)}
    capability_id = operation.execution_class

    try:
      execution = capability_execution_resolver.resolve(capability_id)
    except CapabilityResolutionError as exc:
      return None, _capability_resolution_error(exc)

    try:
      require_commercial_child_provider(
        commercial_work_start,
        execution.bind.provider,
      )
    except CommercialWorkStartError as exc:
      return None, {"code": exc.code, "message": str(exc)}
    effective_model = execution.bind.upstream_model

    if named_operation and operation_mcp_activator is not None:
      try:
        activation_error = operation_mcp_activator(profile)
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
        return None, activation_error

    previous_state: dict[str, Any] | None = None
    methodology_state_instructions: str | None = None
    if profile.persist_state and skill_state_store is not None:
      try:
        previous_state = skill_state_store.get(profile.name)
      except Exception:
        log.warning("Failed to load persisted state for skill %s", profile.name, exc_info=True)
        previous_state = {}
      methodology_state_instructions = _skill_state_prompt(
        profile.name,
        previous_state,
      )
    effective_max_turns = profile.max_turns if profile.max_turns is not None else default_max_turns
    effective_timeout = profile.timeout if profile.timeout is not None else default_timeout
    effective_max_tokens = profile.max_tokens if profile.max_tokens is not None else default_max_tokens
    execution_snapshot = build_agent_execution_snapshot(
      operation=operation,
      result_instructions=render_result_instructions(
        admitted_result_requirement
      ),
      persisted_methodology_state=previous_state,
      methodology_state_instructions=methodology_state_instructions,
      max_turns=effective_max_turns,
      timeout_seconds=effective_timeout,
      client_timeout_seconds=90,
      max_tokens=effective_max_tokens,
      cost_observation_threshold_usd=cost_observation_threshold_usd,
      max_resume_chain_depth=getattr(runner, "_max_resume_chain_depth", 3),
    )
    system_prompt = execution_snapshot.system_prompt

    operation_private_mcp_tool_ids = (
      _operation_private_mcp_tool_ids(
        profile,
        exact_tool_ids=exact_operation_tool_ids,
      )
      if named_operation and profile is not None
      else frozenset()
    )
    inherited_excluded = set(excluded_tools or set())
    inherited_excluded -= operation_private_mcp_tool_ids
    effective_excluded = _DEFAULT_EXCLUDED_TOOLS | inherited_excluded
    role_denied_tools = role_denied_tools_for_session(parent_session)
    if effective_coordinator is not None and effective_coordinator.worker_excluded_tools:
      effective_excluded = effective_excluded | effective_coordinator.worker_excluded_tools
    if profile is not None and not named_operation:
      # Unnamed delegation retains the legacy Phase-0 deny surface. Registered
      # operations instead compile authority from their exact source-owned
      # ceiling, workspace scope, live server effects, and semantic needs.
      from agent.shared.mutation_enforcement import apply_skill_mutation_mode_exclusions

      effective_excluded = apply_skill_mutation_mode_exclusions(
        profile,
        effective_excluded,
        role_denied_tools=role_denied_tools,
        local_tool_names=set(local_tool_handlers or {}) | set(_ARTIFACT_EMIT_TOOLS),
      )
    if profile is not None:
      try:
        effective_excluded |= _skill_extra_excluded_tool_names(profile)
      except ValueError as exc:
        return None, {
          "code": "invalid_skill_config",
          "message": str(exc),
        }
    effective_excluded |= role_denied_tools
    sub_local = {
      name: handler
      for name, handler in (local_tool_handlers or {}).items()
      if name not in effective_excluded
    }
    if (
      fms_rebinder is not None
      and context_research_file_id is not None
      and context_research_file_id > 0
    ):
      fms_rebinder(sub_local, context_research_file_id)
    child_excluded = set(effective_excluded)
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
        skill_state_store=skill_state_store,
        skill_state_lock=skill_state_lock,
        effective_model=effective_model,
        extract_state_update_fn=extract_state_update,
        result_response_text_fn=_result_response_text,
        logger=log,
      )

    if skill_run_id and profile is not None and agent_name:
      child_excluded = _skill_artifact_excluded_tools(effective_excluded, skill_profile=profile)
      if "emit_canvas_artifact" not in child_excluded:
        _install_emit_canvas_artifact_handler(
          sub_local=sub_local, profile=profile, skill_run_id=skill_run_id,
          context_ticker=_admitted_context_ticker,
          context_research_file_id=context_research_file_id,
          parent_session=parent_session, fallback_user_id=effective_parent_user_id,
          emit_parent_event=_emit_parent_event,
        )
      if "emit_dashboard_artifact" not in child_excluded:
        _install_emit_dashboard_artifact_handler(
          sub_local=sub_local,
          profile=profile,
          skill_run_id=skill_run_id,
          context_ticker=_admitted_context_ticker,
          context_research_file_id=context_research_file_id,
          parent_session=parent_session,
          fallback_user_id=effective_parent_user_id,
          emit_parent_event=_emit_parent_event,
        )

    child_excluded |= role_denied_tools
    extra_tool_definitions = (
      _artifact_emit_tool_definitions(set(sub_local))
      if profile is not None
      else []
    )
    if named_operation and profile is not None:
      extra_tool_definitions.extend(_operation_private_mcp_tool_definitions(
        profile=profile,
        mcp_client=mcp_client,
        exact_tool_ids=operation_private_mcp_tool_ids,
      ))

    candidate_definitions_getter = _child_tool_definitions_getter(
      runner=runner,
      mcp_client=mcp_client,
      excluded_tools=child_excluded,
      extra_tool_definitions=extra_tool_definitions,
      local_tool_handlers=sub_local,
    )
    # D-B6-2: the caller's exclusions are an INPUT to authority resolution,
    # not a subtraction applied to a grant that already exists. An excluded
    # tool is never granted, so no later pass has to take it back.
    operation_authority = admit_operation_tools(
      operation,
      grant_id=f"grant:{ordinary_task_id}",
      operation_tool_ids=exact_operation_tool_ids,
      definitions=(
        candidate_definitions_getter()
        if candidate_definitions_getter is not None
        else ()
      ),
      local_tool_handlers=sub_local,
      mcp_client=mcp_client,
      identity=execution_identity_from_session(parent_session),
      exclusions=frozenset(child_excluded),
    )
    if isinstance(operation_authority, OperationUnavailable):
      return None, {
        "code": "operation_unavailable",
        "message": operation_authority.detail,
      }
    if (
      operation.operation.name == GENERIC_EXPLORE_OPERATION_NAME
      and not operation_authority.grant.tools
    ):
      # The generic explore operation declares its evidence domains as
      # *optional* — the retired coarse row meant "at least one read route, of
      # any kind", which the granular vocabulary cannot state, and requiring
      # either domain would refuse a parent that only offers the other.  The
      # floor itself survives here, as the same by-name rule the live workflow
      # catalog already applies: an explore with no evidence route at all is a
      # visible unavailable offer, never a child sent out blind.
      declared = ", ".join(
        sorted(item.name for item in operation.required_capabilities)
      )
      return None, {
        "code": "operation_unavailable",
        "message": (
          "required semantic capability "
          f"{declared} has no compatible admitted route"
        ),
      }
    admitted_dispatch_tools = granted_tool_ids(operation_authority)
    admitted_mcp_scope = derive_dispatcher_allowlist(operation_authority)
    child_excluded |= operation_private_mcp_tool_ids - admitted_dispatch_tools
    candidate_local_names = set(sub_local)
    sub_local = {
      name: handler
      for name, handler in sub_local.items()
      if name in admitted_dispatch_tools
    }
    child_excluded |= candidate_local_names - admitted_dispatch_tools
    admitted_task_factory = _ordinary_admitted_task_factory(
      operation=operation,
      execution_snapshot=execution_snapshot,
      capability_bindings=operation_authority.bindings,
      tool_grant=operation_authority.grant,
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
        semantic_scope=profile.scope,
        context_ticker=context_ticker,
        portfolio_id=parent_portfolio_id,
        event_log_getter=lambda: sub_log,
        tool_ctx=tool_ctx,
        durable_appender=durable_skill_event_transport[0],
        durable_confirmer=durable_skill_event_transport[1],
      )
      if (
        skill_run_id
        and profile is not None
        and agent_name
        and durable_skill_event_transport is not None
      )
      else None
    )
    effective_session_inject_servers = (
      _effective_mcp_session_inject_servers(
        admitted_mcp_scope=admitted_mcp_scope,
        configured_servers=mcp_session_inject_servers,
      )
    )
    sub_dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=sub_local,
      needs_approval=needs_approval,
      approved_tool_types=(
        parent_session.approved_tool_types
        if parent_session is not None
        else None
      ),
      event_log=sub_log,
      session_id=getattr(runner, "_full_session_id", ""),
      should_avoid_permission_prompts=background,
      approval_key_qualifier=approval_key_qualifier,
      mcp_session_inject_servers=effective_session_inject_servers,
      mcp_meta_inject_servers=mcp_meta_inject_servers,
      # D-B6-1: one identity value, carrying the authority's frozen
      # ExecutionIdentity beside the authenticated session the per-user MCP
      # projection reads.
      identity=dispatch_identity(
        session=parent_session,
        execution=operation_authority.identity,
        user_id=effective_parent_user_id,
        credentials_resolver_active=credentials_resolver_active,
      ),
      role=require_inherited_role(parent_session),
      store=getattr(parent_session, "approval_store", None),
      policy=getattr(parent_session, "approval_policy", None),
      run_context=_child_run_context(
        parent_session=parent_session,
        tool_ctx=tool_ctx,
        skill_run_id=skill_run_id,
        skill_name=agent_name,
        research_file_id=_admitted_context_research_file_id(),
        user_id=effective_parent_user_id,
        session_id=getattr(runner, "_full_session_id", ""),
        approval_policy=getattr(parent_session, "approval_policy", None),
      ),
      allowed_mcp_tools_by_server=admitted_mcp_scope,
      get_tool_definitions=_child_tool_definitions_getter(
        runner=runner,
        mcp_client=mcp_client,
        excluded_tools=child_excluded,
        extra_tool_definitions=extra_tool_definitions,
        local_tool_handlers=sub_local,
        granted_tools=admitted_dispatch_tools,
      ),
      **({"commercial_work_start": commercial_work_start} if commercial_work_start is not None else {}),
      **(
        {"commercial_irreversible_recheck": commercial_irreversible_recheck}
        if commercial_irreversible_recheck is not None else {}
      ),
      **({"commercial_mcp_servers": commercial_mcp_servers} if commercial_mcp_servers is not None else {}),
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

      sub_session = None
      if parent_session is not None:
        sub_session = GatewaySession(
          session_id=_derive_sub_agent_id(parent_session, background_call_index),
          api_key_hash=parent_session.api_key_hash,
          created_at=parent_session.created_at,
          expires_at=parent_session.expires_at,
          user_id=effective_parent_user_id or parent_session.user_id,
          user_email=effective_parent_user_email,
          risk_user_id=getattr(parent_session, "risk_user_id", 0),
          role=require_inherited_role(parent_session),
          auth_config=dict(execution.auth_config),
          channel=getattr(parent_session, "channel", None),
          is_public=getattr(parent_session, "is_public", False),
        )
      try:
        await _emit_skill_run_started()
      except DurableSkillEventPersistenceError:
        return None, _durable_skill_event_persistence_error()
      return await runner.spawn_sub_agent(
        task,
        capability_execution=execution,
        skill_name=agent_name,
        logical_task=admitted_task.logical_task,
        attempt=admitted_task.attempt,
        result_requirement=admitted_result_requirement,
        result_provenance=result_provenance,
        system_prompt=system_prompt,
        dispatcher=sub_dispatcher,
        sub_session=sub_session,
        excluded_tools=child_excluded,
        max_turns=execution_snapshot.max_turns,
        timeout=execution_snapshot.timeout_seconds,
        client_timeout=execution_snapshot.client_timeout_seconds,
        max_tokens=execution_snapshot.max_tokens,
        cost_observation_threshold_usd=(
          execution_snapshot.cost_observation_threshold_usd
        ),
        skill_run_id=skill_run_id,
        call_index=background_call_index,
        parent_turn_id=parent_turn_id,
        task_entry=task_entry,
        on_sub_event=_record_sub_event,
      )

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
      elif profile is not None:
        enriched_tool_input["resumable"] = profile.resumable
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
      ) -> dict[str, Any]:
        if skill_event_emitter is None:
          raise RuntimeError("background skill lifecycle is not configured")
        return skill_event_emitter.build_result_captured_event(
          result,
          error,
        )

      return await runner._register_background_task(
        tool_input=enriched_tool_input,
        handler=_dispatch_sub_agent,
        agent_name=agent_name,
        parent_turn_id=parent_turn_id,
        capability_bind_receipt=execution.bind.receipt(),
        admitted_task=admitted_task,
        parent_result_policy=_ordinary_parent_result_policy(
          admitted_result_requirement
        ),
        task_id_override=ordinary_task_id,
        on_before_start=(lambda: on_before_background(agent_name)) if on_before_background else None,
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
    result, error = await _dispatch_sub_agent(tool_input, call_index=call_index)
    if (
      isinstance(error, dict)
      and error.get("code")
      == "durable_skill_event_persistence_failed"
    ):
      return result, error
    try:
      await _emit_skill_result_captured(result, error)
    except DurableSkillEventPersistenceError:
      return None, _durable_skill_event_persistence_error()
    await _persist_skill_state(result, error)
    if (
      error is None
      and result is not None
      and _foreground_completion_transport_ready(runner)
    ):
      try:
        canonical_result = TaskResult.model_validate(result)
        result = await _publish_foreground_completion(
          runner=runner,
          result=canonical_result,
          admitted_task=admitted_task,
          agent_name=agent_name,
          capability_bind_receipt=execution.bind.receipt(),
          parent_turn_id=parent_turn_id,
          call_index=call_index,
        )
      except Exception as exc:
        log.exception("Foreground agent completion publication failed")
        return None, {
          "code": "agent_completion_materialization_failed",
          "message": (
            "Foreground agent completed but its exact parent-readable result "
            f"could not be published: {type(exc).__name__}"
          ),
        }
    return result, error

  return _handle_run_agent


def make_resume_handler(
  runner_ref: list[Any],
  *,
  parent_session: GatewaySession | None = None,
  mcp_client: Any,
  needs_approval: Callable[..., bool] | None = None,
  needs_approval_resolver: NeedsApprovalResolver | None = None,
  mcp_session_inject_servers: AbstractSet[str] | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  user_id: str | None = None,
  credentials_resolver_active: bool = False,
  local_tool_handlers: dict[str, Any] | None = None,
  fms_rebinder: Callable[[dict[str, Any], int], None] | None = None,
  excluded_tools_resolver: ExcludedToolsResolver | None = None,
  default_max_turns: int = 15,
  default_timeout: float | None = DEFAULT_SUB_AGENT_TIMEOUT_SECONDS,
  default_max_tokens: int = 64000,
  capability_execution_resolver: CapabilityExecutionResolver,
  coordinator_config: CoordinatorConfig | None = None,
  commercial_work_start: Any | None = None,
  commercial_irreversible_recheck: Callable[[Any], None] | None = None,
  commercial_mcp_servers: frozenset[str] | None = None,
):
  if excluded_tools_resolver is None:
    raise ValueError("make_resume_handler requires excluded_tools_resolver")

  effective_coordinator = coordinator_config if coordinator_config is not None and coordinator_config.enabled else None
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

  async def _handle_resume(tool_input: dict[str, Any], **kwargs: Any):
    runner = runner_ref[0]
    if runner is None:
      return None, {"code": "internal_error", "message": "Sub-agent runner not initialized"}
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
        entry = runner._task_registry.adopt_interrupted(
          reconstructed_entry
        )
    if entry is None:
      return None, {"code": "not_found", "message": f"Unknown background task: {task_id}"}

    def _source_state_error() -> dict[str, Any] | None:
      current = runner._task_registry.get(task_id) or entry
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
    lineage = await reconstruct_child_run_lineage(durable_log, task_id)
    source_state_error = _source_state_error()
    if source_state_error is not None:
      return None, source_state_error
    metadata = entry.metadata if isinstance(entry.metadata, dict) else {}
    prior_evidence = (
      _prior_result_evidence(lineage)
      if lineage
      else SubAgentResultEvidence.empty()
    )

    async def _abandon(
      code: str,
      message: str,
    ) -> tuple[Any | None, dict[str, Any] | None]:
      return await _finalize_resume_abandoned(
        runner=runner,
        entry=entry,
        code=code,
        message=message,
        evidence=prior_evidence,
      )

    if not lineage or lineage[-1].task_id != task_id:
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has no exact durable child-run lineage",
      )
    raw_bind_receipt = getattr(entry, "capability_bind_receipt", None)
    if raw_bind_receipt is None and isinstance(getattr(entry, "metadata", None), dict):
      raw_bind_receipt = entry.metadata.get("capability_bind")
    try:
      original_bind = CapabilityBind.from_receipt(raw_bind_receipt)
    except (TypeError, ValueError) as exc:
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has no valid capability bind receipt: {exc}",
      )
    try:
      original_admitted_task = (
        entry.admitted_task
        if isinstance(getattr(entry, "admitted_task", None), AdmittedTask)
        else AdmittedTask.model_validate(metadata.get(ADMITTED_TASK_METADATA_KEY))
      )
    except Exception as exc:
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has no canonical admitted task: {exc}",
      )
    if (
      original_admitted_task.attempt.physical_task_id != task_id
      or original_admitted_task.model_bind != original_bind
      or original_admitted_task.operation.execution_class
      != original_bind.capability_id
    ):
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} admission does not match its physical/model bind",
      )
    depth = await runner._resume_chain_depth(task_id)
    source_state_error = _source_state_error()
    if source_state_error is not None:
      return None, source_state_error

    operation = original_admitted_task.operation
    ticker_required = "ticker" in operation.required_context
    trusted_research_origin_required = _tool_grant_has_investment_claim_routes(
      original_admitted_task.tool_grant,
    )
    try:
      logical_invocation_owner = _ordinary_logical_invocation_owner(
        original_admitted_task
      )
      _ticker_from_admitted_inputs(
        original_admitted_task.inputs,
        required=ticker_required,
        owner_invocation_id=logical_invocation_owner,
      )
      admitted_research_file_id = _research_file_id_from_admitted_inputs(
        original_admitted_task.inputs,
        required=trusted_research_origin_required,
        owner_invocation_id=logical_invocation_owner,
      )
    except ValueError as exc:
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has invalid admitted typed context: {exc}",
      )
    original_execution_snapshot = original_admitted_task.execution_snapshot
    if original_execution_snapshot is None:
      return await _abandon(
        "invalid_task_metadata",
        f"Task {task_id} has no immutable execution snapshot",
      )
    max_depth = original_execution_snapshot.resume_mechanics.max_chain_depth
    agent_name = operation.methodology.name
    if not operation.resumable:
      return await _abandon(
        "not_resumable",
        f"Task {task_id} was not admitted as a resumable operation",
      )
    # Resume executes the immutable admitted operation. Methodology files may
    # have changed or disappeared since the original attempt and are never
    # reloaded as execution authority.
    profile = SimpleNamespace(
      name=agent_name,
      scope=("ticker" if "ticker" in operation.required_context else None),
      mutation_mode=operation.workspace_scope,
      extra_excluded_tools=(),
      max_turns=None,
      timeout=None,
      max_tokens=None,
    )
    if operation.workspace_scope == "model_write":
      return await _abandon(
        "model_writer_resume_unsupported",
        "model-writer skills cannot be resumed; re-run the skill",
      )
    result_requirement = original_admitted_task.result_requirement
    prior_evidence = _prior_result_evidence(lineage)
    source_state_error = _source_state_error()
    if source_state_error is not None:
      return None, source_state_error

    if depth >= max_depth:
      return await _abandon(
        "max_resume_chain_depth",
        f"Resume chain depth limit reached ({max_depth}) for {task_id}",
      )

    try:
      execution = capability_execution_resolver.materialize_bind(
        original_bind
      )
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

    transcript = await reconstruct_messages_for_task(durable_log, task_id)
    orphan_ids = detect_orphan_tool_uses(transcript)
    synthetic_results = build_synthetic_tool_results(orphan_ids)
    parent_messages = await reconstruct_parent_messages(
      durable_log,
      task_id,
      datetime.datetime.now(datetime.UTC).timestamp(),
    )
    exact_resume_instruction = (
      additional_context.strip()
      if additional_context is not None and additional_context.strip()
      else "Resume the admitted task from its durable transcript."
    )
    reconstructed_messages = place_resume_messages(
      transcript,
      synthetic_results,
      parent_messages,
      exact_resume_instruction,
    )
    successor_execution_snapshot = resume_agent_execution_snapshot(
      original_execution_snapshot,
      resume_instruction=exact_resume_instruction,
    )
    system_prompt = successor_execution_snapshot.system_prompt
    effective_excluded = set(_DEFAULT_EXCLUDED_TOOLS | excluded_tools_resolver())
    if effective_coordinator is not None and effective_coordinator.worker_excluded_tools:
      effective_excluded = effective_excluded | effective_coordinator.worker_excluded_tools
    role_denied_tools = role_denied_tools_for_session(parent_session)
    effective_excluded |= role_denied_tools
    admitted_artifact_tools = {
      entry.tool_id
      for entry in original_admitted_task.tool_grant.tools
      if entry.tool_id in _ARTIFACT_EMIT_TOOLS
    }
    child_excluded = set(effective_excluded)
    child_excluded.difference_update(
      admitted_artifact_tools - role_denied_tools
    )
    sub_local = {
      name: handler
      for name, handler in (local_tool_handlers or {}).items()
      if name not in effective_excluded
    }
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
    if (
      context_research_file_id is None
      and not trusted_research_origin_required
    ):
      context_research_file_id = _extract_research_file_id_from_resume_messages(
        reconstructed_messages,
        parent_messages,
        additional_context,
      )
    if context_research_file_id == 0:
      context_research_file_id = None
    if fms_rebinder is not None and context_research_file_id is not None and context_research_file_id > 0:
      fms_rebinder(sub_local, context_research_file_id)
    resume_root_id = await runner._resume_root_task_id(task_id)
    successor_task_id = f"{resume_root_id}_r{depth + 1}"
    successor_attempt = AttemptRef(
      attempt_number=original_admitted_task.attempt.attempt_number + 1,
      attempt_id=(
        f"{successor_task_id}:attempt:"
        f"{original_admitted_task.attempt.attempt_number + 1}"
      ),
      physical_task_id=successor_task_id,
      resume_of_task_id=task_id,
    )
    successor_admitted_task_ref: list[AdmittedTask | None] = [None]

    def _successor_context_ticker() -> str | None:
      successor_admitted_task = successor_admitted_task_ref[0]
      if successor_admitted_task is None:
        raise RuntimeError("successor admission is unavailable before dispatch")
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

    if "emit_canvas_artifact" not in child_excluded:
      _install_emit_canvas_artifact_handler(
        sub_local=sub_local, profile=profile, skill_run_id=skill_run_id,
        context_ticker=_successor_context_ticker,
        context_research_file_id=context_research_file_id,
        parent_session=parent_session,
        fallback_user_id=user_id or getattr(parent_session, "user_id", None),
        emit_parent_event=_emit_parent_event,
      )
    if "emit_dashboard_artifact" not in child_excluded:
      _install_emit_dashboard_artifact_handler(
        sub_local=sub_local,
        profile=profile,
        skill_run_id=skill_run_id,
        context_ticker=_successor_context_ticker,
        context_research_file_id=context_research_file_id,
        parent_session=parent_session,
        fallback_user_id=user_id or getattr(parent_session, "user_id", None),
        emit_parent_event=_emit_parent_event,
      )
    child_excluded |= role_denied_tools
    extra_tool_definitions = _artifact_emit_tool_definitions(set(sub_local))
    try:
      admitted_tool_ids, admitted_mcp_scope = scopes_from_tool_grant(
        original_admitted_task.tool_grant,
        local_tool_handlers=sub_local,
        mcp_client=mcp_client,
      )
    except OperationToolAdmissionError as exc:
      return await _abandon("admitted_tool_route_unavailable", str(exc))
    admitted_dispatch_tools = admitted_tool_ids
    candidate_local_names = set(sub_local)
    sub_local = {
      name: handler
      for name, handler in sub_local.items()
      if name in admitted_dispatch_tools
    }
    child_excluded |= candidate_local_names - admitted_dispatch_tools
    successor_factory = _ordinary_admitted_task_factory(
      operation=operation,
      execution_snapshot=successor_execution_snapshot,
      capability_bindings=original_admitted_task.capability_bindings,
      tool_grant=reissue_tool_grant(
        original_admitted_task.tool_grant,
        grant_id=f"grant:{successor_task_id}",
      ),
      model_bind=execution.bind,
      result_requirement=result_requirement,
      objective=f"Resume interrupted delegation {task_id}",
      parent_session=parent_session,
      attempt_number=successor_attempt.attempt_number,
      resume_of_task_id=task_id,
      logical_task_override=original_admitted_task.logical_task,
      inputs=original_admitted_task.inputs,
    )
    successor_admitted_task = successor_factory(
      SimpleNamespace(task_id=successor_task_id)
    )
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
      profile=profile,
      semantic_scope=profile.scope,
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
    sub_dispatcher = ToolDispatcher(
      mcp_client=mcp_client,
      local_tool_handlers=sub_local,
      needs_approval=child_needs_approval,
      approved_tool_types=(
        parent_session.approved_tool_types
        if parent_session is not None
        else None
      ),
      event_log=sub_log,
      session_id=getattr(runner, "_full_session_id", ""),
      should_avoid_permission_prompts=True,
      mcp_session_inject_servers=effective_session_inject_servers,
      mcp_meta_inject_servers=mcp_meta_inject_servers,
      # Resume keeps its authority exactly as persisted (the reissued grant);
      # only the identity threading is new.
      identity=dispatch_identity(
        session=parent_session,
        user_id=user_id or getattr(parent_session, "user_id", None),
        credentials_resolver_active=credentials_resolver_active,
      ),
      role=require_inherited_role(parent_session),
      store=getattr(parent_session, "approval_store", None),
      policy=getattr(parent_session, "approval_policy", None),
      run_context=_child_run_context(
        parent_session=parent_session,
        tool_ctx=tool_ctx,
        skill_run_id=skill_run_id,
        skill_name=agent_name,
        research_file_id=context_research_file_id,
        user_id=user_id or getattr(parent_session, "user_id", None),
        session_id=getattr(runner, "_full_session_id", ""),
        approval_policy=getattr(parent_session, "approval_policy", None),
      ),
      allowed_mcp_tools_by_server=admitted_mcp_scope,
      get_tool_definitions=_child_tool_definitions_getter(
        runner=runner,
        mcp_client=mcp_client,
        excluded_tools=child_excluded,
        extra_tool_definitions=extra_tool_definitions,
        local_tool_handlers=sub_local,
        granted_tools=admitted_dispatch_tools,
      ),
      **({"commercial_work_start": commercial_work_start} if commercial_work_start is not None else {}),
      **(
        {"commercial_irreversible_recheck": commercial_irreversible_recheck}
        if commercial_irreversible_recheck is not None else {}
      ),
      **({"commercial_mcp_servers": commercial_mcp_servers} if commercial_mcp_servers is not None else {}),
    )

    async def _dispatch_resume(_background_input: dict[str, Any], **background_kwargs: Any):
      background_call_index = int(background_kwargs.get("call_index", call_index) or 0)
      task_entry = background_kwargs.get("task_entry")
      sub_session = None
      if parent_session is not None:
        sub_session = GatewaySession(
          session_id=_derive_sub_agent_id(parent_session, background_call_index),
          api_key_hash=parent_session.api_key_hash,
          created_at=parent_session.created_at,
          expires_at=parent_session.expires_at,
          user_id=parent_session.user_id,
          user_email=parent_session.user_email,
          risk_user_id=getattr(parent_session, "risk_user_id", 0),
          role=require_inherited_role(parent_session),
          auth_config=dict(execution.auth_config),
          channel=getattr(parent_session, "channel", None),
          is_public=getattr(parent_session, "is_public", False),
        )
      await _emit_skill_run_started()
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
        system_prompt=system_prompt,
        dispatcher=sub_dispatcher,
        sub_session=sub_session,
        excluded_tools=child_excluded,
        max_turns=successor_execution_snapshot.max_turns,
        timeout=successor_execution_snapshot.timeout_seconds,
        client_timeout=successor_execution_snapshot.client_timeout_seconds,
        max_tokens=successor_execution_snapshot.max_tokens,
        skill_run_id=skill_run_id,
        call_index=background_call_index,
        parent_turn_id=parent_turn_id,
        task_entry=task_entry,
        cost_observation_threshold_usd=(
          successor_execution_snapshot.cost_observation_threshold_usd
        ),
        on_sub_event=lambda event, _sid: sub_log.append(event),
      )

    resume_tool_input = {
      "task_id": task_id,
      "operation": operation.operation.model_dump(mode="json"),
      "resumable": True,
      "result_requirement": result_requirement.model_dump(mode="json"),
    }
    if successor_execution_snapshot.cost_observation_threshold_usd is not None:
      resume_tool_input["cost_observation_threshold_usd"] = (
        successor_execution_snapshot.cost_observation_threshold_usd
      )
    def _build_resumed_skill_result(
      _bg_task: Any,
      result: dict[str, Any] | None,
      error: dict[str, Any] | None,
    ) -> dict[str, Any]:
      return skill_event_emitter.build_result_captured_event(
        result,
        error,
      )

    result, error = await runner._register_background_task(
      tool_input=resume_tool_input,
      handler=_dispatch_resume,
      agent_name=agent_name,
      parent_turn_id=parent_turn_id,
      capability_bind_receipt=execution.bind.receipt(),
      admitted_task=successor_admitted_task,
      parent_result_policy=_ordinary_parent_result_policy(
        result_requirement
      ),
      task_id_override=successor_task_id,
      required_skill_lifecycle=(
        skill_event_emitter.required_lifecycle_metadata()
      ),
      required_skill_result_event_factory=_build_resumed_skill_result,
      required_skill_result_projector=(
        skill_event_emitter.project_result_captured
      ),
      original_task_id=task_id,
      validate_resume_source=True,
      resume_source_entry=entry,
    )
    if error is not None:
      if error.get("code") == "max_resume_chain_depth":
        return await _abandon(
          "max_resume_chain_depth",
          str(error.get("message") or "Resume chain depth limit reached"),
        )
      return None, error
    return result, None

  return _handle_resume


def make_send_message_handler(runner_ref: list[Any]):
  return _make_send_message_handler(runner_ref)


__all__ = [
  "seal_admitted_task_payload",
  "make_get_background_result_handler",
  "make_get_background_result_tool_def",
  "make_send_message_handler",
  "make_send_message_tool_def",
  "make_resume_handler",
  "make_resume_tool_def",
  "make_run_agent_handler",
  "make_run_agent_tool_def",
]
