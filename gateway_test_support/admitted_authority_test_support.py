"""Build a real sealed :class:`AdmittedTask` for settlement-site witnesses.

B-3's mechanical outcome is derived from the authority frozen at admission —
``AdmittedTask.tool_grant`` intersected with ``AdmittedTask.capability_bindings``
(design §4.3, T3-I08).  The runner-to-constructor handoff (each settlement site
passing ``admitted_task=task_entry.admitted_task``) is only witnessed if the
test drives a *real* admitted task through the runner, because a stand-in that
merely satisfies the identity check would not prove the grant and bindings
themselves travelled.

Every digest here is computed the way admission computes it, so the returned
task passes ``AdmittedTask._identity_and_grants`` unchanged.
"""

from __future__ import annotations

from agent_workflow_contracts import (
  AdmittedTask,
  AgentExecutionSnapshot,
  AgentOperationSnapshot,
  AgentResumeMechanics,
  AttemptRef,
  CapabilityBind,
  ContractRef,
  ExecuteTaskDisposition,
  LiveToolCapabilityBinding,
  LogicalTaskRef,
  OutcomePolicy,
  OutcomeRoute,
  ResultRequirement,
  TaskResultProvenance,
  ToolGrant,
  ToolGrantEntry,
  WorkspaceGrant,
  sha256_digest,
)

from agent_gateway.sub_agent import seal_admitted_task_payload


SOURCE_CAPABILITY = "research-web.read/v1"
SOURCE_TOOL_ID = "web_search"

_ADMISSION_DATE = "2026-01-01"
_RESULT_INSTRUCTIONS = "Return one terminal narrative."
_SYSTEM_PROMPT = (
  f"Admitted {_ADMISSION_DATE}. {_RESULT_INSTRUCTIONS}"
)


def _contract(name: str) -> ContractRef:
  return ContractRef(
    namespace="agent-operation",
    name=name,
    version="1",
    digest=sha256_digest({"contract": name}),
  )


def granted_source_authority(
  *,
  tool_id: str = SOURCE_TOOL_ID,
  capability: str = SOURCE_CAPABILITY,
) -> tuple[ToolGrant, tuple[LiveToolCapabilityBinding, ...]]:
  """Return one grant plus the source-capability binding that covers it."""

  grant = ToolGrant(
    grant_id="grant:admitted-authority",
    tools=(
      ToolGrantEntry(tool_id=tool_id, route_id="route-1", effect="read"),
    ),
    digest=sha256_digest({"grant": [tool_id]}),
  )
  bindings = (
    LiveToolCapabilityBinding(
      capability=capability,
      route_id="route-1",
      tool_ids=(tool_id,),
    ),
  )
  return grant, bindings


def sealed_admitted_task(
  *,
  logical_task: LogicalTaskRef,
  attempt: AttemptRef,
  result_requirement: ResultRequirement,
  tool_id: str = SOURCE_TOOL_ID,
  capability: str = SOURCE_CAPABILITY,
) -> AdmittedTask:
  """Return an admitted task granting exactly one source-reading tool."""

  grant, bindings = granted_source_authority(
    tool_id=tool_id,
    capability=capability,
  )
  model_bind = CapabilityBind(
    schema_version="1.0",
    capability_id="node.explore",
    model_key="anthropic.test-sonnet",
    provider="anthropic",
    upstream_model="claude-sonnet-4-6",
    adapter="anthropic.messages",
    protocol_profile="messages.adaptive",
    route="anthropic.public",
    effort="medium",
    credential_principal="user",
    credential_ref="credential-1",
    run_mode="interactive",
    registry_revision="registry-1",
    policy_revision="policy-1",
    selection_source="capability_default",
  )
  operation = AgentOperationSnapshot(
    operation=logical_task.operation,
    methodology=_contract("methodology"),
    prompt=_contract("prompt"),
    description="Admitted authority fixture.",
    instructions="Investigate and report.",
    execution_class="standard",
    workspace_scope="read_only",
    resumable=False,
    result_modes=("narrative",),
  )
  execution_snapshot = AgentExecutionSnapshot(
    system_prompt=_SYSTEM_PROMPT,
    admission_date=_ADMISSION_DATE,
    persisted_methodology_state=None,
    result_instructions=_RESULT_INSTRUCTIONS,
    max_turns=4,
    client_timeout_seconds=60.0,
    max_tokens=4096,
    resume_mechanics=AgentResumeMechanics(
      resumable=False,
      max_chain_depth=0,
      transcript_strategy="durable_reconstruction",
      prompt_strategy="reuse_exact",
      tool_grant_strategy="reissue_exact",
      control_message_strategy="admitted_exact",
    ),
  )
  payload = {
    "schema_version": "1.0",
    "admitted_task_id": f"admitted:{attempt.physical_task_id}",
    "logical_task": logical_task.model_dump(mode="json"),
    "attempt": attempt.model_dump(mode="json"),
    "objective": "Investigate and report.",
    "workflow_identity": None,
    "execution_disposition": ExecuteTaskDisposition().model_dump(mode="json"),
    "execution_snapshot": execution_snapshot.model_dump(mode="json"),
    "operation": operation.model_dump(mode="json"),
    "inputs": [],
    "capability_bindings": [
      binding.model_dump(mode="json") for binding in bindings
    ],
    "tool_grant": grant.model_dump(mode="json"),
    "content_read_grants": [],
    "workspace_grant": WorkspaceGrant(
      workspace_id="workspace:admitted-authority",
      scope="read_only",
    ).model_dump(mode="json"),
    "model_bind": model_bind.model_dump(mode="json"),
    "result_requirement": result_requirement.model_dump(mode="json"),
    "outcome_policy": OutcomePolicy(
      routes=(OutcomeRoute(disposition="complete", action="settle"),),
    ).model_dump(mode="json"),
    "admitted_plan_digest": None,
    "model_bind_digest": sha256_digest(model_bind),
    "capability_binding_digest": sha256_digest([
      binding.model_dump(mode="json") for binding in bindings
    ]),
    "tool_grant_digest": grant.digest,
  }
  seal_admitted_task_payload(payload)
  return AdmittedTask.model_validate(payload)


def provenance_of(admitted: AdmittedTask) -> TaskResultProvenance:
  """Return the provenance the settlement sites check identity against."""

  return TaskResultProvenance(
    admitted_task_digest=admitted.admitted_task_digest,
    model_bind_digest=admitted.model_bind_digest,
    capability_binding_digest=admitted.capability_binding_digest,
    tool_grant_digest=admitted.tool_grant_digest,
  )


__all__ = [
  "SOURCE_CAPABILITY",
  "SOURCE_TOOL_ID",
  "granted_source_authority",
  "provenance_of",
  "sealed_admitted_task",
]
