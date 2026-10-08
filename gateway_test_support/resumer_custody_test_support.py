"""Canonical continuation admissions and durable registrations for read-custody tests."""

from __future__ import annotations

from typing import Any

from agent_workflow_contracts import (
  AdmittedTask,
  AgentOperationRef,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  sha256_digest,
)
from agent_workflow_contracts.models import AdmittedContextWire

from agent_gateway.agent_session_log_records import EVENT_SCHEMA_VERSION
from agent_gateway.runner_background_tasks import (
  task_correlation_payload,
  task_registered_event_payload,
)
from agent_gateway.sub_agent import seal_admitted_task_payload
from agent_gateway.task_registry import TaskEntry

from .admitted_authority_test_support import sealed_admitted_task


def canonical_read_custody_task(
  task_id: str,
  *,
  parent_job_id: str = "original-owner",
  predecessor: AdmittedTask | None = None,
  resumer_job_id: str | None = None,
  tenant_storage_id: str = "alice",
) -> AdmittedTask:
  """Seal a real admitted task, retaining the predecessor's historical lineage."""
  source = predecessor or sealed_admitted_task(
    logical_task=OrdinaryDelegationTaskRef(
      delegation_id=f"delegation:{task_id}",
      operation=AgentOperationRef(
        namespace="agent-operation",
        name="research",
        version="1",
        digest=sha256_digest({"operation": "research"}),
      ),
    ),
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id=f"attempt:{task_id}",
      physical_task_id=task_id,
    ),
    result_requirement=ResultRequirement(outcome=OutcomeRequirement()),
  )
  payload = source.model_dump(mode="json", exclude={"admitted_task_digest"})
  payload["admitted_task_id"] = f"admitted:{task_id}"
  payload["attempt"] = AttemptRef(
    attempt_number=source.attempt.attempt_number + 1 if predecessor is not None else 1,
    attempt_id=f"attempt:{task_id}",
    physical_task_id=task_id,
    resume_of_task_id=predecessor.attempt.physical_task_id if predecessor is not None else None,
    resumer_job_id=resumer_job_id,
  ).model_dump(mode="json")
  snapshot = payload["execution_snapshot"]
  if predecessor is None:
    assert source.model_bind is not None
    payload["schema_version"] = "1.2"
    payload["operation"]["execution_class"] = source.model_bind.capability_id
    payload["operation"]["resumable"] = True
    snapshot["resume_mechanics"]["resumable"] = True
    snapshot["resume_mechanics"]["max_chain_depth"] = 3
    snapshot["provider_tool_definitions"] = [
      {"definition": {
        "name": "web_search",
        "description": "Search public sources.",
        "input_schema": {"type": "object"},
      }},
    ]
    payload["tool_routes"] = [
      {"tool_id": "web_search", "origin": "local", "server_id": None},
    ]
    snapshot["admitted_context"] = AdmittedContextWire.model_validate({
      "tenant": {"tenant_id": None, "storage_id": tenant_storage_id, "role": "owner"},
      "research_binding": {},
      "capability_bind": payload["model_bind"],
      "children": [],
      "tool_grant": ["web_search"],
      "budget_share": None,
      "env_projection": [],
      "approval_route": "no_approval_route",
      "handle_grants": [],
      "placement": "in_process",
      "lineage": {"parent_job_id": parent_job_id, "depth": 1},
    }).model_dump(mode="json")
  else:
    snapshot["resume_instruction"] = f"Continue {predecessor.attempt.physical_task_id}"
  seal_admitted_task_payload(payload)
  return AdmittedTask.model_validate(payload)


def custody_registration(
  admitted: AdmittedTask,
  *,
  parent_session_id: str = "session-alias",
) -> dict[str, Any]:
  """Use the real registration producer without treating its aliases as custody."""
  assert admitted.model_bind is not None
  entry = TaskEntry(
    task_id=admitted.attempt.physical_task_id,
    task_type="background_agent",
    agent_name="research",
    started_at=1.0,
    original_task_id=admitted.attempt.resume_of_task_id,
    admitted_task=admitted,
    capability_bind_receipt=admitted.model_bind.model_dump(mode="json"),
    metadata={"admitted_task": admitted.model_dump(mode="json")},
  )
  return {
    "event_schema_version": EVENT_SCHEMA_VERSION,
    **task_registered_event_payload(
      entry,
      correlation_payload=task_correlation_payload(entry, runner_id="runner-alias", role="owner"),
      agent_name="research",
      parent_session_id=parent_session_id,
    ),
  }
