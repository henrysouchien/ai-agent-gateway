from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_workflow_contracts import (
  CapabilityBind,
  ExecuteTaskDisposition,
  ToolGrant,
  sha256_digest,
)
from agent_gateway.skills import SkillLoader
from agent_gateway.execution_snapshot import (
  build_agent_execution_snapshot,
  render_result_instructions,
  resume_agent_execution_snapshot,
)
from agent_gateway.sub_agent import (
  _canonical_result_requirement,
  _ordinary_admitted_task_factory,
)
from agent_gateway.sub_agent_helpers import make_run_agent_tool_def


def _write_operation(path: Path) -> None:
  path.write_text(
    """---
name: filing-review
version: '2.0'
agent_callable: true
agent_description: Review supplied filing evidence.
mutation_mode: read_only
semantic_metadata:
  tool_refs:
    - kind: local
      tool_id: file_read
  capability_requirements:
    - name: corpus.read/v1
      required: true
      binding_modes: [live_tool]
---
Review the evidence and distinguish facts from inference.
""",
    encoding="utf-8",
  )


def test_loader_resolves_only_full_operation_identity(tmp_path: Path) -> None:
  _write_operation(tmp_path / "filing-review.md")
  loader = SkillLoader(tmp_path)
  operation = next(
    item
    for item in loader.list_callable_operations()
    if item.snapshot.operation.name == "filing-review"
  )

  resolved = loader.resolve_operation(
    operation.snapshot.operation.model_dump(mode="json")
  )

  assert resolved.snapshot == operation.snapshot
  assert resolved.snapshot.required_capabilities[0].name == "corpus.read/v1"
  with pytest.raises(ValueError, match="full AgentOperationRef"):
    loader.resolve_operation("filing-review")  # type: ignore[arg-type]


def test_run_agent_schema_exposes_operation_and_objective(tmp_path: Path) -> None:
  _write_operation(tmp_path / "filing-review.md")

  schema = make_run_agent_tool_def(SkillLoader(tmp_path))["input_schema"]

  assert schema["required"] == ["objective"]
  assert "operation" in schema["properties"]
  assert "agent" not in schema["properties"]
  assert "task" not in schema["properties"]
  description = schema["properties"]["operation"]["description"]
  assert "agent-operation/filing-review@2.0" in description
  assert "sha256:" in description


def test_omitted_operation_selects_registered_generic_explore(tmp_path: Path) -> None:
  resolved = SkillLoader(tmp_path).resolve_operation(None)

  assert resolved.snapshot.operation.name == "explore"
  assert resolved.snapshot.execution_class == "node.explore"
  assert resolved.snapshot.workspace_scope == "read_only"
  assert [
    item.name for item in resolved.snapshot.required_capabilities
  ] == ["corpus.read/v1", "web.read/v1"]


def test_ordinary_and_resumed_admission_are_exact_execute_tasks(
  tmp_path: Path,
) -> None:
  (tmp_path / "explore.md").write_text(
    """---
name: explore
version: '1.0'
agent_callable: true
agent_description: Explore exact evidence.
mutation_mode: read_only
resumable: true
---
Explore the admitted question.
""",
    encoding="utf-8",
  )
  operation = SkillLoader(tmp_path).resolve_operation(None).snapshot
  grant_id = "grant:ordinary"
  grant = ToolGrant(
    grant_id=grant_id,
    tools=(),
    digest=sha256_digest({"grant_id": grant_id, "tools": []}),
  )
  bind = CapabilityBind(
    schema_version="1.0",
    capability_id="node.explore",
    model_key="codex.gpt-5-6-sol",
    provider="codex",
    upstream_model="gpt-5.6-sol",
    adapter="codex.responses",
    protocol_profile="codex.reasoning",
    route="codex.chatgpt",
    effort="high",
    credential_principal="user",
    credential_ref="user:test:alice:codex",
    run_mode="interactive",
    registry_revision="test-registry.1",
    policy_revision="test-policy.1",
    selection_source="internal_policy",
  )
  requirement = _canonical_result_requirement(operation=operation)
  execution_snapshot = build_agent_execution_snapshot(
    operation=operation,
    result_instructions=render_result_instructions(requirement),
    admission_date="2026-08-10",
    persisted_methodology_state={
      "reference": "/tmp/admitted-methodology.json",
    },
    methodology_state_instructions="Use the exact persisted methodology state.",
    max_turns=10,
    timeout_seconds=600,
    client_timeout_seconds=90,
    max_tokens=64_000,
    cost_observation_threshold_usd=50,
    max_resume_chain_depth=3,
  )
  first = _ordinary_admitted_task_factory(
    operation=operation,
    execution_snapshot=execution_snapshot,
    capability_bindings=(),
    tool_grant=grant,
    model_bind=bind,
    result_requirement=requirement,
    objective={
      "goal": "Investigate the admitted question.",
      "reference": "../selected/FY2026.md",
    },
    parent_session=None,
  )(SimpleNamespace(task_id="bg_1"))
  resumed = _ordinary_admitted_task_factory(
    operation=operation,
    execution_snapshot=resume_agent_execution_snapshot(
      execution_snapshot,
      resume_instruction="Resume the exact admitted delegation.",
    ),
    capability_bindings=(),
    tool_grant=grant,
    model_bind=bind,
    result_requirement=requirement,
    objective="Resume the admitted question.",
    parent_session=None,
    attempt_number=2,
    resume_of_task_id="bg_1",
    logical_task_override=first.logical_task,
  )(SimpleNamespace(task_id="bg_1_r1"))

  assert isinstance(first.execution_disposition, ExecuteTaskDisposition)
  assert isinstance(resumed.execution_disposition, ExecuteTaskDisposition)
  assert first.objective["reference"] == "../selected/FY2026.md"
  assert first.execution_snapshot is not None
  assert first.execution_snapshot.persisted_methodology_state == {
    "reference": "/tmp/admitted-methodology.json",
  }
  assert resumed.logical_task == first.logical_task
  assert resumed.attempt.resume_of_task_id == "bg_1"
