from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_workflow_contracts import (
  AdmittedTask,
  AgentExecutionSnapshot,
  CapabilityBind,
  ExecuteTaskDisposition,
  OrdinaryDelegationTaskRef,
  ToolGrant,
  ToolGrantEntry,
  canonical_json_bytes,
  sha256_digest,
)
from agent_gateway.skills import SkillLoader
from agent_gateway.execution_snapshot import (
  build_agent_execution_snapshot,
  provider_tool_definitions_for_grant,
  render_result_instructions,
  resume_agent_execution_snapshot,
)
from agent_gateway.sub_agent import (
  _canonical_result_requirement,
  _ordinary_admitted_task_factory,
  seal_admitted_task_payload,
)
from agent_gateway.sub_agent_helpers import make_run_agent_tool_def


def test_provider_definitions_follow_exact_grant_order_and_require_coverage() -> None:
  tools = (
    ToolGrantEntry(tool_id="tool_b", route_id="local", effect="read"),
    ToolGrantEntry(tool_id="tool_a", route_id="local", effect="read"),
  )
  grant = ToolGrant(
    grant_id="grant:ordered-definitions",
    tools=tools,
    digest=sha256_digest([
      entry.model_dump(mode="json") for entry in tools
    ]),
  )
  definitions = (
    {"name": "tool_a", "input_schema": {"type": "object"}},
    {"name": "tool_b", "input_schema": {"type": "object"}},
  )

  snapshot = provider_tool_definitions_for_grant(definitions, grant=grant)

  assert tuple(item.name for item in snapshot) == ("tool_b", "tool_a")
  with pytest.raises(ValueError, match="missing for grant: tool_b"):
    provider_tool_definitions_for_grant(definitions[:1], grant=grant)


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
    client_timeout_seconds=90,
    max_tokens=64_000,
    cost_observation_threshold_usd=50,
    provider_tool_definitions=(),
    max_resume_chain_depth=3,
  )
  first = _ordinary_admitted_task_factory(
    operation=operation,
    execution_snapshot=execution_snapshot,
    capability_bindings=(),
    tool_grant=grant,
    tool_routes=(),
    model_bind=bind,
    result_requirement=requirement,
    objective={
      "goal": "Investigate the admitted question.",
      "reference": "../selected/FY2026.md",
    },
    parent_session=None,
  )(SimpleNamespace(task_id="bg_1"))
  assert isinstance(first.logical_task, OrdinaryDelegationTaskRef)
  resumed = _ordinary_admitted_task_factory(
    operation=operation,
    execution_snapshot=resume_agent_execution_snapshot(
      execution_snapshot,
      resume_instruction="Resume the exact admitted delegation.",
    ),
    capability_bindings=(),
    tool_grant=grant,
    tool_routes=(),
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
  assert isinstance(first.objective, dict)
  assert first.objective["reference"] == "../selected/FY2026.md"
  assert first.execution_snapshot is not None
  assert first.execution_snapshot.persisted_methodology_state == {
    "reference": "/tmp/admitted-methodology.json",
  }
  assert resumed.logical_task == first.logical_task
  assert resumed.attempt.resume_of_task_id == "bg_1"

  legacy_payload = execution_snapshot.model_dump(mode="json")
  assert "max_budget_usd" not in legacy_payload
  legacy_bytes = canonical_json_bytes(legacy_payload)
  legacy_digest = sha256_digest(legacy_payload)
  parsed_legacy = AgentExecutionSnapshot.model_validate_json(legacy_bytes)
  assert parsed_legacy.max_budget_usd is None
  assert canonical_json_bytes(parsed_legacy) == legacy_bytes
  assert sha256_digest(parsed_legacy) == legacy_digest

  explicit_null = dict(legacy_payload, max_budget_usd=None)
  parsed_null = AgentExecutionSnapshot.model_validate(explicit_null)
  assert "max_budget_usd" not in parsed_null.model_dump(mode="json")
  assert canonical_json_bytes(parsed_null) == legacy_bytes

  budgeted = AgentExecutionSnapshot.model_validate({
    **legacy_payload,
    "max_budget_usd": 6.0,
  })
  assert budgeted.max_budget_usd == pytest.approx(6.0)
  assert canonical_json_bytes(budgeted) != legacy_bytes
  assert sha256_digest(budgeted) != legacy_digest
  resumed_budgeted = resume_agent_execution_snapshot(
    budgeted,
    resume_instruction="Resume the exact admitted delegation.",
  )
  assert resumed_budgeted.max_budget_usd == pytest.approx(6.0)
  budgeted_task_payload = first.model_dump(mode="json")
  budgeted_task_payload.pop("admitted_task_digest")
  budgeted_task_payload["execution_snapshot"] = budgeted.model_dump(
    mode="json"
  )
  budgeted_task = AdmittedTask.model_validate(
    seal_admitted_task_payload(budgeted_task_payload)
  )
  assert budgeted_task.admitted_task_digest != first.admitted_task_digest

  assert first.schema_version == "1.2"
  assert first.tool_routes == ()

  v12_payload = first.model_dump(mode="json")
  v12_payload.pop("admitted_task_digest")
  v12_payload["schema_version"] = "1.2"
  v12_payload["tool_routes"] = []
  v12_task = AdmittedTask.model_validate(
    seal_admitted_task_payload(dict(v12_payload))
  )
  assert v12_task.schema_version == "1.2"
  assert v12_task.tool_routes == ()
  assert v12_task.admitted_task_digest == first.admitted_task_digest

  missing_routes = dict(v12_payload)
  missing_routes.pop("tool_routes")
  with pytest.raises(ValueError, match="schema 1.2 requires tool routes"):
    AdmittedTask.model_validate(seal_admitted_task_payload(missing_routes))

  v12_without_definitions = dict(v12_payload)
  v12_without_definitions["execution_snapshot"] = dict(
    v12_payload["execution_snapshot"]
  )
  v12_without_definitions["execution_snapshot"].pop(
    "provider_tool_definitions"
  )
  with pytest.raises(ValueError, match="schema 1.2 requires provider"):
    AdmittedTask.model_validate(
      seal_admitted_task_payload(v12_without_definitions)
    )

  v11_with_routes = first.model_dump(mode="json")
  v11_with_routes.pop("admitted_task_digest")
  v11_with_routes["schema_version"] = "1.1"
  with pytest.raises(ValueError, match="schema 1.1 cannot carry tool routes"):
    AdmittedTask.model_validate(seal_admitted_task_payload(v11_with_routes))

  v11_without_definitions = first.model_dump(mode="json")
  v11_without_definitions.pop("admitted_task_digest")
  v11_without_definitions["schema_version"] = "1.1"
  v11_without_definitions.pop("tool_routes")
  v11_without_definitions["execution_snapshot"].pop(
    "provider_tool_definitions"
  )
  with pytest.raises(ValueError, match="schema 1.1 requires provider"):
    AdmittedTask.model_validate(
      seal_admitted_task_payload(v11_without_definitions)
    )

  v10_payload = first.model_dump(mode="json")
  v10_payload.pop("admitted_task_digest")
  v10_payload["schema_version"] = "1.0"
  v10_payload.pop("tool_routes")
  v10_payload["execution_snapshot"].pop("provider_tool_definitions")
  v10_task = AdmittedTask.model_validate(seal_admitted_task_payload(v10_payload))
  assert v10_task.schema_version == "1.0"
  assert v10_task.execution_snapshot is not None
  assert v10_task.execution_snapshot.provider_tool_definitions is None

  v10_with_routes = v10_task.model_dump(mode="json")
  v10_with_routes.pop("admitted_task_digest")
  v10_with_routes["tool_routes"] = []
  with pytest.raises(ValueError, match="schema 1.0 cannot carry tool routes"):
    AdmittedTask.model_validate(seal_admitted_task_payload(v10_with_routes))

  v10_with_definitions = first.model_dump(mode="json")
  v10_with_definitions.pop("admitted_task_digest")
  v10_with_definitions["schema_version"] = "1.0"
  v10_with_definitions.pop("tool_routes")
  with pytest.raises(ValueError, match="schema 1.0 cannot carry"):
    AdmittedTask.model_validate(
      seal_admitted_task_payload(v10_with_definitions)
    )

  grant_tools = [
    {"tool_id": "file_read", "route_id": "local", "effect": "read"},
    {"tool_id": "web_search", "route_id": "web", "effect": "read"},
  ]
  grant_id = "grant:route-coverage"
  grant_payload = {
    "kind": "tool_grant",
    "grant_id": grant_id,
    "tools": grant_tools,
    "digest": sha256_digest({"grant_id": grant_id, "tools": grant_tools}),
  }
  ordered_payload = dict(v12_payload)
  ordered_payload["tool_grant"] = grant_payload
  ordered_payload["tool_grant_digest"] = grant_payload["digest"]
  ordered_payload["execution_snapshot"] = {
    **v12_payload["execution_snapshot"],
    "provider_tool_definitions": [
      {"definition": {"name": "file_read"}},
      {"definition": {"name": "web_search"}},
    ],
  }
  ordered_payload["tool_routes"] = [
    {"tool_id": "file_read", "origin": "local", "server_id": None},
    {"tool_id": "web_search", "origin": "mcp", "server_id": "server-a"},
  ]
  AdmittedTask.model_validate(
    seal_admitted_task_payload(dict(ordered_payload))
  )

  reversed_routes = dict(ordered_payload)
  reversed_routes["tool_routes"] = list(
    reversed(ordered_payload["tool_routes"])
  )
  with pytest.raises(ValueError, match="exact ordered tool grant"):
    AdmittedTask.model_validate(seal_admitted_task_payload(reversed_routes))


@pytest.mark.parametrize(
  "value",
  [True, "6", 0, -1, float("nan"), float("inf")],
)
def test_execution_snapshot_rejects_invalid_budget(
  tmp_path: Path,
  value: object,
) -> None:
  (tmp_path / "explore.md").write_text(
    "---\nname: explore\nversion: '1.0'\nagent_callable: true\n"
    "agent_description: Explore.\nmutation_mode: read_only\n---\nExplore.\n",
    encoding="utf-8",
  )
  operation = SkillLoader(tmp_path).resolve_operation(None).snapshot
  requirement = _canonical_result_requirement(operation=operation)
  with pytest.raises(ValueError, match="max_budget_usd"):
    build_agent_execution_snapshot(
      operation=operation,
      result_instructions=render_result_instructions(requirement),
      admission_date="2026-08-19",
      max_turns=1,
      client_timeout_seconds=1,
      max_tokens=1,
      cost_observation_threshold_usd=None,
      provider_tool_definitions=(),
      max_resume_chain_depth=0,
      max_budget_usd=value,  # type: ignore[arg-type]
    )
