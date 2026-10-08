from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from agent_gateway.execution_snapshot import (
  build_agent_execution_snapshot,
  render_result_instructions,
)
from agent_gateway.operation_snapshot import (
  COMMON_CHILD_OPERATION_INSTRUCTIONS,
  build_agent_operation_snapshot,
  compose_operation_instructions,
  operation_listing_description,
  operation_workspace_scope,
)
from agent_gateway.skills import SkillProfile, compile_agent_operation
from agent_workflow_contracts import (
  AdmittedTask,
  AgentExecutionSnapshot,
  AgentOperationSnapshot,
  EvidencePort,
  OutcomeRequirement,
  ProjectionRequirement,
  ResultRequirement,
  canonical_json_bytes,
  sha256_digest,
)
from agent_workflow_contracts.models import AdmittedLineage


def test_legacy_profile_adapter_preserves_golden_snapshot_and_digests() -> None:
  profile = SkillProfile(
    name="golden-operation",
    version="2.1",
    system_prompt="Raw methodology.",
    agent_callable=True,
    agent_description="Golden description.",
    resumable=True,
    mutation_mode="preview",
    metadata={
      "semantic_metadata": {
        "required_context": ["ticker", "as_of", "ticker"],
        "allowed_effects": ["workspace_write"],
      },
      "evidence_ports": [
        {"name": "upstream", "min_selections": 1, "max_selections": 2},
      ],
    },
  )

  legacy_adapter = compile_agent_operation(
    profile,
    execution_class="analysis",
    resolved_instructions="Resolved instructions.",
  )
  primitive = build_agent_operation_snapshot(
    name="golden-operation",
    version="2.1",
    methodology_instructions="Raw methodology.",
    resolved_instructions="Resolved instructions.",
    description="Golden description.",
    execution_class="analysis",
    workspace_scope="workspace_write",
    required_context=("as_of", "ticker"),
    evidence_ports=(
      EvidencePort(name="upstream", min_selections=1, max_selections=2),
    ),
    resumable=True,
  )

  assert isinstance(primitive, AgentOperationSnapshot)
  assert legacy_adapter == primitive
  assert legacy_adapter.methodology.digest == (
    "sha256:81367651d4877bdf37611f33c347a62e6a5063af73714272e2c202437219444f"
  )
  assert legacy_adapter.prompt.digest == (
    "sha256:f211f570d5fc8943a23e07f0e62504c94c8f22a38bd2222f55f0059965e44a1c"
  )
  assert legacy_adapter.operation.digest == (
    "sha256:0581ae552a2cf33a46a71ee89f4843455846477960ca0994b514e10c8a6619cd"
  )


def test_builder_hashes_raw_methodology_but_executes_resolved_blocks() -> None:
  snapshot = build_agent_operation_snapshot(
    name="block-operation",
    version="1.0",
    methodology_instructions="Review {{EVIDENCE_BLOCK}}.",
    resolved_instructions="Review the supplied evidence and cite it.",
    description="Review evidence.",
    execution_class="analysis",
  )

  assert snapshot.methodology.digest == sha256_digest({
    "namespace": "skill-methodology",
    "name": "block-operation",
    "version": "1.0",
    "instructions": "Review {{EVIDENCE_BLOCK}}.",
  })
  assert snapshot.instructions == (
    "Review the supplied evidence and cite it.\n\n"
    f"{COMMON_CHILD_OPERATION_INSTRUCTIONS}"
  )
  assert "{{EVIDENCE_BLOCK}}" not in snapshot.instructions
  assert snapshot.prompt.digest == sha256_digest({
    "namespace": "agent-prompt",
    "name": "block-operation",
    "version": "1.0",
    "instructions": snapshot.instructions,
  })

  canonical = snapshot.model_dump(mode="json")
  canonical["operation"] = {
    key: value
    for key, value in canonical["operation"].items()
    if key != "digest"
  }
  assert snapshot.operation.digest == (
    "sha256:" + hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()
  )


@pytest.mark.parametrize(
  ("role", "effort"),
  [
    (None, None),
    ("custom.review-role", None),
    (None, "custom-effort"),
    ("custom.review-role", "high"),
  ],
)
def test_declared_selection_round_trips_and_enters_operation_identity(
  role: str | None,
  effort: str | None,
) -> None:
  arguments: dict[str, Any] = {
    "name": "selection-operation",
    "version": "1.0",
    "methodology_instructions": "Review the evidence.",
    "resolved_instructions": "Review the supplied evidence.",
    "description": "Review evidence.",
    "execution_class": "analysis",
  }
  historical = build_agent_operation_snapshot(**arguments)
  declared = build_agent_operation_snapshot(
    **arguments,
    role=role,
    effort=effort,
  )
  historical_wire = historical.model_dump(mode="json")
  wire = declared.model_dump(mode="json")

  assert "role" not in historical_wire
  assert "effort" not in historical_wire
  assert ("role" in wire) is (role is not None)
  assert ("effort" in wire) is (effort is not None)
  assert declared.role == role
  assert declared.effort == effort
  assert declared.execution_class == historical.execution_class
  assert declared.methodology == historical.methodology
  assert declared.prompt == historical.prompt
  if role is None and effort is None:
    assert declared.operation == historical.operation
    assert canonical_json_bytes(declared) == canonical_json_bytes(historical)
  else:
    assert declared.operation != historical.operation
    assert declared.operation.digest != historical.operation.digest

  semantic_wire = {
    **wire,
    "operation": {
      key: value
      for key, value in wire["operation"].items()
      if key != "digest"
    },
  }
  assert declared.operation.digest == sha256_digest(semantic_wire)

  encoded = canonical_json_bytes(declared)
  restored = AgentOperationSnapshot.model_validate_json(encoded)
  assert restored.role == role
  assert restored.effort == effort
  assert restored.operation == declared.operation
  assert canonical_json_bytes(restored) == encoded


def test_stop_rule_is_omitted_at_its_default_and_digested_when_declared() -> None:
  arguments: dict[str, Any] = {
    "name": "persist-operation",
    "version": "1.0",
    "methodology_instructions": "Persist each filing, then report.",
    "resolved_instructions": "Persist each filing, then report.",
    "description": "Persist filings.",
    "execution_class": "analysis",
  }
  default = build_agent_operation_snapshot(**arguments)
  declared = build_agent_operation_snapshot(
    **arguments,
    terminal_on_persist=False,
  )

  assert default.terminal_on_persist is True
  assert "terminal_on_persist" not in default.model_dump(mode="json")
  assert declared.terminal_on_persist is False
  assert declared.model_dump(mode="json")["terminal_on_persist"] is False
  assert declared.operation.digest != default.operation.digest



@pytest.mark.parametrize(
  ("overrides", "error_type", "match"),
  [
    ({"execution_class": " analysis "}, ValueError, "execution_class"),
    ({"resumable": "false"}, TypeError, "resumable"),
    ({"resumable": 0}, TypeError, "resumable"),
    ({"description": None}, TypeError, "description"),
    ({"description": 42}, TypeError, "description"),
    ({"description": b"description"}, TypeError, "description"),
    ({"description": []}, TypeError, "description"),
    ({"description": {}}, TypeError, "description"),
  ],
)
def test_builder_rejects_scalar_values_that_would_alias_after_coercion(
  overrides: dict[str, Any],
  error_type: type[Exception],
  match: str,
) -> None:
  arguments: dict[str, Any] = {
    "name": "strict-operation",
    "version": "1.0",
    "methodology_instructions": "Methodology.",
    "resolved_instructions": "Resolved methodology.",
    "description": "Strict description.",
    "execution_class": "analysis",
  }

  with pytest.raises(error_type, match=match):
    build_agent_operation_snapshot(**{**arguments, **overrides})


@pytest.mark.parametrize(
  ("field_name", "malformed_value"),
  [
    ("required_capabilities", set()),
    ("required_context", "ticker"),
    ("required_context", {"ticker"}),
    ("evidence_ports", (item for item in ())),
  ],
)
def test_builder_rejects_non_sequence_or_unordered_collection_shapes(
  field_name: str,
  malformed_value: object,
) -> None:
  arguments: dict[str, Any] = {
    "name": "strict-operation",
    "version": "1.0",
    "methodology_instructions": "Methodology.",
    "resolved_instructions": "Resolved methodology.",
    "description": "Strict description.",
    "execution_class": "analysis",
  }

  arguments[field_name] = malformed_value
  with pytest.raises(TypeError, match=field_name):
    build_agent_operation_snapshot(**arguments)


def test_composed_instructions_round_trip_past_the_retired_length_ceiling() -> None:
  prefix = "x" * 262_145
  composed = compose_operation_instructions(prefix)
  assert composed == f"{prefix}\n\n{COMMON_CHILD_OPERATION_INSTRUCTIONS}"

  operation = build_agent_operation_snapshot(
    name="long-operation",
    version="1.0",
    methodology_instructions="Raw methodology.",
    resolved_instructions=prefix,
    description="Long operation.",
    execution_class="analysis",
  )
  assert operation.instructions == composed
  assert AgentOperationSnapshot.model_validate_json(
    operation.model_dump_json()
  ) == operation

  snapshot = build_agent_execution_snapshot(
    operation=operation,
    result_instructions="Report the evidence.",
    admission_date="2026-10-08",
    max_tokens=256_001,
    cost_observation_threshold_usd=None,
    provider_tool_definitions=(),
    max_resume_chain_depth=1_001,
  )
  assert composed in snapshot.system_prompt
  assert snapshot.max_tokens == 256_001
  assert snapshot.resume_mechanics.max_chain_depth == 1_001
  assert "client_timeout_seconds" not in snapshot.model_dump(mode="json")
  assert AgentExecutionSnapshot.model_validate_json(
    snapshot.model_dump_json()
  ) == snapshot


@pytest.mark.parametrize(
  ("resumable", "depth"),
  [(True, 0), (False, 7), (True, 1_001), (False, 1_001)],
)
def test_resume_depth_is_recorded_independently_of_resumability(
  resumable: bool,
  depth: int,
) -> None:
  operation = build_agent_operation_snapshot(
    name="deep-operation",
    version="1.0",
    methodology_instructions="Review the evidence.",
    resolved_instructions="Review the evidence.",
    description="Deep operation.",
    execution_class="analysis",
    resumable=resumable,
  )
  snapshot = build_agent_execution_snapshot(
    operation=operation,
    result_instructions="Report the evidence.",
    admission_date="2026-10-08",
    max_tokens=256_001,
    cost_observation_threshold_usd=None,
    provider_tool_definitions=(),
    max_resume_chain_depth=depth,
  )
  restored = AgentExecutionSnapshot.model_validate_json(snapshot.model_dump_json())
  assert restored.resume_mechanics.resumable is resumable
  assert restored.resume_mechanics.max_chain_depth == depth

  lineage = AdmittedLineage(parent_job_id="parent-job", depth=1_001)
  assert AdmittedLineage.model_validate_json(lineage.model_dump_json()) == lineage


def test_result_metadata_does_not_refuse_declared_modes_or_projections() -> None:
  ordinary = build_agent_operation_snapshot(
    name="result-operation",
    version="1.0",
    methodology_instructions="Review the evidence.",
    resolved_instructions="Review the evidence.",
    description="Result operation.",
    execution_class="analysis",
  )
  operation = AgentOperationSnapshot.model_validate({
    **ordinary.model_dump(),
    "result_modes": ("narrative", "domain_result"),
    "projection_contracts": (ordinary.prompt,),
  })
  assert AgentOperationSnapshot.model_validate_json(
    operation.model_dump_json()
  ) == operation

  requirement = ResultRequirement(
    mode="domain_result",
    projection=ProjectionRequirement(contract=ordinary.prompt),
    terminal_narrative="optional",
    outcome=OutcomeRequirement(),
  )
  assert ResultRequirement.model_validate_json(
    requirement.model_dump_json()
  ) == requirement
  assert "complete substantive result" in render_result_instructions(requirement)


@pytest.mark.parametrize(
  ("variant", "expected_digest"),
  [
    (
      "narrative",
      "sha256:8ad1e10f103631018a6bfc8d2c50f4508fd4b3b61c9a5a9ee45ac87ce87ed79b",
    ),
    (
      "projection",
      "sha256:06194bd9d897dcde6f8ec0ed82f8f78e66720c6bcc553946558d4e509814b22d",
    ),
    (
      "long",
      "sha256:6b437abc8e1f188bd191e8d281b37a56b578ee1524f7b02b18c0d02c589855ec",
    ),
  ],
)
def test_historical_admitted_task_serialization_keeps_its_sealed_digest(
  variant: str,
  expected_digest: str,
) -> None:
  # Captured with the models at 632de9a90437. Narrative and long tasks passed
  # that revision's validators. The projection capture used model_construct
  # to preserve its serializer's bytes for the shape its validator refused.
  wire = json.loads(r"""
  {
    "schema_version": "1.2",
    "admitted_task_id": "admitted:historical",
    "logical_task": {
      "kind": "ordinary_delegation",
      "delegation_id": "historical",
      "operation": {
        "namespace": "agent-gateway", "name": "historical-operation",
        "version": "1.0",
        "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      }
    },
    "attempt": {
      "attempt_number": 1, "attempt_id": "historical:attempt:1",
      "physical_task_id": "historical", "restart_of_attempt_id": null,
      "resume_of_task_id": null
    },
    "objective": "Review historical evidence.",
    "workflow_identity": null,
    "execution_disposition": {"kind": "execute"},
    "execution_snapshot": {
      "system_prompt": "Historical methodology.\n\nCurrent date: 2026-10-08\n\nHistorical result instructions.",
      "admission_date": "2026-10-08",
      "persisted_methodology_state": null,
      "result_instructions": "Historical result instructions.",
      "max_turns": null, "timeout_seconds": null,
      "client_timeout_seconds": 120.0, "max_tokens": 64000,
      "cost_observation_threshold_usd": null,
      "provider_tool_definitions": [],
      "resume_mechanics": {
        "resumable": true, "max_chain_depth": 3,
        "transcript_strategy": "durable_reconstruction",
        "prompt_strategy": "reuse_exact",
        "tool_grant_strategy": "reissue_exact",
        "control_message_strategy": "admitted_exact"
      },
      "resume_instruction": null
    },
    "operation": {
      "operation": {
        "namespace": "agent-gateway", "name": "historical-operation",
        "version": "1.0",
        "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "methodology": {
        "namespace": "agent-gateway", "name": "historical-operation",
        "version": "1.0",
        "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "prompt": {
        "namespace": "agent-gateway", "name": "historical-operation",
        "version": "1.0",
        "digest": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
      },
      "description": "Historical operation.",
      "instructions": "Historical methodology.",
      "execution_class": "node.explore",
      "required_capabilities": [], "workspace_scope": "read_only",
      "required_context": [], "resumable": true,
      "result_modes": ["narrative"], "projection_contracts": []
    },
    "inputs": [], "capability_bindings": [],
    "tool_grant": {
      "kind": "tool_grant", "grant_id": "grant:historical", "tools": [],
      "digest": "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    },
    "tool_routes": [], "content_read_grants": [],
    "workspace_grant": {
      "kind": "workspace_grant", "workspace_id": "workspace:historical",
      "scope": "read_only"
    },
    "model_bind": {
      "schema_version": "1.0", "capability_id": "node.explore",
      "model_key": "codex.gpt-5-6-sol", "provider": "codex",
      "upstream_model": "gpt-5.6-sol", "adapter": "codex.responses",
      "protocol_profile": "codex.reasoning", "route": "codex.chatgpt",
      "effort": "high", "credential_principal": "user",
      "credential_ref": "user:test:alice:codex", "run_mode": "interactive",
      "registry_revision": "test-registry.1", "policy_revision": "test-policy.1",
      "selection_source": "internal_policy"
    },
    "result_requirement": {
      "mode": "narrative", "projection": null, "terminal_narrative": "required",
      "outcome": {"required": false, "source": "none"}
    },
    "outcome_policy": {"routes": [{"disposition": "complete", "action": "settle"}]},
    "admitted_plan_digest": null,
    "admitted_task_digest": "sha256:8ad1e10f103631018a6bfc8d2c50f4508fd4b3b61c9a5a9ee45ac87ce87ed79b",
    "model_bind_digest": "sha256:71a01a4bbf585f1a8dcdd48ae4caff44bf564902725d703cd356bb5a9f591fa3",
    "capability_binding_digest": "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "tool_grant_digest": "sha256:4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
  }
  """)
  if variant == "projection":
    wire["result_requirement"] = {
      "mode": "domain_result",
      "projection": {"contract": wire["operation"]["prompt"], "required": False},
      "terminal_narrative": "optional",
      "outcome": {"required": False, "source": "none"},
    }
  elif variant == "long":
    wire["operation"]["instructions"] = "x" * 8_192
    wire["execution_snapshot"]["system_prompt"] = (
      "x" * 8_192
      + "\n\nCurrent date: 2026-10-08\n\nHistorical result instructions."
    )
    wire["execution_snapshot"]["max_tokens"] = 256_000
    wire["execution_snapshot"]["resume_mechanics"]["max_chain_depth"] = 100
  wire["admitted_task_digest"] = expected_digest

  restored = AdmittedTask.model_validate_json(json.dumps(wire))

  assert restored.admitted_task_digest == expected_digest
  assert sha256_digest(restored.model_dump(
    mode="json",
    exclude={"admitted_task_digest"},
  )) == expected_digest
  assert canonical_json_bytes(restored.model_dump(mode="json")) == (
    canonical_json_bytes(wire)
  )


@pytest.mark.parametrize(
  ("mutation_mode", "allowed_effects", "expected"),
  [
    (None, (), "read_only"),
    ("unknown", ("external_write",), "read_only"),
    ("read-only", (), "read_only"),
    ("preview", (), "workspace_write"),
    ("apply", (), "workspace_write"),
    ("model_writer", ("artifact_write",), "workspace_write"),
    (
      "model_writer",
      ("artifact_write", "state_write"),
      "model_write",
    ),
    (
      "model_writer",
      ("artifact_write", "external_write"),
      "model_write",
    ),
    ("model_writer", (), "model_write"),
    ("thesis_writer", ("artifact_write",), "model_write"),
  ],
)
def test_workspace_scope_is_a_fail_closed_declaration_projection(
  mutation_mode: str | None,
  allowed_effects: tuple[str, ...],
  expected: str,
) -> None:
  assert operation_workspace_scope(mutation_mode, allowed_effects) == expected


def test_listing_description_preserves_short_text_and_caps_long_text() -> None:
  exact = "d" * 240
  assert operation_listing_description(exact) == exact

  long_with_space = "d" * 238 + " " + "overflow"
  listed = operation_listing_description(long_with_space)
  assert listed == "d" * 238 + "…"
  assert len(listed) <= 240

  with pytest.raises(TypeError, match="description"):
    operation_listing_description(b"not text")  # type: ignore[arg-type]
