from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import pytest

import agent_gateway.operation_snapshot as operation_snapshot
from agent_gateway.operation_snapshot import (
  AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS,
  COMMON_CHILD_OPERATION_INSTRUCTIONS,
  build_agent_operation_snapshot,
  compose_operation_instructions,
  operation_listing_description,
  operation_workspace_scope,
)
from agent_gateway.skills import SkillProfile, compile_agent_operation
from agent_workflow_contracts import (
  AgentOperationSnapshot,
  EvidencePort,
  canonical_json_bytes,
  compose_operation_instructions as shared_compose_operation_instructions,
  sha256_digest,
)


def test_gateway_reexports_shared_operation_prompt_composer() -> None:
  assert (
    operation_snapshot.compose_operation_instructions
    is shared_compose_operation_instructions
  )


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
    "sha256:5bebd137ecafbbd59e18c4a8b6fe5e750ec176d502f27a8cf27557385b3b5447"
  )
  assert legacy_adapter.operation.digest == (
    "sha256:ae0efd4ffa77629469673a3ab2d4cc1cdd01c1095e44305c94ed490fc68647c1"
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

  with pytest.raises(TypeError, match=field_name):
    build_agent_operation_snapshot(
      **arguments,
      **{field_name: malformed_value},
    )


def test_composed_prompt_accepts_exact_snapshot_text_bound() -> None:
  separator_and_suffix_chars = len(COMMON_CHILD_OPERATION_INSTRUCTIONS) + 2
  prefix = "x" * (
    AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS - separator_and_suffix_chars
  )

  composed = compose_operation_instructions(prefix)

  assert len(composed) == AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS
  snapshot = build_agent_operation_snapshot(
    name="boundary-operation",
    version="1.0",
    methodology_instructions="Raw methodology.",
    resolved_instructions=prefix,
    description="Boundary operation.",
    execution_class="analysis",
  )
  assert len(snapshot.instructions) == AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS
  with pytest.raises(ValueError, match="262144-character wire bound"):
    compose_operation_instructions(prefix + "x")


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


def test_snapshot_primitive_has_no_application_or_parser_imports() -> None:
  source_path = (
    Path(__file__).resolve().parents[1]
    / "agent_gateway"
    / "operation_snapshot.py"
  )
  tree = ast.parse(source_path.read_text(encoding="utf-8"))
  imported_roots: set[str] = set()
  relative_imports: list[ast.ImportFrom] = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      if node.level:
        relative_imports.append(node)
      if node.module:
        imported_roots.add(node.module.split(".", 1)[0])

  assert not relative_imports
  assert imported_roots <= {
    "__future__",
    "agent_workflow_contracts",
    "collections",
    "hashlib",
    "typing",
  }
