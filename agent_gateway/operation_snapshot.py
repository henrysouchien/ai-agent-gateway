"""Pure construction helpers for immutable child-operation snapshots.

This module depends only on the shared wire contracts.  Skill parsers and
application-owned source models adapt their declarations into these scalar and
wire arguments at their own boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from agent_workflow_contracts import (
  AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS,
  AgentOperationRef,
  AgentOperationSnapshot,
  COMMON_CHILD_OPERATION_INSTRUCTIONS,
  ContractRef,
  EvidencePort,
  SemanticCapabilityRequirement,
  canonical_json_bytes,
  compose_operation_instructions,
  sha256_digest,
)


OPERATION_LISTING_DESCRIPTION_MAX_CHARS = 240
WorkspaceScope = Literal["read_only", "workspace_write", "model_write"]


def _snapshot_sequence(value: object, *, field_name: str) -> tuple[Any, ...]:
  if not isinstance(value, Sequence) or isinstance(
    value,
    (str, bytes, bytearray),
  ):
    raise TypeError(f"{field_name} must be an ordered sequence")
  return tuple(value)


def operation_workspace_scope(
  mutation_mode: str | None,
  allowed_effects: Iterable[str] = (),
) -> WorkspaceScope:
  """Project declaration mechanics into the child workspace authority."""

  mode = str(mutation_mode or "read_only").strip().lower().replace("-", "_")
  if mode == "read_only":
    return "read_only"
  if mode in {"preview", "apply"}:
    return "workspace_write"
  if mode in {"model_writer", "thesis_writer"}:
    effects = frozenset(
      str(effect).strip().lower()
      for effect in allowed_effects
      if str(effect).strip()
    )
    if (
      mode == "model_writer"
      and "artifact_write" in effects
      and "state_write" not in effects
      and "external_write" not in effects
    ):
      return "workspace_write"
    return "model_write"
  # Missing and unknown legacy metadata must not grant write authority.
  return "read_only"


def operation_listing_description(description: str) -> str:
  """Return the deterministic, display-only operation description prefix."""

  if type(description) is not str:
    raise TypeError("operation description must be a string")
  if len(description) <= OPERATION_LISTING_DESCRIPTION_MAX_CHARS:
    return description
  return (
    description[: OPERATION_LISTING_DESCRIPTION_MAX_CHARS - 1].rstrip()
    + "…"
  )


def _contract_ref(
  *,
  namespace: str,
  name: str,
  version: str,
  instructions: str,
) -> ContractRef:
  return ContractRef(
    namespace=namespace,
    name=name,
    version=version,
    digest=sha256_digest({
      "namespace": namespace,
      "name": name,
      "version": version,
      "instructions": instructions,
    }),
  )


def _operation_semantic_digest(payload: dict[str, Any]) -> str:
  operation = payload.get("operation")
  if not isinstance(operation, dict):
    raise ValueError("operation semantic payload requires an operation object")
  canonical = {
    **payload,
    "operation": {
      key: value for key, value in operation.items() if key != "digest"
    },
  }
  return "sha256:" + hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def build_agent_operation_snapshot(
  *,
  name: str,
  version: str,
  methodology_instructions: str,
  resolved_instructions: str,
  description: str,
  execution_class: str,
  required_capabilities: Sequence[SemanticCapabilityRequirement] = (),
  workspace_scope: WorkspaceScope = "read_only",
  required_context: Sequence[str] = (),
  evidence_ports: Sequence[EvidencePort] = (),
  resumable: bool = False,
) -> AgentOperationSnapshot:
  """Build the canonical immutable snapshot from scalar and wire values."""

  for field_name, value in (
    ("name", name),
    ("version", version),
    ("description", description),
    ("execution_class", execution_class),
  ):
    if type(value) is not str:
      raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
      raise ValueError(f"{field_name} must be a canonical non-empty string")
  if type(resumable) is not bool:
    raise TypeError("resumable must be a bool")
  if workspace_scope not in {"read_only", "workspace_write", "model_write"}:
    raise ValueError("workspace_scope is invalid")
  if type(methodology_instructions) is not str:
    raise TypeError("methodology instructions must be a string")
  methodology_text = methodology_instructions.strip()
  if not methodology_text:
    raise ValueError("methodology instructions must be non-empty")
  capabilities = _snapshot_sequence(
    required_capabilities,
    field_name="required_capabilities",
  )
  if any(
    type(requirement) is not SemanticCapabilityRequirement
    for requirement in capabilities
  ):
    raise TypeError(
      "required_capabilities must contain exact SemanticCapabilityRequirement values"
    )
  context_names = _snapshot_sequence(
    required_context,
    field_name="required_context",
  )
  if any(type(context_name) is not str for context_name in context_names):
    raise TypeError("required_context must contain strings")
  ports = _snapshot_sequence(evidence_ports, field_name="evidence_ports")
  if any(type(port) is not EvidencePort for port in ports):
    raise TypeError("evidence_ports must contain exact EvidencePort values")
  prompt_text = compose_operation_instructions(resolved_instructions)
  methodology = _contract_ref(
    namespace="skill-methodology",
    name=name,
    version=version,
    instructions=methodology_text,
  )
  prompt = _contract_ref(
    namespace="agent-prompt",
    name=name,
    version=version,
    instructions=prompt_text,
  )
  placeholder = AgentOperationSnapshot(
    operation=AgentOperationRef(
      namespace="agent-operation",
      name=name,
      version=version,
      digest="sha256:" + "0" * 64,
    ),
    methodology=methodology,
    prompt=prompt,
    description=description,
    instructions=prompt_text,
    execution_class=execution_class,
    required_capabilities=capabilities,
    workspace_scope=workspace_scope,
    required_context=context_names,
    evidence_ports=ports,
    resumable=resumable,
    result_modes=("narrative",),
    projection_contracts=(),
  )
  payload: dict[str, Any] = placeholder.model_dump(mode="json")
  operation = AgentOperationRef(
    namespace="agent-operation",
    name=name,
    version=version,
    digest=_operation_semantic_digest(payload),
  )
  return AgentOperationSnapshot.model_validate({
    **payload,
    "operation": operation.model_dump(mode="json"),
  })


__all__ = [
  "AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS",
  "COMMON_CHILD_OPERATION_INSTRUCTIONS",
  "OPERATION_LISTING_DESCRIPTION_MAX_CHARS",
  "WorkspaceScope",
  "build_agent_operation_snapshot",
  "compose_operation_instructions",
  "operation_listing_description",
  "operation_workspace_scope",
]
