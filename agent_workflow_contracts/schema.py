"""Deterministic JSON Schema exports for public wire roots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from .models import (
  AdmittedDataRef,
  AdmittedTask,
  AgentCompletionEnvelope,
  AgentOperationRef,
  CanonicalProjection,
  CapabilityBind,
  ContentHandle,
  ContractRef,
  DeliveryEnvelope,
  DeliverySettlement,
  DependencyAcceptancePolicy,
  LogicalTaskRef,
  OutcomePolicy,
  ParentResultMaterialization,
  ParentResultPolicy,
  PublishedOutput,
  PublishedOutputRef,
  RequestedDataRef,
  RequestedDataSelector,
  ResultRequirement,
  TaskResult,
  WorkflowDeliverySpec,
  WorkflowResult,
)

SCHEMA_BUNDLE_VERSION = "1.0"

_PUBLIC_ROOTS: tuple[tuple[str, Any], ...] = (
  ("admitted-data-ref", AdmittedDataRef),
  ("admitted-task", AdmittedTask),
  ("agent-completion-envelope", AgentCompletionEnvelope),
  ("agent-operation-ref", AgentOperationRef),
  ("canonical-projection", CanonicalProjection),
  ("capability-bind", CapabilityBind),
  ("content-handle", ContentHandle),
  ("contract-ref", ContractRef),
  ("delivery-envelope", DeliveryEnvelope),
  ("delivery-settlement", DeliverySettlement),
  ("dependency-acceptance-policy", DependencyAcceptancePolicy),
  ("logical-task-ref", LogicalTaskRef),
  ("outcome-policy", OutcomePolicy),
  ("parent-result-materialization", ParentResultMaterialization),
  ("parent-result-policy", ParentResultPolicy),
  ("published-output", PublishedOutput),
  ("published-output-ref", PublishedOutputRef),
  ("requested-data-ref", RequestedDataRef),
  ("requested-data-selector", RequestedDataSelector),
  ("result-requirement", ResultRequirement),
  ("task-result", TaskResult),
  ("workflow-delivery-spec", WorkflowDeliverySpec),
  ("workflow-result", WorkflowResult),
)


def _schema_for(root: Any) -> dict[str, Any]:
  if isinstance(root, type) and hasattr(root, "model_json_schema"):
    schema = root.model_json_schema(mode="serialization")
  else:
    schema = TypeAdapter(root).json_schema(mode="serialization")
  # Normalize insertion order too, so callers that do not sort at write time
  # still receive deterministic mappings.
  return json.loads(json.dumps(schema, sort_keys=True, separators=(",", ":")))


def public_json_schemas() -> dict[str, dict[str, Any]]:
  """Return all public root schemas in stable name and key order."""

  return {name: _schema_for(root) for name, root in _PUBLIC_ROOTS}


def public_schema_bundle() -> dict[str, Any]:
  return {
    "bundle_version": SCHEMA_BUNDLE_VERSION,
    "schemas": public_json_schemas(),
  }


def public_schema_bundle_json() -> str:
  """Return reproducible canonical JSON with one trailing newline."""

  return json.dumps(
    public_schema_bundle(),
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
  ) + "\n"


def export_public_json_schemas(destination: str | Path) -> tuple[Path, ...]:
  """Write stable per-root files and a bundle to *destination*."""

  target = Path(destination)
  target.mkdir(parents=True, exist_ok=True)
  written: list[Path] = []
  schemas = public_json_schemas()
  for name in sorted(schemas):
    path = target / f"{name}.schema.json"
    path.write_text(
      json.dumps(
        schemas[name],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
      ) + "\n",
      encoding="utf-8",
    )
    written.append(path)
  bundle_path = target / "agent-workflow-contracts.schema-bundle.json"
  bundle_path.write_text(public_schema_bundle_json(), encoding="utf-8")
  written.append(bundle_path)
  return tuple(written)


__all__ = [
  "SCHEMA_BUNDLE_VERSION",
  "export_public_json_schemas",
  "public_json_schemas",
  "public_schema_bundle",
  "public_schema_bundle_json",
]
