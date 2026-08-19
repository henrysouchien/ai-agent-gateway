from __future__ import annotations

import json

from agent_workflow_contracts import (
  export_public_json_schemas,
  public_json_schemas,
  public_schema_bundle_json,
)
from jsonschema import Draft202012Validator


def test_public_schema_export_is_deterministic_and_complete(tmp_path) -> None:
  first = public_schema_bundle_json()
  second = public_schema_bundle_json()
  assert first == second
  assert first.endswith("\n")
  bundle = json.loads(first)
  assert bundle["bundle_version"] == "1.0"
  assert tuple(bundle["schemas"]) == tuple(sorted(bundle["schemas"]))
  assert {
    "admitted-task",
    "agent-completion-envelope",
    "content-handle",
    "delivery-settlement",
    "dependency-acceptance-policy",
    "task-result",
    "workflow-result",
  } <= set(bundle["schemas"])

  written = export_public_json_schemas(tmp_path)
  assert written[-1].read_text(encoding="utf-8") == first
  assert all(path.read_text(encoding="utf-8").endswith("\n") for path in written)


def test_union_schemas_have_explicit_discriminators() -> None:
  schemas = public_json_schemas()
  assert schemas["requested-data-selector"]["discriminator"]["propertyName"] == "kind"
  assert schemas["logical-task-ref"]["discriminator"]["propertyName"] == "kind"
  assert schemas["dependency-acceptance-policy"]["discriminator"]["propertyName"] == "kind"
  assert schemas["parent-result-materialization"]["discriminator"]["propertyName"] == "kind"
  assert schemas["delivery-envelope"]["discriminator"]["propertyName"] == (
    "schema_version"
  )


def test_delivery_schema_preserves_absent_version_v1_and_requires_explicit_v2() -> None:
  schemas = public_json_schemas()
  spec_schema = schemas["workflow-delivery-spec"]
  variants = spec_schema["anyOf"]
  definitions = spec_schema["$defs"]
  assert variants == [
    {"$ref": "#/$defs/WorkflowDeliverySpecV1"},
    {"$ref": "#/$defs/WorkflowDeliverySpecV2"},
  ]
  assert "schema_version" not in definitions["WorkflowDeliverySpecV1"]["properties"]
  assert "schema_version" in definitions["WorkflowDeliverySpecV2"]["required"]

  envelope_schema = schemas["delivery-envelope"]
  assert "schema_version" in envelope_schema["$defs"]["DeliveryEnvelopeV1"]["required"]
  assert "schema_version" in envelope_schema["$defs"]["DeliveryEnvelopeV2"]["required"]
  assert {"kind", "source_start_byte"} <= set(
    envelope_schema["$defs"]["DeliveryPreview"]["required"]
  )


def test_result_requirement_schema_exposes_cross_field_invariants() -> None:
  schema = public_json_schemas()["result-requirement"]
  assert schema["properties"]["mode"]["const"] == "narrative"
  assert schema["properties"]["projection"]["type"] == "null"
  assert schema["properties"]["terminal_narrative"]["const"] == "required"

  validator = Draft202012Validator(schema)
  incoherent = {
    "mode": "strict_projection",
    "projection": {"contract": {}},
    "terminal_narrative": "required",
    "outcome": {"required": False, "source": "none"},
  }
  assert list(validator.iter_errors(incoherent))


def test_public_wire_schemas_do_not_expose_storage_or_secret_fields() -> None:
  bundle = public_schema_bundle_json()
  forbidden_properties = (
    '"artifact_ref"',
    '"credential_handle"',
    '"filesystem_path"',
    '"private_key"',
    '"storage_path"',
  )
  assert all(name not in bundle for name in forbidden_properties)
