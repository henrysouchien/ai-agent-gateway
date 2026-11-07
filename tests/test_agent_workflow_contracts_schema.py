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




def test_result_requirement_schema_preserves_declared_result_metadata() -> None:
  schema = public_json_schemas()["result-requirement"]

  validator = Draft202012Validator(schema)
  declared = {
    "mode": "domain_result",
    "projection": {
      "contract": {
        "namespace": "agent-gateway",
        "name": "result-projection",
        "version": "1.0",
        "digest": "sha256:" + "a" * 64,
      },
      "required": False,
    },
    "terminal_narrative": "optional",
    "outcome": {"required": False, "source": "none"},
  }
  assert list(validator.iter_errors(declared)) == []


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
