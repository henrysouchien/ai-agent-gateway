"""Pure bundle-backed validation for submitted UI-block payloads."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Iterable, Mapping

from .ui_blocks_canonical import payload_too_large
from .ui_blocks_contract import contract_version, envelope_schema, manifest


PAYLOAD_KIND = "hank_ui_blocks.v1"


class FailureCode(StrEnum):
  ENVELOPE_INVALID = "envelope_invalid"
  UNKNOWN_BLOCK = "unknown_block"
  LAYOUT_DEPTH_EXCEEDED = "layout_depth_exceeded"
  VIEW_IN_LAYOUT = "view_in_layout"
  PROPS_INVALID = "props_invalid"
  UNKNOWN_SOURCE = "unknown_source"
  FIELD_MAPPING_INVALID = "field_mapping_invalid"
  MAPPING_PRESET_INVALID = "mapping_preset_invalid"
  UNKNOWN_VIEW = "unknown_view"
  SCOPE_FORBIDDEN_FIELD = "scope_forbidden_field"
  SCOPE_MISSING_REQUIRED_FIELD = "scope_missing_required_field"
  SCOPE_INVALID_TYPE = "scope_invalid_type"
  VIEW_ARTIFACT_MISSING = "view_artifact_missing"
  VIEW_ARTIFACT_STALE = "view_artifact_stale"
  VIEW_PREFLIGHT_ERROR = "view_preflight_error"
  MANIFEST_MISMATCH = "manifest_mismatch"
  PAYLOAD_TOO_LARGE = "payload_too_large"


VIEW_ARTIFACT_MISSING = FailureCode.VIEW_ARTIFACT_MISSING.value
VIEW_ARTIFACT_STALE = FailureCode.VIEW_ARTIFACT_STALE.value
VIEW_PREFLIGHT_ERROR = FailureCode.VIEW_PREFLIGHT_ERROR.value
MANIFEST_MISMATCH = FailureCode.MANIFEST_MISMATCH.value
PAYLOAD_TOO_LARGE = FailureCode.PAYLOAD_TOO_LARGE.value


def build_payload_submitted(tool_input: Mapping[str, Any]) -> dict[str, Any]:
  """Stamp contract constants and omit optional fields absent from tool input."""

  payload: dict[str, Any] = {
    "kind": PAYLOAD_KIND,
    "contract_version": contract_version(),
  }
  if "lead_text" in tool_input:
    payload["lead_text"] = tool_input["lead_text"]
  if "tail_text" in tool_input:
    payload["tail_text"] = tool_input["tail_text"]
  if "blocks" in tool_input:
    payload["blocks"] = tool_input["blocks"]
  return payload


def _failure(stage: int, index: int | None, code: FailureCode, detail: str) -> tuple[int, dict[str, Any]]:
  return stage, {"block_index": index, "code": code.value, "detail": detail}


def _type_matches(value: Any, expected: str) -> bool:
  return {
    "object": isinstance(value, dict),
    "array": isinstance(value, list),
    "string": isinstance(value, str),
    "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    "integer": isinstance(value, int) and not isinstance(value, bool),
    "boolean": isinstance(value, bool),
    "null": value is None,
  }.get(expected, True)


def _resolve_ref(ref: str, bundle: dict[str, Any], local_root: dict[str, Any]) -> dict[str, Any]:
  if ref.startswith("#/"):
    value: Any = local_root
    parts = ref[2:].split("/")
  elif ref.startswith("primitive-blocks.json#/"):
    value = bundle["primitive_blocks"]
    parts = ref.split("#/", 1)[1].split("/")
  elif ref.startswith("data-sources.json#/"):
    value = bundle["data_sources"]
    parts = ref.split("#/", 1)[1].split("/")
  else:
    return {}
  for part in parts:
    value = value[part.replace("~1", "/").replace("~0", "~")]
  return value if isinstance(value, dict) else {}


def _schema_issues(
  value: Any,
  schema: dict[str, Any],
  bundle: dict[str, Any],
  path: str = "$",
  local_root: dict[str, Any] | None = None,
) -> list[str]:
  root = schema if local_root is None else local_root
  if "$ref" in schema:
    return _schema_issues(value, _resolve_ref(schema["$ref"], bundle, root), bundle, path, root)
  if "oneOf" in schema:
    matches = [not _schema_issues(value, option, bundle, path, root) for option in schema["oneOf"]]
    return [] if sum(matches) == 1 else [f"{path}: does not match exactly one permitted schema"]
  expected = schema.get("type")
  if expected is not None:
    expected_types = [expected] if isinstance(expected, str) else expected
    if not any(_type_matches(value, item) for item in expected_types):
      return [f"{path}: expected {' or '.join(expected_types)}"]
  if "const" in schema and value != schema["const"]:
    return [f"{path}: expected constant {schema['const']!r}"]
  if "enum" in schema and value not in schema["enum"]:
    return [f"{path}: value is outside the permitted vocabulary"]
  issues: list[str] = []
  if isinstance(value, dict):
    required = schema.get("required", [])
    issues.extend(f"{path}.{key}: required field is missing" for key in required if key not in value)
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)
    for key, item in value.items():
      if key in properties:
        issues.extend(_schema_issues(item, properties[key], bundle, f"{path}.{key}", root))
      elif additional is False:
        issues.append(f"{path}.{key}: field is not permitted")
      elif isinstance(additional, dict):
        issues.extend(_schema_issues(item, additional, bundle, f"{path}.{key}", root))
  if isinstance(value, list):
    if len(value) < schema.get("minItems", 0):
      issues.append(f"{path}: has too few items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
      issues.append(f"{path}: has too many items")
    if isinstance(schema.get("items"), dict):
      for index, item in enumerate(value):
        issues.extend(_schema_issues(item, schema["items"], bundle, f"{path}[{index}]", root))
  if isinstance(value, str) and len(value) < schema.get("minLength", 0):
    issues.append(f"{path}: string is too short")
  if isinstance(value, (int, float)) and not isinstance(value, bool):
    if "minimum" in schema and value < schema["minimum"]:
      issues.append(f"{path}: value is below minimum")
    if "maximum" in schema and value > schema["maximum"]:
      issues.append(f"{path}: value is above maximum")
  return issues


def _walk_blocks(blocks: Iterable[Any]) -> Iterable[tuple[int, Any, int, bool]]:
  for top_index, block in enumerate(blocks):
    stack = [(block, 0, False)]
    while stack:
      current, layout_depth, in_layout = stack.pop()
      yield top_index, current, layout_depth, in_layout
      if isinstance(current, dict) and isinstance(current.get("children"), list):
        for child in reversed(current["children"]):
          stack.append((child, layout_depth + 1, True))


def _stage_one(payload: Any, bundle: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
  failures: list[tuple[int, dict[str, Any]]] = []
  schema = envelope_schema()
  if not isinstance(payload, dict):
    return [_failure(1, None, FailureCode.ENVELOPE_INVALID, "$: expected object")]
  allowed = set(schema["properties"])
  for key in sorted(set(payload) - allowed):
    failures.append(_failure(1, None, FailureCode.ENVELOPE_INVALID, f"$.{key}: field is not permitted"))
  for key in schema["required"]:
    if key not in payload:
      failures.append(_failure(1, None, FailureCode.ENVELOPE_INVALID, f"$.{key}: required field is missing"))
  if payload.get("kind") != PAYLOAD_KIND:
    failures.append(_failure(1, None, FailureCode.ENVELOPE_INVALID, "$.kind: invalid contract kind"))
  if payload.get("contract_version") != contract_version():
    failures.append(_failure(1, None, FailureCode.ENVELOPE_INVALID, "$.contract_version: invalid contract version"))
  for key in ("lead_text", "tail_text"):
    value = payload.get(key)
    if key in payload and not (value is None or isinstance(value, str)):
      failures.append(_failure(1, None, FailureCode.ENVELOPE_INVALID, f"$.{key}: expected string or null"))
    elif isinstance(value, str) and len(value) > 2000:
      failures.append(_failure(1, None, FailureCode.ENVELOPE_INVALID, f"$.{key}: exceeds 2000 characters"))
  blocks = payload.get("blocks")
  if not isinstance(blocks, list) or not 1 <= len(blocks) <= 24:
    failures.append(_failure(1, None, FailureCode.ENVELOPE_INVALID, "$.blocks: expected 1 to 24 blocks"))
    return failures
  known_blocks = set(bundle["primitive_blocks"]) | set(bundle["sdk_blocks"])
  for index, block, depth, in_layout in _walk_blocks(blocks):
    path = f"$.blocks[{index}]" + (".children" if in_layout else "")
    if not isinstance(block, dict):
      failures.append(_failure(1, index, FailureCode.ENVELOPE_INVALID, f"{path}: expected object"))
      continue
    if "view" in block:
      if in_layout:
        failures.append(_failure(1, index, FailureCode.VIEW_IN_LAYOUT, f"{path}.view: views are top-level only"))
      elif set(block) != {"view", "scope"} or not isinstance(block.get("view"), str) or not isinstance(block.get("scope"), dict):
        failures.append(_failure(1, index, FailureCode.ENVELOPE_INVALID, f"{path}: invalid view envelope"))
    elif "layout" in block:
      if depth >= 2:
        failures.append(_failure(1, index, FailureCode.LAYOUT_DEPTH_EXCEEDED, f"{path}.children: layout depth exceeds contract schema"))
      if set(block) != {"layout", "children"} or block.get("layout") not in bundle["layout_rules"]["layouts"]:
        failures.append(_failure(1, index, FailureCode.ENVELOPE_INVALID, f"{path}: invalid layout envelope"))
      children = block.get("children")
      if not isinstance(children, list) or not 1 <= len(children) <= 12:
        failures.append(_failure(1, index, FailureCode.ENVELOPE_INVALID, f"{path}.children: expected 1 to 12 children"))
    elif "block" in block:
      if block.get("block") not in known_blocks:
        failures.append(_failure(1, index, FailureCode.UNKNOWN_BLOCK, f"{path}.block: unknown block {block.get('block')!r}"))
      elif set(block) != {"block", "props"} or not isinstance(block.get("props"), dict):
        failures.append(_failure(1, index, FailureCode.ENVELOPE_INVALID, f"{path}: invalid block envelope"))
    else:
      failures.append(_failure(1, index, FailureCode.ENVELOPE_INVALID, f"{path}: block has no recognized discriminator"))
  return failures


def _field_mapping_code(block_name: str, props: dict[str, Any], bundle: dict[str, Any]) -> FailureCode | None:
  source_id = props.get("source")
  source = bundle["data_sources"]["sources"].get(source_id, {})
  presets = source.get("presets", {})
  applicable = [preset for preset in presets.values() if preset.get("block") == block_name]
  if applicable:
    mapping_keys = {"field", "xKey", "yKeys", "chartType"}
    if not any(all(props.get(key) == value for key, value in preset.items() if key in mapping_keys) for preset in applicable):
      return FailureCode.MAPPING_PRESET_INVALID
    return None
  fields = source.get("fields", {})
  if block_name == "sdk:metric-grid":
    keys = [item if isinstance(item, str) else item.get("key") for item in props.get("fields", []) if isinstance(item, (str, dict))]
    scalar_types = {"string", "number", "integer", "boolean"}

    def scalar_compatible(key: Any) -> bool:
      descriptor = fields.get(key)
      if not isinstance(descriptor, dict):
        return False
      field_type = descriptor.get("type")
      if isinstance(field_type, str):
        return field_type in scalar_types
      if isinstance(field_type, list):
        declared = set(field_type)
        return bool(declared & scalar_types) and declared <= scalar_types | {"null"}
      return False

    return None if all(scalar_compatible(key) for key in keys) else FailureCode.FIELD_MAPPING_INVALID
  if block_name in {"sdk:source-table", "sdk:chart-panel"}:
    field = fields.get(props.get("field"), {})
    if field.get("type") != "array" or not isinstance(field.get("items"), dict):
      return FailureCode.FIELD_MAPPING_INVALID
    members = set(field["items"])
    if block_name == "sdk:source-table":
      keys = [item.get("key") for item in props.get("columns", []) if isinstance(item, dict)] + [props.get("rowKey")]
    else:
      keys = [props.get("xKey"), *props.get("yKeys", [])]
    return None if all(key in members for key in keys) else FailureCode.FIELD_MAPPING_INVALID
  return None


def _stage_two(payload: dict[str, Any], bundle: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
  failures: list[tuple[int, dict[str, Any]]] = []
  blocks = payload.get("blocks", [])
  if not isinstance(blocks, list):
    return failures
  admitted_views = {key: value for key, value in bundle["views"]["views"].items() if value.get("status") == "admitted"}
  visible_sources = {key for key, value in bundle["data_sources"]["sources"].items() if value.get("status") == "agent-visible"}
  for index, block, _depth, in_layout in _walk_blocks(blocks):
    if not isinstance(block, dict) or in_layout and "view" in block:
      continue
    if "view" in block:
      if block.get("view") not in admitted_views:
        failures.append(_failure(2, index, FailureCode.UNKNOWN_VIEW, f"unknown view {block.get('view')!r}"))
      continue
    name = block.get("block")
    props = block.get("props")
    schemas = bundle["primitive_blocks"] | bundle["sdk_blocks"]
    if name not in schemas or not isinstance(props, dict):
      continue
    issues = _schema_issues(props, schemas[name], bundle, "$.props")
    if name.startswith("sdk:") and props.get("source") not in visible_sources:
      failures.append(_failure(2, index, FailureCode.UNKNOWN_SOURCE, f"unknown agent-visible source {props.get('source')!r}"))
      continue
    if issues:
      failures.append(_failure(2, index, FailureCode.PROPS_INVALID, "; ".join(sorted(issues))))
      continue
    if name.startswith("sdk:"):
      mapping_code = _field_mapping_code(name, props, bundle)
      if mapping_code is not None:
        failures.append(_failure(2, index, mapping_code, "SDK field mapping is not declared by the selected source"))
  return failures


def _stage_three(payload: dict[str, Any], bundle: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
  failures: list[tuple[int, dict[str, Any]]] = []
  definitions = bundle["views"]
  views = definitions["views"]
  for index, block, _depth, in_layout in _walk_blocks(payload.get("blocks", [])):
    if in_layout or not isinstance(block, dict) or "view" not in block:
      continue
    descriptor = views.get(block.get("view"))
    scope = block.get("scope")
    if not isinstance(descriptor, dict) or descriptor.get("status") != "admitted" or not isinstance(scope, dict):
      continue
    schema = _resolve_ref(descriptor["scope_input"]["$ref"], bundle, definitions)
    allowed = set(schema.get("properties", {}))
    forbidden = sorted(set(scope) - allowed)
    if forbidden:
      failures.append(_failure(3, index, FailureCode.SCOPE_FORBIDDEN_FIELD, f"scope fields are forbidden: {', '.join(forbidden)}"))
    missing = sorted(set(schema.get("required", [])) - set(scope))
    if missing:
      failures.append(_failure(3, index, FailureCode.SCOPE_MISSING_REQUIRED_FIELD, f"scope fields are required: {', '.join(missing)}"))
    for key in sorted(set(scope) & allowed):
      if _schema_issues(scope[key], schema["properties"][key], bundle, f"$.scope.{key}"):
        failures.append(_failure(3, index, FailureCode.SCOPE_INVALID_TYPE, f"scope field {key!r} has an invalid type or value"))
  return failures


def validate_payload(payload_submitted: Any) -> list[dict[str, Any]]:
  """Run the size/envelope/vocabulary/scope ladder in deterministic order."""

  staged: list[tuple[int, dict[str, Any]]] = []
  if payload_too_large(payload_submitted):
    staged.append(_failure(0, None, FailureCode.PAYLOAD_TOO_LARGE, "canonical submitted payload exceeds 32768 UTF-8 bytes"))
  bundle = manifest()
  staged.extend(_stage_one(payload_submitted, bundle))
  if isinstance(payload_submitted, dict):
    staged.extend(_stage_two(payload_submitted, bundle))
    staged.extend(_stage_three(payload_submitted, bundle))
  staged.sort(key=lambda item: (item[0], -1 if item[1]["block_index"] is None else item[1]["block_index"], item[1]["code"], item[1]["detail"]))
  seen: set[tuple[Any, ...]] = set()
  result: list[dict[str, Any]] = []
  for _stage, failure in staged:
    identity = (failure["block_index"], failure["code"], failure["detail"])
    if identity not in seen:
      seen.add(identity)
      result.append(failure)
  return result


validate_payload_submitted = validate_payload


__all__ = [
  "FailureCode",
  "MANIFEST_MISMATCH",
  "PAYLOAD_KIND",
  "PAYLOAD_TOO_LARGE",
  "VIEW_ARTIFACT_MISSING",
  "VIEW_ARTIFACT_STALE",
  "VIEW_PREFLIGHT_ERROR",
  "build_payload_submitted",
  "validate_payload",
  "validate_payload_submitted",
]
