"""The Gemini wire's tool-schema subset, shared by every google.generate_content adapter.

Both the public Gemini API (``google.py``) and Cloud Code Assist
(``google_antigravity.py``) accept only the OpenAPI Schema proto's fields for
``functionDeclarations[].parameters``. A tool's raw JSON Schema ``input_schema``
is reduced here, once, so one owner holds the wire rule.
"""

from __future__ import annotations

from typing import Any

_ABSENT = object()

# Keys the Code Assist wire rejects, observed live 2026-10-11/12: ``$defs``,
# ``$ref``, list-valued ``type``, ``const``, ``exclusiveMinimum``,
# ``uniqueItems`` (31 violations over 11 tools), then ``patternProperties``
# (fms_persist, 11 turns in). OpenAPI-object keywords the Schema proto has no
# field for are dropped; ``$ref`` is inlined and ``const`` becomes a one-value
# ``enum`` so nothing meaningful is lost. The remaining draft-7 keywords with
# no proto field are dropped ahead of the next 400.
WIRE_DROPPED_SCHEMA_KEYS = frozenset({
  "$defs",
  "definitions",
  "$schema",
  "$id",
  "$comment",
  "exclusiveMinimum",
  "exclusiveMaximum",
  "uniqueItems",
  "patternProperties",
  "additionalProperties",
  "propertyNames",
  "dependencies",
  "dependentRequired",
  "dependentSchemas",
  "if",
  "then",
  "else",
  "not",
  "contains",
  "multipleOf",
  "contentEncoding",
  "contentMediaType",
  "examples",
  "readOnly",
  "writeOnly",
  "deprecated",
})


def sanitize_tool_schema(node: Any, defs: dict[str, Any]) -> Any:
  """Reduce one JSON Schema node to the wire's Schema subset (pure)."""
  if isinstance(node, list):
    return [sanitize_tool_schema(value, defs) for value in node]
  if not isinstance(node, dict):
    return node
  ref = node.get("$ref")
  if isinstance(ref, str) and ref.startswith("#/"):
    target = defs.get(ref.split("/")[-1])
    if isinstance(target, dict):
      merged = dict(target)
      merged.update({k: v for k, v in node.items() if k != "$ref"})
      return sanitize_tool_schema(merged, defs)
    node = {k: v for k, v in node.items() if k != "$ref"}
  local_defs: dict[str, Any] = {}
  for key in ("$defs", "definitions"):
    value = node.get(key)
    if isinstance(value, dict):
      local_defs.update(value)
  merged_defs = {**defs, **local_defs}
  out: dict[str, Any] = {}
  for key, value in node.items():
    if key == "$ref" or key == "const" or key in WIRE_DROPPED_SCHEMA_KEYS:
      continue
    out[key] = sanitize_tool_schema(value, merged_defs)
  const = node.get("const", _ABSENT)
  # The wire's enum is strings only: a non-string const (e.g. a boolean flag)
  # cannot be represented, so only the type guides the model there.
  if const is not _ABSENT and "enum" not in out and isinstance(const, str):
    out["enum"] = [const]
  enum = out.get("enum")
  if isinstance(enum, list):
    strings = [v for v in enum if isinstance(v, str)]
    if len(strings) == len(enum):
      out["enum"] = [sanitize_tool_schema(v, merged_defs) for v in enum]
    elif strings:
      out["enum"] = strings
    else:
      del out["enum"]
  schema_type = out.get("type")
  if isinstance(schema_type, list):
    non_null = [t for t in schema_type if t != "null"]
    out["type"] = non_null[0] if non_null else "string"
    if "null" in schema_type:
      out["nullable"] = True
  if out.get("type") == "array" and "items" not in out:
    out["items"] = {"type": "string"}
  return out



__all__ = ["WIRE_DROPPED_SCHEMA_KEYS", "sanitize_tool_schema"]
