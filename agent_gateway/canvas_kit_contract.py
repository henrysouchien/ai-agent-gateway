"""Access to the packaged Canvas Kit v1 runtime contract."""

from __future__ import annotations

from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Any

_MANIFEST_FILE = "canvas_kit_manifest.v1.json"
_COMPONENT_TYPES_FILE = "types/node_modules/@hank/canvas-kit/components/index.d.ts"
_FORMATTER_TYPES_FILE = "types/node_modules/@hank/canvas-kit/fmt.d.ts"


def packaged_contract_directory() -> Path:
  """Return the installed Canvas Kit contract directory."""

  return Path(str(files("agent_gateway") / "contracts" / "canvas-kit-v1"))


def _read_json(path: Path) -> Any:
  try:
    return json.loads(path.read_text(encoding="utf-8"))
  except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise ValueError(f"invalid Canvas Kit contract JSON: {path}") from exc


def manifest() -> dict[str, Any]:
  directory = packaged_contract_directory()
  value = _read_json(directory / _MANIFEST_FILE)
  if not isinstance(value, dict):
    raise ValueError("Canvas Kit manifest must be an object")
  return value


def contract_version() -> int:
  value = manifest().get("contract_version")
  if not isinstance(value, int):
    raise ValueError("Canvas Kit contract version is invalid")
  return value


def limits() -> dict[str, int]:
  value = manifest().get("limits")
  if not isinstance(value, dict) or not all(
    isinstance(key, str) and isinstance(item, int) and item > 0
    for key, item in value.items()
  ):
    raise ValueError("Canvas Kit limits block is invalid")
  return dict(value)


def externals_map() -> dict[str, str]:
  value = manifest().get("externals")
  if not isinstance(value, dict) or set(value) != {
    "react", "recharts", "@hank/canvas-kit"
  } or not all(isinstance(item, str) for item in value.values()):
    raise ValueError("Canvas Kit externals map is invalid")
  return dict(value)


def bundle_format() -> dict[str, Any]:
  value = manifest().get("bundle_format")
  if not isinstance(value, dict):
    raise ValueError("Canvas Kit bundle format is invalid")
  return dict(value)


def pinned_versions() -> dict[str, str]:
  value = manifest().get("pinned_versions")
  if not isinstance(value, dict) or not all(
    isinstance(key, str) and isinstance(item, str) for key, item in value.items()
  ):
    raise ValueError("Canvas Kit pinned versions are invalid")
  return dict(value)


def _compact_typescript(value: str) -> str:
  without_comments = re.sub(r"/\*\*?.*?\*/", " ", value, flags=re.DOTALL)
  return " ".join(without_comments.split())


def _type_interfaces(source: str) -> dict[str, str]:
  return {
    match.group("name"): _compact_typescript(match.group("body"))
    for match in re.finditer(
      r"export\s+interface\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)"
      r"(?:<[^{}]+>)?\s*\{(?P<body>.*?)\}",
      source,
      re.DOTALL,
    )
  }


def _component_prop_shapes(source: str, interfaces: dict[str, str]) -> dict[str, str]:
  compact = _compact_typescript(source)
  shapes: dict[str, str] = {}
  pattern = re.compile(
    r"export declare function (?P<name>[A-Za-z][A-Za-z0-9_]*)"
    r"(?:<[^()]+>)?"
    r"\((?P<parameters>.*?)\): import\(\"react/jsx-runtime\"\)",
  )
  for match in pattern.finditer(compact):
    parameters = match.group("parameters")
    if not parameters.strip():
      shapes[match.group("name")] = "{} (no props or children)"
      continue
    destructured = re.match(r"\{.*?\}\s*:\s*(?P<shape>.+)", parameters)
    if destructured is None:
      continue
    shape = destructured.group("shape").strip()
    children_shape = interfaces.get("ChildrenProps")
    if children_shape and shape == "ChildrenProps":
      shape = "{ " + children_shape + " }"
    elif children_shape and shape.startswith("ChildrenProps & "):
      shape = "{ " + children_shape + " " + shape.removeprefix("ChildrenProps & {")
    shapes[match.group("name")] = shape
  return shapes


def _formatter_signatures(source: str) -> dict[str, str]:
  compact = _compact_typescript(source)
  return {
    match.group("name"): f"({match.group('parameters')}): {match.group('return_type')}"
    for match in re.finditer(
      r"export declare function (?P<name>[A-Za-z][A-Za-z0-9_]*)"
      r"\((?P<parameters>.*?)\): (?P<return_type>[^;]+);",
      compact,
    )
  }


def _javascript_string_set(source: str, variable_name: str) -> list[str]:
  match = re.search(
    rf"const\s+{re.escape(variable_name)}\s*=\s*new Set\(\[(?P<body>.*?)\]\);",
    source,
    re.DOTALL,
  )
  if match is None:
    raise ValueError(f"Canvas module policy is missing {variable_name}")
  return re.findall(r'[\"\']([^\"\']+)[\"\']', match.group("body"))


def authoring_manifest() -> dict[str, Any]:
  """Generate the read-only authoring surface from packaged types and policy."""

  directory = packaged_contract_directory()
  try:
    components_source = (directory / _COMPONENT_TYPES_FILE).read_text(encoding="utf-8")
    formatter_source = (directory / _FORMATTER_TYPES_FILE).read_text(encoding="utf-8")
    policy_source = (
      Path(str(files("agent_gateway") / "canvas_build")) / "policy.mjs"
    ).read_text(encoding="utf-8")
  except (OSError, UnicodeError) as exc:
    raise ValueError("invalid Canvas Kit authoring sources") from exc

  interfaces = _type_interfaces(components_source)
  component_shapes = _component_prop_shapes(components_source, interfaces)
  if not component_shapes:
    raise ValueError("Canvas Kit authoring manifest has no component prop shapes")
  return {
    "contract_version": contract_version(),
    "generated_from": [
      _COMPONENT_TYPES_FILE,
      _FORMATTER_TYPES_FILE,
      "canvas_build/policy.mjs",
    ],
    "components": component_shapes,
    "types": {
      name: shape
      for name, shape in interfaces.items()
      if name != "ChildrenProps"
    },
    "formatters": _formatter_signatures(formatter_source),
    "module_policy": {
      "allowed_imports": _javascript_string_set(policy_source, "allowedImports"),
      "forbidden_identifiers": _javascript_string_set(policy_source, "forbidden"),
      "exactly_one_default_component_export": "defaults !== 1" in policy_source,
      "const_only_module_variables": "ts.NodeFlags.Const" in policy_source,
      "literal_module_data_only": all(
        marker in policy_source
        for marker in (
          "ts.isCallExpression",
          "ts.isNewExpression",
          "ts.isAwaitExpression",
        )
      ),
    },
  }


def authoring_manifest_prompt() -> str:
  """Render the generated manifest into the child-visible emitter description."""

  value = authoring_manifest()
  lines = [
    "Generated read-only Canvas Kit authoring manifest (packaged .d.ts + module policy):",
    "Component prop shapes:",
  ]
  lines.extend(
    f"- {name}: {shape}"
    for name, shape in value["components"].items()
  )
  lines.append("Related item/type shapes:")
  lines.extend(
    f"- {name}: {{ {shape} }}"
    for name, shape in value["types"].items()
  )
  lines.append("Formatters:")
  lines.extend(
    f"- {name}{signature}"
    for name, signature in value["formatters"].items()
  )
  policy = value["module_policy"]
  lines.extend([
    "Module policy:",
    f"- allowed imports: {', '.join(policy['allowed_imports'])}",
    "- exactly one default component export; module variables must be const literal data; "
    "calls, new expressions, and await are forbidden in module-scope initializers",
    f"- forbidden identifiers: {', '.join(policy['forbidden_identifiers'])}",
  ])
  return "\n".join(lines)


def component_repair_hint(source: str, line: int, fallback: str) -> str:
  """Name the nearest Kit component and its expected shape in a diagnostic."""

  source_lines = source.splitlines()
  if not source_lines:
    return fallback
  bounded_line = max(1, min(line, len(source_lines)))
  prefix = "\n".join(source_lines[:bounded_line])
  component_names = re.findall(r"<([A-Z][A-Za-z0-9_]*)\b", prefix)
  if not component_names:
    return fallback
  component = component_names[-1]
  value = authoring_manifest()
  shape = value["components"].get(component)
  if not shape:
    return fallback
  related: list[str] = []
  for type_name, type_shape in value["types"].items():
    if re.search(rf"\b{re.escape(type_name)}\b", shape):
      related.append(f"{type_name} = {{ {type_shape} }}")
  suffix = f" Expected {component} props: {shape}."
  if related:
    suffix += f" Related shape: {'; '.join(related)}."
  return fallback.rstrip() + suffix
