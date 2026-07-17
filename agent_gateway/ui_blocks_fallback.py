"""Normative markdown fallback projection for resolved UI-block payloads."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


def _escape(value: Any, characters: str) -> str:
  text = "" if value is None else str(value)
  return "".join("\\" + char if char in characters else char for char in text)


def _flatten(blocks: Iterable[Any]) -> Iterable[Mapping[str, Any]]:
  for block in blocks:
    if not isinstance(block, Mapping):
      continue
    children = block.get("children")
    if isinstance(children, list):
      yield from _flatten(children)
    else:
      yield block


def _scope_summary(scope: Any, escape_chars: str) -> str:
  if not isinstance(scope, Mapping) or not scope:
    return "current portfolio"
  return ", ".join(
    f"{_escape(key, escape_chars)}={_escape(scope[key], escape_chars)}"
    for key in sorted(scope)
  )


_OPTIONAL_SEGMENT = re.compile(r"\[(\s[^][]*)\]")
_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_]+)(?:\|([^}]*))?\}")


def _template(template: Any, values: Mapping[str, Any]) -> str:
  text = str(template)

  def optional(match: re.Match[str]) -> str:
    names = [item.group(1) for item in _PLACEHOLDER.finditer(match.group(1))]
    return match.group(1) if all(values.get(name) is not None for name in names) else ""

  text = _OPTIONAL_SEGMENT.sub(optional, text)

  def placeholder(match: re.Match[str]) -> str:
    value = values.get(match.group(1))
    if value is None:
      value = match.group(2) if match.group(2) is not None else ""
    return str(value)

  return _PLACEHOLDER.sub(placeholder, text)


def _table(props: Mapping[str, Any], projection: Mapping[str, Any], escape_chars: str) -> str:
  columns = props.get("columns") if isinstance(props.get("columns"), list) else []
  data = props.get("data") if isinstance(props.get("data"), list) else []
  valid_columns = [column for column in columns if isinstance(column, Mapping)]
  if not valid_columns:
    return ""
  labels = [_escape(column.get("label", column.get("key", "")), escape_chars) for column in valid_columns]
  lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
  row_cap = int(projection.get("row_cap", 20))
  for row in data[:row_cap]:
    row_mapping = row if isinstance(row, Mapping) else {}
    cells = [_escape(row_mapping.get(column.get("key"), ""), escape_chars).replace("\n", " ") for column in valid_columns]
    lines.append("| " + " | ".join(cells) + " |")
  remaining = max(0, len(data) - row_cap)
  if remaining:
    overflow = str(projection.get("overflow", "_( +{remaining} more rows )_"))
    lines.append(overflow.format(remaining=remaining))
  return "\n".join(lines)


def _project(block: Mapping[str, Any], table: Mapping[str, Any], manifest: Mapping[str, Any], escape_chars: str) -> str:
  if "view" in block:
    descriptor = manifest.get("views", {}).get("views", {}).get(block.get("view"), {})
    label = descriptor.get("label", block.get("view", "Unknown")) if isinstance(descriptor, Mapping) else block.get("view", "Unknown")
    return _template(table.get("view", "_[view: {manifest_label} ({scope_summary})]_"), {
      "manifest_label": _escape(label, escape_chars),
      "scope_summary": _scope_summary(block.get("scope"), escape_chars),
    })
  name = block.get("block")
  props = block.get("props") if isinstance(block.get("props"), Mapping) else {}

  def esc(key: str, default: Any = "") -> str:
    return _escape(props.get(key, default), escape_chars)

  if name == "metric-card":
    return _template(table.get(name, "**{label}:** {value}[ ({change})]"), {
      "label": esc("label"), "value": esc("value"),
      "change": esc("change") if props.get("change") is not None else None,
    })
  if name == "stat-pair":
    return _template(table.get(name, "**{label}:** {value}"), {"label": esc("label"), "value": esc("value")})
  if name == "status-cell":
    return _template(table.get(name, "{label}: {value}"), {"label": esc("label"), "value": esc("value")})
  if name == "insight-banner":
    return _template(table.get(name, "> **{title}**[ — {subtitle}]"), {
      "title": esc("title"),
      "subtitle": esc("subtitle") if props.get("subtitle") is not None else None,
    })
  if name == "section-header":
    return _template(table.get(name, "### {title}"), {"title": esc("title")})
  if name == "gradient-progress":
    return _template(table.get(name, "{label|Progress}: {value}%"), {
      "label": esc("label") if props.get("label") is not None else None,
      "value": esc("value"),
    })
  if name == "data-table":
    projection = table.get("data-table", {})
    return _table(props, projection if isinstance(projection, Mapping) else {}, escape_chars)
  if name == "sparkline-chart":
    data = props.get("data") if isinstance(props.get("data"), list) else []
    low = min(data) if data else "n/a"
    high = max(data) if data else "n/a"
    return _template(table.get(name, "{label|Series}: {count} points, {min}–{max}"), {
      "label": esc("label") if props.get("label") is not None else None,
      "count": len(data), "min": _escape(low, escape_chars), "max": _escape(high, escape_chars),
    })
  if isinstance(name, str) and name.startswith("sdk:"):
    return _template(table.get("sdk:*", "_[live {block_short_name}: source {source}]_"), {
      "block_short_name": _escape(name[4:], escape_chars), "source": esc("source"),
    })
  return ""


def text_fallback(payload_resolved: Mapping[str, Any], projection_table: Mapping[str, Any]) -> str:
  """Project a resolved payload according to the bundle's normative table."""

  algorithm = projection_table.get("algorithm", {})
  projections = projection_table.get("projections", {})
  if not isinstance(algorithm, Mapping) or not isinstance(projections, Mapping):
    raise ValueError("invalid fallback projection table")
  escape_chars = str(algorithm.get("markdown_escape_characters", "\\`*_[]()#|<>"))
  separator = str(algorithm.get("block_separator", "\n\n"))
  parts: list[str] = []
  if payload_resolved.get("lead_text") is not None:
    parts.append(str(payload_resolved["lead_text"]))
  manifest_value = payload_resolved.get("_manifest")
  if not isinstance(manifest_value, Mapping):
    from .ui_blocks_contract import manifest

    manifest_value = manifest()
  blocks = payload_resolved.get("blocks")
  for block in _flatten(blocks if isinstance(blocks, list) else []):
    projected = _project(block, projections, manifest_value, escape_chars)
    if projected:
      parts.append(projected)
  if payload_resolved.get("tail_text") is not None:
    parts.append(str(payload_resolved["tail_text"]))
  output = separator.join(parts)
  maximum = int(algorithm.get("output_max_chars", 8000))
  suffix = str(algorithm.get("truncation_suffix", "_( …truncated )_"))
  if len(output) > maximum:
    output = output[: max(0, maximum - len(suffix))] + suffix[:maximum]
  return output


__all__ = ["text_fallback"]
