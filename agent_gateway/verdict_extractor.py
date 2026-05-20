from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

import yaml

from .events import Confidence


_FENCED_YAML_RE = re.compile(r"```(?:yaml|yml)?\s*\n(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)
_CONFIDENCE_RE = re.compile(r"\bconfidence:\s*(HIGH|MEDIUM|LOW)\b", re.IGNORECASE)


@dataclass(frozen=True)
class VerdictPayload:
  verdict_token: str
  confidence: Confidence | None
  materiality_cushion: float | None
  one_line_summary: str


class _MalformedVerdictYaml(Exception):
  pass


def extract_verdict_payload(entries: Iterable[Any]) -> VerdictPayload | None:
  """Extract the final skill verdict from completed memory_write events.

  The completion event is authoritative because it carries
  `final_tool_result_blocks`; the original `tool_call_start` input is used as
  the content fallback because current memory_write results only echo write
  metadata back to the model.
  """
  events = [_entry_event(entry) for entry in entries]
  starts_by_id = _memory_write_starts_by_id(events)

  verdict_doc: dict[str, Any] | None = None
  summary_doc: dict[str, Any] | None = None
  summary_text: str | None = None

  for complete_event in reversed(events):
    if not _is_completed_memory_write(complete_event):
      continue
    for content in _memory_write_content_candidates(complete_event, starts_by_id):
      try:
        docs = _yaml_mappings_from_content(content)
      except _MalformedVerdictYaml:
        return None
      for doc in reversed(docs):
        if summary_doc is None:
          candidate_summary = _summary_from_doc(doc)
          if candidate_summary:
            summary_doc = doc
            summary_text = candidate_summary
        if verdict_doc is None and _explicit_verdict_token_from_doc(doc):
          verdict_doc = doc
      if verdict_doc is not None and summary_text is not None:
        break
    if verdict_doc is not None and summary_text is not None:
      break

  if verdict_doc is None and summary_doc is not None and _decision_token_from_doc(summary_doc):
    verdict_doc = summary_doc
  if verdict_doc is None:
    return None

  token = _verdict_token_from_doc(verdict_doc)
  if not token:
    return None

  confidence = _confidence_from_doc(verdict_doc)
  if confidence is None and summary_doc is not None:
    confidence = _confidence_from_doc(summary_doc)

  return VerdictPayload(
    verdict_token=token,
    confidence=confidence,
    materiality_cushion=_materiality_cushion_from_doc(verdict_doc),
    one_line_summary=summary_text or _summary_from_doc(verdict_doc) or token,
  )


def _entry_event(entry: Any) -> dict[str, Any]:
  event = getattr(entry, "event", entry)
  return dict(event) if isinstance(event, dict) else {}


def _memory_write_starts_by_id(events: list[dict[str, Any]]) -> dict[str, str]:
  starts: dict[str, str] = {}
  for event in events:
    if event.get("type") != "tool_call_start" or event.get("tool_name") != "memory_write":
      continue
    tool_call_id = str(event.get("tool_call_id") or "")
    tool_input = event.get("tool_input")
    if not tool_call_id or not isinstance(tool_input, dict):
      continue
    content = tool_input.get("content")
    if isinstance(content, str):
      starts[tool_call_id] = content
  return starts


def _is_completed_memory_write(event: dict[str, Any]) -> bool:
  if event.get("type") != "tool_call_complete" or event.get("tool_name") != "memory_write":
    return False
  if event.get("error") is not None or event.get("is_error") is True:
    return False
  return isinstance(event.get("final_tool_result_blocks"), list)


def _memory_write_content_candidates(
  complete_event: dict[str, Any],
  starts_by_id: dict[str, str],
) -> list[str]:
  candidates: list[str] = []
  for block in complete_event.get("final_tool_result_blocks") or []:
    if not isinstance(block, dict):
      continue
    candidates.extend(_strings_from_tool_result_block(block))

  tool_call_id = str(complete_event.get("tool_call_id") or "")
  start_content = starts_by_id.get(tool_call_id)
  if start_content:
    candidates.append(start_content)

  result = complete_event.get("result")
  if isinstance(result, dict):
    candidates.extend(_strings_from_mapping(result))

  return _dedupe_strings(candidates)


def _strings_from_tool_result_block(block: dict[str, Any]) -> list[str]:
  values: list[str] = []
  content = block.get("content")
  if isinstance(content, str):
    values.append(content)
    try:
      decoded = json.loads(content)
    except Exception:
      decoded = None
    if isinstance(decoded, dict):
      values.extend(_strings_from_mapping(decoded))

  text = block.get("text")
  if isinstance(text, str):
    values.append(text)
  return values


def _strings_from_mapping(payload: dict[str, Any]) -> list[str]:
  values: list[str] = []
  for key in ("content", "payload", "markdown"):
    value = payload.get(key)
    if isinstance(value, str):
      values.append(value)
  tool_input = payload.get("tool_input")
  if isinstance(tool_input, dict) and isinstance(tool_input.get("content"), str):
    values.append(tool_input["content"])
  return values


def _dedupe_strings(values: list[str]) -> list[str]:
  seen: set[str] = set()
  deduped: list[str] = []
  for value in values:
    if value in seen:
      continue
    seen.add(value)
    deduped.append(value)
  return deduped


def _yaml_mappings_from_content(content: str) -> list[dict[str, Any]]:
  if not _looks_verdict_relevant(content):
    return []

  snippets = [match.group("body") for match in _FENCED_YAML_RE.finditer(content)]
  if not snippets:
    snippets = [content]

  docs: list[dict[str, Any]] = []
  for snippet in snippets:
    if not _looks_verdict_relevant(snippet):
      continue
    try:
      parsed = yaml.safe_load(snippet)
    except yaml.YAMLError as exc:
      if _looks_like_verdict_yaml(snippet) or "verdict yaml" in content.lower():
        raise _MalformedVerdictYaml() from exc
      continue
    if isinstance(parsed, dict):
      docs.append(parsed)
  return docs


def _looks_verdict_relevant(text: str) -> bool:
  lower = text.lower()
  return any(
    token in lower
    for token in (
      "verdict",
      "verdict_token",
      "one_line_summary",
      "decision:",
      "confidence:",
    )
  )


def _looks_like_verdict_yaml(text: str) -> bool:
  lower = text.lower()
  return "verdict:" in lower or "verdict_token:" in lower


def _verdict_token_from_doc(doc: dict[str, Any]) -> str | None:
  explicit = _explicit_verdict_token_from_doc(doc)
  if explicit:
    return explicit
  return _decision_token_from_doc(doc)


def _explicit_verdict_token_from_doc(doc: dict[str, Any]) -> str | None:
  for key in ("verdict_token", "verdict"):
    value = doc.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()
  return None


def _decision_token_from_doc(doc: dict[str, Any]) -> str | None:
  decision = doc.get("decision")
  if isinstance(decision, str) and ":" in decision:
    token = decision.split(":", 1)[0].strip()
    if token:
      return token
  return None


def _summary_from_doc(doc: dict[str, Any]) -> str | None:
  for key in ("one_line_summary", "decision"):
    value = doc.get(key)
    if isinstance(value, str) and value.strip():
      return value.strip()
  return None


def _confidence_from_doc(doc: dict[str, Any]) -> Confidence | None:
  value = doc.get("confidence")
  if isinstance(value, str):
    normalized = value.strip().upper()
    if normalized in {"HIGH", "MEDIUM", "LOW"}:
      return normalized  # type: ignore[return-value]

  for key in ("rationale", "one_line_summary", "decision"):
    text = doc.get(key)
    if not isinstance(text, str):
      continue
    match = _CONFIDENCE_RE.search(text)
    if match:
      return match.group(1).upper()  # type: ignore[return-value]
  return None


def _materiality_cushion_from_doc(doc: dict[str, Any]) -> float | None:
  explicit = _coerce_float(doc.get("materiality_cushion"))
  if explicit is not None:
    return explicit

  spread_check = doc.get("spread_check")
  if not isinstance(spread_check, dict):
    return None
  spread = _coerce_float(spread_check.get("bull_minus_bear_pct_of_base"))
  threshold = _coerce_float(spread_check.get("materiality_threshold_pct"))
  if spread is None or threshold in (None, 0.0):
    return None
  return spread / threshold


def _coerce_float(value: Any) -> float | None:
  if isinstance(value, bool) or value is None:
    return None
  if isinstance(value, (int, float)):
    return float(value)
  if isinstance(value, str):
    text = value.strip()
    if not text:
      return None
    try:
      return float(text)
    except ValueError:
      return None
  return None


__all__ = [
  "VerdictPayload",
  "extract_verdict_payload",
]
