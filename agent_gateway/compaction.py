from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from .agent_session_log import AgentSessionLog, LogEntry
from .capability_execution import BoundCapabilityExecution
from .child_result_trust import UNTRUSTED_CHILD_RESULTS_POLICY
from .product_config import gateway_product_id
from .provider_summarize import DEFAULT_SUMMARY_SYSTEM_PROMPT, _provider_summarize
from .secret_boundary import (
  sanitize_boundary_value,
  sanitize_tool_event,
  sanitization_failure_tool_input,
)


LOGGER = logging.getLogger("agent_gateway.compaction")

SummaryFn = Callable[[str], Awaitable[str]]
SUMMARY_PROMPT_CHAR_BUDGET = 600_000
_SUMMARY_STRING_LIMIT = 2_000
_SUMMARY_LIST_PREVIEW_ITEMS = 3
_SUMMARY_MAX_DEPTH = 4
_SUMMARY_BULK_KEYS = {
  "articles",
  "content",
  "data",
  "documents",
  "filings",
  "final_tool_result_blocks",
  "holdings",
  "markdown",
  "metrics",
  "positions",
  "raw",
  "rows",
  "sections",
  "tables",
  "text",
}


def _redact_tool_input_for_summary(tool_name: str, tool_input: Any) -> Any:
  if not isinstance(tool_input, dict):
    return sanitize_boundary_value(tool_input, sink="compaction_prompt")
  try:
    from agent.shared.tool_redaction import get_audit_hmac_secret, redact_tool_input

    return redact_tool_input(tool_name, tool_input, deployment_secret=get_audit_hmac_secret())
  except Exception:
    return sanitization_failure_tool_input()


@dataclass(frozen=True)
class _SummaryPromptSlice:
  text: str
  covered_to_seq: int
  rendered_event_count: int


def _isoformat_timestamp(timestamp: float) -> str:
  return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _stringify(value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _omitted_marker(value: Any) -> str:
  if isinstance(value, str):
    return f"<omitted text chars={len(value)}>"
  if isinstance(value, dict):
    return f"<omitted object keys={len(value)}>"
  if isinstance(value, (list, tuple)):
    return f"<omitted list items={len(value)}>"
  return f"<omitted {type(value).__name__}>"


def _compact_for_summary(value: Any, *, depth: int = 0) -> Any:
  """Keep summary inputs useful without replaying full bulk tool payloads."""
  if depth >= _SUMMARY_MAX_DEPTH:
    return _omitted_marker(value)
  if isinstance(value, dict):
    compacted: dict[str, Any] = {}
    for key, item in value.items():
      key_text = str(key)
      if key_text in _SUMMARY_BULK_KEYS:
        compacted[key_text] = _omitted_marker(item)
      else:
        compacted[key_text] = _compact_for_summary(item, depth=depth + 1)
    return compacted
  if isinstance(value, (list, tuple)):
    if len(value) > _SUMMARY_LIST_PREVIEW_ITEMS:
      preview = [
        _compact_for_summary(item, depth=depth + 1)
        for item in value[:_SUMMARY_LIST_PREVIEW_ITEMS]
      ]
      preview.append(f"<omitted {len(value) - _SUMMARY_LIST_PREVIEW_ITEMS} item(s)>")
      return preview
    return [_compact_for_summary(item, depth=depth + 1) for item in value]
  if isinstance(value, str) and len(value) > _SUMMARY_STRING_LIMIT:
    return f"{value[:_SUMMARY_STRING_LIMIT]}... <truncated chars={len(value)}>"
  return value


def _flatten_content_blocks(content_blocks: Any) -> str:
  if not isinstance(content_blocks, list):
    return _stringify(_compact_for_summary(content_blocks))

  parts: list[str] = []
  for block in content_blocks:
    if not isinstance(block, dict):
      parts.append(_stringify(_compact_for_summary(block)))
      continue
    block_type = str(block.get("type") or "")
    if block_type == "text":
      text = str(block.get("text") or "").strip()
      if text:
        parts.append(text)
      continue
    if block_type == "thinking":
      thinking = str(block.get("thinking") or "").strip()
      if thinking:
        parts.append(f"[thinking] {thinking}")
      continue
    if block_type in {"tool_use", "server_tool_use"}:
      name = str(block.get("name") or "tool")
      tool_input = _stringify(_compact_for_summary(_redact_tool_input_for_summary(name, block.get("input"))))
      parts.append(f"[tool_use] {name} input={tool_input}")
      continue
    if block_type == "compaction":
      content = str(block.get("content") or "").strip()
      if content:
        parts.append(f"[compaction] {content}")
      else:
        parts.append("[compaction]")
      continue
    parts.append(_stringify(_compact_for_summary(block)))
  return "\n".join(part for part in parts if part).strip()


def _format_entry_for_summary(entry: LogEntry) -> str | None:
  event = sanitize_tool_event(entry.event, sink="compaction_prompt")
  event_type = str(event.get("type") or "")
  prefix = f"seq={entry.seq} ts={_isoformat_timestamp(entry.timestamp)} type={event_type}"

  if event_type == "summary":
    return None

  if event_type == "user_message":
    return f"{prefix} content={_stringify(event.get('content')).strip()}"

  if event_type == "assistant_message":
    content = _flatten_content_blocks(event.get("content_blocks"))
    stop_reason = str(event.get("stop_reason") or "")
    model = str(event.get("model") or "")
    details = [detail for detail in [f"model={model}" if model else "", f"stop={stop_reason}" if stop_reason else ""] if detail]
    header = f"{prefix} {' '.join(details)}".strip()
    if content:
      return f"{header}\n{content}"
    return header

  if event_type == "tool_call_start":
    tool_name = str(event.get("tool_name") or "tool")
    tool_call_id = str(event.get("tool_call_id") or "")
    tool_input = _stringify(_compact_for_summary(_redact_tool_input_for_summary(tool_name, event.get("tool_input"))))
    return f"{prefix} tool={tool_name} tool_call_id={tool_call_id} input={tool_input}"

  if event_type == "tool_call_complete":
    tool_name = str(event.get("tool_name") or "tool")
    tool_call_id = str(event.get("tool_call_id") or "")
    if event.get("error") is not None:
      outcome = f"error={_stringify(_compact_for_summary(event.get('error')))}"
    else:
      outcome = f"result={_stringify(_compact_for_summary(event.get('result')))}"
    return f"{prefix} tool={tool_name} tool_call_id={tool_call_id} {outcome}"

  if event_type == "tool_call_interrupted":
    tool_name = str(event.get("tool_name") or "tool")
    tool_call_id = str(event.get("tool_call_id") or "")
    tool_risk = str(event.get("tool_risk") or "side_effecting")
    return f"{prefix} tool={tool_name} tool_call_id={tool_call_id} risk={tool_risk}"

  if event_type in {"attach", "detach", "interrupted"}:
    reason = str(event.get("reason") or "")
    role = str(event.get("role") or "")
    sub_agent_id = str(event.get("sub_agent_id") or "")
    fragments = [prefix]
    if role:
      fragments.append(f"role={role}")
    if sub_agent_id:
      fragments.append(f"sub_agent_id={sub_agent_id}")
    if reason:
      fragments.append(f"reason={reason}")
    return " ".join(fragments)

  if event_type == "state_update":
    return f"{prefix} payload={_stringify(_compact_for_summary(event.get('payload')))}"

  return f"{prefix} payload={_stringify(_compact_for_summary(event))}"


def _truncate_rendered_entry(rendered: str, available_chars: int) -> str:
  if available_chars <= 0:
    return ""
  marker = f"\n[summary input truncated from {len(rendered)} chars to fit prompt budget]"
  if available_chars <= len(marker) + 20:
    return rendered[:available_chars]
  return rendered[: available_chars - len(marker)] + marker


def _build_summary_prompt_slice(
  *,
  prior_summary_text: str | None,
  entries: list[LogEntry],
  prompt: str,
  from_seq: int,
  to_seq: int,
  max_chars: int | None = None,
) -> _SummaryPromptSlice:
  prior_summary_section = prior_summary_text.strip() if isinstance(prior_summary_text, str) and prior_summary_text.strip() else "(none)"
  header = (
    f"Prior cumulative summary:\n{prior_summary_section}\n\n"
    f"Unsummarized event slice (inclusive seq range {from_seq}..{to_seq}):\n"
  )
  footer = "\n"
  body_budget = None if max_chars is None else max(0, max_chars - len(header) - len(footer))

  formatted_entries: list[str] = []
  body_chars = 0
  covered_to_seq = from_seq - 1
  rendered_event_count = 0

  for entry in entries:
    rendered = _format_entry_for_summary(entry)
    if rendered is None or not rendered.strip():
      covered_to_seq = entry.seq
      continue

    separator_chars = 2 if formatted_entries else 0
    rendered_chars = len(rendered)
    if body_budget is not None and body_chars + separator_chars + rendered_chars > body_budget:
      if formatted_entries:
        break
      rendered = _truncate_rendered_entry(rendered, max(0, body_budget - separator_chars))
      if not rendered.strip():
        break
      rendered_chars = len(rendered)

    formatted_entries.append(rendered)
    body_chars += separator_chars + rendered_chars
    rendered_event_count += 1
    covered_to_seq = entry.seq

  events_section = "\n\n".join(formatted_entries).strip() or "(no events)"
  return _SummaryPromptSlice(
    text=f"{header}{events_section}{footer}",
    covered_to_seq=covered_to_seq,
    rendered_event_count=rendered_event_count,
  )


def _format_summary_prompt(
  *,
  prior_summary_text: str | None,
  entries: list[LogEntry],
  prompt: str,
  from_seq: int,
  to_seq: int,
  max_chars: int | None = None,
) -> str:
  return _build_summary_prompt_slice(
    prior_summary_text=prior_summary_text,
    entries=entries,
    prompt=prompt,
    from_seq=from_seq,
    to_seq=to_seq,
    max_chars=max_chars,
  ).text


async def generate_and_append_summary(
  log: AgentSessionLog,
  *,
  from_seq: int,
  to_seq: int,
  prompt: str,
  capability_execution: BoundCapabilityExecution,
  prior_summary_text: str | None = None,
  summarize_fn: SummaryFn | None = None,
  prompt_char_budget: int | None = SUMMARY_PROMPT_CHAR_BUDGET,
) -> LogEntry | None:
  if not isinstance(capability_execution, BoundCapabilityExecution):
    raise TypeError(
      "session summary requires a BoundCapabilityExecution"
    )
  capability_execution.validate()
  model = capability_execution.bind.upstream_model
  if from_seq <= 0:
    raise ValueError("from_seq must be positive")
  if to_seq <= 0:
    raise ValueError("to_seq must be positive")
  if from_seq > to_seq:
    return None

  summaries, _ = await log.query(event_types={"summary"}, order="desc", limit=1)
  prior_summary = summaries[0] if summaries else None
  if prior_summary_text is None and prior_summary is not None:
    text = str(prior_summary.event.get("text") or "").strip()
    prior_summary_text = text or None

  entries, _ = await log.query(after_seq=from_seq, before_seq=to_seq, order="asc")
  if not entries:
    return None

  prompt_prefix = (
    f"{UNTRUSTED_CHILD_RESULTS_POLICY}\n\n{prompt.strip()}\n\n"
  )
  slice_budget = None
  if prompt_char_budget is not None:
    slice_budget = max(0, int(prompt_char_budget) - len(prompt_prefix))
  prompt_slice = _build_summary_prompt_slice(
    prior_summary_text=prior_summary_text,
    entries=entries,
    prompt=prompt,
    from_seq=from_seq,
    to_seq=to_seq,
    max_chars=slice_budget,
  )
  if prompt_slice.covered_to_seq < from_seq:
    LOGGER.warning(
      "Session summary prompt budget too small for %s seq=%d..%d",
      log.path,
      from_seq,
      to_seq,
    )
    return None
  if prompt_slice.rendered_event_count == 0:
    return None
  full_prompt = f"{prompt_prefix}{prompt_slice.text}".strip()

  if summarize_fn is not None:
    generated_summary_text = (await summarize_fn(full_prompt)).strip()
  else:
    generated_summary_text = (await _provider_summarize(
      full_prompt,
      capability_execution=capability_execution,
      system_prompt=(
        f"{DEFAULT_SUMMARY_SYSTEM_PROMPT}\n\n"
        f"{UNTRUSTED_CHILD_RESULTS_POLICY}"
      ),
    )).strip()

  if not generated_summary_text:
    raise RuntimeError(
      (
        f"Session summary generation returned empty text for {log.path} "
        f"seq={from_seq}..{to_seq}"
      )
    )

  event = {
    "type": "summary",
    "covers": {"from_seq": from_seq, "to_seq": prompt_slice.covered_to_seq},
    "summary_kind": "cumulative",
    "text": generated_summary_text,
    "source_model": model,
    "token_estimate": max(1, len(generated_summary_text) // 4),
    "generated_at": time.time(),
    "supersedes_seq": prior_summary.seq if prior_summary is not None else None,
  }
  pid = gateway_product_id()
  if pid is not None:
    event["product_id"] = pid
  return await log.append(event)


__all__ = ["SummaryFn", "generate_and_append_summary"]
