from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

from .capability_execution import BoundCapabilityExecution
from .child_result_trust import UNTRUSTED_CHILD_RESULTS_POLICY
from .provider_summarize import provider_summarize
from .providers.base import ModelProvider
from .runner_limits import estimate_tokens
from .thinking import ThinkingLevel, parse_effort


LOGGER = logging.getLogger("agent_gateway.server_compaction")

KEEP_ROUNDS = 0
TAIL_TOKEN_BUDGET = 24_000
MIN_COMPACT_GAIN_TOKENS = 20_000
SUMMARY_MAX_TOKENS = 4_096
RENDER_CHAR_BUDGET = 600_000
MIN_SUMMARY_CHARS = 200
CONTINUATION_NUDGE = "[History was compacted into the summary above. Continue the task from that summary.]"
PORTABLE_COMPACTION_ENV = "AGENT_GATEWAY_PORTABLE_COMPACTION"

DEFAULT_SERVER_COMPACTION_INSTRUCTIONS = """You are compacting a long tool-using agent conversation.

Do not call tools. Do not ask follow-up questions. Return text only.

First write a private <analysis> scratchpad to make sure nothing important is lost.
Then write the durable summary inside a <summary>...</summary> block.

The summary must preserve:
- the original user task and any later changes to it
- current working state and unfinished work
- open tool threads, tool names, tool call ids, and expected next tool results
- decisions made, the rationale for those decisions, and rejected alternatives
- errors encountered and how they were fixed
- explicit user feedback and constraints
- exact identifiers, file paths, commands, URLs, tickers, dates, numbers, and code snippets that are needed to continue

Prefer conclusions and state over raw bulk data. Include raw data only when it is necessary for the next step.
The <summary> content is the only text that will remain in the model context after compaction."""

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


@dataclass(frozen=True)
class CompactResult:
  applied: bool
  messages: list[dict[str, Any]]
  anchor_block: dict[str, Any] | None
  reason: str
  est_before: int
  est_after: int
  summary_chars: int
  summarize_usage: dict[str, Any] | None


class SummaryResponseRejected(RuntimeError):
  def __init__(self, message: str, *, usage: dict[str, Any] | None = None) -> None:
    super().__init__(message)
    self.usage = usage


class SummaryEffortUnsupported(RuntimeError):
  """The bound turn effort cannot be preserved by the summary request."""


def portable_compaction_enabled() -> bool:
  value = os.environ.get(PORTABLE_COMPACTION_ENV)
  if value is None:
    return True
  return str(value).strip().lower() not in {"0", "false", "no"}


def _json_chars(value: Any) -> int:
  return len(json.dumps(value, ensure_ascii=False, default=str))


def _estimate_messages(messages: list[dict[str, Any]]) -> int:
  return estimate_tokens(json.dumps(messages, ensure_ascii=False, default=str))


def should_portable_compact(
  *,
  est_total_tokens: int,
  est_summarize_tokens: int,
  trigger: int | None,
  native_compaction: bool,
  failed_est_at: int | None = None,
  turn_count: int | None = None,
  last_compact_turn: int | None = None,
  force: bool = False,
) -> tuple[bool, str]:
  if not portable_compaction_enabled():
    return False, "disabled_by_env"
  if native_compaction:
    return False, "native_model"
  if trigger is None or trigger <= 0:
    return False, "disabled"
  if not force and est_total_tokens < trigger:
    return False, "below_trigger"
  if not force and failed_est_at is not None and est_total_tokens < failed_est_at + MIN_COMPACT_GAIN_TOKENS:
    return False, "failed_cooldown"
  if not force and turn_count is not None and last_compact_turn == turn_count:
    return False, "already_compacted_turn"
  if est_summarize_tokens < MIN_COMPACT_GAIN_TOKENS:
    return False, "insufficient_gain"
  return True, "force" if force else "triggered"


def _omitted_marker(value: Any) -> str:
  if isinstance(value, str):
    return f"<omitted text chars={len(value)}>"
  if isinstance(value, dict):
    return f"<omitted object keys={len(value)}>"
  if isinstance(value, (list, tuple)):
    return f"<omitted list items={len(value)}>"
  return f"<omitted {type(value).__name__}>"


def _compact_for_summary(value: Any, *, depth: int = 0) -> Any:
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


def _content_text(content: Any) -> str:
  if isinstance(content, str):
    return content
  if not isinstance(content, list):
    return json.dumps(_compact_for_summary(content), ensure_ascii=False, default=str)
  parts: list[str] = []
  for block in content:
    if not isinstance(block, dict):
      parts.append(json.dumps(_compact_for_summary(block), ensure_ascii=False, default=str))
      continue
    block_type = str(block.get("type") or "")
    if block_type == "text":
      text = str(block.get("text") or "").strip()
      if text:
        parts.append(text)
    elif block_type == "thinking":
      thinking = str(block.get("thinking") or "").strip()
      if thinking:
        parts.append(f"[thinking] {thinking}")
    elif block_type in {"tool_use", "server_tool_use"}:
      name = str(block.get("name") or "tool")
      tool_id = str(block.get("id") or "")
      payload = json.dumps(_compact_for_summary(block.get("input")), ensure_ascii=False, default=str)
      parts.append(f"[tool_use] id={tool_id} name={name} input={payload}")
    elif block_type == "tool_result":
      tool_use_id = str(block.get("tool_use_id") or "")
      payload = json.dumps(_compact_for_summary(block.get("content")), ensure_ascii=False, default=str)
      parts.append(f"[tool_result] tool_use_id={tool_use_id} content={payload}")
    elif block_type == "compaction":
      summary = str(block.get("content") or "").strip()
      parts.append(f"[compaction] {summary}" if summary else "[compaction]")
    else:
      parts.append(json.dumps(_compact_for_summary(block), ensure_ascii=False, default=str))
  return "\n".join(part for part in parts if part).strip()


def render_messages_for_summary(
  messages: list[dict[str, Any]],
  *,
  char_budget: int = RENDER_CHAR_BUDGET,
) -> str:
  rendered: list[str] = []
  used = 0
  for index, message in enumerate(messages, start=1):
    role = str(message.get("role") or "unknown")
    provider = str(message.get("provider") or "")
    model = str(message.get("model") or "")
    stop_reason = str(message.get("stop_reason") or "")
    details = " ".join(
      part for part in [
        f"provider={provider}" if provider else "",
        f"model={model}" if model else "",
        f"stop={stop_reason}" if stop_reason else "",
      ] if part
    )
    body = _content_text(message.get("content")).strip()
    header = f"message[{index}] role={role}" + (f" {details}" if details else "")
    item = f"{header}\n{body}".strip()
    separator = "\n\n" if rendered else ""
    next_len = len(separator) + len(item)
    if used + next_len > char_budget:
      remaining = max(0, char_budget - used - len(separator))
      if remaining > 80:
        rendered.append(separator + item[:remaining] + f"\n[summary input truncated at {char_budget} chars]")
      break
    rendered.append(separator + item)
    used += next_len
  return "".join(rendered).strip()


def build_summary_prompt(rendered: str, instructions: str | None) -> str:
  compact_instructions = (instructions or DEFAULT_SERVER_COMPACTION_INSTRUCTIONS).strip()
  return (
    f"{compact_instructions}\n\n"
    "Conversation transcript to compact:\n"
    "<conversation>\n"
    f"{rendered.strip()}\n"
    "</conversation>\n\n"
    "Return text only. Use <analysis> for private scratchpad and <summary> for the durable summary."
  ).strip()


def _tool_use_ids(message: dict[str, Any]) -> set[str]:
  content = message.get("content")
  if not isinstance(content, list):
    return set()
  return {
    str(block.get("id") or "")
    for block in content
    if isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"} and block.get("id")
  }


def _tool_result_ids(message: dict[str, Any]) -> set[str]:
  content = message.get("content")
  if not isinstance(content, list):
    return set()
  return {
    str(block.get("tool_use_id") or "")
    for block in content
    if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id")
  }


def split_messages_for_compact(
  messages: list[dict[str, Any]],
  *,
  keep_rounds: int = KEEP_ROUNDS,
  tail_token_budget: int = TAIL_TOKEN_BUDGET,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  if keep_rounds <= 0:
    return [dict(message) for message in messages], []

  selected_units: list[tuple[int, list[dict[str, Any]]]] = []
  used_tokens = 0
  index = len(messages) - 1
  rounds = 0
  while index >= 0 and rounds < keep_rounds:
    message = messages[index]
    result_ids = _tool_result_ids(message)
    use_ids = _tool_use_ids(message)
    if result_ids:
      if index <= 0:
        index -= 1
        continue
      previous = messages[index - 1]
      previous_use_ids = _tool_use_ids(previous)
      if not result_ids <= previous_use_ids:
        index -= 1
        continue
      unit = [dict(previous), dict(message)]
      start = index - 1
    elif use_ids:
      index -= 1
      continue
    else:
      unit = [dict(message)]
      start = index

    unit_tokens = _estimate_messages(unit)
    if used_tokens + unit_tokens > tail_token_budget:
      break
    selected_units.append((start, unit))
    used_tokens += unit_tokens
    rounds += 1
    index = start - 1

  if not selected_units:
    return [dict(message) for message in messages], []

  first_tail_index = min(start for start, _unit in selected_units)
  tail: list[dict[str, Any]] = []
  for _start, unit in sorted(selected_units, key=lambda item: item[0]):
    tail.extend(unit)
  return [dict(message) for message in messages[:first_tail_index]], tail


def apply_compaction_anchor(
  tail: list[dict[str, Any]],
  summary: str,
  *,
  provider: str,
  model: str,
) -> list[dict[str, Any]]:
  anchor_block = {"type": "compaction", "content": summary}
  anchor = {
    "role": "assistant",
    "content": [anchor_block],
    "provider": provider,
    "model": model,
    "stop_reason": "compaction",
  }
  if tail:
    return [anchor, *[dict(message) for message in tail]]
  return [anchor, {"role": "user", "content": CONTINUATION_NUDGE}]


_ANALYSIS_RE = re.compile(r"<analysis\b[^>]*>.*?</analysis>", re.IGNORECASE | re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary\b[^>]*>(.*?)</summary>", re.IGNORECASE | re.DOTALL)


def _strip_analysis_and_extract_summary(text: str) -> str:
  without_analysis = _ANALYSIS_RE.sub("", str(text or "")).strip()
  match = _SUMMARY_RE.search(without_analysis)
  if match:
    return match.group(1).strip()
  return re.sub(r"</?summary\b[^>]*>", "", without_analysis, flags=re.IGNORECASE).strip()


def _summary_tools(provider: ModelProvider, tools: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  if tools and getattr(provider, "supports_tool_choice_none", False) is True:
    return [dict(tool) for tool in tools], {"tool_choice": "none"}
  return [], {}


def _require_exact_summary_effort(
  *,
  capability_execution: BoundCapabilityExecution,
) -> ThinkingLevel:
  capability_execution.validate()
  provider = capability_execution.provider
  config = dict(capability_execution.auth_config)
  model = capability_execution.bind.upstream_model
  requested = parse_effort(
    capability_execution.bind.effort,
    field_name="portable_compaction.effort",
  )
  if requested is None:
    raise SummaryEffortUnsupported("Portable compaction requires a bound effort")
  model_info = provider.get_model_info(model)
  model_max_tokens = int(getattr(model_info, "max_output_tokens", 0) or 0)
  summary_max_tokens = (
    min(SUMMARY_MAX_TOKENS, model_max_tokens)
    if model_max_tokens > 0
    else SUMMARY_MAX_TOKENS
  )
  resolution = provider.resolve_effort(
    requested=requested,
    model=model,
    model_info=model_info,
    max_tokens=summary_max_tokens,
    auth_mode=config.get("auth_mode"),
    base_url=config.get("base_url") or config.get("baseURL"),
    compat=config.get("compat"),
  )
  if resolution.requested != requested or resolution.effective != requested:
    raise SummaryEffortUnsupported(
      (
        f"Portable compaction cannot preserve bound effort {requested.value!r} "
        f"for {provider.name}:{model} with max_tokens={summary_max_tokens}"
      )
    )
  return requested


async def summarize_messages(
  messages: list[dict[str, Any]],
  instructions: str | None,
  *,
  capability_execution: BoundCapabilityExecution,
  tools: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
  if not isinstance(capability_execution, BoundCapabilityExecution):
    raise TypeError(
      "portable compaction requires a BoundCapabilityExecution"
    )
  capability_execution.validate()
  provider = capability_execution.provider
  compact_instructions = (instructions or DEFAULT_SERVER_COMPACTION_INSTRUCTIONS).strip()
  summarize_tools, request_kwargs = _summary_tools(provider, tools)
  _require_exact_summary_effort(
    capability_execution=capability_execution,
  )

  async def _summarize_raw() -> tuple[str, dict[str, Any]]:
    raw_messages = [dict(message) for message in messages]
    raw_messages.append({"role": "user", "content": compact_instructions})
    result = await provider_summarize(
      capability_execution=capability_execution,
      messages=raw_messages,
      system_prompt=UNTRUSTED_CHILD_RESULTS_POLICY,
      tools=summarize_tools,
      max_tokens=SUMMARY_MAX_TOKENS,
      request_kwargs=request_kwargs,
    )
    if result.saw_tool_use:
      raise SummaryResponseRejected("Summary generation returned tool_use", usage=result.usage)
    return _strip_analysis_and_extract_summary(result.text), result.usage

  async def _summarize_rendered() -> tuple[str, dict[str, Any]]:
    rendered = render_messages_for_summary(messages)
    prompt = build_summary_prompt(rendered, compact_instructions)
    result = await provider_summarize(
      capability_execution=capability_execution,
      messages=[{"role": "user", "content": prompt}],
      system_prompt=UNTRUSTED_CHILD_RESULTS_POLICY,
      tools=[],
      max_tokens=SUMMARY_MAX_TOKENS,
    )
    if result.saw_tool_use:
      raise SummaryResponseRejected("Summary generation returned tool_use", usage=result.usage)
    return _strip_analysis_and_extract_summary(result.text), result.usage

  if _json_chars(messages) <= RENDER_CHAR_BUDGET:
    try:
      return await _summarize_raw()
    except Exception as exc:
      if not provider.is_context_length_error(exc):
        raise
      LOGGER.warning("Raw portable compaction summarize exceeded context; retrying rendered fallback")
  return await _summarize_rendered()


async def maybe_compact_current_messages(
  messages: list[dict[str, Any]],
  instructions: str | None,
  *,
  capability_execution: BoundCapabilityExecution,
  trigger: int | None,
  est_total_tokens: int,
  turn_count: int | None = None,
  last_compact_turn: int | None = None,
  failed_est_at: int | None = None,
  keep_rounds: int = KEEP_ROUNDS,
  tail_token_budget: int = TAIL_TOKEN_BUDGET,
  tools: list[dict[str, Any]] | None = None,
  force: bool = False,
) -> CompactResult:
  if not isinstance(capability_execution, BoundCapabilityExecution):
    raise TypeError(
      "portable compaction requires a BoundCapabilityExecution"
    )
  capability_execution.validate()
  provider = capability_execution.provider
  model = capability_execution.bind.upstream_model
  model_info = provider.get_model_info(model)
  original_messages = [dict(message) for message in messages]
  to_summarize, tail = split_messages_for_compact(
    messages,
    keep_rounds=keep_rounds,
    tail_token_budget=tail_token_budget,
  )
  est_summarize_tokens = _estimate_messages(to_summarize)
  should_compact, reason = should_portable_compact(
    est_total_tokens=est_total_tokens,
    est_summarize_tokens=est_summarize_tokens,
    trigger=trigger,
    native_compaction=model_info.supports_native_compaction,
    failed_est_at=failed_est_at,
    turn_count=turn_count,
    last_compact_turn=last_compact_turn,
    force=force,
  )
  if not should_compact:
    return CompactResult(
      applied=False,
      messages=original_messages,
      anchor_block=None,
      reason=reason,
      est_before=est_total_tokens,
      est_after=est_total_tokens,
      summary_chars=0,
      summarize_usage=None,
    )

  summary, usage = await summarize_messages(
    to_summarize,
    instructions,
    capability_execution=capability_execution,
    tools=tools,
  )
  summary = summary.strip()
  summary_chars = len(summary)
  if summary_chars < MIN_SUMMARY_CHARS:
    return CompactResult(
      applied=False,
      messages=original_messages,
      anchor_block=None,
      reason="summary_too_short",
      est_before=est_total_tokens,
      est_after=est_total_tokens,
      summary_chars=summary_chars,
      summarize_usage=usage,
    )

  compacted_messages = apply_compaction_anchor(
    tail,
    summary,
    provider=capability_execution.bind.provider,
    model=model,
  )
  anchor_block = dict(compacted_messages[0]["content"][0])
  est_after = max(0, est_total_tokens - est_summarize_tokens) + _estimate_messages(compacted_messages)
  return CompactResult(
    applied=True,
    messages=compacted_messages,
    anchor_block=anchor_block,
    reason=reason,
    est_before=est_total_tokens,
    est_after=est_after,
    summary_chars=summary_chars,
    summarize_usage=usage,
  )

__all__ = [
  "KEEP_ROUNDS",
  "TAIL_TOKEN_BUDGET",
  "MIN_COMPACT_GAIN_TOKENS",
  "SUMMARY_MAX_TOKENS",
  "RENDER_CHAR_BUDGET",
  "MIN_SUMMARY_CHARS",
  "SummaryEffortUnsupported",
  "CONTINUATION_NUDGE",
  "PORTABLE_COMPACTION_ENV",
  "DEFAULT_SERVER_COMPACTION_INSTRUCTIONS",
  "CompactResult",
  "should_portable_compact",
  "render_messages_for_summary",
  "build_summary_prompt",
  "split_messages_for_compact",
  "apply_compaction_anchor",
  "summarize_messages",
  "maybe_compact_current_messages",
  "portable_compaction_enabled",
]
