from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .capability_execution import BoundCapabilityExecution
from .child_result_trust import UNTRUSTED_CHILD_RESULTS_POLICY
from .provider_summarize import provider_summarize
from .providers.base import ModelProvider
from .runner_limits import estimate_tokens
from model_authority.thinking import ThinkingLevel, parse_effort


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

First write a brief private <analysis> scratchpad (at most a few bullet points) so the summary fits within the output budget.
Then write the durable summary inside a <summary>...</summary> block.

The summary must preserve:
- the original root user deliverable on the very first line (never narrow the overall task deliverable to an in-flight invoke_skill stage's local output format such as a '## Skill Summary' receipt; an inline skill is one step inside the root deliverable)
- current working state and unfinished work across the full root deliverable
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
  last_compact_failed_turn: int | None = None,
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
  if not force and turn_count is not None and last_compact_failed_turn == turn_count:
    return False, "failed_cooldown"
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
  preserve_tool_result_ids: frozenset[str] | set[str] = frozenset(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  # Required delivery is a chronological suffix, not an optional round. Keep
  # the whole calling batch, including companion results and later continuations.
  required_start = len(messages)
  for index, message in enumerate(messages):
    if _tool_result_ids(message) & preserve_tool_result_ids:
      required_start = min(required_start, index)
      result_ids = _tool_result_ids(message)
      for call_index in range(index - 1, -1, -1):
        if _tool_use_ids(messages[call_index]) & result_ids:
          required_start = min(required_start, call_index)
          result_ids -= _tool_use_ids(messages[call_index])
        if not result_ids:
          break

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

  first_tail_index = min(
    [required_start, *(start for start, _unit in selected_units)]
  )
  return (
    [dict(message) for message in messages[:first_tail_index]],
    [dict(message) for message in messages[first_tail_index:]],
  )


def apply_compaction_anchor(
  tail: list[dict[str, Any]],
  summary: str,
  *,
  provider: str,
  model: str,
) -> list[dict[str, Any]]:
  retained = []
  for message in tail:
    message = dict(message)
    if isinstance(message.get("content"), list):
      message["content"] = [
        block for block in message["content"]
        if not isinstance(block, dict) or block.get("type") != "compaction"
      ]
    if message.get("content"):
      retained.append(message)
  if not str(summary or "").strip():
    return retained or [{"role": "user", "content": CONTINUATION_NUDGE}]
  anchor_block = {"type": "compaction", "content": summary}
  anchor = {
    "role": "assistant",
    "content": [anchor_block],
    "provider": provider,
    "model": model,
    "stop_reason": "compaction",
  }
  if retained:
    return [anchor, *retained]
  return [anchor, {"role": "user", "content": CONTINUATION_NUDGE}]


_ANALYSIS_RE = re.compile(r"<analysis\b[^>]*>.*?</analysis>", re.IGNORECASE | re.DOTALL)
_UNCLOSED_ANALYSIS_RE = re.compile(r"<analysis\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
_ANALYSIS_BODY_RE = re.compile(r"<analysis\b[^>]*>(.*?)(?:</analysis>|\Z)", re.IGNORECASE | re.DOTALL)
_SUMMARY_RE = re.compile(r"<summary\b[^>]*>(.*?)</summary>", re.IGNORECASE | re.DOTALL)
_SUMMARY_OPEN_RE = re.compile(r"<summary\b", re.IGNORECASE)


def _extract_summary_body(text: str) -> str:
  match = _SUMMARY_RE.search(text)
  if match:
    return match.group(1).strip()
  return re.sub(r"</?summary\b[^>]*>", "", text, flags=re.IGNORECASE).strip()


def _strip_analysis_and_extract_summary(text: str) -> str:
  """The `<summary>` body of a model response, without its `<analysis>` scratchpad.

  A response the output budget cut off inside `<analysis>` (so `<summary>` is
  absent or only just opened) would leave less than ``MIN_SUMMARY_CHARS``;
  the analysis is then the only account of the conversation the model wrote,
  so it is kept, ahead of whatever `<summary>` text followed it.
  """
  raw = str(text or "")
  without_analysis = _ANALYSIS_RE.sub("", raw)
  # An `<analysis>` left open ends the text, unless a `<summary>` follows it:
  # one inside the summary body is the summary's own wording.
  summary_open = _SUMMARY_OPEN_RE.search(without_analysis)
  head = without_analysis if summary_open is None else without_analysis[:summary_open.start()]
  without_analysis = (
    _UNCLOSED_ANALYSIS_RE.sub("", head) + without_analysis[len(head):]
  ).strip()
  extracted = _extract_summary_body(without_analysis)
  if len(extracted) >= MIN_SUMMARY_CHARS:
    return extracted
  analysis = "\n".join(body.strip() for body in _ANALYSIS_BODY_RE.findall(raw) if body.strip())
  salvaged = "\n\n".join(part for part in (analysis, extracted) if part).strip()
  return salvaged if len(salvaged) >= MIN_SUMMARY_CHARS else extracted


def _summary_response_text(result: Any) -> str:
  """The response text to extract a summary from.

  A reasoning model can spend the whole output budget in thinking and stop
  before it writes `<summary>`; its thinking is then the only account of the
  conversation it produced, and precedes whatever text it did write.
  """
  text = str(getattr(result, "text", "") or "")
  thinking = str(getattr(result, "thinking_text", "") or "").strip()
  if len(thinking) >= MIN_SUMMARY_CHARS and len(_strip_analysis_and_extract_summary(text)) < MIN_SUMMARY_CHARS:
    return f"{thinking}\n{text}".strip()
  return text



_ROOT_DELIVERABLE_LINE_PREFIX = "Root deliverable: "
_STAGE_ONLY_RECEIPT_RE = re.compile(
  r"(?i)\s*(?:final\s+|the\s+)?deliverable\s+of\s+(?:the\s+|this\s+)?(?:\*+)?current\s+stage(?:\*+)?[^.\n]*?##\s*Skill\s+Summary[^.\n]*?\.",
)
_TASK_HEADING_RE = re.compile(
  r"(?:^|\n)#+\s*Task[^\n]*\n+([^\n]+)",
  re.IGNORECASE,
)


_RUNNER_USER_TEXT_PREFIXES = (
  "[System:",
  "<background-",
  "<operator-",
  "<task-",
  "[History was compacted",
  "You stopped with ",
  "Current checklist (",
  "The analyst edited the checklist:",
  "[tool_result]",
)
_RUNNER_USER_TEXTS = frozenset({CONTINUATION_NUDGE, "Continue from where you left off."})


def is_runner_injected_user_text(text: str) -> bool:
  """Whether a user message's text is the runner's own words, not the analyst's.

  Owner of the text half of that fact for the compaction header and the todo
  checklist's turn scan: the max-tokens and continue nudges, system notices,
  background/operator/task notifications, the compaction continuation,
  checklist reminders and analyst-edit notes.
  """
  stripped = text.strip()
  return stripped in _RUNNER_USER_TEXTS or stripped.startswith(_RUNNER_USER_TEXT_PREFIXES)


def _real_user_prompt_text(message: dict[str, Any]) -> str | None:
  """The text of a user message that carries the analyst's request, else ``None``.

  Continuation nudges, checklist reminders and analyst-edit notes are the
  runner's own words, and a ``tool_result`` batch is a tool's.
  """
  if str(message.get("role") or "") != "user":
    return None
  content = message.get("content")
  if isinstance(content, list) and any(
    isinstance(block, dict) and block.get("type") == "tool_result" for block in content
  ):
    return None
  text = _content_text(content).strip()
  if not text or is_runner_injected_user_text(text):
    return None
  return text


def _latest_user_prompt_index(messages: list[dict[str, Any]]) -> int | None:
  for index in range(len(messages) - 1, -1, -1):
    if _real_user_prompt_text(messages[index]) is not None:
      return index
  return None


def _current_turn_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """The messages from the analyst's latest request on: an earlier request's
  skills and compaction headers describe a task that is finished."""
  latest = _latest_user_prompt_index(messages)
  return messages if latest is None else messages[latest:]


def _compaction_texts(messages: list[dict[str, Any]]) -> list[str]:
  texts: list[str] = []
  for message in messages:
    content = message.get("content")
    if not isinstance(content, list):
      continue
    for block in content:
      if isinstance(block, dict) and block.get("type") == "compaction":
        texts.append(str(block.get("content") or "").strip())
  return texts


def _message_has_inline_skill_or_stage_narrowing(messages: list[dict[str, Any]], summary: str) -> bool:
  if _STAGE_ONLY_RECEIPT_RE.search(summary) is not None or "## Skill Summary" in summary:
    return True
  current_turn = _current_turn_messages(messages)
  for message in current_turn:
    content = message.get("content")
    if not isinstance(content, list):
      continue
    for block in content:
      if isinstance(block, dict) and block.get("type") in {"tool_use", "server_tool_use"} and block.get("name") == "invoke_skill":
        return True
  return any(
    text.startswith(_ROOT_DELIVERABLE_LINE_PREFIX) or "## Skill Summary" in text
    for text in _compaction_texts(current_turn)
  )


def _extract_root_deliverable(messages: list[dict[str, Any]]) -> str | None:
  """The current turn's request: the latest real user prompt's first line.

  Only when no user prompt survives in ``messages`` (every one was compacted
  into an anchor) does the anchor's own ``Root deliverable:`` line, or its
  ``Task`` heading, stand for it.
  """
  latest = _latest_user_prompt_index(messages)
  current_turn = messages if latest is None else messages[latest:]
  if latest is not None:
    text = _real_user_prompt_text(messages[latest]) or ""
    head = text.split("\n\n## Current Session Context", 1)[0].strip()
    first_line = next((ln.strip() for ln in head.splitlines() if ln.strip()), "")
    if first_line:
      return first_line
  compactions = _compaction_texts(current_turn)
  for comp_text in compactions:
    if comp_text.startswith(_ROOT_DELIVERABLE_LINE_PREFIX):
      first = comp_text.splitlines()[0][len(_ROOT_DELIVERABLE_LINE_PREFIX):].strip()
      if first:
        return first
  for comp_text in compactions:
    match = _TASK_HEADING_RE.search(_STAGE_ONLY_RECEIPT_RE.sub("", comp_text))
    if match:
      task_line = match.group(1).strip()
      if task_line:
        return task_line
  return None


def _ensure_root_deliverable_header(summary: str, messages: list[dict[str, Any]]) -> str:
  cleaned = str(summary or "").strip()
  if not cleaned:
    return cleaned
  if not _message_has_inline_skill_or_stage_narrowing(messages, cleaned):
    return cleaned
  root_deliverable = _extract_root_deliverable(messages)
  cleaned = _STAGE_ONLY_RECEIPT_RE.sub("", cleaned).strip()
  if not root_deliverable:
    return cleaned
  lines = cleaned.splitlines()
  if lines and lines[0].startswith(_ROOT_DELIVERABLE_LINE_PREFIX):
    lines[0] = f"{_ROOT_DELIVERABLE_LINE_PREFIX}{root_deliverable}"
    return "\n".join(lines).strip()
  return f"{_ROOT_DELIVERABLE_LINE_PREFIX}{root_deliverable}\n\n{cleaned}"

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
  on_event: Callable[[dict[str, Any]], object] | None = None,
  session_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
  if not isinstance(capability_execution, BoundCapabilityExecution):
    raise TypeError(
      "portable compaction requires a BoundCapabilityExecution"
    )
  capability_execution.validate()
  if capability_execution.resolver is None:
    raise ValueError("portable compaction requires a CapabilityExecutionResolver")
  capability_execution = capability_execution.resolver.resolve(
    "session.compact",
    parent_bind=capability_execution.bind,
  )
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
      on_event=on_event,
      session_id=session_id,
    )
    if result.saw_tool_use:
      raise SummaryResponseRejected("Summary generation returned tool_use", usage=result.usage)
    return _ensure_root_deliverable_header(_strip_analysis_and_extract_summary(_summary_response_text(result)), messages), result.usage

  async def _summarize_rendered() -> tuple[str, dict[str, Any]]:
    rendered = render_messages_for_summary(messages)
    prompt = build_summary_prompt(rendered, compact_instructions)
    result = await provider_summarize(
      capability_execution=capability_execution,
      messages=[{"role": "user", "content": prompt}],
      system_prompt=UNTRUSTED_CHILD_RESULTS_POLICY,
      tools=[],
      max_tokens=SUMMARY_MAX_TOKENS,
      on_event=on_event,
      session_id=session_id,
    )
    if result.saw_tool_use:
      raise SummaryResponseRejected("Summary generation returned tool_use", usage=result.usage)
    return _ensure_root_deliverable_header(_strip_analysis_and_extract_summary(_summary_response_text(result)), messages), result.usage

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
  last_compact_failed_turn: int | None = None,
  failed_est_at: int | None = None,
  keep_rounds: int = KEEP_ROUNDS,
  tail_token_budget: int = TAIL_TOKEN_BUDGET,
  preserve_tool_result_ids: frozenset[str] | set[str] = frozenset(),
  tools: list[dict[str, Any]] | None = None,
  force: bool = False,
  on_event: Callable[[dict[str, Any]], object] | None = None,
  session_id: str | None = None,
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
    preserve_tool_result_ids=preserve_tool_result_ids,
  )
  # Older logs grafted their anchor onto the following assistant batch.
  # Its summary belongs to the replaced prefix, not to the verbatim tail.
  for message in tail:
    content = message.get("content")
    if isinstance(content, list):
      anchors = [
        block for block in content
        if isinstance(block, dict) and block.get("type") == "compaction"
      ]
      if anchors:
        to_summarize.append({**message, "content": anchors})
  est_summarize_tokens = _estimate_messages(to_summarize)
  should_compact, reason = should_portable_compact(
    est_total_tokens=est_total_tokens,
    est_summarize_tokens=est_summarize_tokens,
    trigger=trigger,
    native_compaction=model_info.supports_native_compaction,
    failed_est_at=failed_est_at,
    turn_count=turn_count,
    last_compact_turn=last_compact_turn,
    last_compact_failed_turn=last_compact_failed_turn,
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
    on_event=on_event,
    session_id=session_id,
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
  est_after = max(0, est_total_tokens - _estimate_messages(messages)) + _estimate_messages(compacted_messages)
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
