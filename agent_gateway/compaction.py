from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from .agent_session_log import AgentSessionLog, LogEntry
from .providers import ModelProvider, ThinkingLevel


LOGGER = logging.getLogger("agent_gateway.compaction")

SummaryFn = Callable[[str], Awaitable[str]]


def _isoformat_timestamp(timestamp: float) -> str:
  return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _stringify(value: Any) -> str:
  if value is None:
    return ""
  if isinstance(value, str):
    return value
  return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _flatten_content_blocks(content_blocks: Any) -> str:
  if not isinstance(content_blocks, list):
    return _stringify(content_blocks)

  parts: list[str] = []
  for block in content_blocks:
    if not isinstance(block, dict):
      parts.append(_stringify(block))
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
      tool_input = _stringify(block.get("input"))
      parts.append(f"[tool_use] {name} input={tool_input}")
      continue
    if block_type == "compaction":
      content = str(block.get("content") or "").strip()
      if content:
        parts.append(f"[compaction] {content}")
      else:
        parts.append("[compaction]")
      continue
    parts.append(_stringify(block))
  return "\n".join(part for part in parts if part).strip()


def _format_entry_for_summary(entry: LogEntry) -> str | None:
  event = entry.event
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
    tool_input = _stringify(event.get("tool_input"))
    return f"{prefix} tool={tool_name} tool_call_id={tool_call_id} input={tool_input}"

  if event_type == "tool_call_complete":
    tool_name = str(event.get("tool_name") or "tool")
    tool_call_id = str(event.get("tool_call_id") or "")
    if event.get("error") is not None:
      outcome = f"error={_stringify(event.get('error'))}"
    else:
      outcome = f"result={_stringify(event.get('result'))}"
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
    return f"{prefix} payload={_stringify(event.get('payload'))}"

  return f"{prefix} payload={_stringify(event)}"


def _format_summary_prompt(
  *,
  prior_summary_text: str | None,
  entries: list[LogEntry],
  prompt: str,
  from_seq: int,
  to_seq: int,
) -> str:
  formatted_entries = [
    rendered
    for entry in entries
    for rendered in [_format_entry_for_summary(entry)]
    if rendered is not None and rendered.strip()
  ]
  prior_summary_section = prior_summary_text.strip() if isinstance(prior_summary_text, str) and prior_summary_text.strip() else "(none)"
  events_section = "\n\n".join(formatted_entries).strip() or "(no events)"
  return (
    f"Prior cumulative summary:\n{prior_summary_section}\n\n"
    f"Unsummarized event slice (inclusive seq range {from_seq}..{to_seq}):\n"
    f"{events_section}\n"
  )


async def _provider_summarize(
  prompt_text: str,
  *,
  provider: ModelProvider,
  auth_config: dict[str, Any],
  model: str,
) -> str:
  config = dict(auth_config or {})
  config["auth_mode"] = str(config.get("auth_mode", "api")).strip().lower() or "api"
  config["model"] = model
  if not provider.has_active_credential(config):
    raise RuntimeError(f"No active credential configured for provider={provider.name}")

  model_info = provider.get_model_info(model)
  max_tokens = int(config.get("max_tokens") or min(model_info.max_output_tokens, 4096))
  client = provider.create_client(config, timeout=float(config.get("client_timeout") or 60.0))
  try:
    params = provider.build_request_params(
      model=model,
      messages=[{"role": "user", "content": prompt_text}],
      system_prompt="You write cumulative narrative summaries of autonomous analyst sessions.",
      tools=[],
      max_tokens=max_tokens,
      thinking_level=ThinkingLevel.NONE,
      auth_mode=config["auth_mode"],
      compaction_trigger=None,
      compaction_instructions=None,
    )

    pieces: list[str] = []
    fallback_blocks: list[str] = []
    async for event in provider.stream(client, params):
      if event.type == "text_delta":
        text = str(event.text or "")
        if text:
          pieces.append(text)
      elif event.type == "text_end" and isinstance(event.raw_block, dict):
        block_text = str(event.raw_block.get("text") or "").strip()
        if block_text:
          fallback_blocks.append(block_text)

    summary_text = "".join(pieces).strip()
    if not summary_text:
      summary_text = "\n".join(fallback_blocks).strip()
    if not summary_text:
      raise RuntimeError("Summary generation returned empty text")
    return summary_text
  finally:
    await provider.close_client(client, timeout=5.0)


async def generate_and_append_summary(
  log: AgentSessionLog,
  *,
  from_seq: int,
  to_seq: int,
  prompt: str,
  model: str,
  auth_config: dict[str, Any],
  prior_summary_text: str | None = None,
  provider: ModelProvider | None = None,
  summarize_fn: SummaryFn | None = None,
) -> LogEntry | None:
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

  prompt_text = _format_summary_prompt(
    prior_summary_text=prior_summary_text,
    entries=entries,
    prompt=prompt,
    from_seq=from_seq,
    to_seq=to_seq,
  )
  full_prompt = f"{prompt.strip()}\n\n{prompt_text}".strip()

  try:
    if summarize_fn is not None:
      generated_summary_text = (await summarize_fn(full_prompt)).strip()
    else:
      if provider is None:
        raise ValueError("provider is required when summarize_fn is not provided")
      generated_summary_text = (await _provider_summarize(
        full_prompt,
        provider=provider,
        auth_config=auth_config,
        model=model,
      )).strip()
  except Exception as exc:
    LOGGER.warning(
      "Session summary generation failed for %s seq=%d..%d: %s",
      log.path,
      from_seq,
      to_seq,
      exc,
    )
    return None

  if not generated_summary_text:
    LOGGER.warning("Session summary generation returned empty text for %s seq=%d..%d", log.path, from_seq, to_seq)
    return None

  return await log.append(
    {
      "type": "summary",
      "covers": {"from_seq": from_seq, "to_seq": to_seq},
      "summary_kind": "cumulative",
      "text": generated_summary_text,
      "source_model": model,
      "token_estimate": max(1, len(generated_summary_text) // 4),
      "generated_at": time.time(),
      "supersedes_seq": prior_summary.seq if prior_summary is not None else None,
    }
  )


__all__ = ["SummaryFn", "generate_and_append_summary"]
