# ruff: noqa: E402

import asyncio
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.agent_session_log as agent_session_log_module
from agent_gateway import AgentSessionLog, generate_and_append_summary
from agent.profiles import analyst_session_summary
from api.agent.profiles import analyst as analyst_config
from api.agent.profiles.analyst import (
  BOOTSTRAP_CAP_SECONDS,
  AnalystContextBuilder,
  generate_analyst_session_summary,
)


def _run(coro):
  return asyncio.run(coro)


def _append_with_timestamp(
  log: AgentSessionLog,
  monkeypatch: pytest.MonkeyPatch,
  *,
  timestamp: float,
  event: dict,
) -> None:
  monkeypatch.setattr(agent_session_log_module.time, "time", lambda: timestamp)
  _run(log.append(event))


def test_analyst_session_summary_helpers_preserve_parent_api() -> None:
  assert analyst_config.BOOTSTRAP_CAP_SECONDS == analyst_session_summary.BOOTSTRAP_CAP_SECONDS
  assert analyst_config.SESSION_LOG_SUMMARY_PROMPT is analyst_session_summary.SESSION_LOG_SUMMARY_PROMPT
  assert analyst_config.SESSION_LOG_SUMMARY_MAX_CHUNKS == analyst_session_summary.SESSION_LOG_SUMMARY_MAX_CHUNKS
  assert (
    analyst_config.SESSION_LOG_SUMMARY_PROMPT_CHAR_BUDGET
    == analyst_session_summary.SESSION_LOG_SUMMARY_PROMPT_CHAR_BUDGET
  )
  assert AnalystContextBuilder is analyst_config.AnalystContextBuilder
  assert issubclass(AnalystContextBuilder, analyst_session_summary.AnalystContextBuilder)
  assert is_dataclass(AnalystContextBuilder)
  assert [field.name for field in fields(AnalystContextBuilder)] == [
    "agent_session_log",
    "tail_window_seconds",
    "tail_token_budget",
  ]
  assert generate_analyst_session_summary is analyst_config.generate_analyst_session_summary
  assert AnalystContextBuilder.__module__ == "agent.profiles.analyst"
  assert generate_analyst_session_summary.__module__ == "agent.profiles.analyst"


def test_generate_analyst_session_summary_forwards_parent_provider_name(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  captured: dict[str, object] = {}
  expected = object()
  session_log = object()

  def summarize_fn(_prompt: str) -> str:
    return "summary"

  async def fake_generate_analyst_session_summary(agent_session_log: object, **kwargs: object) -> object:
    captured["agent_session_log"] = agent_session_log
    captured["kwargs"] = kwargs
    return expected

  monkeypatch.setattr(analyst_config, "PROVIDER", "fixture")
  monkeypatch.setattr(
    analyst_config,
    "_generate_analyst_session_summary",
    fake_generate_analyst_session_summary,
  )

  result = _run(
    analyst_config.generate_analyst_session_summary(
      session_log,  # type: ignore[arg-type]
      provider="provider",  # type: ignore[arg-type]
      auth_config={"api_key": "test"},
      summarize_fn=summarize_fn,  # type: ignore[arg-type]
      model="model",
      prompt="prompt",
      now=123.0,
      max_chunks=2,
      prompt_char_budget=456,
    )
  )

  assert result is expected
  assert captured["agent_session_log"] is session_log
  kwargs = captured["kwargs"]
  assert isinstance(kwargs, dict)
  assert kwargs["provider"] == "provider"
  assert kwargs["auth_config"] == {"api_key": "test"}
  assert kwargs["summarize_fn"] is summarize_fn
  assert kwargs["model"] == "model"
  assert kwargs["prompt"] == "prompt"
  assert kwargs["now"] == 123.0
  assert kwargs["max_chunks"] == 2
  assert kwargs["prompt_char_budget"] == 456
  assert kwargs["provider_name"] == "fixture"


def test_generate_and_append_summary_round_trips_cumulative_summary(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "summary-round-trip.jsonl")
  _run(log.append({"type": "user_message", "content": "Review AAPL and MSFT."}))
  _run(log.append({"type": "assistant_message", "content_blocks": [{"type": "text", "text": "Started screening."}]}))

  captured_prompts: list[str] = []

  async def _summarize_first(prompt_text: str) -> str:
    captured_prompts.append(prompt_text)
    return "First cumulative summary."

  first_summary = _run(
    generate_and_append_summary(
      log,
      from_seq=1,
      to_seq=2,
      prompt="Summarize the analyst session.",
      model="claude-sonnet-4-6",
      auth_config={},
      summarize_fn=_summarize_first,
    )
  )

  assert first_summary is not None
  assert first_summary.event["covers"] == {"from_seq": 1, "to_seq": 2}
  assert first_summary.event["summary_kind"] == "cumulative"
  assert first_summary.event["text"] == "First cumulative summary."
  assert first_summary.event["supersedes_seq"] is None

  _run(log.append({"type": "user_message", "content": "Now extend the analysis to NVDA."}))

  async def _summarize_second(prompt_text: str) -> str:
    captured_prompts.append(prompt_text)
    return "Second cumulative summary."

  second_summary = _run(
    generate_and_append_summary(
      log,
      from_seq=3,
      to_seq=4,
      prompt="Summarize the analyst session.",
      model="claude-sonnet-4-6",
      auth_config={},
      prior_summary_text="First cumulative summary.",
      summarize_fn=_summarize_second,
    )
  )

  assert second_summary is not None
  assert second_summary.event["covers"] == {"from_seq": 3, "to_seq": 4}
  assert second_summary.event["summary_kind"] == "cumulative"
  assert second_summary.event["supersedes_seq"] == first_summary.seq
  assert "Prior cumulative summary:\nFirst cumulative summary." in captured_prompts[1]
  assert "type=summary" not in captured_prompts[1]


def test_generate_and_append_summary_returns_none_on_summarization_failure(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "summary-failure.jsonl")
  _run(log.append({"type": "user_message", "content": "Check the portfolio."}))

  async def _failing_summary(_prompt_text: str) -> str:
    raise RuntimeError("provider timeout")

  result = _run(
    generate_and_append_summary(
      log,
      from_seq=1,
      to_seq=1,
      prompt="Summarize the analyst session.",
      model="claude-sonnet-4-6",
      auth_config={},
      summarize_fn=_failing_summary,
    )
  )

  assert result is None
  summaries, _ = _run(log.query(event_types={"summary"}, order="asc"))
  assert summaries == []


def test_generate_and_append_summary_skips_summary_only_slice(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "summary-only.jsonl")
  _run(
    log.append(
      {
        "type": "summary",
        "covers": {"from_seq": 1, "to_seq": 1},
        "summary_kind": "cumulative",
        "text": "Prior summary.",
      }
    )
  )
  calls = 0

  async def _summarize(_prompt_text: str) -> str:
    nonlocal calls
    calls += 1
    return "Should not be called."

  result = _run(
    generate_and_append_summary(
      log,
      from_seq=1,
      to_seq=1,
      prompt="Summarize the analyst session.",
      model="claude-sonnet-4-6",
      auth_config={},
      summarize_fn=_summarize,
    )
  )

  assert result is None
  assert calls == 0
  summaries, _ = _run(log.query(event_types={"summary"}, order="asc"))
  assert len(summaries) == 1


def test_generate_and_append_summary_is_orthogonal_to_compaction_content_blocks(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "compaction-block.jsonl")
  _run(
    log.append(
      {
        "type": "assistant_message",
        "content_blocks": [
          {"type": "text", "text": "The analyst gathered earnings context."},
          {"type": "compaction", "content": "Server-side compaction preserved prior context."},
        ],
        "stop_reason": "compaction",
      }
    )
  )

  async def _summarize(_prompt_text: str) -> str:
    return "Narrative summary."

  summary = _run(
    generate_and_append_summary(
      log,
      from_seq=1,
      to_seq=1,
      prompt="Summarize the analyst session.",
      model="claude-sonnet-4-6",
      auth_config={},
      summarize_fn=_summarize,
    )
  )

  assert summary is not None
  assistant_entries, _ = _run(log.query(event_types={"assistant_message"}, order="asc"))
  summary_entries, _ = _run(log.query(event_types={"summary"}, order="asc"))
  assert len(assistant_entries) == 1
  assert len(summary_entries) == 1


def test_generate_and_append_summary_preserves_thinking_blocks(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "thinking-preserved.jsonl")
  _run(
    log.append(
      {
        "type": "assistant_message",
        "content_blocks": [
          {"type": "thinking", "thinking": "Reasoning detail that informs the handoff."},
          {"type": "text", "text": "The analyst reached a conclusion."},
        ],
        "stop_reason": "end_turn",
      }
    )
  )
  captured_prompts: list[str] = []

  async def _summarize(prompt_text: str) -> str:
    captured_prompts.append(prompt_text)
    return "Narrative summary."

  summary = _run(
    generate_and_append_summary(
      log,
      from_seq=1,
      to_seq=1,
      prompt="Summarize the analyst session.",
      model="claude-sonnet-4-6",
      auth_config={},
      summarize_fn=_summarize,
    )
  )

  assert summary is not None
  assert "[thinking] Reasoning detail that informs the handoff." in captured_prompts[0]
  assert "The analyst reached a conclusion." in captured_prompts[0]


def test_generate_and_append_summary_compacts_bulk_tool_results(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "bulk-tool-result.jsonl")
  _run(
    log.append(
      {
        "type": "tool_call_complete",
        "tool_call_id": "toolu_bulk",
        "tool_name": "fmp_fetch",
        "result": {
          "status": "success",
          "endpoint": "earnings_calendar",
          "row_count": 100,
          "data": [{"symbol": f"TICK{i}", "payload": "x" * 200} for i in range(100)],
          "summary": {"date_range": {"earliest": "2026-05-01", "latest": "2026-06-15"}},
        },
        "error": None,
      }
    )
  )
  captured_prompts: list[str] = []

  async def _summarize(prompt_text: str) -> str:
    captured_prompts.append(prompt_text)
    return "Narrative summary."

  summary = _run(
    generate_and_append_summary(
      log,
      from_seq=1,
      to_seq=1,
      prompt="Summarize the analyst session.",
      model="claude-sonnet-4-6",
      auth_config={},
      summarize_fn=_summarize,
    )
  )

  assert summary is not None
  assert '"endpoint": "earnings_calendar"' in captured_prompts[0]
  assert '"row_count": 100' in captured_prompts[0]
  assert '"data": "<omitted list items=100>"' in captured_prompts[0]
  assert "TICK99" not in captured_prompts[0]


def test_generate_and_append_summary_respects_prompt_budget_boundary(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "budget-boundary.jsonl")
  _run(log.append({"type": "user_message", "content": "first " + ("a" * 5000)}))
  _run(log.append({"type": "user_message", "content": "second " + ("b" * 5000)}))
  captured_prompts: list[str] = []

  async def _summarize(prompt_text: str) -> str:
    captured_prompts.append(prompt_text)
    return "Budgeted summary."

  summary = _run(
    generate_and_append_summary(
      log,
      from_seq=1,
      to_seq=2,
      prompt="Summarize the analyst session.",
      model="claude-sonnet-4-6",
      auth_config={},
      summarize_fn=_summarize,
      prompt_char_budget=1500,
    )
  )

  assert summary is not None
  assert summary.event["covers"] == {"from_seq": 1, "to_seq": 1}
  assert len(captured_prompts[0]) <= 1500
  assert "summary input truncated" in captured_prompts[0]
  assert "second " not in captured_prompts[0]


def test_generate_analyst_session_summary_applies_bootstrap_cap(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  now = 1_000_000.0
  log = AgentSessionLog(path=tmp_path / "sessions" / "bootstrap-cap.jsonl")
  event_timestamps = [
    now - (5 * 86400),
    now - (4 * 86400),
    now - BOOTSTRAP_CAP_SECONDS + 3600,
    now - (2 * 86400),
    now - 1800,
  ]

  for index, event_timestamp in enumerate(event_timestamps, start=1):
    _append_with_timestamp(
      log,
      monkeypatch,
      timestamp=event_timestamp,
      event={"type": "user_message", "content": f"event-{index}"},
    )

  async def _summarize(_prompt_text: str) -> str:
    return "Bootstrap cumulative summary."

  summary = _run(
    generate_analyst_session_summary(
      log,
      auth_config={},
      summarize_fn=_summarize,
      now=now,
    )
  )

  assert summary is not None
  assert summary.event["covers"]["from_seq"] == 3
  assert summary.event["covers"]["to_seq"] == 5


def test_generate_analyst_session_summary_catches_up_in_bounded_chunks(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "chunked-catchup.jsonl")
  for index in range(3):
    _run(log.append({"type": "user_message", "content": f"event-{index} " + ("x" * 5000)}))

  async def _summarize(prompt_text: str) -> str:
    return f"Chunk summary chars={len(prompt_text)}."

  summary = _run(
    generate_analyst_session_summary(
      log,
      auth_config={},
      summarize_fn=_summarize,
      max_chunks=2,
      prompt_char_budget=1500,
    )
  )

  assert summary is not None
  summaries, _ = _run(log.query(event_types={"summary"}, order="asc"))
  assert len(summaries) == 2
  assert summaries[0].event["covers"] == {"from_seq": 1, "to_seq": 1}
  assert summaries[1].event["covers"] == {"from_seq": 2, "to_seq": 2}
