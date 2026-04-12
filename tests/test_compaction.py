import asyncio
import sys
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

from agent_gateway import AgentSessionLog, generate_and_append_summary
from api.agent.profiles.analyst import BOOTSTRAP_CAP_SECONDS, generate_analyst_session_summary
import agent_gateway.agent_session_log as agent_session_log_module


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
