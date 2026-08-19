import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSessionLog, SessionContextBuilder
import agent_gateway.agent_session_log as agent_session_log_module
import agent_gateway.context_builder as context_builder_module


def _run(coro):
  return asyncio.run(coro)


def test_context_builder_loads_state_update_as_layer_1b_and_filters_tail(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "builder.jsonl")
  _run(
    log.append(
      {
        "type": "state_update",
        "payload": {
          "alerts": ["Check filings"],
          "data_flags": ["Earnings stale"],
          "active_servers": ["fmp-mcp", "macro-mcp"],
          "regime": "risk_off",
          "budget_exceeded": True,
        },
        "runner_id": "runner_test",
        "generated_at": 1.0,
        "model": "claude-sonnet-4-6",
      }
    )
  )
  _run(
    log.append(
      {
        "type": "user_message",
        "content": "Continue the analyst loop.",
        "client_kind": "cron",
        "received_at": 2.0,
      }
    )
  )

  builder = SessionContextBuilder(agent_session_log=log)
  messages = _run(builder.build())

  assert len(messages) == 2
  state_message = messages[0]
  assert state_message["role"] == "user"
  assert "## Previous run state" in state_message["content"]
  assert "Active alerts:\n- Check filings" in state_message["content"]
  assert "Data flags:\n- Earnings stale" in state_message["content"]
  assert "Active MCP servers: fmp-mcp, macro-mcp" in state_message["content"]
  assert "Regime: risk_off" in state_message["content"]
  assert "previous run exceeded budget" in state_message["content"].lower()
  assert messages[1] == {"role": "user", "content": "Continue the analyst loop."}


def test_context_builder_missing_regime_does_not_crash(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "missing-regime.jsonl")
  _run(
    log.append(
      {
        "type": "state_update",
        "payload": {
          "alerts": ["Watch guidance"],
          "active_servers": ["portfolio-reads-mcp"],
        },
        "runner_id": "runner_test",
        "generated_at": 1.0,
        "model": "claude-sonnet-4-6",
      }
    )
  )

  builder = SessionContextBuilder(agent_session_log=log)
  messages = _run(builder.build())

  assert len(messages) == 1
  assert "## Previous run state" in messages[0]["content"]
  assert "Watch guidance" in messages[0]["content"]
  assert "Regime:" not in messages[0]["content"]


def test_context_builder_replays_historical_final_answer_draft_as_assistant_only(
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "guard-draft.jsonl")
  _run(
    log.append(
      {
        "type": "runtime_guard",
        "guard": "final_answer",
        "message": "Verify the arithmetic with code_execute before final.",
        "draft_content_blocks": [{"type": "text", "text": "Rough answer: 7.4% BEAT"}],
        "draft_model": "claude-sonnet-4-6",
        "draft_provider": "anthropic",
        "draft_stop_reason": "end_turn",
      }
    )
  )

  messages = _run(SessionContextBuilder(agent_session_log=log).build())

  assert messages == [
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "Rough answer: 7.4% BEAT"}],
      "model": "claude-sonnet-4-6",
      "stop_reason": "end_turn",
      "provider": "anthropic",
    },
  ]


def test_context_builder_tool_call_complete_backward_compat_with_final_blocks(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "tool-result-compat.jsonl")
  _run(
    log.append(
      {
        "type": "tool_call_complete",
        "tool_call_id": "tool-1",
        "tool_name": "lookup",
        "result": {"ok": True},
        "error": None,
        "final_tool_result_blocks": [
          {
            "type": "tool_result",
            "tool_use_id": "tool-1",
            "content": "{\"ok\": true, \"_runner_warning\": \"annotated\"}",
          }
        ],
      }
    )
  )

  messages = _run(SessionContextBuilder(agent_session_log=log).build())

  assert messages == [
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "tool-1",
          "content": "{\"ok\": true, \"_runner_warning\": \"annotated\"}",
        }
      ],
    }
  ]


def test_context_builder_ignores_unknown_state_update_fields(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "unknown-fields.jsonl")
  _run(
    log.append(
      {
        "type": "state_update",
        "payload": {"unknown_field": "surprise"},
        "runner_id": "runner_test",
        "generated_at": 1.0,
        "model": "claude-sonnet-4-6",
      }
    )
  )

  builder = SessionContextBuilder(agent_session_log=log)
  messages = _run(builder.build())

  assert len(messages) == 1
  assert messages[0]["content"] == "## Previous run state\n\nNo structured state was recorded."
  assert "surprise" not in messages[0]["content"]


def test_context_builder_loads_summary_before_state_update_and_uses_summary_boundary(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "summary-boundary.jsonl")
  _run(log.append({"type": "user_message", "content": "Older context 1"}))
  _run(log.append({"type": "assistant_message", "content_blocks": [{"type": "text", "text": "Older context 2"}]}))
  _run(
    log.append(
      {
        "type": "state_update",
        "payload": {"alerts": ["Carry this forward"]},
        "runner_id": "runner_test",
        "generated_at": 1.0,
        "model": "claude-sonnet-4-6",
      }
    )
  )
  _run(
    log.append(
      {
        "type": "summary",
        "covers": {"from_seq": 1, "to_seq": 3},
        "summary_kind": "cumulative",
        "text": "The analyst had two prior context events.",
        "source_model": "claude-sonnet-4-6",
        "token_estimate": 10,
      }
    )
  )
  _run(log.append({"type": "user_message", "content": "Tail event 1"}))
  _run(log.append({"type": "assistant_message", "content_blocks": [{"type": "text", "text": "Tail event 2"}]}))

  builder = SessionContextBuilder(agent_session_log=log)
  messages = _run(builder.build())

  assert messages[0]["content"] == "## Prior session summary\n\nThe analyst had two prior context events."
  assert "## Previous run state" in messages[1]["content"]
  assert "Carry this forward" in messages[1]["content"]
  assert messages[2] == {"role": "user", "content": "Tail event 1"}
  assert messages[3]["role"] == "assistant"
  assert messages[3]["content"] == [{"type": "text", "text": "Tail event 2"}]
  combined_render = "\n".join(str(message["content"]) for message in messages)
  assert "Older context 1" not in combined_render
  assert "Older context 2" not in combined_render


def test_context_builder_without_summary_uses_temporal_fallback(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "temporal-fallback.jsonl")
  monkeypatch.setattr(agent_session_log_module.time, "time", lambda: 100.0)
  _run(log.append({"type": "user_message", "content": "Old event"}))
  monkeypatch.setattr(agent_session_log_module.time, "time", lambda: 199.0)
  _run(log.append({"type": "user_message", "content": "Recent event"}))

  builder = SessionContextBuilder(agent_session_log=log, tail_window_seconds=10)
  monkeypatch.setattr(context_builder_module.time, "time", lambda: 205.0)
  messages = _run(builder.build())

  assert messages == [{"role": "user", "content": "Recent event"}]


def test_context_builder_can_replay_full_session_without_temporal_cutoff(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "full-replay.jsonl")
  monkeypatch.setattr(agent_session_log_module.time, "time", lambda: 100.0)
  _run(log.append({"type": "user_message", "content": "First turn"}))
  monkeypatch.setattr(agent_session_log_module.time, "time", lambda: 199.0)
  _run(log.append({"type": "user_message", "content": "Second turn"}))

  builder = SessionContextBuilder(agent_session_log=log, tail_window_seconds=None)
  monkeypatch.setattr(context_builder_module.time, "time", lambda: 10_000.0)
  messages = _run(builder.build())

  assert messages == [
    {"role": "user", "content": "First turn"},
    {"role": "user", "content": "Second turn"},
  ]


def test_context_builder_surfaces_previous_writer_interruption_with_sub_agent_work(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "interrupted-run.jsonl")
  _run(log.append({"type": "attach", "role": "writer", "runner_id": "runner_old", "started_at": 100.0}))
  _run(
    log.append(
      {
        "type": "task_registered",
        "task_id": "bg_0",
        "agent_name": "macro-review",
        "sub_agent_id": "sub0:sess-parent",
        "started_at": 101.0,
      }
    )
  )
  _run(
    log.append(
      {
        "type": "assistant_message",
        "role": "sub_agent",
        "sub_agent_id": "sub0:sess-parent",
        "content_blocks": [{"type": "text", "text": "partial macro work"}],
      }
    )
  )
  _run(
    log.append(
      {
        "type": "tool_call_start",
        "role": "sub_agent",
        "sub_agent_id": "sub0:sess-parent",
        "tool_call_id": "tool-sub",
        "tool_name": "macro_pull",
        "started_at": 102.0,
      }
    )
  )
  _run(
    log.append(
      {
        "type": "tool_call_interrupted",
        "role": "writer",
        "tool_call_id": "tool-parent",
        "tool_name": "screen_estimate_revisions",
        "tool_risk": "read_only",
        "original_started_at": 103.0,
      }
    )
  )
  _run(
    log.append(
      {
        "type": "state_update",
        "payload": {
          "budget_exceeded": True,
          "data_flags": ["estimate revisions unavailable"],
          "alerts": ["resume macro review"],
        },
      }
    )
  )
  _run(
    log.append(
      {
        "type": "interrupted",
        "role": "writer",
        "reason": "budget_exceeded",
        "runner_id": "runner_old",
        "last_completed_seq": 4,
      }
    )
  )
  _run(log.append({"type": "detach", "role": "writer", "reason": "completed"}))
  _run(
    log.append(
      {
        "type": "summary",
        "covers": {"from_seq": 1, "to_seq": 8},
        "text": "Earlier clean summary.",
      }
    )
  )
  _run(log.append({"type": "attach", "role": "writer", "runner_id": "runner_new", "started_at": 200.0}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  combined = "\n".join(str(message["content"]) for message in messages)

  assert "## Previous run interrupted" in combined
  assert "Reason: budget_exceeded" in combined
  assert "Runner: runner_old" in combined
  assert "`screen_estimate_revisions` (`tool-parent`), risk read_only" in combined
  assert "macro-review (`sub0:sess-parent`): 3 event(s), 1 tool call(s)" in combined
  assert "- budget_exceeded: true" in combined
  assert "- data flag: estimate revisions unavailable" in combined
  assert "- alert: resume macro review" in combined


def test_context_builder_does_not_surface_old_interruption_after_clean_writer_run(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "stale-interruption.jsonl")
  _run(log.append({"type": "attach", "role": "writer", "runner_id": "runner_old"}))
  _run(
    log.append(
      {
        "type": "interrupted",
        "role": "writer",
        "reason": "budget_exceeded",
        "runner_id": "runner_old",
      }
    )
  )
  _run(log.append({"type": "detach", "role": "writer", "reason": "completed"}))
  _run(log.append({"type": "attach", "role": "writer", "runner_id": "runner_clean"}))
  _run(log.append({"type": "user_message", "content": "Clean prior run"}))
  _run(log.append({"type": "detach", "role": "writer", "reason": "completed"}))
  _run(log.append({"type": "attach", "role": "writer", "runner_id": "runner_new"}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  combined = "\n".join(str(message["content"]) for message in messages)

  assert "## Previous run interrupted" not in combined
  assert "Clean prior run" in combined


def test_context_builder_suppresses_duplicate_interruption_tail_lines(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "interruption-tail.jsonl")
  _run(log.append({"type": "attach", "role": "writer", "runner_id": "runner_old"}))
  _run(
    log.append(
      {
        "type": "tool_call_interrupted",
        "role": "writer",
        "tool_call_id": "tool-1",
        "tool_name": "file_read",
        "tool_risk": "read_only",
      }
    )
  )
  _run(
    log.append(
      {
        "type": "interrupted",
        "role": "writer",
        "reason": "recovered_on_attach",
        "runner_id": "runner_old",
      }
    )
  )
  _run(log.append({"type": "attach", "role": "writer", "runner_id": "runner_new"}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  combined = "\n".join(str(message["content"]) for message in messages)

  assert combined.count("## Previous run interrupted") == 1
  assert "[Session log] Previous run ended with interruption reason" not in combined
  assert "[Session log] Previous run interrupted tool" not in combined
