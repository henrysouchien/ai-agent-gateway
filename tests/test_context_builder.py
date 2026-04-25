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
          "active_servers": ["portfolio-mcp"],
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
          "content": "{\"ok\": true}",
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
