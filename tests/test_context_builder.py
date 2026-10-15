import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentSessionLog, AnthropicProvider, SessionContextBuilder
import agent_gateway.agent_session_log as agent_session_log_module


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


def test_context_builder_replays_annotated_tool_results_with_call(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "tool-result-compat.jsonl")
  assistant = {
    "type": "assistant_message",
    "content_blocks": [{"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}}],
  }
  _run(log.append(assistant))
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
    {"role": "assistant", "content": assistant["content_blocks"]},
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


def test_context_builder_replays_entries_older_than_any_window(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "old-entries.jsonl")
  monkeypatch.setattr(agent_session_log_module.time, "time", lambda: 100.0)
  _run(log.append({"type": "user_message", "content": "What is the risk score?"}))
  _run(log.append({"type": "assistant_message", "content": "Overall risk score: 89.0"}))
  monkeypatch.undo()
  _run(log.append({"type": "user_message", "content": "What was the first number?"}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())

  assert messages == [
    {"role": "user", "content": "What is the risk score?"},
    {"role": "assistant", "content": "Overall risk score: 89.0"},
    {"role": "user", "content": "What was the first number?"},
  ]


def test_context_builder_replays_parent_rows_only(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "parent-only.jsonl")
  child = {"role": "sub_agent", "sub_agent_id": "sub0:sess_parent"}
  parent_call = {
    "type": "assistant_message",
    "role": "writer",
    "content_blocks": [
      {"type": "text", "text": "Delegating the profile."},
      {"type": "tool_use", "id": "run-1", "name": "run_agent", "input": {"task": "Profile ACME"}},
    ],
  }
  _run(log.append({"type": "user_message", "role": "writer", "content": "Profile ACME for me."}))
  _run(log.append(parent_call))
  _run(log.append({"type": "tool_call_start", "role": "writer", "tool_call_id": "run-1", "tool_name": "run_agent"}))
  _run(log.append({"type": "attach", **child}))
  _run(log.append({"type": "user_message", "content": "CHILD_TASK Profile ACME", **child}))
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "tool_use", "id": "child-tool", "name": "fetch_company_profile", "input": {}}],
    **child,
  }))
  _run(log.append({"type": "tool_call_start", "tool_call_id": "child-tool", **child}))
  _run(log.append({
    "type": "tool_call_complete", "tool_call_id": "child-tool", "result": {"CHILD_RESULT": True}, **child,
  }))
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [
      {"type": "compaction", "content": "CHILD_SUMMARY"},
      {"type": "text", "text": "CHILD_DRAFT ACME makes widgets."},
    ],
    **child,
  }))
  _run(log.append({"type": "detach", "reason": "completed", **child}))
  _run(log.append({
    "type": "tool_call_complete",
    "role": "writer",
    "tool_call_id": "run-1",
    "tool_name": "run_agent",
    "result": {"report": "ACME makes widgets."},
  }))
  _run(log.append({"type": "assistant_message", "role": "writer", "content": "ACME makes widgets."}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())

  assert messages == [
    {"role": "user", "content": "Profile ACME for me."},
    {"role": "assistant", "content": parent_call["content_blocks"]},
    {
      "role": "user",
      "content": [{
        "type": "tool_result",
        "tool_use_id": "run-1",
        "content": json.dumps({"report": "ACME makes widgets."}),
      }],
    },
    {"role": "assistant", "content": "ACME makes widgets."},
  ]
  assert "CHILD_" not in json.dumps(messages)


def test_context_builder_resumes_from_last_compaction_block(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "compaction-boundary.jsonl")
  _run(log.append({"type": "user_message", "content": "Old question"}))
  _run(log.append({"type": "assistant_message", "content": "Old answer"}))
  _run(log.append({"type": "user_message", "content": "Run the lookup"}))
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [
      {"type": "text", "text": "Pre-compaction text"},
      {"type": "compaction", "content": "S"},
      {"type": "tool_use", "id": "tool-2", "name": "lookup", "input": {}},
    ],
  }))
  _run(log.append({"type": "tool_call_complete", "tool_call_id": "tool-2", "result": {"ok": True}}))
  _run(log.append({"type": "user_message", "content": "Later question"}))
  _run(log.append({"type": "assistant_message", "content": "Later answer"}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())

  assert messages == [
    {
      "role": "assistant",
      "content": [
        {"type": "compaction", "content": "S"},
        {"type": "tool_use", "id": "tool-2", "name": "lookup", "input": {}},
      ],
    },
    {
      "role": "user",
      "content": [{"type": "tool_result", "tool_use_id": "tool-2", "content": json.dumps({"ok": True})}],
    },
    {"role": "user", "content": "Later question"},
    {"role": "assistant", "content": "Later answer"},
  ]


def test_context_builder_uses_latest_of_multiple_compactions(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "multiple-compactions.jsonl")
  _run(log.append({"type": "user_message", "content": "First turn"}))
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "compaction", "content": "S1"}, {"type": "text", "text": "After S1"}],
  }))
  _run(log.append({"type": "user_message", "content": "Second turn"}))
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "compaction", "content": "S2"}, {"type": "text", "text": "After S2"}],
  }))
  _run(log.append({"type": "user_message", "content": "Third turn"}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())

  assert messages == [
    {
      "role": "assistant",
      "content": [{"type": "compaction", "content": "S2"}, {"type": "text", "text": "After S2"}],
    },
    {"role": "user", "content": "Third turn"},
  ]



def test_context_builder_projects_checklist_after_summary(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "checklist-projection.jsonl")
  _run(log.append({"type": "todo_snapshot", "revision": 2, "phases": [
    {"name": "Research", "items": [{"id": "a1", "content": "Reconcile revenue", "status": "completed"}]},
    {"name": "Synthesis", "items": [{"id": "a2", "content": "Assess thesis", "status": "pending"}]},
  ], "gateway_session_id": "sess_x", "actor": "agent", "op": "done"}))
  _run(log.append({"type": "summary", "summary_kind": "portable_compaction", "covers": {"to_seq": 1}}))
  _run(log.append({"type": "user_message", "content": "Next"}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())

  user = [m for m in messages if m["role"] == "user"]
  assert len(user) == 2
  assert "Current checklist (revision 2)" in user[0]["content"]
  assert "[x] Reconcile revenue" in user[0]["content"]
  assert "Assess thesis" in user[0]["content"]
  assert user[1]["content"] == "Next"


def test_context_builder_resumes_from_anchor_only_compaction_row(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "anchor-only-compaction.jsonl")
  _run(log.append({"type": "user_message", "content": "Heavy turn"}))
  _run(log.append({"type": "assistant_message", "content": "Pre-compaction answer"}))
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "compaction", "content": "Portable summary"}],
    "stop_reason": "compaction",
  }))
  _run(log.append({"type": "user_message", "content": "Next turn"}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())

  assert messages == [
    {
      "role": "assistant",
      "content": [{"type": "compaction", "content": "Portable summary"}],
      "stop_reason": "compaction",
    },
    {"role": "user", "content": "Next turn"},
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
          "error": "Autonomous run interrupted: soft_budget_abort",
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
        "reason": "soft_budget_abort",
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
  assert "Reason: soft_budget_abort" in combined
  assert "Runner: runner_old" in combined
  assert "`screen_estimate_revisions` (`tool-parent`), risk read_only" in combined
  assert "macro-review (`sub0:sess-parent`): 3 event(s), 1 tool call(s)" in combined
  assert "- error: Autonomous run interrupted: soft_budget_abort" in combined
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
        "reason": "soft_budget_abort",
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


def test_context_builder_attaches_delayed_results_to_their_assistant_batch(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "delayed-batch.jsonl")
  for tool_id in ("tool-a", "tool-b"):
    _run(log.append({
      "type": "assistant_message",
      "content_blocks": [{"type": "tool_use", "id": tool_id, "name": "lookup", "input": {}}],
    }))
    _run(log.append({"type": "user_message", "content": f"Continue after {tool_id}"}))
  for tool_id in ("tool-b", "tool-a"):
    _run(log.append({
      "type": "tool_call_interrupted",
      "tool_call_id": tool_id,
      "final_tool_result_blocks": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": "interrupted", "is_error": True},
        {"type": "text", "text": f"Annotation for {tool_id}"},
      ],
    }))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  for index, tool_id in ((0, "tool-a"), (3, "tool-b")):
    assert messages[index]["content"][0]["id"] == tool_id
    assert messages[index + 1] == {
      "role": "user",
      "content": [
        {"type": "tool_result", "tool_use_id": tool_id, "content": "interrupted", "is_error": True},
        {"type": "text", "text": f"Annotation for {tool_id}"},
      ],
    }
    assert messages[index + 2]["content"] == f"Continue after {tool_id}"


def test_context_builder_preserves_terminal_result_outside_replayed_call(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "result-boundary.jsonl")
  call = _run(log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}}],
  }))
  _run(log.append({"type": "summary", "text": "Lookup requested", "covers": {"to_seq": call.seq}}))
  _run(log.append({
    "type": "tool_call_complete", "tool_call_id": "tool-1", "result": {"observed": "terminal evidence"},
  }))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  assert not any(
    block.get("type") == "tool_result"
    for message in messages if isinstance(message["content"], list)
    for block in message["content"]
  )
  rendered = json.dumps(messages)
  assert "tool-1" in rendered
  assert "terminal evidence" in rendered


@pytest.mark.parametrize("provider_recorded", [True, False])
def test_context_builder_resume_replays_same_model_thinking_with_signature(
  tmp_path: Path,
  provider_recorded: bool,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "thinking-resume.jsonl")
  thinking = {"type": "thinking", "thinking": "Weigh the guidance.", "signature": "sig-1"}
  event = {
    "type": "assistant_message",
    "content_blocks": [thinking, {"type": "text", "text": "Guidance is light."}],
    "stop_reason": "end_turn",
    "model": "claude-opus-5",
  }
  if provider_recorded:
    event["provider"] = "anthropic"
  _run(log.append(event))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  provider = AnthropicProvider()
  normalized = provider.normalize_messages(messages, provider.get_model_info("claude-opus-5"))

  # A log written before the event recorded `provider` carries the model id
  # alone; that id never named another provider, so the signed block replays.
  replayed_thinking = thinking
  assert normalized == [
    {
      "role": "assistant",
      "content": [replayed_thinking, {"type": "text", "text": "Guidance is light."}],
    },
  ]


@pytest.mark.parametrize("stop_reason", ["pause_turn", "max_tokens", "error"])
def test_portable_summary_replays_pending_pair_after_failed_or_partial_request(
  tmp_path: Path, stop_reason: str,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / f"portable-{stop_reason}.jsonl")
  call = {"type": "tool_use", "id": "pending-1", "name": "lookup", "input": {"q": "NVDA"}}
  result = {"type": "tool_result", "tool_use_id": "pending-1", "content": "UNDELIVERED_SENTINEL"}
  _run(log.append({"type": "user_message", "content": "Covered question"}))
  _run(log.append({"type": "assistant_message", "content_blocks": [call]}))
  _run(log.append({
    "type": "tool_call_complete", "tool_call_id": "pending-1",
    "final_tool_result_blocks": [result],
  }))
  # An older relocated anchor must not truncate the protected call/result pair.
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [
      {"type": "compaction", "content": "SUPERSEDED_ANCHOR"},
      {"type": "text", "text": "Retained continuation"},
    ],
  }))
  _run(log.append({
    "type": "summary", "summary_kind": "portable_compaction",
    "covers": {"from_seq": 1, "to_seq": 1}, "text": "Current prefix summary",
  }))
  if stop_reason == "error":
    _run(log.append({"type": "error", "error": "provider disconnected"}))
  else:
    _run(log.append({
      "type": "assistant_message", "stop_reason": stop_reason,
      "content_blocks": [{"type": "text", "text": "Partial answer"}],
    }))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  expected = [
    ("assistant", [{"type": "compaction", "content": "Current prefix summary"}]),
    ("assistant", [call]),
    ("user", [result]),
    ("assistant", [{"type": "text", "text": "Retained continuation"}]),
  ]
  if stop_reason != "error":
    expected.append(("assistant", [{"type": "text", "text": "Partial answer"}]))
  # Compare the complete model-visible sequence, not provider bookkeeping:
  # the pair occurs exactly once, after the current anchor, with no stale anchor.
  assert [(message["role"], message["content"]) for message in messages] == expected


def test_newest_portable_summary_cutoff_replays_pending_pair_once(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "portable-newest.jsonl")
  _run(log.append({"type": "user_message", "content": "Covered question"}))
  _run(log.append({
    "type": "summary", "summary_kind": "portable_compaction",
    "covers": {"from_seq": 1, "to_seq": 1}, "text": "Old prefix",
  }))
  _run(log.append({"type": "user_message", "content": "Now covered too"}))
  call = {"type": "tool_use", "id": "pending-2", "name": "lookup", "input": {}}
  result = {"type": "tool_result", "tool_use_id": "pending-2", "content": "SECOND_SENTINEL"}
  _run(log.append({"type": "assistant_message", "content_blocks": [call]}))
  _run(log.append({
    "type": "tool_call_complete", "tool_call_id": "pending-2",
    "final_tool_result_blocks": [result],
  }))
  for text in ("New prefix", "Newest prefix"):
    _run(log.append({
      "type": "summary", "summary_kind": "portable_compaction",
      "covers": {"from_seq": 1, "to_seq": 3}, "text": text,
    }))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  # Exact content and order exclude duplicate pairs and every superseded summary.
  assert [(message["role"], message["content"]) for message in messages] == [
    ("assistant", [{"type": "compaction", "content": "Newest prefix"}]),
    ("assistant", [call]),
    ("user", [result]),
  ]


def test_second_turn_context_exposes_completed_first_turn_recap_measurements(tmp_path: Path) -> None:
  from agent_gateway.events import event_to_dict
  from agent_gateway.session_recap import compute_recap_from_events

  log = AgentSessionLog(path=tmp_path / "sessions" / "two-turn-recap.jsonl")
  _run(log.append({"type": "user_message", "content": "First question"}))
  _run(log.append({"type": "assistant_message", "content": "First answer"}))
  recap = event_to_dict(compute_recap_from_events(
    [{
      "type": "learning_fork_launched", "in_flight": 2, "day_invocations": 7,
      "day_charged_microusd": 125_000, "retained_bytes": 4096,
    }],
    session_id="recap-session",
    started_at=1.0,
    trigger="turn_end",
  ))
  _run(log.append(recap))
  _run(log.append({"type": "stream_complete", "terminal_disposition": "completed"}))
  _run(log.append({"type": "user_message", "content": "What happened last turn?"}))

  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  recap_messages = [
    message for message in messages
    if isinstance(message["content"], str)
    and message["content"].startswith("## Previous turn recap\n\n")
  ]
  assert len(recap_messages) == 1
  assert recap_messages[0]["role"] == "user"
  rendered = json.loads(recap_messages[0]["content"].split("\n\n", 1)[1])
  measurements = rendered["tool_calls_summary"]["measurements"]
  assert measurements == recap["tool_calls_summary"]["measurements"]
  assert measurements
  assert 125_000 in measurements.values()
  assert messages.index(recap_messages[0]) < len(messages) - 1
  assert messages[-1] == {"role": "user", "content": "What happened last turn?"}


def _recap_with_surfaces(*, turns: int, surfaces: int) -> dict:
  from agent_gateway.events import event_to_dict
  from agent_gateway.session_recap import compute_recap_from_events

  recap = event_to_dict(compute_recap_from_events(
    [
      {"type": "tool_call_start", "tool_call_id": "call-1", "tool_name": "load_tools"},
      {"type": "tool_call_complete", "tool_call_id": "call-1", "tool_name": "load_tools", "result": {}},
      {"type": "error", "error": "provider overloaded"},
    ],
    session_id="recap-session",
    started_at=1.0,
    trigger="turn_end",
  ))
  recap["runner_id"] = "runner_abc"
  recap["usage"] = {
    "user_id": "1",
    "session_id": "recap-session",
    "request_id": "req-1",
    "input_tokens": 18,
    "output_tokens": 633,
    "cache_read_tokens": 530_095,
    "cache_creation_tokens": 217_373,
    "cost": 1.25,
    "turns": turns,
    "model": "claude-opus-5-5",
    "usage_event_ids": [f"event-{index}" for index in range(turns)],
    "context_surfaces": [
      {"surface": f"surface-{index}", "path": "/x" * 200, "tokens": index} for index in range(surfaces)
    ],
  }
  return recap


def _recap_texts(messages: list[dict]) -> list[str]:
  return [
    message["content"] for message in messages
    if isinstance(message["content"], str) and message["content"].startswith("## Previous turn recap\n\n")
  ]


def test_recap_replays_byte_identically_and_without_context_surfaces(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "recap-stability.jsonl")
  recap_one = _recap_with_surfaces(turns=7, surfaces=300)
  recap_two = _recap_with_surfaces(turns=3, surfaces=300)
  recap_two["usage"]["cost"] = 0.5
  builder = SessionContextBuilder(agent_session_log=log)

  _run(log.append({"type": "user_message", "content": "turn one"}))
  _run(log.append({"type": "assistant_message", "content": "answer one"}))
  _run(log.append(recap_one))
  _run(log.append({"type": "stream_complete", "terminal_disposition": "completed"}))
  _run(log.append({"type": "user_message", "content": "turn two"}))
  turn_two = _recap_texts(_run(builder.build()))

  _run(log.append({"type": "assistant_message", "content": "answer two"}))
  _run(log.append(recap_two))
  _run(log.append({"type": "stream_complete", "terminal_disposition": "completed"}))
  _run(log.append({"type": "user_message", "content": "turn three"}))
  turn_three = _recap_texts(_run(builder.build()))

  assert len(turn_two) == 1
  assert len(turn_three) == 2
  assert turn_three[0] == turn_two[0]
  assert turn_three[1] != turn_three[0]
  for text in turn_three:
    assert "context_surfaces" not in text
    assert "surface-" not in text
    assert "runner_abc" not in text
    assert "usage_event_ids" not in text
    assert len(text.encode()) < 2_000
  rendered = json.loads(turn_three[0].split("\n\n", 1)[1])
  assert rendered["trigger"] == "turn_end"
  assert rendered["tool_calls_summary"]["by_tool_name"] == {"load_tools": 1}
  assert rendered["failures"] == [{"failure_type": "terminal_error", "detail": "provider overloaded"}]
  assert rendered["usage"]["cache_read_tokens"] == 530_095
  assert rendered["usage"]["turns"] == 7


def _tool_results(messages: list[dict], tool_use_id: str) -> list[dict]:
  return [
    block
    for message in messages if isinstance(message["content"], list)
    for block in message["content"]
    if block.get("type") == "tool_result" and block.get("tool_use_id") == tool_use_id
  ]


@pytest.mark.parametrize("terminal_names_call", [True, False])
@pytest.mark.parametrize("settlement", ["tool_call_complete", "tool_call_interrupted"])
def test_parked_call_replays_awaiting_approval_until_its_real_result(
  tmp_path: Path, terminal_names_call: bool, settlement: str,
) -> None:
  # The live park (GP19 F2): one root turn parks execute_trade, then an
  # unrelated turn runs in the same session before the Continue delivers.
  log = AgentSessionLog(path=tmp_path / "sessions" / "parked.jsonl")
  call_id, approval_id = "toolu_parked", "3e22702e-d988-4947-bcd4-484499f96f39"
  call = {"type": "tool_use", "id": call_id, "name": "execute_trade", "input": {"preview_id": "p-1"}}
  _run(log.append({"type": "user_message", "content": "Execute preview p-1."}))
  _run(log.append({"type": "assistant_message", "content_blocks": [call]}))
  _run(log.append({
    "type": "tool_call_start", "tool_call_id": call_id, "tool_name": "execute_trade",
    "tool_input": {"preview_id": "p-1"},
  }))
  _run(log.append({
    "type": "tool_approval_request", "tool_call_id": call_id, "approval_id": approval_id,
    "nonce": "secret-nonce", "tool_name": "execute_trade", "tool_input": {"preview_id": "p-1"},
  }))
  terminal = {"type": "stream_complete", "terminal_disposition": "awaiting_approval"}
  if terminal_names_call:
    terminal |= {"approval_id": approval_id, "original_tool_call_id": call_id}
  _run(log.append(terminal))
  _run(log.append({"type": "detach", "reason": "awaiting_approval"}))
  _run(log.append({"type": "user_message", "content": "Unrelated: what is 17 times 23?"}))
  provider = AnthropicProvider()
  model = provider.get_model_info("claude-opus-5")

  unrelated = provider.normalize_messages(
    _run(SessionContextBuilder(agent_session_log=log).build()), model,
  )
  parked = _tool_results(unrelated, call_id)
  assert len(parked) == 1
  assert not parked[0].get("is_error")
  assert json.loads(parked[0]["content"]) == {"status": "awaiting_approval", "approval_id": approval_id}
  assert "missing_tool_result" not in json.dumps(unrelated)
  assert "secret-nonce" not in json.dumps(unrelated)
  # A replay projection only: the durable call stays unsettled for Continue.
  settled, _ = _run(log.query(event_types={"tool_call_complete", "tool_call_interrupted"}))
  assert settled == []

  _run(log.append({"type": "assistant_message", "content_blocks": [{"type": "text", "text": "391"}]}))
  real = {"type": "tool_result", "tool_use_id": call_id, "content": "{\"status\": \"filled\"}"}
  _run(log.append({
    "type": settlement, "tool_call_id": call_id, "tool_name": "execute_trade",
    "result": {"status": "filled"}, "error": None, "final_tool_result_blocks": [real],
  }))

  continued = provider.normalize_messages(
    _run(SessionContextBuilder(agent_session_log=log).build()), model,
  )
  assert _tool_results(continued, call_id) == [real]
  assert "awaiting_approval" not in json.dumps(continued)
  calling = next(
    index for index, message in enumerate(continued)
    if message["role"] == "assistant" and call in message["content"]
  )
  # The settled-later result stays with its assistant, before later user text.
  assert _tool_results([continued[calling + 1]], call_id) == [real]
  assert continued[calling + 2]["role"] == "user"
  assert "17 times 23" in json.dumps(continued[calling + 2]["content"])


def _files(root: Path) -> dict[str, bytes]:
  return {
    str(path.relative_to(root)): path.read_bytes()
    for path in sorted(root.rglob("*"))
    if path.is_file()
  }


def _fork_source_log(tmp_path: Path) -> AgentSessionLog:
  log = AgentSessionLog(path=tmp_path / "source" / "source.jsonl")
  _run(log.append({"type": "user_message", "content": "Remember the codeword PELICAN."}))
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [{"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {}}],
  }))
  _run(log.append({
    "type": "tool_call_complete",
    "tool_call_id": "tool-1",
    "tool_name": "lookup",
    "result": {"ok": True},
    "error": None,
    "final_tool_result_blocks": [
      {"type": "tool_result", "tool_use_id": "tool-1", "content": "{\"ok\": true}"},
    ],
  }))
  _run(log.append({"type": "assistant_message", "content": "The codeword is PELICAN."}))
  return log


def _read_only(log: AgentSessionLog) -> AgentSessionLog:
  return AgentSessionLog(
    path=log.path,
    _create_active_on_open=False,
    _repair_manifest_on_open=False,
  )


def _fork_seeds(log: AgentSessionLog) -> list[dict]:
  entries, _ = _run(log.query_current_strict(event_types={"fork_seed"}))
  return [entry.event for entry in entries]


def test_fork_builder_seeds_once_and_replays_the_seed_before_its_own_turns(
  tmp_path: Path,
) -> None:
  source = _fork_source_log(tmp_path)
  source_messages = _run(SessionContextBuilder(agent_session_log=source).build())
  assert len(source_messages) == 4
  source_before = _files(tmp_path / "source")
  own = AgentSessionLog(path=tmp_path / "fork" / "own.jsonl")
  builder = SessionContextBuilder(
    agent_session_log=own,
    fork_source_log=_read_only(source),
  )

  assert _run(builder.build()) == source_messages
  [seed] = _fork_seeds(own)
  assert seed == {
    "type": "fork_seed",
    "source_log_path": str(source.path),
    "messages": source_messages,
    "event_schema_version": seed["event_schema_version"],
  }
  assert _files(tmp_path / "source") == source_before

  # The original keeps its own log; its later turns never reach the fork.
  _run(source.append({"type": "user_message", "content": "Original-only turn."}))
  _run(own.append({"type": "user_message", "content": "What was the codeword?"}))
  _run(own.append({"type": "assistant_message", "content": "PELICAN."}))
  own_turns = [
    {"role": "user", "content": "What was the codeword?"},
    {"role": "assistant", "content": "PELICAN."},
  ]

  assert _run(builder.build()) == source_messages + own_turns
  assert len(_fork_seeds(own)) == 1
  # A restart builds the fork's own log with no source at all.
  restarted = SessionContextBuilder(agent_session_log=AgentSessionLog(path=own.path))
  assert _run(restarted.build()) == source_messages + own_turns


def test_fork_seed_prefix_drops_once_a_summary_covers_it(tmp_path: Path) -> None:
  source = _fork_source_log(tmp_path)
  own = AgentSessionLog(path=tmp_path / "fork" / "own.jsonl")
  builder = SessionContextBuilder(
    agent_session_log=own,
    fork_source_log=_read_only(source),
  )
  _run(builder.build())
  own_entries, _ = _run(own.query_current_strict())
  seed_seq = own_entries[-1].seq
  _run(own.append({"type": "user_message", "content": "Fork turn."}))
  _run(own.append({
    "type": "summary",
    "summary_kind": "cumulative",
    "covers": {"from_seq": 1, "to_seq": seed_seq + 1},
    "text": "The fork recalled the codeword.",
  }))
  _run(own.append({"type": "user_message", "content": "After the summary."}))

  assert _run(builder.build()) == [
    {"role": "user", "content": "## Prior session summary\n\nThe fork recalled the codeword."},
    {"role": "user", "content": "After the summary."},
  ]
  assert len(_fork_seeds(own)) == 1


@pytest.mark.parametrize("blocked_stage", ["seed_lookup", "source_read"])
def test_fork_seed_survives_a_cancelled_first_turn(tmp_path: Path, blocked_stage: str) -> None:
  source = _fork_source_log(tmp_path)
  source_messages = _run(SessionContextBuilder(agent_session_log=source).build())
  own = AgentSessionLog(path=tmp_path / "fork" / "own.jsonl")
  slow_source = _read_only(source)
  read_started = asyncio.Event()
  release_read = asyncio.Event()
  read_latest_seq = slow_source.latest_seq_current_strict

  async def delayed_latest_seq() -> int:
    read_started.set()
    await release_read.wait()
    return await read_latest_seq()

  query_own = own.query_current_strict

  async def delayed_seed_lookup(**kwargs):
    read_started.set()
    await release_read.wait()
    return await query_own(**kwargs)

  if blocked_stage == "seed_lookup":
    own.query_current_strict = delayed_seed_lookup
  else:
    slow_source.latest_seq_current_strict = delayed_latest_seq

  async def cancel_during_seed() -> list:
    turn = asyncio.ensure_future(
      SessionContextBuilder(agent_session_log=own, fork_source_log=slow_source).build()
    )
    await read_started.wait()
    turn.cancel()
    await asyncio.sleep(0.05)
    turn.cancel()
    await asyncio.sleep(0.05)
    # Cancellation stays pending while the seed is not yet durable, so the
    # turn cannot hand off the fork's log before its seed is written.
    assert not turn.done()
    seeds, _cursor = await query_own(event_types={"fork_seed"})
    assert seeds == []
    release_read.set()
    with pytest.raises(asyncio.CancelledError):
      await turn
    # A resume of the fork's own log, which binds no source, keeps the history.
    return await SessionContextBuilder(agent_session_log=AgentSessionLog(path=own.path)).build()

  resumed = _run(cancel_during_seed())
  assert resumed == source_messages
  assert "PELICAN" in json.dumps(resumed)


def test_fork_seed_source_read_error_is_not_suppressed(tmp_path: Path) -> None:
  source = _read_only(_fork_source_log(tmp_path))
  own = AgentSessionLog(path=tmp_path / "fork" / "own.jsonl")

  async def failing_latest_seq() -> int:
    raise RuntimeError("source read failed")

  source.latest_seq_current_strict = failing_latest_seq

  with pytest.raises(RuntimeError, match="source read failed"):
    _run(SessionContextBuilder(agent_session_log=own, fork_source_log=source).build())
  assert _fork_seeds(own) == []
