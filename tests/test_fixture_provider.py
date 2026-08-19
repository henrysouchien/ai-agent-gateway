# ruff: noqa: E402

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
API_DIR = ROOT / "api"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))
if str(API_DIR) not in sys.path:
  sys.path.insert(0, str(API_DIR))

from agent_gateway import AgentRunner, EventLog, ParentMessage, ToolDispatcher
from agent_gateway.approval_policy import RunContext
from agent_gateway.approval_resolver import resolve_policy
from agent_gateway.approval_store import SQLiteApprovalStore
from tests.deterministic_fixture_support import (
  FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME,
  FIXTURE_CANVAS_ARTIFACT_SKILL_NAME,
  FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME,
  FIXTURE_APPROVAL_TOOL_NAME,
  FIXTURE_MODEL_ID,
  FIXTURE_TERMINAL_FAILURE_SKILL_NAME,
  FixtureClient,
  FixtureProvider,
  fixture_approval_handler,
)
from tests.capability_execution_test_support import (
  stub_runner_capability_execution,
)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  def get_server_for_tool(self, _name: str) -> str | None:
    return None

  async def call_tool(self, name: str, _tool_input: dict[str, Any], **_kwargs: Any):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}


async def _collect(provider: FixtureProvider, client: FixtureClient, params: dict[str, Any]):
  return [event async for event in provider.stream(client, params)]


async def _collect_terminal_failure_prefix(provider: FixtureProvider, client: FixtureClient, params: dict[str, Any]):
  stream = provider.stream(client, params)
  events = [await anext(stream), await anext(stream)]
  with pytest.raises(RuntimeError, match="fixture_terminal_failure"):
    await anext(stream)
  return events


def test_fixture_stream_turn_one_yields_complete_gated_tool(monkeypatch) -> None:
  monkeypatch.setattr("tests.deterministic_fixture_support._fixture_run_seconds", lambda: 0.0)
  provider = FixtureProvider()
  client = provider.create_client({"model": FIXTURE_MODEL_ID})

  events = asyncio.run(_collect(provider, client, {"messages": []}))

  assert [event.type for event in events] == [
    "text_delta",
    "text_end",
    "tool_use_start",
    "tool_use_delta",
    "tool_use_end",
    "message_end",
  ]
  assert events[0].text
  assert events[2].tool_name == FIXTURE_APPROVAL_TOOL_NAME
  assert events[4].tool_name == FIXTURE_APPROVAL_TOOL_NAME
  assert events[4].tool_input == {
    "reason": "deterministic fixture approval gate",
    "side_effect": "none",
  }
  assert events[-1].stop_reason == "tool_use"
  assert client.turn == 1


def test_fixture_stream_turn_two_echoes_operator_steering(monkeypatch) -> None:
  monkeypatch.setattr("tests.deterministic_fixture_support._fixture_run_seconds", lambda: 0.0)
  provider = FixtureProvider()
  client = provider.create_client({"model": FIXTURE_MODEL_ID})
  asyncio.run(_collect(provider, client, {"messages": []}))

  events = asyncio.run(
    _collect(
      provider,
      client,
      {
        "messages": [
          {
            "role": "user",
            "content": (
              "Operator update for this task:\n"
              "- id=steer-1: use the shorter path"
            ),
          }
        ]
      },
    )
  )

  assert events[0].type == "text_delta"
  assert "use the shorter path" in events[0].text
  assert events[-1].type == "message_end"
  assert events[-1].stop_reason == "end_turn"
  assert client.turn == 2


def test_fixture_canvas_artifact_stream_emits_complete_canvas_tool_call(monkeypatch) -> None:
  monkeypatch.setattr("tests.deterministic_fixture_support._fixture_run_seconds", lambda: 0.0)
  provider = FixtureProvider()
  client = provider.create_client({"model": FIXTURE_MODEL_ID})

  events = asyncio.run(
    _collect(
      provider,
      client,
      {
        "messages": [],
        "system_prompt": (
          "Today is 2026-06-05. Execute the deterministic fixture skill "
          f"{FIXTURE_CANVAS_ARTIFACT_SKILL_NAME}."
        ),
      },
    )
  )

  assert [event.type for event in events] == [
    "text_delta",
    "text_end",
    "tool_use_start",
    "tool_use_delta",
    "tool_use_end",
    "message_end",
  ]
  assert events[2].tool_name == "emit_canvas_artifact"
  assert events[4].tool_name == "emit_canvas_artifact"
  assert events[4].tool_input["title"] == "Fixture Canvas Artifact"
  assert events[4].tool_input["purpose"] == "exploration"
  assert "@hank/canvas-kit" in events[4].tool_input["tsx_source"]
  assert events[4].tool_input["copy_as_json"] == {
    "fixture": FIXTURE_CANVAS_ARTIFACT_SKILL_NAME,
    "contract_name": "CanvasArtifact",
  }
  assert events[-1].stop_reason == "tool_use"

  done_events = asyncio.run(
    _collect(
      provider,
      client,
      {
        "messages": [
          {"role": "assistant", "content": "tool result: ok"},
        ],
        "system_prompt": f"Execute the deterministic fixture skill {FIXTURE_CANVAS_ARTIFACT_SKILL_NAME}.",
      },
    )
  )
  assert done_events[-1].type == "message_end"
  assert done_events[-1].stop_reason == "end_turn"
  assert client.turn == 2


def test_fixture_dashboard_artifact_stream_emits_checked_in_payload(monkeypatch) -> None:
  monkeypatch.setattr("tests.deterministic_fixture_support._fixture_run_seconds", lambda: 0.0)
  provider = FixtureProvider()
  client = provider.create_client({"model": FIXTURE_MODEL_ID})
  expected_payload = json.loads(
    (ROOT / "tests" / "fixtures" / "dashboard" / "full.payload.json").read_text(encoding="utf-8")
  )

  events = asyncio.run(
    _collect(
      provider,
      client,
      {
        "messages": [],
        "system_prompt": (
          "Today is 2026-06-05. Execute the deterministic fixture skill "
          f"{FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME}."
        ),
      },
    )
  )

  assert [event.type for event in events] == [
    "text_delta",
    "text_end",
    "tool_use_start",
    "tool_use_delta",
    "tool_use_end",
    "message_end",
  ]
  assert events[2].tool_name == "emit_dashboard_artifact"
  assert events[4].tool_name == "emit_dashboard_artifact"
  assert events[4].tool_input == {
    "payload": expected_payload,
    "summary": "Deterministic dev-only DashboardArtifact fixture for Hank web live QA.",
    "profile": "production",
  }
  assert events[-1].stop_reason == "tool_use"

  done_events = asyncio.run(
    _collect(
      provider,
      client,
      {
        "messages": [
          {"role": "assistant", "content": "tool result: ok"},
        ],
        "system_prompt": f"Execute the deterministic fixture skill {FIXTURE_DASHBOARD_ARTIFACT_SKILL_NAME}.",
      },
    )
  )
  assert done_events[-1].type == "message_end"
  assert done_events[-1].stop_reason == "end_turn"
  assert client.turn == 2


def test_fixture_approval_canvas_artifact_stream_requests_approval_with_evidence(monkeypatch) -> None:
  monkeypatch.setattr("tests.deterministic_fixture_support._fixture_run_seconds", lambda: 0.0)
  provider = FixtureProvider()
  client = provider.create_client({"model": FIXTURE_MODEL_ID})

  first_turn = asyncio.run(
    _collect(
      provider,
      client,
      {
        "messages": [],
        "system_prompt": (
          "Today is 2026-06-05. Execute the deterministic fixture skill "
          f"{FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME}."
        ),
      },
    )
  )

  assert first_turn[2].tool_name == "emit_canvas_artifact"
  assert first_turn[4].tool_name == "emit_canvas_artifact"
  assert first_turn[-1].stop_reason == "tool_use"

  second_turn = asyncio.run(
    _collect(
      provider,
      client,
      {
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "tool_result",
                "tool_use_id": "fixture_canvas_artifact_1",
                "content": '{"artifact_id":"artifact-canvas-evidence-1","status":"ok"}',
              }
            ],
          }
        ],
        "system_prompt": f"Execute the deterministic fixture skill {FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME}.",
      },
    )
  )

  assert [event.type for event in second_turn] == [
    "text_delta",
    "text_end",
    "tool_use_start",
    "tool_use_delta",
    "tool_use_end",
    "message_end",
  ]
  assert second_turn[2].tool_name == FIXTURE_APPROVAL_TOOL_NAME
  assert second_turn[4].tool_name == FIXTURE_APPROVAL_TOOL_NAME
  assert second_turn[4].tool_input["evidence_artifact"] == {
    "artifact_id": "artifact-canvas-evidence-1",
    "title": "Fixture Canvas Approval Evidence",
    "skill": "_canvas",
    "contract_name": "CanvasArtifact",
    "artifact_path": "artifacts/_canvas/artifact-canvas-evidence-1.json",
    "binary_artifact_path": "artifacts/_canvas/artifact-canvas-evidence-1.bundle.js",
    "data_source": "fixture",
  }
  assert second_turn[-1].stop_reason == "tool_use"

  done_events = asyncio.run(
    _collect(
      provider,
      client,
      {
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "tool_result",
                "tool_use_id": "fixture_canvas_artifact_approval_1",
                "content": '{"ok":true}',
              }
            ],
          }
        ],
        "system_prompt": f"Execute the deterministic fixture skill {FIXTURE_APPROVAL_CANVAS_ARTIFACT_SKILL_NAME}.",
      },
    )
  )
  assert done_events[-1].type == "message_end"
  assert done_events[-1].stop_reason == "end_turn"
  assert client.turn == 3


def test_fixture_terminal_failure_stream_raises_deterministic_error(monkeypatch) -> None:
  monkeypatch.setattr("tests.deterministic_fixture_support._fixture_run_seconds", lambda: 0.0)
  provider = FixtureProvider()
  client = provider.create_client({"model": FIXTURE_MODEL_ID})

  events = asyncio.run(
    _collect_terminal_failure_prefix(
      provider,
      client,
      {
        "messages": [],
        "system_prompt": (
          "Today is 2026-06-11. Execute the deterministic fixture skill "
          f"{FIXTURE_TERMINAL_FAILURE_SKILL_NAME}."
        ),
      },
    )
  )

  assert [event.type for event in events] == ["text_delta", "text_end"]
  assert "intentionally failing" in events[0].text
  assert client.turn == 1


def test_fixture_runner_reaches_approval_pending_then_completes(monkeypatch, tmp_path) -> None:
  monkeypatch.setenv("APP_ENV", "test")
  monkeypatch.setattr("tests.deterministic_fixture_support._fixture_run_seconds", lambda: 0.0)

  async def _run() -> EventLog:
    provider = FixtureProvider()
    event_log = EventLog()
    store = SQLiteApprovalStore(tmp_path / "approvals.sqlite")
    policy = resolve_policy(store=store)
    session = SimpleNamespace(
      user_id="alice",
      user_email="alice@example.com",
      session_id="fixture-session",
      request_id="fixture-run",
      channel="cli",
      role="owner",
      pending_tools={},
      approval_queues={},
      approval_store=store,
      approval_policy=policy,
      agent_session_log=None,
      run_context=RunContext(
        user_id="alice",
        request_id="fixture-run",
        session_id="fixture-session",
        run_id="fixture-run",
        profile="_fixture",
        channel="cli",
        decider_role="owner",
        policy_bundle_hash=str(getattr(policy, "policy_bundle_hash", "unknown")),
      ),
    )

    dispatcher = ToolDispatcher(
      mcp_client=_NullMcpClient(),
      local_tool_handlers={FIXTURE_APPROVAL_TOOL_NAME: fixture_approval_handler},
      needs_approval=lambda name, _tool_input=None, _qualifier="": name == FIXTURE_APPROVAL_TOOL_NAME,
      event_log=event_log,
      session_id="fixture-session",
      session=session,
      role=session.role,
      store=store,
      policy=policy,
      run_context=session.run_context,
    )
    message_inbox: asyncio.Queue[ParentMessage] = asyncio.Queue()
    runner = AgentRunner(
      event_log=event_log,
      dispatcher=dispatcher,
      session_id="fixture-session",
      capability_execution=stub_runner_capability_execution(
        provider=provider,
        auth_config={
          "auth_mode": "none",
          "api_key": "",
          "auth_token": "",
          "max_tokens": 1_024,
        },
        model=FIXTURE_MODEL_ID,
        effort="none",
      ),
      get_tool_definitions=lambda: [
        {
          "name": FIXTURE_APPROVAL_TOOL_NAME,
          "input_schema": {
            "type": "object",
            "properties": {
              "reason": {"type": "string"},
              "side_effect": {"type": "string"},
            },
            "required": ["reason", "side_effect"],
          },
        }
      ],
      per_turn_timeout=5.0,
      stream_stall_timeout=5.0,
      message_inbox=message_inbox,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
      channel="cli",
    )

    task = asyncio.create_task(
      runner.run(
        [{"role": "user", "content": "start fixture"}],
        system_prompt="fixture",
        max_turns=3,
      )
    )
    pending_entry = None
    for _ in range(50):
      if session.pending_tools:
        pending_entry = next(iter(session.pending_tools.values()))
        break
      await asyncio.sleep(0.01)
    assert pending_entry is not None
    assert pending_entry["status"] == "approval_pending"
    assert pending_entry["tool_name"] == FIXTURE_APPROVAL_TOOL_NAME

    await message_inbox.put(ParentMessage(message_id="steer-1", text="finish with the echo", sent_at=1.0))
    tool_call_id = next(iter(session.pending_tools))
    await session.approval_queues[tool_call_id].put({"approved": True, "allow_tool_type": False})
    await asyncio.wait_for(task, timeout=5.0)
    return event_log

  log = asyncio.run(_run())
  events = [entry.event for entry in log.entries]
  text = "".join(str(event.get("text") or "") for event in events if event.get("type") == "text_delta")
  assert "Fixture turn 1 running" in text
  assert "finish with the echo" in text
  assert any(event.get("type") == "tool_approval_request" for event in events)
  assert any(event.get("type") == "stream_complete" for event in events)
