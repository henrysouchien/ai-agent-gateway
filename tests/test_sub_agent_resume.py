import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, AgentSessionLog, EventLog, TaskRegistry, TaskState, ToolDispatcher
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.skills import SkillLoader
from agent_gateway.sub_agent import make_resume_handler, make_resume_tool_def
from agent_gateway.task_registry import ParentMessage
from agent_gateway.transcript import (
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
  place_resume_messages,
  reconstruct_messages_for_task,
  reconstruct_parent_messages,
)


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _NoCredentialProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return False

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if False:
      yield StreamEvent(type="message_start")


def _dispatcher() -> ToolDispatcher:
  return ToolDispatcher(mcp_client=_NullMcpClient(), local_tool_handlers={}, event_log=EventLog(), session_id="sess")


def _runner(tmp_path: Path, *, max_resume_chain_depth: int = 3) -> AgentRunner:
  return AgentRunner(
    event_log=EventLog(),
    dispatcher=_dispatcher(),
    session_id="sess-parent",
    provider=_NoCredentialProvider(),
    auth_config={"model": "claude-sonnet-4-6"},
    agent_session_log=AgentSessionLog(path=tmp_path / "sessions" / "runner.jsonl"),
    task_registry=TaskRegistry(),
    max_resume_chain_depth=max_resume_chain_depth,
  )


async def _append_task_log(log: AgentSessionLog) -> None:
  await log.append(
    {
      "type": "task_registered",
      "task_id": "bg_3",
      "task_type": "background",
      "agent_name": "earnings-review",
      "sub_agent_id": "sub3:sess-parent",
      "started_at": 1.0,
    }
  )
  await log.append({"type": "user_message", "sub_agent_id": "sub3:sess-parent", "role": "sub_agent", "content": "Review AAPL"})
  await log.append(
    {
      "type": "assistant_message",
      "sub_agent_id": "sub3:sess-parent",
      "role": "sub_agent",
      "content_blocks": [
        {"type": "thinking", "thinking": "private chain state", "signature": "sig-1"},
        {"type": "tool_use", "id": "tool-a", "name": "lookup", "input": {"ticker": "AAPL"}},
        {"type": "tool_use", "id": "tool-b", "name": "lookup", "input": {"ticker": "MSFT"}},
      ],
      "stop_reason": "tool_use",
      "model": "claude-sonnet-4-6",
    }
  )
  await log.append(
    {
      "type": "tool_call_complete",
      "sub_agent_id": "sub3:sess-parent",
      "role": "sub_agent",
      "tool_call_id": "tool-a",
      "tool_name": "lookup",
      "result": {"ok": True},
      "error": None,
      "final_tool_result_blocks": [
        {"type": "tool_result", "tool_use_id": "tool-a", "content": "{\"ok\": true}"}
      ],
    }
  )


def test_transcript_reconstructs_messages_and_preserves_thinking(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "transcript.jsonl")
  _run(_append_task_log(log))

  messages = _run(reconstruct_messages_for_task(log, "bg_3"))

  assert messages[0] == {"role": "user", "content": "Review AAPL"}
  assistant = messages[1]
  assert assistant["role"] == "assistant"
  assert assistant["content"][0] == {"type": "thinking", "thinking": "private chain state", "signature": "sig-1"}
  assert messages[2]["content"][0]["tool_use_id"] == "tool-a"


def test_orphan_detection_parallel_synthesizes_only_missing_and_places_first(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "orphans.jsonl")
  _run(_append_task_log(log))
  transcript = _run(reconstruct_messages_for_task(log, "bg_3"))

  orphan_ids = detect_orphan_tool_uses(transcript)
  synthetic = build_synthetic_tool_results(orphan_ids)
  placed = place_resume_messages(
    transcript,
    synthetic,
    [ParentMessage(message_id="msg-1", text="Use latest 10-Q", sent_at=2.0)],
    "Continue carefully",
  )

  assert orphan_ids == ["tool-b"]
  assert synthetic == [
    {
      "type": "tool_result",
      "tool_use_id": "tool-b",
      "content": json.dumps(
        {
          "status": "interrupted",
          "note": "This tool call did not complete before the sub-agent was interrupted. Verify before retrying.",
        }
      ),
      "is_error": True,
    }
  ]
  closing = placed[2]
  assert closing["role"] == "user"
  assert closing["content"][0]["tool_use_id"] == "tool-b"
  assert closing["content"][0]["is_error"] is True
  assert closing["content"][1]["tool_use_id"] == "tool-a"
  assert placed[3] == {
    "role": "user",
    "content": "[Parent message id=msg-1]: Use latest 10-Q\n[Operator continuation note]: Continue carefully",
  }


def test_reconstruct_parent_messages_filters_by_task_and_before_ts(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "parent-messages.jsonl")
  _run(log.append({"type": "parent_message_sent", "task_id": "bg_3", "message_id": "m1", "message": "first", "sent_at": 10.0}))
  _run(log.append({"type": "parent_message_sent", "task_id": "bg_4", "message_id": "m2", "message": "other", "sent_at": 11.0}))
  _run(log.append({"type": "parent_message_sent", "task_id": "bg_3", "message_id": "m3", "message": "late", "sent_at": 99.0}))

  messages = _run(reconstruct_parent_messages(log, "bg_3", before_ts=50.0))

  assert messages == [ParentMessage(message_id="m1", text="first", sent_at=10.0)]


def test_register_background_task_resume_generates_r_suffix(tmp_path: Path) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      return {"response": "resumed"}, None

    result, error = await runner._register_background_task(
      tool_input={},
      handler=_handler,
      agent_name="earnings-review",
      original_task_id="bg_3",
    )
    await runner._task_registry.get("bg_3_r1").asyncio_task

    assert error is None
    assert result["task_id"] == "bg_3_r1"
    assert runner._task_registry.get("bg_3_r1").original_task_id == "bg_3"

  _run(_case())


def test_resume_chain_depth_and_cap(tmp_path: Path) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path, max_resume_chain_depth=3)
    registry = runner._task_registry
    registry.register("background_agent", task_id="bg_3")
    registry.register("background_agent", task_id="bg_3_r1", original_task_id="bg_3")
    registry.register("background_agent", task_id="bg_3_r2", original_task_id="bg_3_r1")
    registry.register("background_agent", task_id="bg_3_r3", original_task_id="bg_3_r2")

    assert await runner._resume_chain_depth("bg_3") == 0
    assert await runner._resume_chain_depth("bg_3_r2") == 2

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      return {"response": "unused"}, None

    result, error = await runner._register_background_task(
      tool_input={},
      handler=_handler,
      agent_name="earnings-review",
      original_task_id="bg_3_r3",
    )

    assert result is None
    assert error["code"] == "max_resume_chain_depth"

  _run(_case())


def test_background_task_payload_interrupted_includes_resume_fields(tmp_path: Path) -> None:
  runner = _runner(tmp_path)
  registry = runner._task_registry
  original = registry.register("background_agent", agent_name="earnings-review", task_id="bg_3")
  original.state = TaskState.INTERRUPTED
  original.metadata["resumable"] = True
  registry.register("background_agent", agent_name="earnings-review", task_id="bg_3_r1", original_task_id="bg_3")

  payload = runner._background_task_payload(original)

  assert payload["resumable"] is True
  assert payload["resumed_as"] == ["bg_3_r1"]
  assert payload["latest_resume_task_id"] == "bg_3_r1"


def test_resume_tool_def_schema() -> None:
  tool_def = make_resume_tool_def()

  assert tool_def["name"] == "resume_background_agent"
  assert tool_def["input_schema"]["required"] == ["task_id"]


def test_resume_handler_rejects_non_interrupted_task(tmp_path: Path) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    entry = runner._task_registry.register("background_agent", agent_name="earnings-review", task_id="bg_3")
    runner._task_registry.transition(entry.task_id, TaskState.COMPLETED, result={"response": "done"})
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "earnings-review.md").write_text(
      "---\nagent_callable: true\nagent_description: Earnings.\nresumable: true\n---\nPrompt",
      encoding="utf-8",
    )
    handler = make_resume_handler([runner], skill_loader=SkillLoader(skills_dir), mcp_client=_NullMcpClient())

    result, error = await handler({"task_id": "bg_3"})

    assert result is None
    assert error["code"] == "not_interrupted"

  _run(_case())


def test_resume_handler_rejects_non_resumable_skill(tmp_path: Path) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    entry = runner._task_registry.register("background_agent", agent_name="email-responder", task_id="bg_3")
    entry.state = TaskState.INTERRUPTED
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "email-responder.md").write_text(
      "---\nagent_callable: true\nagent_description: Email.\n---\nPrompt",
      encoding="utf-8",
    )
    handler = make_resume_handler([runner], skill_loader=SkillLoader(skills_dir), mcp_client=_NullMcpClient())

    result, error = await handler({"task_id": "bg_3"})

    assert result is None
    assert error["code"] == "not_resumable"

  _run(_case())
