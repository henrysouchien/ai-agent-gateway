import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, AgentSessionLog, EventLog, TaskRegistry, TaskState, ToolDispatcher
from agent_gateway.html_artifact_store import read_html_artifact_content, read_html_artifact_sidecar
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.session import GatewaySession
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

_UNRESOLVED_BLOCK_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


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
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
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


def _write_skill(skills_dir: Path, name: str, frontmatter: str = "", *, body: str = "Resume carefully.") -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  lines = [
    "---",
    f"name: {name}",
    "agent_callable: true",
    "agent_description: Test skill.",
    "resumable: true",
  ]
  if frontmatter:
    lines.extend(frontmatter.strip().splitlines())
  lines.extend(["---", body])
  (skills_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


async def _append_interrupted_skill_task(
  runner: AgentRunner,
  *,
  task_id: str,
  agent_name: str,
  user_message: str,
) -> None:
  entry = runner._task_registry.register("background_agent", agent_name=agent_name, task_id=task_id)
  entry.state = TaskState.INTERRUPTED
  log = runner._agent_session_log
  assert log is not None
  await log.append(
    {
      "type": "task_registered",
      "task_id": task_id,
      "task_type": "background",
      "agent_name": agent_name,
      "sub_agent_id": f"sub:{task_id}",
      "started_at": 1.0,
    }
  )
  await log.append(
    {
      "type": "user_message",
      "task_id": task_id,
      "sub_agent_id": f"sub:{task_id}",
      "role": "sub_agent",
      "content": user_message,
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
    "content": (
      "Operator update for this task:\n"
      "- id=msg-1: Use latest 10-Q\n"
      "[Operator continuation note]: Continue carefully"
    ),
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
    handler = make_resume_handler(
      [runner],
      skill_loader=SkillLoader(skills_dir),
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

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
    handler = make_resume_handler(
      [runner],
      skill_loader=SkillLoader(skills_dir),
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": "bg_3"})

    assert result is None
    assert error["code"] == "not_resumable"

  _run(_case())


def test_resume_handler_uses_sub_agent_default_model_knob(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_opus48",
      agent_name="earnings-review",
      user_message="Resume earnings review.",
    )
    captured: dict[str, Any] = {}

    async def _resume_sub_agent(**kwargs: Any):
      captured.update(kwargs)
      return {"response": "continued"}, None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    monkeypatch.setenv("SUB_AGENT_DEFAULT_MODEL", "claude-opus-4-8")
    handler = make_resume_handler(
      [runner],
      skill_loader=SkillLoader(skills_dir),
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      default_model="claude-sonnet-4-6",
      allowed_models={"claude-sonnet-4-6", "claude-opus-4-8"},
    )

    result, error = await handler({"task_id": "bg_opus48"})

    assert error is None
    assert result is not None
    resumed_entry = runner._task_registry.get(result["task_id"])
    assert resumed_entry is not None
    await resumed_entry.asyncio_task
    assert captured["model"] == "claude-opus-4-8"

  _run(_case())


def test_resume_handler_resolves_skill_blocks_before_resume(tmp_path: Path) -> None:
  async def _case() -> None:
    skills_dir = tmp_path / "skills"
    blocks_dir = skills_dir / "_blocks"
    blocks_dir.mkdir(parents=True)
    (blocks_dir / "citation-contract.md").write_text(
      "Resolved resume citation contract.\nSecond line stays verbatim.\n",
      encoding="utf-8",
    )
    _write_skill(
      skills_dir,
      "earnings-review",
      body="Resume carefully.\n{{CITATION_CONTRACT}}\nFinish.",
    )
    assert "{{CITATION_CONTRACT}}" in SkillLoader(skills_dir).load("earnings-review").system_prompt
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_block",
      agent_name="earnings-review",
      user_message="Resume AAPL earnings review.",
    )
    captured: dict[str, Any] = {}

    async def _resume_sub_agent(**kwargs: Any):
      captured.update(kwargs)
      return {"response": "continued"}, None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    handler = make_resume_handler(
      [runner],
      skill_loader=SkillLoader(skills_dir),
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      default_model="claude-sonnet-4-6",
      allowed_models={"claude-sonnet-4-6"},
    )

    result, error = await handler({"task_id": "bg_block"})

    assert error is None
    assert result is not None
    resumed_entry = runner._task_registry.get(result["task_id"])
    assert resumed_entry is not None
    await resumed_entry.asyncio_task
    prompt = captured["system_prompt"]
    assert "Resolved resume citation contract.\nSecond line stays verbatim.\n" in prompt
    assert not _UNRESOLVED_BLOCK_RE.search(prompt)

  _run(_case())


def test_resume_background_emits_skill_run_started_before_completion(tmp_path: Path) -> None:
  async def _case() -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review", "scope: ticker")
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_delayed",
      agent_name="earnings-review",
      user_message="Resume PCTY earnings review.",
    )
    parent_log = EventLog()
    entered_resume = asyncio.Event()
    release_resume = asyncio.Event()

    async def _resume_sub_agent(**kwargs: Any):
      _ = kwargs
      assert [entry.event["type"] for entry in parent_log.entries] == ["skill_run_started"]
      entered_resume.set()
      await release_resume.wait()
      return {"response": "continued"}, None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    handler = make_resume_handler(
      [runner],
      parent_session=GatewaySession(
        session_id="sess-parent",
        api_key_hash="hash",
        created_at=10,
        expires_at=20,
        user_id="alice",
        auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
      ),
      skill_loader=SkillLoader(skills_dir),
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      default_model="claude-sonnet-4-6",
      allowed_models={"claude-sonnet-4-6"},
    )

    result, error = await handler(
      {"task_id": "bg_delayed", "additional_context": "Ticker: PCTY"},
      tool_ctx=SimpleNamespace(tool_call_id="turn-resume", emit=parent_log.append),
    )

    assert error is None
    assert result is not None
    resumed_entry = runner._task_registry.get(result["task_id"])
    assert resumed_entry is not None
    await asyncio.wait_for(entered_resume.wait(), timeout=1)
    events_before_completion = [entry.event for entry in parent_log.entries]
    assert [event["type"] for event in events_before_completion] == ["skill_run_started"]
    assert events_before_completion[0]["skill"] == "earnings-review"
    assert events_before_completion[0]["ticker"] == "PCTY"

    release_resume.set()
    await resumed_entry.asyncio_task
    assert [entry.event["type"] for entry in parent_log.entries] == [
      "skill_run_started",
      "skill_result_captured",
    ]

  _run(_case())


def test_resume_handler_installs_emit_html_artifact_for_resumable_named_skill(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  memory.set_memory_store_factory(None)
  try:
    async def _case() -> None:
      skills_dir = tmp_path / "skills"
      _write_skill(
        skills_dir,
        "html-research",
        """
scope: ticker
extra_excluded_tools:
  - memory_write
  - file_write
""",
      )
      runner = _runner(tmp_path)
      await _append_interrupted_skill_task(
        runner,
        task_id="bg_html",
        agent_name="html-research",
        user_message="Resume PCTY html analysis.",
      )
      captured: dict[str, Any] = {}

      async def _resume_sub_agent(**kwargs: Any):
        captured.update(kwargs)
        return {"response": "continued"}, None

      async def _stub_emit_html_artifact(_tool_input: dict[str, Any], **_kwargs: Any):
        return None, {"code": "stub", "message": "stub"}

      runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
      parent_log = EventLog()
      handler = make_resume_handler(
        [runner],
        parent_session=GatewaySession(
          session_id="sess-parent",
          api_key_hash="hash",
          created_at=10,
          expires_at=20,
          user_id="alice-slug",
          user_email="alice@example.com",
          risk_user_id=77,
          auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
        ),
        skill_loader=SkillLoader(skills_dir),
        mcp_client=_NullMcpClient(),
        local_tool_handlers={"emit_html_artifact": _stub_emit_html_artifact},
        excluded_tools_resolver=lambda: frozenset({"emit_html_artifact"}),
        default_model="claude-sonnet-4-6",
        allowed_models={"claude-sonnet-4-6"},
      )

      result, error = await handler(
        {"task_id": "bg_html"},
        tool_ctx=SimpleNamespace(tool_call_id="turn-resume", emit=parent_log.append),
      )
      assert error is None
      assert result is not None
      resumed_entry = runner._task_registry.get(result["task_id"])
      assert resumed_entry is not None
      await resumed_entry.asyncio_task

      dispatcher = captured["dispatcher"]
      assert "emit_html_artifact" not in captured["excluded_tools"]
      assert "memory_write" in captured["excluded_tools"]
      assert "file_write" in captured["excluded_tools"]
      assert dispatcher._local["emit_html_artifact"] is not _stub_emit_html_artifact
      emit_result, emit_error = await dispatcher.dispatch(
        "tool_html_resume",
        "emit_html_artifact",
        {
          "title": "PCTY Resume View",
          "purpose": "exploration",
          "summary": "PCTY resumed HTML output.",
          "html": "<main><h1>PCTY</h1></main>",
          "copy_as_json": {"ticker": "PCTY"},
          "sources": [],
        },
      )

      assert emit_error is None
      assert emit_result is not None
      artifact_id = emit_result["artifact_id"]
      workspace_dir = memory.get_workspace_dir("77")
      sidecar = read_html_artifact_sidecar(workspace_dir, artifact_id)
      assert sidecar is not None
      assert sidecar.title == "PCTY Resume View"
      assert sidecar.ticker == "PCTY"
      assert sidecar.source_skill == "html-research"
      assert sidecar.exports.copy_as_json == {"ticker": "PCTY"}
      assert read_html_artifact_content(workspace_dir, artifact_id) == "<main><h1>PCTY</h1></main>"

      events = [entry.event for entry in parent_log.entries]
      assert [event["type"] for event in events] == [
        "skill_run_started",
        "skill_result_captured",
        "artifact_ready",
      ]
      assert events[0]["skill"] == "html-research"
      assert events[0]["ticker"] == "PCTY"
      assert events[1]["skill"] == "html-research"
      ready = events[2]
      assert ready["artifact_id"] == artifact_id
      assert ready["ticker"] == "PCTY"
      assert ready["skill"] == "_html"
      assert ready["artifact_path"] == f"artifacts/_html/{artifact_id}.json"
      assert ready["binary_artifact_path"] == f"artifacts/_html/{artifact_id}.html"
      assert ready["contract_name"] == "HtmlArtifact"

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)


def test_resume_emit_html_ticker_prefers_first_user_or_context_over_later_user_messages(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  memory.set_memory_store_factory(None)
  try:
    async def _case() -> None:
      skills_dir = tmp_path / "skills"
      _write_skill(skills_dir, "html-research", "scope: ticker")
      runner = _runner(tmp_path)
      await _append_interrupted_skill_task(
        runner,
        task_id="bg_scope",
        agent_name="html-research",
        user_message="Resume the html analysis carefully.",
      )
      log = runner._agent_session_log
      assert log is not None
      await log.append(
        {
          "type": "user_message",
          "task_id": "bg_scope",
          "sub_agent_id": "sub:bg_scope",
          "role": "sub_agent",
          "content": '{"ticker":"MSFT","note":"later tool-result-like message"}',
        }
      )
      captured: dict[str, Any] = {}

      async def _resume_sub_agent(**kwargs: Any):
        captured.update(kwargs)
        return {"response": "continued"}, None

      runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
      parent_log = EventLog()
      handler = make_resume_handler(
        [runner],
        parent_session=GatewaySession(
          session_id="sess-parent",
          api_key_hash="hash",
          created_at=10,
          expires_at=20,
          user_id="alice",
          auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
        ),
        skill_loader=SkillLoader(skills_dir),
        mcp_client=_NullMcpClient(),
        local_tool_handlers={},
        excluded_tools_resolver=frozenset,
        default_model="claude-sonnet-4-6",
        allowed_models={"claude-sonnet-4-6"},
      )

      result, error = await handler(
        {"task_id": "bg_scope", "additional_context": "Focus on AAPL."},
        tool_ctx=SimpleNamespace(tool_call_id="turn-resume", emit=parent_log.append),
      )
      assert error is None
      assert result is not None
      resumed_entry = runner._task_registry.get(result["task_id"])
      assert resumed_entry is not None
      await resumed_entry.asyncio_task

      emit_result, emit_error = await captured["dispatcher"].dispatch(
        "tool_html_scope",
        "emit_html_artifact",
        {
          "title": "AAPL Resume View",
          "purpose": "exploration",
          "summary": "AAPL resumed HTML output.",
          "html": "<main><h1>AAPL</h1></main>",
          "sources": [],
        },
      )

      assert emit_error is None
      assert emit_result is not None
      artifact_id = emit_result["artifact_id"]
      sidecar = read_html_artifact_sidecar(memory.get_workspace_dir("alice"), artifact_id)
      assert sidecar is not None
      assert sidecar.ticker == "AAPL"
      events = [entry.event for entry in parent_log.entries]
      assert [event["type"] for event in events] == [
        "skill_run_started",
        "skill_result_captured",
        "artifact_ready",
      ]
      assert events[0]["ticker"] == "AAPL"
      assert events[1]["ticker"] == "AAPL"
      assert events[2]["ticker"] == "AAPL"

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)


def test_resume_emit_html_ticker_uses_parent_resume_message_context(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  memory.set_memory_store_factory(None)
  try:
    async def _case() -> None:
      skills_dir = tmp_path / "skills"
      _write_skill(skills_dir, "html-research", "scope: ticker")
      runner = _runner(tmp_path)
      await _append_interrupted_skill_task(
        runner,
        task_id="bg_parent_scope",
        agent_name="html-research",
        user_message="Resume the html analysis carefully.",
      )
      log = runner._agent_session_log
      assert log is not None
      await log.append(
        {
          "type": "parent_message_sent",
          "task_id": "bg_parent_scope",
          "message_id": "msg-msft",
          "message": "Focus on MSFT for the resumed artifact.",
          "sent_at": 2.0,
        }
      )
      captured: dict[str, Any] = {}

      async def _resume_sub_agent(**kwargs: Any):
        captured.update(kwargs)
        return {"response": "continued"}, None

      runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
      parent_log = EventLog()
      handler = make_resume_handler(
        [runner],
        parent_session=GatewaySession(
          session_id="sess-parent",
          api_key_hash="hash",
          created_at=10,
          expires_at=20,
          user_id="alice",
          auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
        ),
        skill_loader=SkillLoader(skills_dir),
        mcp_client=_NullMcpClient(),
        local_tool_handlers={},
        excluded_tools_resolver=frozenset,
        default_model="claude-sonnet-4-6",
        allowed_models={"claude-sonnet-4-6"},
      )

      result, error = await handler(
        {"task_id": "bg_parent_scope"},
        tool_ctx=SimpleNamespace(tool_call_id="turn-resume", emit=parent_log.append),
      )
      assert error is None
      assert result is not None
      resumed_entry = runner._task_registry.get(result["task_id"])
      assert resumed_entry is not None
      await resumed_entry.asyncio_task

      emit_result, emit_error = await captured["dispatcher"].dispatch(
        "tool_html_parent_scope",
        "emit_html_artifact",
        {
          "title": "MSFT Resume View",
          "purpose": "exploration",
          "summary": "MSFT resumed HTML output.",
          "html": "<main><h1>MSFT</h1></main>",
          "sources": [],
        },
      )

      assert emit_error is None
      assert emit_result is not None
      artifact_id = emit_result["artifact_id"]
      sidecar = read_html_artifact_sidecar(memory.get_workspace_dir("alice"), artifact_id)
      assert sidecar is not None
      assert sidecar.ticker == "MSFT"
      events = [entry.event for entry in parent_log.entries]
      assert [event["type"] for event in events] == [
        "skill_run_started",
        "skill_result_captured",
        "artifact_ready",
      ]
      assert events[0]["ticker"] == "MSFT"
      assert events[0]["scope"] == "ticker"
      assert events[1]["ticker"] == "MSFT"
      assert events[2]["ticker"] == "MSFT"
      assert events[2]["scope"] == "ticker"

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)


def test_resume_emit_html_ticker_scope_falls_back_to_portfolio_when_no_ticker(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  memory.set_memory_store_factory(None)
  try:
    async def _case() -> None:
      skills_dir = tmp_path / "skills"
      _write_skill(skills_dir, "html-research", "scope: ticker")
      runner = _runner(tmp_path)
      await _append_interrupted_skill_task(
        runner,
        task_id="bg_no_scope",
        agent_name="html-research",
        user_message="Resume the html analysis carefully.",
      )
      captured: dict[str, Any] = {}

      async def _resume_sub_agent(**kwargs: Any):
        captured.update(kwargs)
        return {"response": "continued"}, None

      runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
      parent_log = EventLog()
      handler = make_resume_handler(
        [runner],
        parent_session=GatewaySession(
          session_id="sess-parent",
          api_key_hash="hash",
          created_at=10,
          expires_at=20,
          user_id="alice",
          auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
        ),
        skill_loader=SkillLoader(skills_dir),
        mcp_client=_NullMcpClient(),
        local_tool_handlers={},
        excluded_tools_resolver=frozenset,
        default_model="claude-sonnet-4-6",
        allowed_models={"claude-sonnet-4-6"},
      )

      result, error = await handler(
        {"task_id": "bg_no_scope"},
        tool_ctx=SimpleNamespace(tool_call_id="turn-resume", emit=parent_log.append),
      )
      assert error is None
      assert result is not None
      resumed_entry = runner._task_registry.get(result["task_id"])
      assert resumed_entry is not None
      await resumed_entry.asyncio_task

      emit_result, emit_error = await captured["dispatcher"].dispatch(
        "tool_html_no_scope",
        "emit_html_artifact",
        {
          "title": "Portfolio Resume View",
          "purpose": "exploration",
          "summary": "Resumed HTML output without a ticker.",
          "html": "<main><h1>Portfolio</h1></main>",
          "sources": [],
        },
      )

      assert emit_error is None
      assert emit_result is not None
      artifact_id = emit_result["artifact_id"]
      sidecar = read_html_artifact_sidecar(memory.get_workspace_dir("alice"), artifact_id)
      assert sidecar is not None
      assert sidecar.ticker is None
      events = [entry.event for entry in parent_log.entries]
      assert [event["type"] for event in events] == [
        "skill_run_started",
        "skill_result_captured",
        "artifact_ready",
      ]
      assert events[0]["ticker"] is None
      assert events[0]["scope"] == "portfolio"
      assert events[1]["ticker"] is None
      assert events[2]["ticker"] is None
      assert events[2]["scope"] == "portfolio"

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)


def test_resume_emit_html_artifact_failure_emits_tool_write_failed(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import memory

  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "data"))
  memory.set_memory_store_factory(None)
  try:
    async def _case() -> None:
      skills_dir = tmp_path / "skills"
      _write_skill(skills_dir, "portfolio-report", "scope: portfolio")
      runner = _runner(tmp_path)
      await _append_interrupted_skill_task(
        runner,
        task_id="bg_portfolio",
        agent_name="portfolio-report",
        user_message="Resume portfolio HTML report.",
      )
      captured: dict[str, Any] = {}

      async def _resume_sub_agent(**kwargs: Any):
        captured.update(kwargs)
        return {"response": "continued"}, None

      runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
      parent_log = EventLog()
      handler = make_resume_handler(
        [runner],
        parent_session=GatewaySession(
          session_id="sess-parent",
          api_key_hash="hash",
          created_at=10,
          expires_at=20,
          user_id="alice",
          auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
        ),
        skill_loader=SkillLoader(skills_dir),
        mcp_client=_NullMcpClient(),
        local_tool_handlers={},
        excluded_tools_resolver=frozenset,
        default_model="claude-sonnet-4-6",
        allowed_models={"claude-sonnet-4-6"},
      )

      result, error = await handler(
        {"task_id": "bg_portfolio"},
        tool_ctx=SimpleNamespace(tool_call_id="turn-resume", emit=parent_log.append),
      )
      assert error is None
      assert result is not None
      resumed_entry = runner._task_registry.get(result["task_id"])
      assert resumed_entry is not None
      await resumed_entry.asyncio_task

      emit_result, emit_error = await captured["dispatcher"].dispatch(
        "tool_html_bad",
        "emit_html_artifact",
        {
          "title": "Broken report",
          "purpose": "report",
          "summary": "Empty html field.",
          "html": "",
          "sources": [],
        },
      )

      assert emit_result is None
      assert emit_error is not None
      assert emit_error["code"] == "internal_error"
      assert "non-empty string" in emit_error["message"]
      events = [entry.event for entry in parent_log.entries]
      assert [event["type"] for event in events] == [
        "skill_run_started",
        "skill_result_captured",
        "artifact_failed",
      ]
      assert events[0]["ticker"] is None
      assert events[0]["scope"] == "portfolio"
      failed = events[2]
      assert failed["ticker"] is None
      assert failed["skill"] == "_html"
      assert failed["error_code"] == "tool_write_failed"
      assert failed["tool_call_id"] == "tool_html_bad"
      assert not (memory.get_workspace_dir("alice") / "artifacts" / "_html").exists()

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)
