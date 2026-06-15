from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
API_DIR = ROOT / "api"
PKG_DIR = ROOT / "packages" / "agent-gateway"
for path in (ROOT, API_DIR, PKG_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent_gateway import AgentRunner, AgentSessionLog, EventLog, TaskRegistry, TaskState, ToolDispatcher, create_agent
import agent_gateway.autonomous as autonomous_module
from agent_gateway.autonomous import run_autonomous
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.server import ChatRequest
from agent_gateway.skills import SkillLoader
from agent_gateway.sub_agent import _DEFAULT_EXCLUDED_TOOLS, make_resume_handler, make_run_agent_handler


WRITE_TOOLS = {
  "code_execute",
  "code_execute_status",
  "emit_html_artifact",
  "file_edit",
  "file_write",
  "memory_store",
  "memory_write",
  "run_bash",
  "send_message",
}


def _run(coro):
  return asyncio.run(coro)


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _StubProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    _ = config
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    _ = kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if False:
      yield StreamEvent(type="message_start")


class _CapturingRunner:
  def __init__(self) -> None:
    self._full_session_id = "sess-parent"
    self.calls: list[dict[str, Any]] = []

  async def spawn_sub_agent(self, task: str, **kwargs: Any):
    self.calls.append({"task": task, **kwargs})
    return {"response": "ok"}, None


async def _local_tool(_tool_input: dict[str, Any], **_kwargs: Any):
  return {"ok": True}, None


def _local_tool_handlers() -> dict[str, Any]:
  return {
    "code_execute": _local_tool,
    "code_execute_status": _local_tool,
    "emit_html_artifact": _local_tool,
    "file_edit": _local_tool,
    "file_write": _local_tool,
    "memory_store": _local_tool,
    "memory_write": _local_tool,
    "reader_action": _local_tool,
    "run_bash": _local_tool,
    "send_message": _local_tool,
  }


def _write_skill(
  skills_dir: Path,
  name: str,
  *,
  mutation_mode: str | None = None,
  resumable: bool = False,
) -> None:
  skills_dir.mkdir(parents=True, exist_ok=True)
  lines = [
    "---",
    f"name: {name}",
    "agent_callable: true",
    "agent_description: Phase-0 test skill.",
  ]
  if mutation_mode is not None:
    lines.append(f"mutation_mode: {mutation_mode}")
  if resumable:
    lines.append("resumable: true")
  lines.extend(["---", "", "Do the work.", ""])
  (skills_dir / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def _direct_package_run_agent_call(tmp_path: Path, *, mutation_mode: str | None) -> dict[str, Any]:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "phase0-agent", mutation_mode=mutation_mode)
  runner = _CapturingRunner()
  handler = make_run_agent_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_NullMcpClient(),
    local_tool_handlers=_local_tool_handlers(),
    excluded_tools={"parent-base"},
    default_model="stub-model",
    allowed_models={"stub-model"},
  )

  result, error = _run(handler({"agent": "phase0-agent", "task": "Review MSFT"}))

  assert error is None
  assert result == {"response": "ok"}
  assert len(runner.calls) == 1
  return runner.calls[0]


@pytest.mark.parametrize("mutation_mode", ["preview", "read_only"])
def test_package_run_agent_opted_in_mode_excludes_write_tools(
  tmp_path: Path,
  mutation_mode: str,
) -> None:
  call = _direct_package_run_agent_call(tmp_path, mutation_mode=mutation_mode)

  assert WRITE_TOOLS <= set(call["excluded_tools"])
  assert WRITE_TOOLS.isdisjoint(call["dispatcher"]._local)
  if mutation_mode == "read_only":
    assert "reader_action" in call["excluded_tools"]
    assert "reader_action" not in call["dispatcher"]._local
  else:
    assert "reader_action" not in call["excluded_tools"]
    assert "reader_action" in call["dispatcher"]._local


def test_package_run_agent_legacy_mode_none_leaves_effective_excluded_unchanged(
  tmp_path: Path,
) -> None:
  call = _direct_package_run_agent_call(tmp_path, mutation_mode=None)

  assert set(call["excluded_tools"]) == set(_DEFAULT_EXCLUDED_TOOLS) | {"parent-base"}
  assert {"file_write", "file_edit", "run_bash", "memory_store", "memory_write"} <= set(call["dispatcher"]._local)
  assert (WRITE_TOOLS - set(_DEFAULT_EXCLUDED_TOOLS) - {"emit_html_artifact"}).isdisjoint(call["excluded_tools"])


def _resume_runner(tmp_path: Path) -> AgentRunner:
  return AgentRunner(
    event_log=EventLog(),
    dispatcher=ToolDispatcher(
      mcp_client=_NullMcpClient(),
      local_tool_handlers={},
      event_log=EventLog(),
      session_id="sess-parent",
    ),
    session_id="sess-parent",
    provider=_StubProvider(),
    auth_config={"model": "stub-model"},
    agent_session_log=AgentSessionLog(path=tmp_path / "sessions" / "runner.jsonl"),
    task_registry=TaskRegistry(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


async def _append_interrupted_task(runner: AgentRunner, *, task_id: str, agent_name: str) -> None:
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
      "content": "Resume MSFT review.",
    }
  )


async def _resume_call(tmp_path: Path, *, mutation_mode: str | None) -> dict[str, Any]:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "phase0-agent", mutation_mode=mutation_mode, resumable=True)
  runner = _resume_runner(tmp_path)
  await _append_interrupted_task(runner, task_id="bg_phase0", agent_name="phase0-agent")
  captured: dict[str, Any] = {}

  async def _resume_sub_agent(**kwargs: Any):
    captured.update(kwargs)
    return {"response": "continued"}, None

  runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
  handler = make_resume_handler(
    [runner],
    skill_loader=SkillLoader(skills_dir),
    mcp_client=_NullMcpClient(),
    local_tool_handlers=_local_tool_handlers(),
    excluded_tools_resolver=lambda: frozenset({"parent-base"}),
    default_model="stub-model",
    allowed_models={"stub-model"},
  )

  result, error = await handler(
    {"task_id": "bg_phase0"},
    tool_ctx=SimpleNamespace(tool_call_id="turn-resume"),
  )

  assert error is None
  assert result is not None
  resumed_entry = runner._task_registry.get(result["task_id"])
  assert resumed_entry is not None
  await resumed_entry.asyncio_task
  return captured


@pytest.mark.parametrize("mutation_mode", ["preview", "read_only"])
def test_package_resume_opted_in_mode_excludes_write_tools_from_resumed_skill(
  tmp_path: Path,
  mutation_mode: str,
) -> None:
  captured = _run(_resume_call(tmp_path, mutation_mode=mutation_mode))

  assert WRITE_TOOLS <= set(captured["excluded_tools"])
  assert WRITE_TOOLS.isdisjoint(captured["dispatcher"]._local)
  if mutation_mode == "read_only":
    assert "reader_action" in captured["excluded_tools"]
  else:
    assert "reader_action" not in captured["excluded_tools"]


def test_package_resume_legacy_mode_none_leaves_effective_excluded_unchanged(
  tmp_path: Path,
) -> None:
  captured = _run(_resume_call(tmp_path, mutation_mode=None))

  assert set(captured["excluded_tools"]) == set(_DEFAULT_EXCLUDED_TOOLS) | {"parent-base"}
  assert {"file_write", "file_edit", "run_bash", "memory_store", "memory_write"} <= set(
    captured["dispatcher"]._local
  )
  assert (WRITE_TOOLS - set(_DEFAULT_EXCLUDED_TOOLS) - {"emit_html_artifact"}).isdisjoint(
    captured["excluded_tools"]
  )


def _build_easy_runtime(app):
  session = app.state.auth.session_store.create_session(api_key_hash="hash", user_id="alice")
  runtime = _run(
    app.state.gateway_config.build_chat_runtime(
      session=session,
      request=ChatRequest(messages=[{"role": "user", "content": "hello"}], context={}),
      channel=None,
      auth_manager=app.state.auth,
    )
  )
  return session, runtime


@pytest.mark.parametrize("mutation_mode", ["preview", "read_only"])
def test_easy_create_agent_run_agent_wiring_uses_package_enforcement(
  tmp_path: Path,
  mutation_mode: str,
) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "phase0-agent", mutation_mode=mutation_mode)
  app = create_agent(
    "test",
    provider="openai",
    model="gpt-4o-mini",
    skills_dir=skills_dir,
    skills_excluded_tools={"parent-base"},
    tool_handlers=_local_tool_handlers(),
  )
  session, runtime = _build_easy_runtime(app)
  runner = runtime.build_runner(EventLog(), session.session_id)
  captured: dict[str, Any] = {}

  async def _spawn_sub_agent(task: str, **kwargs: Any):
    captured.update({"task": task, **kwargs})
    return {"response": "ok"}, None

  runner.spawn_sub_agent = _spawn_sub_agent  # type: ignore[method-assign]

  result, error = _run(runner._dispatcher._local["run_agent"]({"agent": "phase0-agent", "task": "Review MSFT"}))

  assert error is None
  assert result == {"response": "ok"}
  assert WRITE_TOOLS <= set(captured["excluded_tools"])
  assert WRITE_TOOLS.isdisjoint(captured["dispatcher"]._local)


@pytest.mark.parametrize("mutation_mode", ["preview", "read_only"])
def test_autonomous_run_agent_wiring_uses_package_enforcement(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  mutation_mode: str,
) -> None:
  skills_dir = tmp_path / "skills"
  _write_skill(skills_dir, "phase0-agent", mutation_mode=mutation_mode)
  captured: dict[str, Any] = {}

  async def _fake_run_session(runner: AgentRunner, _event_log: EventLog, **_kwargs: Any):
    async def _spawn_sub_agent(task: str, **spawn_kwargs: Any):
      captured.update({"task": task, **spawn_kwargs})
      return {"response": "ok"}, None

    runner.spawn_sub_agent = _spawn_sub_agent  # type: ignore[method-assign]
    result, error = await runner._dispatcher._local["run_agent"]({"agent": "phase0-agent", "task": "Review MSFT"})
    assert error is None
    assert result == {"response": "ok"}
    return autonomous_module.RunOutput(response="done", tools_used=[], usage={}, error=None, timed_out=False)

  monkeypatch.setattr(autonomous_module, "run_session", _fake_run_session)

  output = _run(
    run_autonomous(
      "test",
      "start",
      provider="openai",
      model="gpt-4o-mini",
      skills_dir=skills_dir,
      skills_excluded_tools={"parent-base"},
      tool_handlers=_local_tool_handlers(),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
  )

  assert output.error is None
  assert WRITE_TOOLS <= set(captured["excluded_tools"])
  assert WRITE_TOOLS.isdisjoint(captured["dispatcher"]._local)
