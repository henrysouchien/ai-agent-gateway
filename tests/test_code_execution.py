import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import SessionStore, ToolDispatcher
from agent_gateway.code_execution import CodeExecutionConfig, build_code_execution, cleanup_code_execution
from agent_gateway.event_log import EventLog
from agent_gateway.runner import ToolResultContext


def _run(coro):
  return asyncio.run(coro)


class _FakeMcp:
  def is_mcp_tool(self, _tool_name: str) -> bool:
    return False

  async def call_tool(self, _tool_name: str, _tool_input: Dict[str, Any]):
    raise AssertionError("MCP should not execute in code_execution tests")


async def _dispatch_bundle_tool(
  session,
  bundle,
  tool_name: str,
  tool_input: Dict[str, Any],
  *,
  event_log: EventLog | None = None,
  needs_approval=None,
  approved_tool_types=None,
) -> tuple[dict | None, dict | None]:
  dispatcher = ToolDispatcher(
    mcp_client=_FakeMcp(),
    local_tool_handlers=bundle.handlers,
    needs_approval=needs_approval or (lambda _name, _tool_input, _qualifier: False),
    approved_tool_types=approved_tool_types,
    event_log=event_log or EventLog(),
    approval_key_qualifier=bundle.approval_qualifier,
  )
  return await dispatcher.dispatch(f"{tool_name}_call", tool_name, tool_input)


def test_tool_defs_default_config_include_code_execution_tools() -> None:
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
  bundle = build_code_execution(session)

  assert [tool_def["name"] for tool_def in bundle.tool_definitions] == [
    "code_execute",
    "code_execute_status",
  ]
  assert bundle.tool_definitions[0]["input_schema"]["properties"]["host"]["enum"] == [
    "auto",
    "subprocess",
    "docker",
  ]


def test_code_execute_basic_execution_and_work_dir_persistence() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    first_result, first_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": 'from pathlib import Path\nPath("artifact.txt").write_text("persisted", encoding="utf-8")'},
    )
    second_result, second_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": 'print(open("artifact.txt").read())'},
    )

    assert first_error is None
    assert first_result is not None
    assert second_error is None
    assert second_result is not None
    assert second_result["stdout"] == "persisted\n"
    assert second_result["stderr"] == ""
    assert second_result["return_code"] == 0
    assert second_result["images"] == []
    assert second_result["timed_out"] is False
    assert second_result["truncated"] is False
    assert session.code_execution_work_dir is not None
    assert list(Path(session.code_execution_work_dir).glob("_code_execute_*.py")) == []

  asyncio.run(_run_test())


def test_code_execute_captures_images_and_extra_env(tmp_path: Path) -> None:
  async def _run_test() -> None:
    support_dir = tmp_path / "support"
    support_dir.mkdir()
    (support_dir / "helpermod.py").write_text('VALUE = "ok"\n', encoding="utf-8")

    session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, extra_env={"PYTHONPATH": str(support_dir)}),
    )

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import helpermod\n"
          "import matplotlib.pyplot as plt\n"
          "print(helpermod.VALUE)\n"
          "plt.plot([1, 2, 3], [3, 1, 4])\n"
          "plt.show()\n"
        )
      },
    )

    assert error is None
    assert result is not None
    assert result["stdout"] == "ok\n"
    assert result["images"]
    assert result["images"][0]["filename"] == "_plot_1.png"
    assert result["images"][0]["media_type"] == "image/png"
    assert result["images"][0]["data_base64"]

  asyncio.run(_run_test())


def test_code_execute_respects_image_base64_limit() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
    bundle = build_code_execution(
      session,
      config=CodeExecutionConfig(register_docker=False, max_image_base64_bytes=10),
    )

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": 'from pathlib import Path\nPath("large.png").write_bytes(b"x" * 64)'},
    )

    assert error is None
    assert result is not None
    assert len(result["images"]) == 1
    assert result["images"][0]["filename"] == "large.png"
    assert result["images"][0]["skipped"] is True
    assert "exceeds" in result["images"][0]["reason"]

  asyncio.run(_run_test())


def test_code_execute_streaming_emits_chunk_events() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
    event_log = EventLog()
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import time\n"
          "for idx in range(5):\n"
          "    print(f'line-{idx}', flush=True)\n"
          "    time.sleep(0.05)\n"
        )
      },
      event_log=event_log,
    )

    chunk_events = [entry.event for entry in event_log.entries if entry.event.get("type") == "tool_output_chunk"]
    assert error is None
    assert result is not None
    assert len(chunk_events) == 5
    assert [event["seq"] for event in chunk_events] == [1, 2, 3, 4, 5]
    assert all(event["tool_call_id"] == "code_execute_call" for event in chunk_events)
    assert result["stdout"] == "".join(f"line-{idx}\n" for idx in range(5))

  asyncio.run(_run_test())


def test_code_execute_background_status_and_cancel_flow() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import time\n"
          "print('started', flush=True)\n"
          "time.sleep(0.5)\n"
          "print('finished', flush=True)\n"
        ),
        "background": True,
      },
    )

    assert start_error is None
    assert start_result is not None
    assert start_result["status"] == "running"
    task_id = start_result["task_id"]

    running_result = None
    for _ in range(10):
      await asyncio.sleep(0.05)
      running_result, running_error = await _dispatch_bundle_tool(
        session,
        bundle,
        "code_execute_status",
        {"task_id": task_id},
      )
      assert running_error is None
      if running_result and "started" in running_result.get("stdout_tail", ""):
        break
    assert running_result is not None
    assert running_result["status"] == "running"
    assert "started" in running_result["stdout_tail"]

    await asyncio.sleep(0.8)

    complete_result, complete_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute_status",
      {"task_id": task_id},
    )
    assert complete_error is None
    assert complete_result is not None
    assert complete_result["stdout"] == "started\nfinished\n"
    assert task_id not in session.background_tasks

    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import time\n"
          "print('begin', flush=True)\n"
          "time.sleep(30)\n"
        ),
        "background": True,
      },
    )
    assert start_error is None
    assert start_result is not None
    cancel_task_id = start_result["task_id"]
    task = session.background_tasks[cancel_task_id]

    cancel_result, cancel_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute_status",
      {"task_id": cancel_task_id, "cancel": True},
    )

    assert cancel_error is None
    assert cancel_result == {"status": "cancelled", "task_id": cancel_task_id}
    assert cancel_task_id not in session.background_tasks
    assert task.handle._backend_data["process"].returncode is not None

  asyncio.run(_run_test())


def test_cleanup_code_execution_cancels_tasks_and_is_idempotent() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import time\n"
          "print('still-running', flush=True)\n"
          "time.sleep(30)\n"
        ),
        "background": True,
      },
    )

    assert start_error is None
    assert start_result is not None
    task = session.background_tasks[start_result["task_id"]]
    work_dir = Path(session.code_execution_work_dir or "")

    await cleanup_code_execution(session)
    await cleanup_code_execution(session)

    assert session.background_tasks == {}
    assert task._terminated is True
    assert task.handle._backend_data["process"].returncode is not None
    assert session.code_execution_work_dir is None
    assert not work_dir.exists()

  asyncio.run(_run_test())


def test_backend_qualified_approval_depends_on_sandboxing() -> None:
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
  subprocess_bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
  docker_bundle = build_code_execution(
    session,
    config=CodeExecutionConfig(register_subprocess=False, register_docker=True),
  )

  subprocess_qualifier = subprocess_bundle.approval_qualifier("code_execute", {"code": "print(1)"})
  docker_qualifier = docker_bundle.approval_qualifier("code_execute", {"code": "print(1)", "host": "docker"})

  assert subprocess_qualifier == "subprocess"
  assert subprocess_bundle.needs_approval("code_execute", {"code": "print(1)"}, subprocess_qualifier) is True
  assert docker_qualifier == "docker"
  assert docker_bundle.needs_approval("code_execute", {"code": "print(1)"}, docker_qualifier) is False


def test_strip_code_execute_base64_hook_rewrites_model_history() -> None:
  result_entry = {
    "content": json.dumps(
      {
        "stdout": "",
        "stderr": "",
        "return_code": 0,
        "images": [{"filename": "plot.png", "media_type": "image/png", "data_base64": "Zm9v"}],
      }
    )
  }
  ctx = ToolResultContext(
    tool_name="code_execute",
    tool_input={"code": "print(1)"},
    result=None,
    error=None,
    duration_ms=1,
    tool_call_id="tool_1",
    session_id="sess_1",
    server=None,
    result_entry=result_entry,
  )
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash")
  bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

  bundle.sanitize_hook(ctx)

  payload = json.loads(result_entry["content"])
  assert payload["images"][0]["data_base64"] == "[image: plot.png]"


def test_per_bundle_backend_isolation_keeps_distinct_docker_images(monkeypatch) -> None:
  async def _fake_execute(self, code, work_dir, **kwargs):
    _ = code, work_dir, kwargs
    return {
      "stdout": f"{self._image}\n",
      "stderr": "",
      "return_code": 0,
      "images": [],
      "timed_out": False,
      "duration_ms": 1,
      "truncated": False,
    }

  monkeypatch.setattr("agent_gateway.code_execution._backends._docker.DockerBackend.available", lambda self: True)
  monkeypatch.setattr("agent_gateway.code_execution._backends._docker.DockerBackend.execute", _fake_execute)

  async def _run_test() -> None:
    session_one = SessionStore(ttl=3600).create_session(api_key_hash="hash")
    session_two = SessionStore(ttl=3600).create_session(api_key_hash="hash")
    bundle_one = build_code_execution(
      session_one,
      config=CodeExecutionConfig(register_subprocess=False, docker_image="image-one:latest"),
    )
    bundle_two = build_code_execution(
      session_two,
      config=CodeExecutionConfig(register_subprocess=False, docker_image="image-two:latest"),
    )

    result_one, error_one = await _dispatch_bundle_tool(
      session_one,
      bundle_one,
      "code_execute",
      {"code": "print(1)"},
    )
    result_two, error_two = await _dispatch_bundle_tool(
      session_two,
      bundle_two,
      "code_execute",
      {"code": "print(1)"},
    )

    assert error_one is None
    assert error_two is None
    assert result_one is not None
    assert result_two is not None
    assert result_one["stdout"] == "image-one:latest\n"
    assert result_two["stdout"] == "image-two:latest\n"

  asyncio.run(_run_test())
