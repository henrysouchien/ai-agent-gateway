from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import json
import sys
from types import MappingProxyType
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import SessionStore, ToolDispatcher
from agent_gateway.agent_telemetry import (
  AGENT_TELEMETRY_TOOL_CALL_ID_ENV,
)
from agent_gateway.code_execution import CodeExecutionConfig, build_code_execution, cleanup_code_execution
from agent_gateway.code_execution._provenance import (
  AGENT_CODE_EXECUTE_WORK_DIR_ENV,
  collect_native_computations,
)
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.event_log import EventLog
from agent_gateway.runner import ToolResultContext
from agent_gateway.tool_dispatch_source_identity import sandbox_computation_source_rows


def _run(coro):
  return asyncio.run(coro)


class _FakeMcp(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  async def call_tool(self, *_args, **_kwargs):
    raise AssertionError("MCP should not execute in code_execution provenance tests")


async def _dispatch_bundle_tool(
  session,
  bundle,
  tool_name: str,
  tool_input: Dict[str, Any],
  *,
  event_log: EventLog | None = None,
) -> tuple[dict | None, dict | None]:
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=_FakeMcp(),
    local_tool_handlers=bundle.handlers,
    role="owner",
    needs_approval=lambda _name, _tool_input, _qualifier: False,
    event_log=event_log or EventLog(),
    approval_key_qualifier=bundle.approval_qualifier,
  )
  return await dispatcher.dispatch(f"{tool_name}_call", tool_name, tool_input)


def _entry(function: str, output_hash: str) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "function": function,
    "tool_version": "sourced-tables-v1",
    "output_sha256": output_hash,
    "params": {"symbol": "MSCI"},
    "summary": {"accessions": ["0001408198-26-000001"]},
  }


def test_native_provenance_reaches_source_rows() -> None:
  provenance = _entry("load_statements", "a" * 64)
  del provenance["schema_version"]
  result: dict[str, Any] = {
    "return_code": 0,
    "nested_calls": [{
      "nested_call_id": "cell.1",
      "provenance": MappingProxyType(provenance),
    }],
  }

  collect_native_computations(result)

  expected = {**provenance, "nested_call_id": "cell.1"}
  assert result["computations"] == [expected]
  assert "nested_call_id" not in provenance
  assert "computations_dropped" not in result
  rows = sandbox_computation_source_rows(result)
  assert len(rows) == 1
  assert rows[0][0] == "sandbox:load_statements:sha=aaaaaaaaaaaa"
  assert rows[0][1] == expected


def test_native_provenance_is_unbounded_and_preserves_ledger_attribution() -> None:
  nested_calls = [
    {
      "nested_call_id": f"cell.{index}",
      "provenance": _entry("load_statements", f"{index:064x}"),
    }
    for index in range(20)
  ]
  result: dict[str, Any] = {"nested_calls": nested_calls}

  collect_native_computations(result)

  assert result["computations"] == [
    {**call["provenance"], "nested_call_id": call["nested_call_id"]}
    for call in nested_calls
  ]
  assert "computations_dropped" not in result


def test_invalid_native_provenance_is_ignored() -> None:
  invalid_entries: list[Any] = [None, "not-a-mapping"]
  for field in ("function", "tool_version", "output_sha256"):
    missing = _entry("load_statements", "a" * 64)
    del missing[field]
    invalid_entries.append(missing)
    invalid_entries.append({**_entry("load_statements", "a" * 64), field: " "})
    invalid_entries.append({**_entry("load_statements", "a" * 64), field: 123})
  result: dict[str, Any] = {
    "nested_calls": [
      {"nested_call_id": f"cell.{index}", "provenance": entry}
      for index, entry in enumerate(invalid_entries)
    ],
  }

  collect_native_computations(result)

  assert "computations" not in result
  assert "computations_dropped" not in result


def _write_sidecars_code(entries: list[tuple[str, dict[str, Any]]], *, sleep_s: float = 0.0) -> str:
  payload = json.dumps(entries)
  return (
    "import json, os, time\n"
    "from pathlib import Path\n"
    f"entries = json.loads({payload!r})\n"
    f"sleep_s = {sleep_s!r}\n"
    f"root = Path(os.environ[{AGENT_CODE_EXECUTE_WORK_DIR_ENV!r}]) / '.agent_provenance'\n"
    f"target = root / os.environ[{AGENT_TELEMETRY_TOOL_CALL_ID_ENV!r}]\n"
    "target.mkdir(parents=True, exist_ok=True)\n"
    "for name, body in entries:\n"
    "    (target / name).write_text(json.dumps(body), encoding='utf-8')\n"
    "(root / 'foreign-tool-call').mkdir(parents=True, exist_ok=True)\n"
    "(root / 'foreign-tool-call' / 'keep.json').write_text('{}', encoding='utf-8')\n"
    "print('sidecar-ready', flush=True)\n"
    "time.sleep(sleep_s)\n"
  )


async def _wait_for_terminal_status(session, bundle, task_id: str) -> tuple[dict | None, dict | None]:
  for _ in range(60):
    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute_status",
      {"task_id": task_id},
    )
    if error is not None or not (isinstance(result, dict) and result.get("status") == "running"):
      return result, error
    await asyncio.sleep(0.05)
  raise AssertionError(f"task did not reach terminal status: {task_id}")


def test_code_execute_sets_work_dir_env_for_foreground_and_background_subprocess() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "code": (
          "import json, os\n"
          "print(json.dumps({\n"
          f"  'work_dir': os.environ.get({AGENT_CODE_EXECUTE_WORK_DIR_ENV!r}),\n"
          f"  'tool_call_id': os.environ.get({AGENT_TELEMETRY_TOOL_CALL_ID_ENV!r}),\n"
          "}))\n"
        )
      },
    )
    assert error is None
    assert result is not None
    payload = json.loads(result["stdout"])
    assert payload == {
      "work_dir": session.code_execution_work_dir,
      "tool_call_id": "code_execute_call",
    }
    assert "computations" not in result
    assert "computations_dropped" not in result

    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "background": True,
        "code": (
          "import json, os\n"
          "print(json.dumps({\n"
          f"  'work_dir': os.environ.get({AGENT_CODE_EXECUTE_WORK_DIR_ENV!r}),\n"
          f"  'tool_call_id': os.environ.get({AGENT_TELEMETRY_TOOL_CALL_ID_ENV!r}),\n"
          "}), flush=True)\n"
        ),
      },
    )
    assert start_error is None
    assert start_result is not None
    task_id = start_result["task_id"]
    assert session.background_tasks[task_id].tool_call_id == "code_execute_call"

    complete_result, complete_error = await _wait_for_terminal_status(session, bundle, task_id)
    assert complete_error is None
    assert complete_result is not None
    payload = json.loads(complete_result["stdout"])
    assert payload == {
      "work_dir": session.code_execution_work_dir,
      "tool_call_id": "code_execute_call",
    }

  _run(_run_test())


def test_foreground_ignores_legacy_sidecars() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
    first = _entry("load_statements", "a" * 64)
    second = _entry("render_sourced_table", "b" * 64)

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"code": _write_sidecars_code([("0002-later.json", second), ("0001-first.json", first)])},
    )

    assert error is None
    assert result is not None
    assert "computations" not in result
    assert "computations_dropped" not in result
    assert session.code_execution_work_dir is not None
    provenance_root = Path(session.code_execution_work_dir) / ".agent_provenance"
    assert (provenance_root / "code_execute_call").is_dir()
    await cleanup_code_execution(session)

  _run(_run_test())


def test_sanitize_hook_strips_computations_from_result_entry_only_for_code_execute_tools() -> None:
  for tool_name in ("code_execute", "code_execute_status"):
    result = {
      "stdout": "",
      "return_code": 0,
      "timed_out": False,
      "computations": [_entry("load_statements", "a" * 64)],
    }
    result_entry = {"content": json.dumps(result)}
    ctx = ToolResultContext(
      tool_name=tool_name,
      tool_input={},
      redacted_tool_input={},
      result=dict(result),
      error=None,
      duration_ms=1,
      tool_call_id="tool-1",
      session_id="sess-1",
      server=None,
      result_entry=result_entry,
    )
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    bundle.sanitize_hook(ctx)

    visible = json.loads(result_entry["content"])
    assert "computations" not in visible
    assert ctx.result is not None
    assert ctx.result["computations"] == result["computations"]

