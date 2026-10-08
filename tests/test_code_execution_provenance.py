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
  collect_computation_sidecars,
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


def test_native_provenance_collects_without_sidecars_and_reaches_source_rows(tmp_path: Path) -> None:
  for work_dir in (None, tmp_path / "absent"):
    provenance = _entry("load_statements", "a" * 64)
    del provenance["schema_version"]
    result: dict[str, Any] = {
      "return_code": 0,
      "nested_calls": [{
        "nested_call_id": "cell.1",
        "provenance": MappingProxyType(provenance),
      }],
    }

    collect_computation_sidecars(result, work_dir=work_dir, tool_call_id="cell")

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

  collect_computation_sidecars(result, work_dir=None, tool_call_id="cell")

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

  collect_computation_sidecars(result, work_dir=None, tool_call_id="cell")

  assert "computations" not in result
  assert "computations_dropped" not in result


def test_native_provenance_deduplicates_sidecars_without_consuming_legacy_limit(tmp_path: Path) -> None:
  sidecar_dir = tmp_path / ".agent_provenance" / "cell"
  sidecar_dir.mkdir(parents=True)
  native = _entry("load_statements", "a" * 64)
  legacy = [_entry("load_statements", f"{index:064x}") for index in range(17)]
  for index, entry in enumerate(legacy):
    (sidecar_dir / f"{index:04d}.json").write_text(json.dumps(entry), encoding="utf-8")
  (sidecar_dir / "9999-duplicate.json").write_text(json.dumps(native), encoding="utf-8")
  result: dict[str, Any] = {
    "nested_calls": [{"nested_call_id": "cell.1", "provenance": native}],
  }

  collect_computation_sidecars(result, work_dir=tmp_path, tool_call_id="cell")

  assert result["computations"] == [{**native, "nested_call_id": "cell.1"}, *legacy[:16]]
  assert result["computations_dropped"] == 1
  assert not sidecar_dir.exists()


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


async def _wait_for_sidecar_dir(session, tool_call_id: str = "code_execute_call") -> Path:
  assert session.code_execution_work_dir is not None
  directory = Path(session.code_execution_work_dir) / ".agent_provenance" / tool_call_id
  for _ in range(60):
    if directory.exists():
      return directory
    await asyncio.sleep(0.05)
  raise AssertionError(f"sidecar dir was not created: {directory}")


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


def test_foreground_collection_attaches_valid_sidecars_once_and_deletes_only_consumed_dir() -> None:
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
    assert result["computations"] == [first, second]
    assert "computations_dropped" not in result
    assert session.code_execution_work_dir is not None
    provenance_root = Path(session.code_execution_work_dir) / ".agent_provenance"
    assert not (provenance_root / "code_execute_call").exists()
    assert (provenance_root / "foreign-tool-call").is_dir()

  _run(_run_test())


def test_collect_computation_sidecars_counts_malformed_oversize_and_truncated_entries(tmp_path: Path, monkeypatch) -> None:
  work_dir = tmp_path / "work"
  sidecar_dir = work_dir / ".agent_provenance" / "tool-1"
  sidecar_dir.mkdir(parents=True)
  valid = _entry("load_statements", "a" * 64)
  (sidecar_dir / "0001-valid.json").write_text(json.dumps(valid), encoding="utf-8")
  (sidecar_dir / "0002-malformed.json").write_text("{not-json", encoding="utf-8")
  (sidecar_dir / "0003-invalid-shape.json").write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
  (sidecar_dir / "0004-oversize.json").write_text("x" * (8 * 1024 + 1), encoding="utf-8")

  import agent_gateway.code_execution._provenance as provenance

  original_read = provenance._read_sidecar_json

  def _read_guard(path: Path):
    if path.name == "0004-oversize.json":
      raise AssertionError("oversize sidecar was read")
    return original_read(path)

  monkeypatch.setattr(provenance, "_read_sidecar_json", _read_guard)

  result: dict[str, Any] = {"stdout": ""}
  collect_computation_sidecars(result, work_dir=work_dir, tool_call_id="tool-1")

  assert result["computations"] == [valid]
  assert result["computations_dropped"] == 3
  assert not sidecar_dir.exists()


def test_collect_computation_sidecars_truncates_after_sixteen_entries(tmp_path: Path) -> None:
  work_dir = tmp_path / "work"
  sidecar_dir = work_dir / ".agent_provenance" / "tool-1"
  sidecar_dir.mkdir(parents=True)
  for index in range(18):
    entry = _entry("load_statements", f"{index:064x}")
    (sidecar_dir / f"{index:04d}.json").write_text(json.dumps(entry), encoding="utf-8")

  result: dict[str, Any] = {"stdout": ""}
  collect_computation_sidecars(result, work_dir=work_dir, tool_call_id="tool-1")

  assert len(result["computations"]) == 16
  assert [entry["output_sha256"] for entry in result["computations"]] == [f"{index:064x}" for index in range(16)]
  assert result["computations_dropped"] == 2
  assert not sidecar_dir.exists()


def test_dot_component_tool_call_ids_never_touch_dirs_outside_provenance_root(tmp_path: Path) -> None:
  work_dir = tmp_path / "work"
  root = work_dir / ".agent_provenance"
  other_dir = root / "other-tool-call"
  other_dir.mkdir(parents=True)
  (other_dir / "keep.json").write_text(json.dumps(_entry("load_statements", "a" * 64)), encoding="utf-8")
  (work_dir / "stray.json").write_text(json.dumps(_entry("load_statements", "b" * 64)), encoding="utf-8")

  from agent_gateway.code_execution._provenance import delete_computation_sidecar_dir

  for hostile_id in (".", ".."):
    result: dict[str, Any] = {"stdout": ""}
    collect_computation_sidecars(result, work_dir=work_dir, tool_call_id=hostile_id)
    assert "computations" not in result
    assert "computations_dropped" not in result
    delete_computation_sidecar_dir(work_dir, hostile_id)

  assert work_dir.exists()
  assert (other_dir / "keep.json").exists()
  assert (work_dir / "stray.json").exists()


def test_code_execute_without_tool_context_does_not_collect_or_delete_sidecars(tmp_path: Path) -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    session.code_execution_work_dir = str(tmp_path / "work")
    sidecar_dir = Path(session.code_execution_work_dir) / ".agent_provenance" / "code_execute_call"
    sidecar_dir.mkdir(parents=True)
    (sidecar_dir / "0001-valid.json").write_text(json.dumps(_entry("load_statements", "a" * 64)), encoding="utf-8")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))

    result, error = await bundle.handlers["code_execute"]({"code": "print(1)"})

    assert result is None
    assert error == {"code": "internal_error", "message": "Backend resolution failed"}
    assert sidecar_dir.exists()

  _run(_run_test())


def test_background_terminal_status_attaches_computations_once_then_task_is_not_found() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
    sidecar = _entry("render_sourced_table", "b" * 64)

    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"background": True, "code": _write_sidecars_code([("0001-render.json", sidecar)])},
    )
    assert start_error is None
    assert start_result is not None
    task_id = start_result["task_id"]
    assert session.background_tasks[task_id].tool_call_id == "code_execute_call"

    complete_result, complete_error = await _wait_for_terminal_status(session, bundle, task_id)
    assert complete_error is None
    assert complete_result is not None
    assert complete_result["computations"] == [sidecar]

    later_result, later_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute_status",
      {"task_id": task_id},
    )
    assert later_result is None
    assert later_error == {"code": "not_found", "message": f"Unknown task_id: {task_id}"}

  _run(_run_test())


def test_background_nonterminal_status_does_not_collect_sidecars() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "background": True,
        "code": _write_sidecars_code([("0001-load.json", _entry("load_statements", "a" * 64))], sleep_s=5),
      },
    )
    assert start_error is None
    assert start_result is not None
    task_id = start_result["task_id"]
    sidecar_dir = await _wait_for_sidecar_dir(session)

    running_result, running_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute_status",
      {"task_id": task_id},
    )

    assert running_error is None
    assert running_result is not None
    assert running_result["status"] == "running"
    assert "computations" not in running_result
    assert sidecar_dir.exists()

    await _dispatch_bundle_tool(session, bundle, "code_execute_status", {"task_id": task_id, "cancel": True})

  _run(_run_test())


def test_background_cancel_after_completion_attaches_computations() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
    sidecar = _entry("render_sourced_table", "b" * 64)
    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {"background": True, "code": _write_sidecars_code([("0001-render.json", sidecar)])},
    )
    assert start_error is None
    assert start_result is not None
    task_id = start_result["task_id"]
    await asyncio.sleep(0.2)

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute_status",
      {"task_id": task_id, "cancel": True},
    )

    assert error is None
    assert result is not None
    assert result["computations"] == [sidecar]
    assert task_id not in session.background_tasks

  _run(_run_test())


def test_background_cancel_while_running_deletes_sidecars_without_attaching() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "background": True,
        "code": _write_sidecars_code([("0001-load.json", _entry("load_statements", "a" * 64))], sleep_s=30),
      },
    )
    assert start_error is None
    assert start_result is not None
    task_id = start_result["task_id"]
    sidecar_dir = await _wait_for_sidecar_dir(session)

    result, error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute_status",
      {"task_id": task_id, "cancel": True},
    )

    assert error is None
    assert result is not None
    assert result == {"status": "cancelled", "task_id": task_id}
    assert "computations" not in result
    assert not sidecar_dir.exists()

  _run(_run_test())


def test_background_timed_out_terminal_result_keeps_computations_for_app_gate() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
    sidecar = _entry("render_sourced_table", "b" * 64)
    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "background": True,
        "timeout_ms": 1000,
        "code": _write_sidecars_code([("0001-render.json", sidecar)], sleep_s=30),
      },
    )
    assert start_error is None
    assert start_result is not None
    task_id = start_result["task_id"]

    result, error = await _wait_for_terminal_status(session, bundle, task_id)

    assert error is None
    assert result is not None
    assert result["timed_out"] is True
    assert result["computations"] == [sidecar]

  _run(_run_test())


def test_cleanup_code_execution_deletes_running_task_sidecars_without_minting() -> None:
  async def _run_test() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig(register_docker=False))
    start_result, start_error = await _dispatch_bundle_tool(
      session,
      bundle,
      "code_execute",
      {
        "background": True,
        "code": _write_sidecars_code([("0001-load.json", _entry("load_statements", "a" * 64))], sleep_s=30),
      },
    )
    assert start_error is None
    assert start_result is not None
    sidecar_dir = await _wait_for_sidecar_dir(session)
    work_dir = Path(session.code_execution_work_dir or "")

    await cleanup_code_execution(session)

    assert session.background_tasks == {}
    assert session.code_execution_work_dir is None
    assert not sidecar_dir.exists()
    assert not work_dir.exists()

  _run(_run_test())


def test_sanitize_hook_strips_computations_from_result_entry_only_for_code_execute_tools() -> None:
  for tool_name in ("code_execute", "code_execute_status"):
    result = {
      "stdout": "",
      "return_code": 0,
      "timed_out": False,
      "computations": [_entry("load_statements", "a" * 64)],
      "computations_dropped": 1,
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
    assert "computations_dropped" not in visible
    assert ctx.result is not None
    assert ctx.result["computations"] == result["computations"]
    assert ctx.result["computations_dropped"] == 1

