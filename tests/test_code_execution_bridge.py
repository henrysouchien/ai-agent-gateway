"""`code_execute` bridge: nested `hank.*` calls from a running cell.

The cell reaches the session's tools through frames on the session kernel's
stdio; the host frame reader dispatches an operation iff
it is in the turn's grant (the bound catalog) and the bound code-callable
registration, through the bound dispatcher under the outer call, records it in
the outer result's `nested_calls`, and carries registered producer settlement
identities plus the bound extractor's richer handles in `sources`.
"""
from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import json
import os
import re
import signal
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_DIR.parent.parent
if str(REPO_ROOT / "api") not in sys.path:
  sys.path.insert(0, str(REPO_ROOT / "api"))
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, SessionStore, ToolDispatcher
from agent_gateway.approval_policy import RunContext
from agent_gateway import claim_signing_authority
from agent_gateway.claim_signing_authority import (
  GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV,
  GatewayClaimSigningAuthority,
  LEGACY_CLAIM_HMAC_KEY_ENV,
  install_gateway_claim_signing_authority,
)
from agent_gateway.user_runtime_credentials import (
  DeskReplica,
  USER_RUNTIME_CREDENTIALS_FILE_ENV,
  UserRuntimeIdentity,
  mint_user_runtime_credentials,
  write_user_runtime_credentials,
)
from agent_gateway.code_execution import (
  CodeExecutionConfig,
  build_code_execution,
  cleanup_code_execution,
)
from agent_gateway.code_execution._backends._kernel import KernelBackend
from agent_gateway.code_execution._frame_codec import decode_frame
from agent_gateway.event_log import EventLog
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.policy_imports import configure_server_policy, load_server_policy_module
from agent_gateway.session import SessionStream
from agent_gateway.task_registry import TaskEntry, TaskState
from agent_gateway.tool_dispatcher_helpers import ToolResult
from agent_gateway.tool_policy_registry import PreparedToolCall
from gateway_test_support.capability_execution_test_support import stub_runner_capability_execution
from agent_gateway import tool_dispatch_declarations
from agent_gateway.tool_dispatch_declarations import ToolDispatchSemantics
from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  VersionedPolicyRef,
)

GET_FILINGS = "mcp__edgar-parser-mcp__get_filings"
FRED_SERIES = "mcp__fred-mcp__fred_get_series"
REFRESH_FILINGS = "mcp__edgar-parser-mcp__refresh_filings"
SEARCH_FILINGS = "mcp__edgar-parser-mcp__search_filings"
EXECUTE_TRADE = "mcp__portfolio-trades-mcp__execute_trade"
OUTER_ID = "toolu_outer_1"
# The code-callable decision the product binds, keyed by exact route here.
# `search_filings` is registered and routable but never in the grant.
_CODE_CALLABLE_ROUTES = frozenset({GET_FILINGS, FRED_SERIES, SEARCH_FILINGS})
_CODE_CALLABLE = _CODE_CALLABLE_ROUTES.__contains__

_SERVERS = {
  GET_FILINGS: "edgar-parser-mcp",
  FRED_SERIES: "fred-mcp",
  REFRESH_FILINGS: "edgar-parser-mcp",
  SEARCH_FILINGS: "edgar-parser-mcp",
  EXECUTE_TRADE: "portfolio-trades-mcp",
}
# Ledger telemetry only: the reader's decision never reads a class.
_POLICY_CLASSES = {
  ("edgar-parser-mcp", "get_filings"): "read",
  ("fred-mcp", "fred_get_series"): "read",
  ("edgar-parser-mcp", "refresh_filings"): "state_write",
  ("edgar-parser-mcp", "search_filings"): "read",
}
# The registration owner's dispatch semantics for the producer route this module
# settles (GP54): the reader settles `sources` only from a declared source identity.
_DISPATCH_SEMANTICS = {
  RegisteredToolIdentity(
    route_kind="mcp", logical_server_id="edgar-parser-mcp", logical_name="get_filings",
  ): ToolDispatchSemantics(
    idempotent=True,
    outcome_policy=VersionedPolicyRef(
      kind="outcome", policy_id="current-declared-status-equals", version="v1",
      parameters={"field": "status", "values": ("success",)},
    ),
    source_identity_policy=VersionedPolicyRef(
      kind="source_identity", policy_id="parser_filings", version="v1", parameters={},
    ),
  ),
}
_FILINGS_RESULT = {
  "status": "success",
  "ticker": "AAPL",
  "filings": [
    {
      "document_id": "edgar:0000320193-24-000123", "accession": "0000320193-24-000123",
      "accession_number": "0000320193-24-000123", "form": "10-K",
    },
    {
      "document_id": "edgar:0000320193-24-000081", "accession": "0000320193-24-000081",
      "accession_number": "0000320193-24-000081", "form": "10-Q",
    },
  ],
}
_FRED_RESULT = {"series_id": "CPIAUCSL", "observations": [{"date": "2026-08-01", "value": "321.4"}]}
_FILING_HANDLES = [
  {"document_id": row["document_id"], "source_kind": "filing", "accession": row["accession"]}
  for row in _FILINGS_RESULT["filings"]
]


def _filings_extractor(operation: str, result: Any) -> list[dict[str, Any]]:
  if operation == GET_FILINGS and isinstance(result, dict) and result.get("filings"):
    return [dict(handle) for handle in _FILING_HANDLES]
  return []


def _raising_extractor(operation: str, result: Any) -> list[dict[str, Any]]:
  raise RuntimeError(f"extractor broke on {operation}")


@pytest.fixture(autouse=True)
def _policy(monkeypatch: pytest.MonkeyPatch) -> Any:
  # The declaration table is cached process-wide on first use; build it from
  # this module's policy and drop it at teardown so it never leaks.
  monkeypatch.setattr(tool_dispatch_declarations, "_CACHED_DECLARATIONS", None)
  previous = load_server_policy_module()
  configure_server_policy(types.SimpleNamespace(
    get_forbidden_tools_for_session=lambda *_args, **_kwargs: frozenset(),
    get_server_for_policy_tool=lambda name: {
      "get_filings": "edgar-parser-mcp",
      "fred_get_series": "fred-mcp",
      "refresh_filings": "edgar-parser-mcp",
      "search_filings": "edgar-parser-mcp",
    }.get(name),
    get_tool_class=lambda server, tool: _POLICY_CLASSES.get((server, tool)),
    get_local_tool_effect=lambda _name: None,
    get_dispatch_semantics_overrides=lambda: _DISPATCH_SEMANTICS,
  ))
  yield
  configure_server_policy(previous)


class _CaptureMcp(McpClientManager):
  def __init__(self, *, delay_s: float = 0.0, frame_reader: Any = None) -> None:
    super().__init__(config_path=None)
    self.calls: list[tuple[str, dict[str, Any]]] = []
    self.active_during_call: list[tuple[str, ...]] = []
    self.cancelled = False
    self._delay_s = delay_s
    self._frame_reader = frame_reader

  def is_mcp_tool(self, name: str) -> bool:
    return name in _SERVERS

  def get_server_for_tool(self, name: str) -> str | None:
    return _SERVERS.get(name)

  async def call_tool(
    self,
    name: str,
    tool_input: dict[str, Any] | PreparedToolCall,
    meta: object | None = None,
    abort_event: asyncio.Event | None = None,
    gateway_session: object | None = None,
    allow_uncertain_replay: bool = True,
    trusted_dispatch_scope: object | None = None,
  ) -> ToolResult:
    _ = (meta, abort_event, gateway_session, allow_uncertain_replay, trusted_dispatch_scope)
    assert isinstance(tool_input, dict)
    self.calls.append((name, dict(tool_input)))
    if self._frame_reader is not None:
      self.active_during_call.append(self._frame_reader.active_cells())
    if self._delay_s:
      try:
        await asyncio.sleep(self._delay_s)
      except asyncio.CancelledError:
        self.cancelled = True
        raise
    if tool_input.get("cik") == "missing":
      return None, {"code": "not_found", "message": "no such company"}
    if name == GET_FILINGS:
      return dict(_FILINGS_RESULT), None
    if name == REFRESH_FILINGS:
      return {"refreshed": True}, None
    return dict(_FRED_RESULT), None


def _tool_definitions(bundle: Any) -> list[dict[str, Any]]:
  object_schema = {"type": "object", "additionalProperties": True}
  return [
    *bundle.tool_definitions,
    {"name": GET_FILINGS, "description": "List SEC filings for a company.", "input_schema": object_schema},
    {
      "name": FRED_SERIES,
      "description": "Fetch one FRED series.",
      "input_schema": {"type": "object", "properties": {"series_id": {"type": "string"}}, "required": ["series_id"]},
    },
    {"name": REFRESH_FILINGS, "description": "Refresh cached filings.", "input_schema": object_schema},
    {"name": EXECUTE_TRADE, "description": "Execute a portfolio trade.", "input_schema": object_schema},
    {"name": "file_read", "description": "Read a file.", "input_schema": object_schema},
  ]


async def _file_read(tool_input: dict[str, Any], **_kwargs: Any) -> ToolResult:
  return {"text": "local"}, None


def _bind(
  bundle: Any,
  mcp: _CaptureMcp,
  *,
  extract_sources: Any = None,
  code_callable: Any = None,
  run_context: RunContext | None = None,
) -> ToolDispatcher:
  """One request's dispatcher over `bundle`, bound as the product's builder binds it."""
  definitions = _tool_definitions(bundle)
  dispatcher = ToolDispatcher(
    capability_bind=TEST_CAPABILITY_BIND,
    mcp_client=mcp,
    local_tool_handlers={**bundle.handlers, "file_read": _file_read},
    role="owner",
    needs_approval=lambda _name, _tool_input, _qualifier: False,
    event_log=EventLog(),
    approval_key_qualifier=bundle.approval_qualifier,
    get_tool_definitions=lambda: definitions,
    run_context=run_context,
  )
  bundle.bind_dispatcher(
    dispatcher,
    tool_definitions=dispatcher.get_tool_definitions,
    extract_sources=extract_sources,
    code_callable=code_callable if code_callable is not None else _CODE_CALLABLE,
  )
  return dispatcher


class _StubProvider:
  name = "stub"


def _job_runner() -> AgentRunner:
  """The runner whose task registry owns the session's background cells."""
  event_log = EventLog()
  return AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=_CaptureMcp(),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess-code-bridge",
    ),
    session_id="sess-code-bridge",
    capability_execution=stub_runner_capability_execution(
      provider=_StubProvider(),
      model="stub-model",
      effort="none",
    ),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def _setup(
  *,
  delay_s: float = 0.0,
  extract_sources: Any = None,
  code_callable: Any = None,
) -> tuple[Any, Any, _CaptureMcp, ToolDispatcher]:
  """A session on a live turn whose runner owns background cells as `code` jobs."""
  session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
  runner = _job_runner()
  session.active_turn = SessionStream(event_log=EventLog(), runner_task=None, runtime_owner=runner)
  bundle = build_code_execution(
    session,
    config=CodeExecutionConfig(),
    job_runner=lambda: runner,
  )
  mcp = _CaptureMcp(delay_s=delay_s, frame_reader=bundle.frame_reader)
  return session, bundle, mcp, _bind(bundle, mcp, extract_sources=extract_sources, code_callable=code_callable)


def _code_job(session: Any, task_id: str) -> TaskEntry:
  entry = session.active_turn.runtime_owner._task_registry.get(task_id)
  assert entry is not None and entry.task_type == "code"
  return entry


def _record_started_handles(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
  handles: list[Any] = []
  start_bridged = KernelBackend.start_bridged

  async def _recording_start_bridged(self: Any, *args: Any, **kwargs: Any) -> Any:
    handle = await start_bridged(self, *args, **kwargs)
    handles.append(handle)
    return handle

  monkeypatch.setattr(KernelBackend, "start_bridged", _recording_start_bridged)
  return handles


async def _until(predicate: Any) -> None:
  for _ in range(1500):
    if predicate():
      return
    await asyncio.sleep(0.01)
  raise AssertionError("condition not reached")


async def _collect_background(session: Any, task_id: str) -> Any:
  """Await the `code` job and read it through the registry's result payload."""
  entry = _code_job(session, task_id)
  assert entry.asyncio_task is not None
  await entry.asyncio_task
  return session.active_turn.runtime_owner._background_task_payload(entry)


async def _run_cell(dispatcher: ToolDispatcher, code: str, **extra: Any) -> tuple[Any, Any]:
  return await dispatcher.dispatch(
    OUTER_ID,
    "code_execute",
    {"code": textwrap.dedent(code), **extra},
  )


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("skill_run_id", ["run-enclosing", "run-foreign"])
def test_nested_dispatch_inherits_enclosing_execution_context(
  monkeypatch: pytest.MonkeyPatch, background: bool, skill_run_id: str,
) -> None:
  async def _run() -> None:
    contexts: list[Any] = []

    async def capture_context(tool_input: dict[str, Any], **kwargs: Any) -> ToolResult:
      contexts.append(kwargs["tool_ctx"])
      return {"text": "captured"}, None

    monkeypatch.setattr(sys.modules[__name__], "_file_read", capture_context)
    session, bundle, mcp, _dispatcher = _setup()
    dispatcher = _bind(
      bundle, mcp, code_callable=lambda operation: operation == "file_read",
      run_context=RunContext(
        user_id="alice", request_id="request-enclosing", run_id="run-enclosing",
      ),
    )
    dispatch_context = {
      "skill_run_id": skill_run_id,
      "step_id": "step-enclosing",
      "workspace_dir": session.code_execution_work_dir,
      "batch_id": "batch-enclosing",
    }
    try:
      direct, error = await dispatcher.dispatch("direct", "file_read", {}, **dispatch_context)
      assert error is None and direct == {"text": "captured"}
      result, error = await dispatcher.dispatch(
        OUTER_ID, "code_execute",
        {
          "code": 'print(hank.call("file_read"))',
          "background": background,
        },
        **dispatch_context,
      )
      assert error is None
      assert result is not None
      if background:
        await _collect_background(session, result["task_id"])
      assert len(contexts) == 2, result
      direct_ctx, nested_ctx = contexts
      # step_id is not part of the dispatch_prepared contract (runner_tool_execution.py).
      for field in ("skill_run_id", "workspace_dir", "batch_id", "run_id", "request_id", "user_id"):
        assert getattr(nested_ctx, field) == getattr(direct_ctx, field), field
      assert nested_ctx.skill_run_id == skill_run_id
      assert nested_ctx.run_id == "run-enclosing"
      assert (nested_ctx.skill_run_id == nested_ctx.run_id) is (skill_run_id == "run-enclosing")
      assert nested_ctx.tool_call_id != direct_ctx.tool_call_id
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_two_reads_compose_in_one_cell() -> None:
  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup()
    try:
      result, error = await _run_cell(dispatcher, f"""
        filings = hank.call({GET_FILINGS!r}, cik="0000320193", limit=2)
        series = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        print(len(filings["filings"]), filings["filings"][0]["form"])
        print(series["observations"][0]["value"])
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "2 10-K\n321.4\n"
      assert mcp.calls == [
        (GET_FILINGS, {"cik": "0000320193", "limit": 2}),
        (FRED_SERIES, {"series_id": "CPIAUCSL"}),
      ]
      nested = result["nested_calls"]
      assert [entry["ordinal"] for entry in nested] == [1, 2]
      assert [entry["nested_call_id"] for entry in nested] == [f"{OUTER_ID}.1", f"{OUTER_ID}.2"]
      assert all(entry["parent_call_id"] == OUTER_ID for entry in nested)
      assert [entry["server"] for entry in nested] == ["edgar-parser-mcp", "fred-mcp"]
      assert all(entry["policy_class"] == "read" for entry in nested)
      assert all(entry["ok"] is True and entry["error_code"] is None for entry in nested)
      assert all(entry["duration_ms"] >= 0 and entry["result_bytes"] > 0 for entry in nested)
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_reader_refuses_ungranted_non_code_callable_and_unknown_operations() -> None:
  async def _run() -> None:
    session, _bundle, mcp, dispatcher = _setup()
    try:
      result, error = await _run_cell(dispatcher, f"""
        for op in ({SEARCH_FILINGS!r}, {REFRESH_FILINGS!r}, "mcp__nowhere__nothing"):
            try:
                hank.call(op, cik="0000320193")
            except HankBridgeError as exc:
                print(op, exc.code)
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"].splitlines() == [
        f"{SEARCH_FILINGS} not_granted",
        f"{REFRESH_FILINGS} not_code_callable",
        "mcp__nowhere__nothing unknown_operation",
      ]
      assert mcp.calls == []
      assert [(entry["operation"], entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [
        (SEARCH_FILINGS, False, "not_granted"),
        (REFRESH_FILINGS, False, "not_code_callable"),
        ("mcp__nowhere__nothing", False, "unknown_operation"),
      ]
      assert result["sources"] == []
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


@pytest.mark.parametrize("load_fails", [False, True])
def test_in_cell_load_tools_refreshes_the_grant_only_after_success(load_fails: bool) -> None:
  async def _run() -> None:
    session, bundle, mcp, _dispatcher = _setup()
    definitions = _tool_definitions(bundle)
    deferred = next(row for row in definitions if row["name"] == FRED_SERIES)
    definitions.remove(deferred)
    definitions.append({
      "name": "load_tools",
      "input_schema": {"type": "object", "additionalProperties": True},
    })

    async def _load_tools(tool_input: dict[str, Any], **_kwargs: Any) -> ToolResult:
      assert tool_input == {"tool_names": [FRED_SERIES]}
      definitions.append(deferred)
      if load_fails:
        return None, {"code": "load_failed", "message": "loader failed after changing the surface"}
      return {"loaded": [FRED_SERIES]}, None

    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=mcp,
      local_tool_handlers={**bundle.handlers, "load_tools": _load_tools},
      role="owner",
      needs_approval=lambda _name, _tool_input, _qualifier: False,
      event_log=EventLog(),
      approval_key_qualifier=bundle.approval_qualifier,
      get_tool_definitions=lambda: definitions,
    )
    bundle.bind_dispatcher(
      dispatcher,
      tool_definitions=dispatcher.get_tool_definitions,
      code_callable=lambda name: name == "load_tools" or _CODE_CALLABLE(name),
    )
    try:
      before, error = await _run_cell(dispatcher, f"""
        try:
            hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        except HankBridgeError as exc:
            print(exc.code)
      """)
      assert error is None, error
      assert before["return_code"] == 0, before["stderr"]
      assert before["stdout"] == "not_granted\n"
      assert mcp.calls == []
      assert before["nested_calls"][0]["error_code"] == "not_granted"

      result, error = await _run_cell(dispatcher, f"""
        try:
            hank.load_tools(tool_names=[{FRED_SERIES!r}])
        except HankBridgeError as exc:
            print(exc.code)
        try:
            series = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
            print(series["series_id"])
        except HankBridgeError as exc:
            print(exc.code)
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == ("load_failed\nnot_granted\n" if load_fails else "CPIAUCSL\n")
      assert mcp.calls == ([] if load_fails else [(FRED_SERIES, {"series_id": "CPIAUCSL"})])
      assert [(entry["operation"], entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [
        ("load_tools", not load_fails, "load_failed" if load_fails else None),
        (FRED_SERIES, not load_fails, "not_granted" if load_fails else None),
      ]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_registration_is_call_scoped_to_the_running_cell() -> None:
  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup()
    try:
      result, error = await _run_cell(dispatcher, f"""
        series = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        print(series["series_id"])
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "CPIAUCSL\n"
      # The call reached MCP while the cell was registered, and only then.
      assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
      assert mcp.active_during_call == [(OUTER_ID,)]
      assert bundle.frame_reader.active_cells() == ()
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_timeout_interrupts_cell_cancels_nested_call_and_closes_channel() -> None:
  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup(delay_s=30.0)
    try:
      # A host wait never times a cell out, so the nested call runs on a spawned
      # thread: its wait is compute, and the cell's own thread hits the deadline.
      warm, error = await _run_cell(dispatcher, "pass")
      assert error is None and warm["return_code"] == 0, error
      result, error = await _run_cell(
        dispatcher,
        f"""
        import threading, time
        worker = threading.Thread(
            target=lambda: hank.call({FRED_SERIES!r}, series_id="CPIAUCSL"), daemon=True,
        )
        worker.start()
        print("calling", flush=True)
        time.sleep(30)
        print("unreachable")
        """,
        timeout_ms=1500,
      )
      assert error is None, error
      assert result["timed_out"] is True
      assert result["return_code"] != 0
      assert "unreachable" not in result["stdout"]
      assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
      assert mcp.cancelled is True
      assert [(entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [(False, "cancelled")]
      assert bundle.frame_reader.active_cells() == ()
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_cell_env_carries_the_cell_token_and_no_domain_secrets_or_bridge_fds(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("EDGAR_API_KEY", "edgar-secret")
  monkeypatch.setenv("FRED_API_KEY", "fred-secret")

  async def _run() -> None:
    session, _bundle, _mcp, dispatcher = _setup()
    try:
      result, error = await _run_cell(dispatcher, """
        import os
        print("\\n".join(sorted(os.environ)))
      """)
      assert error is None, error
      names = set(result["stdout"].splitlines())
      assert "AGENT_CODE_EXECUTE_CELL_TOKEN" in names
      assert "EDGAR_API_KEY" not in names
      assert not {name for name in names if "BRIDGE_FD" in name}
      assert "FRED_API_KEY" not in names
      assert "edgar-secret" not in result["stdout"] and "fred-secret" not in result["stdout"]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


_SOURCES_CELL = f"""
  for _ in range(2):
      hank.call({GET_FILINGS!r}, cik="0000320193", limit=2)
  try:
      hank.call({GET_FILINGS!r}, cik="missing")
  except HankBridgeError as exc:
      print(exc.code)
  hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
  print("done")
"""


def test_sources_are_deduplicated_and_payloads_stay_out_of_the_result() -> None:
  async def _run() -> None:
    session, _bundle, _mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      result, error = await _run_cell(dispatcher, _SOURCES_CELL)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "not_found\ndone\n"
      assert [(entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [
        (True, None),
        (True, None),
        (False, "not_found"),
        (True, None),
      ]
      # Repeated reads retain each filing once; the failed read adds none.
      assert result["sources"] == [{**handle, "produced_by_tool": "get_filings"} for handle in _FILING_HANDLES]
      rendered = json.dumps({key: value for key, value in result.items() if key != "sources"})
      assert "0000320193-24-000123" not in rendered
      assert "321.4" not in json.dumps(result)
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_settled_sources_survive_without_a_bound_extractor() -> None:
  async def _run() -> None:
    session, _bundle, _mcp, dispatcher = _setup()
    try:
      result, error = await _run_cell(dispatcher, _SOURCES_CELL)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert len(result["nested_calls"]) == 4
      assert result["sources"] == [
        {"document_id": handle["document_id"], "source_kind": "filing", "produced_by_tool": "get_filings"}
        for handle in _FILING_HANDLES
      ]
      rendered = json.dumps({key: value for key, value in result.items() if key != "sources"})
      assert "0000320193-24-000123" not in rendered
      assert "321.4" not in rendered
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_a_raising_extractor_leaves_the_cell_and_nested_call_intact() -> None:
  async def _run() -> None:
    session, _bundle, _mcp, dispatcher = _setup(extract_sources=_raising_extractor)
    try:
      result, error = await _run_cell(dispatcher, f"""
        filings = hank.call({GET_FILINGS!r}, cik="0000320193", limit=2)
        print(filings["filings"][0]["form"])
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "10-K\n"
      assert [(entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [(True, None)]
      assert [source["document_id"] for source in result["sources"]] == [
        handle["document_id"] for handle in _FILING_HANDLES
      ]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_search_and_describe_list_only_granted_code_callable_operations() -> None:
  async def _run() -> None:
    session, _bundle, mcp, dispatcher = _setup()
    try:
      result, error = await _run_cell(dispatcher, f"""
        import json
        print(json.dumps(sorted(row["name"] for row in hank.search(""))))
        print(json.dumps(sorted(row["name"] for row in hank.search("filings"))))
        print(json.dumps(hank.describe({FRED_SERIES!r})))
        for op in ({REFRESH_FILINGS!r}, {SEARCH_FILINGS!r}):
            try:
                hank.describe(op)
            except HankBridgeError as exc:
                print(exc.code)
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      everything, searched, described, not_callable, not_granted = result["stdout"].splitlines()
      assert json.loads(everything) == sorted([FRED_SERIES, GET_FILINGS])
      assert json.loads(searched) == [GET_FILINGS]
      assert json.loads(described) == {
        "name": FRED_SERIES,
        "description": "Fetch one FRED series.",
        "input_schema": {"type": "object", "properties": {"series_id": {"type": "string"}}, "required": ["series_id"]},
      }
      assert not_callable == "not_code_callable"
      assert not_granted == "not_granted"
      assert mcp.calls == []
      assert "nested_calls" not in result
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())

def test_bridge_call_forwards_tool_arguments_named_like_bridge_parameters() -> None:
  from agent_gateway.code_execution._bridge_client import HankBridge

  frames: list[dict[str, Any]] = []

  class Transport:
    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
      frames.append(dict(payload))
      if payload["type"] == "hello":
        return {
          "call_id": payload["call_id"], "ok": True,
          "result": {"parent_call_id": "outer", "operations": [], "tools": ["x"]},
        }
      return {"call_id": payload["call_id"], "ok": True, "result": {"ok": True}}

  client = HankBridge(Transport(), "token")
  assert client.call("x", operation="inspect") == {"ok": True}
  assert client.x(operation="inspect") == {"ok": True}
  assert client.call("x", name="n", tool="t", operation="inspect") == {"ok": True}
  client.describe("x")
  tool_frames = [frame for frame in frames if frame["type"] == "tool_call"]
  assert [frame["operation"] for frame in tool_frames] == ["x", "x", "x"]
  assert [frame["input"] for frame in tool_frames] == [
    {"operation": "inspect"},
    {"operation": "inspect"},
    {"name": "n", "tool": "t", "operation": "inspect"},
  ]
  describe_frames = [frame for frame in frames if frame["type"] == "catalog"]
  assert describe_frames[0]["operation"] == "x"


@pytest.mark.parametrize("error_code", ["not_granted", "type_error"])
@pytest.mark.parametrize("next_call", [
  pytest.param("load_tools", id="string"),
  pytest.param(["load_tools"], id="list"),
  pytest.param(
    {"tool_name": "load_tools", "arguments": "not-an-object"},
    id="non-object-arguments",
  ),
  pytest.param({}, id="empty-mapping"),
])
def test_bridge_error_rejects_malformed_next_call(next_call: Any, error_code: str) -> None:
  from agent_gateway.code_execution._bridge_client import HankBridge, HankBridgeError
  from agent_gateway.code_execution._frame_reader import _error_reply
  from pydantic import ValidationError

  class Transport:
    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
      return {
        "call_id": payload["call_id"], "ok": False,
        "error": {"code": error_code, "message": "host error", "next_call": next_call},
      }

  with pytest.raises(HankBridgeError, match="malformed next_call") as exc:
    HankBridge(Transport(), "token").call("x")
  assert exc.value.code == "protocol_error"
  if isinstance(next_call, dict):
    assert (
      "must be an object" if "arguments" in next_call else "needs 'tool_name'"
    ) in exc.value.message
  with pytest.raises(ValidationError, match="undeclared keys"):
    _error_reply(
      "tool_result",
      "call-1",
      error_code,
      "host error",
      {"tool_name": "load_tools", "arguments": {"pack": "x"}, "bogus": 1},
    )

@pytest.mark.parametrize("error_fields", [
  pytest.param({}, id="absent"),
  pytest.param({"next_call": None}, id="none"),
  pytest.param(
    {"next_call": {"tool_name": "load_tools", "arguments": {"pack": "x"}}},
    id="valid-mapping",
  ),
  pytest.param(
    {"next_call": types.MappingProxyType({"tool_name": "load_tools", "arguments": {"pack": "x"}})},
    id="valid-nondict-mapping",
  ),
])
def test_bridge_error_preserves_next_call(error_fields: dict[str, Any]) -> None:
  from agent_gateway.code_execution._bridge_client import HankBridge, HankBridgeError

  class Transport:
    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
      return {
        "call_id": payload["call_id"], "ok": False,
        "error": {"code": "not_granted", "message": "host error", **error_fields},
      }

  with pytest.raises(HankBridgeError) as exc:
    HankBridge(Transport(), "token").call("x")
  assert exc.value.code == "not_granted"
  assert exc.value.message == "host error"
  assert exc.value.next_call is error_fields.get("next_call")


def test_bridge_type_error_with_valid_next_call_remains_type_error() -> None:
  from agent_gateway.code_execution._bridge_client import HankBridge

  class Transport:
    def exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
      return {
        "call_id": payload["call_id"], "ok": False,
        "error": {
          "code": "type_error", "message": "invalid tool arguments",
          "next_call": {"tool_name": "load_tools", "arguments": {"pack": "x"}},
        },
      }

  with pytest.raises(TypeError, match="invalid tool arguments"):
    HankBridge(Transport(), "token").call("x")


@pytest.mark.parametrize("grant_pack", [None, "fms-research"])
def test_unloaded_tool_is_refused_until_it_enters_the_active_grant(grant_pack: str | None) -> None:
  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup()
    try:
      object_schema = {"type": "object", "additionalProperties": True}
      search_definition = {
        "name": SEARCH_FILINGS, "description": "Search SEC filings.", "input_schema": object_schema,
      }
      loaded = _tool_definitions(bundle)
      loaded.append({"name": "load_tools", "input_schema": object_schema})

      async def _load_tools(tool_input: dict[str, Any], **_kwargs: Any) -> ToolResult:
        assert tool_input == {"pack": grant_pack}
        loaded.append(search_definition)
        return {"loaded": [SEARCH_FILINGS]}, None

      dispatcher = ToolDispatcher(
        capability_bind=TEST_CAPABILITY_BIND,
        mcp_client=mcp,
        local_tool_handlers={**bundle.handlers, "load_tools": _load_tools},
        role="owner",
        needs_approval=lambda _name, _tool_input, _qualifier: False,
        event_log=EventLog(),
        approval_key_qualifier=bundle.approval_qualifier,
        get_tool_definitions=lambda: loaded,
      )
      bundle.bind_dispatcher(
        dispatcher,
        tool_definitions=lambda: loaded,
        code_callable=lambda name: name == "load_tools" or _CODE_CALLABLE(name),
      )
      bundle.frame_reader.bind_grant_hint(lambda name: grant_pack if name == SEARCH_FILINGS else None)
      before, error = await _run_cell(dispatcher, f"""
        import json
        print(json.dumps(sorted(row["name"] for row in hank.search(""))))
        try:
            hank.call({SEARCH_FILINGS!r}, operation="inspect")
        except HankBridgeError as exc:
            print(exc.code)
            print(exc.message)
            print(json.dumps(exc.next_call))
            if exc.next_call is not None:
                print(json.dumps(hank.call(exc.next_call)))
                print(hank.describe({SEARCH_FILINGS!r})["name"])
      """)
      assert error is None, error
      assert before["return_code"] == 0, before["stderr"]
      lines = before["stdout"].splitlines()
      assert SEARCH_FILINGS not in json.loads(lines[0])
      assert lines[1] == "not_granted"
      expected_message = (
        f'{SEARCH_FILINGS!r} is not granted to this turn; load its pack first: '
        f'hank.call("load_tools", pack="{grant_pack}") (or call it directly as a tool)'
        if grant_pack is not None else
        f"{SEARCH_FILINGS!r} is not granted to this turn; load its pack with load_tools (or call it directly as a tool)"
      )
      assert lines[2] == expected_message
      expected_next_call = (
        {"tool_name": "load_tools", "arguments": {"pack": grant_pack}}
        if grant_pack is not None else None
      )
      assert json.loads(lines[3]) == expected_next_call
      if grant_pack is not None:
        assert json.loads(lines[4]) == {"loaded": [SEARCH_FILINGS]}
        assert lines[5] == SEARCH_FILINGS
        assert before["nested_calls"][1]["operation"] == "load_tools"
        assert before["nested_calls"][1]["ok"] is True
      assert before["nested_calls"][0]["error_code"] == "not_granted"
      assert mcp.calls == []

      if grant_pack is None:
        loaded.append(search_definition)
      result, error = await _run_cell(dispatcher, f"""
        import json
        print(json.dumps(sorted(row["name"] for row in hank.search(""))))
        print(hank.describe({SEARCH_FILINGS!r})["name"])
        print(hank.call({SEARCH_FILINGS!r}, operation="inspect"))
        print(getattr(hank, {SEARCH_FILINGS!r})(operation="inspect"))
        for op in ({EXECUTE_TRADE!r}, "mcp__nowhere__nothing"):
            try:
                hank.call(op, operation="inspect")
            except HankBridgeError as exc:
                print(op, exc.code)
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      lines = result["stdout"].splitlines()
      assert SEARCH_FILINGS in json.loads(lines[0])
      assert lines[1] == SEARCH_FILINGS
      assert mcp.calls == [
        (SEARCH_FILINGS, {"operation": "inspect"}),
        (SEARCH_FILINGS, {"operation": "inspect"}),
      ]
      assert lines[-2:] == [
        f"{EXECUTE_TRADE} not_code_callable",
        "mcp__nowhere__nothing unknown_operation",
      ]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_dormant_research_door_with_grant_hint_refuses_as_not_granted_and_runs_after_load() -> None:
  async def _run() -> None:
    session, bundle, mcp, _ = _setup()
    try:
      door_name = "get_valuation_policy_context"
      door_pack = "research-corpus-mcp"
      door_def = {
        "name": door_name,
        "description": "Read ValuationPolicy authoring context",
        "input_schema": {
          "type": "object",
          "properties": {"research_file_id": {"type": "integer"}},
        },
      }
      load_def = {
        "name": "load_tools",
        "description": "Load deferred tools",
        "input_schema": {
          "type": "object",
          "properties": {"pack": {"type": "string"}},
        },
      }
      loaded: list[dict[str, Any]] = [*bundle.tool_definitions, load_def]
      local_handlers: dict[str, Any] = dict(bundle.handlers)

      async def _door_handler(tool_input: dict[str, Any], **_kw: Any):
        return {"status": "ready", "research_file_id": tool_input.get("research_file_id")}, None

      async def _load_tools(tool_input: dict[str, Any], **_kw: Any):
        assert tool_input == {"pack": door_pack}
        local_handlers[door_name] = _door_handler
        loaded.append(door_def)
        return {"loaded_servers": [door_pack], "new_tools": {door_pack: [door_name]}}, None

      local_handlers["load_tools"] = _load_tools
      dispatcher = ToolDispatcher(
        capability_bind=TEST_CAPABILITY_BIND,
        mcp_client=mcp,
        local_tool_handlers=local_handlers,
        role="owner",
        needs_approval=lambda _name, _tool_input, _qualifier: False,
        event_log=EventLog(),
        approval_key_qualifier=bundle.approval_qualifier,
        get_tool_definitions=lambda: loaded,
      )
      # Before load_tools runs, the dormant door is in neither _local nor _mcp:
      assert dispatcher.route_origin_for_tool(door_name) is None
      bundle.bind_dispatcher(
        dispatcher,
        tool_definitions=lambda: loaded,
        code_callable=lambda name: name in {"load_tools", door_name},
      )
      bundle.frame_reader.bind_grant_hint(
        lambda name: door_pack if name == door_name else None,
      )
      result, error = await _run_cell(dispatcher, f"""
        import json
        read_first = {{"tool_name": {door_name!r}, "arguments": {{"research_file_id": 1}}}}
        try:
            hank.call(read_first)
        except HankBridgeError as exc:
            print(exc.code)
            print(json.dumps(exc.next_call))
            hank.call(exc.next_call)
            ctx = hank.call(read_first)
            print(json.dumps(ctx))
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      lines = result["stdout"].splitlines()
      assert lines[0] == "not_granted"
      assert json.loads(lines[1]) == {"tool_name": "load_tools", "arguments": {"pack": door_pack}}
      assert json.loads(lines[2]) == {"status": "ready", "research_file_id": 1}
      assert [
        (entry["operation"], entry["ok"], entry["error_code"])
        for entry in result["nested_calls"]
      ] == [
        (door_name, False, "not_granted"),
        ("load_tools", True, None),
        (door_name, True, None),
      ]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_grant_hint_respects_frozen_autonomous_ceiling_and_parent_tool_ids() -> None:
  async def _run() -> None:
    session, bundle, mcp, _ = _setup()
    try:
      from agent.interactive.runtime_tooling import bind_session_code_execution
      from excel_mcp.relay import ChannelRegistry

      door_name = "get_valuation_policy_context"
      door_pack = "research-corpus-mcp"
      local_handlers = dict(bundle.handlers)
      local_handlers.pop(door_name, None)

      async def _door_handler(tool_input: dict[str, Any], **_kw: Any):
        return {"status": "ready", "research_file_id": tool_input.get("research_file_id")}, None

      async def _load_tools(tool_input: dict[str, Any], **_kw: Any):
        return {"loaded_servers": [door_pack]}, None

      local_handlers["load_tools"] = _load_tools
      session.autonomous_task_id = "job-ceiling"
      session.admitted_job_context = types.SimpleNamespace(
        tool_ids=frozenset({"load_tools"}),
        withheld=frozenset(),
      )
      dispatcher = ToolDispatcher(
        capability_bind=TEST_CAPABILITY_BIND,
        mcp_client=mcp,
        session=session,
        local_tool_handlers=local_handlers,
        role="owner",
        needs_approval=lambda _name, _tool_input, _qualifier: False,
        event_log=EventLog(),
        run_context=RunContext(user_id="u", request_id="r", profile_name="analyst"),
        get_tool_definitions=lambda: [],
      )
      session.research_door_grant_ceiling = frozenset()
      bind_session_code_execution(
        bundle,
        dispatcher,
        mcp_client_manager=mcp,
        channel_registry=ChannelRegistry(),
        tool_definitions=lambda: [],
      )
      cell = bundle.frame_reader.open_cell("outer_ceiling", abort_event=None, active=True)

      # Parent authority alone does not widen the frozen local ceiling.
      assert cell._refusal(dispatcher, door_name) == "unknown_operation"
      session.admitted_job_context.tool_ids = frozenset({"load_tools", door_name})
      assert cell._refusal(dispatcher, door_name) == "unknown_operation"

      # Once the frozen ceiling includes the door, the refusal is followable.
      local_handlers[door_name] = _door_handler
      session.research_door_grant_ceiling = frozenset({door_name})
      cell.close()
      bind_session_code_execution(
        bundle,
        dispatcher,
        mcp_client_manager=mcp,
        channel_registry=ChannelRegistry(),
        tool_definitions=lambda: [],
      )
      cell = bundle.frame_reader.open_cell("outer_ceiling_granted", abort_event=None, active=True)
      assert cell._refusal(dispatcher, door_name) == "not_granted"
      assert cell._grant_hint(door_name) == door_pack
      cell.close()
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_packless_autonomous_ceiling_does_not_hint_outside_frozen_handlers() -> None:
  async def _run() -> None:
    session, bundle, mcp, _ = _setup()
    try:
      from agent.interactive.runtime_tooling import bind_session_code_execution
      from excel_mcp.relay import ChannelRegistry

      door_name = "get_historical_foundation_context"
      local_handlers = dict(bundle.handlers)
      local_handlers["load_tools"] = lambda *_args, **_kwargs: ({"loaded": True}, None)
      local_handlers.pop(door_name, None)
      session.autonomous_task_id = "job-packless"
      session.admitted_job_context = None
      dispatcher = ToolDispatcher(
        capability_bind=TEST_CAPABILITY_BIND,
        mcp_client=mcp,
        session=session,
        local_tool_handlers=local_handlers,
        role="owner",
        needs_approval=lambda _name, _tool_input, _qualifier: False,
        event_log=EventLog(),
        run_context=RunContext(user_id="u", request_id="r", profile_name="community"),
        get_tool_definitions=lambda: [],
      )
      session.research_door_grant_ceiling = frozenset()
      bind_session_code_execution(
        bundle,
        dispatcher,
        mcp_client_manager=mcp,
        channel_registry=ChannelRegistry(),
        tool_definitions=lambda: [],
      )
      cell = bundle.frame_reader.open_cell("outer_packless", abort_event=None, active=True)
      assert cell._refusal(dispatcher, door_name) == "unknown_operation"

      async def _door_handler(tool_input: dict[str, Any], **_kw: Any):
        return {"status": "ready"}, None

      local_handlers[door_name] = _door_handler
      session.research_door_grant_ceiling = frozenset({door_name})
      cell.close()
      bind_session_code_execution(
        bundle,
        dispatcher,
        mcp_client_manager=mcp,
        channel_registry=ChannelRegistry(),
        tool_definitions=lambda: [],
      )
      cell = bundle.frame_reader.open_cell("outer_packless_granted", abort_event=None, active=True)
      assert cell._refusal(dispatcher, door_name) == "not_granted"
      assert cell._grant_hint(door_name) == "research-corpus-mcp"
      cell.close()
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_grant_hint_follows_the_load_tools_owner_not_the_profile_defaults() -> None:
  async def _run() -> None:
    from agent.interactive.runtime_tooling import bind_session_code_execution
    from excel_mcp.relay import ChannelRegistry

    async def _bound(*, ceiling: frozenset[str] | None, load_tool_packs: dict[str, Any]):
      session, bundle, mcp, _ = _setup()
      local_handlers = dict(bundle.handlers)
      local_handlers["load_tools"] = lambda *_args, **_kwargs: ({"loaded": True}, None)
      session.autonomous_task_id = "job-load-tools-owner"
      session.admitted_job_context = None
      # The caller recorded exactly what it passed to make_load_tools_handler;
      # these differ from load_profile("analyst").tool_packs and the profile ceiling.
      session.load_tool_packs = load_tool_packs
      session.research_door_grant_ceiling = ceiling
      dispatcher = ToolDispatcher(
        capability_bind=TEST_CAPABILITY_BIND,
        mcp_client=mcp,
        session=session,
        local_tool_handlers=local_handlers,
        role="owner",
        needs_approval=lambda _name, _tool_input, _qualifier: False,
        event_log=EventLog(),
        run_context=RunContext(user_id="u", request_id="r", profile_name="analyst"),
        get_tool_definitions=lambda: [],
      )
      bind_session_code_execution(
        bundle,
        dispatcher,
        mcp_client_manager=mcp,
        channel_registry=ChannelRegistry(),
        tool_definitions=lambda: [],
      )
      return session, bundle, dispatcher

    # 1. Packs the caller actually gave load_tools decide the hinted pack: with no
    #    model-versions pack, load_tools grants compare_model_versions through the
    #    research-corpus-mcp alias, not through the analyst profile's pack.
    session, bundle, dispatcher = await _bound(ceiling=None, load_tool_packs={})
    try:
      cell = bundle.frame_reader.open_cell("outer_packs", abort_event=None, active=True)
      assert cell._refusal(dispatcher, "compare_model_versions") == "not_granted"
      assert cell._grant_hint("compare_model_versions") == "research-corpus-mcp"
      cell.close()
    finally:
      await cleanup_code_execution(session)

    # 2. A door outside the ceiling the caller gave load_tools is not grantable.
    door_name = "get_valuation_policy_context"
    session, bundle, dispatcher = await _bound(ceiling=frozenset(), load_tool_packs={})
    try:
      cell = bundle.frame_reader.open_cell("outer_empty_ceiling", abort_event=None, active=True)
      assert cell._refusal(dispatcher, door_name) == "unknown_operation"
      cell.close()
    finally:
      await cleanup_code_execution(session)

    # 3. A door inside that ceiling is a followable load_tools refusal.
    session, bundle, dispatcher = await _bound(
      ceiling=frozenset({door_name}),
      load_tool_packs={},
    )
    try:
      cell = bundle.frame_reader.open_cell("outer_door_ceiling", abort_event=None, active=True)
      assert cell._refusal(dispatcher, door_name) == "not_granted"
      cell.close()
      result = await cell._tool_call({
        "type": "tool_call",
        "call_id": "c-door-ceiling",
        "next_call": {"tool_name": door_name, "arguments": {"research_file_id": 1}},
      })
      assert result["error"]["code"] == "not_granted"
      assert result["error"]["next_call"] == {
        "tool_name": "load_tools",
        "arguments": {"pack": "research-corpus-mcp"},
      }
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_root_autonomous_context_keeps_research_door_grant_hint() -> None:
  async def _run() -> None:
    from agent.interactive.runtime_tooling import bind_session_code_execution
    from agent_gateway.admission import root_context
    from excel_mcp.relay import ChannelRegistry

    session, bundle, mcp, _ = _setup()
    try:
      door_name = "get_valuation_policy_context"
      local_handlers = dict(bundle.handlers)
      local_handlers.pop(door_name, None)
      local_handlers["load_tools"] = lambda *_args, **_kwargs: ({"loaded": True}, None)
      session.autonomous_task_id = "job-root"
      root_context(session, withheld_tools=frozenset(), research_file_id=None)
      # A root with configured packs has no frozen local ceiling; its root
      # context has tool_ids=None and must not narrow the admitted doors.
      session.research_door_grant_ceiling = None
      session.load_tool_packs = {}
      dispatcher = ToolDispatcher(
        capability_bind=TEST_CAPABILITY_BIND,
        mcp_client=mcp,
        session=session,
        local_tool_handlers=local_handlers,
        role="owner",
        needs_approval=lambda _name, _tool_input, _qualifier: False,
        event_log=EventLog(),
        run_context=RunContext(user_id="u", request_id="r", profile_name="analyst"),
        get_tool_definitions=lambda: [],
      )
      bind_session_code_execution(
        bundle,
        dispatcher,
        mcp_client_manager=mcp,
        channel_registry=ChannelRegistry(),
        tool_definitions=lambda: [],
      )
      cell = bundle.frame_reader.open_cell("outer_root", abort_event=None, active=True)
      assert cell._refusal(dispatcher, door_name) == "not_granted"
      assert cell._grant_hint(door_name) == "research-corpus-mcp"
      cell.close()
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_load_tools_grant_state_is_declared_on_gateway_session() -> None:
  from dataclasses import fields

  from agent_gateway.session import GatewaySession

  declared = {item.name for item in fields(GatewaySession)}
  assert {"research_door_grant_ceiling", "load_tool_packs"} <= declared


def test_hank_outside_a_bridged_cell_and_unbound_dispatcher_report_bridge_unavailable() -> None:
  async def _run() -> None:
    session = SessionStore(ttl=3600).create_session(api_key_hash="hash", user_id="alice")
    bundle = build_code_execution(session, config=CodeExecutionConfig())
    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND,
      mcp_client=_CaptureMcp(),
      local_tool_handlers=bundle.handlers,
      role="owner",
      needs_approval=lambda _name, _tool_input, _qualifier: False,
      event_log=EventLog(),
      approval_key_qualifier=bundle.approval_qualifier,
    )
    try:
      result, error = await _run_cell(dispatcher, f"""
        try:
            hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        except HankBridgeError as exc:
            print(exc.code)
      """)
      assert error is None, error
      assert result["stdout"] == "bridge_unavailable\n"
      assert result["nested_calls"][0]["error_code"] == "bridge_unavailable"
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_background_cell_reports_its_nested_calls_on_collection() -> None:
  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup()
    try:
      started, error = await _run_cell(
        dispatcher,
        f"""
        series = hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        print(series["series_id"])
        """,
        background=True,
      )
      assert error is None, error
      task_id = started["task_id"]
      result = await _collect_background(session, task_id)
      assert result["stdout"] == "CPIAUCSL\n", result
      assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
      assert "nested_calls" not in result and "sources" not in result
      record = bundle.frame_reader.cell_record(task_id)
      assert record == bundle.frame_reader.cell_record(OUTER_ID)
      assert [(entry["nested_call_id"], entry["ok"]) for entry in record["nested_calls"]] == [(f"{OUTER_ID}.1", True)]
      assert bundle.frame_reader.active_cells() == ()
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_background_cell_is_a_code_job_whose_result_carries_its_bridge() -> None:
  """The product builds a bundle per request; the cell's channel lives on its `code` job."""

  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    try:
      started, error = await _run_cell(
        dispatcher,
        f"""
        filings = hank.call({GET_FILINGS!r}, cik="0000320193", limit=2)
        print(filings["filings"][0]["form"])
        """,
        background=True,
      )
      assert error is None, error
      task_id = started["task_id"]
      later_bundle = build_code_execution(
        session,
        config=CodeExecutionConfig(),
        job_runner=lambda: session.active_turn.runtime_owner,
      )
      _bind(later_bundle, mcp, extract_sources=_filings_extractor)
      result = await _collect_background(session, task_id)
      assert _code_job(session, task_id).state == TaskState.COMPLETED
      assert result["status"] == "completed"
      assert result["stdout"] == "10-K\n", result
      assert "nested_calls" not in result and "sources" not in result
      record = bundle.frame_reader.cell_record(task_id)
      assert record == later_bundle.frame_reader.cell_record(OUTER_ID)
      assert [(entry["nested_call_id"], entry["ok"]) for entry in record["nested_calls"]] == [(f"{OUTER_ID}.1", True)]
      assert record["sources"] == [{**handle, "produced_by_tool": "get_filings"} for handle in _FILING_HANDLES]
      assert bundle.frame_reader.active_cells() == ()
      await cleanup_code_execution(session)
      assert session.code_execution_frame_reader is None
      assert bundle.frame_reader.cell_record(task_id) == {}
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_registry_kill_cancels_a_running_background_cell_and_closes_it(monkeypatch: pytest.MonkeyPatch) -> None:
  async def _run() -> None:
    handles = _record_started_handles(monkeypatch)
    session, bundle, mcp, dispatcher = _setup(delay_s=30.0)
    try:
      started, error = await _run_cell(
        dispatcher,
        f"""
        hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        """,
        background=True,
      )
      assert error is None, error
      assert started["status"] == "running"
      entry = _code_job(session, started["task_id"])
      await _until(lambda: bool(mcp.calls))
      assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
      assert bundle.frame_reader.active_cells() == (OUTER_ID,)

      assert session.active_turn.runtime_owner.cancel_background_task(entry.task_id) is True
      assert entry.asyncio_task is not None
      await asyncio.wait_for(entry.asyncio_task, timeout=7)

      assert entry.state == TaskState.KILLED
      assert bundle.frame_reader.active_cells() == ()
      assert bundle.frame_reader._cells == {}
      assert mcp.cancelled is True
      [handle] = handles
      assert handle._backend_data["completed"] is True
      assert handle._backend_data["reader_task"].done()
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_session_cleanup_closes_a_running_background_cell(monkeypatch: pytest.MonkeyPatch) -> None:
  async def _run() -> None:
    handles = _record_started_handles(monkeypatch)
    session, bundle, mcp, dispatcher = _setup(delay_s=30.0)
    started, error = await _run_cell(
      dispatcher,
      f"""
      hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
      """,
      background=True,
    )
    assert error is None, error
    assert started["status"] == "running"
    entry = _code_job(session, started["task_id"])
    await _until(lambda: bool(mcp.calls))
    assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
    assert bundle.frame_reader.active_cells() == (OUTER_ID,)

    await cleanup_code_execution(session)

    assert entry.state == TaskState.KILLED
    assert bundle.frame_reader.active_cells() == ()
    assert mcp.cancelled is True
    [handle] = handles
    assert handle._backend_data["reader_task"].done()

  asyncio.run(_run())


def test_background_teardown_does_not_wait_for_detached_descendant_output(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _run() -> None:
    handles = _record_started_handles(monkeypatch)
    session, bundle, mcp, dispatcher = _setup(delay_s=30.0)
    descendant_pid = None
    try:
      started, error = await _run_cell(
        dispatcher,
        f"""
        import subprocess
        import sys
        from pathlib import Path
        descendant = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            start_new_session=True,
        )
        Path("detached.pid").write_text(str(descendant.pid))
        hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")
        """,
        background=True,
      )
      assert error is None, error
      entry = _code_job(session, started["task_id"])
      await _until(lambda: bool(mcp.calls))
      assert mcp.calls == [(FRED_SERIES, {"series_id": "CPIAUCSL"})]
      [handle] = handles
      descendant_pid = int((Path(handle.work_dir) / "detached.pid").read_text())

      await asyncio.wait_for(cleanup_code_execution(session), timeout=7)
      assert entry.state == TaskState.KILLED
      assert handle._backend_data["reader_task"].done()
      assert mcp.cancelled is True
      assert bundle.frame_reader.active_cells() == ()
    finally:
      if descendant_pid is not None:
        try:
          os.kill(descendant_pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_repeated_registration_revocation_and_new_cell_grant_narrowing() -> None:
  from agent_gateway.code_execution._frame_codec import encode_frame

  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup()
    reader = bundle.frame_reader
    cell = reader.open_cell("first", abort_event=None, active=True)
    try:
      async def request(channel: Any, operation: str, index: int) -> Any:
        return decode_frame(await channel.handle_frame(encode_frame({
          "type": "tool_call", "call_id": f"repeat-{index}",
          "parent_call_id": channel.outer_call_id, "cell_token": channel.token,
          "operation": operation, "input": {"series_id": "CPIAUCSL"},
        })))

      assert (await request(cell, FRED_SERIES, 1))["ok"]
      assert (await request(cell, FRED_SERIES, 2))["ok"]
      assert len(mcp.calls) == 2
      cell.close()
      assert (await request(cell, FRED_SERIES, 3))["error"]["code"] == "bad_cell_token"
      # The same session reader now binds a narrower turn; no cached prior
      # cell catalog may grant FRED to this registration.
      reader.bind_dispatcher(
        dispatcher,
        tool_definitions=lambda: [d for d in _tool_definitions(bundle) if d["name"] == GET_FILINGS],
        code_callable=_CODE_CALLABLE,
      )
      later = reader.open_cell("later", abort_event=None, active=True)
      try:
        assert (await request(later, FRED_SERIES, 4))["error"]["code"] == "not_granted"
        catalog = decode_frame(await later.handle_frame(encode_frame({
          "type": "catalog", "request": "search", "query": "",
          "call_id": "catalog", "parent_call_id": "later", "cell_token": later.token,
        })))
        assert [row["name"] for row in catalog["result"]] == [GET_FILINGS]
        assert len(mcp.calls) == 2
      finally:
        later.close()
    finally:
      cell.close()
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_complete_settled_edgar_and_input_dependent_vendor_identities(monkeypatch: Any) -> None:
  from types import MappingProxyType
  from agent_gateway.code_execution._frame_codec import encode_frame
  from agent_gateway.tool_dispatch_classification import ToolResultSettlement

  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup()
    executed_seen: list[PreparedToolCall] = []
    dispatch_prepared = dispatcher.dispatch_prepared

    async def capture_dispatch(call_id: str, name: str, prepared: PreparedToolCall, **kwargs: Any) -> ToolResult:
      on_executed = kwargs["on_executed_prepared_call"]

      def capture_executed(call: PreparedToolCall) -> None:
        executed_seen.append(call)
        on_executed(call)

      kwargs["on_executed_prepared_call"] = capture_executed
      if name == FRED_SERIES:
        # Exercise a changed execution snapshot, not just an equal copied
        # PreparedToolCall: settlement must follow what MCP actually ran.
        assert prepared.materialize_input()["series_id"] == "requested-CPI"
        prepared = PreparedToolCall({"series_id": "CPIAUCSL"})
      return await dispatch_prepared(call_id, name, prepared, **kwargs)

    def settled(name: str, _entry: Any, result: Any, error: Any, _semantic: Any, *, prepared_call: PreparedToolCall) -> Any:
      assert error is None
      assert prepared_call is executed_seen[-1]
      if name == GET_FILINGS:
        sources = tuple(MappingProxyType({
          "document_id": f"edgar:document-{i}", "source_kind": "filing",
          "source_url": f"https://www.sec.gov/Archives/document-{i}",
        }) for i in range(70))
      else:
        # The actual producer settlement can depend on exact prepared input;
        # a payload-only source extractor cannot reconstruct this identity.
        assert "source_url" not in result
        assert prepared_call.materialize_input()["series_id"] == "CPIAUCSL"
        sources = (MappingProxyType({
          "document_id": "fred:producer-cpiau-identity",
          "source_kind": "vendor",
          "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
        }),)
      return ToolResultSettlement(outcome="ok", sources=sources)

    monkeypatch.setattr(dispatcher, "dispatch_prepared", capture_dispatch)
    monkeypatch.setattr(dispatcher, "settle_tool_result", settled)
    bundle.frame_reader.bind_dispatcher(
      dispatcher,
      tool_definitions=dispatcher.get_tool_definitions,
      code_callable=_CODE_CALLABLE,
      source_key=lambda handle: (handle["document_id"], handle["source_kind"]),
      extract_sources=lambda operation, _result: [{
        "document_id": "edgar:document-0", "source_kind": "filing",
        "source_url": "https://www.sec.gov/Archives/document-0",
        "accession": "0000320193-24-000123",
      }] if operation == GET_FILINGS else [],
    )
    cell = bundle.frame_reader.open_cell("sources", abort_event=None, active=True)
    try:
      for index, operation in enumerate((GET_FILINGS, FRED_SERIES)):
        reply = decode_frame(await cell.handle_frame(encode_frame({
          "type": "tool_call", "call_id": f"s{index}", "cell_token": cell.token,
          "parent_call_id": "sources", "operation": operation,
          "input": {"series_id": "requested-CPI" if operation == FRED_SERIES else "CPIAUCSL"},
        })))
        assert reply["ok"], reply
      sources = cell.summary()["sources"]
      assert len(sources) == 71
      assert [row["document_id"] for row in sources[:70]] == [f"edgar:document-{i}" for i in range(70)]
      assert all(row["produced_by_tool"] == "get_filings" for row in sources[:70])
      assert sources[0]["accession"] == "0000320193-24-000123"
      assert sources[0]["source_url"] == "https://www.sec.gov/Archives/document-0"
      assert sources[-1] == {
        "document_id": "fred:producer-cpiau-identity", "source_kind": "vendor",
        "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
        "produced_by_tool": "fred_get_series",
      }
      assert mcp.calls[-1] == (FRED_SERIES, {"series_id": "CPIAUCSL"})
    finally:
      cell.close()
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_registered_settlement_sources_match_direct_dispatch_and_are_json_serializable() -> None:
  from agent_workflow_contracts.tool_registration import (
    PolicyKind, RegisteredToolIdentity, ToolApprovalPolicy, ToolIntrinsicSemantics,
    ToolRegistrationCatalog, ToolRegistrationDeclaration, VersionedPolicyRef,
  )
  from agent_gateway.code_execution._frame_codec import encode_frame
  from agent_gateway.code_execution._frame_reader import FrameReader
  from agent_gateway.tool_dispatch_classification import (
    build_dispatch_record_from_sources, classify_tool_outcome,
  )
  from agent_gateway.tool_dispatch_source_identity import read_source_identities
  from agent_gateway.tool_policy_registry import (
    PlanDecision, RedactionResult, SourceIdentityResult,
    ToolPolicyImplementation, ToolPolicyImplementationRegistry, reject_policy_parameters,
  )

  async def _run() -> None:
    filings = {
      "status": "success",
      "filings": [{
        "accession": f"0000320193-24-{index:06d}",
        "form": "10-K",
        "source_url": f"https://www.sec.gov/Archives/document-{index}",
      } for index in range(70)],
    }

    async def get_filings(_input: dict[str, Any], **_kwargs: Any) -> ToolResult:
      return filings, None

    async def fred_get_series(tool_input: dict[str, Any], **_kwargs: Any) -> ToolResult:
      return {"series": tool_input["series"], "observations": [{"value": 321.4}]}, None

    def source_identity(_ref: Any, call: Any) -> SourceIdentityResult:
      if call.identity.logical_name == "get_filings":
        return SourceIdentityResult(read_source_identities({"kind": "parser_filings"}, call.result))
      # A registered producer policy reads the executed input. SourceIdentityResult
      # recursively freezes these key fields and metadata, as real vendor policy does.
      key_fields = dict(call.tool_input)
      suffix = ":".join(f"{key}={value}" for key, value in sorted(key_fields.items()))
      return SourceIdentityResult(({
        "document_id": f"fred:fred_get_series:{suffix}",
        "source_kind": "vendor",
        "provider": "fred",
        "endpoint": "fred_get_series",
        "key_fields": key_fields,
        "producer_metadata": {"dimensions": [{"units": "index"}]},
      },))

    implementations: dict[PolicyKind, Any] = {
      "planning": lambda _ref, _call: PlanDecision("none"),
      "input_preparation": lambda _ref, call: PreparedToolCall(call.raw_input),
      "redaction": lambda _ref, call: RedactionResult(call.prepared_input),
      "outcome": lambda _ref, call: classify_tool_outcome(None, call.result, call.error, call.semantic_error),
      "source_identity": source_identity,
    }
    registry = ToolPolicyImplementationRegistry(tuple(
      ToolPolicyImplementation(kind, "bridge-test", "v1", reject_policy_parameters, implementation)
      for kind, implementation in implementations.items()
    ))
    refs = {kind: VersionedPolicyRef(kind, "bridge-test", "v1") for kind in implementations}
    handlers = {"get_filings": get_filings, "fred_get_series": fred_get_series}
    declarations = tuple(ToolRegistrationDeclaration(
      RegisteredToolIdentity(route_kind="local_handler", logical_server_id=None, logical_name=name),
      ToolIntrinsicSemantics(
        effect="read", idempotent=True, semantic_capability="bridge-test.read/v1",
        approval=ToolApprovalPolicy("never"), audience="ordinary",
        redaction_policy=refs["redaction"], planning_policy=refs["planning"],
        input_preparation_policy=refs["input_preparation"], outcome_policy=refs["outcome"],
        source_identity_policy=refs["source_identity"],
      ),
    ) for name in handlers)
    definitions = [{"name": name, "input_schema": {"type": "object"}} for name in handlers]
    dispatcher = ToolDispatcher(
      capability_bind=TEST_CAPABILITY_BIND, mcp_client=_CaptureMcp(), local_tool_handlers=handlers, role="owner",
      get_tool_definitions=lambda: definitions,
      tool_registration_catalog=ToolRegistrationCatalog(declarations, ()),
      tool_policy_implementations=registry, redaction_context_factory=lambda _declaration: None,
    )
    reader = FrameReader()
    reader.bind_dispatcher(
      dispatcher, tool_definitions=dispatcher.get_tool_definitions,
      code_callable=frozenset(handlers).__contains__,
      source_key=lambda handle: (
        handle["document_id"], handle["source_kind"],
        json.dumps(handle.get("key_fields"), sort_keys=True, separators=(",", ":")),
      ),
    )
    cell = reader.open_cell("registered-sources", abort_event=None, active=True)
    calls = [
      ("get_filings", {}),
      ("fred_get_series", {"series": "CPIAUCSL", "tail": 24}),
      ("fred_get_series", {"series": "CPIAUCSL", "tail": 12}),
    ]
    try:
      expected = []
      for operation, tool_input in calls:
        prepared = dispatcher.prepare_tool_call(operation, tool_input)
        executed = [prepared]
        result, error = await dispatcher.dispatch_prepared(
          "direct", operation, prepared, on_executed_prepared_call=executed.append,
        )
        settlement = dispatcher.settle_tool_result(
          operation, None, result, error, prepared_call=executed[-1],
        )
        assert settlement.outcome == "ok"
        direct = build_dispatch_record_from_sources(
          entry=None, outcome=settlement.outcome, sources=settlement.sources,
        )["sources"]
        expected.extend({**source, "produced_by_tool": operation} for source in direct)
      for index, (operation, tool_input) in enumerate([*calls, *calls]):
        reply = decode_frame(await cell.handle_frame(encode_frame({
          "type": "tool_call", "call_id": str(index), "cell_token": cell.token,
          "parent_call_id": cell.outer_call_id, "operation": operation, "input": tool_input,
        })))
        assert reply["ok"], reply
      summary = cell.summary()
      assert summary["sources"] == expected
      assert len(summary["sources"]) == 72
      assert [row["document_id"] for row in expected[-2:]] == [
        "fred:fred_get_series:series=CPIAUCSL:tail=24",
        "fred:fred_get_series:series=CPIAUCSL:tail=12",
      ]
      assert json.loads(json.dumps(summary)) == summary
    finally:
      await reader.aclose()

  asyncio.run(_run())


def test_reader_teardown_awaits_nested_dispatch_and_revokes_all_cells() -> None:
  from agent_gateway.code_execution._frame_codec import encode_frame

  async def _run() -> None:
    session, bundle, mcp, _dispatcher = _setup(delay_s=30.0)
    reader = bundle.frame_reader
    cell = reader.open_cell("teardown", abort_event=None, active=True)
    task = asyncio.create_task(cell.handle_frame(encode_frame({
      "type": "tool_call", "call_id": "wait", "cell_token": cell.token,
      "parent_call_id": "teardown", "operation": FRED_SERIES,
      "input": {"series_id": "CPIAUCSL"},
    })))
    try:
      for _ in range(200):
        if mcp.calls:
          break
        await asyncio.sleep(0.01)
      assert mcp.calls
      await reader.aclose()
      await reader.aclose()
      assert task.done() and task.cancelled()
      assert mcp.cancelled
      assert reader.active_cells() == ()
      with pytest.raises(RuntimeError, match="disposed"):
        reader.open_cell("after-session-end", abort_event=None, active=True)
    finally:
      task.cancel()
      await asyncio.gather(task, return_exceptions=True)
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_actual_over_limit_dispatch_reply_is_a_correlated_error_and_the_cell_continues(monkeypatch: Any) -> None:
  from agent_gateway.code_execution._frame_codec import MAX_CELL_BYTES, encode_frame

  async def _run() -> None:
    session, bundle, mcp, _dispatcher = _setup()
    call_tool = mcp.call_tool

    async def oversized(name: str, value: Any, **kwargs: Any) -> ToolResult:
      if value.get("series_id") == "large":
        return {"text": "x" * MAX_CELL_BYTES}, None
      return await call_tool(name, value, **kwargs)

    monkeypatch.setattr(mcp, "call_tool", oversized)
    reader = bundle.frame_reader
    cell = reader.open_cell("large-reply", abort_event=None, active=True)
    try:
      reply = decode_frame(await cell.handle_frame(encode_frame({
        "type": "tool_call", "call_id": "large", "cell_token": cell.token,
        "parent_call_id": cell.outer_call_id, "operation": FRED_SERIES,
        "input": {"series_id": "large"},
      })))
      assert reply["ok"] is False
      assert reply["call_id"] == "large"
      assert reply["error"]["code"] == "protocol_error"
      # One over-limit reply is that call's error; it does not poison the cell.
      assert cell.budget.error is None
      assert cell.summary()["nested_calls"][0]["ok"] is False
      assert cell.summary()["nested_calls"][0]["error_code"] == "protocol_error"
      reply = decode_frame(await cell.handle_frame(encode_frame({
        "type": "tool_call", "call_id": "small", "cell_token": cell.token,
        "parent_call_id": cell.outer_call_id, "operation": FRED_SERIES,
        "input": {"series_id": "CPIAUCSL"},
      })))
      assert reply["ok"] is True, reply
      assert reply["result"]["series_id"] == "CPIAUCSL"
      assert [entry["ok"] for entry in cell.summary()["nested_calls"]] == [False, True]
      cell.close()
      later = reader.open_cell("small-reply", abort_event=None, active=True)
      try:
        reply = decode_frame(await later.handle_frame(encode_frame({
          "type": "tool_call", "call_id": "small", "cell_token": later.token,
          "parent_call_id": later.outer_call_id, "operation": FRED_SERIES,
          "input": {"series_id": "CPIAUCSL"},
        })))
        assert reply["ok"] is True, reply
        assert reply["result"]["series_id"] == "CPIAUCSL"
      finally:
        later.close()
    finally:
      cell.close()
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_tool_reply_above_two_mib_arrives_intact_in_the_cell(monkeypatch: Any) -> None:
  import hashlib

  size = 3 * 1024 * 1024 + 5
  text = ("abcdefghijklmnopqrstuvwxyz0123456789" * (size // 36 + 1))[:size]
  digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

  async def _run() -> None:
    session, _bundle, mcp, dispatcher = _setup()
    call_tool = mcp.call_tool

    async def large(name: str, value: Any, **kwargs: Any) -> ToolResult:
      if value.get("series_id") == "big":
        return {"series_id": "big", "text": text}, None
      return await call_tool(name, value, **kwargs)

    monkeypatch.setattr(mcp, "call_tool", large)
    try:
      result, error = await _run_cell(dispatcher, f"""
        import hashlib
        reply = hank.call({FRED_SERIES!r}, series_id="big")
        body = reply["text"]
        print(len(body), hashlib.sha256(body.encode("utf-8")).hexdigest())
        print(body[:36], body[-5:])
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == f"{size} {digest}\n{text[:36]} {text[-5:]}\n"
      assert not result.get("protocol_error")
      nested = result["nested_calls"]
      assert [(entry["ok"], entry["error_code"]) for entry in nested] == [(True, None)]
      assert nested[0]["result_bytes"] > size
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


@pytest.mark.parametrize("direction", ["request", "reply"])
def test_over_limit_bridge_traffic_is_a_caught_error_that_keeps_stdout_and_exit_code(
  monkeypatch: Any, direction: str,
) -> None:
  from agent_gateway.code_execution._frame_codec import MAX_CELL_BYTES

  async def _run() -> None:
    session, _bundle, mcp, dispatcher = _setup()
    call_tool = mcp.call_tool

    async def oversized(name: str, value: Any, **kwargs: Any) -> ToolResult:
      if value.get("series_id") == "over":
        return {"text": "x" * MAX_CELL_BYTES}, None
      return await call_tool(name, value, **kwargs)

    monkeypatch.setattr(mcp, "call_tool", oversized)
    argument = f"'x' * {MAX_CELL_BYTES}" if direction == "request" else "'over'"
    try:
      result, error = await _run_cell(dispatcher, f"""
        saved = 88
        print("before", flush=True)
        try:
            hank.call({FRED_SERIES!r}, series_id={argument})
        except HankBridgeError as exc:
            print("caught", exc.code)
        print("after", hank.call({FRED_SERIES!r}, series_id="CPIAUCSL")["series_id"])
      """)
      assert error is None, error
      assert result["return_code"] == 0, result
      assert result["stdout"] == "before\ncaught protocol_error\nafter CPIAUCSL\n"
      assert not result.get("protocol_error")
      if direction == "reply":
        assert [(entry["ok"], entry["error_code"]) for entry in result["nested_calls"]] == [
          (False, "protocol_error"), (True, None),
        ]
      else:
        assert [entry["ok"] for entry in result["nested_calls"]] == [True]
      following, error = await _run_cell(dispatcher, "print(saved)")
      assert error is None and following["return_code"] == 0
      assert following["stdout"] == "88\n"
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_concurrent_acall_gather_returns_every_result_in_gather_order(monkeypatch: Any) -> None:
  async def _run() -> None:
    session, _bundle, mcp, dispatcher = _setup()
    arrived: list[str] = []
    all_in = asyncio.Event()

    async def gated(name: str, value: Any, **_kwargs: Any) -> ToolResult:
      # Every call must be in flight at once before any is answered: a serialized
      # bridge never reaches the third call and this wait times out.
      arrived.append(str(value.get("ticker") or value.get("series_id")))
      if len(arrived) == 3:
        all_in.set()
      await asyncio.wait_for(all_in.wait(), timeout=30)
      if name == GET_FILINGS:
        return {**_FILINGS_RESULT, "ticker": value["ticker"]}, None
      return {**_FRED_RESULT, "series_id": value["series_id"]}, None

    monkeypatch.setattr(mcp, "call_tool", gated)
    try:
      result, error = await _run_cell(dispatcher, f"""
        import asyncio
        results = await asyncio.gather(
            hank.acall({GET_FILINGS!r}, ticker="AAA"),
            hank.acall({FRED_SERIES!r}, series_id="BBB"),
            hank.acall({GET_FILINGS!r}, ticker="CCC"),
        )
        print([row.get("ticker") or row["series_id"] for row in results])
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == "['AAA', 'BBB', 'CCC']\n"
      assert sorted(arrived) == ["AAA", "BBB", "CCC"]
      nested = result["nested_calls"]
      assert sorted(entry["ordinal"] for entry in nested) == [1, 2, 3]
      assert len({entry["nested_call_id"] for entry in nested}) == 3
      assert all(entry["ok"] is True and entry["error_code"] is None for entry in nested)
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_call_accepts_a_next_call_mapping_positionally_and_unpacked() -> None:
  async def _run() -> None:
    session, _bundle, mcp, dispatcher = _setup()
    try:
      result, error = await _run_cell(dispatcher, f"""
        next_call = {{"tool_name": {FRED_SERIES!r}, "arguments": {{"series_id": "CPIAUCSL"}}}}
        first = hank.call(next_call)
        second = hank.call(**next_call)
        third = hank.call(**{{"tool_name": {GET_FILINGS!r}, "arguments": {{"cik": "0000320193"}}}})
        print(first["series_id"], second["series_id"], third["ticker"])
        check_model_entry = {{
            "tool_name": {FRED_SERIES!r},
            "arguments": {{"series_id": "CPIAUCSL"}},
            "repair_owner": "model_builder",
            "finding_codes": ["missing_input"],
            "missing_controls": [],
            "note": "refetch the series",
            "then": "check_model",
        }}
        fourth = hank.call(check_model_entry)
        print(fourth["series_id"])
        try:
            hank.call({{"arguments": {{"series_id": "CPIAUCSL"}}, "note": "no operation"}})
        except TypeError:
            print("TypeError")
        try:
            hank.call({{"tool_name": {FRED_SERIES!r}, "argments": {{"series_id": "CPIAUCSL"}}, "arguments": {{"series_id": "CPIAUCSL"}}}})
        except TypeError as exc:
            print("TypeError", "argments" in str(exc), "undeclared keys" in str(exc), "repair_owner" in str(exc))
        try:
            hank.call({{"tool_name": {FRED_SERIES!r}, "arguments": {{"series_id": "CPIAUCSL"}}, "missing_controls": [{{"item_id": "x", "then": 42}}]}})
        except TypeError as exc:
            print("TypeError", str(exc))
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert result["stdout"] == (
        "CPIAUCSL CPIAUCSL AAPL\n"
        "CPIAUCSL\n"
        "TypeError\n"
        "TypeError True True True\n"
        "TypeError next_call.missing_controls.0.then: Input should be a valid list\n"
      )
      assert mcp.calls == [
        (FRED_SERIES, {"series_id": "CPIAUCSL"}),
        (FRED_SERIES, {"series_id": "CPIAUCSL"}),
        (GET_FILINGS, {"cik": "0000320193"}),
        (FRED_SERIES, {"series_id": "CPIAUCSL"}),
      ]
      assert [entry["ok"] for entry in result["nested_calls"]] == [True, True, True, True]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(_run())


def test_producer_status_error_reaches_cell_and_ledger_records_failure(monkeypatch: Any) -> None:
  from agent_gateway.code_execution._frame_codec import encode_frame

  async def _run() -> None:
    session, bundle, mcp, _dispatcher = _setup()
    call_tool = mcp.call_tool
    refusal = {"status": "error", "error": "FRED series 'DGS30' is not in the curated typed registry"}

    async def producer(name: str, value: Any, **kwargs: Any) -> ToolResult:
      if value.get("series_id") == "DGS30":
        return refusal, None
      return await call_tool(name, value, **kwargs)

    monkeypatch.setattr(mcp, "call_tool", producer)
    cell = bundle.frame_reader.open_cell("status-error", abort_event=None, active=True)
    try:
      reply = decode_frame(await cell.handle_frame(encode_frame({
        "type": "tool_call", "call_id": "dgs30", "cell_token": cell.token,
        "parent_call_id": cell.outer_call_id, "operation": FRED_SERIES,
        "input": {"series_id": "DGS30"},
      })))
      assert reply["ok"] is True
      assert reply["result"] == refusal
      entry = cell.summary()["nested_calls"][0]
      assert entry["ok"] is False
      assert entry["error_code"] == "tool_status_error"
      assert cell.summary()["sources"] == []
    finally:
      cell.close()
      await cleanup_code_execution(session)

  asyncio.run(_run())



def test_native_calls_share_bridge_authority_ordinals_and_sources() -> None:
  from agent_gateway.code_execution._frame_codec import encode_frame
  from agent_gateway.code_execution._frame_reader import lookup_active_cell
  from agent_gateway.tool_policy_registry import ToolInputPreparationError

  async def run() -> None:
    session, bundle, _mcp, _dispatcher = _setup(extract_sources=_filings_extractor)
    reader = bundle.frame_reader
    cell = reader.open_cell("native-parent", abort_event=None, active=False)
    try:
      assert lookup_active_cell(cell.token) is None
      assert lookup_active_cell("unknown") is None
      cell.activate()
      active = lookup_active_cell(cell.token)
      assert active is not None
      assert active.outer_call_id == "native-parent"
      assert active.principal["user_id"] == session.user_id
      assert {route["name"] for route in active.installed_routes()} == {GET_FILINGS, FRED_SERIES}
      first = active.start_native(GET_FILINGS, {"cik": "AAPL"})
      reply = decode_frame(await cell.handle_frame(encode_frame({
        "type": "tool_call", "call_id": "bridge", "cell_token": cell.token,
        "parent_call_id": cell.outer_call_id, "operation": FRED_SERIES,
        "input": {"series_id": "CPIAUCSL"},
      })))
      assert reply["ok"]
      active.settle_native(first, result=dict(_FILINGS_RESULT), error=None)
      final = active.start_native(FRED_SERIES, {"series_id": "missing"})
      active.settle_native(final, result=None, error={"code": "not_found", "message": "missing"})
      summary = cell.summary()
      assert [row["ordinal"] for row in summary["nested_calls"]] == [1, 2, 3]
      assert first["native"] and first["ok"]
      assert final["error_code"] == "not_found" and not final["ok"]
      assert any(source.get("document_id") == _FILING_HANDLES[0]["document_id"] for source in summary["sources"])
      with pytest.raises(ToolInputPreparationError) as refusal:
        active.start_native(SEARCH_FILINGS, {})
      assert refusal.value.materialize_error()["code"] == "not_granted"
      cell.close()
      assert lookup_active_cell(cell.token) is None
      later = reader.open_cell("disposed", abort_event=None, active=True)
      await reader.aclose()
      assert lookup_active_cell(later.token) is None
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_native_nested_computation_sources_settle_without_filesystem_reader(monkeypatch: Any) -> None:
  async def run() -> None:
    sources = [
      {"document_id": "sandbox:load_statements:sha=aaaaaaaaaaaa", "source_kind": "computation"},
      {"document_id": "sandbox:render_sourced_table:sha=bbbbbbbbbbbb", "source_kind": "computation"},
    ]
    session, _bundle, _mcp, dispatcher = _setup(
      extract_sources=lambda _operation, result: result["sources"],
    )
    from dataclasses import replace
    settle = dispatcher.settle_tool_result

    def settle_computation(operation: Any, entry: Any, result: Any, error: Any, *args: Any, **kwargs: Any) -> Any:
      settlement = settle(operation, entry, result, error, *args, **kwargs)
      if operation != GET_FILINGS:
        return settlement
      return replace(settlement, sources=tuple(result["sources"]))

    monkeypatch.setattr(dispatcher, "settle_tool_result", settle_computation)

    async def execute(_input: Any, *, bridge: Any, **_kwargs: Any) -> Any:
      from agent_gateway.code_execution._frame_reader import lookup_active_cell
      bridge.activate()
      active = lookup_active_cell(bridge.token)
      assert active is not None
      for index, function in enumerate(("load_statements", "render_sourced_table")):
        entry = active.start_native(GET_FILINGS, {"cik": "AAPL"})
        active.settle_native(
          entry, result={**_FILINGS_RESULT, "sources": [sources[index]]}, error=None,
          provenance={
            "function": function, "tool_version": "risk-v1",
            "output_sha256": ("a" if index == 0 else "b") * 64,
          },
        )
      return {"return_code": 0, "stdout": "", "stderr": ""}, None

    monkeypatch.setattr("agent_gateway.code_execution._handlers.code_execute", execute)
    try:
      result, error = await _run_cell(dispatcher, "pass")
      assert error is None
      assert result["sources"] == [
        {**source, "produced_by_tool": "get_filings"} for source in sources
      ]
      assert [row["function"] for row in result["computations"]] == [
        "load_statements", "render_sourced_table",
      ]
      assert [row["nested_call_id"] for row in result["computations"]] == [
        row["nested_call_id"] for row in result["nested_calls"]
      ]
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_native_settlement_retrieves_live_entry_without_repreparing_executed_input(monkeypatch: Any) -> None:
  from agent_gateway.code_execution._frame_reader import lookup_active_cell
  from agent_gateway.tool_policy_registry import ToolInputPreparationError

  async def run() -> None:
    session, bundle, mcp, dispatcher = _setup(extract_sources=_filings_extractor)
    cell = bundle.frame_reader.open_cell("native-http", abort_event=None, active=True)
    captured_inputs: list[dict[str, object]] = []
    settle = dispatcher.settle_tool_result

    def reject_model_input(*_args: Any, **_kwargs: Any) -> Any:
      raise ToolInputPreparationError({"code": "invalid_input", "message": "producer kwargs are not model input"})

    def capture_settlement(*args: Any, **kwargs: Any) -> Any:
      captured_inputs.append(kwargs["prepared_call"].materialize_input())
      return settle(*args, **kwargs)

    monkeypatch.setattr(mcp, "uses_registered_tool_catalog", lambda: True)
    monkeypatch.setattr(mcp, "prepare_registered_tool_input", reject_model_input)
    monkeypatch.setattr(dispatcher, "settle_tool_result", capture_settlement)
    executed_input = {"cik": "AAPL", "user_email": "native-user@example.com"}
    provenance = {"tool_version": "v" * 4096, "output_sha256": "a" * 64}
    try:
      active = lookup_active_cell(cell.token)
      assert active is not None
      with pytest.raises(ToolInputPreparationError):
        dispatcher.prepare_tool_call(GET_FILINGS, executed_input)
      entry = active.start_native(GET_FILINGS, executed_input)
      retrieved = active.native_entry(entry["nested_call_id"])
      assert retrieved is not None and retrieved is entry
      assert active.native_entry("another-cell.1") is None
      active.settle_native(
        retrieved,
        result=dict(_FILINGS_RESULT), error=None, provenance=provenance,
      )
      assert captured_inputs == [executed_input]
      assert active.native_entry(entry["nested_call_id"]) is None
      summary = cell.summary()
      assert summary["nested_calls"][0]["ok"]
      assert summary["nested_calls"][0]["provenance"] == provenance
      assert any(source.get("document_id") == _FILING_HANDLES[0]["document_id"] for source in summary["sources"])
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_skills_as_code_bridge_read_path_exercises_library_and_reaches_approval_door() -> None:
  """Phase 3+: cell exercises new library (dilution) over bridge read path;
  asserts permitted writes like refresh_filings are code-callable;
  proves money-class ops (execute_trade) are code-callable and reach the
  dispatcher's approval door (refused with approval_required under NoApprovalRoute),
  while code_execute itself is structurally absent from the cell grant.
  """
  from api.agent.shared.server_policies import server_tool_is_code_callable
  from api.agent.shared.tool_policy import requires_approval

  def _real_code_callable(op: str) -> bool:
    if op.startswith("mcp__"):
      parts = op.split("__")
      if len(parts) >= 3:
        return server_tool_is_code_callable(parts[1], parts[2])
    return False

  async def _run() -> None:
    session, bundle, mcp, dispatcher = _setup(code_callable=_real_code_callable)
    dispatcher._needs_approval = lambda name, tool_input, qualifier: (
      requires_approval(name.split("__", 2)[2], tool_input, qualifier)
      if name.startswith("mcp__")
      else requires_approval(name, tool_input, qualifier)
    )
    try:
      result, error = await _run_cell(dispatcher, f"""
        from dilution.treasury_stock import OptionTranche, treasury_stock_method
        import json

        # 0. Host tool (code_execute) is structurally absent from the cell grant
        assert "code_execute" not in [row["name"] for row in hank.search("")]
        try:
            _ = hank.code_execute
            raise AssertionError("hank.code_execute should not exist")
        except AttributeError:
            pass
        host_err = None
        try:
            hank.call("code_execute", code="1+1")
        except HankBridgeError as exc:
            host_err = exc.code
        print("host_tool_error:", host_err)

        # 1. Read path via domain-clients bridge
        filings = hank.call({GET_FILINGS!r}, cik="0000320193", limit=2)
        assert len(filings["filings"]) == 2

        # 2. Compute dilution with new library
        tranches = [
            OptionTranche(strike_price=50.0, shares=1000.0),
            OptionTranche(strike_price=80.0, shares=1000.0),
        ]
        dilution_res = treasury_stock_method(tranches, stock_price=100.0)
        print("net_dilutive_shares:", dilution_res.net_dilutive_shares)

        # 3. Permitted write (cache refresh) IS code-callable
        refresh_res = hank.call({REFRESH_FILINGS!r}, cik="0000320193")
        assert refresh_res["refreshed"] is True
        print("permitted_write: success")

        # 4. Money-class op (execute_trade) is code-callable and reaches the approval door
        denied_error = None
        try:
            hank.call({EXECUTE_TRADE!r}, preview_id="pv-123")
        except HankBridgeError as exc:
            denied_error = exc.code
        print("protected_write_denied:", denied_error)
      """)
      assert error is None, error
      assert result["return_code"] == 0, result["stderr"]
      assert "host_tool_error: unknown_operation" in result["stdout"]
      assert "net_dilutive_shares: 700.0" in result["stdout"]
      assert "permitted_write: success" in result["stdout"]
      assert "protected_write_denied: approval_required" in result["stdout"]

      # Verify bridge dispatched the read and permitted write, but NEVER executed the unattended trade
      assert mcp.calls == [
        (GET_FILINGS, {"cik": "0000320193", "limit": 2}),
        (REFRESH_FILINGS, {"cik": "0000320193"}),
      ]
      nested = result["nested_calls"]
      assert len(nested) == 4
      assert nested[0]["operation"] == "code_execute"
      assert nested[0]["ok"] is False
      assert nested[0]["error_code"] == "unknown_operation"
      assert nested[1]["operation"] == GET_FILINGS
      assert nested[1]["ok"] is True
      assert nested[2]["operation"] == REFRESH_FILINGS
      assert nested[2]["ok"] is True
      assert nested[3]["operation"] == EXECUTE_TRADE
      assert nested[3]["ok"] is False
      assert nested[3]["error_code"] == "approval_required"
      await cleanup_code_execution(session)
    finally:
      pass

  asyncio.run(_run())


# --- call-scoped cell token: v2 native minted by the installed authority -------

_V2_CELL_TOKEN = re.compile(r"v2_[1-9][0-9]*_[0-9a-f]{64}_[0-9a-f]{64}")
_AUTHORITY_SECRET = b"cell-token-authority-test-key-at-least-32-bytes"


def _install_authority(monkeypatch: pytest.MonkeyPatch) -> GatewayClaimSigningAuthority:
  monkeypatch.setattr(claim_signing_authority, "_configured_authority", None)
  monkeypatch.setenv(GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV, "stale")
  monkeypatch.setenv(LEGACY_CLAIM_HMAC_KEY_ENV, "scrubbed-on-install")
  authority = GatewayClaimSigningAuthority(_AUTHORITY_SECRET)
  install_gateway_claim_signing_authority(authority)
  return authority


def _place_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, storage_id: str) -> None:
  bundle = mint_user_runtime_credentials(
    org_claim_key="cell-token-org-authority-at-least-32-bytes",
    identity=UserRuntimeIdentity(
      key="alice-private-api-key", channel="excel", slug="alice",
      email="alice@example.test", risk_user_id=17, role="analyst", capabilities=(),
    ),
    storage_id=storage_id,
    desk=DeskReplica(
      url="https://desk.example.test", aws_access_key_id=None, aws_secret_access_key=None,
      aws_session_token=None, expires_at=None,
    ),
  )
  path = tmp_path / "bundle.json"
  write_user_runtime_credentials(bundle, path)
  monkeypatch.setenv(USER_RUNTIME_CREDENTIALS_FILE_ENV, str(path))


def test_agentcore_shaped_boot_mints_v2_token_for_the_placed_bundle_storage(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  from agent_gateway.code_execution._frame_reader import lookup_active_cell

  authority = _install_authority(monkeypatch)
  _place_bundle(monkeypatch, tmp_path, "4242")

  async def run() -> None:
    session, bundle, _mcp, _dispatcher = _setup()
    cell = bundle.frame_reader.open_cell("placed", abort_event=None, active=True)
    try:
      assert _V2_CELL_TOKEN.fullmatch(cell.token)
      assert claim_signing_authority.native_cell_token_storage(cell.token) == "4242"
      assert authority.verify_native_cell_token(cell.token) == "4242"
      active = lookup_active_cell(cell.token)
      assert active is not None and active.outer_call_id == "placed"
      cell.close()
      assert lookup_active_cell(cell.token) is None
    finally:
      await cleanup_code_execution(session)

  asyncio.run(run())


def test_zero_risk_session_inside_a_placed_bundle_mints_for_the_bundle_storage(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  from agent_gateway.code_execution._frame_reader import FrameReader

  authority = _install_authority(monkeypatch)
  _place_bundle(monkeypatch, tmp_path, "4242")

  async def run() -> None:
    reader = FrameReader(principal={"user_id": "alice-slug", "risk_user_id": 0})
    cell = reader.open_cell("zero-risk", abort_event=None, active=True)
    try:
      assert authority.verify_native_cell_token(cell.token) == "4242"
    finally:
      await reader.aclose()

  asyncio.run(run())


def test_cell_token_without_a_bundle_mints_for_the_positive_risk_user_id(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway.code_execution._frame_reader import FrameReader

  authority = _install_authority(monkeypatch)
  monkeypatch.delenv(USER_RUNTIME_CREDENTIALS_FILE_ENV, raising=False)

  async def run() -> None:
    reader = FrameReader(principal={"user_id": "alice-slug", "risk_user_id": 17})
    cell = reader.open_cell("risk-id", abort_event=None, active=True)
    try:
      assert authority.verify_native_cell_token(cell.token) == "17"
    finally:
      await reader.aclose()

  asyncio.run(run())


def test_cell_token_without_a_bundle_or_positive_risk_id_is_the_plain_nonce(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway.code_execution._frame_reader import FrameReader, lookup_active_cell

  _install_authority(monkeypatch)
  monkeypatch.delenv(USER_RUNTIME_CREDENTIALS_FILE_ENV, raising=False)

  async def run() -> None:
    reader = FrameReader(principal={"user_id": "alice-slug", "risk_user_id": 0})
    cell = reader.open_cell("direct", abort_event=None, active=True)
    try:
      assert not _V2_CELL_TOKEN.fullmatch(cell.token)
      assert claim_signing_authority.native_cell_token_storage(cell.token) is None
      assert lookup_active_cell(cell.token) is not None
    finally:
      await reader.aclose()

  asyncio.run(run())


def test_v2_cell_token_verifies_only_for_its_minted_storage_and_authority(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  from agent_gateway.code_execution._frame_reader import FrameReader

  authority = _install_authority(monkeypatch)
  _place_bundle(monkeypatch, tmp_path, "4242")

  async def run() -> None:
    reader = FrameReader(principal={"risk_user_id": 17})
    cell = reader.open_cell("wrong-storage", abort_event=None, active=True)
    try:
      version, storage, nonce, mac = cell.token.split("_")
      assert (version, storage) == ("v2", "4242")
      other_storage = "_".join((version, "17", nonce, mac))
      assert claim_signing_authority.native_cell_token_storage(other_storage) == "17"
      assert authority.verify_native_cell_token(other_storage) is None
      foreign = GatewayClaimSigningAuthority(b"another-authority-key-at-least-32-bytes!!")
      assert foreign.verify_native_cell_token(cell.token) is None
    finally:
      await reader.aclose()

  asyncio.run(run())


def test_cell_token_without_an_authority_is_the_plain_nonce(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  from agent_gateway.code_execution._frame_reader import FrameReader, lookup_active_cell

  monkeypatch.setattr(claim_signing_authority, "_configured_authority", None)
  _place_bundle(monkeypatch, tmp_path, "4242")

  async def run() -> None:
    reader = FrameReader(principal={"risk_user_id": 17})
    cell = reader.open_cell("plain", abort_event=None, active=True)
    try:
      assert not _V2_CELL_TOKEN.fullmatch(cell.token)
      assert claim_signing_authority.native_cell_token_storage(cell.token) is None
      assert lookup_active_cell(cell.token) is not None
    finally:
      await reader.aclose()

  asyncio.run(run())
