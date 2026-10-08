import asyncio
import hashlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from agent_workflow_contracts import (
  AgentOperationRef,
  AttemptRef,
  ContentHandle,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  OwnerBinding,
  ResultRequirement,
  SELECTED_CONTENT_UTF8_CONTRACT,
  TaskResultProvenance,
  sha256_digest,
)



ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from gateway_test_support.host_policy import owner_session_host_policy  # noqa: E402

from agent_gateway import (  # noqa: E402
  AgentRunner,
  AgentSessionLog,
  McpClientManager,
  EventLog,
  GatewaySession,
  ModelInfo,
  ModelProvider,
  SessionContextBuilder,
  TaskRegistry,
  TaskState,
  ToolDispatcher,
)
from agent_gateway.auth import ProviderCredentialFailure  # noqa: E402
from agent_gateway.providers import StreamEvent  # noqa: E402
from agent_gateway.providers.base import truncate_to_last_compaction  # noqa: E402
import agent_gateway.server_compaction as server_compaction  # noqa: E402
from agent_gateway.runner_state import assistant_turn_message  # noqa: E402
from agent_gateway.selected_content import SelectedContentBinding  # noqa: E402
from agent_gateway.capability_execution import BoundCapabilityExecution  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from agent_gateway.runner_run_loop import (  # noqa: E402
  _background_success_snapshot,
)
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)


def _run(coro):
  return asyncio.run(coro)


def _child_report(summary: str = "done") -> dict[str, object]:
  return {
    "kind": "report",
    "version": "1",
    "report": {
      "summary": summary,
      "findings": [],
      "artifacts": [],
      "caveats": [],
    },
    "usage": {},
    "tools_used": [],
    "fms_results": None,
    "artifact_events": None,
    "warning": None,
  }


def _subagent_result_identity(
  *,
  delegation_id: str,
  physical_task_id: str,
) -> tuple[
  OrdinaryDelegationTaskRef,
  AttemptRef,
  ResultRequirement,
  TaskResultProvenance,
]:
  operation = AgentOperationRef(
    namespace="agent-operation",
    name="test-child",
    version="1.0",
    digest=sha256_digest({"operation": "test-child", "version": "1.0"}),
  )
  logical_task = OrdinaryDelegationTaskRef(
    delegation_id=delegation_id,
    operation=operation,
  )
  attempt = AttemptRef(
    attempt_number=1,
    attempt_id=f"attempt:{physical_task_id}:1",
    physical_task_id=physical_task_id,
  )
  return (
    logical_task,
    attempt,
    ResultRequirement(
      mode="narrative",
      terminal_narrative="required",
      outcome=OutcomeRequirement(required=False, source="none"),
    ),
    TaskResultProvenance(
      admitted_task_digest=sha256_digest({
        "logical_task": logical_task.model_dump(mode="json"),
        "attempt": attempt.model_dump(mode="json"),
      }),
      model_bind_digest=sha256_digest({"model_bind": "test-child"}),
      capability_binding_digest=sha256_digest({
        "capability": "node.implement"
      }),
      tool_grant_digest=sha256_digest({"tools": ["lookup"]}),
    ),
  )


def _runner_execution(
  provider: ModelProvider,
  *,
  model: str = "claude-sonnet-5",
  capability_id: str = "session.driver",
) -> BoundCapabilityExecution:
  return stub_runner_capability_execution(
    provider=provider,
    model=model,
    effort="none",
    capability_id=capability_id,
  )


def _child_execution(
  provider: ModelProvider,
  *,
  model: str = "claude-sonnet-5",
) -> BoundCapabilityExecution:
  return _runner_execution(
    provider,
    model=model,
    capability_id="node.implement",
  )


class _NullMcpClient(McpClientManager):
  """No-server MCP manager used by dispatcher-focused tests."""



class _ScriptedProvider(ModelProvider):
  name = "stub"

  def __init__(self, turns: list[list[StreamEvent]], after_turn: Any | None = None) -> None:
    self._turns = [list(turn) for turn in turns]
    self._turn_index = 0
    self._after_turn = after_turn

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name)

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    if self._turn_index >= len(self._turns):
      raise AssertionError("unexpected extra turn")
    current_turn = self._turns[self._turn_index]
    self._turn_index += 1
    for event in current_turn:
      yield event
    if self._after_turn is not None:
      self._after_turn(self._turn_index)


class _RetryableFailingProvider(_ScriptedProvider):
  def __init__(self, message: str = "Anthropic API error (status=200)") -> None:
    super().__init__([])
    self.message = message
    self.calls = 0

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    self.calls += 1
    raise RuntimeError(self.message)
    yield  # pragma: no cover

  def is_retryable_error(self, exc: Exception) -> bool:
    _ = exc
    return True


def _make_dispatcher(
  *,
  event_log: EventLog | None = None,
  local_tool_handlers: dict[str, Any] | None = None,
) -> ToolDispatcher:
  handlers = dict(local_tool_handlers or {})
  tool_definitions = [
    {
      "name": name,
      "description": f"Synthetic {name} test tool",
      "input_schema": {
        "type": "object",
        "additionalProperties": True,
      },
    }
    for name in handlers
  ]
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers=handlers,
    event_log=event_log or EventLog(),
    session_id="sess-parent",
    # Owner authority: since S1 fail-closed roles a role-less dispatcher is
    # invite and denies the synthetic local tools these logging tests execute.
    role="owner",
    get_tool_definitions=lambda: list(tool_definitions),
  )


def _make_runner(**kwargs: Any) -> AgentRunner:
  dispatcher = kwargs.get("dispatcher")
  tool_definition_getter = getattr(
    dispatcher,
    "get_tool_definitions",
    None,
  )
  if callable(tool_definition_getter):
    kwargs.setdefault("get_tool_definitions", tool_definition_getter)
  return AgentRunner(**kwargs)


def _tool_turn(
  *,
  tool_id: str = "tool-1",
  tool_name: str = "lookup",
  tool_input: dict[str, Any] | None = None,
) -> list[StreamEvent]:
  payload = tool_input or {"query": "AAPL"}
  return [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name=tool_name,
      tool_input=payload,
      raw_block={
        "type": "tool_use",
        "id": tool_id,
        "name": tool_name,
        "input": payload,
      },
    ),
    StreamEvent(type="usage_update", output_tokens=3),
    StreamEvent(type="message_end", stop_reason="tool_use"),
  ]


def _text_turn(text: str) -> list[StreamEvent]:
  return [
    StreamEvent(type="message_start", input_tokens=12),
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(type="usage_update", output_tokens=5),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]


def _mixed_text_tool_turn(
  *,
  text: str = "scratch text before tool",
  tool_id: str = "tool-1",
  tool_name: str = "lookup",
  tool_input: dict[str, Any] | None = None,
) -> list[StreamEvent]:
  payload = tool_input or {"query": "AAPL"}
  return [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(
      type="tool_use_end",
      tool_id=tool_id,
      tool_name=tool_name,
      tool_input=payload,
      raw_block={
        "type": "tool_use",
        "id": tool_id,
        "name": tool_name,
        "input": payload,
      },
    ),
    StreamEvent(type="usage_update", output_tokens=3),
    StreamEvent(type="message_end", stop_reason="tool_use"),
  ]


async def _lookup_tool(tool_input: dict[str, Any], **kwargs: Any):
  _ = kwargs
  return {"echo": tool_input}, None


async def _warning_tool(tool_input: dict[str, Any], **kwargs: Any):
  _ = kwargs
  return {"ok": True, "low_match_warning": tool_input.get("warning", "only 20% matched")}, None


async def _interceptor_warning_tool(
  tool_input: dict[str, Any],
  **kwargs: Any,
):
  _ = tool_input, kwargs
  return {
    "ok": True,
    "_interceptor_warnings": ["policy warning"],
  }, None


async def _large_result_tool(tool_input: dict[str, Any], **kwargs: Any):
  _ = kwargs
  size = int(tool_input.get("size", 10_000))
  return {"status": "success", "ticker": "BIG", "payload": "x" * size}, None


async def _semantic_error_tool(tool_input: dict[str, Any], **kwargs: Any):
  _ = kwargs
  return {
    "status": "error",
    "error": {"code": "not_found", "message": f"{tool_input.get('query', 'ticker')} not found"},
  }, None


async def _empty_status_error_tool(tool_input: dict[str, Any], **kwargs: Any):
  _ = tool_input, kwargs
  return {"status": "error", "error": ""}, None


async def _decision_log_validation_error_tool(tool_input: dict[str, Any], **kwargs: Any):
  _ = tool_input, kwargs
  return {
    "status": "error",
    "error": "decisions-log entry is invalid",
    "error_code": "invalid_decisions_log_entry",
    "validation_error": True,
    "validation_errors": [
      {"type": "missing", "loc": ["date"], "msg": "Field required"},
    ],
    "required_fields": ["date", "skill", "decision", "rationale"],
  }, None


def test_runner_emits_durable_envelope_in_order(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "runner.jsonl")
  provider = _ScriptedProvider([
    _tool_turn(),
    _text_turn("done"),
  ])
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"lookup": _lookup_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    workspace_dir=str(tmp_path),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run lookup"}]))

  entries, _ = _run(log.query(order="asc"))
  assert entries[1].event["content"] == "Run lookup"
  assert entries[2].event["stop_reason"] == "tool_use"
  assert entries[2].event["usage"]["input_tokens"] == 10
  assert entries[3].event["parent_assistant_message_seq"] == entries[2].seq
  assert "final_tool_result_blocks" in entries[4].event
  assert entries[-1].event["type"] == "detach"
  assert entries[-1].event["reason"] == "completed"


def test_gateway_session_attach_records_the_log_owner(tmp_path: Path) -> None:
  # The first attach is the ownership record init-time resume checks.
  log = AgentSessionLog(path=tmp_path / "sessions" / "owned.jsonl")
  session = GatewaySession(
    session_id="sess-owned",
    api_key_hash="keyhash16hexchars",
    created_at=1,
    expires_at=2,
    user_id="alice",
    channel="cli",
    auth_config={"api_key": "k"},
  )
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log),
    session_id="sess-owned",
    capability_execution=_runner_execution(_ScriptedProvider([_text_turn("ok")])),
    agent_session_log=log,
    gateway_session=session,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  entries, _ = _run(log.query(order="asc"))
  attach = entries[0].event
  assert attach["type"] == "attach"
  assert (attach["user_id"], attach["channel"], attach["api_key_hash"]) == (
    "alice",
    "cli",
    "keyhash16hexchars",
  )


def test_detach_keeps_other_session_responsive(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    logs = [
      AgentSessionLog(path=tmp_path / f"session-{index}.jsonl")
      for index in range(2)
    ]
    events = [EventLog(), EventLog()]
    runners = [
      _make_runner(
        event_log=events[index],
        dispatcher=_make_dispatcher(event_log=events[index]),
        session_id=f"session-{index}",
        capability_execution=_runner_execution(
          _ScriptedProvider([_text_turn(f"answer-{index}")])
        ),
        agent_session_log=logs[index],
        user_id=f"user-{index}",
        billing_mode="byok",
        rate_table_version="unknown",
      )
      for index in range(2)
    ]
    loop = asyncio.get_running_loop()
    scan_started = asyncio.Event()
    release_scan = threading.Event()
    scan_started_at: list[float] = []
    read_entries = logs[0]._iter_current_entries_strict_sync

    def _slow_terminal_scan():
      if runners[0]._terminal_closure_event is not None and not scan_started_at:
        scan_started_at.append(time.monotonic())
        loop.call_soon_threadsafe(scan_started.set)
        # A bounded disk-read delay reproduces a large retained-journal scan
        # without making every test write and parse a 98 MB journal.
        release_scan.wait(timeout=2.0)
      yield from read_entries()

    monkeypatch.setattr(
      logs[0], "_iter_current_entries_strict_sync", _slow_terminal_scan
    )
    first = asyncio.create_task(
      runners[0].run(messages=[{"role": "user", "content": "First"}])
    )
    try:
      await asyncio.wait_for(scan_started.wait(), timeout=5.0)
      loop_stall = time.monotonic() - scan_started_at[0]
      await asyncio.wait_for(
        runners[1].run(messages=[{"role": "user", "content": "Second"}]),
        timeout=5.0,
      )
      assert any(
        entry.event.get("text") == "answer-1"
        for entry in events[1].entries
      )
      assert loop_stall < 0.5, f"Other session stalled for {loop_stall:.3f}s"
    finally:
      release_scan.set()
      await first
    for durable_log in logs:
      entries, _ = await durable_log.query(order="asc")
      assert entries[-1].event["type"] == "detach"

  _run(_case())


def test_cancellation_during_recovery_settles_without_starting_provider(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    durable_log = AgentSessionLog(path=tmp_path / "cancelled-recovery.jsonl")
    await durable_log.append({
      "type": "attach",
      "gateway_session_id": "cancelled-recovery",
      "runner_id": "old-writer",
      "started_at": 100.0,
      "client_kind": "cron",
      "role": "writer",
    })
    provider = _ScriptedProvider([_text_turn("Must not execute")])
    event_log = EventLog()
    runner = _make_runner(
      event_log=event_log,
      dispatcher=_make_dispatcher(event_log=event_log),
      session_id="cancelled-recovery",
      capability_execution=_runner_execution(provider),
      agent_session_log=durable_log,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    loop = asyncio.get_running_loop()
    scan_started = asyncio.Event()
    release_scan = threading.Event()
    query = durable_log.query_current_strict_sync

    def _paused_recovery_scan(**kwargs: Any):
      if not release_scan.is_set():
        loop.call_soon_threadsafe(scan_started.set)
        release_scan.wait(timeout=2.0)
      return query(**kwargs)

    monkeypatch.setattr(
      durable_log, "query_current_strict_sync", _paused_recovery_scan
    )
    task = asyncio.create_task(
      runner.run(messages=[{"role": "user", "content": "Do expensive work"}])
    )
    try:
      await asyncio.wait_for(scan_started.wait(), timeout=5.0)
      # Match the public stream's disconnect-then-cancel ordering.
      await runner.on_disconnect()
      for _ in range(2):
        task.cancel()
        await asyncio.sleep(0)
      assert not task.done()
      assert runner._write_lease_file is not None
    finally:
      release_scan.set()
      try:
        await task
      except asyncio.CancelledError:
        pass
    entries, _ = await durable_log.query(order="asc")
    detach_reasons = [
      entry.event["reason"] for entry in entries
      if entry.event["type"] == "detach"
    ]
    assert (provider._turn_index, task.cancelled(), detach_reasons) == (
      0, True, []
    )
    assert [entry.event["type"] for entry in entries] == [
      "attach", "interrupted"
    ]
    assert entries[-1].event["runner_id"] == "old-writer"
    assert entries[-1].event["reason"] == "recovered_on_attach"
    assert runner._write_lease_file is None

  _run(_case())


def test_detach_commits_after_cancellation_during_terminal_scan(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    durable_log = AgentSessionLog(path=tmp_path / "cancelled-detach.jsonl")
    event_log = EventLog()
    runner = _make_runner(
      event_log=event_log,
      dispatcher=_make_dispatcher(event_log=event_log),
      session_id="cancelled-detach",
      capability_execution=_runner_execution(
        _ScriptedProvider([_text_turn("Completed answer")])
      ),
      agent_session_log=durable_log,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    loop = asyncio.get_running_loop()
    scan_started = asyncio.Event()
    release_scan = threading.Event()
    read_entries = durable_log._iter_current_entries_strict_sync

    def _paused_terminal_scan():
      if runner._terminal_closure_event is not None and not release_scan.is_set():
        loop.call_soon_threadsafe(scan_started.set)
        release_scan.wait(timeout=2.0)
      yield from read_entries()

    monkeypatch.setattr(
      durable_log, "_iter_current_entries_strict_sync", _paused_terminal_scan
    )
    existing_tasks = asyncio.all_tasks()
    task = asyncio.create_task(
      runner.run(messages=[{"role": "user", "content": "Finish"}])
    )
    try:
      await asyncio.wait_for(scan_started.wait(), timeout=5.0)
      for _ in range(2):
        # Portal shutdown cancels every task, including separately shielded
        # child tasks, not just the runner waiting for terminal settlement.
        for pending in asyncio.all_tasks() - existing_tasks:
          pending.cancel()
        await asyncio.sleep(0)
      assert runner._write_lease_file is not None
      assert not task.done()
    finally:
      release_scan.set()
      try:
        await task
      except asyncio.CancelledError:
        pass
    entries, _ = await durable_log.query(order="asc")
    assert entries[-1].event["type"] == "detach"
    assert entries[-1].event["reason"] == "completed"
    assert task.cancelled()
    assert runner._write_lease_file is None

  _run(_case())


def test_post_sse_cancellation_keeps_durable_detach(tmp_path: Path) -> None:
  async def case() -> None:
    durable_log = AgentSessionLog(tmp_path / "post-sse-cancellation.jsonl")
    event_log = EventLog()
    runner = _make_runner(
      event_log=event_log, dispatcher=_make_dispatcher(event_log=event_log),
      session_id="post-sse-cancellation",
      capability_execution=_runner_execution(_ScriptedProvider([_text_turn("done")])),
      agent_session_log=durable_log, user_id="alice",
      billing_mode="byok", rate_table_version="unknown",
    )
    task = asyncio.create_task(runner.run(messages=[{"role": "user", "content": "finish"}]))

    def cancel_after_terminal(event: dict[str, Any], _session_id: str) -> None:
      if event["type"] == "stream_complete":
        task.cancel("SSE disconnect")
        task.cancel("server shutdown")

    event_log._on_event = cancel_after_terminal
    with pytest.raises(asyncio.CancelledError):
      await task
    entries, _ = await durable_log.query(order="asc")
    assert entries[-1].event["type"] == "detach"
    assert entries[-1].event["reason"] == "completed"
    assert event_log.has_terminal
    assert runner._write_lease_file is None

  _run(case())


def test_named_skill_repeated_cancellation_drains_interruption_and_detach(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway.skill_lifecycle import (
    TopLevelSkillAdmission,
    TopLevelSkillCompletionPlan,
    TopLevelSkillLifecycleMetadata,
    TopLevelSkillResultPolicy,
  )
  from agent_gateway.runner_session_events import build_skill_result_failure_event
  from agent_gateway.skill_completion_wal import TopLevelSkillCompletionEffectPlan

  async def case() -> None:
    durable_log = AgentSessionLog(tmp_path / "named-cancellation.jsonl")
    lifecycle = TopLevelSkillLifecycleMetadata(
      skill_run_id="cancelled-skill", skill="fundamental-research",
      scope="ticker", ticker="PCTY", portfolio_id=None,
    )
    event_log = EventLog()
    dispatcher = _make_dispatcher(event_log=event_log)
    runner = AgentRunner(
      event_log=event_log, dispatcher=dispatcher, session_id="named-cancellation",
      capability_execution=_runner_execution(_ScriptedProvider([_text_turn("unused")])),
      agent_session_log=durable_log,
      get_tool_definitions=dispatcher.get_tool_definitions,
      top_level_skill_admission=TopLevelSkillAdmission.acquire(durable_log.path),
      top_level_skill_lifecycle=lifecycle,
      top_level_skill_result_policy=TopLevelSkillResultPolicy(
        prepare_provider=lambda prompt: prompt,
        prepare_completion=lambda _log, terminal: TopLevelSkillCompletionPlan(
          result_event=build_skill_result_failure_event(lifecycle, error="cancelled"),
          terminal_event=terminal,
          effect=TopLevelSkillCompletionEffectPlan.noop(),
        ),
      ),
      workspace_dir=tmp_path, user_id="alice", billing_mode="byok",
      rate_table_version="unknown",
    )
    streaming = asyncio.Event()

    async def stream_turn(**_kwargs: Any):
      streaming.set()
      await asyncio.Future()

    runner._stream_turn = stream_turn
    loop = asyncio.get_running_loop()
    scan_started = asyncio.Event()
    release_scan = threading.Event()
    emit_interrupted = runner._emit_interrupted_event_sync

    def paused_interruption_scan(*args: Any, **kwargs: Any):
      loop.call_soon_threadsafe(scan_started.set)
      release_scan.wait(timeout=5.0)
      return emit_interrupted(*args, **kwargs)

    monkeypatch.setattr(
      runner, "_emit_interrupted_event_sync", paused_interruption_scan,
    )
    task = asyncio.create_task(runner.run(messages=[{"role": "user", "content": "run"}]))
    try:
      await asyncio.wait_for(streaming.wait(), timeout=5.0)
      task.cancel("provider cancellation")
      await asyncio.wait_for(scan_started.wait(), timeout=5.0)
      for _ in range(3):
        task.cancel("repeated cancellation")
        await asyncio.sleep(0)
      assert not task.done()
      assert runner._write_lease_file is not None
    finally:
      release_scan.set()
      with pytest.raises(asyncio.CancelledError):
        await task
    assert await runner.wait_for_top_level_skill_settlement() is True
    entries, _ = await durable_log.query(order="asc")
    assert any(entry.event["type"] == "interrupted" for entry in entries)
    assert entries[-1].event["type"] == "detach"
    assert entries[-1].event["reason"] == "cancelled"
    assert runner._write_lease_file is None

  _run(case())


def test_runner_commits_selected_content_on_user_fact_before_provider_setup(
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "selected.jsonl")
  payload = b"exact selected bytes"
  digest = hashlib.sha256(payload).hexdigest()
  binding = SelectedContentBinding(
    input_name="selection_0123456789abcdef01234567",
    display_name="facts.txt",
    owner=OwnerBinding(tenant_id="tenant-1", session_id="sess-parent"),
    content=ContentHandle(
      content_id=f"sha256:{digest}",
      content_sha256=digest,
      content_chars=len(payload.decode("utf-8")),
      content_bytes=len(payload),
      contract=SELECTED_CONTENT_UTF8_CONTRACT,
      media_type="text/plain",
      encoding="utf-8",
      retention="durable",
    ),
  )

  class _CommitCheckingProvider(_ScriptedProvider):
    def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
      entries, _cursor = log.query_sync(event_types={"user_message"}, order="asc")
      assert entries[-1].event["selected_content"] == [binding.model_dump(mode="json")]
      return super().create_client(config, timeout=timeout)

  provider = _CommitCheckingProvider([_text_turn("done")])
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    workspace_dir=str(tmp_path),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  runner.bind_selected_content((binding,))

  _run(runner.run(messages=[{"role": "user", "content": "Use it"}]))


@pytest.mark.parametrize(
  "tool_only_instruction",
  [
    (
      "Tool-call messages are tool-only. Every assistant message that contains any tool call "
      "must contain zero visible text. Run lookup."
    ),
    "Every assistant message before the FMS call must be tool-only (`text=0`). Run lookup.",
  ],
)
def test_runner_suppresses_text_from_tool_only_turns(
  tmp_path: Path,
  tool_only_instruction: str,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "tool-only.jsonl")
  provider = _ScriptedProvider([
    _mixed_text_tool_turn(text="scratch before tool"),
    _text_turn("done"),
  ])
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"lookup": _lookup_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    workspace_dir=str(tmp_path),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(
    runner.run(
      messages=[
        {
          "role": "user",
          "content": tool_only_instruction,
        }
      ],
    )
  )

  entries, _ = _run(log.query(order="asc"))
  tool_turn = next(entry.event for entry in entries if entry.event.get("stop_reason") == "tool_use")
  assert tool_turn["content_blocks"] == [
    {
      "type": "tool_use",
      "id": "tool-1",
      "name": "lookup",
      "input": {"query": "AAPL"},
    }
  ]
  assert "scratch before tool" not in json.dumps([entry.event for entry in entries])
  text_events = [entry for entry in event_log.entries if entry.event.get("type") == "text_delta"]
  assert [event.event["text"] for event in text_events] == ["done"]


def test_semantic_tool_error_is_visible_in_trace_and_timing(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "semantic-error.jsonl")
  event_log = EventLog()
  timing_calls: list[tuple[Any, ...]] = []
  provider = _ScriptedProvider([
    _tool_turn(tool_input={"query": "X"}),
    _text_turn("handled"),
  ])
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"lookup": _semantic_error_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    on_tool_timing=lambda *args: timing_calls.append(args),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run lookup"}]))

  complete_events = [
    entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"
  ]
  assert len(complete_events) == 1
  complete = complete_events[0]
  assert complete["error"] is None
  assert complete["is_error"] is True
  assert complete["semantic_error"] == {
    "code": "tool_status_error",
    "message": "X not found",
    "source": "status",
    "status": "error",
    "sub_code": "not_found",
  }
  assert complete["final_tool_result_blocks"][0]["is_error"] is True
  assert timing_calls
  assert timing_calls[0][4] is True

  durable_entries, _ = _run(log.query(event_types={"tool_call_complete"}, order="asc"))
  assert durable_entries[0].event["is_error"] is True
  assert durable_entries[0].event["semantic_error"]["message"] == "X not found"


def test_semantic_tool_error_without_detail_warns_model_context(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "semantic-empty-error.jsonl")
  event_log = EventLog()
  provider = _ScriptedProvider([
    _tool_turn(tool_name="get_skill_artifact", tool_input={"ticker": "ADI"}),
    _text_turn("handled"),
  ])
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"get_skill_artifact": _empty_status_error_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run lookup"}]))

  complete_events = [
    entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"
  ]
  assert len(complete_events) == 1
  complete = complete_events[0]
  assert complete["result"] == {"status": "error", "error": ""}
  assert complete["semantic_error"] == {
    "code": "tool_status_error",
    "message": "Tool result reported status=error without error detail",
    "source": "status",
    "status": "error",
    "sub_code": "empty_error_detail",
  }
  final_content = json.loads(complete["final_tool_result_blocks"][0]["content"])
  assert final_content == {
    "status": "error",
    "error": "",
    "_runner_warning": (
      "Tool get_skill_artifact returned status=error without error detail; "
      "do not retry unchanged input unless required context changed or there is new evidence the failure was transient."
    ),
  }
  assert complete["final_tool_result_blocks"][0]["is_error"] is True


def test_native_runner_semantic_error_includes_validation_details(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "semantic-validation-error.jsonl")
  event_log = EventLog()
  provider = _ScriptedProvider([
    _tool_turn(tool_name="thesis_append_decisions_log", tool_input={"research_file_id": 1, "entry": {}}),
    _text_turn("handled"),
  ])
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(
      event_log=event_log,
      local_tool_handlers={"thesis_append_decisions_log": _decision_log_validation_error_tool},
    ),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run invalid decision log"}]))

  complete_events = [
    entry.event for entry in event_log.entries if entry.event.get("type") == "tool_call_complete"
  ]
  assert len(complete_events) == 1
  complete = complete_events[0]
  assert complete["error"] is None
  assert complete["is_error"] is True
  assert complete["semantic_error"]["sub_code"] == "invalid_decisions_log_entry"
  assert "decisions-log entry is invalid" in complete["semantic_error"]["message"]
  assert "date: Field required" in complete["semantic_error"]["message"]
  assert "required_fields: date, skill, decision, rationale" in complete["semantic_error"]["message"]
  final_content = json.loads(complete["final_tool_result_blocks"][0]["content"])
  assert final_content["validation_errors"][0]["loc"] == ["date"]


def test_operator_pause_before_turn_emits_clean_interruption(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "operator-pause-before-turn.jsonl")
  event_log = EventLog()
  pause_event = asyncio.Event()
  pause_event.set()
  provider = _ScriptedProvider([])
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    operator_pause_event=pause_event,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Pause cleanly"}]))

  entries, _ = _run(log.query(order="asc"))
  interruption = next(entry.event for entry in entries if entry.event["type"] == "interrupted")
  assert interruption["reason"] == "operator_pause"
  assert interruption["safe_boundary"] == "before_turn"
  assert entries[-1].event["type"] == "detach"
  assert entries[-1].event["reason"] == "operator_pause"
  assert [entry.event["type"] for entry in event_log.entries] == [
    "operator_pause",
    "session_recap",
    "stream_complete",
  ]
  assert event_log.entries[-1].event["reason"] == "operator_pause"
  assert (
    event_log.entries[-1].event["terminal_disposition"]
    == "interrupted"
  )


def test_operator_pause_after_turn_stops_before_tool_dispatch(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "operator-pause-before-tools.jsonl")
  event_log = EventLog()
  pause_event = asyncio.Event()
  tool_calls: list[dict[str, Any]] = []

  async def _unexpected_tool(tool_input: dict[str, Any], **kwargs: Any):
    _ = kwargs
    tool_calls.append(tool_input)
    return {"unexpected": True}, None

  provider = _ScriptedProvider(
    [_tool_turn()],
    after_turn=lambda _turn_index: pause_event.set(),
  )
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"lookup": _unexpected_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    operator_pause_event=pause_event,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run lookup"}]))

  assert tool_calls == []
  entries, _ = _run(log.query(order="asc"))
  interruptions = [entry.event for entry in entries if entry.event["type"] == "interrupted"]
  assert interruptions[0]["reason"] == "operator_pause"
  assert interruptions[0]["safe_boundary"] == "after_turn_before_tools"
  assert not any(entry.event["type"] == "tool_call_start" for entry in entries)


@pytest.mark.parametrize("ending", ["completed", "operator_pause", "cancelled", "dispatch_error"])
def test_followup_replays_complete_tool_batch_before_user_text(
  tmp_path: Path,
  ending: str,
) -> None:
  from agent_gateway.providers.anthropic import AnthropicProvider

  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "followup.jsonl")
    pause_event = asyncio.Event()

    async def _lookup(tool_input: dict[str, Any], **kwargs: Any):
      if ending == "cancelled":
        raise asyncio.CancelledError()
      return {"observed": tool_input["query"]}, None

    opening = _ScriptedProvider(
      [
        [
          *_tool_turn(tool_id="tool-a", tool_input={"query": "first"})[:-1],
          *_tool_turn(tool_id="tool-b", tool_input={"query": "second"})[1:],
        ],
        _text_turn("Opening answer"),
      ],
      after_turn=(
        (lambda _: pause_event.set())
        if ending == "operator_pause" else None
      ),
    )
    runner = _make_runner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(local_tool_handlers={"lookup": _lookup}),
      session_id="sess-parent",
      capability_execution=_runner_execution(opening),
      agent_session_log=log,
      operator_pause_event=pause_event,
      on_tool_result=lambda _: [{"type": "text", "text": "Source annotation"}],
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    if ending == "dispatch_error":
      async def _fail_dispatch(*args: Any, **kwargs: Any):
        raise RuntimeError("dispatch unavailable")
      runner._execute_single_tool = _fail_dispatch
    if ending in {"cancelled", "dispatch_error"}:
      expected = asyncio.CancelledError if ending == "cancelled" else RuntimeError
      with pytest.raises(expected):
        await runner.run(messages=[{"role": "user", "content": "Look up both"}])
    else:
      await runner.run(messages=[{"role": "user", "content": "Look up both"}])

    entries, _ = await log.query(order="asc")
    terminal_blocks = [
      block
      for entry in entries
      for block in entry.event.get("final_tool_result_blocks", [])
      if block.get("type") == "tool_result"
    ]
    assert {block["tool_use_id"] for block in terminal_blocks} == {"tool-a", "tool-b"}
    if ending != "completed":
      assert all(block.get("is_error") for block in terminal_blocks)

    requests: list[dict[str, Any]] = []
    anthropic = AnthropicProvider()

    class _FollowupProvider(_ScriptedProvider):
      def normalize_messages(self, messages, model_info):
        for index, message in enumerate(messages):
          content = message["content"]
          if message["role"] != "assistant" or not isinstance(content, list):
            continue
          tool_ids = {block["id"] for block in content if block.get("type") == "tool_use"}
          if not tool_ids:
            continue
          next_message = messages[index + 1]
          assert next_message["role"] == "user"
          result_ids = set()
          for block in next_message["content"]:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
              break
            result_ids.add(block["tool_use_id"])
          assert result_ids == tool_ids
        return anthropic.normalize_messages(messages, model_info)

      def build_request_params(self, **kwargs):
        params = anthropic.build_request_params(**kwargs)
        requests.append(params)
        return params

    followup = _FollowupProvider([_text_turn("Follow-up answer")])
    followup_runner = _make_runner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-parent",
      capability_execution=_runner_execution(followup),
      agent_session_log=log,
      context_builder=SessionContextBuilder(agent_session_log=log),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    await followup_runner.run(messages=[{"role": "user", "content": "Follow up"}])
    assert len(requests) == 1
    results = [
      block for message in requests[0]["messages"]
      if isinstance(message["content"], list)
      for block in message["content"] if block.get("type") == "tool_result"
    ]
    assert {block["tool_use_id"] for block in results} == {"tool-a", "tool-b"}
    if ending == "completed":
      assert [json.loads(block["content"])["observed"] for block in results] == ["first", "second"]
    else:
      assert all(block.get("is_error") for block in results)

  _run(_case())


def test_provider_client_creation_failure_terminalizes_without_raising(
  tmp_path: Path,
) -> None:
  class _StartupFailureProvider(_ScriptedProvider):
    def create_client(
      self,
      config: dict[str, Any],
      *,
      timeout: float | None = None,
    ) -> Any:
      _ = config, timeout
      raise RuntimeError("sensitive client construction detail")

  log = AgentSessionLog(
    path=tmp_path / "sessions" / "provider-startup-failure.jsonl"
  )
  provider = _StartupFailureProvider([])
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    shutdown_signal_provider=lambda: {"signal_name": "SIGTERM"},
    emit_session_recap=False,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  returned = _run(
    runner.run(
      messages=[{"role": "user", "content": "Start provider"}]
    )
  )

  assert returned is None
  entries, _ = _run(log.query(order="asc"))
  events = [entry.event for entry in entries]
  assert events[0]["type"] == "attach"
  assert events[1]["type"] == "user_message"
  failure = next(event for event in events if event["type"] == "error")
  assert "Provider startup failed" in failure["error"]
  assert events[-1]["type"] == "detach"
  assert events[-1]["reason"] == "error"
  assert "sensitive client construction detail" not in json.dumps(events)


def test_sub_agent_cancelled_run_emits_sub_agent_interrupted_reason(tmp_path: Path) -> None:
  stream_started = asyncio.Event()

  class _BlockingProvider(_ScriptedProvider):
    async def stream(self, client: Any, params: dict[str, Any]):
      _ = client, params
      stream_started.set()
      while True:
        await asyncio.sleep(60)
      yield  # pragma: no cover

  log = AgentSessionLog(path=tmp_path / "sessions" / "sub-agent-cancelled.jsonl")
  provider = _BlockingProvider([])
  runner = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sub-worker:parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    shutdown_signal_provider=lambda: {"signal_name": "SIGTERM"},
  )

  async def _cancel_runner() -> None:
    task = asyncio.create_task(
      runner.run(messages=[{"role": "user", "content": "Cancel"}])
    )
    await asyncio.wait_for(stream_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

  _run(_cancel_runner())

  entries, _ = _run(log.query(order="asc"))
  interrupted = next(entry.event for entry in entries if entry.event["type"] == "interrupted")
  detach = next(entry.event for entry in entries if entry.event["type"] == "detach")
  assert interrupted["reason"] == "sub_agent_cancelled"
  assert "shutdown" not in interrupted
  assert detach["reason"] == "cancelled"


def test_runner_without_context_builder_does_not_inject_prior_durable_history(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "no-context-builder.jsonl")
  _run(log.append({"type": "summary", "text": "PRIOR SUMMARY SENTINEL"}))
  _run(log.append({"type": "state_update", "payload": {"alerts": ["PRIOR STATE SENTINEL"]}}))
  _run(log.append({"type": "user_message", "content": "PRIOR USER SENTINEL"}))
  _run(log.append({"type": "assistant_message", "content": "PRIOR ASSISTANT SENTINEL"}))
  captured_messages: list[list[dict[str, Any]]] = []

  class _CapturingProvider(_ScriptedProvider):
    def build_request_params(
      self,
      *,
      model: str,
      messages: list[dict[str, Any]],
      system_prompt: str | list[tuple[str, bool]] | None,
      tools: list[dict[str, Any]],
      max_tokens: int,
      **kwargs: Any,
    ) -> dict[str, Any]:
      captured_messages.append([dict(message) for message in messages])
      return super().build_request_params(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        max_tokens=max_tokens,
        **kwargs,
      )

  provider = _CapturingProvider([_text_turn("done")])
  runner = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    context_builder=None,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "fresh dev task"}]))

  assert captured_messages
  model_input = json.dumps(captured_messages[0], default=str)
  assert "fresh dev task" in model_input
  assert "PRIOR SUMMARY SENTINEL" not in model_input
  assert "PRIOR STATE SENTINEL" not in model_input
  assert "PRIOR USER SENTINEL" not in model_input
  assert "PRIOR ASSISTANT SENTINEL" not in model_input


def test_runner_with_context_builder_ignores_fabricated_client_history(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "server-authoritative.jsonl")
  _run(log.append({"type": "user_message", "content": "SERVER PRIOR USER"}))
  _run(log.append({"type": "assistant_message", "content_blocks": [{"type": "text", "text": "SERVER PRIOR ASSISTANT"}]}))
  captured_messages: list[list[dict[str, Any]]] = []

  class _CapturingProvider(_ScriptedProvider):
    def build_request_params(
      self,
      *,
      model: str,
      messages: list[dict[str, Any]],
      system_prompt: str | list[tuple[str, bool]] | None,
      tools: list[dict[str, Any]],
      max_tokens: int,
      **kwargs: Any,
    ) -> dict[str, Any]:
      captured_messages.append([dict(message) for message in messages])
      return super().build_request_params(
        model=model,
        messages=messages,
        system_prompt=system_prompt,
        tools=tools,
        max_tokens=max_tokens,
        **kwargs,
      )

  provider = _CapturingProvider([_text_turn("done")])
  runner = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    context_builder=SessionContextBuilder(agent_session_log=log),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(
    runner.run(
      messages=[
        {"role": "user", "content": "CLIENT FABRICATED USER"},
        {"role": "assistant", "content": "CLIENT FABRICATED ASSISTANT"},
        {"role": "user", "content": "fresh question"},
      ]
    )
  )

  assert captured_messages
  model_input = json.dumps(captured_messages[0], default=str)
  assert "SERVER PRIOR USER" in model_input
  assert "SERVER PRIOR ASSISTANT" in model_input
  assert "fresh question" in model_input
  assert "CLIENT FABRICATED USER" not in model_input
  assert "CLIENT FABRICATED ASSISTANT" not in model_input


def test_stream_retry_and_terminal_error_are_durable(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 2)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)
  provider = _RetryableFailingProvider()
  log = AgentSessionLog(path=tmp_path / "sessions" / "stream-retry.jsonl")
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "review allocation"}]))

  assert provider.calls == 3
  entries, _ = _run(log.query(order="asc"))
  retry_events = [entry.event for entry in entries if entry.event.get("type") == "stream_retry"]
  error_events = [entry.event for entry in entries if entry.event.get("type") == "error"]
  visible_events = [entry.event for entry in event_log.entries]

  assert [event["attempt"] for event in retry_events] == [0, 1]
  assert all("Anthropic API error (status=200)" in event["error"] for event in retry_events)
  assert len(error_events) == 1
  assert "Anthropic API error (status=200)" in error_events[0]["error"]
  assert [event["type"] for event in visible_events if event["type"] in {"stream_retry", "error"}] == [
    "stream_retry",
    "stream_retry",
    "error",
  ]


class _RetrySleepRecorder:
  """The runner's ``asyncio`` with ``sleep`` recorded, guard polls excluded."""

  def __init__(self, guard_poll_interval: float) -> None:
    self.guard_poll_interval = guard_poll_interval
    self.retry_sleeps: list[float] = []

  def __getattr__(self, name: str) -> Any:
    return getattr(asyncio, name)

  async def sleep(self, delay: float, result: Any = None) -> Any:
    if delay != self.guard_poll_interval:
      self.retry_sleeps.append(delay)
    return await asyncio.sleep(delay, result)


def _patch_retry_policy(
  monkeypatch: pytest.MonkeyPatch,
  *,
  retry_max: int,
  delay: float,
  backoff: float,
) -> _RetrySleepRecorder:
  guard_poll_interval = 300.0
  recorder = _RetrySleepRecorder(guard_poll_interval)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", retry_max)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", delay)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_BACKOFF", backoff)
  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", guard_poll_interval)
  monkeypatch.setattr(gateway_runner, "asyncio", recorder)
  return recorder


def _journaled_retry_events(log: AgentSessionLog) -> list[dict[str, Any]]:
  entries, _ = _run(log.query(order="asc"))
  return [entry.event for entry in entries if entry.event.get("type") == "stream_retry"]


def test_stream_retry_records_attempt_elapsed_and_the_slept_backoff(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  attempt_s = 0.05
  recorder = _patch_retry_policy(monkeypatch, retry_max=2, delay=0.4, backoff=2.5)

  class _TimedFailingProvider(_RetryableFailingProvider):
    async def stream(self, client: Any, params: dict[str, Any]):
      _ = client, params
      self.calls += 1
      await asyncio.sleep(attempt_s)
      raise RuntimeError(self.message)
      yield  # pragma: no cover

  provider = _TimedFailingProvider()
  log = AgentSessionLog(path=tmp_path / "sessions" / "stream-retry-timing.jsonl")
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "review allocation"}]))

  assert provider.calls == 3
  retry_events = _journaled_retry_events(log)
  assert [event["attempt"] for event in retry_events] == [0, 1]
  assert [event["backoff_s"] for event in retry_events] == recorder.retry_sleeps
  assert recorder.retry_sleeps == pytest.approx([0.4, 1.0])
  for event in retry_events:
    # Measured from the attempt's own start: the 0.4s/1.0s backoffs before
    # attempts 1 and 2 and the earlier attempts are not counted.
    assert attempt_s <= event["attempt_elapsed_s"] < 0.4


def test_credential_rotation_on_the_last_attempt_records_no_backoff(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  recorder = _patch_retry_policy(monkeypatch, retry_max=1, delay=0.01, backoff=2.0)

  class _SpentCredentialProvider(_RetryableFailingProvider):
    def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
      _ = exc
      return ProviderCredentialFailure(provider=self.name, kind="rate_limit", status_code=429)

    def next_credential(
      self,
      config: Any,
      failure: ProviderCredentialFailure,
    ) -> dict[str, Any] | None:
      _ = config, failure
      return {}

  provider = _SpentCredentialProvider("429 Too Many Requests")
  log = AgentSessionLog(path=tmp_path / "sessions" / "credential-rotation-timing.jsonl")
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  monkeypatch.setattr(runner, "_apply_refreshed_auth_config", lambda config, refreshed: None)

  _run(runner.run(messages=[{"role": "user", "content": "review allocation"}]))

  assert provider.calls == 2
  retry_events = _journaled_retry_events(log)
  assert [event["attempt"] for event in retry_events] == [0, 1]
  # The first rotation is followed by an attempt that sleeps first; the last
  # one ends the loop, so nothing is slept and the journal records null.
  assert recorder.retry_sleeps == [0.01]
  assert [event["backoff_s"] for event in retry_events] == [0.01, None]
  assert all(event["attempt_elapsed_s"] >= 0.0 for event in retry_events)


def test_tool_call_complete_logs_final_model_facing_result_blocks(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "final-tool-result.jsonl")
  provider = _ScriptedProvider([
    _tool_turn(tool_name="warn_lookup", tool_input={"warning": "only 20% matched"}),
    _text_turn("done"),
  ])
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"warn_lookup": _warning_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run lookup"}]))

  entries, _ = _run(log.query(event_types={"tool_call_complete"}, order="asc"))
  event = entries[0].event
  assert event["result"] == {"ok": True, "low_match_warning": "only 20% matched"}
  final_blocks = event["final_tool_result_blocks"]
  assert len(final_blocks) == 1
  assert final_blocks[0]["type"] == "tool_result"
  content = final_blocks[0]["content"]
  assert "_runner_warning" in content
  assert "Low match rate detected: only 20% matched" in content


def test_tool_call_complete_pre_and_final_results_differ_when_annotated(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "pre-post-tool-result.jsonl")
  provider = _ScriptedProvider([
    _tool_turn(tool_name="warn_lookup", tool_input={"warning": "3 of 10 rows matched"}),
    _text_turn("done"),
  ])
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"warn_lookup": _warning_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run lookup"}]))

  entries, _ = _run(log.query(event_types={"tool_call_complete"}, order="asc"))
  event = entries[0].event
  assert "_runner_warning" not in event["result"]
  assert "_runner_warning" in event["final_tool_result_blocks"][0]["content"]
  assert event["final_tool_result_blocks"][0]["content"] != json.dumps(event["result"], default=str)


def test_tool_completion_retains_interceptor_warning_provenance(
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "interceptor-warning.jsonl"
  )
  provider = _ScriptedProvider([
    _tool_turn(tool_name="warning_source", tool_input={}),
    _text_turn("done"),
  ])
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(
      event_log=event_log,
      local_tool_handlers={
        "warning_source": _interceptor_warning_tool,
      },
    ),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run warning"}]))

  entries, _ = _run(
    log.query(event_types={"tool_call_complete"}, order="asc")
  )
  event = entries[0].event
  assert event["result"]["_interceptor_warnings"] == [
    "policy warning"
  ]
  model_result = json.loads(
    event["final_tool_result_blocks"][0]["content"]
  )
  assert "_interceptor_warnings" not in model_result
  assert model_result["_runner_warning"] == (
    "Policy warning: policy warning"
  )


def test_large_tool_result_is_compacted_only_for_model_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv(gateway_runner.MODEL_TOOL_RESULT_MAX_CHARS_ENV, "4000")
  log = AgentSessionLog(path=tmp_path / "sessions" / "large-tool-result.jsonl")
  provider = _ScriptedProvider([
    _tool_turn(tool_name="large_lookup", tool_input={"size": 12_000}),
    _text_turn("done"),
  ])
  event_log = EventLog()
  runner = _make_runner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log=event_log, local_tool_handlers={"large_lookup": _large_result_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Run large lookup"}]))

  entries, _ = _run(log.query(event_types={"tool_call_complete"}, order="asc"))
  event = entries[0].event
  assert event["result"]["payload"] == "x" * 12_000
  final_content = event["final_tool_result_blocks"][0]["content"]
  assert len(final_content) <= 4000
  parsed = json.loads(final_content)
  assert parsed["_runner_truncated"] is True
  assert parsed["tool_name"] == "large_lookup"
  # The model-bound copy keeps the result's own shape and elides only bulk, so the
  # small facts stay readable wherever the producer put them.
  projection = parsed["content_projection"]
  assert projection["status"] == "success"
  assert projection["ticker"] == "BIG"
  assert "...<elided chars=" in projection["payload"]
  assert parsed["original_chars"] > 4000


def test_rebuild_task_registry_ignores_tool_call_complete_final_blocks(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "rebuild-final-blocks.jsonl")
  _run(
    log.append(
      {
        "type": "task_registered",
        "task_id": "bg_0",
        "task_type": "background",
        "agent_name": "writer",
        "started_at": 1.0,
        "capability_bind": _child_execution(
          _ScriptedProvider([])
        ).bind.to_json(),
      }
    )
  )
  _run(
    log.append(
      {
        "type": "tool_call_complete",
        "tool_call_id": "tool-1",
        "tool_name": "lookup",
        "result": {"ok": True},
        "error": None,
        "final_tool_result_blocks": [
          {"type": "tool_result", "tool_use_id": "tool-1", "content": "{\"ok\": true}"}
        ],
      }
    )
  )
  provider = _ScriptedProvider([_text_turn("unused")])
  runner = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(local_tool_handlers={}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    task_registry=TaskRegistry(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner._rebuild_task_registry_from_log())

  entry = runner._task_registry.get("bg_0")
  assert entry is not None
  assert entry.state == TaskState.INTERRUPTED


def test_task_durability_events_use_type_and_query_round_trips(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "task-events.jsonl")
  capability_bind = _child_execution(
    _ScriptedProvider([])
  ).bind.to_json()
  correlation = {
    "task_id": "bg_0",
    "owner_runner_id": "runner_old",
    "owner_role": "writer",
    "sub_agent_id": "sub0:sess-parent",
    "parent_turn_id": "turn-1",
    "call_index": 0,
    "task_type": "background",
    "capability_bind": capability_bind,
  }

  _run(log.append({"type": "task_registered", **correlation, "agent_name": "writer", "started_at": 1.0}))
  _run(log.append({"type": "parent_message_sent", **correlation, "message_id": "msg-1", "message": "go"}))
  _run(log.append({"type": "task_completed", **correlation, "final_state": "completed", "completed_at": 2.0, "result": _child_report(), "error": None}))

  entries, _ = _run(log.query(event_types={"task_registered", "parent_message_sent", "task_completed"}, order="asc"))

  assert [entry.event["type"] for entry in entries] == ["task_registered", "parent_message_sent", "task_completed"]
  for entry in entries:
    for key, value in correlation.items():
      assert entry.event[key] == value


@pytest.mark.parametrize("recovery_delay", [0, 3 * 60 * 60])
def test_runner_stale_recovery_synthesizes_orphan_tool_and_prior_writer_interrupt(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  recovery_delay: int,
  owner_session_host_policy,
) -> None:
  from agent_gateway.providers.codex import CodexProvider
  from agent_gateway.providers.openai import OpenAIProvider
  owner_session_host_policy.get_tool_risk_value = {"file_read": "read_only"}.__getitem__

  now = gateway_runner.time.time()
  monkeypatch.setattr(gateway_runner.time, "time", lambda: now - recovery_delay)
  log = AgentSessionLog(path=tmp_path / "sessions" / "recovery.jsonl")
  _run(
    log.append(
      {
        "type": "attach",
        "gateway_session_id": "analyst:2026-04-11",
        "runner_id": "runner_old",
        "started_at": 100.0,
        "client_kind": "cron",
        "role": "writer",
      }
    )
  )
  _run(log.append({
    "type": "assistant_message",
    "content_blocks": [
      {"type": "tool_use", "id": tool_id, "name": "file_read", "input": {}}
      for tool_id in ("tool-orphan", "tool-not-started")
    ],
    "runner_id": "runner_old",
    "role": "writer",
  }))
  _run(
    log.append(
      {
        "type": "tool_call_start",
        "tool_call_id": "tool-orphan",
        "tool_name": "file_read",
        "tool_input": {"path": "/tmp/report.md"},
        "started_at": 101.0,
        "runner_id": "runner_old",
        "role": "writer",
      }
    )
  )

  monkeypatch.setattr(gateway_runner.time, "time", lambda: now)
  replayed: list[dict[str, Any]] = []

  class _RecoveryProvider(_ScriptedProvider):
    def build_request_params(self, **kwargs):
      replayed.extend(kwargs["messages"])
      unmatched = {}
      for adapter in (OpenAIProvider(), CodexProvider()):
        params = adapter.build_request_params(**{**kwargs, "model": "gpt-6-sol"})
        calls = {item["call_id"] for item in params["input"] if item.get("type") == "function_call"}
        unmatched[adapter.name] = {
          item["call_id"] for item in params["input"]
          if item.get("type") == "function_call_output" and item["call_id"] not in calls
        }
      assert not any(unmatched.values()), unmatched
      return {}

  provider = _RecoveryProvider([_text_turn("recovered")])
  runner = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(local_tool_handlers={}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    context_builder=SessionContextBuilder(agent_session_log=log),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Continue"}]))

  entries, _ = _run(log.query(order="asc"))
  interruptions = [entry.event for entry in entries if entry.event["type"] == "tool_call_interrupted"]
  assert {event["tool_call_id"] for event in interruptions} == {"tool-orphan", "tool-not-started"}
  assert all(event["tool_risk"] == "read_only" for event in interruptions)
  orphan = next(event for event in interruptions if event["tool_call_id"] == "tool-orphan")
  assert orphan["original_started_at"] == 101.0
  recovery = next(entry.event for entry in entries if entry.event["type"] == "interrupted")
  new_attach = next(
    entry.event for entry in entries
    if entry.event["type"] == "attach" and entry.event["runner_id"] != "runner_old"
  )
  assert recovery["reason"] == "recovered_on_attach"
  assert recovery["runner_id"] == "runner_old"
  assert recovery["recovered_by_runner_id"] == new_attach["runner_id"]
  assert new_attach["runner_id"] != "runner_old"
  messages = _run(SessionContextBuilder(agent_session_log=log).build())
  assistant_index = next(index for index, message in enumerate(messages) if message["role"] == "assistant")
  results = messages[assistant_index + 1]["content"]
  assert {block["tool_use_id"] for block in results} == {"tool-orphan", "tool-not-started"}
  assert all(block["is_error"] for block in results)
  assert "tool-orphan" in json.dumps(replayed)
  assert "tool-not-started" in json.dumps(replayed)


def test_second_writer_lease_acquisition_raises(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "lease.jsonl")
  provider_one = _ScriptedProvider([_text_turn("one")])
  runner_one = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-one",
    capability_execution=_runner_execution(provider_one),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  provider_two = _ScriptedProvider([_text_turn("two")])
  runner_two = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-two",
    capability_execution=_runner_execution(provider_two),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  runner_one._runner_id = "runner_one"
  runner_two._runner_id = "runner_two"

  async def _exercise() -> None:
    await runner_one._acquire_writer_lease_and_recover()
    with pytest.raises(RuntimeError, match="Writer lease already held"):
      await runner_two._acquire_writer_lease_and_recover()
    runner_one._release_write_lease()

  _run(_exercise())


def test_register_background_task_emits_task_registered_with_correlation(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "registered.jsonl")
    calls: list[tuple[Any, int]] = []

    def _fake_derive(parent_session: Any, call_index: int) -> str:
      calls.append((parent_session, call_index))
      return f"sub{call_index}:derived-parent"

    monkeypatch.setattr(gateway_runner, "_derive_sub_agent_id", _fake_derive)
    provider = _ScriptedProvider([_text_turn("unused")])
    runner = _make_runner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-parent",
      capability_execution=_runner_execution(provider),
      agent_session_log=log,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    runner._runner_id = "runner_new"

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      return _child_report(), None

    capability_bind_receipt = _child_execution(
      runner._provider,
    ).bind.to_json()
    result, error = await runner._register_background_task(
      tool_input={"task": "Collect"},
      handler=_handler,
      agent_name="writer",
      parent_turn_id="turn-1",
      capability_bind_receipt=capability_bind_receipt,
    )
    assert error is None
    assert result is not None
    task = runner._task_registry.get(result["task_id"])
    assert task is not None
    assert task.asyncio_task is not None
    await asyncio.wait_for(task.asyncio_task, timeout=1.0)

    registered, _ = await log.query(event_types={"task_registered"}, order="asc")
    assert len(registered) == 1
    event = registered[0].event
    assert event["type"] == "task_registered"
    assert event["task_id"] == result["task_id"]
    assert event["owner_runner_id"] == "runner_new"
    assert event["owner_role"] == "writer"
    assert event["sub_agent_id"] == "sub0:derived-parent"
    assert event["parent_turn_id"] == "turn-1"
    assert event["call_index"] == 0
    assert event["task_type"] == "background_agent"
    assert event["capability_bind"] == capability_bind_receipt
    assert "provider_name" not in event
    assert "model" not in event
    assert task.metadata["sub_agent_id"] == "sub0:derived-parent"
    assert calls == [("sess-parent", 0)]

  _run(_case())


# --------------------------------------------------------------------------
# A-M8 / T3-I03 + T3-I04 — transactional background registration (WP2).
# --------------------------------------------------------------------------


def _a_m8_runner(
  log: AgentSessionLog | None,
  *,
  max_concurrent_sub_agents: int | None = None,
) -> AgentRunner:
  provider = _ScriptedProvider([_text_turn("unused")])
  runner = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    max_concurrent_sub_agents=max_concurrent_sub_agents,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  runner._runner_id = "runner_new"
  return runner


def _a_m8_bind_receipt(runner: AgentRunner) -> dict[str, str]:
  return _child_execution(runner._provider).bind.to_json()


async def _durable_task_events(
  log: AgentSessionLog,
  task_id: str,
) -> list[dict[str, Any]]:
  entries, _ = await log.query(
    event_types={"task_registered", "task_completed"},
    order="asc",
  )
  return [
    entry.event
    for entry in entries
    if entry.event.get("task_id") == task_id
  ]


def test_concurrent_registration_admits_every_caller_and_queues_execution(
  tmp_path: Path,
) -> None:
  """The in-flight count is a measurement, never a refusal (rule 8).

  Six concurrent callers past the old ceiling of two are all admitted and
  durably registered; the sub-agent semaphore runs at most two at a time and
  the rest wait for a slot.
  """

  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "race.jsonl")
    runner = _a_m8_runner(log, max_concurrent_sub_agents=2)
    capability_bind_receipt = _a_m8_bind_receipt(runner)
    release = asyncio.Event()
    slow_lookups = 0
    original_lookup = runner._lookup_task_in_log

    async def _slow_lookup(task_id: str):
      nonlocal slow_lookups
      slow_lookups += 1
      await asyncio.sleep(0)
      return await original_lookup(task_id)

    runner._lookup_task_in_log = _slow_lookup

    running = 0
    peak_running = 0

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      nonlocal running, peak_running
      running += 1
      peak_running = max(peak_running, running)
      try:
        await release.wait()
      finally:
        running -= 1
      return _child_report(), None

    async def _register(index: int):
      return await runner._register_background_task(
        tool_input={"task": f"collect-{index}"},
        handler=_handler,
        agent_name="writer",
        capability_bind_receipt=capability_bind_receipt,
        task_id_override=f"bg_race_{index}",
      )

    outcomes = await asyncio.gather(*[_register(index) for index in range(6)])

    admitted = [result for result, error in outcomes if error is None]
    assert len(admitted) == 6
    assert runner._task_registry.admission_count == 6
    assert slow_lookups == 6

    registered, _ = await log.query(
      event_types={"task_registered"},
      order="asc",
    )
    assert len(registered) == 6

    async def _two_running() -> None:
      while running < 2:
        await asyncio.sleep(0)

    await asyncio.wait_for(_two_running(), timeout=1.0)
    for _ in range(20):
      await asyncio.sleep(0)
    assert running == 2

    release.set()
    for result in admitted:
      assert result is not None
      entry = runner._task_registry.get(result["task_id"])
      assert entry is not None
      assert entry.asyncio_task is not None
      await asyncio.wait_for(entry.asyncio_task, timeout=1.0)
    assert peak_running == 2

  _run(_case())


def test_post_append_failure_appends_a_compensating_terminal(
  tmp_path: Path,
) -> None:
  """Kills seam-map Hole 2 (D-A8-2/3): no durable registration without a terminal.

  This is the live CUR-E2E-03 shape: the append committed and then the
  RUNNING transition blew up, leaving a PENDING ghost that held
  ``admission_count > 0`` forever and made the run loop refuse clean success
  with ``background_delivery_incomplete``.
  """

  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "bracket.jsonl")
    runner = _a_m8_runner(log)
    original_transition = runner._task_registry.transition

    def _transition(task_id: str, new_state: TaskState, **kwargs: Any):
      if new_state == TaskState.RUNNING:
        raise RuntimeError("post-append transition failure")
      return original_transition(task_id, new_state, **kwargs)

    runner._task_registry.transition = _transition

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      raise AssertionError("worker must never run")

    with pytest.raises(RuntimeError, match="post-append transition failure"):
      await runner._register_background_task(
        tool_input={"task": "collect"},
        handler=_handler,
        agent_name="writer",
        capability_bind_receipt=_a_m8_bind_receipt(runner),
        task_id_override="bg_bracket",
      )

    entry = runner._task_registry.get("bg_bracket")
    assert entry is not None
    assert entry.state == TaskState.FAILED
    assert entry.asyncio_task is None

    events = await _durable_task_events(log, "bg_bracket")
    assert [event["type"] for event in events] == [
      "task_registered",
      "task_completed",
    ]
    assert events[-1]["final_state"] == "failed"
    assert str(events[-1]["result"]["reason"]).startswith("registration_aborted")

    # The acceptance signal for CUR-E2E-03/04: the completion guard clears.
    assert runner._task_registry.admission_count == 0
    blockers, _fingerprint = _background_success_snapshot(runner)
    assert "pending_or_running_tasks" not in blockers

    # T3-I03: replay rebuilds FAILED, never a ghost INTERRUPTED.
    replayed = TaskRegistry()
    replayed.load_from_events(events)
    rebuilt = replayed.get("bg_bracket")
    assert rebuilt is not None
    assert rebuilt.state == TaskState.FAILED

  _run(_case())


def test_post_append_worker_spawn_failure_appends_a_compensating_terminal(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The second post-append exit the seam map names: ``create_task`` raising."""

  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "spawn.jsonl")
    runner = _a_m8_runner(log)

    def _refuse_create_task(coro: Any, *, name: str | None = None):
      _ = name
      coro.close()
      raise RuntimeError("worker spawn failure")

    monkeypatch.setattr(
      gateway_runner,
      "asyncio",
      _AsyncioProxy(create_task=_refuse_create_task),
    )

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      raise AssertionError("worker must never run")

    with pytest.raises(RuntimeError, match="worker spawn failure"):
      await runner._register_background_task(
        tool_input={"task": "collect"},
        handler=_handler,
        agent_name="writer",
        capability_bind_receipt=_a_m8_bind_receipt(runner),
        task_id_override="bg_spawn",
      )

    events = await _durable_task_events(log, "bg_spawn")
    assert [event["type"] for event in events] == [
      "task_registered",
      "task_completed",
    ]
    assert events[-1]["final_state"] == "failed"
    assert runner._task_registry.admission_count == 0

  _run(_case())


def test_post_append_cancellation_shields_the_compensating_terminal(
  tmp_path: Path,
) -> None:
  """D-A8-2: a ``CancelledError`` after the append still leaves a terminal."""

  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "cancelled.jsonl")
    runner = _a_m8_runner(log)
    original_transition = runner._task_registry.transition

    def _transition(task_id: str, new_state: TaskState, **kwargs: Any):
      if new_state == TaskState.RUNNING:
        raise asyncio.CancelledError()
      return original_transition(task_id, new_state, **kwargs)

    runner._task_registry.transition = _transition

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      raise AssertionError("worker must never run")

    with pytest.raises(asyncio.CancelledError):
      await runner._register_background_task(
        tool_input={"task": "collect"},
        handler=_handler,
        agent_name="writer",
        capability_bind_receipt=_a_m8_bind_receipt(runner),
        task_id_override="bg_cancelled",
      )

    events = await _durable_task_events(log, "bg_cancelled")
    assert [event["type"] for event in events] == [
      "task_registered",
      "task_completed",
    ]
    assert events[-1]["final_state"] == "failed"
    assert runner._task_registry.admission_count == 0

  _run(_case())


def test_pre_append_failure_still_discards_the_reservation(
  tmp_path: Path,
) -> None:
  """D-A8-2's other half: pre-append -> ``discard_unstarted``, no terminal."""

  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "pre-append.jsonl")
    runner = _a_m8_runner(log)

    async def _refuse_append(event: dict[str, Any]):
      raise RuntimeError("durable registration unavailable")

    runner._append_durable_event = _refuse_append

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      raise AssertionError("worker must never run")

    result, error = await runner._register_background_task(
      tool_input={"task": "collect"},
      handler=_handler,
      agent_name="writer",
      capability_bind_receipt=_a_m8_bind_receipt(runner),
      task_id_override="bg_pre_append",
    )

    assert result is None
    assert error is not None
    assert error["code"] == "background_registration_failed"
    assert runner._task_registry.get("bg_pre_append") is None
    assert runner._task_registry.admission_count == 0
    assert await _durable_task_events(log, "bg_pre_append") == []

  _run(_case())


class _AsyncioProxy:
  """Expose the real ``asyncio`` module with a few attributes replaced."""

  def __init__(self, **overrides: Any) -> None:
    self._overrides = overrides

  def __getattr__(self, name: str) -> Any:
    if name in self._overrides:
      return self._overrides[name]
    return getattr(asyncio, name)


def test_task_completed_is_durable_before_terminal_transition() -> None:
  async def _case() -> None:
    provider = _ScriptedProvider([_text_turn("unused")])
    runner = _make_runner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-parent",
      capability_execution=_runner_execution(provider),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    entry = runner._task_registry.register("background_agent")
    capability_bind = _child_execution(provider).bind.to_json()
    entry.capability_bind_receipt = capability_bind
    entry.metadata.update(
      {
        "owner_runner_id": "runner_new",
        "owner_role": "writer",
        "sub_agent_id": "sub0:sess-parent",
        "parent_turn_id": "turn-1",
        "call_index": 0,
        "task_type": "background",
        "capability_bind": capability_bind,
      }
    )
    ordering: list[str] = []

    async def _append(event: dict[str, Any]):
      if event.get("type") == "task_completed":
        ordering.append("append_task_completed")

    original_transition = runner._task_registry.transition

    def _transition(task_id: str, new_state: TaskState, **kwargs: Any):
      if new_state in {TaskState.COMPLETED, TaskState.FAILED}:
        ordering.append(f"transition_{new_state.value}")
      return original_transition(task_id, new_state, **kwargs)

    runner._append_durable_event = _append
    runner._task_registry.transition = _transition

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      return _child_report(), None

    await runner._run_background_agent(entry, _handler, {}, 0)

    assert ordering == ["append_task_completed", "transition_completed"]

  _run(_case())


def test_rebuild_hot_set_lazy_lookup_and_seq_restore(tmp_path: Path) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "rebuild.jsonl")
    provider = _ScriptedProvider([_text_turn("unused")])
    capability_bind = _child_execution(provider).bind.to_json()
    for index in range(4):
      await log.append(
        {
          "type": "task_registered",
          "task_id": f"bg_{index}",
          "owner_runner_id": "runner_old",
          "owner_role": "writer",
          "sub_agent_id": f"sub{index}:sess-parent",
          "parent_turn_id": f"turn-{index}",
          "call_index": index,
          "task_type": "background",
          "capability_bind": capability_bind,
          "agent_name": f"agent-{index}",
          "started_at": 100.0 + index,
        }
      )
      if index == 1:
        await log.append(
          {
            "type": "task_completed",
            "task_id": "bg_1",
            "owner_runner_id": "runner_old",
            "owner_role": "writer",
            "sub_agent_id": "sub1:sess-parent",
            "parent_turn_id": "turn-1",
            "call_index": 1,
            "task_type": "background",
            "capability_bind": capability_bind,
            "final_state": "completed",
            "completed_at": 150.0,
            "result": _child_report(),
            "error": None,
          }
        )
    await log.append(
      {
        "type": "parent_message_sent",
        "task_id": "bg_99",
        "owner_runner_id": "runner_old",
        "owner_role": "writer",
        "sub_agent_id": "sub99:sess-parent",
        "parent_turn_id": "turn-99",
        "call_index": 99,
        "task_type": "background",
        "capability_bind": capability_bind,
        "message_id": "msg-99",
        "message": "dangling event should not count toward hot set cap",
      }
    )
    registry = TaskRegistry(max_retained=2)
    runner = _make_runner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-parent",
      capability_execution=_runner_execution(provider),
      agent_session_log=log,
      task_registry=registry,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )

    all_result, all_error = await runner.get_background_result({"task_id": "*"})
    assert all_error is None
    assert all_result is not None
    assert [task["task_id"] for task in all_result["tasks"]] == ["bg_2", "bg_3"]

    lazy_result, lazy_error = await runner.get_background_result({"task_id": "bg_1"})
    assert lazy_error is None
    assert lazy_result is not None
    assert lazy_result["status"] == "completed"
    assert lazy_result["report"]["summary"] == "done"

    assert registry.register("background_agent").task_id == "bg_4"

  _run(_case())


def test_rebuild_renders_interrupted_and_completed_crash_cases(tmp_path: Path) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "crash-cases.jsonl")
    provider = _ScriptedProvider([_text_turn("unused")])
    base = {
      "owner_runner_id": "runner_old",
      "owner_role": "writer",
      "parent_turn_id": "turn-1",
      "task_type": "background",
      "capability_bind": _child_execution(provider).bind.to_json(),
    }
    await log.append({"type": "task_registered", **base, "task_id": "bg_0", "sub_agent_id": "sub0:sess-parent", "call_index": 0, "agent_name": "running", "started_at": 100.0})
    await log.append({"type": "task_registered", **base, "task_id": "bg_1", "sub_agent_id": "sub1:sess-parent", "call_index": 1, "agent_name": "done", "started_at": 110.0})
    await log.append({"type": "task_completed", **base, "task_id": "bg_1", "sub_agent_id": "sub1:sess-parent", "call_index": 1, "final_state": "completed", "completed_at": 120.0, "result": _child_report(), "error": None})
    await log.append({"type": "task_registered", **base, "task_id": "bg_2", "sub_agent_id": "sub2:sess-parent", "call_index": 2, "agent_name": "killed", "started_at": 130.0})

    runner = _make_runner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-parent",
      capability_execution=_runner_execution(provider),
      agent_session_log=log,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )

    interrupted, interrupted_error = await runner.get_background_result({"task_id": "bg_0"})
    completed, completed_error = await runner.get_background_result({"task_id": "bg_1"})
    killed_as_interrupted, killed_error = await runner.get_background_result({"task_id": "bg_2"})

    assert interrupted_error is None
    assert interrupted is not None
    assert interrupted["status"] == "interrupted"
    assert interrupted["completed"] is True
    assert interrupted["agent"] == "running"
    assert interrupted["started_at"] == 100.0
    assert interrupted["sub_agent_id"] == "sub0:sess-parent"
    assert interrupted["parent_turn_id"] == "turn-1"
    assert completed_error is None
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["report"]["summary"] == "done"
    assert killed_error is None
    assert killed_as_interrupted is not None
    assert killed_as_interrupted["status"] == "interrupted"

  _run(_case())


def test_sub_agent_events_are_written_to_parent_log(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "sub-agent.jsonl")
  provider = _ScriptedProvider([
    _tool_turn(tool_id="tool-sub", tool_input={"query": "MSFT"}),
    _text_turn("sub done"),
  ])
  parent_runner = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(local_tool_handlers={"lookup": _lookup_tool}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    workspace_dir=str(tmp_path),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  sub_session = GatewaySession(
    session_id="sub0:sess-parent",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="alice",
    auth_config={"api_key": "k"},
  )
  logical_task, attempt, result_requirement, result_provenance = (
    _subagent_result_identity(
      delegation_id="test-sub-agent-events",
      physical_task_id="sub0:sess-parent",
    )
  )
  child_runner_ref: list[Any] = []
  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect background context",
      capability_execution=_child_execution(parent_runner._provider),
      bind_job_runner=child_runner_ref.append,
      logical_task=logical_task,
      attempt=attempt,
      result_requirement=result_requirement,
      result_provenance=result_provenance,
      skill_name="test-child",
      dispatcher=_make_dispatcher(
        local_tool_handlers={"lookup": _lookup_tool},
      ),
      sub_session=sub_session,
    )
  )

  assert error is None
  assert result is not None

  entries, _ = _run(log.query(role="sub_agent", order="asc"))
  event_types = [entry.event["type"] for entry in entries]
  assert event_types == [
    "attach",
    "user_message",
    "assistant_message",
    "tool_call_start",
    "tool_call_complete",
    "assistant_message",
    "detach",
  ]
  assert {entry.event["sub_agent_id"] for entry in entries} == {"sub0:sess-parent"}


def test_spawn_sub_agent_uses_shared_sub_agent_id_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "sub-agent-helper.jsonl")
  calls: list[tuple[Any, int]] = []

  def _fake_derive(parent_session: Any, call_index: int) -> str:
    calls.append((parent_session, call_index))
    return f"sub{call_index}:derived-parent"

  monkeypatch.setattr(gateway_runner, "_derive_sub_agent_id", _fake_derive)
  provider = _ScriptedProvider([_text_turn("sub done")])
  parent_runner = _make_runner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    workspace_dir=str(tmp_path),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  logical_task, attempt, result_requirement, result_provenance = (
    _subagent_result_identity(
      delegation_id="test-shared-sub-agent-id",
      physical_task_id="sub3:derived-parent",
    )
  )
  child_runner_ref: list[Any] = []
  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect background context",
      capability_execution=_child_execution(parent_runner._provider),
      bind_job_runner=child_runner_ref.append,
      logical_task=logical_task,
      attempt=attempt,
      result_requirement=result_requirement,
      result_provenance=result_provenance,
      skill_name="test-child",
      dispatcher=_make_dispatcher(),
      sub_session=None,
      call_index=3,
    )
  )

  assert error is None
  assert result is not None
  assert calls == [("sess-parent", 3)]
  entries, _ = _run(log.query(role="sub_agent", order="asc"))
  assert {entry.event["sub_agent_id"] for entry in entries} == {"sub3:derived-parent"}


# The user rows the run loop injects for one request or one stop boundary and
# never writes to the log; resume-invariance is stated modulo exactly this set.
_EPHEMERAL_NUDGE_PREFIXES = (
  "[System: New background-task notifications are included above.",
  "[System: The following background task result(s) were delivered as ",
  "[System: Background result payload delivery is still required for exact ",
  "[System: Your previous response was cut off at the output-token limit",
)
_CONTINUATION_NUDGE_TEXT = (
  "[History was compacted into the summary above. Continue the task from that summary.]"
)
_RESUME_MODEL = "claude-sonnet-5"
_TURN_ONE_TOKEN = "SCORE 89.0"
_CHILD_ONLY_MARKERS = ("child_fg_probe", "CHILD-FG", "child_bg_probe", "CHILD-BG")
_SUMMARY_TEXT = (
  f"Earlier: the portfolio risk score was {_TURN_ONE_TOKEN}. A delegated margin "
  "review found margins stable; a background filings review found filings clean. "
  "Next step: re-run the lookup the user asked for and answer from it. "
) * 2


def _strip_ephemeral_nudges(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  kept = []
  for message in messages:
    content = message.get("content")
    if (
      message.get("role") == "user"
      and isinstance(content, str)
      and (
        content.startswith(_EPHEMERAL_NUDGE_PREFIXES)
        or content == _CONTINUATION_NUDGE_TEXT
      )
    ):
      continue
    kept.append(message)
  return kept


def _model_visible(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  return truncate_to_last_compaction(messages, compaction_as_text=False)


class _ResumeSessionProvider(_ScriptedProvider):
  """Parent model for one interactive session; also answers portable summaries."""

  def __init__(self, turns: list[list[StreamEvent]]) -> None:
    super().__init__(turns)
    self.context_window = 200_000
    self.summary_requests = 0

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      context_window=self.context_window,
      max_output_tokens=4096,
    )

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    messages = kwargs["messages"]
    last_content = str(messages[-1].get("content") or "") if messages else ""
    is_summary = "The <summary> content is the only text" in last_content
    return {"kind": "summary" if is_summary else "main"}

  async def stream(self, client: Any, params: dict[str, Any]):
    if params.get("kind") == "summary":
      self.summary_requests += 1
      for event in _text_turn(_SUMMARY_TEXT):
        yield event
      return
    async for event in super().stream(client, params):
      yield event


def _compaction_tool_turn(summary: str) -> list[StreamEvent]:
  # Native compaction: the provider's compaction block arrives inside the
  # assistant message it compacted for, followed by that message's own work.
  turn = _tool_turn(tool_id="tool-post", tool_input={"query": "post-compaction"})
  return [
    turn[0],
    StreamEvent(type="compaction", raw_block={"type": "compaction", "content": summary}),
    *turn[1:],
  ]


@pytest.mark.parametrize("compaction", ["portable", "native"])
def test_session_context_builder_resumes_the_model_visible_state_of_every_turn(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  compaction: str,
) -> None:
  monkeypatch.delenv(server_compaction.PORTABLE_COMPACTION_ENV, raising=False)
  monkeypatch.setattr(server_compaction, "MIN_COMPACT_GAIN_TOKENS", 500)

  parent_turns = [
    # Turn 1: an answer later turns must still see.
    _text_turn(f"Overall risk score: {_TURN_ONE_TOKEN} / 100."),
    # Turn 2: a foreground child; the parent sees only its run_agent result.
    _tool_turn(
      tool_id="tool-fg",
      tool_name="run_agent",
      tool_input={"objective": "Review margins", "background": False},
    ),
    _text_turn("The delegated review found margins stable."),
    # Turn 3: a background child, delivered while the run holds the turn open.
    _tool_turn(
      tool_id="tool-bg",
      tool_name="run_agent",
      tool_input={"objective": "Review filings", "background": True},
    ),
    _text_turn("Started the background filings review."),
    _text_turn("Background review: filings clean."),
    # Turn 4: a plain follow-up after the delivery.
    _text_turn("Nothing else is outstanding."),
    # Turn 5: mid-run compaction, then one more tool round.
    (
      _tool_turn(tool_id="tool-post", tool_input={"query": "post-compaction"})
      if compaction == "portable"
      else _compaction_tool_turn(_SUMMARY_TEXT)
    ),
    _text_turn("Post-compaction lookup done."),
  ]
  provider = _ResumeSessionProvider(parent_turns)
  foreground_child = _ScriptedProvider([
    _mixed_text_tool_turn(
      text="CHILD-FG scratch note",
      tool_id="child-fg-1",
      tool_name="child_fg_probe",
      tool_input={"query": "CHILD-FG probe input"},
    ),
    _text_turn("CHILD-FG final draft"),
  ])
  background_child = _ScriptedProvider([
    _mixed_text_tool_turn(
      text="CHILD-BG scratch note",
      tool_id="child-bg-1",
      tool_name="child_bg_probe",
      tool_input={"query": "CHILD-BG probe input"},
    ),
    _text_turn("CHILD-BG final draft"),
  ])

  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "resume-invariance.jsonl")
    current: dict[str, AgentRunner] = {}

    async def _spawn_child(
      runner: AgentRunner,
      *,
      objective: str,
      child_provider: ModelProvider,
      child_tool: str,
      call_index: int,
    ) -> None:
      logical_task, attempt, result_requirement, result_provenance = (
        _subagent_result_identity(
          delegation_id=f"resume-invariance-{call_index}",
          physical_task_id=f"sub{call_index}:sess-parent",
        )
      )
      child_runner_ref: list[Any] = []
      result, error = await runner.spawn_sub_agent(
        objective,
        capability_execution=_child_execution(child_provider),
        bind_job_runner=child_runner_ref.append,
        logical_task=logical_task,
        attempt=attempt,
        result_requirement=result_requirement,
        result_provenance=result_provenance,
        skill_name="test-child",
        dispatcher=_make_dispatcher(local_tool_handlers={child_tool: _lookup_tool}),
        sub_session=None,
        call_index=call_index,
      )
      assert error is None and result is not None

    # A local `run_agent` over the runner's real child and background-task
    # machinery: it writes the same durable rows (child rows under
    # `role: sub_agent`, the parent's own `tool_call_complete`) without the
    # operation-catalog admission the resume contract does not read.
    async def _run_agent(tool_input: dict[str, Any], **_kwargs: Any):
      runner = current["runner"]
      if not tool_input.get("background"):
        await _spawn_child(
          runner,
          objective=tool_input["objective"],
          child_provider=foreground_child,
          child_tool="child_fg_probe",
          call_index=7,
        )
        return _child_report("Margins stable per delegated review."), None

      async def _background(child_input: dict[str, Any], *, call_index: int, **_: Any):
        await _spawn_child(
          runner,
          objective=child_input["objective"],
          child_provider=background_child,
          child_tool="child_bg_probe",
          call_index=call_index,
        )
        return _child_report("Filings clean per background review."), None

      return await runner._register_background_task(
        tool_input=tool_input,
        handler=_background,
        agent_name="filings-reviewer",
        capability_bind_receipt=_child_execution(background_child).bind.to_json(),
      )

    async def _turn(user_text: str, *, compaction_trigger: int | None = None) -> list[dict[str, Any]]:
      # One runner per turn, as the interactive path builds it: its context is
      # `SessionContextBuilder.build()` plus the new user message.
      runner = _make_runner(
        event_log=EventLog(),
        dispatcher=_make_dispatcher(
          local_tool_handlers={"run_agent": _run_agent, "lookup": _lookup_tool},
        ),
        session_id="sess-parent",
        capability_execution=_runner_execution(provider, model=_RESUME_MODEL),
        agent_session_log=log,
        context_builder=SessionContextBuilder(agent_session_log=log),
        workspace_dir=str(tmp_path),
        compaction_trigger=compaction_trigger,
        user_id="alice",
        billing_mode="byok",
        rate_table_version="unknown",
      )
      current["runner"] = runner
      # Observation: the run loop hands its live `current_messages` list to
      # every `_stream_turn` call and mutates that same list afterwards (nudge
      # and portable-anchor removal); a final `end_turn` reply is never
      # appended to it. So the post-turn state is the last call's list, read
      # after `run()` returns, plus that call's reply in the run loop's own
      # `assistant_turn_message` shape.
      observed: dict[str, Any] = {}
      stream_turn = runner._stream_turn

      async def _recording_stream_turn(**kwargs: Any):
        result = await stream_turn(**kwargs)
        observed["messages"] = kwargs["current_messages"]
        observed["result"] = result
        return result

      runner._stream_turn = _recording_stream_turn
      await runner.run(messages=[{"role": "user", "content": user_text}])
      _client, final_turn = observed["result"]
      assert final_turn.stop_reason == "end_turn"
      return [
        *observed["messages"],
        assistant_turn_message(
          final_turn.content_blocks,
          provider=provider.name,
          model=_RESUME_MODEL,
          stop_reason=final_turn.stop_reason,
        ),
      ]

    turns = [
      ("What is the portfolio risk score?", None),
      ("Delegate a margin review.", None),
      ("Start a background filings review.", None),
      ("Anything else outstanding?", None),
      ("Re-run the lookup. " + "context " * 400, 20 if compaction == "portable" else None),
    ]
    for index, (user_text, compaction_trigger) in enumerate(turns, start=1):
      if index == 5 and compaction == "portable":
        provider.context_window = 20
      post_turn = await _turn(user_text, compaction_trigger=compaction_trigger)
      if index == 3:
        # The delivery ran in this turn: the stripped set is not vacuous here.
        assert _strip_ephemeral_nudges(post_turn) != post_turn
      replay = await SessionContextBuilder(agent_session_log=log).build()
      replay_text = json.dumps(_model_visible(replay), default=str)
      assert _TURN_ONE_TOKEN in replay_text, index
      if index >= 2:
        assert not [marker for marker in _CHILD_ONLY_MARKERS if marker in replay_text], index

    child_rows, _ = await log.query(role="sub_agent", order="asc")
    assert {entry.event["sub_agent_id"] for entry in child_rows} == {
      "sub7:sess-parent",
      "sub0:sess-parent",
    }
    # build() itself starts at the compaction record: nothing before it replays.
    compacted = await SessionContextBuilder(agent_session_log=log).build()
    assert compacted[0]["role"] == "assistant"
    assert compacted[0]["content"][0] == {"type": "compaction", "content": (
      _SUMMARY_TEXT.strip() if compaction == "portable" else _SUMMARY_TEXT
    )}
    assert "What is the portfolio risk score?" not in json.dumps(compacted)
    if compaction == "portable":
      assert provider.summary_requests == 1

  _run(_case())
