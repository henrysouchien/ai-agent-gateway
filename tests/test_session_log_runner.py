import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_workflow_contracts import (  # noqa: E402
  AgentOperationRef,
  AttemptRef,
  OrdinaryDelegationTaskRef,
  OutcomeRequirement,
  ResultRequirement,
  TaskResultProvenance,
  sha256_digest,
)

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (  # noqa: E402
  AgentRunner,
  AgentSessionLog,
  EventLog,
  GatewaySession,
  ModelInfo,
  ModelProvider,
  SessionContextBuilder,
  TaskRegistry,
  TaskState,
  ToolDispatcher,
)
from agent_gateway.providers import StreamEvent  # noqa: E402
from agent_gateway.capability_execution import BoundCapabilityExecution  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
from tests.capability_execution_test_support import (  # noqa: E402
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
) -> dict[str, object]:
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
  return {
    "logical_task": logical_task,
    "attempt": attempt,
    "result_requirement": ResultRequirement(
      mode="narrative",
      terminal_narrative="required",
      outcome=OutcomeRequirement(required=False, source="none"),
    ),
    "result_provenance": TaskResultProvenance(
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
  }


def _runner_execution(
  provider: ModelProvider,
  *,
  model: str = "claude-sonnet-4-6",
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
  model: str = "claude-sonnet-4-6",
) -> BoundCapabilityExecution:
  return _runner_execution(
    provider,
    model=model,
    capability_id="node.implement",
  )


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


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
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers=local_tool_handlers or {},
    event_log=event_log or EventLog(),
    session_id="sess-parent",
    # Owner authority: since S1 fail-closed roles a role-less dispatcher is
    # invite and denies the synthetic local tools these logging tests execute.
    role="owner",
    get_tool_definitions=None,
  )


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
  runner = AgentRunner(
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
  assert entries[1].event["content"] == "Run lookup"
  assert entries[2].event["stop_reason"] == "tool_use"
  assert entries[2].event["usage"]["input_tokens"] == 10
  assert entries[3].event["parent_assistant_message_seq"] == entries[2].seq
  assert "final_tool_result_blocks" in entries[4].event
  assert entries[6].event["reason"] == "completed"


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
  runner = AgentRunner(
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
  runner = AgentRunner(
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
  runner = AgentRunner(
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
  runner = AgentRunner(
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
  runner = AgentRunner(
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
  event_types = [entry.event["type"] for entry in entries]
  assert event_types == ["attach", "user_message", "interrupted", "detach"]
  assert entries[2].event["reason"] == "operator_pause"
  assert entries[2].event["safe_boundary"] == "before_turn"
  assert entries[3].event["reason"] == "operator_pause"
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
  runner = AgentRunner(
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
  event_types = [entry.event["type"] for entry in entries]
  assert event_types == ["attach", "user_message", "assistant_message", "interrupted", "detach"]
  assert entries[3].event["reason"] == "operator_pause"
  assert entries[3].event["safe_boundary"] == "after_turn_before_tools"
  assert entries[4].event["reason"] == "operator_pause"
  assert "tool_call_start" not in event_types
  assert "tool_call_interrupted" not in event_types


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
  runner = AgentRunner(
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
  assert [event["type"] for event in events] == [
    "attach",
    "user_message",
    "error",
    "detach",
  ]
  assert events[2]["error"] == (
    "Provider startup failed: could not create client for provider=stub."
  )
  assert events[3]["reason"] == "error"
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
  runner = AgentRunner(
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
  runner = AgentRunner(
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
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    context_builder=SessionContextBuilder(agent_session_log=log, tail_window_seconds=None),
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
  runner = AgentRunner(
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


def test_tool_call_complete_logs_final_model_facing_result_blocks(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "final-tool-result.jsonl")
  provider = _ScriptedProvider([
    _tool_turn(tool_name="warn_lookup", tool_input={"warning": "only 20% matched"}),
    _text_turn("done"),
  ])
  event_log = EventLog()
  runner = AgentRunner(
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
  runner = AgentRunner(
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
  runner = AgentRunner(
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
  runner = AgentRunner(
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
  assert parsed["scalar_fields"]["status"] == "success"
  assert parsed["scalar_fields"]["ticker"] == "BIG"
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
  runner = AgentRunner(
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
  ).bind.receipt()
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


def test_runner_stale_recovery_synthesizes_orphan_tool_and_prior_writer_interrupt(tmp_path: Path) -> None:
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

  provider = _ScriptedProvider([_text_turn("recovered")])
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_make_dispatcher(local_tool_handlers={}),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "Continue"}], max_turns=1))

  entries, _ = _run(log.query(order="asc"))
  event_types = [entry.event["type"] for entry in entries]
  assert event_types[:6] == [
    "attach",
    "tool_call_start",
    "tool_call_interrupted",
    "interrupted",
    "attach",
    "user_message",
  ]

  orphan = entries[2].event
  recovery = entries[3].event
  new_attach = entries[4].event
  assert orphan["tool_call_id"] == "tool-orphan"
  assert orphan["tool_risk"] == "read_only"
  assert recovery["reason"] == "recovered_on_attach"
  assert recovery["runner_id"] == "runner_old"
  assert recovery["recovered_by_runner_id"] == new_attach["runner_id"]
  assert new_attach["runner_id"] != "runner_old"


def test_second_writer_lease_acquisition_raises(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "lease.jsonl")
  provider_one = _ScriptedProvider([_text_turn("one")])
  runner_one = AgentRunner(
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
  runner_two = AgentRunner(
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
    runner = AgentRunner(
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
    ).bind.receipt()
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


def test_task_completed_is_durable_before_terminal_transition() -> None:
  async def _case() -> None:
    provider = _ScriptedProvider([_text_turn("unused")])
    runner = AgentRunner(
      event_log=EventLog(),
      dispatcher=_make_dispatcher(),
      session_id="sess-parent",
      capability_execution=_runner_execution(provider),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )
    entry = runner._task_registry.register("background_agent")
    capability_bind = _child_execution(provider).bind.receipt()
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

    runner._append_durable_event = _append  # type: ignore[method-assign]
    runner._task_registry.transition = _transition  # type: ignore[method-assign]

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      return _child_report(), None

    await runner._run_background_agent(entry, _handler, {}, 0)

    assert ordering == ["append_task_completed", "transition_completed"]

  _run(_case())


def test_rebuild_hot_set_lazy_lookup_and_seq_restore(tmp_path: Path) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "rebuild.jsonl")
    provider = _ScriptedProvider([_text_turn("unused")])
    capability_bind = _child_execution(provider).bind.receipt()
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
    runner = AgentRunner(
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
    assert [task["task_id"] for task in all_result["tasks"]] == ["bg_2", "bg_3"]

    lazy_result, lazy_error = await runner.get_background_result({"task_id": "bg_1"})
    assert lazy_error is None
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
      "capability_bind": _child_execution(provider).bind.receipt(),
    }
    await log.append({"type": "task_registered", **base, "task_id": "bg_0", "sub_agent_id": "sub0:sess-parent", "call_index": 0, "agent_name": "running", "started_at": 100.0})
    await log.append({"type": "task_registered", **base, "task_id": "bg_1", "sub_agent_id": "sub1:sess-parent", "call_index": 1, "agent_name": "done", "started_at": 110.0})
    await log.append({"type": "task_completed", **base, "task_id": "bg_1", "sub_agent_id": "sub1:sess-parent", "call_index": 1, "final_state": "completed", "completed_at": 120.0, "result": _child_report(), "error": None})
    await log.append({"type": "task_registered", **base, "task_id": "bg_2", "sub_agent_id": "sub2:sess-parent", "call_index": 2, "agent_name": "killed", "started_at": 130.0})

    runner = AgentRunner(
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
    assert interrupted["status"] == "interrupted"
    assert interrupted["completed"] is True
    assert interrupted["agent"] == "running"
    assert interrupted["started_at"] == 100.0
    assert interrupted["sub_agent_id"] == "sub0:sess-parent"
    assert interrupted["parent_turn_id"] == "turn-1"
    assert completed_error is None
    assert completed["status"] == "completed"
    assert completed["report"]["summary"] == "done"
    assert killed_error is None
    assert killed_as_interrupted["status"] == "interrupted"

  _run(_case())


def test_sub_agent_events_are_written_to_parent_log(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "sub-agent.jsonl")
  provider = _ScriptedProvider([
    _tool_turn(tool_id="tool-sub", tool_input={"query": "MSFT"}),
    _text_turn("sub done"),
  ])
  parent_runner = AgentRunner(
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
  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect background context",
      capability_execution=_child_execution(parent_runner._provider),
      **_subagent_result_identity(
        delegation_id="test-sub-agent-events",
        physical_task_id="sub0:sess-parent",
      ),
      skill_name="test-child",
      dispatcher=_make_dispatcher(
        local_tool_handlers={"lookup": _lookup_tool},
      ),
      sub_session=sub_session,
      max_turns=5,
      timeout=5.0,
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
  parent_runner = AgentRunner(
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
  result, error = _run(
    parent_runner.spawn_sub_agent(
      "Collect background context",
      capability_execution=_child_execution(parent_runner._provider),
      **_subagent_result_identity(
        delegation_id="test-shared-sub-agent-id",
        physical_task_id="sub3:derived-parent",
      ),
      skill_name="test-child",
      dispatcher=_make_dispatcher(),
      sub_session=None,
      max_turns=5,
      timeout=5.0,
      call_index=3,
    )
  )

  assert error is None
  assert result is not None
  assert calls == [("sess-parent", 3)]
  entries, _ = _run(log.query(role="sub_agent", order="asc"))
  assert {entry.event["sub_agent_id"] for entry in entries} == {"sub3:derived-parent"}
