import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_workflow_contracts import (  # noqa: E402
  ActivityHandle,
  AgentCompletionEnvelope,
  AgentOperationRef,
  AttemptRef,
  ContentHandle,
  ContentReadGrant,
  ContractRef,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  TranscriptHandle,
  sha256_digest,
)
from agent_gateway import AgentRunner, AgentSessionLog, EventLog, SessionContextBuilder, TaskState  # noqa: E402
import agent_gateway.runner as gateway_runner  # noqa: E402
import agent_gateway.runner_tool_execution as runner_tool_execution  # noqa: E402
from agent_gateway.runner_background_tasks import (  # noqa: E402
  _BACKGROUND_RESULT_ACK_RESULT_KEY,
  build_agent_completion_envelope,
  ordinary_parent_result_policy,
)
from agent_gateway.runner_tool_execution import RunnerToolExecutionMixin  # noqa: E402
from tests.capability_execution_test_support import (  # noqa: E402
  stub_runner_capability_execution,
)


class _Provider:
  name = "stub"

  def get_model_info(self, model: str) -> SimpleNamespace:
    return SimpleNamespace(model_id=model, context_window=200_000, max_output_tokens=8192)

  def estimate_cost(self, model: str, uncached: int, cache_read: int, cache_write: int, output: int) -> SimpleNamespace:
    _ = model, uncached, cache_read, cache_write, output
    return SimpleNamespace(total_usd=0.0, breakdown={})


def _capability_execution():
  return stub_runner_capability_execution(
    provider=_Provider(),
    model="stub-model",
    effort="none",
    auth_config={"api_key": "k"},
  )


class _Dispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, call_index
    return {"status": "ok", "echo": dict(tool_input)}, None


class _SecretResultDispatcher:
  def __init__(self, secret: str) -> None:
    self.secret = secret
    self.inputs: list[dict[str, Any]] = []

  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, call_index
    self.inputs.append(dict(tool_input))
    return {"status": "ok", "credential": self.secret}, None


class _TaskResultDispatcher:
  def __init__(self, result: Any) -> None:
    self.result = result

  async def dispatch(
    self,
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    call_index: int = 0,
  ):
    _ = tool_id, tool_input, call_index
    assert tool_name == "run_agent"
    return self.result, None


class _UiBlocksDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, call_index
    assert tool_name == "emit_ui_blocks"
    if tool_input.get("valid") is False:
      return {"validation_failed": {"failures": [{"code": "unknown_block"}]}}, None
    return {"accepted": {"ui_blocks_id": "ub_test", "emission_index": 0}}, None


class _ExplodingDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_input, call_index
    raise AssertionError(f"dispatch should not be called for excluded tool {tool_name}")


class _UnavailableDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, tool_input, call_index
    return None, {
      "code": "tool_unavailable",
      "message": "Tool 'get_quote' is not currently available. Load market-data first.",
      "data": {
        "deferred_tool": True,
        "required_tool_pack": "market-data",
        "required_tool_packs": ["market-data"],
        "next_tool": "load_tools",
        "suggested_call": {
          "tool": "load_tools",
          "args": {"pack": "market-data"},
        },
      },
    }


class _HintedErrorDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, call_index
    assert tool_name == "get_price_target"
    assert tool_input == {"ticker": "MSCI"}
    return None, {
      "code": "mcp_tool_error",
      "sub_code": "unknown",
      "message": "ticker\n Unexpected keyword argument [type=unexpected_keyword_argument]",
      "tool_usage_hint": "Pass research_file_id; do not pass ticker.",
    }


class _ApprovalTimeoutDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, tool_input, call_index
    return None, {
      "code": "approval_timeout",
      "message": "User did not respond within timeout",
    }


class _ReadableResourceDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, tool_input, call_index
    content = "## Daily note\n\nCaptured markdown.\n"
    content_bytes = content.encode("utf-8")
    digest = hashlib.sha256(content_bytes).hexdigest()
    return {
      "written": True,
      "file": "daily/2026-06-12.md",
      "mode": "append",
      "indexed_chunks": 1,
      "_readable_resource_snapshot": {
        "contract_name": "MarkdownNote",
        "content_type": "text/markdown",
        "content_class": "human_readable",
        "title": "daily/2026-06-12.md",
        "source_path": "daily/2026-06-12.md",
        "content_snapshot_id": f"sha256:{digest}",
        "content_sha256": digest,
        "content_bytes": len(content_bytes),
        "content": content,
        "truncated": False,
        "byte_start": 128,
        "byte_end": 128 + len(content_bytes),
        "tool_name": "memory_write",
      },
    }, None


class _BackgroundResultDispatcher:
  def __init__(self, result: dict[str, Any]) -> None:
    self.result = result

  async def dispatch(
    self,
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    call_index: int = 0,
  ):
    _ = tool_id, tool_input, call_index
    assert tool_name == "get_background_result"
    return dict(self.result), None


class _WorkflowResultDispatcher:
  def __init__(self, result: dict[str, Any]) -> None:
    self.result = result

  async def dispatch(
    self,
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    call_index: int = 0,
  ):
    _ = tool_id, tool_input, call_index
    assert tool_name == "workflow_run"
    return dict(self.result), None


def _run(coro):
  return asyncio.run(coro)


def _canonical_narrative_task_result() -> TaskResult:
  operation = AgentOperationRef(
    namespace="research",
    name="explore",
    version="1.0",
    digest=sha256_digest({"operation": "research.explore/1.0"}),
  )
  digest = sha256_digest({"fixture": "foreground-task-result"})
  text = "Exact foreground terminal message"
  encoded = text.encode("utf-8")
  content_sha256 = hashlib.sha256(encoded).hexdigest()
  content = ContentHandle(
    content_id=f"sha256:{content_sha256}",
    content_sha256=content_sha256,
    content_bytes=len(encoded),
    content_chars=len(text),
    contract=ContractRef(
      namespace="agent-result",
      name="terminal-assistant-message",
      version="1.0",
      digest=sha256_digest({"contract": "terminal-assistant-message"}),
    ),
    media_type="text/plain",
    encoding="utf-8",
    retention="durable",
  )
  return TaskResult(
    task_result_id="foreground-result",
    logical_task=OrdinaryDelegationTaskRef(
      delegation_id="foreground-delegation",
      operation=operation,
    ),
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id="foreground-attempt",
      physical_task_id="foreground-child",
    ),
    execution=ExecutionSettlement(status="succeeded"),
    values=TaskResultValues(terminal_narrative=content),
    observation=TaskObservation(
      transcript=TranscriptHandle(
        kind="child_transcript",
        owner_id="foreground-child",
      ),
      activity=ActivityHandle(
        kind="child_activity",
        owner_id="foreground-child",
      ),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=digest,
      model_bind_digest=digest,
      capability_binding_digest=digest,
      tool_grant_digest=digest,
    ),
  )


def _runner() -> AgentRunner:
  return AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_Dispatcher(),  # type: ignore[arg-type]
    session_id="test-tool-execution",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_tool_secret_is_absent_from_model_event_durable_and_replay_boundaries(
  tmp_path: Path,
) -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-CODEX-WAVE0-8f21d7"
  session_log = AgentSessionLog(path=tmp_path / "sessions" / "secret-boundary.jsonl")
  execution = stub_runner_capability_execution(
    provider=_Provider(),
    model="stub-model",
    effort="none",
    auth_config={"api_key": secret},
  )
  dispatcher = _SecretResultDispatcher(secret)
  runner = AgentRunner(
    event_log=EventLog(session_id="secret-boundary"),
    dispatcher=dispatcher,  # type: ignore[arg-type]
    session_id="secret-boundary",
    capability_execution=execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    agent_session_log=session_log,
  )
  runner._runner_id = "writer-secret-boundary"

  live_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool(
      "tool-secret",
      "lookup",
      {"query": "ordinary", "credential": secret},
      {"tools": []},
    )
  )
  durable, _ = _run(session_log.query(order="asc"))
  replay = _run(SessionContextBuilder(agent_session_log=session_log).build())

  assert dispatcher.inputs == [{"query": "ordinary", "credential": secret}]
  assert secret not in json.dumps(live_entry)
  assert secret not in json.dumps([entry.event for entry in runner._log.entries])
  assert secret not in json.dumps([entry.event for entry in durable])
  assert secret not in json.dumps(replay)
  assert "<redacted-secret>" in json.dumps(live_entry)


def test_unhandled_tool_exception_is_raw_for_hook_but_sanitized_at_all_boundaries(
  tmp_path: Path,
) -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-TOOL-EXCEPTION-8f21d7"
  session_log = AgentSessionLog(path=tmp_path / "sessions" / "secret-exception.jsonl")
  execution = stub_runner_capability_execution(
    provider=_Provider(),
    model="stub-model",
    effort="none",
    auth_config={"api_key": secret},
  )
  hook_errors: list[Any] = []

  class _SecretExceptionDispatcher:
    async def dispatch(
      self,
      tool_id: str,
      tool_name: str,
      tool_input: dict[str, Any],
      *,
      call_index: int = 0,
    ):
      _ = tool_id, tool_name, tool_input, call_index
      raise RuntimeError(f"privileged failure: {secret}")

  async def _capture_raw_hook(ctx: Any) -> list[dict[str, Any]]:
    hook_errors.append(ctx.error)
    return [{"type": "text", "text": "ordinary hook note", "api_key_set": True}]

  runner = AgentRunner(
    event_log=EventLog(session_id="secret-exception"),
    dispatcher=_SecretExceptionDispatcher(),  # type: ignore[arg-type]
    session_id="secret-exception",
    capability_execution=execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    agent_session_log=session_log,
    on_tool_result=_capture_raw_hook,
  )
  runner._runner_id = "writer-secret-exception"

  live_entry, _tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-secret-exception",
      "lookup",
      {"query": "ordinary"},
      {"tools": []},
    )
  )
  durable, _ = _run(session_log.query(order="asc"))
  replay = _run(SessionContextBuilder(agent_session_log=session_log).build())

  assert secret in json.dumps(hook_errors)
  assert extra_blocks == [
    {"type": "text", "text": "ordinary hook note", "api_key_set": True}
  ]
  for projection in (
    live_entry,
    [entry.event for entry in runner._log.entries],
    [entry.event for entry in durable],
    replay,
  ):
    serialized = json.dumps(projection)
    assert secret not in serialized
    assert "<redacted-secret>" in serialized


def _workflow_result(delivery_status: str) -> dict[str, Any]:
  output_id = f"sha256:{'b' * 64}"
  reference = {
    "kind": "workflow_phase_output",
    "output_id": output_id,
    "content_sha256": "a" * 64,
    "content_chars": 100,
    "content_bytes": 100,
    "encoding": "canonical-json",
    "read": {
      "action": "output",
      "workflow_run_id": "workflow-1",
      "output_id": output_id,
    },
  }
  return {
    "ok": True,
    "action": "result",
    "workflow_run_id": "workflow-1",
    "delivery_status": delivery_status,
    "delivery_contract": {
      "primary_output": "synthesis",
      "presentation_mode": "summary_with_primary_attachment",
    },
    "delivery_phase_number": 1,
    "primary_output_reference": dict(reference),
    "output_manifest": {"synthesis": dict(reference)},
    "missing_terminal_outputs": (
      [] if delivery_status == "complete" else ["synthesis"]
    ),
  }


def test_runner_tool_execution_method_is_inherited_from_mixin() -> None:
  assert issubclass(AgentRunner, RunnerToolExecutionMixin)
  assert gateway_runner.RunnerToolExecutionMixin is RunnerToolExecutionMixin
  assert AgentRunner._execute_single_tool is RunnerToolExecutionMixin._execute_single_tool


def test_incomplete_workflow_result_does_not_stage_final_attachment() -> None:
  runner = _runner()
  runner._dispatcher = _WorkflowResultDispatcher(
    _workflow_result("incomplete")
  )

  _run(runner._execute_single_tool(
    "workflow-result-incomplete",
    "workflow_run",
    {"action": "result", "workflow_run_id": "workflow-1"},
    {"tools": []},
  ))

  assert runner._pending_workflow_output_attachments == {}


def test_malformed_complete_workflow_result_fails_tool_call_closed() -> None:
  result = _workflow_result("complete")
  result["primary_output_reference"]["content_sha256"] = "c" * 64
  runner = _runner()
  runner._dispatcher = _WorkflowResultDispatcher(result)

  _run(runner._execute_single_tool(
    "workflow-result-invalid",
    "workflow_run",
    {"action": "result", "workflow_run_id": "workflow-1"},
    {"tools": []},
  ))

  completion = next(
    entry.event
    for entry in runner._log.entries
    if entry.event.get("type") == "tool_call_complete"
  )
  assert completion["result"] is None
  assert completion["error"]["code"] == (
    "workflow_output_attachment_invalid"
  )
  assert runner._pending_workflow_output_attachments == {}


def test_execute_single_tool_resolves_parent_module_helpers(monkeypatch: Any) -> None:
  start_calls: list[dict[str, Any]] = []
  complete_calls: list[dict[str, Any]] = []
  semantic_calls: list[Any] = []

  def _redact(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    assert tool_name == "file_read"
    assert tool_input == {"path": "secret.txt"}
    return {"path": "[patched]"}

  def _start_event(**kwargs: Any) -> dict[str, Any]:
    start_calls.append(kwargs)
    return {
      "type": "patched_start",
      "tool_call_id": kwargs["tool_call_id"],
      "tool_name": kwargs["tool_name"],
      "tool_input": kwargs["tool_input"],
    }

  def _complete_event(**kwargs: Any) -> dict[str, Any]:
    complete_calls.append(kwargs)
    return {
      "type": "patched_complete",
      "tool_call_id": kwargs["tool_call_id"],
      "tool_name": kwargs["tool_name"],
      "result": kwargs["result"],
      "error": kwargs["error"],
      "is_error": kwargs["error"] is not None,
    }

  def _semantic(result: Any) -> None:
    semantic_calls.append(result)
    return None

  monkeypatch.setattr(gateway_runner, "_redact_tool_input_for_event", _redact)
  monkeypatch.setattr(gateway_runner, "resolve_display", lambda name, tool_input: {"name": name, "input": tool_input})
  monkeypatch.setattr(gateway_runner, "_build_tool_call_start_event", _start_event)
  monkeypatch.setattr(gateway_runner, "_build_tool_call_complete_event", _complete_event)
  monkeypatch.setattr(gateway_runner, "classify_semantic_tool_error", _semantic)

  runner = _runner()
  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-1",
      "file_read",
      {"path": "secret.txt"},
      {"tools": []},
      call_index=3,
    )
  )

  assert tool_name == "file_read"
  assert extra_blocks == []
  assert json.loads(live_entry["content"]) == {"status": "ok", "echo": {"path": "secret.txt"}}
  assert start_calls[0]["tool_input"] == {"path": "[patched]"}
  assert start_calls[0]["call_index"] == 3
  assert complete_calls[0]["result"] == {"status": "ok", "echo": {"path": "secret.txt"}}
  assert semantic_calls == [{"status": "ok", "echo": {"path": "secret.txt"}}]

  events = [entry.event for entry in runner._log.entries]
  assert events[0]["type"] == "patched_start"
  assert events[0]["display"] == {"name": "file_read", "input": {"path": "[patched]"}}
  assert events[-1]["type"] == "patched_complete"
  assert events[-1]["final_tool_result_blocks"][0]["content"] == live_entry["content"]


def test_execute_single_tool_emits_readable_resource_event_without_model_leak() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ReadableResourceDispatcher(),  # type: ignore[arg-type]
    session_id="control-run-readable",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    skill_run_id="skill-run-readable",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-readable",
      "memory_write",
      {"file": "daily/2026-06-12.md", "content": "redacted by event", "mode": "append"},
      {"tools": []},
    )
  )

  assert tool_name == "memory_write"
  assert extra_blocks == []
  model_payload = json.loads(live_entry["content"])
  assert model_payload == {
    "written": True,
    "file": "daily/2026-06-12.md",
    "mode": "append",
    "indexed_chunks": 1,
  }

  events = [entry.event for entry in runner._log.entries]
  complete = next(event for event in events if event.get("type") == "tool_call_complete")
  assert "_readable_resource_snapshot" not in complete["result"]
  resource = next(event for event in events if event.get("type") == "readable_resource_ready")
  assert resource["resource_id"].startswith("rr:")
  assert resource["control_run_id"] == "control-run-readable"
  assert resource["skill_run_id"] == "skill-run-readable"
  assert resource["source_path"] == "daily/2026-06-12.md"
  assert resource["content"] == "## Daily note\n\nCaptured markdown.\n"
  assert resource["content_sha256"] == hashlib.sha256(resource["content"].encode("utf-8")).hexdigest()


def test_background_result_queues_ack_only_after_durable_tool_result() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="background-result-ack"),
    dispatcher=_BackgroundResultDispatcher({}),  # type: ignore[arg-type]
    session_id="background-result-ack",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  entry = runner._task_registry.register("background_agent")
  runner._task_registry.transition(entry.task_id, TaskState.RUNNING)
  runner._task_registry.transition(
    entry.task_id,
    TaskState.COMPLETED,
    result={"status": "completed", "value": 42},
  )
  entry.notification_delivery_state = "queue_omitted"
  runner._dispatcher.result = {
    "status": "completed",
    "value": 42,
    _BACKGROUND_RESULT_ACK_RESULT_KEY: {
      "task_id": entry.task_id,
      "notification_generation": entry.notification_generation,
    },
  }

  live_entry, _, _ = _run(
    runner._execute_single_tool(
      "tool-background-result",
      "get_background_result",
      {"task_id": entry.task_id},
      {"tools": []},
    )
  )

  assert entry.notification_delivery_state == "queue_omitted"
  assert runner._pending_background_result_acks == {
    "tool-background-result": (
      entry.task_id,
      entry.notification_generation,
    )
  }
  assert json.loads(live_entry["content"]) == {
    "status": "completed",
    "value": 42,
  }
  complete = next(
    log_entry.event
    for log_entry in runner._log.entries
    if log_entry.event.get("type") == "tool_call_complete"
  )
  assert _BACKGROUND_RESULT_ACK_RESULT_KEY not in complete["result"]


def test_background_result_durable_failure_retains_omitted_payload() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="background-result-ack-failure"),
    dispatcher=_BackgroundResultDispatcher({}),  # type: ignore[arg-type]
    session_id="background-result-ack-failure",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  entry = runner._task_registry.register("background_agent")
  runner._task_registry.transition(entry.task_id, TaskState.RUNNING)
  runner._task_registry.transition(
    entry.task_id,
    TaskState.COMPLETED,
    result={"status": "completed", "value": 42},
  )
  entry.notification_delivery_state = "queue_omitted"
  runner._dispatcher.result = {
    "status": "completed",
    "value": 42,
    _BACKGROUND_RESULT_ACK_RESULT_KEY: {
      "task_id": entry.task_id,
      "notification_generation": entry.notification_generation,
    },
  }
  append_durable_event = runner._append_durable_event

  async def _fail_tool_result_persistence(event: dict[str, Any]) -> None:
    if event.get("type") == "tool_call_complete":
      raise RuntimeError("durable tool-result write failed")
    await append_durable_event(event)

  runner._append_durable_event = _fail_tool_result_persistence  # type: ignore[method-assign]

  with pytest.raises(
    RuntimeError,
    match="durable tool-result write failed",
  ):
    _run(
      runner._execute_single_tool(
        "tool-background-result-failure",
        "get_background_result",
        {"task_id": entry.task_id},
        {"tools": []},
      )
    )

  assert entry.notification_delivery_state == "queue_omitted"
  assert runner._pending_background_result_acks == {}


def test_background_result_ack_cannot_target_a_different_task() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="background-result-ack-forgery"),
    dispatcher=_BackgroundResultDispatcher({}),  # type: ignore[arg-type]
    session_id="background-result-ack-forgery",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  requested = runner._task_registry.register("background_agent")
  target = runner._task_registry.register("background_agent")
  for entry in (requested, target):
    runner._task_registry.transition(entry.task_id, TaskState.RUNNING)
    runner._task_registry.transition(
      entry.task_id,
      TaskState.COMPLETED,
      result={"status": "completed"},
    )
    entry.notification_delivery_state = "queue_omitted"
  runner._dispatcher.result = {
    "status": "completed",
    _BACKGROUND_RESULT_ACK_RESULT_KEY: {
      "task_id": target.task_id,
      "notification_generation": target.notification_generation,
    },
  }

  live_entry, _, _ = _run(
    runner._execute_single_tool(
      "tool-background-result-forgery",
      "get_background_result",
      {"task_id": requested.task_id},
      {"tools": []},
    )
  )

  assert requested.notification_delivery_state == "queue_omitted"
  assert target.notification_delivery_state == "queue_omitted"
  assert runner._pending_background_result_acks == {}
  assert _BACKGROUND_RESULT_ACK_RESULT_KEY not in json.loads(
    live_entry["content"]
  )


def test_background_result_generation_change_blocks_stale_ack() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="background-result-generation"),
    dispatcher=_BackgroundResultDispatcher({}),  # type: ignore[arg-type]
    session_id="background-result-generation",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  entry = runner._task_registry.register("background_agent")
  runner._task_registry.transition(entry.task_id, TaskState.RUNNING)
  runner._task_registry.transition(
    entry.task_id,
    TaskState.INTERRUPTED,
    error={"code": "uncertain", "message": "interrupted"},
  )
  entry.notification_delivery_state = "queue_omitted"
  interrupted_generation = entry.notification_generation
  runner._dispatcher.result = {
    "status": "interrupted",
    _BACKGROUND_RESULT_ACK_RESULT_KEY: {
      "task_id": entry.task_id,
      "notification_generation": interrupted_generation,
    },
  }
  append_durable_event = runner._append_durable_event

  async def _finalize_during_tool_result_persistence(
    event: dict[str, Any],
  ) -> None:
    if event.get("type") == "tool_call_complete":
      runner._task_registry.finalize_interrupted(
        entry.task_id,
        TaskState.COMPLETED,
        result={
          "status": "completed",
          "value": "new",
          "blob": "x" * 40_000,
        },
      )
    await append_durable_event(event)

  runner._append_durable_event = (  # type: ignore[method-assign]
    _finalize_during_tool_result_persistence
  )

  _run(
    runner._execute_single_tool(
      "tool-background-result-generation",
      "get_background_result",
      {"task_id": entry.task_id},
      {"tools": []},
    )
  )

  assert entry.notification_generation == interrupted_generation + 1
  assert entry.notification_delivery_state in {
    "payload_omitted",
    "queue_omitted",
  }
  assert runner._pending_background_result_acks == {
    "tool-background-result-generation": (
      entry.task_id,
      interrupted_generation,
    )
  }


def test_execute_single_tool_marks_accepted_ui_blocks_terminal() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_UiBlocksDispatcher(),  # type: ignore[arg-type]
    session_id="test-accepted-ui-blocks",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-ui", "emit_ui_blocks", {"blocks": []}, {"tools": []})
  )

  assert tool_name == "emit_ui_blocks"
  assert extra_blocks == []
  assert json.loads(live_entry["content"]) == {
    "accepted": {"ui_blocks_id": "ub_test", "emission_index": 0}
  }
  assert runner._stop_after_tool_results_reason == "accepted_ui_blocks"
  assert runner._stop_after_tool_results_tool_name == "emit_ui_blocks"


def test_execute_single_tool_keeps_ui_blocks_validation_failure_recoverable() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_UiBlocksDispatcher(),  # type: ignore[arg-type]
    session_id="test-invalid-ui-blocks",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool(
      "tool-ui-invalid",
      "emit_ui_blocks",
      {"blocks": [], "valid": False},
      {"tools": []},
    )
  )

  assert "validation_failed" in json.loads(live_entry["content"])
  assert not hasattr(runner, "_stop_after_tool_results_reason")


def test_foreground_run_agent_emits_canonical_task_result_json() -> None:
  task_result = _canonical_narrative_task_result()
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_TaskResultDispatcher(task_result),  # type: ignore[arg-type]
    session_id="test-foreground-task-result",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-run-agent",
      "run_agent",
      {"operation": {"name": "explore"}, "objective": "Research"},
      {"tools": []},
    )
  )

  assert tool_name == "run_agent"
  assert extra_blocks == []
  payload = json.loads(live_entry["content"])
  assert payload == task_result.model_dump(mode="json")
  assert payload["schema_version"] == "2.0"
  assert payload["values"]["terminal_narrative"]["content_chars"] == 33
  assert "TaskResult(" not in live_entry["content"]


def test_execute_single_tool_keeps_generic_excluded_tools_bare() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-generic-excluded",
    capability_execution=_capability_execution(),
    excluded_tools={"run_agent"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-1", "run_agent", {"task": "x"}, {"tools": []})
  )

  assert tool_name == "run_agent"
  assert extra_blocks == []
  payload = json.loads(live_entry["content"])
  assert payload["error"] == {
    "code": "tool_excluded",
    "message": "Tool 'run_agent' is not available in this context",
  }


def test_execute_single_tool_returns_typed_blocker_for_output_file_gated_tool(monkeypatch: Any) -> None:
  monkeypatch.setattr(
    runner_tool_execution,
    "_output_file_gated_tool_names",
    lambda: frozenset({"analyze_stock"}),
  )
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-output-file-gated-excluded",
    capability_execution=_capability_execution(),
    excluded_tools={"analyze_stock"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-1", "analyze_stock", {"ticker": "MSCI"}, {"tools": []})
  )

  assert tool_name == "analyze_stock"
  assert extra_blocks == []
  payload = json.loads(live_entry["content"])
  error = payload["error"]
  assert error["code"] == "tool_excluded"
  assert error["sub_code"] == "output_file_gated_tool_excluded"
  assert "output='file'" in error["message"]
  assert error["data"]["output_file_gated"] is True
  assert error["data"]["suggested_tools"] == ["get_quote", "industry_peer_comparison"]
  assert "quantifying-risk" in error["data"]["resolution"]


def test_execute_single_tool_preserves_dispatcher_error_data_for_model_result() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_UnavailableDispatcher(),  # type: ignore[arg-type]
    session_id="test-unavailable-data",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-1", "get_quote", {}, {"tools": []})
  )

  assert tool_name == "get_quote"
  assert extra_blocks == []
  error = json.loads(live_entry["content"])["error"]
  assert error["code"] == "tool_unavailable"
  assert error["data"] == {
    "deferred_tool": True,
    "required_tool_pack": "market-data",
    "required_tool_packs": ["market-data"],
    "next_tool": "load_tools",
    "suggested_call": {
      "tool": "load_tools",
      "args": {"pack": "market-data"},
    },
  }

  complete_events = [entry.event for entry in runner._log.entries if entry.event.get("type") == "tool_call_complete"]
  assert complete_events
  assert complete_events[-1]["final_tool_result_blocks"][0]["content"] == live_entry["content"]


def test_execute_single_tool_exposes_top_level_tool_usage_hint_in_model_error_data() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_HintedErrorDispatcher(),  # type: ignore[arg-type]
    session_id="test-error-hint",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool("tool-1", "get_price_target", {"ticker": "MSCI"}, {"tools": []})
  )

  assert tool_name == "get_price_target"
  assert extra_blocks == []
  error = json.loads(live_entry["content"])["error"]
  assert error["code"] == "mcp_tool_error"
  assert error["data"] == {"tool_usage_hint": "Pass research_file_id; do not pass ticker."}

  complete_events = [entry.event for entry in runner._log.entries if entry.event.get("type") == "tool_call_complete"]
  assert complete_events
  assert complete_events[-1]["error"]["tool_usage_hint"] == "Pass research_file_id; do not pass ticker."
  assert complete_events[-1]["error"]["data"] == {"tool_usage_hint": "Pass research_file_id; do not pass ticker."}
  assert complete_events[-1]["final_tool_result_blocks"][0]["content"] == live_entry["content"]


def test_execute_single_tool_stops_after_repeated_generic_excluded_tool() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-repeated-generic-excluded",
    capability_execution=_capability_execution(),
    excluded_tools={"apply_patch_ops"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  first_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-1", "apply_patch_ops", {"ops": []}, {"tools": []})
  )
  assert json.loads(first_entry["content"])["error"] == {
    "code": "tool_excluded",
    "message": "Tool 'apply_patch_ops' is not available in this context",
  }
  assert not hasattr(runner, "_stop_after_tool_results_reason")

  second_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-2", "apply_patch_ops", {"ops": []}, {"tools": []})
  )

  error = json.loads(second_entry["content"])["error"]
  assert error["code"] == "tool_excluded"
  assert error["sub_code"] == "repeated_tool_excluded"
  assert error["data"]["blocked_tool"] == "apply_patch_ops"
  assert error["data"]["exclusion_count"] == 2
  assert error["data"]["stop_after_tool_results"] is True
  assert runner._stop_after_tool_results_reason == "repeated_tool_excluded"
  assert runner._stop_after_tool_results_tool_name == "apply_patch_ops"


def test_execute_single_tool_stops_after_expired_approval() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ApprovalTimeoutDispatcher(),  # type: ignore[arg-type]
    session_id="test-approval-timeout",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-1", "file_write", {"path": "x"}, {"tools": []})
  )

  assert tool_name == "file_write"
  error = json.loads(live_entry["content"])["error"]
  assert error["code"] == "approval_timeout"
  assert runner._stop_after_tool_results_reason == "approval_timeout"
  assert runner._stop_after_tool_results_tool_name == "file_write"


def test_execute_single_tool_returns_typed_blocker_for_excluded_fms_commit_tool() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-fms-commit-excluded",
    capability_execution=_capability_execution(),
    excluded_tools={"fms_persist_business_model"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-1",
      "fms_persist_business_model",
      {"judgment": {"ticker": "PCTY"}},
      {"tools": []},
    )
  )

  assert tool_name == "fms_persist_business_model"
  assert extra_blocks == []
  payload = json.loads(live_entry["content"])
  error = payload["error"]
  assert error["code"] == "tool_excluded"
  assert error["sub_code"] == "requires_interactive_approval"
  assert "BUILD_BLOCKED" in error["message"]
  assert error["data"]["recommended_verdict"] == "BUILD_BLOCKED"
  assert error["data"]["pending_action"] == {
    "code": "persist_business_model",
    "stage": "bm",
    "message": "Run fms_persist_business_model interactively with operator approval, then retry the workflow.",
    "severity": "blocking",
    "target": "fms_persist_business_model",
    "source": "runner_tool_exclusion",
    "metadata": {
      "blocked_tool": "fms_persist_business_model",
      "requires_interactive_approval": True,
      "tool_class": "state_write",
      "resolution": (
        "Run this commit tool in an interactive model-writer/thesis-writer "
        "session with operator approval, then retry the blocked workflow."
      ),
    },
  }

  complete_events = [entry.event for entry in runner._log.entries if entry.event.get("type") == "tool_call_complete"]
  assert complete_events
  assert complete_events[0]["error"]["data"] == error["data"]


def test_execute_single_tool_returns_typed_blocker_for_excluded_thesis_writer_tool() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-fms-thesis-excluded",
    capability_execution=_capability_execution(),
    excluded_tools={"fms_report_thesis_consultation"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-1", "fms_report_thesis_consultation", {}, {"tools": []})
  )

  error = json.loads(live_entry["content"])["error"]
  assert error["sub_code"] == "requires_interactive_approval"
  assert error["data"]["pending_action"]["code"] == "report_thesis_consultation"
  assert error["data"]["pending_action"]["stage"] == "diligence"


def test_execute_single_tool_keeps_apply_proposal_exclusions_generic() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_ExplodingDispatcher(),  # type: ignore[arg-type]
    session_id="test-apply-excluded",
    capability_execution=_capability_execution(),
    excluded_tools={"apply_patch_proposal"},
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool("tool-1", "apply_patch_proposal", {}, {"tools": []})
  )

  assert json.loads(live_entry["content"])["error"] == {
    "code": "tool_excluded",
    "message": "Tool 'apply_patch_proposal' is not available in this context",
  }


def _aggregate_workflow_result_payload() -> dict[str, Any]:
  from agent_workflow_contracts import (
    AdmittedPlanRef,
    ContinuationState,
    DeliverySettlement,
    TerminalPhaseRevision,
    WorkflowResult,
    WorkflowView,
  )

  return WorkflowResult(
    workflow_run_id="workflow-1",
    view=WorkflowView(
      workflow_run_id="workflow-1",
      workflow_name="dynamic-workflow",
      state="terminal",
      execution_status="succeeded",
      delivery_status="not_required",
      terminal_status="succeeded",
      legal_actions=(),
      observation_seq=1,
      max_phases=2,
      admitted_plan_ref=AdmittedPlanRef(
        workflow_run_id="workflow-1",
        plan_id="plan-1",
        phase_number=1,
        revision=1,
        digest=f"sha256:{'a' * 64}",
      ),
      terminal_phase_revision=TerminalPhaseRevision(phase_number=1, revision=1),
      estimated_cost_usd=0.0,
      admitted_cost_estimate_usd=0.0,
      author_cost_usd=0.0,
    ),
    delivery=DeliverySettlement(status="not_required"),
    transcript=TranscriptHandle(kind="workflow_transcript", owner_id="workflow-1"),
    activity=ActivityHandle(kind="workflow_activity", owner_id="workflow-1"),
    continuation_state=ContinuationState(status="exhausted"),
  ).model_dump(mode="json")


class _WorkflowEvidenceDispatcher:
  async def dispatch(self, tool_id: str, tool_name: str, tool_input: dict[str, Any], *, call_index: int = 0):
    _ = tool_id, tool_name, tool_input, call_index
    return {
      "ok": True,
      "action": "result",
      **_aggregate_workflow_result_payload(),
      "_workflow_evidence_projection": {
        "workflow_run_id": "workflow-1",
        "evidence_tools": ["filings_search", "get_financials"],
        "observed_sources": [
          {"source_kind": "filing", "document_id": "edgar:1"},
        ],
      },
    }, None


def test_workflow_evidence_projection_does_not_leak_to_model_or_durable_event() -> None:
  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_WorkflowEvidenceDispatcher(),  # type: ignore[arg-type]
    session_id="control-run-evidence",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, tool_name, extra_blocks = _run(
    runner._execute_single_tool(
      "tool-workflow",
      "workflow_run",
      {"action": "result", "workflow_run_id": "workflow-1"},
      {"tools": []},
    )
  )

  assert tool_name == "workflow_run"
  assert extra_blocks == []
  model_payload = json.loads(live_entry["content"])
  assert "_workflow_evidence_projection" not in model_payload
  events = [entry.event for entry in runner._log.entries]
  complete = next(event for event in events if event.get("type") == "tool_call_complete")
  assert "_workflow_evidence_projection" not in complete["result"]


def test_workflow_evidence_projection_reaches_the_tool_result_hook() -> None:
  """The private key is popped before the hook runs, so it must ride the context."""

  seen: list[Any] = []

  async def _capture(ctx: Any) -> list[dict[str, Any]]:
    seen.append(ctx.child_evidence)
    return []

  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_WorkflowEvidenceDispatcher(),  # type: ignore[arg-type]
    session_id="control-run-evidence-context",
    capability_execution=_capability_execution(),
    on_tool_result=_capture,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(
    runner._execute_single_tool(
      "tool-workflow",
      "workflow_run",
      {"action": "result", "workflow_run_id": "workflow-1"},
      {"tools": []},
    )
  )

  assert seen == [
    {
      "workflow_run_id": "workflow-1",
      "evidence_tools": ["filings_search", "get_financials"],
      "observed_sources": [
        {"source_kind": "filing", "document_id": "edgar:1"},
      ],
    },
  ]


def _foreground_completion_envelope(*, with_evidence: bool) -> AgentCompletionEnvelope:
  payload = _canonical_narrative_task_result().model_dump(mode="json")
  if with_evidence:
    payload["evidence"] = {
      "observed_sources": [
        {
          "kind": "observed_source",
          "source_kind": "filing",
          "document_id": "edgar:0000789019-26-000012",
          "produced_by_tool": "filings_read",
          "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
        }
      ],
      "tools_used": ["filings_read"],
    }
  result = TaskResult.model_validate(payload)
  return build_agent_completion_envelope(
    result,
    policy=ordinary_parent_result_policy(result),
    terminal_narrative_reader=lambda _result: "Exact foreground terminal message",
    read_grant_factory=lambda source: ContentReadGrant(
      grant_id=f"content-read:{source.content_sha256}",
      content_id=source.content_id,
      scope="direct_parent",
      principal_id="parent-runner",
    ),
  )


def test_foreground_completion_child_evidence_is_stripped_and_routed_to_the_hook() -> None:
  envelope = _foreground_completion_envelope(with_evidence=True)
  assert envelope.child_evidence is not None

  seen: list[Any] = []

  async def _capture(ctx: Any) -> list[dict[str, Any]]:
    seen.append(ctx.child_evidence)
    return []

  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_TaskResultDispatcher(envelope),  # type: ignore[arg-type]
    session_id="test-foreground-child-evidence",
    capability_execution=_capability_execution(),
    on_tool_result=_capture,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool(
      "tool-run-agent",
      "run_agent",
      {"operation": {"name": "explore"}, "objective": "Research"},
      {"tools": []},
    )
  )

  model_payload = json.loads(live_entry["content"])
  assert "child_evidence" not in model_payload
  events = [entry.event for entry in runner._log.entries]
  complete = next(event for event in events if event.get("type") == "tool_call_complete")
  assert "child_evidence" not in complete["result"]
  assert seen == [
    {
      "evidence_tools": ["filings_read"],
      "observed_sources": [
        {
          "kind": "observed_source",
          "source_kind": "filing",
          "document_id": "edgar:0000789019-26-000012",
          "produced_by_tool": "filings_read",
          "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
          "excerpt_handle_id": None,
        }
      ],
    },
  ]


def test_foreground_completion_without_child_evidence_leaves_the_channel_empty() -> None:
  envelope = _foreground_completion_envelope(with_evidence=False)
  assert envelope.child_evidence is None

  seen: list[Any] = []

  async def _capture(ctx: Any) -> list[dict[str, Any]]:
    seen.append(ctx.child_evidence)
    return []

  runner = AgentRunner(
    event_log=EventLog(session_id="test"),
    dispatcher=_TaskResultDispatcher(envelope),  # type: ignore[arg-type]
    session_id="test-foreground-no-child-evidence",
    capability_execution=_capability_execution(),
    on_tool_result=_capture,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  live_entry, _tool_name, _extra_blocks = _run(
    runner._execute_single_tool(
      "tool-run-agent",
      "run_agent",
      {"operation": {"name": "explore"}, "objective": "Research"},
      {"tools": []},
    )
  )

  assert json.loads(live_entry["content"]) == envelope.model_dump(mode="json")
  assert seen == [None]


# --- B-1 / B-2: the dispatch record and the retry loop ---------------------


class _RecordedDispatcher:
  """Return a scripted sequence of (result, error) pairs, one per attempt."""

  def __init__(self, script: list[tuple[Any, dict[str, Any] | None]]) -> None:
    self.script = list(script)
    self.calls = 0

  async def dispatch(
    self,
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    call_index: int = 0,
  ):
    _ = tool_id, tool_name, tool_input, call_index
    index = min(self.calls, len(self.script) - 1)
    self.calls += 1
    return self.script[index]


class _HangingDispatcher:
  def __init__(self) -> None:
    self.calls = 0

  async def dispatch(
    self,
    tool_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    call_index: int = 0,
  ):
    _ = tool_id, tool_name, tool_input, call_index
    self.calls += 1
    await asyncio.sleep(10)
    raise AssertionError("unreachable")


def _dispatch_runner(dispatcher: Any, **kwargs: Any) -> AgentRunner:
  return AgentRunner(
    event_log=EventLog(session_id="dispatch"),
    dispatcher=dispatcher,  # type: ignore[arg-type]
    session_id="test-dispatch-record",
    capability_execution=_capability_execution(),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
    **kwargs,
  )


def _complete_event(runner: AgentRunner) -> dict[str, Any]:
  events = [
    entry.event
    for entry in runner._log.entries
    if entry.event.get("type") == "tool_call_complete"
  ]
  assert events
  return events[-1]


def _start_event_count(runner: AgentRunner) -> int:
  return len(
    [
      entry
      for entry in runner._log.entries
      if entry.event.get("type") == "tool_call_start"
    ]
  )


@pytest.fixture
def declared_read_effects(monkeypatch: pytest.MonkeyPatch):
  """Pin the derived `effect` column so retry eligibility is deterministic."""

  from agent_gateway import tool_dispatch_declarations as declarations

  monkeypatch.setattr(
    declarations,
    "_CACHED_DECLARATIONS",
    declarations.build_tool_dispatch_declarations(
      effect_resolver=lambda _tool_name: "read"
    ),
  )
  yield


def test_dispatch_record_is_emitted_on_the_ok_exit_path() -> None:
  runner = _dispatch_runner(
    _RecordedDispatcher(
      [
        (
          {
            "status": "success",
            "hits": [
              {
                "document_id": "edgar:0000789019-26-000012",
                "ticker": "MSFT",
                "source": "filing",
                "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
              }
            ],
          },
          None,
        )
      ]
    )
  )

  _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  dispatch = _complete_event(runner)["dispatch"]
  assert dispatch["outcome"] == "ok"
  assert dispatch["attempts"] == 1
  assert dispatch["route_id"] == "local/filings_search"
  assert dispatch["sources"] == [
    {
      "document_id": "edgar:0000789019-26-000012",
      "source_kind": "filing",
      "source_url": "https://www.sec.gov/Archives/msft-10k.htm",
    }
  ]


def test_dispatch_record_is_emitted_on_the_semantic_error_exit_path() -> None:
  runner = _dispatch_runner(
    _RecordedDispatcher([({"status": "error", "error": {"code": "no_rows"}}, None)])
  )

  _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  event = _complete_event(runner)
  assert event["dispatch"]["outcome"] == "error_semantic"
  assert event["dispatch"]["sources"] == []
  # D-B1-4: `semantic_error` stays on the event; the outcome rides beside it.
  assert event["semantic_error"]["code"] == "tool_status_error"


def test_dispatch_record_is_emitted_on_the_exclusion_exit_path() -> None:
  runner = _dispatch_runner(_ExplodingDispatcher())
  runner._excluded_tools = {"filings_search"}

  _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  dispatch = _complete_event(runner)["dispatch"]
  assert dispatch["outcome"] == "error_semantic"
  assert dispatch["attempts"] == 1
  assert dispatch["sources"] == []


def test_dispatch_record_is_emitted_on_the_rate_limited_exit_path() -> None:
  runner = _dispatch_runner(
    _RecordedDispatcher(
      [(None, {"code": "rate_limited", "message": "Rate limit: max 5 calls per 60s"})]
    )
  )

  _run(runner._execute_single_tool("tool-1", "get_quote", {}, {"tools": []}))

  assert _complete_event(runner)["dispatch"]["outcome"] == "error_rate_limited"


def test_dispatch_record_is_emitted_on_the_transport_exit_path() -> None:
  class _Boom:
    async def dispatch(self, tool_id, tool_name, tool_input, *, call_index=0):
      _ = tool_id, tool_name, tool_input, call_index
      raise RuntimeError("connection reset by peer")

  runner = _dispatch_runner(_Boom())

  _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  event = _complete_event(runner)
  assert event["error"]["code"] == "internal_error"
  assert event["dispatch"]["outcome"] == "error_transport"


def test_dispatch_record_is_emitted_on_the_timeout_exit_path() -> None:
  dispatcher = _HangingDispatcher()
  runner = _dispatch_runner(dispatcher, tool_call_timeout=0.01)

  _run(runner._execute_single_tool("tool-1", "file_read", {}, {"tools": []}))

  event = _complete_event(runner)
  assert event["error"]["code"] == "tool_timeout"
  assert event["dispatch"]["outcome"] == "error_timeout"


def test_dispatch_record_is_on_the_event_before_the_cancelled_arm_appends() -> None:
  class _Cancelling:
    async def dispatch(self, tool_id, tool_name, tool_input, *, call_index=0):
      _ = tool_id, tool_name, tool_input, call_index
      raise asyncio.CancelledError()

  runner = _dispatch_runner(_Cancelling())

  with pytest.raises(asyncio.CancelledError):
    _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  event = _complete_event(runner)
  assert event["dispatch"]["outcome"] == "cancelled"
  assert event["final_tool_result_blocks"]


def test_transient_read_failure_is_retried_and_settles_ok(
  declared_read_effects: None,
) -> None:
  dispatcher = _RecordedDispatcher(
    [
      (None, {"code": "internal_error", "message": "connection reset"}),
      ({"status": "success", "hits": []}, None),
    ]
  )
  runner = _dispatch_runner(dispatcher)

  _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  event = _complete_event(runner)
  assert dispatcher.calls == 2
  assert event["dispatch"]["outcome"] == "ok"
  assert event["dispatch"]["attempts"] == 2
  assert "retries_exhausted" not in event["dispatch"]
  # No re-emitted tool_call_start: `attempts` carries the multiplicity.
  assert _start_event_count(runner) == 1


def test_retries_stop_at_two_and_emit_retries_exhausted(
  declared_read_effects: None,
) -> None:
  dispatcher = _RecordedDispatcher(
    [(None, {"code": "rate_limited", "message": "HTTP 429 Too Many Requests"})]
  )
  runner = _dispatch_runner(dispatcher)

  _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  event = _complete_event(runner)
  assert dispatcher.calls == 3
  assert event["dispatch"]["outcome"] == "error_rate_limited"
  assert event["dispatch"]["attempts"] == 3
  assert event["dispatch"]["retries_exhausted"] is True
  assert _start_event_count(runner) == 1


def test_semantic_failures_are_never_retried(declared_read_effects: None) -> None:
  dispatcher = _RecordedDispatcher(
    [({"status": "error", "error": {"code": "no_rows"}}, None)]
  )
  runner = _dispatch_runner(dispatcher)

  _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  assert dispatcher.calls == 1
  assert _complete_event(runner)["dispatch"]["attempts"] == 1


def test_undeclared_tools_are_never_retried(declared_read_effects: None) -> None:
  dispatcher = _RecordedDispatcher(
    [(None, {"code": "internal_error", "message": "connection reset"})]
  )
  runner = _dispatch_runner(dispatcher)

  _run(runner._execute_single_tool("tool-1", "memory_write", {}, {"tools": []}))

  assert dispatcher.calls == 1
  assert _complete_event(runner)["dispatch"]["attempts"] == 1


def test_a_set_abort_event_stops_retrying_between_attempts(
  declared_read_effects: None,
) -> None:
  dispatcher = _RecordedDispatcher(
    [(None, {"code": "internal_error", "message": "connection reset"})]
  )
  runner = _dispatch_runner(dispatcher)
  abort = asyncio.Event()
  abort.set()
  runner._tool_abort_event = abort

  _run(runner._execute_single_tool("tool-1", "filings_search", {}, {"tools": []}))

  assert dispatcher.calls == 1
  assert _complete_event(runner)["dispatch"]["attempts"] == 1
