# ruff: noqa: E402

import asyncio
import hashlib
import json
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, AgentSessionLog, EventLog, TaskRegistry, TaskState, ToolDispatcher
from agent_gateway.capability_binding import (
  CapabilityBind,
)
from agent_gateway.execution_snapshot import (
  build_agent_execution_snapshot,
  render_result_instructions,
)
from agent_gateway.final_narrative_artifact import publish_final_narrative
from agent_gateway.providers import ModelInfo, ModelProvider, StreamEvent
from agent_gateway.openai_history_fence import TEXT_SIGNATURE_MARKER
from agent_gateway.session import GatewaySession
from agent_gateway.skills import SkillLoader, SkillProfile
from agent_gateway.sub_agent import (
  _canonical_result_requirement,
  _finalize_resume_abandoned,
  _ordinary_admitted_task_factory,
  _prior_result_evidence,
  _research_file_id_admitted_input,
  _research_file_id_from_admitted_inputs,
  _ticker_admitted_input,
  make_resume_handler as _make_resume_handler,
  make_resume_tool_def,
)
from agent_gateway.skills import compile_agent_operation
from agent_gateway.sub_agent_scope_receipt import ADMITTED_TASK_METADATA_KEY
from agent_gateway.runner_background_tasks import (
  normalize_required_skill_lifecycle,
)
from agent_gateway.runner_introspection import exception_traceback_already_logged
from agent_gateway.sub_agent_result_contract import (
  CHILD_EVIDENCE_EXTERNALIZATION_MAX_NODES,
  terminal_narrative_content_handle,
)
from agent_gateway.agent_session_log_records import LogEntry
from agent_gateway.task_registry import CoordinatorConfig, ParentMessage, TaskEntry
from agent_workflow_contracts import (
  ActivityHandle,
  AgentOperationRef,
  AttemptRef,
  ContentHandle,
  ContractRef,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  ToolGrant,
  ToolGrantEntry,
  TranscriptHandle,
  sha256_digest,
)
from agent_gateway.transcript import (
  ChildRunSegment,
  TranscriptIntegrityError,
  build_synthetic_tool_results,
  detect_orphan_tool_uses,
  place_resume_messages,
  reconstruct_messages_for_task,
  reconstruct_parent_messages,
)
from tests.capability_execution_test_support import (
  stub_bound_capability_execution,
  stub_capability_execution_resolver,
)

_UNRESOLVED_BLOCK_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


def _bind_receipt(
  *,
  capability_id: str = "node.implement",
  provider: str = "anthropic",
  model: str = "claude-sonnet-4-6",
  effort: str = "high",
) -> dict[str, str]:
  return {
    "schema_version": "1.0",
    "capability_id": capability_id,
    "model_key": f"test.{provider}.{model}",
    "provider": provider,
    "upstream_model": model,
    "adapter": f"test.{provider}",
    "protocol_profile": "test.reasoning",
    "route": "test.in_process",
    "effort": effort,
    "credential_principal": "user",
    "credential_ref": f"test-user:{provider}",
    "run_mode": "interactive",
    "registry_revision": "test-capability-execution.1",
    "policy_revision": "test-capability-execution.1",
    "selection_source": "internal_policy",
  }


def _interrupted_task_correlation(
  runner: AgentRunner,
  *,
  task_id: str,
  capability_bind_receipt: dict[str, str],
) -> dict[str, Any]:
  return {
    "owner_runner_id": getattr(runner, "_runner_id", None),
    "owner_role": getattr(runner, "_role", "writer"),
    "sub_agent_id": f"sub:{task_id}",
    "parent_turn_id": f"turn:{task_id}",
    "call_index": 0,
    "capability_bind": dict(capability_bind_receipt),
  }


def _required_skill_lifecycle(
  skill_run_id: str,
  *,
  skill: str = "earnings-review",
  ticker: str | None = "PCTY",
) -> dict[str, Any]:
  return {
    "schema_version": 2,
    "skill_run_id": skill_run_id,
    "skill": skill,
    "scope": "ticker" if ticker is not None else "portfolio",
    "ticker": ticker,
    "portfolio_id": None,
  }


def _required_skill_result_event(
  _entry: TaskEntry,
  result: dict[str, Any] | None,
  error: dict[str, Any] | None,
) -> dict[str, Any]:
  event = _required_skill_result_core()
  if error is not None:
    event.update({
      "exit_code": 1,
      "outcome": "error",
      "status": "error",
      "error": str(
        error.get("message")
        or error.get("error")
        or error.get("code")
        or "child failed"
      ),
    })
  return event


def _required_skill_result_core() -> dict[str, Any]:
  return {
    "type": "skill_result_captured",
    "exit_code": 0,
    "outcome": "success",
    "status": "success",
    "gate_code": None,
    "artifact_refs": [],
    "proposal_ids": [],
    "verdict_echo": None,
    "fms_results": [],
    "artifact_events": [],
    "output_memory_file": None,
    "cost_usd": None,
    "duration_s": None,
    "compaction_count": 0,
    "error": None,
    "warnings": [],
    "approval_outcome": None,
    "approval_id": None,
    "approval_tool_name": None,
  }


def test_required_skill_lifecycle_normalizer_preserves_exact_v2_identity() -> None:
  lifecycle = {
    "schema_version": 2,
    "skill_run_id": "skill-run-portfolio",
    "skill": "portfolio-review",
    "scope": "portfolio",
    "ticker": None,
    "portfolio_id": "portfolio-1",
  }

  assert normalize_required_skill_lifecycle(lifecycle) == lifecycle


def test_required_skill_lifecycle_accepts_dispatch_scope_portfolio_id_bound() -> None:
  lifecycle = {
    "schema_version": 2,
    "skill_run_id": "skill-run-max-portfolio-id",
    "skill": "portfolio-review",
    "scope": "portfolio",
    "ticker": None,
    "portfolio_id": "p" * 256,
  }

  assert normalize_required_skill_lifecycle(lifecycle) == lifecycle


@pytest.mark.parametrize(
  "invalid_lifecycle",
  [
    {
      "schema_version": 1,
      "skill_run_id": "skill-run-stale-v1",
      "skill": "earnings-review",
      "ticker": "PCTY",
    },
    {
      "schema_version": 2,
      "skill_run_id": "skill-run-missing-scope",
      "skill": "earnings-review",
      "ticker": "PCTY",
      "portfolio_id": None,
    },
    {
      "schema_version": 2,
      "skill_run_id": "skill-run-unhashable-scope",
      "skill": "earnings-review",
      "scope": [],
      "ticker": "PCTY",
      "portfolio_id": None,
    },
    {
      "schema_version": 2,
      "skill_run_id": "skill-run-extra-field",
      "skill": "earnings-review",
      "scope": "ticker",
      "ticker": "PCTY",
      "portfolio_id": None,
      "legacy_scope": "ticker",
    },
    {
      "schema_version": True,
      "skill_run_id": "skill-run-bool-version",
      "skill": "earnings-review",
      "scope": "ticker",
      "ticker": "PCTY",
      "portfolio_id": None,
    },
    {
      "schema_version": 2,
      "skill_run_id": " skill-run-whitespace ",
      "skill": "earnings-review",
      "scope": "ticker",
      "ticker": "PCTY",
      "portfolio_id": None,
    },
    {
      "schema_version": 2,
      "skill_run_id": "skill-run-cross-scope-ticker",
      "skill": "earnings-review",
      "scope": "portfolio",
      "ticker": "PCTY",
      "portfolio_id": None,
    },
    {
      "schema_version": 2,
      "skill_run_id": "skill-run-cross-scope-portfolio",
      "skill": "earnings-review",
      "scope": "ticker",
      "ticker": "PCTY",
      "portfolio_id": "portfolio-1",
    },
    {
      "schema_version": 2,
      "skill_run_id": "skill-run-missing-ticker",
      "skill": "earnings-review",
      "scope": "ticker",
      "ticker": None,
      "portfolio_id": None,
    },
    {
      "schema_version": 2,
      "skill_run_id": "skill-run-oversized-portfolio-id",
      "skill": "portfolio-review",
      "scope": "portfolio",
      "ticker": None,
      "portfolio_id": "p" * 257,
    },
  ],
)
def test_required_skill_lifecycle_normalizer_rejects_noncanonical_v2(
  invalid_lifecycle: dict[str, Any],
) -> None:
  with pytest.raises(ValueError):
    normalize_required_skill_lifecycle(invalid_lifecycle)


def test_required_skill_result_factory_must_emit_complete_core(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    entry = runner._task_registry.register(
      "background_agent",
      task_id="bg_incomplete_skill_result_core",
    )

    def _incomplete_factory(
      _entry: TaskEntry,
      _result: dict[str, Any] | None,
      _error: dict[str, Any] | None,
    ) -> dict[str, Any]:
      event = _required_skill_result_core()
      event.pop("approval_outcome")
      return event

    entry.required_skill_result_event_factory = _incomplete_factory
    with pytest.raises(RuntimeError, match="missing required fields"):
      await runner._build_required_skill_result_event(
        entry,
        _required_skill_lifecycle("skill-run-incomplete-core"),
      )

  _run(_case())


class _TestResumeCapabilityExecutionResolver:
  def __init__(self) -> None:
    self.authorize_calls: list[CapabilityBind] = []
    self.materialize_calls: list[CapabilityBind] = []
    self._resolver = stub_capability_execution_resolver(
      extra_models=(("openai", "gpt-4o-mini"),),
    )

  def authorize_bind(self, bind: CapabilityBind) -> CapabilityBind:
    self.authorize_calls.append(bind)
    return self._resolver.authorize_bind(bind)

  def materialize_bind(self, bind: CapabilityBind):
    self.materialize_calls.append(bind)
    return self._resolver.materialize_bind(bind)


def make_resume_handler(*args: Any, **kwargs: Any):
  kwargs.setdefault(
    "capability_execution_resolver",
    _TestResumeCapabilityExecutionResolver(),
  )
  return _make_resume_handler(*args, **kwargs)


def _run(coro):
  return asyncio.run(coro)


def _report_task_result(
  summary: str,
  *,
  task_id: str = "test-task",
  workspace_dir: Path | None = None,
) -> TaskResult:
  raw = summary.encode("utf-8")
  content_sha = hashlib.sha256(raw).hexdigest()
  operation = AgentOperationRef(
    namespace="agent-operation",
    name="test-resume",
    version="1.0",
    digest=sha256_digest({"operation": "test-resume", "version": "1.0"}),
  )
  logical_task = OrdinaryDelegationTaskRef(
    delegation_id=f"delegation:{task_id}",
    operation=operation,
  )
  attempt = AttemptRef(
    attempt_number=1,
    attempt_id=f"{task_id}:attempt:1",
    physical_task_id=task_id,
  )
  if workspace_dir is None:
    handle = ContentHandle(
      content_id=f"sha256:{content_sha}",
      content_sha256=content_sha,
      content_bytes=len(raw),
      content_chars=len(raw.decode("utf-8")),
      contract=ContractRef(
        namespace="test",
        name="terminal-assistant-message",
        version="1.0",
        digest=sha256_digest({
          "contract": "terminal-assistant-message",
          "version": "1.0",
        }),
      ),
      media_type="text/plain",
      encoding="utf-8",
      retention="durable",
    )
  else:
    handle = terminal_narrative_content_handle(publish_final_narrative(
      workspace_dir=workspace_dir,
      sub_agent_id=task_id,
      terminal_event_seq=1,
      text=summary,
    ))
  digest = sha256_digest({"task_id": task_id, "summary": summary})
  return TaskResult(
    task_result_id=f"task-result:{content_sha}",
    logical_task=logical_task,
    attempt=attempt,
    execution=ExecutionSettlement(status="succeeded"),
    values=TaskResultValues(
      terminal_narrative=handle,
    ),
    observation=TaskObservation(
      transcript=TranscriptHandle(kind="child_transcript", owner_id=task_id),
      activity=ActivityHandle(kind="child_activity", owner_id=task_id),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=digest,
      model_bind_digest=digest,
      capability_binding_digest=digest,
      tool_grant_digest=digest,
    ),
  )


def _report_task_result_payload(
  summary: str,
  *,
  task_id: str = "test-task",
) -> dict[str, Any]:
  return _report_task_result(summary, task_id=task_id).model_dump(mode="json")


async def _successful_resume_task_result(
  kwargs: dict[str, Any],
  summary: str,
  *,
  workspace_dir: Path,
) -> TaskResult:
  handle = terminal_narrative_content_handle(publish_final_narrative(
    workspace_dir=workspace_dir,
    sub_agent_id=kwargs["attempt"].physical_task_id,
    terminal_event_seq=1,
    text=summary,
  ))
  return TaskResult(
    task_result_id=f"resume-result:{kwargs['attempt'].attempt_id}",
    logical_task=kwargs["logical_task"],
    attempt=kwargs["attempt"],
    execution=ExecutionSettlement(status="succeeded"),
    values=TaskResultValues(terminal_narrative=handle),
    observation=TaskObservation(
      transcript=TranscriptHandle(
        kind="child_transcript",
        owner_id=kwargs["attempt"].physical_task_id,
      ),
      activity=ActivityHandle(
        kind="child_activity",
        owner_id=kwargs["attempt"].physical_task_id,
      ),
    ),
    provenance=kwargs["result_provenance"],
  )


def _successful_admitted_task_result(
  admitted_task: Any,
  summary: str,
  *,
  workspace_dir: Path,
) -> TaskResult:
  attempt = admitted_task.attempt
  handle = terminal_narrative_content_handle(publish_final_narrative(
    workspace_dir=workspace_dir,
    sub_agent_id=attempt.physical_task_id,
    terminal_event_seq=1,
    text=summary,
  ))
  content_sha = handle.content_sha256
  return TaskResult(
    task_result_id=f"task-result:{attempt.attempt_id}:{content_sha}",
    logical_task=admitted_task.logical_task,
    attempt=attempt,
    execution=ExecutionSettlement(status="succeeded"),
    values=TaskResultValues(
      terminal_narrative=handle,
    ),
    observation=TaskObservation(
      transcript=TranscriptHandle(
        kind="child_transcript",
        owner_id=attempt.physical_task_id,
      ),
      activity=ActivityHandle(
        kind="child_activity",
        owner_id=attempt.physical_task_id,
      ),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=admitted_task.admitted_task_digest,
      model_bind_digest=admitted_task.model_bind_digest,
      capability_binding_digest=(
        admitted_task.capability_binding_digest
      ),
      tool_grant_digest=admitted_task.tool_grant_digest,
    ),
  )


def _assert_resume_abandoned(
  result: Any | None,
  error: dict[str, Any] | None,
  *,
  code: str,
) -> TaskResult:
  assert error is None
  assert isinstance(result, TaskResult)
  assert result.execution.status == "failed"
  assert result.execution.terminal_reason is not None
  assert result.execution.terminal_reason.startswith(
    f"resume_abandoned:{code}:"
  )
  return result


def test_prior_rejected_evidence_survives_lineage_warning_rebuild() -> None:
  registration = LogEntry(
    seq=1,
    timestamp=1.0,
    event={"type": "task_registered", "task_id": "bg_rejected"},
  )
  evidence_entry = LogEntry(
    seq=2,
    timestamp=2.0,
    event={
      "type": "tool_call_complete",
      "tool_name": "fms_report_demo",
      "result": {
        "status": "ok",
        "wide": (
          [None]
          * (CHILD_EVIDENCE_EXTERNALIZATION_MAX_NODES + 1)
        ),
      },
    },
  )
  prior = _prior_result_evidence(
    [
      ChildRunSegment(
        task_id="bg_rejected",
        original_task_id=None,
        registration=registration,
        completion=None,
        runner_id="runner-test",
        entries=(evidence_entry,),
      ),
    ],
  )

  assert prior.admission_rejected is True
  assert any(
    "was interrupted" in warning
    for warning in prior.warning_parts
  )


class _NullMcpClient:
  def is_mcp_tool(self, _name: str) -> bool:
    return False

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": f"Unknown tool: {name}"}

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []


class _InvestmentMcpClient(_NullMcpClient):
  @staticmethod
  def is_mcp_tool(name: str) -> bool:
    return name == "start_quant_research"

  @staticmethod
  def get_server_for_tool(name: str) -> str | None:
    return (
      "idea-workbench-mcp"
      if name == "start_quant_research"
      else None
    )

  @staticmethod
  def get_tool_definitions() -> list[dict[str, Any]]:
    return [{
      "name": "start_quant_research",
      "description": "Start exact quant research.",
      "input_schema": {"type": "object"},
    }]


class _JobsMcpClient(_NullMcpClient):
  def __init__(self) -> None:
    self.call_count = 0

  @staticmethod
  def is_mcp_tool(name: str) -> bool:
    return name == "list_jobs"

  @staticmethod
  def get_server_for_tool(name: str) -> str | None:
    return "jobs-mcp" if name == "list_jobs" else None

  @staticmethod
  def get_tool_definitions() -> list[dict[str, Any]]:
    return [{
      "name": "list_jobs",
      "description": "List operator jobs.",
      "input_schema": {"type": "object"},
    }]

  async def call_tool(self, name: str, tool_input: dict[str, Any]):
    self.call_count += 1
    return {"name": name, "input": tool_input}, None


class _ResumeTestProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

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


def _runner(
  tmp_path: Path,
  *,
  max_resume_chain_depth: int = 3,
  max_retained_tasks: int = 50,
) -> AgentRunner:
  provider = _ResumeTestProvider()
  runner = AgentRunner(
    event_log=EventLog(),
    dispatcher=_dispatcher(),
    session_id="sess-parent",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="claude-sonnet-4-6",
      effort="none",
      auth_config={
        "auth_mode": "api",
        "api_key": "test-secret",
      },
    ),
    agent_session_log=AgentSessionLog(path=tmp_path / "sessions" / "runner.jsonl"),
    workspace_dir=str(tmp_path),
    task_registry=TaskRegistry(max_retained=max_retained_tasks),
    max_resume_chain_depth=max_resume_chain_depth,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  runner._runner_id = "runner-test"
  return runner


async def _append_unsettled_required_skill_task(
  runner: AgentRunner,
  *,
  task_id: str,
  lifecycle: dict[str, Any],
  sub_agent_id: str,
  result: dict[str, Any] | None = None,
) -> None:
  metadata = {
    "owner_runner_id": runner._runner_id,
    "owner_role": runner._role,
    "sub_agent_id": sub_agent_id,
    "parent_turn_id": f"turn:{task_id}",
    "call_index": 0,
    "task_type": "background",
    "required_skill_lifecycle": lifecycle,
  }
  await runner._append_durable_event({
    "type": "task_registered",
    "capability_bind": _bind_receipt(),
    "task_id": task_id,
    "task_type": "background",
    "agent_name": "earnings-review",
    "sub_agent_id": sub_agent_id,
    "metadata": metadata,
    "started_at": 1.0,
  })
  await runner._append_durable_event({
    "type": "task_completed",
    "task_id": task_id,
    "task_type": "background",
    "sub_agent_id": sub_agent_id,
    "final_state": TaskState.COMPLETED.value,
    "completed_at": 2.0,
    "result": (
      result
      if result is not None
      else _report_task_result_payload(
        f"completed {task_id}",
        task_id=task_id,
      )
    ),
    "error": None,
  })


async def _append_task_log(log: AgentSessionLog) -> None:
  child_runner_id = "runner-child-bg-3"
  await log.append(
    {
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": "bg_3",
      "task_type": "background",
      "agent_name": "earnings-review",
      "sub_agent_id": "sub3:sess-parent",
      "started_at": 1.0,
    }
  )
  await log.append({
    "type": "attach",
    "sub_agent_id": "sub3:sess-parent",
    "runner_id": child_runner_id,
    "role": "sub_agent",
  })
  await log.append({
    "type": "user_message",
    "sub_agent_id": "sub3:sess-parent",
    "runner_id": child_runner_id,
    "role": "sub_agent",
    "content": "Review AAPL",
  })
  await log.append(
    {
      "type": "assistant_message",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": child_runner_id,
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
      "runner_id": child_runner_id,
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
    "mutation_mode: read_only",
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
  capability_bind_receipt: dict[str, str] | None = None,
  original_task_id: str | None = None,
  resumable: bool = True,
  allowed_tools: tuple[str, ...] = (),
  required_context: tuple[str, ...] = ("ticker",),
  ticker: str | None = "PCTY",
  research_file_id: int | None = None,
  max_budget_usd: float | None = None,
) -> None:
  receipt = dict(capability_bind_receipt or _bind_receipt())
  model_bind = CapabilityBind.from_receipt(receipt)
  profile = SkillProfile(
    name=agent_name,
    system_prompt="Resume carefully.",
    version="1.0",
    agent_callable=True,
    resumable=resumable,
    mutation_mode="read_only",
    delegation_role="implement",
    metadata={
      "semantic_metadata": {
        "required_context": list(required_context),
      }
    },
  )
  operation = compile_agent_operation(
    profile,
    execution_class=model_bind.capability_id,
  )
  result_requirement = _canonical_result_requirement(operation=operation)
  grant_id = f"grant:{task_id}"
  grant_entries = tuple(
    ToolGrantEntry(
      tool_id=tool_id,
      route_id=f"test:{tool_id}",
      effect="read",
    )
    for tool_id in sorted(set(allowed_tools))
  )
  tool_grant = ToolGrant(
    grant_id=grant_id,
    tools=grant_entries,
    digest=sha256_digest({
      "grant_id": grant_id,
      "tools": [entry.model_dump(mode="json") for entry in grant_entries],
    }),
  )
  execution_snapshot = build_agent_execution_snapshot(
    operation=operation,
    result_instructions=render_result_instructions(result_requirement),
    max_turns=15,
    timeout_seconds=None,
    client_timeout_seconds=90,
    max_tokens=64_000,
    cost_observation_threshold_usd=5.0,
    max_resume_chain_depth=3,
    max_budget_usd=max_budget_usd,
  )
  admission_parent = SimpleNamespace(
    tenant_id="tenant-test",
    session_id="session-test",
  )
  admitted_inputs = (
    (
      _ticker_admitted_input(
        ticker,
        invocation_id=task_id,
        parent_session=admission_parent,
      ),
    )
    if ticker is not None and "ticker" in required_context
    else ()
  ) + (
    (
      _research_file_id_admitted_input(
        research_file_id,
        invocation_id=task_id,
        parent_session=admission_parent,
      ),
    )
    if research_file_id is not None
    and "research_file_id" in required_context
    else ()
  )
  admitted_task = _ordinary_admitted_task_factory(
    operation=operation,
    execution_snapshot=execution_snapshot,
    capability_bindings=(),
    tool_grant=tool_grant,
    model_bind=model_bind,
    result_requirement=result_requirement,
    objective=user_message,
    parent_session=admission_parent,
    inputs=admitted_inputs,
    attempt_number=1,
  )(SimpleNamespace(task_id=task_id))
  entry = runner._task_registry.register(
    "background_agent",
    agent_name=agent_name,
    task_id=task_id,
    original_task_id=original_task_id,
  )
  entry.state = TaskState.INTERRUPTED
  entry.capability_bind_receipt = receipt
  entry.admitted_task = admitted_task
  entry.metadata.update({
    "cost_observation_threshold_usd": 5.0,
    ADMITTED_TASK_METADATA_KEY: admitted_task.model_dump(mode="json"),
  })
  log = runner._agent_session_log
  assert log is not None
  child_runner_id = f"runner:{task_id}"
  correlation = _interrupted_task_correlation(
    runner,
    task_id=task_id,
    capability_bind_receipt=receipt,
  )
  registration = {
    "type": "task_registered",
    "task_id": task_id,
    "task_type": "background",
    "agent_name": agent_name,
    **correlation,
    "started_at": 1.0,
    "metadata": {
      **correlation,
      "task_type": "background",
      "cost_observation_threshold_usd": 5.0,
      ADMITTED_TASK_METADATA_KEY: admitted_task.model_dump(mode="json"),
    },
  }
  if original_task_id is not None:
    registration["original_task_id"] = original_task_id
  await log.append(registration)
  await log.append(
    {
      "type": "attach",
      "sub_agent_id": f"sub:{task_id}",
      "runner_id": child_runner_id,
      "role": "sub_agent",
    }
  )
  await log.append(
    {
      "type": "user_message",
      "task_id": task_id,
      "sub_agent_id": f"sub:{task_id}",
      "runner_id": child_runner_id,
      "role": "sub_agent",
      "content": user_message,
    }
  )


@pytest.mark.parametrize(
  ("recovered_id", "expects_rebind"),
  [
    (37, True),
    (0, False),
  ],
)
def test_resume_handler_rebinds_only_positive_recovered_research_file_id(
  tmp_path: Path,
  recovered_id: int,
  expects_rebind: bool,
) -> None:
  async def _case() -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id=f"bg_rebind_{recovered_id}",
      agent_name="earnings-review",
      user_message=f"Ticker: ADI\nRESEARCH_FILE_ID={recovered_id}",
      allowed_tools=("fms_probe",),
    )
    captured: dict[str, Any] = {}
    rebound_calls: list[int] = []

    async def _parent_fms_handler(_tool_input: dict[str, Any], **_kwargs: Any):
      return {"handler": "parent"}, None

    async def _rebound_fms_handler(_tool_input: dict[str, Any], **_kwargs: Any):
      return {"handler": "rebound"}, None

    def _rebind_fms(handlers: dict[str, Any], research_file_id: int) -> None:
      rebound_calls.append(research_file_id)
      handlers["fms_probe"] = _rebound_fms_handler

    async def _resume_sub_agent(**kwargs: Any):
      captured.update(kwargs)
      return await _successful_resume_task_result(
        kwargs,
        "continued",
        workspace_dir=tmp_path,
      ), None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      local_tool_handlers={"fms_probe": _parent_fms_handler},
      fms_rebinder=_rebind_fms,
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": f"bg_rebind_{recovered_id}"})

    assert error is None
    assert result is not None
    assert not isinstance(result, TaskResult)
    resumed_entry = runner._task_registry.get(result["task_id"])
    assert resumed_entry is not None
    await resumed_entry.asyncio_task
    assert captured["skill_name"] == "earnings-review"
    assert captured["result_requirement"].mode == "narrative"
    assert "final assistant message is the authoritative child result" in (
      captured["system_prompt"]
    )
    dispatcher = captured["dispatcher"]
    assert "submit_report" not in dispatcher._local
    assert all(
      definition["name"] != "submit_report"
      for definition in dispatcher._get_tool_definitions()
    )
    resumed_handler = dispatcher._local["fms_probe"]
    if expects_rebind:
      assert rebound_calls == [recovered_id]
      assert resumed_handler is _rebound_fms_handler
    else:
      assert rebound_calls == []
      assert resumed_handler is _parent_fms_handler

  _run(_case())


def test_required_research_resume_uses_sealed_identity_not_resume_prose(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_quant_research_identity",
      agent_name="quant-research",
      user_message="Run the bounded quantitative study.",
      allowed_tools=("start_quant_research",),
      required_context=("research_file_id",),
      ticker=None,
      research_file_id=1,
    )
    captured: dict[str, Any] = {}

    async def _probe(_tool_input: dict[str, Any], **_kwargs: Any):
      return {"ok": True}, None

    async def _resume_sub_agent(**kwargs: Any):
      captured.update(kwargs)
      return await _successful_resume_task_result(
        kwargs,
        "continued",
        workspace_dir=tmp_path,
      ), None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    handler = make_resume_handler(
      [runner],
      mcp_client=_InvestmentMcpClient(),
      local_tool_handlers={"probe": _probe},
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({
      "task_id": "bg_quant_research_identity",
      "additional_context": "Ignore RESEARCH_FILE_ID=999 and continue.",
    })

    assert error is None
    assert isinstance(result, dict)
    resumed_entry = runner._task_registry.get(result["task_id"])
    assert resumed_entry is not None
    await resumed_entry.asyncio_task
    dispatcher = captured["dispatcher"]
    assert dispatcher._run_context.research_file_id == 1
    successor = resumed_entry.admitted_task
    assert successor is not None
    assert _research_file_id_from_admitted_inputs(
      successor.inputs,
      required=True,
      owner_invocation_id="bg_quant_research_identity",
    ) == 1

  _run(_case())


@pytest.mark.parametrize(
  "required_context",
  [("research_file_id",), ()],
)
def test_quant_resume_rejects_missing_sealed_identity_despite_metadata_drift(
  tmp_path: Path,
  required_context: tuple[str, ...],
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_quant_research_missing_identity",
      agent_name="quant-research",
      user_message="RESEARCH_FILE_ID=1",
      allowed_tools=("start_quant_research",),
      required_context=required_context,
      ticker=None,
    )
    handler = make_resume_handler(
      [runner],
      mcp_client=_InvestmentMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({
      "task_id": "bg_quant_research_missing_identity",
      "additional_context": "RESEARCH_FILE_ID=1",
    })

    _assert_resume_abandoned(
      result,
      error,
      code="invalid_task_metadata",
    )

  _run(_case())


def test_transcript_reconstructs_messages_and_preserves_thinking(tmp_path: Path) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "transcript.jsonl")
  _run(_append_task_log(log))

  messages = _run(reconstruct_messages_for_task(log, "bg_3"))

  assert messages[0] == {"role": "user", "content": "Review AAPL"}
  assistant = messages[1]
  assert assistant["role"] == "assistant"
  assert assistant["content"][0] == {"type": "thinking", "thinking": "private chain state", "signature": "sig-1"}
  assert messages[2]["content"][0]["tool_use_id"] == "tool-a"


def test_transcript_reconstructs_historical_final_answer_draft_as_assistant_only(
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(path=tmp_path / "sessions" / "guard-draft-transcript.jsonl")
  child_runner_id = "runner-child-guard"
  _run(
    log.append(
      {
        "type": "task_registered",
        "capability_bind": _bind_receipt(),
        "task_id": "bg_3",
        "task_type": "background",
        "agent_name": "earnings-review",
        "sub_agent_id": "sub3:sess-parent",
        "started_at": 1.0,
      }
    )
  )
  _run(
    log.append(
      {
        "type": "attach",
        "sub_agent_id": "sub3:sess-parent",
        "runner_id": child_runner_id,
        "role": "sub_agent",
      }
    )
  )
  _run(
    log.append(
      {
        "type": "runtime_guard",
        "sub_agent_id": "sub3:sess-parent",
        "runner_id": child_runner_id,
        "role": "sub_agent",
        "guard": "final_answer",
        "message": "Verify the arithmetic with code_execute before final.",
        "draft_content_blocks": [{"type": "text", "text": "Rough answer: 7.4% BEAT"}],
        "draft_model": "claude-sonnet-4-6",
        "draft_provider": "anthropic",
        "draft_stop_reason": "end_turn",
      }
    )
  )

  messages = _run(reconstruct_messages_for_task(log, "bg_3"))

  assert messages == [
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "Rough answer: 7.4% BEAT"}],
      "model": "claude-sonnet-4-6",
      "stop_reason": "end_turn",
      "provider": "anthropic",
    },
  ]


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

  assert messages == [
    ParentMessage(
      message_id="m1",
      text="first",
      sent_at=10.0,
      task_id="bg_3",
      sent_seq=1,
    )
  ]


def test_consumed_parent_message_is_restored_before_assistant_not_resume_tail(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "consumed-parent-message.jsonl")
    registration = await log.append({
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": "bg_3",
      "task_type": "background",
      "agent_name": "earnings-review",
      "sub_agent_id": "sub3:sess-parent",
      "started_at": 1.0,
    })
    sent = await log.append({
      "type": "parent_message_sent",
      "task_id": "bg_3",
      "message_id": "m-consumed",
      "message": "Use the amended filing",
      "sent_at": 2.0,
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-parent",
      "role": "writer",
    })
    await log.append({
      "type": "attach",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child",
      "role": "sub_agent",
    })
    await log.append({
      "type": "user_message",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child",
      "role": "sub_agent",
      "content": "Review AAPL",
    })
    assistant = await log.append({
      "type": "assistant_message",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child",
      "role": "sub_agent",
      "content_blocks": [{"type": "text", "text": "Using the amendment."}],
      "stop_reason": "end_turn",
      "model": "stub-model",
      "parent_message_consumptions": [{
        "task_id": "bg_3",
        "message_id": "m-consumed",
        "parent_message_seq": sent.seq,
        "consumer_turn": 2,
      }],
    })
    await log.append({
      "type": "parent_message_consumed",
      "task_id": "bg_3",
      "message_id": "m-consumed",
      "parent_message_seq": sent.seq,
      "consumer_turn": 2,
      "assistant_message_seq": assistant.seq,
      "consumed_at": 3.0,
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child",
      "role": "sub_agent",
    })
    await log.append({
      "type": "parent_message_sent",
      "task_id": "bg_3",
      "message_id": "m-pending",
      "message": "Also check note 7",
      "sent_at": 4.0,
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-parent",
      "role": "writer",
    })

    transcript = await reconstruct_messages_for_task(log, "bg_3")
    pending = await reconstruct_parent_messages(log, "bg_3", before_ts=10.0)

    assert registration.seq < sent.seq < assistant.seq
    assert transcript == [
      {"role": "user", "content": "Review AAPL"},
      {
        "role": "user",
        "content": (
          "Operator update for this task:\n"
          "- id=m-consumed: Use the amended filing"
        ),
      },
      {
        "role": "assistant",
        "content": [{"type": "text", "text": "Using the amendment."}],
        "model": "stub-model",
        "stop_reason": "end_turn",
      },
    ]
    assert [message.message_id for message in pending] == ["m-pending"]

  _run(_case())


def test_assistant_binding_closes_consumption_audit_crash_window(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    log = runner._agent_session_log
    assert log is not None
    await log.append({
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": "bg_3",
      "task_type": "background",
      "agent_name": "earnings-review",
      "sub_agent_id": "sub3:sess-parent",
      "started_at": 1.0,
    })
    sent = await log.append({
      "type": "parent_message_sent",
      "task_id": "bg_3",
      "message_id": "m-crash",
      "message": "Use the amended filing",
      "sent_at": 2.0,
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-parent",
      "role": "writer",
    })
    await log.append({
      "type": "attach",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child-old",
      "role": "sub_agent",
    })
    await log.append({
      "type": "user_message",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child-old",
      "role": "sub_agent",
      "content": "Review AAPL",
    })
    assistant = await log.append({
      "type": "assistant_message",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child-old",
      "role": "sub_agent",
      "content_blocks": [{"type": "text", "text": "Used amendment."}],
      "stop_reason": "end_turn",
      "model": "stub-model",
      "parent_message_consumptions": [{
        "task_id": "bg_3",
        "message_id": "m-crash",
        "parent_message_seq": sent.seq,
        "consumer_turn": 1,
      }],
    })

    pending = await reconstruct_parent_messages(log, "bg_3", before_ts=10.0)
    transcript = await reconstruct_messages_for_task(log, "bg_3")
    assert pending == []
    assert transcript[-2]["content"] == (
      "Operator update for this task:\n"
      "- id=m-crash: Use the amended filing"
    )
    audits, _ = await log.query(
      event_types={"parent_message_consumed"},
      order="asc",
    )
    assert audits == []

    runner._role = "sub_agent"
    runner._sub_agent_id = "sub3:sess-parent"
    runner._runner_id = "runner-child-new"
    await runner._materialize_parent_message_consumption_audits()
    await runner._materialize_parent_message_consumption_audits()

    audits, _ = await log.query(
      event_types={"parent_message_consumed"},
      order="asc",
    )
    assert len(audits) == 1
    audit = audits[0]
    assert sent.seq < assistant.seq < audit.seq
    assert audit.event["assistant_message_seq"] == assistant.seq
    assert audit.event["runner_id"] == "runner-child-old"
    assert audit.event["role"] == "sub_agent"
    assert audit.event["sub_agent_id"] == "sub3:sess-parent"
    assert "message" not in audit.event

  _run(_case())


def test_consumed_parent_message_projection_fails_closed_on_duplicate_ack(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "duplicate-consumed-parent-message.jsonl")
    await log.append({
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": "bg_3",
      "task_type": "background",
      "sub_agent_id": "sub3:sess-parent",
      "started_at": 1.0,
    })
    sent = await log.append({
      "type": "parent_message_sent",
      "task_id": "bg_3",
      "message_id": "m1",
      "message": "Use the amendment",
      "sent_at": 2.0,
      "sub_agent_id": "sub3:sess-parent",
    })
    await log.append({
      "type": "attach",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child",
      "role": "sub_agent",
    })
    assistant = await log.append({
      "type": "assistant_message",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child",
      "role": "sub_agent",
      "content_blocks": [{"type": "text", "text": "done"}],
    })
    event = {
      "type": "parent_message_consumed",
      "task_id": "bg_3",
      "message_id": "m1",
      "parent_message_seq": sent.seq,
      "consumer_turn": 1,
      "assistant_message_seq": assistant.seq,
      "consumed_at": 3.0,
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child",
      "role": "sub_agent",
    }
    await log.append(event)
    await log.append(event)

    with pytest.raises(
      TranscriptIntegrityError,
      match="duplicate durable consumption",
    ):
      await reconstruct_messages_for_task(log, "bg_3")

  _run(_case())


def test_parent_message_projection_fails_closed_on_duplicate_unconsumed_sent(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(path=tmp_path / "sessions" / "duplicate-parent-message.jsonl")
    await log.append({
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": "bg_3",
      "task_type": "background",
      "sub_agent_id": "sub3:sess-parent",
      "started_at": 1.0,
    })
    await log.append({
      "type": "attach",
      "sub_agent_id": "sub3:sess-parent",
      "runner_id": "runner-child",
      "role": "sub_agent",
    })
    sent = {
      "type": "parent_message_sent",
      "task_id": "bg_3",
      "message_id": "m1",
      "message": "Use the amendment",
      "sent_at": 2.0,
      "sub_agent_id": "sub3:sess-parent",
    }
    await log.append(sent)
    await log.append(sent)

    with pytest.raises(
      TranscriptIntegrityError,
      match="duplicate durable sent",
    ):
      await reconstruct_parent_messages(log, "bg_3", before_ts=10.0)

  _run(_case())


def test_register_background_task_resume_generates_r_suffix(tmp_path: Path) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)

    async def _handler(_tool_input: dict[str, Any], **_kwargs: Any):
      return _report_task_result(
        "resumed",
        task_id="bg_3_r1",
        workspace_dir=tmp_path,
      ), None

    result, error = await runner._register_background_task(
      capability_bind_receipt=_bind_receipt(),
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


def test_resume_successor_registration_is_single_claim_across_retries(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    runner._max_background_tasks = 1
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()
    handler_calls = 0
    hook_calls = 0

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[TaskResult, None]:
      nonlocal handler_calls
      handler_calls += 1
      handler_started.set()
      await release_handler.wait()
      return _report_task_result(
        "resumed once",
        task_id="bg_claim_root_r1",
        workspace_dir=tmp_path,
      ), None

    def _on_before_start() -> None:
      nonlocal hook_calls
      hook_calls += 1

    async def _register() -> tuple[
      dict[str, Any] | None,
      dict[str, Any] | None,
    ]:
      return await runner._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "resume"},
        handler=_handler,
        agent_name="earnings-review",
        original_task_id="bg_claim_root",
        on_before_start=_on_before_start,
      )

    concurrent = await asyncio.gather(_register(), _register())
    assert [item[1] for item in concurrent] == [None, None]
    assert {
      item[0]["task_id"]
      for item in concurrent
      if item[0] is not None
    } == {"bg_claim_root_r1"}

    await asyncio.wait_for(handler_started.wait(), timeout=1.0)
    running_replay, running_error = await _register()
    assert running_error is None
    assert running_replay is not None
    assert running_replay["task_id"] == "bg_claim_root_r1"
    assert handler_calls == 1
    assert hook_calls == 1

    entry = runner._task_registry.get("bg_claim_root_r1")
    assert entry is not None
    assert entry.asyncio_task is not None
    release_handler.set()
    await entry.asyncio_task
    assert entry.state == TaskState.COMPLETED

    completed_replay, completed_error = await _register()
    assert completed_error is None
    assert completed_replay is not None
    assert completed_replay["task_id"] == "bg_claim_root_r1"
    assert completed_replay["status"] == "completed"
    assert handler_calls == 1
    assert hook_calls == 1

    log = runner._agent_session_log
    assert log is not None
    task_events, _ = await log.query(
      event_types={"task_registered", "task_completed"},
      order="asc",
    )
    matching = [
      log_entry.event
      for log_entry in task_events
      if log_entry.event.get("task_id") == "bg_claim_root_r1"
    ]
    assert [
      event["type"]
      for event in matching
    ] == ["task_registered", "task_completed"]

    restarted = _runner(tmp_path, max_retained_tasks=0)
    restarted._runner_id = "runner-restarted"
    await restarted._rebuild_task_registry_from_log()
    assert restarted._task_registry.get("bg_claim_root_r1") is None
    restarted_handler_calls = 0

    async def _restarted_handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      nonlocal restarted_handler_calls
      restarted_handler_calls += 1
      return {"response": "must not rerun"}, None

    restarted_result, restarted_error = (
      await restarted._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "resume"},
        handler=_restarted_handler,
        original_task_id="bg_claim_root",
      )
    )

    assert restarted_error is None
    assert restarted_result is not None
    assert restarted_result["task_id"] == "bg_claim_root_r1"
    assert restarted_result["status"] == "completed"
    assert restarted_handler_calls == 0
    task_events_after_restart, _ = await log.query(
      event_types={"task_registered", "task_completed"},
      order="asc",
    )
    assert len([
      log_entry
      for log_entry in task_events_after_restart
      if log_entry.event.get("task_id") == "bg_claim_root_r1"
    ]) == 2

  _run(_case())


def test_required_skill_result_recovers_after_append_failure_and_replay(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    lifecycle = _required_skill_lifecycle("skill-run-durable-recovery")
    append_event = runner._append_durable_event
    marker_attempts = 0
    projected: list[dict[str, Any]] = []

    async def _fail_first_result_marker(
      event: dict[str, Any],
    ) -> Any:
      nonlocal marker_attempts
      if event.get("type") == "skill_result_captured":
        marker_attempts += 1
        if marker_attempts == 1:
          raise RuntimeError("skill marker storage unavailable")
      return await append_event(event)

    runner._append_durable_event = _fail_first_result_marker  # type: ignore[method-assign]

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[TaskResult, None]:
      return _report_task_result(
        "completed before marker failure",
        task_id="bg_skill_marker_recovery_r1",
        workspace_dir=tmp_path,
      ), None

    started, start_error = await runner._register_background_task(
      capability_bind_receipt=_bind_receipt(),
      tool_input={"task": "recover lifecycle"},
      handler=_handler,
      agent_name="earnings-review",
      original_task_id="bg_skill_marker_recovery",
      required_skill_lifecycle=lifecycle,
      required_skill_result_event_factory=(
        _required_skill_result_event
      ),
      required_skill_result_projector=projected.append,
    )
    assert start_error is None
    assert started is not None
    entry = runner._task_registry.get(started["task_id"])
    assert entry is not None
    assert entry.asyncio_task is not None
    with pytest.raises(
      RuntimeError,
      match="skill marker storage unavailable",
    ):
      await entry.asyncio_task

    assert entry.state == TaskState.RUNNING
    assert entry.completion_persistence_state == "committed"
    assert entry.pending_final_state == TaskState.COMPLETED
    assert entry.result is None
    assert entry.notification_delivery_state == "not_queued"
    assert entry.required_skill_result_settled is False
    assert projected == []
    log = runner._agent_session_log
    assert log is not None
    before_replay, _ = await log.query(
      event_types={
        "task_registered",
        "task_completed",
        "skill_result_captured",
      },
      order="asc",
    )
    matching_before = [
      item
      for item in before_replay
      if item.event.get("task_id") == entry.task_id
    ]
    assert [item.event["type"] for item in matching_before] == [
      "task_registered",
      "task_completed",
    ]
    assert (
      matching_before[0].event["metadata"][
        "required_skill_lifecycle"
      ]
      == lifecycle
    )

    restarted = _runner(tmp_path)
    restarted._runner_id = "runner-restarted"
    replay_handler_calls = 0

    async def _must_not_rerun(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      nonlocal replay_handler_calls
      replay_handler_calls += 1
      return _report_task_result_payload("must not rerun"), None

    replay_lifecycle = _required_skill_lifecycle(
      "new-request-must-not-replace-durable-lifecycle"
    )

    async def _replay() -> tuple[
      dict[str, Any] | None,
      dict[str, Any] | None,
    ]:
      return await restarted._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "recover lifecycle"},
        handler=_must_not_rerun,
        original_task_id="bg_skill_marker_recovery",
        required_skill_lifecycle=replay_lifecycle,
        required_skill_result_event_factory=(
          _required_skill_result_event
        ),
      )

    replay_results = await asyncio.gather(_replay(), _replay())
    assert all(error is None for _result, error in replay_results)
    assert all(
      result is not None and result["status"] == "completed"
      for result, _error in replay_results
    )
    assert replay_handler_calls == 0

    after_replay, _ = await log.query(
      event_types={
        "task_completed",
        "skill_result_captured",
      },
      order="asc",
    )
    terminal_events = [
      item
      for item in after_replay
      if item.event.get("task_id") == entry.task_id
    ]
    assert [item.event["type"] for item in terminal_events] == [
      "task_completed",
      "skill_result_captured",
    ]
    completed_entry, marker_entry = terminal_events
    assert completed_entry.seq < marker_entry.seq
    assert marker_entry.event["skill_run_id"] == lifecycle["skill_run_id"]
    assert marker_entry.event["skill"] == lifecycle["skill"]
    assert marker_entry.event["ticker"] == lifecycle["ticker"]
    assert marker_attempts == 1

    replay_again, replay_again_error = await _replay()
    assert replay_again_error is None
    assert replay_again is not None
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    assert len([
      item
      for item in markers
      if item.event.get("task_id") == entry.task_id
    ]) == 1

  _run(_case())


def test_required_skill_result_projection_marks_delivery_only_after_success(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    attempts = 0
    projected: list[dict[str, Any]] = []

    async def _projector(event: dict[str, Any]) -> None:
      nonlocal attempts
      attempts += 1
      if attempts == 1:
        raise RuntimeError("injected projection failure")
      projected.append(dict(event))

    entry = TaskEntry(
      task_id="bg_projection_retry",
      task_type="background_agent",
      agent_name="earnings-review",
      state=TaskState.COMPLETED,
      required_skill_result_projector=_projector,
    )
    confirmed = {
      **_required_skill_result_core(),
      "task_id": entry.task_id,
      "skill_run_id": "skill-run-projection-retry",
      "skill": "earnings-review",
      "scope": "ticker",
      "ticker": "PCTY",
      "portfolio_id": None,
      "runner_id": "runner-confirmed",
      "event_schema_version": 1,
    }

    with pytest.raises(RuntimeError, match="injected") as exc_info:
      await runner._project_required_skill_result_event(
        entry,
        confirmed,
      )
    assert exception_traceback_already_logged(exc_info.value) is True
    assert entry.required_skill_result_projected is False

    await runner._project_required_skill_result_event(
      entry,
      confirmed,
    )
    assert entry.required_skill_result_projected is True
    assert projected == [confirmed]

  _run(_case())


def test_required_skill_result_rejects_mutated_whole_event_confirmation(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    lifecycle = _required_skill_lifecycle(
      "skill-run-mutated-confirmation"
    )
    task_id = "bg_mutated_skill_confirmation"
    sub_agent_id = "sub:mutated-skill-confirmation"
    await _append_unsettled_required_skill_task(
      runner,
      task_id=task_id,
      lifecycle=lifecycle,
      sub_agent_id=sub_agent_id,
    )
    entry = TaskEntry(
      task_id=task_id,
      task_type="background",
      agent_name="earnings-review",
      state=TaskState.COMPLETED,
      result=_report_task_result_payload("completed before forged marker"),
      metadata={
        "owner_runner_id": runner._runner_id,
        "owner_role": runner._role,
        "sub_agent_id": sub_agent_id,
        "parent_turn_id": f"turn:{task_id}",
        "call_index": 0,
        "task_type": "background",
        "required_skill_lifecycle": lifecycle,
      },
      completion_persistence_state="committed",
      required_skill_result_event_factory=(
        _required_skill_result_event
      ),
    )
    append_durable = runner._append_durable_event

    async def _append_mutated_marker(
      event: dict[str, Any],
    ) -> Any:
      if event.get("type") == "skill_result_captured":
        event = {
          **event,
          "owner_role": "forged-owner-role",
        }
      return await append_durable(event)

    runner._append_durable_event = _append_mutated_marker  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="envelope mismatch"):
      await runner._persist_required_skill_result_once(entry)
    assert entry.required_skill_result_settled is False

  _run(_case())


def test_task_registry_rebuild_recovers_required_skill_result_once(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    lifecycle = _required_skill_lifecycle(
      "skill-run-rebuild-recovery"
    )
    task_id = "bg_rebuild_skill_result"
    sub_agent_id = "sub:rebuild-skill-result"
    await _append_unsettled_required_skill_task(
      runner,
      task_id=task_id,
      lifecycle=lifecycle,
      sub_agent_id=sub_agent_id,
    )

    restarted = _runner(tmp_path)
    restarted._runner_id = "runner-rebuild-recovery"
    await restarted._rebuild_task_registry_from_log()

    entry = restarted._task_registry.get(task_id)
    assert entry is not None
    assert entry.state == TaskState.COMPLETED
    assert entry.required_skill_result_settled is True
    log = restarted._agent_session_log
    assert log is not None
    events, _ = await log.query(
      event_types={"task_completed", "skill_result_captured"},
      order="asc",
    )
    matching = [
      item
      for item in events
      if item.event.get("task_id") == task_id
    ]
    assert [item.event["type"] for item in matching] == [
      "task_completed",
      "skill_result_captured",
    ]
    assert matching[0].seq < matching[1].seq

    await restarted._rebuild_task_registry_from_log()
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    assert len([
      item
      for item in markers
      if item.event.get("task_id") == task_id
    ]) == 1

  _run(_case())


@pytest.mark.parametrize(
  "mismatch",
  [
    "schema_version",
    "skill_run_id",
    "skill",
    "scope",
    "scope_type",
    "ticker",
    "portfolio_id",
    "task_id",
    "bounds",
    "bool_seq",
    "zero_seq",
    "duplicate",
    "foreign_task_same_run",
    "same_task_foreign_run",
    "duplicate_registration",
    "duplicate_completion",
  ],
)
def test_task_registry_cold_replay_does_not_settle_invalid_skill_marker(
  mismatch: str,
) -> None:
  task_id = "bg_invalid_cold_replay_marker"
  lifecycle = _required_skill_lifecycle(
    "skill-run-invalid-cold-replay-marker"
  )
  marker_task_id = task_id
  marker_skill = lifecycle["skill"]
  marker_scope = lifecycle["scope"]
  marker_ticker = lifecycle["ticker"]
  marker_portfolio_id = lifecycle["portfolio_id"]
  marker_seq = 30
  replay_lifecycle = dict(lifecycle)
  if mismatch == "schema_version":
    replay_lifecycle["schema_version"] = 1
  elif mismatch == "skill_run_id":
    lifecycle["skill_run_id"] = "skill-run-marker-mismatch"
  elif mismatch == "skill":
    marker_skill = "different-skill"
  elif mismatch == "scope":
    marker_scope = "portfolio"
  elif mismatch == "scope_type":
    replay_lifecycle["scope"] = []
  elif mismatch == "ticker":
    marker_ticker = "MSFT"
  elif mismatch == "portfolio_id":
    marker_portfolio_id = "portfolio-mismatch"
  elif mismatch == "task_id":
    marker_task_id = "bg_different_task"
  elif mismatch == "bounds":
    marker_seq = 15
  elif mismatch == "bool_seq":
    marker_seq = True
  elif mismatch == "zero_seq":
    marker_seq = 0

  events = [
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "capability_bind": _bind_receipt(),
      "task_id": task_id,
      "task_type": "background",
      "agent_name": "earnings-review",
      "metadata": {
        "task_type": "background",
        "required_skill_lifecycle": replay_lifecycle,
      },
      "started_at": 1.0,
      "_durable_seq": 10,
    },
    {
      "type": "task_completed",
      "task_id": task_id,
      "task_type": "background",
      "final_state": TaskState.COMPLETED.value,
      "completed_at": 2.0,
      "result": _report_task_result_payload("completed before invalid marker"),
      "error": None,
      "_durable_seq": 20,
    },
    {
      **_required_skill_result_core(),
      "task_id": marker_task_id,
      "skill_run_id": lifecycle["skill_run_id"],
      "skill": marker_skill,
      "scope": marker_scope,
      "ticker": marker_ticker,
      "portfolio_id": marker_portfolio_id,
      "_durable_seq": marker_seq,
    },
  ]
  if mismatch == "duplicate":
    events.append({
      **events[-1],
      "_durable_seq": 31,
    })
  elif mismatch == "foreign_task_same_run":
    events.append({
      **events[-1],
      "task_id": "bg_foreign_same_run",
      "_durable_seq": 31,
    })
  elif mismatch == "same_task_foreign_run":
    events.append({
      **events[-1],
      "skill_run_id": "skill-run-foreign-on-same-task",
      "_durable_seq": 31,
    })
  elif mismatch == "duplicate_registration":
    events.append({
      **events[0],
      "_durable_seq": 11,
    })
  elif mismatch == "duplicate_completion":
    events.append({
      **events[1],
      "_durable_seq": 21,
    })

  registry = TaskRegistry()
  registry.load_from_events(events)

  entry = registry.get(task_id)
  assert entry is not None
  assert entry.required_skill_result_settled is (
    mismatch in {"duplicate_registration", "duplicate_completion"}
  )


def test_task_registry_cold_replay_ignores_unrelated_skill_marker() -> None:
  task_id = "bg_exact_cold_replay_marker"
  lifecycle = _required_skill_lifecycle(
    "skill-run-exact-cold-replay-marker"
  )
  registry = TaskRegistry()
  registry.load_from_events([
    {
      "type": "task_registered",
      "event_schema_version": 2,
      "capability_bind": _bind_receipt(),
      "task_id": task_id,
      "task_type": "background",
      "metadata": {
        "required_skill_lifecycle": lifecycle,
      },
      "started_at": 1.0,
      "_durable_seq": 10,
    },
    {
      "type": "task_completed",
      "task_id": task_id,
      "task_type": "background",
      "final_state": TaskState.COMPLETED.value,
      "completed_at": 2.0,
      "result": _report_task_result_payload("completed before exact marker"),
      "error": None,
      "_durable_seq": 20,
    },
    {
      **_required_skill_result_core(),
      "task_id": task_id,
      "skill_run_id": lifecycle["skill_run_id"],
      "skill": lifecycle["skill"],
      "scope": lifecycle["scope"],
      "ticker": lifecycle["ticker"],
      "portfolio_id": lifecycle["portfolio_id"],
      "_durable_seq": 30,
    },
    {
      **_required_skill_result_core(),
      "task_id": "bg_unrelated_marker",
      "skill_run_id": "skill-run-unrelated-marker",
      "skill": "unrelated-skill",
      "scope": "ticker",
      "ticker": "MSFT",
      "portfolio_id": None,
      "_durable_seq": 31,
    },
  ])

  entry = registry.get(task_id)
  assert entry is not None
  assert entry.required_skill_result_settled is True


def test_writer_recovery_fails_closed_on_mismatched_skill_marker(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    task_id = "bg_writer_mismatched_skill_marker"
    lifecycle = _required_skill_lifecycle(
      "skill-run-writer-mismatched-marker"
    )
    await _append_unsettled_required_skill_task(
      runner,
      task_id=task_id,
      lifecycle=lifecycle,
      sub_agent_id="sub:writer-mismatched-marker",
    )
    await runner._append_durable_event({
      **_required_skill_result_core(),
      "task_id": task_id,
      "skill_run_id": lifecycle["skill_run_id"],
      "skill": lifecycle["skill"],
      "scope": lifecycle["scope"],
      "ticker": "MSFT",
      "portfolio_id": lifecycle["portfolio_id"],
    })

    restarted = _runner(tmp_path)
    restarted._runner_id = "runner-mismatched-marker-recovery"
    with pytest.raises(RuntimeError, match="mismatched ticker"):
      await restarted._rebuild_task_registry_from_log()

    entry = restarted._task_registry.get(task_id)
    assert entry is not None
    assert entry.required_skill_result_settled is False
    log = runner._agent_session_log
    assert log is not None
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    matching = [
      marker
      for marker in markers
      if marker.event.get("task_id") == task_id
    ]
    assert len(matching) == 1
    assert matching[0].event["ticker"] == "MSFT"

  _run(_case())


def test_writer_recovery_fails_closed_on_lifecycle_global_marker_collision(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    task_id = "bg_writer_global_marker_collision"
    lifecycle = _required_skill_lifecycle(
      "skill-run-writer-global-marker-collision"
    )
    await _append_unsettled_required_skill_task(
      runner,
      task_id=task_id,
      lifecycle=lifecycle,
      sub_agent_id="sub:writer-global-marker-collision",
    )
    exact_marker = {
      **_required_skill_result_core(),
      "task_id": task_id,
      "skill_run_id": lifecycle["skill_run_id"],
      "skill": lifecycle["skill"],
      "scope": lifecycle["scope"],
      "ticker": lifecycle["ticker"],
      "portfolio_id": lifecycle["portfolio_id"],
    }
    await runner._append_durable_event(exact_marker)
    await runner._append_durable_event({
      **exact_marker,
      "task_id": "bg_foreign_global_marker_collision",
    })

    restarted = _runner(tmp_path)
    restarted._runner_id = "runner-global-marker-collision"
    with pytest.raises(
      RuntimeError,
      match="duplicate or conflicting required skill results",
    ):
      await restarted._rebuild_task_registry_from_log()

    entry = restarted._task_registry.get(task_id)
    assert entry is not None
    assert entry.required_skill_result_settled is False
    log = runner._agent_session_log
    assert log is not None
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    matching = [
      marker
      for marker in markers
      if (
        marker.event.get("skill_run_id")
        == lifecycle["skill_run_id"]
      )
    ]
    assert len(matching) == 2

  _run(_case())


@pytest.mark.parametrize(
  "duplicate_event_type",
  ["task_registered", "task_completed"],
)
def test_writer_recovery_fails_closed_on_duplicate_task_bounds(
  tmp_path: Path,
  duplicate_event_type: str,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    task_id = f"bg_writer_duplicate_{duplicate_event_type}"
    lifecycle = _required_skill_lifecycle(
      f"skill-run-writer-duplicate-{duplicate_event_type}"
    )
    sub_agent_id = f"sub:writer-duplicate-{duplicate_event_type}"
    await _append_unsettled_required_skill_task(
      runner,
      task_id=task_id,
      lifecycle=lifecycle,
      sub_agent_id=sub_agent_id,
    )
    if duplicate_event_type == "task_registered":
      await runner._append_durable_event({
        "type": "task_registered",
        "capability_bind": _bind_receipt(),
        "task_id": task_id,
        "task_type": "background",
        "agent_name": "earnings-review",
        "sub_agent_id": sub_agent_id,
        "metadata": {
          "sub_agent_id": sub_agent_id,
          "task_type": "background",
          "required_skill_lifecycle": lifecycle,
        },
        "started_at": 1.5,
      })
    else:
      await runner._append_durable_event({
        "type": "task_completed",
        "task_id": task_id,
        "task_type": "background",
        "sub_agent_id": sub_agent_id,
        "final_state": TaskState.COMPLETED.value,
        "completed_at": 2.5,
        "result": _report_task_result_payload("duplicate completion"),
        "error": None,
      })
    await runner._append_durable_event({
      **_required_skill_result_core(),
      "task_id": task_id,
      "skill_run_id": lifecycle["skill_run_id"],
      "skill": lifecycle["skill"],
      "scope": lifecycle["scope"],
      "ticker": lifecycle["ticker"],
      "portfolio_id": lifecycle["portfolio_id"],
    })

    restarted = _runner(tmp_path)
    restarted._runner_id = (
      f"runner-duplicate-{duplicate_event_type}-recovery"
    )
    with pytest.raises(
      RuntimeError,
      match="conflicting durable",
    ):
      await restarted._rebuild_task_registry_from_log()

    entry = restarted._task_registry.get(task_id)
    assert entry is None
    log = runner._agent_session_log
    assert log is not None
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    matching = [
      marker
      for marker in markers
      if marker.event.get("task_id") == task_id
    ]
    assert len(matching) == 1

  _run(_case())


def test_sub_agent_rebuild_does_not_repair_required_skill_result(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    seeder = _runner(tmp_path)
    task_id = "bg_sub_agent_read_only_rebuild"
    lifecycle = _required_skill_lifecycle(
      "skill-run-sub-agent-read-only-rebuild"
    )
    await _append_unsettled_required_skill_task(
      seeder,
      task_id=task_id,
      lifecycle=lifecycle,
      sub_agent_id="sub:read-only-rebuild",
    )
    log = seeder._agent_session_log
    assert log is not None
    latest_seq_before = await log.latest_seq()

    sub_agent = _runner(tmp_path)
    sub_agent._role = "sub_agent"
    sub_agent._sub_agent_id = "sub:read-only-rebuild"
    sub_agent._runner_id = "runner-sub-agent-read-only"

    async def _forbidden_repair(_entry: TaskEntry) -> bool:
      raise AssertionError("sub-agent rebuild attempted durable repair")

    sub_agent._ensure_required_skill_result_settled = (  # type: ignore[method-assign]
      _forbidden_repair
    )
    await sub_agent._rebuild_task_registry_from_log()

    assert await log.latest_seq() == latest_seq_before
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    assert markers == []
    entry = sub_agent._task_registry.get(task_id)
    assert entry is not None
    assert entry.state == TaskState.COMPLETED
    assert entry.required_skill_result_settled is False
    assert sub_agent._task_registry_rebuilt is True

  _run(_case())


def test_writer_and_sub_agent_concurrent_rebuild_has_one_repair_owner(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    seeder = _runner(tmp_path)
    task_id = "bg_concurrent_writer_repair"
    lifecycle = _required_skill_lifecycle(
      "skill-run-concurrent-writer-repair"
    )
    await _append_unsettled_required_skill_task(
      seeder,
      task_id=task_id,
      lifecycle=lifecycle,
      sub_agent_id="sub:concurrent-writer-repair",
    )

    writer = _runner(tmp_path)
    writer._runner_id = "runner-writer-repair"
    sub_agent = _runner(tmp_path)
    sub_agent._role = "sub_agent"
    sub_agent._sub_agent_id = "sub:concurrent-writer-repair"
    sub_agent._runner_id = "runner-sub-agent-concurrent"

    writer_absence_observed = asyncio.Event()
    release_writer_repair = asyncio.Event()
    original_writer_lookup = (
      writer._durable_required_skill_result_event
    )
    hold_first_absence = True

    async def _hold_writer_after_absence(
      entry: TaskEntry,
      persisted_lifecycle: dict[str, Any],
    ) -> dict[str, Any] | None:
      nonlocal hold_first_absence
      existing = await original_writer_lookup(
        entry,
        persisted_lifecycle,
      )
      if hold_first_absence and existing is None:
        hold_first_absence = False
        writer_absence_observed.set()
        await release_writer_repair.wait()
      return existing

    writer._durable_required_skill_result_event = (  # type: ignore[method-assign]
      _hold_writer_after_absence
    )
    writer_rebuild = asyncio.create_task(
      writer._rebuild_task_registry_from_log()
    )
    await writer_absence_observed.wait()

    sub_agent_rebuild = asyncio.create_task(
      sub_agent._rebuild_task_registry_from_log()
    )
    sub_agent_outcome = await asyncio.gather(
      sub_agent_rebuild,
      return_exceptions=True,
    )
    log = writer._agent_session_log
    assert log is not None
    markers_during_writer_pause, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )

    release_writer_repair.set()
    writer_outcome = await asyncio.gather(
      writer_rebuild,
      return_exceptions=True,
    )

    assert sub_agent_outcome == [None]
    assert markers_during_writer_pause == []
    assert writer_outcome == [None]
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    matching = [
      item
      for item in markers
      if item.event.get("task_id") == task_id
    ]
    assert len(matching) == 1
    writer_entry = writer._task_registry.get(task_id)
    assert writer_entry is not None
    assert writer_entry.required_skill_result_settled is True
    sub_agent_entry = sub_agent._task_registry.get(task_id)
    assert sub_agent_entry is not None
    assert sub_agent_entry.required_skill_result_settled is False

  _run(_case())


def test_required_skill_result_recovery_is_bounded_to_task_interval(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    shared_sub_agent_id = "sub:reused-across-resume"
    old_task_id = "bg_shared_history"
    successor_task_id = "bg_shared_history_r1"
    lifecycle = _required_skill_lifecycle(
      "skill-run-successor-bounded-history"
    )

    await runner._append_durable_event({
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": old_task_id,
      "task_type": "background",
      "agent_name": "earnings-review",
      "sub_agent_id": shared_sub_agent_id,
      "metadata": {
        "sub_agent_id": shared_sub_agent_id,
        "task_type": "background",
      },
      "started_at": 1.0,
    })
    await runner._append_durable_event({
      "type": "tool_call_complete",
      "task_id": old_task_id,
      "sub_agent_id": shared_sub_agent_id,
      "tool_name": "fms_old_history",
      "result": {
        "subcommand": "old_history",
        "mutation_mode": "proposal",
        "proposal_id": "OLD-PROPOSAL",
        "artifact_ref": "artifacts/old-history.json",
        "ticker": "OLD",
      },
    })
    await runner._append_durable_event({
      "type": "task_completed",
      "task_id": old_task_id,
      "task_type": "background",
      "sub_agent_id": shared_sub_agent_id,
      "final_state": TaskState.COMPLETED.value,
      "completed_at": 2.0,
      "result": _report_task_result_payload("old completed run"),
      "error": None,
    })
    await runner._append_durable_event({
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": successor_task_id,
      "task_type": "background",
      "agent_name": "earnings-review",
      "original_task_id": old_task_id,
      "sub_agent_id": shared_sub_agent_id,
      "metadata": {
        "sub_agent_id": shared_sub_agent_id,
        "task_type": "background",
        "original_task_id": old_task_id,
        "required_skill_lifecycle": lifecycle,
      },
      "started_at": 3.0,
    })
    await runner._append_durable_event({
      "type": "task_completed",
      "task_id": successor_task_id,
      "task_type": "background",
      "original_task_id": old_task_id,
      "sub_agent_id": shared_sub_agent_id,
      "final_state": TaskState.COMPLETED.value,
      "completed_at": 4.0,
      "result": _report_task_result_payload("successor completed run"),
      "error": None,
    })

    restarted = _runner(tmp_path)
    restarted._runner_id = "runner-bounded-recovery"
    await restarted._rebuild_task_registry_from_log()

    log = restarted._agent_session_log
    assert log is not None
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    successor_markers = [
      item.event
      for item in markers
      if item.event.get("task_id") == successor_task_id
    ]
    assert len(successor_markers) == 1
    marker = successor_markers[0]
    assert marker["skill_run_id"] == lifecycle["skill_run_id"]
    assert marker["ticker"] == "PCTY"
    assert marker["fms_results"] == []
    assert marker["proposal_ids"] == []
    assert marker["artifact_refs"] == []

  _run(_case())


def test_required_skill_result_owned_append_survives_repeated_cancellation(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    lifecycle = _required_skill_lifecycle(
      "skill-run-cancel-safe-settlement"
    )
    entry = runner._task_registry.register(
      "background_agent",
      agent_name="earnings-review",
      task_id="bg_cancel_safe_skill_result",
    )
    entry.state = TaskState.COMPLETED
    entry.result = _report_task_result_payload("settle despite cancellation")
    entry.metadata.update({
      "owner_runner_id": runner._runner_id,
      "owner_role": runner._role,
      "sub_agent_id": "sub:cancel-safe-result",
      "parent_turn_id": "turn-cancel-safe-result",
      "call_index": 0,
      "task_type": "background",
      "required_skill_lifecycle": lifecycle,
    })
    entry.registration_persistence_state = "committed"
    entry.completion_persistence_state = "committed"
    entry.required_skill_result_event_factory = (
      _required_skill_result_event
    )
    await runner._append_durable_event({
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": entry.task_id,
      "task_type": "background",
      "agent_name": entry.agent_name,
      "metadata": dict(entry.metadata),
      "started_at": entry.started_at,
    })
    await runner._append_durable_event({
      "type": "task_completed",
      "task_id": entry.task_id,
      "final_state": TaskState.COMPLETED.value,
      "completed_at": entry.started_at + 1,
      "result": entry.result,
      "error": None,
    })

    append_event = runner._append_durable_event
    append_entered = asyncio.Event()
    release_append = asyncio.Event()
    append_cancelled = False

    async def _blocked_marker(event: dict[str, Any]) -> Any:
      nonlocal append_cancelled
      if event.get("type") == "skill_result_captured":
        append_entered.set()
        try:
          await release_append.wait()
        except asyncio.CancelledError:
          append_cancelled = True
          raise
      return await append_event(event)

    runner._append_durable_event = _blocked_marker  # type: ignore[method-assign]
    waiter = asyncio.create_task(
      runner._ensure_required_skill_result_settled(entry)
    )
    await append_entered.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)

    assert waiter.done() is False
    assert append_cancelled is False
    release_append.set()
    with pytest.raises(asyncio.CancelledError):
      await waiter
    assert append_cancelled is False
    assert entry.required_skill_result_settled is True

    log = runner._agent_session_log
    assert log is not None
    markers, _ = await log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
    matching = [
      item
      for item in markers
      if item.event.get("task_id") == entry.task_id
    ]
    assert len(matching) == 1
    assert matching[0].event["skill_run_id"] == lifecycle["skill_run_id"]

  _run(_case())


def test_resume_successor_retries_failed_registration_without_ghost(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    runner._max_background_tasks = 1
    append_event = runner._append_durable_event
    registration_attempts = 0
    handler_calls = 0
    hook_calls = 0

    async def _fail_first_registration(
      event: dict[str, Any],
    ) -> Any:
      nonlocal registration_attempts
      if (
        event.get("type") == "task_registered"
        and event.get("task_id") == "bg_registration_retry_r1"
      ):
        registration_attempts += 1
        if registration_attempts == 1:
          raise RuntimeError("registration unavailable")
      return await append_event(event)

    runner._append_durable_event = _fail_first_registration  # type: ignore[method-assign]

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[TaskResult, None]:
      nonlocal handler_calls
      handler_calls += 1
      return _report_task_result(
        "resumed after retry",
        task_id="bg_registration_retry_r1",
        workspace_dir=tmp_path,
      ), None

    def _failing_hook() -> None:
      nonlocal hook_calls
      hook_calls += 1
      raise RuntimeError("non-fatal hook failure")

    with pytest.raises(RuntimeError, match="registration unavailable"):
      await runner._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "resume"},
        handler=_handler,
        original_task_id="bg_registration_retry",
        on_before_start=_failing_hook,
      )

    ghost = runner._task_registry.get("bg_registration_retry_r1")
    assert ghost is not None
    assert ghost.state == TaskState.PENDING
    assert ghost.asyncio_task is None
    assert ghost.registration_persistence_state == "uncertain"
    assert runner._task_registry.admission_count == 1

    result, error = await runner._register_background_task(
      capability_bind_receipt=_bind_receipt(),
      tool_input={"task": "resume"},
      handler=_handler,
      original_task_id="bg_registration_retry",
      on_before_start=_failing_hook,
    )

    assert error is None
    assert result is not None
    assert result["task_id"] == "bg_registration_retry_r1"
    entry = runner._task_registry.get(result["task_id"])
    assert entry is ghost
    assert entry.asyncio_task is not None
    await entry.asyncio_task
    assert entry.state == TaskState.COMPLETED
    assert entry.registration_persistence_state == "committed"
    assert registration_attempts == 2
    assert handler_calls == 1
    assert hook_calls == 1

    registered, _ = await runner._agent_session_log.query(
      event_types={"task_registered"},
      order="asc",
    )
    assert len([
      log_entry
      for log_entry in registered
      if log_entry.event.get("task_id") == entry.task_id
    ]) == 1

  _run(_case())


def test_pending_resume_registration_reserves_capacity_without_durable_ghost(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    runner._max_background_tasks = 1
    source = runner._task_registry.register(
      "background_agent",
      agent_name="earnings-review",
      task_id="bg_capacity_source",
    )
    source.state = TaskState.INTERRUPTED
    await runner._append_durable_event(
      {
        "type": "task_registered",
        "capability_bind": _bind_receipt(),
        "task_id": source.task_id,
        "task_type": source.task_type,
        "agent_name": source.agent_name,
        "started_at": source.started_at,
      }
    )

    append_event = runner._append_durable_event
    registration_started = asyncio.Event()
    release_registration = asyncio.Event()
    side_effects: list[str] = []

    async def _blocked_registration(event: dict[str, Any]) -> Any:
      if (
        event.get("type") == "task_registered"
        and event.get("task_id") == "bg_capacity_source_r1"
      ):
        registration_started.set()
        await release_registration.wait()
      return await append_event(event)

    runner._append_durable_event = _blocked_registration  # type: ignore[method-assign]

    async def _resume_handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[TaskResult, None]:
      side_effects.append("resume_handler")
      return _report_task_result(
        "resumed within reserved capacity",
        task_id="bg_capacity_source_r1",
        workspace_dir=tmp_path,
      ), None

    async def _competing_handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      side_effects.append("competing_handler")
      return _report_task_result_payload("must not run"), None

    caller = asyncio.create_task(
      runner._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "resume"},
        handler=_resume_handler,
        original_task_id=source.task_id,
        on_before_start=lambda: side_effects.append("resume_start"),
        on_complete=lambda _entry: side_effects.append("resume_capture"),
        validate_resume_source=True,
      )
    )
    await registration_started.wait()

    successor = runner._task_registry.get("bg_capacity_source_r1")
    assert successor is not None
    assert successor.state == TaskState.PENDING
    assert runner._task_registry.admission_count == 1

    competing_result, competing_error = (
      await runner._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "competing"},
        handler=_competing_handler,
        on_before_start=lambda: side_effects.append("competing_start"),
        on_complete=lambda _entry: side_effects.append("competing_capture"),
      )
    )
    assert competing_result is None
    assert competing_error is not None
    assert competing_error["code"] == "max_background_tasks"
    assert runner._task_registry.get("bg_0") is None
    assert side_effects == []

    release_registration.set()
    started_result, started_error = await caller
    assert started_error is None
    assert started_result is not None
    assert started_result["task_id"] == successor.task_id
    assert successor.asyncio_task is not None
    await successor.asyncio_task

    assert successor.state == TaskState.COMPLETED
    assert successor.completion_persistence_state == "committed"
    assert runner._task_registry.admission_count == 0
    assert side_effects == [
      "resume_start",
      "resume_handler",
      "resume_capture",
    ]

    log = runner._agent_session_log
    assert log is not None
    events, _ = await log.query(
      event_types={"task_registered", "task_completed"},
      order="asc",
    )
    matching = [
      log_entry.event
      for log_entry in events
      if log_entry.event.get("task_id") == successor.task_id
    ]
    assert [event["type"] for event in matching] == [
      "task_registered",
      "task_completed",
    ]

    restarted = _runner(tmp_path)
    restarted._runner_id = "runner-restarted"
    restarted._max_background_tasks = 1
    await restarted._rebuild_task_registry_from_log()
    restarted_source = restarted._task_registry.get(source.task_id)
    restarted_successor = restarted._task_registry.get(successor.task_id)
    assert restarted_source is not None
    assert restarted_source.state == TaskState.INTERRUPTED
    assert restarted_successor is not None
    assert restarted_successor.state == TaskState.COMPLETED
    assert restarted._task_registry.admission_count == 0

    restarted_handler_calls = 0

    async def _restarted_handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      nonlocal restarted_handler_calls
      restarted_handler_calls += 1
      return _report_task_result_payload("must not rerun"), None

    replay, replay_error = await restarted._register_background_task(
      capability_bind_receipt=_bind_receipt(),
      tool_input={"task": "resume"},
      handler=_restarted_handler,
      original_task_id=source.task_id,
      validate_resume_source=True,
    )
    assert replay_error is None
    assert replay is not None
    assert replay["task_id"] == successor.task_id
    assert replay["status"] == "completed"
    assert restarted_handler_calls == 0

  _run(_case())


def test_resume_successor_initialization_survives_caller_cancellation(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    append_event = runner._append_durable_event
    append_started = asyncio.Event()
    release_append = asyncio.Event()
    handler_calls = 0

    async def _blocked_registration(event: dict[str, Any]) -> Any:
      if (
        event.get("type") == "task_registered"
        and event.get("task_id") == "bg_cancel_registration_r1"
      ):
        append_started.set()
        await release_append.wait()
      return await append_event(event)

    runner._append_durable_event = _blocked_registration  # type: ignore[method-assign]

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      nonlocal handler_calls
      handler_calls += 1
      return _report_task_result_payload("resumed after cancellation"), None

    caller = asyncio.create_task(
      runner._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "resume"},
        handler=_handler,
        original_task_id="bg_cancel_registration",
      )
    )
    await append_started.wait()
    caller.cancel("caller-stopped")
    with pytest.raises(asyncio.CancelledError) as cancelled:
      await caller
    assert cancelled.value.args == ("caller-stopped",)

    entry = runner._task_registry.get("bg_cancel_registration_r1")
    assert entry is not None
    assert entry.initialization_task is not None
    assert not entry.initialization_task.done()
    release_append.set()
    started_result, started_error = await entry.initialization_task
    assert started_error is None
    assert started_result is not None
    assert started_result["task_id"] == entry.task_id
    assert entry.asyncio_task is not None
    await entry.asyncio_task
    assert entry.state == TaskState.COMPLETED
    assert handler_calls == 1

    replay, replay_error = await runner._register_background_task(
      capability_bind_receipt=_bind_receipt(),
      tool_input={"task": "resume"},
      handler=_handler,
      original_task_id="bg_cancel_registration",
    )
    assert replay_error is None
    assert replay is not None
    assert replay["status"] == "completed"
    assert handler_calls == 1

  _run(_case())


def test_resume_successor_does_not_start_if_source_settles_during_registration(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    source = runner._task_registry.register(
      "background_agent",
      agent_name="earnings-review",
      task_id="bg_source_settles",
    )
    source.state = TaskState.INTERRUPTED
    append_event = runner._append_durable_event
    registration_started = asyncio.Event()
    release_registration = asyncio.Event()
    side_effects: list[str] = []

    async def _blocked_registration(event: dict[str, Any]) -> Any:
      if (
        event.get("type") == "task_registered"
        and event.get("task_id") == "bg_source_settles_r1"
      ):
        registration_started.set()
        await release_registration.wait()
      return await append_event(event)

    runner._append_durable_event = _blocked_registration  # type: ignore[method-assign]

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      side_effects.append("handler")
      return _report_task_result_payload("must not run"), None

    caller = asyncio.create_task(
      runner._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "resume"},
        handler=_handler,
        original_task_id=source.task_id,
        on_before_start=lambda: side_effects.append("start"),
        on_complete=lambda _entry: side_effects.append("capture"),
        validate_resume_source=True,
      )
    )
    await registration_started.wait()
    runner._task_registry.finalize_interrupted(
      source.task_id,
      TaskState.COMPLETED,
      result=_report_task_result_payload("original completed"),
    )
    release_registration.set()

    result, error = await caller

    assert result is None
    assert error is not None
    assert error["code"] == "not_interrupted"
    successor = runner._task_registry.get("bg_source_settles_r1")
    assert successor is not None
    assert successor.state == TaskState.FAILED
    assert successor.asyncio_task is None
    assert successor.result is not None
    assert successor.result["reason"] == "resume_abandoned"
    assert successor.completion_persistence_state == "committed"
    assert side_effects == []

    log = runner._agent_session_log
    assert log is not None
    events, _ = await log.query(
      event_types={"task_registered", "task_completed"},
      order="asc",
    )
    matching = [
      log_entry.event
      for log_entry in events
      if log_entry.event.get("task_id") == successor.task_id
    ]
    assert [event["type"] for event in matching] == [
      "task_registered",
      "task_completed",
    ]
    assert matching[-1]["final_state"] == "failed"
    assert matching[-1]["result"]["reason"] == "resume_abandoned"

  _run(_case())


def test_shutdown_owns_hung_resume_initialization_without_late_worker(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    runner._background_cancel_drain_timeout_seconds = 0.01
    append_event = runner._append_durable_event
    append_started = asyncio.Event()
    release_append = asyncio.Event()
    handler_calls = 0
    lifecycle = _required_skill_lifecycle(
      "skill-run-pre-worker-termination"
    )
    projected: list[dict[str, Any]] = []

    async def _blocked_registration(event: dict[str, Any]) -> Any:
      if (
        event.get("type") == "task_registered"
        and event.get("task_id") == "bg_shutdown_registration_r1"
      ):
        append_started.set()
        await release_append.wait()
      return await append_event(event)

    runner._append_durable_event = _blocked_registration  # type: ignore[method-assign]

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      nonlocal handler_calls
      handler_calls += 1
      return _report_task_result_payload("must not run"), None

    caller = asyncio.create_task(
      runner._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input={"task": "resume"},
        handler=_handler,
        original_task_id="bg_shutdown_registration",
        required_skill_lifecycle=lifecycle,
        required_skill_result_event_factory=(
          _required_skill_result_event
        ),
        required_skill_result_projector=projected.append,
      )
    )
    await append_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
      await caller

    await runner._shutdown_background_tasks(was_cancelled=True)
    entry = runner._task_registry.get("bg_shutdown_registration_r1")
    assert entry is not None
    assert entry.state == TaskState.PENDING
    assert entry.termination_intent == "cancelled"
    assert handler_calls == 0

    release_append.set()
    assert entry.initialization_task is not None
    await entry.initialization_task

    assert entry.state == TaskState.KILLED
    assert entry.result is not None
    assert entry.result["reason"] == "cancelled"
    assert handler_calls == 0
    assert entry.required_skill_result_settled is True
    assert len(projected) == 1
    assert projected[0]["task_id"] == entry.task_id
    assert projected[0]["skill_run_id"] == lifecycle["skill_run_id"]
    log = runner._agent_session_log
    assert log is not None
    events, _ = await log.query(
      event_types={
        "task_registered",
        "task_completed",
        "skill_result_captured",
      },
      order="asc",
    )
    matching = [
      log_entry
      for log_entry in events
      if log_entry.event.get("task_id") == entry.task_id
    ]
    assert [item.event["type"] for item in matching] == [
      "task_registered",
      "task_completed",
      "skill_result_captured",
    ]
    assert matching[1].seq < matching[2].seq
    assert (
      matching[0].event["metadata"]["required_skill_lifecycle"]
      == lifecycle
    )

  _run(_case())


def test_resume_successor_reconciles_registration_failure_after_fsync(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    log = runner._agent_session_log
    assert log is not None
    update_manifest = log._update_manifest_latest_seq_locked
    fail_after_fsync = True
    handler_calls = 0

    def _fail_once(seq: int) -> None:
      nonlocal fail_after_fsync
      if fail_after_fsync:
        fail_after_fsync = False
        raise RuntimeError("manifest update failed after fsync")
      update_manifest(seq)

    monkeypatch.setattr(
      log,
      "_update_manifest_latest_seq_locked",
      _fail_once,
    )

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      nonlocal handler_calls
      handler_calls += 1
      return _report_task_result_payload("reconciled registration"), None

    result, error = await runner._register_background_task(
      capability_bind_receipt=_bind_receipt(),
      tool_input={"task": "resume"},
      handler=_handler,
      original_task_id="bg_registration_fsync",
    )

    assert error is None
    assert result is not None
    entry = runner._task_registry.get("bg_registration_fsync_r1")
    assert entry is not None
    assert entry.asyncio_task is not None
    await entry.asyncio_task
    assert entry.state == TaskState.COMPLETED
    assert entry.registration_persistence_state == "committed"
    assert handler_calls == 1

    events, _ = await log.query(
      event_types={"task_registered", "task_completed"},
      order="asc",
    )
    matching = [
      log_entry.event["type"]
      for log_entry in events
      if log_entry.event.get("task_id") == entry.task_id
    ]
    assert matching == ["task_registered", "task_completed"]

  _run(_case())


def test_resume_source_reference_survives_retention_eviction(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path, max_retained_tasks=0)
    runner._runner_id = "runner-test"
    task_id = "bg_retention_source"
    await _append_interrupted_skill_task(
      runner,
      task_id=task_id,
      agent_name="earnings-review",
      user_message="Ticker: AAPL\nResume after restart.",
    )
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    resume_calls = 0

    async def _resume_sub_agent(**kwargs: Any):
      nonlocal resume_calls
      resume_calls += 1
      return await _successful_resume_task_result(
        kwargs,
        "resumed after source eviction",
        workspace_dir=tmp_path,
      ), None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    started, error = await handler({"task_id": task_id})

    assert error is None
    assert started is not None
    successor = runner._task_registry.get(started["task_id"])
    assert successor is not None
    assert runner._task_registry.get(task_id) is None
    assert successor.asyncio_task is not None
    await successor.asyncio_task
    assert successor.state == TaskState.COMPLETED
    assert resume_calls == 1

  _run(_case())


def test_resume_successor_collision_returns_structured_error(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._task_registry.register(
      "background_agent",
      task_id="bg_collision_r1",
      original_task_id="bg_other",
    )
    side_effects: list[str] = []

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any], None]:
      side_effects.append("handler")
      return _report_task_result_payload("must not run"), None

    result, error = await runner._register_background_task(
      capability_bind_receipt=_bind_receipt(),
      tool_input={"task": "resume"},
      handler=_handler,
      original_task_id="bg_collision",
      on_before_start=lambda: side_effects.append("hook"),
    )

    assert result is None
    assert error is not None
    assert error["code"] == "resume_successor_conflict"
    assert "original_task_id='bg_other'" in error["message"]
    assert "requested original_task_id='bg_collision'" in error["message"]
    assert side_effects == []
    assert len(runner._task_registry.list_tasks()) == 1

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
      return _report_task_result_payload("unused"), None

    result, error = await runner._register_background_task(
      capability_bind_receipt=_bind_receipt(),
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
    runner._task_registry.transition(entry.task_id, TaskState.COMPLETED, result=_report_task_result_payload("done"))
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "earnings-review.md").write_text(
      "---\nagent_callable: true\nagent_description: Earnings.\nresumable: true\n---\nPrompt",
      encoding="utf-8",
    )
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": "bg_3"})

    assert result is None
    assert error["code"] == "not_interrupted"

  _run(_case())


@pytest.mark.parametrize(
  "terminal_state",
  [TaskState.COMPLETED, TaskState.FAILED, TaskState.KILLED],
)
def test_resume_handler_rejects_evicted_durable_terminal_task(
  tmp_path: Path,
  terminal_state: TaskState,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    task_id = f"bg_evicted_{terminal_state.value}"
    await _append_interrupted_skill_task(
      runner,
      task_id=task_id,
      agent_name="earnings-review",
      user_message="Original task already settled.",
    )
    log = runner._agent_session_log
    assert log is not None
    await log.append({
      "type": "task_completed",
      "task_id": task_id,
      "final_state": terminal_state.value,
      "completed_at": 2.0,
      "result": _report_task_result_payload("durably terminal"),
      "error": None,
    })
    runner._task_registry._tasks.pop(task_id)
    runner._task_registry_rebuilt = True
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": task_id})

    assert result is None
    assert error is not None
    assert error["code"] == "not_interrupted"
    assert runner._task_registry.get(task_id) is None
    assert runner._task_registry.get(f"{task_id}_r1") is None

  _run(_case())


def test_resume_handler_waits_for_same_run_uncertain_completion(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    runner._background_completion_persist_timeout_seconds = 0.01
    task_id = "bg_completion_pending"
    await _append_interrupted_skill_task(
      runner,
      task_id=task_id,
      agent_name="earnings-review",
      user_message="Ticker: AAPL\nFinish the original work.",
    )
    original = runner._task_registry.get(task_id)
    assert original is not None
    original.state = TaskState.RUNNING
    append_event = runner._append_durable_event
    append_started = asyncio.Event()
    release_append = asyncio.Event()

    async def _blocked_completion(event: dict[str, Any]) -> Any:
      if (
        event.get("type") == "task_completed"
        and event.get("task_id") == task_id
      ):
        append_started.set()
        await release_append.wait()
      return await append_event(event)

    runner._append_durable_event = _blocked_completion  # type: ignore[method-assign]

    async def _original_handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[TaskResult, None]:
      assert original.admitted_task is not None
      return _successful_admitted_task_result(
        original.admitted_task,
        "original completed",
        workspace_dir=tmp_path,
      ), None

    original_worker = asyncio.create_task(
      runner._run_background_agent(
        original,
        _original_handler,
        {},
        0,
      )
    )
    original.asyncio_task = original_worker
    await append_started.wait()
    with pytest.raises(asyncio.TimeoutError):
      await original_worker

    assert original.state == TaskState.INTERRUPTED
    assert original.completion_finalizer_detached is True
    assert original.completion_persistence_state == "uncertain"

    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    parent_log = EventLog()
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await asyncio.wait_for(
      handler(
        {"task_id": task_id},
        tool_ctx=SimpleNamespace(
          tool_call_id="resume-pending",
          emit=parent_log.append,
        ),
      ),
      timeout=0.2,
    )

    assert result is None
    assert error is not None
    assert error["code"] == "completion_pending"
    assert runner._task_registry.get(f"{task_id}_r1") is None
    assert parent_log.entries == []

    release_append.set()
    for _ in range(100):
      if original.state == TaskState.COMPLETED:
        break
      await asyncio.sleep(0)

    assert original.state == TaskState.COMPLETED
    assert original.task_result == _successful_admitted_task_result(
      original.admitted_task,
      "original completed",
      workspace_dir=tmp_path,
    )
    assert runner._task_registry.get(f"{task_id}_r1") is None
    assert parent_log.entries == []

  _run(_case())


def test_resume_handler_accepts_responses_log_after_cutover(tmp_path: Path) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_marked",
      agent_name="earnings-review",
      user_message="Resume marked history.",
    )
    assert runner._agent_session_log is not None
    await runner._agent_session_log.append(
      {
        "type": "assistant_message",
        "task_id": "bg_marked",
        "sub_agent_id": "sub:bg_marked",
        "role": "sub_agent",
        "content_blocks": [{"type": "text", "text": "prior", "signature": TEXT_SIGNATURE_MARKER}],
      }
    )
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")

    async def _resume_sub_agent(**kwargs: Any):
      return await _successful_resume_task_result(
        kwargs,
        "resumed marked history",
        workspace_dir=tmp_path,
      ), None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": "bg_marked"})

    assert error is None
    assert result is not None
    assert result["task_id"] == "bg_marked_r1"
    successor = runner._task_registry.get(result["task_id"])
    assert successor is not None
    assert successor.original_task_id == "bg_marked"
    assert successor.asyncio_task is not None
    await successor.asyncio_task
    assert successor.state == TaskState.COMPLETED

  _run(_case())


def test_resume_handler_rejects_non_resumable_skill(tmp_path: Path) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_3",
      agent_name="email-responder",
      user_message="Resume email response.",
      resumable=False,
    )
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "email-responder.md").write_text(
      "---\nagent_callable: true\nagent_description: Email.\n---\nPrompt",
      encoding="utf-8",
    )
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": "bg_3"})

    abandoned = _assert_resume_abandoned(
      result,
      error,
      code="not_resumable",
    )
    assert runner._task_registry.get("bg_3").state == TaskState.FAILED
    notifications = runner._notification_queue.peek()
    assert len(notifications) == 1
    assert notifications[0].event == "failed"
    abandoned_payload = abandoned.model_dump(mode="json")
    assert notifications[0].payload == abandoned_payload
    log = runner._agent_session_log
    assert log is not None
    completed, _ = await log.query(
      event_types={"task_completed"},
      order="asc",
    )
    matching = [
      log_entry.event
      for log_entry in completed
      if log_entry.event.get("task_id") == "bg_3"
    ]
    assert len(matching) == 1
    assert matching[0]["final_state"] == "failed"
    assert matching[0]["result"] == abandoned_payload

    replayed = TaskRegistry()
    registered, _ = await log.query(
      event_types={"task_registered", "task_completed"},
      order="asc",
    )
    replayed.load_from_events([log_entry.event for log_entry in registered])
    replayed_entry = replayed.get("bg_3")
    assert replayed_entry is not None
    assert replayed_entry.state == TaskState.FAILED
    assert replayed_entry.result == abandoned_payload

  _run(_case())


def test_resume_abandoned_finalization_survives_cancellation_after_durable_commit(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_cancel_finalize",
      agent_name="email-responder",
      user_message="Settle cancelled resume finalization.",
      resumable=False,
    )
    entry = runner._task_registry.get("bg_cancel_finalize")
    assert entry is not None
    durable_committed = asyncio.Event()
    release_appender = asyncio.Event()
    append_calls = 0
    append_task_completed = runner._append_task_completed_event

    async def _append_then_block(
      task_entry: Any,
      final_state: TaskState,
      **kwargs: Any,
    ) -> None:
      nonlocal append_calls
      append_calls += 1
      await append_task_completed(
        task_entry,
        final_state,
        **kwargs,
      )
      durable_committed.set()
      await release_appender.wait()

    runner._append_task_completed_event = _append_then_block  # type: ignore[method-assign]
    caller = asyncio.create_task(
      _finalize_resume_abandoned(
        runner=runner,
        entry=entry,
        code="not_resumable",
        message="Skill is not resumable",
      )
    )

    await durable_committed.wait()
    assert entry.finalization_lock.locked()
    caller.cancel("caller-stopped")
    await asyncio.sleep(0)
    assert not caller.done()

    release_appender.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
      await caller
    assert cancelled.value.args == ("caller-stopped",)

    completed, _ = await runner._agent_session_log.query(
      event_types={"task_completed"},
      order="asc",
    )
    matching = [
      log_entry.event
      for log_entry in completed
      if log_entry.event.get("task_id") == entry.task_id
    ]
    assert len(matching) == 1
    assert append_calls == 1
    assert entry.state == TaskState.FAILED
    assert entry.result == matching[0]["result"]
    notifications = runner._notification_queue.peek()
    assert len(notifications) == 1
    assert notifications[0].event == "failed"
    assert notifications[0].payload == entry.result

    replay_result, replay_error = await _finalize_resume_abandoned(
      runner=runner,
      entry=entry,
      code="not_resumable",
      message="Skill is not resumable",
    )

    assert replay_error is None
    assert replay_result == entry.task_result
    assert append_calls == 1
    completed_after_retry, _ = await runner._agent_session_log.query(
      event_types={"task_completed"},
      order="asc",
    )
    assert len([
      log_entry
      for log_entry in completed_after_retry
      if log_entry.event.get("task_id") == entry.task_id
    ]) == 1
    assert len(runner._notification_queue.peek()) == 1

  _run(_case())


def test_cancelled_resume_abandonment_is_bounded_while_append_hangs(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    runner._background_completion_persist_timeout_seconds = 0.01
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_hung_abandon",
      agent_name="email-responder",
      user_message="Settle hung resume abandonment.",
      resumable=False,
    )
    entry = runner._task_registry.get("bg_hung_abandon")
    assert entry is not None
    log = runner._agent_session_log
    assert log is not None
    append_started = asyncio.Event()
    release_append = asyncio.Event()
    append_task_completed = runner._append_task_completed_event

    async def _hang_before_append(
      task_entry: Any,
      final_state: TaskState,
      **kwargs: Any,
    ) -> None:
      append_started.set()
      await release_append.wait()
      await append_task_completed(
        task_entry,
        final_state,
        **kwargs,
      )

    runner._append_task_completed_event = _hang_before_append  # type: ignore[method-assign]
    caller = asyncio.create_task(
      _finalize_resume_abandoned(
        runner=runner,
        entry=entry,
        code="not_resumable",
        message="Skill is not resumable",
      )
    )
    await append_started.wait()

    started_at = asyncio.get_running_loop().time()
    caller.cancel("caller-stopped")
    with pytest.raises(asyncio.CancelledError) as cancelled:
      await caller
    elapsed = asyncio.get_running_loop().time() - started_at

    assert cancelled.value.args == ("caller-stopped",)
    assert elapsed < 0.2
    assert entry.state == TaskState.INTERRUPTED
    assert entry.completion_persistence_state == "uncertain"

    release_append.set()
    for _ in range(100):
      if entry.state == TaskState.FAILED:
        break
      await asyncio.sleep(0)

    assert entry.state == TaskState.FAILED
    assert entry.completion_persistence_state == "committed"
    completed, _ = await log.query(
      event_types={"task_completed"},
      order="asc",
    )
    assert len([
      log_entry
      for log_entry in completed
      if log_entry.event.get("task_id") == entry.task_id
    ]) == 1
    notifications = runner._notification_queue.peek()
    assert len(notifications) == 1
    assert notifications[0].event == "failed"

  _run(_case())


def test_resume_abandoned_reconciles_append_failure_after_fsync(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_fsync_finalize",
      agent_name="email-responder",
      user_message="Reconcile fsync resume abandonment.",
      resumable=False,
    )
    entry = runner._task_registry.get("bg_fsync_finalize")
    assert entry is not None
    log = runner._agent_session_log
    assert log is not None

    update_manifest = log._update_manifest_latest_seq_locked
    fail_after_fsync = True

    def _fail_once(seq: int) -> None:
      nonlocal fail_after_fsync
      if fail_after_fsync:
        fail_after_fsync = False
        raise RuntimeError("manifest update failed after fsync")
      update_manifest(seq)

    monkeypatch.setattr(
      log,
      "_update_manifest_latest_seq_locked",
      _fail_once,
    )

    result, error = await _finalize_resume_abandoned(
      runner=runner,
      entry=entry,
      code="not_resumable",
      message="Skill is not resumable",
    )

    assert error is None
    assert isinstance(result, TaskResult)
    assert result.execution.status == "failed"
    assert result.execution.terminal_reason is not None
    assert result.execution.terminal_reason.startswith(
      "resume_abandoned:not_resumable:"
    )
    assert entry.state == TaskState.FAILED
    result_payload = result.model_dump(mode="json")
    assert entry.result == result_payload
    assert entry.task_result == result
    assert entry.completion_persistence_state == "committed"
    assert entry.completion_persistence_error is None

    repeated_result, repeated_error = await _finalize_resume_abandoned(
      runner=runner,
      entry=entry,
      code="not_resumable",
      message="Skill is not resumable",
    )
    assert repeated_error is None
    assert repeated_result == result

    completed, _ = await log.query(
      event_types={"task_completed"},
      order="asc",
    )
    assert len([
      log_entry
      for log_entry in completed
      if log_entry.event.get("task_id") == entry.task_id
    ]) == 1
    notifications = runner._notification_queue.peek()
    assert len(notifications) == 1
    assert notifications[0].event == "failed"
    assert notifications[0].payload == result_payload

  _run(_case())


def test_cancelled_resume_abandon_reconciliation_poisons_writer_lease(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_cancelled_fsync_reconcile",
      agent_name="email-responder",
      user_message="Cancel fsync resume reconciliation.",
      resumable=False,
    )
    entry = runner._task_registry.get("bg_cancelled_fsync_reconcile")
    assert entry is not None
    log = runner._agent_session_log
    assert log is not None

    update_manifest = log._update_manifest_latest_seq_locked
    fail_after_fsync = True

    def _fail_once(seq: int) -> None:
      nonlocal fail_after_fsync
      if fail_after_fsync:
        fail_after_fsync = False
        raise RuntimeError("manifest update failed after fsync")
      update_manifest(seq)

    monkeypatch.setattr(
      log,
      "_update_manifest_latest_seq_locked",
      _fail_once,
    )
    lookup_started = asyncio.Event()

    async def _blocked_lookup(_task_id: str) -> TaskEntry | None:
      lookup_started.set()
      await asyncio.Future()

    runner._lookup_task_in_log = _blocked_lookup  # type: ignore[method-assign]
    caller = asyncio.create_task(
      _finalize_resume_abandoned(
        runner=runner,
        entry=entry,
        code="not_resumable",
        message="Skill is not resumable",
      )
    )
    await lookup_started.wait()
    finalizer = next(
      task
      for task in runner._pending_background_initializations
      if task.get_name().endswith(":resume-abandoned-finalize")
    )
    finalizer.cancel()

    with pytest.raises(asyncio.CancelledError):
      await caller
    await asyncio.sleep(0)

    assert finalizer.cancelled()
    assert runner._writer_lease_poisoned is True
    assert entry.state == TaskState.INTERRUPTED
    assert entry.completion_persistence_state == "uncertain"
    completed, _ = await log.query(
      event_types={"task_completed"},
      order="asc",
    )
    assert len([
      log_entry
      for log_entry in completed
      if log_entry.event.get("task_id") == entry.task_id
    ]) == 1

  _run(_case())


def test_failed_resume_abandon_lookup_poisons_writer_lease(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_failed_fsync_reconcile",
      agent_name="email-responder",
      user_message="Fail fsync resume reconciliation.",
      resumable=False,
    )
    entry = runner._task_registry.get("bg_failed_fsync_reconcile")
    assert entry is not None
    log = runner._agent_session_log
    assert log is not None

    update_manifest = log._update_manifest_latest_seq_locked
    fail_after_fsync = True

    def _fail_once(seq: int) -> None:
      nonlocal fail_after_fsync
      if fail_after_fsync:
        fail_after_fsync = False
        raise RuntimeError("manifest update failed after fsync")
      update_manifest(seq)

    monkeypatch.setattr(
      log,
      "_update_manifest_latest_seq_locked",
      _fail_once,
    )

    async def _failed_lookup(_task_id: str) -> TaskEntry | None:
      raise OSError("durable lookup unavailable")

    runner._lookup_task_in_log = _failed_lookup  # type: ignore[method-assign]

    with pytest.raises(
      RuntimeError,
      match="manifest update failed after fsync",
    ):
      await _finalize_resume_abandoned(
        runner=runner,
        entry=entry,
        code="not_resumable",
        message="Skill is not resumable",
      )

    assert runner._writer_lease_poisoned is True
    assert entry.state == TaskState.INTERRUPTED
    assert entry.completion_persistence_state == "uncertain"
    completed, _ = await log.query(
      event_types={"task_completed"},
      order="asc",
    )
    assert len([
      log_entry
      for log_entry in completed
      if log_entry.event.get("task_id") == entry.task_id
    ]) == 1

  _run(_case())


def test_timed_out_completion_reconciles_late_post_fsync_commit(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    runner._background_completion_persist_timeout_seconds = 0.01
    entry = runner._task_registry.register(
      "background_agent",
      agent_name="earnings-review",
      task_id="bg_late_fsync_completion",
    )
    log = runner._agent_session_log
    assert log is not None
    await log.append({
      "type": "task_registered",
      "capability_bind": _bind_receipt(),
      "task_id": entry.task_id,
      "task_type": "background",
      "agent_name": entry.agent_name,
      "started_at": 1.0,
    })
    update_manifest = log._update_manifest_latest_seq_locked
    manifest_started = threading.Event()
    release_manifest = threading.Event()
    fail_after_fsync = True

    def _block_then_fail(seq: int) -> None:
      nonlocal fail_after_fsync
      if fail_after_fsync:
        fail_after_fsync = False
        manifest_started.set()
        assert release_manifest.wait(timeout=2.0)
        raise RuntimeError("manifest update failed after fsync")
      update_manifest(seq)

    monkeypatch.setattr(
      log,
      "_update_manifest_latest_seq_locked",
      _block_then_fail,
    )
    accepted_result = _report_task_result(
      "Accepted before persistence timeout",
      task_id=entry.task_id,
      workspace_dir=tmp_path,
    )
    accepted_payload = accepted_result.model_dump(mode="json")
    completed: list[str] = []

    async def _handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[TaskResult, None]:
      return accepted_result, None

    task = asyncio.create_task(
      runner._run_background_agent(
        entry,
        _handler,
        {},
        0,
        on_complete=lambda completed_entry: completed.append(
          completed_entry.task_id
        ),
      )
    )
    entry.asyncio_task = task
    runner._task_registry.transition(
      entry.task_id,
      TaskState.RUNNING,
    )
    assert await asyncio.to_thread(manifest_started.wait, 1.0)

    with pytest.raises(asyncio.TimeoutError):
      await task

    assert entry.state == TaskState.INTERRUPTED
    assert entry.result is None
    assert completed == []

    release_manifest.set()
    for _ in range(200):
      if entry.state == TaskState.COMPLETED:
        break
      await asyncio.sleep(0)

    assert entry.state == TaskState.COMPLETED
    assert entry.result == accepted_payload
    assert entry.task_result == accepted_result
    assert entry.completion_persistence_state == "committed"
    assert entry.completion_persistence_error is None
    assert completed == [entry.task_id]
    completed_events, _ = await log.query(
      event_types={"task_completed"},
      order="asc",
    )
    matching = [
      log_entry.event
      for log_entry in completed_events
      if log_entry.event.get("task_id") == entry.task_id
    ]
    assert len(matching) == 1
    assert matching[0]["final_state"] == "completed"
    assert matching[0]["result"] == accepted_payload
    notifications = runner._notification_queue.peek()
    assert [notification.event for notification in notifications] == [
      "interrupted",
      "completed",
    ]
    assert [
      notification.notification_generation
      for notification in notifications
    ] == [1, 2]
    assert notifications[1].payload["task_result_ref"][
      "task_result_id"
    ] == accepted_result.task_result_id

  _run(_case())


def test_resume_handler_adopts_old_interrupted_task_before_abandonment(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log_writer = _runner(tmp_path)
    await _append_interrupted_skill_task(
      log_writer,
      task_id="bg_1",
      agent_name="email-responder",
      user_message="Old interrupted work.",
      resumable=False,
    )
    await _append_interrupted_skill_task(
      log_writer,
      task_id="bg_2",
      agent_name="newer-agent",
      user_message="Newer interrupted work.",
    )

    runner = _runner(tmp_path, max_retained_tasks=1)
    runner._runner_id = "runner-test"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "email-responder.md").write_text(
      "---\nagent_callable: true\nagent_description: Email.\n---\nPrompt",
      encoding="utf-8",
    )
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": "bg_1"})

    _assert_resume_abandoned(
      result,
      error,
      code="not_resumable",
    )
    adopted = runner._task_registry.get("bg_1")
    assert adopted is not None
    assert adopted.state == TaskState.FAILED
    assert runner._task_registry.get("bg_2").state == TaskState.INTERRUPTED

  _run(_case())


def test_resume_handler_rejects_persisted_grant_for_now_denied_mcp_server(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_jobs",
      agent_name="legacy-job-reader",
      user_message="Inspect an operator job.",
      allowed_tools=("list_jobs",),
    )
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "legacy-job-reader")
    mcp_client = _JobsMcpClient()
    handler = make_resume_handler(
      [runner],
      mcp_client=mcp_client,
      excluded_tools_resolver=frozenset,
      denied_mcp_servers={"jobs-mcp"},
    )

    result, error = await handler({"task_id": "bg_jobs"})

    _assert_resume_abandoned(
      result,
      error,
      code="admitted_tool_route_unavailable",
    )
    assert mcp_client.call_count == 0
    assert runner._task_registry.get("bg_jobs").state == TaskState.FAILED

  _run(_case())


def test_resume_handler_abandons_chain_at_maximum_depth(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path, max_resume_chain_depth=1)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_depth_root",
      agent_name="earnings-review",
      user_message="Root work",
    )
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_depth_root_r1",
      original_task_id="bg_depth_root",
      agent_name="earnings-review",
      user_message="First resumed work",
    )
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": "bg_depth_root_r1"})

    _assert_resume_abandoned(
      result,
      error,
      code="max_resume_chain_depth",
    )
    assert runner._task_registry.get("bg_depth_root_r1").state == (
      TaskState.FAILED
    )
    assert runner._task_registry.get("bg_depth_root").state == (
      TaskState.INTERRUPTED
    )

  _run(_case())


def test_resume_handler_preserves_original_bind_despite_legacy_model_knob(
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
      return await _successful_resume_task_result(
        kwargs,
        "continued",
        workspace_dir=tmp_path,
      ), None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    monkeypatch.setenv("SUB_AGENT_DEFAULT_MODEL", "claude-opus-4-8")
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": "bg_opus48"})

    assert error is None
    assert result is not None
    resumed_entry = runner._task_registry.get(result["task_id"])
    assert resumed_entry is not None
    await resumed_entry.asyncio_task
    assert (
      captured["capability_execution"].bind.upstream_model
      == "claude-sonnet-4-6"
    )

  _run(_case())


def test_resume_handler_rejects_hidden_model_and_provider_overrides(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_invalid_openai_model",
      agent_name="earnings-review",
      user_message="Resume earnings review.",
    )

    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({
      "task_id": "bg_invalid_openai_model",
      "provider": "openai",
      "model": "bad-model",
    })

    assert result is None
    assert error["code"] == "invalid_input"
    assert error["message"].endswith("model, provider")

  _run(_case())


def test_resume_rebuilds_and_materializes_the_exact_durable_bind(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    receipt = _bind_receipt(
      capability_id="node.verify",
      provider="openai",
      model="gpt-4o-mini",
      effort="max",
    )
    seed_runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      seed_runner,
      task_id="bg_exact_bind",
      agent_name="verify-finding",
      user_message="Resume exact verification.",
      capability_bind_receipt=receipt,
    )

    runner = _runner(tmp_path)
    runner._runner_id = "runner-resume"
    captured: dict[str, Any] = {}

    async def _resume_sub_agent(**kwargs: Any):
      captured.update(kwargs)
      return await _successful_resume_task_result(
        kwargs,
        "continued",
        workspace_dir=tmp_path,
      ), None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    skills_dir = tmp_path / "skills"
    _write_skill(
      skills_dir,
      "verify-finding",
      """
allowed_tools: []
mcp_tools: {}
semantic_metadata:
  tool_refs: []
""",
    )
    resolver = _TestResumeCapabilityExecutionResolver()
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      capability_execution_resolver=resolver,
    )

    result, error = await handler({"task_id": "bg_exact_bind"})

    assert error is None
    assert result is not None
    resumed_entry = runner._task_registry.get(result["task_id"])
    assert resumed_entry is not None
    assert resumed_entry.capability_bind_receipt == receipt
    await resumed_entry.asyncio_task
    assert resolver.materialize_calls[0].receipt() == receipt
    assert captured["capability_execution"].bind.receipt() == receipt
    assert captured["capability_execution"].bind is resolver.materialize_calls[0]

    log = runner._agent_session_log
    assert log is not None
    registered, _cursor = await log.query(event_types={"task_registered"})
    resumed_event = next(
      entry.event
      for entry in registered
      if entry.event.get("task_id") == result["task_id"]
    )
    assert resumed_event["capability_bind"] == receipt
    assert resumed_event["metadata"][ADMITTED_TASK_METADATA_KEY][
      "attempt"
    ]["physical_task_id"] == result["task_id"]

  _run(_case())


def test_resume_refuses_missing_durable_bind_before_materialization(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    seed_runner = _runner(tmp_path)
    log = seed_runner._agent_session_log
    assert log is not None
    await log.append({
      "type": "task_registered",
      "task_id": "bg_missing_bind",
      "task_type": "background",
      "agent_name": "earnings-review",
      "sub_agent_id": "sub:bg_missing_bind",
      "started_at": 1.0,
    })
    await log.append({
      "type": "attach",
      "sub_agent_id": "sub:bg_missing_bind",
      "runner_id": "runner:bg_missing_bind",
      "role": "sub_agent",
    })

    runner = _runner(tmp_path)
    runner.resume_sub_agent = pytest.fail  # type: ignore[method-assign]
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    resolver = _TestResumeCapabilityExecutionResolver()
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      capability_execution_resolver=resolver,
    )

    result, error = await handler({"task_id": "bg_missing_bind"})

    # A bind-less durable registration is loudly skipped at rebuild
    # (drain, not migrate), so resume refuses with a typed not_found
    # before any reconstruction or materialization — never a late
    # invalid_task_metadata failure on a rebuilt bind-less entry.
    assert result is None
    assert error is not None
    assert error["code"] == "not_found"
    assert "bg_missing_bind" in error["message"]
    assert resolver.materialize_calls == []
    assert runner._task_registry.get("bg_missing_bind") is None

  _run(_case())


def test_resume_loudly_skips_pre_cutover_v1_registration(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  """A v1 task_registered record never rebuilds as an INTERRUPTED entry.

  Pre-cutover records are drained with a warn-once loud skip (review
  2026-08-14, B7); resume answers with a typed not_found instead of a
  late invalid_task_metadata on a bind-less reconstruction.
  """

  async def _case() -> None:
    import agent_gateway.agent_session_log as session_log_module

    seed_runner = _runner(tmp_path)
    log = seed_runner._agent_session_log
    assert log is not None
    monkeypatch.setattr(session_log_module, "EVENT_SCHEMA_VERSION", 1)
    await log.append({
      "type": "task_registered",
      "task_id": "bg_v1_record",
      "task_type": "background",
      "agent_name": "earnings-review",
      "sub_agent_id": "sub:bg_v1_record",
      "capability_bind": _bind_receipt(),
      "started_at": 1.0,
    })
    monkeypatch.undo()

    runner = _runner(tmp_path)
    runner.resume_sub_agent = pytest.fail  # type: ignore[method-assign]
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    resolver = _TestResumeCapabilityExecutionResolver()
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      capability_execution_resolver=resolver,
    )

    with caplog.at_level("WARNING", logger="agent_gateway.task_registry"):
      result, error = await handler({"task_id": "bg_v1_record"})

    assert result is None
    assert error is not None
    assert error["code"] == "not_found"
    assert resolver.materialize_calls == []
    assert runner._task_registry.get("bg_v1_record") is None
    skip_messages = [
      record.getMessage()
      for record in caplog.records
      if "Skipping durable task bg_v1_record" in record.getMessage()
    ]
    assert skip_messages
    assert "event_schema_version" in skip_messages[0]

  _run(_case())


def test_resume_handler_persists_original_bind_with_coordinator_enabled(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review")
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_default_provider",
      agent_name="earnings-review",
      user_message="Resume earnings review.",
    )
    captured: dict[str, Any] = {}
    parent_session = GatewaySession(
      session_id="session-proof",
      api_key_hash="hash",
      created_at=1,
      expires_at=2,
      user_id="owner",
    )
    async def _register_background_task(**kwargs: Any):
      captured.update(kwargs)
      return {"task_id": "bg_default_provider_r1", "status": "running"}, None

    runner._register_background_task = _register_background_task  # type: ignore[method-assign]
    handler = make_resume_handler(
      [runner],
      parent_session=parent_session,
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      coordinator_config=CoordinatorConfig(enabled=True),
    )

    result, error = await handler({"task_id": "bg_default_provider"})

    assert error is None
    assert result is not None
    assert "provider" not in captured["tool_input"]
    assert "model" not in captured["tool_input"]
    assert captured["capability_bind_receipt"] == _bind_receipt()

  _run(_case())


def test_resume_handler_uses_exact_admitted_prompt_after_skill_source_changes(
  tmp_path: Path,
) -> None:
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
      max_budget_usd=6.0,
    )
    source_entry = runner._task_registry.get("bg_block")
    assert source_entry is not None
    source_snapshot = source_entry.admitted_task.execution_snapshot
    assert source_snapshot is not None
    captured: dict[str, Any] = {}

    async def _resume_sub_agent(**kwargs: Any):
      captured.update(kwargs)
      return await _successful_resume_task_result(
        kwargs,
        "continued",
        workspace_dir=tmp_path,
      ), None

    runner.resume_sub_agent = _resume_sub_agent  # type: ignore[method-assign]
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
    )

    result, error = await handler({"task_id": "bg_block"})

    assert error is None
    assert result is not None
    resumed_entry = runner._task_registry.get(result["task_id"])
    assert resumed_entry is not None
    await resumed_entry.asyncio_task
    successor_snapshot = resumed_entry.admitted_task.execution_snapshot
    assert successor_snapshot is not None
    prompt = captured["system_prompt"]
    assert prompt.startswith("Resume carefully.")
    assert "Resolved resume citation contract" not in prompt
    assert not _UNRESOLVED_BLOCK_RE.search(prompt)
    assert successor_snapshot.model_dump(
      exclude={"resume_instruction"}
    ) == source_snapshot.model_dump(exclude={"resume_instruction"})
    assert successor_snapshot.resume_instruction == (
      "Resume the admitted task from its durable transcript."
    )
    assert captured["system_prompt"] == successor_snapshot.system_prompt
    assert captured["max_turns"] == successor_snapshot.max_turns
    assert captured["timeout"] == successor_snapshot.timeout_seconds
    assert captured["client_timeout"] == (
      successor_snapshot.client_timeout_seconds
    )
    assert captured["max_tokens"] == successor_snapshot.max_tokens
    assert captured["cost_observation_threshold_usd"] == (
      successor_snapshot.cost_observation_threshold_usd
    )
    assert captured["max_budget_usd"] == pytest.approx(6.0)
    assert captured["dispatcher"]._should_avoid_permission_prompts is True

  _run(_case())


def test_resume_background_emits_skill_run_started_before_completion(tmp_path: Path) -> None:
  async def _case() -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "earnings-review", "scope: ticker")
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
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
      assert [entry.event["type"] for entry in parent_log.entries] == ["skill_run_started"]
      entered_resume.set()
      await release_resume.wait()
      return await _successful_resume_task_result(
        kwargs,
        "continued",
        workspace_dir=tmp_path,
      ), None

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
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
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
    durable_log = runner._agent_session_log
    assert durable_log is not None
    durable_markers, _ = await durable_log.query(
      event_types={"skill_run_started", "skill_result_captured"},
      order="asc",
    )
    assert [entry.event["type"] for entry in durable_markers] == [
      "skill_run_started",
      "skill_result_captured",
    ]
    assert durable_markers[0].seq < durable_markers[1].seq

  _run(_case())


def test_resume_copies_admitted_ticker_and_ignores_later_prose(
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
      runner._runner_id = "runner-test"
      await _append_interrupted_skill_task(
        runner,
        task_id="bg_scope",
        agent_name="html-research",
        user_message="Resume the html analysis carefully.",
      )
      original = runner._task_registry.get("bg_scope")
      assert original is not None and original.admitted_task is not None
      original_inputs = original.admitted_task.inputs
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
        return await _successful_resume_task_result(
          kwargs,
          "continued",
          workspace_dir=tmp_path,
        ), None

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
        mcp_client=_NullMcpClient(),
        local_tool_handlers={},
        excluded_tools_resolver=frozenset,
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

      # INC-4 retired emit_html_artifact; the resolved ticker/scope is observable
      # on the surviving skill_run_started event (no emit tool needed).
      events = [entry.event for entry in parent_log.entries]
      assert [event["type"] for event in events] == [
        "skill_run_started",
        "skill_result_captured",
      ]
      assert events[0]["ticker"] == "PCTY"
      assert events[0]["scope"] == "ticker"
      assert events[1]["ticker"] == "PCTY"
      assert resumed_entry.admitted_task is not None
      assert resumed_entry.admitted_task.inputs == original_inputs

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)


def test_resume_ticker_owner_survives_two_exact_successor_hops(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    runner._runner_id = "runner-test"
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_ticker_root",
      agent_name="earnings-review",
      user_message="Review PCTY earnings.",
      ticker="PCTY",
    )
    original = runner._task_registry.get("bg_ticker_root")
    assert original is not None and original.admitted_task is not None
    original_inputs = original.admitted_task.inputs
    logical_owner = original.admitted_task.logical_task.delegation_id
    first_resolver = _TestResumeCapabilityExecutionResolver()
    dispatches: list[str] = []
    first_dispatch_started = asyncio.Event()

    async def _first_resume_sub_agent(**kwargs: Any):
      dispatches.append(kwargs["attempt"].physical_task_id)
      first_dispatch_started.set()
      await asyncio.Event().wait()

    runner.resume_sub_agent = _first_resume_sub_agent  # type: ignore[method-assign]
    first_handler = make_resume_handler(
      [runner],
      parent_session=GatewaySession(
        session_id="sess-parent",
        api_key_hash="hash",
        created_at=10,
        expires_at=20,
        user_id="alice",
        auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
      ),
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      capability_execution_resolver=first_resolver,
    )

    first, first_error = await first_handler({
      "task_id": "bg_ticker_root",
      "additional_context": "Ignore PCTY and switch to MSFT.",
    })
    assert first_error is None and first is not None
    first_entry = runner._task_registry.get(first["task_id"])
    assert first_entry is not None and first_entry.asyncio_task is not None
    await first_dispatch_started.wait()
    assert first_entry.admitted_task is not None
    assert first_entry.admitted_task.inputs == original_inputs
    assert first_entry.admitted_task.logical_task.delegation_id == logical_owner

    # A fresh process recovers the durably registered, incomplete first
    # successor as interrupted. Its exact admission is the second-hop source.
    restarted = _runner(tmp_path)
    restarted._runner_id = "runner-test-restarted"
    second_resolver = _TestResumeCapabilityExecutionResolver()

    async def _second_resume_sub_agent(**kwargs: Any):
      dispatches.append(kwargs["attempt"].physical_task_id)
      return await _successful_resume_task_result(
        kwargs,
        "second hop completed",
        workspace_dir=tmp_path,
      ), None

    restarted.resume_sub_agent = _second_resume_sub_agent  # type: ignore[method-assign]
    second_handler = make_resume_handler(
      [restarted],
      parent_session=GatewaySession(
        session_id="sess-parent",
        api_key_hash="hash",
        created_at=10,
        expires_at=20,
        user_id="alice",
        auth_config={"api_key": "k", "model": "claude-sonnet-4-6"},
      ),
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      capability_execution_resolver=second_resolver,
    )

    second, second_error = await second_handler({
      "task_id": first_entry.task_id,
      "additional_context": "Now override the subject with GOOGL.",
    })
    assert second_error is None and second is not None
    recovered_first = restarted._task_registry.get(first_entry.task_id)
    assert recovered_first is not None
    assert recovered_first.state == TaskState.INTERRUPTED
    assert recovered_first.admitted_task is not None
    assert recovered_first.admitted_task.inputs == original_inputs
    second_entry = restarted._task_registry.get(second["task_id"])
    assert second_entry is not None and second_entry.asyncio_task is not None
    await second_entry.asyncio_task

    assert second_entry.state == TaskState.COMPLETED
    assert second_entry.admitted_task is not None
    assert second_entry.admitted_task.inputs == original_inputs
    assert second_entry.admitted_task.logical_task.delegation_id == logical_owner
    assert original_inputs[0].context.content == "PCTY"
    assert dispatches == ["bg_ticker_root_r1", "bg_ticker_root_r2"]
    assert len(first_resolver.materialize_calls) == 1
    assert len(second_resolver.materialize_calls) == 1

    first_entry.termination_intent = "killed"
    first_entry.asyncio_task.cancel()
    await asyncio.gather(first_entry.asyncio_task, return_exceptions=True)

  _run(_case())


def test_resume_admitted_ticker_ignores_parent_resume_message_context(
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
      runner._runner_id = "runner-test"
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
          **_interrupted_task_correlation(
            runner,
            task_id="bg_parent_scope",
            capability_bind_receipt=_bind_receipt(),
          ),
          "task_type": "background",
          "message_id": "msg-msft",
          "message": "Focus on MSFT for the resumed artifact.",
          "sent_at": 2.0,
        }
      )
      captured: dict[str, Any] = {}

      async def _resume_sub_agent(**kwargs: Any):
        captured.update(kwargs)
        return await _successful_resume_task_result(
          kwargs,
          "continued",
          workspace_dir=tmp_path,
        ), None

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
        mcp_client=_NullMcpClient(),
        local_tool_handlers={},
        excluded_tools_resolver=frozenset,
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

      # INC-4 retired emit_html_artifact; the parent-resume-message ticker context
      # is observable on the surviving skill_run_started event.
      events = [entry.event for entry in parent_log.entries]
      assert [event["type"] for event in events] == [
        "skill_run_started",
        "skill_result_captured",
      ]
      assert events[0]["ticker"] == "PCTY"
      assert events[0]["scope"] == "ticker"
      assert events[1]["ticker"] == "PCTY"

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)


def test_resume_generic_historical_admission_without_ticker_remains_live(
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
      runner._runner_id = "runner-test"
      await _append_interrupted_skill_task(
        runner,
        task_id="bg_no_scope",
        agent_name="html-research",
        user_message="Resume the html analysis carefully.",
        required_context=(),
        ticker=None,
      )
      captured: dict[str, Any] = {}

      async def _resume_sub_agent(**kwargs: Any):
        captured.update(kwargs)
        return await _successful_resume_task_result(
          kwargs,
          "continued",
          workspace_dir=tmp_path,
        ), None

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
        mcp_client=_NullMcpClient(),
        local_tool_handlers={},
        excluded_tools_resolver=frozenset,
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

      # INC-4 retired emit_html_artifact; the portfolio-scope fallback (no ticker)
      # is observable on the surviving skill_run_started event.
      events = [entry.event for entry in parent_log.entries]
      assert [event["type"] for event in events] == [
        "skill_run_started",
        "skill_result_captured",
      ]
      assert events[0]["ticker"] is None
      assert events[0]["scope"] == "portfolio"
      assert events[1]["ticker"] is None

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)


def test_resume_ticker_required_historical_admission_without_binding_fails_closed(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    runner = _runner(tmp_path)
    await _append_interrupted_skill_task(
      runner,
      task_id="bg_missing_ticker_binding",
      agent_name="html-research",
      user_message="Resume legacy work.",
      required_context=("ticker",),
      ticker=None,
    )
    resolver = _TestResumeCapabilityExecutionResolver()
    handler = make_resume_handler(
      [runner],
      mcp_client=_NullMcpClient(),
      excluded_tools_resolver=frozenset,
      capability_execution_resolver=resolver,
    )

    result, error = await handler({"task_id": "bg_missing_ticker_binding"})

    _assert_resume_abandoned(result, error, code="invalid_task_metadata")
    assert resolver.materialize_calls == []
    assert runner._task_registry.get("bg_missing_ticker_binding").state == (
      TaskState.FAILED
    )

  _run(_case())


def test_resume_emit_dashboard_artifact_failure_emits_tool_write_failed(
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
      runner._runner_id = "runner-test"
      await _append_interrupted_skill_task(
        runner,
        task_id="bg_portfolio",
        agent_name="portfolio-report",
        user_message="Resume portfolio HTML report.",
        allowed_tools=("emit_dashboard_artifact",),
        required_context=(),
      )
      captured: dict[str, Any] = {}

      async def _resume_sub_agent(**kwargs: Any):
        captured.update(kwargs)
        return await _successful_resume_task_result(
          kwargs,
          "continued",
          workspace_dir=tmp_path,
        ), None

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
        mcp_client=_NullMcpClient(),
        local_tool_handlers={},
        excluded_tools_resolver=frozenset,
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

      def _raise_build(*_args: Any, **_kwargs: Any):
        raise RuntimeError("boom dashboard build")

      monkeypatch.setattr(
        "agent_gateway.dashboard_artifact.build_dashboard_artifact",
        _raise_build,
      )
      emit_result, emit_error = await captured["dispatcher"].dispatch(
        "tool_dashboard_bad",
        "emit_dashboard_artifact",
        {
          "payload": {"blocks": []},
          "summary": "Dashboard that fails to build.",
        },
      )

      assert emit_result is None
      assert emit_error is not None
      assert emit_error["code"] == "internal_error"
      assert "boom dashboard build" in emit_error["message"]
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
      assert failed["skill"] == "_dashboard"
      assert failed["error_code"] == "tool_write_failed"
      assert failed["tool_call_id"] == "tool_dashboard_bad"
      assert not (memory.get_workspace_dir("alice") / "artifacts" / "_dashboards").exists()

    _run(_case())
  finally:
    memory.set_memory_store_factory(None)
