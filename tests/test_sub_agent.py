from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_workflow_contracts import (
  ActivityHandle,
  AdmittedTask,
  AgentCompletionEnvelope,
  AnalyticalOutcome,
  CanonicalProjection,
  ContentHandle,
  ContractRef,
  ExecutionSettlement,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  TranscriptHandle,
  UsageObservation,
  canonical_json_bytes,
  sha256_digest,
)
from agent_gateway.agent_result_content import (
  make_get_agent_result_content_handler,
)
from agent_gateway.agent_session_log import AgentSessionLog
from agent_gateway.capability_binding import (
  CapabilityBind,
)
from agent_gateway.event_log import EventLog
from agent_gateway.final_narrative_artifact import publish_final_narrative
from agent_gateway.tool_dispatcher_helpers import ToolExecutionContext
from agent_gateway.session import GatewaySession
from agent_gateway.skills import SkillLoader
from agent_gateway.sub_agent import (
  make_get_background_result_handler,
  make_get_background_result_tool_def,
  make_run_agent_handler,
  make_run_agent_tool_def,
  make_resume_tool_def,
)
from agent_gateway.sub_agent_scope_receipt import ADMITTED_TASK_METADATA_KEY
from agent_gateway.sub_agent_result_contract import (
  terminal_narrative_content_handle,
)
from agent_gateway.sub_agent_skill_events import (
  DurableSkillEventPersistenceError,
  SkillRunEventEmitter,
)
from tests.capability_execution_test_support import (
  stub_capability_execution_resolver,
)


async def _tool(_tool_input: dict[str, Any], **_kwargs: Any):
  return {"ok": True}, None


class _McpClient:
  @staticmethod
  def is_mcp_tool(_name: str) -> bool:
    return False

  @staticmethod
  def get_tool_definitions() -> list[dict[str, Any]]:
    return []

  async def call_tool(self, name: str, _tool_input: dict[str, Any]):
    return None, {"code": "unknown_tool", "message": name}


class _CapabilityResolver:
  def __init__(self) -> None:
    self._resolver = stub_capability_execution_resolver(
      default_provider="openai",
      default_model="gpt-5.6-sol",
      default_effort="high",
    )
    self.calls: list[dict[str, Any]] = []

  def resolve(self, capability_id: str, **kwargs: Any) -> Any:
    self.calls.append({"capability_id": capability_id, **kwargs})
    return self._resolver.resolve(capability_id, **kwargs)


def _content(
  value: object,
  contract: ContractRef,
  *,
  media_type: str = "application/json",
) -> ContentHandle:
  raw = (
    value.encode("utf-8")
    if isinstance(value, str)
    else canonical_json_bytes(value)
  )
  digest = hashlib.sha256(raw).hexdigest()
  return ContentHandle(
    content_id=f"sha256:{digest}",
    content_sha256=digest,
    content_bytes=len(raw),
    content_chars=len(raw.decode("utf-8")),
    contract=contract,
    media_type=media_type,
    encoding="utf-8",
    retention="durable",
  )


def _spawn_result(kwargs: dict[str, Any]) -> TaskResult:
  requirement = kwargs["result_requirement"]
  provenance = kwargs["result_provenance"]
  narrative_contract = ContractRef(
    namespace="agent-gateway",
    name="terminal-narrative",
    version="1.0",
    digest=sha256_digest({"contract": "terminal-narrative"}),
  )
  terminal = (
    _content(
      "Canonical child narrative.",
      narrative_contract,
      media_type="text/plain; charset=utf-8",
    )
    if requirement.terminal_narrative == "required"
    else None
  )
  projection = None
  if requirement.projection is not None:
    inline = {"summary": "Canonical child summary."}
    projection = CanonicalProjection(
      contract=requirement.projection.contract,
      content=_content(inline, requirement.projection.contract),
      inline_view=inline,
    )
  outcome = (
    AnalyticalOutcome(
      disposition="complete",
      assessment_source="domain_tool",
    )
    if requirement.outcome.required
    else None
  )
  attempt = kwargs["attempt"]
  return TaskResult(
    task_result_id=f"result:{attempt.attempt_id}",
    logical_task=kwargs["logical_task"],
    attempt=attempt,
    execution=ExecutionSettlement(status="succeeded"),
    outcome=outcome,
    values=TaskResultValues(
      terminal_narrative=terminal,
      projection=projection,
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
      usage=UsageObservation(),
    ),
    provenance=TaskResultProvenance.model_validate(provenance),
  )


class _Runner:
  def __init__(self) -> None:
    self._full_session_id = "session-test"
    self._agent_session_log: object | None = object()
    self.spawn_calls: list[dict[str, Any]] = []
    self.background_calls: list[dict[str, Any]] = []
    self.durable_events: list[dict[str, Any]] = []
    self.background_result_calls: list[dict[str, Any]] = []

  def _get_tool_definitions(self) -> list[dict[str, Any]]:
    return [{
      "name": "web_search",
      "description": "Read-only evidence search.",
      "input_schema": {"type": "object"},
    }]

  async def _append_durable_event(self, event: dict[str, Any]) -> object:
    self.durable_events.append(dict(event))
    return object()

  async def _confirm_durable_skill_event(
    self,
    event: dict[str, Any],
  ) -> dict[str, Any] | None:
    return dict(event) if event in self.durable_events else None

  async def spawn_sub_agent(self, task: str, **kwargs: Any):
    call = {"task": task, **kwargs}
    self.spawn_calls.append(call)
    return _spawn_result(call).model_dump(mode="json"), None

  async def _register_background_task(self, **kwargs: Any):
    self.background_calls.append(dict(kwargs))
    return {"task_id": kwargs["task_id_override"], "status": "running"}, None

  async def get_background_result(
    self,
    tool_input: dict[str, Any],
  ):
    self.background_result_calls.append(dict(tool_input))
    return {"task_id": tool_input["task_id"], "status": "completed"}, None


def _handler(
  runner: _Runner | None,
  *,
  loader: SkillLoader | None,
  resolver: _CapabilityResolver | None = None,
  background_handlers: dict[str, Any] | None = None,
  operation_mcp_activator: Any | None = None,
  mcp_meta_inject_servers: frozenset[str] | None = None,
  fms_rebinder: Any | None = None,
  approval_store: Any | None = None,
  approval_policy: Any | None = None,
  approved_tool_types: set[str] | None = None,
):
  handlers = {"web_search": _tool}
  handlers.update(background_handlers or {})
  parent_session = GatewaySession(
    session_id="session-test",
    api_key_hash="hash",
    created_at=1,
    expires_at=2,
    user_id="actor-test",
    owner_user_id="actor-test",
    user_email="actor@example.com",
    role="owner",
    tenant_id="tenant-test",
    channel="cli",
    auth_config={"provider": "openai", "api_key": "opaque"},
  )
  parent_session.approval_store = approval_store
  parent_session.approval_policy = approval_policy
  parent_session.approved_tool_types = set(approved_tool_types or ())
  return make_run_agent_handler(
    [runner],
    parent_session=parent_session,
    skill_loader=loader,
    mcp_client=_McpClient(),
    local_tool_handlers=handlers,
    capability_execution_resolver=resolver or _CapabilityResolver(),
    operation_mcp_activator=operation_mcp_activator,
    mcp_meta_inject_servers=mcp_meta_inject_servers,
    fms_rebinder=fms_rebinder,
  )


def _write_operation(path: Path, *, resumable: bool = True) -> SkillLoader:
  path.mkdir(parents=True, exist_ok=True)
  (path / "filing-review.md").write_text(
    f"""---
name: filing-review
version: '1.0'
agent_callable: true
agent_description: Review filing evidence.
mutation_mode: read_only
resumable: {str(resumable).lower()}
allowed_tools:
  - web_search
mcp_tools: {{}}
semantic_metadata:
  tool_refs:
    - kind: local
      tool_id: web_search
  capability_requirements:
    - name: research-evidence.read/v1
      required: true
      binding_modes: [live_tool]
---
Review the admitted filing evidence and return a source-aware conclusion.
""",
    encoding="utf-8",
  )
  return SkillLoader(path)


def _operation(loader: SkillLoader) -> dict[str, Any]:
  return next(
    item.snapshot.operation.model_dump(mode="json")
    for item in loader.list_callable_operations()
    if item.snapshot.operation.name == "filing-review"
  )


def test_run_agent_tool_schema_is_operation_first(tmp_path: Path) -> None:
  schema = make_run_agent_tool_def(_write_operation(tmp_path))["input_schema"]

  assert schema["required"] == ["objective"]
  assert schema["additionalProperties"] is False
  assert set(schema["properties"]) >= {
    "operation", "objective", "ticker", "research_file_id", "background",
  }
  assert schema["properties"]["ticker"]["pattern"].startswith("^[A-Z0-9]")
  assert schema["properties"]["research_file_id"]["minimum"] == 1
  threshold = schema["properties"]["cost_observation_threshold_usd"]
  assert threshold["exclusiveMinimum"] == 0
  assert "never stops or fails" in threshold["description"]
  assert "max_budget_usd" not in schema["properties"]
  assert "agent" not in schema["properties"]
  assert "task" not in schema["properties"]


def test_direct_handler_rejects_retired_budget_input_without_dispatch(
  tmp_path: Path,
) -> None:
  runner = _Runner()

  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
  )({
    "objective": "Research the question.",
    "max_budget_usd": 10,
  }))

  assert result is None
  assert error is not None and error["code"] == "invalid_input"
  assert "max_budget_usd" in error["message"]
  assert runner.spawn_calls == []
  assert runner.background_calls == []


def test_direct_handler_rejects_ticker_that_conflicts_with_objective(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "ticker": "PCTY",
    "background": False,
  }))

  assert result is None
  assert error == {
    "code": "context_ticker_mismatch",
    "message": "ticker must match the exact ticker stated in objective",
  }
  assert runner.spawn_calls == []


def test_direct_handler_rejects_research_file_that_conflicts_with_objective(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review MSFT with research_file_id=42.",
    "research_file_id": 43,
    "background": False,
  }))

  assert result is None
  assert error == {
    "code": "context_research_file_id_mismatch",
    "message": "research_file_id must match the exact ID stated in objective",
  }
  assert runner.spawn_calls == []


def test_generic_delegation_uses_exact_canonical_contracts(tmp_path: Path) -> None:
  runner = _Runner()
  resolver = _CapabilityResolver()

  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
    resolver=resolver,
  )({
    "objective": "Find the load-bearing evidence.",
    "background": False,
    "cost_observation_threshold_usd": 2.75,
  }))

  assert error is None
  canonical = TaskResult.model_validate(result)
  call = runner.spawn_calls[0]
  assert canonical.logical_task == call["logical_task"]
  assert canonical.attempt == call["attempt"]
  assert call["result_requirement"].mode == "narrative"
  assert call["result_provenance"] == canonical.provenance
  assert call["skill_name"] == "explore"
  assert call["cost_observation_threshold_usd"] == 2.75
  assert resolver.calls[0]["capability_id"] == "node.explore"
  assert set(call["dispatcher"]._local) == {"web_search"}


def test_foreground_delegation_returns_readable_completion_handle(
  tmp_path: Path,
) -> None:
  terminal_text = "Exact child result.\n" * 1_100

  class _DurableForegroundRunner(_Runner):
    def __init__(self) -> None:
      super().__init__()
      self._gateway_session_id = "session-test"
      self._runner_id = "parent-runner"
      self._role = "writer"
      self._workspace_dir = tmp_path / "workspace"
      self._workspace_dir.mkdir()
      self._agent_session_log = AgentSessionLog(
        tmp_path / "agent-session.jsonl"
      )

    async def _append_durable_event(
      self,
      event: dict[str, Any],
    ) -> object:
      return await self._agent_session_log.append(dict(event))

    async def spawn_sub_agent(self, task: str, **kwargs: Any):
      call = {"task": task, **kwargs}
      self.spawn_calls.append(call)
      result = _spawn_result(call)
      reference = publish_final_narrative(
        workspace_dir=self._workspace_dir,
        sub_agent_id=kwargs["attempt"].physical_task_id,
        terminal_event_seq=1,
        text=terminal_text,
      )
      result = result.model_copy(update={
        "values": TaskResultValues(
          terminal_narrative=terminal_narrative_content_handle(reference),
        ),
      })
      return result, None

  runner = _DurableForegroundRunner()
  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path / "skills"),
  )({
    "objective": "Return the complete evidence report.",
    "background": False,
  }))

  assert error is None
  envelope = AgentCompletionEnvelope.model_validate(result)
  assert envelope.parent_materialization.kind == "result_handle"
  source = envelope.parent_materialization.source
  grant = envelope.parent_materialization.read_grant
  page, read_error = asyncio.run(
    make_get_agent_result_content_handler([runner])({
      "content_id": source.content_id,
      "read_grant_id": grant.grant_id,
    })
  )
  assert read_error is None
  assert page is not None
  assert page["content"] == terminal_text
  assert page["end"] is True
  events, _cursor = asyncio.run(runner._agent_session_log.query(
    event_types={
      "task_registered",
      "task_completed",
      "agent_completion",
    },
    order="asc",
  ))
  assert [entry.event["type"] for entry in events] == [
    "task_registered",
    "task_completed",
    "agent_completion",
  ]


def test_registered_operation_requires_full_ref_and_emits_lifecycle(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  call = runner.spawn_calls[0]
  assert call["logical_task"].operation.name == "filing-review"
  assert call["result_requirement"].mode == "narrative"
  assert [event["type"] for event in runner.durable_events] == [
    "skill_run_started",
    "skill_result_captured",
  ]


def test_registered_operation_captures_recoverable_fms_error_without_internal_error(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()
  spawn_sub_agent = runner.spawn_sub_agent

  async def _spawn_with_fms_error(task: str, **kwargs: Any):
    result, error = await spawn_sub_agent(task, **kwargs)
    kwargs["dispatcher"]._event_log.append({
      "type": "tool_call_complete",
      "tool_name": "fms_persist_test_artifact",
      "result": {
        "status": "error",
        "subcommand": "persist_test_artifact",
        "mutation_mode": "model_writer",
        "error": {
          "type": "INVALID_JUDGMENT",
          "message": "active research file is required",
          "recoverable": True,
        },
      },
    })
    return result, error

  runner.spawn_sub_agent = _spawn_with_fms_error  # type: ignore[method-assign]
  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  captured = runner.durable_events[-1]
  assert captured["type"] == "skill_result_captured"
  assert captured["exit_code"] == 1
  assert captured["outcome"] == "error"
  assert captured["status"] == "error"
  assert captured["error"] == "active research file is required"


def test_registered_operation_activates_declared_mcp_before_admission(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()
  activated: list[str] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    operation_mcp_activator=lambda profile: activated.append(
      profile.name
    ),
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "ticker": "MSFT",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  assert activated == ["filing-review"]
  assert len(runner.spawn_calls) == 1


def test_registered_foreground_operation_inherits_parent_approval_lifecycle(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()
  approval_store = object()
  approval_policy = object()

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    approval_store=approval_store,
    approval_policy=approval_policy,
    approved_tool_types={"fms_persist_test_artifact"},
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  dispatcher = runner.spawn_calls[0]["dispatcher"]
  assert dispatcher._approval_store is approval_store
  assert dispatcher._approval_policy is approval_policy
  assert dispatcher._session is not None
  assert dispatcher._approved_tool_types == {"fms_persist_test_artifact"}
  assert dispatcher._should_avoid_permission_prompts is False
  started = next(
    event for event in runner.durable_events
    if event.get("type") == "skill_run_started"
  )
  assert dispatcher._run_context.run_id == started["skill_run_id"]
  assert runner.spawn_calls[0]["skill_run_id"] == started["skill_run_id"]
  assert dispatcher._run_context.skill == "filing-review"
  assert dispatcher._run_context.profile == "filing-review"


def test_registered_operation_rebinds_fms_to_objective_research_file(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()
  rebound: list[tuple[int, set[str]]] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    fms_rebinder=lambda handlers, research_file_id: rebound.append((
      research_file_id,
      set(handlers),
    )),
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT with research_file_id=42.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  assert rebound == [(42, {"web_search"})]


def test_registered_operation_binds_explicit_research_file_to_child_run(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()
  rebound: list[int] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    fms_rebinder=lambda _handlers, research_file_id: rebound.append(
      research_file_id
    ),
  )({
    "operation": _operation(loader),
    "objective": "Review the exact PCTY research file.",
    "ticker": "PCTY",
    "research_file_id": 42,
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  assert rebound == [42]
  dispatcher = runner.spawn_calls[0]["dispatcher"]
  assert dispatcher._run_context.research_file_id == 42


def test_registered_operation_mcp_activation_error_prevents_dispatch(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()
  activation_error = {
    "code": "mcp_tool_unavailable",
    "message": "Declared MCP tool is unavailable.",
  }

  result, error = asyncio.run(_handler(
    runner,
    loader=loader,
    operation_mcp_activator=lambda _profile: activation_error,
  )({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert result is None
  assert error == activation_error
  assert runner.spawn_calls == []


def test_generic_delegation_does_not_activate_named_operation_mcp(
  tmp_path: Path,
) -> None:
  runner = _Runner()
  activated: list[str] = []

  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
    operation_mcp_activator=lambda profile: activated.append(
      profile.name
    ),
  )({
    "objective": "Find the load-bearing evidence.",
    "background": False,
  }))

  assert error is None
  assert TaskResult.model_validate(result).execution.status == "succeeded"
  assert activated == []


def test_registered_operation_fails_closed_without_durable_log(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()
  runner._agent_session_log = None

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review MSFT filing evidence.",
    "background": False,
  }))

  assert result is None
  assert error is not None
  assert error["code"] == "durable_session_log_required"
  assert runner.spawn_calls == []


def test_bare_operation_name_is_not_an_authority_reference(tmp_path: Path) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": "filing-review",
    "objective": "Review the filing.",
  }))

  assert result is None
  assert error is not None and error["code"] == "invalid_operation"
  assert "full AgentOperationRef" in error["message"]


def test_background_registration_persists_exact_admitted_task(
  tmp_path: Path,
) -> None:
  loader = _write_operation(tmp_path)
  runner = _Runner()

  result, error = asyncio.run(_handler(runner, loader=loader)({
    "operation": _operation(loader),
    "objective": "Review the filing in the background.",
    "background": True,
    "cost_observation_threshold_usd": 3.25,
  }))

  assert error is None
  assert result is not None and result["status"] == "running"
  registration = runner.background_calls[0]
  admitted = registration["admitted_task"]
  assert isinstance(admitted, AdmittedTask)
  assert admitted.execution_disposition.kind == "execute"
  assert admitted.operation.operation.name == "filing-review"
  assert admitted.execution_snapshot.cost_observation_threshold_usd == 3.25
  assert registration["task_id_override"] == admitted.attempt.physical_task_id
  assert registration["tool_input"]["operation"] == _operation(loader)
  assert registration["tool_input"]["result_requirement"] == (
    admitted.result_requirement.model_dump(mode="json")
  )
  assert registration["tool_input"]["cost_observation_threshold_usd"] == 3.25
  assert ADMITTED_TASK_METADATA_KEY not in registration["tool_input"]
  assert "agent" not in registration["tool_input"]
  assert "task" not in registration["tool_input"]
  assert "child_tool_scope_receipt" not in registration["tool_input"]


@pytest.mark.parametrize("objective", [None, "", 0])
def test_objective_validation_is_canonical(
  tmp_path: Path,
  objective: object,
) -> None:
  result, error = asyncio.run(_handler(
    _Runner(),
    loader=SkillLoader(tmp_path),
  )({"objective": objective}))

  assert result is None
  assert error == {"code": "invalid_input", "message": "objective is required"}


def test_missing_runner_fails_before_admission(tmp_path: Path) -> None:
  result, error = asyncio.run(_handler(
    None,
    loader=SkillLoader(tmp_path),
  )({"objective": "Explore."}))

  assert result is None
  assert error == {
    "code": "internal_error",
    "message": "Sub-agent runner not initialized",
  }


def test_background_result_handler_proxies_exact_task_id() -> None:
  runner = _Runner()

  result, error = asyncio.run(
    make_get_background_result_handler([runner])({"task_id": "bg_7"})
  )

  assert error is None
  assert result == {"task_id": "bg_7", "status": "completed"}
  assert runner.background_result_calls == [{"task_id": "bg_7"}]


def test_background_and_resume_tool_schemas_are_explicit() -> None:
  background = make_get_background_result_tool_def()["input_schema"]
  resume = make_resume_tool_def()["input_schema"]

  assert background["required"] == ["task_id"]
  assert resume["required"] == ["task_id"]
  assert "additional_context" in resume["properties"]


def test_skill_lifecycle_emitter_projects_canonical_task_result_once(
  tmp_path: Path,
) -> None:
  runner = _Runner()
  result, error = asyncio.run(_handler(
    runner,
    loader=SkillLoader(tmp_path),
  )({"objective": "Collect evidence.", "background": False}))
  assert error is None
  durable: list[dict[str, Any]] = []
  projected: list[dict[str, Any]] = []
  event_log = EventLog()

  async def append(event: dict[str, Any]) -> object:
    durable.append(dict(event))
    return object()

  async def confirm(event: dict[str, Any]) -> dict[str, Any] | None:
    return dict(event) if event in durable else None

  emitter = SkillRunEventEmitter(
    skill_run_id="skill-run-1",
    profile=SimpleNamespace(name="filing-review"),
    semantic_scope="ticker",
    context_ticker="msft",
    portfolio_id=None,
    event_log_getter=lambda: event_log,
    tool_ctx=SimpleNamespace(emit=projected.append),
    durable_appender=append,
    durable_confirmer=confirm,
    time_fn=lambda: 1.0,
  )

  assert asyncio.run(emitter.emit_started()) is True
  assert asyncio.run(emitter.emit_started()) is True
  assert asyncio.run(emitter.emit_result_captured(result, None)) is True
  assert [event["type"] for event in durable] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  assert [event["type"] for event in projected] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  assert projected[-1]["status"] == "not_assessed"
  assert len(event_log.entries) == 2


def test_skill_lifecycle_result_projects_after_child_stream_closes() -> None:
  durable: list[dict[str, Any]] = []
  parent_log = EventLog()
  child_log = EventLog()

  async def append(event: dict[str, Any]) -> object:
    durable.append(dict(event))
    return object()

  async def confirm(event: dict[str, Any]) -> dict[str, Any] | None:
    return dict(event) if event in durable else None

  emitter = SkillRunEventEmitter(
    skill_run_id="skill-run-terminal-child",
    profile=SimpleNamespace(name="valuation-policy-precompile"),
    semantic_scope="ticker",
    context_ticker="PCTY",
    portfolio_id=None,
    event_log_getter=lambda: child_log,
    tool_ctx=ToolExecutionContext(
      tool_call_id="tool-1",
      tool_name="run_agent",
      event_log=parent_log,
    ),
    durable_appender=append,
    durable_confirmer=confirm,
    time_fn=lambda: 1.0,
  )

  assert asyncio.run(emitter.emit_started()) is True
  child_log.append({"type": "stream_complete"})
  assert asyncio.run(
    emitter.emit_result_captured(
      {"status": "completed", "result": "done"},
      None,
    )
  ) is True

  assert [entry.event["type"] for entry in parent_log.entries] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  assert [entry.event["type"] for entry in child_log.entries] == [
    "stream_complete",
  ]


def test_skill_lifecycle_emitter_fails_closed_without_confirmation() -> None:
  async def append(_event: dict[str, Any]) -> None:
    return None

  async def never_confirm(
    _event: dict[str, Any],
  ) -> dict[str, Any] | None:
    return None

  emitter = SkillRunEventEmitter(
    skill_run_id="skill-run-unconfirmed",
    profile=SimpleNamespace(name="filing-review"),
    semantic_scope=None,
    context_ticker=None,
    portfolio_id=None,
    event_log_getter=EventLog,
    tool_ctx=None,
    durable_appender=append,
    durable_confirmer=never_confirm,
  )

  with pytest.raises(DurableSkillEventPersistenceError):
    asyncio.run(emitter.emit_started())
