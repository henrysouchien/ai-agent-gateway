import asyncio
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
TESTS_DIR = Path(__file__).resolve().parent
for path in (PKG_DIR, TESTS_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent_gateway import AgentRunner, AgentSessionLog, EventLog  # noqa: E402
from agent_gateway.agent_session_log_records import EVENT_SCHEMA_VERSION  # noqa: E402
import agent_gateway.skill_completion_wal as completion_wal_module  # noqa: E402
from agent_gateway.autonomous import run_session  # noqa: E402
from agent_gateway.providers import StreamEvent  # noqa: E402
from agent_gateway.runner_session_events import (  # noqa: E402
  build_skill_run_started_event,
)
from agent_gateway.skill_lifecycle import (  # noqa: E402
  TopLevelSkillAdmission,
  TopLevelSkillCompletionPlan,
  TopLevelSkillLifecycleMetadata,
  TopLevelSkillResultPolicy,
  resolve_skill_lifecycle_artifact_identity,
)
from agent_gateway.skill_completion_wal import (  # noqa: E402
  SkillCompletionEffectConflict,
  SkillCompletionWal,
  SkillCompletionWalCorruptError,
  TopLevelSkillCompletionEffectPlan,
  apply_completion_effect,
)
from test_session_log_runner import (  # noqa: E402
  _ScriptedProvider,
  _child_report,
  _make_dispatcher,
  _runner_execution,
  _text_turn,
  _tool_turn,
)


def _run(coro: Any) -> Any:
  return asyncio.run(coro)


def _bind_receipt() -> dict[str, str]:
  return {
    "schema_version": "1.0",
    "capability_id": "node.implement",
    "model_key": "test.stub.claude-sonnet-4-6",
    "provider": "stub",
    "upstream_model": "claude-sonnet-4-6",
    "adapter": "test.stub",
    "protocol_profile": "test.reasoning",
    "route": "test.in_process",
    "effort": "none",
    "credential_principal": "user",
    "credential_ref": "test-user:stub",
    "run_mode": "interactive",
    "registry_revision": "test-capability-execution.1",
    "policy_revision": "test-capability-execution.1",
    "selection_source": "internal_policy",
  }


class _RecordingContextCapture:
  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  def persist(
    self,
    *,
    surfaces: list[dict[str, Any]],
    rendered_system_prompt: Any,
  ) -> str:
    self.calls.append(
      {
        "surfaces": surfaces,
        "rendered_system_prompt": rendered_system_prompt,
      }
    )
    return "sha256:prompt"


def _metadata() -> TopLevelSkillLifecycleMetadata:
  return TopLevelSkillLifecycleMetadata(
    skill_run_id="skill-run-1",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )


def _top_level_started_event(
  lifecycle: TopLevelSkillLifecycleMetadata,
  *,
  started_at: float,
  runner_id: str = "runner-original",
) -> dict[str, Any]:
  return _durable_writer_event(
    {
      **build_skill_run_started_event(
        lifecycle,
        started_at=started_at,
      ),
      "lifecycle_origin": "top_level",
    },
    runner_id=runner_id,
  )


def _durable_writer_event(
  event: dict[str, Any],
  *,
  runner_id: str = "runner-original",
) -> dict[str, Any]:
  return {
    **event,
    "runner_id": runner_id,
    "role": "writer",
    "event_schema_version": EVENT_SCHEMA_VERSION,
  }


def _result_event(
  lifecycle: TopLevelSkillLifecycleMetadata,
  terminal_event: dict[str, Any],
) -> dict[str, Any]:
  succeeded = (
    terminal_event.get("type") == "stream_complete"
    and terminal_event.get("terminal_disposition") != "interrupted"
  )
  timed_out = terminal_event.get("reason") == "timeout"
  outcome = (
    "success"
    if succeeded
    else ("timeout" if timed_out else "error")
  )
  return {
    "type": "skill_result_captured",
    **lifecycle.identity_fields(),
    "exit_code": 0 if succeeded else (124 if timed_out else 1),
    "outcome": outcome,
    "status": outcome,
    "gate_code": None,
    "artifact_refs": [],
    "proposal_ids": [],
    "verdict_echo": None,
    "fms_results": [],
    "artifact_events": [],
    "output_memory_file": "skills/fundamental-research/result.md",
    "cost_usd": 0.0,
    "duration_s": 0.1,
    "compaction_count": 0,
    "error": None if succeeded else str(
      terminal_event.get("error") or "interrupted"
    ),
    "warnings": [],
    "approval_outcome": None,
    "approval_id": None,
    "approval_tool_name": None,
  }


def _completion_plan(
  result_event: dict[str, Any],
  terminal_event: dict[str, Any],
  *,
  effect: TopLevelSkillCompletionEffectPlan | None = None,
) -> TopLevelSkillCompletionPlan:
  return TopLevelSkillCompletionPlan(
    result_event=result_event,
    terminal_event=terminal_event,
    effect=effect or TopLevelSkillCompletionEffectPlan.noop(),
  )


def _wal_intent_record(
  *,
  skill_run_id: str = "skill-run-wal",
) -> dict[str, Any]:
  lifecycle = {
    **_metadata().identity_fields(),
    "skill_run_id": skill_run_id,
  }
  return {
    "record_type": "intent",
    "skill_run_id": skill_run_id,
    "lifecycle": lifecycle,
    "result": {"type": "skill_result_captured"},
    "terminal": {"type": "stream_complete"},
    "effect": TopLevelSkillCompletionEffectPlan.noop().durable_payload(),
    "fence": {
      "generation": 1,
      "owner_token": "a" * 64,
    },
  }


def _wal_settled_record(
  *,
  skill_run_id: str = "skill-run-wal",
) -> dict[str, Any]:
  return {
    "record_type": "settled",
    "skill_run_id": skill_run_id,
    "result_digest": "sha256:" + "b" * 64,
    "terminal_digest": "sha256:" + "c" * 64,
    "fence": {
      "generation": 1,
      "owner_token": "a" * 64,
    },
    "reason": "exact_result_and_terminal_committed",
  }


def _policy(
  lifecycle: TopLevelSkillLifecycleMetadata,
) -> TopLevelSkillResultPolicy:
  return TopLevelSkillResultPolicy(
    prepare_provider=lambda proposed_prompt: proposed_prompt,
    prepare_completion=lambda _event_log, terminal_event: (
      _completion_plan(
        _result_event(lifecycle, terminal_event),
        terminal_event,
      )
    ),
  )


def _runner(
  *,
  log: AgentSessionLog,
  provider: Any,
  event_log: EventLog | None = None,
  dispatcher: Any | None = None,
  lifecycle: TopLevelSkillLifecycleMetadata | None = None,
  policy: TopLevelSkillResultPolicy | None = None,
  context_capture: Any | None = None,
  workspace_dir: Path | None = None,
) -> AgentRunner:
  lifecycle = lifecycle or _metadata()
  policy = policy or _policy(lifecycle)
  active_event_log = event_log or EventLog()
  admission = TopLevelSkillAdmission.acquire(log.path)
  return AgentRunner(
    event_log=active_event_log,
    dispatcher=dispatcher or _make_dispatcher(
      event_log=active_event_log
    ),
    session_id="sess-parent",
    capability_execution=_runner_execution(provider),
    agent_session_log=log,
    context_capture=context_capture,
    top_level_skill_admission=admission,
    top_level_skill_lifecycle=lifecycle,
    top_level_skill_result_policy=policy,
    workspace_dir=workspace_dir or log.path.parent,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


@pytest.mark.parametrize(
  ("semantic_scope", "ticker", "portfolio_id", "expected"),
  [
    (
      "ticker",
      "brk.b",
      "ignored-for-ticker",
      {
        "scope": "ticker",
        "ticker": "BRKB",
        "portfolio_id": None,
      },
    ),
    (
      "ticker",
      None,
      "portfolio-1",
      {
        "scope": "portfolio",
        "ticker": None,
        "portfolio_id": "portfolio-1",
      },
    ),
    (
      "industry",
      "PCTY",
      "portfolio-2",
      {
        "scope": "portfolio",
        "ticker": None,
        "portfolio_id": "portfolio-2",
      },
    ),
  ],
)
def test_canonical_lifecycle_identity_keeps_semantic_scope_separate(
  semantic_scope: str,
  ticker: str | None,
  portfolio_id: str | None,
  expected: dict[str, Any],
) -> None:
  identity = resolve_skill_lifecycle_artifact_identity(
    semantic_scope=semantic_scope,
    context_ticker=ticker,
    portfolio_id=portfolio_id,
  )

  assert identity.identity_fields() == expected


def test_ticker_lifecycle_identity_requires_concrete_ticker() -> None:
  with pytest.raises(ValueError, match="ticker scope requires ticker"):
    TopLevelSkillLifecycleMetadata(
      skill_run_id="skill-run-1",
      skill="fundamental-research",
      scope="ticker",
      ticker=None,
      portfolio_id=None,
    )


def test_unhashable_semantic_scope_is_rejected_as_invalid() -> None:
  with pytest.raises(ValueError, match="semantic_scope"):
    resolve_skill_lifecycle_artifact_identity(
      semantic_scope=[],  # type: ignore[arg-type]
      context_ticker=None,
      portfolio_id=None,
    )


@pytest.mark.parametrize(
  "nested_value",
  [
    {"bad": float("nan")},
    {1: "non-string-key"},
    {"bad": object()},
  ],
)
def test_result_normalizer_rejects_noncanonical_nested_json(
  nested_value: dict[Any, Any],
) -> None:
  lifecycle = _metadata()
  event = _result_event(
    lifecycle,
    {
      "type": "stream_complete",
      "terminal_disposition": "completed",
    },
  )
  event["verdict_echo"] = nested_value

  with pytest.raises(RuntimeError):
    lifecycle.normalize_result_event(event)


def test_result_normalizer_rejects_nested_cycle() -> None:
  lifecycle = _metadata()
  event = _result_event(
    lifecycle,
    {
      "type": "stream_complete",
      "terminal_disposition": "completed",
    },
  )
  cyclic: dict[str, Any] = {}
  cyclic["self"] = cyclic
  event["verdict_echo"] = cyclic

  with pytest.raises(RuntimeError, match="cycle"):
    lifecycle.normalize_result_event(event)


@pytest.mark.parametrize(
  ("causes", "authoritative_cause"),
  [
    (
      ("caller_cancellation", "shutdown", "timeout"),
      "caller_cancellation",
    ),
    (("shutdown", "timeout", "caller_cancellation"), "shutdown"),
    (("timeout", "caller_cancellation", "shutdown"), "timeout"),
  ],
)
def test_server_terminal_cause_keeps_first_authoritative_trigger(
  causes: tuple[str, ...],
  authoritative_cause: str,
) -> None:
  seen_terminal_events: list[dict[str, Any]] = []

  def _prepare_completion(
    _event_log: Any,
    terminal_event: dict[str, Any],
  ) -> TopLevelSkillCompletionPlan:
    seen_terminal_events.append(terminal_event)
    return _completion_plan({}, terminal_event)

  policy = TopLevelSkillResultPolicy(
    prepare_provider=lambda proposed: proposed,
    prepare_completion=_prepare_completion,
  )
  for index, cause in enumerate(causes):
    accepted = policy.set_server_terminal_cause(  # type: ignore[arg-type]
      cause
    )
    assert accepted is (index == 0)
  assert policy.server_terminal_cause == authoritative_cause

  receipt = _run(
    policy.prepare(
      EventLog(),
      {
        "type": "stream_complete",
        "terminal_disposition": "completed",
        "usage": {"cost_usd": 0.1},
      },
    )
  )

  assert receipt.result_event == {}
  assert seen_terminal_events == [{
    "type": "stream_complete",
    "terminal_disposition": "interrupted",
    "reason": authoritative_cause,
    "server_terminal_cause": authoritative_cause,
    "usage": {"cost_usd": 0.1},
  }]
  assert policy.set_server_terminal_cause(  # type: ignore[arg-type]
    authoritative_cause
  )


def test_provider_policy_reuse_requires_exact_nested_prompt() -> None:
  async def _case() -> None:
    policy = TopLevelSkillResultPolicy(
      prepare_provider=lambda proposed: proposed,
      prepare_completion=lambda _log, terminal: (
        _completion_plan({}, terminal)
      ),
    )
    assert await policy.prepare_system_prompt(
      {"nested": {"value": 1}}
    ) == {"nested": {"value": 1}}

    with pytest.raises(RuntimeError, match="another prompt"):
      await policy.prepare_system_prompt(
        {"nested": {"value": True}}
      )

  _run(_case())


def test_completion_policy_reuse_requires_exact_nested_terminal() -> None:
  async def _case() -> None:
    event_log = EventLog()
    policy = TopLevelSkillResultPolicy(
      prepare_provider=lambda proposed: proposed,
      prepare_completion=lambda _log, terminal: (
        _completion_plan({}, terminal)
      ),
    )
    assert (
      await policy.prepare(
        event_log,
        {
          "type": "stream_complete",
          "usage": {"output_tokens": 1},
        },
      )
    ).result_event == {}

    with pytest.raises(RuntimeError, match="another terminal"):
      await policy.prepare(
        event_log,
        {
          "type": "stream_complete",
          "usage": {"output_tokens": True},
        },
      )

  _run(_case())


def test_repeated_cancellation_drains_single_completion_effect() -> None:
  async def _case() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def _prepare_completion(
      _event_log: Any,
      terminal_event: dict[str, Any],
    ) -> TopLevelSkillCompletionPlan:
      nonlocal calls
      calls += 1
      entered.set()
      await release.wait()
      return _completion_plan(
        {"receipt": {"nested": ["owned"]}},
        terminal_event,
      )

    policy = TopLevelSkillResultPolicy(
      prepare_provider=lambda proposed: proposed,
      prepare_completion=_prepare_completion,
    )
    task = asyncio.create_task(
      policy.prepare(
        EventLog(),
        {
          "type": "stream_complete",
          "terminal_disposition": "interrupted",
          "reason": "caller_cancellation",
        },
      )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    plan = await asyncio.wait_for(task, timeout=1.0)
    assert plan.result_event == {
      "receipt": {"nested": ["owned"]}
    }
    assert calls == 1

  _run(_case())


def test_top_level_skill_lifecycle_is_exactly_once_and_ordered(
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "top-level.jsonl"
  )
  event_log = EventLog()
  context_capture = _RecordingContextCapture()
  provider_seen_types: list[str] = []
  provider_preparation_seen_types: list[str] = []

  async def _prepare_provider(proposed_prompt: Any) -> str:
    assert proposed_prompt == "system prompt"
    entries, _ = await log.query(order="asc")
    provider_preparation_seen_types.extend(
      entry.event["type"] for entry in entries
    )
    return "prepared system prompt"

  lifecycle = _metadata()
  policy = TopLevelSkillResultPolicy(
    prepare_provider=_prepare_provider,
    prepare_completion=lambda _event_log, terminal_event: (
      _completion_plan(
        _result_event(lifecycle, terminal_event),
        terminal_event,
      )
    ),
  )

  class _OrderingProvider(_ScriptedProvider):
    async def stream(self, client: Any, params: dict[str, Any]):
      entries, _ = await log.query(order="asc")
      provider_seen_types.extend(
        entry.event["type"] for entry in entries
      )
      async for event in super().stream(client, params):
        yield event

  runner = _runner(
    log=log,
    provider=_OrderingProvider([_text_turn("done")]),
    event_log=event_log,
    lifecycle=lifecycle,
    policy=policy,
    context_capture=context_capture,
  )

  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run the skill"}],
      system_prompt="system prompt",
    )
  )

  entries, _ = _run(log.query(order="asc"))
  durable_types = [entry.event["type"] for entry in entries]
  assert durable_types == [
    "attach",
    "user_message",
    "skill_run_started",
    "context_manifest",
    "assistant_message",
    "skill_result_captured",
    "stream_complete",
    "detach",
  ]
  assert provider_seen_types == [
    "attach",
    "user_message",
    "skill_run_started",
    "context_manifest",
  ]
  assert provider_preparation_seen_types == [
    "attach",
    "user_message",
    "skill_run_started",
  ]
  assert len(context_capture.calls) == 1
  assert context_capture.calls[0]["surfaces"] == []
  assert context_capture.calls[0]["rendered_system_prompt"].startswith(
    "prepared system prompt"
  )
  assert durable_types.index("user_message") < durable_types.index(
    "skill_run_started"
  ) < durable_types.index("context_manifest") < durable_types.index(
    "assistant_message"
  )
  lifecycle_events = [
    entry.event
    for entry in entries
    if entry.event["type"] in {
      "skill_run_started",
      "skill_result_captured",
    }
  ]
  assert len(lifecycle_events) == 2
  for event in lifecycle_events:
    assert {
      field_name: event[field_name]
      for field_name in _metadata().identity_fields()
    } == _metadata().identity_fields()

  live_types = [entry.event["type"] for entry in event_log.entries]
  assert live_types.count("skill_run_started") == 1
  assert live_types.count("skill_result_captured") == 1
  assert live_types.index("skill_result_captured") < live_types.index(
    "stream_complete"
  )
  committed = runner.committed_top_level_skill_result_event
  assert committed is not None
  assert all(
    lifecycle_events[1][field_name] == value
    for field_name, value in committed.items()
  )
  committed["artifact_refs"].append("mutated")
  assert "mutated" not in (
    runner.committed_top_level_skill_result_event or {}
  ).get(
    "artifact_refs",
    [],
  )


def test_top_level_result_follows_background_completion(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(
      path=tmp_path / "sessions" / "background.jsonl"
    )
    event_log = EventLog()
    provider = _ScriptedProvider([
      _tool_turn(
        tool_id="tool-bg",
        tool_name="spawn_background",
        tool_input={"task": "Collect context"},
      ),
      _text_turn("done"),
    ])
    runner_ref: list[AgentRunner | None] = [None]

    async def _background_handler(
      _tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, object], None]:
      await asyncio.sleep(0.02)
      return _child_report(), None

    async def _spawn_background(
      tool_input: dict[str, Any],
      **_kwargs: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
      runner = runner_ref[0]
      assert runner is not None
      return await runner._register_background_task(
        capability_bind_receipt=_bind_receipt(),
        tool_input=tool_input,
        handler=_background_handler,
        agent_name="researcher",
        parent_turn_id="turn-1",
      )

    dispatcher = _make_dispatcher(
      event_log=event_log,
      local_tool_handlers={
        "spawn_background": _spawn_background,
      },
    )
    runner = _runner(
      log=log,
      provider=provider,
      event_log=event_log,
      dispatcher=dispatcher,
    )
    runner_ref[0] = runner

    await runner.run(
      messages=[{"role": "user", "content": "Run with research"}]
    )

    entries, _ = await log.query(order="asc")
    durable_types = [entry.event["type"] for entry in entries]
    assert durable_types.index("skill_run_started") < (
      durable_types.index("task_registered")
    )
    assert durable_types.index("task_completed") < durable_types.index(
      "skill_result_captured"
    )
    assert durable_types.index("skill_result_captured") < (
      durable_types.index("error")
    )
    assert durable_types.count("skill_result_captured") == 1
    assert durable_types.count("error") == 1

    live_types = [
      entry.event["type"] for entry in event_log.entries
    ]
    assert live_types.index("skill_result_captured") < live_types.index(
      "error"
    )
    assert live_types.count("skill_result_captured") == 1
    assert live_types.count("error") == 1

  _run(_case())


def test_live_callback_cannot_mutate_committed_result(
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "callback-ownership.jsonl"
  )

  def _mutate_callback(
    event: dict[str, Any],
    _session_id: str,
  ) -> None:
    if event.get("type") == "skill_result_captured":
      event["artifact_refs"].append("callback-mutation")

  event_log = EventLog(on_event=_mutate_callback)
  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    event_log=event_log,
  )

  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run the skill"}]
    )
  )

  durable, _ = _run(
    log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
  )
  assert durable[0].event["artifact_refs"] == []
  assert (
    runner.committed_top_level_skill_result_event or {}
  )["artifact_refs"] == []
  live_result = next(
    entry.event
    for entry in event_log.entries
    if entry.event["type"] == "skill_result_captured"
  )
  assert live_result["artifact_refs"] == []


def test_startup_error_uses_exact_deferred_terminal_for_result(
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "startup-error.jsonl"
  )
  event_log = EventLog()
  provider = _ScriptedProvider([])
  runner = _runner(
    log=log,
    provider=provider,
    event_log=event_log,
  )
  def _fail_client_creation(
    _config: dict[str, Any],
    *,
    timeout: float | None = None,
  ) -> Any:
    _ = timeout
    raise RuntimeError("provider client creation failed")

  provider.create_client = _fail_client_creation  # type: ignore[method-assign]

  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run the skill"}]
    )
  )

  entries, _ = _run(log.query(order="asc"))
  durable_events = [entry.event for entry in entries]
  durable_types = [event["type"] for event in durable_events]
  assert durable_types == [
    "attach",
    "user_message",
    "skill_run_started",
    "skill_result_captured",
    "error",
    "detach",
  ]
  result_event = next(
    event
    for event in durable_events
    if event["type"] == "skill_result_captured"
  )
  error_event = next(
    event for event in durable_events if event["type"] == "error"
  )
  assert result_event["error"] == error_event["error"]
  assert error_event["error"] == (
    "Provider startup failed: could not create client for provider=stub."
  )
  detach_event = next(
    event
    for event in durable_events
    if event["type"] == "detach"
  )
  assert detach_event["reason"] == "error"

  live_types = [entry.event["type"] for entry in event_log.entries]
  assert live_types.index("skill_result_captured") < live_types.index(
    "error"
  )
  assert live_types.count("skill_result_captured") == 1
  assert live_types.count("error") == 1


@pytest.mark.parametrize("prepared", [False, True])
def test_top_level_failure_result_preserves_only_product_gate(
  prepared: bool,
) -> None:
  lifecycle = _metadata()
  runner = object.__new__(AgentRunner)
  runner._top_level_skill_lifecycle = lifecycle
  prepared_event = _result_event(
    lifecycle,
    {"type": "stream_complete"},
  )
  prepared_event.update(
    {
      "status": "noop",
      "gate_code": "PROCEED",
      "verdict_echo": {"gate_code": "PROCEED"},
      "fms_results": [
        {"status": "noop", "gate_code": "PROCEED"}
      ],
    }
  )
  runner._top_level_skill_result_event = (
    prepared_event if prepared else None
  )

  event = runner._top_level_skill_failure_result_event(
    failure=RuntimeError("fixed failure"),
    failure_code="fixed_gate",
  )

  assert event["gate_code"] == (
    "PROCEED" if prepared else None
  )
  assert event["error"] == (
    "fixed_gate: RuntimeError: fixed failure"
  )
  if prepared:
    assert event["verdict_echo"] == {"gate_code": "PROCEED"}
    assert event["fms_results"] == [
      {"status": "noop", "gate_code": "PROCEED"}
    ]


def test_interrupted_stream_complete_is_deferred_until_after_result(
  tmp_path: Path,
) -> None:
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "interrupted.jsonl"
  )
  event_log = EventLog()
  runner = _runner(
    log=log,
    provider=_ScriptedProvider([]),
    event_log=event_log,
  )
  runner.request_operator_pause()

  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run the skill"}]
    )
  )

  entries, _ = _run(log.query(order="asc"))
  durable_events = [entry.event for entry in entries]
  durable_types = [event["type"] for event in durable_events]
  assert durable_types.index("skill_result_captured") < (
    durable_types.index("stream_complete")
  ) < durable_types.index("detach")
  assert durable_types.count("skill_result_captured") == 1
  assert durable_types.count("stream_complete") == 1
  terminal_event = next(
    event
    for event in durable_events
    if event["type"] == "stream_complete"
  )
  assert terminal_event["terminal_disposition"] == "interrupted"
  assert terminal_event["reason"] == "operator_pause"

  live_types = [entry.event["type"] for entry in event_log.entries]
  assert live_types.index("skill_result_captured") < live_types.index(
    "stream_complete"
  )
  assert live_types.count("skill_result_captured") == 1
  assert live_types.count("stream_complete") == 1


def test_stale_lifecycle_is_recovered_before_new_session_admission(
  tmp_path: Path,
) -> None:
  lifecycle = _metadata()
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "stale.jsonl"
  )
  _run(
    log.append(
      _top_level_started_event(
        lifecycle,
        started_at=1.0,
      )
    )
  )
  context_capture = _RecordingContextCapture()
  provider_preparation_calls: list[Any] = []

  def _prepare_provider(proposed_prompt: Any) -> Any:
    provider_preparation_calls.append(proposed_prompt)
    return proposed_prompt

  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("must not run")]),
    lifecycle=lifecycle,
    policy=TopLevelSkillResultPolicy(
      prepare_provider=_prepare_provider,
      prepare_completion=lambda _event_log, terminal_event: (
        _completion_plan(
          _result_event(lifecycle, terminal_event),
          terminal_event,
        )
      ),
    ),
    context_capture=context_capture,
  )

  with pytest.raises(RuntimeError, match="Stale top-level"):
    _run(
      runner.run(
        messages=[{"role": "user", "content": "Retry stale run"}]
      )
    )

  after, _ = _run(log.query(order="asc"))
  assert context_capture.calls == []
  assert provider_preparation_calls == []
  assert [entry.event["type"] for entry in after] == [
    "skill_run_started",
    "skill_result_captured",
    "error",
    "interrupted",
  ]
  assert after[1].event["exit_code"] == 1
  assert after[1].event["outcome"] == "error"
  assert after[1].event["gate_code"] is None
  assert after[2].event["skill_run_id"] == lifecycle.skill_run_id
  assert (
    after[3].event["recovery_kind"]
    == "top_level_skill_orphan_reconciled"
  )
  assert not any(
    entry.event["type"] in {"attach", "user_message"}
    for entry in after
  )


def test_recovery_repairs_only_explicit_top_level_lifecycles(
  tmp_path: Path,
) -> None:
  child_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="child-skill-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  stale_top_level = TopLevelSkillLifecycleMetadata(
    skill_run_id="stale-top-level-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_top_level = TopLevelSkillLifecycleMetadata(
    skill_run_id="fresh-top-level-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "mixed-lifecycles.jsonl"
  )
  _run(
    log.append(
      build_skill_run_started_event(
        child_lifecycle,
        started_at=1.0,
      )
    )
  )
  _run(
    log.append(
      _result_event(
        child_lifecycle,
        {
          "type": "stream_complete",
          "terminal_disposition": "completed",
        },
      )
    )
  )
  _run(
    log.append(
      _top_level_started_event(
        stale_top_level,
        started_at=2.0,
      )
    )
  )

  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    lifecycle=fresh_top_level,
    policy=_policy(fresh_top_level),
  )
  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run fresh skill"}]
    )
  )

  entries, _ = _run(log.query(order="asc"))
  child_events = [
    entry.event
    for entry in entries
    if entry.event.get("skill_run_id")
    == child_lifecycle.skill_run_id
  ]
  assert [
    event["type"] for event in child_events
  ] == [
    "skill_run_started",
    "skill_result_captured",
  ]
  assert all(
    "lifecycle_origin" not in event
    for event in child_events
  )

  recovered_top_level_events = [
    entry.event
    for entry in entries
    if entry.event.get("skill_run_id")
    == stale_top_level.skill_run_id
  ]
  assert [
    event["type"] for event in recovered_top_level_events
  ] == [
    "skill_run_started",
    "skill_result_captured",
    "error",
    "interrupted",
  ]


def test_recovery_ignores_unmarked_pre_cutover_start(
  tmp_path: Path,
) -> None:
  unmarked_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="unmarked-pre-cutover-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="fresh-after-cutover-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "unmarked-cutover.jsonl"
  )
  _run(
    log.append(
      build_skill_run_started_event(
        unmarked_lifecycle,
        started_at=1.0,
      )
    )
  )

  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    lifecycle=fresh_lifecycle,
    policy=_policy(fresh_lifecycle),
  )
  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run fresh skill"}]
    )
  )

  entries, _ = _run(log.query(order="asc"))
  unmarked_events = [
    entry.event
    for entry in entries
    if entry.event.get("skill_run_id")
    == unmarked_lifecycle.skill_run_id
  ]
  assert [
    event["type"] for event in unmarked_events
  ] == ["skill_run_started"]
  assert "lifecycle_origin" not in unmarked_events[0]


def test_recovery_rejects_forged_marker_without_writer_wrapper(
  tmp_path: Path,
) -> None:
  forged_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="forged-marker-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="fresh-after-forged-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "forged-marker.jsonl"
  )
  _run(
    log.append(
      {
        **build_skill_run_started_event(
          forged_lifecycle,
          started_at=1.0,
        ),
        "lifecycle_origin": "top_level",
      }
    )
  )
  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("must not run")]),
    lifecycle=fresh_lifecycle,
    policy=_policy(fresh_lifecycle),
  )

  with pytest.raises(
    SkillCompletionWalCorruptError,
    match="invalid lifecycle identity",
  ):
    _run(
      runner.run(
        messages=[{"role": "user", "content": "Run fresh skill"}]
      )
    )

  entries, _ = _run(log.query(order="asc"))
  assert [entry.event["type"] for entry in entries] == [
    "skill_run_started"
  ]


@pytest.mark.parametrize(
  ("malformed_envelope", "error_match"),
  [
    ("result_missing_core_field", "Recovered top-level result is invalid"),
    ("result_identity_mismatch", "Recovered top-level result is invalid"),
    ("terminal_missing_identity", "terminal identity is invalid"),
  ],
)
def test_recovery_validates_complete_log_envelopes_before_shortcut(
  tmp_path: Path,
  malformed_envelope: str,
  error_match: str,
) -> None:
  stale_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id=f"malformed-complete-{malformed_envelope}",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id=f"fresh-after-{malformed_envelope}",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  terminal = {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    **stale_lifecycle.identity_fields(),
  }
  result = _result_event(stale_lifecycle, terminal)
  if malformed_envelope == "result_missing_core_field":
    result.pop("status")
  elif malformed_envelope == "result_identity_mismatch":
    result["ticker"] = "OTHR"
  else:
    terminal.pop("scope")
  log = AgentSessionLog(
    path=(
      tmp_path
      / "sessions"
      / f"malformed-complete-{malformed_envelope}.jsonl"
    )
  )
  _run(
    log.append(
      _top_level_started_event(
        stale_lifecycle,
        started_at=1.0,
      )
    )
  )
  _run(log.append(_durable_writer_event(result)))
  _run(log.append(_durable_writer_event(terminal)))

  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("must not run")]),
    lifecycle=fresh_lifecycle,
    policy=_policy(fresh_lifecycle),
  )
  with pytest.raises(
    SkillCompletionWalCorruptError,
    match=error_match,
  ):
    _run(
      runner.run(
        messages=[{"role": "user", "content": "Run fresh skill"}]
      )
    )

  entries, _ = _run(log.query(order="asc"))
  assert [entry.event["type"] for entry in entries] == [
    "skill_run_started",
    "skill_result_captured",
    "stream_complete",
  ]


_OTHER_GENERATION_FIELDS = {
  "skill_entry_digest": "sha256:" + "f" * 64,
  "skill_catalog_digest": "sha256:" + "9" * 64,
}


def test_recovery_tolerates_unknown_result_fields_from_other_generation(
  tmp_path: Path,
) -> None:
  """A settled lifecycle written by another generation's schema must
  not poison recovery: unknown durable envelope fields are ignored on
  read while the core result stays strictly validated."""
  stale_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="other-generation-settled-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="fresh-after-other-generation-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  terminal = {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    **stale_lifecycle.identity_fields(),
    **_OTHER_GENERATION_FIELDS,
  }
  result = {
    **_result_event(stale_lifecycle, terminal),
    **_OTHER_GENERATION_FIELDS,
  }
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "other-generation-settled.jsonl"
  )
  _run(
    log.append(
      {
        **_top_level_started_event(
          stale_lifecycle,
          started_at=1.0,
        ),
        **_OTHER_GENERATION_FIELDS,
      }
    )
  )
  _run(log.append(_durable_writer_event(result)))
  _run(log.append(_durable_writer_event(terminal)))

  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    lifecycle=fresh_lifecycle,
    policy=_policy(fresh_lifecycle),
  )
  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run fresh skill"}]
    )
  )

  entries, _ = _run(log.query(order="asc"))
  stale_events = [
    entry.event
    for entry in entries
    if entry.event.get("skill_run_id")
    == stale_lifecycle.skill_run_id
  ]
  assert [event["type"] for event in stale_events] == [
    "skill_run_started",
    "skill_result_captured",
    "stream_complete",
  ]
  for event in stale_events:
    for field_name, value in _OTHER_GENERATION_FIELDS.items():
      assert event[field_name] == value
  fresh_events = [
    entry.event
    for entry in entries
    if entry.event.get("skill_run_id")
    == fresh_lifecycle.skill_run_id
  ]
  assert "skill_result_captured" in {
    event["type"] for event in fresh_events
  }


def test_recovery_reconciles_orphaned_result_with_unknown_fields(
  tmp_path: Path,
) -> None:
  """An orphaned result carrying unknown fields is recovered: the core
  result is reused (never dropped or re-invented), the existing durable
  envelope is preserved byte-for-byte, and later runs start."""
  stale_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="other-generation-orphan-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="fresh-after-orphan-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  success_terminal = {
    "type": "stream_complete",
    "terminal_disposition": "completed",
  }
  result = {
    **_result_event(stale_lifecycle, success_terminal),
    **_OTHER_GENERATION_FIELDS,
  }
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "other-generation-orphan.jsonl"
  )
  _run(
    log.append(
      _top_level_started_event(
        stale_lifecycle,
        started_at=1.0,
      )
    )
  )
  _run(log.append(_durable_writer_event(result)))

  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    lifecycle=fresh_lifecycle,
    policy=_policy(fresh_lifecycle),
  )
  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run fresh skill"}]
    )
  )

  entries, _ = _run(log.query(order="asc"))
  stale_events = [
    entry.event
    for entry in entries
    if entry.event.get("skill_run_id")
    == stale_lifecycle.skill_run_id
  ]
  assert [event["type"] for event in stale_events] == [
    "skill_run_started",
    "skill_result_captured",
    "stream_complete",
    "interrupted",
  ]
  recovered_result = stale_events[1]
  assert recovered_result["outcome"] == "success"
  for field_name, value in _OTHER_GENERATION_FIELDS.items():
    assert recovered_result[field_name] == value
  recovered_terminal = stale_events[2]
  assert (
    recovered_terminal["reason"]
    == "recovered_after_result_commit"
  )
  assert (
    stale_events[3]["recovery_kind"]
    == "top_level_skill_orphan_reconciled"
  )
  fresh_events = [
    entry.event
    for entry in entries
    if entry.event.get("skill_run_id")
    == fresh_lifecycle.skill_run_id
  ]
  assert "skill_result_captured" in {
    event["type"] for event in fresh_events
  }


@pytest.mark.parametrize(
  ("drift_event", "field_name", "field_value"),
  [
    ("result", "runner_id", "runner-other"),
    ("terminal", "role", "parent"),
    ("terminal", "sub_agent_id", "child-1"),
  ],
)
def test_recovery_rejects_durable_wrapper_drift(
  tmp_path: Path,
  drift_event: str,
  field_name: str,
  field_value: str,
) -> None:
  stale_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id=(
      f"wrapper-drift-{drift_event}-{field_name}"
    ),
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id=(
      f"fresh-wrapper-drift-{drift_event}-{field_name}"
    ),
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  log = AgentSessionLog(
    path=(
      tmp_path
      / "sessions"
      / f"wrapper-drift-{drift_event}-{field_name}.jsonl"
    )
  )
  terminal = {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    "reason": "completed",
    **stale_lifecycle.identity_fields(),
  }
  result = _durable_writer_event(
    _result_event(stale_lifecycle, terminal)
  )
  durable_terminal = _durable_writer_event(terminal)
  if drift_event == "result":
    result[field_name] = field_value
  else:
    durable_terminal[field_name] = field_value
  _run(
    log.append(
      _top_level_started_event(
        stale_lifecycle,
        started_at=1.0,
      )
    )
  )
  _run(log.append(result))
  _run(log.append(durable_terminal))

  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("must not run")]),
    lifecycle=fresh_lifecycle,
    policy=_policy(fresh_lifecycle),
  )
  with pytest.raises(SkillCompletionWalCorruptError):
    _run(
      runner.run(
        messages=[{"role": "user", "content": "Run fresh skill"}]
      )
    )


def test_recovery_preserves_genuine_writer_wrapper(
  tmp_path: Path,
) -> None:
  stale_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="genuine-wrapper-stale-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="genuine-wrapper-fresh-run",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "genuine-wrapper.jsonl"
  )
  successful_terminal = {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    **stale_lifecycle.identity_fields(),
  }
  _run(
    log.append(
      _top_level_started_event(
        stale_lifecycle,
        started_at=1.0,
      )
    )
  )
  _run(
    log.append(
      _durable_writer_event(
        _result_event(stale_lifecycle, successful_terminal)
      )
    )
  )

  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    lifecycle=fresh_lifecycle,
    policy=_policy(fresh_lifecycle),
  )
  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run fresh skill"}]
    )
  )

  entries, _ = _run(log.query(order="asc"))
  recovered = [
    entry.event
    for entry in entries
    if entry.event.get("skill_run_id")
    == stale_lifecycle.skill_run_id
  ]
  assert [event["type"] for event in recovered] == [
    "skill_run_started",
    "skill_result_captured",
    "stream_complete",
    "interrupted",
  ]
  for event in recovered[:3]:
    assert event["runner_id"] == "runner-original"
    assert event["role"] == "writer"
    assert "sub_agent_id" not in event
    assert event["event_schema_version"] == EVENT_SCHEMA_VERSION


class _AmbiguousResultLog(AgentSessionLog):
  def append_sync(self, event: dict[str, Any]) -> Any:
    entry = super().append_sync(event)
    if event.get("type") == "skill_result_captured":
      raise OSError("append acknowledgement lost")
    return entry


class _DefiniteAppendLog(AgentSessionLog):
  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self.started_appended = False
    self.result_appended = False

  async def append(self, event: dict[str, Any]) -> Any:
    entry = await super().append(event)
    if event.get("type") == "skill_run_started":
      self.started_appended = True
    return entry

  def append_sync(self, event: dict[str, Any]) -> Any:
    entry = super().append_sync(event)
    if event.get("type") == "skill_result_captured":
      self.result_appended = True
    return entry

  async def query(self, **kwargs: Any) -> Any:
    event_types = kwargs.get("event_types")
    if (
      self.started_appended
      and event_types == {"skill_run_started"}
    ):
      raise AssertionError(
        "definite start append must not be requeried"
      )
    if (
      self.result_appended
      and event_types == {"skill_result_captured"}
    ):
      raise AssertionError(
        "definite result append must not be requeried"
      )
    return await super().query(**kwargs)


def test_definite_lifecycle_appends_do_not_requery(
  tmp_path: Path,
) -> None:
  log = _DefiniteAppendLog(
    path=tmp_path / "sessions" / "definite-result.jsonl"
  )
  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
  )

  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run the skill"}]
    )
  )

  assert log.started_appended
  assert log.result_appended


def test_ambiguous_result_append_is_confirmed_without_duplicate(
  tmp_path: Path,
) -> None:
  log = _AmbiguousResultLog(
    path=tmp_path / "sessions" / "ambiguous-result.jsonl"
  )
  event_log = EventLog()
  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    event_log=event_log,
  )

  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run the skill"}]
    )
  )

  captured, _ = _run(
    log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
  )
  assert len(captured) == 1
  live_types = [entry.event["type"] for entry in event_log.entries]
  assert live_types.count("skill_result_captured") == 1
  assert "stream_complete" in live_types


class _FailedResultLog(AgentSessionLog):
  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self._failed_result_once = False

  def append_sync(self, event: dict[str, Any]) -> Any:
    if (
      event.get("type") == "skill_result_captured"
      and not self._failed_result_once
    ):
      self._failed_result_once = True
      raise OSError("result append failed before write")
    return super().append_sync(event)


def test_sync_result_failure_retries_exact_success_after_state_effect(
  tmp_path: Path,
) -> None:
  lifecycle = _metadata()
  state_path = tmp_path / "state.json"
  state_path.write_bytes(b'{"state":"before"}\n')

  def _prepare_state_plan(
    _event_log: Any,
    terminal_event: dict[str, Any],
  ) -> TopLevelSkillCompletionPlan:
    return _completion_plan(
      _result_event(lifecycle, terminal_event),
      terminal_event,
      effect=(
        TopLevelSkillCompletionEffectPlan.canonical_json_update(
          workspace_path=tmp_path,
          target_path=state_path,
          update=lambda _exists, _before: {
            "state": "success",
          },
        )
      ),
    )

  log = _FailedResultLog(
    path=tmp_path / "sessions" / "failed-result.jsonl"
  )
  event_log = EventLog()
  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    event_log=event_log,
    lifecycle=lifecycle,
    policy=TopLevelSkillResultPolicy(
      prepare_provider=lambda proposed: proposed,
      prepare_completion=_prepare_state_plan,
    ),
    workspace_dir=tmp_path,
  )

  _run(
    runner.run(
      messages=[{"role": "user", "content": "Run the skill"}]
    )
  )

  captured, _ = _run(
    log.query(
      event_types={"skill_result_captured"},
      order="asc",
    )
  )
  assert len(captured) == 1
  assert captured[0].event["exit_code"] == 0
  assert captured[0].event["outcome"] == "success"
  assert captured[0].event["error"] is None
  assert state_path.read_bytes() == b'{"state":"success"}\n'
  live_types = [entry.event["type"] for entry in event_log.entries]
  assert live_types.count("skill_result_captured") == 1
  assert "stream_complete" in live_types


def test_completion_effect_failure_leaves_recoverable_intent(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  lifecycle = _metadata()

  policy = TopLevelSkillResultPolicy(
    prepare_provider=lambda proposed: proposed,
    prepare_completion=lambda _event_log, terminal_event: (
      _completion_plan(
        _result_event(lifecycle, terminal_event),
        terminal_event,
      )
    ),
  )
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "state-failure.jsonl"
  )
  event_log = EventLog()
  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("done")]),
    event_log=event_log,
    lifecycle=lifecycle,
    policy=policy,
  )

  monkeypatch.setattr(
    "agent_gateway.runner_session_lifecycle.apply_completion_effect",
    lambda *_args, **_kwargs: (_ for _ in ()).throw(
      OSError("atomic state write failed")
    ),
  )
  with pytest.raises(OSError, match="atomic state write failed"):
    _run(
      runner.run(
        messages=[{"role": "user", "content": "Run the skill"}]
      )
    )

  assert runner.committed_top_level_skill_result_event is None
  wal_path = (
    log.path.parent
    / f".{log.path.name}.skill_completion"
    / "completion.json"
  )
  assert wal_path.exists()


def test_conflicting_partial_result_blocks_before_recovery_effect(
  tmp_path: Path,
) -> None:
  stale_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="stale-conflicting-partial-result",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  fresh_lifecycle = TopLevelSkillLifecycleMetadata(
    skill_run_id="fresh-after-conflicting-partial-result",
    skill="fundamental-research",
    scope="ticker",
    ticker="PCTY",
    portfolio_id=None,
  )
  log = AgentSessionLog(
    path=tmp_path / "sessions" / "conflicting-partial-result.jsonl"
  )
  runner = _runner(
    log=log,
    provider=_ScriptedProvider([_text_turn("must not run")]),
    lifecycle=fresh_lifecycle,
    policy=_policy(fresh_lifecycle),
    workspace_dir=tmp_path,
  )
  terminal = {
    "type": "stream_complete",
    "terminal_disposition": "completed",
    **stale_lifecycle.identity_fields(),
  }
  expected_result = _result_event(stale_lifecycle, terminal)
  conflicting_result = {
    **expected_result,
    "warnings": ["conflicting durable evidence"],
  }
  durable_result = _durable_writer_event(expected_result)
  durable_terminal = _durable_writer_event(terminal)
  _run(
    log.append(
      _top_level_started_event(
        stale_lifecycle,
        started_at=1.0,
      )
    )
  )
  _run(log.append(_durable_writer_event(conflicting_result)))
  state_path = tmp_path / "state.json"
  state_path.write_bytes(b'{"state":"before"}\n')
  effect = TopLevelSkillCompletionEffectPlan.canonical_json_update(
    workspace_path=tmp_path,
    target_path=state_path,
    update=lambda _exists, _before: {"state": "after"},
  )
  SkillCompletionWal(log.path).store({
    "record_type": "intent",
    "skill_run_id": stale_lifecycle.skill_run_id,
    "lifecycle": stale_lifecycle.identity_fields(),
    "result": durable_result,
    "terminal": durable_terminal,
    "effect": effect.durable_payload(),
    "fence": runner._top_level_skill_admission.fence,
  })

  with pytest.raises(
    RuntimeError,
    match="Conflicting durable top-level lifecycle envelope",
  ):
    _run(
      runner.run(
        messages=[{"role": "user", "content": "Run fresh skill"}]
      )
    )

  assert state_path.read_bytes() == b'{"state":"before"}\n'


class _BlockingLifecycleLog(AgentSessionLog):
  def __init__(
    self,
    *args: Any,
    blocked_type: str,
    **kwargs: Any,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.blocked_type = blocked_type
    self.append_entered = asyncio.Event()
    self.append_release = asyncio.Event()

  async def append(self, event: dict[str, Any]) -> Any:
    if event.get("type") == self.blocked_type:
      self.append_entered.set()
      await self.append_release.wait()
    return await super().append(event)


def test_cancellation_during_start_append_drains_owned_write(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = _BlockingLifecycleLog(
      path=tmp_path / "sessions" / "cancel-start.jsonl",
      blocked_type="skill_run_started",
    )
    runner = _runner(
      log=log,
      provider=_ScriptedProvider([_text_turn("must not run")]),
    )
    task = asyncio.create_task(
      runner.run(
        messages=[{"role": "user", "content": "Run the skill"}]
      )
    )
    await asyncio.wait_for(log.append_entered.wait(), timeout=1.0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    log.append_release.set()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task, timeout=1.0)

    entries, _ = await log.query(order="asc")
    types = [entry.event["type"] for entry in entries]
    assert types.count("skill_run_started") == 1
    assert types.count("skill_result_captured") == 1

  _run(_case())


class _BlockingProvider(_ScriptedProvider):
  def __init__(self) -> None:
    super().__init__([])
    self.stream_entered = asyncio.Event()

  async def stream(
    self,
    client: Any,
    params: dict[str, Any],
  ) -> Any:
    _ = client, params
    self.stream_entered.set()
    await asyncio.Event().wait()
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def test_real_timeout_cause_agrees_across_settlement_artifacts(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(
      path=tmp_path / "sessions" / "real-timeout.jsonl"
    )
    event_log = EventLog()
    provider = _BlockingProvider()
    runner = _runner(
      log=log,
      provider=provider,
      event_log=event_log,
    )

    # A real (non-mocked) session timeout must interrupt a run that is stably
    # blocked inside the provider stream. A 10ms timeout instead races session
    # STARTUP under CPU starvation: the cancel lands at an arbitrary early
    # await (attach-event append, settlement enrollment, the completion
    # handshake), cancelling the handshake task itself so
    # drain_owned_lifecycle_task's shield re-raises CancelledError. The 2s
    # timeout still fires promptly (the provider blocks forever) but leaves
    # startup ample time to finish first; the stream_entered gate asserts that
    # precondition event-wise instead of relying on the clock.
    session_task = asyncio.create_task(
      run_session(
        runner,
        event_log,
        max_turns=4,
        timeout_seconds=2.0,
        initial_message="Run the skill",
        system_prompt=None,
      )
    )
    await asyncio.wait_for(
      provider.stream_entered.wait(),
      timeout=30.0,
    )
    output = await asyncio.wait_for(session_task, timeout=120.0)

    assert output.timed_out
    assert runner._top_level_skill_settlement_complete.is_set()
    assert runner._write_lease_file is None
    events, _ = await log.query(order="asc")
    by_type = {
      event_type: [
        entry.event
        for entry in events
        if entry.event["type"] == event_type
      ]
      for event_type in {
        "skill_result_captured",
        "stream_complete",
        "run_error",
        "interrupted",
        "detach",
      }
    }
    assert by_type["skill_result_captured"][0]["exit_code"] == 124
    assert by_type["skill_result_captured"][0]["outcome"] == "timeout"
    for event_type in (
      "stream_complete",
      "run_error",
      "interrupted",
      "detach",
    ):
      assert by_type[event_type][0]["reason"] == "timeout"
    for event_type in (
      "stream_complete",
      "run_error",
      "interrupted",
    ):
      assert (
        by_type[event_type][0]["server_terminal_cause"]
        == "timeout"
      )

  _run(_case())


def test_caller_cancel_during_timeout_plan_drain_keeps_timeout(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(
      path=tmp_path / "sessions" / "timeout-drain.jsonl"
    )
    event_log = EventLog()
    lifecycle = _metadata()
    plan_entered = asyncio.Event()
    plan_release = asyncio.Event()

    async def _prepare_completion(
      _event_log: Any,
      terminal_event: dict[str, Any],
    ) -> TopLevelSkillCompletionPlan:
      plan_entered.set()
      await plan_release.wait()
      return _completion_plan(
        _result_event(lifecycle, terminal_event),
        terminal_event,
      )

    provider = _BlockingProvider()
    runner = _runner(
      log=log,
      provider=provider,
      event_log=event_log,
      lifecycle=lifecycle,
      policy=TopLevelSkillResultPolicy(
        prepare_provider=lambda proposed: proposed,
        prepare_completion=_prepare_completion,
      ),
    )
    # Same de-flake as test_real_timeout_cause_agrees_across_settlement_artifacts:
    # a 10ms session timeout races session STARTUP under CPU starvation, so the
    # cancel can land before settlement enrollment and prepare_completion is
    # never reached (plan_entered never fires). The 2s timeout still fires
    # promptly against the forever-blocking provider, and the stream_entered
    # gate asserts the run reached the steady blocked-stream state first. The
    # generous wait_for bounds only cap how long the test waits — the plan gate
    # itself stays event-driven.
    task = asyncio.create_task(
      run_session(
        runner,
        event_log,
        max_turns=4,
        timeout_seconds=2.0,
        initial_message="Run the skill",
        system_prompt=None,
      )
    )
    await asyncio.wait_for(
      provider.stream_entered.wait(),
      timeout=30.0,
    )
    await asyncio.wait_for(
      plan_entered.wait(),
      timeout=30.0,
    )
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    plan_release.set()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task, timeout=30.0)

    assert runner._top_level_skill_settlement_complete.is_set()
    assert runner._write_lease_file is None
    events, _ = await log.query(order="asc")
    result = next(
      entry.event
      for entry in events
      if entry.event["type"] == "skill_result_captured"
    )
    terminal = next(
      entry.event
      for entry in events
      if entry.event["type"] == "stream_complete"
    )
    assert result["exit_code"] == 124
    assert result["outcome"] == "timeout"
    assert terminal["reason"] == "timeout"
    assert terminal["server_terminal_cause"] == "timeout"

  _run(_case())


def test_repeated_cancellation_during_plan_drain_commits_once(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(
      path=tmp_path / "sessions" / "cancel-result.jsonl"
    )
    provider = _BlockingProvider()
    lifecycle = _metadata()
    plan_entered = asyncio.Event()
    plan_release = asyncio.Event()

    async def _prepare_completion(
      _event_log: Any,
      terminal_event: dict[str, Any],
    ) -> TopLevelSkillCompletionPlan:
      plan_entered.set()
      await plan_release.wait()
      return _completion_plan(
        _result_event(lifecycle, terminal_event),
        terminal_event,
      )

    runner = _runner(
      log=log,
      provider=provider,
      lifecycle=lifecycle,
      policy=TopLevelSkillResultPolicy(
        prepare_provider=lambda proposed: proposed,
        prepare_completion=_prepare_completion,
      ),
    )
    task = asyncio.create_task(
      runner.run(
        messages=[{"role": "user", "content": "Run the skill"}]
      )
    )
    await asyncio.wait_for(provider.stream_entered.wait(), timeout=1.0)
    task.cancel()
    await asyncio.wait_for(plan_entered.wait(), timeout=1.0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    plan_release.set()
    with pytest.raises(asyncio.CancelledError):
      await asyncio.wait_for(task, timeout=1.0)

    entries, _ = await log.query(order="asc")
    types = [entry.event["type"] for entry in entries]
    assert types.count("skill_run_started") == 1
    assert types.count("skill_result_captured") == 1

  _run(_case())


@pytest.mark.parametrize(
  ("field_name", "invalid_value", "message"),
  [
    ("exit_code", True, "non-negative integer"),
    ("exit_code", -1, "non-negative integer"),
    ("gate_code", 1, "must be a string"),
    ("artifact_refs", "not-a-list", "must be a list"),
    ("artifact_refs", [1], "must be a string"),
    ("fms_results", ["not-a-mapping"], "must be a mapping"),
    ("cost_usd", -0.1, "finite and non-negative"),
    ("duration_s", float("inf"), "finite and non-negative"),
    ("compaction_count", True, "non-negative integer"),
    ("compaction_count", -1, "non-negative integer"),
  ],
)
def test_result_policy_payload_types_are_strict(
  field_name: str,
  invalid_value: Any,
  message: str,
) -> None:
  lifecycle = _metadata()
  event = _result_event(
    lifecycle,
    {"type": "stream_complete"},
  )
  event[field_name] = invalid_value

  with pytest.raises(RuntimeError, match=message):
    lifecycle.normalize_result_event(event)


def test_result_policy_allows_recoverable_fms_retry_before_success() -> None:
  lifecycle = _metadata()
  event = _result_event(
    lifecycle,
    {"type": "stream_complete"},
  )
  event["status"] = "applied"
  event["fms_results"] = [
    {
      "tool_name": "fms_report_idea_to_thesis",
      "status": "error",
      "error": {
        "recoverable": True,
        "message": "repair the judgment shape",
      },
    },
    {
      "tool_name": "fms_report_idea_to_thesis",
      "status": "applied",
      "artifact_ref": "artifacts/PCTY/idea-to-thesis/result.json",
    },
  ]

  normalized = lifecycle.normalize_result_event(event)

  assert normalized["outcome"] == "success"
  assert normalized["exit_code"] == 0
  assert normalized["fms_results"] == event["fms_results"]


def test_result_policy_rejects_unrecoverable_fms_failure_on_success() -> None:
  lifecycle = _metadata()
  event = _result_event(
    lifecycle,
    {"type": "stream_complete"},
  )
  event["fms_results"] = [{
    "tool_name": "fms_report_idea_to_thesis",
    "status": "error",
    "error": {
      "recoverable": False,
      "message": "write refused",
    },
  }]

  with pytest.raises(RuntimeError, match="unrecoverable FMS failure"):
    lifecycle.normalize_result_event(event)


def test_result_policy_rejects_unexpected_fields() -> None:
  lifecycle = _metadata()
  event = _result_event(
    lifecycle,
    {"type": "stream_complete"},
  )
  event["legacy_receipt"] = True

  with pytest.raises(RuntimeError, match="unexpected fields"):
    lifecycle.normalize_result_event(event)


def test_projection_flag_changes_only_after_successful_append() -> None:
  runner = object.__new__(AgentRunner)
  runner._top_level_skill_started_projected = False
  runner._top_level_skill_result_projected = False
  event = build_skill_run_started_event(
    _metadata(),
    started_at=1.0,
  )

  def _fail_append(_event: dict[str, Any]) -> None:
    raise RuntimeError("live projection failed")

  runner._append = _fail_append  # type: ignore[method-assign]
  with pytest.raises(RuntimeError, match="live projection failed"):
    runner._project_top_level_skill_event(event)
  assert not runner._top_level_skill_started_projected

  projected: list[dict[str, Any]] = []
  runner._append = (  # type: ignore[method-assign]
    lambda projected_event: (
      projected.append(projected_event),
      SimpleNamespace(event=projected_event),
    )[1]
  )
  runner._project_top_level_skill_event(event)
  assert runner._top_level_skill_started_projected
  assert projected == [event]

  runner._top_level_skill_started_projected = False
  runner._append = lambda _event: None  # type: ignore[method-assign]
  with pytest.raises(RuntimeError, match="Live EventLog rejected"):
    runner._project_top_level_skill_event(event)
  assert not runner._top_level_skill_started_projected


class _ForgedStartWriterWrapperLog(AgentSessionLog):
  async def append(self, event: dict[str, Any]) -> Any:
    if event.get("type") == "skill_run_started":
      event = {
        **event,
        "role": "parent",
      }
    return await super().append(event)


def test_top_level_start_requires_exact_writer_acknowledgement(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = _ForgedStartWriterWrapperLog(
      path=tmp_path / "sessions" / "forged-start-wrapper.jsonl"
    )
    runner = _runner(
      log=log,
      provider=_ScriptedProvider([_text_turn("unused")]),
    )
    runner._runner_id = "runner-forged-start"
    runner._role = "writer"
    runner._sub_agent_id = None
    event = {
      **build_skill_run_started_event(
        _metadata(),
        started_at=1.0,
      ),
      "lifecycle_origin": "top_level",
    }
    try:
      with pytest.raises(RuntimeError, match="envelope mismatch"):
        await runner._persist_top_level_skill_started(event)
    finally:
      runner._top_level_skill_admission.release()

  _run(_case())


def test_child_skill_durable_confirmer_requires_exact_whole_envelope(
  tmp_path: Path,
) -> None:
  async def _case() -> None:
    log = AgentSessionLog(
      path=tmp_path / "sessions" / "child-confirm.jsonl"
    )
    runner = object.__new__(AgentRunner)
    runner._agent_session_log = log
    runner._runner_id = "runner-child-confirm"
    runner._role = "parent"
    runner._sub_agent_id = None
    runner._last_durable_seq = 0
    event = build_skill_run_started_event(
      _metadata(),
      started_at=1.0,
    )

    await runner._append_durable_event(event)
    confirmed = await runner._confirm_durable_skill_event(event)

    assert confirmed is not None
    assert confirmed["runner_id"] == "runner-child-confirm"
    assert confirmed["role"] == "parent"
    assert confirmed["event_schema_version"] == EVENT_SCHEMA_VERSION
    mutated = {
      **event,
      "ts": 2.0,
    }
    with pytest.raises(RuntimeError, match="envelope mismatch"):
      await runner._confirm_durable_skill_event(mutated)

  _run(_case())


def test_admission_fence_generation_is_durable_and_monotonic(
  tmp_path: Path,
) -> None:
  log_path = tmp_path / "sessions" / "generation.jsonl"
  log_path.parent.mkdir(parents=True)
  log_path.write_text("", encoding="utf-8")

  first = TopLevelSkillAdmission.acquire(log_path)
  assert first.lease_generation == 1
  assert len(bytes.fromhex(first.lease_owner_token)) == 32
  assert stat.S_IMODE(first.write_lease_path.stat().st_mode) == 0o600
  first.release()
  assert first.release() is False
  assert first.release_count == 1

  second = TopLevelSkillAdmission.acquire(log_path)
  try:
    assert second.lease_generation == 2
    assert second.lease_owner_token != first.lease_owner_token
    second.validate_fence()
  finally:
    second.release()


@pytest.mark.parametrize("unsafe_kind", ["hardlink", "writable"])
def test_admission_rejects_unsafe_existing_session_log_before_lease(
  tmp_path: Path,
  unsafe_kind: str,
) -> None:
  log_path = tmp_path / "sessions" / "unsafe.jsonl"
  log_path.parent.mkdir(parents=True)
  if unsafe_kind == "hardlink":
    source = tmp_path / "source.jsonl"
    source.write_text("", encoding="utf-8")
    os.link(source, log_path)
  else:
    log_path.write_text("", encoding="utf-8")
    log_path.chmod(0o622)

  with pytest.raises(RuntimeError):
    TopLevelSkillAdmission.acquire(log_path)

  assert not log_path.with_name(
    f"{log_path.name}.write_lease"
  ).exists()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "writable"])
def test_admission_rejects_unsafe_existing_lease_without_repair(
  tmp_path: Path,
  unsafe_kind: str,
) -> None:
  log_path = tmp_path / "sessions" / "unsafe-lease.jsonl"
  log_path.parent.mkdir(parents=True)
  lease_path = log_path.with_name(f"{log_path.name}.write_lease")
  source = tmp_path / "lease-source"
  source.write_text("", encoding="utf-8")
  source.chmod(0o600)
  if unsafe_kind == "symlink":
    lease_path.symlink_to(source)
  elif unsafe_kind == "hardlink":
    os.link(source, lease_path)
  else:
    lease_path.write_text("", encoding="utf-8")
    lease_path.chmod(0o622)

  with pytest.raises((OSError, RuntimeError)):
    TopLevelSkillAdmission.acquire(log_path)

  if unsafe_kind == "writable":
    assert stat.S_IMODE(lease_path.stat().st_mode) == 0o622


def test_completion_wal_is_private_checksummed_and_exact(
  tmp_path: Path,
) -> None:
  log_path = tmp_path / "sessions" / "wal.jsonl"
  log_path.parent.mkdir(parents=True)
  wal = SkillCompletionWal(log_path)

  stored = wal.store(_wal_intent_record())

  assert wal.load() == stored
  assert stored["checksum"].startswith("sha256:")
  assert stat.S_IMODE(wal.directory_path.stat().st_mode) == 0o700
  assert stat.S_IMODE(
    (wal.directory_path / "completion.json").stat().st_mode
  ) == 0o600


def test_completion_wal_rejects_dangling_directory_symlink(
  tmp_path: Path,
) -> None:
  log_path = tmp_path / "sessions" / "dangling.jsonl"
  log_path.parent.mkdir(parents=True)
  wal = SkillCompletionWal(log_path)
  wal.directory_path.symlink_to(tmp_path / "missing-target")

  with pytest.raises(OSError):
    wal.load()


def test_completion_wal_does_not_overwrite_corrupt_existing_record(
  tmp_path: Path,
) -> None:
  log_path = tmp_path / "sessions" / "corrupt.jsonl"
  log_path.parent.mkdir(parents=True)
  wal = SkillCompletionWal(log_path)
  wal.directory_path.mkdir(mode=0o700)
  record_path = wal.directory_path / "completion.json"
  record_path.write_text("{broken", encoding="utf-8")
  record_path.chmod(0o600)

  with pytest.raises(SkillCompletionWalCorruptError):
    wal.store(_wal_intent_record())

  assert record_path.read_text(encoding="utf-8") == "{broken"


def test_completion_wal_fsync_failure_leaves_blocking_temp_remnant(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_path = tmp_path / "sessions" / "fsync.jsonl"
  log_path.parent.mkdir(parents=True)
  wal = SkillCompletionWal(log_path)
  wal.store(_wal_intent_record())
  real_fsync = completion_wal_module.os.fsync
  calls = 0

  def _fail_first_fsync(fd: int) -> None:
    nonlocal calls
    calls += 1
    if calls == 1:
      raise OSError("injected temp fsync failure")
    real_fsync(fd)

  monkeypatch.setattr(completion_wal_module.os, "fsync", _fail_first_fsync)
  with pytest.raises(OSError, match="injected"):
    wal.store(_wal_settled_record())

  with pytest.raises(
    SkillCompletionWalCorruptError,
    match="crash-remnant",
  ):
    wal.load()


def test_completion_wal_post_rename_failure_recovers_new_exact_record(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  log_path = tmp_path / "sessions" / "rename.jsonl"
  log_path.parent.mkdir(parents=True)
  wal = SkillCompletionWal(log_path)
  wal.store(_wal_intent_record())
  real_rename = completion_wal_module.os.rename

  def _rename_then_fail(*args: Any, **kwargs: Any) -> None:
    real_rename(*args, **kwargs)
    raise OSError("injected post-rename failure")

  monkeypatch.setattr(
    completion_wal_module.os,
    "rename",
    _rename_then_fail,
  )
  with pytest.raises(OSError, match="post-rename"):
    wal.store(_wal_settled_record())

  loaded = wal.load()
  assert loaded is not None
  assert loaded["record_type"] == "settled"


def test_completion_effect_is_cas_idempotent_and_supports_json_null(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "workspace"
  target = workspace / "state" / "result.json"
  target.parent.mkdir(parents=True)
  plan = TopLevelSkillCompletionEffectPlan.canonical_json_update(
    workspace_path=workspace,
    target_path=target,
    update=lambda _exists, _before: None,
  )
  payload = plan.durable_payload()

  assert apply_completion_effect(
    payload,
    expected_workspace=workspace,
  ) == "applied"
  assert json.loads(target.read_text(encoding="utf-8")) is None
  assert apply_completion_effect(
    payload,
    expected_workspace=workspace,
  ) == "already_applied"


def test_completion_effect_rejects_cas_conflict_without_overwrite(
  tmp_path: Path,
) -> None:
  workspace = tmp_path / "workspace"
  target = workspace / "state" / "result.json"
  target.parent.mkdir(parents=True)
  target.write_text('{"version":1}\n', encoding="utf-8")
  plan = TopLevelSkillCompletionEffectPlan.canonical_json_update(
    workspace_path=workspace,
    target_path=target,
    update=lambda _exists, _before: {"version": 2},
  )
  target.write_text('{"version":3}\n', encoding="utf-8")

  with pytest.raises(SkillCompletionEffectConflict):
    apply_completion_effect(
      plan.durable_payload(),
      expected_workspace=workspace,
    )

  assert json.loads(target.read_text(encoding="utf-8")) == {
    "version": 3,
  }


def test_direct_artifact_handlers_do_not_produce_lifecycle_start() -> None:
  for relative_path in (
    "api/agent/shared/canvas_artifact_tool.py",
    "api/agent/shared/dashboard_artifact_tool.py",
  ):
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    assert "SkillRunStartedEvent" not in source
    assert '"type": "skill_run_started"' not in source
