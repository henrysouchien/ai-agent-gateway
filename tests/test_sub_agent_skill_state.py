# ruff: noqa: E402

from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.sub_agent as sub_agent_module
import agent_gateway.sub_agent_skill_state as skill_state
from agent_workflow_contracts import (
  ActivityHandle,
  AgentOperationRef,
  AnalyticalOutcome,
  AttemptRef,
  CanonicalProjection,
  ContentHandle,
  ContractRef,
  ExecutionSettlement,
  OrdinaryDelegationTaskRef,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  TranscriptHandle,
  UsageObservation,
  canonical_json_bytes,
  sha256_digest,
)
from agent_gateway.autonomous_output import extract_state_update


class _Store:
  def __init__(self, initial: dict[str, Any]) -> None:
    self.initial = initial
    self.saved: dict[str, dict[str, Any]] = {}

  def get(self, name: str) -> dict[str, Any]:
    _ = name
    return dict(self.initial)

  def set(self, name: str, value: dict[str, Any]) -> None:
    self.saved[name] = dict(value)

  def update(self, name: str, mutation: Any) -> dict[str, Any]:
    value = mutation(dict(self.initial))
    self.saved[name] = dict(value)
    return dict(value)


def _contract(name: str) -> ContractRef:
  return ContractRef(
    namespace="skill-state-test",
    name=name,
    version="1.0",
    digest=sha256_digest({"contract": name}),
  )


def _task_result(
  summary: str,
  *,
  status: str = "succeeded",
  terminal_reason: str | None = None,
) -> dict[str, Any]:
  operation = AgentOperationRef(
    namespace="agent-operation",
    name="skill-a",
    version="1.0",
    digest=sha256_digest({"operation": "skill-a"}),
  )
  task_id = "bg_skill_state"
  values = TaskResultValues()
  outcome = None
  if status == "succeeded":
    contract = _contract("summary")
    inline = {"summary": summary}
    raw = canonical_json_bytes(inline)
    digest = hashlib.sha256(raw).hexdigest()
    values = TaskResultValues(
      projection=CanonicalProjection(
        contract=contract,
        content=ContentHandle(
          content_id=f"sha256:{digest}",
          content_sha256=digest,
          content_bytes=len(raw),
          content_chars=len(raw.decode("utf-8")),
          contract=contract,
          media_type="application/json",
          encoding="utf-8",
          retention="durable",
        ),
        inline_view=inline,
      )
    )
    outcome = AnalyticalOutcome(
      disposition="complete",
      assessment_source="domain_tool",
    )
  result = TaskResult(
    task_result_id="result-skill-state",
    logical_task=OrdinaryDelegationTaskRef(
      delegation_id=task_id,
      operation=operation,
    ),
    attempt=AttemptRef(
      attempt_number=1,
      attempt_id="attempt-skill-state",
      physical_task_id=task_id,
    ),
    execution=ExecutionSettlement(
      status=status,  # type: ignore[arg-type]
      terminal_reason=terminal_reason,
    ),
    outcome=outcome,
    values=values,
    observation=TaskObservation(
      transcript=TranscriptHandle(
        kind="child_transcript",
        owner_id=task_id,
      ),
      activity=ActivityHandle(
        kind="child_activity",
        owner_id=task_id,
      ),
      usage=UsageObservation(),
    ),
    provenance=TaskResultProvenance(
      admitted_task_digest=sha256_digest({"admitted": task_id}),
      model_bind_digest=sha256_digest({"model": task_id}),
      capability_binding_digest=sha256_digest({"capability": task_id}),
      tool_grant_digest=sha256_digest({"tools": task_id}),
    ),
  )
  return result.model_dump(mode="json")


def test_result_response_text_reads_canonical_report_summary() -> None:
  assert skill_state.result_response_text(
    _task_result("Canonical summary")
  ) == "Canonical summary"
  assert skill_state.result_response_text(
    _task_result("", status="interrupted", terminal_reason="killed")
  ) == ""


def test_task_result_outcome_drives_skill_state_classification() -> None:
  result = _task_result("Visible workflow summary")

  assert skill_state.classify_child_outcome(result, None).succeeded is True
  assert skill_state.classify_child_outcome(result, None).outcome == "complete"
  assert skill_state.result_response_text(result) == (
    "Visible workflow summary"
  )


def test_parent_skill_state_wrappers_delegate_to_sidecar(monkeypatch) -> None:
  calls: list[tuple[str, Any]] = []

  def fake_response_text(result: Any | None) -> str:
    calls.append(("response", result))
    return "patched-response"

  def fake_prompt(skill_name: str, previous_state: dict[str, Any]) -> str:
    calls.append(("prompt", (skill_name, previous_state)))
    return "patched-prompt"

  monkeypatch.setattr(skill_state, "result_response_text", fake_response_text)
  monkeypatch.setattr(skill_state, "skill_state_prompt", fake_prompt)

  task_result = _task_result("raw")
  assert sub_agent_module._result_response_text(task_result) == "patched-response"
  assert sub_agent_module._skill_state_prompt("skill-a", {"runs": 1}) == "patched-prompt"
  assert calls == [
    ("response", task_result),
    ("prompt", ("skill-a", {"runs": 1})),
  ]


def test_classify_child_outcome_uses_execution_settlement_failure() -> None:
  result = _task_result("", status="interrupted", terminal_reason="timeout")
  classification = skill_state.classify_child_outcome(result, None)

  assert classification.outcome == "interrupted"
  assert classification.succeeded is False
  assert classification.error == {
    "code": "timeout",
    "message": "Sub-agent ended with timeout",
    "child_outcome": "interrupted",
  }


def test_classify_child_outcome_rejects_success_claim_from_error() -> None:
  classification = skill_state.classify_child_outcome(
    _task_result("partial report"),
    {
      "code": "provider_failed",
      "message": "provider failed",
      "child_outcome": "success",
    },
  )

  assert classification.outcome == "error"
  assert classification.succeeded is False
  assert classification.error == {
    "code": "provider_failed",
    "message": "provider failed",
    "child_outcome": "error",
  }


def test_persist_skill_state_merges_model_state_only_for_task_success() -> None:
  async def scenario() -> None:
    store = _Store({"run_count": 2, "keep": "yes", "last_error": {"old": True}})
    profile = SimpleNamespace(name="skill-a", persist_state=True, version="1.2.3")
    warnings: list[tuple[str, tuple[Any, ...]]] = []
    logger = SimpleNamespace(warning=lambda message, *args, **_kwargs: warnings.append((message, args)))

    await skill_state.persist_skill_state(
      _task_result(
        "## STATE_UPDATE_JSON\n```json\n"
        '{"fresh":"state"}\n'
        "```"
      ),
      None,
      agent_name="skill-a",
      profile=profile,
      skill_state_store=store,
      skill_state_lock=asyncio.Lock(),
      effective_model="model-a",
      extract_state_update_fn=extract_state_update,
      logger=logger,
    )

    saved = store.saved["skill-a"]
    assert saved["keep"] == "yes"
    assert saved["fresh"] == "state"
    assert saved["model"] == "model-a"
    assert saved["run_count"] == 3
    assert saved["version"] == "1.2.3"
    assert saved["last_outcome"] == "complete"
    assert saved["outcome_counts"] == {"complete": 1}
    assert "last_error" not in saved
    assert "last_run" in saved
    assert warnings == []

  asyncio.run(scenario())


def test_persist_skill_state_preserves_domain_state_on_external_error() -> None:
  async def scenario() -> None:
    store = _Store({
      "run_count": 2,
      "fresh": "previous",
      "last_error": {"old": True},
    })
    profile = SimpleNamespace(name="skill-a", persist_state=True, version=None)

    await skill_state.persist_skill_state(
      _task_result(
        "## STATE_UPDATE_JSON\n```json\n"
        '{"fresh":"partial"}\n'
        "```"
      ),
      {
        "code": "failed",
        "message": "provider failed",
        "child_outcome": "success",
      },
      agent_name="skill-a",
      profile=profile,
      skill_state_store=store,
      skill_state_lock=asyncio.Lock(),
      effective_model="model-a",
      extract_state_update_fn=extract_state_update,
      logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    )

    saved = store.saved["skill-a"]
    assert saved["fresh"] == "previous"
    assert saved["last_outcome"] == "error"
    assert saved["outcome_counts"] == {"error": 1}
    assert saved["last_error"] == {
      "code": "failed",
      "message": "provider failed",
      "child_outcome": "error",
    }
    assert saved["run_count"] == 3

  asyncio.run(scenario())


def test_persist_skill_state_skips_when_profile_does_not_persist() -> None:
  async def scenario() -> None:
    store = _Store({})
    profile = SimpleNamespace(name="skill-a", persist_state=False, version=None)

    await skill_state.persist_skill_state(
      _task_result("ignored"),
      None,
      agent_name="skill-a",
      profile=profile,
      skill_state_store=store,
      skill_state_lock=asyncio.Lock(),
      effective_model="model-a",
      extract_state_update_fn=lambda _text: {"fresh": "state"},
      logger=SimpleNamespace(warning=lambda *_args, **_kwargs: None),
    )

    assert store.saved == {}

  asyncio.run(scenario())
