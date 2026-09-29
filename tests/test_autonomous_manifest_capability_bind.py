"""The durable manifest bind write is unconditional (review §3, durable items).

The capability_bind field previously wrote only when a test-patchable runtime
version compare matched, so a patched ``_TASK_MANIFEST_VERSION`` could persist
a bind-less manifest that failed only at rehydrate. The write must not depend
on any version compare.
"""

from __future__ import annotations

from pathlib import Path
from agent_gateway.autonomous_launch_envelope import AutonomousControlAuthority
from model_authority.bind import CapabilityBind
from agent_gateway.autonomous_runner import AutonomousRegistry
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY

import pytest

from agent_gateway import autonomous_runner_state


def _capability_bind() -> CapabilityBind:
  entry = INITIAL_MODEL_REGISTRY.require("anthropic.claude-opus-5")
  return CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key=entry.key,
    provider=entry.provider,
    upstream_model=entry.upstream_model,
    adapter=entry.adapter,
    protocol_profile=entry.protocol_profile,
    route=entry.route,
    effort="high",
    credential_principal="service",
    credential_ref="service:manifest-capability-bind:anthropic",
    run_mode="autonomous",
    registry_revision=INITIAL_MODEL_REGISTRY.revision,
    policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
    selection_source="capability_default",
  )

def _registry(tmp_path: Path) -> AutonomousRegistry:
  return AutonomousRegistry(
    api_dir=tmp_path / "api",
    log_dir=tmp_path / "autonomous",
  )


def _record(*, capability_bind: CapabilityBind) -> autonomous_runner_state.AutonomousTask:
  return autonomous_runner_state.AutonomousTask(
    manifest_version=autonomous_runner_state._TASK_MANIFEST_VERSION,
    task_id="bg_test",
    control_run_id="run-1",
    session_id="bg_test",
    channel_id="c" * 64,
    owner_user_id="henry",
    user_id="henry",
    raw_user_id="henry",
    user_slug="henry",
    risk_user_id=0,
    user_email=None,
    user_aliases=["henry"],
    identity_status="verified",
    role="owner",
    profile="analyst",
    mode="task",
    task="do the thing",
    skill=None,
    pack=None,
    deliver=True,
    skill_resume_allowed=False,
    admitted_skill_execution_limits=None,
    context=None,
    ticker=None,
    channel="tui",
    dev_mode=False,
    max_budget_usd=1.0,
    research_file_id=None,
    dispatch_scope=None,
    cmd=["python", "-m", "job"],
    log_path=Path("/tmp/bg_test.log"),
    events_path=None,
    operator_inbox_path=Path("/tmp/bg_test.inbox"),
    approval_decisions_path=None,
    owner_lease_path=Path("/tmp/bg_test.lease"),
    owner_lease_device=1,
    owner_lease_inode=2,
    control_authority=AutonomousControlAuthority(
      control_mode="file",
      admission_ledger_path="/tmp/bg_test.ledger",
      admission_ledger_device=1,
      admission_ledger_inode=2,
      operator_inbox_path="/tmp/bg_test.inbox",
      operator_inbox_device=1,
      operator_inbox_inode=3,
    ),
    started_at=100.0,
    state="running",
    exit_code=None,
    error=None,
    terminal_reason=None,
    completed_at=None,
    resumed_from=None,
    resumed_as=[],
    schedule_id=None,
    schedule_name=None,
    tool_result_spill_dir=None,
    capability_bind=capability_bind,
  )


@pytest.mark.parametrize("patched_version_delta", [0, 1])
def test_manifest_payload_always_writes_capability_bind(
  monkeypatch: pytest.MonkeyPatch,
  patched_version_delta: int,
  tmp_path: Path,
) -> None:
  capability_bind = _capability_bind()
  record = _record(capability_bind=capability_bind)
  bind_receipt = capability_bind.to_json()
  if patched_version_delta:
    # Simulate the runtime-attr divergence that previously suppressed the
    # write: the payload must still carry the bind.
    monkeypatch.setattr(
      autonomous_runner_state,
      "_TASK_MANIFEST_VERSION",
      autonomous_runner_state._TASK_MANIFEST_VERSION + patched_version_delta,
    )

  registry = _registry(tmp_path)
  payload = registry._manifest_payload(record)

  assert "capability_bind" in payload
  assert payload["capability_bind"] == bind_receipt


def test_manifest_payload_writes_explicit_null_for_absent_bind(
  tmp_path: Path,
) -> None:
  record = _record(capability_bind=_capability_bind())
  record.capability_bind = None

  registry = _registry(tmp_path)
  payload = registry._manifest_payload(record)

  assert "capability_bind" in payload
  assert payload["capability_bind"] is None
