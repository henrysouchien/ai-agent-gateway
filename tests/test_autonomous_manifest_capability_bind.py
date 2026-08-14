"""The durable manifest bind write is unconditional (review §3, durable items).

The capability_bind field previously wrote only when a test-patchable runtime
version compare matched, so a patched ``_TASK_MANIFEST_VERSION`` could persist
a bind-less manifest that failed only at rehydrate. The write must not depend
on any version compare.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_gateway import autonomous_runner_state


class _Receipt:
  def __init__(self, payload: dict) -> None:
    self._payload = payload

  def receipt(self) -> dict:
    return dict(self._payload)


def _record(*, capability_bind: _Receipt | None) -> SimpleNamespace:
  return SimpleNamespace(
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
    context=None,
    ticker=None,
    channel="tui",
    dev_mode=False,
    max_budget_usd=1.0,
    dispatch_scope=None,
    cmd=["python", "-m", "job"],
    log_path=Path("/tmp/bg_test.log"),
    events_path=None,
    operator_inbox_path=None,
    approval_decisions_path=None,
    owner_lease_path=Path("/tmp/bg_test.lease"),
    owner_lease_device=1,
    owner_lease_inode=2,
    control_authority=_Receipt({"control_mode": "file"}),
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
) -> None:
  bind_receipt = {"schema_version": "1.0", "capability_id": "session.driver"}
  record = _record(capability_bind=_Receipt(bind_receipt))
  if patched_version_delta:
    # Simulate the runtime-attr divergence that previously suppressed the
    # write: the payload must still carry the bind.
    monkeypatch.setattr(
      autonomous_runner_state,
      "_TASK_MANIFEST_VERSION",
      autonomous_runner_state._TASK_MANIFEST_VERSION + patched_version_delta,
    )

  payload = autonomous_runner_state.AutonomousRegistryStateMixin._manifest_payload(
    None,
    record,
  )

  assert "capability_bind" in payload
  assert payload["capability_bind"] == bind_receipt


def test_manifest_payload_writes_explicit_null_for_absent_bind() -> None:
  record = _record(capability_bind=None)

  payload = autonomous_runner_state.AutonomousRegistryStateMixin._manifest_payload(
    None,
    record,
  )

  assert "capability_bind" in payload
  assert payload["capability_bind"] is None
