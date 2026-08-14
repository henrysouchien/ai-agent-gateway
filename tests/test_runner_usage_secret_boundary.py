from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.runner_usage as runner_usage  # noqa: E402
from agent_workflow_contracts import CapabilityBind  # noqa: E402


class _Aggregator:
  async def record(self, _event) -> bool:
    return True


class _FailingProducer:
  async def reconcile(self, _summary) -> None:
    raise RuntimeError("CUSTOM-ACTIVE-CREDENTIAL-usage-observer-8f21d7")


def _usage_event():
  totals = runner_usage.empty_usage_totals()
  totals["input_tokens"] = 1
  totals["capability_bind"] = CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key="test.anthropic.model-1",
    provider="anthropic",
    upstream_model="model-1",
    adapter="test.anthropic",
    protocol_profile="test.reasoning",
    route="test.in_process",
    effort="none",
    credential_principal="user",
    credential_ref="test-user:anthropic",
    run_mode="interactive",
    registry_revision="test-runner-usage-secret-boundary.1",
    policy_revision="test-runner-usage-secret-boundary.1",
    selection_source="capability_default",
  ).receipt()
  totals["provider_reported_model"] = None
  return runner_usage.build_usage_event(
    user_id="alice",
    session_id="session-1",
    request_id="request-1",
    parent_turn_id=None,
    timestamp=1.0,
    model="model-1",
    provider_name="anthropic",
    usage_totals=totals,
    cost_total=0.0,
    rate_table_version="v1",
    billing_mode="byok",
    channel="excel",
  )


def test_usage_observer_and_durability_failure_logs_are_value_free(
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
  tmp_path: Path,
) -> None:
  secret = "CUSTOM-ACTIVE-CREDENTIAL-usage-observer-8f21d7"
  logger = logging.getLogger("test.runner_usage.secret_boundary")

  def fail(_value) -> None:
    raise RuntimeError(secret)

  def fail_dlq(_event, _path) -> None:
    raise RuntimeError(secret)

  monkeypatch.setattr(runner_usage, "write_dlq", fail_dlq)

  async def case() -> None:
    event = _usage_event()
    await runner_usage.call_late_usage_event_hook(
      fail,
      event,
      log_session_id="safe-session",
      logger=logger,
    )
    await runner_usage.call_session_summary_hook(
      fail,
      SimpleNamespace(),
      log_session_id="safe-session",
      logger=logger,
      commercial_usage_producer=_FailingProducer(),
    )
    await runner_usage.call_usage_event_hook(
      _Aggregator(),
      event,
      is_summary_emitted=lambda: False,
      on_usage=fail,
      on_late_usage_event=None,
      emit_metric=lambda _name, _value: None,
      dlq_path=tmp_path / "usage.dlq",
      log_session_id="safe-session",
      logger=logger,
    )

  with caplog.at_level(logging.ERROR, logger=logger.name):
    asyncio.run(case())

  assert secret not in caplog.text
  assert caplog.text.count("exception_type=RuntimeError") == 5
