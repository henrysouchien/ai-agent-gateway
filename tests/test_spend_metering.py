# ruff: noqa: E402

import asyncio
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import AgentRunner, ToolDispatcher
from agent_gateway.event_log import EventLog
from agent_gateway.mcp_client import McpClientManager
from agent_gateway.provider_summarize import provider_summarize
from agent_gateway.providers import CostEstimate, ModelInfo, ModelProvider, StreamEvent
from agent_gateway.sdk_runner import AgentSDKRunner
from agent_gateway.send_prompt import send_prompt
from agent_gateway.spend_metering import (
  ProviderCallMeter,
  SpendBudgetBlocked,
  spend_credential,
)
import agent_gateway.runner as gateway_runner
from gateway_test_support.capability_execution_test_support import (
  stub_bound_capability_execution,
)
from model_authority.spend_ledger import fingerprint_credential, ledger_path, spend_context

API_CONFIG = {"provider": "anthropic", "auth_mode": "api", "api_key": "sk-ant-api-key-1"}
OAUTH_CONFIG = {
  "provider": "anthropic",
  "auth_mode": "oauth",
  "auth_mechanism": "oauth_bearer",
  "auth_token": "sk-ant-oat-token-1",
  "api_key": "",
}


def _rows() -> list[dict[str, Any]]:
  path = ledger_path()
  if not path.exists():
    return []
  with closing(sqlite3.connect(path)) as conn:
    conn.row_factory = sqlite3.Row
    return [
      dict(row)
      for row in conn.execute("SELECT * FROM api_spend_events ORDER BY ts, rowid")
    ]


def _freeze_payg() -> None:
  Path(os.environ["HANK_SPEND_SETTINGS_PATH"]).write_text(json.dumps({"freeze_payg": True}))


class _Provider(ModelProvider):
  """Scripted provider: each call to ``stream`` plays the next script step."""

  name = "anthropic"

  def __init__(self, *steps: Any) -> None:
    self.steps = list(steps)
    self.stream_calls = 0

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return True

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> object:
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    return None

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(id=model, provider=self.name, supports_thinking=True)

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return {}

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    return CostEstimate(total=0.0042, cost_known=True)

  def is_retryable_error(self, exc: Exception) -> bool:
    return str(exc) == "transient"

  async def stream(self, client: Any, params: dict[str, Any]) -> AsyncIterator[StreamEvent]:
    step = self.steps[min(self.stream_calls, len(self.steps) - 1)]
    self.stream_calls += 1
    for item in step:
      if isinstance(item, BaseException):
        raise item
      if isinstance(item, StreamEvent):
        yield item
      else:  # "hang": the peer goes silent mid-stream
        await asyncio.Event().wait()


def _ok_step(text: str = "ok") -> list[Any]:
  return [
    StreamEvent(type="message_start", input_tokens=10, cache_read_tokens=2, cache_creation_tokens=3),
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(type="usage_update", output_tokens=4, reasoning_tokens=1),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]


def _usage_then(failure: BaseException) -> list[Any]:
  return [
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="usage_update", output_tokens=4),
    failure,
  ]


def _meter(provider: ModelProvider, config: dict[str, Any] = API_CONFIG, **changes: Any) -> ProviderCallMeter:
  kwargs: dict[str, Any] = dict(
    provider=provider,
    provider_name="anthropic",
    model="claude-sonnet-4-6",
    model_key="claude-sonnet-4-6",
    config=config,
    call_site="chat_turn",
    session_id="sess-1",
    request_id="req-1",
  )
  kwargs.update(changes)
  return ProviderCallMeter(**kwargs)


async def _drain(meter: ProviderCallMeter) -> None:
  async for _ in meter.stream(object(), {}):
    pass


# ── credential classification ────────────────────────────────────────────────────


def test_credential_kind_follows_the_wire_mechanism_not_billing_mode() -> None:
  api = spend_credential("anthropic", API_CONFIG)
  oauth = spend_credential("anthropic", OAUTH_CONFIG)
  local = spend_credential("ollama", {"provider": "ollama", "auth_mode": "api"})
  plan = spend_credential("ollama_cloud", {"provider": "ollama_cloud", "auth_mode": "api", "api_key": "k"})

  assert (api.auth_kind, api.fingerprint) == ("payg_api_key", fingerprint_credential("sk-ant-api-key-1"))
  assert (oauth.auth_kind, oauth.fingerprint) == (
    "oauth_subscription", fingerprint_credential("sk-ant-oat-token-1"),
  )
  assert (local.auth_kind, local.fingerprint) == ("local_or_free", None)
  assert plan.auth_kind == "plan_api_key"


# ── the metered stream: one row per call ─────────────────────────────────────────


def test_a_completed_stream_writes_one_row_with_normalized_tokens_and_the_adapter_cost() -> None:
  provider = _Provider(_ok_step())
  asyncio.run(_drain(_meter(provider)))

  [row] = _rows()
  assert (row["api_category"], row["provider"], row["model_or_endpoint"]) == (
    "llm", "anthropic", "claude-sonnet-4-6",
  )
  assert row["model_key"] == "claude-sonnet-4-6"
  assert row["auth_kind"] == "payg_api_key"
  assert row["credential_fingerprint"] == fingerprint_credential("sk-ant-api-key-1")
  assert (row["session_id"], row["request_id"], row["call_site"]) == ("sess-1", "req-1", "chat_turn")
  assert (
    row["input_tokens"], row["cache_read_tokens"], row["cache_write_tokens"],
    row["output_tokens"], row["reasoning_tokens"],
  ) == (10, 2, 3, 4, 1)
  assert (row["status"], row["error_type"], row["api_calls"]) == ("ok", None, 1)
  assert row["shadow_cost_usd"] == 0.0042 and row["payg_cost_usd"] == 0.0042
  assert row["duration_ms"] is not None


def test_an_unpriced_model_is_an_explicit_null_cost_not_zero() -> None:
  class Unpriced(_Provider):
    def estimate_cost(self, *args: Any, **kwargs: Any) -> CostEstimate:
      return CostEstimate(total=0.0, cost_known=False)

  asyncio.run(_drain(_meter(Unpriced(_ok_step()))))

  [row] = _rows()
  assert row["shadow_cost_usd"] is None and row["payg_cost_usd"] is None


def test_a_mid_stream_error_after_usage_writes_one_error_row_with_the_partial_usage() -> None:
  provider = _Provider(_usage_then(RuntimeError("boom")))
  with pytest.raises(RuntimeError, match="boom"):
    asyncio.run(_drain(_meter(provider)))

  [row] = _rows()
  assert (row["status"], row["error_type"]) == ("error", "RuntimeError")
  assert (row["input_tokens"], row["output_tokens"]) == (10, 4)
  assert row["shadow_cost_usd"] == 0.0042


def test_a_429_is_a_rate_limit_row_even_with_no_usage() -> None:
  class Limited(RuntimeError):
    status_code = 429

  with pytest.raises(Limited):
    asyncio.run(_drain(_meter(_Provider([Limited("slow down")]))))

  [row] = _rows()
  assert (row["status"], row["http_status"]) == ("rate_limit", 429)
  assert row["input_tokens"] == 0 and row["shadow_cost_usd"] == 0.0


def test_a_failure_the_adapter_stream_raises_is_a_row_even_before_any_event() -> None:
  with pytest.raises(RuntimeError, match="provider said no"):
    asyncio.run(_drain(_meter(_Provider([RuntimeError("provider said no")]))))

  [row] = _rows()
  assert (row["status"], row["error_type"], row["input_tokens"]) == ("error", "RuntimeError", 0)
  assert row["shadow_cost_usd"] == 0.0


def test_a_failure_raised_before_the_stream_starts_writes_nothing() -> None:
  class BadParams(_Provider):
    def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
      raise ValueError("cannot build the request")

  provider = BadParams(_ok_step())
  with pytest.raises(ValueError, match="cannot build"):
    asyncio.run(send_prompt(
      "hello",
      capability_execution=_execution(provider),
      user_id="alice",
      completion_record_root=None,
    ))

  assert provider.stream_calls == 0
  assert _rows() == []


def test_an_error_raised_after_message_end_is_an_error_row_not_ok() -> None:
  provider = _Provider([
    StreamEvent(type="message_start", input_tokens=10),
    StreamEvent(type="usage_update", output_tokens=4),
    StreamEvent(type="message_end", stop_reason="error"),
    RuntimeError("response failed after inference"),
  ])
  with pytest.raises(RuntimeError, match="after inference"):
    asyncio.run(_drain(_meter(provider)))

  [row] = _rows()
  assert (row["status"], row["error_type"], row["input_tokens"], row["output_tokens"]) == (
    "error", "RuntimeError", 10, 4,
  )


def test_a_consumer_that_stops_after_message_end_is_a_completed_call() -> None:
  async def stop_at_end() -> None:
    meter = _meter(_Provider(_ok_step()))
    stream = meter.stream(object(), {})
    async for event in stream:
      if event.type == "message_end":
        break
    await stream.aclose()

  asyncio.run(stop_at_end())

  [row] = _rows()
  assert (row["status"], row["error_type"]) == ("ok", None)


def test_a_cancelled_stream_writes_one_row_with_its_partial_usage() -> None:
  async def scenario() -> None:
    provider = _Provider([StreamEvent(type="message_start", input_tokens=10), "hang"])
    task = asyncio.create_task(_drain(_meter(provider)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task

  asyncio.run(scenario())

  [row] = _rows()
  assert (row["status"], row["error_type"]) == ("error", "CancelledError")
  assert row["input_tokens"] == 10
  assert json.loads(row["attrs_json"])["canceled"] is True


def test_an_enclosing_spend_context_call_site_names_the_real_caller() -> None:
  with spend_context(call_site="eval_jury", run_id="run-9"):
    asyncio.run(_drain(_meter(_Provider(_ok_step()))))

  [row] = _rows()
  assert (row["call_site"], row["run_id"]) == ("eval_jury", "run-9")


# ── ceilings ─────────────────────────────────────────────────────────────────────


def test_a_frozen_payg_call_is_refused_and_recorded_as_budget_blocked() -> None:
  _freeze_payg()
  with pytest.raises(SpendBudgetBlocked) as refused:
    _meter(_Provider(_ok_step())).enforce()

  assert [breach.ceiling for breach in refused.value.breaches] == ["freeze_payg"]
  assert "hank_spend.py set" in str(refused.value)
  [row] = _rows()
  assert (row["status"], row["api_calls"], row["input_tokens"], row["auth_kind"]) == (
    "budget_blocked", 0, 0, "payg_api_key",
  )


def test_an_oauth_call_is_never_refused_by_a_freeze() -> None:
  _freeze_payg()
  _meter(_Provider(_ok_step()), OAUTH_CONFIG).enforce()
  asyncio.run(_drain(_meter(_Provider(_ok_step()), OAUTH_CONFIG)))

  [row] = _rows()
  assert (row["status"], row["auth_kind"], row["payg_cost_usd"]) == ("ok", "oauth_subscription", 0.0)
  assert row["shadow_cost_usd"] == 0.0042


def test_a_per_key_ceiling_met_by_the_shared_ledger_refuses_the_next_payg_call() -> None:
  Path(os.environ["HANK_SPEND_SETTINGS_PATH"]).write_text(json.dumps({"per_key_daily_payg_usd": 0.004}))
  asyncio.run(_drain(_meter(_Provider(_ok_step()))))

  with pytest.raises(SpendBudgetBlocked) as refused:
    _meter(_Provider(_ok_step())).enforce()

  [breach] = refused.value.breaches
  assert breach.ceiling == "per_key_daily_payg_usd"
  assert breach.scope == fingerprint_credential("sk-ant-api-key-1")
  assert [row["status"] for row in _rows()] == ["ok", "budget_blocked"]


# ── runner: retries, refusal ──────────────────────────────────────────────────────


def _runner(provider: ModelProvider, config: dict[str, Any] | None = None) -> tuple[AgentRunner, list[dict]]:
  event_log = EventLog()
  execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-opus-5",
    effort="none",
    auth_config=config or {"api_key": "k"},
  )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=ToolDispatcher(
      capability_bind=execution.bind,
      mcp_client=McpClientManager(config_path=None),
      local_tool_handlers={},
      event_log=event_log,
      session_id="sess-spend",
      get_tool_definitions=None,
    ),
    session_id="sess-spend",
    capability_execution=execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  durable: list[dict] = []

  async def _append_durable_event(event: dict) -> Any:
    durable.append(dict(event))
    return SimpleNamespace(seq=len(durable))

  runner._append_durable_event = _append_durable_event
  return runner, durable


def test_a_retried_runner_turn_writes_one_row_per_upstream_attempt(monkeypatch) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0)
  provider = _Provider(_usage_then(RuntimeError("transient")), _ok_step("done"))
  runner, _ = _runner(provider)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  failed, served = _rows()
  assert provider.stream_calls == 2
  assert (failed["status"], failed["error_type"], failed["input_tokens"]) == ("error", "RuntimeError", 10)
  assert (served["status"], served["input_tokens"], served["output_tokens"]) == ("ok", 10, 4)
  assert {row["call_site"] for row in (failed, served)} == {"chat_turn"}
  assert {row["session_id"] for row in (failed, served)} == {"sess-spend"}
  assert failed["model_key"] == served["model_key"]
  assert failed["request_id"] == served["request_id"]


def test_a_runner_turn_refused_by_a_freeze_never_starts_the_provider_call_or_retries(monkeypatch) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0)
  _freeze_payg()
  provider = _Provider(_ok_step())
  runner, _ = _runner(provider)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  assert provider.stream_calls == 0
  [row] = _rows()
  assert (row["status"], row["input_tokens"], row["api_calls"]) == ("budget_blocked", 0, 0)
  errors = [entry.event for entry in runner._log.entries if entry.event.get("type") == "error"]
  assert len(errors) == 1
  assert "freeze_payg" in str(errors[0]) and "hank_spend.py set" in str(errors[0])


def test_a_runner_turn_on_an_oauth_credential_runs_under_a_freeze() -> None:
  _freeze_payg()
  provider = _Provider(_ok_step())
  runner, _ = _runner(provider, OAUTH_CONFIG)

  asyncio.run(runner.run(messages=[{"role": "user", "content": "go"}]))

  [row] = _rows()
  assert (row["status"], row["auth_kind"]) == ("ok", "oauth_subscription")


# ── send_prompt and compaction ────────────────────────────────────────────────────


def _execution(provider: ModelProvider, **config: Any):
  return stub_bound_capability_execution(
    provider=provider,
    model="claude-sonnet-4-6",
    effort="none",
    credential_principal="user",
    auth_config={"provider": "anthropic", "max_tokens": 4096, "auth_mode": "api", "api_key": "k", **config},
  )


def test_send_prompt_writes_one_row_for_its_provider_call() -> None:
  provider = _Provider(_ok_step())

  result = asyncio.run(send_prompt(
    "hello",
    capability_execution=_execution(provider),
    user_id="alice",
    session_id="sess-sp",
    request_id="req-sp",
    channel="mcp",
    completion_record_root=None,
  ))

  assert result == "ok"
  [row] = _rows()
  assert (row["call_site"], row["session_id"], row["request_id"]) == ("send_prompt:mcp", "sess-sp", "req-sp")
  assert (row["status"], row["input_tokens"], row["output_tokens"]) == ("ok", 10, 4)
  assert json.loads(row["attrs_json"])["capability_id"]


def test_send_prompt_refuses_a_frozen_payg_call_before_creating_a_client() -> None:
  _freeze_payg()
  provider = _Provider(_ok_step())

  with pytest.raises(SpendBudgetBlocked):
    asyncio.run(send_prompt(
      "hello",
      capability_execution=_execution(provider),
      user_id="alice",
      completion_record_root=None,
    ))

  assert provider.stream_calls == 0
  assert [row["status"] for row in _rows()] == ["budget_blocked"]


def test_send_prompt_on_oauth_credentials_runs_under_a_freeze() -> None:
  _freeze_payg()
  provider = _Provider(_ok_step())

  asyncio.run(send_prompt(
    "hello",
    capability_execution=_execution(provider, auth_mode="oauth", auth_mechanism="oauth_bearer",
                                    auth_token="tok", api_key=""),
    user_id="alice",
    completion_record_root=None,
  ))

  [row] = _rows()
  assert (row["status"], row["auth_kind"]) == ("ok", "oauth_subscription")


def test_send_prompt_failure_with_usage_writes_one_error_row() -> None:
  provider = _Provider(_usage_then(RuntimeError("boom")))

  with pytest.raises(RuntimeError, match="boom"):
    asyncio.run(send_prompt(
      "hello",
      capability_execution=_execution(provider),
      user_id="alice",
      completion_record_root=None,
    ))

  [row] = _rows()
  assert (row["status"], row["error_type"], row["input_tokens"]) == ("error", "RuntimeError", 10)


def test_compaction_summary_call_writes_one_row() -> None:
  provider = _Provider(_ok_step("summary"))

  result = asyncio.run(provider_summarize(
    capability_execution=_execution(provider),
    messages=[{"role": "user", "content": "summarize"}],
    system_prompt=None,
    max_tokens=1000,
    session_id="sess-compact",
  ))

  assert result.text == "summary"
  [row] = _rows()
  assert (row["call_site"], row["session_id"], row["status"]) == ("compaction", "sess-compact", "ok")
  assert (row["input_tokens"], row["output_tokens"]) == (10, 4)


# ── Agent SDK ────────────────────────────────────────────────────────────────────


def _sdk_self(usage: dict[str, Any], num_turns: int = 3) -> Any:
  return SimpleNamespace(_usage=usage, _num_turns=num_turns, _effective_model="claude-opus-5")


def test_an_sdk_query_writes_one_row_from_its_cumulative_result_usage() -> None:
  meter = _meter(None, OAUTH_CONFIG, call_site="agent_sdk")  # type: ignore[arg-type]
  usage = {
    "input_tokens": 100, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 300,
    "output_tokens": 40, "reasoning_tokens_observed": 0, "estimated_cost": 0.5,
    "provider_reported_cost_usd": 0.5, "provider_reported_model": None, "provider_unit_deltas": {},
  }

  AgentSDKRunner._record_sdk_spend(
    _sdk_self(usage), meter, None, error=None, started=time.monotonic(),
  )

  [row] = _rows()
  assert (row["call_site"], row["auth_kind"], row["status"], row["api_calls"]) == (
    "agent_sdk", "oauth_subscription", "ok", 3,
  )
  assert (row["input_tokens"], row["cache_write_tokens"], row["cache_read_tokens"], row["output_tokens"]) == (
    100, 20, 300, 40,
  )
  assert (row["shadow_cost_usd"], row["payg_cost_usd"]) == (0.5, 0.0)


def test_an_sdk_query_that_observed_nothing_and_was_not_cancelled_writes_nothing() -> None:
  meter = _meter(None, API_CONFIG, call_site="agent_sdk")  # type: ignore[arg-type]
  usage = {"input_tokens": 0, "output_tokens": 0, "provider_unit_deltas": {}}

  AgentSDKRunner._record_sdk_spend(
    _sdk_self(usage), meter, None, error=RuntimeError("connect failed"), started=time.monotonic(),
  )

  assert _rows() == []


def test_a_cancelled_sdk_query_records_the_calls_observed_before_the_result() -> None:
  meter = _meter(None, API_CONFIG, call_site="agent_sdk")  # type: ignore[arg-type]
  def observed() -> dict[str, Any]:
    return {
      "input_tokens": 50, "output_tokens": 7, "cache_read_input_tokens": 0,
      "cache_creation_input_tokens": 0, "provider_unit_deltas": {},
    }

  AgentSDKRunner._record_sdk_spend(
    _sdk_self({"input_tokens": 0, "output_tokens": 0}, num_turns=0),
    meter, observed, error=asyncio.CancelledError(), started=time.monotonic(),
  )

  [row] = _rows()
  assert (row["status"], row["error_type"], row["input_tokens"], row["output_tokens"]) == (
    "error", "CancelledError", 50, 7,
  )
  assert row["api_calls"] == 1 and row["auth_kind"] == "payg_api_key"
  assert row["shadow_cost_usd"] is not None


# ── rotation must not carry the previous credential's class ─────────────────────


def _stamped(mechanism: str, entitlement: str, **material: Any) -> dict[str, Any]:
  return {
    "provider": "anthropic",
    "max_tokens": 4096,
    "auth_mechanism": mechanism,
    "entitlement_kind": entitlement,
    **material,
  }


SEAT = _stamped(
  "oauth_bearer", "operator_personal_subscription",
  auth_mode="oauth", auth_token="sk-ant-oat-seat", api_key="",
)
KEY = _stamped(
  "api_key", "org_commercial", auth_mode="api", api_key="sk-ant-api-key-1", auth_token="",
)


def _unstamped_material(**material: Any) -> dict[str, Any]:
  return dict(material)


def test_a_seat_rotated_to_an_api_key_is_classified_and_refused_as_payg() -> None:
  from agent_gateway.runner_auth import merge_refreshed_auth_config

  merged = merge_refreshed_auth_config(
    SEAT, _unstamped_material(auth_mode="api", api_key="sk-ant-api-key-1", auth_token=""),
  )

  assert (merged["auth_mechanism"], merged["entitlement_kind"]) == ("api_key", "org_commercial")
  _freeze_payg()
  meter = _meter(_Provider(_ok_step()), merged)
  assert meter.credential.auth_kind == "payg_api_key"
  assert meter.credential.fingerprint == fingerprint_credential("sk-ant-api-key-1")
  with pytest.raises(SpendBudgetBlocked):
    meter.enforce()


def test_an_api_key_rotated_to_a_seat_is_classified_as_a_subscription_and_not_refused() -> None:
  from agent_gateway.runner_auth import merge_refreshed_auth_config

  merged = merge_refreshed_auth_config(
    KEY, _unstamped_material(auth_mode="oauth", api_key="", auth_token="sk-ant-oat-seat"),
  )

  assert (merged["auth_mechanism"], merged["entitlement_kind"]) == (
    "oauth_bearer", "operator_personal_subscription",
  )
  _freeze_payg()
  meter = _meter(_Provider(_ok_step()), merged)
  assert meter.credential.auth_kind == "oauth_subscription"
  meter.enforce()
  asyncio.run(_drain(meter))
  [row] = _rows()
  assert (row["status"], row["auth_kind"], row["payg_cost_usd"]) == ("ok", "oauth_subscription", 0.0)


def test_a_refresh_that_restamps_its_class_keeps_the_stamp_it_supplied() -> None:
  from agent_gateway.runner_auth import merge_refreshed_auth_config

  merged = merge_refreshed_auth_config(
    SEAT,
    _stamped("api_key", "commercial_byok", auth_mode="api", api_key="k2", auth_token=""),
  )

  assert (merged["auth_mechanism"], merged["entitlement_kind"]) == ("api_key", "commercial_byok")


def test_a_refresh_of_the_same_material_keeps_its_stamps() -> None:
  from agent_gateway.runner_auth import merge_refreshed_auth_config

  merged = merge_refreshed_auth_config(KEY, _unstamped_material(account_identity="acct"))

  assert (merged["auth_mechanism"], merged["entitlement_kind"]) == ("api_key", "org_commercial")


@pytest.mark.parametrize(
  ("record_kind", "token", "expected"),
  [
    ("api_key", "sk-ant-api-key-1", ("api_key", "org_commercial")),
    ("subscription", "sk-ant-oat-seat", ("oauth_bearer", "operator_personal_subscription")),
  ],
)
def test_anthropic_rotation_material_stamps_the_class_of_the_member_it_names(
  record_kind: str, token: str, expected: tuple[str, str],
) -> None:
  from agent_gateway.providers.anthropic import _credential_material

  selected = SimpleNamespace(
    record=SimpleNamespace(credential_kind=record_kind),
    token=token,
    store_path=None,
    prepared=False,
    source_metadata=lambda: {"account_identity": "acct"},
  )

  material = _credential_material(selected, SEAT)  # type: ignore[arg-type]

  assert (material["auth_mechanism"], material["entitlement_kind"]) == expected


# ── subscriptions whose wire credential is a minted key ───────────────────────────

MUSE_SEAT = {
  "provider": "muse", "auth_mode": "oauth", "api_key": "muse-minted-key", "auth_token": "",
  "auth_mechanism": "api_key", "entitlement_kind": "operator_personal_subscription",
}
MUSE_KEY = {
  "provider": "muse", "auth_mode": "api", "api_key": "muse-real-key", "auth_token": "",
  "auth_mechanism": "api_key", "entitlement_kind": "org_commercial",
}


@pytest.mark.parametrize(
  ("entitlement", "mechanism", "expected"),
  [
    ("consumer_subscription", "api_key", "oauth_subscription"),
    ("operator_personal_subscription", "api_key", "oauth_subscription"),
    ("consumer_subscription", "oauth_bearer", "oauth_subscription"),
    ("local_unauthenticated", "api_key", "local_or_free"),
    ("commercial_byok", "api_key", "payg_api_key"),
    ("org_commercial", "api_key", "payg_api_key"),
    ("platform_metered", "api_key", "payg_api_key"),
  ],
)
def test_the_entitlement_overrides_the_wire_mechanism_for_every_kind(
  entitlement: str, mechanism: str, expected: str,
) -> None:
  config = {
    "provider": "anthropic", "auth_mode": "api", "api_key": "k", "auth_token": "",
    "auth_mechanism": mechanism, "entitlement_kind": entitlement,
  }

  assert spend_credential("anthropic", config).auth_kind == expected


def test_a_muse_subscription_is_a_seat_and_is_not_refused_under_a_freeze() -> None:
  _freeze_payg()
  meter = _meter(_Provider(_ok_step()), MUSE_SEAT, provider_name="muse")

  assert meter.credential.auth_kind == "oauth_subscription"
  assert meter.credential.fingerprint == fingerprint_credential("muse-minted-key")
  meter.enforce()


def test_a_real_muse_api_key_is_payg_and_refused_under_a_freeze() -> None:
  _freeze_payg()
  meter = _meter(_Provider(_ok_step()), MUSE_KEY, provider_name="muse")

  assert meter.credential.auth_kind == "payg_api_key"
  with pytest.raises(SpendBudgetBlocked):
    meter.enforce()


# ── the real OpenAI and Codex adapters on recorded event sequences ─────────────────


class _OpenAIEvents:
  def __init__(self, events: list[dict[str, Any]]) -> None:
    self._events = events

  def __aiter__(self):
    self._iterator = iter(self._events)
    return self

  async def __anext__(self):
    try:
      return next(self._iterator)
    except StopIteration as exc:
      raise StopAsyncIteration from exc

  async def __aenter__(self):
    return self

  async def __aexit__(self, *_exc: object) -> None:
    return None


class _OpenAIClient:
  def __init__(self, events: list[dict[str, Any]]) -> None:
    outer = self

    class Responses:
      async def create(self, **params: Any) -> _OpenAIEvents:
        return _OpenAIEvents(outer.events)

    self.events = events
    self.responses = Responses()


def _openai_meter(provider: ModelProvider, model: str = "gpt-5.6") -> ProviderCallMeter:
  return _meter(
    provider,
    {"provider": "openai", "auth_mode": "api", "api_key": "sk-openai"},
    provider_name="openai",
    model=model,
    call_site="chat_turn",
  )


def test_openai_response_failed_after_inference_is_one_error_row_with_its_usage() -> None:
  from agent_gateway.providers import OpenAIProvider

  client = _OpenAIClient([{
    "type": "response.failed",
    "response": {
      "status": "failed",
      "error": {"message": "request failed after inference"},
      "usage": {"input_tokens": 18, "output_tokens": 7},
    },
  }])

  async def drain() -> None:
    async for _ in _openai_meter(OpenAIProvider()).stream(client, {"model": "gpt-5.6"}):
      pass

  with pytest.raises(RuntimeError, match="after inference"):
    asyncio.run(drain())

  [row] = _rows()
  assert (row["status"], row["error_type"], row["input_tokens"], row["output_tokens"]) == (
    "error", "RuntimeError", 18, 7,
  )


def test_openai_error_after_response_created_is_one_error_row() -> None:
  from agent_gateway.providers import OpenAIProvider

  client = _OpenAIClient([
    {"type": "response.created", "response": {"model": "gpt-5.6"}},
    {"type": "error", "message": "server had an error"},
  ])

  async def drain() -> None:
    async for _ in _openai_meter(OpenAIProvider()).stream(client, {"model": "gpt-5.6"}):
      pass

  with pytest.raises(RuntimeError, match="server had an error"):
    asyncio.run(drain())

  [row] = _rows()
  assert (row["status"], row["error_type"], row["input_tokens"]) == ("error", "RuntimeError", 0)


@pytest.mark.parametrize(
  "failure_event",
  [
    {"type": "error", "message": "something broke"},
    {"type": "response.failed", "response": {"error": {"message": "no capacity"}}},
  ],
)
def test_codex_failure_after_the_request_was_sent_is_one_error_row(failure_event: dict[str, Any]) -> None:
  import base64

  import httpx

  from agent_gateway.providers import CodexProvider
  from agent_gateway.providers.codex_helpers import JWT_CLAIM_PATH

  def segment(value: dict[str, Any]) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

  token = ".".join([
    segment({"alg": "none", "typ": "JWT"}),
    segment({JWT_CLAIM_PATH: {"chatgpt_account_id": "acct_test"}}),
    "sig",
  ])
  payload = "".join(
    "data: " + json.dumps(event) + "\n\n"
    for event in ({"type": "response.created", "response": {"model": "gpt-6-sol"}}, failure_event)
  )

  def handle(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)

  async def scenario() -> None:
    provider = CodexProvider()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
      provider._client_state[client] = {
        "token": token, "account_id": "acct_test",
        "endpoint_url": "https://codex.example/responses", "init_headers": {},
      }
      meter = _meter(
        provider,
        {"provider": "codex", "auth_mode": "oauth", "auth_mechanism": "oauth_bearer",
         "auth_token": token, "api_key": ""},
        provider_name="codex",
        model="gpt-6-sol",
      )
      async for _ in meter.stream(client, {"model": "gpt-6-sol"}):
        pass

  with pytest.raises(RuntimeError):
    asyncio.run(scenario())

  [row] = _rows()
  assert (row["status"], row["error_type"], row["auth_kind"]) == ("error", "RuntimeError", "oauth_subscription")


# ── Agent SDK: run() writes the row ──────────────────────────────────────────────────


def _install_sdk(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> SimpleNamespace:
  import types

  from agent_gateway.providers.agent_sdk import SDK_PINNED_VERSION

  state = SimpleNamespace(clients=0)

  class Messages:
    def __init__(self) -> None:
      self.items = list(messages)

    def __aiter__(self):
      return self

    async def __anext__(self):
      if not self.items:
        raise StopAsyncIteration
      item = self.items.pop(0)
      if isinstance(item, BaseException):
        raise item
      return item

    async def aclose(self) -> None:
      self.items.clear()

  class Client:
    def __init__(self, *, options: Any) -> None:
      state.clients += 1
      self.iterator: Messages | None = None

    async def connect(self) -> None:
      return None

    async def query(self, prompt: Any, session_id: str = "default") -> None:
      self.iterator = Messages()

    def receive_response(self) -> Any:
      return self.iterator

    async def disconnect(self) -> None:
      return None

  module = types.ModuleType("claude_agent_sdk")
  module.__dict__.update({
    "__version__": SDK_PINNED_VERSION,
    "HookMatcher": lambda **kwargs: SimpleNamespace(**kwargs),
    "ClaudeAgentOptions": lambda **kwargs: SimpleNamespace(**kwargs),
    "ClaudeSDKClient": Client,
  })
  monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
  return state


def _sdk_result() -> Any:
  return SimpleNamespace(
    subtype="success", is_error=False, duration_ms=5, num_turns=2, total_cost_usd=0.5,
    usage={
      "input_tokens": 100, "output_tokens": 40,
      "cache_read_input_tokens": 300, "cache_creation_input_tokens": 20,
    },
  )


def _sdk_runner(**auth: Any) -> AgentSDKRunner:
  from agent_gateway import AgentSDKConfig, EventLog
  from gateway_test_support.sdk_capability_execution_test_support import stub_sdk_capability_execution

  return AgentSDKRunner(
    event_log=EventLog(),
    session_id="sess-sdk-spend",
    sdk_config=AgentSDKConfig(
      user_id="alice", billing_mode="byok", rate_table_version="unknown", request_id="req-sdk-spend",
    ),
    capability_execution=stub_sdk_capability_execution(**auth),
    system_prompt="test",
  )


def test_an_sdk_run_writes_one_row_from_its_result_message(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_sdk(monkeypatch, [_sdk_result()])
  runner = _sdk_runner()

  asyncio.run(runner.run([{"role": "user", "content": "go"}]))

  [row] = _rows()
  assert (row["call_site"], row["session_id"], row["request_id"]) == ("agent_sdk", "sess-sdk-spend", "req-sdk-spend")
  assert (row["status"], row["auth_kind"], row["api_calls"]) == ("ok", "payg_api_key", 2)
  assert (row["input_tokens"], row["cache_read_tokens"], row["cache_write_tokens"], row["output_tokens"]) == (
    100, 300, 20, 40,
  )
  assert row["shadow_cost_usd"] == 0.5


def test_an_sdk_run_that_fails_after_its_usage_is_an_error_row(monkeypatch: pytest.MonkeyPatch) -> None:
  failed = _sdk_result()
  failed.subtype = "error_during_execution"
  failed.is_error = True
  _install_sdk(monkeypatch, [failed])

  with pytest.raises(RuntimeError, match="agent_sdk_result_error"):
    asyncio.run(_sdk_runner().run([{"role": "user", "content": "go"}]))

  [row] = _rows()
  assert (row["status"], row["error_type"], row["input_tokens"]) == ("error", "RuntimeError", 100)


def test_an_sdk_run_under_a_freeze_is_refused_before_the_sdk_starts(monkeypatch: pytest.MonkeyPatch) -> None:
  state = _install_sdk(monkeypatch, [_sdk_result()])
  _freeze_payg()

  with pytest.raises(SpendBudgetBlocked):
    asyncio.run(_sdk_runner().run([{"role": "user", "content": "go"}]))

  assert state.clients == 0
  [row] = _rows()
  assert (row["status"], row["call_site"], row["api_calls"]) == ("budget_blocked", "agent_sdk", 0)


def test_an_sdk_run_on_an_oauth_seat_is_not_refused_under_a_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
  _install_sdk(monkeypatch, [_sdk_result()])
  _freeze_payg()

  asyncio.run(_sdk_runner(auth_mode="oauth", api_key="", auth_token="sk-ant-oat-seat").run(
    [{"role": "user", "content": "go"}],
  ))

  [row] = _rows()
  assert (row["status"], row["auth_kind"], row["payg_cost_usd"]) == ("ok", "oauth_subscription", 0.0)


def test_compaction_summary_under_a_freeze_is_refused_for_payg_only() -> None:
  _freeze_payg()
  provider = _Provider(_ok_step("summary"))

  with pytest.raises(SpendBudgetBlocked):
    asyncio.run(provider_summarize(
      capability_execution=_execution(provider),
      messages=[{"role": "user", "content": "summarize"}],
      system_prompt=None,
      max_tokens=1000,
    ))

  assert provider.stream_calls == 0
  assert [row["status"] for row in _rows()] == ["budget_blocked"]
