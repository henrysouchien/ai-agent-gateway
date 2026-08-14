from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.multi_user.billing import (  # noqa: E402
  SqliteUsageLedger,
  UsageEvent,
  _UsageAggregator,
)
from agent_gateway.events import _session_usage_summary  # noqa: E402
from agent_gateway.capability_binding import CapabilityResolutionError  # noqa: E402
from agent_gateway.model_registry import ProductModelRegistry  # noqa: E402
from agent_gateway.providers.anthropic import AnthropicProvider  # noqa: E402
from agent_gateway.providers.codex_helpers import (  # noqa: E402
  _ResponsesStreamState as CodexStreamState,
  _map_event as map_codex_event,
)
from agent_gateway.providers.openai_responses_helpers import (  # noqa: E402
  _ResponsesStreamState as OpenAIStreamState,
  map_event as map_openai_event,
)
from agent_gateway.providers.xai_helpers import map_event as map_xai_event  # noqa: E402
from agent_gateway.provider_summarize import provider_summarize  # noqa: E402
from agent_gateway.runner_usage import build_usage_event, usage_delta  # noqa: E402
from agent_gateway.runner_run_loop import _merge_usage_totals  # noqa: E402
from agent_gateway.send_prompt import send_prompt  # noqa: E402
from agent_gateway.providers import (  # noqa: E402
  ModelInfo,
  ModelProvider,
  StreamEvent,
)
from tests.capability_execution_test_support import (  # noqa: E402
  stub_bound_capability_execution,
)
from agent_workflow_contracts import CapabilityBind  # noqa: E402


class _StaticStream:
  def __init__(self, events: list[Any]) -> None:
    self._events = events

  async def __aenter__(self) -> _StaticStream:
    self._iterator = iter(self._events)
    return self

  async def __aexit__(self, *_args: Any) -> None:
    return None

  def __aiter__(self) -> _StaticStream:
    return self

  async def __anext__(self) -> Any:
    try:
      return next(self._iterator)
    except StopIteration as exc:
      raise StopAsyncIteration from exc


class _Messages:
  def __init__(self, events: list[Any]) -> None:
    self._events = events

  def stream(self, **_kwargs: Any) -> _StaticStream:
    return _StaticStream(self._events)


def _terminal_response(model: str) -> dict[str, Any]:
  return {
    "type": "response.completed",
    "response": {
      "model": model,
      "status": "completed",
      "usage": {"input_tokens": 3, "output_tokens": 2},
    },
  }


def test_response_mappers_preserve_provider_reported_model_verbatim() -> None:
  openai_events = map_openai_event(
    _terminal_response("gpt-5.6-2026-08-01"), OpenAIStreamState()
  )
  codex_events = map_codex_event(
    _terminal_response("gpt-5.6-sol"), CodexStreamState()
  )
  xai_events = map_xai_event(
    _terminal_response("grok-4.5-2026-08-01"), CodexStreamState()
  )

  assert openai_events[0].provider_reported_model == "gpt-5.6-2026-08-01"
  assert codex_events[0].provider_reported_model == "gpt-5.6-sol"
  assert xai_events[0].provider_reported_model == "grok-4.5-2026-08-01"


def test_response_mapper_does_not_infer_missing_reported_model() -> None:
  events = map_openai_event(
    {
      "type": "response.completed",
      "response": {
        "status": "completed",
        "usage": {"input_tokens": 3, "output_tokens": 2},
      },
    },
    OpenAIStreamState(),
  )

  assert events[0].type == "message_start"
  assert events[0].provider_reported_model is None


def test_anthropic_message_start_preserves_provider_reported_model() -> None:
  provider = AnthropicProvider()
  message = SimpleNamespace(
    type="message_start",
    message=SimpleNamespace(
      model="claude-opus-5-20260801",
      usage=SimpleNamespace(
        input_tokens=3,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
      ),
    ),
  )
  messages = _Messages([message])
  client = SimpleNamespace(
    messages=messages,
    beta=SimpleNamespace(messages=messages),
  )

  async def collect() -> list[Any]:
    return [
      event
      async for event in provider.stream(
        client,
        {"model": "claude-opus-5", "messages": []},
      )
    ]

  events = asyncio.run(collect())
  assert events[0].type == "message_start"
  assert events[0].provider_reported_model == "claude-opus-5-20260801"


def _bind_receipt() -> dict[str, str]:
  return CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key="anthropic.claude-opus-5",
    provider="anthropic",
    upstream_model="claude-opus-5",
    adapter="anthropic.sdk.messages",
    protocol_profile="anthropic.messages",
    route="direct",
    effort="none",
    credential_principal="user",
    credential_ref="test-credential",
    run_mode="interactive",
    registry_revision="test-v1",
    policy_revision="test-v1",
    selection_source="explicit_user",
  ).receipt()


def _usage_event() -> UsageEvent:
  return UsageEvent(
    user_id="user-1",
    session_id="session-1",
    request_id="request-1",
    parent_turn_id=None,
    timestamp=1.0,
    model="claude-opus-5",
    provider="anthropic",
    capability_bind=_bind_receipt(),
    provider_reported_model="claude-opus-5-20260801",
    input_tokens=3,
    output_tokens=2,
    cache_read_tokens=0,
    cache_creation_tokens=0,
    cost_usd=0.01,
    rate_table_version="rates-v1",
    billing_mode="byok",
    channel="cli",
  )


def test_usage_persistence_keeps_admitted_and_reported_identity_distinct(
  tmp_path: Path,
) -> None:
  path = tmp_path / "usage.sqlite3"
  ledger = SqliteUsageLedger(path)
  event = _usage_event()

  asyncio.run(ledger.record(event))
  with sqlite3.connect(path) as conn:
    row = conn.execute(
      """
      SELECT capability_bind_json, provider_reported_model
      FROM usage_events
      """
    ).fetchone()
  ledger.close()

  assert row == (
    json.dumps(
      _bind_receipt(), sort_keys=True, separators=(",", ":")
    ),
    "claude-opus-5-20260801",
  )


def test_runner_usage_delta_and_event_keep_identity_fields_distinct() -> None:
  before = {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0,
  }
  after = {
    **before,
    "input_tokens": 3,
    "output_tokens": 2,
    "capability_bind": _bind_receipt(),
    "provider_reported_model": "claude-opus-5-20260801",
  }
  delta = usage_delta(before, after)
  event = build_usage_event(
    user_id="user-1",
    session_id="session-1",
    request_id="request-1",
    parent_turn_id=None,
    timestamp=1.0,
    model="claude-opus-5",
    provider_name="anthropic",
    usage_totals=delta,
    cost_total=0.01,
    rate_table_version="rates-v1",
    billing_mode="byok",
    channel="cli",
  )

  assert event.capability_bind == _bind_receipt()
  assert event.provider_reported_model == "claude-opus-5-20260801"
  assert event.model == event.capability_bind["upstream_model"]


def test_session_summary_keeps_admitted_and_reported_identity_distinct() -> None:
  aggregator = _UsageAggregator(
    user_id="user-1",
    session_id="session-1",
    request_id="request-1",
    channel="cli",
  )

  async def summarize():
    assert await aggregator.record(_usage_event()) is True
    return await aggregator.snapshot(ended_at=2.0)

  summary = asyncio.run(summarize())
  assert summary.capability_bind == _bind_receipt()
  assert summary.provider_reported_model == "claude-opus-5-20260801"

  restored = _session_usage_summary(asdict(summary))
  assert restored is not None
  assert restored.capability_bind == _bind_receipt()
  assert restored.provider_reported_model == "claude-opus-5-20260801"


class _SendPromptProvider(ModelProvider):
  name = "anthropic"

  def __init__(self, reported_model: str) -> None:
    self.reported_model = reported_model

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=4096,
      supports_thinking=True,
    )

  def create_client(
    self,
    config: dict[str, Any],
    *,
    timeout: float | None = None,
  ) -> object:
    _ = config, timeout
    return object()

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return kwargs

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = client, params
    yield StreamEvent(
      type="message_start",
      input_tokens=3,
      provider_reported_model=self.reported_model,
    )
    yield StreamEvent(type="text_delta", text="ok")
    yield StreamEvent(type="usage_update", output_tokens=2)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


def _send_prompt_execution(reported_model: str):
  provider = _SendPromptProvider(reported_model)
  execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-opus-5",
    effort="none",
    auth_config={"max_tokens": 4096},
  )
  entry = execution.registry.require(execution.bind.model_key)
  admitted_entry = replace(
    entry,
    reported_identities=frozenset({entry.upstream_model, "claude-opus-5-20260801"}),
  )
  registry = ProductModelRegistry(
    schema=execution.registry.schema,
    revision=execution.registry.revision,
    models={admitted_entry.key: admitted_entry},
  )
  return replace(execution, registry=registry)


def test_send_prompt_validates_and_emits_reported_identity_without_rebinding() -> None:
  execution = _send_prompt_execution("claude-opus-5-20260801")
  events: list[UsageEvent] = []

  result = asyncio.run(
    send_prompt(
      "hello",
      capability_execution=execution,
      user_id="user-1",
      on_usage=events.append,
    )
  )

  assert result == "ok"
  assert len(events) == 1
  assert events[0].capability_bind == execution.bind.receipt()
  assert events[0].provider_reported_model == "claude-opus-5-20260801"
  assert events[0].model == execution.bind.upstream_model
  assert execution.bind.upstream_model == "claude-opus-5"


def test_send_prompt_rejects_unadmitted_reported_identity() -> None:
  execution = _send_prompt_execution("claude-unknown")

  try:
    asyncio.run(
      send_prompt(
        "hello",
        capability_execution=execution,
        user_id="user-1",
      )
    )
  except CapabilityResolutionError as exc:
    assert exc.code == "reported_identity_mismatch"
  else:
    raise AssertionError("unadmitted provider identity must be refused")


def test_provider_summarize_validates_and_carries_reported_identity() -> None:
  execution = _send_prompt_execution("claude-opus-5-20260801")

  result = asyncio.run(
    provider_summarize(
      capability_execution=execution,
      messages=[{"role": "user", "content": "summarize"}],
      system_prompt=None,
      max_tokens=1024,
    )
  )

  assert result.text == "ok"
  assert result.usage["capability_bind"] == execution.bind.receipt()
  assert result.usage["provider_reported_model"] == "claude-opus-5-20260801"


def test_provider_summarize_rejects_unadmitted_reported_identity() -> None:
  execution = _send_prompt_execution("claude-unknown")

  try:
    asyncio.run(
      provider_summarize(
        capability_execution=execution,
        messages=[{"role": "user", "content": "summarize"}],
        system_prompt=None,
        max_tokens=1024,
      )
    )
  except CapabilityResolutionError as exc:
    assert exc.code == "reported_identity_mismatch"
  else:
    raise AssertionError("unadmitted provider identity must be refused")


def test_compaction_usage_merge_keeps_validated_identity_fields() -> None:
  totals = {
    "input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens_observed": 0,
    "provider_units": 0,
    "provider_unit_deltas": {},
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
  }

  _merge_usage_totals(totals, {
    "input_tokens": 3,
    "output_tokens": 2,
    "capability_bind": _bind_receipt(),
    "provider_reported_model": "claude-opus-5-20260801",
  })

  assert totals["capability_bind"] == _bind_receipt()
  assert totals["provider_reported_model"] == "claude-opus-5-20260801"
