"""Model fallback hop: a spent account pool or a lasting outage moves the turn
to the next entry of the role's frozen chain (model-roles design §6.5, W7b)."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from uuid import uuid4
from pathlib import Path
from typing import Any, Literal

import pytest

from agent_gateway import AgentRunner, EventLog, McpClientManager, ModelInfo, ModelProvider, ToolDispatcher
import agent_gateway.runner as gateway_runner
from agent_gateway.admission import root_context
from agent_gateway.auth import ProviderCredentialFailure
from agent_gateway.execution_identity import DispatchIdentity
from agent_gateway.fork_request_handoff import (
  ForkRequestHandoff,
  build_mid_turn_handoff,
  fork_credential_identity_available,
)
from agent_gateway.capability_execution import (
  BoundCapabilityExecution,
  CapabilityExecutionResolver,
  MaterializedCredential,
)
from agent_gateway.providers import StreamEvent
from hank_contracts.commercial_work import WorkStartSpendLedger
from agent_gateway.providers.anthropic_oauth import (
  ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES,
  resolve_anthropic_credentials,
)
from agent_gateway.providers.anthropic_quota import Cooldown, record_cooldown
from agent_gateway.runner_sub_agents import _build_child_event_log, _ChildActivity
from agent_gateway.session import GatewaySession
from agent_gateway.tool_dispatcher_helpers import ToolResult
from gateway_test_support.model_defaults import compat_for_profile
from agent_workflow_contracts.models import SelectionProvenance
from model_authority.bind import CapabilityBind
from model_authority.binding import AuthContext, CredentialHandle, ModelSelectionIntent
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.registry import ModelRegistryEntry, ProductModelRegistry
from model_authority.rates import ModelRates, RateTable
from model_authority.schema import SCHEMA
from model_authority.selection import (
  CapabilityDefault,
  CapabilitySelectionPolicy,
  FallbackEntry,
  ProductModelSelectionPolicy,
  SelectionSnapshot,
  TierAliasDefault,
  TierModelsDefault,
  TierSelectionPolicy,
  capture_selection_snapshot,
)


OPUS = "test.anthropic.opus"
SONNET = "test.anthropic.sonnet"
SONNET_SDK = "test.anthropic.sonnet-sdk"
SOL = "test.codex.sol"
OPENAI = "test.openai.sol"
GROK = "test.xai.grok"
_UPSTREAM = {
  OPUS: "claude-opus-test", SONNET: "claude-sonnet-test", SONNET_SDK: "claude-sonnet-sdk-test",
  SOL: "gpt-sol-test", OPENAI: "gpt-openai-test", GROK: "grok-test",
}
_PROVIDER = {
  OPUS: "anthropic", SONNET: "anthropic", SONNET_SDK: "anthropic", SOL: "codex",
  OPENAI: "openai", GROK: "xai",
}
_TOKEN_A = "sk-ant-oat01-fallback-account-a-000000000000"
_TOKEN_B = "sk-ant-oat01-fallback-account-b-000000000000"
_CODEX_KEY = "codex-account"
_OPENAI_KEY = "sk-openai-fallback"
_GROK_KEY = "sk-xai-fallback"


class _Limited(RuntimeError):
  def __init__(self, reset_at: float) -> None:
    super().__init__("usage limit reached")
    self.reset_at = reset_at


class _Outage(RuntimeError):
  def __init__(self) -> None:
    super().__init__("provider overloaded")


class _Denied(RuntimeError):
  kind: Literal["auth", "billing"]
  reset_at: float | None

  def __init__(self, kind: Literal["auth", "billing"], reset_at: float | None) -> None:
    super().__init__("access denied")
    self.kind = kind
    self.reset_at = reset_at


class _World:
  """Every provider's scripted state: limits, outages, and what was requested."""

  def __init__(self) -> None:
    self.reset_at = time.time() + 3600
    self.limited_models: set[str] = set()
    self.limit_after: dict[str, int] = {}
    self.down: set[str] = set()
    self.siblings: dict[str, str] = {}
    self.turns: dict[str, list[list[StreamEvent]]] = {}
    self.windows: dict[str, int] = {}
    self.calls: list[tuple[str, str, str]] = []
    self.requests: list[dict[str, Any]] = []
    # Accounts whose organisation refuses OAuth, as a parked account did live.
    self.refusing_tokens: set[str] = set()
    # (provider that closed it, the client's token), in order.
    self.closed: list[tuple[str, Any]] = []
    # An outage that streams some tokens before failing, and a hook run on
    # each request with the request count.
    self.partial_outage = False
    self.denied_models: set[str] = set()
    self.denied_kind: Literal["auth", "billing"] = "auth"
    self.denied_reset_at: float | None = None
    self.on_call: Callable[[int], Awaitable[None]] | None = None
    # Providers that accept the request and then never send a stream event.
    self.stalled: set[str] = set()


class _ScriptedProvider(ModelProvider):
  def __init__(self, name: str, world: _World) -> None:
    self.name = name
    self._world = world

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("auth_token") or config.get("api_key"))

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = timeout
    return {"token": config.get("auth_token") or config.get("api_key")}

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = timeout
    if client is not None:
      self._world.closed.append((self.name, client["token"]))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model, provider=self.name, max_output_tokens=4096,
      context_window=self._world.windows.get(model, 200_000),
    )

  def build_request_params(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    _ = kwargs
    rendered = copy.deepcopy(messages)
    # One cache marker on the final block, as a cache-sharing request carries.
    last = rendered[-1]
    if isinstance(last.get("content"), str):
      last["content"] = [{"type": "text", "text": last["content"]}]
    last["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return {"model": model, "messages": rendered}

  async def stream(self, client: Any, params: dict[str, Any]):
    world = self._world
    model = params["model"]
    world.calls.append((self.name, model, client["token"]))
    world.requests.append(params)
    if world.on_call is not None:
      await world.on_call(len(world.calls))
    if self.name in world.down:
      if world.partial_outage:
        yield StreamEvent(type="message_start", input_tokens=10)
      raise _Outage()
    if self.name in world.stalled:
      await asyncio.sleep(3600)
    if model in world.denied_models:
      raise _Denied(world.denied_kind, world.denied_reset_at)
    served = sum(1 for _, called, _ in world.calls[:-1] if called == model)
    if model in world.limited_models or served >= world.limit_after.get(model, 1 << 30):
      raise _Limited(world.reset_at)
    if client["token"] in world.refusing_tokens:
      raise RuntimeError("OAuth authentication is currently not allowed for this organization")
    queued = world.turns.get(model) or []
    events = queued.pop(0) if queued else _text_turn(f"answer from {model}")
    yield StreamEvent(type="message_start", input_tokens=10)
    for event in events:
      yield event
    yield StreamEvent(type="usage_update", output_tokens=2)

  def classify_credential_failure(self, exc: Exception) -> ProviderCredentialFailure | None:
    if isinstance(exc, _Limited):
      return ProviderCredentialFailure(
        provider=self.name, kind="rate_limit", status_code=429, reset_at=exc.reset_at,
      )
    if isinstance(exc, _Denied):
      return ProviderCredentialFailure(
        provider=self.name, kind=exc.kind, status_code=403, reset_at=exc.reset_at,
      )
    return None

  def next_credential(self, config: Any, failure: ProviderCredentialFailure) -> dict[str, Any] | None:
    _ = failure
    sibling = self._world.siblings.get(str(config.get("auth_token") or ""))
    return {"auth_token": sibling} if sibling is not None else None

  def is_retryable_error(self, exc: Exception) -> bool:
    return isinstance(exc, (_Limited, _Outage))


def _text_turn(text: str) -> list[StreamEvent]:
  return [
    StreamEvent(type="text_delta", text=text),
    StreamEvent(type="text_end", raw_block={"type": "text", "text": text}),
    StreamEvent(type="message_end", stop_reason="end_turn"),
  ]


def _tool_use_turn(tool_id: str, name: str) -> list[StreamEvent]:
  return [
    StreamEvent(type="tool_use_start", tool_id=tool_id, tool_name=name),
    StreamEvent(
      type="tool_use_end", tool_id=tool_id, tool_name=name, tool_input={},
      raw_block={"type": "tool_use", "id": tool_id, "name": name, "input": {}},
    ),
    StreamEvent(type="message_end", stop_reason="tool_use"),
  ]


class _NullMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)

  def is_mcp_tool(self, name: str) -> bool:
    _ = name
    return False

  def get_tool_definitions(self) -> list[dict[str, Any]]:
    return []

  def get_server_for_tool(self, name: str) -> str | None:
    _ = name
    return None


def _entry(key: str) -> ModelRegistryEntry:
  provider = _PROVIDER[key]
  return ModelRegistryEntry(
    key=key,
    label=f"{provider} {_UPSTREAM[key]}",
    provider=provider,
    upstream_model=_UPSTREAM[key],
    adapter=f"test.{provider}",
    protocol_profile="responses.reasoning",
    route="test.in_process",
    lifecycle="active",
    capabilities={
      capability_id: "user_selectable" if capability_id == "session.driver" else "internal"
      for capability_id in CAPABILITY_IDS
    },
    supported_efforts=frozenset({"none", "low"}) if key == SOL else frozenset({"none"}),
    default_effort="none",
    features=frozenset({"tools", "streaming"}),
    reported_identities=frozenset({_UPSTREAM[key]}),
    compat=compat_for_profile("responses.reasoning"),
  )


def _resolver(
  world: _World,
  *,
  chain: tuple[FallbackEntry, ...] = (FallbackEntry(model=SOL), FallbackEntry(model=SONNET)),
  providers_with_credentials: frozenset[str] = frozenset({"anthropic", "codex"}),
  tiered: bool = True,
  plan_fallback: tuple[FallbackEntry, ...] | None = None,
  fast_fallback: tuple[FallbackEntry, ...] = (),
) -> CapabilityExecutionResolver:
  """``tiered=False`` is a later policy revision with concrete defaults and no roles.

  ``plan`` aliases ``deep``; ``plan_fallback`` gives it a chain of its own.
  """
  keys = frozenset(_UPSTREAM)
  registry = ProductModelRegistry(
    schema=SCHEMA, revision="test-fallback.1", models={key: _entry(key) for key in keys},
  )

  def _default(capability_id: str) -> CapabilityDefault:
    if capability_id == "node.fork":
      return CapabilityDefault(kind="inherit_parent")
    if tiered:
      return CapabilityDefault(kind="tier", tier="deep")
    return CapabilityDefault(kind="model", model_key=OPUS, effort="none")

  policy = ProductModelSelectionPolicy(
    schema=SCHEMA,
    revision="test-fallback.1",
    capabilities={
      capability_id: CapabilitySelectionPolicy(
        capability_id=capability_id,
        default=_default(capability_id),
        by_channel={},
        allowed_model_keys=keys,
        allow_saved_preference=capability_id == "session.driver",
        allow_explicit_user=capability_id == "session.driver",
      )
      for capability_id in CAPABILITY_IDS
    },
    tiers={
      # As shipped, `deep`'s own ceiling is narrower than its alias `plan`'s.
      "deep": TierSelectionPolicy(
        default=TierModelsDefault((OPUS,), "none"), allowed_model_keys=keys - {SONNET_SDK}, fallback=chain,
      ),
      "fast": TierSelectionPolicy(
        default=TierModelsDefault((SONNET,), "none"),
        allowed_model_keys=frozenset({
          SONNET, *(entry.model for entry in fast_fallback if entry.model is not None),
        }),
        **({"fallback": fast_fallback} if fast_fallback else {}),
      ),
      "plan": TierSelectionPolicy(
        default=TierAliasDefault("deep"), allowed_model_keys=keys, fallback=plan_fallback,
      ),
    } if tiered else {},
  )
  handles = {
    provider: CredentialHandle(
      handle_id=f"user:{provider}", provider=provider, principal="user",
      tenant_id="test-tenant", actor_id="test-actor",
    )
    for provider in providers_with_credentials
  }
  material = {
    "anthropic": {"auth_mode": "oauth", "auth_token": _TOKEN_A, "provider": "anthropic"},
    "codex": {"auth_mode": "api", "api_key": _CODEX_KEY, "provider": "codex"},
    "openai": {"auth_mode": "api", "api_key": _OPENAI_KEY, "provider": "openai"},
    "xai": {"auth_mode": "api", "api_key": _GROK_KEY, "provider": "xai"},
  }
  adapters = {
    "test.anthropic": _ScriptedProvider("anthropic", world),
    "test.codex": _ScriptedProvider("codex", world),
    "test.openai": _ScriptedProvider("openai", world),
    "test.xai": _ScriptedProvider("xai", world),
  }
  return CapabilityExecutionResolver(
    registry=registry,
    selection_policy=policy,
    auth_context=AuthContext(
      run_mode="interactive",
      actor_id="test-actor",
      tenant_id="test-tenant",
      user_provider_handles=handles,
      service_provider_handles={},
      entitled_capabilities=CAPABILITY_IDS,
      entitled_model_keys=keys,
    ),
    credential_materializer=lambda handle: MaterializedCredential(
      handle=handle, auth_config=material[handle.provider],
    ),
    adapter_resolver=lambda adapter: adapters[adapter],
  )


class _Turn:
  """One admitted interactive turn: snapshot, primary execution, runner."""

  def __init__(
    self,
    resolver: CapabilityExecutionResolver,
    *,
    snapshot: SelectionSnapshot | None = None,
    tool_calls: list[str] | None = None,
    explicit: str | None = None,
    gateway_session: Any | None = None,
    on_tool: Callable[[], None] | None = None,
    event_log: EventLog | None = None,
  ) -> None:
    self.snapshot = snapshot or capture_selection_snapshot(resolver.selection_policy)
    self.primary = resolver.resolve(
      "session.driver",
      explicit_intent=(
        ModelSelectionIntent(model_key=explicit, effort=None, source="explicit_user")
        if explicit is not None else None
      ),
      selection_snapshot=self.snapshot,
    )
    self.event_log = event_log if event_log is not None else EventLog()
    self.usage: list[Any] = []
    calls = tool_calls if tool_calls is not None else []

    async def _echo(_tool_input: dict[str, Any], *, call_index: int = 0, tool_ctx: Any = None) -> ToolResult:
      _ = call_index, tool_ctx
      calls.append("lookup")
      if on_tool is not None:
        on_tool()
      return {"status": "success"}, None

    self.dispatcher = ToolDispatcher(
      capability_bind=self.primary.bind,
      mcp_client=_NullMcpClient(),
      local_tool_handlers={"lookup": _echo},
      event_log=self.event_log,
      session_id="sess-fallback",
      role="owner",
      # The admitted job context carries the turn's frozen selection snapshot.
      identity=DispatchIdentity(
        context=root_context(
          None, withheld_tools=frozenset(), research_file_id=None, selection_snapshot=self.snapshot,
        ),
      ),
    )
    self.runner = AgentRunner(
      event_log=self.event_log,
      dispatcher=self.dispatcher,
      session_id="sess-fallback",
      gateway_session=gateway_session,
      capability_execution=self.primary,
      get_tool_definitions=lambda: [
        {"name": "lookup", "description": "", "input_schema": {"type": "object", "properties": {}}},
      ],
      on_usage=self.usage.append,
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )

  def run(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
    run = self.runner.run(messages=[{"role": "user", "content": "go"}], system_prompt="x")
    asyncio.run(asyncio.wait_for(run, timeout) if timeout is not None else run)
    return [entry.event for entry in self.event_log.entries]


@pytest.fixture(autouse=True)
def _fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 2)


@pytest.fixture(autouse=True)
def _isolated_anthropic_accounts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """The real Anthropic account pool over two accounts, isolated from this machine's.

  Parked accounts are cooldown rows in the store's selection sidecar, so a private
  store path isolates them too.
  """
  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_TOKEN_A, _TOKEN_B]))
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))


def _park(token: str, *, until: float) -> None:
  member = resolve_anthropic_credentials().by_token(token)
  assert member is not None
  record_cooldown(
    member.sidecar_store_path, member.locator,
    Cooldown(scope="shared", until=until, reason="rate_limit"),
  )


def _of_type(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
  return [event for event in events if event["type"] == kind]


def test_spent_pool_hops_to_the_next_chain_entry_and_bills_the_serving_bind() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS]}
  world.siblings = {_TOKEN_A: _TOKEN_B}
  turn = _Turn(_resolver(world))
  primary_bind = turn.primary.bind

  events = turn.run()

  # Same-model rotation runs first; only the spent pool hops.
  assert world.calls == [
    ("anthropic", "claude-opus-test", _TOKEN_A),
    ("anthropic", "claude-opus-test", _TOKEN_B),
    ("codex", "gpt-sol-test", _CODEX_KEY),
  ]
  [rotation] = _of_type(events, "credential_refreshed")
  [hop] = _of_type(events, "model_fallback")
  assert events.index(rotation) < events.index(hop)
  assert hop == {
    "type": "model_fallback",
    "from": {
      "model_key": OPUS, "label": "anthropic claude-opus-test", "provider": "anthropic",
      "upstream_model": "claude-opus-test", "effort": "none",
    },
    "to": {
      "model_key": SOL, "label": "codex gpt-sol-test", "provider": "codex",
      "upstream_model": "gpt-sol-test", "effort": "none",
    },
    "reason": "usage_limit",
    "reset_at": pytest.approx(world.reset_at),
    "role": "deep",
    "chain_index": 0,
    "skipped": [],
  }
  assert not _of_type(events, "error")
  [complete] = _of_type(events, "stream_complete")
  assert complete["terminal_disposition"] == "completed"

  hop_bind = turn.runner.capability_execution.bind
  assert hop_bind.model_key == SOL
  assert hop_bind.provider == "codex"
  assert hop_bind.credential_ref == "user:codex"
  assert hop_bind.selection_source == primary_bind.selection_source
  assert hop_bind.to_json() != primary_bind.to_json()
  assert turn.runner._auth_config["api_key"] == _CODEX_KEY
  assert "auth_token" not in turn.runner._auth_config or turn.runner._auth_config["auth_token"] != _TOKEN_B
  # The dispatcher's tool captures follow the hop (different model, same principal).
  assert turn.dispatcher._capability_bind is hop_bind
  # The failed attempt's client is closed by the provider that created it.
  assert ("anthropic", _TOKEN_B) in world.closed
  assert ("codex", _TOKEN_B) not in world.closed
  billed = [event for event in turn.usage if event.capability_bind is not None]
  assert billed[-1].capability_bind == hop_bind.to_json()
  assert billed[-1].model == "gpt-sol-test"
  [turn_complete] = _of_type(events, "turn_complete")
  assert turn_complete["usage"]["capability_bind"]["model_key"] == SOL
  [recap] = _of_type(events, "session_recap")
  assert recap["usage"]["model"] == "gpt-sol-test"


def test_refused_entries_are_skipped_and_an_exhausted_chain_fails_as_today() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS], _UPSTREAM[SONNET]}
  # No Codex login: the first entry is skipped with its typed refusal.
  turn = _Turn(_resolver(world, providers_with_credentials=frozenset({"anthropic"})))

  events = turn.run()

  [hop] = _of_type(events, "model_fallback")
  assert hop["to"]["model_key"] == SONNET
  assert hop["chain_index"] == 1
  assert [skip["model_key"] for skip in hop["skipped"]] == [SOL]
  assert hop["skipped"][0]["code"] == "default_not_eligible"
  assert {provider for provider, _, _ in world.calls} == {"anthropic"}
  assert world.calls[0][1] == "claude-opus-test"
  assert {model for _, model, _ in world.calls[1:]} == {"claude-sonnet-test"}
  [error] = _of_type(events, "error")
  assert "usage limit reached" in error["error"]
  assert turn.runner.capability_execution.bind.model_key == SONNET



def test_fallback_rebind_failure_logs_evaluated_candidates_and_refusal_codes(caplog: pytest.LogCaptureFixture) -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS], _UPSTREAM[SONNET]}
  turn = _Turn(_resolver(world, chain=(FallbackEntry(model=SOL),), providers_with_credentials=frozenset({"anthropic"})))

  with caplog.at_level(logging.WARNING, logger="agent_gateway.runner"):
    events = turn.run()

  assert not _of_type(events, "model_fallback")
  assert any(
    "with no eligible fallback chain entry (evaluated candidates:" in record.getMessage()
    and f"{SOL}: default_not_eligible" in record.getMessage()
    for record in caplog.records
  )

def test_entry_whose_context_cannot_hold_the_transcript_is_skipped() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS]}
  world.windows = {"gpt-sol-test": 1}
  turn = _Turn(_resolver(world))

  events = turn.run()

  [hop] = _of_type(events, "model_fallback")
  assert hop["to"]["model_key"] == SONNET
  assert hop["skipped"] == [{"model_key": SOL, "code": "context_window_exceeded"}]
  assert not _of_type(events, "error")


def test_outage_after_the_retry_budget_hops_and_spends_no_extra_primary_attempt() -> None:
  world = _World()
  world.down = {"anthropic"}
  turn = _Turn(_resolver(world))

  events = turn.run()

  assert world.calls == [("anthropic", "claude-opus-test", _TOKEN_A)] * (1 + gateway_runner.STREAM_RETRY_MAX) + [
    ("codex", "gpt-sol-test", _CODEX_KEY),
  ]
  [hop] = _of_type(events, "model_fallback")
  assert hop["reason"] == "provider_outage"
  assert hop["reset_at"] is None
  assert hop["to"]["model_key"] == SOL
  assert not _of_type(events, "error")


def test_outage_with_no_chain_takes_todays_failure_path() -> None:
  world = _World()
  world.down = {"anthropic"}
  turn = _Turn(_resolver(world, chain=()))

  events = turn.run()

  assert len(world.calls) == 1 + gateway_runner.STREAM_RETRY_MAX
  assert not _of_type(events, "model_fallback")
  assert len(_of_type(events, "error")) == 1
  assert turn.runner.capability_execution is turn.primary


@pytest.fixture
def _fast_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_STALL_TIMEOUT", 0.2)
  monkeypatch.setattr(gateway_runner, "STREAM_GUARD_POLL_INTERVAL", 0.02)


@pytest.mark.usefixtures("_fast_watchdog")
def test_stall_through_the_retry_budget_hops_as_an_outage() -> None:
  world = _World()
  world.stalled = {"anthropic"}
  turn = _Turn(_resolver(world))

  events = turn.run(timeout=30)

  assert world.calls == [("anthropic", "claude-opus-test", _TOKEN_A)] * (1 + gateway_runner.STREAM_RETRY_MAX) + [
    ("codex", "gpt-sol-test", _CODEX_KEY),
  ]
  [hop] = _of_type(events, "model_fallback")
  assert hop["reason"] == "provider_outage"
  assert hop["to"]["model_key"] == SOL
  assert not _of_type(events, "error")


@pytest.mark.usefixtures("_fast_watchdog")
def test_stall_with_no_chain_takes_todays_failure_path() -> None:
  world = _World()
  world.stalled = {"anthropic"}
  turn = _Turn(_resolver(world, chain=()))

  events = turn.run(timeout=30)

  assert len(world.calls) == 1 + gateway_runner.STREAM_RETRY_MAX
  assert not _of_type(events, "model_fallback")
  [error] = _of_type(events, "error")
  assert "Stream watchdog" in str(error.get("error"))


def test_completed_tool_effects_are_not_replayed_after_a_mid_run_hop() -> None:
  world = _World()
  world.turns = {"claude-opus-test": [_tool_use_turn("toolu_1", "lookup")]}
  world.limit_after = {"claude-opus-test": 1}
  tool_calls: list[str] = []
  turn = _Turn(_resolver(world), tool_calls=tool_calls)

  events = turn.run()

  assert tool_calls == ["lookup"]
  assert [call[1] for call in world.calls] == ["claude-opus-test", "claude-opus-test", "gpt-sol-test"]
  fallback_request = world.requests[-1]["messages"]
  tool_results = [
    block
    for message in fallback_request
    if isinstance(message.get("content"), list)
    for block in message["content"]
    if isinstance(block, dict) and block.get("type") == "tool_result"
  ]
  assert [block["tool_use_id"] for block in tool_results] == ["toolu_1"]
  assert len(_of_type(events, "tool_call_start")) == 1
  assert [
    event["usage"]["capability_bind"]["model_key"] for event in _of_type(events, "turn_complete")
  ] == [OPUS, SOL]
  assert not _of_type(events, "error")


def test_next_turn_returns_to_the_primary_once_its_block_window_ends() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS]}
  resolver = _resolver(world)
  first = _Turn(resolver)
  first.run()
  assert first.runner.capability_execution.bind.model_key == SOL

  world.limited_models = set()
  world.calls.clear()
  second = _Turn(resolver)
  events = second.run()

  assert second.primary.bind.model_key == OPUS
  assert world.calls == [("anthropic", "claude-opus-test", _TOKEN_A)]
  assert not _of_type(events, "model_fallback")
  assert second.runner.capability_execution.bind.model_key == OPUS


def test_a_fork_captured_before_the_hop_keeps_its_bind_and_none_is_captured_after_it() -> None:
  world = _World()
  world.turns = {"claude-opus-test": [_tool_use_turn("toolu_1", "lookup")]}
  world.limit_after = {"claude-opus-test": 1}
  handoffs: list[ForkRequestHandoff] = []
  resolver = _resolver(world)
  primary_handle = resolver.auth_context.user_provider_handles["anthropic"]
  # The session's bound credential handle is the turn's admitted primary.
  session = GatewaySession(
    session_id="sess-fallback", api_key_hash="hash", created_at=1, expires_at=2,
    user_id="alice", owner_user_id="alice", role="owner", tenant_id="test-tenant",
    session_credential_handle=primary_handle,
  )
  turn = _Turn(
    resolver,
    gateway_session=session,
    on_tool=lambda: handoffs.append(
      build_mid_turn_handoff(turn.runner, [{"role": "user", "content": "go"}]),
    ),
  )

  turn.run()

  live = turn.runner.capability_execution
  assert live.bind.model_key == SOL
  [before] = handoffs
  assert before.capability_bind == turn.primary.bind
  # The learning fork's admission selects node.fork from the handoff through
  # the parent's live resolver (runner_fork_agents); after the hop it is still A.
  assert live.resolver is not None
  fork_bind = live.resolver.select_bind(
    "node.fork", parent_bind=before.capability_bind, selection_snapshot=turn.snapshot,
  )
  assert (fork_bind.model_key, fork_bind.credential_ref) == (OPUS, primary_handle.handle_id)
  assert live.resolver.materialize_bind(fork_bind).bind == fork_bind
  # The run loop's fork capture sites read this gate: the session's credential
  # names the primary, not the serving bind, so no fork is captured after it.
  assert not fork_credential_identity_available(turn.runner)


def _no_role() -> SelectionProvenance:
  return SelectionProvenance(role=None, role_source=None, effort_source=None, preset=None)


def test_replace_capability_bind_admits_a_model_change_but_not_a_principal_change() -> None:
  world = _World()
  turn = _Turn(_resolver(world))
  hop = CapabilityExecutionResolver.rebind_model_fallback(
    turn.primary, turn.snapshot, selection_provenance=_no_role(), history_fits=lambda _execution: True,
  )
  assert hop is not None
  assert isinstance(hop.execution, BoundCapabilityExecution)

  turn.dispatcher.replace_capability_bind(hop.execution.bind)
  assert turn.dispatcher._capability_bind.model_key == SOL

  service_bind = CapabilityBind.model_validate({
    **hop.execution.bind.to_json(), "credential_principal": "service",
  })
  with pytest.raises(ValueError, match="changes execution authority"):
    turn.dispatcher.replace_capability_bind(service_bind)


def test_hop_walks_forward_from_the_last_position_and_never_revisits() -> None:
  world = _World()
  turn = _Turn(_resolver(world))
  first = CapabilityExecutionResolver.rebind_model_fallback(
    turn.primary, turn.snapshot, selection_provenance=_no_role(), history_fits=lambda _execution: True,
  )
  assert first is not None and first.execution.bind.model_key == SOL
  second = CapabilityExecutionResolver.rebind_model_fallback(
    first.execution, turn.snapshot, history_fits=lambda _execution: True, previous=first,
    selection_provenance=_no_role(),
  )
  assert second is not None and second.execution.bind.model_key == SONNET
  assert second.chain_index == 1
  assert CapabilityExecutionResolver.rebind_model_fallback(
    second.execution, turn.snapshot, history_fits=lambda _execution: True, previous=second,
    selection_provenance=_no_role(),
  ) is None


def test_explicit_pick_named_later_in_the_chain_hops_to_the_other_entries() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[SONNET]}
  turn = _Turn(_resolver(world), explicit=SONNET)
  assert (turn.primary.bind.model_key, turn.primary.bind.selection_source) == (SONNET, "explicit_user")

  events = turn.run()

  [hop] = _of_type(events, "model_fallback")
  assert (hop["to"]["model_key"], hop["chain_index"]) == (SOL, 0)
  assert turn.runner.capability_execution.bind.selection_source == "explicit_user"
  assert not _of_type(events, "error")


def test_requested_role_hops_along_its_chain_when_it_differs_from_the_slot_default() -> None:
  world = _World()
  resolver = _resolver(
    world, chain=(FallbackEntry(model=SOL),), fast_fallback=(FallbackEntry(model=OPUS),),
  )
  snapshot = capture_selection_snapshot(resolver.selection_policy)
  child = resolver.resolve("node.explore", role="fast", selection_snapshot=snapshot)
  assert child.bind.model_key == SONNET

  hop = CapabilityExecutionResolver.rebind_model_fallback(
    child, snapshot,
    selection_provenance=SelectionProvenance(
      role="fast", role_source="spawn", effort_source=None, preset=None,
    ),
    history_fits=lambda _execution: True,
  )
  assert hop is not None and (hop.role, hop.execution.bind.model_key) == ("fast", OPUS)


def test_no_provenance_does_not_hop() -> None:
  world = _World()
  turn = _Turn(_resolver(world))
  assert CapabilityExecutionResolver.rebind_model_fallback(
    turn.primary, turn.snapshot, selection_provenance=None, history_fits=lambda _execution: True,
  ) is None


def test_root_default_bind_hops_along_the_slot_default_chain() -> None:
  world = _World()
  turn = _Turn(_resolver(world))
  hop = CapabilityExecutionResolver.rebind_model_fallback(
    turn.primary, turn.snapshot, selection_provenance=_no_role(), history_fits=lambda _execution: True,
  )
  assert hop is not None and (hop.role, hop.execution.bind.model_key) == ("deep", SOL)



def test_a_chain_naming_a_model_twice_hops_once_per_entry_and_ends() -> None:
  world = _World()
  world.down = {"anthropic", "codex"}
  turn = _Turn(_resolver(world, chain=(FallbackEntry(model=SOL), FallbackEntry(model=SOL))))

  events = turn.run(timeout=10.0)

  assert [
    (hop["to"]["model_key"], hop["chain_index"]) for hop in _of_type(events, "model_fallback")
  ] == [(SOL, 0), (SOL, 1)]
  assert len(_of_type(events, "error")) == 1


def test_fallback_usage_names_the_table_that_priced_each_hop(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  world = _World()
  world.down = {"anthropic"}
  world.partial_outage = True
  resolver = _resolver(world)
  for model_key, version, price in (
    (OPUS, "2026-10-07", 1.0),
    (SOL, "2026-10-15", 2.0),
  ):
    execution = resolver.resolve(
      "session.driver",
      explicit_intent=ModelSelectionIntent(
        model_key=model_key, effort=None, source="explicit_user",
      ),
    )
    provider = execution.provider
    table = RateTable(
      version=version,
      source="fallback-test",
      providers={provider.name: {execution.bind.upstream_model: ModelRates(
        display_name=model_key,
        input_cost_per_mtok=price,
        output_cost_per_mtok=price,
        cache_read_cost_per_mtok=0.0,
        cache_write_cost_per_mtok=0.0,
      )}},
      _path=tmp_path / f"{provider.name}.json",
    )

    def model_info(
      model: str, *, provider: ModelProvider = provider, table: RateTable = table,
    ) -> ModelInfo:
      rates = table.lookup(provider.name, model)
      return ModelInfo(
        id=model, provider=provider.name, max_output_tokens=4096, context_window=200_000,
        input_cost_per_mtok=rates.input_cost_per_mtok,
        output_cost_per_mtok=rates.output_cost_per_mtok,
      )

    monkeypatch.setattr(provider, "_rate_table", table, raising=False)
    monkeypatch.setattr(provider, "get_model_info", model_info)
  turn = _Turn(resolver)
  turn.runner._rate_table_version = "2026-10-07"

  events = turn.run()

  assert len(_of_type(events, "model_fallback")) == 1
  primary_usage, fallback_usage = turn.usage[:-1], turn.usage[-1]
  assert len(primary_usage) == 1 + gateway_runner.STREAM_RETRY_MAX
  assert all(
    event.provider == "anthropic"
    and event.rate_table_version == "2026-10-07"
    and event.cost_usd == pytest.approx(10 / 1_000_000)
    for event in primary_usage
  )
  assert fallback_usage.provider == "codex"
  assert fallback_usage.rate_table_version == "2026-10-15"
  assert fallback_usage.cost_usd == pytest.approx(24 / 1_000_000)


def test_partial_usage_is_billed_when_the_client_disconnects_at_budget_exhaustion() -> None:
  world = _World()
  world.down = {"anthropic"}
  world.partial_outage = True
  turn = _Turn(_resolver(world))
  last_attempt = 1 + gateway_runner.STREAM_RETRY_MAX

  async def disconnect_on_last_attempt(count: int) -> None:
    if count == last_attempt:
      await turn.runner.on_disconnect()

  world.on_call = disconnect_on_last_attempt

  events = turn.run()

  assert len(world.calls) == last_attempt
  assert not _of_type(events, "model_fallback")
  assert len(turn.usage) == last_attempt
  assert all(event.input_tokens == 10 for event in turn.usage)


def test_a_captured_role_the_live_policy_no_longer_has_takes_todays_failure_path() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS]}
  captured = capture_selection_snapshot(_resolver(world).selection_policy)
  turn = _Turn(_resolver(world, tiered=False), snapshot=captured, explicit=OPUS)

  events = turn.run()

  assert not _of_type(events, "model_fallback")
  [error] = _of_type(events, "error")
  assert "usage limit reached" in error["error"]
  assert turn.runner.capability_execution is turn.primary


def test_frozen_tree_hops_along_its_capture_time_chain_after_a_policy_revision() -> None:
  world = _World()
  captured = capture_selection_snapshot(_resolver(world).selection_policy)
  revised = _resolver(world, chain=(FallbackEntry(model=SONNET),))
  turn = _Turn(revised, snapshot=captured)

  hop = CapabilityExecutionResolver.rebind_model_fallback(
    turn.primary, captured, selection_provenance=_no_role(), history_fits=lambda _execution: True,
  )

  assert hop is not None and hop.execution.bind.model_key == SOL


def test_entry_whose_whole_account_pool_is_parked_is_skipped_and_recorded() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS]}
  world.refusing_tokens = {_TOKEN_A, _TOKEN_B}
  for token in (_TOKEN_A, _TOKEN_B):
    _park(token, until=world.reset_at)
  turn = _Turn(_resolver(world, chain=(FallbackEntry(model=SONNET), FallbackEntry(model=SOL))))

  events = turn.run()

  [hop] = _of_type(events, "model_fallback")
  assert hop["skipped"] == [{"model_key": SONNET, "code": "credential_pool_parked"}]
  assert hop["to"]["model_key"] == SOL
  assert "claude-sonnet-test" not in [model for _, model, _ in world.calls]
  assert not _of_type(events, "error")


def test_a_chain_of_parked_entries_exhausts_and_fails_as_today() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS]}
  world.refusing_tokens = {_TOKEN_A, _TOKEN_B}
  for token in (_TOKEN_A, _TOKEN_B):
    _park(token, until=world.reset_at)
  turn = _Turn(_resolver(world, chain=(FallbackEntry(model=SONNET),)))

  events = turn.run()

  assert not _of_type(events, "model_fallback")
  assert {model for _, model, _ in world.calls} == {"claude-opus-test"}
  [error] = _of_type(events, "error")
  assert "usage limit reached" in error["error"]
  assert turn.runner.capability_execution is turn.primary


def test_entry_binds_the_account_whose_block_window_has_ended() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS]}
  world.refusing_tokens = {_TOKEN_A}
  _park(_TOKEN_A, until=world.reset_at)
  _park(_TOKEN_B, until=time.time() - 1)
  turn = _Turn(_resolver(world, chain=(FallbackEntry(model=SONNET),)))

  events = turn.run()

  [hop] = _of_type(events, "model_fallback")
  assert hop["to"]["model_key"] == SONNET
  assert hop["skipped"] == []
  assert world.calls[-1] == ("anthropic", "claude-sonnet-test", _TOKEN_B)
  member_b = resolve_anthropic_credentials().by_token(_TOKEN_B)
  assert member_b is not None
  live = turn.runner.capability_execution
  assert live.bind.model_key == SONNET
  assert live.bind.credential_ref == member_b.source_metadata()["generation_ref"]
  assert live.auth_config["auth_token"] == _TOKEN_B
  assert turn.dispatcher._capability_bind is live.bind
  assert not _of_type(events, "error")


def test_a_childs_model_fallback_reaches_the_parent_stream_tagged_with_its_sub_agent_id() -> None:
  world = _World()
  world.limited_models = {_UPSTREAM[OPUS]}
  forwarded: list[dict[str, Any]] = []
  # The parent chat log: the SSE subscriber and transcript read its entries.
  parent_log = EventLog(session_id="sess-parent", on_event=lambda event, _sid: forwarded.append(event))
  child_log = _build_child_event_log(
    parent_log=parent_log,
    event_log_cls=EventLog,
    sub_session_id="sub0:sess-parent",
    activity=_ChildActivity(),
    progress_cb=None,
    on_sub_event=None,
  )
  child = _Turn(_resolver(world), event_log=child_log)

  child.run()

  [hop] = [entry.event for entry in parent_log.entries if entry.event["type"] == "model_fallback"]
  assert hop["sub_agent_id"] == "sub0:sess-parent"
  assert (hop["from"]["model_key"], hop["to"]["model_key"]) == (OPUS, SOL)
  # The parent's append is its one delivery to the live forwarder.
  assert [event["type"] for event in forwarded].count("model_fallback") == 1


@pytest.mark.parametrize("kind", ["auth", "billing"])
def test_denied_pool_hops_with_credential_denied(kind: Literal["auth", "billing"]) -> None:
  world = _World()
  world.denied_models = {_UPSTREAM[OPUS]}
  world.denied_kind = kind
  world.denied_reset_at = world.reset_at
  turn = _Turn(_resolver(world))

  events = turn.run()

  assert world.calls == [
    ("anthropic", "claude-opus-test", _TOKEN_A),
    ("codex", "gpt-sol-test", _CODEX_KEY),
  ]
  assert not _of_type(events, "credential_refreshed")
  [hop] = _of_type(events, "model_fallback")
  assert hop["reason"] == "credential_denied"
  assert hop["reset_at"] == pytest.approx(world.reset_at)
  assert hop["to"]["model_key"] == SOL
  assert not _of_type(events, "error")


def test_denied_codex_and_openai_entries_walk_on_to_grok() -> None:
  world = _World()
  world.denied_models = {_UPSTREAM[OPUS], _UPSTREAM[SOL], _UPSTREAM[OPENAI]}
  world.denied_kind = "billing"
  world.denied_reset_at = world.reset_at
  turn = _Turn(_resolver(
    world,
    chain=(FallbackEntry(model=SOL), FallbackEntry(model=OPENAI), FallbackEntry(model=GROK)),
    providers_with_credentials=frozenset({"anthropic", "codex", "openai", "xai"}),
  ))

  events = turn.run()

  hops = _of_type(events, "model_fallback")
  assert [hop["to"]["model_key"] for hop in hops] == [SOL, OPENAI, GROK]
  assert [hop["reason"] for hop in hops] == ["credential_denied", "credential_denied", "credential_denied"]
  assert [hop["chain_index"] for hop in hops] == [0, 1, 2]
  assert hops[0]["reset_at"] == pytest.approx(world.reset_at)
  assert world.calls[-1] == ("xai", "grok-test", _GROK_KEY)
  assert not _of_type(events, "error")
  assert turn.runner.capability_execution.bind.model_key == GROK


def test_denied_chain_exhausts_and_fails_as_today() -> None:
  world = _World()
  world.denied_models = {_UPSTREAM[OPUS], _UPSTREAM[SOL], _UPSTREAM[OPENAI]}
  world.denied_kind = "auth"
  turn = _Turn(_resolver(
    world,
    chain=(FallbackEntry(model=SOL), FallbackEntry(model=OPENAI)),
    providers_with_credentials=frozenset({"anthropic", "codex", "openai"}),
  ))

  events = turn.run()

  hops = _of_type(events, "model_fallback")
  assert [hop["to"]["model_key"] for hop in hops] == [SOL, OPENAI]
  assert all(hop["reason"] == "credential_denied" for hop in hops)
  [error] = _of_type(events, "error")
  assert "access denied" in error["error"]
  assert turn.runner.capability_execution.bind.model_key == OPENAI


def test_reached_budget_stops_before_the_fallback_hop() -> None:
  world = _World()
  world.denied_models = {_UPSTREAM[OPUS]}
  world.denied_kind = "billing"
  world.denied_reset_at = world.reset_at
  turn = _Turn(_resolver(world))
  ledger = WorkStartSpendLedger(
    authorization_id=uuid4(),
    reserved_spend_microusd=1,
  )
  turn.runner._commercial_usage_producer = SimpleNamespace(
    _work_start=SimpleNamespace(spend_ledger=ledger),
  )
  turn.event_log.append({"type": "partial_result", "text": "kept"})

  async def _record_the_attempt(_n: int) -> None:
    ledger.record_usd("0.01")

  world.on_call = _record_the_attempt
  events = turn.run()

  assert ("codex", "gpt-sol-test", _CODEX_KEY) not in world.calls
  assert world.calls
  assert not _of_type(events, "model_fallback")
  assert any(
    event.get("type") == "partial_result" and event.get("text") == "kept"
    for event in events
  )
  completion = events[-1]
  assert completion["type"] == "stream_complete"
  assert completion["terminal_disposition"] == "interrupted"
  assert completion["reason"] == "commercial_budget_reached"


def test_credential_rotation_and_model_fallback_hops_do_not_compound_retry_backoff(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 3.0)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_BACKOFF", 2.0)
  retry_sleeps: list[float] = []

  async def record_sleep(delay: float, result: Any = None) -> Any:
    if delay != gateway_runner.STREAM_GUARD_POLL_INTERVAL:
      retry_sleeps.append(delay)
    return await asyncio.sleep(0, result)

  monkeypatch.setattr(
    gateway_runner, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": record_sleep}),
  )

  world = _World()
  world.siblings = {_TOKEN_A: _TOKEN_B}
  world.limited_models = {_UPSTREAM[OPUS], _UPSTREAM[SOL]}
  world.denied_models = {_UPSTREAM[OPENAI]}
  world.denied_kind = "billing"
  world.denied_reset_at = world.reset_at
  turn = _Turn(_resolver(
    world,
    chain=(FallbackEntry(model=SOL), FallbackEntry(model=OPENAI), FallbackEntry(model=GROK)),
    providers_with_credentials=frozenset({"anthropic", "codex", "openai", "xai"}),
  ))

  events = turn.run()

  assert world.calls == [
    ("anthropic", "claude-opus-test", _TOKEN_A),
    ("anthropic", "claude-opus-test", _TOKEN_B),
    ("codex", "gpt-sol-test", _CODEX_KEY),
    ("openai", "gpt-openai-test", _OPENAI_KEY),
    ("xai", "grok-test", _GROK_KEY),
  ]
  retry_events = _of_type(events, "stream_retry")
  assert [event["attempt"] for event in retry_events] == [0, 1, 2, 3]
  # The scripted sibling rotation does not change credential_ref, so it is a
  # revisit and spends one transient retry. Every model hop is immediate.
  assert [(event["error"], event["backoff_s"]) for event in retry_events] == [
    ("credential rotated after provider rate_limit failure", 3.0),
    ("model fallback after provider rate_limit failure", 0.0),
    ("model fallback after provider rate_limit failure", 0.0),
    ("model fallback after provider billing failure", 0.0),
  ]
  # The 3s sleep is the same-credential revisit. Model hops do not sleep.
  assert retry_sleeps == [3.0, 0.0, 0.0, 0.0]
  assert not _of_type(events, "error")
  assert turn.runner.capability_execution.bind.model_key == GROK
