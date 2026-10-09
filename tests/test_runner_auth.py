from gateway_test_support.autonomous_launch_test_support import TEST_CAPABILITY_BIND
import asyncio
import io
import json
import sys
import time
from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import httpx2
import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway import (  # noqa: E402
  AgentRunner,
  EventLog,
  McpClientManager,
  ModelInfo,
  ModelProvider,
  ToolDispatcher,
)
import agent_gateway.runner as gateway_runner  # noqa: E402
import agent_gateway.providers.anthropic as anthropic_provider_module  # noqa: E402
from agent_gateway.auth import ProviderCredentialFailure  # noqa: E402
from agent_gateway.autonomous_credential_handoff import (  # noqa: E402
  encode_autonomous_credential_handoff,
  read_autonomous_credential_handoff,
)
from agent_gateway.autonomous_runner_start import (  # noqa: E402
  _positive_autonomous_child_env,
)
from model_authority.binding import CredentialHandle
from agent_gateway.capability_execution import MaterializedCredential  # noqa: E402
from agent_gateway.providers import StreamEvent  # noqa: E402
from agent_gateway.providers.anthropic import AnthropicProvider  # noqa: E402
from agent_gateway.providers.anthropic_oauth import (  # noqa: E402
  ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES,
  ANTHROPIC_USER_SCOPE_FIELD,
  AnthropicCredentialPool,
  AnthropicOAuthRecord,
  resolve_anthropic_auth_store_path,
  resolve_anthropic_credentials,
  rotate_anthropic_credential_source,
  select_anthropic_credential_source,
  upsert_anthropic_oauth_record,
)
from agent_gateway.runner_auth import merge_refreshed_auth_config  # noqa: E402
from agent_gateway.runner_fork_agents import _bind_learning_fork_execution
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_bound_capability_execution,
)


_TOKEN_A = "sk-ant-oat01-pool-primary-credential-000000000000"
_TOKEN_B = "sk-ant-oat01-pool-sibling-credential-000000000000"


class _UsageLimitedError(RuntimeError):
  """The shape `api/credentials.py` hands the runner for a 429.

  Same message and same projected reset fields as the product sanitizer, so the
  classification and the block window under test are the real ones.
  """

  def __init__(self, *, reset_at: float) -> None:
    super().__init__("Anthropic rate limit exceeded")
    self.retryable = True
    self.status_code = 429
    self.rate_limit_5h_status = "rejected"
    self.rate_limit_5h_reset = (
      datetime.fromtimestamp(reset_at, timezone.utc).isoformat().replace("+00:00", "Z")
    )
    self.retry_after = str(max(0, int(reset_at - time.time())))


class _PooledAnthropicProvider(AnthropicProvider):
  """The real credential pool with the Anthropic network seam stubbed out."""

  def __init__(self, *, limited_tokens: tuple[str, ...], reset_at: float) -> None:
    super().__init__()
    self._limited_tokens = limited_tokens
    self._reset_at = reset_at
    self.streamed_tokens: list[str] = []

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    _ = timeout
    return {"auth_token": str(config.get("auth_token") or "")}

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    _ = client, timeout

  def get_model_info(self, model: str) -> ModelInfo:
    from gateway_test_support.model_defaults import compat_for_profile

    return ModelInfo(
      id=model, provider=self.name, max_output_tokens=4096,
      compat=compat_for_profile("messages.standard"),
    )

  def build_request_params(
    self,
    *,
    model: str,
    messages: list[dict[str, Any]],
    system_prompt: str | list[tuple[str, bool]] | None,
    tools: list[dict[str, Any]],
    max_tokens: int,
    **kwargs: Any,
  ) -> dict[str, Any]:
    _ = model, messages, system_prompt, tools, max_tokens, kwargs
    return {}

  async def stream(self, client: Any, params: dict[str, Any]):
    _ = params
    token = str(client["auth_token"])
    self.streamed_tokens.append(token)
    if token in self._limited_tokens:
      raise _UsageLimitedError(reset_at=self._reset_at)
    yield StreamEvent(type="message_start", input_tokens=10)
    yield StreamEvent(type="text_delta", text="ok")
    yield StreamEvent(type="text_end", raw_block={"type": "text", "text": "ok"})
    yield StreamEvent(type="usage_update", output_tokens=2)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


class _SanitizingPooledProvider(_PooledAnthropicProvider):
  """Adds the retryability seam `api/credentials.py` wraps this provider with.

  `SanitizingAnthropicProvider.is_retryable_error` reads the sanitized error's
  own `retryable` flag before falling back to SDK-shaped classification, so a
  sanitized 429 really does spend the gateway's stream-retry budget in product.
  """

  def is_retryable_error(self, exc: Exception) -> bool:
    retryable = getattr(exc, "retryable", None)
    if isinstance(retryable, bool):
      return retryable
    return super().is_retryable_error(exc)


class _NullMcpClient(McpClientManager):
  def __init__(self) -> None:
    super().__init__(config_path=None)


class _StubProvider(ModelProvider):
  name = "stub"

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=4096,
      supports_thinking=True,
    )


def _run(coro):
  return asyncio.run(coro)


def _make_dispatcher(event_log: EventLog | None = None, *, capability_bind = TEST_CAPABILITY_BIND) -> ToolDispatcher:
  return ToolDispatcher(
    capability_bind=capability_bind,
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-auth",
  )


def _make_runner() -> AgentRunner:
  event_log = EventLog()
  provider = _StubProvider()
  capability_execution = stub_bound_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
      auth_config={"api_key": "old", "max_tokens": 512},
    )
  return AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log, capability_bind=capability_execution.bind),
    session_id="sess-auth",
    capability_execution=capability_execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )


def test_runner_auth_wrappers_resolve_parent_module_helpers(monkeypatch: Any) -> None:
  runner = _make_runner()
  original_execution = runner.capability_execution
  config = dict(original_execution.auth_config)
  config["api_key"] = "request-old"

  monkeypatch.setattr(
    gateway_runner,
    "_merge_refreshed_auth_config",
    lambda original, refreshed: {
      **original,
      "api_key": f"{original['api_key']}->{refreshed['api_key']}",
    },
  )

  AgentRunner._apply_refreshed_auth_config(runner, config, {"api_key": "new"})

  assert config["api_key"] == "old->new"
  assert runner._auth_config["api_key"] == "old->new"
  assert runner.capability_execution is not original_execution
  assert runner.capability_execution.bind is original_execution.bind
  assert runner.capability_execution.provider is original_execution.provider
  assert runner._secret_boundary.sanitize(
    ["old", "old->new"],
    sink="test",
  ) == ["<redacted-secret>", "<redacted-secret>"]


def test_merge_refreshed_auth_config_preserves_runtime_controls() -> None:
  merged = merge_refreshed_auth_config(
    {
      "api_key": "old",
      "auth_mode": "oauth",
      "auth_token": "old-token",
      "provider": "stub",
      "max_tokens": 4096,
      "billing_mode": "metered",
      "rate_table_version": "2026-04-08",
    },
    {
      "api_key": "new",
      "auth_mode": "oauth",
      "auth_token": "new-token",
      "billing_mode": "byok",
      "rate_table_version": "unknown",
    },
  )

  assert merged == {
    "api_key": "new",
    "auth_mode": "oauth",
    "auth_token": "new-token",
    "provider": "stub",
    "max_tokens": 4096,
    "billing_mode": "metered",
    "rate_table_version": "2026-04-08",
  }


def test_merge_refreshed_auth_config_rejects_selection_material() -> None:
  config = {
    "api_key": "old",
    "auth_mode": "api",
    "provider": "stub",
    "max_tokens": 4096,
  }

  with pytest.raises(ValueError, match="must not contain model selection"):
    merge_refreshed_auth_config(config, {"model": "other-model"})
  with pytest.raises(ValueError, match="must not contain model selection"):
    merge_refreshed_auth_config(config, {"thinking": False})
  with pytest.raises(ValueError, match="must not contain model selection"):
    merge_refreshed_auth_config(config, {"execution_transport": "native"})


@pytest.mark.parametrize("has_generation_metadata", [True, False])
def test_usage_limited_credential_rotates_to_sibling_and_keeps_bound_model(
  monkeypatch: Any, tmp_path: Path, has_generation_metadata: bool,
) -> None:
  reset_at = time.time() + 4 * 3600
  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_TOKEN_A, _TOKEN_B]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)

  provider = _PooledAnthropicProvider(limited_tokens=(_TOKEN_A,), reset_at=reset_at)
  sources = resolve_anthropic_credentials()
  member_a = sources.by_token(_TOKEN_A)
  member_b = sources.by_token(_TOKEN_B)
  assert member_a is not None and member_b is not None
  if not has_generation_metadata:
    monkeypatch.setattr(
      provider, "next_credential",
      lambda config, failure: {"auth_token": _TOKEN_B},
    )
  event_log = EventLog()
  capability_execution = stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      credential_ref=member_a.source_metadata()["generation_ref"],
      auth_config={
        "auth_mode": "oauth",
        "auth_token": _TOKEN_A,
        "credential_source": member_a.source_metadata(),
        "max_tokens": 512,
      },
    )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log, capability_bind=capability_execution.bind),
    session_id="sess-pool",
    capability_execution=capability_execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  bound_model_key = runner.capability_execution.bind.model_key

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  event_types = [entry.event.get("type") for entry in event_log.entries]
  if has_generation_metadata:
    assert provider.streamed_tokens == [_TOKEN_A, _TOKEN_B]
    assert "credential_refreshed" in event_types
    assert "error" not in event_types
    assert runner.capability_execution.bind.credential_ref == member_b.source_metadata()["generation_ref"]
    assert runner.capability_execution.auth_config["auth_token"] == _TOKEN_B
    assert runner._auth_config["auth_token"] == _TOKEN_B
    assert pool.blocked_until(_TOKEN_A) == pytest.approx(reset_at, abs=2.0)
    assert pool.blocked_until(_TOKEN_B) == 0.0
  else:
    assert provider.streamed_tokens == [_TOKEN_A]
    assert "credential_refreshed" not in event_types
    assert "error" in event_types
    assert runner.capability_execution is capability_execution
    assert runner._auth_config["auth_token"] == _TOKEN_A
  assert runner.capability_execution.bind.model_key == bound_model_key
  dispatcher = runner._dispatcher
  assert isinstance(dispatcher, ToolDispatcher)
  assert dispatcher._capability_bind is runner.capability_execution.bind
  # Child admission's parent-binding reader sees the current execution's bind.
  assert runner.capability_execution.bind.credential_ref == (
    member_b if has_generation_metadata else member_a
  ).source_metadata()["generation_ref"]


def _make_rotation_budget_runner(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  provider: _PooledAnthropicProvider,
  tokens: tuple[str, ...],
) -> tuple[AgentRunner, EventLog]:
  monkeypatch.setattr(
    anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", AnthropicCredentialPool(),
  )
  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", tokens[0])
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps(tokens))
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_MAX", 3)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)
  member = resolve_anthropic_credentials().by_token(tokens[0])
  assert member is not None
  event_log = EventLog()
  capability_execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-pool-test",
    effort="none",
    credential_ref=member.source_metadata()["generation_ref"],
    auth_config={
      "auth_mode": "oauth",
      "auth_token": tokens[0],
      "credential_source": member.source_metadata(),
      "max_tokens": 512,
    },
  )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log, capability_bind=capability_execution.bind),
    session_id="sess-rotation-budget",
    capability_execution=capability_execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  return runner, event_log


def test_five_credential_generations_reach_the_usable_member_beyond_retry_budget(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  tokens = (
    _TOKEN_A,
    _TOKEN_B,
    "sk-ant-oat01-pool-third-credential-000000000000",
    "sk-ant-oat01-pool-fourth-credential-000000000000",
    "sk-ant-oat01-pool-fifth-credential-000000000000",
  )
  provider = _PooledAnthropicProvider(
    limited_tokens=tokens[:-1], reset_at=time.time() + 4 * 3600,
  )
  runner, event_log = _make_rotation_budget_runner(monkeypatch, tmp_path, provider, tokens)
  member_five = resolve_anthropic_credentials().by_token(tokens[-1])
  assert member_five is not None
  retry_sleeps: list[float] = []
  retry_delay = 0.001
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", retry_delay)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_BACKOFF", 2.0)

  async def record_sleep(delay: float, result: Any = None) -> Any:
    if delay != gateway_runner.STREAM_GUARD_POLL_INTERVAL:
      retry_sleeps.append(delay)
    return await asyncio.sleep(0, result)

  monkeypatch.setattr(
    gateway_runner, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": record_sleep}),
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert provider.streamed_tokens == list(tokens)
  assert runner.capability_execution.bind.credential_ref == member_five.source_metadata()["generation_ref"]
  assert runner.capability_execution.auth_config["auth_token"] == tokens[-1]
  events = [entry.event for entry in event_log.entries]
  assert len([event for event in events if event["type"] == "credential_refreshed"]) == 4
  retry_events = [event for event in events if event["type"] == "stream_retry"]
  assert [event["attempt"] for event in retry_events] == [0, 1, 2, 3]
  assert retry_sleeps == pytest.approx([0.001, 0.002, 0.004, 0.008])
  assert [event["backoff_s"] for event in retry_events] == retry_sleeps
  assert not any(event["type"] == "error" for event in events)
  [completion] = [event for event in events if event["type"] == "stream_complete"]
  assert completion["terminal_disposition"] == "completed"


def test_free_rotation_preserves_the_full_transient_retry_budget(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  class _TransientSiblingProvider(_PooledAnthropicProvider):
    async def stream(self, client: Any, params: dict[str, Any]):
      if client["auth_token"] == _TOKEN_B:
        self.streamed_tokens.append(_TOKEN_B)
        raise RuntimeError("transient transport error")
      async for event in super().stream(client, params):
        yield event

    def is_retryable_error(self, exc: Exception) -> bool:
      return str(exc) == "transient transport error" or super().is_retryable_error(exc)

  provider = _TransientSiblingProvider(
    limited_tokens=(_TOKEN_A,), reset_at=time.time() + 4 * 3600,
  )
  runner, event_log = _make_rotation_budget_runner(
    monkeypatch, tmp_path, provider, (_TOKEN_A, _TOKEN_B),
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert provider.streamed_tokens == [_TOKEN_A] + [_TOKEN_B] * (1 + gateway_runner.STREAM_RETRY_MAX)
  events = [entry.event for entry in event_log.entries]
  assert len([event for event in events if event["type"] == "credential_refreshed"]) == 1
  retry_events = [event for event in events if event["type"] == "stream_retry"]
  assert [event["attempt"] for event in retry_events] == [0, 1, 2, 3]
  assert sum("transient transport error" in event["error"] for event in retry_events) == gateway_runner.STREAM_RETRY_MAX
  [error] = [event for event in events if event["type"] == "error"]
  assert "transient transport error" in error["error"]


@pytest.mark.parametrize("cycle_generations", [False, True])
def test_rotations_to_tried_generations_consume_transient_retry_budget(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cycle_generations: bool,
) -> None:
  provider = _PooledAnthropicProvider(
    limited_tokens=(_TOKEN_A, _TOKEN_B), reset_at=time.time() + 4 * 3600,
  )
  runner, event_log = _make_rotation_budget_runner(
    monkeypatch, tmp_path, provider, (_TOKEN_A, _TOKEN_B),
  )
  sources = resolve_anthropic_credentials()

  def next_credential(
    config: Any, failure: ProviderCredentialFailure,
  ) -> dict[str, Any]:
    _ = failure
    token = (
      (_TOKEN_B if config["auth_token"] == _TOKEN_A else _TOKEN_A)
      if cycle_generations else config["auth_token"]
    )
    member = sources.by_token(token)
    assert member is not None
    return {"auth_token": token, "credential_source": member.source_metadata()}

  monkeypatch.setattr(provider, "next_credential", next_credential)

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  expected_tokens = (
    [_TOKEN_A, _TOKEN_B, _TOKEN_A, _TOKEN_B, _TOKEN_A]
    if cycle_generations else [_TOKEN_A] * (1 + gateway_runner.STREAM_RETRY_MAX)
  )
  assert provider.streamed_tokens == expected_tokens
  events = [entry.event for entry in event_log.entries]
  assert len([event for event in events if event["type"] == "credential_refreshed"]) == len(expected_tokens)
  retry_events = [event for event in events if event["type"] == "stream_retry"]
  assert [event["attempt"] for event in retry_events] == list(range(len(expected_tokens)))
  assert [event["backoff_s"] for event in retry_events] == [0.0] * (len(expected_tokens) - 1) + [None]
  assert len([event for event in events if event["type"] == "error"]) == 1


def test_cancellation_during_a_free_rotation_retry_cancels_the_turn(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
  provider = _PooledAnthropicProvider(
    limited_tokens=(_TOKEN_A,), reset_at=time.time() + 4 * 3600,
  )
  runner, event_log = _make_rotation_budget_runner(
    monkeypatch, tmp_path, provider, (_TOKEN_A, _TOKEN_B),
  )
  retry_started = asyncio.Event()
  retry_delay = 45.0
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", retry_delay)

  async def record_sleep(delay: float, result: Any = None) -> Any:
    if delay == retry_delay:
      retry_started.set()
    return await asyncio.sleep(delay, result)

  monkeypatch.setattr(
    gateway_runner, "asyncio", SimpleNamespace(**{**vars(asyncio), "sleep": record_sleep}),
  )

  async def cancel_retry() -> None:
    task = asyncio.create_task(runner.run(messages=[{"role": "user", "content": "hello"}]))
    await asyncio.wait_for(retry_started.wait(), timeout=5.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task
    assert task.cancelled()

  _run(cancel_retry())

  assert provider.streamed_tokens == [_TOKEN_A]
  member_b = resolve_anthropic_credentials().by_token(_TOKEN_B)
  assert member_b is not None
  assert runner.capability_execution.bind.credential_ref == member_b.source_metadata()["generation_ref"]
  events = [entry.event for entry in event_log.entries]
  assert len([event for event in events if event["type"] == "credential_refreshed"]) == 1
  assert not any(event["type"] == "error" for event in events)


def test_learning_fork_rotates_runner_and_dispatcher_to_sibling(
  monkeypatch: Any, tmp_path: Path,
) -> None:
  reset_at = time.time() + 4 * 3600
  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_TOKEN_A, _TOKEN_B]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)
  provider = _PooledAnthropicProvider(limited_tokens=(_TOKEN_A,), reset_at=reset_at)
  sources = resolve_anthropic_credentials()
  member_a = sources.by_token(_TOKEN_A)
  member_b = sources.by_token(_TOKEN_B)
  assert member_a is not None and member_b is not None
  parent_execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-pool-test",
    effort="none",
    credential_ref=member_a.source_metadata()["generation_ref"],
    auth_config={
      "auth_mode": "oauth",
      "auth_token": _TOKEN_A,
      "credential_source": member_a.source_metadata(),
      "max_tokens": 512,
    },
  )
  resolver = parent_execution.resolver
  assert resolver is not None
  entry = parent_execution.registry.models[parent_execution.bind.model_key]
  registry = replace(parent_execution.registry, models={
    **parent_execution.registry.models,
    entry.key: replace(entry, capabilities={
      **entry.capabilities, "node.fork": "internal",
    }),
  })
  resolver = replace(
    resolver,
    registry=registry,
    auth_context=replace(resolver.auth_context, entitled_capabilities=frozenset({
      *resolver.auth_context.entitled_capabilities, "node.fork",
    })),
  )
  parent_execution = replace(parent_execution, registry=registry, resolver=resolver)
  fork_execution = _bind_learning_fork_execution(parent_execution)
  parent_dispatcher = _make_dispatcher(capability_bind=parent_execution.bind)
  event_log = EventLog()
  fork_dispatcher = parent_dispatcher.fork_child(
    event_log=event_log,
    session_id="sess-learning-fork",
    capability_bind=fork_execution.bind,
    spawning_turn=parent_dispatcher.capture_spawning_turn(),
  )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=fork_dispatcher,
    session_id="sess-learning-fork",
    capability_execution=fork_execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "remember this"}]))

  assert provider.streamed_tokens == [_TOKEN_A, _TOKEN_B]
  event_types = [entry.event.get("type") for entry in event_log.entries]
  assert "credential_refreshed" in event_types
  assert "error" not in event_types
  assert runner.capability_execution.bind.capability_id == "node.fork"
  assert runner.capability_execution.bind.credential_ref == (
    member_b.source_metadata()["generation_ref"]
  )
  assert runner.capability_execution.auth_config["auth_token"] == _TOKEN_B
  assert fork_dispatcher._capability_bind is runner.capability_execution.bind
  assert parent_dispatcher._capability_bind is parent_execution.bind
  assert pool.blocked_until(_TOKEN_A) == pytest.approx(reset_at, abs=2.0)
  assert pool.blocked_until(_TOKEN_B) == 0.0


def test_single_credential_pool_is_never_withheld_from_a_run(
  monkeypatch: Any, tmp_path: Path,
) -> None:
  reset_at = time.time() + 3600
  pool = AnthropicCredentialPool()
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKENS", raising=False)
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "oauth.json"))
  sources = resolve_anthropic_credentials()

  assert sources.tokens == (_TOKEN_A,)
  selected = select_anthropic_credential_source(sources, pool=pool)
  assert selected is not None and selected.token == _TOKEN_A

  # Nothing to swap in, and the only credential is still handed out: a pool of
  # one leaves the turn to the gateway's own bounded stream retry, which ends it
  # with the sanitized rate-limit error (see the pool-of-one turn test below).
  assert rotate_anthropic_credential_source(
    sources, current=_TOKEN_A, until=reset_at, pool=pool
  ) is None
  selected = select_anthropic_credential_source(sources, pool=pool)
  assert selected is not None and selected.token == _TOKEN_A


def test_usage_limited_enrolled_account_rotates_to_the_next_enrolled_account(
  monkeypatch: Any, tmp_path: Path
) -> None:
  """An account enrolled by this product's login rotates like any sibling.

  Both credentials come from the store rather than the environment, and the
  parked one is parked by account identity, so the rejected account stays out
  of rotation even after its access token is refreshed.
  """

  store = tmp_path / "oauth.json"
  for identity in ("primary@example.com", "sibling@example.com"):
    upsert_anthropic_oauth_record(
      store,
      AnthropicOAuthRecord(
        identity=identity,
        access_token=f"sk-ant-oat01-{identity}",
        refresh_token=f"sk-ant-ort01-{identity}",
        expires_at=time.time() + 3600,
      ),
    )
  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKENS", raising=False)
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(store))

  reset_at = time.time() + 4 * 3600
  failure = ProviderCredentialFailure(
    provider="anthropic",
    kind="rate_limit",
    status_code=429,
    reset_at=reset_at,
  )
  provider = AnthropicProvider()
  rotated = provider.next_credential(
    {"auth_mode": "oauth", "auth_token": "sk-ant-oat01-primary@example.com"},
    failure,
  )

  sibling = resolve_anthropic_credentials().by_token("sk-ant-oat01-sibling@example.com")
  assert sibling is not None
  assert rotated == {
    "auth_token": "sk-ant-oat01-sibling@example.com",
    "credential_source": sibling.source_metadata(),
  }
  assert pool.blocked_until("primary@example.com") == pytest.approx(reset_at, abs=2.0)
  assert pool.blocked_until("sibling@example.com") == 0.0
  # Every account limited: no sibling is handed back, so the run cannot circle.
  assert provider.next_credential(
    {"auth_mode": "oauth", "auth_token": "sk-ant-oat01-sibling@example.com"},
    failure,
  ) is None


def test_pool_of_one_ends_the_turn_with_the_reported_reset_window(
  monkeypatch: Any,
) -> None:
  """A 20-hour reset is reported to the analyst, not waited out.

  With no sibling to rotate to, the turn spends the gateway's bounded stream
  retries and then fails with the sanitized rate-limit error and the window the
  limiter reported — never a stall the caller reads as a watchdog timeout.
  """
  reset_at = time.time() + 73443
  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_TOKEN_A]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)

  provider = _SanitizingPooledProvider(limited_tokens=(_TOKEN_A,), reset_at=reset_at)
  event_log = EventLog()
  capability_execution = stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config={
        "auth_mode": "oauth",
        "auth_token": _TOKEN_A,
        "max_tokens": 512,
      },
    )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log, capability_bind=capability_execution.bind),
    session_id="sess-pool-of-one",
    capability_execution=capability_execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  # Every attempt is the same credential and the count is the gateway's own
  # retry budget: nothing in this path is derived from the reported reset.
  assert provider.streamed_tokens == [_TOKEN_A] * (1 + gateway_runner.STREAM_RETRY_MAX)
  errors = [
    entry.event for entry in event_log.entries if entry.event.get("type") == "error"
  ]
  assert errors and "rate limit" in errors[0].get("error", "")
  assert errors[0].get("status_code") == 429
  assert int(errors[0]["retry_after"]) == pytest.approx(73443, abs=5)
  assert errors[0].get("rate_limit_5h_status") == "rejected"


def test_every_credential_limited_stops_rotating_instead_of_circling(
  monkeypatch: Any,
) -> None:
  reset_at = time.time() + 4 * 3600
  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_TOKEN_A, _TOKEN_B]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)

  provider = _PooledAnthropicProvider(
    limited_tokens=(_TOKEN_A, _TOKEN_B),
    reset_at=reset_at,
  )
  event_log = EventLog()
  capability_execution = stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config={
        "auth_mode": "oauth",
        "auth_token": _TOKEN_A,
        "max_tokens": 512,
      },
    )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log, capability_bind=capability_execution.bind),
    session_id="sess-pool-exhausted",
    capability_execution=capability_execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  # Each credential is tried once and the turn then fails as it does today; a
  # blocked sibling is never handed back, so the run cannot rotate in circles.
  assert provider.streamed_tokens == [_TOKEN_A, _TOKEN_B]
  errors = [
    entry.event for entry in event_log.entries if entry.event.get("type") == "error"
  ]
  assert errors and "rate limit" in errors[0].get("error", "")


def test_autonomous_child_rotates_on_the_pool_its_projection_carries(
  monkeypatch: Any,
  tmp_path: Path,
) -> None:
  """An autonomous child rotates off a 429 exactly like a chat turn.

  The parent binds one credential per run and hands it to the child over the
  stdin handoff; which pool that credential belongs to reaches the child only
  through the launch environment projection. A child projected with no pool saw
  one credential and every autonomous run died on the first 429 of whichever
  account the parent happened to bind, while the same gateway process rotated
  on chat (docs/qa/autonomous-child-credential-pool-of-one-2026-09-17.md).
  """

  reset_at = time.time() + 4 * 3600
  gateway_environ = {
    "PATH": "/usr/bin",
    "HOME": str(tmp_path),
    "ANTHROPIC_AUTH_TOKEN": _TOKEN_A,
    "ANTHROPIC_AUTH_TOKENS": json.dumps([_TOKEN_A, _TOKEN_B]),
    "ANTHROPIC_AUTH_STORE_PATH": str(tmp_path / "oauth.json"),
  }
  child_environ = _positive_autonomous_child_env(
    gateway_environ,
    provider="anthropic",
    profile="analyst",
    deliver=False,
  )

  # The parent selects out of the pool it resolves, as it does for every run.
  parent_pool = AnthropicCredentialPool()
  selected = select_anthropic_credential_source(
    resolve_anthropic_credentials(environ=gateway_environ),
    pool=parent_pool,
  )
  assert selected is not None and selected.token == _TOKEN_A
  bound_token = selected.token

  handle = CredentialHandle(
    handle_id="handle-autonomous-child",
    provider="anthropic",
    principal="service",
    tenant_id="hank-test",
    actor_id=None,
  )
  child_credential = read_autonomous_credential_handoff(
    expected_handle_id=handle.handle_id,
    expected_provider=handle.provider,
    expected_principal=handle.principal,
    expected_tenant_id=handle.tenant_id,
    expected_actor_id=handle.actor_id,
    stream=io.BytesIO(encode_autonomous_credential_handoff(
      MaterializedCredential(
        handle=handle,
        auth_config={
          "provider": "anthropic",
          "auth_mode": "oauth",
          "api_key": "",
          "auth_token": bound_token,
          "max_tokens": 512,
        },
      ),
    )),
  )

  # The child process resolves credentials out of exactly what was projected
  # onto it, and it starts with no parked credential of its own.
  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    projected = child_environ.get(name)
    if projected is None:
      monkeypatch.delenv(name, raising=False)
    else:
      monkeypatch.setenv(name, projected)
  child_pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", child_pool)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)

  provider = _SanitizingPooledProvider(limited_tokens=(_TOKEN_A,), reset_at=reset_at)
  event_log = EventLog()
  capability_execution = stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config=dict(child_credential.auth_config),
    )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log, capability_bind=capability_execution.bind),
    session_id="sess-autonomous-child",
    capability_execution=capability_execution,
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  bound_model_key = runner.capability_execution.bind.model_key

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert provider.streamed_tokens == [_TOKEN_A, _TOKEN_B]
  event_types = [entry.event.get("type") for entry in event_log.entries]
  assert "error" not in event_types
  assert runner._auth_config["auth_token"] == _TOKEN_B
  assert runner.capability_execution.bind.model_key == bound_model_key
  assert child_pool.blocked_until(_TOKEN_A) == pytest.approx(reset_at, abs=2.0)


def _enroll(store_path: Path, identity: str) -> str:
  upsert_anthropic_oauth_record(
    store_path,
    AnthropicOAuthRecord(
      identity=identity,
      access_token=f"sk-ant-oat01-{identity}",
      refresh_token=f"sk-ant-ort01-{identity}",
      expires_at=time.time() + 3600,
    ),
  )
  return f"sk-ant-oat01-{identity}"


def test_a_users_turn_rotates_inside_that_users_pool_and_never_onto_another(
  monkeypatch: Any, tmp_path: Path
) -> None:
  """Rotation stays inside the pool bound to the turn's user identity.

  Henry's personal subscription accounts are enrolled under his own
  `risk_user_id`; a 429 on one of them must rotate to his sibling and then to
  the org credential, and a turn for any other user must never be handed one of
  them — that is the whole reason the pool is user-scoped.
  """

  environ = {
    "USER_DATA_DIR": str(tmp_path),
    "ANTHROPIC_AUTH_TOKEN": _TOKEN_A,
  }
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKENS", raising=False)
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.delenv("ANTHROPIC_AUTH_STORE_PATH", raising=False)
  mine = resolve_anthropic_auth_store_path(environ=environ, user_scope=1)
  primary = _enroll(mine, "henry-primary@example.com")
  sibling = _enroll(mine, "henry-sibling@example.com")
  _enroll(resolve_anthropic_auth_store_path(environ=environ, user_scope=2), "other@example.com")

  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  reset_at = time.time() + 4 * 3600
  failure = ProviderCredentialFailure(
    provider="anthropic",
    kind="rate_limit",
    status_code=429,
    reset_at=reset_at,
  )
  provider = AnthropicProvider()
  mine_bound = {
    "auth_mode": "oauth",
    "auth_token": primary,
    ANTHROPIC_USER_SCOPE_FIELD: "1",
  }

  selected = select_anthropic_credential_source(
    resolve_anthropic_credentials(environ=environ, user_scope=1), pool=pool
  )
  assert selected is not None and selected.token == primary
  def source_of(token: str, user_scope: int) -> dict[str, str]:
    credential = resolve_anthropic_credentials(
      environ=environ, user_scope=user_scope,
    ).by_token(token)
    assert credential is not None
    return credential.source_metadata()

  assert provider.next_credential(mine_bound, failure) == {
    "auth_token": sibling, "credential_source": source_of(sibling, 1),
  }
  assert provider.next_credential(
    {**mine_bound, "auth_token": sibling}, failure
  ) == {"auth_token": _TOKEN_A, "credential_source": source_of(_TOKEN_A, 1)}
  assert pool.blocked_until("henry-primary@example.com") == pytest.approx(reset_at, abs=2.0)

  # Another user's turn is served the org credential and nothing of user 1's,
  # even though user 1's accounts are the ones no limiter is holding.
  theirs = resolve_anthropic_credentials(environ=environ, user_scope=2)
  assert theirs.tokens == ("sk-ant-oat01-other@example.com", _TOKEN_A)
  assert primary not in theirs.tokens and sibling not in theirs.tokens
  assert provider.next_credential(
    {
      "auth_mode": "oauth",
      "auth_token": "sk-ant-oat01-other@example.com",
      ANTHROPIC_USER_SCOPE_FIELD: "2",
    },
    failure,
  ) == {"auth_token": _TOKEN_A, "credential_source": source_of(_TOKEN_A, 2)}


def test_autonomous_child_of_a_user_scoped_run_inherits_that_users_pool(
  monkeypatch: Any, tmp_path: Path
) -> None:
  """A child rotates inside the pool of the user whose run launched it.

  The child learns the pool from two things and nothing else: the launch
  environment projection (which carries the per-user data root) and the
  credential material on the stdin handoff (which names the user whose pool the
  bound credential came from). A child that lost the scope would rotate onto
  the org credential while its parent's user still had unspent accounts.
  """

  reset_at = time.time() + 4 * 3600
  gateway_environ = {
    "PATH": "/usr/bin",
    "HOME": str(tmp_path),
    "USER_DATA_DIR": str(tmp_path / "state"),
    "ANTHROPIC_AUTH_TOKEN": _TOKEN_A,
  }
  mine = resolve_anthropic_auth_store_path(environ=gateway_environ, user_scope=3)
  primary = _enroll(mine, "child-primary@example.com")
  sibling = _enroll(mine, "child-sibling@example.com")
  child_environ = _positive_autonomous_child_env(
    gateway_environ,
    provider="anthropic",
    profile="analyst",
    deliver=False,
  )

  parent_pool = AnthropicCredentialPool()
  selected = select_anthropic_credential_source(
    resolve_anthropic_credentials(environ=gateway_environ, user_scope=3),
    pool=parent_pool,
  )
  assert selected is not None and selected.token == primary
  bound_token = selected.token

  handle = CredentialHandle(
    handle_id="handle-user-scoped-child",
    provider="anthropic",
    principal="service",
    tenant_id="hank-test",
    actor_id=None,
  )
  child_credential = read_autonomous_credential_handoff(
    expected_handle_id=handle.handle_id,
    expected_provider=handle.provider,
    expected_principal=handle.principal,
    expected_tenant_id=handle.tenant_id,
    expected_actor_id=handle.actor_id,
    stream=io.BytesIO(encode_autonomous_credential_handoff(
      MaterializedCredential(
        handle=handle,
        auth_config={
          "provider": "anthropic",
          "auth_mode": "oauth",
          "api_key": "",
          "auth_token": bound_token,
          "max_tokens": 512,
          ANTHROPIC_USER_SCOPE_FIELD: "3",
        },
      ),
    )),
  )

  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    projected = child_environ.get(name)
    if projected is None:
      monkeypatch.delenv(name, raising=False)
    else:
      monkeypatch.setenv(name, projected)
  child_pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", child_pool)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)

  provider = _SanitizingPooledProvider(limited_tokens=(primary,), reset_at=reset_at)
  event_log = EventLog()
  capability_execution = stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config=dict(child_credential.auth_config),
    )
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log, capability_bind=capability_execution.bind),
    session_id="sess-user-scoped-child",
    capability_execution=capability_execution,
    user_id="3",
    billing_mode="byok",
    rate_table_version="unknown",
  )

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  assert provider.streamed_tokens == [primary, sibling]
  assert "error" not in [entry.event.get("type") for entry in event_log.entries]
  assert runner._auth_config["auth_token"] == sibling
  assert child_pool.blocked_until("child-primary@example.com") == pytest.approx(
    reset_at, abs=2.0
  )


def test_rotated_credential_source_matches_prepared_provider_authentication(
  monkeypatch: Any, tmp_path: Path,
) -> None:
  for name in ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES:
    monkeypatch.delenv(name, raising=False)
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "state"))
  store_path = tmp_path / "oauth.json"
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(store_path))
  token_a = _enroll(store_path, "rotation-primary@example.com")
  token_b = _enroll(store_path, "rotation-sibling@example.com")

  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  sources = resolve_anthropic_credentials()
  selected = select_anthropic_credential_source(sources, pool=pool)
  assert selected is not None and selected.token == token_a
  replacement = next(source for source in sources.credentials if source.token == token_b)
  config = {
    "auth_mode": "oauth",
    "auth_token": selected.token,
    "credential_source": selected.source_metadata(),
  }
  provider = AnthropicProvider()
  refreshed = provider.next_credential(
    config,
    ProviderCredentialFailure(
      provider="anthropic",
      kind="rate_limit",
      status_code=429,
      reset_at=time.time() + 3600,
    ),
  )
  assert refreshed is not None
  assert refreshed["auth_token"] == token_b
  assert refreshed["credential_source"] == replacement.source_metadata()
  assert refreshed["credential_source"] != selected.source_metadata()
  merged = merge_refreshed_auth_config(config, refreshed)
  assert merged["credential_source"] == replacement.source_metadata()

  async def prepare_request() -> None:
    client = provider.create_client(merged)
    try:
      request = httpx2.Request(
        "POST",
        "https://api.anthropic.com/v1/messages",
        headers={"Authorization": f"Bearer {token_a}"},
      )
      await client._prepare_request(request)
      assert request.headers["Authorization"] == f"Bearer {token_b}"
      assert request.headers["Authorization"] != f"Bearer {token_a}"
    finally:
      await client.close()

  _run(prepare_request())


@pytest.mark.parametrize(
  ("field", "value"),
  [("capability_id", "node.fork"), ("run_mode", "batch"), ("credential_principal", "user")],
)
def test_dispatcher_credential_rebind_preserves_execution_authority(
  field: str, value: str,
) -> None:
  runner = _make_runner()
  original = runner.capability_execution.bind
  incompatible = type(original).model_validate({**original.to_json(), field: value})

  with pytest.raises(ValueError, match="changes execution authority"):
    runner._dispatcher.replace_capability_bind(incompatible)

  dispatcher = runner._dispatcher
  assert isinstance(dispatcher, ToolDispatcher)
  assert dispatcher._capability_bind is original
