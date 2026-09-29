import asyncio
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
  rotate_anthropic_credential,
  select_anthropic_credential,
  upsert_anthropic_oauth_record,
)
from agent_gateway.runner_auth import merge_refreshed_auth_config  # noqa: E402
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


def _make_dispatcher(event_log: EventLog | None = None) -> ToolDispatcher:
  return ToolDispatcher(
    mcp_client=_NullMcpClient(),
    local_tool_handlers={},
    event_log=event_log or EventLog(),
    session_id="sess-auth",
  )


def _make_runner() -> AgentRunner:
  event_log = EventLog()
  provider = _StubProvider()
  return AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-auth",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="stub-model",
      effort="none",
      auth_config={"api_key": "old", "max_tokens": 512},
    ),
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


def test_usage_limited_credential_rotates_to_sibling_and_keeps_bound_model(
  monkeypatch: Any,
) -> None:
  reset_at = time.time() + 4 * 3600
  pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", pool)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_TOKEN_A, _TOKEN_B]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setattr(gateway_runner, "STREAM_RETRY_DELAY", 0.0)

  provider = _PooledAnthropicProvider(limited_tokens=(_TOKEN_A,), reset_at=reset_at)
  event_log = EventLog()
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-pool",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config={
        "auth_mode": "oauth",
        "auth_token": _TOKEN_A,
        "max_tokens": 512,
      },
    ),
    user_id="alice",
    billing_mode="byok",
    rate_table_version="unknown",
  )
  bound_model_key = runner.capability_execution.bind.model_key

  _run(runner.run(messages=[{"role": "user", "content": "hello"}]))

  event_types = [entry.event.get("type") for entry in event_log.entries]
  assert provider.streamed_tokens == [_TOKEN_A, _TOKEN_B]
  assert "credential_refreshed" in event_types
  assert "error" not in event_types
  assert runner.capability_execution.bind.model_key == bound_model_key
  assert runner._auth_config["auth_token"] == _TOKEN_B
  assert pool.blocked_until(_TOKEN_A) == pytest.approx(reset_at, abs=2.0)
  assert pool.blocked_until(_TOKEN_B) == 0.0


def test_single_credential_pool_is_never_withheld_from_a_run(
  monkeypatch: Any,
) -> None:
  reset_at = time.time() + 3600
  pool = AnthropicCredentialPool()
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _TOKEN_A)
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKENS", raising=False)
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", "/nonexistent/anthropic/oauth.json")
  sources = resolve_anthropic_credentials()

  assert sources.tokens == (_TOKEN_A,)
  assert select_anthropic_credential(sources, pool=pool) == _TOKEN_A

  # Nothing to swap in, and the only credential is still handed out: a pool of
  # one leaves the turn to the gateway's own bounded stream retry, which ends it
  # with the sanitized rate-limit error (see the pool-of-one turn test below).
  assert rotate_anthropic_credential(
    sources, current=_TOKEN_A, until=reset_at, pool=pool
  ) == ""
  assert select_anthropic_credential(sources, pool=pool) == _TOKEN_A


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

  assert rotated == {"auth_token": "sk-ant-oat01-sibling@example.com"}
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
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-pool-of-one",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config={
        "auth_mode": "oauth",
        "auth_token": _TOKEN_A,
        "max_tokens": 512,
      },
    ),
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
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-pool-exhausted",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config={
        "auth_mode": "oauth",
        "auth_token": _TOKEN_A,
        "max_tokens": 512,
      },
    ),
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
  bound_token = select_anthropic_credential(
    resolve_anthropic_credentials(environ=gateway_environ),
    pool=parent_pool,
  )
  assert bound_token == _TOKEN_A

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
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-autonomous-child",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config=dict(child_credential.auth_config),
    ),
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

  assert select_anthropic_credential(
    resolve_anthropic_credentials(environ=environ, user_scope=1), pool=pool
  ) == primary
  assert provider.next_credential(mine_bound, failure) == {"auth_token": sibling}
  assert provider.next_credential(
    {**mine_bound, "auth_token": sibling}, failure
  ) == {"auth_token": _TOKEN_A}
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
  ) == {"auth_token": _TOKEN_A}


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
  bound_token = select_anthropic_credential(
    resolve_anthropic_credentials(environ=gateway_environ, user_scope=3),
    pool=parent_pool,
  )
  assert bound_token == primary

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
  runner = AgentRunner(
    event_log=event_log,
    dispatcher=_make_dispatcher(event_log),
    session_id="sess-user-scoped-child",
    capability_execution=stub_bound_capability_execution(
      provider=provider,
      model="claude-pool-test",
      effort="none",
      auth_config=dict(child_credential.auth_config),
    ),
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
