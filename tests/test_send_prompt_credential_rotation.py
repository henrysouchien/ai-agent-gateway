# ruff: noqa: E402

import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.providers import StreamEvent
from agent_gateway.providers.anthropic import AnthropicProvider
from agent_gateway import CapabilityExecutionResolver, resolve_auth_config
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.providers.anthropic_oauth import (
  is_anthropic_generation_ref, resolve_anthropic_credentials,
)
from agent_gateway.send_prompt import _rotated_execution, send_prompt
from gateway_test_support.capability_execution_test_support import stub_bound_capability_execution
from model_authority.binding import AuthContext, CredentialHandle, ModelSelectionIntent
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY

_REFUSED = "sk-ant-oat01-refused-org"
_SERVING = "sk-ant-oat01-serving"
_ORG_REFUSAL_BODY = {
  "type": "error",
  "error": {
    "type": "permission_error",
    "message": "OAuth authentication is currently not allowed for this organization.",
    "details": {"error_code": "oauth_not_allowed_for_organization"},
  },
}
_RATE_LIMIT_BODY = {
  "type": "error",
  "error": {"type": "rate_limit_error", "message": "This request would exceed your rate limit."},
}


def _status_error(status_code: int, body: dict[str, object]) -> Exception:
  anthropic = pytest.importorskip("anthropic")
  request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
  headers = {"request-id": "req_x"}
  if status_code == 429:
    # A bare 429 is backoff, not rotation: the account parks only on an exhausted window.
    headers["anthropic-ratelimit-unified-5h-status"] = "rejected"
    headers["anthropic-ratelimit-unified-5h-reset"] = str(int(time.time()) + 3600)
  response = httpx2.Response(status_code, request=request, headers=headers, json=body)
  error_type = anthropic.PermissionDeniedError if status_code == 403 else anthropic.RateLimitError
  return error_type(f"Error code: {status_code} - {body}", response=response, body=body)


class _RejectingStream:
  def __init__(self, exc: Exception) -> None:
    self._exc = exc

  async def __aenter__(self) -> Any:
    raise self._exc

  async def __aexit__(self, *exc_info: object) -> bool:
    return False


class _PoolProvider(AnthropicProvider):
  """The real Anthropic classifier and pool rotation over a scripted API.

  A token scripted with an exception is rejected by the Anthropic SDK path
  (so the error the caller sees is the provider's own wrapped rejection); a
  token scripted with text answers it.
  """

  def __init__(self, outcomes: dict[str, Exception | str]) -> None:
    super().__init__()
    self.outcomes = outcomes
    self.tokens_used: list[str] = []

  def create_client(self, config: dict[str, Any], *, timeout: float | None = None) -> Any:
    del timeout
    token = str(config["auth_token"])
    self.tokens_used.append(token)
    return token

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    del client, timeout

  async def stream(self, client: Any, params: dict[str, Any]):
    outcome = self.outcomes[client]
    if isinstance(outcome, Exception):
      messages = SimpleNamespace(stream=lambda **_: _RejectingStream(outcome))
      sdk_client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))
      async for event in super().stream(sdk_client, params):
        yield event
      return
    yield StreamEvent(type="message_start", input_tokens=3)
    yield StreamEvent(type="text_delta", text=outcome)
    yield StreamEvent(type="message_end", stop_reason="end_turn")


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  # Cooldowns live in the store's selection sidecar; the private store path and
  # USER_DATA_DIR below isolate them from this machine's.
  # The root tests/conftest.py forces api mode with a fake API key process-wide;
  # these oauth-pool tests declare their own mode.
  monkeypatch.setenv("ANTHROPIC_AUTH_MODE", "oauth")
  monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _REFUSED)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_REFUSED, _SERVING]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "absent-oauth.json"))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))


def test_resolve_auth_config_retains_the_selected_pool_generation(
  pool: None,
) -> None:
  del pool
  config = resolve_auth_config(max_tokens=1024)
  selected = resolve_anthropic_credentials().by_token(_REFUSED)
  assert selected is not None
  assert config["auth_token"] == _REFUSED
  assert config["credential_source"] == {**selected.source_metadata(), "automatic": True}
  assert is_anthropic_generation_ref(config["credential_source"]["generation_ref"])


def _run(provider: _PoolProvider) -> str:
  execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-sonnet-5",
    effort="none",
    auth_config={"auth_mode": "oauth", "auth_token": _REFUSED, "max_tokens": 1024},
  )
  return asyncio.run(send_prompt("hello", capability_execution=execution, user_id="alice", completion_record_root=None))


@pytest.mark.parametrize(
  "rejection",
  [
    pytest.param((403, _ORG_REFUSAL_BODY), id="org-refuses-oauth"),
    pytest.param((429, _RATE_LIMIT_BODY), id="rate-limited"),
  ],
)
def test_send_prompt_rotates_a_rejected_pool_member_and_answers_on_the_next(
  pool: None,
  rejection: tuple[int, dict[str, object]],
) -> None:
  provider = _PoolProvider({_REFUSED: _status_error(*rejection), _SERVING: "answered"})

  assert _run(provider) == "answered"
  assert provider.tokens_used == [_REFUSED, _SERVING]


def test_send_prompt_tries_each_pool_member_once_then_raises_the_provider_rejection(
  pool: None,
) -> None:
  anthropic = pytest.importorskip("anthropic")
  provider = _PoolProvider({
    _REFUSED: _status_error(403, _ORG_REFUSAL_BODY),
    _SERVING: _status_error(429, _RATE_LIMIT_BODY),
  })

  # The last member's own rejection surfaces; no member is tried twice.
  with pytest.raises(anthropic.RateLimitError):
    _run(provider)

  assert provider.tokens_used == [_REFUSED, _SERVING]


@pytest.mark.parametrize("has_generation_metadata", [True, False])
def test_send_prompt_rotation_names_the_sibling_generation(
  pool: None,
  monkeypatch: pytest.MonkeyPatch,
  has_generation_metadata: bool,
) -> None:
  rejection = _status_error(429, _RATE_LIMIT_BODY)
  provider = _PoolProvider({_REFUSED: rejection, _SERVING: "answered"})
  sources = resolve_anthropic_credentials()
  member_a = sources.by_token(_REFUSED)
  member_b = sources.by_token(_SERVING)
  assert member_a is not None and member_b is not None
  execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-sonnet-5",
    effort="none",
    credential_ref=member_a.source_metadata()["generation_ref"],
    auth_config={
      "auth_mode": "oauth", "auth_token": _REFUSED, "max_tokens": 1024,
      "credential_source": member_a.source_metadata(),
    },
  )
  if not has_generation_metadata:
    monkeypatch.setattr(
      provider, "next_credential", lambda config, failure: {"auth_token": _SERVING},
    )

  rotated = _rotated_execution(execution, rejection)

  if has_generation_metadata:
    assert rotated is not None
    assert rotated.bind.credential_ref == member_b.source_metadata()["generation_ref"]
    assert rotated.auth_config["auth_token"] == _SERVING
    assert rotated.bind.to_json() == {
      **execution.bind.to_json(),
      "credential_ref": member_b.source_metadata()["generation_ref"],
    }
  else:
    assert rotated is None
    assert execution.bind.credential_ref == member_a.source_metadata()["generation_ref"]
    assert execution.auth_config["auth_token"] == _REFUSED


def test_send_prompt_parking_persists_across_second_bind(
  pool: None,
) -> None:
  model = "anthropic.claude-opus-5-5"

  def bind(config: dict[str, Any], provider: _PoolProvider):
    generation_ref = config["credential_source"]["generation_ref"]
    assert is_anthropic_generation_ref(generation_ref)
    handle = CredentialHandle(
      handle_id=generation_ref, provider="anthropic",
      principal="service", tenant_id="tenant", actor_id=None,
    )
    resolver = CapabilityExecutionResolver(
      registry=INITIAL_MODEL_REGISTRY,
      selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      auth_context=AuthContext(
        run_mode="batch", actor_id="alice", tenant_id="tenant",
        user_provider_handles={}, service_provider_handles={"anthropic": handle},
        entitled_capabilities=frozenset({"session.driver"}),
        entitled_model_keys=frozenset({model}),
      ),
      credential_materializer=lambda selected: MaterializedCredential(
        handle=selected, auth_config=config,
      ),
      adapter_resolver=lambda adapter_id: provider,
    )
    execution = resolver.resolve(
      "session.driver",
      explicit_intent=ModelSelectionIntent(model_key=model, effort=None, source="explicit_user"),
    )
    assert execution.bind.credential_ref == generation_ref
    assert execution.resolver is resolver
    return execution

  first_config = resolve_auth_config(max_tokens=4096)
  assert first_config["auth_token"] == _REFUSED
  provider1 = _PoolProvider({_REFUSED: _status_error(429, _RATE_LIMIT_BODY), _SERVING: "answered-1"})
  first_execution = bind(first_config, provider1)
  observed_events = []
  assert asyncio.run(send_prompt(
    "hello", capability_execution=first_execution, user_id="alice",
    on_usage=observed_events.append, completion_record_root=None,
  )) == "answered-1"
  assert provider1.tokens_used == [_REFUSED, _SERVING]

  # Fresh resolution must honor the parking from the first execution.
  second_config = resolve_auth_config(max_tokens=4096)
  assert second_config["auth_token"] == _SERVING
  assert second_config["credential_source"]["generation_ref"] != first_execution.bind.credential_ref
  assert len(observed_events) == 1
  assert observed_events[0].capability_bind["credential_ref"] == second_config["credential_source"]["generation_ref"]
  provider2 = _PoolProvider({_SERVING: "answered-2"})
  second_execution = bind(second_config, provider2)
  assert asyncio.run(send_prompt("hello again", capability_execution=second_execution, user_id="alice", completion_record_root=None)) == "answered-2"
  assert provider2.tokens_used == [_SERVING]


def test_send_prompt_pool_of_one_selects_and_does_not_rotate_infinitely(
  pool: None,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  anthropic = pytest.importorskip("anthropic")
  # Single token configured: pool of one
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-solo")
  monkeypatch.delenv("ANTHROPIC_AUTH_TOKENS", raising=False)

  # 1. Initial resolution selects it
  initial_config = resolve_auth_config(max_tokens=1024)
  assert initial_config["auth_token"] == "token-solo"

  # 2. 429 raises provider rejection; no sibling exists, so no infinite rotation
  provider = _PoolProvider({"token-solo": _status_error(429, _RATE_LIMIT_BODY)})
  execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-sonnet-5",
    effort="none",
    auth_config=initial_config,
  )
  with pytest.raises(anthropic.RateLimitError):
    asyncio.run(send_prompt("hello", capability_execution=execution, user_id="alice", completion_record_root=None))

  assert provider.tokens_used == ["token-solo"]

  # 3. Subsequent bind on pool of one still binds it (select fallback)
  parked_config = resolve_auth_config(max_tokens=1024)
  assert parked_config["auth_token"] == "token-solo"
