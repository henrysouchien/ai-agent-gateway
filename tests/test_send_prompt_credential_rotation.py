# ruff: noqa: E402

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest

PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.providers.anthropic as anthropic_provider_module
from agent_gateway.providers import StreamEvent
from agent_gateway.providers.anthropic import AnthropicProvider
from agent_gateway.providers.anthropic_oauth import AnthropicCredentialPool
from agent_gateway.send_prompt import send_prompt
from gateway_test_support.capability_execution_test_support import stub_bound_capability_execution

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
  response = httpx2.Response(status_code, request=request, headers={"request-id": "req_x"}, json=body)
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
def pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AnthropicCredentialPool:
  process_pool = AnthropicCredentialPool()
  monkeypatch.setattr(anthropic_provider_module, "ANTHROPIC_CREDENTIAL_POOL", process_pool)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", _REFUSED)
  monkeypatch.setenv("ANTHROPIC_AUTH_TOKENS", json.dumps([_REFUSED, _SERVING]))
  monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
  monkeypatch.setenv("ANTHROPIC_AUTH_STORE_PATH", str(tmp_path / "absent-oauth.json"))
  monkeypatch.setenv("USER_DATA_DIR", str(tmp_path))
  return process_pool


def _run(provider: _PoolProvider) -> str:
  execution = stub_bound_capability_execution(
    provider=provider,
    model="claude-sonnet-5",
    effort="none",
    auth_config={"auth_mode": "oauth", "auth_token": _REFUSED, "max_tokens": 1024},
  )
  return asyncio.run(send_prompt("hello", capability_execution=execution, user_id="alice"))


@pytest.mark.parametrize(
  "rejection",
  [
    pytest.param((403, _ORG_REFUSAL_BODY), id="org-refuses-oauth"),
    pytest.param((429, _RATE_LIMIT_BODY), id="rate-limited"),
  ],
)
def test_send_prompt_rotates_a_rejected_pool_member_and_answers_on_the_next(
  pool: AnthropicCredentialPool,
  rejection: tuple[int, dict[str, object]],
) -> None:
  provider = _PoolProvider({_REFUSED: _status_error(*rejection), _SERVING: "answered"})

  assert _run(provider) == "answered"
  assert provider.tokens_used == [_REFUSED, _SERVING]


def test_send_prompt_tries_each_pool_member_once_then_raises_the_provider_rejection(
  pool: AnthropicCredentialPool,
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
