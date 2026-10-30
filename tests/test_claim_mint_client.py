import asyncio

import httpx2
import pytest
from hank_contracts.agent_api import AGENT_API_CLAIM_AUDIENCE, AGENT_API_CLAIM_ENV_VARS

from agent_gateway.claim_mint_client import (
  GatewayClaimMintError,
  GatewayClaimMintSchemaError,
  mint_gateway_user_claim,
)


_BASE_URL = "https://gateway.example.test/private-gateway"
_API_KEY = "synthetic-roster-bearer-for-mint-tests"
_SECOND_API_KEY = "synthetic-second-user-bearer-for-mint-tests"
_RESPONSE_SECRET = "synthetic-private-response-detail"


def _require(condition: bool, message: str) -> None:
  if not condition:
    pytest.fail(message, pytrace=False)


def _claim() -> dict[str, str]:
  fields = {
    "audience": AGENT_API_CLAIM_AUDIENCE,
    "issued_at": "1800000000",
    "expiry": "1800000300",
    "user_id": "test-risk-user-41",
    "user_email": "analyst@example.test",
    "nonce": "ab" * 16,
    "signature": "cd" * 32,
  }
  return {name: fields[field] for field, name in AGENT_API_CLAIM_ENV_VARS.items()}


def _assert_redacted(error: GatewayClaimMintError) -> None:
  rendered = f"{error!s} {error!r} {error.args!r}"
  private_values = (
    _BASE_URL, _API_KEY, _SECOND_API_KEY, _RESPONSE_SECRET,
    *_claim().values(),
  )
  _require(
    all(value not in rendered for value in private_values),
    "mint errors must not expose credentials, identities, URLs, or response fields",
  )
  assert error.__cause__ is None
  assert error.__suppress_context__


@pytest.mark.parametrize("suffix", ["", "/", "///"])
@pytest.mark.parametrize("status_code", [200, 201, 299])
def test_mint_posts_only_roster_bearer_and_returns_seven_fields(
  suffix: str, status_code: int,
) -> None:
  calls = 0
  expected = _claim()

  def respond(request: httpx2.Request) -> httpx2.Response:
    nonlocal calls
    calls += 1
    assert request.method == "POST"
    _require(
      str(request.url) == f"{_BASE_URL}/api/claims/mint",
      "mint must use the exact configured endpoint without identity selectors",
    )
    _require(
      request.headers.get("Authorization") == f"Bearer {_API_KEY}",
      "mint must send only the caller's roster bearer",
    )
    _require(request.content == b"", "mint must not send a request body")
    assert "content-type" not in request.headers
    assert not any(name.lower().startswith("x-agent-claim-") for name in request.headers)
    return httpx2.Response(status_code, json=expected)

  async def scenario() -> None:
    async with httpx2.AsyncClient(
      transport=httpx2.MockTransport(respond), auth=("other-user", "synthetic-password"),
    ) as client:
      result = await mint_gateway_user_claim(
        base_url=f"{_BASE_URL}{suffix}", api_key=_API_KEY, client=client,
      )
      assert not client.is_closed
    assert set(result) == set(AGENT_API_CLAIM_ENV_VARS.values())
    assert len(result) == 7
    _require(result == expected, "mint must return the exact unmodified claim environment")

  asyncio.run(scenario())
  assert calls == 1


def test_second_roster_user_forbidden_response_stays_redacted() -> None:
  calls = 0

  def respond(request: httpx2.Request) -> httpx2.Response:
    nonlocal calls
    calls += 1
    _require(
      request.headers.get("Authorization") == f"Bearer {_SECOND_API_KEY}",
      "mint must not replace a second user's roster bearer",
    )
    _require(request.content == b"", "mint must not select a user in a body")
    return httpx2.Response(403, json={"detail": _RESPONSE_SECRET})

  async def scenario() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
      with pytest.raises(GatewayClaimMintError) as caught:
        await mint_gateway_user_claim(
          base_url=_BASE_URL, api_key=_SECOND_API_KEY, client=client,
        )
    assert type(caught.value) is GatewayClaimMintError
    assert caught.value.status_code == 403
    _assert_redacted(caught.value)

  asyncio.run(scenario())
  assert calls == 1


@pytest.mark.parametrize(
  "status_code",
  [100, 199, 300, 301, 302, 303, 304, 307, 308, 399, 400, 401, 403, 404, 429, 499, 500, 503, 599],
)
def test_non_success_status_is_preserved_without_redirect_or_retry(status_code: int) -> None:
  calls = 0

  def respond(request: httpx2.Request) -> httpx2.Response:
    nonlocal calls
    calls += 1
    return httpx2.Response(
      status_code,
      headers={"Location": "https://other.example.test/private"},
      content=_RESPONSE_SECRET.encode(),
    )

  async def scenario() -> None:
    async with httpx2.AsyncClient(
      transport=httpx2.MockTransport(respond), follow_redirects=True,
    ) as client:
      with pytest.raises(GatewayClaimMintError) as caught:
        await mint_gateway_user_claim(base_url=_BASE_URL, api_key=_API_KEY, client=client)
    assert type(caught.value) is GatewayClaimMintError
    assert caught.value.status_code == status_code
    _assert_redacted(caught.value)

  asyncio.run(scenario())
  assert calls == 1


@pytest.mark.parametrize("field", list(AGENT_API_CLAIM_ENV_VARS))
def test_missing_claim_field_is_rejected(field: str) -> None:
  payload = _claim()
  del payload[AGENT_API_CLAIM_ENV_VARS[field]]
  _assert_schema_rejected(payload)


def test_extra_claim_field_is_rejected() -> None:
  payload = {**_claim(), "unexpected": _RESPONSE_SECRET}
  _assert_schema_rejected(payload)


@pytest.mark.parametrize("payload", [None, [], [_RESPONSE_SECRET], _RESPONSE_SECRET, 4, True])
def test_non_object_claim_is_rejected(payload: object) -> None:
  _assert_schema_rejected(payload)


@pytest.mark.parametrize("field", list(AGENT_API_CLAIM_ENV_VARS))
@pytest.mark.parametrize("value", [None, True, 41, 1.5, [], {}])
def test_claim_values_are_strict_strings(field: str, value: object) -> None:
  payload: dict[str, object] = dict(_claim())
  payload[AGENT_API_CLAIM_ENV_VARS[field]] = value
  _assert_schema_rejected(payload)


def test_wrong_claim_audience_is_rejected() -> None:
  payload = _claim()
  payload[AGENT_API_CLAIM_ENV_VARS["audience"]] = "another-audience"
  _assert_schema_rejected(payload)


def _assert_schema_rejected(payload: object) -> None:
  def respond(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=payload)

  async def scenario() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
      with pytest.raises(GatewayClaimMintSchemaError) as caught:
        await mint_gateway_user_claim(base_url=_BASE_URL, api_key=_API_KEY, client=client)
    assert isinstance(caught.value, GatewayClaimMintError)
    assert caught.value.status_code == 200
    _assert_redacted(caught.value)

  asyncio.run(scenario())


def test_empty_optional_claim_email_is_preserved() -> None:
  expected = _claim()
  expected[AGENT_API_CLAIM_ENV_VARS["user_email"]] = ""

  def respond(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, json=expected)

  async def scenario() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
      result = await mint_gateway_user_claim(base_url=_BASE_URL, api_key=_API_KEY, client=client)
    _require(result == expected, "mint must preserve the signer's optional empty email")

  asyncio.run(scenario())


@pytest.mark.parametrize("content", [b"", b"not-json", b'{"private":"synthetic-private-response-detail"', b"\xff"])
def test_invalid_json_is_rejected_without_exposing_decoder_details(content: bytes) -> None:
  def respond(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, content=content)

  async def scenario() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
      with pytest.raises(GatewayClaimMintSchemaError) as caught:
        await mint_gateway_user_claim(base_url=_BASE_URL, api_key=_API_KEY, client=client)
    assert caught.value.status_code == 200
    _assert_redacted(caught.value)

  asyncio.run(scenario())


@pytest.mark.parametrize("error_type", [httpx2.ConnectError, httpx2.ReadTimeout, httpx2.InvalidURL])
def test_transport_errors_are_redacted_without_raw_exception_chains(error_type: type[Exception]) -> None:
  calls = 0

  def respond(request: httpx2.Request) -> httpx2.Response:
    nonlocal calls
    calls += 1
    raise error_type(f"{_BASE_URL} {_API_KEY} {_RESPONSE_SECRET}")

  async def scenario() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
      with pytest.raises(GatewayClaimMintError) as caught:
        await mint_gateway_user_claim(base_url=_BASE_URL, api_key=_API_KEY, client=client)
    assert type(caught.value) is GatewayClaimMintError
    assert caught.value.status_code is None
    _assert_redacted(caught.value)

  asyncio.run(scenario())
  assert calls == 1


@pytest.mark.parametrize("base_url", ["", " ", "\t\n"])
def test_empty_base_url_is_an_entry_error_without_request(base_url: str) -> None:
  def respond(request: httpx2.Request) -> httpx2.Response:
    pytest.fail("empty base URL must fail before making a request", pytrace=False)

  async def scenario() -> None:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
      with pytest.raises(ValueError):
        await mint_gateway_user_claim(base_url=base_url, api_key=_API_KEY, client=client)

  asyncio.run(scenario())


def test_owned_client_has_no_timeout_cap_and_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
  real_client = httpx2.AsyncClient
  owned_clients: list[httpx2.AsyncClient] = []

  def respond(request: httpx2.Request) -> httpx2.Response:
    assert all(value is None for value in request.extensions["timeout"].values())
    return httpx2.Response(200, json=_claim())

  def make_client(**kwargs) -> httpx2.AsyncClient:
    assert kwargs["timeout"] is None
    assert kwargs["follow_redirects"] is False
    client = real_client(transport=httpx2.MockTransport(respond), **kwargs)
    owned_clients.append(client)
    return client

  monkeypatch.setattr("agent_gateway.claim_mint_client.httpx2.AsyncClient", make_client)
  result = asyncio.run(mint_gateway_user_claim(base_url=_BASE_URL, api_key=_API_KEY))
  _require(result == _claim(), "owned client must return the minted claim")
  assert len(owned_clients) == 1
  assert owned_clients[0].is_closed
