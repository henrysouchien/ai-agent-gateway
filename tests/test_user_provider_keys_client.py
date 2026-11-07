"""Consumer cases from frozen user-provider-keys v2 (§§3, 4, 6); no network."""

import hashlib
import hmac
import json
import logging
from typing import Any, Literal

import httpx
import pytest
from hank_contracts.user_provider_keys import (
  AddKeyRequest,
  EnrolmentOk,
  FailureContext,
  KeyStatusRequest,
  ListKeysRequest,
  Provider,
  ResolveCredentialRequest,
)
from hank_model_transport.user_provider_keys import (
  IdentityUnresolved,
  ResolveContext,
  ResolvedUserKeys,
  UserKeyMissing,
  UserKeyResolverUnavailable,
  UserKeysSuspended,
  consumer_key_hash_for,
)

from agent_gateway import user_provider_keys_client as client_module
from agent_gateway.user_provider_keys_client import (
  EnrolmentRefused,
  RiskUserKeyClient,
  encode_body,
  internal_route_url,
  signed_v2_headers,
  v2_signature_payload,
)


URL = "https://risk.example.test/prefix/api/internal/resolve-credential"
HMAC_KEY = "synthetic-resolver-hmac"
API_KEY = "synthetic-user-api-key-never-log"
FP = "0123456789ab"
USER_ID = 101
CONTEXT = ResolveContext(
  consumer_key_hash=consumer_key_hash_for("risk-api-key"), channel="cli", session_id="session-1",
  model="model-1", request_id="request-1",
)


def _ok(provider: Provider = "anthropic") -> dict[str, Any]:
  return {
    "contract_version": 2, "provider": provider, "risk_user_id": USER_ID,
    "billing_mode": "byok", "credential_principal": "user", "identity": {"role": "invite"},
    "keys": [{"fingerprint": FP, "label": "研究", "api_key": API_KEY, "workspace_id": "workspace-1"}],
  }


def _enrolment_ok() -> dict[str, Any]:
  return {
    "contract_version": 2, "provider": "anthropic", "risk_user_id": USER_ID,
    "keys": [{
      "fingerprint": FP, "label": "研究", "status": "active", "stop_kind": None,
      "suspended_until": None, "created_at": "2026-11-03T00:00:00Z", "last_validated_at": None,
    }],
  }


def _refusal(code: str) -> dict[str, Any]:
  document: dict[str, Any] = {"contract_version": 2, "error": code}
  if code in ("user_key_missing", "user_keys_suspended"):
    document.update(provider="anthropic", risk_user_id=USER_ID)
  if code == "user_key_missing":
    document["message"] = "connect your API key"
  if code == "user_keys_suspended":
    document["keys"] = [{"fingerprint": FP, "kind": "auth_invalid", "suspended_until": None}]
  return document


class StubPost:
  def __init__(self, status: int, response: object) -> None:
    self.status = status
    self.response = json.dumps(response).encode("utf-8")
    self.calls: list[tuple[str, bytes, dict[str, str], float]] = []

  def __call__(self, url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    self.calls.append((url, body, headers, timeout))
    return self.status, self.response


def _client(post: StubPost) -> RiskUserKeyClient:
  return RiskUserKeyClient(resolve_url=URL, hmac_key=HMAC_KEY, timeout_seconds=1.25, post=post)


def _assert_signature(body: bytes, headers: dict[str, str], consumer_key_hash: str, provider: str = "anthropic") -> None:
  # Independent reconstruction of the documented canonical string from the posted bytes.
  normalised = httpx.Headers(headers)
  canonical = "\n".join([
    "v2", normalised["X-Resolver-Timestamp"], str(USER_ID), provider, consumer_key_hash,
    hashlib.sha256(body).hexdigest(),
  ])
  expected = hmac.new(HMAC_KEY.encode(), canonical.encode(), hashlib.sha256).hexdigest()
  assert normalised["X-Resolver-Signature"] == expected
  assert normalised["Content-Type"] == "application/json"


def test_encode_body_is_sorted_compact_utf8() -> None:
  assert encode_body({"z": "研究", "a": [1, True]}) == b'{"a":[1,true],"z":"\xe7\xa0\x94\xe7\xa9\xb6"}'


def test_signature_payload_and_explicit_timestamp() -> None:
  body = b'{"provider":"anthropic"}'
  expected_payload = f"v2\n1800000000\n101\nanthropic\n{CONTEXT.consumer_key_hash}\n" + hashlib.sha256(body).hexdigest()
  assert v2_signature_payload(
    timestamp="1800000000", risk_user_id=USER_ID, provider="anthropic",
    consumer_key_hash=CONTEXT.consumer_key_hash, body=body,
  ) == expected_payload
  headers = signed_v2_headers(
    timestamp="1800000000", hmac_key=HMAC_KEY, risk_user_id=USER_ID, provider="anthropic",
    consumer_key_hash=CONTEXT.consumer_key_hash, body=body,
  )
  assert headers["X-Resolver-Timestamp"] == "1800000000"
  _assert_signature(body, headers, CONTEXT.consumer_key_hash)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_resolve_posts_exact_signed_bytes_and_returns_keys(provider: Provider, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(client_module.time, "time", lambda: 1800000000.9)
  post = StubPost(200, _ok(provider))
  result = _client(post).resolve(risk_user_id=USER_ID, provider=provider, context=CONTEXT)
  assert isinstance(result, ResolvedUserKeys)
  assert (result.provider, result.risk_user_id) == (provider, USER_ID)
  assert result.keys[0].api_key == API_KEY
  assert result.keys[0].fingerprint == FP
  assert result.keys[0].workspace_id == "workspace-1"
  assert len(post.calls) == 1
  url, body, headers, timeout = post.calls[0]
  assert url == URL
  assert timeout == 1.25
  assert headers["X-Resolver-Timestamp"] == "1800000000"
  expected = {
    "contract_version": 2, "risk_user_id": USER_ID, "provider": provider,
    "consumer_key_hash": CONTEXT.consumer_key_hash, "request_id": "request-1",
    "session_id": "session-1", "channel": "cli", "model": "model-1",
  }
  assert body == json.dumps(expected, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode("utf-8")
  assert ResolveCredentialRequest.from_wire(json.loads(body)).refresh is False
  _assert_signature(body, headers, CONTEXT.consumer_key_hash, provider)


def _invalid_responses() -> list[tuple[int, object]]:
  ok = _ok()
  missing = _refusal("user_key_missing")
  return [
    (200, {**ok, "unknown": API_KEY}),
    (200, {**ok, "keys": []}),
    (200, {**ok, "contract_version": 1}),
    (200, {**ok, "provider": "openai"}),
    (200, {**ok, "risk_user_id": 102}),
    (200, {**ok, "billing_mode": "metered"}),
    (200, {**ok, "credential_principal": "service"}),
    (200, {**ok, "keys": [{**ok["keys"][0], "unknown": True}]}),
    (404, {**_refusal("user_keys_suspended"), "keys": []}),
    (404, {key: value for key, value in missing.items() if key != "message"}),
    (404, {**missing, "provider": "openai"}),
    (404, {**missing, "risk_user_id": 102}),
    (404, {**missing, "unknown": API_KEY}),
    (404, {**missing, "contract_version": 1}),
    (200, missing),
    (404, ok),
    (200, []), (404, None), (200, "not-an-object"),
    (500, {"contract_version": 2, "error": "unexpected"}),
  ]


@pytest.mark.parametrize(("status", "document"), _invalid_responses())
def test_malformed_resolver_envelopes_are_never_passed_through(status: int, document: object) -> None:
  with pytest.raises(UserKeyResolverUnavailable) as caught:
    _client(StubPost(status, document)).resolve(risk_user_id=USER_ID, provider="anthropic", context=CONTEXT)
  assert caught.value.stage == "resolver_response_invalid"
  assert caught.value.provider == "anthropic"
  assert caught.value.__cause__ is None
  assert caught.value.__suppress_context__


@pytest.mark.parametrize("body", [b"not json", b"\xff"])
def test_resolver_rejects_unparseable_response_bytes(body: bytes) -> None:
  post = StubPost(200, {})
  post.response = body
  with pytest.raises(UserKeyResolverUnavailable) as caught:
    _client(post).resolve(risk_user_id=USER_ID, provider="anthropic", context=CONTEXT)
  assert caught.value.stage == "resolver_response_invalid"


@pytest.mark.parametrize(("status", "code", "error_type"), [
  (404, "user_key_missing", UserKeyMissing),
  (404, "user_keys_suspended", UserKeysSuspended),
  (400, "identity_unresolved", IdentityUnresolved),
  (400, "provider_unsupported", UserKeyResolverUnavailable),
  (401, "auth_failed", UserKeyResolverUnavailable),
  (503, "resolver_transient", UserKeyResolverUnavailable),
])
def test_resolver_refusals_are_typed(status: int, code: str, error_type: type[Exception]) -> None:
  with pytest.raises(error_type) as caught:
    _client(StubPost(status, _refusal(code))).resolve(risk_user_id=USER_ID, provider="anthropic", context=CONTEXT)
  error = caught.value
  if isinstance(error, UserKeyMissing):
    assert "connect your" in str(error)
    assert error.risk_user_id == USER_ID
  if isinstance(error, UserKeysSuspended):
    assert error.keys[0].fingerprint == FP
    assert error.keys[0].kind == "auth_invalid"
  if isinstance(error, UserKeyResolverUnavailable):
    assert error.stage == code


@pytest.mark.parametrize(("status", "code"), [
  (400, "user_key_missing"), (503, "user_keys_suspended"),
  (404, "identity_unresolved"), (401, "provider_unsupported"),
  (503, "auth_failed"), (500, "resolver_transient"),
])
def test_resolver_refusal_requires_documented_http_status(status: int, code: str) -> None:
  with pytest.raises(UserKeyResolverUnavailable) as caught:
    _client(StubPost(status, _refusal(code))).resolve(risk_user_id=USER_ID, provider="anthropic", context=CONTEXT)
  assert caught.value.stage == "resolver_response_invalid"


def test_report_posts_refresh_and_failure_with_same_signature_binding() -> None:
  post = StubPost(200, _ok())
  failure = FailureContext(
    fingerprint=FP, kind="out_of_credit", reason="provider limit", reset_at="2026-12-01T00:00:00Z",
  )
  result = _client(post).report(risk_user_id=USER_ID, provider="anthropic", failure=failure, context=CONTEXT)
  assert result.keys[0].fingerprint == FP
  _, body, headers, _ = post.calls[0]
  request = ResolveCredentialRequest.from_wire(json.loads(body))
  assert request.refresh is True
  assert request.failure == failure
  assert request.consumer_key_hash == CONTEXT.consumer_key_hash
  _assert_signature(body, headers, CONTEXT.consumer_key_hash)


@pytest.mark.parametrize("operation", ["add", "list", "disable", "enable", "remove"])
def test_enrolment_routes_bodies_and_empty_consumer_hash(
  operation: Literal["add", "list", "disable", "enable", "remove"],
) -> None:
  post = StubPost(200, _enrolment_ok())
  client = _client(post)
  if operation == "add":
    result = client.add_key(
      risk_user_id=USER_ID, provider="anthropic", api_key=API_KEY, label="研究", workspace_id="workspace-1",
    )
  elif operation == "list":
    result = client.list_keys(risk_user_id=USER_ID, provider="anthropic")
  else:
    result = client.set_key_status(risk_user_id=USER_ID, provider="anthropic", fingerprint=FP, operation=operation)
  assert isinstance(result, EnrolmentOk)
  assert result.keys[0].fingerprint == FP
  url, body, headers, timeout = post.calls[0]
  assert url == "https://risk.example.test/prefix/api/internal/user-provider-keys/" + operation
  assert timeout == 1.25
  _assert_signature(body, headers, "")
  document = json.loads(body)
  assert "consumer_key_hash" not in document
  assert "operation" not in document
  if operation == "add":
    request = AddKeyRequest.from_wire(document)
    assert request.api_key == API_KEY
    assert request.label == "研究"
    assert request.workspace_id == "workspace-1"
    assert "研究".encode() in body
  elif operation == "list":
    request = ListKeysRequest.from_wire(document)
  else:
    request = KeyStatusRequest.from_wire(document)
    assert request.fingerprint == FP
  assert request.risk_user_id == USER_ID
  assert request.provider == "anthropic"
  assert request.request_id


def test_add_key_omits_unstated_workspace() -> None:
  post = StubPost(200, _enrolment_ok())
  _client(post).add_key(risk_user_id=USER_ID, provider="anthropic", api_key=API_KEY, label="desk")
  assert "workspace_id" not in json.loads(post.calls[0][1])


@pytest.mark.parametrize(("code", "message"), [
  ("key_invalid", "the provider rejected this API key"),
  ("workspace_required", "this key needs a workspace id"),
  ("key_already_enrolled", "this key is already connected"),
  ("key_not_found", "no connected key has that fingerprint"),
  ("provider_unsupported", "the API key service is unavailable; try again shortly"),
  ("auth_failed", "the API key service is unavailable; try again shortly"),
  ("resolver_transient", "the API key service is unavailable; try again shortly"),
])
def test_enrolment_refusals_preserve_codes_not_server_messages(code: str, message: str) -> None:
  post = StubPost(400, {"contract_version": 2, "error": code, "fingerprint": FP, "message": API_KEY})
  with pytest.raises(EnrolmentRefused) as caught:
    _client(post).add_key(risk_user_id=USER_ID, provider="anthropic", api_key=API_KEY, label="desk")
  assert caught.value.code == code
  assert caught.value.fingerprint == FP
  assert str(caught.value) == caught.value.user_message == message
  assert API_KEY not in str(caught.value)


def test_enrolment_identity_refusal_uses_shared_typed_error() -> None:
  post = StubPost(400, {"contract_version": 2, "error": "identity_unresolved"})
  with pytest.raises(IdentityUnresolved) as caught:
    _client(post).list_keys(risk_user_id=USER_ID, provider="anthropic")
  assert caught.value.provider == "anthropic"


@pytest.mark.parametrize(("status", "document"), [
  (200, {**_enrolment_ok(), "unknown": API_KEY}),
  (200, {**_enrolment_ok(), "provider": "openai"}),
  (200, {**_enrolment_ok(), "risk_user_id": 102}),
  (200, {**_enrolment_ok(), "contract_version": 1}),
  (200, {"contract_version": 2, "error": "key_invalid"}),
  (400, _enrolment_ok()),
  (400, {"contract_version": 2, "error": "key_invalid", "api_key": API_KEY}),
  (400, {"contract_version": 2, "error": "unknown"}),
  (400, {"contract_version": 1, "error": "key_invalid"}),
  (200, []), (400, None),
])
def test_invalid_enrolment_response_is_unavailable(status: int, document: object) -> None:
  with pytest.raises(UserKeyResolverUnavailable) as caught:
    _client(StubPost(status, document)).list_keys(risk_user_id=USER_ID, provider="anthropic")
  assert caught.value.stage == "enrolment_response_invalid"


def test_enrolment_rejects_unparseable_response_bytes() -> None:
  post = StubPost(200, {})
  post.response = b"not json"
  with pytest.raises(UserKeyResolverUnavailable) as caught:
    _client(post).list_keys(risk_user_id=USER_ID, provider="anthropic")
  assert caught.value.stage == "enrolment_response_invalid"


@pytest.mark.parametrize(("error_type", "stage"), [(httpx.ReadTimeout, "timeout"), (httpx.ConnectError, "request")])
@pytest.mark.parametrize("enrolment", [False, True])
def test_transport_failures_are_typed_and_redacted(
  error_type: type[httpx.RequestError], stage: str, enrolment: bool, caplog: pytest.LogCaptureFixture,
) -> None:
  def post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    raise error_type(API_KEY)

  client = RiskUserKeyClient(resolve_url=URL, hmac_key=HMAC_KEY, post=post)
  with caplog.at_level(logging.INFO), pytest.raises(UserKeyResolverUnavailable) as caught:
    if enrolment:
      client.list_keys(risk_user_id=USER_ID, provider="anthropic")
    else:
      client.resolve(risk_user_id=USER_ID, provider="anthropic", context=CONTEXT)
  assert caught.value.stage == stage
  assert caught.value.__cause__ is None
  assert caught.value.__suppress_context__
  assert API_KEY not in caplog.text
  assert API_KEY not in str(caught.value)


def test_default_post_uses_content_timeout_and_no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
  real_client = httpx.Client
  requests: list[httpx.Request] = []
  clients: list[httpx.Client] = []

  def receive(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    _assert_signature(request.content, dict(request.headers), CONTEXT.consumer_key_hash)
    return httpx.Response(200, json=_ok())

  def factory(**kwargs: Any) -> httpx.Client:
    assert kwargs == {"timeout": 1.25, "trust_env": False}
    client = real_client(transport=httpx.MockTransport(receive), **kwargs)
    clients.append(client)
    return client

  monkeypatch.setattr(client_module.httpx, "Client", factory)
  client = RiskUserKeyClient(resolve_url=URL, hmac_key=HMAC_KEY, timeout_seconds=1.25)
  result = client.resolve(risk_user_id=USER_ID, provider="anthropic", context=CONTEXT)
  assert result.keys[0].api_key == API_KEY
  assert len(requests) == 1
  assert requests[0].method == "POST"
  assert str(requests[0].url) == URL
  assert clients[0].is_closed


@pytest.mark.parametrize(("resolve_url", "hmac_key"), [("", HMAC_KEY), (URL, ""), ("https://risk.example.test/wrong", HMAC_KEY), (URL + "/", HMAC_KEY)])
def test_constructor_rejects_missing_or_wrong_configuration(resolve_url: str, hmac_key: str) -> None:
  with pytest.raises(ValueError):
    RiskUserKeyClient(resolve_url=resolve_url, hmac_key=hmac_key)


def test_internal_route_requires_resolver_suffix() -> None:
  with pytest.raises(ValueError):
    internal_route_url("https://risk.example.test/wrong", "add")


def test_success_refusal_and_invalid_body_logs_never_contain_key_bytes(caplog: pytest.LogCaptureFixture) -> None:
  with caplog.at_level(logging.INFO):
    _client(StubPost(200, _ok())).resolve(risk_user_id=USER_ID, provider="anthropic", context=CONTEXT)
    _client(StubPost(200, _enrolment_ok())).add_key(
      risk_user_id=USER_ID, provider="anthropic", api_key=API_KEY, label=API_KEY,
    )
    with pytest.raises(EnrolmentRefused):
      _client(StubPost(400, {"contract_version": 2, "error": "key_invalid", "message": API_KEY})).list_keys(
        risk_user_id=USER_ID, provider="anthropic",
      )
    with pytest.raises(UserKeyResolverUnavailable):
      _client(StubPost(200, {"api_key": API_KEY})).resolve(
        risk_user_id=USER_ID, provider="anthropic", context=CONTEXT,
      )
  assert FP in caplog.text
  assert "key_invalid" in caplog.text
  assert "resolver_response_invalid" in caplog.text
  assert API_KEY not in caplog.text
  assert HMAC_KEY not in caplog.text
  assert "研究" not in caplog.text
