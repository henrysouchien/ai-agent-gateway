from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient
import pytest

from agent_gateway.capability_binding import (
  CredentialHandle,
)
from agent_gateway.auth import AuthConfig, ResolverResult
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.commercial_claims import CommercialClaimError
from agent_gateway.commercial_work_start import (
  COMMERCIAL_CLAIM_HEADER,
  COMMERCIAL_WORK_AUTHORIZATION_HEADER,
  CommercialWorkStartError,
  CommercialWorkStartFacts,
  CommercialWorkStartGate,
  CommercialWorkStartContext,
  require_commercial_child_provider,
)
from agent_gateway.server import (
  ChatRuntime,
  GatewayServerConfig,
  MaterializedCredential,
  create_gateway_app,
)
from agent_gateway.work_authorization_consumption import (
  WorkAuthorizationAlreadyAttached,
  WorkAuthorizationConsumptionConflict,
  WorkAuthorizationConsumptionError,
)


CLAIM_TOKEN = "claim-token-must-not-persist"
WORK_TOKEN = "work-token-must-not-persist"
WORK_TOKEN_2 = "second-work-token-must-not-persist"
_SERVICE_HANDLE = CredentialHandle(
  handle_id="service:commercial-work-start-tests:anthropic",
  provider="anthropic",
  principal="service",
  tenant_id="commercial-work-start-tests",
  actor_id=None,
)


def _materialize_service_credential(
  handle: CredentialHandle,
) -> MaterializedCredential:
  assert handle is _SERVICE_HANDLE
  return MaterializedCredential(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "api_key": "test-key",
      "billing_mode": "metered",
      "rate_table_version": "test-v1",
    },
  )


async def _resolve_credentials(
  api_key: str,
  payload: Any,
) -> ResolverResult:
  assert api_key == "gateway-key"
  assert payload.user_id == "101"
  return ResolverResult(
    user_id="101",
    channel="mcp",
    auth_config=AuthConfig.from_dict({
      "provider": "anthropic",
      "billing_mode": "metered",
      "auth_mode": "api",
      "api_key": "test-key",
      "rate_table_version": "test-v1",
    }),
    credential_principal="service",
    allow_service_for_interactive=True,
    risk_user_id=101,
    role="owner",
    model_entitled_capabilities=CAPABILITY_IDS,
    model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
  )


def _gateway_config(
  *,
  build_chat_runtime,
  **overrides: Any,
) -> GatewayServerConfig:
  return GatewayServerConfig(
    tenant_id=_SERVICE_HANDLE.tenant_id,
    allow_service_credentials_for_interactive=True,
    credentials_resolver=_resolve_credentials,
    model_registry=INITIAL_MODEL_REGISTRY,
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    service_provider_handles={"anthropic": _SERVICE_HANDLE},
    service_auth_config_resolver=_materialize_service_credential,
    build_chat_runtime=build_chat_runtime,
    **overrides,
  )


class _ClaimVerifier:
  def __init__(self, order: list[str], claim: object) -> None:
    self.order = order
    self.claim = claim
    self.error: Exception | None = None

  def verify_for_work_start(self, token: str):
    self.order.append("verify_claim")
    assert token == CLAIM_TOKEN
    if self.error is not None:
      raise self.error
    return self.claim


class _AuthorizationVerifier:
  def __init__(self, order: list[str], authorization: object) -> None:
    self.order = order
    self.authorization = authorization
    self.calls: list[dict[str, Any]] = []

  def verify_for_attach(self, token: str, **facts):
    self.order.append("verify_authorization")
    assert token in {WORK_TOKEN, WORK_TOKEN_2}
    self.calls.append(facts)
    return SimpleNamespace(
      name=f"{self.authorization.name}:{len(self.calls)}:{facts['request_id']}"
    )


class _ConsumptionStore:
  def __init__(self, order: list[str], record: object) -> None:
    self.order = order
    self.record = record
    self.error: Exception | None = None
    self.attached: list[object] = []

  def attach_once(self, authorization):
    self.order.append("consume")
    self.attached.append(authorization)
    if self.error is not None:
      raise self.error
    return self.record


def _gate(order: list[str], *, pre_consume=None):
  claim = SimpleNamespace(name="verified-claim", subject="user:101")
  authorization = SimpleNamespace(name="verified-authorization")
  record = SimpleNamespace(name="consumption-record")
  claim_verifier = _ClaimVerifier(order, claim)
  authorization_verifier = _AuthorizationVerifier(order, authorization)
  store = _ConsumptionStore(order, record)

  def facts_resolver(session, request, channel):
    order.append("resolve_facts")
    assert session.session_id == "session-1"
    assert request.request_id == "request-1"
    assert channel == "mcp"
    return CommercialWorkStartFacts(
      operation="messages.create",
      provider="anthropic",
      billing_mode="metered",
      capability_id="portfolio.review",
    )

  gate = CommercialWorkStartGate(
    enabled=True,
    claim_verifier=claim_verifier,
    authorization_verifier=authorization_verifier,
    consumption_store=store,
    facts_resolver=facts_resolver,
    pre_consume=pre_consume,
  )
  return gate, claim_verifier, authorization_verifier, store


def _request_facts():
  session = SimpleNamespace(session_id="session-1", owner_user_id="101")
  request = SimpleNamespace(request_id="request-1")
  headers = {
    COMMERCIAL_CLAIM_HEADER: CLAIM_TOKEN,
    COMMERCIAL_WORK_AUTHORIZATION_HEADER: WORK_TOKEN,
  }
  return session, request, headers


def test_gate_is_default_off_and_rejects_false_authority_signals() -> None:
  gate = CommercialWorkStartGate(enabled=False)
  session, request, headers = _request_facts()
  assert gate.verify_request(
    {}, session=session, request=request, channel="mcp"
  ) is None
  with pytest.raises(
    CommercialWorkStartError, match="disabled"
  ) as raised:
    gate.verify_request(
      headers, session=session, request=request, channel="mcp"
    )
  assert raised.value.status_code == 403
  assert raised.value.code == "commercial_work_start_disabled"


def test_enabled_gate_requires_complete_dependencies_and_both_headers() -> None:
  with pytest.raises(ValueError, match="requires both verifiers"):
    CommercialWorkStartGate(enabled=True)
  order: list[str] = []
  gate, *_ = _gate(order)
  session, request, _ = _request_facts()
  with pytest.raises(CommercialWorkStartError) as raised:
    gate.verify_request(
      {COMMERCIAL_CLAIM_HEADER: CLAIM_TOKEN},
      session=session,
      request=request,
      channel="mcp",
    )
  assert raised.value.status_code == 401
  assert raised.value.code == "commercial_work_authority_required"
  assert order == []


def test_gate_verifies_exact_facts_then_consumes_without_retaining_tokens() -> None:
  order: list[str] = []
  gate, _, authorization_verifier, store = _gate(order)
  session, request, headers = _request_facts()

  pending = gate.verify_request(
    headers, session=session, request=request, channel="mcp"
  )
  context = gate.consume(pending)

  assert context is not None
  assert context.authorization is store.attached[0]
  assert authorization_verifier.calls == [{
    "execution_claim": context.claim,
    "request_id": "request-1",
    "session_id": "session-1",
    "operation": "messages.create",
    "provider": "anthropic",
    "billing_mode": "metered",
    "capability_id": "portfolio.review",
  }]
  assert order == [
    "resolve_facts",
    "verify_claim",
    "verify_authorization",
    "consume",
  ]
  assert CLAIM_TOKEN not in repr(pending)
  assert WORK_TOKEN not in repr(pending)
  assert CLAIM_TOKEN not in repr(context)
  assert WORK_TOKEN not in repr(context)


def test_gate_rejects_claim_for_a_different_session_owner_before_authorization() -> None:
  order: list[str] = []
  gate, _, authorization_verifier, store = _gate(order)
  session, request, headers = _request_facts()
  session.owner_user_id = "202"

  with pytest.raises(CommercialWorkStartError) as raised:
    gate.verify_request(headers, session=session, request=request, channel="mcp")

  assert raised.value.code == "commercial_work_subject_mismatch"
  assert raised.value.status_code == 403
  assert authorization_verifier.calls == []
  assert store.attached == []


def test_gate_runs_durability_preflight_before_consuming_authority() -> None:
  order: list[str] = []

  def unavailable(_pending):
    order.append("preflight")
    raise RuntimeError("circuit open")

  gate, _, _, store = _gate(order, pre_consume=unavailable)
  session, request, headers = _request_facts()

  pending = gate.verify_request(
    headers, session=session, request=request, channel="mcp"
  )
  with pytest.raises(CommercialWorkStartError) as raised:
    gate.consume(pending)

  assert raised.value.code == "commercial_work_start_unavailable"
  assert raised.value.status_code == 503
  assert store.attached == []
  assert order == [
    "resolve_facts", "verify_claim", "verify_authorization", "preflight",
  ]


def test_child_provider_must_match_verified_root_authority() -> None:
  work_start = CommercialWorkStartContext(
    claim=SimpleNamespace(),
    authorization=SimpleNamespace(provider="anthropic"),
    consumption=SimpleNamespace(),
  )

  require_commercial_child_provider(work_start, "anthropic")
  with pytest.raises(CommercialWorkStartError) as mismatch:
    require_commercial_child_provider(work_start, "openai")
  assert mismatch.value.code == "commercial_child_provider_mismatch"
  assert mismatch.value.status_code == 403

  with pytest.raises(CommercialWorkStartError) as invalid:
    require_commercial_child_provider(SimpleNamespace(), "anthropic")
  assert invalid.value.code == "commercial_child_authority_invalid"
  assert invalid.value.status_code == 503


def test_gate_maps_invalid_replay_conflict_and_storage_failure_to_no_start() -> None:
  order: list[str] = []
  gate, claim_verifier, _, store = _gate(order)
  session, request, headers = _request_facts()
  claim_verifier.error = CommercialClaimError("expired")
  with pytest.raises(CommercialWorkStartError) as invalid:
    gate.verify_request(
      headers, session=session, request=request, channel="mcp"
    )
  assert invalid.value.code == "commercial_work_authority_invalid"
  assert invalid.value.status_code == 403
  assert store.attached == []

  claim_verifier.error = None
  pending = gate.verify_request(
    headers, session=session, request=request, channel="mcp"
  )
  cases = (
    (
      WorkAuthorizationAlreadyAttached(SimpleNamespace()),
      "commercial_work_authority_already_consumed",
      409,
    ),
    (
      WorkAuthorizationConsumptionConflict("conflict"),
      "commercial_work_authority_conflict",
      409,
    ),
    (
      WorkAuthorizationConsumptionError("disk unavailable"),
      "commercial_work_start_unavailable",
      503,
    ),
  )
  for error, code, status in cases:
    store.error = error
    with pytest.raises(CommercialWorkStartError) as raised:
      gate.consume(pending)
    assert raised.value.code == code
    assert raised.value.status_code == status


class _CompleteRunner:
  def __init__(
    self,
    event_log,
    order: list[str],
    delay: float,
    capability_execution,
  ) -> None:
    self._event_log = event_log
    self._order = order
    self._delay = delay
    self.capability_execution = capability_execution

  async def run(self, **_kwargs) -> None:
    self._order.append("provider_start")
    if self._delay:
      await asyncio.sleep(self._delay)
    self._event_log.append({"type": "stream_complete", "usage": {}})


def _server_app(tmp_path: Path, *, provider_delay: float = 0):
  order: list[str] = []
  gate, _, _, store = _gate(order)
  observed_contexts = []

  def facts_resolver(_session, request, channel):
    order.append("resolve_facts")
    assert request.request_id in {"request-1", "request-2"}
    assert channel == "mcp"
    return CommercialWorkStartFacts(
      operation="messages.create",
      provider="anthropic",
      billing_mode="metered",
      capability_id="portfolio.review",
    )

  gate._facts_resolver = facts_resolver

  async def build_runtime(_session, request, _channel, _auth_manager):
    order.append("runtime")
    observed_contexts.append(request.commercial_work_start)
    assert order.index("consume") < order.index("runtime")
    capability_execution = request.capability_execution
    assert capability_execution is not None
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda event_log, _sid, _started_at: _CompleteRunner(
        event_log,
        order,
        provider_delay,
        capability_execution,
      ),
      capability_execution=capability_execution,
    )

  app = create_gateway_app(
    _gateway_config(
      build_chat_runtime=build_runtime,
      commercial_work_start_gate=gate,
      transcript_dir=tmp_path,
    )
  )
  return app, gate, store, order, observed_contexts


def _session_token(client: TestClient) -> str:
  response = client.post(
    "/api/chat/init",
    json={
      "api_key": "gateway-key",
      "user_id": "101",
      "context": {"channel": "mcp"},
    },
  )
  assert response.status_code == 200, response.text
  return response.json()["session_token"]


def _chat_headers(token: str, work_token: str = WORK_TOKEN) -> dict[str, str]:
  return {
    "Authorization": f"Bearer {token}",
    COMMERCIAL_CLAIM_HEADER: CLAIM_TOKEN,
    COMMERCIAL_WORK_AUTHORIZATION_HEADER: work_token,
  }


def test_server_consumes_before_runtime_and_never_transcripts_raw_tokens(
  tmp_path: Path,
) -> None:
  app, _, store, order, observed = _server_app(tmp_path)
  with TestClient(app) as client:
    token = _session_token(client)
    with client.stream(
      "POST",
      "/api/chat",
      headers=_chat_headers(token),
      json={
        "request_id": "request-1",
        "user_id": "101",
        "context": {"channel": "mcp"},
        "messages": [{"role": "user", "content": "hello"}],
      },
    ) as response:
      assert response.status_code == 200
      list(response.iter_lines())

  assert order.index("verify_authorization") < order.index("consume")
  assert order.index("consume") < order.index("runtime")
  assert order.index("runtime") < order.index("provider_start")
  assert len(store.attached) == 1
  assert observed[0] is not None
  persisted = b"".join(path.read_bytes() for path in tmp_path.glob("*.jsonl"))
  assert CLAIM_TOKEN.encode() not in persisted
  assert WORK_TOKEN.encode() not in persisted


def test_server_rejects_missing_or_replayed_authority_before_runtime(
  tmp_path: Path,
) -> None:
  app, _, store, order, _ = _server_app(tmp_path)
  with TestClient(app) as client:
    token = _session_token(client)
    missing = client.post(
      "/api/chat",
      headers={"Authorization": f"Bearer {token}"},
      json={
        "request_id": "request-1",
        "user_id": "101",
        "context": {"channel": "mcp"},
        "messages": [{"role": "user", "content": "hello"}],
      },
    )
    assert missing.status_code == 401
    assert missing.json()["error"] == "commercial_work_authority_required"
    assert "runtime" not in order

    store.error = WorkAuthorizationAlreadyAttached(SimpleNamespace())
    replay = client.post(
      "/api/chat",
      headers=_chat_headers(token),
      json={
        "request_id": "request-1",
        "user_id": "101",
        "context": {"channel": "mcp"},
        "messages": [{"role": "user", "content": "hello"}],
      },
    )
    assert replay.status_code == 409
    assert replay.json()["error"] == "commercial_work_authority_already_consumed"
    assert "runtime" not in order
    assert "provider_start" not in order


def test_concurrent_requests_reserve_one_dispatch_before_consumption(
  tmp_path: Path,
) -> None:
  app, _, store, _, _ = _server_app(tmp_path, provider_delay=0.25)
  barrier = threading.Barrier(2)
  with TestClient(app) as client:
    token = _session_token(client)

    def dispatch(request_id: str, work_token: str) -> int:
      barrier.wait(timeout=2)
      response = client.post(
        "/api/chat",
        headers=_chat_headers(token, work_token),
        json={
          "request_id": request_id,
          "user_id": "101",
          "context": {"channel": "mcp"},
          "messages": [{"role": "user", "content": "hello"}],
        },
      )
      return response.status_code

    with ThreadPoolExecutor(max_workers=2) as executor:
      futures = (
        executor.submit(dispatch, "request-1", WORK_TOKEN),
        executor.submit(dispatch, "request-2", WORK_TOKEN_2),
      )
      statuses = sorted(future.result() for future in futures)

  assert statuses == [200, 409]
  assert len(store.attached) == 1


def _commercial_jwt(issuer: str) -> str:
  def segment(payload: dict[str, object]) -> str:
    return base64.urlsafe_b64encode(
      json.dumps(payload, separators=(",", ":")).encode()
    ).rstrip(b"=").decode()

  return f"{segment({'alg': 'EdDSA'})}.{segment({'iss': issuer})}.x"


@pytest.mark.parametrize(
  "body_changes",
  (
    {COMMERCIAL_CLAIM_HEADER: CLAIM_TOKEN},
    {"context": {"commercial_claim_token": CLAIM_TOKEN}},
    {
      "metadata": {
        "nested": {
          "opaque": _commercial_jwt("risk-module-commercial-work-control")
        }
      }
    },
  ),
)
def test_server_rejects_commercial_bearer_material_anywhere_in_body(
  tmp_path: Path,
  body_changes: dict[str, object],
) -> None:
  app, _, store, order, _ = _server_app(tmp_path)
  with TestClient(app) as client:
    token = _session_token(client)
    body: dict[str, object] = {
      "request_id": "request-1",
      "user_id": "101",
      "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(body_changes)
    response = client.post(
      "/api/chat",
      headers=_chat_headers(token),
      json=body,
    )

  assert response.status_code == 422
  assert response.json()["error"] == "commercial_bearer_material_in_body"
  assert store.attached == []
  assert "runtime" not in order
  for value in _nested_string_values(body_changes):
    assert value not in response.text
  persisted = b"".join(path.read_bytes() for path in tmp_path.glob("*.jsonl"))
  assert CLAIM_TOKEN.encode() not in persisted
  assert WORK_TOKEN.encode() not in persisted


@pytest.mark.parametrize(
  ("path", "payload", "secret_markers"),
  (
    (
      "/api/chat",
      {
        "api_key": "sk_fake_chat_secret",
        "session_token": "fake_session_secret",
        "nested": {"access_token": "fake_nested_secret"},
      },
      (
        "sk_fake_chat_secret",
        "fake_session_secret",
        "fake_nested_secret",
      ),
    ),
    (
      "/api/chat/init",
      [
        {
          "api_key": "sk_fake_init_secret",
          "anthropic_api_key": "sk_fake_provider_secret",
        }
      ],
      ("sk_fake_init_secret", "sk_fake_provider_secret"),
    ),
  ),
)
def test_request_validation_errors_never_echo_request_values(
  tmp_path: Path,
  path: str,
  payload: object,
  secret_markers: tuple[str, ...],
) -> None:
  app, _, _, order, _ = _server_app(tmp_path)

  with TestClient(app) as client:
    response = client.post(path, json=payload)

  assert response.status_code == 422
  detail = response.json()["detail"]
  assert detail
  assert all({"type", "loc", "msg"} <= set(error) for error in detail)
  assert all("input" not in error and "ctx" not in error for error in detail)
  assert all(marker not in response.text for marker in secret_markers)
  assert "runtime" not in order


def _nested_string_values(value: object) -> list[str]:
  if isinstance(value, str):
    return [value]
  if isinstance(value, dict):
    return [
      item
      for nested in value.values()
      for item in _nested_string_values(nested)
    ]
  if isinstance(value, list):
    return [item for nested in value for item in _nested_string_values(nested)]
  return []


def test_unconfigured_server_rejects_commercial_headers() -> None:
  runtime_calls: list[str] = []

  async def build_runtime(*_args, **_kwargs):
    runtime_calls.append("runtime")
    raise AssertionError("runtime must not be constructed")

  app = create_gateway_app(
    _gateway_config(build_chat_runtime=build_runtime)
  )
  with TestClient(app) as client:
    token = _session_token(client)
    response = client.post(
      "/api/chat",
      headers=_chat_headers(token),
      json={
        "request_id": "request-1",
        "user_id": "101",
        "context": {"channel": "mcp"},
        "messages": [{"role": "user", "content": "hello"}],
      },
    )
  assert response.status_code == 403
  assert response.json()["error"] == "commercial_work_start_disabled"
  assert runtime_calls == []
