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
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.work_authorization_consumption import (
  WorkAuthorizationAlreadyAttached,
  WorkAuthorizationConsumptionConflict,
  WorkAuthorizationConsumptionError,
)


CLAIM_TOKEN = "claim-token-must-not-persist"
WORK_TOKEN = "work-token-must-not-persist"
WORK_TOKEN_2 = "second-work-token-must-not-persist"


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
  claim = SimpleNamespace(name="verified-claim")
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
  session = SimpleNamespace(session_id="session-1")
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
  def __init__(self, event_log, order: list[str], delay: float) -> None:
    self._event_log = event_log
    self._order = order
    self._delay = delay

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
    return ChatRuntime(
      system_prompt="test",
      build_runner=lambda event_log, _sid: _CompleteRunner(
        event_log, order, provider_delay
      ),
    )

  app = create_gateway_app(GatewayServerConfig(
    auth_config={"model": "claude-sonnet-4-6"},
    build_chat_runtime=build_runtime,
    commercial_work_start_gate=gate,
    transcript_dir=tmp_path,
  ))
  return app, gate, store, order, observed_contexts


def _session_token(client: TestClient) -> str:
  response = client.post(
    "/api/chat/init",
    json={
      "api_key": "gateway-key",
      "user_id": "alice",
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

  app = create_gateway_app(GatewayServerConfig(build_chat_runtime=build_runtime))
  with TestClient(app) as client:
    token = _session_token(client)
    response = client.post(
      "/api/chat",
      headers=_chat_headers(token),
      json={
        "request_id": "request-1",
        "context": {"channel": "mcp"},
        "messages": [{"role": "user", "content": "hello"}],
      },
    )
  assert response.status_code == 403
  assert response.json()["error"] == "commercial_work_start_disabled"
  assert runtime_calls == []
