from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from agent_gateway import AgentRunner, McpClientManager, ToolDispatcher
from agent_gateway.auth import AuthConfig, ResolverResult
from hank_contracts.commercial_claims import VerifiedCommercialClaim
from agent_gateway.model_preferences import ModelPreferenceStore
from agent_gateway.server import ChatRuntime, GatewayServerConfig, create_gateway_app
from agent_gateway.server_models import ChatRequest
from agent_gateway.session import AuthManager, GatewaySession
from gateway_test_support.control_plane_identity import fake_identity_resolver
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY


def test_user_preset_store_round_trip_and_owner_isolation(tmp_path: Path) -> None:
  path = tmp_path / "preferences.sqlite3"
  store = ModelPreferenceStore(path)
  reference = store.put_user_preset(
    tenant_id="tenant-a", actor_id="alice", name="economy", policy_revision="candidate-1",
  )
  assert reference.name == "economy"
  assert reference.policy_revision == "candidate-1"
  assert reference.updated_at > 0
  assert ModelPreferenceStore(path).get_user_preset(
    tenant_id="tenant-a", actor_id="alice",
  ) == reference
  assert store.get_user_preset(tenant_id="tenant-a", actor_id="bob") is None
  assert store.get_user_preset(tenant_id="tenant-b", actor_id="alice") is None
  store.put_user_preset(
    tenant_id="tenant-a", actor_id="bob", name="base", policy_revision="candidate-1",
  )
  replacement = store.put_user_preset(
    tenant_id="tenant-a", actor_id="alice", name="base", policy_revision="candidate-2",
  )
  assert store.get_user_preset(tenant_id="tenant-a", actor_id="alice") == replacement
  assert store.delete_user_preset(tenant_id="tenant-a", actor_id="alice")
  assert not store.delete_user_preset(tenant_id="tenant-a", actor_id="alice")
  assert store.get_user_preset(tenant_id="tenant-a", actor_id="alice") is None
  assert store.get_user_preset(tenant_id="tenant-a", actor_id="bob") is not None


def test_firm_preset_requires_verified_claim_account_uuid(tmp_path: Path) -> None:
  path = tmp_path / "preferences.sqlite3"
  store = ModelPreferenceStore(path)
  claim = VerifiedCommercialClaim(
    schema_version=1, key_id="test-key", subject="user:101", environment="dev",
    surface="mcp", commercial_account_id=UUID(int=1), agreement_id=UUID(int=2),
    agreement_terms_revision=1, offer_code="test-offer",
    effective_scopes=("portfolio.review",), entitlement_revision=1,
    payer_policy_version="payer-v1", budget_policy_version="budget-v1",
    shadow_rate_version="shadow-v1", manifest_version="manifest-v1",
    authorized_work_start_deadline=2_000_000_000, usage_accept_until=2_000_000_100,
    issued_at=1_900_000_000, expires_at=2_000_000_000, context_id=UUID(int=3),
  )
  reference = store.put_firm_preset(
    commercial_account_id=claim.commercial_account_id,
    name="economy", policy_revision="candidate-1",
  )
  assert ModelPreferenceStore(path).get_firm_preset(
    commercial_account_id=claim.commercial_account_id,
  ) == reference
  assert store.get_firm_preset(commercial_account_id=uuid4()) is None
  assert store.get_user_preset(tenant_id="tenant-a", actor_id=claim.subject) is None
  for unrelated_identity in (None, "tenant-a", "owner", str(claim.commercial_account_id)):
    with pytest.raises(ValueError):
      store.get_firm_preset(commercial_account_id=cast(UUID, unrelated_identity))
  other_account = uuid4()
  store.put_firm_preset(
    commercial_account_id=other_account, name="base", policy_revision="candidate-1",
  )
  replacement = store.put_firm_preset(
    commercial_account_id=claim.commercial_account_id,
    name="base", policy_revision="candidate-2",
  )
  assert store.get_firm_preset(commercial_account_id=claim.commercial_account_id) == replacement
  assert store.delete_firm_preset(commercial_account_id=claim.commercial_account_id)
  assert not store.delete_firm_preset(commercial_account_id=claim.commercial_account_id)
  assert store.get_firm_preset(commercial_account_id=claim.commercial_account_id) is None
  assert store.get_firm_preset(commercial_account_id=other_account) is not None


def _preset_app(tmp_path: Path) -> tuple[Any, ModelPreferenceStore]:
  async def credentials_resolver(api_key: str, _request: Any) -> ResolverResult:
    return ResolverResult(
      user_id="bob" if api_key == "bob-key" else "alice",
      risk_user_id=102 if api_key == "bob-key" else 101,
      channel="cli",
      auth_config=AuthConfig.from_dict({
        "provider": "anthropic", "billing_mode": "byok", "api_key": "operator-key",
      }),
      credential_principal="service", allow_service_for_interactive=True,
      role="owner", model_entitled_capabilities=CAPABILITY_IDS,
      model_entitled_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
    )

  store = ModelPreferenceStore(
    tmp_path / "preferences.sqlite3", known_preset_names=frozenset({"economy", "base"}),
  )
  async def build_runtime(
    _session: GatewaySession, request: ChatRequest, _channel: str | None,
    _auth: AuthManager | None, *, storage_root: Path | None = None,
  ) -> ChatRuntime:
    execution = request.capability_execution
    assert execution is not None
    def build_runner(event_log, session_id, started_at):
      return AgentRunner(
        event_log,
        ToolDispatcher(
          capability_bind=execution.bind,
          mcp_client=McpClientManager(config_path=None),
        ),
        session_id, capability_execution=execution, started_at=started_at,
      )
    return ChatRuntime(
      system_prompt="test", build_runner=build_runner,
      capability_execution=execution,
    )
  app = create_gateway_app(GatewayServerConfig(
    identity_resolver=fake_identity_resolver, tenant_id="test-product",
    model_registry=INITIAL_MODEL_REGISTRY,
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    credentials_resolver=credentials_resolver, model_preference_store=store,
    build_chat_runtime=build_runtime,
  ))
  return app, store


def _headers(client: TestClient, api_key: str = "alice-key") -> dict[str, str]:
  response = client.post(
    "/api/chat/init", json={"api_key": api_key, "context": {"channel": "cli"}},
  )
  assert response.status_code == 200, response.text
  return {"Authorization": f"Bearer {response.json()['session_token']}"}


def test_authenticated_user_preset_doors_round_trip_and_actor_isolation(tmp_path: Path) -> None:
  app, store = _preset_app(tmp_path)
  with TestClient(app) as client:
    alice = _headers(client)
    bob = _headers(client, "bob-key")
    assert client.get("/api/model-presets/self", headers=alice).json() == {
      "name": None, "policy_revision": None, "updated_at": None,
      "known_preset_names": ["base", "economy"],
    }
    put = client.put("/api/model-presets/self", headers=alice, json={"name": "economy"})
    assert put.status_code == 200, put.text
    reference = put.json()
    assert reference["name"] == "economy"
    assert reference["policy_revision"] == INITIAL_MODEL_SELECTION_POLICY.revision
    assert reference["updated_at"] > 0
    get = client.get("/api/model-presets/self", headers=alice)
    assert get.status_code == 200, get.text
    assert get.json() == reference
    assert client.get("/api/model-presets/self", headers=bob).json()["name"] is None
    bob_put = client.put("/api/model-presets/self", headers=bob, json={"name": "base"})
    assert bob_put.status_code == 200, bob_put.text
    deleted = client.delete("/api/model-presets/self", headers=alice)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {
      "name": None, "policy_revision": None, "updated_at": None,
      "known_preset_names": ["base", "economy"],
    }
    assert client.get("/api/model-presets/self", headers=alice).json()["name"] is None
    assert client.get("/api/model-presets/self", headers=bob).json()["name"] == "base"
  assert store.get_user_preset(tenant_id="test-product", actor_id="101") is None
  assert store.get_user_preset(tenant_id="test-product", actor_id="102") is not None


def test_unknown_preset_name_returns_existing_error_envelope_without_overwriting(tmp_path: Path) -> None:
  app, store = _preset_app(tmp_path)
  with TestClient(app) as client:
    headers = _headers(client)
    accepted = client.put("/api/model-presets/self", headers=headers, json={"name": "economy"})
    assert accepted.status_code == 200, accepted.text
    response = client.put("/api/model-presets/self", headers=headers, json={"name": "unknown"})
    assert response.status_code == 422, response.text
    payload = response.json()
    assert payload["error_code"] == "model_preset_unknown"
    assert payload["name"] == "unknown"
    assert payload["policy_revision"] == INITIAL_MODEL_SELECTION_POLICY.revision
    assert set(payload["known_preset_names"]) == {"economy", "base"}
    assert isinstance(payload["message"], str)
    assert client.get("/api/model-presets/self", headers=headers).json() == accepted.json()
  reference = store.get_user_preset(tenant_id="test-product", actor_id="101")
  assert reference is not None and reference.name == "economy"


def test_every_preset_door_lists_the_names_it_accepts(tmp_path: Path) -> None:
  app, _store = _preset_app(tmp_path)
  with TestClient(app) as client:
    headers = _headers(client)
    listed = client.get("/api/model-presets/self", headers=headers)
    assert listed.status_code == 200, listed.text
    assert listed.json()["known_preset_names"] == ["base", "economy"]
    selected = client.put(
      "/api/model-presets/self", headers=headers,
      json={"name": listed.json()["known_preset_names"][0]},
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["name"] == "base"
    assert selected.json()["known_preset_names"] == ["base", "economy"]
    cleared = client.delete("/api/model-presets/self", headers=headers)
    assert cleared.json()["known_preset_names"] == ["base", "economy"]


@pytest.mark.parametrize("method", ["get", "put", "delete"])
def test_user_preset_doors_require_authentication(tmp_path: Path, method: str) -> None:
  app, _store = _preset_app(tmp_path)
  with TestClient(app) as client:
    response = client.request(
      method, "/api/model-presets/self",
      json={"name": "economy"} if method == "put" else None,
    )
    assert response.status_code == 401, response.text


def test_concrete_preferences_and_preset_references_are_independent(tmp_path: Path) -> None:
  store = ModelPreferenceStore(tmp_path / "preferences.sqlite3")
  concrete = store.put(
    tenant_id="tenant-a", actor_id="alice", capability_id="session.driver",
    model_key="anthropic.claude-sonnet-5-5", effort="high",
  )
  store.put_user_preset(
    tenant_id="tenant-a", actor_id="alice", name="economy", policy_revision="candidate-1",
  )
  assert store.get(
    tenant_id="tenant-a", actor_id="alice", capability_id="session.driver",
  ) == concrete
  assert store.delete_user_preset(tenant_id="tenant-a", actor_id="alice")
  assert store.get(
    tenant_id="tenant-a", actor_id="alice", capability_id="session.driver",
  ) == concrete
