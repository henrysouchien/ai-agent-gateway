from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from model_authority.bind import CapabilityBind
from model_authority.selection import TierModelsDefault
from model_authority.binding import (
  AuthContext,
  CapabilityResolutionError,
  CredentialHandle,
  ModelSelectionIntent,
  resolve_capability_model,
)
from hank_model_transport.capability_execution import MaterializedCredential
from hank_model_transport.user_provider_keys import (
  IdentityUnresolved, NO_API_KEY_CONSUMER, UserKeyMissing, UserKeyResolverUnavailable, consumer_key_hash_for,
)
from agent_gateway.claim_signing_authority import (
  GatewayClaimSigningAuthority,
)
from agent_gateway.control_plane import batches
from hank_model_transport.providers import ModelInfo, ModelProvider
from model_authority.current import INITIAL_MODEL_REGISTRY, INITIAL_MODEL_SELECTION_POLICY
from agent_gateway.session import AuthManager, GatewaySession, SessionStore
from agent_gateway.admission import default_root_provenance, AdmittedContext, root_context
from agent_gateway.approval_route import child_approval_route
from agent_workflow_contracts import ResolvedAuthority
from agent_workflow_contracts.models import (
  AdmittedContextWire,
  AdmittedLineage,
  ToolGrant,
)

@pytest.fixture(autouse=True)
def _batch_user_roster(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("GATEWAY_USER_KEYS", json.dumps([
    {"key": "batch-owner", "slug": "alice", "risk_user_id": 7, "role": "owner"},
  ]))
  # Pools are process-global: a test elsewhere in the worker that leaves them installed
  # must not decide this module's enrollment. Tests that need pools install their own.
  _uninstall_batch_pools(monkeypatch)



class _Provider(ModelProvider):
  name = "anthropic"

  def __init__(self, name: str = "anthropic") -> None:
    self.name = name

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=16_000,
      supports_thinking=True,
    )


def _session(
  *,
  handle: CredentialHandle,
  auth_config: dict[str, Any],
) -> GatewaySession:
  return GatewaySession(
    session_id="sess-batch-test",
    api_key_hash=hashlib.sha256(b"risk-api-key").hexdigest(),
    created_at=int(time.time()),
    expires_at=int(time.time()) + 60,
    user_id="alice",
    owner_user_id="alice",
    kind="control",
    risk_user_id=7,
    auth_config=auth_config,
    tenant_id="tenant-a",
    session_credential_handle=handle,
  )


class _BatchPools:
  """A stand-in for the installed UserKeyPools that records every probe and lease."""

  def __init__(self, *, enrolled: frozenset[str], outage: bool) -> None:
    self._enrolled = enrolled
    self._outage = outage
    self.probes: list[dict[str, Any]] = []
    self.leases: list[dict[str, Any]] = []

  def enrolled(self, **kwargs: Any) -> bool:
    self.probes.append(kwargs)
    if self._outage:
      raise UserKeyResolverUnavailable(kwargs["provider"], "transport")
    return kwargs["provider"] in self._enrolled

  def lease(self, **kwargs: Any) -> SimpleNamespace:
    self.leases.append(kwargs)
    provider = kwargs["provider"]
    if provider not in self._enrolled:
      raise UserKeyMissing(provider, kwargs["risk_user_id"])
    return SimpleNamespace(auth_config=lambda: {
      "provider": provider, "auth_mode": "api",
      "api_key": "user-key-secret", "billing_mode": "byok",
      "entitlement_kind": "commercial_byok",
    })


def _install_batch_pools(
  monkeypatch: pytest.MonkeyPatch, *, enrolled: frozenset[str], outage: bool = False,
) -> _BatchPools:
  pools = _BatchPools(enrolled=enrolled, outage=outage)
  monkeypatch.setattr(batches, "installed_user_key_pools", lambda provider=None: pools)
  return pools


def _uninstall_batch_pools(monkeypatch: pytest.MonkeyPatch) -> _BatchPools:
  """No pools installed: the real accessor raises ``not_configured``."""

  def not_installed(provider: str | None = None) -> _BatchPools:
    raise UserKeyResolverUnavailable(provider, "not_configured")

  monkeypatch.setattr(batches, "installed_user_key_pools", not_installed)
  return _BatchPools(enrolled=frozenset(), outage=True)


def _deployment_config(provider: str) -> tuple[SimpleNamespace, CredentialHandle]:
  deployment_handle = CredentialHandle(
    handle_id=f"service:deployment:{provider}", provider=provider,
    principal="service", tenant_id="tenant-a", actor_id=None,
  )
  config = _config()
  config.service_provider_handles = {provider: deployment_handle}
  config.service_admission = None
  config.service_auth_config_resolver = lambda *_args, **_kwargs: pytest.fail(
    "service materializer called for the session handle",
  )
  config.capability_adapter_resolver = lambda _adapter: _Provider(provider)
  return config, deployment_handle


def _deployment_session(handle: CredentialHandle, provider: str) -> GatewaySession:
  return _session(handle=handle, auth_config={
    "provider": provider, "api_key": "deployment-secret", "entitlement_kind": "org_commercial",
  })


def _config(
  *,
  service_handle: CredentialHandle | None = None,
  materialized_handles: list[CredentialHandle] | None = None,
  materialized_scopes: list[Any] | None = None,
) -> SimpleNamespace:
  provider = _Provider()
  material_by_identity = (
    {
      id(service_handle): MaterializedCredential(
        handle=service_handle,
        auth_config={
          "provider": "anthropic",
          "api_key": "service-secret",
          "entitlement_kind": "operator_personal_subscription",
        },
      )
    }
    if service_handle is not None
    else {}
  )

  def _materialize(
    handle: CredentialHandle,
    *,
    user_scope: Any = None,
  ) -> MaterializedCredential:
    if materialized_handles is not None:
      materialized_handles.append(handle)
    if materialized_scopes is not None:
      materialized_scopes.append(user_scope)
    return material_by_identity[id(handle)]

  return SimpleNamespace(
    tenant_id="tenant-a",
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    model_registry=INITIAL_MODEL_REGISTRY,
    service_provider_handles=(
      {"anthropic": service_handle}
      if service_handle is not None
      else {}
    ),
    service_auth_config_resolver=(
      _materialize if service_handle is not None else None
    ),
    service_admission=lambda handle, *, risk_user_id, auth_config=None: (
      risk_user_id in {1, 7} and (
        handle is service_handle
        or (auth_config is not None and auth_config.get("entitlement_kind") == "operator_personal_subscription")
      )
    ),
    capability_adapter_resolver=lambda adapter: (
      provider
      if adapter == "anthropic.messages"
      else pytest.fail(f"unexpected adapter: {adapter}")
    ),
    default_provider=provider,
  )


def test_authenticated_batch_marks_exact_user_handle_run_scoped(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_batch_pools(monkeypatch, enrolled=frozenset({"anthropic", "openai"}))
  handle = CredentialHandle(
    handle_id="user:alice:anthropic",
    provider="anthropic",
    principal="user",
    tenant_id="tenant-a",
    actor_id="alice",
  )
  session = _session(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "api_key": "user-secret",
    },
  )

  resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=_config()),
    user_id="alice",
    authenticated_session=session,
  )

  assert isinstance(resolver.auth_context, AuthContext)
  assert resolver.auth_context.run_mode == "batch"
  assert resolver.auth_context.user_provider_handles["anthropic"] is handle
  assert resolver.auth_context.run_scoped_user_providers == frozenset({
    "anthropic", "openai"
  })
  assert execution.bind.credential_principal == "user"
  assert execution.auth_config["api_key"] == "user-secret"
  assert "node.implement" in resolver.auth_context.entitled_capabilities
  assert "node.explore" in resolver.auth_context.entitled_capabilities
  assert "session.driver" in resolver.auth_context.entitled_capabilities
  assert "anthropic.claude-opus-5-5" in resolver.auth_context.entitled_model_keys


def test_standalone_batch_materializes_admitted_owner_seat() -> None:
  service_handle = CredentialHandle(
    handle_id="service:batch:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="tenant-a",
    actor_id=None,
  )

  materialized_scopes: list[Any] = []

  resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(
      gateway_config=_config(
        service_handle=service_handle,
        materialized_scopes=materialized_scopes,
      )
    ),
    user_id="1",
    authenticated_session=None,
  )

  assert materialized_scopes == [1]
  assert set(resolver.auth_context.user_provider_handles) == {"openai"}
  assert (
    resolver.auth_context.service_provider_handles["anthropic"]
    is service_handle
  )
  assert execution.bind.credential_principal == "service"
  assert execution.auth_config["api_key"] == "service-secret"


def test_authenticated_batch_materializes_service_principal_for_the_session_risk_user() -> None:
  service_handle = CredentialHandle(
    handle_id="service:batch:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="tenant-a",
    actor_id=None,
  )
  session = GatewaySession(
    session_id="sess-batch-service-scope",
    api_key_hash=hashlib.sha256(b"batch-api-key").hexdigest(),
    created_at=int(time.time()),
    expires_at=int(time.time()) + 60,
    user_id="alice",
    owner_user_id="alice",
    risk_user_id=7,
    kind="control",
    tenant_id="tenant-a",
  )
  materialized_scopes: list[Any] = []

  _resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(
      gateway_config=_config(
        service_handle=service_handle,
        materialized_scopes=materialized_scopes,
      )
    ),
    user_id="alice",
    authenticated_session=session,
  )

  assert execution.bind.credential_principal == "service"
  assert materialized_scopes == [7]


def test_authenticated_batch_rejects_session_owner_mismatch() -> None:
  handle = CredentialHandle(
    handle_id="user:alice:anthropic",
    provider="anthropic",
    principal="user",
    tenant_id="tenant-a",
    actor_id="alice",
  )
  session = _session(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "api_key": "user-secret",
    },
  )

  with pytest.raises(
    RuntimeError,
    match="session owner does not match",
  ):
    batches._batch_capability_execution_context(
      app_state=SimpleNamespace(gateway_config=_config()),
      user_id="bob",
      authenticated_session=session,
    )


def test_batch_parent_rebinds_authenticated_alias_to_canonical_owner() -> None:
  handle = CredentialHandle(
    handle_id="service:batch-alias:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="tenant-a",
    actor_id=None,
  )
  now = int(time.time())
  session = GatewaySession(
    session_id="sess-batch-alias",
    api_key_hash=hashlib.sha256(b"batch-api-key").hexdigest(),
    created_at=now,
    expires_at=now + 60,
    user_id="henry",
    owner_user_id="1",
    user_email="henry@example.com",
    risk_user_id=1,
    kind="chat",
    auth_config={
      "provider": "anthropic",
      "api_key": "user-secret",
      "entitlement_kind": "operator_personal_subscription",
    },
    tenant_id="tenant-a",
    session_credential_handle=handle,
    channel="mcp",
    raw_user_id="henry",
    user_slug="henry",
    user_aliases=("1", "henry", "henry@example.com"),
    identity_status="risk_user_id_authoritative",
  )
  _resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=_config()),
    user_id="1",
    authenticated_session=session,
  )

  parent = batches._batch_parent_session(
    user_id="1",
    user_email=session.user_email,
    role=session.role,
    tenant_id="tenant-a",
    channel=session.channel,
    authenticated_session=session,
    session_driver_execution=execution,
  )

  assert type(parent) is GatewaySession
  assert parent is not session
  assert parent.user_id == "1"
  assert parent.owner_user_id == "1"
  assert parent.user_email == "henry@example.com"
  assert parent.risk_user_id == 1
  assert parent.kind == "control"
  assert parent.channel == "mcp"
  assert parent.auth_config == execution.auth_config
  assert parent.session_credential_handle is None
  assert parent.raw_user_id == "henry"
  assert parent.user_slug == "henry"
  assert parent.user_aliases == ("1", "henry", "henry@example.com")
  assert parent.identity_status == "risk_user_id_authoritative"
  assert session.user_id == "henry"
  assert session.owner_user_id == "1"


def test_batch_parent_rebind_carries_authenticated_session_token() -> None:
  handle = CredentialHandle(
    handle_id="service:batch-alias:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="tenant-a",
    actor_id=None,
  )
  now = int(time.time())
  session = GatewaySession(
    session_id="sess-batch-alias-token",
    api_key_hash=hashlib.sha256(b"batch-api-key").hexdigest(),
    created_at=now,
    expires_at=now + 60,
    user_id="henry",
    owner_user_id="1",
    risk_user_id=1,
    kind="chat",
    auth_config={
      "provider": "anthropic",
      "api_key": "user-secret",
      "entitlement_kind": "operator_personal_subscription",
    },
    session_token="caller-issued-jwt",
    tenant_id="tenant-a",
    session_credential_handle=handle,
    channel="mcp",
  )
  _resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=_config()),
    user_id="1",
    authenticated_session=session,
  )

  parent = batches._batch_parent_session(
    user_id="1",
    user_email=None,
    role=session.role,
    tenant_id="tenant-a",
    channel=session.channel,
    authenticated_session=session,
    session_driver_execution=execution,
  )

  assert parent is not session
  assert parent.user_id == "1"
  assert parent.session_token == "caller-issued-jwt"


def test_sessionless_batch_parent_carries_no_session_token() -> None:
  service_handle = CredentialHandle(
    handle_id="service:batch:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="tenant-a",
    actor_id=None,
  )
  _resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(
      gateway_config=_config(service_handle=service_handle)
    ),
    user_id="alice",
    authenticated_session=None,
  )

  parent = batches._batch_parent_session(
    user_id="alice",
    user_email=None,
    role="owner",
    tenant_id="tenant-a",
    channel=None,
    authenticated_session=None,
    session_driver_execution=execution,
  )

  assert parent.user_id == "alice"
  assert parent.session_token is None


def test_batch_parent_preserves_exact_authenticated_owner_session() -> None:
  handle = CredentialHandle(
    handle_id="user:alice:anthropic",
    provider="anthropic",
    principal="user",
    tenant_id="tenant-a",
    actor_id="alice",
  )
  session = _session(
    handle=handle,
    auth_config={
      "provider": "anthropic",
      "api_key": "user-secret",
    },
  )
  _resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=_config()),
    user_id="alice",
    authenticated_session=session,
  )

  parent = batches._batch_parent_session(
    user_id="alice",
    user_email=None,
    role=session.role,
    tenant_id="tenant-a",
    channel=session.channel,
    authenticated_session=session,
    session_driver_execution=execution,
  )

  assert parent is session


def test_authenticated_cross_provider_batch_preserves_session_selection(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  _install_batch_pools(monkeypatch, enrolled=frozenset({"anthropic", "openai"}))
  user_handle = CredentialHandle(
    handle_id="user:alice:openai",
    provider="openai",
    principal="user",
    tenant_id="tenant-a",
    actor_id="alice",
  )
  service_handle = CredentialHandle(
    handle_id="service:batch:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="tenant-a",
    actor_id=None,
  )
  service_material = MaterializedCredential(
    handle=service_handle,
    auth_config={
      "provider": "anthropic",
      "api_key": "service-secret",
    },
  )
  providers = {
    "anthropic": _Provider("anthropic"),
    "openai": _Provider("openai"),
  }
  session = _session(
    handle=user_handle,
    auth_config={
      "provider": "openai",
      "api_key": "user-secret",
    },
  )
  required_bind = resolve_capability_model(
    "session.driver",
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth=AuthContext(
      run_mode="batch",
      actor_id="alice",
      tenant_id="tenant-a",
      user_provider_handles={"openai": user_handle},
      service_provider_handles={"anthropic": service_handle},
      entitled_capabilities=frozenset({"session.driver"}),
      entitled_model_keys=INITIAL_MODEL_SELECTION_POLICY.capabilities[
        "session.driver"
      ].allowed_model_keys,
      run_scoped_user_providers=frozenset({"openai"}),
    ),
    explicit_intent=ModelSelectionIntent(
      model_key="openai.gpt-6-1-sol",
      effort="medium",
      source="explicit_user",
    ),
  )
  gateway_config = SimpleNamespace(
    tenant_id="tenant-a",
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    model_registry=INITIAL_MODEL_REGISTRY,
    service_provider_handles={"anthropic": service_handle},
    service_auth_config_resolver=lambda handle, *, user_scope=None: (
      service_material
      if handle is service_handle
      else pytest.fail("unexpected service handle")
    ),
    capability_adapter_resolver=lambda adapter: providers[
      INITIAL_MODEL_REGISTRY.require(required_bind.model_key).provider
      if adapter == required_bind.adapter
      else "anthropic"
    ],
    default_provider=providers["anthropic"],
  )

  resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=gateway_config),
    user_id="alice",
    authenticated_session=session,
    required_bind=required_bind,
  )

  assert resolver.auth_context.run_scoped_user_providers == frozenset({
    "anthropic", "openai"
  })
  assert execution.bind.provider == "openai"
  assert execution.bind.model_key == "openai.gpt-6-1-sol"
  assert execution.bind.upstream_model == "gpt-6.1-sol"
  assert execution.bind.credential_principal == "user"
  assert execution.auth_config["api_key"] == "user-secret"


class _DispatchRegistry:
  def __init__(self) -> None:
    self.status = "running"
    self.launch_record = {
      "approval_policy": "approve_all",
      "approve_live_orders": False,
    }

  def launch_approval_settings(self, _batch_id: int) -> dict[str, object]:
    return dict(self.launch_record)

  def close(self) -> None:
    return None

  def lookup_batch_dispatch(self, **_kwargs: Any) -> None:
    return None

  def list_active_batches(self, _user_id: str) -> list[dict[str, Any]]:
    return []

  def get_batch_digest(self, _batch_id: int) -> dict[str, Any]:
    return {
      "batch_id": 17,
      "user_id": "alice",
      "status": self.status,
      "cost_usd": 0.0,
    }

  def transition_status_if_current(
    self,
    _batch_id: int,
    status: str,
    *,
    expected_statuses: set[str],
    error: str | None = None,
  ) -> bool:
    _ = error
    if self.status not in expected_statuses:
      return False
    self.status = status
    return True


class _DispatchEventBus:
  async def publish_terminal_if_absent(
    self,
    _user_id: str,
    _control_run_id: str,
    _event: dict[str, Any],
  ) -> bool:
    return True


def _dispatch_app_state(
  *,
  tmp_path: Path,
  registry: _DispatchRegistry,
  controller: Any,
  task_registry: batches.BatchTaskRegistry,
  gateway_config: SimpleNamespace,
) -> SimpleNamespace:
  return SimpleNamespace(
    batch_backend=batches.BatchBackend(
      registry_for_user=lambda _user_id: registry,
      controller=controller,
      open_captured_run_context=controller.open_captured_run_context,
      stage_admitter=lambda *_args: pytest.fail(
        "capture-only dispatch controller must not admit stages"
      ),
      terminal_event_payload=lambda batch_id, digest: {
        "type": "run_state_changed",
        "run_id": f"batch_{batch_id}",
        "control_run_id": f"batch_{batch_id}",
        "batch_id": batch_id,
        "user_id": digest["user_id"],
        "state": digest["status"],
      },
      active_batch_error=RuntimeError,
      dispatch_replay=batches.BatchDispatchReplayError,
      dispatch_rejected=batches.BatchDispatchRejectedError,
      dispatch_rejection_record=SimpleNamespace,
    ),
    gateway_config=gateway_config,
    batch_task_registry=task_registry,
    gateway_approval_store=None,
    gateway_approval_policy=None,
    user_event_bus=_DispatchEventBus(),
    gateway_claim_signing_authority=GatewayClaimSigningAuthority(
      "batch-capability-test-key-at-least-32-bytes"
    ),
    autonomous_storage_root=tmp_path,
    gateway_skill_application=object(),
  )


def test_dispatch_threads_one_pre_materialized_execution_into_task(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  async def _run() -> None:
    service_handle = CredentialHandle(
      handle_id="service:batch:anthropic",
      provider="anthropic",
      principal="service",
      tenant_id="tenant-a",
      actor_id=None,
    )
    materialized_handles: list[CredentialHandle] = []
    materialized_scopes: list[Any] = []
    started = asyncio.Event()
    release = asyncio.Event()
    captured: dict[str, Any] = {}
    build_parent_session = batches._batch_parent_session

    def capture_parent_session(**kwargs: Any) -> GatewaySession:
      session = build_parent_session(**kwargs)
      captured["parent_session"] = session
      return session

    monkeypatch.setattr(batches, "_batch_parent_session", capture_parent_session)

    class _Controller:
      def acquire_batch_run(self, _payload: dict[str, Any], **_kwargs: Any):
        return 17, "alice", None

      async def run_acquired_batch(
        self,
        _batch_id: int,
        _payload: dict[str, Any],
        **kwargs: Any,
      ) -> None:
        captured.update(kwargs)
        started.set()
        await release.wait()
        registry.status = "completed"

      async def open_captured_run_context(
        self,
        **kwargs: Any,
      ) -> dict[str, Any]:
        captured["admission_kwargs"] = kwargs
        return kwargs

    registry = _DispatchRegistry()
    controller = _Controller()
    task_registry = batches.BatchTaskRegistry()
    app_state = _dispatch_app_state(
      tmp_path=tmp_path,
      registry=registry,
      controller=controller,
      task_registry=task_registry,
      gateway_config=_config(
        service_handle=service_handle,
        materialized_handles=materialized_handles,
        materialized_scopes=materialized_scopes,
      ),
    )
    monkeypatch.setattr(
      batches,
      "require_corpus_readiness",
      lambda payload, **_kwargs: asyncio.sleep(
        0,
        result=(payload, None),
      ),
    )

    result = await batches.dispatch_batch_in_process(
      {"source": "quality_screen", "universe": ["MSFT"]},
      app_state=app_state,
      user_id="alice",
      role="invite",
      dispatch_key="batch-capability-binding-test",
    )
    task = task_registry.get(owner_user_id="alice", batch_id=17)
    assert task is not None
    await started.wait()

    resolver = captured["capability_execution_resolver"]
    execution = captured["session_driver_execution"]
    assert execution.bind.capability_id == "session.driver"
    assert execution.bind.run_mode == "batch"
    assert execution.bind.registry_revision == resolver.registry.revision
    assert execution.bind.policy_revision == resolver.selection_policy.revision
    assert materialized_handles == [service_handle]
    assert materialized_scopes == [7]
    parent = root_context(
      captured["parent_session"],
      withheld_tools=frozenset(),
      research_file_id=None,
      selection_provenance=default_root_provenance(resolver.selection_policy),
    )
    assert parent.tenant is not None
    admitted = AdmittedContext(
      wire=AdmittedContextWire(
        tenant=parent.tenant,
        research_binding=parent.research_binding,
        capability_bind=execution.bind,
        children=(),
        tool_grant=(),
        budget_share=None,
        env_projection=(),
        approval_route="no_approval_route",
        handle_grants=(),
        placement="in_process",
        lineage=AdmittedLineage(
          parent_job_id=parent.job_id,
          depth=parent.depth + 1,
        ),
      ),
      tool_authority=ResolvedAuthority(
        operation_name="test-stage",
        grant=ToolGrant(grant_id="test-stage-grant", digest="sha256:" + "0" * 64),
      ),
      approval_route=child_approval_route(parent.approval_route),
    )
    await captured["captured_run_context_factory"](
      task_id="agent-1",
      session_driver_execution=execution,
      admitted=admitted,
    )
    admission_kwargs = captured["admission_kwargs"]
    service_session = admission_kwargs["parent_session"]
    assert type(service_session) is GatewaySession
    assert service_session.role == "invite"
    assert service_session.kind == "control"
    assert service_session.user_id == "alice"
    assert service_session.auth_config == execution.auth_config
    assert service_session.session_credential_handle is None
    assert admission_kwargs["origin"] == "service"
    assert admission_kwargs["run_id"] == "batch_17"
    assert admission_kwargs["task_id"] == "agent-1"
    assert admission_kwargs["storage_root"] == tmp_path
    assert result == {
      "batch_id": 17,
      "status": "running",
      "replayed": False,
    }

    release.set()
    await task
    await asyncio.sleep(0)

  asyncio.run(_run())


def test_authenticated_dispatch_keeps_its_session_token_live_until_terminal(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  async def _run() -> None:
    now = [1_000]
    monkeypatch.setattr("agent_gateway.session.time.time", lambda: now[0])
    started = asyncio.Event()
    release = asyncio.Event()
    registry = _DispatchRegistry()

    class _Controller:
      def acquire_batch_run(self, _payload: dict[str, Any], **_kwargs: Any):
        return 17, "alice", None

      async def run_acquired_batch(
        self,
        _batch_id: int,
        _payload: dict[str, Any],
        **_kwargs: Any,
      ) -> None:
        started.set()
        await release.wait()
        registry.status = "completed"

      async def open_captured_run_context(self, **_kwargs: Any) -> object:
        return object()

    store = SessionStore(ttl=900)
    auth = AuthManager(
      secret="batch-liveness-test-secret-at-least-32-bytes",
      valid_keys={"gateway-key"},
      session_store=store,
    )
    session = store.create_session(
      api_key_hash=hashlib.sha256(b"batch-api-key").hexdigest(),
      user_id="alice",
      owner_user_id="alice",
      kind="control",
      risk_user_id=7,
      auth_config={"provider": "anthropic", "api_key": "user-secret"},
      tenant_id="tenant-a",
      session_credential_handle=CredentialHandle(
        handle_id="user:alice:anthropic",
        provider="anthropic",
        principal="user",
        tenant_id="tenant-a",
        actor_id="alice",
      ),
    )
    token = auth.issue_token(session)
    task_registry = batches.BatchTaskRegistry()
    controller = _Controller()
    app_state = _dispatch_app_state(
      tmp_path=tmp_path,
      registry=registry,
      controller=controller,
      task_registry=task_registry,
      gateway_config=_config(),
    )
    monkeypatch.setattr(
      batches,
      "require_corpus_readiness",
      lambda payload, **_kwargs: asyncio.sleep(
        0,
        result=(payload, None),
      ),
    )

    await batches.dispatch_batch_in_process(
      {"source": "quality_screen", "universe": ["MSFT"]},
      app_state=app_state,
      user_id="alice",
      dispatch_key="batch-liveness-test",
      authenticated_session=session,
    )
    await started.wait()
    assert session.admitted_batches == {("alice", 17)}

    # The batch's stages carry this token past the session's TTL.
    now[0] = session.expires_at + 1
    assert auth.verify_token(token) is session

    release.set()
    while task_registry.has_admitted_batch(owner_user_id="alice", batch_id=17):
      await asyncio.sleep(0)
    assert session.admitted_batches == set()
    assert auth.verify_token(token) is session

  asyncio.run(_run())


def test_sessionless_batch_dispatch_without_role_fails_closed() -> None:
  with pytest.raises(
    ValueError,
    match="role must be exactly 'owner' or 'invite'",
  ):
    asyncio.run(
      batches.dispatch_batch_in_process(
        {"source": "quality_screen", "universe": ["MSFT"]},
        app_state=SimpleNamespace(),
        user_id="alice",
        dispatch_key="roleless-batch-dispatch",
      )
    )


def test_batch_credential_unavailable_remains_a_503(monkeypatch: pytest.MonkeyPatch) -> None:
  def _unavailable(**_kwargs: Any) -> None:
    raise CapabilityResolutionError(
      "credential_unavailable", "no active credential", capability_id="session.driver",
    )

  registry = SimpleNamespace(
    lookup_batch_dispatch=lambda **_kwargs: None,
    close=lambda: None,
  )
  backend = SimpleNamespace(registry_for_user=lambda _user_id: registry)
  monkeypatch.setattr(batches, "_batch_backend", lambda _state: backend)
  monkeypatch.setattr(batches, "_task_registry_for_state", lambda _state: SimpleNamespace())
  async def _no_reconcile(**_kwargs: Any) -> None:
    return None

  monkeypatch.setattr(batches, "_reconcile_active_batches_before_fresh_admission", _no_reconcile)
  monkeypatch.setattr(batches, "_batch_capability_execution_context", _unavailable)
  with pytest.raises(batches._BatchDispatchNotAdmitted) as raised:
    asyncio.run(
      batches.dispatch_batch_in_process(
        {"source": "quality_screen", "universe": ["MSFT"]},
        app_state=SimpleNamespace(),
        user_id="1",
        role="owner",
        dispatch_key="credential-unavailable-503",
      )
    )
  assert raised.value.status_code == 503


def _batch_bind_for_handle(handle: CredentialHandle) -> CapabilityBind:
  auth = AuthContext(
    run_mode="batch",
    actor_id="alice",
    tenant_id="tenant-a",
    user_provider_handles={handle.provider: handle} if handle.principal == "user" else {},
    service_provider_handles={handle.provider: handle} if handle.principal == "service" else {},
    entitled_capabilities=frozenset({"session.driver"}),
    entitled_model_keys=INITIAL_MODEL_SELECTION_POLICY.capabilities["session.driver"].allowed_model_keys,
    run_scoped_user_providers=frozenset({handle.provider}) if handle.principal == "user" else frozenset(),
  )
  entry = next(
    entry for entry in INITIAL_MODEL_REGISTRY.models.values()
    if entry.provider == handle.provider and entry.lifecycle in {"active", "hidden"}
    and "session.driver" in entry.capabilities
  )
  if entry.lifecycle == "active":
    return resolve_capability_model(
      "session.driver", auth=auth,
      registry=INITIAL_MODEL_REGISTRY,
      selection_policy=INITIAL_MODEL_SELECTION_POLICY,
      explicit_intent=ModelSelectionIntent(
        model_key=entry.key, effort=entry.default_effort, source="explicit_user",
      ),
    )
  return resolve_capability_model(
    "session.driver", auth=auth,
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    role_map={"default": TierModelsDefault(
      models=(entry.key,),
      effort="none" if "none" in entry.supported_efforts else entry.default_effort,
    )},
    role_map_source="internal_policy",
  )


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_batch_unenrolled_session_user_keeps_deployment_handle_without_a_lease(
  monkeypatch: pytest.MonkeyPatch,
  provider: str,
) -> None:
  pools = _install_batch_pools(monkeypatch, enrolled=frozenset())
  config, deployment_handle = _deployment_config(provider)
  session = _deployment_session(deployment_handle, provider)

  resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=config),
    user_id="alice", authenticated_session=session,
    required_bind=_batch_bind_for_handle(deployment_handle),
  )

  assert resolver.user_key_handles == {}
  assert resolver.auth_context.service_provider_handles[provider] is deployment_handle
  assert provider not in resolver.auth_context.user_provider_handles
  assert execution.bind.credential_principal == "service"
  assert execution.auth_config["api_key"] == "deployment-secret"
  assert {call["provider"] for call in pools.probes} == {"anthropic", "openai"}
  assert pools.leases == []


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("outage", ["resolver_unavailable", "not_installed"])
def test_batch_risk_outage_is_not_enrolled_and_serves_the_deployment_handle(
  monkeypatch: pytest.MonkeyPatch,
  provider: str,
  outage: str,
) -> None:
  pools = (
    _install_batch_pools(monkeypatch, enrolled=frozenset(), outage=True)
    if outage == "resolver_unavailable"
    else _uninstall_batch_pools(monkeypatch)
  )
  config, deployment_handle = _deployment_config(provider)
  session = _deployment_session(deployment_handle, provider)

  resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=config),
    user_id="alice", authenticated_session=session,
    required_bind=_batch_bind_for_handle(deployment_handle),
  )

  assert resolver.user_key_handles == {}
  assert resolver.auth_context.service_provider_handles[provider] is deployment_handle
  assert execution.auth_config["api_key"] == "deployment-secret"
  assert pools.leases == []


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
@pytest.mark.parametrize("with_session", [False, True])
def test_batch_enrolled_user_leases_its_key_and_never_uses_the_deployment_handle(
  monkeypatch: pytest.MonkeyPatch,
  provider: str,
  with_session: bool,
) -> None:
  pools = _install_batch_pools(monkeypatch, enrolled=frozenset({provider}))
  config, deployment_handle = _deployment_config(provider)
  session = _deployment_session(deployment_handle, provider) if with_session else None
  key_handle = CredentialHandle(
    handle_id=f"user-key:7:{provider}", provider=provider,
    principal="user", tenant_id="tenant-a", actor_id="alice",
  )

  resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=config),
    user_id="alice", authenticated_session=session,
    required_bind=_batch_bind_for_handle(key_handle),
  )

  assert provider not in resolver.auth_context.service_provider_handles
  assert resolver.auth_context.user_provider_handles[provider] == key_handle
  # A session offers only enrolled providers; sessionless dispatch has no
  # deployment fallback, so every provider stays a user-key handle (N2).
  assert set(resolver.user_key_handles) == (
    {provider} if with_session else {"anthropic", "openai"}
  )
  assert execution.bind.credential_ref == key_handle.handle_id
  assert execution.auth_config["api_key"] == "user-key-secret"
  assert execution.auth_config["billing_mode"] == "byok"
  assert resolver.risk_user_id == 7
  assert [call["provider"] for call in pools.leases] == [provider]
  assert pools.leases[0]["risk_user_id"] == 7
  assert pools.leases[0]["context"].consumer_key_hash == (
    consumer_key_hash_for("risk-api-key") if with_session else NO_API_KEY_CONSUMER
  )


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_batch_sessionless_dispatch_keeps_the_typed_credential_error_not_the_deployment_handle(
  monkeypatch: pytest.MonkeyPatch,
  provider: str,
) -> None:
  pools = _install_batch_pools(monkeypatch, enrolled=frozenset())
  config, deployment_handle = _deployment_config(provider)
  key_handle = CredentialHandle(
    handle_id=f"user-key:7:{provider}", provider=provider,
    principal="user", tenant_id="tenant-a", actor_id="alice",
  )

  with pytest.raises(CapabilityResolutionError, match="connect your .* API key") as refused:
    batches._batch_capability_execution_context(
      app_state=SimpleNamespace(gateway_config=config),
      user_id="alice", authenticated_session=None,
      required_bind=_batch_bind_for_handle(key_handle),
    )
  assert refused.value.code == "credential_unavailable"
  assert isinstance(refused.value.__cause__, UserKeyMissing)
  assert pools.probes == []
  assert pools.leases[0]["provider"] == provider
  assert pools.leases[0]["context"].consumer_key_hash == NO_API_KEY_CONSUMER


@pytest.mark.parametrize("with_session", [False, True])
def test_batch_unmapped_identity_has_typed_refusal(
  monkeypatch: pytest.MonkeyPatch,
  with_session: bool,
) -> None:
  monkeypatch.setenv("GATEWAY_USER_KEYS", "[]")
  session = GatewaySession(
    session_id="unmapped", api_key_hash=hashlib.sha256(b"batch-api-key").hexdigest(), created_at=1, expires_at=2,
    user_id="alice", owner_user_id="alice", tenant_id="tenant-a", kind="control",
  ) if with_session else None
  with pytest.raises(CapabilityResolutionError) as refused:
    batches._batch_capability_execution_context(
      app_state=SimpleNamespace(gateway_config=_config()),
      user_id="alice", authenticated_session=session,
    )
  if with_session:
    # A session's legacy 0 identity names no user key to lease: no handle
    # reaches the resolver, so it refuses by selection, never by a lease.
    assert refused.value.code == "default_not_eligible"
    return
  assert refused.value.code == "credential_unavailable"
  assert isinstance(refused.value.__cause__, IdentityUnresolved)
  assert str(refused.value) == refused.value.__cause__.user_message


@pytest.mark.parametrize("provider", ["xai", "google", "kimi_code", "muse", "ollama_cloud", "anthropic_bedrock"])
def test_batch_preserves_r4_and_admitted_bedrock_service_material(
  provider: str,
) -> None:
  handle = CredentialHandle(
    handle_id=f"service:batch:{provider}", provider=provider,
    principal="service", tenant_id="tenant-a", actor_id=None,
  )
  seen: list[int] = []
  config = _config()
  config.service_provider_handles = {provider: handle}

  def materialize(selected: CredentialHandle, *, user_scope: int) -> MaterializedCredential:
    assert selected is handle
    seen.append(user_scope)
    return MaterializedCredential(handle=selected, auth_config={
      "provider": provider, "api_key": "existing-service-secret",
    })

  config.service_auth_config_resolver = materialize
  config.service_admission = lambda selected, *, risk_user_id, auth_config=None: (
    provider == "anthropic_bedrock" and selected is handle and risk_user_id == 7
  )
  config.capability_adapter_resolver = lambda _adapter: _Provider(provider)
  resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(gateway_config=config),
    user_id="alice", authenticated_session=None,
    required_bind=_batch_bind_for_handle(handle),
  )
  assert resolver.auth_context.service_provider_handles[provider] is handle
  assert execution.auth_config["api_key"] == "existing-service-secret"
  assert seen == [7]


@pytest.mark.parametrize(
  ("launch_value", "flipped"), [("ask_all", "approve_all"), ("approve_all", "ask_all")],
)
def test_batch_keeps_its_launch_approval_settings_after_the_operator_file_flips(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
  launch_value: str,
  flipped: str,
) -> None:
  """A headless batch decides under the snapshot taken when its row was created."""

  from agent_gateway.run_settings import set_operator_setting
  from agent_gateway.single_user_policy import current_approval_settings

  async def _run() -> None:
    monkeypatch.setenv("HANK_RUN_SETTINGS_PATH", str(tmp_path / "operator.json"))
    set_operator_setting("approval_policy", launch_value)
    launch_record = {
      "approval_policy": launch_value,
      "approve_live_orders": False,
    }
    seen: list[dict[str, object]] = []
    service_handle = CredentialHandle(
      handle_id="service:batch:anthropic",
      provider="anthropic",
      principal="service",
      tenant_id="tenant-a",
      actor_id=None,
    )
    registry = _DispatchRegistry()
    registry.launch_record = launch_record

    def read_persisted_launch(_batch_id: int) -> dict[str, object]:
      # The row is the launch snapshot. Flipping the file in this read is the
      # mid-batch change: a pin that re-snapshots the live file binds the flip.
      set_operator_setting("approval_policy", flipped)
      return dict(launch_record)

    registry.launch_approval_settings = read_persisted_launch

    class _Controller:
      def acquire_batch_run(self, _payload: dict[str, Any], **_kwargs: Any):
        return 17, "alice", None

      async def run_acquired_batch(self, *_args: Any, **_kwargs: Any) -> None:
        async def stage_body() -> None:
          seen.append(current_approval_settings().record())

        await stage_body()
        registry.status = "completed"

      async def open_captured_run_context(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    task_registry = batches.BatchTaskRegistry()
    app_state = _dispatch_app_state(
      tmp_path=tmp_path,
      registry=registry,
      controller=_Controller(),
      task_registry=task_registry,
      gateway_config=_config(service_handle=service_handle),
    )
    monkeypatch.setattr(
      batches,
      "require_corpus_readiness",
      lambda payload, **_kwargs: asyncio.sleep(0, result=(payload, None)),
    )
    await batches.dispatch_batch_in_process(
      {"source": "quality_screen", "universe": ["MSFT"]},
      app_state=app_state,
      user_id="alice",
      role="invite",
      dispatch_key="batch-approval-pin",
    )
    task = task_registry.get(owner_user_id="alice", batch_id=17)
    assert task is not None
    await task
    assert seen == [launch_record]
    assert current_approval_settings().approval_policy == flipped

  asyncio.run(_run())
