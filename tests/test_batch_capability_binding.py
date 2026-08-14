from __future__ import annotations

# ruff: noqa: E402

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
for path in (ROOT, PKG_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent_gateway.capability_binding import (
  AuthContext,
  CredentialHandle,
  ModelSelectionIntent,
  resolve_capability_model,
)
from agent_gateway.capability_execution import MaterializedCredential
from agent_gateway.claim_signing_authority import (
  GatewayClaimSigningAuthority,
)
from agent_gateway.control_plane import batches
from agent_gateway.providers import ModelInfo, ModelProvider
from agent_gateway.model_registry import (
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.session import GatewaySession


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
    api_key_hash="hash",
    created_at=int(time.time()),
    expires_at=int(time.time()) + 60,
    user_id="alice",
    owner_user_id="alice",
    kind="control",
    auth_config=auth_config,
    tenant_id="tenant-a",
    session_credential_handle=handle,
  )


def _config(
  *,
  service_handle: CredentialHandle | None = None,
  materialized_handles: list[CredentialHandle] | None = None,
) -> SimpleNamespace:
  provider = _Provider()
  material_by_identity = (
    {
      id(service_handle): MaterializedCredential(
        handle=service_handle,
        auth_config={
          "provider": "anthropic",
          "api_key": "service-secret",
        },
      )
    }
    if service_handle is not None
    else {}
  )

  def _materialize(handle: CredentialHandle) -> MaterializedCredential:
    if materialized_handles is not None:
      materialized_handles.append(handle)
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
    capability_adapter_resolver=lambda adapter: (
      provider
      if adapter == "anthropic.messages"
      else pytest.fail(f"unexpected adapter: {adapter}")
    ),
    default_provider=provider,
  )


def test_authenticated_batch_marks_exact_user_handle_run_scoped() -> None:
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
    "anthropic"
  })
  assert execution.bind.credential_principal == "user"
  assert execution.auth_config["api_key"] == "user-secret"


def test_standalone_batch_materializes_only_service_principal() -> None:
  service_handle = CredentialHandle(
    handle_id="service:batch:anthropic",
    provider="anthropic",
    principal="service",
    tenant_id="tenant-a",
    actor_id=None,
  )

  resolver, execution = batches._batch_capability_execution_context(
    app_state=SimpleNamespace(
      gateway_config=_config(service_handle=service_handle)
    ),
    user_id="alice",
    authenticated_session=None,
  )

  assert resolver.auth_context.user_provider_handles == {}
  assert (
    resolver.auth_context.service_provider_handles["anthropic"]
    is service_handle
  )
  assert execution.bind.credential_principal == "service"
  assert execution.auth_config["api_key"] == "service-secret"


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
    api_key_hash="hash",
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


def test_authenticated_cross_provider_batch_preserves_session_selection() -> None:
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
      model_key="openai.gpt-5-6",
      effort="medium",
      source="explicit_user",
    ),
  )
  gateway_config = SimpleNamespace(
    tenant_id="tenant-a",
    model_selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    model_registry=INITIAL_MODEL_REGISTRY,
    service_provider_handles={"anthropic": service_handle},
    service_auth_config_resolver=lambda handle: (
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
    "openai"
  })
  assert execution.bind.provider == "openai"
  assert execution.bind.model_key == "openai.gpt-5-6"
  assert execution.bind.upstream_model == "gpt-5.6"
  assert execution.bind.credential_principal == "user"
  assert execution.auth_config["api_key"] == "user-secret"


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
    started = asyncio.Event()
    release = asyncio.Event()
    captured: dict[str, Any] = {}

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

      async def admit_in_process_runtime_authority(
        self,
        **kwargs: Any,
      ) -> object:
        captured["admission_kwargs"] = kwargs
        return object()

    class _Registry:
      def __init__(self) -> None:
        self.status = "running"

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

    class _EventBus:
      async def publish_terminal_if_absent(
        self,
        _user_id: str,
        _control_run_id: str,
        _event: dict[str, Any],
      ) -> bool:
        return True

    registry = _Registry()
    task_registry = batches.BatchTaskRegistry()
    app_state = SimpleNamespace(
      gateway_config=_config(
        service_handle=service_handle,
        materialized_handles=materialized_handles,
      ),
      batch_task_registry=task_registry,
      gateway_approval_store=None,
      gateway_approval_policy=None,
      user_event_bus=_EventBus(),
      gateway_claim_signing_authority=GatewayClaimSigningAuthority(
        "batch-capability-test-key-at-least-32-bytes"
      ),
      autonomous_storage_root=tmp_path,
    )
    controller = _Controller()
    monkeypatch.setattr(batches, "_controller", lambda: controller)
    monkeypatch.setattr(
      batches,
      "_registry_for_user",
      lambda _user_id: registry,
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
    admission = await captured["captured_run_admission_factory"](
      task_id="agent-1",
      session_driver_execution=execution,
    )
    assert type(admission) is object
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
