from __future__ import annotations

from typing import Any

import pytest

from agent_gateway.capability_binding import (
  AuthContext,
  CapabilityResolutionError,
  CredentialHandle,
  ModelSelectionIntent,
)
from agent_gateway.capability_execution import (
  BoundCapabilityExecution,
  CapabilityExecutionResolver,
  MaterializedCredential,
  derive_batch_capability_execution,
)
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_gateway.providers import ModelInfo, ModelProvider


class _ExactProvider(ModelProvider):
  def __init__(self, name: str) -> None:
    self.name = name

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    return bool(config.get("api_key") or config.get("auth_token"))

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=64_000,
      supports_thinking=True,
    )


def _handle(
  provider: str = "anthropic",
  *,
  principal: str = "user",
) -> CredentialHandle:
  return CredentialHandle(
    handle_id=f"{principal}:tenant:{provider}",
    provider=provider,
    principal=principal,
    tenant_id="tenant",
    actor_id="alice" if principal == "user" else None,
  )


def _auth(
  *,
  providers: tuple[str, ...] = ("anthropic",),
  run_mode: str = "interactive",
  principal: str = "user",
) -> AuthContext:
  handles = {provider: _handle(provider, principal=principal) for provider in providers}
  return AuthContext(
    run_mode=run_mode,
    actor_id="alice",
    tenant_id="tenant",
    user_provider_handles=handles if principal == "user" else {},
    service_provider_handles=handles if principal == "service" else {},
    entitled_capabilities=CAPABILITY_IDS,
    entitled_model_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
  )


def _resolver(
  *,
  auth: AuthContext | None = None,
  providers: dict[str, _ExactProvider] | None = None,
  materializer=None,
  overrides: dict[str, ModelSelectionIntent] | None = None,
) -> CapabilityExecutionResolver:
  resolved_auth = auth or _auth()
  provider_map = providers or {
    family: _ExactProvider(family)
    for family in {"anthropic", "codex", "openai", "xai"}
  }
  handles = {
    **dict(resolved_auth.user_provider_handles),
    **dict(resolved_auth.service_provider_handles),
  }

  def _materialize(handle: CredentialHandle) -> MaterializedCredential:
    assert handles[handle.provider] is handle
    return MaterializedCredential(
      handle=handle,
      auth_config={"api_key": "test-secret", "auth_mode": "api"},
    )

  return CapabilityExecutionResolver(
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth_context=resolved_auth,
    credential_materializer=materializer or _materialize,
    adapter_resolver=lambda adapter: provider_map[
      INITIAL_MODEL_REGISTRY.models[
        next(
          key
          for key, entry in INITIAL_MODEL_REGISTRY.models.items()
          if entry.adapter == adapter
        )
      ].provider
    ],
    authenticated_run_overrides=overrides or {},
  )


def test_resolver_materializes_one_complete_immutable_execution() -> None:
  execution = _resolver().resolve("session.driver")

  assert execution.bind.model_key == "anthropic.claude-opus-5"
  assert execution.bind.adapter == "anthropic.messages"
  assert execution.registry is INITIAL_MODEL_REGISTRY
  assert execution.adapter.name == "anthropic"
  assert execution.auth_config == {
    "api_key": "test-secret",
    "auth_mode": "api",
  }
  with pytest.raises(TypeError):
    execution.auth_config["api_key"] = "changed"  # type: ignore[index]


def test_materializer_cannot_change_selected_credential_handle() -> None:
  replacement = CredentialHandle(
    handle_id="user:tenant:anthropic:replacement",
    provider="anthropic",
    principal="user",
    tenant_id="tenant",
    actor_id="alice",
  )

  def _replace(_handle: CredentialHandle) -> MaterializedCredential:
    return MaterializedCredential(
      handle=replacement,
      auth_config={"api_key": "test-secret"},
    )

  with pytest.raises(CapabilityResolutionError) as refused:
    _resolver(materializer=_replace).resolve("session.driver")
  assert refused.value.code == "credential_unavailable"


def test_materializer_expected_failure_is_secret_free_typed_refusal() -> None:
  def _missing(_handle: CredentialHandle) -> MaterializedCredential:
    raise LookupError("credential secret must not escape")

  with pytest.raises(CapabilityResolutionError) as refused:
    _resolver(materializer=_missing).resolve("session.driver")
  assert refused.value.code == "credential_unavailable"
  assert "secret" not in str(refused.value)


def test_materializer_unexpected_exception_is_not_a_credential_refusal() -> None:
  """A programming/configuration error must not masquerade as a typed refusal."""

  def _explode(_handle: CredentialHandle) -> MaterializedCredential:
    raise RuntimeError("materializer misconfigured")

  with pytest.raises(RuntimeError, match="materializer misconfigured"):
    _resolver(materializer=_explode).resolve("session.driver")


def test_adapter_resolver_unexpected_exception_propagates_unwrapped() -> None:
  resolver = _resolver()
  object.__setattr__(
    resolver,
    "adapter_resolver",
    lambda adapter: (_ for _ in ()).throw(RuntimeError("adapter wiring bug")),
  )

  with pytest.raises(RuntimeError, match="adapter wiring bug"):
    resolver.resolve("session.driver")


def test_adapter_resolver_failure_is_typed_before_runtime_dispatch() -> None:
  resolver = _resolver(providers={})
  object.__setattr__(
    resolver,
    "adapter_resolver",
    lambda adapter: (_ for _ in ()).throw(LookupError(adapter)),
  )

  with pytest.raises(CapabilityResolutionError) as refused:
    resolver.resolve("session.driver")
  assert refused.value.code == "provider_unavailable"
  assert refused.value.model_key == "anthropic.claude-opus-5"


def test_bound_execution_rejects_provider_family_mismatch() -> None:
  execution = _resolver().resolve("session.driver")

  with pytest.raises(CapabilityResolutionError) as refused:
    BoundCapabilityExecution(
      bind=execution.bind,
      registry=execution.registry,
      adapter=_ExactProvider("openai"),
      auth_config=execution.auth_config,
    )
  assert refused.value.code == "provider_unavailable"


def test_bound_execution_rejects_bind_identity_not_admitted_by_registry() -> None:
  execution = _resolver().resolve("session.driver")
  bind = execution.bind.model_copy(update={"upstream_model": "claude-sonnet-5"})

  with pytest.raises(ValueError, match="execution identity does not match"):
    BoundCapabilityExecution(
      bind=bind,
      registry=execution.registry,
      adapter=execution.adapter,
      auth_config=execution.auth_config,
    )


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("model", "claude-opus-5"),
    ("model_key", "anthropic.claude-opus-5"),
    ("effort", "high"),
    ("execution_transport", "native"),
    ("thinking", True),
    ("thinking_enabled_requested", True),
  ],
)
def test_bound_execution_rejects_selection_fields_in_auth_config(
  field: str,
  value: object,
) -> None:
  execution = _resolver().resolve("session.driver")
  config = dict(execution.auth_config)
  config[field] = value

  with pytest.raises(ValueError, match="must not contain model-selection fields"):
    BoundCapabilityExecution(
      bind=execution.bind,
      registry=execution.registry,
      adapter=execution.adapter,
      auth_config=config,
    )


@pytest.mark.parametrize(
  "field",
  [
    "model",
    "model_key",
    "effort",
    "execution_transport",
    "thinking",
    "thinking_enabled_requested",
  ],
)
def test_materialized_credential_rejects_selection_fields(field: str) -> None:
  with pytest.raises(ValueError, match="must not contain model-selection fields"):
    MaterializedCredential(
      handle=_handle(),
      auth_config={field: "forbidden"},
    )


def test_materialize_durable_bind_reauthorizes_without_reselection() -> None:
  resolver = _resolver()
  initial = resolver.resolve(
    "session.driver",
    explicit_intent=ModelSelectionIntent(
      model_key="anthropic.claude-sonnet-5",
      effort="xhigh",
      source="explicit_user",
    ),
  )

  resumed = resolver.materialize_bind(initial.bind)
  assert resumed.bind is initial.bind
  assert resumed.bind.model_key == "anthropic.claude-sonnet-5"
  assert resumed.bind.effort == "xhigh"


def test_materialize_bind_refuses_changed_run_mode() -> None:
  resolver = _resolver()
  bind = resolver.resolve("session.driver").bind.model_copy(
    update={"run_mode": "batch"}
  )

  with pytest.raises(CapabilityResolutionError) as refused:
    resolver.materialize_bind(bind)
  assert refused.value.code == "credential_unavailable"


def test_authenticated_run_override_is_policy_scoped() -> None:
  override = ModelSelectionIntent(
    model_key="anthropic.claude-fable-5",
    effort="high",
    source="explicit_user",
  )
  resolver = _resolver(overrides={"plan.author": override})
  assert resolver.resolve("plan.author").bind.model_key == override.model_key

  with pytest.raises(CapabilityResolutionError):
    _resolver(overrides={"node.explore": override})

  assert INITIAL_MODEL_SELECTION_POLICY.capabilities[
    "plan.author"
  ].allow_authenticated_run_override is True
  assert all(
    not INITIAL_MODEL_SELECTION_POLICY.capabilities[
      capability_id
    ].allow_authenticated_run_override
    for capability_id in (
      "node.explore",
      "node.implement",
      "node.verify",
    )
  )


def test_exact_parent_binding_is_passed_to_plan_author_resolution() -> None:
  resolver = _resolver()
  parent = resolver.resolve(
    "session.driver",
    explicit_intent=ModelSelectionIntent(
      model_key="anthropic.claude-sonnet-5",
      effort="high",
      source="explicit_user",
    ),
  )
  author = resolver.resolve("plan.author", parent_bind=parent.bind)

  assert author.bind.model_key == parent.bind.model_key
  assert author.bind.effort == parent.bind.effort
  assert author.bind.selection_source == "parent_binding"


def test_standard_protocol_profile_cannot_gain_reasoning_effort() -> None:
  resolver = _resolver()
  execution = resolver.resolve("risk.completion")
  assert execution.bind.protocol_profile == "messages.standard"
  assert execution.bind.effort == "none"
  assert not set(execution.auth_config) & {
    "model",
    "model_key",
    "effort",
    "execution_transport",
    "thinking",
    "thinking_enabled_requested",
  }

  tampered = execution.bind.model_copy(update={"effort": "high"})
  with pytest.raises(CapabilityResolutionError) as refused:
    BoundCapabilityExecution(
      bind=tampered,
      registry=execution.registry,
      adapter=execution.adapter,
      auth_config=execution.auth_config,
    )
  assert refused.value.code == "capability_effort_unsupported"


def test_batch_derivation_preserves_exact_bind_and_scopes_user_handle() -> None:
  resolver = _resolver()
  interactive = resolver.resolve("session.driver")

  batch_resolver, batch = derive_batch_capability_execution(
    resolver,
    interactive,
  )

  assert batch.bind == interactive.bind.model_copy(update={"run_mode": "batch"})
  assert batch.bind.credential_ref == interactive.bind.credential_ref
  assert "anthropic" in batch_resolver.auth_context.run_scoped_user_providers


def test_batch_derivation_narrows_user_scope_to_the_inherited_binds_provider() -> None:
  """Only the bind's own user credential crosses into the batch authorization."""

  resolver = _resolver(auth=_auth(providers=("anthropic", "openai", "xai")))
  interactive = resolver.resolve("session.driver")
  assert interactive.bind.credential_principal == "user"
  assert interactive.bind.provider == "anthropic"

  batch_resolver, batch = derive_batch_capability_execution(
    resolver,
    interactive,
  )

  assert batch_resolver.auth_context.run_scoped_user_providers == frozenset(
    {"anthropic"}
  )
  assert set(batch_resolver.auth_context.user_provider_handles) == {"anthropic"}
  assert batch.bind.run_mode == "batch"


def test_batch_derivation_from_service_parent_carries_no_user_handles() -> None:
  auth = AuthContext(
    run_mode="interactive",
    actor_id="alice",
    tenant_id="tenant",
    user_provider_handles={"openai": _handle("openai")},
    service_provider_handles={
      "anthropic": _handle("anthropic", principal="service"),
    },
    entitled_capabilities=CAPABILITY_IDS,
    entitled_model_keys=frozenset(INITIAL_MODEL_REGISTRY.models),
    allow_service_for_interactive=True,
  )
  resolver = _resolver(auth=auth)
  interactive = resolver.resolve("session.driver")
  assert interactive.bind.credential_principal == "service"

  batch_resolver, batch = derive_batch_capability_execution(
    resolver,
    interactive,
  )

  assert batch_resolver.auth_context.run_scoped_user_providers == frozenset()
  assert dict(batch_resolver.auth_context.user_provider_handles) == {}
  assert batch.bind.credential_principal == "service"
  assert batch.bind.run_mode == "batch"


def test_batch_service_default_uses_service_principal() -> None:
  resolver = _resolver(
    auth=_auth(run_mode="batch", principal="service"),
  )
  execution = resolver.resolve("risk.document_ingest")
  assert execution.bind.credential_principal == "service"
  assert execution.bind.adapter == "anthropic.sdk.messages"
