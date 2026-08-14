from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_gateway.capability_binding import (
  AuthContext,
  CapabilityResolutionError,
  CredentialHandle,
  ModelSelectionIntent,
  eligible_model_choices,
  reauthorize_capability_bind,
  resolve_capability_model,
  validate_reported_identity,
)
from agent_gateway.model_registry import (
  CAPABILITY_IDS,
  INITIAL_ADAPTER_ROUTE_SUPPORT,
  INITIAL_MODEL_REGISTRY,
  INITIAL_MODEL_SELECTION_POLICY,
)
from agent_workflow_contracts import CapabilityBind


def _user_handle(provider: str) -> CredentialHandle:
  return CredentialHandle(
    handle_id=f"user:tenant:alice:{provider}",
    provider=provider,
    principal="user",
    tenant_id="tenant",
    actor_id="alice",
  )


def _service_handle(provider: str) -> CredentialHandle:
  return CredentialHandle(
    handle_id=f"service:tenant:{provider}",
    provider=provider,
    principal="service",
    tenant_id="tenant",
    actor_id=None,
  )


def _auth(
  *,
  run_mode: str = "interactive",
  providers: tuple[str, ...] = ("anthropic",),
  capabilities: frozenset[str] = CAPABILITY_IDS,
  model_keys: frozenset[str] | None = None,
  service: bool = False,
) -> AuthContext:
  handles = {
    provider: (
      _service_handle(provider)
      if service
      else _user_handle(provider)
    )
    for provider in providers
  }
  return AuthContext(
    run_mode=run_mode,
    actor_id="alice",
    tenant_id="tenant",
    user_provider_handles={} if service else handles,
    service_provider_handles=handles if service else {},
    entitled_capabilities=capabilities,
    entitled_model_keys=(
      frozenset(INITIAL_MODEL_REGISTRY.models)
      if model_keys is None
      else model_keys
    ),
  )


def _resolve(
  capability_id: str,
  *,
  auth: AuthContext | None = None,
  explicit_intent: ModelSelectionIntent | None = None,
  saved_preference: ModelSelectionIntent | None = None,
  authenticated_run_override: ModelSelectionIntent | None = None,
  parent_bind: CapabilityBind | None = None,
) -> CapabilityBind:
  return resolve_capability_model(
    capability_id,
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth=auth or _auth(),
    explicit_intent=explicit_intent,
    saved_preference=saved_preference,
    authenticated_run_override=authenticated_run_override,
    parent_bind=parent_bind,
  )


def test_initial_registry_is_closed_over_installed_adapter_support() -> None:
  INITIAL_MODEL_REGISTRY.admit_adapter_support(INITIAL_ADAPTER_ROUTE_SUPPORT)
  INITIAL_MODEL_SELECTION_POLICY.admit_registry(INITIAL_MODEL_REGISTRY)
  assert len(INITIAL_MODEL_REGISTRY.models) == 16
  assert set(INITIAL_MODEL_SELECTION_POLICY.capabilities) == CAPABILITY_IDS


def test_initial_registry_has_ten_user_selectable_session_models() -> None:
  selectable = {
    entry.key
    for entry in INITIAL_MODEL_REGISTRY.models.values()
    if entry.capabilities.get("session.driver") == "user_selectable"
  }
  assert selectable == {
    "anthropic.claude-fable-5",
    "anthropic.claude-haiku-4-5",
    "anthropic.claude-mythos-5",
    "anthropic.claude-opus-5",
    "anthropic.claude-sonnet-5",
    "codex.gpt-5-6-luna",
    "codex.gpt-5-6-sol",
    "codex.gpt-5-6-terra",
    "openai.gpt-5-6",
    "xai.grok-4-5",
  }
  assert not any("gpt-4-1" in key or "gpt-4o-mini" in key for key in selectable)


def test_default_session_binding_freezes_complete_execution_identity() -> None:
  bind = _resolve("session.driver")

  assert bind == CapabilityBind(
    schema_version="1.0",
    capability_id="session.driver",
    model_key="anthropic.claude-opus-5",
    provider="anthropic",
    upstream_model="claude-opus-5",
    adapter="anthropic.messages",
    protocol_profile="messages.adaptive",
    route="anthropic.public",
    effort="high",
    credential_principal="user",
    credential_ref="user:tenant:alice:anthropic",
    run_mode="interactive",
    registry_revision=INITIAL_MODEL_REGISTRY.revision,
    policy_revision=INITIAL_MODEL_SELECTION_POLICY.revision,
    selection_source="capability_default",
  )
  assert CapabilityBind.from_receipt(bind.receipt()) == bind


def test_wire_binding_has_no_permissive_legacy_fields() -> None:
  receipt = _resolve("session.driver").receipt()
  receipt["model"] = receipt["upstream_model"]
  receipt["policy_id"] = "legacy"

  with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
    CapabilityBind.from_receipt(receipt)


def test_explicit_intent_uses_stable_key_and_exact_effort() -> None:
  bind = _resolve(
    "session.driver",
    auth=_auth(providers=("openai",)),
    explicit_intent=ModelSelectionIntent(
      model_key="openai.gpt-5-6",
      effort="xhigh",
      source="explicit_user",
      catalog_revision=INITIAL_MODEL_REGISTRY.revision,
    ),
  )

  assert (
    bind.model_key,
    bind.provider,
    bind.upstream_model,
    bind.effort,
    bind.selection_source,
  ) == (
    "openai.gpt-5-6",
    "openai",
    "gpt-5.6",
    "xhigh",
    "explicit_user",
  )


@pytest.mark.parametrize(
  "selector",
  ["openai:gpt-5.6", "gpt-5.6", "anthropic:claude-opus-5"],
)
def test_provider_or_upstream_selectors_are_not_model_keys(selector: str) -> None:
  with pytest.raises(CapabilityResolutionError) as refused:
    _resolve(
      "session.driver",
      explicit_intent=ModelSelectionIntent(
        model_key=selector,
        effort=None,
        source="explicit_user",
      ),
    )

  assert refused.value.code == "capability_model_unavailable"
  assert refused.value.receipt() == {
    "error_code": "capability_model_unavailable",
    "capability_id": "session.driver",
    "model_key": selector,
  }


def test_explicit_unsupported_effort_refuses_before_credential_selection() -> None:
  with pytest.raises(CapabilityResolutionError) as refused:
    _resolve(
      "session.driver",
      explicit_intent=ModelSelectionIntent(
        model_key="anthropic.claude-haiku-4-5",
        effort="low",
        source="explicit_user",
      ),
    )

  assert refused.value.code == "capability_effort_unsupported"
  assert refused.value.model_key == "anthropic.claude-haiku-4-5"


def test_explicit_selection_requires_independent_capability_entitlement() -> None:
  with pytest.raises(CapabilityResolutionError) as refused:
    _resolve(
      "session.driver",
      auth=_auth(capabilities=frozenset()),
      explicit_intent=ModelSelectionIntent(
        model_key="anthropic.claude-opus-5",
        effort="high",
        source="explicit_user",
      ),
    )

  assert refused.value.code == "capability_entitlement_required"


def test_default_does_not_fall_through_to_another_eligible_model() -> None:
  with pytest.raises(CapabilityResolutionError) as refused:
    _resolve(
      "session.driver",
      auth=_auth(
        model_keys=frozenset({"anthropic.claude-sonnet-5"}),
      ),
    )

  assert refused.value.code == "default_not_eligible"
  assert refused.value.model_key == "anthropic.claude-opus-5"
  assert refused.value.eligible_model_keys == ("anthropic.claude-sonnet-5",)


def test_ineligible_saved_preference_is_not_execution_intent() -> None:
  bind = _resolve(
    "session.driver",
    saved_preference=ModelSelectionIntent(
      model_key="openai.gpt-5-6",
      effort="medium",
      source="saved_preference",
    ),
  )

  assert bind.model_key == "anthropic.claude-opus-5"
  assert bind.selection_source == "capability_default"


def test_eligible_choices_are_derived_from_entitlement_and_handles() -> None:
  choices = eligible_model_choices(
    "session.driver",
    registry=INITIAL_MODEL_REGISTRY,
    selection_policy=INITIAL_MODEL_SELECTION_POLICY,
    auth=_auth(
      providers=("anthropic", "openai"),
      model_keys=frozenset({
        "anthropic.claude-haiku-4-5",
        "anthropic.claude-opus-5",
        "openai.gpt-5-6",
      }),
    ),
  )

  assert [choice.key for choice in choices] == [
    "anthropic.claude-haiku-4-5",
    "anthropic.claude-opus-5",
    "openai.gpt-5-6",
  ]
  assert choices[0].supported_efforts == ("none",)


def test_plan_author_inherits_exact_parent_identity_and_effort() -> None:
  parent = _resolve(
    "session.driver",
    explicit_intent=ModelSelectionIntent(
      model_key="anthropic.claude-sonnet-5",
      effort="xhigh",
      source="explicit_user",
    ),
  )
  author = _resolve("plan.author", parent_bind=parent)

  assert (
    author.model_key,
    author.provider,
    author.upstream_model,
    author.adapter,
    author.protocol_profile,
    author.route,
    author.effort,
  ) == (
    parent.model_key,
    parent.provider,
    parent.upstream_model,
    parent.adapter,
    parent.protocol_profile,
    parent.route,
    parent.effort,
  )
  assert author.selection_source == "parent_binding"
  assert author.capability_id == "plan.author"


def test_plan_author_requires_parent_without_authenticated_override() -> None:
  with pytest.raises(CapabilityResolutionError) as refused:
    _resolve("plan.author")

  assert refused.value.code == "parent_binding_required"


def test_authenticated_plan_author_override_precedes_parent_inheritance() -> None:
  bind = _resolve(
    "plan.author",
    authenticated_run_override=ModelSelectionIntent(
      model_key="anthropic.claude-fable-5",
      effort="high",
      source="explicit_user",
    ),
  )

  assert bind.model_key == "anthropic.claude-fable-5"
  assert bind.selection_source == "explicit_user"


def test_internal_workload_uses_frozen_dated_model_key() -> None:
  bind = _resolve("risk.overview_editorial")
  assert bind.model_key == "anthropic.claude-haiku-4-5-20251001-sdk"
  assert bind.upstream_model == "claude-haiku-4-5-20251001"
  assert bind.adapter == "anthropic.sdk.messages"
  assert bind.selection_source == "internal_policy"


def test_investment_quant_uses_service_authority_in_batch() -> None:
  bind = _resolve(
    "investment.quant_worker",
    auth=_auth(
      run_mode="batch",
      providers=("openai",),
      service=True,
    ),
  )
  assert bind.model_key == "openai.gpt-5-6"
  assert bind.credential_principal == "service"
  assert bind.credential_ref == "service:tenant:openai"
  assert bind.run_mode == "batch"


def test_reauthorization_preserves_durable_bind_instead_of_reselecting() -> None:
  auth = _auth()
  bind = _resolve("session.driver", auth=auth)

  handle = reauthorize_capability_bind(
    bind,
    registry=INITIAL_MODEL_REGISTRY,
    auth=auth,
  )
  assert handle.handle_id == bind.credential_ref


def test_reauthorization_rejects_tampered_registry_identity() -> None:
  bind = _resolve("session.driver").model_copy(
    update={"route": "attacker.route"}
  )
  with pytest.raises(CapabilityResolutionError) as refused:
    reauthorize_capability_bind(
      bind,
      registry=INITIAL_MODEL_REGISTRY,
      auth=_auth(),
    )
  assert refused.value.code == "capability_model_unavailable"


def test_reauthorization_rejects_changed_credential_reference() -> None:
  bind = _resolve("session.driver").model_copy(
    update={"credential_ref": "user:tenant:alice:replacement"}
  )
  with pytest.raises(CapabilityResolutionError) as refused:
    reauthorize_capability_bind(
      bind,
      registry=INITIAL_MODEL_REGISTRY,
      auth=_auth(),
    )
  assert refused.value.code == "credential_unavailable"


def test_reported_identity_must_be_explicitly_admitted() -> None:
  bind = _resolve("session.driver")
  assert validate_reported_identity(
    bind,
    "claude-opus-5",
    registry=INITIAL_MODEL_REGISTRY,
  ) == "claude-opus-5"

  with pytest.raises(CapabilityResolutionError) as refused:
    validate_reported_identity(
      bind,
      "claude-opus-latest",
      registry=INITIAL_MODEL_REGISTRY,
    )
  assert refused.value.code == "reported_identity_mismatch"


def test_auth_context_rejects_cross_actor_user_handle() -> None:
  with pytest.raises(ValueError, match="actor does not match"):
    AuthContext(
      run_mode="interactive",
      actor_id="bob",
      tenant_id="tenant",
      user_provider_handles={"anthropic": _user_handle("anthropic")},
      service_provider_handles={},
      entitled_capabilities=frozenset({"session.driver"}),
      entitled_model_keys=frozenset({"anthropic.claude-opus-5"}),
    )


def test_interactive_service_handle_requires_explicit_server_policy() -> None:
  handle = _service_handle("anthropic")
  auth = AuthContext(
    run_mode="interactive",
    actor_id="alice",
    tenant_id="tenant",
    user_provider_handles={},
    service_provider_handles={"anthropic": handle},
    entitled_capabilities=frozenset({"session.driver"}),
    entitled_model_keys=frozenset({"anthropic.claude-opus-5"}),
  )
  with pytest.raises(CapabilityResolutionError) as refused:
    _resolve("session.driver", auth=auth)
  assert refused.value.code == "default_not_eligible"

  admitted = AuthContext(
    run_mode="interactive",
    actor_id="alice",
    tenant_id="tenant",
    user_provider_handles={},
    service_provider_handles={"anthropic": handle},
    entitled_capabilities=frozenset({"session.driver"}),
    entitled_model_keys=frozenset({"anthropic.claude-opus-5"}),
    allow_service_for_interactive=True,
  )
  assert _resolve("session.driver", auth=admitted).credential_principal == "service"
