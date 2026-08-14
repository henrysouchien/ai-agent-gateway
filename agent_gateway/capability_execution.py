"""Materialization boundary for complete capability bindings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from .capability_binding import (
  AuthContext,
  CapabilityBind,
  CapabilityId,
  CapabilityResolutionError,
  CredentialHandle,
  ModelSelectionIntent,
  reauthorize_capability_bind,
  require_capability_execution_bind,
  resolve_capability_model,
)
from .model_registry import (
  CAPABILITY_EXECUTION_PROCESS,
  CAPABILITY_IDS,
  ProductModelRegistry,
  ProductModelSelectionPolicy,
)
from .providers import ModelProvider


CredentialMaterializer = Callable[[CredentialHandle], "MaterializedCredential"]
CapabilityAdapterResolver = Callable[[str], ModelProvider]
_SELECTION_AUTH_CONFIG_FIELDS = frozenset({
  "effort",
  "execution_transport",
  "model",
  "model_key",
  "thinking",
  "thinking_enabled_requested",
})


def _reject_selection_auth_config(auth_config: Mapping[str, Any]) -> None:
  duplicated = sorted(_SELECTION_AUTH_CONFIG_FIELDS & set(auth_config))
  if duplicated:
    raise ValueError(
      "credential auth_config must not contain model-selection fields: "
      + ", ".join(duplicated)
    )


@dataclass(frozen=True, slots=True)
class MaterializedCredential:
  handle: CredentialHandle
  auth_config: Mapping[str, Any] = field(repr=False)

  def __post_init__(self) -> None:
    if not isinstance(self.handle, CredentialHandle):
      raise TypeError("materialized credential handle must be CredentialHandle")
    if not isinstance(self.auth_config, Mapping):
      raise TypeError("materialized credential auth_config must be a mapping")
    config = dict(self.auth_config)
    _reject_selection_auth_config(config)
    object.__setattr__(self, "auth_config", MappingProxyType(config))


@dataclass(frozen=True, slots=True)
class BoundCapabilityExecution:
  """One immutable binding paired with adapter code and secret material."""

  bind: CapabilityBind
  registry: ProductModelRegistry
  adapter: ModelProvider
  auth_config: Mapping[str, Any] = field(repr=False)

  def __post_init__(self) -> None:
    if not isinstance(self.bind, CapabilityBind):
      raise TypeError("bound execution requires a CapabilityBind")
    if not isinstance(self.registry, ProductModelRegistry):
      raise TypeError("bound execution requires its admitting ProductModelRegistry")
    if not isinstance(self.auth_config, Mapping):
      raise TypeError("bound execution auth_config must be a mapping")
    config = dict(self.auth_config)
    _reject_selection_auth_config(config)
    configured_provider = config.get("provider")
    if configured_provider is not None and (
      not isinstance(configured_provider, str)
      or configured_provider.strip().lower() != self.bind.provider
    ):
      raise ValueError("auth_config provider does not match the capability bind")

    try:
      entry = self.registry.require(self.bind.model_key)
    except KeyError as exc:
      raise ValueError(
        "bound execution registry does not contain the capability bind model_key"
      ) from exc
    bound_identity = (
      self.bind.provider,
      self.bind.upstream_model,
      self.bind.adapter,
      self.bind.protocol_profile,
      self.bind.route,
    )
    registry_identity = (
      entry.provider,
      entry.upstream_model,
      entry.adapter,
      entry.protocol_profile,
      entry.route,
    )
    if bound_identity != registry_identity:
      raise ValueError("capability bind execution identity does not match its registry")
    if self.bind.capability_id not in entry.capabilities:
      raise ValueError("capability bind is not qualified by its registry entry")
    object.__setattr__(self, "auth_config", MappingProxyType(config))
    self.validate()

  @property
  def provider(self) -> ModelProvider:
    """The adapter's provider object; selection identity remains in ``bind``."""

    return self.adapter

  def validate(self) -> None:
    require_capability_execution_bind(
      self.bind,
      provider=self.adapter,
      auth_config=self.auth_config,
    )


@dataclass(frozen=True, slots=True)
class CapabilityExecutionResolver:
  """Resolve stable-key intent once, then materialize its exact binding."""

  registry: ProductModelRegistry
  selection_policy: ProductModelSelectionPolicy
  auth_context: AuthContext
  credential_materializer: CredentialMaterializer = field(repr=False)
  adapter_resolver: CapabilityAdapterResolver = field(repr=False)
  trusted_channel: str | None = None
  authenticated_run_overrides: Mapping[CapabilityId, ModelSelectionIntent] = field(
    default_factory=dict
  )
  # Explicit designation of which capabilities THIS serving process executes
  # (for the gateway server, `GATEWAY_EXECUTED_CAPABILITY_IDS`).  Capabilities
  # outside the set are executed by another serving process; attempting to
  # resolve or materialize one here is a typed refusal, not a stream-time
  # `provider_unavailable`.  `None` leaves the resolver unrestricted for
  # processes (Risk, Investment workers) that designate their own executable
  # sets at their own boundaries.
  executable_capability_ids: frozenset[str] | None = None

  def __post_init__(self) -> None:
    if not callable(self.credential_materializer):
      raise TypeError("credential_materializer must be callable")
    if not callable(self.adapter_resolver):
      raise TypeError("adapter_resolver must be callable")
    if self.executable_capability_ids is not None:
      executable = frozenset(self.executable_capability_ids)
      unknown = executable - CAPABILITY_IDS
      if unknown:
        raise ValueError(
          "executable_capability_ids has unknown capabilities: "
          + ", ".join(sorted(unknown))
        )
      object.__setattr__(self, "executable_capability_ids", executable)
    self.selection_policy.admit_registry(self.registry)
    run_overrides = dict(self.authenticated_run_overrides)
    for capability_id, intent in run_overrides.items():
      policy = self.selection_policy.capabilities.get(capability_id)
      if policy is None or not policy.allow_authenticated_run_override:
        raise CapabilityResolutionError(
          "capability_model_not_allowed",
          f"authenticated run override is not allowed for {capability_id}",
          capability_id=capability_id,
          model_key=intent.model_key,
        )
      if intent.source != "explicit_user":
        raise ValueError("authenticated run overrides must be explicit_user intent")
    object.__setattr__(
      self,
      "authenticated_run_overrides",
      MappingProxyType(run_overrides),
    )
    if self.trusted_channel is not None:
      object.__setattr__(
        self,
        "trusted_channel",
        str(self.trusted_channel).strip() or None,
      )

  def _require_executable_here(self, capability_id: str) -> None:
    """Refuse capabilities designated as executed by another serving process."""

    if (
      self.executable_capability_ids is None
      or capability_id in self.executable_capability_ids
    ):
      return
    executing_process = CAPABILITY_EXECUTION_PROCESS.get(capability_id)
    raise CapabilityResolutionError(
      "capability_externally_executed",
      f"capability {capability_id!r} is executed by the "
      f"{executing_process or 'designated external'} serving process, "
      "not this gateway process",
      capability_id=capability_id,
    )

  def resolve(
    self,
    capability_id: CapabilityId,
    *,
    explicit_intent: ModelSelectionIntent | None = None,
    saved_preference: ModelSelectionIntent | None = None,
    parent_bind: CapabilityBind | None = None,
  ) -> BoundCapabilityExecution:
    self._require_executable_here(str(capability_id or "").strip())
    bind = resolve_capability_model(
      capability_id,
      registry=self.registry,
      selection_policy=self.selection_policy,
      auth=self.auth_context,
      explicit_intent=explicit_intent,
      saved_preference=saved_preference,
      authenticated_run_override=self.authenticated_run_overrides.get(capability_id),
      trusted_channel=self.trusted_channel,
      parent_bind=parent_bind,
    )
    return self._materialize(bind)

  def materialize_bind(self, bind: CapabilityBind) -> BoundCapabilityExecution:
    """Reauthorize and materialize a durable bind without reselection."""

    self._require_executable_here(bind.capability_id)
    handle = reauthorize_capability_bind(
      bind,
      registry=self.registry,
      auth=self.auth_context,
    )
    return self._materialize(bind, handle=handle)

  def authorize_bind(self, bind: CapabilityBind) -> CapabilityBind:
    """Reauthorize a durable bind without credentials or adapter construction."""

    self._require_executable_here(bind.capability_id)
    reauthorize_capability_bind(
      bind,
      registry=self.registry,
      auth=self.auth_context,
    )
    return bind

  def _materialize(
    self,
    bind: CapabilityBind,
    *,
    handle: CredentialHandle | None = None,
  ) -> BoundCapabilityExecution:
    self._require_executable_here(bind.capability_id)
    selected_handle = handle or reauthorize_capability_bind(
      bind,
      registry=self.registry,
      auth=self.auth_context,
    )
    try:
      materialized = self.credential_materializer(selected_handle)
    except CapabilityResolutionError:
      raise
    except (LookupError, ValueError) as exc:
      # Expected credential-authority failures (missing/rejected material)
      # become the typed secret-free refusal.  Anything else is a programming
      # or configuration error and propagates as such, distinct from a
      # credential refusal.
      raise CapabilityResolutionError(
        "credential_unavailable",
        f"credential material is unavailable for {bind.provider!r}",
        capability_id=bind.capability_id,
        model_key=bind.model_key,
        provider=bind.provider,
        upstream_model=bind.upstream_model,
      ) from exc
    if not isinstance(materialized, MaterializedCredential):
      raise CapabilityResolutionError(
        "credential_unavailable",
        "credential materializer must return MaterializedCredential",
        capability_id=bind.capability_id,
        model_key=bind.model_key,
      )
    if materialized.handle != selected_handle:
      raise CapabilityResolutionError(
        "credential_unavailable",
        "credential materializer returned a different credential handle",
        capability_id=bind.capability_id,
        model_key=bind.model_key,
      )

    auth_config = dict(materialized.auth_config)
    try:
      _reject_selection_auth_config(auth_config)
    except ValueError as exc:
      raise CapabilityResolutionError(
        "credential_unavailable",
        "credential material contains forbidden model-selection fields",
        capability_id=bind.capability_id,
        model_key=bind.model_key,
      ) from exc
    configured_provider = auth_config.get("provider")
    if configured_provider is not None and (
      not isinstance(configured_provider, str)
      or configured_provider.strip().lower() != bind.provider
    ):
      raise CapabilityResolutionError(
        "credential_unavailable",
        "credential material provider does not match the capability bind",
        capability_id=bind.capability_id,
        model_key=bind.model_key,
      )
    try:
      adapter = self.adapter_resolver(bind.adapter)
    except CapabilityResolutionError:
      raise
    except (LookupError, ValueError) as exc:
      # Expected adapter-resolution failures (adapter unknown/uninstalled)
      # become the typed refusal; unexpected exceptions are configuration or
      # programming errors and propagate unchanged.
      raise CapabilityResolutionError(
        "provider_unavailable",
        f"adapter is unavailable for {bind.adapter!r}",
        capability_id=bind.capability_id,
        model_key=bind.model_key,
        provider=bind.provider,
        upstream_model=bind.upstream_model,
      ) from exc
    return BoundCapabilityExecution(
      bind=bind,
      registry=self.registry,
      adapter=adapter,
      auth_config=auth_config,
    )


def derive_batch_capability_execution(
  parent_resolver: CapabilityExecutionResolver,
  parent_execution: BoundCapabilityExecution,
) -> tuple[CapabilityExecutionResolver, BoundCapabilityExecution]:
  """Carry the exact session binding into a dedicated batch authorization."""

  if parent_execution.bind.capability_id != "session.driver":
    raise ValueError("batch derivation requires a session.driver execution")
  parent_execution.validate()
  parent_auth = parent_resolver.auth_context
  parent_bind = parent_execution.bind
  # The batch context authorizes exactly the inherited bind's credential use:
  # a user credential is run-scoped to the bind's provider family only, and no
  # other user handle crosses into the derived batch authorization.
  if parent_bind.credential_principal == "user":
    user_provider_handles = {
      provider: handle
      for provider, handle in parent_auth.user_provider_handles.items()
      if provider == parent_bind.provider
    }
    run_scoped = frozenset({parent_bind.provider})
  else:
    user_provider_handles = {}
    run_scoped = frozenset()
  batch_auth = AuthContext(
    run_mode="batch",
    actor_id=parent_auth.actor_id,
    tenant_id=parent_auth.tenant_id,
    user_provider_handles=user_provider_handles,
    service_provider_handles=parent_auth.service_provider_handles,
    entitled_capabilities=parent_auth.entitled_capabilities,
    entitled_model_keys=parent_auth.entitled_model_keys,
    run_scoped_user_providers=run_scoped,
  )
  batch_resolver = CapabilityExecutionResolver(
    registry=parent_resolver.registry,
    selection_policy=parent_resolver.selection_policy,
    auth_context=batch_auth,
    credential_materializer=parent_resolver.credential_materializer,
    adapter_resolver=parent_resolver.adapter_resolver,
    trusted_channel=parent_resolver.trusted_channel,
    authenticated_run_overrides=parent_resolver.authenticated_run_overrides,
    executable_capability_ids=parent_resolver.executable_capability_ids,
  )
  # Validated construction of the derived bind: every contract validator runs
  # rather than being bypassed by model_copy(update=...).
  batch_bind = CapabilityBind.model_validate({
    **parent_bind.model_dump(mode="json"),
    "run_mode": "batch",
  })
  batch_execution = batch_resolver.materialize_bind(batch_bind)
  return batch_resolver, batch_execution


__all__ = [
  "BoundCapabilityExecution",
  "CapabilityAdapterResolver",
  "CapabilityExecutionResolver",
  "CredentialMaterializer",
  "MaterializedCredential",
  "derive_batch_capability_execution",
]
