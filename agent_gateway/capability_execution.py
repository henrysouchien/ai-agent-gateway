"""Materialization boundary for complete capability bindings."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal

from agent_workflow_contracts.models import SelectionProvenance
from model_authority.bind import CapabilityBind
from model_authority.binding import (
  AuthContext,
  CapabilityId,
  CapabilityResolutionError,
  CredentialHandle,
  ModelSelectionIntent,
  reauthorize_capability_bind,
  require_capability_execution_bind,
  require_registered_capability_bind,
  resolve_capability_model,
)
from model_authority.capabilities import (
  CAPABILITY_EXECUTION_PROCESS,
  CAPABILITY_IDS,
  SESSION_DRIVER_CAPABILITY,
)
from model_authority.registry import ProductModelRegistry
from model_authority.selection import (
  ProductModelSelectionPolicy,
  SelectionSnapshot,
  TierDefault,
  TierModelsDefault,
  TierProjectionError,
  capture_selection_snapshot,
  project_fallback_chain,
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
class FallbackSkip:
  """A chain entry a hop passed over, with the resolver's typed refusal code."""

  model_key: str
  code: str


@dataclass(frozen=True, slots=True)
class ModelFallbackHop:
  """The next chain entry, bound and materialized for a turn's remaining attempts.

  ``chain_index`` is this entry's position in the projected chain and
  ``primary_model_key`` the model the turn was bound to; a later hop walks on
  from the position, never by matching a model key.
  """

  execution: BoundCapabilityExecution
  role: str
  chain_index: int
  skipped: tuple[FallbackSkip, ...]
  primary_model_key: str


@dataclass(frozen=True, slots=True)
class BoundCapabilityExecution:
  """One immutable binding paired with adapter code and secret material."""

  bind: CapabilityBind
  registry: ProductModelRegistry
  adapter: ModelProvider
  auth_config: Mapping[str, Any] = field(repr=False)
  resolver: CapabilityExecutionResolver | None = field(default=None, repr=False, compare=False)

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

    object.__setattr__(self, "auth_config", MappingProxyType(config))
    self.validate()

  @property
  def provider(self) -> ModelProvider:
    """The adapter's provider object; selection identity remains in ``bind``."""

    return self.adapter

  def validate(self) -> None:
    require_registered_capability_bind(self.bind, registry=self.registry)
    require_capability_execution_bind(
      self.bind,
      provider=self.adapter,
      auth_config=self.auth_config,
    )

  def revalidate(self) -> None:
    """Reauthorize the binding against the live authorization context and registry.

    Fails closed with CapabilityResolutionError if the credential was revoked,
    the user has no active roster entry, or the capability's disposition no
    longer admits the use.
    """
    self.validate()
    if self.resolver is not None:
      self.resolver.materialize_bind(self.bind)

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
    # An authenticated run override applies only where the capability's
    # policy admits one; this check is that rule's owner. Its authority is the
    # authenticated request that built this resolver, so the intent's source
    # label is not read here or in resolve_capability_model.
    for capability_id, intent in run_overrides.items():
      policy = self.selection_policy.capabilities.get(capability_id)
      if policy is None or not policy.allow_authenticated_run_override:
        raise CapabilityResolutionError(
          "capability_model_not_allowed",
          f"authenticated run override is not allowed for {capability_id}",
          capability_id=capability_id,
          model_key=intent.model_key,
        )
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

  def select_bind(
    self,
    capability_id: CapabilityId,
    *,
    explicit_intent: ModelSelectionIntent | None = None,
    saved_preference: ModelSelectionIntent | None = None,
    parent_bind: CapabilityBind | None = None,
    effort: str | None = None,
    role: str | None = None,
    selection_snapshot: SelectionSnapshot | None = None,
  ) -> CapabilityBind:
    """Select and authorize an exact bind without credentials or adapter effects.

    ``role`` and ``effort`` are already-lowered spawn/operation requests;
    the admission owner supplies the frozen ``selection_snapshot``.
    """

    self._require_executable_here(str(capability_id or "").strip())
    return resolve_capability_model(
      capability_id,
      registry=self.registry,
      selection_policy=self.selection_policy,
      auth=self.auth_context,
      explicit_intent=explicit_intent,
      saved_preference=saved_preference,
      authenticated_run_override=self.authenticated_run_overrides.get(capability_id),
      trusted_channel=self.trusted_channel,
      parent_bind=parent_bind,
      requested_effort=effort,
      requested_role=role,
      selection_snapshot=selection_snapshot,
    )

  def resolve(
    self,
    capability_id: CapabilityId,
    *,
    explicit_intent: ModelSelectionIntent | None = None,
    saved_preference: ModelSelectionIntent | None = None,
    parent_bind: CapabilityBind | None = None,
    effort: str | None = None,
    role: str | None = None,
    selection_snapshot: SelectionSnapshot | None = None,
  ) -> BoundCapabilityExecution:
    bind = self.select_bind(
      capability_id,
      explicit_intent=explicit_intent,
      saved_preference=saved_preference,
      parent_bind=parent_bind,
      effort=effort,
      role=role,
      selection_snapshot=selection_snapshot,
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

  @staticmethod
  def rebind_rotated_credential(
    execution: BoundCapabilityExecution,
    rotated_auth_config: Mapping[str, Any],
    *,
    merged_auth_config: Mapping[str, Any],
  ) -> BoundCapabilityExecution | None:
    """Reauthorize a sibling generation; never put it under the previous bind."""
    from .providers.anthropic_oauth import is_anthropic_generation_ref

    source = rotated_auth_config.get("credential_source")
    generation_ref = source.get("generation_ref") if isinstance(source, Mapping) else None
    if not isinstance(generation_ref, str) or not is_anthropic_generation_ref(generation_ref):
      if is_anthropic_generation_ref(execution.bind.credential_ref):
        return None
      # Providers without generation handles retain their per-session identity.
      return replace(execution, auth_config=merged_auth_config)

    resolver = execution.resolver
    if resolver is None:
      return None
    # CapabilityBind is a validated Pydantic value, not a dataclass. Its wire
    # fields are the identity; there is no separately cached bind digest.
    bind = CapabilityBind.model_validate({
      **execution.bind.to_json(),
      "credential_ref": generation_ref,
    })
    return resolver.with_frozen_credential(bind, merged_auth_config).materialize_bind(bind)

  @staticmethod
  def rebind_model_fallback(
    execution: BoundCapabilityExecution,
    selection_snapshot: SelectionSnapshot,
    *,
    selection_provenance: SelectionProvenance | None,
    history_fits: Callable[[BoundCapabilityExecution], bool],
    previous: ModelFallbackHop | None = None,
    skipped_out: list[FallbackSkip] | None = None,
  ) -> ModelFallbackHop | None:
    """Bind the next eligible entry of the snapshot's chain; never a reselection.

    The chain is the recorded role's capture-time chain when
    ``selection_provenance.role`` is set, otherwise the slot default role's
    (design §6.5: a default bind and an explicit or saved concrete pick both
    use it). A first hop with no provenance has an unknown origin and does
    not hop. The first hop walks the chain from the start, skipping entries
    identical to the bound primary, so an explicit or saved pick uses the
    chain even when it names the pick; ``previous`` continues after the last
    hop's position, so hops are finite even when a chain names a model twice.
    Each entry resolves through ``resolve_capability_model`` as a one-model
    assignment of that role, so qualification, both ceilings, lifecycle,
    effort support,
    entitlement and the entry provider's own credential all apply; a refused
    entry is skipped with its code. An entry whose own account pool is
    entirely parked is skipped as ``credential_pool_parked``; otherwise it
    binds an unblocked account. The bind keeps the primary's
    ``selection_source`` and principal. ``None`` means no entry serves (or
    the captured role no longer projects) and the caller's existing failure
    path stands.
    """

    resolver = execution.resolver
    if resolver is None:
      return None
    bind = execution.bind
    try:
      if previous is None:
        if selection_provenance is None:
          return None
        if selection_provenance.role is not None:
          role = selection_provenance.role
        else:
          defaults = selection_snapshot.frozen_slot(bind.capability_id)
          channel = str(resolver.trusted_channel or "").strip()
          slot_default = defaults.by_channel.get(channel, defaults.default)
          if slot_default.kind != "tier" or slot_default.tier is None:
            return None
          role = slot_default.tier
        primary_model_key = bind.model_key
        start = 0
      else:
        role = previous.role
        primary_model_key = previous.primary_model_key
        start = previous.chain_index + 1
      chain = project_fallback_chain(
        resolver.registry,
        resolver.selection_policy,
        bind.capability_id,
        role,
        selection_snapshot=selection_snapshot,
      )
    except TierProjectionError:
      return None
    skipped: list[FallbackSkip] = [] if previous is not None else [
      FallbackSkip(omitted.entry.models[0], omitted.code)
      for omitted in chain.omitted
    ]
    for index in range(start, len(chain.entries)):
      entry = chain.entries[index]
      if entry.model_key == primary_model_key:
        continue
      effort = entry.effort or bind.effort
      try:
        resolved = resolve_capability_model(
          bind.capability_id,
          registry=resolver.registry,
          selection_policy=resolver.selection_policy,
          auth=resolver.auth_context,
          trusted_channel=resolver.trusted_channel,
          requested_effort=effort,
          requested_role=role,
          role_map={role: TierModelsDefault((entry.model_key,), effort)},
          role_map_source="internal_policy",
        )
        if (
          resolved.credential_principal != bind.credential_principal
          or resolved.run_mode != bind.run_mode
        ):
          skipped.append(FallbackSkip(entry.model_key, "credential_principal_changed"))
          continue
        hop_execution = CapabilityExecutionResolver._rebind_unparked_account(
          resolver.materialize_bind(CapabilityBind.model_validate({
            **resolved.to_json(),
            "selection_source": bind.selection_source,
          })),
        )
        if hop_execution is None:
          skipped.append(FallbackSkip(entry.model_key, "credential_pool_parked"))
          continue
      except CapabilityResolutionError as exc:
        skipped.append(FallbackSkip(entry.model_key, exc.code))
        continue
      if not history_fits(hop_execution):
        # Until portable compaction (W6) carries history across providers,
        # an entry whose context cannot hold this transcript is not used.
        skipped.append(FallbackSkip(entry.model_key, "context_window_exceeded"))
        continue
      return ModelFallbackHop(hop_execution, role, index, tuple(skipped), primary_model_key)
    if skipped_out is not None:
      skipped_out.extend(skipped)
    return None


  @staticmethod
  def _rebind_unparked_account(
    execution: BoundCapabilityExecution,
  ) -> BoundCapabilityExecution | None:
    """The execution on an unblocked account of its pool; ``None`` when all are parked.

    Materialization binds a member the way a turn start does, which hands
    back a parked one when nothing else is left; a hop never serves on it.
    Anthropic retains its generation-aware binding; other enrolled providers
    use their registered candidate resolver and materializer.
    """
    from .providers import anthropic as anthropic_provider
    from .providers.anthropic_oauth import (
      resolve_anthropic_credentials,
      select_anthropic_credential_source,
    )
    from .providers.credential_pool import (
      get_provider_credential_pool,
      get_provider_pool_hooks,
      select_pool_candidate,
    )

    config = execution.auth_config
    if execution.bind.provider != "anthropic":
      hooks = get_provider_pool_hooks(execution.bind.provider)
      if hooks is None:
        return execution
      candidate_resolver, materializer = hooks
      pool_config = {
        key: value for key, value in config.items()
        if key != "credential_source"
      }
      candidates, refresh_fn = candidate_resolver(pool_config)
      if not candidates:
        # The bound credential was supplied to the run directly and no enrolled
        # pool exists: it is valid material and nothing is parked on it.
        return execution
      provider_pool = get_provider_credential_pool(execution.bind.provider)
      if candidates and not provider_pool.first_unblocked([candidate.key for candidate in candidates]):
        return None
      candidate = select_pool_candidate(candidates, pool=provider_pool, refresh_fn=refresh_fn)
      if candidate is None or provider_pool.blocked_until(candidate.key) > time.time():
        return None
      account = materializer(candidate, config)
      if all(config.get(key) == value for key, value in account.items()):
        return execution
      return CapabilityExecutionResolver.rebind_rotated_credential(
        execution, account, merged_auth_config={**config, **account},
      )
    pool = anthropic_provider.ANTHROPIC_CREDENTIAL_POOL
    pool_config = {key: value for key, value in config.items() if key != "credential_source"}
    sources = resolve_anthropic_credentials(pool_config)
    if not pool.first_unblocked(sources.keys):
      return None
    selected = select_anthropic_credential_source(sources, pool=pool)
    if selected is None or pool.blocked_until(selected.key) > time.time():
      # The unblocked account's refresh failed and parked it.
      return None
    is_api_key = selected.record is not None and selected.record.credential_kind == "api_key"
    credential_field = "api_key" if is_api_key else "auth_token"
    if selected.token == config.get(credential_field):
      return execution
    account = {
      "auth_mode": "api" if is_api_key else "oauth",
      "api_key": selected.token if is_api_key else "",
      "auth_token": "" if is_api_key else selected.token,
      "credential_source": selected.source_metadata(),
    }
    rebound = CapabilityExecutionResolver.rebind_rotated_credential(
      execution, account, merged_auth_config={**config, **account},
    )
    if rebound is None:
      raise CapabilityResolutionError(
        "credential_unavailable",
        "the unblocked Anthropic account has no credential generation",
        capability_id=execution.bind.capability_id,
        model_key=execution.bind.model_key,
      )
    return rebound

  def with_frozen_credential(
    self, bind: CapabilityBind, auth_config: Mapping[str, Any],
  ) -> CapabilityExecutionResolver:
    """Offer an exact durable credential under the current ownership facts."""
    handle_field = (
      "user_provider_handles"
      if bind.credential_principal == "user"
      else "service_provider_handles"
    )
    handles = getattr(self.auth_context, handle_field)
    current_handle = handles.get(bind.provider)
    if current_handle is None:
      raise CapabilityResolutionError(
        "credential_unavailable",
        "frozen credential ownership is unavailable",
        capability_id=bind.capability_id,
      )
    exact_handle = replace(current_handle, handle_id=bind.credential_ref)
    materialized = MaterializedCredential(
      handle=exact_handle, auth_config=auth_config,
    )

    def _materialize(handle: CredentialHandle) -> MaterializedCredential:
      if handle == exact_handle:
        return materialized
      return self.credential_materializer(handle)

    return replace(
      self,
      auth_context=replace(
        self.auth_context,
        **{handle_field: {**handles, bind.provider: exact_handle}},
      ),
      credential_materializer=_materialize,
    )

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
      resolver=self,
    )


def preset_role_map(
  selection_policy: ProductModelSelectionPolicy,
  name: str,
) -> Mapping[str, TierDefault]:
  """The one place a ``model_preset`` name becomes a published role map.

  An explicit root or subtree selection the policy does not publish fails
  clearly (design §6.2); it is never replaced by base policy.
  """
  role_map = selection_policy.presets.get(name)
  if role_map is None:
    raise CapabilityResolutionError(
      "capability_selection_invalid",
      f"model preset {name!r} is not published by model-selection policy "
      f"revision {selection_policy.revision!r}; list published names with "
      "GET /model-presets/self",
      capability_id=SESSION_DRIVER_CAPABILITY,
    )
  return role_map


def preset_selection_snapshot(
  selection_policy: ProductModelSelectionPolicy,
  name: str,
) -> SelectionSnapshot:
  """Freeze an explicit root or subtree preset's whole map (design §7.4)."""
  return capture_selection_snapshot(
    selection_policy,
    role_map=preset_role_map(selection_policy, name),
    role_map_source="internal_policy",
  )


def materialize_frozen_bind(
  resolver: CapabilityExecutionResolver,
  bind: CapabilityBind,
  *,
  user_scope: Any,
  base_config: Mapping[str, Any],
) -> BoundCapabilityExecution:
  """Reauthorize a durable bind on its exact frozen credential generation.

  An Anthropic generation-ref bind continues on that generation, offered
  under the current ownership facts, never on whatever the owning session
  holds now. A generation that cannot be read is ``credential_unavailable``.
  """
  from .providers.anthropic_oauth import (
    AnthropicSourceError,
    anthropic_generation_auth_config,
    is_anthropic_generation_ref,
  )

  if bind.provider == "anthropic" and is_anthropic_generation_ref(bind.credential_ref):
    try:
      exact_config = anthropic_generation_auth_config(
        bind.credential_ref, user_scope=user_scope, base_config=base_config,
      )
    except AnthropicSourceError as exc:
      raise CapabilityResolutionError(
        "credential_unavailable",
        "resumed Anthropic credential generation is unavailable",
        capability_id=bind.capability_id,
      ) from exc
    resolver = resolver.with_frozen_credential(bind, exact_config)
  return resolver.materialize_bind(bind)


def derive_batch_capability_execution(
  parent_resolver: CapabilityExecutionResolver,
  parent_execution: BoundCapabilityExecution,
) -> tuple[CapabilityExecutionResolver, BoundCapabilityExecution]:
  """Carry the exact session binding into a dedicated batch authorization."""

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


ModelCallKind = Literal["completion", "judgment"]


@dataclass(frozen=True, slots=True)
class ModelCallOperation:
  """A registered leaf model-call operation.

  The registration owns execution class (capability slot id), rubric schema
  (for judgments), system prompt, role/effort ceilings and provider constraints.
  A cell or caller passes the operation name, never a raw slot id.
  """

  name: str
  execution_class: CapabilityId
  kind: ModelCallKind
  description: str
  rubric_schema: Mapping[str, Any] | None = None
  system_prompt: str | None = None
  default_role: str | None = None
  default_effort: str | None = None
  allowed_roles: frozenset[str] | None = None
  provider_constraints: frozenset[str] | None = None


_MODEL_CALL_OPERATIONS: dict[str, ModelCallOperation] = {}


def register_model_call_operation(operation: ModelCallOperation) -> None:
  """Register one leaf model-call operation."""
  if not isinstance(operation, ModelCallOperation):
    raise TypeError("operation must be an exact ModelCallOperation")
  name = str(operation.name or "").strip()
  if not name:
    raise ValueError("operation name must be non-empty")
  if name in _MODEL_CALL_OPERATIONS:
    raise ValueError(f"model-call operation already registered: {name!r}")
  _MODEL_CALL_OPERATIONS[name] = operation


def lookup_model_call_operation(name: str) -> ModelCallOperation | None:
  """Look up one registered model-call operation by name."""
  return _MODEL_CALL_OPERATIONS.get(str(name or "").strip())


def list_model_call_operations(
  *,
  kind: ModelCallKind | None = None,
) -> tuple[ModelCallOperation, ...]:
  """Return all registered model-call operations, optionally filtered by kind."""
  ops = _MODEL_CALL_OPERATIONS.values()
  if kind is not None:
    ops = (op for op in ops if op.kind == kind)
  return tuple(sorted(ops, key=lambda op: op.name))


_DEFAULT_MODEL_CALL_OPERATIONS: tuple[ModelCallOperation, ...] = (
  ModelCallOperation(
    name="general_completion",
    execution_class="edgar.general_completion",
    kind="completion",
    description="General text generation, summarization, and synthesis from supplied material.",
    system_prompt="You are a helpful fundamental research assistant. Synthesize the provided material accurately and concisely.",
  ),
  ModelCallOperation(
    name="citation_review",
    execution_class="citation.review",
    kind="judgment",
    description="Evaluate citation coverage, source fidelity, and factual consistency against sources.",
    system_prompt="You are a precise citation and evidence auditor. Evaluate the claims against the sources. Return a structured judgment.",
    rubric_schema=MappingProxyType({
      "type": "object",
      "properties": {
        "verdict": {
          "type": "string",
          "enum": ["supported", "unsupported", "partially_supported"],
        },
        "findings": {"type": "array", "items": {"type": "string"}},
      },
      "required": ["verdict"],
    }),
  ),
  ModelCallOperation(
    name="verification",
    execution_class="node.verify",
    kind="judgment",
    description="Verify hypotheses, financial statements, and calculations against reference data.",
    system_prompt="You are a rigorous verification evaluator. Check the state against the provided questions and rubric.",
    rubric_schema=MappingProxyType({
      "type": "object",
      "properties": {
        "valid": {"type": "boolean"},
        "reasons": {"type": "array", "items": {"type": "string"}},
      },
      "required": ["valid"],
    }),
  ),
)

for _default_op in _DEFAULT_MODEL_CALL_OPERATIONS:
  _MODEL_CALL_OPERATIONS[_default_op.name] = _default_op


def reset_model_call_operations_for_tests() -> None:
  """Reset the model-call operation registry to default operations."""
  _MODEL_CALL_OPERATIONS.clear()
  for op in _DEFAULT_MODEL_CALL_OPERATIONS:
    _MODEL_CALL_OPERATIONS[op.name] = op


async def execute_leaf_model_call(
  operation: str,
  *,
  material: str | None = None,
  state: Any | None = None,
  questions: Any | None = None,
  role: str | None = None,
  effort: str | None = None,
  evidence_handles: Iterable[str] = (),
  runner: Any = None,
  resolver: CapabilityExecutionResolver | None = None,
  user_id: str | None = None,
  session_id: str | None = None,
  task_id: str | None = None,
  wait: bool = False,
  timeout: float | None = None,
) -> dict[str, Any]:
  """Execute one leaf model-call operation under D13 proposal.

  The operation owns execution class and rubric schema. Under proposal D13
  each leaf call registers as a P4 job on the runner's registry, so wait,
  cancel and journal usage share one surface with agent jobs.
  """
  import asyncio
  import json
  import os

  op = lookup_model_call_operation(operation)
  if op is None:
    registered = [o.name for o in list_model_call_operations()]
    raise ValueError(
      f"unknown model-call operation {operation!r}; registered: {registered}"
    )

  if op.allowed_roles is not None and role is not None:
    if role not in op.allowed_roles:
      raise ValueError(
        f"role {role!r} not allowed for operation {op.name!r}; "
        f"allowed: {sorted(op.allowed_roles)}"
      )

  effective_role = role or op.default_role
  effective_effort = effort or op.default_effort

  if resolver is None:
    cap_exec = getattr(runner, "capability_execution", None)
    resolver = getattr(cap_exec, "resolver", None) if cap_exec is not None else None
  if resolver is None:
    raise RuntimeError(
      "execute_leaf_model_call requires a resolver or runner with capability_execution.resolver"
    )

  bound_execution = resolver.resolve(
    op.execution_class,
    role=effective_role,
    effort=effective_effort,
  )
  if bound_execution.auth_config.get("max_tokens") is None:
    model_info = getattr(bound_execution.adapter, "get_model_info", lambda _m: None)(
      bound_execution.bind.upstream_model
    )
    default_max = getattr(model_info, "max_output_tokens", 4096) if model_info else 4096
    config = {**bound_execution.auth_config, "max_tokens": int(default_max or 4096)}
    bound_execution = replace(bound_execution, auth_config=MappingProxyType(config))

  handles = tuple(str(h) for h in evidence_handles if str(h).strip())
  if op.kind == "completion":
    prompt_text = str(material or "")
    if handles:
      prompt_text = f"{prompt_text}\n\nEvidence handles: {', '.join(handles)}"
    system_prompt = op.system_prompt
  else:
    q_str = (
      json.dumps(questions, indent=2)
      if isinstance(questions, (dict, list))
      else str(questions or "")
    )
    s_str = (
      json.dumps(state, indent=2)
      if isinstance(state, (dict, list))
      else str(state or "")
    )
    prompt_text = f"State:\n{s_str}\n\nQuestions/Rubric:\n{q_str}"
    if handles:
      prompt_text = f"{prompt_text}\n\nEvidence handles: {', '.join(handles)}"
    system_prompt = (
      op.system_prompt
      or "Evaluate the provided state against the questions and rubric. Return a structured judgment."
    )

  assigned_task_id = task_id or f"mc_{os.urandom(4).hex()}"

  def _usage_recorder() -> tuple[list[Any], Any]:
    events: list[Any] = []
    return events, events.append

  def _usage_summary(events: list[Any]) -> dict[str, Any]:
    return {
      "input_tokens": sum(e.input_tokens or 0 for e in events),
      "output_tokens": sum(e.output_tokens or 0 for e in events),
      "cache_read_tokens": sum(e.cache_read_tokens or 0 for e in events),
      "cache_creation_tokens": sum(e.cache_creation_tokens or 0 for e in events),
      "cost_usd": sum(e.cost_usd or 0.0 for e in events),
      "model": events[-1].model if events else None,
      "provider": events[-1].provider if events else None,
      "turn_count": len(events),
    }

  if runner is not None and hasattr(runner, "_register_background_task"):
    async def _run_model_call_job(
      _tool_input: dict[str, Any],
      *,
      call_index: int = 0,
      task_entry: Any = None,
      **_job_kwargs: Any,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
      from .send_prompt import send_prompt
      usage_events, record_usage = _usage_recorder()
      try:
        text = await send_prompt(
          prompt_text,
          system_prompt=system_prompt,
          capability_execution=bound_execution,
          user_id=user_id or getattr(runner, "user_id", "anonymous"),
          session_id=session_id or getattr(runner, "session_id", "model-call"),
          request_id=getattr(runner, "_request_id", None),
          rate_table_version=getattr(runner, "_rate_table_version", None),
          billing_mode=getattr(runner, "_billing_mode", None),
          channel=getattr(runner, "_channel", None),
          on_usage=record_usage,
          # send_prompt bills each attempt with its real outcome (succeeded,
          # failed_billable, canceled); the runner's producer is the one billing
          # door for agent and leaf usage alike.
          commercial_usage_producer=getattr(runner, "_commercial_usage_producer", None),
          completion_record_root=None,
        )
        res: dict[str, Any] = {
          "operation": op.name,
          "kind": op.kind,
          "content": text,
          "task_id": assigned_task_id,
          "capability_id": bound_execution.bind.capability_id,
          "model": bound_execution.bind.model_key,
          "provider": bound_execution.bind.provider,
          "selection_provenance": bound_execution.bind.model_dump(mode="json"),
          "usage": _usage_summary(usage_events),
        }
        if op.kind == "judgment":
          res["state"] = state
          res["questions"] = questions
          try:
            res["judgment"] = json.loads(text)
          except Exception:
            res["judgment"] = text
        return res, None
      except asyncio.CancelledError:
        raise
      except Exception as exc:
        return None, {"code": "model_call_failed", "message": str(exc)}

    launched, error = await runner._register_background_task(
      tool_input={
        "operation": op.name,
        "kind": op.kind,
        "prompt_preview": prompt_text[:200],
      },
      handler=_run_model_call_job,
      task_type=op.kind,
      task_id_override=assigned_task_id,
      capability_bind_receipt=bound_execution.bind.to_json(),
      on_complete=None,
    )
    if error is not None:
      raise RuntimeError(f"failed to register model call job: {error}")

    job_info: dict[str, Any] = {
      **(launched or {}),
      "operation": op.name,
      "kind": op.kind,
      "task_id": assigned_task_id,
      "bind": bound_execution.bind.to_json(),
      "selection_provenance": bound_execution.bind.model_dump(mode="json"),
    }

    if wait:
      registry = getattr(runner, "_task_registry", None)
      entry = registry.get(assigned_task_id) if registry is not None else None
      if entry is not None and entry.asyncio_task is not None:
        if timeout is not None:
          await asyncio.wait_for(entry.asyncio_task, timeout=timeout)
        else:
          await entry.asyncio_task
      return {
        **job_info,
        "result": entry.result if entry is not None else None,
        "status": entry.state.value if entry is not None else "completed",
      }
    return job_info

  # Direct execution fallback without runner
  from .send_prompt import send_prompt

  usage_events, record_usage = _usage_recorder()
  text = await send_prompt(
    prompt_text,
    system_prompt=system_prompt,
    capability_execution=bound_execution,
    user_id=user_id or "anonymous",
    session_id=session_id or "model-call",
    on_usage=record_usage,
    completion_record_root=None,
  )
  res = {
    "operation": op.name,
    "kind": op.kind,
    "content": text,
    "task_id": assigned_task_id,
    "bind": bound_execution.bind.to_json(),
    "capability_id": bound_execution.bind.capability_id,
    "model": bound_execution.bind.model_key,
    "provider": bound_execution.bind.provider,
    "selection_provenance": bound_execution.bind.model_dump(mode="json"),
    "usage": _usage_summary(usage_events),
  }
  if op.kind == "judgment":
    res["state"] = state
    res["questions"] = questions
    try:
      res["judgment"] = json.loads(text)
    except Exception:
      res["judgment"] = text
  return res


__all__ = [
  "BoundCapabilityExecution",
  "CapabilityAdapterResolver",
  "CapabilityExecutionResolver",
  "CredentialMaterializer",
  "FallbackSkip",
  "MaterializedCredential",
  "ModelFallbackHop",
  "derive_batch_capability_execution",
  "execute_leaf_model_call",
  "list_model_call_operations",
  "lookup_model_call_operation",
  "materialize_frozen_bind",
  "register_model_call_operation",
  "reset_model_call_operations_for_tests",
]
