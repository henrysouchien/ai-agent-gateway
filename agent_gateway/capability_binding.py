"""Single model-selection resolver and complete capability binding authority."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, NoReturn, TypeAlias

from agent_workflow_contracts import CapabilityBind

from .model_registry import (
  CAPABILITY_IDS,
  CapabilitySelectionPolicy,
  ModelRegistryEntry,
  ProductModelRegistry,
  ProductModelSelectionPolicy,
  SelectionSource,
)
from .thinking import EffortResolution, parse_effort


CapabilityId: TypeAlias = str
RunMode: TypeAlias = Literal["interactive", "fleet", "batch", "autonomous", "cron"]
CredentialPrincipal: TypeAlias = Literal["user", "service"]
IntentSource: TypeAlias = Literal["explicit_user", "saved_preference"]
CapabilityResolutionCode: TypeAlias = Literal[
  "unknown_capability",
  "capability_policy_missing",
  "capability_catalog_stale",
  "capability_model_not_allowed",
  "capability_model_unavailable",
  "capability_model_deprecated",
  "capability_model_disabled",
  "capability_model_revoked",
  "capability_effort_invalid",
  "capability_effort_unsupported",
  "capability_entitlement_required",
  "capability_externally_executed",
  "credential_unavailable",
  "default_not_eligible",
  "parent_binding_required",
  "parent_binding_incompatible",
  "provider_unavailable",
  "reported_identity_mismatch",
]

_RUN_MODES = frozenset({"interactive", "fleet", "batch", "autonomous", "cron"})
_CREDENTIAL_PRINCIPALS = frozenset({"user", "service"})


class CapabilityResolutionError(ValueError):
  """Typed, actionable, secret-free refusal raised before provider work."""

  def __init__(
    self,
    code: CapabilityResolutionCode,
    message: str,
    *,
    capability_id: str,
    model_key: str | None = None,
    provider: str | None = None,
    upstream_model: str | None = None,
    eligible_model_keys: tuple[str, ...] = (),
    catalog_revision: str | None = None,
  ) -> None:
    super().__init__(message)
    self.code = code
    self.capability_id = capability_id
    self.model_key = model_key
    self.provider = provider
    self.upstream_model = upstream_model
    self.eligible_model_keys = eligible_model_keys
    self.catalog_revision = catalog_revision

  def receipt(self) -> dict[str, object]:
    receipt: dict[str, object] = {
      "error_code": self.code,
      "capability_id": self.capability_id,
    }
    if self.model_key is not None:
      receipt["model_key"] = self.model_key
    if self.provider is not None:
      receipt["provider"] = self.provider
    if self.upstream_model is not None:
      receipt["upstream_model"] = self.upstream_model
    if self.eligible_model_keys:
      receipt["eligible_model_keys"] = list(self.eligible_model_keys)
    if self.catalog_revision is not None:
      receipt["catalog_revision"] = self.catalog_revision
    return receipt


def _required_text(value: object, *, field_name: str) -> str:
  text = str(value or "").strip()
  if not text:
    raise ValueError(f"{field_name} must be non-empty")
  return text


def _provider_family(value: object, *, field_name: str = "provider") -> str:
  provider = _required_text(value, field_name=field_name).lower()
  if provider == "agent-sdk":
    raise ValueError(
      f"{field_name} must name a credential provider family; "
      "'agent-sdk' is an adapter, not a provider"
    )
  return provider


def _canonical_effort(value: object, *, field_name: str) -> str:
  _required_text(value, field_name=field_name)
  effort = parse_effort(value, field_name=field_name)
  if effort is None:
    raise ValueError(f"{field_name} must be non-empty")
  return effort.value


def _refuse(
  code: CapabilityResolutionCode,
  message: str,
  *,
  capability_id: str,
  entry: ModelRegistryEntry | None = None,
  model_key: str | None = None,
  eligible_model_keys: tuple[str, ...] = (),
) -> NoReturn:
  raise CapabilityResolutionError(
    code,
    message,
    capability_id=capability_id,
    model_key=entry.key if entry is not None else model_key,
    provider=entry.provider if entry is not None else None,
    upstream_model=entry.upstream_model if entry is not None else None,
    eligible_model_keys=eligible_model_keys,
  )


@dataclass(frozen=True, slots=True)
class CredentialHandle:
  """Opaque reference to credential material owned by a credential authority."""

  handle_id: str
  provider: str
  principal: CredentialPrincipal
  tenant_id: str
  actor_id: str | None

  def __post_init__(self) -> None:
    handle_id = _required_text(self.handle_id, field_name="handle_id")
    provider = _provider_family(self.provider)
    principal = _required_text(self.principal, field_name="principal").lower()
    if principal not in _CREDENTIAL_PRINCIPALS:
      raise ValueError(f"unknown credential principal: {principal}")
    tenant_id = _required_text(self.tenant_id, field_name="tenant_id")
    actor_id = (
      _required_text(self.actor_id, field_name="actor_id")
      if self.actor_id is not None
      else None
    )
    if principal == "user" and actor_id is None:
      raise ValueError("user credential handles require actor_id")
    if principal == "service" and actor_id is not None:
      raise ValueError("service credential handles must not carry actor_id")
    object.__setattr__(self, "handle_id", handle_id)
    object.__setattr__(self, "provider", provider)
    object.__setattr__(self, "principal", principal)
    object.__setattr__(self, "tenant_id", tenant_id)
    object.__setattr__(self, "actor_id", actor_id)


@dataclass(frozen=True, slots=True)
class AuthContext:
  """Authenticated execution and independently admitted eligibility facts."""

  run_mode: RunMode
  actor_id: str
  tenant_id: str
  user_provider_handles: Mapping[str, CredentialHandle]
  service_provider_handles: Mapping[str, CredentialHandle]
  entitled_capabilities: frozenset[str]
  entitled_model_keys: frozenset[str]
  run_scoped_user_providers: frozenset[str] = frozenset()
  allow_service_for_interactive: bool = False

  def __post_init__(self) -> None:
    run_mode = _required_text(self.run_mode, field_name="run_mode").lower()
    if run_mode not in _RUN_MODES:
      raise ValueError(f"unknown run_mode: {run_mode}")
    actor_id = _required_text(self.actor_id, field_name="actor_id")
    tenant_id = _required_text(self.tenant_id, field_name="tenant_id")
    user_handles = self._validate_handles(
      self.user_provider_handles,
      principal="user",
      tenant_id=tenant_id,
      actor_id=actor_id,
    )
    service_handles = self._validate_handles(
      self.service_provider_handles,
      principal="service",
      tenant_id=tenant_id,
      actor_id=actor_id,
    )
    entitled_capabilities = frozenset(self.entitled_capabilities)
    unknown_capabilities = entitled_capabilities - CAPABILITY_IDS
    if unknown_capabilities:
      raise ValueError(
        "unknown entitled capabilities: "
        f"{', '.join(sorted(unknown_capabilities))}"
      )
    entitled_model_keys = frozenset(
      _required_text(value, field_name="entitled model key")
      for value in self.entitled_model_keys
    )
    run_scoped = frozenset(
      _provider_family(value) for value in self.run_scoped_user_providers
    )
    missing = run_scoped - set(user_handles)
    if missing:
      raise ValueError(
        "run-scoped user providers are missing credential handles: "
        f"{', '.join(sorted(missing))}"
      )
    if not isinstance(self.allow_service_for_interactive, bool):
      raise ValueError("allow_service_for_interactive must be a bool")
    object.__setattr__(self, "run_mode", run_mode)
    object.__setattr__(self, "actor_id", actor_id)
    object.__setattr__(self, "tenant_id", tenant_id)
    object.__setattr__(self, "user_provider_handles", MappingProxyType(user_handles))
    object.__setattr__(self, "service_provider_handles", MappingProxyType(service_handles))
    object.__setattr__(self, "entitled_capabilities", entitled_capabilities)
    object.__setattr__(self, "entitled_model_keys", entitled_model_keys)
    object.__setattr__(self, "run_scoped_user_providers", run_scoped)

  @staticmethod
  def _validate_handles(
    handles: Mapping[str, CredentialHandle],
    *,
    principal: CredentialPrincipal,
    tenant_id: str,
    actor_id: str,
  ) -> dict[str, CredentialHandle]:
    normalized: dict[str, CredentialHandle] = {}
    for raw_provider, handle in dict(handles).items():
      provider = _provider_family(raw_provider)
      if provider != handle.provider:
        raise ValueError("credential handle provider must match its mapping key")
      if handle.principal != principal:
        raise ValueError(
          f"{principal} credential map contains a {handle.principal} handle"
        )
      if handle.tenant_id != tenant_id:
        raise ValueError("credential handle tenant does not match AuthContext")
      if principal == "user" and handle.actor_id != actor_id:
        raise ValueError("user credential handle actor does not match AuthContext")
      normalized[provider] = handle
    return normalized

  def credential_handles(self, provider: str) -> tuple[CredentialHandle, ...]:
    family = _provider_family(provider)
    handles: list[CredentialHandle] = []
    user = self.user_provider_handles.get(family)
    if user is not None and (
      self.run_mode == "interactive" or family in self.run_scoped_user_providers
    ):
      handles.append(user)
    service = self.service_provider_handles.get(family)
    if service is not None and (
      self.run_mode in {"fleet", "batch", "autonomous", "cron"}
      or (self.run_mode == "interactive" and self.allow_service_for_interactive)
    ):
      handles.append(service)
    return tuple(handles)


@dataclass(frozen=True, slots=True)
class ModelSelectionIntent:
  """Stable-key selection intent; provider and upstream IDs are not accepted."""

  model_key: str
  effort: str | None
  source: IntentSource
  catalog_revision: str | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "model_key",
      _required_text(self.model_key, field_name="model_key"),
    )
    if self.effort is not None:
      object.__setattr__(
        self,
        "effort",
        _canonical_effort(self.effort, field_name="effort"),
      )
    if self.source not in {"explicit_user", "saved_preference"}:
      raise ValueError(f"unknown selection intent source: {self.source}")
    if self.catalog_revision is not None:
      object.__setattr__(
        self,
        "catalog_revision",
        _required_text(self.catalog_revision, field_name="catalog_revision"),
      )


@dataclass(frozen=True, slots=True)
class EligibleModelChoice:
  key: str
  label: str
  supported_efforts: tuple[str, ...]
  default_effort: str
  lifecycle: str


def _entry_for_key(
  registry: ProductModelRegistry,
  *,
  capability_id: str,
  model_key: str,
) -> ModelRegistryEntry:
  try:
    return registry.require(model_key)
  except KeyError:
    _refuse(
      "capability_model_unavailable",
      f"unknown stable model key {model_key!r}",
      capability_id=capability_id,
      model_key=model_key,
    )


def _user_exposed(
  *,
  capability_id: str,
  entry: ModelRegistryEntry,
  policy: CapabilitySelectionPolicy,
) -> bool:
  """Whether the entry is exposed to user selection, as eligible choices are."""

  exposure = entry.capabilities.get(capability_id)
  return (
    exposure == "user_selectable"
    or (
      exposure == "internal"
      and policy.allow_authenticated_run_override
    )
  )


def _validate_entry_for_policy(
  *,
  capability_id: str,
  entry: ModelRegistryEntry,
  policy: CapabilitySelectionPolicy,
  user_selected: bool = False,
) -> None:
  if entry.key not in policy.allowed_model_keys:
    _refuse(
      "capability_model_not_allowed",
      f"{entry.key} is not allowed for {capability_id}",
      capability_id=capability_id,
      entry=entry,
    )
  if capability_id not in entry.capabilities:
    _refuse(
      "capability_model_not_allowed",
      f"{entry.key} is not qualified for {capability_id}",
      capability_id=capability_id,
      entry=entry,
    )
  if user_selected:
    # The resolver is the security boundary: a user-driven selection may only
    # admit what eligible_model_choices would advertise for this capability.
    if not _user_exposed(
      capability_id=capability_id,
      entry=entry,
      policy=policy,
    ):
      _refuse(
        "capability_model_not_allowed",
        f"{entry.key} is not user-selectable for {capability_id}",
        capability_id=capability_id,
        entry=entry,
      )
  if entry.lifecycle == "deprecated":
    _refuse(
      "capability_model_deprecated",
      f"{entry.key} is deprecated and cannot start new work",
      capability_id=capability_id,
      entry=entry,
    )
  if entry.lifecycle == "disabled":
    _refuse(
      "capability_model_disabled",
      f"{entry.key} is disabled",
      capability_id=capability_id,
      entry=entry,
    )
  if entry.lifecycle == "revoked":
    _refuse(
      "capability_model_revoked",
      f"{entry.key} is revoked",
      capability_id=capability_id,
      entry=entry,
    )
  if user_selected and entry.lifecycle != "active":
    # Only "hidden" remains after the specific lifecycle refusals above:
    # eligible_model_choices never advertises it, so user-driven selection
    # refuses it rather than admitting a broader set than choices expose.
    _refuse(
      "capability_model_unavailable",
      f"{entry.key} is not available for selection for {capability_id}",
      capability_id=capability_id,
      entry=entry,
    )


def _policy_effort(
  policy: CapabilitySelectionPolicy,
  trusted_channel: str | None,
) -> str | None:
  """The applicable channel/capability policy effort for sourced selections.

  Design § Selection rules: effort follows the selection source only when the
  source explicitly carries an effort; otherwise the applicable capability or
  channel policy effort applies.  Only when the policy carries no effort (for
  example an inherit_parent default) does the registry entry's own default
  effort apply.
  """

  channel = str(trusted_channel or "").strip()
  if channel and channel in policy.by_channel:
    channel_effort = policy.by_channel[channel].effort
    if channel_effort is not None:
      return channel_effort
  return policy.default.effort


def _parent_entry(
  *,
  capability_id: str,
  registry: ProductModelRegistry,
  parent_bind: CapabilityBind,
) -> ModelRegistryEntry:
  """The current registry entry for an exact parent binding, identity-checked."""

  entry = _entry_for_key(
    registry,
    capability_id=capability_id,
    model_key=parent_bind.model_key,
  )
  if (
    entry.provider != parent_bind.provider
    or entry.upstream_model != parent_bind.upstream_model
    or entry.adapter != parent_bind.adapter
    or entry.protocol_profile != parent_bind.protocol_profile
    or entry.route != parent_bind.route
  ):
    _refuse(
      "parent_binding_incompatible",
      f"{capability_id} parent binding identity is inconsistent with its stable key",
      capability_id=capability_id,
      entry=entry,
    )
  return entry


def _eligible_handle(
  *,
  auth: AuthContext,
  capability_id: str,
  entry: ModelRegistryEntry,
  credential_ref: str | None = None,
  credential_principal: CredentialPrincipal | None = None,
) -> CredentialHandle | None:
  if capability_id not in auth.entitled_capabilities:
    return None
  if entry.key not in auth.entitled_model_keys:
    return None
  for handle in auth.credential_handles(entry.provider):
    if credential_ref is not None and handle.handle_id != credential_ref:
      continue
    if credential_principal is not None and handle.principal != credential_principal:
      continue
    return handle
  return None


def eligible_model_choices(
  capability_id: str,
  *,
  registry: ProductModelRegistry,
  selection_policy: ProductModelSelectionPolicy,
  auth: AuthContext,
) -> tuple[EligibleModelChoice, ...]:
  normalized = _required_text(capability_id, field_name="capability_id")
  policy = selection_policy.capabilities.get(normalized)
  if policy is None:
    _refuse(
      "capability_policy_missing",
      f"model-selection policy has no rule for {normalized}",
      capability_id=normalized,
    )
  choices: list[EligibleModelChoice] = []
  for key in sorted(policy.allowed_model_keys):
    entry = registry.require(key)
    if not _user_exposed(
      capability_id=normalized,
      entry=entry,
      policy=policy,
    ):
      continue
    if entry.lifecycle != "active":
      continue
    if _eligible_handle(
      auth=auth,
      capability_id=normalized,
      entry=entry,
    ) is None:
      continue
    choices.append(EligibleModelChoice(
      key=entry.key,
      label=entry.label,
      supported_efforts=tuple(sorted(entry.supported_efforts)),
      default_effort=entry.default_effort,
      lifecycle=entry.lifecycle,
    ))
  return tuple(choices)


SavedPreferenceIneligibilityReason: TypeAlias = Literal[
  "model_unknown",
  "model_deprecated",
  "model_disabled",
  "model_revoked",
  "model_hidden",
  "model_not_allowed",
  "effort_unsupported",
  "credential_ineligible",
]

_LIFECYCLE_INELIGIBILITY: Mapping[str, SavedPreferenceIneligibilityReason] = {
  "deprecated": "model_deprecated",
  "disabled": "model_disabled",
  "revoked": "model_revoked",
  "hidden": "model_hidden",
}


def saved_preference_ineligibility(
  saved_preference: ModelSelectionIntent,
  *,
  capability_id: str,
  registry: ProductModelRegistry,
  policy: CapabilitySelectionPolicy,
  auth: AuthContext,
  trusted_channel: str | None = None,
) -> SavedPreferenceIneligibilityReason | None:
  """Why a saved preference cannot be applied now, or None if it is eligible.

  This is the same eligibility rule that admits entries into
  ``eligible_model_choices`` (lifecycle, exposure, policy allowance,
  credential), plus the stored effort.  It never mutates or interprets the
  stored preference: an ineligible preference is reported as not applied and
  the applicable eligible default resolves instead.
  """

  try:
    entry = registry.require(saved_preference.model_key)
  except KeyError:
    return "model_unknown"
  if entry.lifecycle != "active":
    return _LIFECYCLE_INELIGIBILITY.get(entry.lifecycle, "model_hidden")
  if entry.key not in policy.allowed_model_keys:
    return "model_not_allowed"
  if not _user_exposed(
    capability_id=capability_id,
    entry=entry,
    policy=policy,
  ):
    return "model_not_allowed"
  effort = (
    saved_preference.effort
    or _policy_effort(policy, trusted_channel)
    or entry.default_effort
  )
  if effort not in entry.supported_efforts:
    return "effort_unsupported"
  if _eligible_handle(
    auth=auth,
    capability_id=capability_id,
    entry=entry,
  ) is None:
    return "credential_ineligible"
  return None


def _selection_candidate(
  capability_id: str,
  *,
  policy: CapabilitySelectionPolicy,
  registry: ProductModelRegistry,
  explicit_intent: ModelSelectionIntent | None,
  saved_preference: ModelSelectionIntent | None,
  authenticated_run_override: ModelSelectionIntent | None,
  trusted_channel: str | None,
  parent_bind: CapabilityBind | None,
) -> tuple[ModelRegistryEntry, str, SelectionSource]:
  sourced_policy_effort = _policy_effort(policy, trusted_channel)
  if explicit_intent is not None:
    if explicit_intent.source != "explicit_user" or not policy.allow_explicit_user:
      _refuse(
        "capability_model_not_allowed",
        f"explicit user selection is not allowed for {capability_id}",
        capability_id=capability_id,
        model_key=explicit_intent.model_key,
      )
    entry = _entry_for_key(
      registry,
      capability_id=capability_id,
      model_key=explicit_intent.model_key,
    )
    effort = (
      explicit_intent.effort
      or sourced_policy_effort
      or entry.default_effort
    )
    return entry, effort, "explicit_user"

  if saved_preference is not None:
    if saved_preference.source != "saved_preference" or not policy.allow_saved_preference:
      _refuse(
        "capability_model_not_allowed",
        f"saved preference is not allowed for {capability_id}",
        capability_id=capability_id,
        model_key=saved_preference.model_key,
      )
    entry = _entry_for_key(
      registry,
      capability_id=capability_id,
      model_key=saved_preference.model_key,
    )
    effort = (
      saved_preference.effort
      or sourced_policy_effort
      or entry.default_effort
    )
    return entry, effort, "saved_preference"

  if authenticated_run_override is not None:
    if (
      authenticated_run_override.source != "explicit_user"
      or not policy.allow_authenticated_run_override
    ):
      _refuse(
        "capability_model_not_allowed",
        f"authenticated run override is not allowed for {capability_id}",
        capability_id=capability_id,
        model_key=authenticated_run_override.model_key,
      )
    entry = _entry_for_key(
      registry,
      capability_id=capability_id,
      model_key=authenticated_run_override.model_key,
    )
    effort = (
      authenticated_run_override.effort
      or sourced_policy_effort
      or entry.default_effort
    )
    return entry, effort, "explicit_user"

  channel = str(trusted_channel or "").strip()
  if channel and channel in policy.by_channel:
    channel_default = policy.by_channel[channel]
    if channel_default.kind == "inherit_parent":
      if parent_bind is None:
        _refuse(
          "parent_binding_required",
          f"{capability_id} requires a parent binding",
          capability_id=capability_id,
        )
      entry = _parent_entry(
        capability_id=capability_id,
        registry=registry,
        parent_bind=parent_bind,
      )
      return entry, parent_bind.effort, "parent_binding"
    entry = registry.require(channel_default.model_key or "")
    return entry, channel_default.effort or entry.default_effort, "channel_default"

  if policy.default.kind == "inherit_parent":
    if parent_bind is None:
      _refuse(
        "parent_binding_required",
        f"{capability_id} requires an exact parent binding",
        capability_id=capability_id,
      )
    entry = _parent_entry(
      capability_id=capability_id,
      registry=registry,
      parent_bind=parent_bind,
    )
    return entry, parent_bind.effort, "parent_binding"

  entry = registry.require(policy.default.model_key or "")
  source: SelectionSource = (
    "capability_default" if capability_id == "session.driver" else "internal_policy"
  )
  return entry, policy.default.effort or entry.default_effort, source


_STALE_CATALOG_REFUSAL_CODES: frozenset[str] = frozenset({
  "capability_model_unavailable",
  "capability_model_not_allowed",
  "capability_model_deprecated",
  "capability_model_disabled",
  "capability_model_revoked",
  "capability_effort_unsupported",
  "capability_entitlement_required",
  "credential_unavailable",
})


def resolve_capability_model(
  capability_id: CapabilityId,
  *,
  registry: ProductModelRegistry,
  selection_policy: ProductModelSelectionPolicy,
  auth: AuthContext,
  explicit_intent: ModelSelectionIntent | None = None,
  saved_preference: ModelSelectionIntent | None = None,
  authenticated_run_override: ModelSelectionIntent | None = None,
  trusted_channel: str | None = None,
  parent_bind: CapabilityBind | None = None,
) -> CapabilityBind:
  """Resolve once from stable-key intent and freeze complete execution identity."""

  normalized = _required_text(capability_id, field_name="capability_id")
  if normalized not in CAPABILITY_IDS:
    _refuse(
      "unknown_capability",
      f"unknown capability_id: {normalized}",
      capability_id=normalized,
    )
  policy = selection_policy.capabilities.get(normalized)
  if policy is None:
    _refuse(
      "capability_policy_missing",
      f"model-selection policy has no rule for {normalized}",
      capability_id=normalized,
    )

  try:
    return _resolve_admitted_capability_model(
      normalized,
      registry=registry,
      selection_policy=selection_policy,
      policy=policy,
      auth=auth,
      explicit_intent=explicit_intent,
      saved_preference=saved_preference,
      authenticated_run_override=authenticated_run_override,
      trusted_channel=trusted_channel,
      parent_bind=parent_bind,
    )
  except CapabilityResolutionError as exc:
    if not exc.eligible_model_keys:
      # Refusals always carry the current eligible choices so a client can
      # recover explicitly (design § Failure and fallback behavior).
      choices = eligible_model_choices(
        normalized,
        registry=registry,
        selection_policy=selection_policy,
        auth=auth,
      )
      exc.eligible_model_keys = tuple(choice.key for choice in choices)
    observed_revision = (
      explicit_intent.catalog_revision if explicit_intent is not None else None
    )
    if (
      observed_revision is not None
      and observed_revision != registry.revision
      and exc.code in _STALE_CATALOG_REFUSAL_CODES
    ):
      # The catalog revision is concurrency context, not authority: a
      # still-eligible key is accepted despite a stale revision, but a key
      # that no longer resolves under the current catalog names the stale
      # revision so the client refreshes choices instead of retrying blind.
      raise CapabilityResolutionError(
        "capability_catalog_stale",
        (
          f"the selection was made against catalog revision "
          f"{observed_revision!r}; the current revision is "
          f"{registry.revision!r} and the selection no longer resolves"
        ),
        capability_id=normalized,
        model_key=exc.model_key,
        provider=exc.provider,
        upstream_model=exc.upstream_model,
        eligible_model_keys=exc.eligible_model_keys,
        catalog_revision=registry.revision,
      ) from exc
    raise


def _resolve_admitted_capability_model(
  normalized: str,
  *,
  registry: ProductModelRegistry,
  selection_policy: ProductModelSelectionPolicy,
  policy: CapabilitySelectionPolicy,
  auth: AuthContext,
  explicit_intent: ModelSelectionIntent | None,
  saved_preference: ModelSelectionIntent | None,
  authenticated_run_override: ModelSelectionIntent | None,
  trusted_channel: str | None,
  parent_bind: CapabilityBind | None,
) -> CapabilityBind:
  effective_saved_preference = saved_preference
  if saved_preference is not None:
    if saved_preference.source != "saved_preference" or not policy.allow_saved_preference:
      _refuse(
        "capability_model_not_allowed",
        f"saved preference is not allowed for {normalized}",
        capability_id=normalized,
        model_key=saved_preference.model_key,
      )
    if saved_preference_ineligibility(
      saved_preference,
      capability_id=normalized,
      registry=registry,
      policy=policy,
      auth=auth,
      trusted_channel=trusted_channel,
    ) is not None:
      # A saved preference is not current execution intent.  Preserve it in the
      # preference store, report it as not applied at the API boundary, and
      # continue with the applicable policy default.
      effective_saved_preference = None

  entry, effort, source = _selection_candidate(
    normalized,
    policy=policy,
    registry=registry,
    explicit_intent=explicit_intent,
    saved_preference=effective_saved_preference,
    authenticated_run_override=authenticated_run_override,
    trusted_channel=trusted_channel,
    parent_bind=parent_bind,
  )
  _validate_entry_for_policy(
    capability_id=normalized,
    entry=entry,
    policy=policy,
    user_selected=source in {"explicit_user", "saved_preference"},
  )
  resolved_effort = _canonical_effort(effort, field_name=f"{normalized}.effort")
  if resolved_effort not in entry.supported_efforts:
    _refuse(
      "capability_effort_unsupported",
      f"effort {resolved_effort!r} is unsupported by {entry.key}",
      capability_id=normalized,
      entry=entry,
    )

  if source == "parent_binding":
    if parent_bind is None:  # pragma: no cover - guarded by _selection_candidate
      _refuse(
        "parent_binding_required",
        f"{normalized} requires an exact parent binding",
        capability_id=normalized,
      )
    return _inherit_parent_binding_whole(
      normalized,
      entry=entry,
      auth=auth,
      parent_bind=parent_bind,
      resolved_effort=resolved_effort,
    )

  handle = _eligible_handle(
    auth=auth,
    capability_id=normalized,
    entry=entry,
  )
  if handle is None:
    code: CapabilityResolutionCode
    if source in {"capability_default", "channel_default", "internal_policy"}:
      code = "default_not_eligible"
    elif normalized not in auth.entitled_capabilities or entry.key not in auth.entitled_model_keys:
      code = "capability_entitlement_required"
    else:
      code = "credential_unavailable"
    _refuse(
      code,
      f"{entry.key} is not eligible for {normalized}; credential or entitlement action is required",
      capability_id=normalized,
      entry=entry,
    )

  return CapabilityBind(
    schema_version="1.0",
    capability_id=normalized,
    model_key=entry.key,
    provider=entry.provider,
    upstream_model=entry.upstream_model,
    adapter=entry.adapter,
    protocol_profile=entry.protocol_profile,
    route=entry.route,
    effort=resolved_effort,
    credential_principal=handle.principal,
    credential_ref=handle.handle_id,
    run_mode=auth.run_mode,
    registry_revision=registry.revision,
    policy_revision=selection_policy.revision,
    selection_source=source,
  )


def _inherit_parent_binding_whole(
  normalized: str,
  *,
  entry: ModelRegistryEntry,
  auth: AuthContext,
  parent_bind: CapabilityBind,
  resolved_effort: str,
) -> CapabilityBind:
  """Copy the exact parent binding whole for the inheriting capability.

  Product decision 4 / plan §6.D: exact parent inheritance copies the whole
  binding — execution identity, effort, credential principal and reference,
  run mode, and registry/policy revision provenance — with only the capability
  and ``selection_source: parent_binding`` naming the child.  The parent's
  credential is REauthorized against current facts (it may have been rotated
  or revoked); it is never re-selected, so an ineligible parent credential is
  a typed refusal, not a substitution.
  """

  handle = _eligible_handle(
    auth=auth,
    capability_id=normalized,
    entry=entry,
    credential_ref=parent_bind.credential_ref,
    credential_principal=parent_bind.credential_principal,
  )
  if handle is None:
    code: CapabilityResolutionCode
    if normalized not in auth.entitled_capabilities or entry.key not in auth.entitled_model_keys:
      code = "capability_entitlement_required"
    else:
      code = "credential_unavailable"
    _refuse(
      code,
      f"the parent binding's credential authority is not eligible for {normalized}",
      capability_id=normalized,
      entry=entry,
    )

  return CapabilityBind(
    schema_version="1.0",
    capability_id=normalized,
    model_key=parent_bind.model_key,
    provider=parent_bind.provider,
    upstream_model=parent_bind.upstream_model,
    adapter=parent_bind.adapter,
    protocol_profile=parent_bind.protocol_profile,
    route=parent_bind.route,
    effort=resolved_effort,
    credential_principal=parent_bind.credential_principal,
    credential_ref=parent_bind.credential_ref,
    run_mode=parent_bind.run_mode,
    registry_revision=parent_bind.registry_revision,
    policy_revision=parent_bind.policy_revision,
    selection_source="parent_binding",
  )


def reauthorize_capability_bind(
  bind: CapabilityBind,
  *,
  registry: ProductModelRegistry,
  auth: AuthContext,
) -> CredentialHandle:
  """Reauthorize a durable bind without consulting current defaults."""

  if not isinstance(bind, CapabilityBind):
    raise TypeError("reauthorization requires CapabilityBind")
  # The policy revision on the bind is provenance. Current selection policy
  # governs new work only; retry/resume rechecks current registry lifecycle,
  # entitlement, credential, and route facts without reinterpreting selection.
  entry = _entry_for_key(
    registry,
    capability_id=bind.capability_id,
    model_key=bind.model_key,
  )
  if (
    entry.provider != bind.provider
    or entry.upstream_model != bind.upstream_model
    or entry.adapter != bind.adapter
    or entry.protocol_profile != bind.protocol_profile
    or entry.route != bind.route
  ):
    _refuse(
      "capability_model_unavailable",
      "durable binding identity does not match its stable registry key",
      capability_id=bind.capability_id,
      entry=entry,
    )
  if bind.capability_id not in entry.capabilities:
    _refuse(
      "capability_model_not_allowed",
      "durable binding is no longer qualified for its capability",
      capability_id=bind.capability_id,
      entry=entry,
    )
  if entry.lifecycle == "disabled":
    _refuse(
      "capability_model_disabled",
      f"durable binding model {entry.key} is disabled",
      capability_id=bind.capability_id,
      entry=entry,
    )
  if entry.lifecycle == "revoked":
    _refuse(
      "capability_model_revoked",
      f"durable binding model {entry.key} is revoked",
      capability_id=bind.capability_id,
      entry=entry,
    )
  if bind.effort not in entry.supported_efforts:
    _refuse(
      "capability_effort_unsupported",
      f"durable binding effort is unsupported by {entry.key}",
      capability_id=bind.capability_id,
      entry=entry,
    )
  if bind.run_mode != auth.run_mode:
    _refuse(
      "credential_unavailable",
      "durable binding run mode does not match the authorization context",
      capability_id=bind.capability_id,
      entry=entry,
    )
  handle = _eligible_handle(
    auth=auth,
    capability_id=bind.capability_id,
    entry=entry,
    credential_ref=bind.credential_ref,
    credential_principal=bind.credential_principal,
  )
  if handle is None:
    _refuse(
      "credential_unavailable",
      "the durable binding's credential authority is no longer eligible",
      capability_id=bind.capability_id,
      entry=entry,
    )
  return handle


def require_capability_execution_bind(
  bind: CapabilityBind,
  *,
  provider: Any,
  auth_config: Mapping[str, Any],
) -> EffortResolution:
  """Validate exact provider, adapter, credential, model, and effort at execution."""

  provider_family = str(getattr(provider, "name", "") or "").strip().lower()
  if provider_family == "agent-sdk":
    provider_family = "anthropic"
  if provider_family != bind.provider:
    _refuse(
      "provider_unavailable",
      f"provider {provider_family!r} does not match binding provider {bind.provider!r}",
      capability_id=bind.capability_id,
      model_key=bind.model_key,
    )
  if not provider.has_active_credential(dict(auth_config)):
    _refuse(
      "credential_unavailable",
      f"credential material is unavailable for {bind.provider!r}",
      capability_id=bind.capability_id,
      model_key=bind.model_key,
    )

  requested = parse_effort(bind.effort, field_name=f"{bind.capability_id}.effort")
  if requested is None:
    _refuse(
      "capability_effort_invalid",
      f"{bind.capability_id} has no bound effort",
      capability_id=bind.capability_id,
      model_key=bind.model_key,
    )
  if bind.protocol_profile in {
    "messages.standard",
    "messages.oauth",
    "chat_completions.standard",
  }:
    if requested.value != "none":
      _refuse(
        "capability_effort_unsupported",
        f"protocol profile {bind.protocol_profile!r} admits only effort 'none'",
        capability_id=bind.capability_id,
        model_key=bind.model_key,
      )
    return EffortResolution(
      requested=requested,
      effective=requested,
      thinking_enabled_effective=False,
      payload_fragments={},
    )
  if bind.adapter == "anthropic.agent_sdk":
    if bind.provider != "anthropic" or requested.value not in {
      "none", "low", "medium", "high", "max",
    }:
      _refuse(
        "capability_effort_unsupported",
        f"bound effort is unsupported by {bind.adapter}",
        capability_id=bind.capability_id,
        model_key=bind.model_key,
      )
    return EffortResolution(
      requested=requested,
      effective=requested,
      thinking_enabled_effective=requested.value != "none",
      payload_fragments={},
    )

  model_info = provider.get_model_info(bind.upstream_model)
  configured_max_tokens = int(auth_config.get("max_tokens", 16_000))
  model_max_tokens = int(getattr(model_info, "max_output_tokens", 0) or 0)
  effective_max_tokens = (
    min(configured_max_tokens, model_max_tokens)
    if model_max_tokens > 0
    else configured_max_tokens
  )
  resolution = provider.resolve_effort(
    requested=requested,
    model=bind.upstream_model,
    model_info=model_info,
    max_tokens=effective_max_tokens,
    auth_mode=auth_config.get("auth_mode"),
    base_url=auth_config.get("base_url") or auth_config.get("baseURL"),
    compat=auth_config.get("compat"),
  )
  if resolution.requested != requested or resolution.effective != requested:
    _refuse(
      "capability_effort_unsupported",
      f"adapter cannot preserve bound effort for {bind.model_key}",
      capability_id=bind.capability_id,
      model_key=bind.model_key,
    )
  return resolution


def validate_reported_identity(
  bind: CapabilityBind,
  reported_identity: str,
  *,
  registry: ProductModelRegistry,
) -> str:
  """Record provider identity verbatim only when the registry admits it."""

  identity = _required_text(reported_identity, field_name="reported_identity")
  entry = _entry_for_key(
    registry,
    capability_id=bind.capability_id,
    model_key=bind.model_key,
  )
  if identity not in entry.reported_identities:
    _refuse(
      "reported_identity_mismatch",
      f"provider reported an unadmitted identity for {entry.key}",
      capability_id=bind.capability_id,
      entry=entry,
    )
  return identity


__all__ = [
  "AuthContext",
  "CAPABILITY_IDS",
  "CapabilityBind",
  "CapabilityId",
  "CapabilityResolutionCode",
  "CapabilityResolutionError",
  "CredentialHandle",
  "CredentialPrincipal",
  "EligibleModelChoice",
  "ModelSelectionIntent",
  "RunMode",
  "SavedPreferenceIneligibilityReason",
  "eligible_model_choices",
  "saved_preference_ineligibility",
  "reauthorize_capability_bind",
  "require_capability_execution_bind",
  "resolve_capability_model",
  "validate_reported_identity",
]
