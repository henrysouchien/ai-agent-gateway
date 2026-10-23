from __future__ import annotations

from dataclasses import dataclass, field
import logging
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Literal, Mapping, Sequence, cast

from model_authority.binding import CredentialPrincipal
from model_authority.capabilities import CAPABILITY_IDS
from .session_capabilities import normalize_session_capabilities


logger = logging.getLogger(__name__)


def _installed_provider_names() -> frozenset[str]:
  """Provider families the installed adapters declare; their own declarations own the set."""

  # Imported here: the providers package imports this module for its failure types.
  from .providers import installed_adapter_route_support

  return frozenset(support.provider for support in installed_adapter_route_support().values())


_VALID_BILLING_MODES = {"byok", "metered"}
CredentialFailureKind = Literal["rate_limit", "billing", "auth"]
AuthMechanism = Literal["api_key", "oauth_bearer", "none"]
EntitlementKind = Literal[
  "commercial_byok",
  "consumer_subscription",
  "org_commercial",
  "operator_personal_subscription",
  "platform_metered",
  "local_unauthenticated",
]
Placement = Literal["local", "hosted_external"]
AUTH_MECHANISMS: frozenset[str] = frozenset({"api_key", "oauth_bearer", "none"})
ENTITLEMENT_KINDS: frozenset[str] = frozenset({
  "commercial_byok",
  "consumer_subscription",
  "org_commercial",
  "operator_personal_subscription",
  "platform_metered",
  "local_unauthenticated",
})
PLACEMENTS: frozenset[str] = frozenset({"local", "hosted_external"})
LOCAL_DAEMON_ROUTES: frozenset[str] = frozenset({"ollama.local", "ollama.daemon_cloud"})
HOSTED_REJECTED_ROUTES: frozenset[str] = frozenset({
  "ollama.local",
  "ollama.daemon_cloud",
  "codex.chatgpt",
  "google.antigravity",
  "copilot.api",
  "anthropic.oauth",
})
# Flat-rate seats: the entitlement kinds whose credential is a subscription,
# whatever form it takes on the wire (Muse's seat reaches the wire as a key).
SUBSCRIPTION_ENTITLEMENT_KINDS: frozenset[str] = frozenset({
  "consumer_subscription",
  "operator_personal_subscription",
})
HOSTED_REJECTED_ENTITLEMENT_KINDS: frozenset[str] = SUBSCRIPTION_ENTITLEMENT_KINDS | {
  "local_unauthenticated",
}
# Channels whose requests run on hosted infrastructure for external users.
# Hosted key-channel egress is hosted_external by construction
# (`api/runtime_egress.py`), not by channel.
HOSTED_EXTERNAL_CHANNELS: frozenset[str] = frozenset({"web"})


def placement_for_channel(channel: str | None) -> Placement:
  """Single owner of which trusted channel runs on `hosted_external`."""

  if str(channel or "").strip().lower() in HOSTED_EXTERNAL_CHANNELS:
    return "hosted_external"
  return "local"


@dataclass(frozen=True)
class PlacementEligibility:
  """Whether a credential may run on a placement, and how it is billed there."""

  eligible: bool
  credential_principal: Literal["user", "service"] | None
  billing_mode: Literal["byok", "metered"] | None
  reason: str | None = None


def derive_auth_mechanism_and_entitlement(
  raw: Mapping[str, Any],
  *,
  is_user_store: bool | None = None,
) -> tuple[AuthMechanism, EntitlementKind]:
  """Classify a resolver or store record's wire mechanism and entitlement."""

  explicit_mechanism = raw.get("auth_mechanism")
  if explicit_mechanism in AUTH_MECHANISMS:
    auth_mechanism = cast(AuthMechanism, explicit_mechanism)
  else:
    provider = str(raw.get("provider") or "").strip().lower()
    auth_mode = str(raw.get("auth_mode") or "").strip().lower()
    if provider == "ollama":
      auth_mechanism = "none"
    elif auth_mode == "oauth" or (not raw.get("api_key") and raw.get("auth_token")):
      # Muse's device login is OAuth, but the wire credential is the minted key.
      if provider == "muse" and raw.get("api_key"):
        auth_mechanism = "api_key"
      else:
        auth_mechanism = "oauth_bearer"
    elif raw.get("api_key") or auth_mode == "api":
      if str(raw.get("api_key") or "").strip().startswith("sk-ant-oat"):
        auth_mechanism = "oauth_bearer"
      else:
        auth_mechanism = "api_key"
    else:
      auth_mechanism = "none"

  explicit_kind = raw.get("entitlement_kind")
  if explicit_kind in ENTITLEMENT_KINDS:
    entitlement_kind = cast(EntitlementKind, explicit_kind)
  else:
    if is_user_store is not None:
      user_scoped = is_user_store
    else:
      store_path = str(
        raw.get("auth_store_path") or raw.get("api_key_store_path") or ""
      ).replace("\\", "/")
      ownership = str(raw.get("ownership") or "")
      user_scoped = (
        "/users/" in store_path
        or (
          ownership.endswith((":personal", ":byok"))
          and not ownership.startswith("deployment:")
        )
      )
    token_text = str(raw.get("auth_token") or raw.get("api_key") or "").strip()
    source = str(raw.get("source") or "").strip().lower()
    provider = str(raw.get("provider") or "").strip().lower()
    if provider == "ollama":
      entitlement_kind = "local_unauthenticated"
    elif (
      token_text.startswith("sk-ant-oat")
      or source in {"subscription", "claude-setup-token", "anthropic-oauth-login"}
      or auth_mechanism == "oauth_bearer"
    ):
      entitlement_kind = (
        "consumer_subscription" if user_scoped else "operator_personal_subscription"
      )
    elif auth_mechanism == "api_key":
      entitlement_kind = "commercial_byok" if user_scoped else "org_commercial"
    else:
      entitlement_kind = "local_unauthenticated"
  return auth_mechanism, entitlement_kind


def evaluate_placement_eligibility(
  *,
  provider: str,
  route: str | None,
  entitlement_kind: str,
  placement: str = "local",
) -> PlacementEligibility:
  """Single owner of which entitlement may run on `placement`."""

  norm_placement = str(placement or "local").strip().lower()
  if norm_placement not in PLACEMENTS:
    raise ValueError(f"Unsupported placement: {placement!r}")
  norm_kind = str(entitlement_kind or "").strip().lower()
  norm_route = str(route or "").strip().lower()
  norm_provider = str(provider or "").strip().lower()
  if norm_placement == "hosted_external":
    if (
      norm_route in LOCAL_DAEMON_ROUTES
      or norm_provider == "ollama"
      or norm_kind == "local_unauthenticated"
    ):
      return PlacementEligibility(
        False, None, None, "local_daemon_route_not_allowed_on_hosted_external",
      )
    if norm_route in HOSTED_REJECTED_ROUTES:
      return PlacementEligibility(
        False, None, None, "subscription_route_not_allowed_on_hosted_external",
      )
    if norm_kind in HOSTED_REJECTED_ENTITLEMENT_KINDS:
      return PlacementEligibility(
        False, None, None, f"{norm_kind}_not_allowed_on_hosted_external",
      )
    if norm_kind == "commercial_byok":
      return PlacementEligibility(True, "user", "byok", None)
    if norm_kind in {"org_commercial", "platform_metered"}:
      return PlacementEligibility(True, "service", "metered", None)
    return PlacementEligibility(False, None, None, f"unsupported_entitlement_kind:{norm_kind}")
  if norm_kind in {"commercial_byok", "consumer_subscription"}:
    return PlacementEligibility(True, "user", "byok", None)
  if norm_kind in {
    "org_commercial",
    "operator_personal_subscription",
    "platform_metered",
    "local_unauthenticated",
  }:
    return PlacementEligibility(True, "service", "metered", None)
  return PlacementEligibility(False, None, None, f"unsupported_entitlement_kind:{norm_kind}")


def filter_fallback_entries_for_placement(
  entries: Sequence[Any],
  registry: Any,
  *,
  placement: str = "local",
  entitlement_by_provider: Mapping[str, str] | None = None,
) -> list[Any]:
  """Keep fallback entries whose route and entitlement may run on `placement`."""

  kept: list[Any] = []
  entitlements = entitlement_by_provider or {}
  for entry in entries:
    if hasattr(entry, "model_key"):
      model_key = entry.model_key
    elif hasattr(entry, "key"):
      model_key = entry.key
    else:
      model_key = str(entry)
    reg_entry = registry.models.get(model_key)
    if reg_entry is None:
      continue
    entitlement_kind = entitlements.get(reg_entry.provider)
    if not entitlement_kind:
      if reg_entry.route in LOCAL_DAEMON_ROUTES:
        entitlement_kind = "local_unauthenticated"
      elif reg_entry.route in HOSTED_REJECTED_ROUTES:
        entitlement_kind = "consumer_subscription"
      else:
        entitlement_kind = "org_commercial"
    if evaluate_placement_eligibility(
      provider=reg_entry.provider,
      route=reg_entry.route,
      entitlement_kind=entitlement_kind,
      placement=placement,
    ).eligible:
      kept.append(entry)
  return kept


@dataclass(frozen=True)
class AuthConfig:
  """Typed wrapper over the existing auth_config dict shape."""

  provider: str
  billing_mode: Literal["byok", "metered"]
  max_tokens: int | None
  _raw: Mapping[str, Any] = field(repr=False)
  auth_mechanism: AuthMechanism = "api_key"
  entitlement_kind: EntitlementKind = "org_commercial"

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> "AuthConfig":
    if "provider" not in d:
      raise ValueError("AuthConfig requires 'provider'. Include the provider name in the resolver output.")
    if "billing_mode" not in d:
      raise ValueError(
        "AuthConfig requires 'billing_mode'. Set it to 'byok' or 'metered' in the resolver output."
      )

    provider = str(d["provider"]).strip().lower()
    billing_mode = str(d["billing_mode"]).strip().lower()
    installed = _installed_provider_names()
    if provider not in installed:
      expected = ", ".join(sorted(installed))
      raise ValueError(f"Unsupported provider '{provider}'. Expected one of: {expected}.")
    if billing_mode not in _VALID_BILLING_MODES:
      raise ValueError(f"Unsupported billing_mode '{billing_mode}'. Expected 'byok' or 'metered'.")

    raw = dict(d)
    max_tokens = raw.get("max_tokens")
    if max_tokens is not None:
      max_tokens = int(max_tokens)
    auth_mechanism, entitlement_kind = derive_auth_mechanism_and_entitlement(raw)

    return cls(
      provider=provider,
      billing_mode=billing_mode,  # type: ignore[arg-type]
      max_tokens=max_tokens,
      auth_mechanism=auth_mechanism,
      entitlement_kind=entitlement_kind,
      _raw=MappingProxyType(raw),
    )

  def to_dict(self) -> dict[str, Any]:
    return dict(self._raw)


@dataclass
class ResolverResult:
  user_id: str
  channel: str
  auth_config: AuthConfig = field(repr=False)
  credential_principal: CredentialPrincipal
  allow_service_for_interactive: bool = False
  risk_user_id: int | None = None
  role: Literal["owner", "invite"] = "invite"
  user_email: str | None = None
  capabilities: frozenset[str] = frozenset()
  model_entitled_capabilities: frozenset[str] = frozenset()
  model_entitled_keys: frozenset[str] = frozenset()
  # Entitlements the resolver named that no registered capability matches.
  # They grant nothing (the intersection below); this field is where a session
  # reads which ones were dropped.
  unknown_model_entitled_capabilities: frozenset[str] = field(
    default=frozenset(), init=False
  )

  def __post_init__(self) -> None:
    if self.credential_principal not in {"user", "service"}:
      raise ValueError(
        "credential_principal must be declared as 'user' or 'service'"
      )
    if not isinstance(self.allow_service_for_interactive, bool):
      raise ValueError("allow_service_for_interactive must be a bool")
    self.capabilities = normalize_session_capabilities(self.capabilities)
    self.model_entitled_capabilities = frozenset(
      str(value or "").strip()
      for value in self.model_entitled_capabilities
      if str(value or "").strip()
    )
    self.unknown_model_entitled_capabilities = (
      self.model_entitled_capabilities - CAPABILITY_IDS
    )
    if self.unknown_model_entitled_capabilities:
      logger.warning(
        "unknown model-entitled capabilities grant nothing: %s",
        ", ".join(sorted(self.unknown_model_entitled_capabilities)),
      )
    self.model_entitled_capabilities &= CAPABILITY_IDS
    self.model_entitled_keys = frozenset(
      str(value or "").strip()
      for value in self.model_entitled_keys
      if str(value or "").strip()
    )


CredentialsResolver = Callable[
  [str, Any],
  Awaitable[ResolverResult],
]


@dataclass(frozen=True)
class ProviderCredentialFailure:
  """Sanitized provider failure that may be recoverable with fresh credentials."""

  provider: str
  kind: CredentialFailureKind
  status_code: int | None = None
  error_code: str | None = None
  message: str = ""
  retryable_with_new_credentials: bool = True
  # Epoch seconds at which the limiter said this credential becomes usable
  # again, when it said so at all. Derived from the provider projection of the
  # response's own reset/retry-after fields, never from a local policy.
  reset_at: float | None = None
  # What the limit covers: "shared" (the whole account, the default when None)
  # or a model family when the provider scoped the claim to one.
  limit_scope: str | None = None


@dataclass(frozen=True)
class CredentialRefreshRequest:
  """Request passed to a deployment-specific credential refresh resolver."""

  user_id: str
  user_email: str | None
  session_id: str
  api_key_hash: str
  channel: str | None
  provider: str
  billing_mode: str | None
  rate_table_version: str | None
  model: str | None
  auth_mode: str | None
  request_id: str | None
  failure: ProviderCredentialFailure


CredentialsRefreshResolver = Callable[
  [CredentialRefreshRequest],
  Awaitable[AuthConfig | None],
]


class NoCredentialError(Exception):
  def __init__(self, message: str | None = None) -> None:
    super().__init__(
      message
      or "No credential is configured for this user. Provide a BYOK credential or route the user to metered billing."
    )


class CredentialsTimeoutError(Exception):
  def __init__(self, message: str | None = None) -> None:
    super().__init__(
      message
      or "Credential resolution timed out. Check the resolver latency or raise resolver_timeout_seconds."
    )


class MissingUserIdError(Exception):
  def __init__(self, message: str | None = None) -> None:
    super().__init__(
      message or "user_id is required. Send the stable end-user ID at the gateway boundary."
    )


class ChannelMismatchError(Exception):
  """Request context.channel does not match the key's bound channel."""


class CrossUserReuseError(Exception):
  def __init__(self, message: str | None = None) -> None:
    super().__init__(
      message
      or "This session belongs to a different user. Create a separate gateway session for each end user."
    )


class AuthExpiredError(Exception):
  def __init__(self, message: str | None = None) -> None:
    super().__init__(
      message or "Provider credentials expired. Re-initialize the gateway session and retry the user's last message."
    )


__all__ = [
  "AUTH_MECHANISMS",
  "AuthConfig",
  "AuthExpiredError",
  "AuthMechanism",
  "ChannelMismatchError",
  "CredentialFailureKind",
  "CredentialRefreshRequest",
  "CredentialsRefreshResolver",
  "CredentialsResolver",
  "CredentialsTimeoutError",
  "CrossUserReuseError",
  "ENTITLEMENT_KINDS",
  "EntitlementKind",
  "HOSTED_REJECTED_ENTITLEMENT_KINDS",
  "SUBSCRIPTION_ENTITLEMENT_KINDS",
  "HOSTED_REJECTED_ROUTES",
  "LOCAL_DAEMON_ROUTES",
  "MissingUserIdError",
  "NoCredentialError",
  "PLACEMENTS",
  "Placement",
  "PlacementEligibility",
  "ProviderCredentialFailure",
  "ResolverResult",
  "derive_auth_mechanism_and_entitlement",
  "evaluate_placement_eligibility",
  "filter_fallback_entries_for_placement",
  "placement_for_channel",
]
