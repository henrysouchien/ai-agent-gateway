from __future__ import annotations

from dataclasses import dataclass, field, fields
import logging
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Literal, Mapping

from model_authority.binding import CredentialPrincipal
from model_authority.capabilities import CAPABILITY_IDS
from hank_model_transport.auth import (
  AuthMechanism,
  EntitlementKind,
  ProviderCredentialFailure,
  derive_auth_mechanism_and_entitlement,
)
from .session_capabilities import normalize_session_capabilities
from hank_model_transport.user_provider_keys import USER_KEY_PROVIDERS


logger = logging.getLogger(__name__)


def _installed_provider_names() -> frozenset[str]:
  """Provider families the installed adapters declare; their own declarations own the set."""

  from hank_model_transport.providers import installed_adapter_route_support

  return frozenset(support.provider for support in installed_adapter_route_support().values())


_VALID_BILLING_MODES = {"byok", "metered"}


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


PUBLIC_AUTH_CONFIG_FIELDS: frozenset[str] = frozenset(
  f.name for f in fields(AuthConfig) if not f.name.startswith("_")
) | frozenset({
  "auth_mode",
  "base_url",
  "baseURL",
  "compat",
  "integration_id",
  "ownership",
  "placement",
  "rate_table_version",
})

def validate_user_key_providers(providers: frozenset[str], *, field_name: str) -> frozenset[str]:
  if not isinstance(providers, frozenset) or not providers <= frozenset(USER_KEY_PROVIDERS):
    raise ValueError(f"{field_name} must be a frozenset of supported user-key providers")
  return providers


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
  enrolled_user_key_providers: frozenset[str] = frozenset()
  unavailable_user_key_providers: frozenset[str] = frozenset()
  # Entitlements the resolver named that no registered capability matches.
  # They grant nothing (the intersection below); this field is where a session
  # reads which ones were dropped.
  unknown_model_entitled_capabilities: frozenset[str] = field(
    default=frozenset(), init=False
  )

  def __post_init__(self) -> None:
    self.enrolled_user_key_providers = validate_user_key_providers(
      self.enrolled_user_key_providers, field_name="enrolled_user_key_providers",
    )
    self.unavailable_user_key_providers = validate_user_key_providers(
      self.unavailable_user_key_providers, field_name="unavailable_user_key_providers",
    )
    if self.enrolled_user_key_providers & self.unavailable_user_key_providers:
      raise ValueError("user-key provider enrollment cannot be both known and unavailable")
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


class SessionDriverUserKeyMissing(NoCredentialError):
  """The session driver's provider has no enrolled user key (``user_key_missing``).

  Every caller refuses it as the ``NoCredentialError`` it is. Only ``/chat/init``
  with a verified commercial claim opens ``pending`` as a metered-pending session:
  it holds no credential, and each turn binds only a verified metered V4's seat.
  """

  def __init__(self, message: str, *, provider: str, pending: ResolverResult) -> None:
    super().__init__(message)
    if (
      provider not in USER_KEY_PROVIDERS
      or pending.credential_principal != "user"
      or pending.allow_service_for_interactive
      or pending.auth_config.provider != provider
      or provider in pending.enrolled_user_key_providers
    ):
      raise ValueError("a metered-pending start is a keyless user start for its provider")
    self.provider = provider
    self.pending = pending


class CredentialsTimeoutError(Exception):
  def __init__(self, message: str | None = None) -> None:
    super().__init__(
      message
      or "Credential resolution timed out. Check the resolver latency."
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
  "AuthConfig",
  "AuthExpiredError",
  "ChannelMismatchError",
  "CredentialRefreshRequest",
  "CredentialsRefreshResolver",
  "CredentialsResolver",
  "CredentialsTimeoutError",
  "CrossUserReuseError",
  "MissingUserIdError",
  "NoCredentialError",
  "SessionDriverUserKeyMissing",
  "ResolverResult",
]
