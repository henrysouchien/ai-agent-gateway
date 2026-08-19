from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Literal, Mapping

from .capability_binding import CredentialPrincipal
from .model_registry import CAPABILITY_IDS
from .session_capabilities import normalize_session_capabilities


_VALID_PROVIDERS = {"anthropic", "openai", "codex", "xai"}
_VALID_BILLING_MODES = {"byok", "metered"}
CredentialFailureKind = Literal["rate_limit", "billing", "auth"]


@dataclass(frozen=True)
class AuthConfig:
  """Typed wrapper over the existing auth_config dict shape."""

  provider: Literal["anthropic", "openai", "codex", "xai"]
  billing_mode: Literal["byok", "metered"]
  max_tokens: int | None
  _raw: Mapping[str, Any] = field(repr=False)

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
    if provider not in _VALID_PROVIDERS:
      expected = ", ".join(sorted(_VALID_PROVIDERS))
      raise ValueError(f"Unsupported provider '{provider}'. Expected one of: {expected}.")
    if billing_mode not in _VALID_BILLING_MODES:
      raise ValueError(f"Unsupported billing_mode '{billing_mode}'. Expected 'byok' or 'metered'.")

    raw = dict(d)
    forbidden_selection = sorted(
      field_name
      for field_name in ("model", "model_key", "effort", "thinking")
      if field_name in raw
    )
    if forbidden_selection:
      raise ValueError(
        "AuthConfig credential material must not select execution identity: "
        + ", ".join(forbidden_selection)
      )
    max_tokens = raw.get("max_tokens")
    if max_tokens is not None:
      max_tokens = int(max_tokens)

    return cls(
      provider=provider,  # type: ignore[arg-type]
      billing_mode=billing_mode,  # type: ignore[arg-type]
      max_tokens=max_tokens,
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
    unknown = self.model_entitled_capabilities - CAPABILITY_IDS
    if unknown:
      raise ValueError(
        "unknown model-entitled capabilities: "
        f"{', '.join(sorted(unknown))}"
      )
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
  "AuthConfig",
  "AuthExpiredError",
  "ChannelMismatchError",
  "CredentialFailureKind",
  "CredentialRefreshRequest",
  "CredentialsRefreshResolver",
  "CredentialsResolver",
  "CredentialsTimeoutError",
  "CrossUserReuseError",
  "MissingUserIdError",
  "NoCredentialError",
  "ProviderCredentialFailure",
  "ResolverResult",
]
