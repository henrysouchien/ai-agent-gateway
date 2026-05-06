from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Literal, Mapping


_VALID_PROVIDERS = {"anthropic", "openai", "codex"}
_VALID_BILLING_MODES = {"byok", "metered"}
CredentialFailureKind = Literal["rate_limit", "billing", "auth"]


@dataclass(frozen=True)
class AuthConfig:
  """Typed wrapper over the existing auth_config dict shape."""

  provider: Literal["anthropic", "openai", "codex"]
  billing_mode: Literal["byok", "metered"]
  model: str | None
  max_tokens: int | None
  _raw: Mapping[str, Any]

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
      raise ValueError(f"Unsupported provider '{provider}'. Expected one of: anthropic, openai, codex.")
    if billing_mode not in _VALID_BILLING_MODES:
      raise ValueError(f"Unsupported billing_mode '{billing_mode}'. Expected 'byok' or 'metered'.")

    raw = dict(d)
    model = raw.get("model")
    max_tokens = raw.get("max_tokens")
    if max_tokens is not None:
      max_tokens = int(max_tokens)

    return cls(
      provider=provider,  # type: ignore[arg-type]
      billing_mode=billing_mode,  # type: ignore[arg-type]
      model=str(model) if model is not None else None,
      max_tokens=max_tokens,
      _raw=MappingProxyType(raw),
    )

  def to_dict(self) -> dict[str, Any]:
    return dict(self._raw)


CredentialsResolver = Callable[
  [str, Any],
  Awaitable[AuthConfig],
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


class StrictModeDefaultUserError(Exception):
  def __init__(self, message: str | None = None) -> None:
    super().__init__(
      message
      or (
        "Gateway is in strict multi-user mode. Supply a non-default user_id and thread it from the consumer auth "
        "middleware into /chat/init."
      )
    )


class MissingUserIdError(Exception):
  def __init__(self, message: str | None = None) -> None:
    super().__init__(
      message or "user_id is required in strict multi-user mode. Send the end-user ID in the /chat request body."
    )


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
  "CredentialFailureKind",
  "CredentialRefreshRequest",
  "CredentialsRefreshResolver",
  "CredentialsResolver",
  "CredentialsTimeoutError",
  "CrossUserReuseError",
  "MissingUserIdError",
  "NoCredentialError",
  "ProviderCredentialFailure",
  "StrictModeDefaultUserError",
]
