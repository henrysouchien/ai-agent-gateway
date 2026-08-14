from __future__ import annotations

import os
import threading
import weakref
from typing import Any, Mapping, NamedTuple, Protocol

from .autonomous_runner_claims import sign_user_claim


GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV = (
  "AGENT_GATEWAY_CLAIM_SIGNING_AUTHORITY"
)
GATEWAY_CLAIM_SIGNING_AUTHORITY_FD_V1 = "fd-v1"
LEGACY_CLAIM_HMAC_KEY_ENV = "AGENT_API_USER_CLAIM_HMAC_KEY"
CLAIM_SIGNING_KEY_MIN_BYTES = 32
CLAIM_SIGNING_KEY_MAX_BYTES = 4096

_configured_authority: GatewayClaimSigningAuthority | None = None
_configured_authority_lock = threading.Lock()


def _canonical_secret(secret: bytes | str) -> str:
  if isinstance(secret, bytes):
    try:
      value = secret.decode("utf-8")
    except UnicodeDecodeError as exc:
      raise ValueError(
        "gateway claim-signing key must be valid UTF-8"
      ) from exc
  elif type(secret) is str:
    value = secret
  else:
    raise TypeError(
      "gateway claim-signing key must be bytes or str"
    )
  encoded = value.encode("utf-8")
  if not (
    CLAIM_SIGNING_KEY_MIN_BYTES
    <= len(encoded)
    <= CLAIM_SIGNING_KEY_MAX_BYTES
  ):
    raise ValueError(
      "gateway claim-signing key must be between "
      f"{CLAIM_SIGNING_KEY_MIN_BYTES} and "
      f"{CLAIM_SIGNING_KEY_MAX_BYTES} bytes"
    )
  if value != value.strip() or "\x00" in value:
    raise ValueError(
      "gateway claim-signing key must be canonical"
    )
  return value


class GatewayClaimSigningAuthority:
  """Process-local signing authority whose secret is never child authority."""

  __slots__ = ("__secret",)

  def __init__(self, secret: bytes | str) -> None:
    self.__secret = _canonical_secret(secret)

  def __repr__(self) -> str:
    return "GatewayClaimSigningAuthority(secret=<redacted>)"

  @classmethod
  def from_one_shot_fd(
    cls,
    fd: int,
  ) -> GatewayClaimSigningAuthority:
    if type(fd) is not int or fd < 0:
      raise ValueError(
        "claim-signing key descriptor must be a nonnegative integer"
      )
    chunks: list[bytes] = []
    remaining = CLAIM_SIGNING_KEY_MAX_BYTES + 1
    try:
      os.set_inheritable(fd, False)
      while remaining:
        chunk = os.read(fd, min(remaining, 4096))
        if not chunk:
          break
        chunks.append(chunk)
        remaining -= len(chunk)
    except OSError as exc:
      raise RuntimeError(
        "claim-signing key descriptor could not be consumed"
      ) from exc
    finally:
      try:
        os.close(fd)
      except OSError:
        pass
    secret = b"".join(chunks)
    if len(secret) > CLAIM_SIGNING_KEY_MAX_BYTES:
      raise ValueError(
        "gateway claim-signing key exceeds its byte bound"
      )
    return cls(secret)

  def sign_user_claim(
    self,
    *,
    user_id: str,
    user_email: str | None,
    ttl_seconds: int,
  ) -> dict[str, str]:
    return sign_user_claim(
      self.__secret,
      user_id=user_id,
      user_email=user_email,
      ttl_seconds=ttl_seconds,
    )

  def bind_user(
    self,
    *,
    user_id: str,
    user_email: str | None,
  ) -> GatewayUserClaimSigner:
    return GatewayUserClaimSigner(
      self,
      user_id=user_id,
      user_email=user_email,
    )

  def sign_autonomous_launch_envelope(
    self,
    **kwargs: Any,
  ) -> str:
    from .autonomous_launch_envelope import (
      sign_autonomous_launch_envelope,
    )

    return sign_autonomous_launch_envelope(
      self.__secret,
      **kwargs,
    )

  def verify_autonomous_launch_envelope(
    self,
    envelope_json: str,
    **kwargs: Any,
  ) -> Any:
    from .autonomous_launch_envelope import (
      verify_autonomous_launch_envelope,
    )

    return verify_autonomous_launch_envelope(
      self.__secret,
      envelope_json,
      **kwargs,
    )

  def verify_user_claim(
    self,
    claim_headers: Mapping[str, str],
    *,
    ttl_ceiling: int,
    now: int | None = None,
  ) -> dict[str, Any] | None:
    """Verify a claim without making the root secret caller-observable."""

    from .server_artifact_helpers import (
      _verify_agent_claim_headers,
    )

    return _verify_agent_claim_headers(
      self.__secret,
      claim_headers,
      ttl_ceiling=ttl_ceiling,
      now=now,
    )


class _GatewayUserClaimBinding(NamedTuple):
  authority: GatewayClaimSigningAuthority
  user_email: str | None
  user_id: str


_gateway_user_claim_bindings: dict[
  int,
  tuple[weakref.ReferenceType[object], _GatewayUserClaimBinding],
] = {}
_gateway_user_claim_bindings_lock = threading.Lock()


def _discard_gateway_user_claim_binding(
  signer_identity: int,
  signer_reference: weakref.ReferenceType[object],
) -> None:
  with _gateway_user_claim_bindings_lock:
    current = _gateway_user_claim_bindings.get(signer_identity)
    if current is not None and current[0] is signer_reference:
      _gateway_user_claim_bindings.pop(signer_identity, None)


def _gateway_user_claim_binding(
  signer: object,
) -> _GatewayUserClaimBinding:
  if type(signer) is not GatewayUserClaimSigner:
    raise TypeError("gateway user claim signer type changed")
  with _gateway_user_claim_bindings_lock:
    current = _gateway_user_claim_bindings.get(id(signer))
    if current is None or current[0]() is not signer:
      raise RuntimeError("gateway user claim signer is not bound")
    binding = current[1]
  if type(binding) is not _GatewayUserClaimBinding:
    raise RuntimeError("gateway user claim signer is not bound")
  return binding


class GatewayUserClaimSigner:
  """Parent-process signer bound to exactly one gateway user."""

  __slots__ = ("__weakref__",)

  def __init__(
    self,
    authority: GatewayClaimSigningAuthority,
    *,
    user_id: str,
    user_email: str | None,
  ) -> None:
    if type(authority) is not GatewayClaimSigningAuthority:
      raise TypeError(
        "gateway user claim signer requires exact signing authority"
      )
    if type(user_id) is not str or not user_id or user_id != user_id.strip():
      raise ValueError(
        "gateway user claim signer requires canonical user_id"
      )
    if user_email is not None and (
      type(user_email) is not str
      or not user_email
      or user_email != user_email.strip()
    ):
      raise ValueError(
        "gateway user claim signer requires canonical user_email"
      )
    binding = _GatewayUserClaimBinding(
      authority=authority,
      user_email=user_email,
      user_id=user_id,
    )
    signer_identity = id(self)
    signer_reference = weakref.ref(
      self,
      lambda reference: _discard_gateway_user_claim_binding(
        signer_identity,
        reference,
      ),
    )
    with _gateway_user_claim_bindings_lock:
      current = _gateway_user_claim_bindings.get(signer_identity)
      if current is not None and current[0]() is self:
        raise RuntimeError(
          "gateway user claim signer is already bound"
        )
      if current is not None and current[0]() is not None:
        raise RuntimeError(
          "gateway user claim signer identity collision"
        )
      _gateway_user_claim_bindings[signer_identity] = (
        signer_reference,
        binding,
      )

  def __init_subclass__(cls, **kwargs: Any) -> None:
    raise TypeError("gateway user claim signer cannot be subclassed")

  def __repr__(self) -> str:
    return (
      "GatewayUserClaimSigner("
      "user_id=<bound>, user_email=<bound>, authority=<redacted>)"
    )

  @property
  def user_id(self) -> str:
    return _gateway_user_claim_binding(self).user_id

  @property
  def user_email(self) -> str | None:
    return _gateway_user_claim_binding(self).user_email


def gateway_user_claim_signer_identity(
  signer: GatewayUserClaimSigner,
) -> tuple[str, str | None]:
  """Resolve one exact gateway signer without virtual dispatch."""

  binding = _gateway_user_claim_binding(signer)
  return binding.user_id, binding.user_email


def sign_gateway_user_claim(
  signer: GatewayUserClaimSigner,
  *,
  ttl_seconds: int,
) -> dict[str, str]:
  """Sign through the module-owned exact-type gateway capability."""

  binding = _gateway_user_claim_binding(signer)
  return binding.authority.sign_user_claim(
    user_id=binding.user_id,
    user_email=binding.user_email,
    ttl_seconds=ttl_seconds,
  )


class UserClaimSigner(Protocol):
  """Identity-bound signer marker used by shared runtime plumbing."""

  @property
  def user_id(self) -> str: ...

  @property
  def user_email(self) -> str | None: ...


def _publish_claim_signing_authority_environment() -> None:
  environment = os.environ
  try:
    environment.pop(
      GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV,
      None,
    )
    environment.pop(LEGACY_CLAIM_HMAC_KEY_ENV, None)
    environment[GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV] = (
      GATEWAY_CLAIM_SIGNING_AUTHORITY_FD_V1
    )
  except BaseException:
    # A failed final write may have changed the mapping before raising.
    # Best-effort removal leaves child launch fail-closed instead of
    # advertising a signing authority that was never installed.
    try:
      environment.pop(
        GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV,
        None,
      )
    except BaseException:
      pass
    raise


def install_gateway_claim_signing_authority(
  authority: GatewayClaimSigningAuthority,
) -> None:
  if type(authority) is not GatewayClaimSigningAuthority:
    raise TypeError(
      "configured claim-signing authority must be exact"
    )
  global _configured_authority
  with _configured_authority_lock:
    if (
      _configured_authority is not None
      and _configured_authority is not authority
    ):
      raise RuntimeError(
        "gateway claim-signing authority is already installed"
      )
    # Supported readers take this same lock. Publish the environment contract
    # first and the process-local authority pointer last, so no reader can
    # observe an installed authority before the legacy secret is scrubbed.
    _publish_claim_signing_authority_environment()
    _configured_authority = authority


def configured_gateway_claim_signing_authority(
  *,
  required: bool = False,
) -> GatewayClaimSigningAuthority | None:
  with _configured_authority_lock:
    authority = _configured_authority
  if required and authority is None:
    raise RuntimeError(
      "gateway claim-signing authority is not installed"
    )
  return authority


__all__ = [
  "CLAIM_SIGNING_KEY_MAX_BYTES",
  "CLAIM_SIGNING_KEY_MIN_BYTES",
  "GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV",
  "GATEWAY_CLAIM_SIGNING_AUTHORITY_FD_V1",
  "GatewayClaimSigningAuthority",
  "GatewayUserClaimSigner",
  "UserClaimSigner",
  "LEGACY_CLAIM_HMAC_KEY_ENV",
  "configured_gateway_claim_signing_authority",
  "gateway_user_claim_signer_identity",
  "install_gateway_claim_signing_authority",
  "sign_gateway_user_claim",
]
