from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import threading
import weakref
from typing import Any, Callable, Generator, Literal, Mapping, NamedTuple, Protocol, overload

import httpx2
from hank_contracts.agent_api import (
  AGENT_API_CLAIM_ENV_VARS,
  AGENT_API_CLAIM_HEADERS,
)

from .autonomous_runner_claims import get_agent_api_claim_ttl_seconds, sign_user_claim
from .runtime_placement import _storage_id
from .ttl_policy import AGENT_API_CLAIM_TTL, TtlResolver, resolve_ttl_seconds


GATEWAY_CLAIM_SIGNING_AUTHORITY_ENV = (
  "AGENT_GATEWAY_CLAIM_SIGNING_AUTHORITY"
)
GATEWAY_CLAIM_SIGNING_AUTHORITY_FD_V1 = "fd-v1"
LEGACY_CLAIM_HMAC_KEY_ENV = "AGENT_API_USER_CLAIM_HMAC_KEY"
CLAIM_SIGNING_KEY_MIN_BYTES = 32
CLAIM_SIGNING_KEY_MAX_BYTES = 4096
NATIVE_PRODUCER_SIGNATURE_HEADER = "X-Agent-Code-Producer-Signature"

# The local multi-user gateway lists tools without selecting a user's data.
# The reserved .invalid address cannot name a Risk user for tools/call.
# Placed runtimes and one-user runs instead list as their bound user.
MCP_CATALOG_PRINCIPAL = ("mcp-catalog", "mcp-catalog@hank.invalid")

_configured_authority: GatewayClaimSigningAuthority | None = None
_configured_authority_lock = threading.Lock()


_NATIVE_RISK_REQUEST_DOMAIN = b"hank/native-risk-cell/v1\0"
_NATIVE_CELL_TOKEN_DOMAIN = b"hank/native-cell-token/v2\0"
_NATIVE_CELL_TOKEN_NONCE_BYTES = 32
_NATIVE_CELL_TOKEN_PATTERN = re.compile(
  r"v2_([0-9]+)_([0-9a-f]{64})_([0-9a-f]{64})", re.ASCII,
)


def _parse_native_cell_token(token: str) -> tuple[str, str, str] | None:
  """The one v2 token shape; the storage segment is `_storage_id`'s to judge."""
  match = _NATIVE_CELL_TOKEN_PATTERN.fullmatch(token)
  if match is None:
    return None
  storage, nonce, mac = match.groups()
  try:
    _storage_id(storage)
  except ValueError:
    return None
  return storage, nonce, mac


def native_cell_token_storage(token: str) -> str | None:
  """Parse the one v2 grammar without a key; None when `token` is not v2."""
  parsed = _parse_native_cell_token(token)
  return None if parsed is None else parsed[0]


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

  __slots__ = ("__secret", "_ttl_resolver")

  def __init__(
    self,
    secret: bytes | str,
    *,
    ttl_resolver: TtlResolver | None = None,
  ) -> None:
    self.__secret = _canonical_secret(secret)
    self._ttl_resolver = ttl_resolver

  def __repr__(self) -> str:
    return "GatewayClaimSigningAuthority(secret=<redacted>)"

  @classmethod
  def from_one_shot_fd(
    cls,
    fd: int,
    *,
    ttl_resolver: TtlResolver | None = None,
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
    return cls(secret, ttl_resolver=ttl_resolver)

  def resolve_user_claim_ttl_seconds(
    self,
    *,
    user_id: str,
    ttl_seconds: int | None = None,
  ) -> int:
    fallback = (
      get_agent_api_claim_ttl_seconds()
      if ttl_seconds is None else ttl_seconds
    )
    if self._ttl_resolver is None:
      return fallback
    return resolve_ttl_seconds(
      self._ttl_resolver, user_id, AGENT_API_CLAIM_TTL, fallback,
    )

  def sign_user_claim(
    self,
    *,
    user_id: str,
    user_email: str | None,
    ttl_seconds: int | None = None,
  ) -> dict[str, str]:
    resolved_ttl = self.resolve_user_claim_ttl_seconds(
      user_id=user_id,
      ttl_seconds=ttl_seconds,
    )
    return self._sign_resolved_user_claim(
      user_id=user_id,
      user_email=user_email,
      ttl_seconds=resolved_ttl,
    )

  def _sign_resolved_user_claim(
    self,
    *,
    user_id: str,
    user_email: str | None,
    ttl_seconds: int,
  ) -> dict[str, str]:
    """Mint the exact TTL already resolved and validated by the caller."""
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
    now: int | None = None,
  ) -> dict[str, Any] | None:
    """Verify a claim without making the root secret caller-observable."""

    from .server_artifact_helpers import (
      _verify_agent_claim_headers,
    )

    return _verify_agent_claim_headers(
      self.__secret,
      claim_headers,
      now=now,
    )

  def _native_mac(self, domain: bytes, *parts: bytes) -> bytes:
    mac = hmac.new(self.__secret.encode("utf-8"), digestmod=hashlib.sha256)
    mac.update(domain)
    for part in parts:
      mac.update(part)
    return mac.digest()

  def sign_native_risk_request(self, body: bytes) -> str:
    """Sign a producer callback body without exposing holder key material."""
    return self._native_mac(_NATIVE_RISK_REQUEST_DOMAIN, body).hex()

  def verify_native_risk_request(self, body: bytes, signature: str) -> bool:
    """Authenticate the producer callback without exposing holder key material."""
    try:
      supplied = bytes.fromhex(signature)
    except ValueError:
      return False
    return hmac.compare_digest(
      self._native_mac(_NATIVE_RISK_REQUEST_DOMAIN, body), supplied,
    )

  def _native_cell_token_mac(self, storage: str, nonce: str) -> str:
    return self._native_mac(
      _NATIVE_CELL_TOKEN_DOMAIN,
      storage.encode("ascii"), b"\0", nonce.encode("ascii"),
    ).hex()

  def mint_native_cell_token(self, storage_id: str) -> str:
    """Mint a key-bound cell token; the cell registry owns its liveness."""
    nonce = secrets.token_hex(_NATIVE_CELL_TOKEN_NONCE_BYTES)
    token = f"v2_{storage_id}_{nonce}_{self._native_cell_token_mac(storage_id, nonce)}"
    if native_cell_token_storage(token) != storage_id:
      raise ValueError("native cell token storage must be a canonical positive numeric identity")
    return token

  def verify_native_cell_token(self, token: str) -> str | None:
    """Return the token's storage id when it carries this key's MAC."""
    parsed = _parse_native_cell_token(token)
    if parsed is None:
      return None
    storage, nonce, supplied = parsed
    if not hmac.compare_digest(
      self._native_cell_token_mac(storage, nonce).encode("ascii"),
      supplied.encode("ascii"),
    ):
      return None
    return storage


class _GatewayUserClaimBinding(NamedTuple):
  authority: GatewayClaimSigningAuthority
  user_email: str | None
  user_id: str


# Entries leave with their signer. WeakKeyDictionary drops them without taking
# the lock below, so a collection that runs inside the locked section cannot
# wait on a lock its own thread holds.
_gateway_user_claim_bindings: weakref.WeakKeyDictionary[
  GatewayUserClaimSigner,
  _GatewayUserClaimBinding,
] = weakref.WeakKeyDictionary()
_gateway_user_claim_bindings_lock = threading.Lock()


def _gateway_user_claim_binding(
  signer: object,
) -> _GatewayUserClaimBinding:
  if type(signer) is not GatewayUserClaimSigner:
    raise TypeError("gateway user claim signer type changed")
  with _gateway_user_claim_bindings_lock:
    binding = _gateway_user_claim_bindings.get(signer)
  if binding is None:
    raise RuntimeError("gateway user claim signer is not bound")
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
    with _gateway_user_claim_bindings_lock:
      if _gateway_user_claim_bindings.get(self) is not None:
        raise RuntimeError(
          "gateway user claim signer is already bound"
        )
      _gateway_user_claim_bindings[self] = binding

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
  ttl_seconds: int | None = None,
) -> dict[str, str]:
  """Sign through the module-owned exact-type gateway capability."""

  binding = _gateway_user_claim_binding(signer)
  return binding.authority.sign_user_claim(
    user_id=binding.user_id,
    user_email=binding.user_email,
    ttl_seconds=ttl_seconds,
  )


def agent_api_claim_http_headers(
  claim: Mapping[str, str],
) -> dict[str, str]:
  """The seven `X-Agent-Claim-*` request headers of one signed claim."""

  return {
    AGENT_API_CLAIM_HEADERS[field]: claim[AGENT_API_CLAIM_ENV_VARS[field]]
    for field in AGENT_API_CLAIM_HEADERS
  }


class UserClaimHttpAuth(httpx2.Auth):
  """Signs every request of one connection with a fresh user claim.

  A claim's lifetime is short by design, so one set of headers fixed at
  connect time would expire under a long-lived connection. `mint_claim`
  returns a newly signed claim (env-named fields, as the signers produce)
  for the one user the connection is bound to.
  """

  def __init__(self, mint_claim: Callable[[], Mapping[str, str]]) -> None:
    self._mint_claim = mint_claim

  def auth_flow(
    self, request: httpx2.Request,
  ) -> Generator[httpx2.Request, httpx2.Response, None]:
    request.headers.update(agent_api_claim_http_headers(self._mint_claim()))
    yield request


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


@overload
def configured_gateway_claim_signing_authority(
  *,
  required: Literal[True],
) -> GatewayClaimSigningAuthority: ...


@overload
def configured_gateway_claim_signing_authority(
  *,
  required: bool = False,
) -> GatewayClaimSigningAuthority | None: ...


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
  "UserClaimHttpAuth",
  "UserClaimSigner",
  "LEGACY_CLAIM_HMAC_KEY_ENV",
  "MCP_CATALOG_PRINCIPAL",
  "agent_api_claim_http_headers",
  "NATIVE_PRODUCER_SIGNATURE_HEADER",
  "configured_gateway_claim_signing_authority",
  "gateway_user_claim_signer_identity",
  "install_gateway_claim_signing_authority",
  "sign_gateway_user_claim",
]
