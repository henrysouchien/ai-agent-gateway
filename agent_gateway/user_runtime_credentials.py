"""Edge-minted single-user credentials, loaded only from a private bundle file."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
from typing import Literal, cast

from hank_contracts.user_claims import USER_CLAIM_KEY_SCHEME, derive_user_claim_key

USER_RUNTIME_CREDENTIALS_FILE_ENV = "HANK_USER_RUNTIME_CREDENTIALS_FILE"
# The placed runtime's storage id: a non-secret identifier, unlike the bundle,
# so children inherit it and per-user state roots derive from it alone.
USER_STORAGE_ID_ENV = "HANK_USER_STORAGE_ID"
APPROVAL_CLAIM_KEY_SCHEME = "user-claim-approval.v1"


@dataclass(frozen=True)
class UserRuntimeIdentity:
  key: str = field(repr=False)
  channel: str
  slug: str
  email: str
  risk_user_id: int
  role: str
  capabilities: tuple[str, ...]


@dataclass(frozen=True)
class DeskReplica:
  url: str
  aws_access_key_id: str | None = field(repr=False)
  aws_secret_access_key: str | None = field(repr=False)
  aws_session_token: str | None = field(repr=False)
  expires_at: str | None


@dataclass(frozen=True)
class UserClaimSigningDelegation:
  scheme: str
  user_id: str
  user_email: str
  key: str = field(repr=False)


@dataclass(frozen=True)
class UserRuntimeCredentials:
  schema_version: Literal["user-runtime-credentials.v1"]
  storage_id: str
  generation: int
  user: UserRuntimeIdentity
  desk: DeskReplica
  claim_signing: UserClaimSigningDelegation
  approval_claim_signing: UserClaimSigningDelegation


def _text(value: object, name: str) -> str:
  if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
    raise ValueError(f"{name} must be a canonical non-empty string")
  return value


def derive_approval_claim_key(
  org_hmac_secret: str, user_id: str, user_email: str, generation: int,
) -> str:
  """Derive an approval-only generation key directly from the organization secret.

  A runtime's generic claim key cannot derive this key or a replacement's key.
  Length-prefixed identity fields prevent newline-separated identity collisions.
  Inputs are canonical typed values supplied by the credential/placement owner.
  """
  uid = user_id.encode("utf-8")
  email = user_email.encode("utf-8")
  message = (
    APPROVAL_CLAIM_KEY_SCHEME.encode("ascii") + b"\n"
    + str(len(uid)).encode("ascii") + b":" + uid + b"\n"
    + str(len(email)).encode("ascii") + b":" + email + b"\n"
    + str(generation).encode("ascii")
  )
  return hmac.new(org_hmac_secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _object(value: object, name: str) -> dict[str, object]:
  if not isinstance(value, dict):
    raise ValueError(f"{name} must be an object")
  return value


def bundle_from_json(value: dict[str, object]) -> UserRuntimeCredentials:
  """Validate the external bundle shape; product auth owns channel/role policy."""
  if value.get("schema_version") != "user-runtime-credentials.v1":
    raise ValueError("schema_version must be user-runtime-credentials.v1")
  generation = value.get("generation")
  if type(generation) is not int or generation <= 0:
    raise ValueError("generation must be a positive integer")
  user = _object(value.get("user"), "user")
  desk = _object(value.get("desk"), "desk")
  delegations = {
    name: _object(value.get(name), name)
    for name in ("claim_signing", "approval_claim_signing")
  }
  risk_id = user.get("risk_user_id")
  if type(risk_id) is not int or risk_id <= 0:
    raise ValueError("user.risk_user_id must be a positive integer")
  capabilities = user.get("capabilities")
  if not isinstance(capabilities, list):
    raise ValueError("user.capabilities must be an array")
  identity = UserRuntimeIdentity(
    key=_text(user.get("key"), "user.key"),
    channel=_text(user.get("channel"), "user.channel"),
    slug=_text(user.get("slug"), "user.slug"),
    email=_text(user.get("email"), "user.email"),
    risk_user_id=risk_id,
    role=_text(user.get("role"), "user.role"),
    capabilities=tuple(_text(item, "user.capabilities") for item in capabilities),
  )
  signing: dict[str, UserClaimSigningDelegation] = {}
  for name, scheme in (
    ("claim_signing", USER_CLAIM_KEY_SCHEME),
    ("approval_claim_signing", APPROVAL_CLAIM_KEY_SCHEME),
  ):
    claim = delegations[name]
    if claim.get("scheme") != scheme:
      raise ValueError(f"{name}.scheme must equal {scheme}")
    user_id = _text(claim.get("user_id"), f"{name}.user_id")
    if user_id != identity.slug:
      raise ValueError(f"{name}.user_id must equal user.slug")
    user_email = _text(claim.get("user_email"), f"{name}.user_email")
    if user_email != identity.email:
      raise ValueError(f"{name}.user_email must equal user.email")
    signing[name] = UserClaimSigningDelegation(
      scheme=scheme, user_id=user_id, user_email=user_email,
      key=_text(claim.get("key"), f"{name}.key"),
    )
  optional: dict[str, str | None] = {}
  for name in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token", "expires_at"):
    if name not in desk:
      raise ValueError(f"desk.{name} is required (may be null)")
    optional[name] = None if desk[name] is None else _text(desk[name], f"desk.{name}")
  return UserRuntimeCredentials(
    schema_version="user-runtime-credentials.v1",
    storage_id=_text(value.get("storage_id"), "storage_id"),
    generation=generation,
    user=identity,
    desk=DeskReplica(url=_text(desk.get("url"), "desk.url"), **optional),
    claim_signing=signing["claim_signing"],
    approval_claim_signing=signing["approval_claim_signing"],
  )


def bundle_to_json(bundle: UserRuntimeCredentials) -> dict[str, object]:
  value = asdict(bundle)
  value["user"]["capabilities"] = list(bundle.user.capabilities)
  return value


def mint_user_runtime_credentials(
  *, org_claim_key: str, identity: UserRuntimeIdentity, storage_id: str, desk: DeskReplica,
  generation: int,
) -> UserRuntimeCredentials:
  bundle = UserRuntimeCredentials(
    schema_version="user-runtime-credentials.v1", storage_id=storage_id, generation=generation,
    user=identity, desk=desk,
    claim_signing=UserClaimSigningDelegation(
      scheme=USER_CLAIM_KEY_SCHEME, user_id=identity.slug, user_email=identity.email,
      key=derive_user_claim_key(org_claim_key, identity.slug, identity.email),
    ),
    approval_claim_signing=UserClaimSigningDelegation(
      scheme=APPROVAL_CLAIM_KEY_SCHEME, user_id=identity.slug, user_email=identity.email,
      key=derive_approval_claim_key(org_claim_key, identity.slug, identity.email, generation),
    ),
  )
  return bundle_from_json(bundle_to_json(bundle))


def load_user_runtime_credentials(environ: Mapping[str, str]) -> UserRuntimeCredentials | None:
  if USER_RUNTIME_CREDENTIALS_FILE_ENV not in environ:
    return None
  path = Path(_text(environ[USER_RUNTIME_CREDENTIALS_FILE_ENV], USER_RUNTIME_CREDENTIALS_FILE_ENV))
  try:
    with path.open("r", encoding="utf-8") as stream:
      metadata = os.fstat(stream.fileno())
      if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("user runtime credentials must be a regular file")
      mode = stat.S_IMODE(metadata.st_mode)
      if mode & 0o077:
        raise ValueError(f"user runtime credentials file mode {mode:04o} permits group/other access")
      value = json.load(stream)
  except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise ValueError(f"cannot read user runtime credentials file: {exc}") from exc
  return bundle_from_json(_object(value, "bundle"))


def positive_int(value: object) -> int | None:
  if isinstance(value, bool):
    return None
  if isinstance(value, int):
    return value if value > 0 else None
  if isinstance(value, str):
    cleaned = value.strip()
    if cleaned.isdecimal():
      parsed = int(cleaned)
      return parsed if parsed > 0 else None
  return None


def normalized_str(value: object) -> str | None:
  if isinstance(value, bool):
    return None
  cleaned = str(value or "").strip()
  return cleaned or None


class MalformedGatewayUserKeys(ValueError):
  """GATEWAY_USER_KEYS is not a JSON array. `base` and `detail` are the facts."""

  def __init__(self, *, code: Literal["invalid_json", "not_array"], detail: str | None = None) -> None:
    self.code = code
    self.detail = detail
    self.base = (
      "GATEWAY_USER_KEYS must be a JSON array"
      if code == "not_array"
      else "GATEWAY_USER_KEYS not valid JSON"
    )
    message = self.base if detail is None else f"{self.base}: {detail}"
    super().__init__(message)


def gateway_user_key_entries(environ: Mapping[str, str]) -> list[dict[str, object]]:
  """Parse GATEWAY_USER_KEYS. Absent or blank is []; bad JSON or a non-array raises MalformedGatewayUserKeys."""
  raw_keys = (environ.get("GATEWAY_USER_KEYS") or "").strip()
  if not raw_keys:
    return []
  try:
    raw_entries: object = json.loads(raw_keys)
  except json.JSONDecodeError as exc:
    raise MalformedGatewayUserKeys(code="invalid_json", detail=str(exc)) from exc
  if not isinstance(raw_entries, list):
    raise MalformedGatewayUserKeys(code="not_array")
  return [
    cast(dict[str, object], entry)
    for entry in raw_entries
    if isinstance(entry, dict)
  ]


def deployment_owner_scopes(environ: Mapping[str, str]) -> frozenset[str]:
  """Resolve deployment owners, preferring the runtime bundle over legacy keys."""
  bundle = load_user_runtime_credentials(environ)
  if bundle is not None:
    risk_uid = positive_int(bundle.user.risk_user_id)
    role = (normalized_str(bundle.user.role) or "").lower()
    if risk_uid is not None and role == "owner":
      return frozenset({str(risk_uid)})
    return frozenset()
  scopes: set[str] = set()
  for entry in gateway_user_key_entries(environ):
    risk_uid = positive_int(entry.get("risk_user_id"))
    slug = normalized_str(entry.get("slug"))
    if risk_uid is None or slug is None:
      continue
    role = (normalized_str(entry.get("role")) or "").lower()
    if role == "owner":
      scopes.add(str(risk_uid))
  return frozenset(scopes)


def bundle_storage_id(bundle: UserRuntimeCredentials, user_id: str, user_email: str | None) -> str | None:
  """The bundle's storage id for its own slug or email, else None: the one claim→storage map."""
  if user_id == bundle.user.slug or (
    user_email is not None and user_email.strip().lower() == bundle.user.email.lower()
  ):
    return bundle.storage_id
  return None


def write_user_runtime_credentials(bundle: UserRuntimeCredentials, path: str | os.PathLike[str]) -> None:
  value = bundle_to_json(bundle_from_json(bundle_to_json(bundle)))
  fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
      fd = -1
      json.dump(value, stream)
      stream.write("\n")
  finally:
    if fd >= 0:
      os.close(fd)
