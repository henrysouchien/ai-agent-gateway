"""Edge-minted single-user credentials, loaded only from a private bundle file."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import stat
from typing import Literal

from hank_contracts.user_claims import USER_CLAIM_KEY_SCHEME, derive_user_claim_key

USER_RUNTIME_CREDENTIALS_FILE_ENV = "HANK_USER_RUNTIME_CREDENTIALS_FILE"


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
  user: UserRuntimeIdentity
  desk: DeskReplica
  claim_signing: UserClaimSigningDelegation


def _text(value: object, name: str) -> str:
  if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
    raise ValueError(f"{name} must be a canonical non-empty string")
  return value


def _object(value: object, name: str) -> dict[str, object]:
  if not isinstance(value, dict):
    raise ValueError(f"{name} must be an object")
  return value


def bundle_from_json(value: dict[str, object]) -> UserRuntimeCredentials:
  """Validate the external bundle shape; product auth owns channel/role policy."""
  if value.get("schema_version") != "user-runtime-credentials.v1":
    raise ValueError("schema_version must be user-runtime-credentials.v1")
  user = _object(value.get("user"), "user")
  desk = _object(value.get("desk"), "desk")
  claim = _object(value.get("claim_signing"), "claim_signing")
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
  if claim.get("scheme") != USER_CLAIM_KEY_SCHEME:
    raise ValueError("claim_signing.scheme must equal USER_CLAIM_KEY_SCHEME")
  user_id = _text(claim.get("user_id"), "claim_signing.user_id")
  if user_id != identity.slug:
    raise ValueError("claim_signing.user_id must equal user.slug")
  user_email = _text(claim.get("user_email"), "claim_signing.user_email")
  if user_email != identity.email:
    raise ValueError("claim_signing.user_email must equal user.email")
  optional: dict[str, str | None] = {}
  for name in ("aws_access_key_id", "aws_secret_access_key", "aws_session_token", "expires_at"):
    if name not in desk:
      raise ValueError(f"desk.{name} is required (may be null)")
    optional[name] = None if desk[name] is None else _text(desk[name], f"desk.{name}")
  return UserRuntimeCredentials(
    schema_version="user-runtime-credentials.v1",
    storage_id=_text(value.get("storage_id"), "storage_id"),
    user=identity,
    desk=DeskReplica(url=_text(desk.get("url"), "desk.url"), **optional),
    claim_signing=UserClaimSigningDelegation(
      scheme=USER_CLAIM_KEY_SCHEME, user_id=user_id, user_email=user_email,
      key=_text(claim.get("key"), "claim_signing.key"),
    ),
  )


def bundle_to_json(bundle: UserRuntimeCredentials) -> dict[str, object]:
  value = asdict(bundle)
  value["user"]["capabilities"] = list(bundle.user.capabilities)
  return value


def mint_user_runtime_credentials(
  *, org_claim_key: str, identity: UserRuntimeIdentity, storage_id: str, desk: DeskReplica,
) -> UserRuntimeCredentials:
  bundle = UserRuntimeCredentials(
    schema_version="user-runtime-credentials.v1", storage_id=storage_id,
    user=identity, desk=desk,
    claim_signing=UserClaimSigningDelegation(
      scheme=USER_CLAIM_KEY_SCHEME, user_id=identity.slug, user_email=identity.email,
      key=derive_user_claim_key(org_claim_key, identity.slug, identity.email),
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
