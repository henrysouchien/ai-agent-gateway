"""Map a storage id, slug, or email to a positive risk_user_id.

One owner. The runtime bundle is the only identity when it is present;
otherwise the GATEWAY_USER_KEYS roster. Not the deployment-owner roster,
and never a legacy risk_user_id of 0.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from hank_model_transport.user_runtime_credentials import (
  MalformedGatewayUserKeys,
  bundle_to_json,
  gateway_user_key_entries as load_gateway_user_key_entries,
  load_user_runtime_credentials,
  normalized_str,
  positive_int,
)


@dataclass(frozen=True)
class StorageIdentity:
  """A parsed desk owner, separate from its login aliases."""

  storage_id: str
  form: Literal["numeric", "legacy"]


class StorageIdentityFinding(ValueError):
  """An external desk-owner input that the storage grammar cannot admit."""

  def __init__(self, input: object, reason: str) -> None:
    self.input = input
    self.reason = reason
    super().__init__(f"storage identity Finding: {reason}: {input!r}")


_NUMERIC_STORAGE_ID_RE = re.compile(r"[1-9][0-9]*")
_LEGACY_STORAGE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def parse_storage_identity(value: object) -> StorageIdentity:
  """Parse one canonical numeric owner or a lowercase, single-segment legacy id."""
  if type(value) is not str:
    raise StorageIdentityFinding(value, "storage id must be a string")
  if _NUMERIC_STORAGE_ID_RE.fullmatch(value) is not None:
    return StorageIdentity(value, "numeric")
  if value.isdecimal():
    raise StorageIdentityFinding(value, "numeric storage id must be canonical and positive")
  if _LEGACY_STORAGE_ID_RE.fullmatch(value) is not None:
    return StorageIdentity(value, "legacy")
  raise StorageIdentityFinding(value, "storage id must be a lowercase safe single segment")


class RunOwnerUnresolved(ValueError):
  """The identifier names no user with a positive risk_user_id."""

  def __init__(self, identifier: object, *, reason: str = "unknown") -> None:
    self.identifier = identifier
    self.reason = reason
    super().__init__(
      f"the identifier names no user with a positive risk_user_id: {identifier!r}"
    )


def risk_user_id_for(
  identifier: str,
  *,
  environ: Mapping[str, str] | None = None,
) -> int:
  """storage_id (numeric) -> int; slug/email -> roster/bundle risk_user_id; else RunOwnerUnresolved.

  Every unresolvable case is RunOwnerUnresolved. Ambiguity is raised where it
  is detected. A malformed roster is MalformedGatewayUserKeys, translated here.
  """
  if not isinstance(identifier, str):
    raise RunOwnerUnresolved(identifier, reason="unknown")
  cleaned = identifier.strip()
  storage_id = _self_mapped_storage_id(cleaned)
  if storage_id is not None:
    return storage_id
  if not cleaned:
    raise RunOwnerUnresolved(identifier, reason="unknown")
  field = "email" if "@" in cleaned else "slug"
  try:
    return mapped_risk_user_id(cleaned, field=field, environ=environ)
  except RunOwnerUnresolved:
    raise
  except MalformedGatewayUserKeys as exc:
    raise RunOwnerUnresolved(identifier, reason="malformed_roster") from exc
  except ValueError as exc:
    raise RunOwnerUnresolved(identifier, reason="unknown") from exc


def gateway_user_key_entries(
  channel: str | None = None,
  *,
  environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
  """Bundle identity, else the GATEWAY_USER_KEYS roster.

  A user runtime admits one identity, the bundle's, whatever its channel;
  the shared gateway reads the roster. A malformed roster raises MalformedGatewayUserKeys.
  """
  env = os.environ if environ is None else environ
  bundle = load_user_runtime_credentials(env)
  if bundle is not None:
    user = bundle_to_json(bundle)["user"]
    entries: list[dict[str, Any]] = [user] if isinstance(user, dict) else []
  else:
    entries = list(load_gateway_user_key_entries(env))
  normalized_channel = str(channel or "").strip()
  if not normalized_channel:
    return entries
  return [
    entry
    for entry in entries
    if str(entry.get("channel", "")).strip() == normalized_channel
  ]


def validated_identity_entries(
  *,
  environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
  """Roster/bundle rows, each already parsed to a slug and a positive risk_user_id by the roster owner."""
  validated: list[dict[str, Any]] = []
  for entry in gateway_user_key_entries(environ=environ):
    risk_user_id = positive_int(entry["risk_user_id"])
    slug = normalized_str(entry["slug"])
    email = _normalized_email(entry.get("email"))
    validated.append({
      "slug": slug,
      "email": email,
      "risk_user_id": risk_user_id,
      "channel": normalized_str(entry.get("channel")),
      "role": normalized_str(entry.get("role")),
    })
  return validated


def mapped_risk_user_id(
  identifier: str,
  *,
  field: str,
  environ: Mapping[str, str] | None = None,
) -> int:
  """Slug or email → risk_user_id. A numeric token is still a slug here.

  Unresolvable slug/email results are RunOwnerUnresolved, including an
  ambiguous mapping raised here. A malformed roster raises MalformedGatewayUserKeys.
  """
  if field not in {"slug", "email"}:
    raise RunOwnerUnresolved(identifier, reason="unknown")
  cleaned = identifier.strip()
  if field == "email":
    cleaned = cleaned.lower()
    if "@" not in cleaned:
      raise RunOwnerUnresolved(identifier, reason="unknown")
  if not cleaned:
    raise RunOwnerUnresolved(identifier, reason="unknown")
  entries = validated_identity_entries(environ=environ)
  matched = [entry for entry in entries if entry.get(field) == cleaned]
  if not matched:
    raise RunOwnerUnresolved(identifier, reason="unknown")
  owners = {int(entry["risk_user_id"]) for entry in matched}
  if len(owners) > 1:
    raise RunOwnerUnresolved(identifier, reason="ambiguous") from ValueError(
      f"ambiguous GATEWAY_USER_KEYS {field} mapping for {cleaned!r}: "
      f"{sorted(str(owner) for owner in owners)}"
    )
  return int(matched[0]["risk_user_id"])


def entry_risk_user_id(value: object) -> int:
  """Positive int() of a roster risk_user_id, else RunOwnerUnresolved.

  Matches the historical storage-path int() rules, including acceptance of
  values int() accepts. Callers that must SystemExit translate reason.
  """
  try:
    parsed = int(value)  # type: ignore[arg-type]
  except (TypeError, ValueError) as exc:
    raise RunOwnerUnresolved(value, reason="not_integer") from exc
  if parsed <= 0:
    raise RunOwnerUnresolved(value, reason="non_positive")
  return parsed


def _self_mapped_storage_id(identifier: str) -> int | None:
  """Positive decimal storage id, else None. Non-positive decimal raises."""
  if not identifier:
    return None
  negative = identifier[0] == "-"
  digits = identifier[1:] if negative else identifier
  if not digits.isdecimal():
    return None
  parsed = int(identifier)
  if parsed <= 0:
    raise RunOwnerUnresolved(identifier, reason="non_positive")
  return parsed


def _normalized_email(value: object) -> str | None:
  if not isinstance(value, str):
    return None
  cleaned = value.strip().lower()
  return cleaned if "@" in cleaned else None


__all__ = [
  "RunOwnerUnresolved",
  "StorageIdentity",
  "StorageIdentityFinding",
  "parse_storage_identity",
  "entry_risk_user_id",
  "gateway_user_key_entries",
  "mapped_risk_user_id",
  "risk_user_id_for",
  "validated_identity_entries",
]
