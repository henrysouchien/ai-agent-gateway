from __future__ import annotations

import json
import sys
from typing import Any, BinaryIO

from .capability_binding import CredentialHandle
from .capability_execution import MaterializedCredential


AUTONOMOUS_CREDENTIAL_HANDOFF_ENV = (
  "AGENT_AUTONOMOUS_CREDENTIAL_HANDOFF"
)
AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN = "stdin-json-v1"
AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES = 128 * 1024

_HANDOFF_FIELDS = frozenset({
  "version",
  "transport",
  "credential_handle",
  "auth_config",
})
_HANDLE_FIELDS = frozenset({
  "handle_id",
  "provider",
  "principal",
  "tenant_id",
  "actor_id",
})


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
  payload: dict[str, Any] = {}
  for key, value in pairs:
    if key in payload:
      raise ValueError(
        "autonomous credential handoff contains duplicate fields"
      )
    payload[key] = value
  return payload


def _require_closed_fields(
  payload: dict[str, Any],
  *,
  expected: frozenset[str],
  object_name: str,
) -> None:
  fields = set(payload)
  if fields != expected:
    raise ValueError(
      f"autonomous credential handoff {object_name} has invalid fields"
    )


def _handle_payload(handle: CredentialHandle) -> dict[str, str | None]:
  return {
    "handle_id": handle.handle_id,
    "provider": handle.provider,
    "principal": handle.principal,
    "tenant_id": handle.tenant_id,
    "actor_id": handle.actor_id,
  }


def encode_autonomous_credential_handoff(
  materialized: MaterializedCredential,
) -> bytes:
  """Encode the launch's sole credential for an anonymous subprocess pipe."""

  if not isinstance(materialized, MaterializedCredential):
    raise TypeError(
      "autonomous credential handoff requires MaterializedCredential"
    )
  payload = {
    "version": 1,
    "transport": AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN,
    "credential_handle": _handle_payload(materialized.handle),
    "auth_config": dict(materialized.auth_config),
  }
  try:
    encoded = json.dumps(
      payload,
      sort_keys=True,
      separators=(",", ":"),
      ensure_ascii=False,
    ).encode("utf-8")
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "autonomous credential material is not JSON serializable"
    ) from exc
  if not encoded or len(encoded) > AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES:
    raise ValueError(
      "autonomous credential handoff exceeds the allowed size"
    )
  return encoded


def read_autonomous_credential_handoff(
  *,
  expected_handle_id: str,
  expected_provider: str,
  expected_principal: str,
  expected_tenant_id: str,
  expected_actor_id: str | None,
  stream: BinaryIO | None = None,
) -> MaterializedCredential:
  """Read and validate the sole credential from the inherited stdin pipe."""

  source = stream if stream is not None else sys.stdin.buffer
  try:
    raw = source.read(AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES + 1)
  finally:
    if stream is None:
      source.close()
  if not isinstance(raw, bytes) or not raw:
    raise ValueError("autonomous credential handoff is missing")
  if len(raw) > AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES:
    raise ValueError(
      "autonomous credential handoff exceeds the allowed size"
    )
  try:
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_closed_object)
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise ValueError("autonomous credential handoff is invalid") from exc
  if not isinstance(payload, dict):
    raise ValueError("autonomous credential handoff must be an object")
  _require_closed_fields(
    payload,
    expected=_HANDOFF_FIELDS,
    object_name="payload",
  )
  if payload["version"] != 1:
    raise ValueError("autonomous credential handoff version is unsupported")
  if payload["transport"] != AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN:
    raise ValueError("autonomous credential handoff transport is invalid")

  raw_handle = payload["credential_handle"]
  if not isinstance(raw_handle, dict):
    raise ValueError(
      "autonomous credential handoff credential_handle must be an object"
    )
  _require_closed_fields(
    raw_handle,
    expected=_HANDLE_FIELDS,
    object_name="credential_handle",
  )
  try:
    handle = CredentialHandle(
      handle_id=raw_handle["handle_id"],
      provider=raw_handle["provider"],
      principal=raw_handle["principal"],
      tenant_id=raw_handle["tenant_id"],
      actor_id=raw_handle["actor_id"],
    )
  except (TypeError, ValueError) as exc:
    raise ValueError(
      "autonomous credential handoff handle is invalid"
    ) from exc
  if (
    handle.handle_id != expected_handle_id
    or handle.provider != expected_provider
    or handle.principal != expected_principal
    or handle.tenant_id != expected_tenant_id
    or handle.actor_id != expected_actor_id
  ):
    raise ValueError(
      "autonomous credential handoff does not match the signed launch"
    )

  auth_config = payload["auth_config"]
  if not isinstance(auth_config, dict):
    raise ValueError(
      "autonomous credential handoff auth_config must be an object"
    )
  configured_provider = auth_config.get("provider")
  if (
    not isinstance(configured_provider, str)
    or configured_provider.strip().lower() != handle.provider
  ):
    raise ValueError(
      "autonomous credential handoff provider does not match its handle"
    )
  return MaterializedCredential(
    handle=handle,
    auth_config=auth_config,
  )


__all__ = [
  "AUTONOMOUS_CREDENTIAL_HANDOFF_ENV",
  "AUTONOMOUS_CREDENTIAL_HANDOFF_MAX_BYTES",
  "AUTONOMOUS_CREDENTIAL_HANDOFF_STDIN",
  "encode_autonomous_credential_handoff",
  "read_autonomous_credential_handoff",
]
