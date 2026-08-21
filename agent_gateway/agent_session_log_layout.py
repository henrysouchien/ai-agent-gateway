"""Authenticated durable session-log layout selection and v2 storage binding.

``AGENT_SESSION_LOG_LAYOUT`` is a temporary operational choice owned by the
session-log storage/deploy owner.  ``v1`` remains supported until the canonical
flat streams have been cut over and a later v2-capable release replaces the
Foundation release as the rollback floor.
"""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

from .agent_session_log_records import slugify
from .agent_session_log_sidecars import V2_ACTIVE_SIDECAR_FIELDS
from .descriptor_paths import (
  DirectoryChainSecurityError,
  absolute_lexical_path,
  open_directory_chain,
)
from .openai_history_fence import scope_provider_session_id

if TYPE_CHECKING:
  from .agent_session_log import AgentSessionLogLocation


SESSION_LOG_LAYOUT_ENV = "AGENT_SESSION_LOG_LAYOUT"
SESSION_LOG_ROOT_FD_ENV = "AGENT_SESSION_LOG_ROOT_FD"
SESSION_LOG_ACTIVE_FD_ENV = "AGENT_SESSION_LOG_ACTIVE_FD"
SESSION_LOG_META_FD_ENV = "AGENT_SESSION_LOG_META_FD"
SESSION_LOG_V2_DIRECTORY = ".session-log-v2"
SESSION_LOG_LAYOUT_V1 = "v1"
SESSION_LOG_LAYOUT_V2 = "v2"

_LAYOUTS = frozenset({SESSION_LOG_LAYOUT_V1, SESSION_LOG_LAYOUT_V2})
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PRODUCT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_IDENTITY_FIELDS = (
  "layout",
  "tenant",
  "owner",
  "workload_profile",
  "provider",
  "provider_session_epoch",
)
_AUTHORITY_FIELDS = frozenset({
  "layout",
  "provider_session_epoch",
  "base_path",
  "root_path",
  "root_device",
  "root_inode",
  "active_path",
  "active_device",
  "active_inode",
  "meta_path",
  "meta_device",
  "meta_inode",
  "storage_identity_digest",
})


class AgentSessionLogLayoutError(RuntimeError):
  """The configured session-log layout or its exact storage is unsafe."""


def _canonical_absolute_path(value: object, *, field_name: str) -> str:
  if type(value) is not str or not value or value != value.strip():
    raise ValueError(f"{field_name} must be a canonical absolute path")
  path = Path(value)
  if not path.is_absolute() or str(path) != value or ".." in path.parts:
    raise ValueError(f"{field_name} must be a canonical absolute path")
  return value


def _required_identity_text(value: object, *, field_name: str) -> str:
  if (
    type(value) is not str
    or not value
    or value != value.strip()
    or len(value.encode("utf-8")) > 512
    or any(ord(character) < 0x20 for character in value)
  ):
    raise ValueError(f"{field_name} must be canonical bounded text")
  return value


def _optional_identity_text(value: object, *, field_name: str) -> str | None:
  if value is None:
    return None
  return _required_identity_text(value, field_name=field_name)


def _file_identity_value(value: object, *, field_name: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < 0:
    raise ValueError(f"{field_name} must be a nonnegative integer")
  if field_name.endswith("inode") and value == 0:
    raise ValueError(f"{field_name} must be positive")
  return value


def resolve_agent_session_log_layout(
  environ: Mapping[str, str] | None = None,
) -> Literal["v1", "v2"]:
  """Return the explicit supported storage choice, defaulting to Foundation v1."""

  env = os.environ if environ is None else environ
  value = str(env.get(SESSION_LOG_LAYOUT_ENV, SESSION_LOG_LAYOUT_V1) or "").strip()
  if value not in _LAYOUTS:
    raise AgentSessionLogLayoutError(
      f"{SESSION_LOG_LAYOUT_ENV} must be v1 or v2"
    )
  return value  # type: ignore[return-value]


def resolve_agent_session_log_archive_product_ids(
  environ: Mapping[str, str] | None = None,
) -> frozenset[str]:
  """Return the explicit temporary v1/archival tenant compatibility set."""

  env = os.environ if environ is None else environ
  raw = str(env.get("AGENT_SESSION_LOG_ARCHIVE_PRODUCT_IDS", "") or "")
  values = tuple(part.strip() for part in raw.split(",") if part.strip())
  if (
    len(values) != len(set(values))
    or any(not _PRODUCT_ID_RE.fullmatch(value) for value in values)
  ):
    raise AgentSessionLogLayoutError(
      "AGENT_SESSION_LOG_ARCHIVE_PRODUCT_IDS is invalid"
    )
  return frozenset(values)


def canonical_session_log_storage_identity(
  *,
  tenant: str,
  owner: str,
  workload_profile: str,
  provider: str,
  provider_session_epoch: str | None,
) -> tuple[dict[str, Any], str]:
  """Return the canonical v2 identity object and its full SHA-256 digest."""

  normalized_provider = _required_identity_text(
    str(provider or "").strip().lower(),
    field_name="provider",
  )
  normalized_epoch = _optional_identity_text(
    provider_session_epoch,
    field_name="provider_session_epoch",
  )
  # Reuse the provider history owner to validate the epoch contract.
  scope_provider_session_id(
    "session-log-identity",
    provider=normalized_provider,
    durable=True,
    openai_epoch=normalized_epoch,
  )
  if normalized_provider != "openai" and normalized_epoch is not None:
    raise ValueError(
      "provider_session_epoch must be null for non-OpenAI storage"
    )
  identity = {
    "layout": 2,
    "tenant": _required_identity_text(tenant, field_name="tenant"),
    "owner": _required_identity_text(owner, field_name="owner"),
    "workload_profile": _required_identity_text(
      str(workload_profile or "").strip().lower(),
      field_name="workload_profile",
    ),
    "provider": normalized_provider,
    "provider_session_epoch": normalized_epoch,
  }
  if tuple(identity) != _IDENTITY_FIELDS:
    raise AssertionError("session-log storage identity field order changed")
  encoded = json.dumps(
    identity,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
  ).encode("utf-8")
  return identity, hashlib.sha256(encoded).hexdigest()


def derive_v2_agent_session_log_paths(
  base_dir: str | Path,
  *,
  tenant: str,
  owner: str,
  workload_profile: str,
  provider: str,
  provider_session_epoch: str | None,
) -> tuple[Path, Path, Path, str]:
  """Derive the one-directory v2 namespace without touching storage."""

  base = absolute_lexical_path(base_dir)
  _identity, digest = canonical_session_log_storage_identity(
    tenant=tenant,
    owner=owner,
    workload_profile=workload_profile,
    provider=provider,
    provider_session_epoch=provider_session_epoch,
  )
  stream_slug = f"s-{digest[:52]}"
  root = base / SESSION_LOG_V2_DIRECTORY / stream_slug
  active = root / f"agentsess_{stream_slug}_{slugify(owner)}.jsonl"
  return root, active, active.with_suffix(".meta.json"), digest


@dataclass(frozen=True, slots=True)
class AutonomousSessionLogAuthority:
  """Signed storage coordinates for one autonomous canonical stream."""

  layout: Literal["v1", "v2"]
  provider_session_epoch: str | None
  base_path: str
  root_path: str | None
  root_device: int | None
  root_inode: int | None
  active_path: str | None
  active_device: int | None
  active_inode: int | None
  meta_path: str | None
  meta_device: int | None
  meta_inode: int | None
  storage_identity_digest: str | None

  def __post_init__(self) -> None:
    if self.layout not in _LAYOUTS:
      raise ValueError("session-log authority layout is invalid")
    object.__setattr__(
      self,
      "base_path",
      _canonical_absolute_path(self.base_path, field_name="base_path"),
    )
    epoch = _optional_identity_text(
      self.provider_session_epoch,
      field_name="provider_session_epoch",
    )
    object.__setattr__(self, "provider_session_epoch", epoch)
    storage_values = (
      self.root_path,
      self.root_device,
      self.root_inode,
      self.active_path,
      self.active_device,
      self.active_inode,
      self.meta_path,
      self.meta_device,
      self.meta_inode,
      self.storage_identity_digest,
    )
    if self.layout == SESSION_LOG_LAYOUT_V1:
      if any(value is not None for value in storage_values):
        raise ValueError("v1 session-log authority cannot carry v2 storage")
      return
    if any(value is None for value in storage_values):
      raise ValueError("v2 session-log authority requires exact storage")
    root_path = _canonical_absolute_path(self.root_path, field_name="root_path")
    active_path = _canonical_absolute_path(self.active_path, field_name="active_path")
    meta_path = _canonical_absolute_path(self.meta_path, field_name="meta_path")
    if Path(active_path).parent != Path(root_path) or Path(meta_path).parent != Path(root_path):
      raise ValueError("v2 active and metadata must be direct root children")
    if Path(meta_path) != Path(active_path).with_suffix(".meta.json"):
      raise ValueError("v2 metadata path does not match active path")
    object.__setattr__(self, "root_path", root_path)
    object.__setattr__(self, "active_path", active_path)
    object.__setattr__(self, "meta_path", meta_path)
    for name in (
      "root_device", "root_inode", "active_device", "active_inode",
      "meta_device", "meta_inode",
    ):
      object.__setattr__(
        self,
        name,
        _file_identity_value(getattr(self, name), field_name=name),
      )
    digest = self.storage_identity_digest
    if type(digest) is not str or _DIGEST_RE.fullmatch(digest) is None:
      raise ValueError("v2 storage_identity_digest is invalid")

  @classmethod
  def from_receipt(cls, value: object) -> "AutonomousSessionLogAuthority":
    if not isinstance(value, Mapping) or set(value) != _AUTHORITY_FIELDS:
      raise ValueError("session-log authority violates its closed contract")
    return cls(**dict(value))

  def receipt(self) -> dict[str, Any]:
    return {
      "layout": self.layout,
      "provider_session_epoch": self.provider_session_epoch,
      "base_path": self.base_path,
      "root_path": self.root_path,
      "root_device": self.root_device,
      "root_inode": self.root_inode,
      "active_path": self.active_path,
      "active_device": self.active_device,
      "active_inode": self.active_inode,
      "meta_path": self.meta_path,
      "meta_device": self.meta_device,
      "meta_inode": self.meta_inode,
      "storage_identity_digest": self.storage_identity_digest,
    }


@dataclass(slots=True)
class PreparedAutonomousSessionLog:
  authority: AutonomousSessionLogAuthority
  root_fd: int = -1
  active_fd: int = -1
  meta_fd: int = -1

  @property
  def pass_fds(self) -> tuple[int, ...]:
    if self.authority.layout == SESSION_LOG_LAYOUT_V1:
      return ()
    if min(self.root_fd, self.active_fd, self.meta_fd) < 0:
      raise AgentSessionLogLayoutError("v2 session-log descriptors are closed")
    return self.root_fd, self.active_fd, self.meta_fd

  def close(self) -> None:
    for name in ("meta_fd", "active_fd", "root_fd"):
      descriptor = getattr(self, name)
      setattr(self, name, -1)
      if descriptor >= 0:
        os.close(descriptor)


def _require_safe_directory(
  info: os.stat_result,
  *,
  target: str,
  exact_private_mode: bool = True,
) -> None:
  if (
    not stat.S_ISDIR(info.st_mode)
    or info.st_uid != os.geteuid()
    or (
      stat.S_IMODE(info.st_mode) != 0o700
      if exact_private_mode
      else stat.S_IMODE(info.st_mode) & 0o022
    )
  ):
    raise AgentSessionLogLayoutError(f"{target} is unsafe")


def _require_safe_regular(info: os.stat_result, *, target: str) -> None:
  if (
    not stat.S_ISREG(info.st_mode)
    or info.st_uid != os.geteuid()
    or info.st_nlink != 1
    or stat.S_IMODE(info.st_mode) != 0o600
  ):
    raise AgentSessionLogLayoutError(f"{target} is unsafe")


def _open_regular_at(
  parent_fd: int,
  name: str,
  *,
  create: bool,
) -> tuple[int, bool]:
  flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
  created = False
  if create:
    try:
      descriptor = os.open(
        name,
        flags | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=parent_fd,
      )
      created = True
    except FileExistsError:
      descriptor = os.open(name, flags, dir_fd=parent_fd)
  else:
    descriptor = os.open(name, flags, dir_fd=parent_fd)
  try:
    opened = os.fstat(descriptor)
    named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    _require_safe_regular(opened, target=name)
    _require_safe_regular(named, target=name)
    if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
      raise AgentSessionLogLayoutError(f"{name} identity changed")
  except Exception:
    os.close(descriptor)
    raise
  return descriptor, created


def _open_managed_v2_root(
  base: Path,
  root: Path,
  *,
  create: bool,
) -> tuple[int, tuple[tuple[int, int], ...]]:
  """Open B→.session-log-v2→R with explicit owner/mode checks."""

  if root.parent != base / SESSION_LOG_V2_DIRECTORY:
    raise AgentSessionLogLayoutError("v2 session-log root grammar is invalid")
  base_fd = version_fd = root_fd = -1
  try:
    base_fd, base_identity = open_directory_chain(base)
    _require_safe_directory(
      os.fstat(base_fd),
      target="session-log base",
      exact_private_mode=False,
    )

    def open_child(parent_fd: int, name: str, target: str) -> int:
      if create:
        try:
          os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError:
          pass
      descriptor = os.open(
        name,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
      )
      try:
        opened = os.fstat(descriptor)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        _require_safe_directory(opened, target=target)
        _require_safe_directory(named, target=target)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
          raise AgentSessionLogLayoutError(f"{target} identity changed")
      except Exception:
        os.close(descriptor)
        raise
      return descriptor

    version_fd = open_child(
      base_fd,
      SESSION_LOG_V2_DIRECTORY,
      "v2 session-log namespace",
    )
    root_fd = open_child(version_fd, root.name, "v2 session-log root")
    root_info = os.fstat(root_fd)
    result = root_fd
    root_fd = -1
    return result, (
      *base_identity,
      (os.fstat(version_fd).st_dev, os.fstat(version_fd).st_ino),
      (root_info.st_dev, root_info.st_ino),
    )
  except FileNotFoundError as exc:
    raise AgentSessionLogLayoutError(
      "managed v2 session-log directory is missing"
    ) from exc
  finally:
    for descriptor in (root_fd, version_fd, base_fd):
      if descriptor >= 0:
        os.close(descriptor)


def _canonical_sidecar_bytes(payload: Mapping[str, Any]) -> bytes:
  return (json.dumps(dict(payload), sort_keys=True, indent=2) + "\n").encode("utf-8")


def _replace_descriptor_bytes(descriptor: int, payload: bytes) -> None:
  """Publish deterministic initialization bytes to an already pinned file."""

  os.ftruncate(descriptor, 0)
  os.lseek(descriptor, 0, os.SEEK_SET)
  offset = 0
  while offset < len(payload):
    written = os.write(descriptor, payload[offset:])
    if written <= 0:
      raise OSError("session-log metadata write made no progress")
    offset += written
  os.fsync(descriptor)


def _read_descriptor_json(descriptor: int, *, max_bytes: int = 64 * 1024) -> dict[str, Any]:
  os.lseek(descriptor, 0, os.SEEK_SET)
  chunks: list[bytes] = []
  total = 0
  while True:
    chunk = os.read(descriptor, min(8192, max_bytes + 1 - total))
    if not chunk:
      break
    chunks.append(chunk)
    total += len(chunk)
    if total > max_bytes:
      raise AgentSessionLogLayoutError("session-log sidecar exceeds its byte bound")
  try:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
      result: dict[str, Any] = {}
      for key, value in pairs:
        if key in result:
          raise AgentSessionLogLayoutError(
            "session-log sidecar contains duplicate fields"
          )
        result[key] = value
      return result

    payload = json.loads(
      b"".join(chunks).decode("utf-8"),
      object_pairs_hook=strict_object,
    )
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise AgentSessionLogLayoutError("session-log sidecar is invalid") from exc
  if not isinstance(payload, dict):
    raise AgentSessionLogLayoutError("session-log sidecar is not an object")
  return payload


def _v2_sidecar_payload(
  *,
  active: Path,
  digest: str,
  tenant: str,
  owner: str,
  workload_profile: str,
  provider: str,
  provider_session_epoch: str | None,
  now_iso: str,
) -> dict[str, Any]:
  stream_hash = hashlib.sha1(str(active).encode("utf-8")).hexdigest()[:16]
  logical_agent_id = scope_provider_session_id(
    workload_profile,
    provider=provider,
    durable=True,
    openai_epoch=provider_session_epoch,
  )
  return {
    "schema_version": 2,
    "agent_session_id": active.stem,
    # Consumer attribution remains the authenticated profile identity; the
    # directory slug is an opaque storage coordinate only.
    "agent_id": logical_agent_id,
    "user_id": owner,
    "product_id": tenant,
    "tenant_id": tenant,
    "file_kind": "canonical",
    "channel": None,
    "profile": workload_profile,
    "created_at": now_iso,
    "file_role": "active",
    "logical_stream_id": str(active),
    "telemetry_source_id": f"agent_session_log:{stream_hash}:active:000000",
    "active_generation": 0,
    "storage_layout": 2,
    "workload_profile": workload_profile,
    "provider": provider,
    "provider_session_epoch": provider_session_epoch,
    "storage_identity_digest": digest,
  }


def validate_v2_sidecar_payload(
  payload: Mapping[str, Any],
  *,
  active: Path,
  digest: str,
  tenant: str,
  owner: str,
  workload_profile: str,
  provider: str,
  provider_session_epoch: str | None,
) -> dict[str, Any]:
  """Require every v2 classification field to match its trusted identity."""

  if not isinstance(payload, Mapping):
    raise AgentSessionLogLayoutError("v2 session-log sidecar is not an object")
  if set(payload) != V2_ACTIVE_SIDECAR_FIELDS:
    raise AgentSessionLogLayoutError(
      "v2 session-log sidecar violates its closed field contract"
    )
  exact = {
    "schema_version": 2,
    "agent_session_id": active.stem,
    "agent_id": scope_provider_session_id(
      workload_profile,
      provider=str(provider).strip().lower(),
      durable=True,
      openai_epoch=provider_session_epoch,
    ),
    "user_id": owner,
    "product_id": tenant,
    "tenant_id": tenant,
    "file_kind": "canonical",
    "profile": workload_profile,
    "file_role": "active",
    "logical_stream_id": str(active),
    "storage_layout": 2,
    "workload_profile": workload_profile,
    "provider": str(provider).strip().lower(),
    "provider_session_epoch": provider_session_epoch,
    "storage_identity_digest": digest,
  }
  for field_name, expected in exact.items():
    if payload.get(field_name) != expected:
      raise AgentSessionLogLayoutError(
        f"v2 session-log sidecar {field_name} does not match storage authority"
      )
  if payload.get("run_kind") is not None or payload.get("gateway_session_id") is not None:
    raise AgentSessionLogLayoutError(
      "v2 canonical sidecar contains another stream-kind identity"
    )
  generation = payload.get("active_generation")
  if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
    raise AgentSessionLogLayoutError(
      "v2 session-log sidecar active_generation is invalid"
    )
  stream_hash = hashlib.sha1(str(active).encode("utf-8")).hexdigest()[:16]
  expected_source_id = (
    f"agent_session_log:{stream_hash}:active:{generation:06d}"
  )
  if payload.get("telemetry_source_id") != expected_source_id:
    raise AgentSessionLogLayoutError(
      "v2 session-log sidecar telemetry_source_id does not match storage"
    )
  created_at = payload.get("created_at")
  if type(created_at) is not str or not created_at:
    raise AgentSessionLogLayoutError(
      "v2 session-log sidecar created_at is invalid"
    )
  return dict(payload)


def prepare_autonomous_session_log(
  *,
  base_dir: str | Path,
  layout: Literal["v1", "v2"],
  tenant: str,
  owner: str,
  workload_profile: str,
  provider: str,
  provider_session_epoch: str | None,
  now_iso: Callable[[], str],
) -> PreparedAutonomousSessionLog:
  """Create and pin the exact v2 namespace before an autonomous child starts."""

  base = absolute_lexical_path(base_dir)
  if layout == SESSION_LOG_LAYOUT_V1:
    return PreparedAutonomousSessionLog(
      AutonomousSessionLogAuthority(
        layout=SESSION_LOG_LAYOUT_V1,
        provider_session_epoch=provider_session_epoch,
        base_path=str(base),
        root_path=None,
        root_device=None,
        root_inode=None,
        active_path=None,
        active_device=None,
        active_inode=None,
        meta_path=None,
        meta_device=None,
        meta_inode=None,
        storage_identity_digest=None,
      )
    )
  if layout != SESSION_LOG_LAYOUT_V2:
    raise AgentSessionLogLayoutError("session-log layout is unsupported")
  root, active, meta, digest = derive_v2_agent_session_log_paths(
    base,
    tenant=tenant,
    owner=owner,
    workload_profile=workload_profile,
    provider=provider,
    provider_session_epoch=provider_session_epoch,
  )
  root_fd = active_fd = meta_fd = init_mutex_fd = -1
  try:
    root_fd, _root_chain = _open_managed_v2_root(base, root, create=True)
    root_info = os.fstat(root_fd)
    _require_safe_directory(root_info, target="v2 session-log root")
    active_fd, _active_created = _open_regular_at(root_fd, active.name, create=True)
    active_info = os.fstat(active_fd)
    init_mutex_fd, _mutex_created = _open_regular_at(
      root_fd,
      f"{active.name}.append_mutex",
      create=True,
    )
    fcntl.flock(init_mutex_fd, fcntl.LOCK_EX)
    meta_payload: dict[str, Any] | None = None
    try:
      meta_fd, _meta_created = _open_regular_at(root_fd, meta.name, create=False)
    except FileNotFoundError:
      if active_info.st_size != 0:
        raise AgentSessionLogLayoutError(
          "nonempty v2 session log is missing mandatory metadata"
        ) from None
      meta_fd, created = _open_regular_at(root_fd, meta.name, create=True)
      if not created:
        raise AgentSessionLogLayoutError(
          "v2 session-log metadata appeared during initialization"
        )
    else:
      try:
        meta_payload = _read_descriptor_json(meta_fd)
      except AgentSessionLogLayoutError as exc:
        # The only recoverable published state is syntactically incomplete
        # metadata next to an empty active file. The append mutex serializes
        # this rewrite with both concurrent initializers and future writers.
        if (
          str(exc) != "session-log sidecar is invalid"
          or active_info.st_size != 0
        ):
          raise
    if meta_payload is None:
      payload = _v2_sidecar_payload(
        active=active,
        digest=digest,
        tenant=tenant,
        owner=owner,
        workload_profile=workload_profile,
        provider=str(provider).strip().lower(),
        provider_session_epoch=provider_session_epoch,
        now_iso=now_iso(),
      )
      _replace_descriptor_bytes(
        meta_fd,
        _canonical_sidecar_bytes(payload),
      )
      os.fsync(root_fd)
      meta_payload = payload
    meta_info = os.fstat(meta_fd)
    _require_safe_regular(meta_info, target="v2 session-log metadata")
    validate_v2_sidecar_payload(
      meta_payload,
      active=active,
      digest=digest,
      tenant=tenant,
      owner=owner,
      workload_profile=workload_profile,
      provider=provider,
      provider_session_epoch=provider_session_epoch,
    )
    authority = AutonomousSessionLogAuthority(
      layout=SESSION_LOG_LAYOUT_V2,
      provider_session_epoch=provider_session_epoch,
      base_path=str(base),
      root_path=str(root),
      root_device=int(root_info.st_dev),
      root_inode=int(root_info.st_ino),
      active_path=str(active),
      active_device=int(active_info.st_dev),
      active_inode=int(active_info.st_ino),
      meta_path=str(meta),
      meta_device=int(meta_info.st_dev),
      meta_inode=int(meta_info.st_ino),
      storage_identity_digest=digest,
    )
    result = PreparedAutonomousSessionLog(
      authority=authority,
      root_fd=root_fd,
      active_fd=active_fd,
      meta_fd=meta_fd,
    )
    root_fd = active_fd = meta_fd = -1
    return result
  except (DirectoryChainSecurityError, OSError) as exc:
    if isinstance(exc, AgentSessionLogLayoutError):
      raise
    raise AgentSessionLogLayoutError(
      "v2 session-log storage cannot be initialized"
    ) from exc
  finally:
    for descriptor in (init_mutex_fd, meta_fd, active_fd, root_fd):
      if descriptor >= 0:
        os.close(descriptor)


@dataclass(slots=True)
class VerifiedAutonomousSessionLog:
  authority: AutonomousSessionLogAuthority
  location: AgentSessionLogLocation | None
  root_fd: int = -1
  active_fd: int = -1
  meta_fd: int = -1

  def close(self) -> None:
    for name in ("meta_fd", "active_fd", "root_fd"):
      descriptor = getattr(self, name)
      setattr(self, name, -1)
      if descriptor >= 0:
        os.close(descriptor)


def _verify_descriptor(
  descriptor: int,
  *,
  expected_device: int,
  expected_inode: int,
  directory: bool,
  target: str,
) -> os.stat_result:
  try:
    info = os.fstat(descriptor)
  except OSError as exc:
    raise AgentSessionLogLayoutError(f"{target} descriptor is unavailable") from exc
  if (info.st_dev, info.st_ino) != (expected_device, expected_inode):
    raise AgentSessionLogLayoutError(f"{target} descriptor identity changed")
  if directory:
    _require_safe_directory(info, target=target)
  else:
    _require_safe_regular(info, target=target)
  return info


def verify_autonomous_session_log(
  authority: AutonomousSessionLogAuthority,
  *,
  root_fd: int | None,
  active_fd: int | None,
  meta_fd: int | None,
  tenant: str,
  owner: str,
  workload_profile: str,
  provider: str,
  projected_base_path: str | None,
) -> VerifiedAutonomousSessionLog:
  """Bind inherited descriptors to the signed storage identity in the child."""

  if type(authority) is not AutonomousSessionLogAuthority:
    raise TypeError("session-log verification requires exact authority")
  if str(projected_base_path or "") != authority.base_path:
    raise AgentSessionLogLayoutError(
      "projected session-log base does not match signed authority"
    )
  if authority.layout == SESSION_LOG_LAYOUT_V1:
    if any(fd is not None for fd in (root_fd, active_fd, meta_fd)):
      raise AgentSessionLogLayoutError("v1 child inherited v2 descriptors")
    return VerifiedAutonomousSessionLog(authority=authority, location=None)
  if any(type(fd) is not int or fd < 0 for fd in (root_fd, active_fd, meta_fd)):
    raise AgentSessionLogLayoutError("v2 child requires all storage descriptors")
  assert root_fd is not None and active_fd is not None and meta_fd is not None
  root, active, meta, digest = derive_v2_agent_session_log_paths(
    authority.base_path,
    tenant=tenant,
    owner=owner,
    workload_profile=workload_profile,
    provider=provider,
    provider_session_epoch=authority.provider_session_epoch,
  )
  if (
    str(root) != authority.root_path
    or str(active) != authority.active_path
    or str(meta) != authority.meta_path
    or digest != authority.storage_identity_digest
  ):
    raise AgentSessionLogLayoutError(
      "signed session-log paths do not match authenticated identity"
    )
  _verify_descriptor(
    root_fd,
    expected_device=authority.root_device or -1,
    expected_inode=authority.root_inode or -1,
    directory=True,
    target="v2 session-log root",
  )
  _verify_descriptor(
    active_fd,
    expected_device=authority.active_device or -1,
    expected_inode=authority.active_inode or -1,
    directory=False,
    target="v2 session-log active",
  )
  _verify_descriptor(
    meta_fd,
    expected_device=authority.meta_device or -1,
    expected_inode=authority.meta_inode or -1,
    directory=False,
    target="v2 session-log metadata",
  )
  root_chain_fd = reopened_active = reopened_meta = -1
  try:
    root_chain_fd, parent_identity = _open_managed_v2_root(
      Path(authority.base_path),
      root,
      create=False,
    )
    if os.fstat(root_chain_fd).st_ino != authority.root_inode or os.fstat(root_chain_fd).st_dev != authority.root_device:
      raise AgentSessionLogLayoutError("v2 session-log root path was displaced")
    reopened_active = os.open(active.name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=root_chain_fd)
    reopened_meta = os.open(meta.name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=root_chain_fd)
    for descriptor, expected, target in (
      (reopened_active, (authority.active_device, authority.active_inode), "active"),
      (reopened_meta, (authority.meta_device, authority.meta_inode), "metadata"),
    ):
      info = os.fstat(descriptor)
      if (info.st_dev, info.st_ino) != expected:
        raise AgentSessionLogLayoutError(
          f"v2 session-log {target} path was displaced"
        )
    validate_v2_sidecar_payload(
      _read_descriptor_json(meta_fd),
      active=active,
      digest=digest,
      tenant=tenant,
      owner=owner,
      workload_profile=workload_profile,
      provider=provider,
      provider_session_epoch=authority.provider_session_epoch,
    )
    segments = active.with_name(f"{active.stem}.segments")
    try:
      segments_info = os.stat(
        segments.name,
        dir_fd=root_chain_fd,
        follow_symlinks=False,
      )
    except FileNotFoundError:
      segments_identity = None
    else:
      if stat.S_ISLNK(segments_info.st_mode) or not stat.S_ISDIR(segments_info.st_mode):
        raise AgentSessionLogLayoutError("v2 session-log segments are unsafe")
      segments_identity = (segments_info.st_dev, segments_info.st_ino)
    from .agent_session_log import AgentSessionLogLocation

    location = AgentSessionLogLocation(
      path=active,
      parent_identity=parent_identity,
      active_identity=(authority.active_device, authority.active_inode),
      segments_identity=segments_identity,
    )
    return VerifiedAutonomousSessionLog(
      authority=authority,
      location=location,
      root_fd=root_fd,
      active_fd=active_fd,
      meta_fd=meta_fd,
    )
  except (DirectoryChainSecurityError, OSError) as exc:
    if isinstance(exc, AgentSessionLogLayoutError):
      raise
    raise AgentSessionLogLayoutError(
      "v2 session-log child verification failed"
    ) from exc
  finally:
    for descriptor in (reopened_meta, reopened_active, root_chain_fd):
      if descriptor >= 0:
        os.close(descriptor)


__all__ = [
  "AgentSessionLogLayoutError",
  "AutonomousSessionLogAuthority",
  "PreparedAutonomousSessionLog",
  "SESSION_LOG_ACTIVE_FD_ENV",
  "SESSION_LOG_LAYOUT_ENV",
  "SESSION_LOG_LAYOUT_V1",
  "SESSION_LOG_LAYOUT_V2",
  "SESSION_LOG_META_FD_ENV",
  "SESSION_LOG_ROOT_FD_ENV",
  "SESSION_LOG_V2_DIRECTORY",
  "VerifiedAutonomousSessionLog",
  "canonical_session_log_storage_identity",
  "derive_v2_agent_session_log_paths",
  "prepare_autonomous_session_log",
  "resolve_agent_session_log_layout",
  "resolve_agent_session_log_archive_product_ids",
  "verify_autonomous_session_log",
  "validate_v2_sidecar_payload",
]
