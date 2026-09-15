"""Durable one-time launch-nonce consume for verified ordinary launches.

The parent prepares one dedicated SQLite database and signs the returned
path/device/inode identity into every launch envelope.  The child reconstructs
that identity from the verified envelope and must consume the envelope nonce
here before it performs any autonomous side effect.

The persisted SQLite schema (table, index, and trigger names plus their
RAISE message text) predates this module's rename and is frozen: it is
exact-compared against live databases at ``_validate_schema``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import sqlite3
import stat
import time
from typing import Any, Callable

from .autonomous_launch_envelope import (
  AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE,
  AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS,
  AutonomousLaunchEnvelope,
)

from .named_refusal import NamedRefusal, NamedRefusalTransport


LAUNCH_NONCE_STORE_SCHEMA_VERSION = 2
LAUNCH_NONCE_STORE_BUSY_TIMEOUT_MS = 5_000
LAUNCH_NONCE_STORE_PAGE_SIZE = 4_096
LAUNCH_NONCE_STORE_MAX_PAGE_COUNT = 65_536
LAUNCH_NONCE_STORE_MAX_ROWS = 100_000
LAUNCH_NONCE_STORE_CLEANUP_BATCH_SIZE = 512

_MAX_SQLITE_INTEGER = (1 << 63) - 1
_MAX_PATH_BYTES = 4_096
_MAX_ID_TEXT_LENGTH = 512
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_CHANNEL_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_FACTS_FIELDS = frozenset({
  "audience",
  "nonce",
  "task_id",
  "control_run_id",
  "owner_user_id",
  "channel_id",
  "issued_at_ns",
  "expires_at_ns",
})
_CONSUMED_NONCE_WIRE_FIELDS = frozenset({
  "receipt",
  "admitted_at_ns",
})

_METADATA_TABLE_SQL = f"""
  CREATE TABLE IF NOT EXISTS autonomous_admission_ledger_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL CHECK (
      schema_version = {LAUNCH_NONCE_STORE_SCHEMA_VERSION}
    ),
    page_size INTEGER NOT NULL CHECK (page_size > 0),
    max_page_count INTEGER NOT NULL CHECK (max_page_count > 0),
    max_rows INTEGER NOT NULL CHECK (max_rows > 0),
    cleanup_batch_size INTEGER NOT NULL CHECK (cleanup_batch_size > 0)
  )
"""
_CLOCK_TABLE_SQL = """
  CREATE TABLE IF NOT EXISTS autonomous_admission_ledger_clock (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    last_admitted_wall_ns INTEGER NOT NULL CHECK (
      typeof(last_admitted_wall_ns) = 'integer'
      AND last_admitted_wall_ns >= 0
    )
  )
"""
_ADMISSIONS_TABLE_SQL = f"""
  CREATE TABLE IF NOT EXISTS ordinary_autonomous_launch_admissions (
    nonce TEXT PRIMARY KEY CHECK (
      typeof(nonce) = 'text'
      AND length(nonce) = 32
      AND nonce NOT GLOB '*[^0-9a-f]*'
    ),
    schema_version INTEGER NOT NULL CHECK (
      typeof(schema_version) = 'integer'
      AND schema_version = {LAUNCH_NONCE_STORE_SCHEMA_VERSION}
    ),
    audience TEXT NOT NULL CHECK (
      typeof(audience) = 'text'
      AND length(audience) BETWEEN 1 AND {_MAX_ID_TEXT_LENGTH}
      AND audience = trim(audience)
    ),
    task_id TEXT NOT NULL CHECK (
      typeof(task_id) = 'text'
      AND length(task_id) BETWEEN 1 AND {_MAX_ID_TEXT_LENGTH}
      AND task_id = trim(task_id)
    ),
    control_run_id TEXT NOT NULL CHECK (
      typeof(control_run_id) = 'text'
      AND length(control_run_id) BETWEEN 1 AND {_MAX_ID_TEXT_LENGTH}
      AND control_run_id = trim(control_run_id)
    ),
    owner_user_id TEXT NOT NULL CHECK (
      typeof(owner_user_id) = 'text'
      AND length(owner_user_id) BETWEEN 1 AND {_MAX_ID_TEXT_LENGTH}
      AND owner_user_id = trim(owner_user_id)
    ),
    channel_id TEXT NOT NULL CHECK (
      typeof(channel_id) = 'text'
      AND length(channel_id) = 64
      AND channel_id NOT GLOB '*[^0-9a-f]*'
    ),
    issued_at_ns INTEGER NOT NULL CHECK (
      typeof(issued_at_ns) = 'integer'
      AND issued_at_ns > 0
    ),
    expires_at_ns INTEGER NOT NULL CHECK (
      typeof(expires_at_ns) = 'integer'
      AND expires_at_ns > issued_at_ns
    ),
    admitted_at_ns INTEGER NOT NULL CHECK (
      typeof(admitted_at_ns) = 'integer'
      AND admitted_at_ns >= issued_at_ns
      AND admitted_at_ns < expires_at_ns
    )
  ) WITHOUT ROWID
"""
_EXPIRY_INDEX_SQL = """
  CREATE INDEX IF NOT EXISTS idx_autonomous_admissions_expiry
  ON ordinary_autonomous_launch_admissions (expires_at_ns, nonce)
"""
_NO_ADMISSION_UPDATE_TRIGGER_SQL = """
  CREATE TRIGGER IF NOT EXISTS trg_autonomous_admission_no_update
  BEFORE UPDATE ON ordinary_autonomous_launch_admissions
  BEGIN
    SELECT RAISE(ABORT, 'autonomous admission rows are immutable');
  END
"""
_NO_METADATA_UPDATE_TRIGGER_SQL = """
  CREATE TRIGGER IF NOT EXISTS trg_autonomous_admission_metadata_no_update
  BEFORE UPDATE ON autonomous_admission_ledger_metadata
  BEGIN
    SELECT RAISE(ABORT, 'autonomous admission metadata is immutable');
  END
"""
_NO_METADATA_DELETE_TRIGGER_SQL = """
  CREATE TRIGGER IF NOT EXISTS trg_autonomous_admission_metadata_no_delete
  BEFORE DELETE ON autonomous_admission_ledger_metadata
  BEGIN
    SELECT RAISE(ABORT, 'autonomous admission metadata is required');
  END
"""
_NO_CLOCK_DELETE_TRIGGER_SQL = """
  CREATE TRIGGER IF NOT EXISTS trg_autonomous_admission_clock_no_delete
  BEFORE DELETE ON autonomous_admission_ledger_clock
  BEGIN
    SELECT RAISE(ABORT, 'autonomous admission clock is required');
  END
"""
_SCHEMA_OBJECTS = {
  "autonomous_admission_ledger_metadata": (
    "table",
    _METADATA_TABLE_SQL,
  ),
  "ordinary_autonomous_launch_admissions": (
    "table",
    _ADMISSIONS_TABLE_SQL,
  ),
  "autonomous_admission_ledger_clock": (
    "table",
    _CLOCK_TABLE_SQL,
  ),
  "idx_autonomous_admissions_expiry": (
    "index",
    _EXPIRY_INDEX_SQL,
  ),
  "trg_autonomous_admission_no_update": (
    "trigger",
    _NO_ADMISSION_UPDATE_TRIGGER_SQL,
  ),
  "trg_autonomous_admission_metadata_no_update": (
    "trigger",
    _NO_METADATA_UPDATE_TRIGGER_SQL,
  ),
  "trg_autonomous_admission_metadata_no_delete": (
    "trigger",
    _NO_METADATA_DELETE_TRIGGER_SQL,
  ),
  "trg_autonomous_admission_clock_no_delete": (
    "trigger",
    _NO_CLOCK_DELETE_TRIGGER_SQL,
  ),
}


class LaunchNonceStoreError(NamedRefusal):
  """An autonomous launch could not be safely admitted."""

  def __init__(
    self,
    message: str,
    *,
    code: str = "launch_nonce_store",
    transport: NamedRefusalTransport = "internal",
  ) -> None:
    super().__init__(code, message, transport=transport)


class LaunchNonceStoreIdentityError(
  LaunchNonceStoreError
):
  """The signed store file identity is no longer authoritative."""

  def __init__(self, message: str) -> None:
    super().__init__(message, code="launch_nonce_store_identity", transport="unavailable")


class LaunchNonceStoreUnavailable(
  LaunchNonceStoreError
):
  """The durable store cannot currently earn a consume decision."""

  def __init__(self, message: str) -> None:
    super().__init__(message, code="launch_nonce_store_unavailable", transport="unavailable")


class LaunchNonceReplay(
  LaunchNonceStoreError
):
  """The launch nonce has already been consumed."""

  def __init__(self, message: str) -> None:
    super().__init__(message, code="launch_nonce_replay", transport="conflict")


class LaunchNonceExpired(
  LaunchNonceStoreError
):
  """The launch envelope is not live at the durable admission boundary."""

  def __init__(self, message: str, *, consumed: bool) -> None:
    self.consumed = consumed
    super().__init__(message, code="launch_nonce_expired", transport="conflict")


class LaunchNonceStoreCapacityExceeded(
  LaunchNonceStoreError
):
  """The fixed store capacity was reached; the consume failed closed."""


class LaunchNonceClockRollback(
  LaunchNonceStoreError
):
  """The wall clock moved behind a durably observed admission time."""



def _closed_mapping(
  value: object,
  *,
  field_name: str,
  expected_fields: frozenset[str],
) -> dict[str, Any]:
  if type(value) is not dict:
    raise ValueError(f"{field_name} must be an object")
  if len(value) != len(expected_fields):
    raise ValueError(f"{field_name} violates its closed contract")
  payload = dict(value)
  if set(payload) != expected_fields:
    raise ValueError(f"{field_name} violates its closed contract")
  return payload


def _exact_integer(
  value: object,
  *,
  field_name: str,
  minimum: int,
) -> int:
  if (
    type(value) is not int
    or value < minimum
    or value > _MAX_SQLITE_INTEGER
  ):
    raise ValueError(
      f"{field_name} must be an integer in "
      f"[{minimum}, {_MAX_SQLITE_INTEGER}]"
    )
  return value


def _canonical_identifier(value: object, *, field_name: str) -> str:
  if (
    type(value) is not str
    or not value
    or value != value.strip()
    or len(value) > _MAX_ID_TEXT_LENGTH
    or "\x00" in value
    or any(ord(character) < 0x20 for character in value)
  ):
    raise ValueError(f"{field_name} must be a canonical identifier")
  return value


def _canonical_absolute_path(value: object) -> str:
  if isinstance(value, Path):
    raw_path = str(value)
  elif type(value) is str:
    raw_path = value
  else:
    raise TypeError("launch nonce store path must be str or Path")
  if (
    not raw_path
    or raw_path != raw_path.strip()
    or "\x00" in raw_path
    or len(raw_path.encode("utf-8")) > _MAX_PATH_BYTES
    or any(ord(character) < 0x20 for character in raw_path)
  ):
    raise ValueError(
      "launch nonce store path must be a canonical absolute path"
    )
  path = Path(raw_path)
  if (
    not path.is_absolute()
    or str(path) != raw_path
    or ".." in path.parts
    or path.name in {"", ".", ".."}
  ):
    raise ValueError(
      "launch nonce store path must be a canonical absolute path"
    )
  return raw_path


@dataclass(frozen=True, slots=True)
class LaunchNonceStoreIdentity:
  """Immutable file identity intended for the signed launch envelope."""

  schema_version: int
  path: str
  device: int
  inode: int

  def __post_init__(self) -> None:
    if (
      type(self.schema_version) is not int
      or self.schema_version
      != LAUNCH_NONCE_STORE_SCHEMA_VERSION
    ):
      raise ValueError(
        "launch nonce store schema version is unsupported"
      )
    object.__setattr__(
      self,
      "path",
      _canonical_absolute_path(self.path),
    )
    _exact_integer(
      self.device,
      field_name="launch nonce store device",
      minimum=0,
    )
    _exact_integer(
      self.inode,
      field_name="launch nonce store inode",
      minimum=1,
    )

  @classmethod
  def from_verified_envelope(
    cls,
    envelope: AutonomousLaunchEnvelope,
  ) -> "LaunchNonceStoreIdentity":
    if type(envelope) is not AutonomousLaunchEnvelope:
      raise TypeError(
        "autonomous admission identity requires a verified launch envelope"
      )
    file_authority = envelope.control_authority.file_authority()
    return cls(
      schema_version=LAUNCH_NONCE_STORE_SCHEMA_VERSION,
      path=file_authority.admission_ledger_path,
      device=file_authority.admission_ledger_device,
      inode=file_authority.admission_ledger_inode,
    )


@dataclass(frozen=True, slots=True)
class LaunchNonceFacts:
  """Closed security facts copied from a verified ordinary envelope."""

  audience: str
  nonce: str
  task_id: str
  control_run_id: str
  owner_user_id: str
  channel_id: str
  issued_at_ns: int
  expires_at_ns: int

  def __post_init__(self) -> None:
    if (
      type(self.audience) is not str
      or self.audience != AUTONOMOUS_CAPABILITY_ENVELOPE_AUDIENCE
    ):
      raise ValueError("autonomous admission audience is invalid")
    if (
      type(self.nonce) is not str
      or _NONCE_RE.fullmatch(self.nonce) is None
    ):
      raise ValueError(
        "autonomous admission nonce must be 32 lowercase hex characters"
      )
    for field_name in (
      "task_id",
      "control_run_id",
      "owner_user_id",
    ):
      object.__setattr__(
        self,
        field_name,
        _canonical_identifier(
          getattr(self, field_name),
          field_name=f"autonomous admission {field_name}",
        ),
      )
    if (
      type(self.channel_id) is not str
      or _CHANNEL_ID_RE.fullmatch(self.channel_id) is None
    ):
      raise ValueError(
        "autonomous admission channel_id "
        "must be 64 lowercase hex characters"
      )
    issued_at_ns = _exact_integer(
      self.issued_at_ns,
      field_name="autonomous admission issued_at_ns",
      minimum=1,
    )
    expires_at_ns = _exact_integer(
      self.expires_at_ns,
      field_name="autonomous admission expires_at_ns",
      minimum=1,
    )
    if expires_at_ns <= issued_at_ns:
      raise ValueError(
        "autonomous admission expiry must follow issuance"
      )
    if (
      expires_at_ns - issued_at_ns
      > AUTONOMOUS_CAPABILITY_ENVELOPE_MAX_TTL_SECONDS
      * 1_000_000_000
    ):
      raise ValueError(
        "autonomous admission TTL exceeds the envelope maximum"
      )

  @classmethod
  def from_mapping(
    cls,
    value: object,
  ) -> "LaunchNonceFacts":
    return cls(**_closed_mapping(
      value,
      field_name="launch nonce facts",
      expected_fields=_FACTS_FIELDS,
    ))

  @classmethod
  def from_verified_envelope(
    cls,
    envelope: AutonomousLaunchEnvelope,
  ) -> "LaunchNonceFacts":
    if type(envelope) is not AutonomousLaunchEnvelope:
      raise TypeError(
        "ordinary autonomous admission requires a verified launch envelope"
      )
    return cls(
      audience=envelope.audience,
      nonce=envelope.nonce,
      task_id=envelope.task_id,
      control_run_id=envelope.control_run_id,
      owner_user_id=envelope.owner_user_id,
      channel_id=envelope.channel_id,
      issued_at_ns=envelope.iat_ns,
      expires_at_ns=envelope.exp_ns,
    )

  def to_mapping(self) -> dict[str, int | str]:
    return {
      "audience": self.audience,
      "nonce": self.nonce,
      "task_id": self.task_id,
      "control_run_id": self.control_run_id,
      "owner_user_id": self.owner_user_id,
      "channel_id": self.channel_id,
      "issued_at_ns": self.issued_at_ns,
      "expires_at_ns": self.expires_at_ns,
    }


@dataclass(frozen=True, slots=True)
class ConsumedLaunchNonce:
  """Durably committed one-time launch-nonce consume."""

  facts: LaunchNonceFacts
  admitted_at_ns: int

  def __post_init__(self) -> None:
    if type(self.facts) is not LaunchNonceFacts:
      raise TypeError(
        "consumed launch nonce requires exact launch nonce facts"
      )
    admitted_at_ns = _exact_integer(
      self.admitted_at_ns,
      field_name="autonomous admission admitted_at_ns",
      minimum=1,
    )
    if not (
      self.facts.issued_at_ns
      <= admitted_at_ns
      < self.facts.expires_at_ns
    ):
      raise ValueError(
        "autonomous admission record time is outside its launch lifetime"
      )

  @classmethod
  def from_wire(
    cls,
    value: object,
  ) -> "ConsumedLaunchNonce":
    payload = _closed_mapping(
      value,
      field_name="consumed launch nonce",
      expected_fields=_CONSUMED_NONCE_WIRE_FIELDS,
    )
    return cls(
      facts=LaunchNonceFacts.from_mapping(
        payload["receipt"]
      ),
      admitted_at_ns=payload["admitted_at_ns"],
    )

  def to_wire(self) -> dict[str, object]:
    return {
      "receipt": self.facts.to_mapping(),
      "admitted_at_ns": self.admitted_at_ns,
    }


def _normalized_ddl(sql: str | None) -> str:
  return " ".join((sql or "").replace("IF NOT EXISTS ", "").split())


def _add_exception_note(primary: BaseException, note: str) -> None:
  add_note = getattr(primary, "add_note", None)
  if callable(add_note):
    add_note(note)
    return
  try:
    notes = tuple(
      getattr(primary, "_autonomous_admission_notes", ())
    )
    setattr(
      primary,
      "_autonomous_admission_notes",
      (*notes, note),
    )
  except Exception:
    # Diagnostic attachment only: never replace the primary failure on 3.10.
    return


def _expected_metadata() -> tuple[int, int, int, int, int, int]:
  return (
    1,
    LAUNCH_NONCE_STORE_SCHEMA_VERSION,
    LAUNCH_NONCE_STORE_PAGE_SIZE,
    LAUNCH_NONCE_STORE_MAX_PAGE_COUNT,
    LAUNCH_NONCE_STORE_MAX_ROWS,
    LAUNCH_NONCE_STORE_CLEANUP_BATCH_SIZE,
  )


def _validate_policy_constants() -> None:
  policies = (
    (
      LAUNCH_NONCE_STORE_BUSY_TIMEOUT_MS,
      1,
      60_000,
      "busy timeout",
    ),
    (
      LAUNCH_NONCE_STORE_PAGE_SIZE,
      512,
      65_536,
      "page size",
    ),
    (
      LAUNCH_NONCE_STORE_MAX_PAGE_COUNT,
      16,
      1_073_741_823,
      "max page count",
    ),
    (
      LAUNCH_NONCE_STORE_MAX_ROWS,
      1,
      1_000_000,
      "max rows",
    ),
    (
      LAUNCH_NONCE_STORE_CLEANUP_BATCH_SIZE,
      1,
      LAUNCH_NONCE_STORE_MAX_ROWS,
      "cleanup batch size",
    ),
  )
  for value, minimum, maximum, field_name in policies:
    if (
      type(value) is not int
      or not minimum <= value <= maximum
    ):
      raise LaunchNonceStoreError(
        "launch nonce store "
        f"{field_name} is outside its fixed bound"
      )
  page_size = LAUNCH_NONCE_STORE_PAGE_SIZE
  if page_size & (page_size - 1):
    raise LaunchNonceStoreError(
      "launch nonce store page size must be a power of two"
    )


def _verify_parent_directory(path: Path) -> None:
  try:
    resolved_parent = path.parent.resolve(strict=True)
    parent_stat = os.lstat(path.parent)
  except OSError as exc:
    raise LaunchNonceStoreIdentityError(
      "launch nonce store parent directory is unavailable"
    ) from exc
  if resolved_parent != path.parent or not stat.S_ISDIR(parent_stat.st_mode):
    raise LaunchNonceStoreIdentityError(
      "launch nonce store parent path is not canonical"
    )
  if parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
    raise LaunchNonceStoreIdentityError(
      "launch nonce store parent directory is writable by others"
    )


def _database_stat(path: Path) -> os.stat_result:
  try:
    file_stat = os.lstat(path)
  except OSError as exc:
    raise LaunchNonceStoreIdentityError(
      "launch nonce store file is unavailable"
    ) from exc
  if stat.S_ISLNK(file_stat.st_mode):
    raise LaunchNonceStoreIdentityError(
      "launch nonce store must not be a symlink"
    )
  if not stat.S_ISREG(file_stat.st_mode):
    raise LaunchNonceStoreIdentityError(
      "launch nonce store must be a regular file"
    )
  if file_stat.st_nlink != 1:
    raise LaunchNonceStoreIdentityError(
      "launch nonce store must have exactly one hard link"
    )
  if (
    hasattr(os, "geteuid")
    and file_stat.st_uid != os.geteuid()
  ):
    raise LaunchNonceStoreIdentityError(
      "launch nonce store must be owned by the current user"
    )
  if stat.S_IMODE(file_stat.st_mode) != 0o600:
    raise LaunchNonceStoreIdentityError(
      "launch nonce store mode must be 0600"
    )
  return file_stat


def _verify_sidecars(path: Path) -> None:
  for suffix in ("-journal", "-wal", "-shm"):
    sidecar = Path(str(path) + suffix)
    try:
      sidecar_stat = os.lstat(sidecar)
    except FileNotFoundError:
      continue
    except OSError as exc:
      raise LaunchNonceStoreIdentityError(
        "launch nonce store sidecar cannot be inspected"
      ) from exc
    if (
      not stat.S_ISREG(sidecar_stat.st_mode)
      or sidecar_stat.st_nlink != 1
      or stat.S_IMODE(sidecar_stat.st_mode) != 0o600
      or (
        hasattr(os, "geteuid")
        and sidecar_stat.st_uid != os.geteuid()
      )
    ):
      raise LaunchNonceStoreIdentityError(
        "launch nonce store sidecar identity is unsafe"
      )


def _verify_identity(
  identity: LaunchNonceStoreIdentity,
) -> os.stat_result:
  if type(identity) is not LaunchNonceStoreIdentity:
    raise TypeError(
      "autonomous admission requires an exact signed ledger identity"
    )
  path = Path(identity.path)
  _verify_parent_directory(path)
  file_stat = _database_stat(path)
  if (
    file_stat.st_dev != identity.device
    or file_stat.st_ino != identity.inode
  ):
    raise LaunchNonceStoreIdentityError(
      "launch nonce store file identity changed"
    )
  _verify_sidecars(path)
  return file_stat


def _open_parent_fd(parent: Path) -> int:
  required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
  if os.name != "posix" or any(
    not hasattr(os, flag)
    for flag in required_flags
  ):
    raise LaunchNonceStoreIdentityError(
      "secure launch nonce store preparation requires POSIX"
    )
  flags = (
    os.O_RDONLY
    | os.O_CLOEXEC
    | os.O_DIRECTORY
    | os.O_NOFOLLOW
  )
  try:
    return os.open(parent, flags)
  except OSError as exc:
    raise LaunchNonceStoreIdentityError(
      "launch nonce store parent cannot be opened securely"
    ) from exc


def _prepare_database_file(path: Path) -> os.stat_result:
  _verify_parent_directory(path)
  parent_fd = _open_parent_fd(path.parent)
  try:
    flags = (
      os.O_RDWR
      | os.O_CREAT
      | os.O_CLOEXEC
      | os.O_NOFOLLOW
    )
    try:
      file_fd = os.open(
        path.name,
        flags,
        0o600,
        dir_fd=parent_fd,
      )
    except OSError as exc:
      raise LaunchNonceStoreIdentityError(
        "launch nonce store cannot be opened securely"
      ) from exc
    try:
      file_stat = os.fstat(file_fd)
      if (
        not stat.S_ISREG(file_stat.st_mode)
        or file_stat.st_nlink != 1
        or (
          hasattr(os, "geteuid")
          and file_stat.st_uid != os.geteuid()
        )
      ):
        raise LaunchNonceStoreIdentityError(
          "launch nonce store file identity is unsafe"
        )
      os.fchmod(file_fd, 0o600)
      os.fsync(file_fd)
      secured_stat = os.fstat(file_fd)
      if stat.S_IMODE(secured_stat.st_mode) != 0o600:
        raise LaunchNonceStoreIdentityError(
          "launch nonce store mode could not be secured"
        )
    finally:
      os.close(file_fd)
    os.fsync(parent_fd)
  except OSError as exc:
    raise LaunchNonceStoreUnavailable(
      "launch nonce store preparation is not durable"
    ) from exc
  finally:
    os.close(parent_fd)
  path_stat = _database_stat(path)
  if (
    path_stat.st_dev != secured_stat.st_dev
    or path_stat.st_ino != secured_stat.st_ino
  ):
    raise LaunchNonceStoreIdentityError(
      "launch nonce store changed during preparation"
    )
  return path_stat


def _configure_connection(connection: sqlite3.Connection) -> None:
  connection.execute(
    f"PRAGMA busy_timeout={LAUNCH_NONCE_STORE_BUSY_TIMEOUT_MS}"
  )
  connection.execute("PRAGMA foreign_keys=ON")
  connection.execute("PRAGMA trusted_schema=OFF")
  journal_mode = connection.execute(
    "PRAGMA journal_mode=DELETE"
  ).fetchone()[0]
  connection.execute("PRAGMA synchronous=FULL")
  synchronous = connection.execute(
    "PRAGMA synchronous"
  ).fetchone()[0]
  connection.execute(
    f"PRAGMA page_size={LAUNCH_NONCE_STORE_PAGE_SIZE}"
  )
  page_size = connection.execute("PRAGMA page_size").fetchone()[0]
  max_page_count = connection.execute(
    "PRAGMA max_page_count="
    f"{LAUNCH_NONCE_STORE_MAX_PAGE_COUNT}"
  ).fetchone()[0]
  page_count = connection.execute("PRAGMA page_count").fetchone()[0]
  if (
    str(journal_mode).lower() != "delete"
    or synchronous != 2
    or page_size != LAUNCH_NONCE_STORE_PAGE_SIZE
    or max_page_count
    != LAUNCH_NONCE_STORE_MAX_PAGE_COUNT
    or page_count > max_page_count
  ):
    raise LaunchNonceStoreUnavailable(
      "launch nonce store durability bounds are unavailable"
    )


def _connect(
  identity: LaunchNonceStoreIdentity,
) -> sqlite3.Connection:
  _verify_identity(identity)
  connection = sqlite3.connect(
    Path(identity.path).as_uri() + "?mode=rw",
    uri=True,
    timeout=(
      LAUNCH_NONCE_STORE_BUSY_TIMEOUT_MS / 1_000
    ),
    isolation_level=None,
  )
  try:
    connection.row_factory = sqlite3.Row
    _configure_connection(connection)
    _verify_identity(identity)
    return connection
  except BaseException as exc:
    try:
      connection.close()
    except sqlite3.Error as close_error:
      _add_exception_note(
        exc,
        "autonomous admission connection close also failed: "
        f"{close_error!r}",
      )
    raise


def _validate_schema(connection: sqlite3.Connection) -> None:
  user_version = connection.execute("PRAGMA user_version").fetchone()[0]
  if user_version != LAUNCH_NONCE_STORE_SCHEMA_VERSION:
    raise LaunchNonceStoreUnavailable(
      "launch nonce store schema version is incompatible"
    )
  rows = connection.execute(
    """
    SELECT type, name, sql
      FROM sqlite_master
     WHERE name NOT LIKE 'sqlite_%'
       AND sql IS NOT NULL
     ORDER BY name
     LIMIT ?
    """,
    (len(_SCHEMA_OBJECTS) + 1,),
  ).fetchall()
  observed = {row["name"]: (row["type"], row["sql"]) for row in rows}
  if set(observed) != set(_SCHEMA_OBJECTS):
    raise LaunchNonceStoreUnavailable(
      "launch nonce store schema objects are incompatible"
    )
  for name, (expected_type, expected_sql) in _SCHEMA_OBJECTS.items():
    observed_type, observed_sql = observed[name]
    if (
      observed_type != expected_type
      or _normalized_ddl(observed_sql)
      != _normalized_ddl(expected_sql)
    ):
      raise LaunchNonceStoreUnavailable(
        "launch nonce store schema constraints are incompatible"
      )
  metadata = connection.execute(
    """
    SELECT singleton, schema_version, page_size, max_page_count,
           max_rows, cleanup_batch_size
      FROM autonomous_admission_ledger_metadata
    """
  ).fetchall()
  if (
    len(metadata) != 1
    or tuple(metadata[0]) != _expected_metadata()
  ):
    raise LaunchNonceStoreUnavailable(
      "launch nonce store bounds are incompatible"
    )
  clock_rows = connection.execute(
    """
    SELECT singleton, last_admitted_wall_ns
      FROM autonomous_admission_ledger_clock
    """
  ).fetchall()
  if (
    len(clock_rows) != 1
    or type(clock_rows[0]["last_admitted_wall_ns"]) is not int
    or clock_rows[0]["singleton"] != 1
    or clock_rows[0]["last_admitted_wall_ns"] < 0
  ):
    raise LaunchNonceStoreUnavailable(
      "launch nonce store clock state is incompatible"
    )


def _initialize_or_validate_schema(
  connection: sqlite3.Connection,
) -> None:
  connection.execute("BEGIN EXCLUSIVE")
  try:
    existing = connection.execute(
      """
      SELECT name
        FROM sqlite_master
       WHERE name NOT LIKE 'sqlite_%'
         AND sql IS NOT NULL
       LIMIT 1
      """
    ).fetchall()
    if not existing:
      connection.execute(_METADATA_TABLE_SQL)
      connection.execute(_CLOCK_TABLE_SQL)
      connection.execute(_ADMISSIONS_TABLE_SQL)
      connection.execute(_EXPIRY_INDEX_SQL)
      connection.execute(_NO_ADMISSION_UPDATE_TRIGGER_SQL)
      connection.execute(_NO_METADATA_UPDATE_TRIGGER_SQL)
      connection.execute(_NO_METADATA_DELETE_TRIGGER_SQL)
      connection.execute(_NO_CLOCK_DELETE_TRIGGER_SQL)
      connection.execute(
        """
        INSERT INTO autonomous_admission_ledger_metadata (
          singleton, schema_version, page_size, max_page_count,
          max_rows, cleanup_batch_size
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        _expected_metadata(),
      )
      connection.execute(
        """
        INSERT INTO autonomous_admission_ledger_clock (
          singleton, last_admitted_wall_ns
        ) VALUES (1, 0)
        """
      )
      connection.execute(
        "PRAGMA user_version="
        f"{LAUNCH_NONCE_STORE_SCHEMA_VERSION}"
      )
    _validate_schema(connection)
    connection.commit()
  except BaseException as exc:
    try:
      connection.rollback()
    except sqlite3.Error as rollback_error:
      _add_exception_note(
        exc,
        "autonomous admission schema rollback also failed: "
        f"{rollback_error!r}",
      )
    raise


def prepare_launch_nonce_store(
  path: str | Path,
) -> LaunchNonceStoreIdentity:
  """Create or validate the sole canonical ledger and return its identity."""

  _validate_policy_constants()
  canonical_path = Path(_canonical_absolute_path(path))
  file_stat = _prepare_database_file(canonical_path)
  identity = LaunchNonceStoreIdentity(
    schema_version=LAUNCH_NONCE_STORE_SCHEMA_VERSION,
    path=str(canonical_path),
    device=file_stat.st_dev,
    inode=file_stat.st_ino,
  )
  try:
    connection = _connect(identity)
  except sqlite3.Error as exc:
    raise LaunchNonceStoreUnavailable(
      "launch nonce store cannot be prepared"
    ) from exc
  primary_error: BaseException | None = None
  try:
    _initialize_or_validate_schema(connection)
    _verify_identity(identity)
  except sqlite3.Error as exc:
    error = LaunchNonceStoreUnavailable(
      "launch nonce store schema preparation failed"
    )
    primary_error = error
    raise error from exc
  except BaseException as exc:
    primary_error = exc
    raise
  finally:
    try:
      connection.close()
    except sqlite3.Error as close_error:
      if primary_error is not None:
        _add_exception_note(
          primary_error,
          "autonomous admission connection close also failed: "
          f"{close_error!r}",
        )
      else:
        raise LaunchNonceStoreUnavailable(
          "autonomous admission connection could not be closed"
        ) from close_error
  _verify_identity(identity)
  return identity


def _wall_time_ns(clock_ns: Callable[[], int]) -> int:
  now_ns = clock_ns()
  return _exact_integer(
    now_ns,
    field_name="autonomous admission wall clock",
    minimum=1,
  )


def _ensure_live(
  facts: LaunchNonceFacts,
  *,
  now_ns: int,
  consumed: bool,
) -> None:
  if now_ns < facts.issued_at_ns:
    raise LaunchNonceExpired(
      "autonomous launch is not yet valid at durable admission",
      consumed=consumed,
    )
  if now_ns >= facts.expires_at_ns:
    raise LaunchNonceExpired(
      "autonomous launch expired at durable admission",
      consumed=consumed,
    )


def _rollback_preserving(
  connection: sqlite3.Connection,
  primary: BaseException,
) -> None:
  try:
    connection.rollback()
  except sqlite3.Error as rollback_error:
    _add_exception_note(
      primary,
      "autonomous admission rollback also failed: "
      f"{rollback_error!r}",
    )


def _record_from_row(row: sqlite3.Row) -> ConsumedLaunchNonce:
  facts = LaunchNonceFacts(
    audience=row["audience"],
    nonce=row["nonce"],
    task_id=row["task_id"],
    control_run_id=row["control_run_id"],
    owner_user_id=row["owner_user_id"],
    channel_id=row["channel_id"],
    issued_at_ns=row["issued_at_ns"],
    expires_at_ns=row["expires_at_ns"],
  )
  admitted_at_ns = _exact_integer(
    row["admitted_at_ns"],
    field_name="autonomous admission admitted_at_ns",
    minimum=1,
  )
  return ConsumedLaunchNonce(
    facts=facts,
    admitted_at_ns=admitted_at_ns,
  )


def consume_launch_nonce(
  expected_identity: LaunchNonceStoreIdentity,
  facts: LaunchNonceFacts,
  *,
  clock_ns: Callable[[], int] = time.time_ns,
) -> ConsumedLaunchNonce:
  """Atomically consume one verified ordinary launch before child admission."""

  _validate_policy_constants()
  if type(expected_identity) is not LaunchNonceStoreIdentity:
    raise TypeError(
      "autonomous admission requires an exact signed ledger identity"
    )
  if type(facts) is not LaunchNonceFacts:
    raise TypeError(
      "launch nonce consume requires exact launch nonce facts"
    )
  if not callable(clock_ns):
    raise TypeError("autonomous admission clock must be callable")
  _verify_identity(expected_identity)
  _ensure_live(
    facts,
    now_ns=_wall_time_ns(clock_ns),
    consumed=False,
  )

  try:
    connection = _connect(expected_identity)
  except sqlite3.Error as exc:
    raise LaunchNonceStoreUnavailable(
      "launch nonce store cannot be opened"
    ) from exc
  committed = False
  admitted_at_ns = 0
  primary_error: BaseException | None = None
  try:
    try:
      connection.execute("BEGIN IMMEDIATE")
      _verify_identity(expected_identity)
      _validate_schema(connection)
      admitted_at_ns = _wall_time_ns(clock_ns)
      _ensure_live(
        facts,
        now_ns=admitted_at_ns,
        consumed=False,
      )
      clock_row = connection.execute(
        """
        SELECT last_admitted_wall_ns
          FROM autonomous_admission_ledger_clock
         WHERE singleton = 1
        """
      ).fetchone()
      if (
        clock_row is None
        or type(clock_row["last_admitted_wall_ns"]) is not int
      ):
        raise LaunchNonceStoreUnavailable(
          "launch nonce store clock state is unavailable"
        )
      last_admitted_wall_ns = clock_row["last_admitted_wall_ns"]
      if admitted_at_ns < last_admitted_wall_ns:
        raise LaunchNonceClockRollback(
          "autonomous admission wall clock moved backward"
        )
      clock_update = connection.execute(
        """
        UPDATE autonomous_admission_ledger_clock
           SET last_admitted_wall_ns = ?
         WHERE singleton = 1
           AND last_admitted_wall_ns = ?
        """,
        (admitted_at_ns, last_admitted_wall_ns),
      )
      if clock_update.rowcount != 1:
        raise LaunchNonceStoreUnavailable(
          "launch nonce store clock update was not exclusive"
        )
      connection.execute(
        """
        DELETE FROM ordinary_autonomous_launch_admissions
         WHERE nonce IN (
           SELECT nonce
             FROM ordinary_autonomous_launch_admissions
            WHERE expires_at_ns <= ?
            ORDER BY expires_at_ns, nonce
            LIMIT ?
         )
        """,
        (
          admitted_at_ns,
          LAUNCH_NONCE_STORE_CLEANUP_BATCH_SIZE,
        ),
      )
      try:
        connection.execute(
          """
          INSERT INTO ordinary_autonomous_launch_admissions (
            nonce, schema_version, audience, task_id, control_run_id,
            owner_user_id, channel_id, issued_at_ns, expires_at_ns,
            admitted_at_ns
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          """,
          (
            facts.nonce,
            LAUNCH_NONCE_STORE_SCHEMA_VERSION,
            facts.audience,
            facts.task_id,
            facts.control_run_id,
            facts.owner_user_id,
            facts.channel_id,
            facts.issued_at_ns,
            facts.expires_at_ns,
            admitted_at_ns,
          ),
        )
      except sqlite3.IntegrityError as exc:
        existing_nonce = connection.execute(
          """
          SELECT 1
            FROM ordinary_autonomous_launch_admissions
           WHERE nonce = ?
          """,
          (facts.nonce,),
        ).fetchone()
        if existing_nonce is not None:
          raise LaunchNonceReplay(
            "autonomous launch nonce has already been consumed"
          ) from exc
        raise
      row_count = connection.execute(
        "SELECT COUNT(*) FROM ordinary_autonomous_launch_admissions"
      ).fetchone()[0]
      if (
        type(row_count) is not int
        or row_count > LAUNCH_NONCE_STORE_MAX_ROWS
      ):
        raise LaunchNonceStoreCapacityExceeded(
          "launch nonce store reached its fixed row capacity"
        )
      _verify_identity(expected_identity)
      connection.commit()
      committed = True
    except BaseException as exc:
      if not committed:
        _rollback_preserving(connection, exc)
      raise

    _verify_identity(expected_identity)
    row = connection.execute(
      """
      SELECT nonce, audience, task_id, control_run_id, owner_user_id,
             channel_id, issued_at_ns, expires_at_ns, admitted_at_ns
        FROM ordinary_autonomous_launch_admissions
       WHERE nonce = ?
      """,
      (facts.nonce,),
    ).fetchone()
    if row is None:
      raise LaunchNonceStoreUnavailable(
        "committed autonomous admission is not visible"
      )
    record = _record_from_row(row)
    if (
      record.facts != facts
      or record.admitted_at_ns != admitted_at_ns
    ):
      raise LaunchNonceStoreUnavailable(
        "committed launch nonce facts changed"
      )
    observed_after_commit_ns = _wall_time_ns(clock_ns)
    if observed_after_commit_ns < admitted_at_ns:
      raise LaunchNonceClockRollback(
        "autonomous admission wall clock moved backward"
      )
    _ensure_live(
      facts,
      now_ns=observed_after_commit_ns,
      consumed=True,
    )
    _verify_identity(expected_identity)
    return record
  except sqlite3.Error as exc:
    error = LaunchNonceStoreUnavailable(
      "launch nonce store transaction failed"
    )
    primary_error = error
    raise error from exc
  except BaseException as exc:
    primary_error = exc
    raise
  finally:
    try:
      connection.close()
    except sqlite3.Error as close_error:
      if primary_error is not None:
        _add_exception_note(
          primary_error,
          "autonomous admission connection close also failed: "
          f"{close_error!r}",
        )
      else:
        raise LaunchNonceStoreUnavailable(
          "autonomous admission connection could not be closed"
        ) from close_error


__all__ = [
  "LAUNCH_NONCE_STORE_SCHEMA_VERSION",
  "ConsumedLaunchNonce",
  "LaunchNonceClockRollback",
  "LaunchNonceExpired",
  "LaunchNonceFacts",
  "LaunchNonceReplay",
  "LaunchNonceStoreCapacityExceeded",
  "LaunchNonceStoreError",
  "LaunchNonceStoreIdentity",
  "LaunchNonceStoreIdentityError",
  "LaunchNonceStoreUnavailable",
  "consume_launch_nonce",
  "prepare_launch_nonce_store",
]
