from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import fcntl
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Awaitable, BinaryIO, Callable, Literal, Mapping

from .artifact_paths import canonicalize_ticker
from .skill_completion_wal import (
  TopLevelSkillCompletionEffectPlan,
  canonical_json_bytes,
)


SkillLifecycleScope = Literal["ticker", "portfolio"]
SemanticSkillScope = Literal["ticker", "portfolio", "industry"]
TopLevelServerTerminalCause = Literal[
  "caller_cancellation",
  "shutdown",
  "timeout",
]
TopLevelSkillTerminalToolDisposition = Literal[
  "success",
  "failure",
]
TopLevelSkillCompletionPreparer = Callable[
  [Any, dict[str, Any]],
  (
    "TopLevelSkillCompletionPlan"
    | Awaitable["TopLevelSkillCompletionPlan"]
  ),
]
TopLevelSkillAdmissionState = Literal[
  "held",
  "transferred",
  "released",
]
TopLevelSkillProviderPreparer = Callable[
  [Any],
  Any | Awaitable[Any],
]
TopLevelSkillTerminalToolObserver = Callable[
  [Any],
  (
    TopLevelSkillTerminalToolDisposition
    | None
    | Awaitable[TopLevelSkillTerminalToolDisposition | None]
  ),
]

_IDENTITY_FIELDS = (
  "skill_run_id",
  "skill",
  "scope",
  "ticker",
  "portfolio_id",
)
SKILL_RESULT_CORE_FIELDS = frozenset({
  "type",
  *_IDENTITY_FIELDS,
  "exit_code",
  "outcome",
  "status",
  "gate_code",
  "artifact_refs",
  "proposal_ids",
  "verdict_echo",
  "fms_results",
  "artifact_events",
  "output_memory_file",
  "cost_usd",
  "duration_s",
  "compaction_count",
  "error",
  "warnings",
  "approval_outcome",
  "approval_id",
  "approval_tool_name",
})
_FAILURE_RESULT_STATUSES = frozenset({
  "error",
  "failed",
  "failure",
  "invalid",
  "rejected",
})
_LEASE_FENCE_SCHEMA_VERSION = 1
_MAX_LEASE_FENCE_BYTES = 4 * 1024 * 1024


class WriterLeaseAlreadyHeldError(RuntimeError):
  """Raised when another writer owns the durable session lease."""


def _normalized_absolute_path(path: str | Path) -> Path:
  expanded = Path(path).expanduser()
  return Path(os.path.abspath(os.path.normpath(os.fspath(expanded))))


def _write_all_fd(fd: int, payload: bytes) -> None:
  offset = 0
  while offset < len(payload):
    written = os.write(fd, payload[offset:])
    if written <= 0:
      raise OSError("short write while persisting writer lease fence")
    offset += written


def _require_regular_owned_path(
  path: Path,
  *,
  expected_uid: int,
  expected_identity: tuple[int, int] | None = None,
  required_mode: int | None = None,
) -> os.stat_result:
  try:
    path_stat = path.lstat()
  except FileNotFoundError as exc:
    raise RuntimeError(
      f"Top-level skill admission path disappeared: {path}"
    ) from exc
  if stat.S_ISLNK(path_stat.st_mode):
    raise RuntimeError(
      f"Top-level skill admission rejects symlink path: {path}"
    )
  if not stat.S_ISREG(path_stat.st_mode):
    raise RuntimeError(
      f"Top-level skill admission requires a regular file: {path}"
    )
  if path_stat.st_uid != expected_uid:
    raise RuntimeError(
      f"Top-level skill admission owner changed for {path}"
    )
  if path_stat.st_nlink != 1:
    raise RuntimeError(
      f"Top-level skill admission requires one hard link for {path}"
    )
  path_mode = stat.S_IMODE(path_stat.st_mode)
  if (
    required_mode is not None
    and path_mode != required_mode
  ):
    raise RuntimeError(
      "Top-level skill admission requires mode "
      f"{required_mode:04o} for {path}"
    )
  if required_mode is None and path_mode & 0o022:
    raise RuntimeError(
      f"Top-level skill admission rejects writable mode for {path}"
    )
  if (
    expected_identity is not None
    and (path_stat.st_dev, path_stat.st_ino) != expected_identity
  ):
    raise RuntimeError(
      f"Top-level skill admission file identity changed for {path}"
    )
  return path_stat


def _lease_fence_record(
  *,
  generation: int,
  owner_token: str,
) -> dict[str, Any]:
  body = {
    "schema_version": _LEASE_FENCE_SCHEMA_VERSION,
    "generation": generation,
    "owner_token": owner_token,
  }
  checksum = (
    "sha256:"
    + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
  )
  return {**body, "checksum": checksum}


def _parse_lease_fence_records(raw: bytes) -> list[dict[str, Any]]:
  if not raw:
    return []
  if len(raw) > _MAX_LEASE_FENCE_BYTES:
    raise RuntimeError("Writer lease fence history exceeds 4 MiB")
  if not raw.endswith(b"\n"):
    raise RuntimeError("Writer lease fence history has a torn tail")
  records: list[dict[str, Any]] = []
  prior_generation = 0
  for index, line in enumerate(raw.splitlines()):
    try:
      record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise RuntimeError(
        f"Writer lease fence record {index} is unreadable"
      ) from exc
    if (
      type(record) is not dict
      or set(record)
      != {
        "schema_version",
        "generation",
        "owner_token",
        "checksum",
      }
      or type(record["schema_version"]) is not int
      or record["schema_version"] != _LEASE_FENCE_SCHEMA_VERSION
      or type(record["generation"]) is not int
      or record["generation"] != prior_generation + 1
      or type(record["owner_token"]) is not str
      or len(record["owner_token"]) != 64
      or type(record["checksum"]) is not str
    ):
      raise RuntimeError(
        f"Writer lease fence record {index} is invalid"
      )
    try:
      bytes.fromhex(record["owner_token"])
    except ValueError as exc:
      raise RuntimeError(
        f"Writer lease fence record {index} owner token is invalid"
      ) from exc
    body = {
      "schema_version": record["schema_version"],
      "generation": record["generation"],
      "owner_token": record["owner_token"],
    }
    expected_checksum = (
      "sha256:"
      + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    )
    if record["checksum"] != expected_checksum:
      raise RuntimeError(
        f"Writer lease fence record {index} checksum mismatch"
      )
    records.append(record)
    prior_generation = record["generation"]
  return records


@dataclass(slots=True)
class TopLevelSkillAdmission:
  """Single-owner transfer of one pre-acquired top-level writer lease."""

  log_path: Path
  write_lease_path: Path
  _lease_file: BinaryIO = field(repr=False)
  _owner_uid: int = field(repr=False)
  _lease_identity: tuple[int, int] = field(repr=False)
  lease_generation: int
  lease_owner_token: str = field(repr=False)
  _initial_log_identity: tuple[int, int] | None = field(
    repr=False,
    default=None,
  )
  _state: TopLevelSkillAdmissionState = field(
    init=False,
    default="held",
  )
  _release_count: int = field(init=False, default=0, repr=False)

  @classmethod
  def acquire(
    cls,
    log_path: str | Path,
  ) -> "TopLevelSkillAdmission":
    """Acquire the exact derived write lease without constructing the log."""

    canonical_log_path = _normalized_absolute_path(log_path)
    if canonical_log_path.name in {"", ".", ".."}:
      raise ValueError("log_path must identify one exact file")
    canonical_lease_path = canonical_log_path.with_name(
      f"{canonical_log_path.name}.write_lease"
    )
    canonical_log_path.parent.mkdir(parents=True, exist_ok=True)

    owner_uid = os.geteuid()
    initial_log_identity: tuple[int, int] | None = None
    try:
      initial_log_stat = canonical_log_path.lstat()
    except FileNotFoundError:
      pass
    else:
      initial_log_stat = _require_regular_owned_path(
        canonical_log_path,
        expected_uid=owner_uid,
      )
      initial_log_identity = (
        initial_log_stat.st_dev,
        initial_log_stat.st_ino,
      )

    flags = os.O_RDWR | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    lease_file: BinaryIO | None = None
    lease_created = False
    try:
      try:
        fd = os.open(
          canonical_lease_path,
          flags | os.O_CREAT | os.O_EXCL,
          0o600,
        )
        lease_created = True
      except FileExistsError:
        fd = os.open(canonical_lease_path, flags)
      lease_file = os.fdopen(fd, "a+b", buffering=0)
      fd = None
      lease_stat = os.fstat(lease_file.fileno())
      if not stat.S_ISREG(lease_stat.st_mode):
        raise RuntimeError(
          "Top-level skill admission lease is not a regular file: "
          f"{canonical_lease_path}"
        )
      if lease_stat.st_uid != owner_uid:
        raise RuntimeError(
          "Top-level skill admission rejects lease owner drift: "
          f"{canonical_lease_path}"
        )
      if lease_stat.st_nlink != 1:
        raise RuntimeError(
          "Top-level skill admission requires a single-link lease: "
          f"{canonical_lease_path}"
        )
      if lease_created:
        os.fchmod(lease_file.fileno(), 0o600)
      elif stat.S_IMODE(lease_stat.st_mode) != 0o600:
        raise RuntimeError(
          "Top-level skill admission requires mode 0600 for "
          f"{canonical_lease_path}"
        )
      lease_identity = (lease_stat.st_dev, lease_stat.st_ino)
      _require_regular_owned_path(
        canonical_lease_path,
        expected_uid=owner_uid,
        expected_identity=lease_identity,
        required_mode=0o600,
      )
      try:
        fcntl.flock(
          lease_file.fileno(),
          fcntl.LOCK_EX | fcntl.LOCK_NB,
        )
      except BlockingIOError as exc:
        raise WriterLeaseAlreadyHeldError(
          f"Writer lease already held for {canonical_log_path}"
        ) from exc
      lease_file.seek(0)
      lease_history = lease_file.read(
        _MAX_LEASE_FENCE_BYTES + 1
      )
      fence_records = _parse_lease_fence_records(lease_history)
      lease_generation = len(fence_records) + 1
      lease_owner_token = secrets.token_hex(32)
      fence_record = _lease_fence_record(
        generation=lease_generation,
        owner_token=lease_owner_token,
      )
      fence_line = canonical_json_bytes(fence_record) + b"\n"
      if len(lease_history) + len(fence_line) > _MAX_LEASE_FENCE_BYTES:
        raise RuntimeError(
          "Writer lease fence history cannot accept another generation"
      )
      lease_file.seek(0, os.SEEK_END)
      _write_all_fd(lease_file.fileno(), fence_line)
      lease_file.flush()
      os.fsync(lease_file.fileno())
      lease_file.seek(0)
      persisted_records = _parse_lease_fence_records(
        lease_file.read(_MAX_LEASE_FENCE_BYTES + 1)
      )
      if (
        not persisted_records
        or persisted_records[-1] != fence_record
      ):
        raise RuntimeError(
          "Writer lease fence readback does not match the new generation"
        )
      if lease_created:
        parent_flags = os.O_RDONLY | getattr(
          os,
          "O_DIRECTORY",
          0,
        )
        parent_flags |= getattr(os, "O_CLOEXEC", 0)
        parent_flags |= getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(
          canonical_lease_path.parent,
          parent_flags,
        )
        try:
          os.fsync(parent_fd)
        finally:
          os.close(parent_fd)
      _require_regular_owned_path(
        canonical_lease_path,
        expected_uid=owner_uid,
        expected_identity=lease_identity,
        required_mode=0o600,
      )
      return cls(
        log_path=canonical_log_path,
        write_lease_path=canonical_lease_path,
        _lease_file=lease_file,
        _owner_uid=owner_uid,
        _lease_identity=lease_identity,
        lease_generation=lease_generation,
        lease_owner_token=lease_owner_token,
        _initial_log_identity=initial_log_identity,
      )
    except BaseException:
      if lease_file is not None:
        try:
          fcntl.flock(lease_file.fileno(), fcntl.LOCK_UN)
        except (OSError, ValueError):
          pass
        lease_file.close()
      elif fd is not None:
        os.close(fd)
      raise

  @property
  def state(self) -> TopLevelSkillAdmissionState:
    return self._state

  @property
  def release_count(self) -> int:
    return self._release_count

  @property
  def fence(self) -> dict[str, Any]:
    return {
      "generation": self.lease_generation,
      "owner_token": self.lease_owner_token,
    }

  def _validate_storage_identity(self) -> None:
    lease_stat = os.fstat(self._lease_file.fileno())
    if not stat.S_ISREG(lease_stat.st_mode):
      raise RuntimeError(
        "Top-level skill admission lease descriptor is not regular"
      )
    if lease_stat.st_uid != self._owner_uid:
      raise RuntimeError(
        "Top-level skill admission lease descriptor owner changed"
      )
    if lease_stat.st_nlink != 1:
      raise RuntimeError(
        "Top-level skill admission lease descriptor link count changed"
      )
    if stat.S_IMODE(lease_stat.st_mode) != 0o600:
      raise RuntimeError(
        "Top-level skill admission lease descriptor mode changed"
      )
    if (
      lease_stat.st_dev,
      lease_stat.st_ino,
    ) != self._lease_identity:
      raise RuntimeError(
        "Top-level skill admission lease descriptor identity changed"
      )
    _require_regular_owned_path(
      self.write_lease_path,
      expected_uid=self._owner_uid,
      expected_identity=self._lease_identity,
      required_mode=0o600,
    )
    current_log_stat = _require_regular_owned_path(
      self.log_path,
      expected_uid=self._owner_uid,
      expected_identity=self._initial_log_identity,
    )
    if self._initial_log_identity is None:
      self._initial_log_identity = (
        current_log_stat.st_dev,
        current_log_stat.st_ino,
      )

  def validate_fence(self) -> None:
    """Prove the exact held descriptor still owns the latest generation."""

    if self._state not in {"held", "transferred"}:
      raise RuntimeError(
        "Top-level skill admission fence is no longer held"
      )
    self._validate_storage_identity()
    self._lease_file.seek(0)
    records = _parse_lease_fence_records(
      self._lease_file.read(_MAX_LEASE_FENCE_BYTES + 1)
    )
    if not records:
      raise RuntimeError("Writer lease fence history is empty")
    latest = records[-1]
    if (
      latest["generation"] != self.lease_generation
      or latest["owner_token"] != self.lease_owner_token
    ):
      raise RuntimeError(
        "Top-level skill admission fence is stale"
      )

  def transfer(
    self,
    *,
    log_path: str | Path,
    write_lease_path: str | Path,
  ) -> BinaryIO:
    """Transfer the held descriptor once to the exact admitted runner."""

    if self._state != "held":
      raise RuntimeError(
        "Top-level skill admission can transfer only from held state"
      )
    if _normalized_absolute_path(log_path) != self.log_path:
      raise RuntimeError(
        "AgentRunner session log does not match top-level admission"
      )
    if (
      _normalized_absolute_path(write_lease_path)
      != self.write_lease_path
    ):
      raise RuntimeError(
        "AgentRunner write lease does not match top-level admission"
      )
    self._validate_storage_identity()
    self.validate_fence()
    self._state = "transferred"
    return self._lease_file

  def release(self) -> bool:
    """Release once from either owner; repeated release is a no-op."""

    if self._state == "released":
      return False
    release_error: BaseException | None = None
    try:
      fcntl.flock(self._lease_file.fileno(), fcntl.LOCK_UN)
    except BaseException as exc:
      release_error = exc
    try:
      self._lease_file.close()
    except BaseException as exc:
      if release_error is None:
        release_error = exc
    self._state = "released"
    self._release_count += 1
    if release_error is not None:
      raise release_error
    return True


def _require_exact_nonempty_string(value: Any, *, field_name: str) -> str:
  if type(value) is not str or not value or value != value.strip():
    raise ValueError(
      f"{field_name} must be a non-empty string without surrounding whitespace"
    )
  return value


def _require_optional_exact_string(
  value: Any,
  *,
  field_name: str,
) -> str | None:
  if value is None:
    return None
  return _require_exact_nonempty_string(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class SkillLifecycleArtifactIdentity:
  """Canonical lifecycle identity derived from a semantic skill scope."""

  scope: SkillLifecycleScope
  ticker: str | None
  portfolio_id: str | None

  def __post_init__(self) -> None:
    if self.scope not in ("ticker", "portfolio"):
      raise ValueError("scope must be exactly 'ticker' or 'portfolio'")
    _require_optional_exact_string(self.ticker, field_name="ticker")
    _require_optional_exact_string(
      self.portfolio_id,
      field_name="portfolio_id",
    )
    if self.scope == "ticker":
      if self.ticker is None:
        raise ValueError("ticker lifecycle scope requires ticker")
      if self.portfolio_id is not None:
        raise ValueError("ticker lifecycle scope cannot carry portfolio_id")
    elif self.ticker is not None:
      raise ValueError("portfolio lifecycle scope cannot carry ticker")

  def identity_fields(self) -> dict[str, Any]:
    return {
      "scope": self.scope,
      "ticker": self.ticker,
      "portfolio_id": self.portfolio_id,
    }


def resolve_skill_lifecycle_artifact_identity(
  *,
  semantic_scope: str | None,
  context_ticker: str | None,
  portfolio_id: str | None,
) -> SkillLifecycleArtifactIdentity:
  """Map semantic skill scope to the one canonical artifact identity."""

  if semantic_scope not in (None, "ticker", "portfolio", "industry"):
    raise ValueError(
      "semantic_scope must be exactly None, 'ticker', 'portfolio', "
      "or 'industry'"
    )
  _require_optional_exact_string(
    context_ticker,
    field_name="context_ticker",
  )
  if semantic_scope == "ticker" and context_ticker is not None:
    return SkillLifecycleArtifactIdentity(
      scope="ticker",
      ticker=canonicalize_ticker(context_ticker),
      portfolio_id=None,
    )
  _require_optional_exact_string(
    portfolio_id,
    field_name="portfolio_id",
  )
  return SkillLifecycleArtifactIdentity(
    scope="portfolio",
    ticker=None,
    portfolio_id=portfolio_id,
  )


def _require_string_value(value: Any, *, field_name: str) -> None:
  if type(value) is not str:
    raise RuntimeError(f"{field_name} must be a string")


def _require_optional_string_value(
  value: Any,
  *,
  field_name: str,
) -> None:
  if value is not None:
    _require_string_value(value, field_name=field_name)


def _require_string_list(value: Any, *, field_name: str) -> None:
  if type(value) is not list:
    raise RuntimeError(f"{field_name} must be a list")
  for index, item in enumerate(value):
    if type(item) is not str:
      raise RuntimeError(
        f"{field_name}[{index}] must be a string"
      )


def _require_mapping_list(value: Any, *, field_name: str) -> None:
  if type(value) is not list:
    raise RuntimeError(f"{field_name} must be a list")
  for index, item in enumerate(value):
    if type(item) is not dict:
      raise RuntimeError(
        f"{field_name}[{index}] must be a mapping"
      )


def _require_nonnegative_number(
  value: Any,
  *,
  field_name: str,
  optional: bool,
) -> None:
  if value is None and optional:
    return
  if type(value) not in {int, float}:
    raise RuntimeError(f"{field_name} must be a number")
  number = float(value)
  if not math.isfinite(number) or number < 0:
    raise RuntimeError(
      f"{field_name} must be finite and non-negative"
    )


def _canonical_json_value(
  value: Any,
  *,
  field_name: str,
  ancestors: set[int] | None = None,
) -> Any:
  """Validate and deep-own one strict, finite canonical JSON value."""

  if value is None or type(value) in {str, bool, int}:
    return value
  if type(value) is float:
    if not math.isfinite(value):
      raise RuntimeError(f"{field_name} must be finite")
    return value
  if type(value) not in {list, dict}:
    raise RuntimeError(
      f"{field_name} contains non-JSON value "
      f"{type(value).__name__}"
    )
  active = ancestors if ancestors is not None else set()
  identity = id(value)
  if identity in active:
    raise RuntimeError(f"{field_name} contains a cycle")
  active.add(identity)
  try:
    if type(value) is list:
      return [
        _canonical_json_value(
          item,
          field_name=f"{field_name}[{index}]",
          ancestors=active,
        )
        for index, item in enumerate(value)
      ]
    canonical: dict[str, Any] = {}
    for key, item in value.items():
      if type(key) is not str:
        raise RuntimeError(
          f"{field_name} contains non-string object key {key!r}"
        )
      canonical[key] = _canonical_json_value(
        item,
        field_name=f"{field_name}.{key}",
        ancestors=active,
      )
    return canonical
  finally:
    active.remove(identity)


def _exact_value_match(left: Any, right: Any) -> bool:
  if type(left) is not type(right):
    return False
  if type(left) is dict:
    if set(left) != set(right):
      return False
    return all(
      _exact_value_match(left[key], right[key])
      for key in left
    )
  if type(left) in {list, tuple}:
    return len(left) == len(right) and all(
      _exact_value_match(left_item, right_item)
      for left_item, right_item in zip(left, right)
    )
  return bool(left == right)


@dataclass(frozen=True, slots=True)
class TopLevelSkillCompletionPlan:
  """Pure exact envelopes plus one bounded idempotent completion effect."""

  result_event: dict[str, Any]
  terminal_event: dict[str, Any]
  effect: TopLevelSkillCompletionEffectPlan

  def __post_init__(self) -> None:
    if type(self.result_event) is not dict:
      raise TypeError("completion plan result_event must be an exact dict")
    if type(self.terminal_event) is not dict:
      raise TypeError(
        "completion plan terminal_event must be an exact dict"
      )
    if not isinstance(
      self.effect,
      TopLevelSkillCompletionEffectPlan,
    ):
      raise TypeError(
        "completion plan effect must be "
        "TopLevelSkillCompletionEffectPlan"
      )
    canonical_result = _canonical_json_value(
      self.result_event,
      field_name="completion_plan.result_event",
    )
    canonical_terminal = _canonical_json_value(
      self.terminal_event,
      field_name="completion_plan.terminal_event",
    )
    object.__setattr__(
      self,
      "result_event",
      canonical_result,
    )
    object.__setattr__(
      self,
      "terminal_event",
      canonical_terminal,
    )


@dataclass(frozen=True, slots=True)
class TopLevelSkillLifecycleMetadata:
  """Immutable server-owned identity for one fresh top-level named-skill run."""

  skill_run_id: str
  skill: str
  scope: SkillLifecycleScope
  ticker: str | None = None
  portfolio_id: str | None = None

  def __post_init__(self) -> None:
    _require_exact_nonempty_string(
      self.skill_run_id,
      field_name="skill_run_id",
    )
    _require_exact_nonempty_string(self.skill, field_name="skill")
    if self.scope not in ("ticker", "portfolio"):
      raise ValueError("scope must be exactly 'ticker' or 'portfolio'")
    _require_optional_exact_string(self.ticker, field_name="ticker")
    _require_optional_exact_string(
      self.portfolio_id,
      field_name="portfolio_id",
    )
    if self.ticker is not None and self.portfolio_id is not None:
      raise ValueError("ticker and portfolio_id are mutually exclusive")
    if self.scope == "ticker":
      if self.ticker is None:
        raise ValueError("ticker scope requires ticker")
      if self.portfolio_id is not None:
        raise ValueError("ticker scope cannot carry portfolio_id")
    if self.scope == "portfolio" and self.ticker is not None:
      raise ValueError("portfolio scope cannot carry ticker")

  def identity_fields(self) -> dict[str, Any]:
    return {
      "skill_run_id": self.skill_run_id,
      "skill": self.skill,
      "scope": self.scope,
      "ticker": self.ticker,
      "portfolio_id": self.portfolio_id,
    }

  def require_event_identity(
    self,
    event: Mapping[str, Any],
    *,
    event_type: str,
  ) -> None:
    if event.get("type") != event_type:
      raise RuntimeError(
        f"Top-level skill lifecycle expected {event_type}, "
        f"got {event.get('type')!r}"
      )
    for field_name, expected in self.identity_fields().items():
      if field_name not in event:
        raise RuntimeError(
          f"{event_type} is missing lifecycle identity field "
          f"{field_name}"
        )
      actual = event[field_name]
      if actual != expected or type(actual) is not type(expected):
        raise RuntimeError(
          f"{event_type} lifecycle identity mismatch for {field_name}: "
          f"expected {expected!r}, got {actual!r}"
        )

  def normalize_result_event(
    self,
    event: Mapping[str, Any],
  ) -> dict[str, Any]:
    event_fields = set(event)
    missing = SKILL_RESULT_CORE_FIELDS - event_fields
    if missing:
      raise RuntimeError(
        "skill_result_captured is missing required fields: "
        + ", ".join(sorted(missing))
      )
    unexpected = event_fields - SKILL_RESULT_CORE_FIELDS
    if unexpected:
      raise RuntimeError(
        "skill_result_captured has unexpected fields: "
        + ", ".join(sorted(unexpected))
      )
    self.require_event_identity(
      event,
      event_type="skill_result_captured",
    )
    if type(event["exit_code"]) is not int or event["exit_code"] < 0:
      raise RuntimeError(
        "skill_result_captured.exit_code must be a non-negative integer"
      )
    _require_string_value(
      event["outcome"],
      field_name="skill_result_captured.outcome",
    )
    _require_string_value(
      event["status"],
      field_name="skill_result_captured.status",
    )
    for field_name in (
      "gate_code",
      "output_memory_file",
      "error",
      "approval_outcome",
      "approval_id",
      "approval_tool_name",
    ):
      _require_optional_string_value(
        event[field_name],
        field_name=f"skill_result_captured.{field_name}",
      )
    for field_name in (
      "artifact_refs",
      "proposal_ids",
      "warnings",
    ):
      _require_string_list(
        event[field_name],
        field_name=f"skill_result_captured.{field_name}",
      )
    for field_name in ("fms_results", "artifact_events"):
      _require_mapping_list(
        event[field_name],
        field_name=f"skill_result_captured.{field_name}",
      )
    exit_code = event["exit_code"]
    outcome = event["outcome"]
    status = event["status"].strip().lower()
    error = event["error"]
    if outcome == "success":
      if exit_code != 0:
        raise RuntimeError(
          "skill_result_captured success requires exit_code 0"
        )
      if error is not None:
        raise RuntimeError(
          "skill_result_captured success cannot carry an error"
        )
    if status in _FAILURE_RESULT_STATUSES and exit_code == 0:
      raise RuntimeError(
        "skill_result_captured failure status requires nonzero exit_code"
      )
    if outcome in {"error", "post_run_guard_failed"} and (
      exit_code == 0
      or not isinstance(error, str)
      or not error.strip()
    ):
      raise RuntimeError(
        "skill_result_captured error outcome requires nonzero exit_code "
        "and a non-empty error"
      )
    unrecoverable_failed_fms_statuses = [
      str(item.get("status") or "").strip().lower()
      for item in event["fms_results"]
      if str(item.get("status") or "").strip().lower()
      in _FAILURE_RESULT_STATUSES
      and not (
        isinstance(item.get("error"), Mapping)
        and item["error"].get("recoverable") is True
      )
    ]
    if unrecoverable_failed_fms_statuses and exit_code == 0:
      raise RuntimeError(
        "skill_result_captured unrecoverable FMS failure requires "
        "nonzero exit_code"
      )
    verdict_echo = event["verdict_echo"]
    if verdict_echo is not None and type(verdict_echo) is not dict:
      raise RuntimeError(
        "skill_result_captured.verdict_echo must be a mapping"
      )
    for field_name in ("cost_usd", "duration_s"):
      _require_nonnegative_number(
        event[field_name],
        field_name=f"skill_result_captured.{field_name}",
        optional=True,
      )
    compaction_count = event["compaction_count"]
    if (
      type(compaction_count) is not int
      or compaction_count < 0
    ):
      raise RuntimeError(
        "skill_result_captured.compaction_count must be a "
        "non-negative integer"
      )
    canonical = _canonical_json_value(
      dict(event),
      field_name="skill_result_captured",
    )
    if type(canonical) is not dict:
      raise RuntimeError(
        "skill_result_captured must normalize to an object"
      )
    return canonical


@dataclass(frozen=True, slots=True)
class TopLevelSkillResultPolicy:
  """Single-use completion policy owned by one top-level named-skill run."""

  prepare_provider: TopLevelSkillProviderPreparer
  prepare_completion: TopLevelSkillCompletionPreparer
  terminal_tool_result_observer: TopLevelSkillTerminalToolObserver | None = None
  _completion_task: asyncio.Task[TopLevelSkillCompletionPlan] | None = field(
    init=False,
    default=None,
    repr=False,
    compare=False,
  )
  _completion_event_log: Any = field(
    init=False,
    default=None,
    repr=False,
    compare=False,
  )
  _completion_terminal_event: dict[str, Any] | None = field(
    init=False,
    default=None,
    repr=False,
    compare=False,
  )
  _provider_task: asyncio.Task[Any] | None = field(
    init=False,
    default=None,
    repr=False,
    compare=False,
  )
  _provider_input: Any = field(
    init=False,
    default=None,
    repr=False,
    compare=False,
  )
  _server_terminal_cause: TopLevelServerTerminalCause | None = field(
    init=False,
    default=None,
    repr=False,
    compare=False,
  )
  _prepared_terminal_event: dict[str, Any] | None = field(
    init=False,
    default=None,
    repr=False,
    compare=False,
  )
  def __post_init__(self) -> None:
    if not callable(self.prepare_provider):
      raise TypeError("prepare_provider must be callable")
    if not callable(self.prepare_completion):
      raise TypeError("prepare_completion must be callable")
    if (
      self.terminal_tool_result_observer is not None
      and not callable(self.terminal_tool_result_observer)
    ):
      raise TypeError("terminal_tool_result_observer must be callable")

  async def prepare_system_prompt(self, proposed_prompt: Any) -> Any:
    """Run admitted pre-provider preparation exactly once."""

    task = self._provider_task
    if task is None:
      object.__setattr__(
        self,
        "_provider_input",
        deepcopy(proposed_prompt),
      )

      async def _prepare_provider_once() -> Any:
        prepared = self.prepare_provider(deepcopy(proposed_prompt))
        if inspect.isawaitable(prepared):
          prepared = await prepared
        return deepcopy(prepared)

      task = asyncio.create_task(
        _prepare_provider_once(),
        name="top-level-skill:prepare-provider",
      )
      object.__setattr__(self, "_provider_task", task)
    elif not _exact_value_match(
      proposed_prompt,
      self._provider_input,
    ):
      raise RuntimeError(
        "Top-level provider preparation was reused with another prompt"
      )
    try:
      prepared = await asyncio.shield(task)
    except asyncio.CancelledError:
      await drain_owned_lifecycle_task(task)
      raise
    return deepcopy(prepared)

  def set_server_terminal_cause(
    self,
    cause: TopLevelServerTerminalCause,
  ) -> bool:
    """Set the typed server cause before completion preparation begins."""

    if cause not in (
      "caller_cancellation",
      "shutdown",
      "timeout",
    ):
      raise ValueError(f"Unsupported server terminal cause: {cause!r}")
    if self._completion_task is not None:
      return self._server_terminal_cause == cause
    current = self._server_terminal_cause
    if current is None:
      object.__setattr__(self, "_server_terminal_cause", cause)
    return self._server_terminal_cause == cause

  def _effective_terminal_event(
    self,
    proposed: dict[str, Any],
  ) -> dict[str, Any]:
    cause = self._server_terminal_cause
    if cause is None:
      return deepcopy(proposed)
    effective = {
      "type": "stream_complete",
      "terminal_disposition": "interrupted",
      "reason": cause,
      "server_terminal_cause": cause,
    }
    usage = proposed.get("usage")
    if isinstance(usage, dict):
      effective["usage"] = deepcopy(usage)
    return effective

  @property
  def prepared_terminal_event(self) -> dict[str, Any] | None:
    return deepcopy(self._prepared_terminal_event)

  @property
  def server_terminal_cause(
    self,
  ) -> TopLevelServerTerminalCause | None:
    return self._server_terminal_cause

  async def prepare(
    self,
    event_log: Any,
    terminal_event: Mapping[str, Any],
  ) -> TopLevelSkillCompletionPlan:
    """Purely prepare exact envelopes and one recoverable effect plan."""

    if type(terminal_event) is not dict:
      raise TypeError("terminal_event must be an exact dict")
    terminal_snapshot = deepcopy(terminal_event)
    task = self._completion_task
    if task is None:
      object.__setattr__(self, "_completion_event_log", event_log)
      object.__setattr__(
        self,
        "_completion_terminal_event",
        terminal_snapshot,
      )

      async def _prepare_once() -> TopLevelSkillCompletionPlan:
        effective_terminal_event = self._effective_terminal_event(
          terminal_snapshot
        )
        object.__setattr__(
          self,
          "_prepared_terminal_event",
          deepcopy(effective_terminal_event),
        )
        prepared = self.prepare_completion(
          event_log,
          effective_terminal_event,
        )
        if inspect.isawaitable(prepared):
          prepared = await prepared
        if not isinstance(prepared, TopLevelSkillCompletionPlan):
          raise TypeError(
            "prepare_completion must return TopLevelSkillCompletionPlan"
          )
        if not _exact_value_match(
          prepared.terminal_event,
          effective_terminal_event,
        ):
          raise RuntimeError(
            "completion plan terminal does not match the exact "
            "effective terminal proposal"
          )
        return deepcopy(prepared)

      task = asyncio.create_task(
        _prepare_once(),
        name="top-level-skill:prepare-completion",
      )
      object.__setattr__(self, "_completion_task", task)
    else:
      if event_log is not self._completion_event_log:
        raise RuntimeError(
          "Top-level completion policy was reused with another EventLog"
        )
      if not _exact_value_match(
        terminal_snapshot,
        self._completion_terminal_event,
      ):
        raise RuntimeError(
          "Top-level completion policy was reused with another terminal "
          "proposal"
        )
    try:
      prepared = await asyncio.shield(task)
    except asyncio.CancelledError:
      prepared = await drain_owned_lifecycle_task(task)
    return deepcopy(prepared)

  @property
  def prepared_plan(self) -> TopLevelSkillCompletionPlan | None:
    task = self._completion_task
    if task is None or not task.done() or task.cancelled():
      return None
    try:
      plan = task.result()
    except BaseException:
      return None
    return deepcopy(plan)


async def drain_owned_lifecycle_task(
  task: "Any",
) -> Any:
  """Drain a lifecycle task despite repeated cancellation of its waiter."""

  import asyncio

  while True:
    try:
      return await asyncio.shield(task)
    except asyncio.CancelledError:
      if task.done():
        return task.result()


__all__ = [
  "SemanticSkillScope",
  "SKILL_RESULT_CORE_FIELDS",
  "SkillLifecycleArtifactIdentity",
  "SkillLifecycleScope",
  "TopLevelSkillLifecycleMetadata",
  "TopLevelSkillAdmission",
  "TopLevelSkillAdmissionState",
  "TopLevelSkillCompletionPlan",
  "TopLevelSkillCompletionPreparer",
  "TopLevelSkillResultPolicy",
  "TopLevelSkillProviderPreparer",
  "TopLevelServerTerminalCause",
  "TopLevelSkillTerminalToolObserver",
  "TopLevelSkillTerminalToolDisposition",
  "WriterLeaseAlreadyHeldError",
  "drain_owned_lifecycle_task",
  "resolve_skill_lifecycle_artifact_identity",
]
