from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
import stat
from typing import Any, Callable, Literal, Mapping

from .canonical_json_target_lock import (
  LockedCanonicalJsonTarget,
  lock_canonical_json_target,
  read_locked_canonical_json_target,
  write_locked_canonical_json_target,
)

WAL_SCHEMA_VERSION = 1
EFFECT_SCHEMA_VERSION = 1
MAX_AFTER_IMAGE_BYTES = 4 * 1024 * 1024
MAX_WAL_BYTES = MAX_AFTER_IMAGE_BYTES + 512 * 1024
_WAL_DIRECTORY_MODE = 0o700
_WAL_FILE_MODE = 0o600
_WAL_FILE_NAME = "completion.json"
_WAL_TEMP_PREFIX = ".completion."
_WAL_TEMP_SUFFIX = ".tmp"
_MISSING_DIGEST_PREFIX = b"missing\x00"
_PRESENT_DIGEST_PREFIX = b"present\x00"

CompletionWalRecordType = Literal["intent", "settled", "ambiguous"]
CompletionEffectKind = Literal[
  "noop_v1",
  "canonical_json_replace_v1",
]


class SkillCompletionWalError(RuntimeError):
  """Base error for private completion-WAL integrity failures."""


class SkillCompletionWalCorruptError(SkillCompletionWalError):
  """The private WAL exists but cannot be trusted."""


class SkillCompletionEffectConflict(SkillCompletionWalError):
  """The effect target matches neither the before nor after image."""


def _canonical_json_value(
  value: Any,
  *,
  field_name: str,
  ancestors: set[int] | None = None,
) -> Any:
  if value is None or type(value) in {str, bool, int}:
    return value
  if type(value) is float:
    if not math.isfinite(value):
      raise ValueError(f"{field_name} must be finite")
    return value
  if type(value) not in {list, dict}:
    raise TypeError(
      f"{field_name} contains unsupported value "
      f"{type(value).__name__}"
    )
  active = ancestors if ancestors is not None else set()
  identity = id(value)
  if identity in active:
    raise ValueError(f"{field_name} contains a cycle")
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
        raise TypeError(
          f"{field_name} contains non-string key {key!r}"
        )
      canonical[key] = _canonical_json_value(
        item,
        field_name=f"{field_name}.{key}",
        ancestors=active,
      )
    return canonical
  finally:
    active.remove(identity)


def canonical_json_bytes(value: Any) -> bytes:
  canonical = _canonical_json_value(
    value,
    field_name="canonical_json",
  )
  return json.dumps(
    canonical,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")


def semantic_json_digest(
  value: Any | None,
  *,
  exists: bool,
) -> str:
  prefix = _PRESENT_DIGEST_PREFIX if exists else _MISSING_DIGEST_PREFIX
  payload = canonical_json_bytes(value) if exists else b""
  return f"sha256:{hashlib.sha256(prefix + payload).hexdigest()}"


def _require_sha256(value: Any, *, field_name: str) -> str:
  if (
    type(value) is not str
    or len(value) != 71
    or not value.startswith("sha256:")
  ):
    raise ValueError(f"{field_name} must be a sha256 digest")
  try:
    bytes.fromhex(value[7:])
  except ValueError as exc:
    raise ValueError(
      f"{field_name} must be a sha256 digest"
    ) from exc
  return value


def _normalize_relative_target(value: str) -> str:
  if type(value) is not str or not value:
    raise ValueError("effect target must be a non-empty relative path")
  path = PurePosixPath(value)
  if path.is_absolute() or any(
    part in {"", ".", ".."} for part in path.parts
  ):
    raise ValueError(
      "effect target must be a normalized workspace-relative path"
    )
  normalized = path.as_posix()
  if normalized != value:
    raise ValueError("effect target must already be normalized")
  return normalized


def _require_owned_directory_stat(
  info: os.stat_result,
  *,
  path_label: str,
  exact_mode: int | None = None,
) -> None:
  if not stat.S_ISDIR(info.st_mode):
    raise SkillCompletionWalError(
      f"{path_label} is not a directory"
    )
  if info.st_uid != os.geteuid():
    raise SkillCompletionWalError(
      f"{path_label} is not owned by the service user"
    )
  if info.st_nlink < 1:
    raise SkillCompletionWalError(
      f"{path_label} has an invalid link count"
    )
  mode = stat.S_IMODE(info.st_mode)
  if exact_mode is not None and mode != exact_mode:
    raise SkillCompletionWalError(
      f"{path_label} must have mode {exact_mode:04o}, got {mode:04o}"
    )
  if exact_mode is None and mode & 0o022:
    raise SkillCompletionWalError(
      f"{path_label} is group/world writable"
    )


def _require_owned_regular_stat(
  info: os.stat_result,
  *,
  path_label: str,
  exact_mode: int | None = None,
) -> None:
  if not stat.S_ISREG(info.st_mode):
    raise SkillCompletionWalError(
      f"{path_label} is not a regular file"
    )
  if info.st_uid != os.geteuid():
    raise SkillCompletionWalError(
      f"{path_label} is not owned by the service user"
    )
  if info.st_nlink != 1:
    raise SkillCompletionWalError(
      f"{path_label} must have exactly one hard link"
    )
  mode = stat.S_IMODE(info.st_mode)
  if exact_mode is not None and mode != exact_mode:
    raise SkillCompletionWalError(
      f"{path_label} must have mode {exact_mode:04o}, got {mode:04o}"
    )
  if exact_mode is None and mode & 0o022:
    raise SkillCompletionWalError(
      f"{path_label} is group/world writable"
    )


def _open_directory(
  path: Path,
  *,
  exact_mode: int | None = None,
) -> int:
  flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
  flags |= getattr(os, "O_CLOEXEC", 0)
  flags |= getattr(os, "O_NOFOLLOW", 0)
  fd = os.open(path, flags)
  try:
    _require_owned_directory_stat(
      os.fstat(fd),
      path_label=str(path),
      exact_mode=exact_mode,
    )
  except BaseException:
    os.close(fd)
    raise
  return fd


def _open_child_directory(
  parent_fd: int,
  name: str,
  *,
  exact_mode: int | None = None,
) -> int:
  flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
  flags |= getattr(os, "O_CLOEXEC", 0)
  flags |= getattr(os, "O_NOFOLLOW", 0)
  fd = os.open(name, flags, dir_fd=parent_fd)
  try:
    _require_owned_directory_stat(
      os.fstat(fd),
      path_label=name,
      exact_mode=exact_mode,
    )
  except BaseException:
    os.close(fd)
    raise
  return fd


def _write_all(fd: int, payload: bytes) -> None:
  offset = 0
  while offset < len(payload):
    written = os.write(fd, payload[offset:])
    if written <= 0:
      raise OSError("short write while persisting completion state")
    offset += written


def _read_all(fd: int, *, max_bytes: int, path_label: str) -> bytes:
  chunks: list[bytes] = []
  total = 0
  while True:
    chunk = os.read(fd, min(64 * 1024, max_bytes + 1 - total))
    if not chunk:
      return b"".join(chunks)
    chunks.append(chunk)
    total += len(chunk)
    if total > max_bytes:
      raise SkillCompletionWalCorruptError(
        f"{path_label} exceeds its bounded size"
      )


@dataclass(frozen=True, slots=True)
class TopLevelSkillCompletionEffectPlan:
  kind: CompletionEffectKind
  workspace_path: Path | None
  workspace_identity: tuple[int, int, int] | None
  target: str | None
  before_digest: str
  after_digest: str
  after_image: Any | None

  @classmethod
  def noop(cls) -> "TopLevelSkillCompletionEffectPlan":
    digest = semantic_json_digest(None, exists=False)
    return cls(
      kind="noop_v1",
      workspace_path=None,
      workspace_identity=None,
      target=None,
      before_digest=digest,
      after_digest=digest,
      after_image=None,
    )

  @classmethod
  def from_durable_payload(
    cls,
    payload: Mapping[str, Any],
    *,
    expected_workspace: str | Path | None,
  ) -> "TopLevelSkillCompletionEffectPlan":
    canonical = validate_effect_payload(payload)
    if canonical["kind"] == "noop_v1":
      return cls.noop()
    if expected_workspace is None:
      raise SkillCompletionWalCorruptError(
        "completion effect requires an exact recovery workspace"
      )
    workspace = Path(expected_workspace).expanduser()
    if not workspace.is_absolute():
      workspace = Path(os.path.abspath(workspace))
    identity = canonical["workspace_identity"]
    plan = cls(
      kind="canonical_json_replace_v1",
      workspace_path=workspace,
      workspace_identity=(
        identity["dev"],
        identity["ino"],
        identity["uid"],
      ),
      target=canonical["target"],
      before_digest=canonical["before_digest"],
      after_digest=canonical["after_digest"],
      after_image=canonical["after_image"],
    )
    if plan.durable_payload() != canonical:
      raise SkillCompletionWalCorruptError(
        "completion effect recovery plan is not exact"
      )
    return plan

  @classmethod
  def canonical_json_update(
    cls,
    *,
    workspace_path: str | Path,
    target_path: str | Path,
    update: Callable[[bool, Any | None], Any],
  ) -> "TopLevelSkillCompletionEffectPlan":
    if not callable(update):
      raise TypeError("completion effect update must be callable")
    workspace = Path(workspace_path).expanduser()
    if not workspace.is_absolute():
      workspace = Path(os.path.abspath(workspace))
    target_path_obj = Path(target_path).expanduser()
    if not target_path_obj.is_absolute():
      target_path_obj = workspace / target_path_obj
    try:
      relative = target_path_obj.relative_to(workspace).as_posix()
    except ValueError as exc:
      raise ValueError(
        "effect target must be inside the exact workspace"
      ) from exc
    relative = _normalize_relative_target(relative)
    workspace_fd = _open_directory(workspace)
    try:
      workspace_stat = os.fstat(workspace_fd)
      parent_fd, target_name = _open_effect_parent(
        workspace_fd,
        relative,
      )
      try:
        with lock_canonical_json_target(
          parent_fd,
          target_name,
        ) as locked_target:
          (
            before_exists,
            canonical_before,
            before_digest,
          ) = _read_effect_target(locked_target)
          canonical_after = _canonical_json_value(
            update(before_exists, canonical_before),
            field_name="after_image",
          )
          if (
            len(canonical_json_bytes(canonical_after))
            > MAX_AFTER_IMAGE_BYTES
          ):
            raise ValueError(
              "completion effect after image exceeds 4 MiB"
            )
      finally:
        os.close(parent_fd)
    finally:
      os.close(workspace_fd)
    return cls(
      kind="canonical_json_replace_v1",
      workspace_path=workspace,
      workspace_identity=(
        workspace_stat.st_dev,
        workspace_stat.st_ino,
        workspace_stat.st_uid,
      ),
      target=relative,
      before_digest=before_digest,
      after_digest=semantic_json_digest(
        canonical_after,
        exists=True,
      ),
      after_image=canonical_after,
    )

  def durable_payload(self) -> dict[str, Any]:
    payload: dict[str, Any] = {
      "schema_version": EFFECT_SCHEMA_VERSION,
      "kind": self.kind,
      "before_digest": _require_sha256(
        self.before_digest,
        field_name="before_digest",
      ),
      "after_digest": _require_sha256(
        self.after_digest,
        field_name="after_digest",
      ),
      "target": self.target,
      "workspace_identity": None,
      "after_image": None,
    }
    if self.kind == "noop_v1":
      missing_digest = semantic_json_digest(None, exists=False)
      if (
        self.workspace_path is not None
        or self.workspace_identity is not None
        or self.target is not None
        or self.after_image is not None
        or self.before_digest != missing_digest
        or self.after_digest != missing_digest
      ):
        raise ValueError("noop effect plan carries mutable state")
      return payload
    if self.kind != "canonical_json_replace_v1":
      raise ValueError(f"Unsupported effect kind: {self.kind!r}")
    if (
      self.workspace_path is None
      or self.workspace_identity is None
      or self.target is None
    ):
      raise ValueError(
        "canonical JSON effect plan is incomplete"
      )
    _normalize_relative_target(self.target)
    after_bytes = canonical_json_bytes(self.after_image)
    if len(after_bytes) > MAX_AFTER_IMAGE_BYTES:
      raise ValueError(
        "completion effect after image exceeds 4 MiB"
      )
    if semantic_json_digest(
      self.after_image,
      exists=True,
    ) != self.after_digest:
      raise ValueError("completion effect after digest is inconsistent")
    payload["workspace_identity"] = {
      "dev": self.workspace_identity[0],
      "ino": self.workspace_identity[1],
      "uid": self.workspace_identity[2],
    }
    payload["after_image"] = self.after_image
    return payload


def validate_effect_payload(payload: Any) -> dict[str, Any]:
  if type(payload) is not dict:
    raise SkillCompletionWalCorruptError(
      "completion effect payload must be an object"
    )
  expected_fields = {
    "schema_version",
    "kind",
    "before_digest",
    "after_digest",
    "target",
    "workspace_identity",
    "after_image",
  }
  if set(payload) != expected_fields:
    raise SkillCompletionWalCorruptError(
      "completion effect payload fields are invalid"
    )
  if (
    type(payload["schema_version"]) is not int
    or payload["schema_version"] != EFFECT_SCHEMA_VERSION
  ):
    raise SkillCompletionWalCorruptError(
      "unsupported completion effect schema version"
    )
  kind = payload["kind"]
  if kind not in {"noop_v1", "canonical_json_replace_v1"}:
    raise SkillCompletionWalCorruptError(
      "unsupported completion effect kind"
    )
  try:
    before_digest = _require_sha256(
      payload["before_digest"],
      field_name="before_digest",
    )
    after_digest = _require_sha256(
      payload["after_digest"],
      field_name="after_digest",
    )
  except ValueError as exc:
    raise SkillCompletionWalCorruptError(str(exc)) from exc
  if kind == "noop_v1":
    missing_digest = semantic_json_digest(None, exists=False)
    if (
      payload["target"] is not None
      or payload["workspace_identity"] is not None
      or payload["after_image"] is not None
      or before_digest != missing_digest
      or after_digest != missing_digest
    ):
      raise SkillCompletionWalCorruptError(
        "noop completion effect is inconsistent"
      )
  else:
    try:
      _normalize_relative_target(payload["target"])
    except (TypeError, ValueError) as exc:
      raise SkillCompletionWalCorruptError(str(exc)) from exc
    identity = payload["workspace_identity"]
    if (
      type(identity) is not dict
      or set(identity) != {"dev", "ino", "uid"}
      or any(type(identity[key]) is not int for key in identity)
    ):
      raise SkillCompletionWalCorruptError(
        "completion effect workspace identity is invalid"
      )
    try:
      canonical_after = _canonical_json_value(
        payload["after_image"],
        field_name="after_image",
      )
    except (TypeError, ValueError) as exc:
      raise SkillCompletionWalCorruptError(str(exc)) from exc
    if len(canonical_json_bytes(canonical_after)) > MAX_AFTER_IMAGE_BYTES:
      raise SkillCompletionWalCorruptError(
        "completion effect after image exceeds 4 MiB"
      )
    if semantic_json_digest(
      canonical_after,
      exists=True,
    ) != after_digest:
      raise SkillCompletionWalCorruptError(
        "completion effect after digest mismatch"
      )
  return _canonical_json_value(
    payload,
    field_name="effect",
  )


class SkillCompletionWal:
  """Strict private single-record WAL adjacent to one session log."""

  def __init__(self, session_log_path: str | Path) -> None:
    log_path = Path(session_log_path).expanduser()
    if not log_path.is_absolute():
      log_path = Path(os.path.abspath(log_path))
    self.session_log_path = log_path
    self.parent_path = log_path.parent
    self.directory_name = f".{log_path.name}.skill_completion"
    self.directory_path = self.parent_path / self.directory_name

  def _open_private_directory(self, *, create: bool) -> tuple[int, int]:
    parent_fd = _open_directory(self.parent_path)
    try:
      if create:
        try:
          os.mkdir(
            self.directory_name,
            _WAL_DIRECTORY_MODE,
            dir_fd=parent_fd,
          )
        except FileExistsError:
          pass
        else:
          os.fsync(parent_fd)
      directory_fd = _open_child_directory(
        parent_fd,
        self.directory_name,
        exact_mode=_WAL_DIRECTORY_MODE,
      )
      return parent_fd, directory_fd
    except BaseException:
      os.close(parent_fd)
      raise

  @staticmethod
  def _reject_temp_remnants(directory_fd: int) -> None:
    for name in os.listdir(directory_fd):
      if (
        name.startswith(_WAL_TEMP_PREFIX)
        and name.endswith(_WAL_TEMP_SUFFIX)
      ):
        raise SkillCompletionWalCorruptError(
          "completion WAL has a crash-remnant temp file"
        )

  def load(self) -> dict[str, Any] | None:
    try:
      self.directory_path.lstat()
    except FileNotFoundError:
      return None
    parent_fd, directory_fd = self._open_private_directory(
      create=False
    )
    try:
      self._reject_temp_remnants(directory_fd)
      flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
      flags |= getattr(os, "O_NOFOLLOW", 0)
      try:
        fd = os.open(
          _WAL_FILE_NAME,
          flags,
          dir_fd=directory_fd,
        )
      except FileNotFoundError:
        return None
      try:
        _require_owned_regular_stat(
          os.fstat(fd),
          path_label=str(self.directory_path / _WAL_FILE_NAME),
          exact_mode=_WAL_FILE_MODE,
        )
        raw = _read_all(
          fd,
          max_bytes=MAX_WAL_BYTES,
          path_label="completion WAL",
        )
      finally:
        os.close(fd)
    finally:
      os.close(directory_fd)
      os.close(parent_fd)
    if not raw:
      raise SkillCompletionWalCorruptError(
        "completion WAL is empty"
      )
    try:
      record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise SkillCompletionWalCorruptError(
        "completion WAL is unreadable"
      ) from exc
    return validate_wal_record(record)

  def store(self, record: Mapping[str, Any]) -> dict[str, Any]:
    canonical = build_wal_record(record)
    payload = canonical_json_bytes(canonical) + b"\n"
    if len(payload) > MAX_WAL_BYTES:
      raise ValueError("completion WAL record exceeds bounded size")
    parent_fd, directory_fd = self._open_private_directory(
      create=True
    )
    temp_name = (
      f"{_WAL_TEMP_PREFIX}{secrets.token_hex(16)}"
      f"{_WAL_TEMP_SUFFIX}"
    )
    temp_fd: int | None = None
    try:
      self._reject_temp_remnants(directory_fd)
      try:
        self._read_record_from_open_directory(directory_fd)
      except FileNotFoundError:
        pass
      flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
      flags |= getattr(os, "O_CLOEXEC", 0)
      flags |= getattr(os, "O_NOFOLLOW", 0)
      temp_fd = os.open(
        temp_name,
        flags,
        _WAL_FILE_MODE,
        dir_fd=directory_fd,
      )
      os.fchmod(temp_fd, _WAL_FILE_MODE)
      _require_owned_regular_stat(
        os.fstat(temp_fd),
        path_label=temp_name,
        exact_mode=_WAL_FILE_MODE,
      )
      _write_all(temp_fd, payload)
      os.fsync(temp_fd)
      os.close(temp_fd)
      temp_fd = None
      os.rename(
        temp_name,
        _WAL_FILE_NAME,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
      )
      os.fsync(directory_fd)
      readback = self._read_record_from_open_directory(
        directory_fd
      )
      if readback != canonical:
        raise SkillCompletionWalCorruptError(
          "completion WAL readback mismatch"
        )
      return readback
    except BaseException:
      if temp_fd is not None:
        os.close(temp_fd)
      raise
    finally:
      os.close(directory_fd)
      os.close(parent_fd)

  def _read_record_from_open_directory(
    self,
    directory_fd: int,
  ) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(
      _WAL_FILE_NAME,
      flags,
      dir_fd=directory_fd,
    )
    try:
      _require_owned_regular_stat(
        os.fstat(fd),
        path_label=_WAL_FILE_NAME,
        exact_mode=_WAL_FILE_MODE,
      )
      raw = _read_all(
        fd,
        max_bytes=MAX_WAL_BYTES,
        path_label="completion WAL",
      )
    finally:
      os.close(fd)
    try:
      record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
      raise SkillCompletionWalCorruptError(
        "completion WAL readback is unreadable"
      ) from exc
    return validate_wal_record(record)


def build_wal_record(record: Mapping[str, Any]) -> dict[str, Any]:
  body = _canonical_json_value(
    dict(record),
    field_name="wal_record",
  )
  body.pop("checksum", None)
  body["schema_version"] = WAL_SCHEMA_VERSION
  checksum = f"sha256:{hashlib.sha256(canonical_json_bytes(body)).hexdigest()}"
  return {**body, "checksum": checksum}


def validate_wal_record(record: Any) -> dict[str, Any]:
  if type(record) is not dict:
    raise SkillCompletionWalCorruptError(
      "completion WAL record must be an object"
    )
  checksum = record.get("checksum")
  try:
    _require_sha256(checksum, field_name="checksum")
  except ValueError as exc:
    raise SkillCompletionWalCorruptError(str(exc)) from exc
  body = dict(record)
  body.pop("checksum", None)
  if (
    type(body.get("schema_version")) is not int
    or body["schema_version"] != WAL_SCHEMA_VERSION
  ):
    raise SkillCompletionWalCorruptError(
      "unsupported completion WAL schema version"
    )
  expected = f"sha256:{hashlib.sha256(canonical_json_bytes(body)).hexdigest()}"
  if checksum != expected:
    raise SkillCompletionWalCorruptError(
      "completion WAL checksum mismatch"
    )
  record_type = body.get("record_type")
  if record_type not in {"intent", "settled", "ambiguous"}:
    raise SkillCompletionWalCorruptError(
      "completion WAL record type is invalid"
    )
  fence = body.get("fence")
  if (
    type(fence) is not dict
    or set(fence) != {"generation", "owner_token"}
    or type(fence["generation"]) is not int
    or fence["generation"] <= 0
    or type(fence["owner_token"]) is not str
    or len(fence["owner_token"]) != 64
  ):
    raise SkillCompletionWalCorruptError(
      "completion WAL fence is invalid"
    )
  try:
    bytes.fromhex(fence["owner_token"])
  except ValueError as exc:
    raise SkillCompletionWalCorruptError(
      "completion WAL fence owner_token is invalid"
    ) from exc
  if record_type == "intent":
    expected_fields = {
      "schema_version",
      "record_type",
      "skill_run_id",
      "lifecycle",
      "result",
      "terminal",
      "effect",
      "fence",
    }
    if set(body) != expected_fields:
      raise SkillCompletionWalCorruptError(
        "completion intent fields are invalid"
      )
    if (
      type(body["skill_run_id"]) is not str
      or not body["skill_run_id"]
      or type(body["lifecycle"]) is not dict
      or type(body["result"]) is not dict
      or type(body["terminal"]) is not dict
    ):
      raise SkillCompletionWalCorruptError(
        "completion intent envelopes are invalid"
      )
    validate_effect_payload(body["effect"])
  else:
    expected_fields = {
      "schema_version",
      "record_type",
      "skill_run_id",
      "result_digest",
      "terminal_digest",
      "fence",
      "reason",
    }
    if set(body) != expected_fields:
      raise SkillCompletionWalCorruptError(
        f"completion {record_type} fields are invalid"
      )
    if (
      type(body["skill_run_id"]) is not str
      or not body["skill_run_id"]
      or type(body["reason"]) is not str
    ):
      raise SkillCompletionWalCorruptError(
        f"completion {record_type} identity is invalid"
      )
    try:
      _require_sha256(
        body["result_digest"],
        field_name="result_digest",
      )
      _require_sha256(
        body["terminal_digest"],
        field_name="terminal_digest",
      )
    except ValueError as exc:
      raise SkillCompletionWalCorruptError(str(exc)) from exc
  return _canonical_json_value(
    record,
    field_name="wal_record",
  )


def _workspace_from_effect(
  effect: Mapping[str, Any],
  *,
  expected_workspace: str | Path,
) -> tuple[int, dict[str, Any]]:
  canonical = validate_effect_payload(effect)
  if canonical["kind"] == "noop_v1":
    raise ValueError("noop effect has no workspace")
  workspace = Path(expected_workspace).expanduser()
  if not workspace.is_absolute():
    workspace = Path(os.path.abspath(workspace))
  workspace_fd = _open_directory(workspace)
  identity = canonical["workspace_identity"]
  actual = os.fstat(workspace_fd)
  if (
    actual.st_dev != identity["dev"]
    or actual.st_ino != identity["ino"]
    or actual.st_uid != identity["uid"]
  ):
    os.close(workspace_fd)
    raise SkillCompletionEffectConflict(
      "completion effect workspace identity changed"
    )
  return workspace_fd, canonical


def _open_effect_parent(
  workspace_fd: int,
  target: str,
) -> tuple[int, str]:
  parts = PurePosixPath(_normalize_relative_target(target)).parts
  parent_fd = os.dup(workspace_fd)
  try:
    for part in parts[:-1]:
      next_fd = _open_child_directory(parent_fd, part)
      os.close(parent_fd)
      parent_fd = next_fd
    return parent_fd, parts[-1]
  except BaseException:
    os.close(parent_fd)
    raise


def ensure_completion_effect_parent(
  *,
  workspace_path: str | Path,
  target_path: str | Path,
) -> None:
  """Securely create only the directory infrastructure for an effect."""

  workspace = Path(workspace_path).expanduser()
  if not workspace.is_absolute():
    workspace = Path(os.path.abspath(workspace))
  target_path_obj = Path(target_path).expanduser()
  if not target_path_obj.is_absolute():
    target_path_obj = workspace / target_path_obj
  try:
    relative = target_path_obj.relative_to(workspace).as_posix()
  except ValueError as exc:
    raise ValueError(
      "effect target must be inside the exact workspace"
    ) from exc
  parts = PurePosixPath(_normalize_relative_target(relative)).parts
  workspace_fd = _open_directory(workspace)
  parent_fd = os.dup(workspace_fd)
  os.close(workspace_fd)
  try:
    for part in parts[:-1]:
      try:
        next_fd = _open_child_directory(parent_fd, part)
      except FileNotFoundError:
        try:
          os.mkdir(part, _WAL_DIRECTORY_MODE, dir_fd=parent_fd)
        except FileExistsError:
          pass
        else:
          os.fsync(parent_fd)
        next_fd = _open_child_directory(parent_fd, part)
      os.close(parent_fd)
      parent_fd = next_fd
  finally:
    os.close(parent_fd)


def _read_effect_target(
  target: LockedCanonicalJsonTarget,
) -> tuple[bool, Any | None, str]:
  snapshot = read_locked_canonical_json_target(target)
  if not snapshot.exists:
    return False, None, semantic_json_digest(None, exists=False)
  if snapshot.parse_error is not None:
    raise SkillCompletionEffectConflict(
      "completion effect target is not canonical JSON"
    )
  canonical = _canonical_json_value(
    snapshot.value,
    field_name="effect_target",
  )
  return (
    True,
    canonical,
    semantic_json_digest(canonical, exists=True),
  )


def apply_completion_effect(
  effect: Mapping[str, Any],
  *,
  expected_workspace: str | Path | None,
) -> Literal["noop", "already_applied", "applied"]:
  canonical = validate_effect_payload(effect)
  if canonical["kind"] == "noop_v1":
    return "noop"
  if expected_workspace is None:
    raise SkillCompletionEffectConflict(
      "completion effect requires an exact workspace"
    )
  workspace_fd, canonical = _workspace_from_effect(
    canonical,
    expected_workspace=expected_workspace,
  )
  try:
    parent_fd, target_name = _open_effect_parent(
      workspace_fd,
      canonical["target"],
    )
  finally:
    os.close(workspace_fd)
  try:
    with lock_canonical_json_target(
      parent_fd,
      target_name,
    ) as locked_target:
      _exists, _image, current_digest = _read_effect_target(
        locked_target,
      )
      if current_digest == canonical["after_digest"]:
        return "already_applied"
      if current_digest != canonical["before_digest"]:
        raise SkillCompletionEffectConflict(
          "completion effect target matches neither before nor after digest"
        )
      if canonical["before_digest"] == canonical["after_digest"]:
        return "noop"
      write_locked_canonical_json_target(
        locked_target,
        canonical["after_image"],
      )
      _exists, _image, readback_digest = _read_effect_target(
        locked_target,
      )
      if readback_digest != canonical["after_digest"]:
        raise SkillCompletionEffectConflict(
          "completion effect readback digest mismatch"
        )
      return "applied"
  finally:
    os.close(parent_fd)


def envelope_digest(envelope: Mapping[str, Any]) -> str:
  return f"sha256:{hashlib.sha256(canonical_json_bytes(dict(envelope))).hexdigest()}"


__all__ = [
  "MAX_AFTER_IMAGE_BYTES",
  "SkillCompletionEffectConflict",
  "SkillCompletionWal",
  "SkillCompletionWalCorruptError",
  "SkillCompletionWalError",
  "TopLevelSkillCompletionEffectPlan",
  "apply_completion_effect",
  "build_wal_record",
  "canonical_json_bytes",
  "ensure_completion_effect_parent",
  "envelope_digest",
  "semantic_json_digest",
  "validate_effect_payload",
  "validate_wal_record",
]
