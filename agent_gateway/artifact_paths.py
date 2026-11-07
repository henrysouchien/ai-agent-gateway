from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Mapping
from contextlib import closing
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

from agent_workflow_contracts.ticker_contract import normalize_contract_ticker
from .issue_inbox_schema import IssueInboxRowError, StoredIssueV1


# Component length cap (128 chars) is owned by the artifact-store and skill-catalog
# contracts (api/agent/shared/canvas_artifact_store.py:19,
# api/agent/shared/document_artifact_erase.py:25, api/agent/skills/catalog.py:737-738)
# so `<id>.payload.json` / `<id>.docx` stays within the 255-byte POSIX NAME_MAX limit.
MAX_ARTIFACT_COMPONENT_LENGTH = 128
MAX_ARTIFACT_FILENAME_LENGTH = MAX_ARTIFACT_COMPONENT_LENGTH + len(".payload.json")
_SKILL_NAME_RE = re.compile(r"^(?=.{1,128}$)[a-z0-9]+(?:-[a-z0-9]+)*$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
ISSUE_INBOX_DB_ENV = "AGENT_ISSUE_INBOX_DB_PATH"
FRICTIONS_DIR_NAME = "frictions"
_ISSUE_INBOX_STORE_DIRECTORY = ("gateway", "issue-inbox")
_ISSUE_INBOX_DB_FILENAME = "issue-inbox.sqlite3"
_ISSUE_INBOX_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")

log = logging.getLogger(__name__)


class ArtifactPathError(ValueError):
  """Raised when a requested artifact path is not safe to resolve."""


class _LegacyIssueInboxRefused(ValueError):
  """The legacy issue-inbox rows cannot be carried into the new store."""


@dataclass(frozen=True)
class ArtifactPath:
  workspace_root: Path
  path: Path
  ticker: str
  skill: str | None = None
  artifact_id: str | None = None


def artifact_json_path_for_request(
  user_id: str,
  *,
  ticker: str,
  skill: str,
  artifact_id: str,
) -> ArtifactPath:
  normalized_ticker = _validate_ticker(ticker)
  normalized_skill = _validate_skill(skill)
  normalized_artifact_id = _validate_artifact_id(artifact_id)
  workspace_root = user_workspace_root(user_id)
  path = _resolve_under_workspace(
    workspace_root,
    "artifacts",
    normalized_ticker,
    normalized_skill,
    f"{normalized_artifact_id}.json",
  )
  return ArtifactPath(
    workspace_root=workspace_root,
    path=path,
    ticker=normalized_ticker,
    skill=normalized_skill,
    artifact_id=normalized_artifact_id,
  )


def artifact_json_paths_for_request(
  user_id: str,
  *,
  ticker: str,
  skill: str,
) -> list[ArtifactPath]:
  normalized_ticker = _validate_ticker(ticker)
  normalized_skill = _validate_skill(skill)
  workspace_root = user_workspace_root(user_id)
  directory = _resolve_under_workspace(
    workspace_root,
    "artifacts",
    normalized_ticker,
    normalized_skill,
  )
  if not directory.is_dir():
    return []
  return [
    ArtifactPath(
      workspace_root=workspace_root,
      path=path,
      ticker=normalized_ticker,
      skill=normalized_skill,
      artifact_id=path.stem,
    )
    for path in _safe_json_children(directory, workspace_root)
  ]


def ticker_artifact_paths_for_request(user_id: str, *, ticker: str) -> dict[str, list[ArtifactPath]]:
  normalized_ticker = _validate_ticker(ticker)
  workspace_root = user_workspace_root(user_id)
  ticker_dir = _resolve_under_workspace(workspace_root, "artifacts", normalized_ticker)
  if not ticker_dir.is_dir():
    return {}

  by_skill: dict[str, list[ArtifactPath]] = {}
  for skill_dir in sorted(ticker_dir.iterdir(), key=lambda path: path.name):
    if not skill_dir.is_dir():
      continue
    try:
      normalized_skill = _validate_skill(skill_dir.name)
    except ArtifactPathError:
      # Only skill writers create subdirectories here; a foreign-named
      # directory must not hide, or refuse the listing of, valid skills.
      log.warning(
        "artifact ticker directory %s contains a foreign-named subdirectory %r; ignoring it",
        ticker_dir,
        skill_dir.name,
      )
      continue
    safe_skill_dir = _ensure_under_workspace(skill_dir, workspace_root)
    artifacts = _safe_json_children(safe_skill_dir, workspace_root)
    if not artifacts:
      continue
    by_skill[normalized_skill] = [
      ArtifactPath(
        workspace_root=workspace_root,
        path=path,
        ticker=normalized_ticker,
        skill=normalized_skill,
        artifact_id=path.stem,
      )
      for path in artifacts
    ]
  return by_skill


def letter_docx_path_for_request(
  user_id: str,
  *,
  ticker: str,
  artifact_id: str,
) -> ArtifactPath:
  normalized_ticker = _validate_ticker(ticker)
  normalized_artifact_id = _validate_artifact_id(artifact_id)
  workspace_root = user_workspace_root(user_id)
  path = _resolve_under_workspace(
    workspace_root,
    "letters",
    normalized_ticker,
    f"{normalized_artifact_id}.docx",
  )
  return ArtifactPath(
    workspace_root=workspace_root,
    path=path,
    ticker=normalized_ticker,
    artifact_id=normalized_artifact_id,
  )


def reject_unsafe_path(path: str) -> None:
  for component in _split_path_components(path):
    _validate_path_component(component, "path")


def user_desk_root(data_dir: Path, storage_id: str) -> Path:
  """Locate a desk using the storage id already parsed at its entry boundary."""
  return data_dir / "users" / storage_id


def user_friction_queue_dir(data_dir: Path, storage_id: str) -> Path:
  """The directory `report_friction` writes a desk's friction reports to."""
  return user_desk_root(data_dir, storage_id) / "workspace" / FRICTIONS_DIR_NAME


def user_workspace_root(user_id: str) -> Path:
  return (user_desk_root(user_data_dir(), user_id) / "workspace").resolve()


def user_issue_inbox_db_path(user_id: str) -> Path:
  """Resolve durable issue intake within the authenticated user's workspace."""
  root = user_workspace_root(user_id)
  path = _resolve_under_workspace(root, "issue-inbox", _ISSUE_INBOX_DB_FILENAME)
  path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  return path


def existing_user_issue_inbox_db_paths(
  environ: Mapping[str, str] | None = None,
) -> list[Path]:
  """Enumerate per-user issue inboxes that already exist, ordered by user id.

  Read-only counterpart of `user_issue_inbox_db_path` for operator drains: it
  creates no directories and returns only stores a user's session has written.
  An entry that is not a plain user-id directory or whose inbox file resolves
  outside that user's workspace is logged and skipped, never followed.
  """
  data_dir = user_data_dir(environ=environ)
  users_root = data_dir / "users"
  try:
    user_dirs = sorted(users_root.iterdir(), key=lambda child: child.name)
  except FileNotFoundError:
    return []
  from agent_gateway.user_identity_mapping import (
    StorageIdentityFinding,
    parse_storage_identity,
  )

  paths: list[Path] = []
  for user_dir in user_dirs:
    if not user_dir.is_dir():
      continue
    try:

      identity = parse_storage_identity(user_dir.name)
      workspace_root = (user_desk_root(data_dir, identity.storage_id) / "workspace").resolve()
      path = _resolve_under_workspace(
        workspace_root, "issue-inbox", _ISSUE_INBOX_DB_FILENAME,
      )
    except (ArtifactPathError, StorageIdentityFinding) as exc:
      log.warning("issue-inbox enumeration skipped user directory %r: %s", user_dir.name, exc)
      continue
    if path.is_file():
      paths.append(path)
  return paths


def user_error_signals_cursor_path(user_id: str) -> Path:
  """Resolve the user's independent error observation cursor."""
  root = user_workspace_root(user_id)
  return _resolve_under_workspace(root, ".error_signals_cursor.json")


def user_data_dir(
  *, environ: Mapping[str, str] | None = None, home: Path | None = None,
) -> Path:
  """Resolve durable user data independently of any source checkout."""
  env = os.environ if environ is None else environ
  configured = env.get("USER_DATA_DIR", "").strip()
  if configured:
    return Path(configured).expanduser()
  state_home = env.get("XDG_STATE_HOME", "").strip()
  state_root = (
    Path(state_home).expanduser()
    if state_home
    else (Path.home() if home is None else home) / ".local" / "state"
  )
  return state_root / "hank" / "data"


INDUSTRY_ARTIFACTS_DIR_ENV = "INDUSTRY_ARTIFACTS_DIR"


def industry_artifacts_dir(
  *, environ: Mapping[str, str] | None = None, home: Path | None = None,
) -> Path:
  """Runtime industry data root shared by the writer, bootstrap and Risk MCP children.

  ``INDUSTRY_ARTIFACTS_DIR`` is the contract with readers that cannot import the
  gateway (``industry_slice_core``). When it is unset the root derives from
  ``user_data_dir``, so an unset ``USER_DATA_DIR`` names the same place for the
  writer and for the environment bootstrap hands to those readers.
  """
  env = os.environ if environ is None else environ
  configured = env.get(INDUSTRY_ARTIFACTS_DIR_ENV, "").strip()
  if configured:
    return Path(configured).expanduser()
  return user_data_dir(environ=env, home=home) / "industry_artifacts"


def autonomous_log_dir(*, environ: Mapping[str, str] | None = None) -> Path | None:
  """The autonomous run store the gateway writes, or None when none is configured.

  Writer (``server.py`` autonomous storage root) and offline readers
  (``session_archive``) resolve it here, so they cannot name different roots.
  """
  env = os.environ if environ is None else environ
  explicit = env.get("AGENT_GATEWAY_AUTONOMOUS_LOG_DIR", "").strip()
  if explicit:
    return Path(explicit).expanduser()
  gateway_log_dir = env.get("GATEWAY_LOG_DIR", "").strip()
  if gateway_log_dir:
    return Path(gateway_log_dir).expanduser() / "autonomous"
  legacy_agents_log_dir = env.get("AGENTS_MCP_LOG_DIR", "").strip()
  if legacy_agents_log_dir:
    return Path(legacy_agents_log_dir).expanduser()
  return None


def resolve_issue_inbox_db_path(
  environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, dict[str, str] | None]:
  """Locate the durable `log_issue` inbox inside the gateway state root.

  The store keeps a directory of its own. SQLite creates the rollback
  journal beside the database, so whoever writes the inbox needs write
  access to its directory — and the contained autonomous child opens this
  store in-process through the `log_issue` handler. The `gateway/` root it
  sits under also holds the parent-only approval ledger, which the child is
  never admitted to, so the child's containment grants this directory
  rather than that root.
  """
  env = os.environ if environ is None else environ
  explicit = str(env.get(ISSUE_INBOX_DB_ENV, "")).strip()
  if explicit:
    target = Path(explicit).expanduser()
    source = ISSUE_INBOX_DB_ENV
  else:
    user_data = str(env.get("USER_DATA_DIR", "")).strip()
    if not user_data:
      return None, {
        "code": "issue_sink_unavailable",
        "message": (
          f"Set {ISSUE_INBOX_DB_ENV} or USER_DATA_DIR to a persistent path "
          "outside the immutable runtime."
        ),
      }
    target = Path(user_data).expanduser().joinpath(
      *_ISSUE_INBOX_STORE_DIRECTORY,
      _ISSUE_INBOX_DB_FILENAME,
    )
    source = "USER_DATA_DIR"
  if not target.is_absolute():
    return None, {
      "code": "issue_sink_invalid",
      "message": f"{source} must resolve to an absolute issue-inbox path: {target}",
    }
  runtime_value = str(env.get("LOCAL_GATEWAY_RUNTIME_VERSION_ROOT", "")).strip()
  if runtime_value:
    runtime_root = Path(runtime_value).expanduser()
    if runtime_root.is_absolute() and target.resolve().is_relative_to(runtime_root.resolve()):
      return None, {
        "code": "issue_sink_invalid",
        "message": f"Issue inbox must be outside the immutable runtime: {target}",
      }
  parent = target.parent
  try:
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
  except OSError as exc:
    return None, {
      "code": "issue_sink_unavailable",
      "message": f"Issue-inbox directory is unusable: {parent}: {exc}",
    }
  if not os.access(parent, os.W_OK):
    return None, {
      "code": "issue_sink_unavailable",
      "message": f"Issue-inbox parent directory is not writable: {parent}",
    }
  if source == "USER_DATA_DIR":
    adoption_error = _adopt_legacy_issue_inbox_store(target)
    if adoption_error is not None:
      return None, adoption_error
  return target, None


def _adopt_legacy_issue_inbox_store(target: Path) -> dict[str, str] | None:
  """Fold a pre-directory inbox file into the store, exactly once.

  The derived default was `<state>/gateway/issue-inbox.sqlite3` before the
  store took a directory of its own, and rows left there are undelivered
  intake that nothing reads any more. When the new path is still empty,
  adoption is a rename inside the same gateway state root: it keeps the
  inode, so an open handle and the rollback journal beside it stay valid,
  and it cannot half-copy. When a run already created the new store, the
  legacy rows are folded into it by issue id instead — the id is the payload
  digest, so an id already present is the same observation and its live
  status/attempt counters win. Either way the legacy path is left behind
  under a different name, so adoption runs once and cannot double-publish:
  each row's delivery marker travels with the row. A legacy row the store
  rejects is never skipped: the import is refused whole, the legacy path is
  not retired, and the refusal is returned for `resolve_issue_inbox_db_path`.
  """
  legacy = target.parent.parent / target.name
  if not legacy.is_file():
    return None
  if not target.exists():
    _rename_legacy_issue_inbox_store(legacy, target)
    return None
  return _merge_legacy_issue_inbox_store(legacy, target)


def _rename_legacy_issue_inbox_store(legacy: Path, target: Path) -> None:
  adopted = [legacy.name]
  try:
    os.replace(legacy, target)
    for suffix in _ISSUE_INBOX_SIDECAR_SUFFIXES:
      sidecar = legacy.with_name(legacy.name + suffix)
      if sidecar.exists():
        os.replace(sidecar, target.with_name(target.name + suffix))
        adopted.append(sidecar.name)
  except OSError as exc:
    log.warning("Issue-inbox legacy store %s could not be adopted: %s", legacy, exc)
    return
  log.info(
    "Issue-inbox adopted legacy store from %s into %s: %s",
    legacy.parent,
    target.parent,
    ", ".join(adopted),
  )


def _merge_legacy_issue_inbox_store(legacy: Path, target: Path) -> dict[str, str] | None:
  try:
    imported = _import_issue_inbox_rows(legacy, target)
  except (IssueInboxRowError, sqlite3.IntegrityError, _LegacyIssueInboxRefused) as exc:
    return {
      "code": "issue_sink_invalid",
      "message": (
        f"Legacy issue inbox {legacy} cannot be imported into {target} ({exc}); "
        "nothing was imported and the legacy store was not retired."
      ),
    }
  except (OSError, sqlite3.Error) as exc:
    log.warning("Issue-inbox legacy store %s could not be imported: %s", legacy, exc)
    return None
  stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  retired = legacy.with_name(f"{legacy.name}.adopted-{stamp}")
  try:
    for suffix in _ISSUE_INBOX_SIDECAR_SUFFIXES:
      sidecar = legacy.with_name(legacy.name + suffix)
      if sidecar.exists():
        os.replace(sidecar, retired.with_name(retired.name + suffix))
    os.replace(legacy, retired)
  except OSError as exc:
    # The rows are in the new store; without the rename the next resolve
    # re-imports them, which the id-keyed insert makes a no-op.
    log.warning("Issue-inbox legacy store %s could not be retired: %s", legacy, exc)
    return None
  log.info(
    "Issue-inbox imported %d legacy row(s) from %s into %s; legacy store kept as %s",
    imported,
    legacy,
    target,
    retired.name,
  )
  return None


def _import_issue_inbox_rows(legacy: Path, target: Path) -> int:
  with closing(sqlite3.connect(target, timeout=5.0)) as connection:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    columns = [row[1] for row in connection.execute("PRAGMA table_info(issue_inbox)")]
    if not columns:
      raise _LegacyIssueInboxRefused(
        f"the target store has no issue_inbox table; remove {target} if it is an "
        "empty file and retry, and the legacy store is then adopted whole"
      )
    connection.execute("ATTACH DATABASE ? AS legacy", (str(legacy),))
    try:
      legacy_columns = [
        row[1] for row in connection.execute("PRAGMA legacy.table_info(issue_inbox)")
      ]
      if not legacy_columns:
        return 0
      if "issue_id" not in legacy_columns:
        raise _LegacyIssueInboxRefused("the legacy issue_inbox table has no issue_id column")
      projection = ", ".join(f'"{name}"' for name in legacy_columns)
      with connection:
        # Keep the validated source snapshot through the INSERT and commit.
        connection.execute("BEGIN IMMEDIATE")
        for row in connection.execute("SELECT * FROM legacy.issue_inbox"):
          StoredIssueV1.parse(dict(row))
        cursor = connection.execute(
          f"INSERT INTO main.issue_inbox ({projection}) "
          f"SELECT {projection} FROM legacy.issue_inbox AS source "
          "WHERE NOT EXISTS ("
          "SELECT 1 FROM main.issue_inbox AS present WHERE present.issue_id = source.issue_id"
          ")"
        )
      return cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
    finally:
      connection.execute("DETACH DATABASE legacy")


def _safe_json_children(directory: Path, workspace_root: Path) -> list[Path]:
  artifacts: list[Path] = []
  for path in sorted(directory.glob("*.json"), key=lambda child: child.name):
    safe_path = _ensure_under_workspace(path, workspace_root)
    if not safe_path.is_file():
      continue
    try:
      _validate_artifact_id(safe_path.stem)
    except ArtifactPathError:
      # Only the artifact writer names .json children here; a foreign-named
      # file must not vanish silently or refuse the enumeration.
      log.warning(
        "artifact directory %s contains a foreign-named json file %r; ignoring it",
        directory,
        path.name,
      )
      continue
    artifacts.append(safe_path)
  return artifacts


def _resolve_under_workspace(workspace_root: Path, *parts: str) -> Path:
  for part in parts:
    _validate_path_component(part, "path")
  return _ensure_under_workspace(Path(workspace_root).joinpath(*parts), workspace_root)


def _ensure_under_workspace(path: Path, workspace_root: Path) -> Path:
  resolved_workspace = Path(workspace_root).resolve()
  resolved_path = Path(path).resolve()
  try:
    resolved_path.relative_to(resolved_workspace)
  except ValueError as exc:
    raise ArtifactPathError("resolved artifact path escapes user workspace") from exc
  return resolved_path


def _validate_ticker(ticker: str) -> str:
  return normalize_ticker_for_artifact_request(ticker)


def normalize_ticker_for_artifact_request(ticker: str) -> str:
  decoded = _validate_path_component(ticker, "ticker")
  try:
    return normalize_contract_ticker(decoded)
  except ValueError as exc:
    raise ArtifactPathError("invalid ticker path component") from exc


def canonicalize_ticker(ticker: object) -> str:
  """Canonicalize and validate an explicit ticker without app imports."""
  return normalize_contract_ticker(ticker)


def _validate_skill(skill: str) -> str:
  decoded = _validate_path_component(skill, "skill")
  if not _SKILL_NAME_RE.match(decoded):
    raise ArtifactPathError("invalid skill path component")
  return decoded


def _validate_artifact_id(artifact_id: str) -> str:
  decoded = _validate_path_component(artifact_id, "artifact_id")
  if not _SAFE_ID_RE.match(decoded):
    raise ArtifactPathError("invalid artifact_id path component")
  return decoded


def _validate_path_component(value: str, label: str) -> str:
  decoded = _decode_path_component(value).strip()
  max_len = MAX_ARTIFACT_FILENAME_LENGTH if label == "path" else MAX_ARTIFACT_COMPONENT_LENGTH
  if not decoded or decoded == "." or len(decoded) > max_len:
    raise ArtifactPathError(f"invalid {label} path component")
  if ".." in decoded or "/" in decoded or "\\" in decoded:
    raise ArtifactPathError(f"invalid {label} path component")
  candidate = Path(decoded)
  if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
    raise ArtifactPathError(f"invalid {label} path component")
  return decoded


def _decode_path_component(value: str) -> str:
  decoded = str(value or "")
  for _ in range(3):
    next_decoded = unquote(decoded)
    if next_decoded == decoded:
      break
    decoded = next_decoded
  return decoded


def _split_path_components(path: str) -> list[str]:
  decoded = _decode_path_component(path)
  return [component for component in decoded.split("/") if component]


__all__ = [
  "ArtifactPath",
  "ArtifactPathError",
  "artifact_json_paths_for_request",
  "artifact_json_path_for_request",
  "canonicalize_ticker",
  "existing_user_issue_inbox_db_paths",
  "ISSUE_INBOX_DB_ENV",
  "letter_docx_path_for_request",
  "industry_artifacts_dir",
  "INDUSTRY_ARTIFACTS_DIR_ENV",
  "normalize_ticker_for_artifact_request",
  "reject_unsafe_path",
  "resolve_issue_inbox_db_path",
  "ticker_artifact_paths_for_request",
  "user_data_dir",
  "user_error_signals_cursor_path",
  "user_issue_inbox_db_path",
  "user_workspace_root",
]
