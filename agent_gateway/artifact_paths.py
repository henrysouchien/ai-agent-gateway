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



_EXCHANGE_SUFFIXES = (
  ".TO",
  ".HK",
  ".AX",
  ".PA",
  ".DE",
  ".SW",
  ".AS",
  ".SS",
  ".SZ",
  ".OL",
  ".MI",
  ".CO",
  ".ST",
  ".HE",
  ".BR",
  ".SA",
  ".SI",
  ".KS",
  ".TW",
  ".BO",
  ".NS",
  ".L",
  ".T",
)
# Share-class suffixes are collapsed to a trailing letter (BRK.B / BRK-B -> BRKB).
# Preferred lines (for example EFC-PC, PPL-PA) are intentionally left hyphenated
# so they fail _TICKER_RE and cannot become common-equity artifact paths.
_SHARE_CLASS_SUFFIXES = (".A", ".B", "-A", "-B")
_TICKER_RE = re.compile(r"^[A-Z]{1,6}$")
# Non-US listings carry digits and exchange-suffix dots (B3 "TAEE11", HK
# "0700"). The writer side (api/research/artifact_paths.py) accepts them
# verbatim under this same path-safe rule; the read/list side must mirror it or
# persisted artifacts become unreadable (Lane H LH-12 reader symmetry).
# Hyphens stay excluded so preferred lines (EFC-PC, PPL-PA) keep failing per
# the non-common-equity policy above.
_EXTENDED_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.]{0,14}$")
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
ISSUE_INBOX_DB_ENV = "AGENT_ISSUE_INBOX_DB_PATH"
_ISSUE_INBOX_STORE_DIRECTORY = ("gateway", "issue-inbox")
_ISSUE_INBOX_DB_FILENAME = "issue-inbox.sqlite3"
_ISSUE_INBOX_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")

log = logging.getLogger(__name__)


class ArtifactPathError(ValueError):
  """Raised when a requested artifact path is not safe to resolve."""


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


def user_workspace_root(user_id: str) -> Path:
  normalized_user_id = _validate_user_id(user_id)
  return (user_data_dir() / "users" / normalized_user_id / "workspace").resolve()


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
    _adopt_legacy_issue_inbox_store(target)
  return target, None


def _adopt_legacy_issue_inbox_store(target: Path) -> None:
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
  each row's delivery marker travels with the row.
  """
  legacy = target.parent.parent / target.name
  if not legacy.is_file():
    return
  if not target.exists():
    _rename_legacy_issue_inbox_store(legacy, target)
    return
  _merge_legacy_issue_inbox_store(legacy, target)


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


def _merge_legacy_issue_inbox_store(legacy: Path, target: Path) -> None:
  try:
    imported = _import_issue_inbox_rows(legacy, target)
  except (OSError, sqlite3.Error) as exc:
    log.warning("Issue-inbox legacy store %s could not be imported: %s", legacy, exc)
    return
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
    return
  log.info(
    "Issue-inbox imported %d legacy row(s) from %s into %s; legacy store kept as %s",
    imported,
    legacy,
    target,
    retired.name,
  )


def _import_issue_inbox_rows(legacy: Path, target: Path) -> int:
  with closing(sqlite3.connect(target, timeout=5.0)) as connection:
    connection.execute("PRAGMA busy_timeout = 5000")
    columns = [row[1] for row in connection.execute("PRAGMA table_info(issue_inbox)")]
    if not columns:
      return 0
    connection.execute("ATTACH DATABASE ? AS legacy", (str(legacy),))
    try:
      legacy_columns = {
        row[1] for row in connection.execute("PRAGMA legacy.table_info(issue_inbox)")
      }
      carried = [name for name in columns if name in legacy_columns]
      if "issue_id" not in carried:
        return 0
      projection = ", ".join(f'"{name}"' for name in carried)
      with connection:
        cursor = connection.execute(
          f"INSERT OR IGNORE INTO main.issue_inbox ({projection}) "
          f"SELECT {projection} FROM legacy.issue_inbox"
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


def _validate_user_id(user_id: str) -> str:
  raw = str(user_id or "").strip()
  if not raw:
    raise ArtifactPathError("missing user_id")
  candidate = Path(raw)
  if candidate.is_absolute():
    raise ArtifactPathError("invalid user_id path component")
  for part in candidate.parts:
    _validate_path_component(part, "user_id")
  return raw


def _validate_ticker(ticker: str) -> str:
  return normalize_ticker_for_artifact_request(ticker)


def normalize_ticker_for_artifact_request(ticker: str) -> str:
  decoded = _validate_path_component(ticker, "ticker")
  normalized = _normalize_ticker(decoded)
  if _TICKER_RE.match(normalized):
    return normalized
  raw = decoded.strip().upper()
  if _EXTENDED_TICKER_RE.match(raw) and ".." not in raw and not raw.endswith("."):
    return raw
  raise ArtifactPathError("invalid ticker path component")


def canonicalize_ticker(ticker: object) -> str:
  """Canonicalize and validate an explicit ticker without app imports."""
  from agent_workflow_contracts.ticker_contract import normalize_contract_ticker

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
  if not decoded:
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


def _normalize_ticker(raw: str) -> str:
  value = raw.strip().upper()
  if value.endswith("."):
    value = value[:-1]

  for suffix in _EXCHANGE_SUFFIXES:
    if value.endswith(suffix):
      value = value[: -len(suffix)]
      break

  for suffix in _SHARE_CLASS_SUFFIXES:
    if value.endswith(suffix) and len(value) > len(suffix):
      value = value[: -len(suffix)] + suffix[-1]
      break

  return value


__all__ = [
  "ArtifactPath",
  "ArtifactPathError",
  "artifact_json_paths_for_request",
  "artifact_json_path_for_request",
  "canonicalize_ticker",
  "ISSUE_INBOX_DB_ENV",
  "letter_docx_path_for_request",
  "normalize_ticker_for_artifact_request",
  "reject_unsafe_path",
  "resolve_issue_inbox_db_path",
  "ticker_artifact_paths_for_request",
  "user_data_dir",
  "user_workspace_root",
]
