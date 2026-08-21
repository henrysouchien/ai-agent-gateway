from __future__ import annotations

from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable

from schema.canvas_artifact import CanvasArtifact
from schema.dashboard_artifact import DashboardArtifact
from schema.html_artifact import HtmlArtifact

from .artifact_sidecar_index import (
  delete_artifact_sidecar_index_rows,
  list_artifact_sidecar_index_rows,
)


_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_STORE_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,128}$")
_SKILL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.]{0,14}$")
_UI_BLOCKS_ID_RE = re.compile(r"^ub_[0-9a-f]{16}$")
_ERASABLE_KINDS = frozenset({"canvas", "dashboard", "html", "skill_artifact"})


@dataclass(frozen=True)
class _TypedArtifact:
  artifact_kind: str
  artifact_id: str
  artifact_ref: str
  research_file_id: int | None
  ticker: str | None
  paths: tuple[Path, ...]
  payload_ref: str | None = None
  expected_identities: tuple[tuple[Path, _FileIdentity], ...] = ()


@dataclass(frozen=True)
class _FileIdentity:
  device: int
  inode: int
  owner_uid: int
  size: int
  ctime_ns: int


@dataclass(frozen=True)
class _OpenDeleteTarget:
  path: Path
  name: str
  parent_descriptor: int
  file_descriptor: int
  identity: _FileIdentity


def purge_ui_blocks_payloads(
  workspace_dir: Path,
  *,
  user_id: str,
  ui_blocks_ids: tuple[str, ...],
) -> None:
  """Remove exact UI payloads named by typed research-message anchors."""

  workspace = _validated_workspace(workspace_dir, user_id=user_id)
  normalized_ids = _validated_ui_blocks_ids(ui_blocks_ids)
  indexed = {
    str(row["artifact_id"]): row
    for row in list_artifact_sidecar_index_rows(
      workspace_dir=workspace,
      artifact_kind="ui_blocks",
      user_id=user_id,
    )
  }
  paths: list[Path] = []
  keys: list[tuple[str, str]] = []
  expected_identities: dict[Path, _FileIdentity] = {}
  for ui_blocks_id in normalized_ids:
    expected_ref = f"artifacts/_ui_blocks/{ui_blocks_id}.json"
    row = indexed.get(ui_blocks_id)
    if row is not None and (
      row.get("artifact_ref") != ui_blocks_id
      or row.get("payload_ref") != expected_ref
    ):
      raise ValueError("ui blocks index identity is invalid")
    path = workspace / expected_ref
    if os.path.lexists(path):
      envelope, identity = _read_json_object(path, workspace=workspace)
      if envelope.get("ui_blocks_id") != ui_blocks_id:
        raise ValueError("ui blocks envelope identity is invalid")
      expected_identities[path] = identity
    paths.append(path)
    keys.append(("ui_blocks", ui_blocks_id))

  _delete_exact_paths(
    tuple(paths),
    workspace=workspace,
    expected_identities=expected_identities,
  )
  delete_artifact_sidecar_index_rows(
    workspace_dir=workspace,
    user_id=user_id,
    keys=tuple(keys),
  )


def purge_research_file_artifacts(
  workspace_dir: Path,
  *,
  user_id: str,
  research_file_ids: tuple[int, ...],
  research_file_tickers: tuple[str, ...] = (),
) -> None:
  """Remove current typed artifacts bound to exact owner research files."""

  workspace = _validated_workspace(workspace_dir, user_id=user_id)
  target_file_ids = _validated_research_file_ids(research_file_ids)
  target_tickers = _validated_tickers(research_file_tickers)
  if not target_file_ids and not target_tickers:
    return

  # This is the bounded, source-owned index-lag fallback. Selection reads only
  # typed research-file/ticker bindings; artifact prose, names, and origin text
  # never participate in deletion targeting.
  records = {
    (record.artifact_kind, record.artifact_ref): record
    for record in (
      *_scan_store_records(
        workspace,
        artifact_kind="canvas",
        directory_name="_canvas",
        model=CanvasArtifact,
        payload_suffixes=(".tsx", ".bundle.js"),
      ),
      *_scan_store_records(
        workspace,
        artifact_kind="dashboard",
        directory_name="_dashboards",
        model=DashboardArtifact,
        payload_suffixes=(".payload.json",),
      ),
      *_scan_store_records(
        workspace,
        artifact_kind="html",
        directory_name="_html",
        model=HtmlArtifact,
        payload_suffixes=(".html",),
      ),
      *_scan_skill_records(workspace),
      *_scan_legacy_portfolio_skill_records(workspace),
    )
  }
  index_rows = list_artifact_sidecar_index_rows(
    workspace_dir=workspace,
    user_id=user_id,
  )

  candidates: dict[tuple[str, str], _TypedArtifact] = {
    key: record
    for key, record in records.items()
    if _artifact_matches_targets(
      record,
      research_file_ids=target_file_ids,
      research_file_tickers=target_tickers,
    )
  }
  index_keys: set[tuple[str, str]] = set(candidates)
  for row in index_rows:
    artifact_kind = str(row.get("artifact_kind") or "")
    if artifact_kind not in _ERASABLE_KINDS:
      continue
    artifact_ref = str(row.get("artifact_ref") or "")
    key = (artifact_kind, artifact_ref)
    current = records.get(key)
    if current is not None:
      if _artifact_matches_targets(
        current,
        research_file_ids=target_file_ids,
        research_file_tickers=target_tickers,
      ):
        _require_index_agrees(row, current)
        candidates[key] = current
        index_keys.add(key)
      continue
    if not _index_row_matches_targets(
      row,
      research_file_ids=target_file_ids,
      research_file_tickers=target_tickers,
    ):
      continue
    candidate = _candidate_from_index_row(workspace, row)
    candidates[key] = candidate
    index_keys.add(key)

  ordered = tuple(
    candidates[key]
    for key in sorted(candidates)
  )
  _delete_exact_paths(
    tuple(path for candidate in ordered for path in candidate.paths),
    workspace=workspace,
    expected_identities={
      path: identity
      for candidate in ordered
      for path, identity in candidate.expected_identities
    },
  )
  delete_artifact_sidecar_index_rows(
    workspace_dir=workspace,
    user_id=user_id,
    keys=tuple(sorted(index_keys)),
  )


def _scan_store_records(
  workspace: Path,
  *,
  artifact_kind: str,
  directory_name: str,
  model: Callable[..., Any],
  payload_suffixes: tuple[str, ...],
) -> tuple[_TypedArtifact, ...]:
  directory = workspace / "artifacts" / directory_name
  records: list[_TypedArtifact] = []
  for path in _regular_files(directory, workspace=workspace):
    if path.suffix != ".json" or path.name.endswith(".payload.json"):
      continue
    artifact_id = path.stem
    if _STORE_ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
      continue
    payload, identity = _read_json_object(path, workspace=workspace)
    artifact = model.model_validate(payload)
    if str(artifact.artifact_id) != artifact_id:
      raise ValueError(f"{artifact_kind} artifact identity is invalid")
    artifact_ref = path.relative_to(workspace).as_posix()
    payload_paths = tuple(
      directory / f"{artifact_id}{suffix}"
      for suffix in payload_suffixes
    )
    records.append(_TypedArtifact(
      artifact_kind=artifact_kind,
      artifact_id=artifact_id,
      artifact_ref=artifact_ref,
      research_file_id=artifact.research_file_id,
      ticker=_optional_ticker(artifact.ticker),
      paths=(*payload_paths, path),
      payload_ref=(
        payload_paths[-1].relative_to(workspace).as_posix()
        if payload_paths
        else None
      ),
      expected_identities=((path, identity),),
    ))
  return tuple(records)


def _scan_skill_records(workspace: Path) -> tuple[_TypedArtifact, ...]:
  artifacts_dir = workspace / "artifacts"
  records: list[_TypedArtifact] = []
  for scope_dir in _directories(artifacts_dir, workspace=workspace):
    if scope_dir.name.startswith("_") and scope_dir.name != "_portfolio":
      continue
    if scope_dir.name != "_portfolio" and not _valid_ticker(scope_dir.name):
      continue
    for skill_dir in _directories(scope_dir, workspace=workspace):
      if _SKILL_RE.fullmatch(skill_dir.name) is None:
        continue
      for path in _regular_files(skill_dir, workspace=workspace):
        if path.suffix != ".json":
          continue
        artifact_id = path.stem
        if _ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
          continue
        payload, identity = _read_json_object(path, workspace=workspace)
        payload_artifact_id = str(payload.get("artifact_id") or artifact_id)
        if payload_artifact_id != artifact_id:
          raise ValueError("skill artifact identity is invalid")
        research_file_id = _optional_research_file_id(
          payload.get("research_file_id")
        )
        ticker = _skill_artifact_ticker(scope_dir, payload)
        payload_path, payload_ref = _skill_payload_path(
          workspace,
          artifact_id=artifact_id,
          payload=payload,
          required=research_file_id is not None,
        )
        records.append(_TypedArtifact(
          artifact_kind="skill_artifact",
          artifact_id=artifact_id,
          artifact_ref=path.relative_to(workspace).as_posix(),
          research_file_id=research_file_id,
          ticker=ticker,
          paths=(*((payload_path,) if payload_path is not None else ()), path),
          payload_ref=payload_ref,
          expected_identities=((path, identity),),
        ))
  return tuple(records)


def _scan_legacy_portfolio_skill_records(
  workspace: Path,
) -> tuple[_TypedArtifact, ...]:
  skills_dir = workspace / "notes" / "skills"
  records: list[_TypedArtifact] = []
  for skill_dir in _directories(skills_dir, workspace=workspace):
    if _SKILL_RE.fullmatch(skill_dir.name) is None:
      continue
    for path in _regular_files(skill_dir, workspace=workspace):
      if not path.name.endswith(".typed.json"):
        continue
      artifact_id = path.name[: -len(".typed.json")]
      if _ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
        continue
      payload, identity = _read_json_object(path, workspace=workspace)
      research_file_id = _optional_research_file_id(
        payload.get("research_file_id")
      )
      records.append(_TypedArtifact(
        artifact_kind="skill_artifact",
        artifact_id=artifact_id,
        artifact_ref=path.relative_to(workspace).as_posix(),
        research_file_id=research_file_id,
        ticker=_optional_ticker(payload.get("ticker")),
        paths=(path.with_name(f"{artifact_id}.md"), path),
        expected_identities=((path, identity),),
      ))
  return tuple(records)


def _candidate_from_index_row(
  workspace: Path,
  row: dict[str, Any],
) -> _TypedArtifact:
  artifact_kind = str(row.get("artifact_kind") or "")
  artifact_id = str(row.get("artifact_id") or "")
  artifact_ref = str(row.get("artifact_ref") or "")
  if artifact_kind in {"canvas", "dashboard", "html"}:
    if _STORE_ARTIFACT_ID_RE.fullmatch(artifact_id) is None:
      raise ValueError("artifact index identity is invalid")
    directory, suffixes = {
      "canvas": ("_canvas", (".tsx", ".bundle.js", ".json")),
      "dashboard": ("_dashboards", (".payload.json", ".json")),
      "html": ("_html", (".html", ".json")),
    }[artifact_kind]
    expected_ref = f"artifacts/{directory}/{artifact_id}.json"
    if artifact_ref != expected_ref:
      raise ValueError("artifact index ref is invalid")
    paths = tuple(
      workspace / "artifacts" / directory / f"{artifact_id}{suffix}"
      for suffix in suffixes
    )
    expected_payload_ref = paths[-2].relative_to(workspace).as_posix()
    if row.get("payload_ref") not in {None, expected_payload_ref}:
      raise ValueError("artifact index payload ref is invalid")
    return _TypedArtifact(
      artifact_kind=artifact_kind,
      artifact_id=artifact_id,
      artifact_ref=artifact_ref,
      research_file_id=_optional_research_file_id(row.get("research_file_id")),
      ticker=_optional_ticker(row.get("ticker")),
      paths=paths,
      payload_ref=expected_payload_ref,
    )
  if artifact_kind != "skill_artifact":
    raise ValueError("artifact index kind is invalid")
  path = _canonical_skill_sidecar_path(workspace, artifact_ref, artifact_id)
  payload_path = _canonical_index_docx_path(
    workspace,
    artifact_id=artifact_id,
    ticker=str(row.get("ticker") or ""),
    payload_ref=row.get("payload_ref"),
  )
  return _TypedArtifact(
    artifact_kind=artifact_kind,
    artifact_id=artifact_id,
    artifact_ref=artifact_ref,
    research_file_id=_optional_research_file_id(row.get("research_file_id")),
    ticker=_optional_ticker(row.get("ticker")),
    paths=(*((payload_path,) if payload_path is not None else ()), path),
    payload_ref=(
      payload_path.relative_to(workspace).as_posix()
      if payload_path is not None
      else None
    ),
  )


def _require_index_agrees(
  row: dict[str, Any],
  current: _TypedArtifact,
) -> None:
  if str(row.get("artifact_id") or "") != current.artifact_id:
    raise ValueError("artifact index identity disagrees with sidecar")
  indexed_payload_ref = row.get("payload_ref")
  if (
    indexed_payload_ref is not None
    and current.payload_ref is not None
    and indexed_payload_ref != current.payload_ref
  ):
    raise ValueError("artifact index payload ref disagrees with sidecar")


def _artifact_matches_targets(
  artifact: _TypedArtifact,
  *,
  research_file_ids: frozenset[int],
  research_file_tickers: frozenset[str],
) -> bool:
  return artifact.research_file_id in research_file_ids or (
    artifact.research_file_id is None
    and artifact.ticker in research_file_tickers
  )


def _index_row_matches_targets(
  row: dict[str, Any],
  *,
  research_file_ids: frozenset[int],
  research_file_tickers: frozenset[str],
) -> bool:
  research_file_id = _optional_research_file_id(row.get("research_file_id"))
  return research_file_id in research_file_ids or (
    research_file_id is None
    and _optional_ticker(row.get("ticker")) in research_file_tickers
  )


def _skill_artifact_ticker(
  scope_dir: Path,
  payload: dict[str, Any],
) -> str | None:
  payload_ticker = _optional_ticker(payload.get("ticker"))
  path_ticker = (
    None
    if scope_dir.name == "_portfolio"
    else _optional_ticker(scope_dir.name)
  )
  if (
    payload_ticker is not None
    and path_ticker is not None
    and payload_ticker != path_ticker
  ):
    raise ValueError("skill artifact ticker disagrees with typed path")
  return payload_ticker or path_ticker


def _skill_payload_path(
  workspace: Path,
  *,
  artifact_id: str,
  payload: dict[str, Any],
  required: bool,
) -> tuple[Path | None, str | None]:
  raw_ref = payload.get("binary_artifact_path") or payload.get("payload_ref")
  if raw_ref is None:
    return None, None
  if not isinstance(raw_ref, str) or not raw_ref.endswith(".docx"):
    return None, None
  ticker = str(payload.get("ticker") or "").strip().upper()
  expected_ref = f"letters/{ticker}/{artifact_id}.docx"
  if not _valid_ticker(ticker) or raw_ref != expected_ref:
    if required:
      raise ValueError("skill artifact payload ref is not canonical DOCX")
    return None, None
  return workspace / expected_ref, expected_ref


def _canonical_index_docx_path(
  workspace: Path,
  *,
  artifact_id: str,
  ticker: str,
  payload_ref: object,
) -> Path | None:
  if payload_ref is None:
    return None
  if not isinstance(payload_ref, str) or not payload_ref.endswith(".docx"):
    return None
  normalized_ticker = ticker.strip().upper()
  expected_ref = f"letters/{normalized_ticker}/{artifact_id}.docx"
  if not _valid_ticker(normalized_ticker) or payload_ref != expected_ref:
    raise ValueError("skill artifact index payload ref is not canonical DOCX")
  return workspace / expected_ref


def _canonical_skill_sidecar_path(
  workspace: Path,
  artifact_ref: str,
  artifact_id: str,
) -> Path:
  parts = Path(artifact_ref).parts
  if (
    len(parts) != 4
    or parts[0] != "artifacts"
    or parts[-1] != f"{artifact_id}.json"
    or _ARTIFACT_ID_RE.fullmatch(artifact_id) is None
    or _SKILL_RE.fullmatch(parts[2]) is None
    or (
      parts[1] != "_portfolio"
      and not _valid_ticker(parts[1])
    )
  ):
    raise ValueError("skill artifact index ref is invalid")
  return workspace.joinpath(*parts)


def _validated_workspace(workspace_dir: Path, *, user_id: str) -> Path:
  normalized_user_id = str(user_id or "").strip()
  if not normalized_user_id:
    raise ValueError("artifact erase user_id is required")
  raw_workspace = Path(workspace_dir).expanduser()
  if raw_workspace.is_symlink():
    raise ValueError("artifact erase workspace cannot be a symlink")
  workspace = raw_workspace.resolve()
  if (
    len(workspace.parts) >= 3
    and workspace.parts[-1] == "workspace"
    and workspace.parts[-3] == "users"
    and workspace.parts[-2] != normalized_user_id
  ):
    raise ValueError("artifact erase workspace does not match user_id")
  return workspace


def _validated_ui_blocks_ids(values: tuple[str, ...]) -> tuple[str, ...]:
  if type(values) is not tuple:
    raise TypeError("ui_blocks_ids must be a tuple")
  normalized = tuple(sorted(set(values)))
  if any(type(value) is not str or _UI_BLOCKS_ID_RE.fullmatch(value) is None for value in normalized):
    raise ValueError("invalid ui_blocks_id")
  return normalized


def _validated_research_file_ids(values: tuple[int, ...]) -> frozenset[int]:
  if type(values) is not tuple:
    raise TypeError("research_file_ids must be a tuple")
  normalized: set[int] = set()
  for value in values:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value < (1 << 63):
      raise ValueError("invalid research_file_id")
    normalized.add(value)
  return frozenset(normalized)


def _validated_tickers(values: tuple[str, ...]) -> frozenset[str]:
  if type(values) is not tuple:
    raise TypeError("research_file_tickers must be a tuple")
  return frozenset(_required_ticker(value) for value in values)


def _required_ticker(value: object) -> str:
  if type(value) is not str or not _valid_ticker(value):
    raise ValueError("invalid research file ticker")
  return value


def _optional_ticker(value: object) -> str | None:
  if value is None or value == "":
    return None
  return _required_ticker(value)


def _optional_research_file_id(value: object) -> int | None:
  if value is None or isinstance(value, bool):
    return None
  try:
    normalized = int(value)
  except (TypeError, ValueError):
    return None
  return normalized if 0 < normalized < (1 << 63) else None


def _valid_ticker(value: str) -> bool:
  return (
    _TICKER_RE.fullmatch(value) is not None
    and ".." not in value
    and not value.endswith(".")
  )


def _directories(directory: Path, *, workspace: Path) -> tuple[Path, ...]:
  descriptor = _open_directory_descriptor(
    directory,
    workspace=workspace,
    required=False,
  )
  if descriptor is None:
    return ()
  children: list[Path] = []
  try:
    with os.scandir(descriptor) as entries:
      for entry in entries:
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
          raise ValueError("artifact erase encountered a symlink")
        if stat.S_ISDIR(info.st_mode):
          children.append(directory / entry.name)
  finally:
    os.close(descriptor)
  return tuple(sorted(children, key=lambda path: path.name))


def _regular_files(directory: Path, *, workspace: Path) -> tuple[Path, ...]:
  descriptor = _open_directory_descriptor(
    directory,
    workspace=workspace,
    required=False,
  )
  if descriptor is None:
    return ()
  children: list[Path] = []
  try:
    with os.scandir(descriptor) as entries:
      for entry in entries:
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
          raise ValueError("artifact erase encountered a symlink")
        if stat.S_ISREG(info.st_mode):
          children.append(directory / entry.name)
  finally:
    os.close(descriptor)
  return tuple(sorted(children, key=lambda path: path.name))


def _read_json_object(
  path: Path,
  *,
  workspace: Path,
) -> tuple[dict[str, Any], _FileIdentity]:
  target = _open_delete_target(path, workspace=workspace, required=True)
  assert target is not None
  try:
    read_descriptor = os.dup(target.file_descriptor)
    try:
      handle = os.fdopen(
        read_descriptor,
        "r",
        encoding="utf-8",
      )
    except BaseException:
      os.close(read_descriptor)
      raise
    with handle:
      payload = json.load(handle)
    _require_open_target_stable(target)
  finally:
    _close_delete_target(target)
  if not isinstance(payload, dict):
    raise ValueError("artifact sidecar must be a JSON object")
  return payload, target.identity


def _require_lexical_workspace_path(path: Path, *, workspace: Path) -> None:
  absolute = Path(os.path.abspath(path))
  try:
    absolute.relative_to(workspace)
  except ValueError as exc:
    raise ValueError("artifact erase path escapes workspace") from exc


def _open_directory_descriptor(
  directory: Path,
  *,
  workspace: Path,
  required: bool,
) -> int | None:
  _require_lexical_workspace_path(directory, workspace=workspace)
  try:
    relative = Path(os.path.abspath(directory)).relative_to(workspace)
  except ValueError as exc:
    raise ValueError("artifact erase directory escapes workspace") from exc
  flags = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
  )
  try:
    workspace_path_info = workspace.lstat()
  except FileNotFoundError:
    if not required:
      return None
    raise
  if (
    stat.S_ISLNK(workspace_path_info.st_mode)
    or not stat.S_ISDIR(workspace_path_info.st_mode)
  ):
    raise ValueError("artifact erase workspace is unsafe")
  try:
    descriptor = os.open(workspace, flags)
  except OSError as exc:
    if not required and exc.errno == errno.ENOENT:
      return None
    raise ValueError("artifact erase workspace is unsafe") from exc
  try:
    workspace_info = os.fstat(descriptor)
    if (
      not stat.S_ISDIR(workspace_info.st_mode)
      or (workspace_info.st_dev, workspace_info.st_ino)
      != (workspace_path_info.st_dev, workspace_path_info.st_ino)
      or workspace_info.st_uid != os.geteuid()
      or stat.S_IMODE(workspace_info.st_mode) & 0o022
    ):
      raise ValueError("artifact erase workspace is unsafe")
    owner_uid = workspace_info.st_uid
    for component in relative.parts:
      try:
        child_descriptor = os.open(component, flags, dir_fd=descriptor)
      except OSError as exc:
        if not required and exc.errno == errno.ENOENT:
          os.close(descriptor)
          return None
        raise ValueError("artifact erase directory is unsafe") from exc
      child_info = os.fstat(child_descriptor)
      if (
        not stat.S_ISDIR(child_info.st_mode)
        or child_info.st_uid != owner_uid
        or stat.S_IMODE(child_info.st_mode) & 0o022
      ):
        os.close(child_descriptor)
        raise ValueError("artifact erase directory is unsafe")
      os.close(descriptor)
      descriptor = child_descriptor
    return descriptor
  except BaseException:
    os.close(descriptor)
    raise


def _open_delete_target(
  path: Path,
  *,
  workspace: Path,
  required: bool,
) -> _OpenDeleteTarget | None:
  _require_lexical_workspace_path(path, workspace=workspace)
  parent_descriptor = _open_directory_descriptor(
    path.parent,
    workspace=workspace,
    required=required,
  )
  if parent_descriptor is None:
    return None
  flags = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
  )
  try:
    try:
      file_descriptor = os.open(
        path.name,
        flags,
        dir_fd=parent_descriptor,
      )
    except OSError as exc:
      if not required and exc.errno == errno.ENOENT:
        os.close(parent_descriptor)
        return None
      raise ValueError("artifact erase target is unsafe") from exc
    try:
      file_info = os.fstat(file_descriptor)
      parent_info = os.fstat(parent_descriptor)
      if (
        not stat.S_ISREG(file_info.st_mode)
        or file_info.st_nlink != 1
        or file_info.st_uid != parent_info.st_uid
      ):
        raise ValueError("artifact erase target is unsafe")
      identity = _file_identity(file_info)
      target = _OpenDeleteTarget(
        path=path,
        name=path.name,
        parent_descriptor=parent_descriptor,
        file_descriptor=file_descriptor,
        identity=identity,
      )
      _require_open_target_stable(target)
      return target
    except BaseException:
      os.close(file_descriptor)
      raise
  except BaseException:
    os.close(parent_descriptor)
    raise


def _file_identity(info: os.stat_result) -> _FileIdentity:
  return _FileIdentity(
    device=info.st_dev,
    inode=info.st_ino,
    owner_uid=info.st_uid,
    size=info.st_size,
    ctime_ns=info.st_ctime_ns,
  )


def _require_open_target_stable(target: _OpenDeleteTarget) -> None:
  descriptor_info = os.fstat(target.file_descriptor)
  if (
    not stat.S_ISREG(descriptor_info.st_mode)
    or descriptor_info.st_nlink != 1
    or _file_identity(descriptor_info) != target.identity
  ):
    raise ValueError("artifact erase target changed during validation")
  try:
    entry_info = os.stat(
      target.name,
      dir_fd=target.parent_descriptor,
      follow_symlinks=False,
    )
  except FileNotFoundError as exc:
    raise ValueError("artifact erase target changed during validation") from exc
  if (
    not stat.S_ISREG(entry_info.st_mode)
    or entry_info.st_nlink != 1
    or _file_identity(entry_info) != target.identity
  ):
    raise ValueError("artifact erase target changed during validation")


def _close_delete_target(target: _OpenDeleteTarget) -> None:
  os.close(target.file_descriptor)
  os.close(target.parent_descriptor)


def _delete_exact_paths(
  paths: tuple[Path, ...],
  *,
  workspace: Path,
  expected_identities: dict[Path, _FileIdentity] | None = None,
) -> None:
  expected = expected_identities or {}
  ordered_paths = tuple(dict.fromkeys(paths))
  targets: list[_OpenDeleteTarget] = []
  try:
    for path in ordered_paths:
      target = _open_delete_target(
        path,
        workspace=workspace,
        required=False,
      )
      if target is None:
        continue
      expected_identity = expected.get(path)
      if (
        expected_identity is not None
        and target.identity != expected_identity
      ):
        _close_delete_target(target)
        raise ValueError("artifact erase target changed after discovery")
      targets.append(target)

    # Validate the complete set while every file and parent descriptor remains
    # open. No unlink occurs unless the whole deletion set is still identical.
    for target in targets:
      _require_open_target_stable(target)
    for target in targets:
      _require_open_target_stable(target)
      os.unlink(target.name, dir_fd=target.parent_descriptor)
      if os.fstat(target.file_descriptor).st_nlink != 0:
        raise ValueError("artifact erase target retained an unexpected link")
  finally:
    for target in reversed(targets):
      _close_delete_target(target)


__all__ = [
  "purge_research_file_artifacts",
  "purge_ui_blocks_payloads",
]
