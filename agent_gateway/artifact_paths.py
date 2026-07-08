from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_ARTIFACT_INDEX_RECENT_LIMIT = 5


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


def latest_artifact_json_path_for_request(
  user_id: str,
  *,
  ticker: str,
  skill: str,
) -> ArtifactPath | None:
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
    return None
  artifacts = _safe_json_children(directory, workspace_root)
  if not artifacts:
    return None
  path = artifacts[-1]
  return ArtifactPath(
    workspace_root=workspace_root,
    path=path,
    ticker=normalized_ticker,
    skill=normalized_skill,
    artifact_id=path.stem,
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


def ticker_artifact_index_for_request(user_id: str, *, ticker: str) -> list[dict[str, Any]]:
  normalized_ticker = _validate_ticker(ticker)
  workspace_root = user_workspace_root(user_id)
  ticker_dir = _resolve_under_workspace(workspace_root, "artifacts", normalized_ticker)
  if not ticker_dir.is_dir():
    return []

  index: list[dict[str, Any]] = []
  for skill_dir in sorted(ticker_dir.iterdir(), key=lambda path: path.name):
    if not skill_dir.is_dir():
      continue
    try:
      normalized_skill = _validate_skill(skill_dir.name)
    except ArtifactPathError:
      continue
    safe_skill_dir = _ensure_under_workspace(skill_dir, workspace_root)
    artifacts = _safe_json_children(safe_skill_dir, workspace_root)
    if not artifacts:
      continue
    artifact_ids = [path.stem for path in artifacts]
    index.append({
      "skill": normalized_skill,
      "latest_artifact_id": artifact_ids[-1],
      "artifact_count": len(artifact_ids),
      "recent_artifact_ids": list(reversed(artifact_ids[-_ARTIFACT_INDEX_RECENT_LIMIT:])),
    })
  return index


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


def user_data_dir() -> Path:
  configured = os.getenv("USER_DATA_DIR", "").strip()
  if configured:
    return Path(configured).expanduser()
  return _repo_root() / "data"


def _repo_root() -> Path:
  return Path(__file__).resolve().parents[3]


def _safe_json_children(directory: Path, workspace_root: Path) -> list[Path]:
  artifacts: list[Path] = []
  for path in sorted(directory.glob("*.json"), key=lambda child: child.name):
    safe_path = _ensure_under_workspace(path, workspace_root)
    if not safe_path.is_file():
      continue
    try:
      _validate_artifact_id(safe_path.stem)
    except ArtifactPathError:
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
  if not _TICKER_RE.match(normalized):
    raise ArtifactPathError("invalid ticker path component")
  return normalized


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
  "latest_artifact_json_path_for_request",
  "letter_docx_path_for_request",
  "normalize_ticker_for_artifact_request",
  "reject_unsafe_path",
  "ticker_artifact_index_for_request",
  "ticker_artifact_paths_for_request",
  "user_data_dir",
  "user_workspace_root",
]
