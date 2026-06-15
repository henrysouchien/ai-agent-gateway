from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from schema.dashboard_artifact import DashboardArtifact


_DASHBOARD_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,128}$")


def write_dashboard_artifact(
  *,
  workspace_dir: Path,
  artifact: DashboardArtifact,
  payload_json: Any,
) -> None:
  """Atomic dual-write: payload first, sidecar rename as the commit point."""
  artifact_id = _validate_dashboard_artifact_id(artifact.artifact_id)
  directory = _dashboard_artifacts_dir(workspace_dir, create=True)

  payload_path = _ensure_under_workspace(directory / f"{artifact_id}.payload.json", workspace_dir)
  sidecar_path = _ensure_under_workspace(directory / f"{artifact_id}.json", workspace_dir)
  payload_tmp_path = _ensure_under_workspace(directory / f"{artifact_id}.payload.json.tmp", workspace_dir)
  sidecar_tmp_path = _ensure_under_workspace(directory / f"{artifact_id}.json.tmp", workspace_dir)

  payload = json.dumps(
    payload_json,
    ensure_ascii=False,
    separators=(",", ":"),
  )
  sidecar = json.dumps(
    artifact.model_dump(mode="json"),
    ensure_ascii=False,
    separators=(",", ":"),
  )

  try:
    payload_tmp_path.write_text(payload, encoding="utf-8")
    sidecar_tmp_path.write_text(sidecar, encoding="utf-8")
    payload_tmp_path.replace(payload_path)
    sidecar_tmp_path.replace(sidecar_path)
  except Exception:
    _unlink_if_exists(payload_tmp_path)
    _unlink_if_exists(sidecar_tmp_path)
    raise


def read_dashboard_artifact_sidecar(workspace_dir: Path, artifact_id: str) -> DashboardArtifact | None:
  sidecar_path = _dashboard_artifact_path(workspace_dir, artifact_id, suffix=".json")
  if not sidecar_path.is_file():
    return None
  return DashboardArtifact.model_validate_json(sidecar_path.read_bytes())


def read_dashboard_artifact_payload(workspace_dir: Path, artifact_id: str) -> Any | None:
  payload_path = _dashboard_artifact_path(workspace_dir, artifact_id, suffix=".payload.json")
  if not payload_path.is_file():
    return None
  return json.loads(payload_path.read_text(encoding="utf-8"))


def list_dashboard_artifacts(
  workspace_dir: Path,
  *,
  ticker: str | None = None,
  since: float | None = None,
  limit: int = 50,
) -> list[DashboardArtifact]:
  directory = _dashboard_artifacts_dir(workspace_dir)
  if not directory.is_dir() or limit <= 0:
    return []

  artifacts: list[tuple[int, Path, DashboardArtifact]] = []
  for sidecar_path in sorted(directory.glob("*.json"), key=lambda path: path.name):
    try:
      artifact_id = _validate_dashboard_artifact_id(sidecar_path.stem)
      safe_sidecar_path = _ensure_under_workspace(sidecar_path, workspace_dir)
    except ValueError:
      continue
    if artifact_id != sidecar_path.stem:
      continue
    try:
      sidecar_stat = safe_sidecar_path.stat()
      artifact = DashboardArtifact.model_validate_json(safe_sidecar_path.read_bytes())
    except (OSError, ValueError):
      continue
    if ticker is not None and artifact.ticker != ticker:
      continue
    if since is not None and sidecar_stat.st_mtime < since:
      continue
    artifacts.append((sidecar_stat.st_mtime_ns, sidecar_path, artifact))

  artifacts.sort(key=lambda entry: (entry[0], entry[1].as_posix()), reverse=True)
  return [artifact for _mtime_ns, _path, artifact in artifacts[:limit]]


def _dashboard_artifact_path(workspace_dir: Path, artifact_id: str, *, suffix: str) -> Path:
  normalized_artifact_id = _validate_dashboard_artifact_id(artifact_id)
  return _ensure_under_workspace(
    _dashboard_artifacts_dir(workspace_dir) / f"{normalized_artifact_id}{suffix}",
    workspace_dir,
  )


def _dashboard_artifacts_dir(workspace_dir: Path, *, create: bool = False) -> Path:
  workspace_root = Path(workspace_dir).resolve()
  artifacts_dir = _ensure_under_workspace(workspace_root / "artifacts", workspace_root)
  if create:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
  elif artifacts_dir.exists():
    _ensure_under_workspace(artifacts_dir, workspace_root)
  directory = _ensure_under_workspace(artifacts_dir / "_dashboards", workspace_root)
  if create:
    directory.mkdir(parents=True, exist_ok=True)
  elif directory.exists():
    _ensure_under_workspace(directory, workspace_root)
  return directory


def _ensure_under_workspace(path: Path, workspace_dir: Path) -> Path:
  workspace_root = Path(workspace_dir).resolve()
  resolved_path = Path(path).resolve()
  try:
    resolved_path.relative_to(workspace_root)
  except ValueError as exc:
    raise ValueError("resolved dashboard artifact path escapes workspace") from exc
  return resolved_path


def _validate_dashboard_artifact_id(artifact_id: str) -> str:
  normalized = str(artifact_id or "").strip()
  if not _DASHBOARD_ARTIFACT_ID_RE.match(normalized):
    raise ValueError("invalid dashboard artifact_id")
  return normalized


def _unlink_if_exists(path: Path) -> None:
  try:
    path.unlink()
  except FileNotFoundError:
    pass


__all__ = [
  "list_dashboard_artifacts",
  "read_dashboard_artifact_payload",
  "read_dashboard_artifact_sidecar",
  "write_dashboard_artifact",
]
