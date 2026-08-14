from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import re

from .artifact_paths import canonicalize_ticker
from .artifact_sidecar_index import artifact_sidecar_index_path, register_canvas_artifact_sidecar
from schema.canvas_artifact import CanvasArtifact


log = logging.getLogger(__name__)
_CANVAS_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,128}$")


def write_canvas_artifact(
  *,
  workspace_dir: Path,
  artifact: CanvasArtifact,
  source: str,
  bundle: bytes,
  user_id: str = "",
) -> None:
  """Write source and bundle first, with the sidecar rename as the commit point."""
  artifact_id = _validate_canvas_artifact_id(artifact.artifact_id)
  if not isinstance(source, str) or not source.strip():
    raise ValueError("canvas artifact source must be a non-empty string")
  if not isinstance(bundle, bytes) or not bundle:
    raise ValueError("canvas artifact bundle must be non-empty bytes")
  source_bytes = source.encode("utf-8")
  if hashlib.sha256(source_bytes).hexdigest() != artifact.source_digest:
    raise ValueError("canvas artifact source digest does not match source bytes")
  if hashlib.sha256(bundle).hexdigest() != artifact.bundle_digest:
    raise ValueError("canvas artifact bundle digest does not match bundle bytes")

  directory = _canvas_artifacts_dir(workspace_dir, create=True)
  source_path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".tsx")
  bundle_path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".bundle.js")
  sidecar_path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".json")
  source_tmp_path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".tsx.tmp")
  bundle_tmp_path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".bundle.js.tmp")
  sidecar_tmp_path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".json.tmp")
  del directory

  payload = json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
  committed_paths: list[Path] = []
  try:
    source_tmp_path.write_bytes(source_bytes)
    bundle_tmp_path.write_bytes(bundle)
    sidecar_tmp_path.write_text(payload, encoding="utf-8")
    source_tmp_path.replace(source_path)
    committed_paths.append(source_path)
    bundle_tmp_path.replace(bundle_path)
    committed_paths.append(bundle_path)
    sidecar_tmp_path.replace(sidecar_path)
    committed_paths.append(sidecar_path)
  except Exception:
    for path in (source_tmp_path, bundle_tmp_path, sidecar_tmp_path, *reversed(committed_paths)):
      _unlink_if_exists(path)
    raise

  try:
    register_canvas_artifact_sidecar(
      workspace_dir=workspace_dir,
      artifact=artifact,
      sidecar_path=sidecar_path,
      bundle_path=bundle_path,
      user_id=user_id,
    )
  except Exception:
    log.warning(
      "artifact_index_failure",
      extra={
        "artifact_kind": "canvas",
        "artifact_id": artifact_id,
        "user_id": user_id,
        "workspace_dir": str(Path(workspace_dir).resolve()),
        "index_path": str(artifact_sidecar_index_path(workspace_dir)),
      },
      exc_info=True,
    )


def read_canvas_artifact_sidecar(workspace_dir: Path, artifact_id: str) -> CanvasArtifact | None:
  path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".json")
  if not path.is_file():
    return None
  return CanvasArtifact.model_validate_json(path.read_bytes())


def read_canvas_artifact_source(workspace_dir: Path, artifact_id: str) -> str | None:
  path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".tsx")
  if not path.is_file():
    return None
  return path.read_text(encoding="utf-8")


def read_canvas_artifact_bundle(workspace_dir: Path, artifact_id: str) -> bytes | None:
  path = _canvas_artifact_path(workspace_dir, artifact_id, suffix=".bundle.js")
  if not path.is_file():
    return None
  return path.read_bytes()


def list_canvas_artifacts(
  workspace_dir: Path,
  *,
  ticker: str | None = None,
  purpose: str | None = None,
  since: float | None = None,
  limit: int = 50,
) -> list[CanvasArtifact]:
  directory = _canvas_artifacts_dir(workspace_dir)
  if not directory.is_dir() or limit <= 0:
    return []
  ticker = canonicalize_ticker(ticker) if ticker is not None else None
  artifacts: list[tuple[int, Path, CanvasArtifact]] = []
  for sidecar_path in sorted(directory.glob("*.json"), key=lambda path: path.name):
    try:
      artifact_id = _validate_canvas_artifact_id(sidecar_path.stem)
      safe_sidecar_path = _ensure_under_workspace(sidecar_path, workspace_dir)
    except ValueError:
      continue
    if artifact_id != sidecar_path.stem:
      continue
    try:
      sidecar_stat = safe_sidecar_path.stat()
      artifact = CanvasArtifact.model_validate_json(safe_sidecar_path.read_bytes())
    except (OSError, ValueError):
      continue
    if ticker is not None and artifact.ticker != ticker:
      continue
    if purpose is not None and artifact.purpose != purpose:
      continue
    if since is not None and sidecar_stat.st_mtime < since:
      continue
    artifacts.append((sidecar_stat.st_mtime_ns, sidecar_path, artifact))
  artifacts.sort(key=lambda entry: (entry[0], entry[1].as_posix()), reverse=True)
  return [artifact for _mtime_ns, _path, artifact in artifacts[:limit]]


def _canvas_artifact_path(workspace_dir: Path, artifact_id: str, *, suffix: str) -> Path:
  normalized = _validate_canvas_artifact_id(artifact_id)
  return _ensure_under_workspace(_canvas_artifacts_dir(workspace_dir) / f"{normalized}{suffix}", workspace_dir)


def _canvas_artifacts_dir(workspace_dir: Path, *, create: bool = False) -> Path:
  workspace_root = Path(workspace_dir).resolve()
  artifacts_dir = _ensure_under_workspace(workspace_root / "artifacts", workspace_root)
  if create:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
  elif artifacts_dir.exists():
    _ensure_under_workspace(artifacts_dir, workspace_root)
  directory = _ensure_under_workspace(artifacts_dir / "_canvas", workspace_root)
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
    raise ValueError("resolved canvas artifact path escapes workspace") from exc
  return resolved_path


def _validate_canvas_artifact_id(artifact_id: str) -> str:
  normalized = str(artifact_id or "").strip()
  if not _CANVAS_ARTIFACT_ID_RE.match(normalized):
    raise ValueError("invalid canvas artifact_id")
  return normalized


def _unlink_if_exists(path: Path) -> None:
  try:
    path.unlink()
  except FileNotFoundError:
    pass


__all__ = [
  "list_canvas_artifacts",
  "read_canvas_artifact_bundle",
  "read_canvas_artifact_sidecar",
  "read_canvas_artifact_source",
  "write_canvas_artifact",
]
