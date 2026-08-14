from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping

from .artifact_paths import user_workspace_root
from .artifact_sidecar_index import artifact_sidecar_index_path, register_ui_blocks_payload_sidecar


log = logging.getLogger(__name__)
_UI_BLOCKS_ID_RE = re.compile(r"^ub_[0-9a-f]{16}$")


@dataclass(frozen=True)
class UiBlocksWriteOutcome:
  path: Path
  index_lag: bool


def write_ui_blocks_payload(
  workspace_dir: Path,
  envelope: Mapping[str, Any],
  *,
  user_id: str,
) -> UiBlocksWriteOutcome:
  workspace = _validated_user_workspace(workspace_dir, user_id=user_id)
  ui_blocks_id = _validate_ui_blocks_id(envelope.get("ui_blocks_id"))
  directory = _ensure_under_workspace(workspace / "artifacts" / "_ui_blocks", workspace)
  directory.mkdir(parents=True, exist_ok=True)
  path = _ensure_under_workspace(directory / f"{ui_blocks_id}.json", workspace)
  temp_path = _ensure_under_workspace(directory / f"{ui_blocks_id}.json.tmp", workspace)
  payload = json.dumps(dict(envelope), ensure_ascii=False, separators=(",", ":"))

  try:
    temp_path.write_text(payload, encoding="utf-8")
    temp_path.replace(path)
  except Exception:
    _unlink_if_exists(temp_path)
    raise

  try:
    register_ui_blocks_payload_sidecar(
      workspace_dir=workspace,
      user_id=user_id,
      ui_blocks_id=ui_blocks_id,
      path=path,
      session_id=str(envelope["session_id"]),
      turn_key=str(envelope["turn_key"]),
      emission_index=int(envelope["emission_index"]),
      ts=float(envelope["ts"]),
    )
  except Exception:
    log.warning(
      "ui_blocks_index_failure",
      extra={
        "artifact_kind": "ui_blocks",
        "artifact_id": ui_blocks_id,
        "user_id": user_id,
        "workspace_dir": str(workspace),
        "index_path": str(artifact_sidecar_index_path(workspace)),
      },
      exc_info=True,
    )
    return UiBlocksWriteOutcome(path=path, index_lag=True)
  return UiBlocksWriteOutcome(path=path, index_lag=False)


def read_ui_blocks_payload(workspace_dir: Path, ui_blocks_id: str) -> dict[str, Any] | None:
  workspace = Path(workspace_dir).resolve()
  normalized_id = _validate_ui_blocks_id(ui_blocks_id)
  path = _ensure_under_workspace(
    workspace / "artifacts" / "_ui_blocks" / f"{normalized_id}.json",
    workspace,
  )
  if not path.is_file():
    return None
  payload = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(payload, dict):
    raise ValueError("ui blocks envelope must be a JSON object")
  return payload


def _validated_user_workspace(workspace_dir: Path, *, user_id: str) -> Path:
  expected = user_workspace_root(user_id)
  workspace = Path(workspace_dir).resolve()
  if workspace != expected:
    raise ValueError("ui blocks workspace does not match user workspace")
  return workspace


def _validate_ui_blocks_id(value: object) -> str:
  ui_blocks_id = str(value or "")
  if not _UI_BLOCKS_ID_RE.fullmatch(ui_blocks_id):
    raise ValueError("invalid ui_blocks_id")
  return ui_blocks_id


def _ensure_under_workspace(path: Path, workspace_dir: Path) -> Path:
  workspace = Path(workspace_dir).resolve()
  resolved = Path(path).resolve()
  try:
    resolved.relative_to(workspace)
  except ValueError as exc:
    raise ValueError("ui blocks path escapes workspace") from exc
  return resolved


def _unlink_if_exists(path: Path) -> None:
  try:
    path.unlink()
  except FileNotFoundError:
    pass


__all__ = ["UiBlocksWriteOutcome", "read_ui_blocks_payload", "write_ui_blocks_payload"]
