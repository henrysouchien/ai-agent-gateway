from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath

from .canonical_json_target_lock import canonical_json_target_lock_name


def run_control_root(workspace_dir: Path) -> Path:
  """Run metadata root, a sibling of the compared workspace.

  Locks, the artifact sidecar index, excerpt ledgers, source registries,
  effect-lineage and skill state live here so they stay out of the workspace
  tree a run is compared on. A user's workspace is ``users/<id>/workspace``
  and its control root is ``users/<id>/control``. Any other directory used as
  a workspace (an eval or test fixture) gets the sibling ``<name>.control``.
  """

  resolved = Path(workspace_dir).resolve()
  if resolved.name == "workspace":
    return resolved.parent / "control"
  return resolved.parent / f"{resolved.name}.control"


# Run-record stores written under the workspace before they moved to the run
# control root. Each keeps its workspace-relative path beneath the control root.
_RELOCATED_DIRECTORIES = (
  PurePosixPath("artifacts/_excerpt_ledger"),
  PurePosixPath("artifacts/_source_registry"),
  PurePosixPath("artifacts/_lineage"),
  PurePosixPath("artifacts/_index"),
  PurePosixPath(".locks/research_file_activity"),
)
_SKILL_STATE_NAMES = (
  "skill_state.json",
  "skill_state.json.bak",
  canonical_json_target_lock_name("skill_state.json"),
)


class RunStoreRelocationConflict(RuntimeError):
  """A legacy run store and its control-root destination both exist."""


@dataclass(frozen=True)
class RunStoreMove:
  source: Path
  destination: Path


def pending_run_store_moves(workspace_dir: Path) -> tuple[RunStoreMove, ...]:
  """Every legacy run store still under ``workspace_dir``, with its destination."""

  workspace = Path(workspace_dir).resolve()
  control = run_control_root(workspace)
  relatives = [relative for relative in _RELOCATED_DIRECTORIES if _present(workspace / relative)]
  notes = workspace / "notes"
  if notes.is_dir() and not notes.is_symlink():
    for profile in sorted(notes.iterdir(), key=lambda child: child.name):
      if profile.is_symlink() or not profile.is_dir():
        continue
      relatives.extend(
        PurePosixPath("notes", profile.name, name)
        for name in _SKILL_STATE_NAMES
        if _present(profile / name)
      )
  return tuple(
    RunStoreMove(source=workspace / relative, destination=control / relative)
    for relative in relatives
  )


def relocate_workspace_run_stores(workspace_dir: Path) -> tuple[RunStoreMove, ...]:
  """Move one workspace's legacy run stores beneath its run control root.

  All moves for the workspace happen or none do: a destination that already
  exists refuses the whole workspace before anything moves, and a failed rename
  restores the ones already made. A workspace with nothing left to move returns
  ``()``, so a second run is a no-op. Renames stay on one filesystem (the
  control root is the workspace's sibling), so every byte, SQLite sidecar and
  lock file moves as is and handles keep resolving from the same records.
  """

  moves = pending_run_store_moves(workspace_dir)
  conflicts = [move for move in moves if _present(move.destination)]
  if conflicts:
    raise RunStoreRelocationConflict(
      "run control destinations already exist: "
      + ", ".join(str(move.destination) for move in conflicts)
    )
  done: list[RunStoreMove] = []
  try:
    for move in moves:
      move.destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
      os.rename(move.source, move.destination)
      done.append(move)
  except BaseException:
    for move in reversed(done):
      os.rename(move.destination, move.source)
    raise
  for parent in {move.destination.parent for move in done} | {move.source.parent for move in done}:
    _fsync_directory(parent)
  return tuple(done)


def _present(path: Path) -> bool:
  return path.is_symlink() or path.exists()


def _fsync_directory(directory: Path) -> None:
  if not directory.is_dir():
    return
  descriptor = os.open(directory, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


__all__ = [
  "RunStoreMove",
  "RunStoreRelocationConflict",
  "pending_run_store_moves",
  "relocate_workspace_run_stores",
  "run_control_root",
]
