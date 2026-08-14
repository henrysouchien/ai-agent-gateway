from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from agent_gateway.approval_store import SQLiteApprovalStore


def test_approval_store_creates_private_inode_and_sidecars(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"

  prior_umask = os.umask(0o022)
  try:
    store = SQLiteApprovalStore(path)
  finally:
    os.umask(prior_umask)

  file_stat = path.stat()
  assert stat.S_IMODE(file_stat.st_mode) == 0o600
  assert file_stat.st_uid == os.geteuid()
  assert file_stat.st_nlink == 1

  with store._connection():
    for suffix in ("-wal", "-shm"):
      sidecar_stat = Path(f"{path}{suffix}").stat()
      assert stat.S_IMODE(sidecar_stat.st_mode) == 0o600
      assert sidecar_stat.st_uid == os.geteuid()
      assert sidecar_stat.st_nlink == 1


def test_approval_store_rejects_bound_inode_replacement(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  store = SQLiteApprovalStore(path)
  original = path.stat()
  os.rename(path, tmp_path / "approvals.original.sqlite3")
  replacement_fd = os.open(
    path,
    os.O_RDWR | os.O_CREAT | os.O_EXCL,
    0o600,
  )
  os.close(replacement_fd)

  with pytest.raises(RuntimeError, match="identity"):
    store._connect()

  assert path.stat().st_ino != original.st_ino


def test_signed_approval_store_rejects_permission_drift(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  SQLiteApprovalStore(path)
  identity = path.stat()
  path.chmod(0o640)

  with pytest.raises(RuntimeError, match="permissions"):
    SQLiteApprovalStore(
      path,
      expected_device=identity.st_dev,
      expected_inode=identity.st_ino,
    )


def test_approval_store_rejects_writable_parent(
  tmp_path: Path,
) -> None:
  parent = tmp_path / "unsafe"
  parent.mkdir(mode=0o700)
  parent.chmod(0o777)

  with pytest.raises(RuntimeError, match="parent"):
    SQLiteApprovalStore(parent / "approvals.sqlite3")


def test_approval_store_rejects_missing_parent_without_creating_it(
  tmp_path: Path,
) -> None:
  parent = tmp_path / "missing"

  with pytest.raises(RuntimeError, match="parent is unavailable"):
    SQLiteApprovalStore(parent / "approvals.sqlite3")

  assert not parent.exists()


def test_approval_store_rejects_sidecar_symlink(
  tmp_path: Path,
) -> None:
  path = tmp_path / "approvals.sqlite3"
  target = tmp_path / "target"
  target.write_bytes(b"")
  Path(f"{path}-wal").symlink_to(target)

  with pytest.raises(RuntimeError, match="unavailable"):
    SQLiteApprovalStore(path)
