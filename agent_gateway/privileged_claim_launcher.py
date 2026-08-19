#!/usr/bin/python3
"""Root-only one-shot launcher for gateway claim-signing authority."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import pwd
import resource
import stat
import sys
from typing import NoReturn, Sequence


_SECRET_PATH = Path(
  "/etc/agent_gateway/agent-api-user-claim-hmac-key"
)
_TARGET_USER = "ubuntu"
_TARGET_ARGV = (
  "/var/www/agent_gateway/venv/bin/python3",
  "-m",
  "agent_gateway.gateway_server",
  "--workers",
  "1",
  "--timeout-keep-alive",
  "120",
  "--timeout-graceful-shutdown",
  "30",
)
_TARGET_ENV = {
  "PATH": (
    "/opt/hank/node-v24.16.0/bin:"
    "/var/www/agent_gateway/venv/bin:"
    "/usr/local/bin:/usr/bin:/bin"
  ),
  "PYTHONPATH": (
    "/var/www/agent_gateway:"
    "/var/www/agent_gateway/api"
  ),
  "CANVAS_BUILD_DIR": "/opt/hank/canvas-build",
  "CANVAS_NODE_BINARY": "/opt/hank/node-v24.16.0/bin/node",
  "BROKERAGE_CONNECT_REQUIRED": "true",
  "USER_DATA_DIR": "/mnt/hank-data/agent_gateway/data",
  "AGENT_SESSION_LOG_BASE_DIR": (
    "/mnt/hank-data/agent_gateway/data/agent-sessions"
  ),
  "AGENT_GATEWAY_SKILLS_DIR": (
    "/var/www/agent_gateway/api/memory/workspace/notes/skills"
  ),
  "GATEWAY_APPROVAL_DB_PATH": (
    "/mnt/hank-data/agent_gateway/data/gateway/approvals.sqlite3"
  ),
  "COMMERCIAL_STATE_DIR": (
    "/mnt/hank-data/agent_gateway/data/commercial"
  ),
  "GATEWAY_LOG_DIR": "/mnt/hank-data/agent_gateway/logs",
  "MCP_CONFIG_PATH": (
    "/var/www/agent_gateway/deploy/mcp.production.json"
  ),
  "MCP_CONFIG_TMP_DIR": "/mnt/hank-data/agent_gateway/tmp",
  "EXCEL_MCP_RELAY_STATE_DIR": (
    "/mnt/hank-data/agent_gateway/data/excel_mcp/relay"
  ),
  "EXCEL_MCP_RELAY_RESTART_LOCK_PATH": (
    "/run/agent-gateway-deploy/restart.lock"
  ),
  "EXCEL_MCP_RELAY_RESTART_LOCK_REQUIRED": "true",
  "GATEWAY_APPROVAL_AUDIT_DIR": (
    "/mnt/hank-data/agent_gateway/data/audit/approvals"
  ),
  "PRODUCT_ID": "hank",
}
_MIN_SECRET_BYTES = 32
_MAX_SECRET_BYTES = 4096
_PR_SET_DUMPABLE = 4
_PR_SET_KEEPCAPS = 8
_PR_SET_NO_NEW_PRIVS = 38
_PR_CAPBSET_DROP = 24
_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_CLEAR_ALL = 4


def _open_secret(
  path: Path,
  *,
  expected_owner_uid: int,
) -> int:
  if (
    not path.is_absolute()
    or path != Path(os.path.normpath(str(path)))
  ):
    raise RuntimeError(
      "claim-signing credential path is not canonical"
    )
  parent = path.parent
  try:
    resolved_parent = parent.resolve(strict=True)
    parent_info = parent.lstat()
    before = path.lstat()
  except OSError as exc:
    raise RuntimeError(
      "claim-signing credential is unavailable"
    ) from exc
  if (
    resolved_parent != parent
    or not stat.S_ISDIR(parent_info.st_mode)
    or parent_info.st_uid != expected_owner_uid
    or stat.S_IMODE(parent_info.st_mode) & 0o022
  ):
    raise RuntimeError(
      "claim-signing credential parent is untrusted"
    )
  if (
    stat.S_ISLNK(before.st_mode)
    or not stat.S_ISREG(before.st_mode)
    or before.st_uid != expected_owner_uid
    or stat.S_IMODE(before.st_mode) != 0o600
    or before.st_nlink != 1
  ):
    raise RuntimeError(
      "claim-signing credential is untrusted"
    )
  flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
  if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
  try:
    fd = os.open(path, flags)
    opened = os.fstat(fd)
  except OSError as exc:
    raise RuntimeError(
      "claim-signing credential could not be opened"
    ) from exc
  if (
    (opened.st_dev, opened.st_ino)
    != (before.st_dev, before.st_ino)
    or not stat.S_ISREG(opened.st_mode)
    or opened.st_uid != expected_owner_uid
    or stat.S_IMODE(opened.st_mode) != 0o600
    or opened.st_nlink != 1
  ):
    os.close(fd)
    raise RuntimeError(
      "claim-signing credential changed while opening"
    )
  return fd


def _consume_secret(
  path: Path,
  *,
  expected_owner_uid: int,
) -> bytes:
  fd = _open_secret(
    path,
    expected_owner_uid=expected_owner_uid,
  )
  try:
    value = os.read(fd, _MAX_SECRET_BYTES + 1)
    if os.read(fd, 1):
      raise RuntimeError(
        "claim-signing credential exceeds its byte bound"
      )
  except OSError as exc:
    raise RuntimeError(
      "claim-signing credential could not be read"
    ) from exc
  finally:
    os.close(fd)
  if not _MIN_SECRET_BYTES <= len(value) <= _MAX_SECRET_BYTES:
    raise RuntimeError(
      "claim-signing credential length is invalid"
    )
  try:
    text = value.decode("utf-8")
  except UnicodeDecodeError as exc:
    raise RuntimeError(
      "claim-signing credential is not UTF-8"
    ) from exc
  if text != text.strip() or "\x00" in text:
    raise RuntimeError(
      "claim-signing credential is not canonical"
    )
  return value


def _anonymous_secret_fd(secret: bytes) -> int:
  read_fd, write_fd = os.pipe2(
    getattr(os, "O_CLOEXEC", 0)
  )
  try:
    view = memoryview(secret)
    while view:
      written = os.write(write_fd, view)
      if written <= 0:
        raise RuntimeError(
          "anonymous claim-signing transfer made no progress"
        )
      view = view[written:]
  except BaseException:
    os.close(read_fd)
    raise
  finally:
    os.close(write_fd)
  os.set_inheritable(read_fd, True)
  if not os.get_inheritable(read_fd):
    os.close(read_fd)
    raise RuntimeError(
      "anonymous claim-signing descriptor is not inheritable"
    )
  return read_fd


def _prctl(option: int, value: int) -> None:
  libc = ctypes.CDLL(None, use_errno=True)
  result = libc.prctl(
    ctypes.c_int(option),
    ctypes.c_ulong(value),
    ctypes.c_ulong(0),
    ctypes.c_ulong(0),
    ctypes.c_ulong(0),
  )
  if result != 0:
    error = ctypes.get_errno()
    raise OSError(
      error,
      f"prctl({option}) failed",
    )


def _drop_privileges(user_name: str) -> None:
  if os.geteuid() != 0:
    raise RuntimeError(
      "privileged claim launcher must start as root"
    )
  account = pwd.getpwnam(user_name)
  os.initgroups(account.pw_name, account.pw_gid)
  resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
  for capability in range(64):
    try:
      _prctl(_PR_CAPBSET_DROP, capability)
    except OSError as exc:
      if exc.errno != 22:
        raise
  _prctl(_PR_CAP_AMBIENT, _PR_CAP_AMBIENT_CLEAR_ALL)
  _prctl(_PR_SET_KEEPCAPS, 0)
  os.setgid(account.pw_gid)
  os.setuid(account.pw_uid)
  _prctl(_PR_SET_NO_NEW_PRIVS, 1)
  _prctl(_PR_SET_DUMPABLE, 0)
  if (
    os.getuid() != account.pw_uid
    or os.geteuid() != account.pw_uid
    or os.getgid() != account.pw_gid
    or os.getegid() != account.pw_gid
  ):
    raise RuntimeError(
      "privileged claim launcher did not drop identity"
    )
  if resource.getrlimit(resource.RLIMIT_CORE) != (0, 0):
    raise RuntimeError(
      "privileged claim launcher did not disable core dumps"
    )
  status = Path("/proc/self/status").read_text(
    encoding="utf-8"
  )
  status_fields = {
    line.split(":", 1)[0]: line.split(":", 1)[1].strip()
    for line in status.splitlines()
    if ":" in line
  }
  if (
    status_fields.get("NoNewPrivs") != "1"
    or any(
      int(status_fields.get(field, "1"), 16) != 0
      for field in (
        "CapInh",
        "CapPrm",
        "CapEff",
        "CapBnd",
        "CapAmb",
      )
    )
  ):
    raise RuntimeError(
      "privileged claim launcher retained capabilities"
    )


def launch_with_claim_signing_fd(
  *,
  secret_path: Path,
  target_user: str,
  target_argv: Sequence[str],
  expected_owner_uid: int = 0,
) -> NoReturn:
  if (
    type(target_user) is not str
    or not target_user
    or not target_argv
    or any(
      type(arg) is not str or not arg or "\x00" in arg
      for arg in target_argv
    )
    or not Path(target_argv[0]).is_absolute()
  ):
    raise ValueError(
      "privileged claim launcher target is invalid"
    )
  secret = _consume_secret(
    Path(secret_path),
    expected_owner_uid=expected_owner_uid,
  )
  secret_fd = _anonymous_secret_fd(secret)
  del secret
  try:
    _drop_privileges(target_user)
    env = dict(os.environ)
    env.pop("AGENT_API_USER_CLAIM_HMAC_KEY", None)
    env.pop("CREDENTIALS_DIRECTORY", None)
    env.update(_TARGET_ENV)
    argv = [
      *target_argv,
      "--claim-signing-key-fd",
      str(secret_fd),
    ]
    os.execve(target_argv[0], argv, env)
  finally:
    os.close(secret_fd)
  raise AssertionError("unreachable")


def main() -> NoReturn:
  if len(sys.argv) != 1:
    raise SystemExit(
      "privileged claim launcher accepts no arguments"
    )
  launch_with_claim_signing_fd(
    secret_path=_SECRET_PATH,
    target_user=_TARGET_USER,
    target_argv=_TARGET_ARGV,
  )


if __name__ == "__main__":
  main()
