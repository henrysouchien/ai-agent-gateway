from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys

import pytest

import agent_gateway.privileged_claim_launcher as launcher_module
from agent_gateway.privileged_claim_launcher import (
  _SESSION_LOG_LAYOUT_PATH,
  _TARGET_ARGV,
  _TARGET_ENV,
  _consume_secret,
  _consume_session_log_layout,
  launch_with_claim_signing_fd,
)


_SECRET = b"privileged-launcher-test-key-at-least-32-bytes"


def test_production_target_bounds_uvicorn_graceful_shutdown() -> None:
  index = _TARGET_ARGV.index("--timeout-graceful-shutdown")
  assert _TARGET_ARGV[index + 1] == "30"


def test_production_target_uses_deployed_skill_catalog() -> None:
  assert _TARGET_ENV["AGENT_GATEWAY_SKILLS_DIR"] == (
    "/var/www/agent_gateway/api/memory/workspace/notes/skills"
  )


def test_production_target_keeps_foundation_session_log_layout_v1() -> None:
  repo_root = Path(__file__).resolve().parents[3]
  assert "AGENT_SESSION_LOG_LAYOUT" not in _TARGET_ENV
  assert _SESSION_LOG_LAYOUT_PATH == Path(
    "/etc/agent_gateway/session-log-layout"
  )
  assert (
    repo_root / "deploy" / "agent-session-log-layout.production"
  ).read_bytes() == b"v1\n"
  assert _TARGET_ENV["AGENT_SESSION_LOG_ARCHIVE_PRODUCT_IDS"] == "hank-dev"


def test_launcher_reads_only_canonical_private_source(
  tmp_path: Path,
) -> None:
  tmp_path.chmod(0o700)
  secret_path = tmp_path / "claim-key"
  secret_path.write_bytes(_SECRET)
  secret_path.chmod(0o600)

  assert _consume_secret(
    secret_path,
    expected_owner_uid=os.geteuid(),
  ) == _SECRET

  secret_path.chmod(0o640)
  with pytest.raises(RuntimeError, match="untrusted"):
    _consume_secret(
      secret_path,
      expected_owner_uid=os.geteuid(),
    )


@pytest.mark.parametrize(
  ("raw", "expected"),
  [(b"v1\n", "v1"), (b"v2\n", "v2")],
)
def test_launcher_accepts_exact_trusted_session_log_layout_choice(
  tmp_path: Path,
  raw: bytes,
  expected: str,
) -> None:
  tmp_path.chmod(0o700)
  layout_path = tmp_path / "session-log-layout"
  layout_path.write_bytes(raw)
  layout_path.chmod(0o600)

  assert _consume_session_log_layout(
    layout_path,
    expected_owner_uid=os.geteuid(),
  ) == expected


@pytest.mark.parametrize(
  "raw",
  [b"", b"v1", b"v2", b"v1\r\n", b"V1\n", b"v3\n", b"v1\nextra"],
)
def test_launcher_rejects_noncanonical_session_log_layout_choice(
  tmp_path: Path,
  raw: bytes,
) -> None:
  tmp_path.chmod(0o700)
  layout_path = tmp_path / "session-log-layout"
  layout_path.write_bytes(raw)
  layout_path.chmod(0o600)

  with pytest.raises(RuntimeError, match="session-log layout choice"):
    _consume_session_log_layout(
      layout_path,
      expected_owner_uid=os.geteuid(),
    )


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "writable"])
def test_launcher_rejects_untrusted_session_log_layout_source(
  tmp_path: Path,
  unsafe_kind: str,
) -> None:
  tmp_path.chmod(0o700)
  layout_path = tmp_path / "session-log-layout"
  target = tmp_path / "target"
  target.write_bytes(b"v1\n")
  target.chmod(0o600)
  if unsafe_kind == "symlink":
    layout_path.symlink_to(target)
  elif unsafe_kind == "hardlink":
    os.link(target, layout_path)
  else:
    layout_path.write_bytes(b"v1\n")
    layout_path.chmod(0o640)

  with pytest.raises(RuntimeError, match="untrusted"):
    _consume_session_log_layout(
      layout_path,
      expected_owner_uid=os.geteuid(),
    )


def test_launcher_ignores_ambient_layout_and_injects_only_trusted_choice(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  order: list[str] = []
  read_fd, write_fd = os.pipe()
  os.close(write_fd)
  captured: dict[str, object] = {}

  def consume_layout(*_args: object, **_kwargs: object) -> str:
    order.append("layout")
    return "v2"

  def consume_secret(*_args: object, **_kwargs: object) -> bytes:
    order.append("secret")
    return _SECRET

  def drop_privileges(_user: str) -> None:
    order.append("drop")

  def execve(path: str, argv: list[str], env: dict[str, str]) -> None:
    captured.update(path=path, argv=argv, env=env)
    raise RuntimeError("captured exec")

  monkeypatch.setenv("AGENT_SESSION_LOG_LAYOUT", "hostile-ambient")
  monkeypatch.setattr(
    launcher_module,
    "_consume_session_log_layout",
    consume_layout,
  )
  monkeypatch.setattr(launcher_module, "_consume_secret", consume_secret)
  monkeypatch.setattr(launcher_module, "_anonymous_secret_fd", lambda _value: read_fd)
  monkeypatch.setattr(launcher_module, "_drop_privileges", drop_privileges)
  monkeypatch.setattr(launcher_module.os, "execve", execve)

  with pytest.raises(RuntimeError, match="captured exec"):
    launch_with_claim_signing_fd(
      secret_path=Path("/trusted/secret"),
      session_log_layout_path=Path("/trusted/layout"),
      target_user="service-user",
      target_argv=("/trusted/python", "-m", "gateway"),
    )

  assert order == ["layout", "secret", "drop"]
  assert captured["path"] == "/trusted/python"
  assert isinstance(captured["env"], dict)
  assert captured["env"]["AGENT_SESSION_LOG_LAYOUT"] == "v2"


def _root_launcher_test_available() -> bool:
  if sys.platform != "linux" or os.geteuid() != 0:
    return False
  try:
    pwd.getpwnam("nobody")
  except KeyError:
    return False
  status = Path("/proc/self/status").read_text(encoding="utf-8")
  effective = next(
    (
      int(line.split(":", 1)[1].strip(), 16)
      for line in status.splitlines()
      if line.startswith("CapEff:")
    ),
    0,
  )
  return bool(effective & (1 << 8))


@pytest.mark.skipif(
  not _root_launcher_test_available(),
  reason="requires Linux root with CAP_SETPCAP",
)
def test_real_dropped_child_cannot_read_named_or_parent_authority(
  tmp_path: Path,
) -> None:
  tmp_path.chmod(0o700)
  secret_path = tmp_path / "claim-key"
  secret_path.write_bytes(_SECRET)
  secret_path.chmod(0o600)
  layout_path = tmp_path / "session-log-layout"
  layout_path.write_bytes(b"v1\n")
  layout_path.chmod(0o600)
  assert secret_path.stat().st_uid == 0
  assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
  repo_root = Path(__file__).resolve().parents[3]
  probe_source = r"""
import ctypes
import json
import os
import resource
import sys

secret_path = sys.argv[1]
secret_fd = int(sys.argv[3])
secret = os.read(secret_fd, 4097)
os.close(secret_fd)
parent_pid = os.getpid()
read_fd, write_fd = os.pipe()
child_pid = os.fork()
if child_pid == 0:
  os.close(read_fd)
  def readable(path):
    try:
      fd = os.open(path, os.O_RDONLY)
    except OSError:
      return False
    else:
      os.close(fd)
      return True
  result = {
    "env_key": "AGENT_API_USER_CLAIM_HMAC_KEY" in os.environ,
    "root_source_readable": readable(secret_path),
    "systemd_credential_readable": readable(
      "/run/credentials/gateway.service/agent-api-user-claim-hmac-key"
    ),
    "parent_mem_readable": readable(f"/proc/{parent_pid}/mem"),
    "parent_secret_fd_readable": readable(f"/proc/{parent_pid}/fd/{secret_fd}"),
  }
  os.write(write_fd, json.dumps(result, sort_keys=True).encode("utf-8"))
  os.close(write_fd)
  os._exit(0)
os.close(write_fd)
child_result = json.loads(os.read(read_fd, 65536))
os.close(read_fd)
_, status = os.waitpid(child_pid, 0)
proc_status = open("/proc/self/status", encoding="utf-8").read()
fields = {
  line.split(":", 1)[0]: line.split(":", 1)[1].strip()
  for line in proc_status.splitlines()
  if ":" in line
}
libc = ctypes.CDLL(None, use_errno=True)
print(json.dumps({
  "caps_zero": all(
    int(fields[name], 16) == 0
    for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
  ),
  "child": child_result,
  "child_status": status,
  "core_limit": list(resource.getrlimit(resource.RLIMIT_CORE)),
  "dumpable": libc.prctl(3, 0, 0, 0, 0),
  "no_new_privs": fields["NoNewPrivs"],
  "secret_matches": secret == b"privileged-launcher-test-key-at-least-32-bytes",
}, sort_keys=True))
"""
  wrapper_source = r"""
from pathlib import Path
import sys
from agent_gateway.privileged_claim_launcher import (
  launch_with_claim_signing_fd,
)

launch_with_claim_signing_fd(
  secret_path=Path(sys.argv[1]),
  session_log_layout_path=Path(sys.argv[2]),
  target_user="nobody",
  target_argv=("/usr/bin/python3", "-c", sys.argv[3], sys.argv[1]),
)
"""
  env = dict(os.environ)
  env["AGENT_API_USER_CLAIM_HMAC_KEY"] = "must-be-scrubbed"
  env["PYTHONPATH"] = os.pathsep.join((
    str(repo_root / "packages" / "agent-gateway"),
    str(repo_root),
  ))
  completed = subprocess.run(
    [
      sys.executable,
      "-c",
      wrapper_source,
      str(secret_path),
      str(layout_path),
      probe_source,
    ],
    cwd="/",
    env=env,
    check=True,
    capture_output=True,
    text=True,
    timeout=15,
  )
  result = json.loads(completed.stdout)
  assert result == {
    "caps_zero": True,
    "child": {
      "env_key": False,
      "parent_mem_readable": False,
      "parent_secret_fd_readable": False,
      "root_source_readable": False,
      "systemd_credential_readable": False,
    },
    "child_status": 0,
    "core_limit": [0, 0],
    "dumpable": 0,
    "no_new_privs": "1",
    "secret_matches": True,
  }
