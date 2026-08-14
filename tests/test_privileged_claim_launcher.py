from __future__ import annotations

import json
import os
from pathlib import Path
import pwd
import stat
import subprocess
import sys

import pytest

from agent_gateway.privileged_claim_launcher import (
  _consume_secret,
)


_SECRET = b"privileged-launcher-test-key-at-least-32-bytes"


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
  target_user="nobody",
  target_argv=("/usr/bin/python3", "-c", sys.argv[2], sys.argv[1]),
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
