"""Trusted placement state and the local single-writer user-runtime backend."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Iterator, Literal, Mapping, Protocol
from urllib.error import URLError
from urllib.request import urlopen

import httpx

from hank_contracts.user_claims import derive_user_claim_key

from .user_runtime_credentials import (
  DeskReplica,
  USER_RUNTIME_CREDENTIALS_FILE_ENV,
  UserRuntimeCredentials,
  UserRuntimeIdentity,
  load_user_runtime_credentials,
  mint_user_runtime_credentials,
  write_user_runtime_credentials,
)

RuntimeState = Literal["starting", "ready", "stopped", "replaced"]

# The runtime's writable state layout, relative to its placement root (the
# parent of USER_DATA_DIR). The product pins the approval stores under
# USER_DATA_DIR/gateway; logs, audit and telemetry sit beside the desk.
_RUNTIME_STATE_PATHS: Mapping[str, str] = {
  "GATEWAY_APPROVAL_DB_PATH": "user-data/gateway/approvals.sqlite3",
  "AGENT_AUTONOMOUS_APPROVALS_DB_PATH": "user-data/gateway/approvals.sqlite3",
  "AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH": "user-data/gateway/autonomous-approval-decisions.jsonl",
  "GATEWAY_APPROVAL_AUDIT_DIR": "user-data/audit/approvals",
  "COMMERCIAL_STATE_DIR": "user-data/commercial",
  "GATEWAY_LOG_DIR": "logs/gateway",
  "TOOL_TIMING_DB_PATH": "logs/gateway/cost.db",
  "GATEWAY_CHAT_LOG_DIR": "logs/chat",
  "AGENT_SESSION_LOG_BASE_DIR": "logs/sessions",
  "LOG_DIR": "logs/service",
  "MCP_CONFIG_TMP_DIR": "tmp/mcp",
  "MODEL_ENGINE_OVERRIDES_DIR": "model-engine-overrides",
  "GATEWAY_SURFACE_CONTENT_PATH": "telemetry/surface_content.jsonl",
}


def _now() -> float:
  return time.time()


@dataclass(frozen=True)
class RuntimePlacement:
  storage_id: str
  url: str
  state: RuntimeState
  backend: str
  handle: str | None
  bundle_path: str
  generation: int
  started_at: float
  updated_at: float


class RuntimePlacementStore:
  """Persist routing facts only; credentials stay in private bundle files."""

  def __init__(self, path: str | os.PathLike[str]) -> None:
    self.path = Path(path).expanduser().resolve()
    self.path.parent.mkdir(parents=True, exist_ok=True)
    with self._connection() as connection:
      connection.execute("PRAGMA journal_mode=WAL")
      connection.execute("""
        CREATE TABLE IF NOT EXISTS runtime_placements (
          storage_id TEXT PRIMARY KEY, url TEXT NOT NULL, state TEXT NOT NULL,
          backend TEXT NOT NULL, handle TEXT, bundle_path TEXT NOT NULL,
          generation INTEGER NOT NULL, started_at REAL NOT NULL, updated_at REAL NOT NULL
        )
      """)

  @contextmanager
  def _connection(self) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(self.path, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
      with connection:
        yield connection
    finally:
      connection.close()

  def get(self, storage_id: str) -> RuntimePlacement | None:
    with self._connection() as connection:
      row = connection.execute(
        "SELECT * FROM runtime_placements WHERE storage_id = ?", (storage_id,),
      ).fetchone()
    return RuntimePlacement(**dict(row)) if row is not None else None

  def upsert(self, placement: RuntimePlacement) -> RuntimePlacement:
    with self._connection() as connection:
      connection.execute("""
        INSERT INTO runtime_placements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(storage_id) DO UPDATE SET
          url=excluded.url, state=excluded.state, backend=excluded.backend,
          handle=excluded.handle, bundle_path=excluded.bundle_path,
          generation=excluded.generation, started_at=excluded.started_at,
          updated_at=excluded.updated_at
      """, (
        placement.storage_id, placement.url, placement.state, placement.backend,
        placement.handle, placement.bundle_path, placement.generation,
        placement.started_at, placement.updated_at,
      ))
    return placement

  def mark(self, storage_id: str, state: RuntimeState) -> RuntimePlacement | None:
    with self._connection() as connection:
      connection.execute(
        "UPDATE runtime_placements SET state = ?, updated_at = ? WHERE storage_id = ?",
        (state, _now(), storage_id),
      )
      row = connection.execute(
        "SELECT * FROM runtime_placements WHERE storage_id = ?", (storage_id,),
      ).fetchone()
    return RuntimePlacement(**dict(row)) if row is not None else None

  def list(self) -> list[RuntimePlacement]:
    with self._connection() as connection:
      rows = connection.execute("SELECT * FROM runtime_placements ORDER BY storage_id").fetchall()
    return [RuntimePlacement(**dict(row)) for row in rows]


class RuntimeBackend(Protocol):
  """Blocking placement operations; ASGI callers offload these to a worker thread."""

  store: RuntimePlacementStore

  def start(
    self, storage_id: str, identity: UserRuntimeIdentity, desk: DeskReplica,
  ) -> RuntimePlacement: ...

  def stop(self, placement: RuntimePlacement) -> RuntimePlacement: ...

  def replace(self, placement: RuntimePlacement) -> RuntimePlacement: ...

  def health(self, placement: RuntimePlacement) -> bool: ...

  def transport(self, placement: RuntimePlacement) -> httpx.AsyncBaseTransport: ...


def _storage_id(storage_id: str) -> str:
  # This path is a private data boundary, not a product admission policy.
  if (
    not isinstance(storage_id, str) or not storage_id.isascii()
    or not storage_id.isdigit() or storage_id.startswith("0")
  ):
    raise ValueError("storage_id must be a canonical positive numeric identity")
  return storage_id


class RuntimePlacementAuthority:
  """Shared placement locks and edge-owned credential bundle minting."""

  def __init__(self, *, placement_root: str | os.PathLike[str], org_claim_key: str) -> None:
    self.placement_root = Path(placement_root).expanduser().resolve()
    self.placement_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    self.store = RuntimePlacementStore(self.placement_root / "runtime_host.sqlite3")
    self._org_claim_key = org_claim_key
    self._locks_guard = threading.Lock()
    self._locks: dict[str, threading.RLock] = {}

  def _lock(self, storage_id: str) -> threading.RLock:
    _storage_id(storage_id)
    with self._locks_guard:
      return self._locks.setdefault(storage_id, threading.RLock())

  def _mint_bundle(self, storage_id: str, identity: UserRuntimeIdentity, desk: DeskReplica) -> Path:
    if type(identity.risk_user_id) is not int or storage_id != str(identity.risk_user_id):
      raise ValueError("storage_id must match identity.risk_user_id")
    _storage_id(storage_id)
    user_root = self.placement_root / storage_id
    user_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    bundle_path = user_root / "bundle.json"
    bundle = mint_user_runtime_credentials(
      org_claim_key=self._org_claim_key, identity=identity, storage_id=storage_id, desk=desk,
    )
    write_user_runtime_credentials(bundle, bundle_path)
    return bundle_path

  def _bundle_serves(self, running: UserRuntimeCredentials | None, identity: UserRuntimeIdentity) -> bool:
    """A running bundle is reusable only for the current identity under the current org key.

    Bundles store the derived claim key, so a rotated org key leaves every
    running runtime signing with a dead key until it is re-minted.
    """
    return (
      running is not None and running.user == identity
      and running.claim_signing.key == derive_user_claim_key(
        self._org_claim_key, identity.slug, identity.email,
      )
    )


class LocalProcessRuntimeBackend(RuntimePlacementAuthority):
  """Local process placement, not an OS isolation boundary for arbitrary code."""

  def __init__(
    self, *, placement_root: str | os.PathLike[str], org_claim_key: str,
    app: str = "api.main:app", python: str | os.PathLike[str] = sys.executable,
    extra_env: Mapping[str, str] | None = None, litestream_bin: str | None = None,
  ) -> None:
    super().__init__(placement_root=placement_root, org_claim_key=org_claim_key)
    self._app = app
    self._python = str(python)
    self._extra_env = dict(extra_env or {})
    self._litestream_bin = litestream_bin
    # This implementation deliberately boots the checkout's existing custody owner.
    self._checkout = Path(__file__).resolve().parents[3]
    self._scripts = self._checkout / "deploy" / "user-runtime"
    self._processes: dict[str, subprocess.Popen[bytes]] = {}

  def __repr__(self) -> str:
    return "LocalProcessRuntimeBackend(credentials=<redacted>)"


  def _environment(self, user_data_dir: Path) -> dict[str, str]:
    environ = dict(self._extra_env)
    for name in (
      "AGENT_API_USER_CLAIM_HMAC_KEY", "GATEWAY_USER_KEYS", "GATEWAY_API_KEY_WEB",
    ):
      environ.pop(name, None)
    environ["USER_DATA_DIR"] = str(user_data_dir)
    # Every writable state root is the runtime's own: an inherited shared-tier
    # path would put two users' logs, approvals or telemetry in one place.
    runtime_root = user_data_dir.parent
    for name, relative in _RUNTIME_STATE_PATHS.items():
      path = runtime_root / relative
      (path if name.endswith("_DIR") else path.parent).mkdir(parents=True, exist_ok=True, mode=0o700)
      environ[name] = str(path)
    return environ

  @staticmethod
  def _probe(url: str) -> bool:
    for path in ("/api/health", "/health"):
      try:
        with urlopen(url + path, timeout=1) as response:
          if response.status == 200:
            return True
      except (OSError, URLError):
        pass
    return False

  def health(self, placement: RuntimePlacement) -> bool:
    _storage_id(placement.storage_id)
    return placement.state == "ready" and self._probe(placement.url)

  def transport(self, placement: RuntimePlacement) -> httpx.AsyncBaseTransport:
    return httpx.AsyncHTTPTransport()

  def start(
    self, storage_id: str, identity: UserRuntimeIdentity, desk: DeskReplica,
  ) -> RuntimePlacement:
    with self._lock(storage_id):
      if type(identity.risk_user_id) is not int or storage_id != str(identity.risk_user_id):
        raise ValueError("storage_id must match identity.risk_user_id")
      existing = self.store.get(storage_id)
      if existing and existing.state == "ready" and self.health(existing):
        running = load_user_runtime_credentials({
          USER_RUNTIME_CREDENTIALS_FILE_ENV: existing.bundle_path,
        })
        # The caller's current authenticated identity and the current org key
        # are the authority: a rotated key, a revoked role, a changed email or
        # a rotated org signing key re-mints the runtime.
        if self._bundle_serves(running, identity) and running is not None and running.desk == desk:
          return existing
      if existing:
        self.stop(existing)
      user_root = self.placement_root / storage_id
      user_root.mkdir(parents=True, exist_ok=True, mode=0o700)
      user_data_dir = user_root / "user-data"
      (user_data_dir / "users" / storage_id).mkdir(parents=True, exist_ok=True, mode=0o700)
      bundle_path = self._mint_bundle(storage_id, identity, desk)
      environ = self._environment(user_data_dir)
      environ[USER_RUNTIME_CREDENTIALS_FILE_ENV] = str(bundle_path)
      if desk.aws_access_key_id is not None:
        # The user's own session credentials for its desk prefix (ruling 2);
        # restore and the replicator read them.
        environ["AWS_ACCESS_KEY_ID"] = desk.aws_access_key_id
        environ["AWS_SECRET_ACCESS_KEY"] = desk.aws_secret_access_key or ""
        if desk.aws_session_token is not None:
          environ["AWS_SESSION_TOKEN"] = desk.aws_session_token
      previous_port = int(existing.url.rsplit(":", 1)[1]) if existing else None
      while True:
        with socket.socket() as listener:
          listener.bind(("127.0.0.1", 0))
          port = listener.getsockname()[1]
        if port != previous_port:
          break
      timestamp = _now()
      placement = RuntimePlacement(
        storage_id=storage_id, url=f"http://127.0.0.1:{port}", state="starting",
        backend="local-process", handle=None, bundle_path=str(bundle_path),
        generation=existing.generation + 1 if existing else 1,
        started_at=timestamp, updated_at=timestamp,
      )
      self.store.upsert(placement)
      try:
        if self._litestream_bin is not None:
          desk_command = [
            self._python, str(self._scripts / "desk_sync.py"),
            "restore", "--storage-id", storage_id, "--user-data-dir", str(user_data_dir),
            "--replica", desk.url, "--litestream", self._litestream_bin,
          ]
          subprocess.run(
            desk_command, cwd=self._checkout, env=environ, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
          )
        boot_command = [
          self._python, str(self._scripts / "user_runtime_boot.py"),
          "--bundle", str(bundle_path), "--port", str(port), "--host", "127.0.0.1",
          "--app", self._app,
        ]
        if self._litestream_bin is not None:
          # The replica is the desk (ISOLATION-SHAPE-A ruling 1): Litestream
          # replicates while it supervises the runtime, and flushes when the
          # session group is terminated.
          boot_command = [
            self._python, str(self._scripts / "desk_sync.py"),
            "replicate", "--storage-id", storage_id, "--user-data-dir", str(user_data_dir),
            "--replica", desk.url, "--litestream", self._litestream_bin,
            "--exec", *boot_command,
          ]
        process = self._spawn(boot_command, environ)
        self._processes[storage_id] = process
        placement = replace(placement, handle=str(process.pid), updated_at=_now())
        self.store.upsert(placement)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
          if process.poll() is not None:
            raise RuntimeError("user runtime process exited before readiness")
          if self._probe(placement.url):
            return self.store.upsert(replace(placement, state="ready", updated_at=_now()))
          time.sleep(0.1)
        raise RuntimeError("user runtime did not become ready within 60 seconds")
      except BaseException:
        self.stop(placement)
        self.store.upsert(replace(placement, state="stopped", handle=None, updated_at=_now()))
        raise

  def _spawn(self, command: list[str], environ: Mapping[str, str]) -> subprocess.Popen[bytes]:
    # User apps may print credentials: their output goes to the runtime's own
    # log beside its desk, never into the edge's logs.
    log_path = Path(environ["GATEWAY_LOG_DIR"]).parent / "runtime.log"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
      return subprocess.Popen(
        command, cwd=self._checkout, env=environ, start_new_session=True,
        stdout=descriptor, stderr=subprocess.STDOUT,
      )
    finally:
      os.close(descriptor)

  @staticmethod
  def _terminate(pid: int, process: subprocess.Popen[bytes] | None = None) -> None:
    if process is not None and process.poll() is not None:
      return
    try:
      os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
      if process is not None:
        process.wait()
      return
    if process is not None:
      try:
        process.wait(timeout=10)
        return
      except subprocess.TimeoutExpired:
        os.killpg(pid, signal.SIGKILL)
        process.wait(timeout=5)
        return
    # A reloaded backend has only persisted pids. Reap if this is still our child,
    # otherwise wait for the process to disappear before another desk writer starts.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
      try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
        if waited == pid:
          return
      except ChildProcessError:
        pass
      try:
        os.kill(pid, 0)
      except ProcessLookupError:
        return
      time.sleep(0.05)
    try:
      os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
      return
    try:
      os.waitpid(pid, 0)
    except ChildProcessError:
      pass

  def stop(self, placement: RuntimePlacement) -> RuntimePlacement:
    storage_id = placement.storage_id
    with self._lock(storage_id):
      stored = self.store.get(storage_id)
      if stored and (stored.generation != placement.generation or stored.handle != placement.handle):
        # A stale caller does not own the replacement's process or routing row.
        return replace(placement, state="stopped", handle=None, updated_at=_now())
      process = self._processes.pop(storage_id, None)
      if process is not None:
        self._terminate(process.pid, process)
      elif placement.state in ("ready", "starting") and placement.handle:
        self._terminate(int(placement.handle))
      return self.store.upsert(replace(placement, state="stopped", handle=None, updated_at=_now()))

  def replace(self, placement: RuntimePlacement) -> RuntimePlacement:
    storage_id = placement.storage_id
    with self._lock(storage_id):
      stored = self.store.get(storage_id)
      if stored is None:
        raise KeyError(storage_id)
      placement = stored
      bundle = load_user_runtime_credentials({
        USER_RUNTIME_CREDENTIALS_FILE_ENV: placement.bundle_path,
      })
      assert bundle is not None
      if bundle.storage_id != storage_id or storage_id != str(bundle.user.risk_user_id):
        raise ValueError("placement bundle must match its storage identity")
      # Load before stopping; failed bundle reads must leave the original runtime intact.
      self.stop(placement)
      self.store.mark(storage_id, "replaced")
      return self.start(storage_id, bundle.user, bundle.desk)
