"""Trusted placement state and the local single-writer user-runtime backend."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import logging
import os
from pathlib import Path
import signal
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Iterator, Literal, Mapping, Protocol
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from hank_contracts.user_claims import derive_user_claim_key

from hank_model_transport.user_runtime_credentials import (
  DeskReplica,
  USER_RUNTIME_CREDENTIALS_FILE_ENV,
  USER_STORAGE_ID_ENV,
  UserRuntimeCredentials,
  UserRuntimeIdentity,
  derive_approval_claim_key,
  load_user_runtime_credentials,
  mint_user_runtime_credentials,
  write_user_runtime_credentials,
)

log = logging.getLogger(__name__)

RuntimeState = Literal["starting", "ready", "stopped", "replaced"]

DESK_LEASE_SECONDS = 90.0
LEASE_REFUSED_EXIT = 3


class RestoreEpoch(BaseModel):
  """The writer epoch acquired by desk_sync before runtime placement."""

  model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
  epoch: int = Field(ge=1)


class WriterRecord(BaseModel):
  """The replica writer record carried from desk_sync to the placement edge."""

  model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
  epoch: int = Field(ge=1)
  holder: str
  expires_at: float
  state: Literal["held", "released"]


class RestoreRefused(BaseModel):
  """A refused restore's current holder, absent only when no holder was read."""

  model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
  refused: WriterRecord | None


RestoreResult = TypeAdapter(RestoreEpoch | RestoreRefused)


class DeskLeaseRefused(RuntimeError):
  """The replica refused a runtime's writer lease before it could start."""

  def __init__(self, message: str, record: WriterRecord | None) -> None:
    super().__init__(message)
    self.record = record


# How long a caller waits after SIGTERM to a placement's desk_sync process group
# before SIGKILL. A stop is graceful only if this covers everything desk_sync does
# after the signal: the gateway drains up to its uvicorn graceful-shutdown bound
# (30 s: the `timeout_graceful_shutdown` default in gateway_server.py, which
# user_runtime_boot.py leaves unset), then desk_sync runs its final file pass
# (allowance below) and stops Litestream with SIGTERM, waiting up to
# LITESTREAM_STOP_TIMEOUT (10 s, deploy/user-runtime/desk_sync.py). 30 + 20 + 10.
# The AgentCore adapter's stop_product imports this one constant.
PLACEMENT_STOP_GRACE_SECONDS = 30 + 20 + 10

# A local placement cannot outlive the edge that started it. Its stdin is the
# read end of a pipe whose only write end the edge holds; this preamble runs
# first, forks one watcher in the placement's own session, and execs the boot
# command unchanged (same pid, same birth token). However the edge goes away,
# the kernel closes the write end: the watcher reads EOF and stops the session
# group the way `_terminate` does (SIGTERM, the stop grace, then SIGKILL if the
# leader still runs). `stop()` closes the write end after its own stop.
_EDGE_LIFELINE = """\
import os, signal, sys, time
grace = float(sys.argv[1])
leader = os.getpid()
if os.fork() == 0:
  try:
    while os.read(0, 4096):
      pass
  except OSError:
    pass
  signal.signal(signal.SIGTERM, signal.SIG_IGN)
  os.killpg(0, signal.SIGTERM)
  deadline = time.monotonic() + grace
  while os.getppid() == leader and time.monotonic() < deadline:
    time.sleep(0.1)
  if os.getppid() == leader:
    os.killpg(0, signal.SIGKILL)
  os._exit(0)
null = os.open(os.devnull, os.O_RDONLY)
os.dup2(null, 0)
os.close(null)
os.execvp(sys.argv[2], sys.argv[2:])
"""

def _personal_credential_env_names() -> frozenset[str]:
  from hank_model_transport.providers.anthropic_oauth import ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES, ANTHROPIC_STORE_BASE_ENV
  from hank_model_transport.providers.codex import CODEX_PERSONAL_OAUTH_ENV_NAMES
  from hank_model_transport.providers.copilot_oauth import COPILOT_PERSONAL_OAUTH_ENV_NAMES
  from hank_model_transport.providers.openai import OPENAI_PERSONAL_OAUTH_ENV_NAMES
  from hank_model_transport.providers.google_antigravity_oauth import GOOGLE_ANTIGRAVITY_PERSONAL_OAUTH_ENV_NAMES
  from hank_model_transport.providers.google_oauth import GOOGLE_PERSONAL_OAUTH_ENV_NAMES
  from hank_model_transport.providers.kimi_code_oauth import KIMI_CODE_PERSONAL_OAUTH_ENV_NAMES
  from hank_model_transport.providers.muse_oauth import MUSE_PERSONAL_OAUTH_ENV_NAMES
  from hank_model_transport.providers.xai_oauth import XAI_PERSONAL_OAUTH_ENV_NAMES

  # Environment names that carry one person's OAuth identity or store path,
  # exported beside each provider adapter; `api/credentials.py` reads OpenAI/Codex.
  # A placement that does not admit the operator's seats
  # (`scope_admits_operator_seats`) must not inherit them; org API keys are not in
  # this set. `ANTHROPIC_STORE_BASE_ENV` (`USER_DATA_DIR`) is dropped from the pool
  # names because `_environment` rebinds it per placement.
  return frozenset().union(
    ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES - {ANTHROPIC_STORE_BASE_ENV},
    XAI_PERSONAL_OAUTH_ENV_NAMES,
    KIMI_CODE_PERSONAL_OAUTH_ENV_NAMES,
    GOOGLE_ANTIGRAVITY_PERSONAL_OAUTH_ENV_NAMES,
    GOOGLE_PERSONAL_OAUTH_ENV_NAMES,
    COPILOT_PERSONAL_OAUTH_ENV_NAMES,
    MUSE_PERSONAL_OAUTH_ENV_NAMES,
    OPENAI_PERSONAL_OAUTH_ENV_NAMES,
    CODEX_PERSONAL_OAUTH_ENV_NAMES,
  )

# Approval state stays at the edge; placed processes must not inherit local pins.
APPROVAL_LOCAL_PATH_ENV_NAMES = (
  "GATEWAY_APPROVAL_DB_PATH",
  "AGENT_AUTONOMOUS_APPROVALS_DB_PATH",
  "AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH",
  "GATEWAY_APPROVAL_AUDIT_DIR",
)

# The runtime's writable state layout, relative to its placement root (the
# parent of USER_DATA_DIR). Approval stores and their audit remain edge-owned.
_RUNTIME_STATE_PATHS: Mapping[str, str] = {
  "COMMERCIAL_STATE_DIR": "user-data/commercial",
  "GATEWAY_LOG_DIR": "logs/gateway",
  "TOOL_TIMING_DB_PATH": "logs/gateway/cost.db",
  "GATEWAY_CHAT_LOG_DIR": "logs/chat",
  "LOG_DIR": "logs/service",
  "MCP_CONFIG_TMP_DIR": "tmp/mcp",
  "MODEL_ENGINE_OVERRIDES_DIR": "model-engine-overrides",
  "GATEWAY_SURFACE_CONTENT_PATH": "telemetry/surface_content.jsonl",
}

# The user's own data, relative to the desk root: what deploy/user-runtime/
# desk_sync.py replicates and restores. Investment private state is the desk's
# ``investment-tools`` directory: the placement sets INVESTMENT_TOOLS_STATE_DIR to
# it as a locator (investment_state_dir) but never creates it, since a fresh state
# root must be absent for investment_tools state_paths to initialise it.
_DESK_STATE_PATHS: Mapping[str, str] = {
  "AGENT_SESSION_LOG_BASE_DIR": "agent-sessions",
}


def desk_root(user_data_dir: Path, storage_id: str) -> Path:
  """The one directory a user's desk replicates: ``USER_DATA_DIR/users/<storage_id>``."""
  return user_data_dir / "users" / storage_id


def investment_state_dir(user_data_dir: Path, storage_id: str) -> Path:
  """The desk's Investment private-state root, ``<desk root>/investment-tools``.

  The placement owns this layout and hands it to the runtime and its children as an
  explicit ``INVESTMENT_TOOLS_STATE_DIR``. It is a locator only: nothing here creates it.
  """
  return desk_root(user_data_dir, storage_id) / "investment-tools"


def runtime_state_environment(user_data_dir: Path, storage_id: str) -> dict[str, str]:
  """Every writable state root of one placed runtime, created and as env values.

  Everything is the runtime's own: an inherited shared-tier path would put two
  users' logs or telemetry in one place. User data (session
  journals) sits under the desk root, the rest under the placement root, the
  parent of ``user_data_dir``. The storage id is exported too, and the
  Investment private-state root as ``INVESTMENT_TOOLS_STATE_DIR`` (the desk's
  ``investment-tools``, see ``investment_state_dir``): that locator is the only
  way investment_tools places its state, in this runtime and in every child
  that inherits it. Investment state is the one root not created here;
  investment_tools initialises it on first use.
  """
  runtime_root = user_data_dir.parent
  paths = {name: runtime_root / relative for name, relative in _RUNTIME_STATE_PATHS.items()}
  desk = desk_root(user_data_dir, storage_id)
  paths.update({name: desk / relative for name, relative in _DESK_STATE_PATHS.items()})
  environ: dict[str, str] = {
    USER_STORAGE_ID_ENV: storage_id,
    "INVESTMENT_TOOLS_STATE_DIR": str(investment_state_dir(user_data_dir, storage_id)),
  }
  for name, path in paths.items():
    (path if name.endswith("_DIR") else path.parent).mkdir(parents=True, exist_ok=True, mode=0o700)
    environ[name] = str(path)
  return environ


def _now() -> float:
  return time.time()


def parse_positive_pid(raw: object) -> int:
  """The one pid shape the system signals or records: a positive int32."""
  if type(raw) is int and 0 < raw <= 0x7FFFFFFF:
    return raw
  raise ValueError(f"pid must be a positive 32-bit integer, got {raw!r}")


def _linux_proc_stat_birth_token(raw: str) -> str:
  """Return Linux ``/proc/<pid>/stat`` field 22, or an empty token."""

  if not isinstance(raw, str):
    return ""
  comm_end = raw.rfind(")")
  if comm_end < 0:
    return ""
  fields_from_state = raw[comm_end + 1 :].split()
  if len(fields_from_state) <= 19:
    return ""
  token = fields_from_state[19]
  return token if token.isdecimal() else ""


def pid_birth_token(
  pid: int, *, platform: str | None = None, run_ps: Any = None, path_cls: Any = Path,
) -> str:
  """Process-birth identity; empty string when unavailable or pid is not a positive int32."""

  if type(pid) is not int or not (0 < pid <= 0x7FFFFFFF):
    return ""
  if (sys.platform if platform is None else platform) == "linux":
    try:
      raw = path_cls(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
      return ""
    return _linux_proc_stat_birth_token(raw)
  try:
    out = (subprocess.run if run_ps is None else run_ps)(
      ["ps", "-p", str(pid), "-o", "lstart="],
      capture_output=True, text=True, timeout=5,
    )
  except (OSError, subprocess.SubprocessError):
    return ""
  return out.stdout.strip()


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
  handle_birth: str | None = None


@dataclass(frozen=True)
class ApprovalLedgerMigration:
  source: Path
  destination: Path
  pending_count: int
  placement: RuntimePlacement


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
          generation INTEGER NOT NULL, started_at REAL NOT NULL, updated_at REAL NOT NULL,
          handle_birth TEXT
        )
      """)
      columns = {row[1] for row in connection.execute("PRAGMA table_info(runtime_placements)").fetchall()}
      if "handle_birth" not in columns:
        # One-time migration: pre-existing rows have no recorded spawn birth
        # token (and under rule 7 a deploy drains live work before restart),
        # so their handle_birth stays NULL (unverified) and stop never signals them.
        connection.execute("ALTER TABLE runtime_placements ADD COLUMN handle_birth TEXT")
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
        INSERT INTO runtime_placements (
          storage_id, url, state, backend, handle, bundle_path, generation,
          started_at, updated_at, handle_birth
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(storage_id) DO UPDATE SET
          url=excluded.url, state=excluded.state, backend=excluded.backend,
          handle=excluded.handle, bundle_path=excluded.bundle_path,
          generation=excluded.generation, started_at=excluded.started_at,
          updated_at=excluded.updated_at, handle_birth=excluded.handle_birth
      """, (
        placement.storage_id, placement.url, placement.state, placement.backend,
        placement.handle, placement.bundle_path, placement.generation,
        placement.started_at, placement.updated_at, placement.handle_birth,
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


def _fsync_directory(path: Path) -> None:
  fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
  try:
    os.fsync(fd)
  finally:
    os.close(fd)


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

  def _mint_bundle(
    self, storage_id: str, identity: UserRuntimeIdentity, desk: DeskReplica, generation: int,
  ) -> Path:
    if type(identity.risk_user_id) is not int or storage_id != str(identity.risk_user_id):
      raise ValueError("storage_id must match identity.risk_user_id")
    _storage_id(storage_id)
    user_root = self.placement_root / storage_id
    user_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    bundle_path = user_root / "bundle.json"
    bundle = mint_user_runtime_credentials(
      org_claim_key=self._org_claim_key, identity=identity, storage_id=storage_id, desk=desk,
      generation=generation,
    )
    write_user_runtime_credentials(bundle, bundle_path)
    return bundle_path

  def _bundle_serves(
    self, running: UserRuntimeCredentials | None, identity: UserRuntimeIdentity,
    placement: RuntimePlacement,
  ) -> bool:
    """Reuse only the current generation and both identity-bound signing keys.

    Replacement rotates approval authority; org-key rotation rotates both keys.
    """
    return (
      running is not None and running.user == identity
      and running.generation == placement.generation
      and running.claim_signing.key == derive_user_claim_key(
        self._org_claim_key, identity.slug, identity.email,
      )
      and running.approval_claim_signing.key == derive_approval_claim_key(
        self._org_claim_key, identity.slug, identity.email, placement.generation,
      )
    )


class LocalProcessRuntimeBackend(RuntimePlacementAuthority):
  """Local process placement, not an OS isolation boundary for arbitrary code."""

  def __init__(
    self, *, placement_root: str | os.PathLike[str], org_claim_key: str,
    app: str = "api.main:app", python: str | os.PathLike[str] = sys.executable,
    extra_env: Mapping[str, str] | None = None, litestream_bin: str | None = None,
    stop_grace: float = PLACEMENT_STOP_GRACE_SECONDS,
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
    # storage_id -> the write end of that placement's lifeline (_EDGE_LIFELINE).
    self._lifelines: dict[str, int] = {}
    # A stop proves only this generation; restore consumes the proof once.
    self._confirmed_stops: dict[str, int] = {}
    self._stop_grace = stop_grace

  def __repr__(self) -> str:
    return "LocalProcessRuntimeBackend(credentials=<redacted>)"

  def _environment(self, user_data_dir: Path, storage_id: str) -> dict[str, str]:
    from .approval_remote import APPROVALS_EGRESS_URL_ENV

    environ = dict(self._extra_env)
    for name in (
      *APPROVAL_LOCAL_PATH_ENV_NAMES,
      "AGENT_API_USER_CLAIM_HMAC_KEY", "GATEWAY_USER_KEYS", "GATEWAY_API_KEY_WEB",
    ):
      environ.pop(name, None)
    from hank_model_transport.providers.credential_pool import scope_admits_operator_seats

    if not scope_admits_operator_seats(storage_id, environ=self._extra_env):
      for name in _personal_credential_env_names():
        environ.pop(name, None)
    environ["USER_DATA_DIR"] = str(user_data_dir)
    environ["MEMORY_STARTUP_USER_ID"] = storage_id
    approval_url = environ.get(APPROVALS_EGRESS_URL_ENV)
    if not approval_url:
      raise ValueError(f"{APPROVALS_EGRESS_URL_ENV} is required for local placement")
    try:
      parsed = urlsplit(approval_url)
      if (
        parsed.scheme not in ("http", "https") or not parsed.hostname
        or parsed.port == 0 or "?" in approval_url or "#" in approval_url
        or any(character.isspace() for character in approval_url)
      ):
        raise ValueError("invalid approval egress URL")
    except ValueError as exc:
      raise ValueError(
        f"{APPROVALS_EGRESS_URL_ENV} must be an explicit absolute http(s) URL without query or fragment"
      ) from exc
    environ[APPROVALS_EGRESS_URL_ENV] = approval_url
    environ["HANK_APPROVAL_TRANSPORT"] = "local"

    environ.update(runtime_state_environment(user_data_dir, storage_id))
    return environ

  def _seed_org_credentials(
    self, user_data_dir: Path, environ: dict[str, str], storage_id: str,
  ) -> None:
    """Seed the placement's private org root from the edge's own stores.

    Sources are the edge's `USER_DATA_DIR` store, then the operator's
    `~/.agent_gateway`; an existing target is never overwritten and nothing
    lands under `users/<sid>/`. Anthropic copies only non-rotating entries.
    Kimi and xAI OAuth hold rotating grants, so they are pointed at in place
    unless the child's environment already names a store. Operator personal
    seats belong to deployment owners: for any other `storage_id` they are
    neither copied nor bound, and org/API-key stores still seed.
    """
    from hank_model_transport.providers.anthropic_oauth import seed_non_rotating_org_store
    from hank_model_transport.providers.credential_pool import (
      first_org_credential_source,
      scope_admits_operator_seats,
      seed_org_credential_stores,
      store_base_dir,
    )

    bases: list[Path] = []
    for base in (store_base_dir(self._extra_env), Path.home() / ".agent_gateway"):
      if base not in bases:
        bases.append(base)
    sources = tuple(bases)
    seed_org_credential_stores(
      sources, user_data_dir, user_scope=storage_id, environ=self._extra_env,
    )
    seed_non_rotating_org_store(
      first_org_credential_source(sources, "anthropic", "oauth.json"),
      user_data_dir / "anthropic" / "oauth.json",
      user_scope=storage_id, environ=self._extra_env,
    )
    kimi_xai = (
      ("kimi_code", ("KIMI_CODE_AUTH_STORE_PATH", "KIMI_AUTH_STORE_PATH")),
      ("xai", ("XAI_AUTH_STORE_PATH",)),
    )
    if not scope_admits_operator_seats(storage_id, environ=self._extra_env):
      # An inherited path would still be a live binding of the operator's seat.
      for _, names in kimi_xai:
        for name in names:
          environ.pop(name, None)
      return
    for family, names in kimi_xai:
      if any(str(environ.get(name) or "").strip() for name in names):
        continue
      source = first_org_credential_source(sources, family, "oauth.json")
      if source is not None:
        environ[names[0]] = str(source)

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
    process = self._processes.get(placement.storage_id)
    if process is not None and placement.handle == str(process.pid) and process.poll() is not None:
      return False
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
      running: UserRuntimeCredentials | None = None
      if existing and existing.state == "ready" and self.health(existing):
        running = load_user_runtime_credentials({
          USER_RUNTIME_CREDENTIALS_FILE_ENV: existing.bundle_path,
        })
        # The caller's current authenticated identity and the current org key
        # are the authority: a rotated key, a revoked role, a changed email or
        # a rotated org signing key re-mints the runtime.
        if self._bundle_serves(running, identity, existing) and running is not None and running.desk == desk:
          return existing
      if existing:
        self.stop(existing)
      user_root = self.placement_root / storage_id
      user_root.mkdir(parents=True, exist_ok=True, mode=0o700)
      user_data_dir = user_root / "user-data"
      desk_root(user_data_dir, storage_id).mkdir(parents=True, exist_ok=True, mode=0o700)
      environ = self._environment(user_data_dir, storage_id)
      self._seed_org_credentials(user_data_dir, environ, storage_id)
      if desk.aws_access_key_id is not None:
        # The user's own session credentials for its desk prefix (ruling 2);
        # restore and the replicator read them.
        environ["AWS_ACCESS_KEY_ID"] = desk.aws_access_key_id
        environ["AWS_SECRET_ACCESS_KEY"] = desk.aws_secret_access_key or ""
        if desk.aws_session_token is not None:
          environ["AWS_SESSION_TOKEN"] = desk.aws_session_token
      desk_command = [
        self._python, str(self._scripts / "desk_sync.py"),
        "restore", "--storage-id", storage_id, "--user-data-dir", str(user_data_dir),
        "--replica", desk.url, "--lease-seconds", str(DESK_LEASE_SECONDS),
      ]
      takeover_epoch = self._confirmed_stops.pop(storage_id, None)
      if existing is not None and takeover_epoch == existing.generation:
        if running is None:
          running = load_user_runtime_credentials({
            USER_RUNTIME_CREDENTIALS_FILE_ENV: existing.bundle_path,
          })
        if running is not None and running.desk.url == desk.url:
          desk_command.extend(("--takeover-epoch", str(takeover_epoch)))
      if self._litestream_bin is not None:
        desk_command.extend(("--litestream", self._litestream_bin))
      restored = self._run_restore(desk_command, environ)
      if restored.returncode == LEASE_REFUSED_EXIT:
        refused = RestoreResult.validate_json(restored.stdout)
        assert isinstance(refused, RestoreRefused)
        raise DeskLeaseRefused(restored.stderr, record=refused.refused)
      # Any other failure is the restore's own error, raised with its output, never a parse error.
      restored.check_returncode()
      won = RestoreResult.validate_json(restored.stdout)
      assert isinstance(won, RestoreEpoch)
      generation = won.epoch
      bundle_path = self._mint_bundle(storage_id, identity, desk, generation)
      environ[USER_RUNTIME_CREDENTIALS_FILE_ENV] = str(bundle_path)
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
        generation=generation,
        started_at=timestamp, updated_at=timestamp,
      )
      self.store.upsert(placement)
      # The placement's own origin: everything in this runtime that asks "where is
      # my gateway" (gateway_address) reaches this placement, never the edge's
      # inherited value. Imported here, as the seed's provider imports are, so the
      # sparse launch fixture needs no extra module.
      from .gateway_address import GATEWAY_BASE_URL_ENV

      environ[GATEWAY_BASE_URL_ENV] = placement.url
      try:
        boot_command = [
          self._python, str(self._scripts / "user_runtime_boot.py"),
          "--bundle", str(bundle_path), "--port", str(port), "--host", "127.0.0.1",
          "--app", self._app,
        ]
        # desk_sync owns the writer lease even without Litestream: a runtime
        # must never run outside the replica's epoch supervisor.
        boot_command = [
          self._python, str(self._scripts / "desk_sync.py"),
          "replicate", "--storage-id", storage_id, "--user-data-dir", str(user_data_dir),
          "--replica", desk.url, "--epoch", str(generation),
          "--lease-seconds", str(DESK_LEASE_SECONDS),
          *(
            ["--litestream", self._litestream_bin]
            if self._litestream_bin is not None else []
          ),
          "--exec", *boot_command,
        ]
        process = self._spawn(storage_id, boot_command, environ)
        self._processes[storage_id] = process
        birth = pid_birth_token(process.pid) or None
        placement = replace(placement, handle=str(process.pid), handle_birth=birth, updated_at=_now())
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
        self.store.upsert(
          replace(placement, state="stopped", handle=None, handle_birth=None, updated_at=_now()),
        )
        raise

  def _run_restore(
    self, command: list[str], environ: Mapping[str, str],
  ) -> subprocess.CompletedProcess[str]:
    # desk_sync restore: wins (or is refused) the desk's writer lease, then restores.
    return subprocess.run(command, cwd=self._checkout, env=environ, capture_output=True, text=True)

  def _spawn(
    self, storage_id: str, command: list[str], environ: Mapping[str, str],
  ) -> subprocess.Popen[bytes]:
    # User apps may print credentials: their output goes to the runtime's own
    # log beside its desk, never into the edge's logs.
    log_path = Path(environ["GATEWAY_LOG_DIR"]).parent / "runtime.log"
    lifeline_read, lifeline_write = os.pipe()
    try:
      descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
      try:
        process = subprocess.Popen(
          [self._python, "-I", "-c", _EDGE_LIFELINE, str(self._stop_grace), *command],
          cwd=self._checkout, env=environ, start_new_session=True,
          stdin=lifeline_read, stdout=descriptor, stderr=subprocess.STDOUT,
        )
      finally:
        os.close(descriptor)
    except BaseException:
      os.close(lifeline_write)
      raise
    finally:
      os.close(lifeline_read)
    self._lifelines[storage_id] = lifeline_write
    return process

  def _terminate(
    self, pid: int, process: subprocess.Popen[bytes] | None = None, *,
    expected_birth: str | None = None,
  ) -> bool:
    if process is None or process.poll() is None:
      signum = signal.SIGTERM
      if expected_birth is not None:
        current_birth = pid_birth_token(pid)
        if current_birth and current_birth != expected_birth:
          return False
        if not current_birth:
          # An unavailable birth token cannot authorize a signal. The existing
          # group lookup can still prove that the recorded group is gone.
          signum = 0
      try:
        os.killpg(pid, signum)
      except ProcessLookupError:
        if process is not None:
          process.wait()
        return True
      except PermissionError:
        return False
      if signum == 0:
        return False
      if process is not None:
        try:
          process.wait(timeout=self._stop_grace)
        except subprocess.TimeoutExpired:
          pass
      else:
        # Reap a persisted leader if it is still our child. Its exit is not
        # enough to release the desk: a gateway member can outlive the leader.
        deadline = time.monotonic() + self._stop_grace
        while time.monotonic() < deadline:
          try:
            waited, _ = os.waitpid(pid, os.WNOHANG)
            if waited == pid:
              break
          except ChildProcessError:
            pass
          try:
            os.kill(pid, 0)
          except ProcessLookupError:
            break
          time.sleep(0.05)
    # Even a reaped or gracefully exited leader may leave a live gateway or
    # lifeline watcher. Only an empty process group proves writer termination.
    try:
      os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
      return True
    except PermissionError:
      # macOS may deny a signal for a group whose leader has exited. Only the
      # group lookup below, never leader disappearance, can confirm the stop.
      pass
    deadline = time.monotonic() + 5
    while True:
      if process is not None:
        process.poll()
      else:
        try:
          os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
          pass
      try:
        os.killpg(pid, 0)
      except ProcessLookupError:
        return True
      except PermissionError:
        # macOS answers EPERM for a group whose remaining members are zombies
        # awaiting reap. EPERM proves nothing either way; only ESRCH proves the
        # group empty, so keep polling within the same bound.
        pass
      if time.monotonic() >= deadline:
        return False
      time.sleep(0.05)

  def stop(self, placement: RuntimePlacement) -> RuntimePlacement:
    storage_id = placement.storage_id
    with self._lock(storage_id):
      stored = self.store.get(storage_id)
      if stored and (stored.generation != placement.generation or stored.handle != placement.handle):
        # A stale caller does not own the replacement's process or routing row.
        return replace(placement, state="stopped", handle=None, handle_birth=None, updated_at=_now())
      process = self._processes.pop(storage_id, None)
      lifeline = self._lifelines.pop(storage_id, None)
      confirmed = False
      if process is not None:
        self._confirmed_stops.pop(storage_id, None)
        confirmed = self._terminate(process.pid, process) and placement.handle == str(process.pid)
      if process is None or placement.handle != str(process.pid):
        # A reloaded backend holds only a persisted pid; the OS may have reused
        # it, so the group is signalled only while the pid is still the
        # process whose birth token was recorded at spawn.
        target = stored if stored is not None else placement
        expected_birth = target.handle_birth
        if (
          target.state in ("ready", "starting") and target.handle
          and target.handle.isascii() and target.handle.isdigit()
          and expected_birth
        ):
          pid = int(target.handle)
          self._confirmed_stops.pop(storage_id, None)
          confirmed = self._terminate(pid, expected_birth=expected_birth)
      if lifeline is not None:
        os.close(lifeline)
      if confirmed:
        self._confirmed_stops[storage_id] = placement.generation
      return self.store.upsert(
        replace(placement, state="stopped", handle=None, handle_birth=None, updated_at=_now()),
      )

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

  def migrate_approvals(
    self, storage_id: str, *, destination: Path,
    begin_migration: Callable[[str], None],
    note_placement_stopped: Callable[[str, int], None],
    complete_migration: Callable[[str], None],
  ) -> ApprovalLedgerMigration:
    """Move one explicitly selected legacy ledger to the edge and restart its runtime.

    The caller supplies the edge service's canonical destination and migration
    callbacks. This is never called by start, and never selects storage ids.
    A relocation failure leaves the edge fenced and the generation stopped.
    """
    from .approval_store import SQLiteApprovalStore

    with self._lock(storage_id):
      placement = self.store.get(storage_id)
      if placement is None:
        raise KeyError(storage_id)
      bundle = load_user_runtime_credentials({
        USER_RUNTIME_CREDENTIALS_FILE_ENV: placement.bundle_path,
      })
      assert bundle is not None
      source = self.placement_root / storage_id / "user-data" / "gateway" / "approvals.sqlite3"
      # Opening first prevents the store owner from creating an absent legacy DB.
      with source.open("rb"):
        pass
      begin_migration(storage_id)
      self.stop(placement)
      note_placement_stopped(storage_id, placement.generation)
      store = SQLiteApprovalStore(source)
      pending_count = store.pending_approval_count()
      log.info(
        "Approval ledger migration storage_id=%s generation=%s pending_count=%s",
        storage_id, placement.generation, pending_count,
      )
      store.checkpoint_for_migration()
      destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
      # Exclusive creation preserves an existing edge ledger. Copying before
      # unlink supports distinct filesystems and retains every SQLite table/id.
      with source.open("rb") as old:
        relocated = destination.open("xb")
        try:
          with relocated:
            os.fchmod(relocated.fileno(), 0o600)
            shutil.copyfileobj(old, relocated)
            relocated.flush()
            os.fsync(relocated.fileno())
          # Persist new parent directories as well as the database's entry.
          for directory in (destination.parent, *destination.parent.parents):
            _fsync_directory(directory)
        except BaseException:
          # Only this attempt's exclusively created, incomplete destination.
          destination.unlink()
          raise
      source.unlink()
      _fsync_directory(source.parent)
      complete_migration(storage_id)
      replacement = self.start(storage_id, bundle.user, bundle.desk)
      return ApprovalLedgerMigration(source, destination, pending_count, replacement)
