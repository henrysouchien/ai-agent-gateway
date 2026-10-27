from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
from textwrap import dedent
import time
from typing import Iterator
from urllib.request import urlopen

import pytest

from hank_contracts.agent_api import AGENT_API_CLAIM_ENV_VARS
from hank_contracts.user_claims import derive_user_claim_key
from agent_gateway import AgentSessionLog, AgentSessionRef, resolve_agent_session_id
from agent_gateway.child_environment import AUTONOMOUS_CHILD_ENVIRONMENT, filter_child_environment
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.runtime_placement import (
  LocalProcessRuntimeBackend,
  RuntimePlacement,
  RuntimePlacementStore,
  RuntimeState,
  parse_positive_pid,
  pid_birth_token,
)
import agent_gateway.runtime_placement as placement_module
from agent_gateway.user_runtime_credentials import (
  DeskReplica,
  USER_RUNTIME_CREDENTIALS_FILE_ENV,
  USER_STORAGE_ID_ENV,
  UserRuntimeIdentity,
  load_user_runtime_credentials,
  mint_user_runtime_credentials,
  write_user_runtime_credentials,
)
import investment_tools.state_paths as investment_state_paths
from investment_tools.state_paths import ready_state_paths


ORG_KEY = "placement-test-org-authority-at-least-32-bytes"
IDENTITY = UserRuntimeIdentity(
  key="placement-private-api-key", channel="excel", slug="alice",
  email="alice@example.test", risk_user_id=17, role="analyst", capabilities=("research",),
)

SUBJECT_PUBLIC_KEYS = json.dumps({"risk-test": "public-assertion-verification-key"})


def test_store_round_trip_wal_and_states(tmp_path: Path) -> None:
  store = RuntimePlacementStore(tmp_path / "host" / "runtime_host.sqlite3")
  placement = RuntimePlacement(
    storage_id="17", url="http://127.0.0.1:12345", state="starting",
    backend="local-process", handle="123", bundle_path=str(tmp_path / "17" / "bundle.json"),
    generation=1, started_at=1791849600.0, updated_at=1791849600.0,
  )
  assert store.path == tmp_path / "host" / "runtime_host.sqlite3"
  assert store.get("17") is None
  assert store.mark("missing", "stopped") is None
  assert store.upsert(placement) == placement
  assert RuntimePlacementStore(store.path).get("17") == placement
  states: tuple[RuntimeState, ...] = ("ready", "stopped", "replaced", "starting")
  for state in states:
    marked = store.mark("17", state)
    assert marked is not None
    assert marked.state == state
    assert marked.generation == 1
    assert marked.started_at == placement.started_at
    assert marked.updated_at != placement.updated_at
  newer = replace(placement, generation=2, handle="456", state="ready")
  store.upsert(newer)
  assert store.list() == [newer]
  with sqlite3.connect(store.path) as connection:
    assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    columns = {row[1]: row[2] for row in connection.execute("PRAGMA table_info(runtime_placements)")}
    assert columns["started_at"] == "REAL"
    assert columns["updated_at"] == "REAL"
  with pytest.raises(FrozenInstanceError):
    setattr(placement, "generation", 9)


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[LocalProcessRuntimeBackend]:
  (tmp_path / "placement_smoke_app.py").write_text(dedent("""
    import os
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from agent_gateway.claim_signing_authority import configured_gateway_claim_signing_authority
    from agent_gateway.user_runtime_credentials import load_user_runtime_credentials

    async def health(request):
      return JSONResponse({"status": "ok"})

    async def claim(request):
      bundle = load_user_runtime_credentials(os.environ)
      authority = configured_gateway_claim_signing_authority(required=True)
      return JSONResponse({
        "claim_environment": authority.sign_user_claim(
          user_id=bundle.user.slug, user_email=bundle.user.email,
        ),
        "storage_id": bundle.storage_id,
        "pid": os.getpid(),
        "data_dir": os.environ["USER_DATA_DIR"],
        "approval_db": os.environ.get("GATEWAY_APPROVAL_DB_PATH"),
        "chat_log_dir": os.environ.get("GATEWAY_CHAT_LOG_DIR"),
        "subject_public_keys": os.environ.get("GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS"),
        "unsafe_environment": [
          name for name in (
            "AGENT_API_USER_CLAIM_HMAC_KEY", "GATEWAY_USER_KEYS", "GATEWAY_API_KEY_WEB",
          ) if name in os.environ
        ],
      })

    app = Starlette(routes=[Route("/health", health), Route("/claim", claim)])
  """))
  checkout = Path(__file__).resolve().parents[3]
  runtime = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY,
    app="placement_smoke_app:app", python=sys.executable,
    extra_env={
      **os.environ,
      "PYTHONPATH": os.pathsep.join((
        str(tmp_path), str(checkout), str(checkout / "packages" / "agent-gateway"),
        str(checkout / "packages" / "contracts"), os.environ.get("PYTHONPATH", ""),
      )),
      "AGENT_API_USER_CLAIM_HMAC_KEY": ORG_KEY,
      "GATEWAY_USER_KEYS": json.dumps([{"key": "other-user-secret"}]),
      "GATEWAY_API_KEY_WEB": "shared-web-operator-authority",
      "GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS": SUBJECT_PUBLIC_KEYS,
      # The shared tier's own state roots must not reach a user runtime.
      "GATEWAY_APPROVAL_DB_PATH": str(tmp_path / "shared" / "approvals.sqlite3"),
      "GATEWAY_CHAT_LOG_DIR": str(tmp_path / "shared" / "chat"),
    },
  )
  try:
    yield runtime
  finally:
    for placement in runtime.store.list():
      runtime.stop(placement)


def _desk(tmp_path: Path) -> DeskReplica:
  return DeskReplica(
    url=str(tmp_path / "replicas"), aws_access_key_id=None, aws_secret_access_key=None,
    aws_session_token=None, expires_at=None,
  )


def _gone(pid: str | None) -> None:
  assert pid is not None
  with pytest.raises(ProcessLookupError):
    os.kill(int(pid), 0)


def test_runtime_retains_subject_public_keys_without_shared_secrets(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  placement = backend.start("17", IDENTITY, _desk(tmp_path))
  with urlopen(placement.url + "/claim") as response:
    runtime_environment = json.load(response)
  assert runtime_environment["subject_public_keys"] == SUBJECT_PUBLIC_KEYS
  assert runtime_environment["unsafe_environment"] == []


def _bundle(path: Path, tmp_path: Path, storage_id: str) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  write_user_runtime_credentials(
    mint_user_runtime_credentials(
      org_claim_key=ORG_KEY, identity=replace(IDENTITY, risk_user_id=int(storage_id)),
      storage_id=storage_id, desk=_desk(tmp_path),
    ),
    path,
  )
  return path


def _local_environment(tmp_path: Path, storage_id: str) -> dict[str, str]:
  local = LocalProcessRuntimeBackend(placement_root=tmp_path / "local", org_claim_key=ORG_KEY)
  user_root = tmp_path / "local" / storage_id
  environ = local._environment(user_root / "user-data", storage_id)
  environ[USER_RUNTIME_CREDENTIALS_FILE_ENV] = str(_bundle(user_root / "bundle.json", tmp_path, storage_id))
  return environ


def _agentcore_environment(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage_id: str,
) -> dict[str, str]:
  checkout = Path(__file__).resolve().parents[3]
  adapter_path = checkout / "deploy/user-runtime/agentcore/runtime_adapter.py"
  spec = importlib.util.spec_from_file_location("_placement_test_runtime_adapter", adapter_path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  monkeypatch.setitem(sys.modules, spec.name, module)
  spec.loader.exec_module(module)
  root = tmp_path / "agentcore"
  adapter = module.RuntimeAdapter(module.AdapterConfig(root=root, image_env={}, litestream_bin=None))
  try:
    environ = adapter.environment({}, storage_id)
  finally:
    asyncio.run(adapter.client.aclose())
  _bundle(root / "bundle.json", tmp_path, storage_id)
  return environ


def _initialise_investment_state(environ: dict[str, str], tmp_path: Path, name: str):
  package_root = tmp_path / f"package-{name}"
  package_root.mkdir()
  return ready_state_paths(environ, home=tmp_path / "home", package_root=package_root)


def test_placed_runtime_initialises_investment_state_in_its_desk(tmp_path: Path) -> None:
  environ = _local_environment(tmp_path, "7331")
  desk = (Path(environ["USER_DATA_DIR"]) / "users" / "7331").resolve()

  paths = _initialise_investment_state(environ, tmp_path, "local")

  assert paths.root.is_relative_to(desk)
  assert paths.root != desk
  assert paths.cutover_marker.is_file()
  assert paths.cutover_marker.is_relative_to(desk)


def test_edge_state_dir_does_not_move_a_placed_runtime_off_its_desk(tmp_path: Path) -> None:
  local = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "local", org_claim_key=ORG_KEY,
    extra_env={investment_state_paths.STATE_DIR_ENV: "/shared/x"},
  )
  user_data = tmp_path / "local" / "7331" / "user-data"
  environ = local._environment(user_data, "7331")

  desk_state = user_data / "users" / "7331" / "investment-tools"
  assert environ[investment_state_paths.STATE_DIR_ENV] == str(desk_state)
  assert not desk_state.exists()  # a locator only: investment_tools initialises it on first use
  root = investment_state_paths.resolve_state_paths(
    environ, home=tmp_path / "home", package_root=tmp_path / "package",
  ).root
  assert root == desk_state.resolve()


def test_agentcore_and_local_runtimes_keep_state_and_journals_in_their_own_desk(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  environments = {
    "agentcore": _agentcore_environment(tmp_path, monkeypatch, "7331"),
    "local": _local_environment(tmp_path, "7331"),
  }
  for name, environ in environments.items():
    desk = (Path(environ["USER_DATA_DIR"]) / "users" / "7331").resolve()
    paths = _initialise_investment_state(environ, tmp_path, name)
    assert paths.root.is_relative_to(desk)
    assert paths.cutover_marker.is_file()
    session = AgentSessionLog(
      session_ref=AgentSessionRef(
        user_id="alice", agent_id="analyst", agent_session_id=resolve_agent_session_id("alice", "analyst"),
      ),
      base_dir=environ["AGENT_SESSION_LOG_BASE_DIR"],
    )
    assert session.path.is_file()
    assert session.path.resolve().is_relative_to(desk)
    assert not session.path.resolve().is_relative_to(paths.root)


def test_investment_tools_and_placement_agree_on_the_storage_id_env_name() -> None:
  # investment_tools ships as its own distribution pinned to an older agent_gateway,
  # so it declares the name itself; this pins it to the placement's.
  assert investment_state_paths.USER_STORAGE_ID_ENV == USER_STORAGE_ID_ENV


def test_children_of_a_placed_runtime_resolve_the_same_investment_state_root(tmp_path: Path) -> None:
  gateway_environment = _local_environment(tmp_path, "7331")
  gateway_paths = _initialise_investment_state(gateway_environment, tmp_path, "gateway")

  child_environment = filter_child_environment(gateway_environment, purpose=AUTONOMOUS_CHILD_ENVIRONMENT)

  # The credential bundle never reaches a child; the storage id and the placement's
  # state locator alone place its state.
  assert USER_RUNTIME_CREDENTIALS_FILE_ENV in gateway_environment
  assert USER_RUNTIME_CREDENTIALS_FILE_ENV not in child_environment
  child_package_root = tmp_path / "package-child"
  child_package_root.mkdir()
  child_paths = ready_state_paths(child_environment, home=tmp_path / "home", package_root=child_package_root)
  desk = (Path(gateway_environment["USER_DATA_DIR"]) / "users" / "7331").resolve()
  assert child_paths.root == gateway_paths.root
  assert child_paths.root.is_relative_to(desk)


def test_local_process_claim_custody_and_replacement(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  desk = _desk(tmp_path)
  # Two same-desk callers must share one writer and one initial generation.
  with ThreadPoolExecutor(max_workers=2) as threads:
    placements = list(threads.map(lambda _: backend.start("17", IDENTITY, desk), range(2)))
  first = placements[0]
  assert placements == [first, first]
  assert first.state == "ready"
  assert first.generation == 1
  assert first.handle is not None
  assert isinstance(first.started_at, float)
  assert isinstance(first.updated_at, float)
  assert first.bundle_path == str(backend.placement_root / "17" / "bundle.json")
  assert (backend.placement_root / "17" / "user-data" / "users" / "17").is_dir()
  assert backend.health(first)
  assert stat.S_IMODE(Path(first.bundle_path).stat().st_mode) == 0o600
  bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: first.bundle_path})
  assert bundle is not None
  assert bundle.user == IDENTITY
  assert bundle.desk == desk
  assert bundle.claim_signing.key == derive_user_claim_key(ORG_KEY, "alice", "alice@example.test")
  with urlopen(first.url + "/claim") as response:
    claim = json.load(response)
  assert claim["storage_id"] == "17"
  assert claim["pid"] == int(first.handle)
  assert claim["data_dir"] == str(backend.placement_root / "17" / "user-data")
  # The product pins approvals to USER_DATA_DIR/gateway; logs stay per runtime.
  assert claim["approval_db"] == str(backend.placement_root / "17" / "user-data" / "gateway" / "approvals.sqlite3")
  assert claim["chat_log_dir"] == str(backend.placement_root / "17" / "logs" / "chat")
  assert claim["unsafe_environment"] == []
  alice = GatewayClaimSigningAuthority(derive_user_claim_key(ORG_KEY, "alice", "alice@example.test"))
  bob = GatewayClaimSigningAuthority(derive_user_claim_key(ORG_KEY, "bob", "bob@example.com"))
  normalized_claim = {
    field: claim["claim_environment"][env_name]
    for field, env_name in AGENT_API_CLAIM_ENV_VARS.items()
  }
  verified = alice.verify_user_claim(normalized_claim)
  assert verified is not None
  assert verified["user_id"] == IDENTITY.slug
  assert verified["user_email"] == IDENTITY.email
  assert bob.verify_user_claim(normalized_claim) is None
  assert GatewayClaimSigningAuthority(ORG_KEY).verify_user_claim(normalized_claim) is None
  for secret in (ORG_KEY, IDENTITY.key, bundle.claim_signing.key, "other-user-secret"):
    assert secret not in repr(backend)
    assert secret not in repr(bundle)
    assert secret.encode() not in backend.store.path.read_bytes()
    wal = Path(str(backend.store.path) + "-wal")
    if wal.exists():
      assert secret.encode() not in wal.read_bytes()
  replacement = backend.replace(first)
  assert replacement.state == "ready"
  assert replacement.generation == 2
  assert replacement.url != first.url
  assert replacement.handle != first.handle
  _gone(first.handle)
  assert load_user_runtime_credentials({
    USER_RUNTIME_CREDENTIALS_FILE_ENV: replacement.bundle_path,
  }) == bundle
  backend.stop(first)  # A stale stop must not kill a newer generation.
  assert backend.store.get("17") == replacement
  assert backend.health(replacement)
  stopped = backend.stop(replacement)
  assert stopped.state == "stopped"
  _gone(replacement.handle)
  restarted = backend.start("17", IDENTITY, desk)
  assert restarted.generation == 3


def test_backend_reload_stops_persisted_pid(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  placement = backend.start("17", IDENTITY, _desk(tmp_path))
  reloaded = LocalProcessRuntimeBackend(
    placement_root=backend.placement_root, org_claim_key=ORG_KEY,
  )
  try:
    assert reloaded.health(placement)
    assert reloaded.stop(placement).state == "stopped"
    _gone(placement.handle)
  finally:
    reloaded.stop(placement)


def _sigterm_ignoring_product(
  directory: Path, exit_after: float | None,
) -> tuple[subprocess.Popen[bytes], Path]:
  """A session-leader product that logs SIGTERM and ignores it (exiting `exit_after` s later, or never)."""
  directory.mkdir()
  marker = directory / "sigterm-received"
  ready = directory / "ready"
  script = directory / "stubborn_product.py"
  script.write_text(dedent(f"""
    import signal, sys, time
    def on_term(signum, frame):
      open({str(marker)!r}, "w").write(repr(time.monotonic()))
      if {exit_after!r} is not None:
        time.sleep({exit_after!r})
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)
    open({str(ready)!r}, "w").write("1")
    while True:
      time.sleep(0.05)
  """))
  process = subprocess.Popen([sys.executable, str(script)], start_new_session=True)
  deadline = time.monotonic() + 10
  while not ready.exists():
    assert time.monotonic() < deadline and process.poll() is None
    time.sleep(0.02)
  return process, marker


def test_stop_sigkills_only_after_the_grace_has_elapsed(tmp_path: Path) -> None:
  grace = 1.5
  runtime = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY, stop_grace=grace,
  )
  # A product still draining at (grace - margin) is not killed: SIGTERM only.
  draining, marker = _sigterm_ignoring_product(tmp_path / "draining", exit_after=grace - 0.7)
  runtime._terminate(draining.pid, draining)
  assert draining.returncode == 0
  assert marker.exists()

  # A product that ignores SIGTERM past the grace gets SIGKILL, and not before.
  stubborn, marker = _sigterm_ignoring_product(tmp_path / "stubborn", exit_after=None)
  started = time.monotonic()
  runtime._terminate(stubborn.pid, stubborn)
  elapsed = time.monotonic() - started
  assert stubborn.returncode == -signal.SIGKILL
  assert marker.exists()  # SIGTERM was delivered first
  assert elapsed >= grace


def test_backend_reload_does_not_signal_a_reused_pid(
  tmp_path: Path, backend: LocalProcessRuntimeBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
  placement = backend.start("17", IDENTITY, _desk(tmp_path))
  assert placement.handle is not None and placement.handle_birth
  assert backend.store.get("17") == placement
  reloaded = LocalProcessRuntimeBackend(
    placement_root=backend.placement_root, org_claim_key=ORG_KEY,
  )
  signalled: list[int] = []
  with monkeypatch.context() as patched:
    # The persisted pid now belongs to a different process.
    patched.setattr(placement_module, "pid_birth_token", lambda pid: "recycled-birth")
    patched.setattr(LocalProcessRuntimeBackend, "_terminate", staticmethod(
      lambda pid, process=None: signalled.append(pid),
    ))
    stopped = reloaded.stop(placement)
  assert stopped.state == "stopped"
  assert stopped.handle is None and stopped.handle_birth is None
  assert signalled == []
  stored = reloaded.store.get("17")
  assert stored is not None and stored.handle is None and stored.handle_birth is None
  process = backend._processes.pop("17")
  backend._terminate(process.pid, process)
  _gone(placement.handle)


def test_store_migration_adds_handle_birth_leaving_preexisting_rows_unverified(tmp_path: Path) -> None:
  path = tmp_path / "host" / "runtime_host.sqlite3"
  path.parent.mkdir()
  columns = (
    "storage_id TEXT PRIMARY KEY, url TEXT NOT NULL, state TEXT NOT NULL, "
    "backend TEXT NOT NULL, handle TEXT, bundle_path TEXT NOT NULL, "
    "generation INTEGER NOT NULL, started_at REAL NOT NULL, updated_at REAL NOT NULL"
  )
  rows = (
    ("17", "ready", str(os.getpid())),
    ("18", "starting", "not-a-pid"),
    ("19", "stopped", str(os.getpid())),
    ("20", "ready", None),
    ("21", "ready", "99999999999"),
  )
  with sqlite3.connect(path) as connection:
    connection.execute(f"CREATE TABLE runtime_placements ({columns})")
    connection.executemany(
      "INSERT INTO runtime_placements VALUES (?, 'http://127.0.0.1:1', ?, 'local-process', ?, 'b', 1, 1.0, 1.0)",
      rows,
    )
  connection.close()

  store = RuntimePlacementStore(path)
  births = {placement.storage_id: placement.handle_birth for placement in store.list()}
  assert births == {"17": None, "18": None, "19": None, "20": None, "21": None}
  with sqlite3.connect(path) as inspect:
    names = [row[1] for row in inspect.execute("PRAGMA table_info(runtime_placements)")]
  inspect.close()
  assert names[-1] == "handle_birth"
  assert {placement.storage_id: placement.handle_birth for placement in RuntimePlacementStore(path).list()} == births

def test_pid_primitives_accept_only_positive_int32() -> None:
  assert parse_positive_pid(1) == 1
  assert parse_positive_pid(0x7FFFFFFF) == 0x7FFFFFFF
  for raw in (0, -1, 0x80000000, True, 1.0, float("inf"), "17", None):
    with pytest.raises(ValueError, match="positive 32-bit integer"):
      parse_positive_pid(raw)
    assert pid_birth_token(raw) == ""  # type: ignore[arg-type]


@pytest.mark.parametrize("storage_id", ("../17", "/17", "017", "0", "１８", "18"))
def test_backend_rejects_unsafe_or_mismatched_storage_identity(
  tmp_path: Path, backend: LocalProcessRuntimeBackend, storage_id: str,
) -> None:
  with pytest.raises(ValueError, match="storage_id"):
    backend.start(storage_id, IDENTITY, _desk(tmp_path))
  assert backend.store.list() == []


def test_start_failure_stops_children_and_marks_stopped(
  tmp_path: Path,
) -> None:
  broken = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "broken", org_claim_key=ORG_KEY,
    app="missing_placement_app:app", python=sys.executable,
    extra_env=dict(os.environ),
  )
  try:
    with pytest.raises(RuntimeError, match="exited before readiness"):
      broken.start("17", IDENTITY, _desk(tmp_path))
    placement = broken.store.get("17")
    assert placement is not None
    assert placement.state == "stopped"
    assert placement.handle is None
    assert not broken.health(placement)
    assert not broken._processes
  finally:
    for placement in broken.store.list():
      broken.stop(placement)


def test_replace_loads_bundle_before_stopping_original(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  placement = backend.start("17", IDENTITY, _desk(tmp_path))
  path = Path(placement.bundle_path)
  original = path.read_bytes()
  try:
    path.write_text("not-json")
    with pytest.raises(ValueError, match="cannot read"):
      backend.replace(placement)
    assert backend.store.get("17") == placement
    assert backend.health(placement)
    invalid = json.loads(original)
    invalid["storage_id"] = "18"
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="storage identity"):
      backend.replace(placement)
    assert backend.store.get("17") == placement
    assert backend.health(placement)
  finally:
    path.write_bytes(original)


def test_replica_is_the_desk_across_a_wiped_replacement(tmp_path: Path) -> None:
  # ISOLATION-SHAPE-A ruling 1: the runtime runs under the desk replicator, so
  # a replacement whose local disk is gone restores the user's rows and files
  # from the replica rather than from the sleeve filesystem.
  litestream = os.environ.get("LITESTREAM_BIN") or shutil.which("litestream")
  if not litestream:
    pytest.skip("litestream binary not configured (LITESTREAM_BIN)")
  checkout = Path(__file__).resolve().parents[3]
  (tmp_path / "desk_smoke_app.py").write_text(dedent("""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(request):
      return JSONResponse({"status": "ok"})

    app = Starlette(routes=[Route("/health", health)])
  """))
  runtime = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY,
    app="desk_smoke_app:app", python=sys.executable, litestream_bin=litestream,
    extra_env={
      **os.environ,
      "PYTHONPATH": os.pathsep.join((
        str(tmp_path), str(checkout), str(checkout / "packages" / "agent-gateway"),
        str(checkout / "packages" / "contracts"), os.environ.get("PYTHONPATH", ""),
      )),
    },
  )
  desk = _desk(tmp_path)
  try:
    first = runtime.start("17", IDENTITY, desk)
    user_root = runtime.placement_root / "17" / "user-data" / "users" / "17"
    db_path = user_root / "research.db"
    with sqlite3.connect(db_path) as connection:
      connection.execute("PRAGMA journal_mode=WAL")
      connection.execute("CREATE TABLE sentinel(note TEXT)")
      connection.execute("INSERT INTO sentinel VALUES ('alice-desk')")
    note = user_root / "workspace" / "notes" / "alice.md"
    note.parent.mkdir(parents=True)
    note.write_text("ALICE-FILE-SENTINEL\n")
    time.sleep(3)
    runtime.stop(first)
    # Replacement compute starts from an empty disk: databases and files alike
    # come back from the replica.
    shutil.rmtree(user_root)
    second = runtime.start("17", IDENTITY, desk)
    assert second.generation == first.generation + 1
    with sqlite3.connect(db_path) as connection:
      assert connection.execute("SELECT note FROM sentinel").fetchall() == [("alice-desk",)]
    assert note.read_text() == "ALICE-FILE-SENTINEL\n"
  finally:
    for placement in runtime.store.list():
      runtime.stop(placement)


def test_rotated_org_claim_key_remints_a_running_runtime(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  # Bundles store the derived claim key; after the org key rotates, a running
  # runtime would sign every claim with a dead key until it is re-minted.
  desk = _desk(tmp_path)
  first = backend.start("17", IDENTITY, desk)
  rotated_key = "rotated-org-authority-at-least-32-bytes-long"
  backend._org_claim_key = rotated_key
  second = backend.start("17", IDENTITY, desk)
  assert second.generation == first.generation + 1
  bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: second.bundle_path})
  assert bundle is not None
  assert bundle.claim_signing.key == derive_user_claim_key(rotated_key, IDENTITY.slug, IDENTITY.email)
  # Unchanged identity under the same key reuses the runtime.
  assert backend.start("17", IDENTITY, desk) == second
