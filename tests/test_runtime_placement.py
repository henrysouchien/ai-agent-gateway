from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import errno
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
  DESK_LEASE_SECONDS,
  DeskLeaseRefused,
  LocalProcessRuntimeBackend,
  RestoreEpoch,
  RestoreRefused,
  RestoreResult,
  RuntimePlacement,
  RuntimePlacementStore,
  RuntimeState,
  WriterRecord,
  parse_positive_pid,
  pid_birth_token,
)
import agent_gateway.runtime_placement as placement_module
from hank_model_transport.providers import api_key_store
from hank_model_transport.providers.anthropic_oauth import (
  ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES,
  ANTHROPIC_STORE_BASE_ENV,
  AnthropicOAuthRecord,
  load_anthropic_oauth_store,
  upsert_anthropic_oauth_record,
)
from hank_model_transport.providers.codex import CODEX_PERSONAL_OAUTH_ENV_NAMES
from hank_model_transport.providers.copilot_oauth import COPILOT_PERSONAL_OAUTH_ENV_NAMES
from hank_model_transport.providers.google_antigravity_oauth import GOOGLE_ANTIGRAVITY_PERSONAL_OAUTH_ENV_NAMES
from hank_model_transport.providers.openai import OPENAI_PERSONAL_OAUTH_ENV_NAMES
from hank_model_transport.providers.google_oauth import GOOGLE_PERSONAL_OAUTH_ENV_NAMES
from hank_model_transport.providers.kimi_code_oauth import KIMI_CODE_PERSONAL_OAUTH_ENV_NAMES
from hank_model_transport.providers.muse_oauth import MUSE_PERSONAL_OAUTH_ENV_NAMES
from hank_model_transport.providers.xai_oauth import XAI_PERSONAL_OAUTH_ENV_NAMES
from hank_model_transport.providers.ollama_cloud import resolve_ollama_cloud_pool_candidates
from hank_model_transport.user_runtime_credentials import (
  DeskReplica,
  USER_RUNTIME_CREDENTIALS_FILE_ENV,
  USER_STORAGE_ID_ENV,
  UserRuntimeIdentity,
  derive_approval_claim_key,
  load_user_runtime_credentials,
  mint_user_runtime_credentials,
  write_user_runtime_credentials,
)
import investment_tools.state_paths as investment_state_paths
from investment_tools.state_paths import ready_state_paths


# Real desk_sync restore/replicate here runs the host-wide replica-writer census; another
# worker's census-test fakes (deliberately unreadable configs) are unproven writers to it.
# Same serialization boundary as tests/test_desk_sync_files.py.
pytestmark = pytest.mark.xdist_group("desk_sync_process_census")


ORG_KEY = "placement-test-org-authority-at-least-32-bytes"
IDENTITY = UserRuntimeIdentity(
  key="placement-private-api-key", channel="excel", slug="alice",
  email="alice@example.test", risk_user_id=17, role="analyst", capabilities=("research",),
)

SUBJECT_PUBLIC_KEYS = json.dumps({"risk-test": "public-assertion-verification-key"})
APPROVALS_EGRESS_URL = "http://127.0.0.1:18350/egress/approvals"


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


@pytest.mark.parametrize("epoch", [0, -1, True, "23", 23.0, None])
def test_restore_epoch_refuses_non_positive_integer_epochs(epoch: object) -> None:
  with pytest.raises(ValueError):
    RestoreEpoch.model_validate({"epoch": epoch})


def test_lease_boundary_models_are_frozen_and_forbid_extra_fields() -> None:
  epoch = RestoreEpoch.model_validate_json('{"epoch":23}')
  record = WriterRecord.model_validate_json(
    '{"epoch":23,"holder":"host:17","expires_at":1791849600.0,"state":"held"}',
  )
  assert epoch.epoch == record.epoch == 23
  with pytest.raises(ValueError):
    setattr(epoch, "epoch", 24)
  with pytest.raises(ValueError):
    setattr(record, "state", "released")
  with pytest.raises(ValueError):
    RestoreEpoch.model_validate({"epoch": 23, "generation": 23})
  with pytest.raises(ValueError):
    WriterRecord.model_validate({**record.model_dump(), "generation": 23})
  refused = RestoreRefused(refused=record)
  with pytest.raises(ValueError):
    setattr(refused, "refused", None)
  with pytest.raises(ValueError):
    RestoreRefused.model_validate({"refused": None, "epoch": 23})


@pytest.mark.parametrize("raw", [
  "{}", '{"epoch":23,"refused":null}', '{"refused":false}',
  '{"refused":{"epoch":23}}', '{"refused":null,"holder":"host:17"}',
])
def test_restore_result_refuses_malformed_union_members(raw: str) -> None:
  with pytest.raises(ValueError):
    RestoreResult.validate_json(raw)


def test_restore_result_parses_success_and_structured_refusals() -> None:
  success = RestoreResult.validate_json('{"epoch":23}')
  assert isinstance(success, RestoreEpoch)
  assert success.epoch == 23
  known_holder = RestoreResult.validate_json(
    '{"refused":{"epoch":23,"holder":"host:17","expires_at":1791849600.0,"state":"held"}}',
  )
  assert isinstance(known_holder, RestoreRefused)
  assert known_holder.refused == WriterRecord(
    epoch=23, holder="host:17", expires_at=1791849600.0, state="held",
  )
  unknown_holder = RestoreResult.validate_json('{"refused":null}')
  assert isinstance(unknown_holder, RestoreRefused)
  assert unknown_holder.refused is None


@pytest.mark.parametrize("change", [
  {"epoch": "23"}, {"epoch": True}, {"epoch": 0}, {"holder": 17},
  {"expires_at": "1791849600"}, {"expires_at": float("inf")},
  {"expires_at": float("nan")}, {"state": "expired"},
])
def test_writer_record_refuses_malformed_lease_events(change: dict[str, object]) -> None:
  with pytest.raises(ValueError):
    WriterRecord.model_validate({
      "epoch": 23, "holder": "host:17", "expires_at": 1791849600.0,
      "state": "held", **change,
    })


@pytest.fixture
def backend(tmp_path: Path) -> Iterator[LocalProcessRuntimeBackend]:
  (tmp_path / "placement_smoke_app.py").write_text(dedent("""
    import os
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority, configured_gateway_claim_signing_authority
    from hank_model_transport.user_runtime_credentials import load_user_runtime_credentials

    async def health(request):
      return JSONResponse({"status": "ok"})

    async def claim(request):
      bundle = load_user_runtime_credentials(os.environ)
      authority = configured_gateway_claim_signing_authority(required=True)
      approval_authority = GatewayClaimSigningAuthority(bundle.approval_claim_signing.key)
      return JSONResponse({
        "claim_environment": authority.sign_user_claim(
          user_id=bundle.user.slug, user_email=bundle.user.email,
        ),
        "approval_claim_environment": approval_authority.sign_user_claim(
          user_id=bundle.user.slug, user_email=bundle.user.email,
        ),
        "storage_id": bundle.storage_id,
        "pid": os.getpid(),
        "data_dir": os.environ["USER_DATA_DIR"],
        "approval_local_paths": [
          name for name in (
            "GATEWAY_APPROVAL_DB_PATH", "AGENT_AUTONOMOUS_APPROVALS_DB_PATH",
            "AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH", "GATEWAY_APPROVAL_AUDIT_DIR",
          ) if name in os.environ
        ],
        "approvals_egress_url": os.environ.get("HANK_APPROVALS_EGRESS_URL"),
        "chat_log_dir": os.environ.get("GATEWAY_CHAT_LOG_DIR"),
        "subject_public_keys": os.environ.get("GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS"),
        "gateway_base_url": os.environ.get("GATEWAY_BASE_URL"),
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
      "AGENT_AUTONOMOUS_APPROVALS_DB_PATH": str(tmp_path / "shared" / "approvals.sqlite3"),
      "AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH": str(tmp_path / "shared" / "decisions.jsonl"),
      "GATEWAY_APPROVAL_AUDIT_DIR": str(tmp_path / "shared" / "audit"),
      "HANK_APPROVALS_EGRESS_URL": APPROVALS_EGRESS_URL,
      "GATEWAY_CHAT_LOG_DIR": str(tmp_path / "shared" / "chat"),
      # The edge's own origin (the shared pair's .env value) must not name a user runtime's gateway.
      "GATEWAY_BASE_URL": "https://localhost:8000",
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


def test_refused_desk_lease_never_mints_or_starts_a_runtime(
  tmp_path: Path, backend: LocalProcessRuntimeBackend, monkeypatch: pytest.MonkeyPatch,
) -> None:
  desk = _desk(tmp_path)
  writer_path = Path(desk.url) / "users" / "17" / "writer.json"
  writer_path.parent.mkdir(parents=True)
  writer = WriterRecord(
    epoch=31, holder="other-host:123", expires_at=time.time() + DESK_LEASE_SECONDS,
    state="held",
  )
  writer_path.write_text(writer.model_dump_json())

  def unexpected_spawn(*args: object, **kwargs: object) -> None:
    pytest.fail("a refused lease must not spawn a runtime")

  monkeypatch.setattr(backend, "_spawn", unexpected_spawn)
  with pytest.raises(DeskLeaseRefused, match="holder epoch 31") as caught:
    backend.start("17", IDENTITY, desk)

  assert "other-host:123" in str(caught.value)
  assert "Finding: desk writer lease LOST/refused" in str(caught.value)
  assert caught.value.record == writer
  assert backend.store.get("17") is None
  assert not (backend.placement_root / "17" / "bundle.json").exists()
  assert WriterRecord.model_validate_json(writer_path.read_bytes()) == writer


def test_won_replica_epoch_is_the_generation_despite_local_history(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  desk = _desk(tmp_path)
  writer_path = Path(desk.url) / "users" / "17" / "writer.json"
  writer_path.parent.mkdir(parents=True)
  writer_path.write_text(WriterRecord(
    epoch=40, holder="previous-host:123", expires_at=time.time(), state="released",
  ).model_dump_json())
  backend.store.upsert(RuntimePlacement(
    storage_id="17", url="http://127.0.0.1:12345", state="stopped",
    backend="local-process", handle=None,
    bundle_path=str(backend.placement_root / "17" / "bundle.json"),
    generation=999, started_at=1791849600.0, updated_at=1791849600.0,
  ))
  replica_note = writer_path.parent / "files" / "workspace" / "replica-only.txt"
  replica_note.parent.mkdir(parents=True)
  replica_note.write_text("not restored without Litestream")

  placement = backend.start("17", IDENTITY, desk)

  assert placement.generation == 41
  assert backend.store.get("17") == placement
  bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: placement.bundle_path})
  assert bundle is not None
  assert bundle.generation == 41
  assert bundle.approval_claim_signing.key == derive_approval_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email, 41,
  )
  writer = WriterRecord.model_validate_json(writer_path.read_bytes())
  assert writer.epoch == 41
  assert writer.state == "held"
  assert not (
    backend.placement_root / "17" / "user-data" / "users" / "17"
    / "workspace" / "replica-only.txt"
  ).exists()
  backend.stop(placement)
  assert WriterRecord.model_validate_json(writer_path.read_bytes()).state == "released"
  assert replica_note.read_text() == "not restored without Litestream"


@pytest.mark.parametrize("action", ["start", "replace"])
@pytest.mark.parametrize("reload_backend", [False, True], ids=["managed", "persisted"])
def test_killed_placement_takes_over_its_held_epoch_without_waiting(
  tmp_path: Path, backend: LocalProcessRuntimeBackend, action: str, reload_backend: bool,
) -> None:
  desk = _desk(tmp_path)
  first = backend.start("17", IDENTITY, desk)
  assert first.handle is not None and first.handle_birth
  writer_path = Path(desk.url) / "users" / "17" / "writer.json"
  os.killpg(int(first.handle), signal.SIGKILL)
  backend._processes["17"].wait(timeout=5)
  held = WriterRecord.model_validate_json(writer_path.read_bytes())
  assert held.state == "held"
  assert held.epoch == first.generation
  assert time.time() < held.expires_at
  runtime = (
    LocalProcessRuntimeBackend(
      placement_root=backend.placement_root, org_claim_key=ORG_KEY,
      app="placement_smoke_app:app", python=sys.executable, extra_env=backend._extra_env,
    )
    if reload_backend else backend
  )
  try:
    replacement = (
      runtime.start("17", IDENTITY, desk) if action == "start" else runtime.replace(first)
    )
    assert replacement.state == "ready"
    assert replacement.generation == first.generation + 1
    assert time.time() < held.expires_at
    assert runtime.health(replacement)
    writer = WriterRecord.model_validate_json(writer_path.read_bytes())
    assert writer.epoch == replacement.generation
    assert writer.state == "held"
    bundle = load_user_runtime_credentials({
      USER_RUNTIME_CREDENTIALS_FILE_ENV: replacement.bundle_path,
    })
    assert bundle is not None and bundle.generation == writer.epoch
    runtime.stop(first)
    assert runtime.store.get("17") == replacement
    assert runtime.health(replacement)
  finally:
    if reload_backend:
      current = runtime.store.get("17")
      assert current is not None
      runtime.stop(current)


def test_killed_leader_cannot_leave_a_gateway_alive_during_takeover(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  desk = _desk(tmp_path)
  first = backend.start("17", IDENTITY, desk)
  assert first.handle is not None
  with urlopen(first.url + "/claim") as response:
    gateway_pid = json.load(response)["pid"]
  os.kill(int(first.handle), signal.SIGKILL)
  backend._processes["17"].wait(timeout=5)
  os.kill(gateway_pid, 0)
  with urlopen(first.url + "/health") as response:
    assert response.status == 200
  assert not backend.health(first)
  writer_path = Path(desk.url) / "users" / "17" / "writer.json"
  held = WriterRecord.model_validate_json(writer_path.read_bytes())
  assert held.state == "held"
  assert time.time() < held.expires_at
  replacement = backend.start("17", IDENTITY, desk)
  assert replacement.generation == first.generation + 1
  assert time.time() < held.expires_at
  assert backend.health(replacement)
  _gone(str(gateway_pid))
  writer = WriterRecord.model_validate_json(writer_path.read_bytes())
  assert writer.epoch == replacement.generation
  assert writer.state == "held"


def test_confirmed_takeover_refuses_other_epoch_and_consumes_the_proof(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  desk = _desk(tmp_path)
  first = backend.start("17", IDENTITY, desk)
  assert first.handle is not None
  os.killpg(int(first.handle), signal.SIGKILL)
  backend._processes["17"].wait(timeout=5)
  writer_path = Path(desk.url) / "users" / "17" / "writer.json"
  held = WriterRecord.model_validate_json(writer_path.read_bytes())
  other = held.model_copy(update={"epoch": held.epoch + 1, "holder": "other-host:123"})
  writer_path.write_text(other.model_dump_json())
  with pytest.raises(DeskLeaseRefused) as caught:
    backend.start("17", IDENTITY, desk)
  assert caught.value.record == other
  assert WriterRecord.model_validate_json(writer_path.read_bytes()) == other
  assert not backend._processes
  writer_path.write_text(held.model_dump_json())
  with pytest.raises(DeskLeaseRefused) as caught:
    backend.start("17", IDENTITY, desk)
  assert caught.value.record == held
  assert WriterRecord.model_validate_json(writer_path.read_bytes()) == held


def test_confirmed_takeover_does_not_cross_replica_urls(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  desk = _desk(tmp_path)
  first = backend.start("17", IDENTITY, desk)
  assert first.handle is not None
  os.killpg(int(first.handle), signal.SIGKILL)
  backend._processes["17"].wait(timeout=5)
  other_desk = _desk(tmp_path / "other")
  writer_path = Path(other_desk.url) / "users" / "17" / "writer.json"
  writer_path.parent.mkdir(parents=True)
  other = WriterRecord(
    epoch=first.generation, holder="other-host:123",
    expires_at=time.time() + DESK_LEASE_SECONDS, state="held",
  )
  writer_path.write_text(other.model_dump_json())
  with pytest.raises(DeskLeaseRefused) as caught:
    backend.start("17", IDENTITY, other_desk)
  assert caught.value.record == other
  assert WriterRecord.model_validate_json(writer_path.read_bytes()) == other
  assert not backend._processes


@pytest.mark.parametrize("proof", ["missing-handle", "missing-birth", "mismatched-birth", "unknown-birth"])
@pytest.mark.parametrize("state", ["ready", "stopped"])
def test_unconfirmed_placement_cannot_take_over_a_held_writer(
  tmp_path: Path, backend: LocalProcessRuntimeBackend, monkeypatch: pytest.MonkeyPatch,
  proof: str, state: RuntimeState,
) -> None:
  desk = _desk(tmp_path)
  first = backend.start("17", IDENTITY, desk)
  assert first.handle is not None and first.handle_birth
  reloaded = LocalProcessRuntimeBackend(
    placement_root=backend.placement_root, org_claim_key=ORG_KEY,
    app="placement_smoke_app:app", python=sys.executable, extra_env=backend._extra_env,
  )
  unconfirmed = replace(
    first, state=state,
    handle=None if proof == "missing-handle" else first.handle,
    handle_birth=None if proof == "missing-birth" else first.handle_birth,
  )
  backend.store.upsert(unconfirmed)
  signals: list[tuple[int, int]] = []
  real_killpg = os.killpg

  def killpg(pid: int, signum: int) -> None:
    signals.append((pid, signum))
    assert signum == 0, "unconfirmed ownership must never signal a placement"
    real_killpg(pid, signum)

  with monkeypatch.context() as patched:
    patched.setattr(reloaded, "_probe", lambda url: False)
    patched.setattr(placement_module.os, "killpg", killpg)
    if proof in ("mismatched-birth", "unknown-birth"):
      patched.setattr(
        placement_module, "pid_birth_token",
        lambda pid: "another-process-birth" if proof == "mismatched-birth" else "",
      )
    with pytest.raises(DeskLeaseRefused) as caught:
      reloaded.start("17", IDENTITY, desk)
  assert caught.value.record is not None
  assert caught.value.record.epoch == first.generation
  assert caught.value.record.state == "held"
  assert not reloaded._processes
  assert all(signum == 0 for _, signum in signals)
  assert backend._processes["17"].poll() is None
  assert backend.health(first)


def test_two_backend_roots_share_exactly_one_replica_writer(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  other = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "other-placements", org_claim_key=ORG_KEY,
    app="placement_smoke_app:app", python=sys.executable, extra_env=backend._extra_env,
  )
  desk = _desk(tmp_path)

  def start(runtime: LocalProcessRuntimeBackend) -> RuntimePlacement | DeskLeaseRefused:
    try:
      return runtime.start("17", IDENTITY, desk)
    except DeskLeaseRefused as exc:
      return exc

  try:
    with ThreadPoolExecutor(max_workers=2) as threads:
      outcomes = list(threads.map(start, (backend, other)))
    started = [outcome for outcome in outcomes if isinstance(outcome, RuntimePlacement)]
    refused = [outcome for outcome in outcomes if isinstance(outcome, DeskLeaseRefused)]
    assert len(started) == len(refused) == 1
    assert started[0].state == "ready"
    assert started[0].generation == 1
    assert sum(
      runtime.store.get("17") is not None for runtime in (backend, other)
    ) == 1
    writer = WriterRecord.model_validate_json(
      (Path(desk.url) / "users" / "17" / "writer.json").read_bytes(),
    )
    assert writer.epoch == started[0].generation
    assert writer.state == "held"
  finally:
    for placement in other.store.list():
      other.stop(placement)


def test_runtime_retains_subject_public_keys_without_shared_secrets(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  placement = backend.start("17", IDENTITY, _desk(tmp_path))
  with urlopen(placement.url + "/claim") as response:
    runtime_environment = json.load(response)
  assert runtime_environment["subject_public_keys"] == SUBJECT_PUBLIC_KEYS
  assert runtime_environment["unsafe_environment"] == []


def test_a_placed_runtime_names_its_own_origin_as_its_gateway(
  tmp_path: Path, backend: LocalProcessRuntimeBackend,
) -> None:
  placement = backend.start("17", IDENTITY, _desk(tmp_path))
  with urlopen(placement.url + "/claim") as response:
    runtime_environment = json.load(response)
  assert runtime_environment["gateway_base_url"] == placement.url
  assert runtime_environment["approvals_egress_url"] == APPROVALS_EGRESS_URL
  assert runtime_environment["approval_local_paths"] == []


@pytest.mark.parametrize("approval_url", [
  "http://127.0.0.1:18350/egress/approvals", "https://edge.example.test/egress/approvals",
])
def test_local_approvals_egress_uses_explicit_url(
  tmp_path: Path, approval_url: str,
) -> None:
  runtime = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY,
    extra_env={"HANK_APPROVALS_EGRESS_URL": approval_url, "GATEWAY_BASE_URL": "https://other.test"},
  )
  environ = runtime._environment(tmp_path / "placements" / "17" / "user-data", "17")
  assert environ["HANK_APPROVALS_EGRESS_URL"] == approval_url
  assert environ["HANK_APPROVAL_TRANSPORT"] == "local"


@pytest.mark.parametrize("approval_url", [
  None, "", "/egress/approvals", "edge.example.test/egress/approvals",
  "ftp://edge.example.test/egress/approvals", "https:///egress/approvals",
  "https://edge.example.test/egress/approvals?x=1",
  "https://edge.example.test/egress/approvals#fragment",
  "https://edge.example.test/egress/approvals?", "https://edge.example.test/egress/approvals#",
  "http://edge.example.test:invalid/egress/approvals", "http://edge.example.test:0/egress/approvals",
  "https://edge .example.test/egress/approvals",
])
def test_local_placement_refuses_missing_or_malformed_approval_url(
  tmp_path: Path, approval_url: str | None,
) -> None:
  extra_env = {"GATEWAY_BASE_URL": "https://shared.example.test"}
  if approval_url is not None:
    extra_env["HANK_APPROVALS_EGRESS_URL"] = approval_url
  runtime = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY, extra_env=extra_env,
  )
  with pytest.raises(ValueError, match="HANK_APPROVALS_EGRESS_URL"):
    runtime.start("17", IDENTITY, _desk(tmp_path))
  assert runtime.store.get("17") is None


def _bundle(path: Path, tmp_path: Path, storage_id: str) -> Path:
  path.parent.mkdir(parents=True, exist_ok=True)
  write_user_runtime_credentials(
    mint_user_runtime_credentials(
      org_claim_key=ORG_KEY, identity=replace(IDENTITY, risk_user_id=int(storage_id)),
      storage_id=storage_id, desk=_desk(tmp_path), generation=1,
    ),
    path,
  )
  return path


def _local_environment(tmp_path: Path, storage_id: str) -> dict[str, str]:
  local = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "local", org_claim_key=ORG_KEY,
    extra_env={"HANK_APPROVALS_EGRESS_URL": APPROVALS_EGRESS_URL},
  )
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
    extra_env={
      investment_state_paths.STATE_DIR_ENV: "/shared/x",
      "HANK_APPROVALS_EGRESS_URL": APPROVALS_EGRESS_URL,
    },
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
  assert bundle.generation == first.generation
  assert bundle.claim_signing.key == derive_user_claim_key(
    ORG_KEY, "alice", "alice@example.test",
  )
  assert bundle.approval_claim_signing.key == derive_approval_claim_key(
    ORG_KEY, "alice", "alice@example.test", first.generation,
  )
  with urlopen(first.url + "/claim") as response:
    claim = json.load(response)
  assert claim["storage_id"] == "17"
  assert os.getpgid(claim["pid"]) == int(first.handle)
  assert claim["data_dir"] == str(backend.placement_root / "17" / "user-data")
  # Approval state remains at the edge; logs stay per runtime.
  assert claim["approval_local_paths"] == []
  assert claim["approvals_egress_url"] == APPROVALS_EGRESS_URL
  assert claim["chat_log_dir"] == str(backend.placement_root / "17" / "logs" / "chat")
  assert claim["unsafe_environment"] == []
  alice = GatewayClaimSigningAuthority(
    derive_user_claim_key(ORG_KEY, "alice", "alice@example.test"),
  )
  bob = GatewayClaimSigningAuthority(
    derive_user_claim_key(ORG_KEY, "bob", "bob@example.com"),
  )
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
  approval_alice = GatewayClaimSigningAuthority(bundle.approval_claim_signing.key)
  normalized_approval_claim = {
    field: claim["approval_claim_environment"][env_name]
    for field, env_name in AGENT_API_CLAIM_ENV_VARS.items()
  }
  assert approval_alice.verify_user_claim(normalized_approval_claim) is not None
  assert alice.verify_user_claim(normalized_approval_claim) is None
  for secret in (
    ORG_KEY, IDENTITY.key, bundle.claim_signing.key,
    bundle.approval_claim_signing.key, "other-user-secret",
  ):
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
  replacement_bundle = load_user_runtime_credentials({
    USER_RUNTIME_CREDENTIALS_FILE_ENV: replacement.bundle_path,
  })
  assert replacement_bundle is not None
  assert replacement_bundle.user == bundle.user
  assert replacement_bundle.desk == bundle.desk
  assert replacement_bundle.generation == replacement.generation
  assert replacement_bundle.claim_signing.key == derive_user_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email,
  )
  assert replacement_bundle.claim_signing.key == bundle.claim_signing.key
  assert replacement_bundle.approval_claim_signing.key == derive_approval_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email, replacement.generation,
  )
  assert replacement_bundle.approval_claim_signing.key != bundle.approval_claim_signing.key
  replacement_authority = GatewayClaimSigningAuthority(replacement_bundle.claim_signing.key)
  assert replacement_authority.verify_user_claim(normalized_claim) is not None
  replacement_approval_authority = GatewayClaimSigningAuthority(
    replacement_bundle.approval_claim_signing.key,
  )
  assert replacement_approval_authority.verify_user_claim(normalized_approval_claim) is None
  with urlopen(replacement.url + "/claim") as response:
    replacement_claim = json.load(response)
  normalized_replacement_claim = {
    field: replacement_claim["claim_environment"][env_name]
    for field, env_name in AGENT_API_CLAIM_ENV_VARS.items()
  }
  assert replacement_authority.verify_user_claim(normalized_replacement_claim) is not None
  assert alice.verify_user_claim(normalized_replacement_claim) is not None
  normalized_replacement_approval_claim = {
    field: replacement_claim["approval_claim_environment"][env_name]
    for field, env_name in AGENT_API_CLAIM_ENV_VARS.items()
  }
  assert replacement_approval_authority.verify_user_claim(normalized_replacement_approval_claim) is not None
  assert approval_alice.verify_user_claim(normalized_replacement_approval_claim) is None
  retained_bundle = Path(replacement.bundle_path).read_bytes()
  assert backend.start("17", IDENTITY, desk) == replacement
  assert Path(replacement.bundle_path).read_bytes() == retained_bundle
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
    assert reloaded.start("17", IDENTITY, _desk(tmp_path)) == placement
    assert reloaded.stop(placement).state == "stopped"
    _gone(placement.handle)
  finally:
    reloaded.stop(placement)


@pytest.mark.parametrize(("group_probe_errors", "confirmed"), [
  pytest.param([ProcessLookupError(errno.ESRCH, "group gone")], True, id="group-gone"),
  # macOS: EPERM while the group's last members are zombies awaiting reap, then ESRCH.
  pytest.param(
    [PermissionError(errno.EPERM, "zombie group"), PermissionError(errno.EPERM, "zombie group"),
     ProcessLookupError(errno.ESRCH, "group gone")],
    True, id="eperm-then-gone",
  ),
  pytest.param([PermissionError(errno.EPERM, "probe denied")], False, id="group-unknown"),
])
def test_backend_reload_sigkill_permission_error_requires_confirmed_group_exit(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, group_probe_errors: list[OSError], confirmed: bool,
) -> None:
  runtime = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY, stop_grace=0,
  )
  pid = 12345
  signals: list[tuple[int, int]] = []
  clock = [0.0]
  probes = list(group_probe_errors)

  def killpg(group: int, signum: int) -> None:
    signals.append((group, signum))
    if signum == signal.SIGKILL:
      raise PermissionError(errno.EPERM, "process group denied")
    if signum == 0:
      # The last listed answer repeats: a persistent EPERM never proves the group empty.
      raise probes.pop(0) if len(probes) > 1 else probes[0]

  def kill(process: int, signum: int) -> None:
    pytest.fail("leader disappearance must not substitute for process-group exit")

  def sleep(seconds: float) -> None:
    clock[0] += seconds

  monkeypatch.setattr(placement_module.os, "killpg", killpg)
  monkeypatch.setattr(placement_module.os, "kill", kill)
  monkeypatch.setattr(placement_module.os, "waitpid", lambda pid, options: (0, 0))
  monkeypatch.setattr(placement_module.time, "monotonic", lambda: clock[0])
  monkeypatch.setattr(placement_module.time, "sleep", sleep)
  assert runtime._terminate(pid) is confirmed
  assert signals[:2] == [(pid, signal.SIGTERM), (pid, signal.SIGKILL)]
  assert signals[2:] and all(signum == 0 for _, signum in signals[2:])
  if not confirmed:
    assert 5 <= clock[0] < 6  # EPERM alone is unconfirmed only after the full bound


def test_termination_without_group_exit_is_unconfirmed_after_the_bound(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  runtime = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY, stop_grace=0,
  )
  pid = 12345
  clock = [0.0]
  signals: list[tuple[int, int]] = []

  def no_child(pid: int, options: int) -> tuple[int, int]:
    raise ChildProcessError

  def sleep(seconds: float) -> None:
    clock[0] += seconds

  monkeypatch.setattr(placement_module.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
  monkeypatch.setattr(placement_module.os, "waitpid", no_child)
  monkeypatch.setattr(placement_module.time, "monotonic", lambda: clock[0])
  monkeypatch.setattr(placement_module.time, "sleep", sleep)
  assert runtime._terminate(pid) is False
  assert signals[:2] == [(pid, signal.SIGTERM), (pid, signal.SIGKILL)]
  assert signals[2:] and all(signum == 0 for _, signum in signals[2:])
  assert 5 <= clock[0] < 6


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


def _placement_under_an_edge(
  tmp_path: Path, *, grace: float, ignore_sigterm: bool,
) -> tuple[subprocess.Popen[bytes], int, int, Path]:
  """An edge process that spawned one placement; the placement runs a child in its group.

  Returns the edge, the placement's pid, its child's pid, and the placement's
  SIGTERM marker.
  """
  marker = tmp_path / "placement-sigterm"
  pids = tmp_path / "placement-pids"
  product = tmp_path / "placement_product.py"
  product.write_text(dedent(f"""
    import os, signal, subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    def on_term(signum, frame):
      open({str(marker)!r}, "w").write("1")
      if not {ignore_sigterm!r}:
        child.terminate()
        sys.exit(0)
    signal.signal(signal.SIGTERM, on_term)
    open({str(pids)!r} + ".tmp", "w").write(f"{{os.getpid()}} {{child.pid}}")
    os.replace({str(pids)!r} + ".tmp", {str(pids)!r})
    while True:
      time.sleep(0.05)
  """))
  edge_script = tmp_path / "edge.py"
  edge_script.write_text(dedent(f"""
    import os, sys, time
    from agent_gateway.runtime_placement import LocalProcessRuntimeBackend
    backend = LocalProcessRuntimeBackend(
      placement_root={str(tmp_path / "placements")!r}, org_claim_key={ORG_KEY!r},
      stop_grace={grace!r},
    )
    backend._spawn("17", [sys.executable, {str(product)!r}], {{
      "GATEWAY_LOG_DIR": {str(tmp_path / "logs" / "gateway")!r}, "PATH": os.environ["PATH"],
    }})
    time.sleep(600)
  """))
  (tmp_path / "logs" / "gateway").mkdir(parents=True)
  edge = subprocess.Popen(
    [sys.executable, str(edge_script)],
    env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
  )
  deadline = time.monotonic() + 30
  while not pids.exists():
    assert time.monotonic() < deadline and edge.poll() is None
    time.sleep(0.02)
  leader, child = (int(value) for value in pids.read_text().split())
  return edge, leader, child, marker


def _wait_gone(pids: tuple[int, ...], timeout: float) -> float:
  started = time.monotonic()
  alive = list(pids)
  while time.monotonic() - started < timeout:
    alive = []
    for pid in pids:
      try:
        os.kill(pid, 0)
        alive.append(pid)
      except ProcessLookupError:
        pass
    if not alive:
      return time.monotonic() - started
    time.sleep(0.05)
  for pid in alive:
    try:
      os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
  raise AssertionError(f"still running {timeout}s after the edge died: {alive}")


def test_a_placement_and_its_children_stop_when_their_edge_is_killed(tmp_path: Path) -> None:
  edge, leader, child, marker = _placement_under_an_edge(
    tmp_path, grace=30.0, ignore_sigterm=False,
  )
  edge.kill()
  edge.wait()
  # SIGTERM ends a placement that honours it, well before the 30 s grace.
  _wait_gone((leader, child), timeout=10)
  assert marker.exists()


def test_a_placement_ignoring_sigterm_is_killed_after_the_grace_when_its_edge_dies(
  tmp_path: Path,
) -> None:
  grace = 1.5
  edge, leader, child, marker = _placement_under_an_edge(
    tmp_path, grace=grace, ignore_sigterm=True,
  )
  edge.kill()
  edge.wait()
  elapsed = _wait_gone((leader, child), timeout=grace + 10)
  assert marker.exists()  # SIGTERM came first
  assert elapsed >= grace - 0.2


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
    patched.setattr(placement_module.os, "killpg", lambda pid, signum: signalled.append(pid))
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
    extra_env={**os.environ, "HANK_APPROVALS_EGRESS_URL": APPROVALS_EGRESS_URL},
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
      "HANK_APPROVALS_EGRESS_URL": APPROVALS_EGRESS_URL,
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
  assert bundle.generation == second.generation
  assert bundle.claim_signing.key == derive_user_claim_key(
    rotated_key, IDENTITY.slug, IDENTITY.email,
  )
  assert bundle.approval_claim_signing.key == derive_approval_claim_key(
    rotated_key, IDENTITY.slug, IDENTITY.email, second.generation,
  )
  # Unchanged identity under the same key reuses the runtime.
  assert backend.start("17", IDENTITY, desk) == second


@pytest.mark.parametrize("mismatch", ("generation", "generic_key", "approval_key", "both"))
def test_ready_bundle_must_match_current_placement_generation(
  tmp_path: Path, backend: LocalProcessRuntimeBackend, mismatch: str,
) -> None:
  desk = _desk(tmp_path)
  first = backend.start("17", IDENTITY, desk)
  bundle_path = Path(first.bundle_path)
  stale_bundle = json.loads(bundle_path.read_text())
  if mismatch in ("generation", "both"):
    stale_bundle["generation"] = first.generation + 1
  if mismatch == "generic_key":
    stale_bundle["claim_signing"]["key"] = derive_user_claim_key(
      "other-org-authority-at-least-32-bytes", IDENTITY.slug, IDENTITY.email,
    )
  if mismatch in ("approval_key", "both"):
    stale_bundle["approval_claim_signing"]["key"] = derive_approval_claim_key(
      ORG_KEY, IDENTITY.slug, IDENTITY.email, first.generation + 1,
    )
  bundle_path.write_text(json.dumps(stale_bundle))

  replacement = backend.start("17", IDENTITY, desk)

  assert replacement.generation == first.generation + 1
  assert replacement.handle != first.handle
  _gone(first.handle)
  bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: replacement.bundle_path})
  assert bundle is not None
  assert bundle.generation == replacement.generation
  assert bundle.claim_signing.key == derive_user_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email,
  )
  assert bundle.approval_claim_signing.key == derive_approval_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email, replacement.generation,
  )
  assert backend.start("17", IDENTITY, desk) == replacement


def _seeding_backend(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_env: dict[str, str] | None = None,
) -> tuple[LocalProcessRuntimeBackend, Path, Path]:
  """A backend whose edge store is `tmp/shared` and whose operator home is `tmp/home`."""
  shared = tmp_path / "shared"
  home = tmp_path / "home"
  (home / ".agent_gateway").mkdir(parents=True)
  shared.mkdir()
  monkeypatch.setenv("HOME", str(home))
  backend = LocalProcessRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY,
    app="missing_placement_app:app", python=sys.executable,
    extra_env={
      **os.environ, "USER_DATA_DIR": str(shared),
      "HANK_APPROVALS_EGRESS_URL": APPROVALS_EGRESS_URL, **(extra_env or {}),
    },
  )
  return backend, shared, home / ".agent_gateway"


_OWNER_17_ENV = {
  "GATEWAY_USER_KEYS": json.dumps([{"slug": "alice", "risk_user_id": 17, "role": "owner"}]),
}


def _ollama_store(base: Path, api_key: str) -> Path:
  store = base / "ollama_cloud" / "credentials.json"
  api_key_store.save_api_key_record(store, {"api_key": api_key, "source": "api_key", "identity": "seat"})
  return store


def _start_until_runtime_fails(backend: LocalProcessRuntimeBackend, tmp_path: Path) -> Path:
  try:
    with pytest.raises(RuntimeError, match="exited before readiness"):
      backend.start("17", IDENTITY, _desk(tmp_path))
  finally:
    for placement in backend.store.list():
      backend.stop(placement)
  return backend.placement_root / "17" / "user-data"


def test_fresh_placement_resolves_ollama_candidates_from_its_private_org_root(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend, shared, operator = _seeding_backend(tmp_path, monkeypatch)
  _ollama_store(shared, "shared-edge-key")
  _ollama_store(operator, "operator-home-key")

  private = _start_until_runtime_fails(backend, tmp_path)

  candidates, _ = resolve_ollama_cloud_pool_candidates(
    environ={"USER_DATA_DIR": str(private)}, user_scope="17",
  )
  assert [candidate.record["api_key"] for candidate in candidates] == ["shared-edge-key"]
  store_path = candidates[0].store_path
  assert store_path == private / "ollama_cloud" / "credentials.json"
  assert store_path is not None
  assert stat.S_IMODE(store_path.stat().st_mode) == 0o600
  # Seeded stores sit at the org root; users/<sid> holds only the user's own enrollments.
  desk = private / "users" / "17"
  assert not [
    path for path in desk.rglob("*")
    if path.is_file() and path.name.startswith(("credentials.json", "oauth.json"))
  ]


def test_placement_seeds_from_operator_home_when_the_edge_store_lacks_a_file(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend, _, operator = _seeding_backend(tmp_path, monkeypatch)
  _ollama_store(operator, "operator-home-key")

  private = _start_until_runtime_fails(backend, tmp_path)

  candidates, _ = resolve_ollama_cloud_pool_candidates(environ={"USER_DATA_DIR": str(private)})
  assert [candidate.record["api_key"] for candidate in candidates] == ["operator-home-key"]


def test_placement_seeding_never_overwrites_an_existing_target(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend, shared, _ = _seeding_backend(tmp_path, monkeypatch)
  _ollama_store(shared, "shared-edge-key")
  rotating = AnthropicOAuthRecord(
    identity="rotating@example.test", access_token="rotating-access-marker",
    refresh_token="rotating-refresh-marker", expires_at=9_999_999_999.0,
    created_at=1000.0, updated_at=1000.0,
  )
  upsert_anthropic_oauth_record(shared / "anthropic" / "oauth.json", rotating)
  setup = replace(rotating, identity="source-setup@example.test", refresh_token="")
  upsert_anthropic_oauth_record(shared / "anthropic" / "oauth.json", setup)
  private = backend.placement_root / "17" / "user-data"
  kept_ollama = _ollama_store(private, "already-here-key")
  kept_anthropic = private / "anthropic" / "oauth.json"
  upsert_anthropic_oauth_record(kept_anthropic, replace(rotating, identity="kept@example.test"))
  before = (kept_ollama.read_bytes(), kept_anthropic.read_bytes())

  assert _start_until_runtime_fails(backend, tmp_path) == private

  assert (kept_ollama.read_bytes(), kept_anthropic.read_bytes()) == before


def test_placement_anthropic_seed_excludes_rotating_grants(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend, shared, _ = _seeding_backend(tmp_path, monkeypatch, _OWNER_17_ENV)
  rotating = AnthropicOAuthRecord(
    identity="rotating@example.test", access_token="rotating-access-marker",
    refresh_token="rotating-refresh-marker", expires_at=9_999_999_999.0,
    created_at=1000.0, updated_at=1000.0,
  )
  setup = replace(
    rotating, identity="setup@example.test", access_token="setup-access-marker",
    refresh_token="", credential_kind="auth_token",
  )
  for record in (rotating, setup):
    upsert_anthropic_oauth_record(shared / "anthropic" / "oauth.json", record)

  private = _start_until_runtime_fails(backend, tmp_path)

  seeded = private / "anthropic" / "oauth.json"
  assert [record.identity for record in load_anthropic_oauth_store(seeded)] == ["setup@example.test"]
  text = seeded.read_text(encoding="utf-8")
  assert "rotating-refresh-marker" not in text and "rotating-access-marker" not in text
  assert not (private / "users" / "17" / "anthropic").exists()


def test_placement_binds_kimi_and_xai_oauth_in_place_without_copying(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend, shared, operator = _seeding_backend(tmp_path, monkeypatch, _OWNER_17_ENV)
  for family in ("kimi_code", "xai"):
    (shared / family).mkdir()
    (shared / family / "oauth.json").write_text('{"marker": "rotating-grant"}', encoding="utf-8")
  (operator / "xai").mkdir()
  (operator / "xai" / "oauth.json").write_text('{"marker": "operator"}', encoding="utf-8")
  private = backend.placement_root / "17" / "user-data"
  environ = backend._environment(private, "17")
  environ.pop("KIMI_CODE_AUTH_STORE_PATH", None)
  environ.pop("KIMI_AUTH_STORE_PATH", None)
  environ.pop("XAI_AUTH_STORE_PATH", None)

  backend._seed_org_credentials(private, environ, "17")

  assert environ["KIMI_CODE_AUTH_STORE_PATH"] == str(shared / "kimi_code" / "oauth.json")
  assert environ["XAI_AUTH_STORE_PATH"] == str(shared / "xai" / "oauth.json")
  assert not (private / "kimi_code" / "oauth.json").exists()
  assert not (private / "xai" / "oauth.json").exists()


def test_non_owner_placement_gets_no_personal_seat_or_live_binding(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend, shared, operator = _seeding_backend(
    tmp_path, monkeypatch,
    {
      "GATEWAY_USER_KEYS": json.dumps([{"slug": "henry", "risk_user_id": 1, "role": "owner"}]),
      "KIMI_CODE_AUTH_STORE_PATH": "/inherited/kimi/oauth.json",
      "XAI_AUTH_STORE_PATH": "/inherited/xai/oauth.json",
    },
  )
  _ollama_store(shared, "shared-edge-key")
  (shared / "xai").mkdir()
  (shared / "xai" / "credentials.json").write_text(
    json.dumps({"api_key": "xai-org-key"}), encoding="utf-8",
  )
  (shared / "google").mkdir()
  (shared / "google" / "oauth.json").write_text(
    json.dumps({"access_token": "personal-google-access"}), encoding="utf-8",
  )
  for base in (shared, operator):
    for family in ("kimi_code", "xai"):
      (base / family).mkdir(exist_ok=True)
      (base / family / "oauth.json").write_text('{"marker": "rotating-grant"}', encoding="utf-8")
  rotating = AnthropicOAuthRecord(
    identity="rotating@example.test", access_token="rotating-access-marker",
    refresh_token="rotating-refresh-marker", expires_at=9_999_999_999.0,
    created_at=1000.0, updated_at=1000.0,
  )
  setup = replace(
    rotating, identity="setup@example.test", access_token="personal-setup-marker",
    refresh_token="", credential_kind="auth_token",
  )
  api_key = replace(
    rotating, identity="api-key-org", access_token="sk-ant-api03-org-marker",
    refresh_token="", credential_kind="api_key", scope="",
  )
  for record in (rotating, setup, api_key):
    upsert_anthropic_oauth_record(shared / "anthropic" / "oauth.json", record)
  private = backend.placement_root / "17" / "user-data"
  environ = backend._environment(private, "17")

  backend._seed_org_credentials(private, environ, "17")

  candidates, _ = resolve_ollama_cloud_pool_candidates(
    environ={"USER_DATA_DIR": str(private)}, user_scope="17",
  )
  assert [candidate.record["api_key"] for candidate in candidates] == ["shared-edge-key"]
  xai_store = private / "xai" / "credentials.json"
  assert json.loads(xai_store.read_text(encoding="utf-8"))["api_key"] == "xai-org-key"
  assert not (private / "google").exists()
  seeded = private / "anthropic" / "oauth.json"
  assert [entry["access_token"] for entry in json.loads(seeded.read_text(encoding="utf-8"))["static_credentials"]] == [
    "sk-ant-api03-org-marker",
  ]
  assert json.loads(seeded.read_text(encoding="utf-8"))["accounts"] == []
  for path in private.rglob("*.json"):
    text = path.read_text(encoding="utf-8")
    assert "personal" not in text and "rotating-" not in text
  assert "KIMI_CODE_AUTH_STORE_PATH" not in environ
  assert "KIMI_AUTH_STORE_PATH" not in environ
  assert "XAI_AUTH_STORE_PATH" not in environ
  assert not (private / "kimi_code" / "oauth.json").exists()
  assert not (private / "xai" / "oauth.json").exists()


def test_non_owner_environment_carries_no_personal_credential_material(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  owner_env_names = sorted(
    (ANTHROPIC_CREDENTIAL_POOL_ENV_NAMES - {ANTHROPIC_STORE_BASE_ENV})
    | XAI_PERSONAL_OAUTH_ENV_NAMES
    | KIMI_CODE_PERSONAL_OAUTH_ENV_NAMES
    | GOOGLE_ANTIGRAVITY_PERSONAL_OAUTH_ENV_NAMES
    | GOOGLE_PERSONAL_OAUTH_ENV_NAMES
    | COPILOT_PERSONAL_OAUTH_ENV_NAMES
    | MUSE_PERSONAL_OAUTH_ENV_NAMES
    | OPENAI_PERSONAL_OAUTH_ENV_NAMES
    | CODEX_PERSONAL_OAUTH_ENV_NAMES
    # Literal wire names, independent of the product's own sets: a rename or a
    # dropped member of a provider set must still fail this regression.
    | {"OPENAI_AUTH_TOKEN", "CODEX_AUTH_TOKEN", "CODEX_AUTH_JSON"}
  )
  assert {
    "ANTHROPIC_AUTH_TOKEN", "XAI_REFRESH_TOKEN", "MUSE_AUTH_STORE_PATH",
    "OPENAI_AUTH_TOKEN", "CODEX_AUTH_TOKEN", "CODEX_AUTH_JSON",
  } <= set(owner_env_names)
  backend, _, _ = _seeding_backend(tmp_path, monkeypatch, {
    **{name: f"personal-{name}" for name in owner_env_names},
    "ANTHROPIC_API_KEY": "org-api-key",
  })
  private = backend.placement_root / "17" / "user-data"

  environ = backend._environment(private, "17")

  assert [name for name in owner_env_names if name in environ] == []
  assert environ["ANTHROPIC_API_KEY"] == "org-api-key"


def test_placement_keeps_an_oauth_store_the_child_environment_already_names(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  backend, shared, _ = _seeding_backend(tmp_path, monkeypatch, _OWNER_17_ENV)
  for family in ("kimi_code", "xai"):
    (shared / family).mkdir()
    (shared / family / "oauth.json").write_text('{"marker": "rotating-grant"}', encoding="utf-8")
  private = backend.placement_root / "17" / "user-data"
  environ = backend._environment(private, "17")
  environ["KIMI_CODE_AUTH_STORE_PATH"] = "/explicit/kimi/oauth.json"
  environ["XAI_AUTH_STORE_PATH"] = "/explicit/xai/oauth.json"

  backend._seed_org_credentials(private, environ, "17")

  assert environ["KIMI_CODE_AUTH_STORE_PATH"] == "/explicit/kimi/oauth.json"
  assert environ["XAI_AUTH_STORE_PATH"] == "/explicit/xai/oauth.json"


def test_agentcore_runtime_gets_no_seeded_org_credentials_or_oauth_binding(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway import runtime_placement_agentcore

  monkeypatch.setenv("HOME", str(tmp_path / "home"))
  _ollama_store(tmp_path / "home" / ".agent_gateway", "operator-home-key")
  backend = object.__new__(runtime_placement_agentcore.AgentCoreRuntimeBackend)
  backend._extra_env = {
    "KIMI_CODE_AUTH_STORE_PATH": str(tmp_path / "shared/kimi/oauth.json"),
    "XAI_AUTH_STORE_PATH": str(tmp_path / "shared/xai/oauth.json"),
  }

  projected = backend._environment()
  assert "KIMI_CODE_AUTH_STORE_PATH" not in projected
  assert "XAI_AUTH_STORE_PATH" not in projected
  environ = _agentcore_environment(tmp_path, monkeypatch, "7331")

  root = Path(environ["USER_DATA_DIR"])
  assert not (root / "ollama_cloud").exists()
  assert not (root / "anthropic").exists()
  assert "KIMI_CODE_AUTH_STORE_PATH" not in environ
  assert "XAI_AUTH_STORE_PATH" not in environ


def _legacy_approval_request():
  from datetime import datetime, timedelta, timezone
  from agent_gateway.approval_policy import ApprovalRequest

  now = datetime.now(timezone.utc)
  return ApprovalRequest(
    approval_id="approval-before-placement-cutover",
    tool_call_id="tool-before-placement-cutover",
    parent_approval_id=None, approval_chain_id="chain-before-placement-cutover",
    request_id="request-before-placement-cutover", session_id="session-before-placement-cutover",
    run_id="run-before-placement-cutover", user_id="17", profile="analyst", channel="tui",
    tool_name="memory_write", tool_class="state_write", tool_args_redacted={},
    args_hash="migration-args-hash", reason="review before migration",
    blast_radius_summary="state_write:memory_write", state="pending_user",
    requested_at=now, expires_at=now + timedelta(days=1), approval_constraint="standard",
  )


def test_explicit_approval_migration_preserves_pending_ids_for_edge_decision(
  tmp_path: Path, backend: LocalProcessRuntimeBackend, caplog: pytest.LogCaptureFixture,
) -> None:
  from datetime import datetime, timezone
  from agent_gateway.approval_policy import ApprovalVote
  from agent_gateway.approval_remote import ApprovalRemoteMigrationIncomplete
  from agent_gateway.approval_store import SQLiteApprovalStore
  from api.runtime_approval_service import RuntimeApprovalService, storage_ledger_path

  first = backend.start("17", IDENTITY, _desk(tmp_path))
  source = backend.placement_root / "17" / "user-data" / "gateway" / "approvals.sqlite3"
  source.parent.mkdir(parents=True, mode=0o700)
  legacy = SQLiteApprovalStore(source)
  request = _legacy_approval_request()
  asyncio.run(legacy.create(request))
  leftover = source.parent / "autonomous-approval-decisions.jsonl"
  leftover.write_text("legacy decision journal remains untouched\n")
  unplaced = tmp_path / "shared" / "gateway" / "approvals.sqlite3"
  unplaced.parent.mkdir(parents=True)
  unplaced.write_bytes(b"shared unplaced ledger remains untouched")
  services_root = tmp_path / "approval-services"
  destination = storage_ledger_path(services_root, "17")
  events = []

  with RuntimeApprovalService(
    services_root=services_root, audit_hmac_secret=b"migration-test-audit-secret",
    audit_hmac_key_id="migration-test",
  ) as service:
    previous = service.bind("17", first.generation)

    def begin(storage_id):
      service.begin_migration(storage_id)
      events.append("begin")
      assert backend.health(first)
      with pytest.raises(ApprovalRemoteMigrationIncomplete):
        previous.call("request.get", {"approval_id": request.approval_id})

    def complete(storage_id):
      assert not source.exists()
      assert not backend.health(first)
      assert any("pending_count=1" in record.message for record in caplog.records)
      service.complete_migration(storage_id)
      events.append("complete")

    with caplog.at_level("INFO", logger=placement_module.__name__):
      result = backend.migrate_approvals(
        "17", destination=destination, begin_migration=begin, complete_migration=complete,
        note_placement_stopped=service.note_placement_stopped,
      )

    assert events == ["begin", "complete"]
    assert result.source == source
    assert result.destination == destination
    assert result.pending_count == 1
    assert result.placement.generation == first.generation + 1
    assert result.placement.state == "ready"
    assert not source.exists()
    assert leftover.read_text() == "legacy decision journal remains untouched\n"
    assert unplaced.read_bytes() == b"shared unplaced ledger remains untouched"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    current = service.bind("17", result.placement.generation)
    restored = current.call("request.get", {"approval_id": request.approval_id})
    assert restored == request
    decided = current.call("request.vote", {
      "approval_id": request.approval_id,
      "vote": ApprovalVote(
        vote_id="vote-after-placement-cutover", approval_id=request.approval_id,
        decider_id="17", decider_role="owner", decision="approved",
        decision_reason="reviewed after migration", decided_at=datetime.now(timezone.utc),
      ),
    })
    assert decided.state == "approved"
    assert decided.approval_id == request.approval_id
    assert decided.tool_call_id == request.tool_call_id


@pytest.mark.parametrize("failure", ["existing_destination", "checkpoint", "partial_copy"])
def test_failed_approval_migration_keeps_generation_stopped_and_edge_fenced(
  tmp_path: Path, backend: LocalProcessRuntimeBackend, monkeypatch: pytest.MonkeyPatch,
  failure: str,
) -> None:
  from agent_gateway.approval_remote import ApprovalRemoteMigrationIncomplete
  from agent_gateway.approval_store import SQLiteApprovalStore
  from api.runtime_approval_service import RuntimeApprovalService, storage_ledger_path

  first = backend.start("17", IDENTITY, _desk(tmp_path))
  source = backend.placement_root / "17" / "user-data" / "gateway" / "approvals.sqlite3"
  source.parent.mkdir(parents=True, mode=0o700)
  legacy = SQLiteApprovalStore(source)
  request = _legacy_approval_request()
  asyncio.run(legacy.create(request))
  services_root = tmp_path / "approval-services"
  destination = storage_ledger_path(services_root, "17")
  with RuntimeApprovalService(
    services_root=services_root, audit_hmac_secret=b"migration-test-audit-secret",
    audit_hmac_key_id="migration-test",
  ) as service:
    closure = service.bind("17", first.generation)
    if failure == "existing_destination":
      destination.parent.mkdir(parents=True, mode=0o700)
      destination.write_bytes(b"do not overwrite existing edge ledger")
      expected_error = FileExistsError
    elif failure == "checkpoint":
      def blocked_checkpoint(self):
        raise RuntimeError("checkpoint remains busy")

      monkeypatch.setattr(SQLiteApprovalStore, "checkpoint_for_migration", blocked_checkpoint)
      expected_error = RuntimeError
    else:
      def interrupted_copy(old, relocated):
        relocated.write(old.read(128))
        raise RuntimeError("copy interrupted")

      monkeypatch.setattr(placement_module.shutil, "copyfileobj", interrupted_copy)
      expected_error = RuntimeError

    with pytest.raises(expected_error):
      backend.migrate_approvals(
        "17", destination=destination,
        begin_migration=service.begin_migration, complete_migration=service.complete_migration,
        note_placement_stopped=service.note_placement_stopped,
      )
    stopped = backend.store.get("17")
    assert stopped is not None
    assert stopped.state == "stopped"
    assert not backend.health(first)
    assert source.exists()
    assert asyncio.run(legacy.get(request.approval_id)) == request
    if failure == "existing_destination":
      assert destination.read_bytes() == b"do not overwrite existing edge ledger"
    else:
      assert not destination.exists()
    with pytest.raises(ApprovalRemoteMigrationIncomplete):
      closure.call("request.get", {"approval_id": request.approval_id})
