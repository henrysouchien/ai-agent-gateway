from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
import json
from pathlib import Path
import stat
import threading
from typing import Any
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest
from botocore.credentials import Credentials
from botocore.exceptions import ClientError
from websockets.asyncio.client import connect as websocket_connect
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from hank_contracts.agent_api import AGENT_API_CLAIM_ENV_VARS
from hank_contracts.user_claims import derive_user_claim_key
from agent_gateway.claim_signing_authority import GatewayClaimSigningAuthority
from agent_gateway.runtime_placement import DESK_LEASE_SECONDS, DeskLeaseRefused, RuntimePlacement
from agent_gateway.runtime_placement_agentcore import (
  AgentCoreRuntimeBackend,
  desk_prelease_policy,
  desk_session_policy,
  generate_ws_connection,
)
from agent_gateway.runtime_tunnel import TunnelEndpoint, TunnelRequest, TunnelResponse
from hank_model_transport.user_runtime_credentials import (
  DeskReplica,
  USER_RUNTIME_CREDENTIALS_FILE_ENV,
  UserRuntimeIdentity,
  derive_approval_claim_key,
  load_user_runtime_credentials,
)


ORG_KEY = "agentcore-test-org-authority-at-least-32-bytes"
RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-2:123456789012:runtime/test_runtime-abc123"
ROLE_ARN = "arn:aws:iam::123456789012:role/desk-replica"
BUCKET = "private-desk-replicas"
SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"
IDENTITY = UserRuntimeIdentity(
  key="alice-private-api-key", channel="excel", slug="alice",
  email="alice@example.test", risk_user_id=17, role="analyst", capabilities=("research",),
)
SHARED_ENV = {
  "AGENT_API_USER_CLAIM_HMAC_KEY": ORG_KEY,
  "GATEWAY_USER_KEYS": '[{"key":"other-user-secret"}]',
  "GATEWAY_API_KEY_WEB": "shared-web-authority",
  "GATEWAY_RESOLVER_HMAC_KEY": "shared-resolver-signing-authority",
  "RISK_MODULE_RESOLVER_URL": "https://private-resolver.example.test",
  "RISK_API_URL": "https://private-risk.example.test",
  "EDGAR_API_KEY": "shared-edgar-authority",
  "ANTHROPIC_API_KEY": "shared-model-authority",
  "AWS_ACCESS_KEY_ID": "shared-aws-access",
  "AWS_SECRET_ACCESS_KEY": "shared-aws-secret",
  "AWS_SESSION_TOKEN": "shared-aws-session",
  "GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS": '{"test":"public-key"}',
  "GATEWAY_APPROVAL_DB_PATH": "/edge/private/approvals.sqlite3",
  "AGENT_AUTONOMOUS_APPROVALS_DB_PATH": "/edge/private/autonomous-approvals.sqlite3",
  "AGENT_AUTONOMOUS_APPROVAL_DECISIONS_PATH": "/edge/private/approval-decisions.jsonl",
  "GATEWAY_APPROVAL_AUDIT_DIR": "/edge/private/approval-audit",
  "HANK_APPROVALS_EGRESS_URL": "https://private-approvals.example.test",
}
FIRST_CHUNK = b'data: {"type":"delta","text":"first"}\n\n'
LAST_CHUNK = b'data: {"type":"complete"}\n\n'
LEASE_DEADLINE = datetime(2026, 10, 30, 12, 34, 56, 123456, tzinfo=timezone.utc)


async def _body(*chunks: bytes) -> AsyncIterator[bytes]:
  for chunk in chunks:
    yield chunk


class _FakeSession:
  def __init__(self) -> None:
    self.credentials = Credentials("EDGEACCESS", "edge-secret", "edge-session-token")
    self.client_calls: list[tuple[str, str | None]] = []
    self.assume_calls: list[dict[str, Any]] = []
    self.stop_calls: list[dict[str, Any]] = []
    self.expirations: list[datetime] = []
    self.stop_error: ClientError | None = None

  def client(self, name: str, region_name: str | None = None) -> _FakeSession:
    assert name in {"sts", "bedrock-agentcore"}
    self.client_calls.append((name, region_name))
    return self

  def get_credentials(self) -> Credentials:
    return self.credentials

  def assume_role(self, **kwargs: Any) -> dict[str, Any]:
    self.assume_calls.append(kwargs)
    ordinal = len(self.assume_calls)
    expiration = (
      self.expirations.pop(0) if self.expirations
      else datetime.now(timezone.utc) + timedelta(hours=1)
    )
    return {"Credentials": {
      "AccessKeyId": f"DESKACCESS{ordinal}",
      "SecretAccessKey": f"desk-secret-{ordinal}",
      "SessionToken": f"desk-token-{ordinal}",
      "Expiration": expiration,
    }}

  def stop_runtime_session(self, **kwargs: Any) -> dict[str, Any]:
    assert kwargs["agentRuntimeArn"] == RUNTIME_ARN
    assert isinstance(kwargs["runtimeSessionId"], str)
    self.stop_calls.append(kwargs)
    if self.stop_error is not None:
      raise self.stop_error
    return {}


class _ExpiredWriterOwner:
  """Stub the lease owner's successful CAS after its prior deadline."""

  def __init__(self, prior: dict[str, Any], epoch: int) -> None:
    self.prior = prior
    self.now = prior["expires_at"] + 1
    self.epoch = epoch

  def acquire(self, holder: str) -> dict[str, Any]:
    assert self.now > self.prior["expires_at"]
    return {
      "epoch": self.epoch, "holder": holder,
      "expires_at": self.now + DESK_LEASE_SECONDS, "state": "held",
    }


class _FakeRuntime:
  """Serve on another loop: backend.start()/health() intentionally block the caller."""

  def __init__(self) -> None:
    self.loop = asyncio.new_event_loop()
    self.thread = threading.Thread(target=self._run, daemon=True, name="fake-agentcore-runtime")
    self.server: Server | None = None
    self.url = ""
    self.connections: dict[str, ServerConnection] = {}
    self.bootstraps: list[tuple[str, dict[str, Any]]] = []
    self.credential_updates: list[tuple[str, dict[str, Any]]] = []
    self.initializations: dict[str, dict[str, Any]] = {}
    self.leases: dict[str, dict[str, Any]] = {}
    self.next_epoch = 41
    self.refusal: str | None = None
    self.refusal_writer: dict[str, Any] | None = None
    self.lease_override: dict[str, Any] | None = None
    self.writer_owner: Any = None
    self.product_requests: list[tuple[str, TunnelRequest]] = []
    self.edge_frames: list[dict[str, Any]] = []
    self.runtime_frames: list[dict[str, Any]] = []
    self.connect_calls: list[tuple[str, dict[str, str]]] = []
    self.changed = threading.Condition()
    self.stream_release: asyncio.Event | None = None
    self.ready_gate: asyncio.Event | None = None
    self.health_status = 200
    self.thread.start()
    try:
      asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(timeout=5)
    except BaseException:
      self.loop.call_soon_threadsafe(self.loop.stop)
      self.thread.join(timeout=5)
      self.loop.close()
      raise

  def _run(self) -> None:
    asyncio.set_event_loop(self.loop)
    self.loop.run_forever()

  async def _start(self) -> None:
    self.server = await serve(self._connection, "127.0.0.1", 0)
    assert self.server.sockets
    port = self.server.sockets[0].getsockname()[1]
    self.url = f"ws://127.0.0.1:{port}/ws"

  def connect(
    self, url: str, *, additional_headers: Mapping[str, str], **kwargs: Any,
  ) -> Any:
    self.connect_calls.append((url, dict(additional_headers)))
    return websocket_connect(self.url, additional_headers=additional_headers, **kwargs)

  async def _connection(self, websocket: ServerConnection) -> None:
    assert websocket.request is not None
    session_id = websocket.request.headers[SESSION_HEADER]
    self.connections[session_id] = websocket

    async def send(text: str) -> None:
      self.runtime_frames.append(json.loads(text))
      await websocket.send(text)

    async def control(kind: str, body: dict[str, Any]) -> None:
      with self.changed:
        if kind == "bootstrap":
          self.bootstraps.append((session_id, body))
          self.initializations.setdefault(session_id, body)
        elif kind == "credentials":
          self.credential_updates.append((session_id, body))
        else:
          raise AssertionError(f"Unexpected edge control: {kind}")
        self.changed.notify_all()
      if kind == "bootstrap":
        if self.writer_owner is not None and session_id not in self.leases:
          try:
            self.leases[session_id] = self.writer_owner.acquire(session_id)
          except DeskLeaseRefused as exc:
            await endpoint.control("failed", {
              "error": "DeskLeaseRefused", "finding": str(exc),
              "writer": exc.record.model_dump() if exc.record is not None else None,
            })
            return
        if self.refusal is not None:
          await endpoint.control("failed", {
            "error": "DeskLeaseRefused", "finding": self.refusal, "writer": self.refusal_writer,
          })
          return
        if session_id not in self.leases:
          self.leases[session_id] = {
            "epoch": self.next_epoch, "holder": f"runtime-{session_id}",
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=DESK_LEASE_SECONDS)).timestamp(),
            "state": "held",
          }
          self.next_epoch += 1
        await endpoint.control("lease", self.leases[session_id] if self.lease_override is None else self.lease_override)
      elif kind == "credentials":
        if "bundle" in body:
          assert body["bundle"]["generation"] == self.leases[session_id]["epoch"]
        if self.ready_gate is not None:
          await self.ready_gate.wait()
        await endpoint.control("ready", {"generation": self.leases[session_id]["epoch"]})

    async def product(request: TunnelRequest) -> TunnelResponse:
      self.product_requests.append((session_id, request))
      path = urlsplit(request.path).path
      if path == "/api/health":
        return TunnelResponse(
          self.health_status, [("content-type", "application/json")],
          _body(json.dumps({"status": "ok" if self.health_status == 200 else "error"}).encode()),
        )
      if path == "/api/chat/init":
        assert request.method == "POST"
        assert session_id in self.initializations
        return TunnelResponse(200, [("content-type", "application/json")], _body(
          json.dumps({"session_id": f"product-{session_id}", "session_token": "user-token"}).encode(),
        ))
      if path == "/api/chat":
        self.stream_release = asyncio.Event()

        async def stream() -> AsyncIterator[bytes]:
          yield FIRST_CHUNK
          assert self.stream_release is not None
          await self.stream_release.wait()
          yield LAST_CHUNK

        return TunnelResponse(201, [
          ("content-type", "text/event-stream"), ("x-runtime-response", "preserved"),
          ("set-cookie", "one=1"), ("set-cookie", "two=2"),
        ], stream())
      if path == "/api/egress-probe":
        response = await endpoint.request(
          "POST", "/risk/api/positions?book=long%2Fshort",
          [("x-user-request", "runtime-value")], request.body,
        )
        return TunnelResponse(response.status, response.headers, response.body)
      return TunnelResponse(404, [], _body(b"missing"))

    endpoint = TunnelEndpoint(send=send, handler=product, on_control=control, odd_ids=True)
    try:
      async for text in websocket:
        assert isinstance(text, str)
        self.edge_frames.append(json.loads(text))
        await endpoint.feed(text)
    except ConnectionClosed:
      pass
    finally:
      endpoint.close()
      if self.connections.get(session_id) is websocket:
        self.connections.pop(session_id, None)

  def wait_bootstraps(self, count: int) -> None:
    with self.changed:
      assert self.changed.wait_for(lambda: len(self.bootstraps) >= count, timeout=5)

  def wait_credentials(self, count: int) -> None:
    with self.changed:
      assert self.changed.wait_for(lambda: len(self.credential_updates) >= count, timeout=5)

  def release_stream(self) -> None:
    assert self.stream_release is not None
    self.loop.call_soon_threadsafe(self.stream_release.set)

  def renew(self, session_id: str, expires_at: float) -> None:
    async def notify() -> None:
      record = {**self.leases[session_id], "expires_at": expires_at}
      self.leases[session_id] = record
      await self.connections[session_id].send(json.dumps({
        "t": "control", "kind": "lease", "body": record,
      }))

    asyncio.run_coroutine_threadsafe(notify(), self.loop).result(timeout=5)

  def drop(self, session_id: str) -> None:
    async def disconnect() -> None:
      await self.connections[session_id].close(code=1012, reason="transient runtime socket loss")

    asyncio.run_coroutine_threadsafe(disconnect(), self.loop).result(timeout=5)

  def hold_ready(self) -> None:
    async def hold() -> None:
      self.ready_gate = asyncio.Event()

    asyncio.run_coroutine_threadsafe(hold(), self.loop).result(timeout=5)

  def release_ready(self) -> None:
    if self.ready_gate is not None:
      self.loop.call_soon_threadsafe(self.ready_gate.set)

  def close(self) -> None:
    async def shutdown() -> None:
      if self.stream_release is not None:
        self.stream_release.set()
      if self.ready_gate is not None:
        self.ready_gate.set()
      if self.server is not None:
        self.server.close()
        await self.server.wait_closed()
      tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
      for task in tasks:
        task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

    try:
      asyncio.run_coroutine_threadsafe(shutdown(), self.loop).result(timeout=5)
    finally:
      self.loop.call_soon_threadsafe(self.loop.stop)
      self.thread.join(timeout=5)
      assert not self.thread.is_alive()
      self.loop.close()


@pytest.fixture
def fake_runtime() -> Iterator[_FakeRuntime]:
  runtime = _FakeRuntime()
  try:
    yield runtime
  finally:
    runtime.close()


@pytest.fixture
def aws_session() -> _FakeSession:
  return _FakeSession()


@pytest.fixture
def egress_calls() -> list[tuple[UserRuntimeIdentity, str, int, Mapping[str, str]]]:
  return []


@pytest.fixture
def egress_requests() -> list[TunnelRequest]:
  return []


@pytest.fixture
def backend(
  tmp_path: Path, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
  egress_calls: list[tuple[UserRuntimeIdentity, str, int, Mapping[str, str]]],
  egress_requests: list[TunnelRequest],
) -> Iterator[AgentCoreRuntimeBackend]:
  def egress_factory(
    identity: UserRuntimeIdentity, storage_id: str, placement_generation: int, config: Mapping[str, str],
  ) -> Any:
    egress_calls.append((identity, storage_id, placement_generation, dict(config)))

    async def handle(request: TunnelRequest) -> TunnelResponse:
      egress_requests.append(request)
      return TunnelResponse(202, [("x-edge-response", "preserved")], _body(b"edge-response"))

    return handle

  runtime = AgentCoreRuntimeBackend(
    placement_root=tmp_path / "placements", org_claim_key=ORG_KEY,
    runtime_arn=RUNTIME_ARN, desk_bucket=BUCKET, desk_role_arn=ROLE_ARN,
    extra_env=SHARED_ENV, egress_factory=egress_factory,
    session=aws_session, connect=fake_runtime.connect, ready_timeout=5,
  )
  try:
    yield runtime
  finally:
    runtime.close()
    assert not runtime._thread.is_alive()
    assert runtime._loop.is_closed()


def _desk() -> DeskReplica:
  return DeskReplica(
    url=f"s3://{BUCKET}", aws_access_key_id=None,
    aws_secret_access_key=None, aws_session_token=None, expires_at=None,
  )


def test_desk_session_policy_grants_only_the_users_replica_prefix() -> None:
  assert desk_session_policy(BUCKET, "17", LEASE_DEADLINE.timestamp()) == {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        "Resource": f"arn:aws:s3:::{BUCKET}/users/17/*",
        "Condition": {"DateLessThan": {"aws:CurrentTime": "2026-10-30T12:34:56.123456Z"}},
      },
      {
        "Effect": "Allow",
        "Action": "s3:ListBucket",
        "Resource": f"arn:aws:s3:::{BUCKET}",
        "Condition": {
          "StringLike": {"s3:prefix": ["users/17/*", "users/17/"]},
        },
      },
    ],
  }


def test_prelease_policy_can_only_cas_writer_and_read_desk_until_deadline() -> None:
  policy = desk_prelease_policy(BUCKET, "17", LEASE_DEADLINE.timestamp())
  deadline = {"DateLessThan": {"aws:CurrentTime": "2026-10-30T12:34:56.123456Z"}}
  assert policy == {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
        "Resource": f"arn:aws:s3:::{BUCKET}/users/17/writer.json", "Condition": deadline,
      },
      {
        "Effect": "Allow", "Action": "s3:GetObject",
        "Resource": f"arn:aws:s3:::{BUCKET}/users/17/*", "Condition": deadline,
      },
      {
        "Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{BUCKET}",
        "Condition": {"StringLike": {"s3:prefix": ["users/17/*", "users/17/"]}},
      },
    ],
  }


@pytest.mark.parametrize(("storage_id", "prefix", "allowed"), [
  ("17", "users/17/", True),
  ("17", "users/17/research.db", True),
  ("17", "users/17", False),
  ("17", "users/1", False),
  ("17", "users/170/", False),
  ("17", "users/", False),
  ("1", "users/1/", True),
  ("1", "users/1/research.db", True),
  ("1", "users/1", False),
  ("1", "users/10/", False),
])
def test_listbucket_prefix_cannot_enumerate_numeric_siblings(
  storage_id: str, prefix: str, allowed: bool,
) -> None:
  policy = desk_session_policy(BUCKET, storage_id, LEASE_DEADLINE.timestamp())
  patterns = policy["Statement"][1]["Condition"]["StringLike"]["s3:prefix"]
  assert any(fnmatchcase(prefix, pattern) for pattern in patterns) is allowed


def test_websocket_sigv4_url_and_headers_are_deterministic(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  instant = datetime(2026, 10, 13, 12, 34, 56, tzinfo=timezone.utc)
  monkeypatch.setattr("botocore.auth.get_current_datetime", lambda: instant)
  credentials = Credentials(
    "AKIDEXAMPLE", "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", "fixed-session-token",
  )
  url, headers = generate_ws_connection(RUNTIME_ARN, "test-session-one", "us-east-2", credentials)
  assert url == (
    "wss://bedrock-agentcore.us-east-2.amazonaws.com/runtimes/"
    "arn%3Aaws%3Abedrock-agentcore%3Aus-east-2%3A123456789012%3Aruntime%2F"
    "test_runtime-abc123/ws?qualifier=DEFAULT"
  )
  assert headers == {
    "host": "bedrock-agentcore.us-east-2.amazonaws.com",
    "X-Amz-Date": "20261013T123456Z",
    "X-Amz-Security-Token": "fixed-session-token",
    "Authorization": (
      "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261013/us-east-2/bedrock-agentcore/aws4_request, "
      "SignedHeaders=host;x-amz-date;x-amz-security-token, "
      "Signature=45530c2616ea75f0f98b8352bb17843fefe9ca3039b78273959143b2d2dc7d4c"
    ),
    SESSION_HEADER: "test-session-one",
  }
  second_url, second_headers = generate_ws_connection(
    RUNTIME_ARN, "test-session-two", "us-east-2", credentials,
  )
  assert second_url == url
  assert second_headers == {**headers, SESSION_HEADER: "test-session-two"}
  # The SDK appends the session id after signing; the websocket library supplies
  # its own Upgrade, Connection, and Sec-WebSocket-* handshake headers.
  assert SESSION_HEADER.lower() not in headers["Authorization"].lower()
  assert not {"upgrade", "connection", "sec-websocket-key", "sec-websocket-version"} & {
    name.lower() for name in headers
  }


def test_cold_start_mints_prelease_only_after_websocket_connects(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import agent_gateway.runtime_placement_agentcore as owner

  now = [LEASE_DEADLINE.timestamp()]
  monkeypatch.setattr(owner, "time", SimpleNamespace(time=lambda: now[0]))

  @asynccontextmanager
  async def cold_connect(*args: Any, **kwargs: Any):
    # Connection establishment includes container cold start, not desk lease time.
    now[0] += DESK_LEASE_SECONDS * 2
    async with fake_runtime.connect(*args, **kwargs) as socket:
      yield socket

  backend._connect = cold_connect
  placement = backend.start("17", IDENTITY, _desk())
  assert placement.state == "ready"
  bootstrap = fake_runtime.bootstraps[0][1]
  deadline = datetime.fromisoformat(bootstrap["desk"]["credentials"]["Expiration"]).timestamp()
  assert deadline == now[0] + DESK_LEASE_SECONDS
  prelease = aws_session.assume_calls[0]
  assert prelease["RoleSessionName"] == "hank-desk-17-prelease"
  assert json.loads(prelease["Policy"]) == desk_prelease_policy(BUCKET, "17", deadline)
  assert bootstrap["lease_seconds"] == DESK_LEASE_SECONDS


def test_start_bootstraps_private_bundle_without_shared_authority(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
  egress_calls: list[tuple[UserRuntimeIdentity, str, int, Mapping[str, str]]],
) -> None:
  placement = backend.start("17", IDENTITY, _desk())
  assert placement.state == "ready"
  assert placement.backend == "agentcore"
  assert placement.generation == 41
  assert placement.handle is not None
  assert placement.url == f"agentcore://{placement.handle}"
  assert placement.bundle_path == str(backend.placement_root / "17" / "bundle.json")
  bundle_path = Path(placement.bundle_path)
  assert stat.S_IMODE(bundle_path.stat().st_mode) == 0o600
  bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: placement.bundle_path})
  assert bundle is not None
  assert bundle.user == IDENTITY
  assert bundle.desk == _desk()
  assert bundle.generation == placement.generation
  assert bundle.claim_signing.key == derive_user_claim_key(
    ORG_KEY, "alice", "alice@example.test",
  )
  assert bundle.approval_claim_signing.key == derive_approval_claim_key(
    ORG_KEY, "alice", "alice@example.test", placement.generation,
  )
  assert fake_runtime.bootstraps[0][0] == placement.handle
  bootstrap = fake_runtime.bootstraps[0][1]
  assert set(bootstrap) == {"storage_id", "desk", "env", "lease_seconds"}
  assert bootstrap["storage_id"] == placement.storage_id
  assert bootstrap["lease_seconds"] == DESK_LEASE_SECONDS
  assert fake_runtime.credential_updates[0][1]["bundle"] == json.loads(bundle_path.read_text())
  assert fake_runtime.credential_updates[0][1]["desk"]["AccessKeyId"] == "DESKACCESS2"
  assert bootstrap["desk"]["url"] == f"s3://{BUCKET}"
  assert bootstrap["desk"]["credentials"] == {
    "AccessKeyId": "DESKACCESS1", "SecretAccessKey": "desk-secret-1",
    "SessionToken": "desk-token-1",
    "Expiration": bootstrap["desk"]["credentials"]["Expiration"],
  }
  assert datetime.fromisoformat(bootstrap["desk"]["credentials"]["Expiration"]).tzinfo is not None
  environment = bootstrap["env"]
  assert environment["CHAT_API_SECRET"]
  assert {name: value for name, value in environment.items() if name != "CHAT_API_SECRET"} == {
    "GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS": SHARED_ENV["GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS"],
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8090/egress/anthropic",
    "RISK_API_URL": "http://127.0.0.1:8090/egress/risk",
    "EDGAR_API_URL": "http://127.0.0.1:8090/egress/edgar",
    "RISK_MODULE_RESOLVER_URL": "http://127.0.0.1:8090/egress/resolver/api/internal/resolve-credential",
    "HANK_APPROVALS_EGRESS_URL": "http://127.0.0.1:8090/egress/approvals",
    "HANK_APPROVAL_TRANSPORT": "agentcore",
  }
  wire = json.dumps(bootstrap)
  for name, authority in SHARED_ENV.items():
    if name != "GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS":
      assert authority not in wire
  for authority in ("EDGEACCESS", "edge-secret", "edge-session-token"):
    assert authority not in wire
  assert aws_session.client_calls == [("sts", "us-east-2"), ("bedrock-agentcore", "us-east-2")]
  assert len(aws_session.assume_calls) == 2
  prelease, held = aws_session.assume_calls
  prelease_deadline = json.loads(prelease["Policy"])["Statement"][0]["Condition"]["DateLessThan"]["aws:CurrentTime"]
  assert bootstrap["desk"]["credentials"]["Expiration"] == prelease_deadline
  assert prelease == {
    "RoleArn": ROLE_ARN, "RoleSessionName": "hank-desk-17-prelease", "DurationSeconds": 900,
    "Policy": json.dumps(desk_prelease_policy(
      BUCKET, "17", datetime.fromisoformat(prelease_deadline.replace("Z", "+00:00")).timestamp(),
    )),
  }
  assert held == {
    "RoleArn": ROLE_ARN, "RoleSessionName": "hank-desk-17-g41", "DurationSeconds": 900,
    "Policy": json.dumps(desk_session_policy(BUCKET, "17", fake_runtime.leases[placement.handle]["expires_at"])),
  }
  held_deadline = json.loads(held["Policy"])["Statement"][0]["Condition"]["DateLessThan"]["aws:CurrentTime"]
  assert fake_runtime.credential_updates[0][1]["desk"]["Expiration"] == held_deadline
  assert egress_calls == [(IDENTITY, "17", placement.generation, SHARED_ENV)]
  assert fake_runtime.product_requests == []
  assert backend.health(placement)
  assert fake_runtime.product_requests[-1] == (
    placement.handle, TunnelRequest("GET", "/api/health", [], b""),
  )
  fake_runtime.health_status = 503
  assert not backend.health(placement)
  fake_runtime.health_status = 200
  retained_bundle = bundle_path.read_bytes()
  assert backend.start("17", IDENTITY, _desk()) == placement
  assert len(fake_runtime.bootstraps) == 1
  assert len(aws_session.assume_calls) == 2
  assert bundle_path.read_bytes() == retained_bundle
  # SQLite is a routing index; neither derived authority nor AWS material lives
  # in it, and its sidecar WAL is held to the same boundary.
  for path in (backend.store.path, Path(str(backend.store.path) + "-wal")):
    if path.exists():
      for secret in (
        ORG_KEY, IDENTITY.key, bundle.claim_signing.key,
        bundle.approval_claim_signing.key, "desk-secret-1", "edge-secret",
      ):
        assert secret.encode() not in path.read_bytes()
  assert ORG_KEY not in repr(backend)
  assert IDENTITY.key not in repr(backend)


@pytest.mark.parametrize("writer_state", ("held", "released"))
def test_stopped_session_replacement_refuses_unexpired_writer(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
  egress_calls: list[tuple[UserRuntimeIdentity, str, int, Mapping[str, str]]],
  writer_state: str,
) -> None:
  from agent_gateway.runtime_placement import WriterRecord

  first = backend.start("17", IDENTITY, _desk())
  assert first.handle is not None
  writer = WriterRecord.model_validate({
    **fake_runtime.leases[first.handle], "state": writer_state,
  })
  assert writer.expires_at > datetime.now(timezone.utc).timestamp()
  attempts: list[str] = []

  class UnexpiredWriterOwner:
    def acquire(self, holder: str) -> dict[str, Any]:
      attempts.append(holder)
      raise DeskLeaseRefused("prior writer deadline has not expired", record=writer)

  fake_runtime.writer_owner = UnexpiredWriterOwner()
  bundle_before = Path(first.bundle_path).read_bytes()
  updates_before = len(fake_runtime.credential_updates)
  backend.stop(first)
  with pytest.raises(DeskLeaseRefused, match="prior writer deadline") as caught:
    backend.replace(first)
  assert caught.value.record is not None
  assert caught.value.record == writer
  assert caught.value.record.holder == writer.holder
  assert caught.value.record.epoch == first.generation
  assert len(attempts) == 1
  assert attempts[0] != first.handle
  stored = backend.store.get("17")
  assert stored is not None
  assert stored.state != "ready"
  assert stored.handle is None
  assert stored.generation == first.generation
  assert Path(first.bundle_path).read_bytes() == bundle_before
  assert len(egress_calls) == 1
  assert len(fake_runtime.credential_updates) == updates_before
  assert len(fake_runtime.leases) == 1
  assert fake_runtime.next_epoch == first.generation + 1
  assert len(aws_session.stop_calls) == 2


def test_stop_uses_runtime_session_and_replacement_advances_generation(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
  egress_calls: list[tuple[UserRuntimeIdentity, str, int, Mapping[str, str]]],
) -> None:
  first = backend.start("17", IDENTITY, _desk())
  original_bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: first.bundle_path})
  assert original_bundle is not None
  assert first.handle is not None
  fake_runtime.writer_owner = _ExpiredWriterOwner(fake_runtime.leases[first.handle], 508)
  replacement = backend.replace(first)
  assert replacement.state == "ready"
  assert replacement.generation == 508
  assert replacement.handle is not None
  assert replacement.handle != first.handle
  assert replacement.url != first.url
  assert replacement.bundle_path == first.bundle_path
  replacement_bundle = load_user_runtime_credentials({
    USER_RUNTIME_CREDENTIALS_FILE_ENV: replacement.bundle_path,
  })
  assert replacement_bundle is not None
  assert replacement_bundle.user == original_bundle.user
  assert replacement_bundle.desk == original_bundle.desk
  assert replacement_bundle.generation == replacement.generation
  assert replacement_bundle.claim_signing.key == derive_user_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email,
  )
  assert replacement_bundle.claim_signing.key == original_bundle.claim_signing.key
  assert replacement_bundle.approval_claim_signing.key == derive_approval_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email, replacement.generation,
  )
  assert replacement_bundle.approval_claim_signing.key != original_bundle.approval_claim_signing.key
  original_approval_authority = GatewayClaimSigningAuthority(original_bundle.approval_claim_signing.key)
  replacement_approval_authority = GatewayClaimSigningAuthority(replacement_bundle.approval_claim_signing.key)
  original_approval_environment = original_approval_authority.sign_user_claim(
    user_id=IDENTITY.slug, user_email=IDENTITY.email,
  )
  replacement_approval_environment = replacement_approval_authority.sign_user_claim(
    user_id=IDENTITY.slug, user_email=IDENTITY.email,
  )
  original_approval_claim = {
    field: original_approval_environment[env_name]
    for field, env_name in AGENT_API_CLAIM_ENV_VARS.items()
  }
  replacement_approval_claim = {
    field: replacement_approval_environment[env_name]
    for field, env_name in AGENT_API_CLAIM_ENV_VARS.items()
  }
  assert original_approval_authority.verify_user_claim(original_approval_claim) is not None
  assert replacement_approval_authority.verify_user_claim(original_approval_claim) is None
  assert replacement_approval_authority.verify_user_claim(replacement_approval_claim) is not None
  assert original_approval_authority.verify_user_claim(replacement_approval_claim) is None
  retained_bundle = Path(replacement.bundle_path).read_bytes()
  assert backend.start("17", IDENTITY, _desk()) == replacement
  assert Path(replacement.bundle_path).read_bytes() == retained_bundle
  assert aws_session.stop_calls == [{
    "agentRuntimeArn": RUNTIME_ARN, "runtimeSessionId": first.handle, "qualifier": "DEFAULT",
  }]
  assert [session_id for session_id, _ in fake_runtime.bootstraps] == [
    first.handle, replacement.handle,
  ]
  assert all("takeover_epoch" not in body for _, body in fake_runtime.bootstraps)
  assert [body["bundle"]["generation"] for _, body in fake_runtime.credential_updates] == [41, 508]
  assert egress_calls == [
    (IDENTITY, "17", first.generation, SHARED_ENV),
    (IDENTITY, "17", replacement.generation, SHARED_ENV),
  ]
  assert aws_session.assume_calls[3]["RoleSessionName"] == "hank-desk-17-g508"
  backend.stop(first)
  assert backend.store.get("17") == replacement
  assert len(aws_session.stop_calls) == 1
  assert backend.health(replacement)
  stopped = backend.stop(replacement)
  assert stopped.state == "stopped"
  assert stopped.handle is None
  assert backend.store.get("17") == stopped
  assert not backend.health(stopped)
  assert aws_session.stop_calls[-1] == {
    "agentRuntimeArn": RUNTIME_ARN, "runtimeSessionId": replacement.handle, "qualifier": "DEFAULT",
  }


@pytest.mark.parametrize("mismatch", ("generation", "generic_key", "approval_key", "both"))
def test_ready_bundle_must_match_current_placement_generation(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, mismatch: str,
) -> None:
  first = backend.start("17", IDENTITY, _desk())
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

  replacement = backend.start("17", IDENTITY, _desk())

  assert replacement.generation == first.generation + 1
  assert replacement.handle != first.handle
  bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: replacement.bundle_path})
  assert bundle is not None
  assert bundle.generation == replacement.generation
  assert bundle.claim_signing.key == derive_user_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email,
  )
  assert bundle.approval_claim_signing.key == derive_approval_claim_key(
    ORG_KEY, IDENTITY.slug, IDENTITY.email, replacement.generation,
  )
  assert fake_runtime.credential_updates[-1][1]["bundle"]["generation"] == replacement.generation
  assert backend.start("17", IDENTITY, _desk()) == replacement
  assert len(fake_runtime.bootstraps) == 2


def test_transport_initialization_and_streaming_preserve_http_request_and_response(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime,
) -> None:
  placement = backend.start("17", IDENTITY, _desk())
  path = "/api/chat?model=a%2Fb&name=Alice%20Desk&tag=one&tag=two"
  payload = b'{"message":"research"}\x00\xff'

  async def scenario() -> None:
    async with httpx.AsyncClient(
      transport=backend.transport(placement), base_url="http://user-runtime", timeout=5,
    ) as client:
      initialized = await client.post("/api/chat/init", json={"api_key": IDENTITY.key})
      assert initialized.status_code == 200
      assert initialized.json()["session_id"] == f"product-{placement.handle}"
      token = initialized.json()["session_token"]
      async with client.stream(
        "POST", path, content=payload,
        headers=[("Authorization", f"Bearer {token}"), ("X-Client-Detail", "one"), ("X-Client-Detail", "two")],
      ) as response:
        assert response.status_code == 201
        assert response.headers["content-type"] == "text/event-stream"
        assert response.headers["x-runtime-response"] == "preserved"
        assert response.headers.get_list("set-cookie") == ["one=1", "two=2"]
        chunks = response.aiter_raw()
        try:
          # The peer withholds its final chunk. Seeing this before releasing it
          # proves the cross-loop transport does not buffer an SSE response.
          assert await asyncio.wait_for(anext(chunks), timeout=5) == FIRST_CHUNK
        finally:
          fake_runtime.release_stream()
        assert b"".join([chunk async for chunk in chunks]) == LAST_CHUNK

  asyncio.run(asyncio.wait_for(scenario(), timeout=10))
  session_id, request = fake_runtime.product_requests[-1]
  assert session_id == placement.handle
  assert request.method == "POST"
  assert request.path == path
  assert request.body == payload
  assert ("authorization", "Bearer user-token") in request.headers
  assert [value for name, value in request.headers if name == "x-client-detail"] == ["one", "two"]
  assert all(frame["id"] % 2 == 0 for frame in fake_runtime.edge_frames if frame["t"] == "req")


def test_runtime_odd_id_egress_requests_share_the_product_tunnel(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime,
  egress_requests: list[TunnelRequest],
) -> None:
  placement = backend.start("17", IDENTITY, _desk())
  payload = b"runtime-body\x00\xff"

  async def scenario() -> None:
    async with httpx.AsyncClient(
      transport=backend.transport(placement), base_url="http://user-runtime", timeout=5,
    ) as client:
      response = await client.post("/api/egress-probe", content=payload)
      assert response.status_code == 202
      assert response.headers["x-edge-response"] == "preserved"
      assert response.content == b"edge-response"

  asyncio.run(asyncio.wait_for(scenario(), timeout=10))
  assert egress_requests == [TunnelRequest(
    "POST", "/risk/api/positions?book=long%2Fshort",
    [("x-user-request", "runtime-value")], payload,
  )]
  odd_requests = [frame for frame in fake_runtime.runtime_frames if frame["t"] == "req"]
  assert len(odd_requests) == 1
  assert odd_requests[0]["id"] == 1
  edge_requests = [frame for frame in fake_runtime.edge_frames if frame["t"] == "req"]
  assert len(edge_requests) == 1
  assert edge_requests[0]["id"] == 2


def test_socket_reconnect_reuses_session_and_idempotent_bootstrap(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
  egress_calls: list[tuple[UserRuntimeIdentity, str, int, Mapping[str, str]]],
) -> None:
  placement = backend.start("17", IDENTITY, _desk())
  assert placement.handle is not None
  first_bootstrap = fake_runtime.bootstraps[0][1]
  assert backend.health(placement)
  fake_runtime.drop(placement.handle)
  fake_runtime.wait_bootstraps(2)
  assert backend.health(placement)
  assert backend.store.get("17") == placement
  assert fake_runtime.bootstraps == [
    (placement.handle, first_bootstrap),
    (placement.handle, {
      **first_bootstrap,
      "desk": {**first_bootstrap["desk"], "credentials": fake_runtime.credential_updates[0][1]["desk"]},
    }),
  ]
  assert fake_runtime.initializations == {placement.handle: first_bootstrap}
  assert len(fake_runtime.connect_calls) == 2
  assert all(
    headers[SESSION_HEADER] == placement.handle for _, headers in fake_runtime.connect_calls
  )
  assert fake_runtime.connect_calls[0][0] == fake_runtime.connect_calls[1][0]
  assert len(aws_session.assume_calls) == 3
  assert aws_session.stop_calls == []
  assert len(egress_calls) == 1

  async def scenario() -> None:
    async with httpx.AsyncClient(
      transport=backend.transport(placement), base_url="http://user-runtime", timeout=5,
    ) as client:
      response = await client.post("/api/chat/init", json={"api_key": IDENTITY.key})
      assert response.status_code == 200
      assert response.json()["session_id"] == f"product-{placement.handle}"

  asyncio.run(asyncio.wait_for(scenario(), timeout=10))


def test_desk_credentials_remint_only_on_actual_lease_deadline_and_reconnect(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
) -> None:
  # An expiring STS session does not allocate a fresh write deadline.
  aws_session.expirations.extend([
    datetime.now(timezone.utc) + timedelta(seconds=1),
    datetime.now(timezone.utc) + timedelta(seconds=1),
  ])
  placement = backend.start("17", IDENTITY, _desk())
  assert placement.handle is not None
  original_bundle = Path(placement.bundle_path).read_bytes()
  initial_update = fake_runtime.credential_updates[0][1]
  initial_record = fake_runtime.leases[placement.handle]
  with fake_runtime.changed:
    assert not fake_runtime.changed.wait_for(lambda: len(fake_runtime.credential_updates) > 1, timeout=1.2)
  assert len(aws_session.assume_calls) == 2
  assert len(fake_runtime.credential_updates) == 1
  renewed_deadline = initial_record["expires_at"] + 37.125
  fake_runtime.renew(placement.handle, renewed_deadline)
  fake_runtime.wait_credentials(2)
  session_id, update = fake_runtime.credential_updates[1]
  assert session_id == placement.handle
  assert set(update) == {"desk"}
  assert update["desk"]["AccessKeyId"] == "DESKACCESS3"
  assert initial_update["desk"]["AccessKeyId"] == "DESKACCESS2"
  assert len(aws_session.assume_calls) == 3
  assert aws_session.assume_calls[2] == {
    "RoleArn": ROLE_ARN, "RoleSessionName": f"hank-desk-17-g{placement.generation}",
    "DurationSeconds": 900,
    "Policy": json.dumps(desk_session_policy(BUCKET, "17", renewed_deadline)),
  }
  old_condition = json.loads(aws_session.assume_calls[1]["Policy"])["Statement"][0]["Condition"]
  new_condition = json.loads(aws_session.assume_calls[2]["Policy"])["Statement"][0]["Condition"]
  assert old_condition != new_condition
  assert new_condition == {"DateLessThan": {
    "aws:CurrentTime": datetime.fromtimestamp(renewed_deadline, timezone.utc).isoformat().replace("+00:00", "Z"),
  }}
  assert update["desk"]["Expiration"] == new_condition["DateLessThan"]["aws:CurrentTime"]
  assert Path(placement.bundle_path).read_bytes() == original_bundle
  assert backend.store.get("17") == placement
  assert backend.health(placement)
  fake_runtime.drop(placement.handle)
  fake_runtime.wait_bootstraps(2)
  latest_bootstrap = fake_runtime.bootstraps[-1][1]
  assert latest_bootstrap["desk"]["credentials"] == update["desk"]
  assert "generation" not in latest_bootstrap
  assert "bundle" not in latest_bootstrap
  assert backend.health(placement)
  assert len(aws_session.assume_calls) == 4
  assert aws_session.assume_calls[3]["Policy"] == aws_session.assume_calls[2]["Policy"]
  assert len(fake_runtime.initializations) == 1
  assert aws_session.stop_calls == []


def test_missing_aws_session_stops_locally_and_allows_a_new_generation(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
) -> None:
  placement = backend.start("17", IDENTITY, _desk())
  aws_session.stop_error = ClientError({
    "Error": {"Code": "ResourceNotFoundException", "Message": "Runtime session already expired"},
  }, "StopRuntimeSession")
  stopped = backend.stop(placement)
  assert stopped.state == "stopped"
  assert stopped.handle is None
  assert backend.store.get("17") == stopped
  assert not backend.health(stopped)
  aws_session.stop_error = None
  assert placement.handle is not None
  fake_runtime.writer_owner = _ExpiredWriterOwner(
    fake_runtime.leases[placement.handle], placement.generation + 1,
  )
  restarted = backend.start("17", IDENTITY, _desk())
  assert restarted.state == "ready"
  assert restarted.generation == placement.generation + 1
  assert restarted.handle is not None
  assert restarted.handle != placement.handle
  assert [session_id for session_id, _ in fake_runtime.bootstraps] == [
    placement.handle, restarted.handle,
  ]
  assert all("takeover_epoch" not in body for _, body in fake_runtime.bootstraps)
  assert [body["bundle"]["generation"] for _, body in fake_runtime.credential_updates] == [
    placement.generation, restarted.generation,
  ]
  assert aws_session.stop_calls == [{
    "agentRuntimeArn": RUNTIME_ARN, "runtimeSessionId": placement.handle, "qualifier": "DEFAULT",
  }]
  assert backend.health(restarted)


def test_aws_stop_errors_preserve_the_handle_for_retry(
  backend: AgentCoreRuntimeBackend, aws_session: _FakeSession,
) -> None:
  placement = backend.start("17", IDENTITY, _desk())
  aws_session.stop_error = ClientError({
    "Error": {"Code": "AccessDeniedException", "Message": "Runtime termination was denied"},
  }, "StopRuntimeSession")
  with pytest.raises(ClientError, match="AccessDeniedException"):
    backend.stop(placement)
  retained = backend.store.get("17")
  assert retained is not None
  assert retained.handle == placement.handle
  assert retained.generation == placement.generation
  assert aws_session.stop_calls == [{
    "agentRuntimeArn": RUNTIME_ARN, "runtimeSessionId": placement.handle, "qualifier": "DEFAULT",
  }]
  aws_session.stop_error = None
  stopped = backend.stop(retained)
  assert stopped.state == "stopped"
  assert stopped.handle is None
  assert backend.store.get("17") == stopped
  assert len(aws_session.stop_calls) == 2
  assert aws_session.stop_calls[1] == aws_session.stop_calls[0]


def test_stop_wakes_http_request_waiting_for_reconnect_readiness(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  placement = backend.start("17", IDENTITY, _desk())
  assert placement.handle is not None
  fake_runtime.hold_ready()
  fake_runtime.drop(placement.handle)
  fake_runtime.wait_bootstraps(2)
  waiting = threading.Event()
  ensure_session = backend._ensure_session

  async def observed_ensure_session(current: RuntimePlacement) -> Any:
    assert current.handle is not None
    assert not backend._sessions[current.handle].ready.is_set()
    waiting.set()
    return await ensure_session(current)

  monkeypatch.setattr(backend, "_ensure_session", observed_ensure_session)

  async def scenario() -> None:
    async with httpx.AsyncClient(
      transport=backend.transport(placement), base_url="http://user-runtime", timeout=5,
    ) as client:
      request = asyncio.create_task(client.post("/api/chat/init", json={"api_key": IDENTITY.key}))
      try:
        assert await asyncio.to_thread(waiting.wait, 5)
        stopped = backend.stop(placement)
        assert stopped.state == "stopped"
        assert stopped.handle is None
        # The ready timeout is five seconds; stop must settle this captured
        # waiter now, rather than leaving it attached to a disconnected session.
        with pytest.raises(httpx.ReadError, match="TunnelClosed"):
          await asyncio.wait_for(request, timeout=1)
      finally:
        fake_runtime.release_ready()
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)

  asyncio.run(asyncio.wait_for(scenario(), timeout=10))
  stored = backend.store.get("17")
  assert stored is not None
  assert stored.state == "stopped"


@pytest.mark.parametrize("refused_record", [
  None,
  {"epoch": 900, "holder": "another-runtime", "expires_at": 1793370000.0, "state": "held"},
])
def test_refused_restore_surfaces_finding_without_placement_bundle_or_generation(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
  egress_calls: list[tuple[UserRuntimeIdentity, str, int, Mapping[str, str]]],
  refused_record: dict[str, Any] | None,
) -> None:
  fake_runtime.refusal = "Finding: writer.json is held by another runtime at epoch 900"
  fake_runtime.refusal_writer = refused_record
  with pytest.raises(DeskLeaseRefused, match="Finding: writer.json is held") as caught:
    backend.start("17", IDENTITY, _desk())
  if refused_record is None:
    assert caught.value.record is None
  else:
    assert caught.value.record is not None
    assert caught.value.record.model_dump() == refused_record
  assert backend.store.get("17") is None
  assert not (backend.placement_root / "17" / "bundle.json").exists()
  assert egress_calls == []
  assert fake_runtime.credential_updates == []
  assert len(aws_session.assume_calls) == 1
  assert len(aws_session.stop_calls) == 1
  fake_runtime.refusal = None
  fake_runtime.next_epoch = 901
  placement = backend.start("17", IDENTITY, _desk())
  assert placement.generation == 901


@pytest.mark.parametrize("bad_record", [
  {"epoch": "41", "holder": "holder", "expires_at": 1793370000.0, "state": "held"},
  {"epoch": True, "holder": "holder", "expires_at": 1793370000.0, "state": "held"},
  {"epoch": 41, "holder": "holder", "expires_at": float("inf"), "state": "held"},
  {"epoch": 41, "holder": "holder", "expires_at": 1793370000.0, "state": "held", "extra": 1},
])
def test_lease_control_parses_writer_record_before_minting_or_placement(
  backend: AgentCoreRuntimeBackend, fake_runtime: _FakeRuntime, aws_session: _FakeSession,
  bad_record: dict[str, Any],
) -> None:
  from pydantic import ValidationError

  fake_runtime.lease_override = bad_record
  with pytest.raises(ValidationError):
    backend.start("17", IDENTITY, _desk())
  assert backend.store.get("17") is None
  assert not (backend.placement_root / "17" / "bundle.json").exists()
  assert len(aws_session.assume_calls) == 1
  assert fake_runtime.credential_updates == []
