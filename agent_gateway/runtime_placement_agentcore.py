"""AgentCore microVM placements, with shared authority retained at the edge.

The blocking placement API owns one asyncio thread. HTTPX transports bridge to
that loop without buffering responses; reconnects preserve the AWS session id.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import logging
import os
import secrets
import threading
import time
from typing import Any, TypeVar
from urllib.parse import quote, urlencode
import uuid

import httpx

from pydantic import BaseModel, ConfigDict

from .runtime_placement import (
  DESK_LEASE_SECONDS, DeskLeaseRefused, RuntimePlacement, RuntimePlacementAuthority,
  WriterRecord, _storage_id,
)
from .runtime_tunnel import Handler, TunnelClosed, TunnelEndpoint, TunnelRequest, TunnelResponse
from .user_runtime_credentials import (
  DeskReplica, USER_RUNTIME_CREDENTIALS_FILE_ENV,
  UserRuntimeIdentity, bundle_to_json, load_user_runtime_credentials,
)

log = logging.getLogger(__name__)
T = TypeVar("T")
EgressFactory = Callable[[UserRuntimeIdentity, str, int, Mapping[str, str]], Handler]


def _default_aws_session(region: str) -> Any:
  from boto3.session import Session

  return Session(region_name=region)


def _default_websocket_connect() -> Any:
  from websockets.asyncio.client import connect

  return connect


def _lease_deadline(expires_at: float) -> str:
  return datetime.fromtimestamp(expires_at, timezone.utc).isoformat().replace("+00:00", "Z")


def desk_session_policy(bucket: str, storage_id: str, expires_at: float) -> dict[str, Any]:
  """Replica writes expire at the won writer lease, not the STS session expiry."""
  _storage_id(storage_id)
  prefix = f"users/{storage_id}"
  deadline = {"DateLessThan": {"aws:CurrentTime": _lease_deadline(expires_at)}}
  return {
    "Version": "2012-10-17",
    "Statement": [
      {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
       "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*", "Condition": deadline},
      {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}",
       "Condition": {"StringLike": {"s3:prefix": [f"{prefix}/*", f"{prefix}/"]}}},
    ],
  }


def desk_prelease_policy(bucket: str, storage_id: str, expires_at: float) -> dict[str, Any]:
  """Restore can read the desk and CAS writer.json, but cannot write its data."""
  _storage_id(storage_id)
  prefix = f"users/{storage_id}"
  deadline = {"DateLessThan": {"aws:CurrentTime": _lease_deadline(expires_at)}}
  return {
    "Version": "2012-10-17",
    "Statement": [
      {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject"],
       "Resource": f"arn:aws:s3:::{bucket}/{prefix}/writer.json", "Condition": deadline},
      {"Effect": "Allow", "Action": "s3:GetObject",
       "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*", "Condition": deadline},
      {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}",
       "Condition": {"StringLike": {"s3:prefix": [f"{prefix}/*", f"{prefix}/"]}}},
    ],
  }


class _RuntimeReady(BaseModel):
  model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
  generation: int


class _RuntimeFailure(BaseModel):
  model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
  error: str
  finding: str | None = None
  writer: WriterRecord | None = None


def generate_ws_connection(
  runtime_arn: str, session_id: str, region: str, credentials: Any,
) -> tuple[str, dict[str, str]]:
  """Mirror AgentCoreRuntimeClient.generate_ws_connection (SDK lines 153-305).

  The SDK signs the HTTPS upgrade URL, then appends the *unsigned* runtime
  session-id header. websockets owns Upgrade/Connection/Sec-WebSocket headers;
  adding the SDK's copies would produce duplicate handshake headers.
  """
  from botocore.auth import SigV4Auth
  from botocore.awsrequest import AWSRequest

  host = f"bedrock-agentcore.{region}.amazonaws.com"
  path = f"/runtimes/{quote(runtime_arn, safe='')}/ws"
  url = f"https://{host}{path}?{urlencode({'qualifier': 'DEFAULT'})}"
  frozen = credentials.get_frozen_credentials()
  request = AWSRequest(method="GET", url=url, headers={"host": host})
  SigV4Auth(frozen, "bedrock-agentcore", region).add_auth(request)
  headers = dict(request.headers.items())
  headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] = session_id
  return url.replace("https://", "wss://", 1), headers


@dataclass
class _Session:
  storage_id: str
  identity: UserRuntimeIdentity
  handle: str
  placement: RuntimePlacement | None = None
  bootstrap: dict[str, Any] = field(default_factory=dict)
  handler: Handler | None = None
  lease: WriterRecord | None = None
  ready: asyncio.Event = field(default_factory=asyncio.Event)
  endpoint: TunnelEndpoint | None = None
  socket: Any = None
  task: asyncio.Task[None] | None = None
  error: BaseException | None = None


class _ResponseStream(httpx.AsyncByteStream):
  def __init__(self, backend: AgentCoreRuntimeBackend, body: AsyncIterator[bytes]) -> None:
    self._backend = backend
    self._body = body

  async def _next(self) -> bytes:
    return await anext(self._body)

  async def __aiter__(self) -> AsyncIterator[bytes]:
    while True:
      try:
        yield await asyncio.wrap_future(self._backend._submit(self._next()))
      except StopAsyncIteration:
        return

  async def aclose(self) -> None:
    close = getattr(self._body, "aclose", None)
    if close is not None:
      await asyncio.wrap_future(self._backend._submit(close()))


class _TunnelTransport(httpx.AsyncBaseTransport):
  def __init__(self, backend: AgentCoreRuntimeBackend, placement: RuntimePlacement) -> None:
    self._backend = backend
    self._placement = placement

  async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
    body = await request.aread()
    try:
      response = await asyncio.wrap_future(self._backend._submit(self._backend._request(
        self._placement, request.method, request.url.raw_path.decode("ascii"),
        list(request.headers.multi_items()), body,
      )))
    except Exception as exc:
      raise httpx.ReadError(f"runtime tunnel request failed: {type(exc).__name__}", request=request) from None
    return httpx.Response(
      response.status, headers=response.headers,
      stream=_ResponseStream(self._backend, response.body),
    )


class AgentCoreRuntimeBackend(RuntimePlacementAuthority):
  """One full product app per AgentCore session; no provider keys in bootstrap."""

  def __init__(
    self, *, placement_root: str | os.PathLike[str], org_claim_key: str,
    runtime_arn: str, desk_bucket: str, desk_role_arn: str, region: str = "us-east-2",
    extra_env: Mapping[str, str] | None = None, egress_factory: EgressFactory | None = None,
    session: Any = None, connect: Any = None, ready_timeout: float = 300,
  ) -> None:
    super().__init__(placement_root=placement_root, org_claim_key=org_claim_key)
    self._runtime_arn = runtime_arn
    self._desk_bucket = desk_bucket
    self._desk_role_arn = desk_role_arn
    self._region = region
    self._extra_env = dict(extra_env or {})
    self._egress_factory = egress_factory
    self._aws = session if session is not None else _default_aws_session(region)
    self._sts = self._aws.client("sts", region_name=region)
    self._agentcore = self._aws.client("bedrock-agentcore", region_name=region)
    self._connect = connect if connect is not None else _default_websocket_connect()
    self._ready_timeout = ready_timeout
    self._sessions: dict[str, _Session] = {}
    self._session_locks: dict[str, asyncio.Lock] = {}
    self._loop = asyncio.new_event_loop()
    self._thread = threading.Thread(target=self._loop.run_forever, name="agentcore-tunnels", daemon=True)
    self._thread.start()

  def __repr__(self) -> str:
    return "AgentCoreRuntimeBackend(credentials=<redacted>)"

  def _submit(self, coroutine: Coroutine[Any, Any, T]) -> Future[T]:
    return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

  def _environment(self) -> dict[str, str]:
    from .approval_remote import APPROVALS_EGRESS_URL, APPROVALS_EGRESS_URL_ENV

    # Projection, not scrubbing: new edge secrets cannot accidentally enter a VM.
    names = ("GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS",)
    environ = {name: self._extra_env[name] for name in names if name in self._extra_env}
    environ["CHAT_API_SECRET"] = secrets.token_urlsafe(32)
    for name, service in (
      ("ANTHROPIC_BASE_URL", "anthropic"), ("RISK_API_URL", "risk"), ("EDGAR_API_URL", "edgar"),
    ):
      environ[name] = f"http://127.0.0.1:8090/egress/{service}"
    environ["RISK_MODULE_RESOLVER_URL"] = "http://127.0.0.1:8090/egress/resolver/api/internal/resolve-credential"
    environ[APPROVALS_EGRESS_URL_ENV] = APPROVALS_EGRESS_URL
    environ["HANK_APPROVAL_TRANSPORT"] = "agentcore"
    return environ

  def _desk_credentials(
    self, storage_id: str, expires_at: float, generation: int | None = None,
  ) -> dict[str, str]:
    policy = (
      desk_prelease_policy(self._desk_bucket, storage_id, expires_at) if generation is None
      else desk_session_policy(self._desk_bucket, storage_id, expires_at)
    )
    suffix = "prelease" if generation is None else f"g{generation}"
    response = self._sts.assume_role(
      RoleArn=self._desk_role_arn, RoleSessionName=f"hank-desk-{storage_id}-{suffix}",
      DurationSeconds=900, Policy=json.dumps(policy),
    )
    credentials = response["Credentials"]
    return {
      "AccessKeyId": credentials["AccessKeyId"], "SecretAccessKey": credentials["SecretAccessKey"],
      "SessionToken": credentials["SessionToken"],
      "Expiration": _lease_deadline(expires_at),
    }

  async def _accept_lease(self, state: _Session, record: WriterRecord) -> None:
    if record.state != "held":
      raise DeskLeaseRefused("runtime no longer holds its desk writer lease", record=record)
    if state.placement is not None and record.epoch != state.placement.generation:
      raise DeskLeaseRefused("runtime lease epoch no longer matches its placement", record=record)
    if state.lease is not None and record.holder != state.lease.holder:
      raise DeskLeaseRefused("runtime lease holder changed", record=record)
    credentials = await asyncio.to_thread(
      self._desk_credentials, state.storage_id, record.expires_at, record.epoch,
    )
    # Stop may win while STS is running; it must not resurrect a placement.
    if self._sessions.get(state.handle) is not state:
      raise TunnelClosed("runtime session stopped while minting desk credentials")
    if state.placement is not None:
      current = self.store.get(state.storage_id)
      if current is None or current.state not in ("starting", "ready") or current.handle != state.handle or current.generation != record.epoch:
        raise TunnelClosed("placement no longer owns this runtime session")
    update: dict[str, Any] = {"desk": credentials}
    if state.placement is None:
      scoped_desk = DeskReplica(f"s3://{self._desk_bucket}", None, None, None, None)
      bundle_path = self._mint_bundle(state.storage_id, state.identity, scoped_desk, record.epoch)
      bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: str(bundle_path)})
      assert bundle is not None
      state.handler = self._egress_factory(state.identity, state.storage_id, record.epoch, self._extra_env) if self._egress_factory else None
      timestamp = time.time()
      state.placement = self.store.upsert(RuntimePlacement(
        state.storage_id, f"agentcore://{state.handle}", "starting", "agentcore", state.handle,
        str(bundle_path), record.epoch, timestamp, timestamp,
      ))
      update["bundle"] = bundle_to_json(bundle)
    state.lease = record
    state.bootstrap["desk"]["credentials"] = credentials
    assert state.endpoint is not None
    await state.endpoint.control("credentials", update)

  async def _maintain(self, state: _Session) -> None:
    while True:
      try:
        credentials = await asyncio.to_thread(self._aws.get_credentials)
        if credentials is None:
          raise RuntimeError("edge AWS credentials unavailable")
        url, headers = await asyncio.to_thread(
          generate_ws_connection, self._runtime_arn, state.handle, self._region, credentials,
        )
        # websockets supplies Host from the same signed URL.
        upgrade_headers = {key: value for key, value in headers.items() if key.lower() != "host"}
        async with self._connect(url, additional_headers=upgrade_headers, max_size=None, proxy=None) as socket:
          state.socket = socket

          async def control(kind: str, body: dict[str, Any]) -> None:
            try:
              if kind == "lease":
                await self._accept_lease(state, WriterRecord.model_validate(body))
              elif kind == "ready":
                ready = _RuntimeReady.model_validate(body)
                if state.placement is None or ready.generation != state.placement.generation:
                  raise DeskLeaseRefused("runtime became ready without its placement lease epoch", record=state.lease)
                state.ready.set()
              elif kind == "failed":
                failure = _RuntimeFailure.model_validate(body)
                detail = failure.error if failure.finding is None else failure.finding
                if failure.error == "DeskLeaseRefused":
                  raise DeskLeaseRefused(f"user runtime failed to boot: {detail}", record=failure.writer)
                raise RuntimeError(f"user runtime failed to boot: {detail}")
            except Exception as exc:
              state.error = exc
              state.ready.set()
              raise

          async def egress(request: TunnelRequest) -> TunnelResponse:
            if state.handler is None:
              raise TunnelClosed("runtime has not acquired its desk writer lease")
            return await state.handler(request)

          endpoint = TunnelEndpoint(send=socket.send, handler=egress, on_control=control, odd_ids=False)
          state.endpoint = endpoint
          if state.lease is None:
            # Opening the websocket may cold-start the container. Its startup
            # time must not consume the prelease's bounded restore authority.
            credentials = await asyncio.to_thread(
              self._desk_credentials, state.storage_id, time.time() + DESK_LEASE_SECONDS,
            )
            if self._sessions.get(state.handle) is not state:
              raise TunnelClosed("runtime session stopped before bootstrap")
            state.bootstrap["desk"]["credentials"] = credentials
          await endpoint.control("bootstrap", state.bootstrap)
          async for frame in socket:
            await endpoint.feed(frame.decode() if isinstance(frame, bytes) else frame)
      except asyncio.CancelledError:
        raise
      except Exception as exc:
        # No exception text: upstream errors may contain signed URLs or secrets.
        log.warning("AgentCore tunnel reconnect storage_id=%s error=%s", state.storage_id, type(exc).__name__)
      finally:
        if state.error is None:
          state.ready.clear()
        if state.endpoint is not None:
          state.endpoint.close()
        state.socket = None
      if state.error is not None:
        return
      await asyncio.sleep(1)

  async def _launch_session(self, state: _Session) -> None:
    self._sessions[state.handle] = state
    state.bootstrap = {
      "storage_id": state.storage_id,
      "desk": {"url": f"s3://{self._desk_bucket}"},
      "env": self._environment(), "lease_seconds": DESK_LEASE_SECONDS,
    }
    state.task = asyncio.create_task(self._maintain(state))

  async def _wait_ready(self, state: _Session) -> _Session:
    await asyncio.wait_for(state.ready.wait(), timeout=self._ready_timeout)
    if state.error is not None:
      raise state.error
    return state

  async def _start_session(self, state: _Session) -> _Session:
    await self._launch_session(state)
    return await self._wait_ready(state)

  async def _ensure_session(self, placement: RuntimePlacement) -> _Session:
    if placement.handle is None:
      raise TunnelClosed("placement has no runtime session")
    async with self._session_locks.setdefault(placement.storage_id, asyncio.Lock()):
      state = self._sessions.get(placement.handle)
      if state is None:
        bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: placement.bundle_path})
        assert bundle is not None
        state = _Session(placement.storage_id, bundle.user, placement.handle, placement=placement)
        state.handler = self._egress_factory(bundle.user, placement.storage_id, placement.generation, self._extra_env) if self._egress_factory else None
        await self._launch_session(state)
    return await self._wait_ready(state)

  async def _request(
    self, placement: RuntimePlacement, method: str, path: str,
    headers: list[tuple[str, str]], body: bytes,
  ) -> TunnelResponse:
    stored = self.store.get(placement.storage_id)
    if stored is None or stored.state != "ready" or stored.handle != placement.handle or stored.generation != placement.generation:
      raise TunnelClosed("placement no longer owns this runtime session")
    state = await self._ensure_session(placement)
    if state.endpoint is None or state.endpoint.closed:
      raise TunnelClosed("runtime tunnel is disconnected")
    return await state.endpoint.request(method, path, headers, body)

  async def _health(self, placement: RuntimePlacement) -> bool:
    try:
      async with asyncio.timeout(5):
        response = await self._request(placement, "GET", "/api/health", [], b"")
        async for _ in response.body:
          pass
        return response.status == 200
    except (TimeoutError, ConnectionError, RuntimeError):
      return False

  def health(self, placement: RuntimePlacement) -> bool:
    return placement.state == "ready" and self._submit(self._health(placement)).result()

  def start(self, storage_id: str, identity: UserRuntimeIdentity, desk: DeskReplica) -> RuntimePlacement:
    with self._lock(storage_id):
      if type(identity.risk_user_id) is not int or storage_id != str(identity.risk_user_id):
        raise ValueError("storage_id must match identity.risk_user_id")
      existing = self.store.get(storage_id)
      if existing and existing.state == "ready":
        running = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: existing.bundle_path})
        if self._bundle_serves(running, identity, existing) and self.health(existing):
          return existing
      if existing:
        self.stop(existing)
      # The runtime's restore CAS chooses the epoch. Until its lease event there
      # is no placement row, bundle, handler, or locally allocated generation.
      state = _Session(storage_id, identity, f"hank-u{storage_id}-{uuid.uuid4().hex}")
      try:
        self._submit(self._start_session(state)).result(timeout=self._ready_timeout + 30)
        assert state.placement is not None
        ready = self.store.upsert(replace(state.placement, state="ready", updated_at=time.time()))
        state.placement = ready
        return ready
      except BaseException:
        if state.placement is not None:
          self.stop(state.placement)
        else:
          self._submit(self._disconnect(state.handle)).result()
          self._stop_runtime_session(state.handle)
        raise

  async def _disconnect(self, handle: str) -> None:
    state = self._sessions.pop(handle, None)
    if state is not None:
      state.error = TunnelClosed("runtime session stopped")
      if state.task is not None:
        state.task.cancel()
        await asyncio.gather(state.task, return_exceptions=True)
      if state.endpoint is not None:
        state.endpoint.close()
      state.ready.set()

  def _stop_runtime_session(self, handle: str) -> None:
    from botocore.exceptions import ClientError

    try:
      self._agentcore.stop_runtime_session(
        agentRuntimeArn=self._runtime_arn, runtimeSessionId=handle, qualifier="DEFAULT",
      )
    except ClientError as exc:
      # Idle-terminated or never-created sessions are already stopped.
      if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
        raise

  def stop(self, placement: RuntimePlacement) -> RuntimePlacement:
    with self._lock(placement.storage_id):
      stored = self.store.get(placement.storage_id)
      if stored and (stored.generation != placement.generation or stored.handle != placement.handle):
        return replace(placement, state="stopped", handle=None, updated_at=time.time())
      # Stop reconnection before StopRuntimeSession; otherwise a concurrent WS
      # retry can resurrect the old session while replacement compute starts.
      self.store.mark(placement.storage_id, "stopped")
      if placement.handle is not None:
        self._submit(self._disconnect(placement.handle)).result()
        self._stop_runtime_session(placement.handle)
      return self.store.upsert(replace(placement, state="stopped", handle=None, updated_at=time.time()))

  def replace(self, placement: RuntimePlacement) -> RuntimePlacement:
    with self._lock(placement.storage_id):
      stored = self.store.get(placement.storage_id)
      if stored is None:
        raise KeyError(placement.storage_id)
      bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: stored.bundle_path})
      assert bundle is not None
      self.stop(stored)
      self.store.mark(placement.storage_id, "replaced")
      return self.start(placement.storage_id, bundle.user, bundle.desk)

  def transport(self, placement: RuntimePlacement) -> httpx.AsyncBaseTransport:
    return _TunnelTransport(self, placement)

  def close(self) -> None:
    """Release edge sockets, not user compute (stop owns compute termination)."""
    async def disconnect_all() -> None:
      for handle in list(self._sessions):
        await self._disconnect(handle)
      await self._loop.shutdown_default_executor()
    self._submit(disconnect_all()).result()
    self._loop.call_soon_threadsafe(self._loop.stop)
    self._thread.join()
    self._loop.close()
