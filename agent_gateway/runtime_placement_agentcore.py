"""AgentCore microVM placements, with shared authority retained at the edge.

The blocking placement API owns one asyncio thread. HTTPX transports bridge to
that loop without buffering responses; reconnects preserve the AWS session id.
"""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import itertools
import json
import logging
import os
import signal
import secrets
import subprocess
import sys
import threading
import time
from typing import Any, TypeVar
from urllib.parse import quote, urlencode
import uuid

import httpx

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .runtime_placement import (
  DESK_LEASE_SECONDS, DeskLeaseRefused, RuntimePlacement, RuntimePlacementAuthority,
  WriterRecord, _storage_id,
)
from .runtime_tunnel import Handler, TunnelClosed, TunnelEndpoint, TunnelRequest, TunnelResponse
from hank_model_transport.user_runtime_credentials import (
  DeskReplica, USER_RUNTIME_CREDENTIALS_FILE_ENV,
  UserRuntimeIdentity, bundle_to_json, load_user_runtime_credentials,
)

log = logging.getLogger(__name__)
T = TypeVar("T")
EgressFactory = Callable[[UserRuntimeIdentity, str, int, Mapping[str, str]], Handler]
# Bound on one runtime's acknowledgement of an admission close, inside the edge
# readiness the deploy drain reads (its read timeout is 10 s).
SELF_ADMISSION_CONFIRM_SECONDS = 3.0


async def _sdk_call(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
  """Run blocking SDK work without a non-daemon executor atexit join."""
  future: Future[T] = Future()

  def call() -> None:
    if not future.set_running_or_notify_cancel():
      return
    try:
      result = function(*args, **kwargs)
    except BaseException as exc:
      future.set_exception(exc)
    else:
      future.set_result(result)

  threading.Thread(target=call, name="agentcore-sdk", daemon=True).start()
  return await asyncio.wrap_future(future)


def _default_aws_session(region: str, profile: str | None = None) -> Any:
  from boto3.session import Session

  return Session(profile_name=profile, region_name=region)


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


def desk_observer_policy(bucket: str, storage_id: str, expires_at: float) -> dict[str, Any]:
  """Read only the writer record; exact-prefix listing proves its absence."""
  _storage_id(storage_id)
  key = f"users/{storage_id}/writer.json"
  deadline = {"DateLessThan": {"aws:CurrentTime": _lease_deadline(expires_at)}}
  return {
    "Version": "2012-10-17",
    "Statement": [
      {"Effect": "Allow", "Action": "s3:GetObject",
       "Resource": f"arn:aws:s3:::{bucket}/{key}", "Condition": deadline},
      {"Effect": "Allow", "Action": "s3:ListBucket",
       "Resource": f"arn:aws:s3:::{bucket}",
       "Condition": {"StringEquals": {"s3:prefix": key}, **deadline}},
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


class _RuntimeAdmission(BaseModel):
  model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
  closed: bool
  seq: int = Field(ge=0)


def generate_ws_connection(
  runtime_arn: str, session_id: str, region: str, credentials: Any, *, runtime_qualifier: str,
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
  url = f"https://{host}{path}?{urlencode({'qualifier': runtime_qualifier})}"
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
  # Admission-frame seq -> whether the runtime acknowledged it closed.
  admission_acks: dict[int, asyncio.Future[bool]] = field(default_factory=dict)


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
    runtime_arn: str, runtime_qualifier: str, desk_bucket: str, desk_role_arn: str,
    region: str = "us-east-2",
    extra_env: Mapping[str, str] | None = None, egress_factory: EgressFactory | None = None,
    session: Any = None, connect: Any = None, ready_timeout: float = 300,
    shutdown_timeout: float = 180, aws_profile: str | None = None, tunnels: bool = True,
  ) -> None:
    super().__init__(placement_root=placement_root, org_claim_key=org_claim_key)
    self._runtime_arn = runtime_arn
    self._runtime_qualifier = runtime_qualifier
    self._desk_bucket = desk_bucket
    self._desk_role_arn = desk_role_arn
    self._region = region
    self._aws_profile = aws_profile
    self._extra_env = dict(extra_env or {})
    self._egress_factory = egress_factory
    self._aws = session if session is not None else _default_aws_session(region, aws_profile)
    self._sts = self._aws.client("sts", region_name=region)
    self._agentcore = self._aws.client("bedrock-agentcore", region_name=region)
    self._connect = connect if connect is not None else _default_websocket_connect()
    self._ready_timeout = ready_timeout
    self._shutdown_timeout = shutdown_timeout
    self._sessions: dict[str, _Session] = {}
    self._sessions_guard = threading.Lock()
    self._closing = threading.Event()
    self._session_locks: dict[str, asyncio.Lock] = {}
    # Runtime self-admission (schedule ticks, approval resumes), backend-wide:
    # every connect delivers it after bootstrap. Read and written on the tunnel loop.
    self._admission_closed = False
    self._admission_seq = itertools.count()
    self._loop: asyncio.AbstractEventLoop | None = None
    self._thread: threading.Thread | None = None
    if tunnels:
      self._loop = asyncio.new_event_loop()
      self._thread = threading.Thread(target=self._loop.run_forever, name="agentcore-tunnels", daemon=True)
      self._thread.start()

  def __repr__(self) -> str:
    return "AgentCoreRuntimeBackend(credentials=<redacted>)"

  def _submit(self, coroutine: Coroutine[Any, Any, T]) -> Future[T]:
    loop = self._loop
    if loop is None or loop.is_closed():
      raise TunnelClosed("runtime tunnels are closed")
    return asyncio.run_coroutine_threadsafe(coroutine, loop)

  def _owns_session(self, state: _Session) -> bool:
    with self._sessions_guard:
      return not self._closing.is_set() and self._sessions.get(state.handle) is state

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

  def _scoped_desk_credentials(
    self, storage_id: str, expires_at: float, suffix: str, policy: Mapping[str, Any],
  ) -> dict[str, str]:
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

  def _desk_credentials(
    self, storage_id: str, expires_at: float, generation: int | None = None,
  ) -> dict[str, str]:
    policy = (
      desk_prelease_policy(self._desk_bucket, storage_id, expires_at) if generation is None
      else desk_session_policy(self._desk_bucket, storage_id, expires_at)
    )
    suffix = "prelease" if generation is None else f"g{generation}"
    return self._scoped_desk_credentials(storage_id, expires_at, suffix, policy)

  def _observer_credentials(self, storage_id: str, expires_at: float) -> dict[str, str]:
    return self._scoped_desk_credentials(
      storage_id, expires_at, "observer", desk_observer_policy(self._desk_bucket, storage_id, expires_at),
    )

  async def _accept_lease(self, state: _Session, record: WriterRecord) -> None:
    if record.state != "held":
      raise DeskLeaseRefused("runtime no longer holds its desk writer lease", record=record)
    if state.placement is not None and record.epoch != state.placement.generation:
      raise DeskLeaseRefused("runtime lease epoch no longer matches its placement", record=record)
    if state.lease is not None and record.holder != state.lease.holder:
      raise DeskLeaseRefused("runtime lease holder changed", record=record)
    self.store.retain_shutdown_session(state.storage_id, state.handle, record.epoch)
    credentials = await _sdk_call(
      self._desk_credentials, state.storage_id, record.expires_at, record.epoch,
    )
    # Stop may win while STS is running; it must not resurrect a placement.
    if not self._owns_session(state):
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
    while self._owns_session(state):
      try:
        credentials = await _sdk_call(self._aws.get_credentials)
        if credentials is None:
          raise RuntimeError("edge AWS credentials unavailable")
        url, headers = await _sdk_call(
          generate_ws_connection, self._runtime_arn, state.handle, self._region, credentials,
          runtime_qualifier=self._runtime_qualifier,
        )
        # websockets supplies Host from the same signed URL.
        upgrade_headers = {key: value for key, value in headers.items() if key.lower() != "host"}
        async with self._connect(url, additional_headers=upgrade_headers, max_size=None, proxy=None) as socket:
          if not self._owns_session(state):
            raise TunnelClosed("runtime session stopped before reconnect")
          state.socket = socket

          async def control(kind: str, body: dict[str, Any]) -> None:
            if kind == "admission":
              # An acknowledgement never fails the session: an unmatched or
              # malformed one leaves the close unconfirmed, so drains stay busy.
              try:
                ack = _RuntimeAdmission.model_validate(body)
              except ValidationError:
                log.warning("AgentCore runtime sent a malformed admission acknowledgement storage_id=%s", state.storage_id)
                return
              acknowledged = state.admission_acks.get(ack.seq)
              if acknowledged is not None and not acknowledged.done():
                acknowledged.set_result(ack.closed)
              return
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
            credentials = await _sdk_call(
              self._desk_credentials, state.storage_id, time.time() + DESK_LEASE_SECONDS,
            )
            if not self._owns_session(state):
              raise TunnelClosed("runtime session stopped before bootstrap")
            state.bootstrap["desk"]["credentials"] = credentials
          await endpoint.control("bootstrap", state.bootstrap)
          # A reconnect or new session receives the edge's current admission
          # state; a fresh edge's open state reopens one a prior edge closed.
          await endpoint.control("admission", {"closed": self._admission_closed, "seq": next(self._admission_seq)})
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
    with self._sessions_guard:
      if self._closing.is_set():
        raise TunnelClosed("runtime backend is shutting down")
      self._sessions[state.handle] = state
    generation = state.placement.generation if state.placement is not None else None
    self.store.retain_shutdown_session(state.storage_id, state.handle, generation)
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
      with self._sessions_guard:
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

  async def _confirm_admission_closed(self, placement: RuntimePlacement) -> None:
    async with asyncio.timeout(SELF_ADMISSION_CONFIRM_SECONDS):
      state = await self._ensure_session(placement)
      endpoint = state.endpoint
      if endpoint is None or endpoint.closed:
        raise TunnelClosed("runtime tunnel is disconnected")
      seq = next(self._admission_seq)
      acknowledged: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
      state.admission_acks[seq] = acknowledged
      try:
        await endpoint.control("admission", {"closed": True, "seq": seq})
        if not await acknowledged:
          raise RuntimeError("runtime could not close its admission")
      finally:
        state.admission_acks.pop(seq, None)

  async def _close_self_admission(self, placements: Sequence[RuntimePlacement]) -> frozenset[str]:
    self._admission_closed = True
    results = await asyncio.gather(
      *(self._confirm_admission_closed(placement) for placement in placements), return_exceptions=True,
    )
    confirmed: set[str] = set()
    for placement, result in zip(placements, results):
      if result is None:
        confirmed.add(placement.storage_id)
      elif isinstance(result, Exception):
        # No exception text: upstream errors may contain signed URLs or secrets.
        log.warning(
          "AgentCore runtime did not confirm its admission close storage_id=%s error=%s",
          placement.storage_id, type(result).__name__,
        )
      else:
        raise result
    return frozenset(confirmed)

  def close_self_admission(self, placements: Sequence[RuntimePlacement]) -> frozenset[str]:
    """Close the runtimes' own admission; return the storage ids whose runtime acknowledged.

    The adapter holds the product's restart lock exclusively once in-progress
    admissions end, so a confirmed runtime's counts already hold every run it
    admitted. The closed state stays backend-wide until `open_self_admission`.
    """
    return self._submit(self._close_self_admission(placements)).result()

  async def _open_self_admission(self) -> None:
    self._admission_closed = False
    with self._sessions_guard:
      sessions = list(self._sessions.values())
    for state in sessions:
      endpoint = state.endpoint
      if endpoint is None or endpoint.closed:
        continue
      try:
        await endpoint.control("admission", {"closed": False, "seq": next(self._admission_seq)})
      except Exception as exc:
        # Its reconnect delivers the open state after bootstrap.
        log.warning(
          "AgentCore runtime admission reopen not delivered storage_id=%s error=%s",
          state.storage_id, type(exc).__name__,
        )

  def open_self_admission(self) -> None:
    """Reopen every session's own admission; reconnects deliver it to the rest."""
    self._submit(self._open_self_admission()).result()

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
    with self._sessions_guard:
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
        agentRuntimeArn=self._runtime_arn, runtimeSessionId=handle, qualifier=self._runtime_qualifier,
      )
    except ClientError as exc:
      # Idle-terminated or never-created sessions are already stopped.
      if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
        raise

  def _stop_for_shutdown(self, storage_id: str, handle: str) -> None:
    """Request compute termination without discarding the unproven handle."""
    with self._lock(storage_id):
      stored = self.store.get(storage_id)
      if stored is not None and stored.handle == handle:
        self.store.mark(storage_id, "stopped")
      self._stop_runtime_session(handle)

  def _complete_shutdown(self, storage_id: str, handle: str) -> None:
    with self._lock(storage_id):
      stored = self.store.get(storage_id)
      if stored is not None and stored.handle == handle:
        self.store.upsert(replace(stored, state="stopped", handle=None, updated_at=time.time()))
      self.store.complete_shutdown_session(handle)

  def stop(self, placement: RuntimePlacement) -> RuntimePlacement:
    with self._lock(placement.storage_id):
      stored = self.store.get(placement.storage_id)
      if stored and (stored.generation != placement.generation or stored.handle != placement.handle):
        return replace(placement, state="stopped", handle=None, updated_at=time.time())
      # The routing row may stop naming this compute, but teardown retains its
      # handle and epoch until both stop and the old writer proof complete.
      if placement.handle is not None:
        self.store.retain_shutdown_session(placement.storage_id, placement.handle, placement.generation)
      self.store.mark(placement.storage_id, "stopped")
      if placement.handle is not None:
        self._submit(self._disconnect(placement.handle)).result()
        self._stop_runtime_session(placement.handle)
      return self.store.upsert(replace(placement, state="stopped", handle=None, updated_at=time.time()))

  def _writer_client(self, storage_id: str, expires_at: float) -> Any:
    credentials = self._observer_credentials(storage_id, expires_at)
    return self._aws.client(
      "s3", region_name=self._region,
      aws_access_key_id=credentials["AccessKeyId"],
      aws_secret_access_key=credentials["SecretAccessKey"],
      aws_session_token=credentials["SessionToken"],
    )

  def _read_writer(self, client: Any, storage_id: str) -> tuple[WriterRecord | None, float]:
    """Read G6's record and its own clock; never allocate or mutate the lease."""
    from botocore.exceptions import ClientError

    key = f"users/{storage_id}/writer.json"
    listing = client.list_objects_v2(Bucket=self._desk_bucket, Prefix=key)
    listed_now = parsedate_to_datetime(listing["ResponseMetadata"]["HTTPHeaders"]["date"]).timestamp()
    if "Contents" not in listing:
      return None, listed_now
    try:
      response = client.get_object(Bucket=self._desk_bucket, Key=key)
    except ClientError as exc:
      if exc.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404"):
        raise
      now = parsedate_to_datetime(
        exc.response["ResponseMetadata"]["HTTPHeaders"]["date"],
      ).timestamp()
      return None, now
    now = parsedate_to_datetime(response["ResponseMetadata"]["HTTPHeaders"]["date"]).timestamp()
    body = response["Body"]
    try:
      record = WriterRecord.model_validate_json(body.read())
    finally:
      body.close()
    return record, now

  def _await_writer_safe(
    self, storage_id: str, expected_generation: int | None, shutdown_deadline: float,
  ) -> None:
    # Read authority covers the whole bounded observation, plus clock skew for
    # the final S3 Date proof; it does not alter the writer's own deadline.
    observer_expiry = time.time() + max(0, shutdown_deadline - time.monotonic()) + 60
    client = self._writer_client(storage_id, observer_expiry)
    while True:
      record, server_now = self._read_writer(client, storage_id)
      if record is None:
        log.info("AgentCore shutdown writer absent storage_id=%s", storage_id)
        return
      if expected_generation is not None and record.epoch > expected_generation:
        # G6 could not have admitted the newer epoch until our old credential
        # deadline passed. Its live renewals belong to the new owner, not us.
        log.info(
          "AgentCore shutdown old writer superseded storage_id=%s generation=%s current_generation=%s",
          storage_id, expected_generation, record.epoch,
        )
        return
      if expected_generation is not None and record.epoch < expected_generation:
        # A regressed record cannot prove the newer owned epoch's deadline.
        # Keep the handle pending until the shared budget fails this proof.
        remaining = shutdown_deadline - time.monotonic()
      elif server_now >= record.expires_at:
        log.info(
          "AgentCore shutdown writer safe storage_id=%s generation=%s deadline=%s",
          storage_id, record.epoch, _lease_deadline(record.expires_at),
        )
        return
      else:
        remaining = min(record.expires_at - server_now, shutdown_deadline - time.monotonic())
      if remaining <= 0:
        raise TimeoutError(
          f"writer deadline {_lease_deadline(record.expires_at)} not safely expired for storage {storage_id}"
        )
      log.info(
        "AgentCore shutdown awaiting writer storage_id=%s generation=%s deadline=%s",
        storage_id, record.epoch, _lease_deadline(record.expires_at),
      )
      time.sleep(min(1.0, remaining + 0.05))

  def _capture_shutdown_targets(self) -> None:
    """Fence reconnects and durably capture live and pre-placement handles."""
    with self._sessions_guard:
      self._closing.set()
      live = list(self._sessions.values())
    targets: dict[str, tuple[str, int | None]] = {
      handle: (storage_id, generation)
      for handle, storage_id, generation in self.store.list_shutdown_sessions()
    }
    for placement in self.store.list():
      if placement.handle is not None:
        targets[placement.handle] = (placement.storage_id, placement.generation)
    for state in live:
      previous = targets.get(state.handle)
      generation = (
        state.lease.epoch if state.lease is not None
        else state.placement.generation if state.placement is not None
        else previous[1] if previous is not None else None
      )
      targets[state.handle] = (state.storage_id, generation)
    self.store.retain_shutdown_sessions([
      (storage_id, handle, generation) for handle, (storage_id, generation) in targets.items()
    ])

  def _shutdown_owned(self) -> None:
    """The isolated worker's stop/proof transaction; never wait here at exit."""
    shutdown_deadline = time.monotonic() + self._shutdown_timeout
    targets: dict[str, tuple[str, int | None]] = {
      handle: (storage_id, generation)
      for handle, storage_id, generation in self.store.list_shutdown_sessions()
    }
    for placement in self.store.list():
      if placement.handle is not None:
        targets[placement.handle] = (placement.storage_id, placement.generation)
    self.store.retain_shutdown_sessions([
      (storage_id, handle, generation) for handle, (storage_id, generation) in targets.items()
    ])

    executor = ThreadPoolExecutor(max_workers=min(32, max(1, len(targets))))
    errors: list[Exception] = []
    try:
      stops = {
        handle: executor.submit(self._stop_for_shutdown, storage_id, handle)
        for handle, (storage_id, _generation) in targets.items()
      }
      failed_stops: set[str] = set()
      for handle, future in stops.items():
        try:
          future.result(timeout=max(0, shutdown_deadline - time.monotonic()))
        except Exception as exc:
          failed_stops.add(handle)
          errors.append(exc)
          log.error(
            "AgentCore shutdown stop failed storage_id=%s handle=%s error=%s",
            targets[handle][0], handle, type(exc).__name__,
          )

      observations = {
        handle: executor.submit(self._await_writer_safe, storage_id, generation, shutdown_deadline)
        for handle, (storage_id, generation) in targets.items()
      }
      for handle, future in observations.items():
        try:
          future.result(timeout=max(0, shutdown_deadline - time.monotonic()))
          if handle not in failed_stops:
            self._complete_shutdown(targets[handle][0], handle)
        except Exception as exc:
          errors.append(exc)
          log.error(
            "AgentCore shutdown writer proof failed storage_id=%s handle=%s error=%s",
            targets[handle][0], handle, type(exc).__name__,
          )
    finally:
      executor.shutdown(wait=False, cancel_futures=True)
    if errors:
      raise ExceptionGroup("AgentCore runtime shutdown did not complete", errors)

  def shutdown(self) -> None:
    """Capture once, then bound remote teardown and actual worker exit."""
    shutdown_deadline = time.monotonic() + self._shutdown_timeout
    self._capture_shutdown_targets()
    close_error: BaseException | None = None
    try:
      self.close(timeout=min(5.0, max(0, shutdown_deadline - time.monotonic()) / 10))
    except BaseException as exc:
      close_error = exc
    try:
      _supervise_agentcore_shutdown(
        shutdown_deadline=shutdown_deadline, placement_root=self.placement_root,
        org_claim_key=self._org_claim_key, runtime_arn=self._runtime_arn,
        runtime_qualifier=self._runtime_qualifier, desk_bucket=self._desk_bucket,
        desk_role_arn=self._desk_role_arn, region=self._region, aws_profile=self._aws_profile,
      )
    except BaseException as exc:
      if close_error is not None:
        raise BaseExceptionGroup("AgentCore runtime shutdown did not complete", [close_error, exc])
      raise
    if close_error is not None:
      raise close_error

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

  def close(self, *, timeout: float | None = None) -> None:
    """Release edge sockets without ever joining SDK executor threads."""
    self._closing.set()
    loop = self._loop
    thread = self._thread
    if loop is None or loop.is_closed():
      return
    deadline = time.monotonic() + (self._shutdown_timeout if timeout is None else timeout)

    async def disconnect_all() -> None:
      with self._sessions_guard:
        handles = list(self._sessions)
      for handle in handles:
        await self._disconnect(handle)

    future = self._submit(disconnect_all())
    error: BaseException | None = None
    remaining = max(0, deadline - time.monotonic())
    disconnect_deadline = deadline - remaining / 2
    try:
      future.result(timeout=max(0, disconnect_deadline - time.monotonic()))
    except TimeoutError:
      error = TimeoutError("AgentCore tunnel cleanup exceeded its time budget")
      future.cancel()
    except BaseException as exc:
      error = exc
      future.cancel()
    loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
      thread.join(timeout=max(0, deadline - time.monotonic()))
      if thread.is_alive():
        raise TimeoutError("AgentCore tunnel cleanup exceeded its time budget")
    loop.close()
    if error is not None:
      raise error


def _supervise_agentcore_shutdown(
  *, shutdown_deadline: float, placement_root: str | os.PathLike[str], org_claim_key: str,
  runtime_arn: str, runtime_qualifier: str, desk_bucket: str, desk_role_arn: str,
  region: str, aws_profile: str | None,
) -> None:
  """Kill and reap the whole SDK/cleanup lifecycle at the shared hard wall."""
  command = [
    sys.executable, "-B", "-m", "agent_gateway.runtime_placement_agentcore",
    "--placement-root", str(placement_root),
    "--agentcore-runtime-arn", runtime_arn,
    "--agentcore-qualifier", runtime_qualifier,
    "--desk-bucket", desk_bucket, "--desk-role-arn", desk_role_arn,
    "--region", region, "--shutdown-timeout", str(max(0, shutdown_deadline - time.monotonic())),
  ]
  if aws_profile is not None:
    command.extend(("--aws-profile", aws_profile))
  environ = dict(os.environ)
  environ["AGENT_API_USER_CLAIM_HMAC_KEY"] = org_claim_key
  process = subprocess.Popen(command, env=environ, start_new_session=True)
  remaining = max(0, shutdown_deadline - time.monotonic())
  # Reserve part of the same budget for SIGKILL and actual wait/reaping.
  wait_deadline = shutdown_deadline - min(1.0, remaining / 10)
  try:
    returncode = process.wait(timeout=max(0, wait_deadline - time.monotonic()))
  except subprocess.TimeoutExpired:
    try:
      os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
      pass
    try:
      process.wait(timeout=max(0, shutdown_deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
      raise TimeoutError("AgentCore shutdown worker did not exit after SIGKILL") from exc
    raise TimeoutError("AgentCore shutdown exceeded its hard time budget") from None
  if returncode != 0:
    raise RuntimeError(f"AgentCore shutdown worker failed with exit status {returncode}")


def shutdown_agentcore_runtime(
  *, placement_root: str | os.PathLike[str], org_claim_key: str,
  runtime_arn: str, runtime_qualifier: str, desk_bucket: str, desk_role_arn: str,
  region: str = "us-east-2", aws_profile: str | None = None, shutdown_timeout: float = 180,
) -> None:
  """Supervise cleanup before this process constructs any AWS SDK resource."""
  _supervise_agentcore_shutdown(
    shutdown_deadline=time.monotonic() + shutdown_timeout,
    placement_root=placement_root, org_claim_key=org_claim_key,
    runtime_arn=runtime_arn, runtime_qualifier=runtime_qualifier,
    desk_bucket=desk_bucket, desk_role_arn=desk_role_arn,
    region=region, aws_profile=aws_profile,
  )


def _shutdown_worker_main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Isolated AgentCore shutdown worker")
  parser.add_argument("--placement-root", type=str, required=True)
  parser.add_argument("--agentcore-runtime-arn", required=True)
  parser.add_argument("--agentcore-qualifier", required=True)
  parser.add_argument("--desk-bucket", required=True)
  parser.add_argument("--desk-role-arn", required=True)
  parser.add_argument("--region", default="us-east-2")
  parser.add_argument("--shutdown-timeout", type=float, default=180)
  parser.add_argument("--aws-profile")
  args = parser.parse_args(argv)
  try:
    backend = AgentCoreRuntimeBackend(
      placement_root=args.placement_root, org_claim_key=os.environ["AGENT_API_USER_CLAIM_HMAC_KEY"],
      runtime_arn=args.agentcore_runtime_arn, runtime_qualifier=args.agentcore_qualifier,
      desk_bucket=args.desk_bucket, desk_role_arn=args.desk_role_arn, region=args.region,
      shutdown_timeout=args.shutdown_timeout, aws_profile=args.aws_profile, tunnels=False,
    )
    backend._shutdown_owned()
  except BaseException as exc:
    log.error("AgentCore isolated shutdown failed error=%s", type(exc).__name__)
    os._exit(1)
  # A failed SDK future can leave a non-daemon executor thread. The supervisor
  # owns the wall clock; worker exit must never join those threads in atexit.
  os._exit(0)


if __name__ == "__main__":
  _shutdown_worker_main()
