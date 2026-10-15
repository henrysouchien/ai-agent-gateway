"""AgentCore microVM placements, with shared authority retained at the edge.

The blocking placement API owns one asyncio thread. HTTPX transports bridge to
that loop without buffering responses; reconnects preserve the AWS session id.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from datetime import datetime
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

from .runtime_placement import RuntimePlacement, RuntimePlacementAuthority, _storage_id
from .runtime_tunnel import Handler, TunnelClosed, TunnelEndpoint, TunnelResponse
from .user_runtime_credentials import (
  DeskReplica, USER_RUNTIME_CREDENTIALS_FILE_ENV,
  UserRuntimeIdentity, bundle_to_json, load_user_runtime_credentials,
)

log = logging.getLogger(__name__)
T = TypeVar("T")
EgressFactory = Callable[[UserRuntimeIdentity, str, Mapping[str, str]], Handler]


def _default_aws_session(region: str) -> Any:
  from boto3.session import Session

  return Session(region_name=region)


def _default_websocket_connect() -> Any:
  from websockets.asyncio.client import connect

  return connect


def desk_session_policy(bucket: str, storage_id: str) -> dict[str, Any]:
  """The desk STS session can address only this user's replica objects."""
  _storage_id(storage_id)
  prefix = f"users/{storage_id}"
  return {
    "Version": "2012-10-17",
    "Statement": [
      {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
       "Resource": f"arn:aws:s3:::{bucket}/{prefix}/*"},
      {"Effect": "Allow", "Action": "s3:ListBucket", "Resource": f"arn:aws:s3:::{bucket}",
       "Condition": {"StringLike": {"s3:prefix": [f"{prefix}/*", f"{prefix}/"]}}},
    ],
  }


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
  placement: RuntimePlacement
  bootstrap: dict[str, Any]
  handler: Handler | None
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
    # Projection, not scrubbing: new edge secrets cannot accidentally enter a VM.
    names = ("GATEWAY_SUBJECT_ASSERTION_ED25519_PUBLIC_KEYS",)
    environ = {name: self._extra_env[name] for name in names if name in self._extra_env}
    environ["CHAT_API_SECRET"] = secrets.token_urlsafe(32)
    for name, service in (
      ("ANTHROPIC_BASE_URL", "anthropic"), ("RISK_API_URL", "risk"), ("EDGAR_API_URL", "edgar"),
    ):
      environ[name] = f"http://127.0.0.1:8090/egress/{service}"
    environ["RISK_MODULE_RESOLVER_URL"] = "http://127.0.0.1:8090/egress/resolver/api/internal/resolve-credential"
    return environ

  def _desk_credentials(self, storage_id: str, generation: int) -> dict[str, str]:
    response = self._sts.assume_role(
      RoleArn=self._desk_role_arn, RoleSessionName=f"hank-desk-{storage_id}-g{generation}",
      DurationSeconds=3600, Policy=json.dumps(desk_session_policy(self._desk_bucket, storage_id)),
    )
    credentials = response["Credentials"]
    expiration = credentials["Expiration"]
    return {
      "AccessKeyId": credentials["AccessKeyId"], "SecretAccessKey": credentials["SecretAccessKey"],
      "SessionToken": credentials["SessionToken"],
      "Expiration": expiration.isoformat() if isinstance(expiration, datetime) else str(expiration),
    }

  async def _renew(self, state: _Session) -> None:
    while True:
      expiration = datetime.fromisoformat(state.bootstrap["desk"]["credentials"]["Expiration"].replace("Z", "+00:00"))
      await asyncio.sleep(max(1, expiration.timestamp() - time.time() - 300))
      try:
        credentials = await asyncio.to_thread(self._desk_credentials, state.placement.storage_id, state.placement.generation)
        state.bootstrap["desk"]["credentials"] = credentials
        if state.endpoint is not None and not state.endpoint.closed:
          await state.endpoint.control("credentials", {"desk": credentials})
      except Exception as exc:
        log.warning("Desk credential renewal storage_id=%s error=%s", state.placement.storage_id, type(exc).__name__)
        await asyncio.sleep(5)

  async def _maintain(self, state: _Session) -> None:
    renewal = asyncio.create_task(self._renew(state))
    try:
      while True:
        try:
          credentials = await asyncio.to_thread(self._aws.get_credentials)
          if credentials is None:
            raise RuntimeError("edge AWS credentials unavailable")
          url, headers = await asyncio.to_thread(
            generate_ws_connection, self._runtime_arn, str(state.placement.handle), self._region, credentials,
          )
          # websockets supplies Host from the same signed URL; a second Host
          # header makes the HTTP upgrade invalid.
          upgrade_headers = {key: value for key, value in headers.items() if key.lower() != "host"}
          async with self._connect(url, additional_headers=upgrade_headers, max_size=None, proxy=None) as socket:
            state.socket = socket

            async def control(kind: str, body: dict[str, Any]) -> None:
              if kind == "ready" and body.get("generation") == state.placement.generation:
                state.ready.set()
              elif kind == "failed":
                state.error = RuntimeError("user runtime failed to boot")
                state.ready.set()

            endpoint = TunnelEndpoint(send=socket.send, handler=state.handler, on_control=control, odd_ids=False)
            state.endpoint = endpoint
            await endpoint.control("bootstrap", state.bootstrap)
            async for frame in socket:
              await endpoint.feed(frame.decode() if isinstance(frame, bytes) else frame)
        except asyncio.CancelledError:
          raise
        except Exception as exc:
          # No exception text: upstream errors may contain signed URLs or secrets.
          log.warning("AgentCore tunnel reconnect storage_id=%s error=%s", state.placement.storage_id, type(exc).__name__)
        finally:
          if state.error is None:
            state.ready.clear()
          if state.endpoint is not None:
            state.endpoint.close()
          state.socket = None
        if state.error is not None:
          return
        await asyncio.sleep(1)
    finally:
      renewal.cancel()
      await asyncio.gather(renewal, return_exceptions=True)

  async def _ensure_session(self, placement: RuntimePlacement) -> _Session:
    if placement.handle is None:
      raise TunnelClosed("placement has no runtime session")
    async with self._session_locks.setdefault(placement.storage_id, asyncio.Lock()):
      state = self._sessions.get(placement.handle)
      if state is None:
        bundle = load_user_runtime_credentials({USER_RUNTIME_CREDENTIALS_FILE_ENV: placement.bundle_path})
        assert bundle is not None
        credentials = await asyncio.to_thread(self._desk_credentials, placement.storage_id, placement.generation)
        # STS is an await boundary: stop/replacement may have won while it ran.
        current = self.store.get(placement.storage_id)
        if current is None or current.state not in ("starting", "ready") or current.handle != placement.handle or current.generation != placement.generation:
          raise TunnelClosed("placement no longer owns this runtime session")
        handler = self._egress_factory(bundle.user, placement.storage_id, self._extra_env) if self._egress_factory else None
        state = _Session(placement, {
          "bundle": bundle_to_json(bundle),
          "desk": {"url": f"s3://{self._desk_bucket}", "credentials": credentials},
          "env": self._environment(), "generation": placement.generation,
        }, handler)
        self._sessions[placement.handle] = state
        state.task = asyncio.create_task(self._maintain(state))
    await asyncio.wait_for(state.ready.wait(), timeout=self._ready_timeout)
    if state.error is not None:
      raise state.error
    return state

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
        if self._bundle_serves(running, identity) and self.health(existing):
          return existing
      if existing:
        self.stop(existing)
      generation = existing.generation + 1 if existing else 1
      # Bundle carries identity/derived claim authority; rotating STS material
      # belongs to the credentials control channel, not a persisted provider file.
      scoped_desk = DeskReplica(f"s3://{self._desk_bucket}", None, None, None, None)
      bundle_path = self._mint_bundle(storage_id, identity, scoped_desk)
      session_id = f"hank-u{storage_id}-g{generation}-{uuid.uuid4().hex}"
      timestamp = time.time()
      placement = self.store.upsert(RuntimePlacement(
        storage_id, f"agentcore://{session_id}", "starting", "agentcore", session_id,
        str(bundle_path), generation, timestamp, timestamp,
      ))
      try:
        self._submit(self._ensure_session(placement)).result(timeout=self._ready_timeout + 30)
        return self.store.upsert(replace(placement, state="ready", updated_at=time.time()))
      except BaseException:
        self.stop(placement)
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
        from botocore.exceptions import ClientError

        try:
          self._agentcore.stop_runtime_session(
            agentRuntimeArn=self._runtime_arn, runtimeSessionId=placement.handle, qualifier="DEFAULT",
          )
        except ClientError as exc:
          # AWS reports an idle-terminated or never-created session as missing:
          # that is already the stopped state this operation owns.
          if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
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
