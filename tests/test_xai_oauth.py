from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import multiprocessing
import os
from pathlib import Path
import stat
import threading
import time
from urllib.parse import parse_qs

import httpx
import pytest

from agent_gateway.providers.xai import XAIProvider
from agent_gateway.providers import xai_oauth
from agent_gateway.providers.xai_oauth import (
  DEFAULT_XAI_OAUTH_CLIENT_ID,
  DEFAULT_XAI_OAUTH_SCOPE,
  _record_is_valid,
  _store_lock,
  load_xai_token_record,
  login_xai_device_code,
  oauth_record_from_config,
  refresh_xai_oauth_token,
  resolve_xai_auth_mode,
  resolve_xai_oauth_settings,
  save_xai_token_record,
)


@pytest.fixture(autouse=True)
def _reset_store_refresh_consumed() -> None:
  xai_oauth._STORE_REFRESH_CONSUMED.clear()
  yield
  xai_oauth._STORE_REFRESH_CONSUMED.clear()


def _run(coro):
  return asyncio.run(coro)


def _record(**overrides):
  return {
    "access_token": "access-1",
    "refresh_token": "refresh-1",
    "expires_at": 4_000_000_000,
    "scope": DEFAULT_XAI_OAUTH_SCOPE,
    "issuer": "https://auth.x.ai",
    "client_id": DEFAULT_XAI_OAUTH_CLIENT_ID,
    "token_endpoint": "https://auth.x.ai/oauth2/token",
    **overrides,
  }


def _token_form(request: httpx.Request) -> dict[str, list[str]]:
  return parse_qs(request.content.decode())


def _completed_response(text: str = "summary") -> httpx.Response:
  events = [
    {"type": "response.output_text.delta", "delta": text},
    {
      "type": "response.completed",
      "response": {
        "status": "completed",
        "usage": {"input_tokens": 1, "output_tokens": 1},
      },
    },
  ]
  return httpx.Response(200, text="".join(f"data: {json.dumps(event)}\n\n" for event in events))


def _install_retry_transport(
  monkeypatch: pytest.MonkeyPatch,
  transport: httpx.AsyncBaseTransport,
  *,
  created: list[tuple[httpx.AsyncClient, dict]] | None = None,
) -> None:
  real_httpx = xai_oauth.httpx

  class RetryHTTPX:
    def __getattr__(self, name: str):
      return getattr(real_httpx, name)

    def AsyncClient(self, *args, **kwargs) -> httpx.AsyncClient:
      assert "transport" not in kwargs
      client_kwargs = {**kwargs, "transport": transport}
      client = real_httpx.AsyncClient(*args, **client_kwargs)
      if created is not None:
        created.append((client, dict(kwargs)))
      return client

  monkeypatch.setattr(xai_oauth, "httpx", RetryHTTPX())


class _GatedRefreshTransport(httpx.AsyncBaseTransport):
  def __init__(
    self,
    *,
    expires_in: int = 3600,
    same_access: bool = False,
    response_status: int = 200,
  ) -> None:
    self.expires_in = expires_in
    self.same_access = same_access
    self.response_status = response_status
    self.posts: list[str] = []
    self.first_posted = asyncio.Event()
    self.release_first = asyncio.Event()

  async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
    if not request.url.path.endswith("/oauth2/token"):
      return _completed_response()
    refresh_token = _token_form(request)["refresh_token"][0]
    self.posts.append(refresh_token)
    ordinal = len(self.posts)
    if ordinal == 1:
      self.first_posted.set()
      await self.release_first.wait()
    if self.response_status != 200:
      return httpx.Response(self.response_status, json={"error": "server_error"})
    access_token = "access-1" if self.same_access and ordinal == 1 else f"access-{ordinal + 1}"
    return httpx.Response(
      200,
      json={
        "access_token": access_token,
        "refresh_token": f"refresh-{ordinal + 1}",
        "expires_in": self.expires_in,
      },
    )


def _cross_process_refresh_worker(
  store_path: str,
  server_url: str,
  barrier: multiprocessing.synchronize.Barrier,
  captured: multiprocessing.queues.Queue,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": store_path})
  caller_record = load_xai_token_record(Path(store_path))
  captured.put(caller_record["refresh_token"])
  barrier.wait()

  class ForwardTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
      async with httpx.AsyncClient() as forwarding_client:
        response = await forwarding_client.post(
          server_url,
          content=request.content,
          headers={"Content-Type": request.headers["content-type"]},
        )
      return httpx.Response(
        response.status_code,
        content=response.content,
        headers=response.headers,
        request=request,
      )

  async def run() -> None:
    async with httpx.AsyncClient(transport=ForwardTransport()) as client:
      await refresh_xai_oauth_token(
        caller_record,
        settings=settings,
        client=client,
        force=True,
      )

  asyncio.run(run())


def test_store_defaults_under_user_data_dir_and_is_mode_0600(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings(environ={"USER_DATA_DIR": str(tmp_path)})
  assert settings.store_path == tmp_path / "xai" / "oauth.json"
  save_xai_token_record(settings.store_path, _record())
  assert stat.S_IMODE(settings.store_path.stat().st_mode) == 0o600
  assert load_xai_token_record(settings.store_path) == _record()


def test_explicit_auth_mode_wins_and_auto_detects_refreshable_store(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record())
  env = {"XAI_AUTH_STORE_PATH": str(store), "XAI_API_KEY": "api-key"}
  assert resolve_xai_auth_mode(environ=env) == "oauth"
  assert resolve_xai_auth_mode(environ={**env, "XAI_AUTH_MODE": "api"}) == "api"
  assert resolve_xai_auth_mode(environ={**env, "XAI_AUTH_MODE": "oauth"}) == "oauth"


def test_device_code_login_discovers_polls_and_persists(tmp_path: Path) -> None:
  requests: list[httpx.Request] = []

  def handler(request: httpx.Request) -> httpx.Response:
    requests.append(request)
    if request.url.path.endswith("openid-configuration"):
      return httpx.Response(
        200,
        json={
          "device_authorization_endpoint": "https://auth.x.ai/oauth2/device/code",
          "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
      )
    if request.url.path.endswith("/device/code"):
      return httpx.Response(
        200,
        json={
          "device_code": "device-1",
          "user_code": "ABCD-1234",
          "verification_uri": "https://accounts.x.ai/oauth2/device",
          "verification_uri_complete": "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234",
          "expires_in": 900,
          "interval": 1,
        },
      )
    return httpx.Response(
      200,
      json={
        "access_token": "access-1",
        "refresh_token": "refresh-1",
        "expires_in": 3600,
      },
    )

  seen_codes = []
  client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
  try:
    record, path = _run(
      login_xai_device_code(
        config={"auth_store_path": str(tmp_path / "oauth.json")},
        on_verification=lambda code: seen_codes.append(code),
        client=client,
      )
    )
  finally:
    _run(client.aclose())

  assert path == tmp_path / "oauth.json"
  assert record["refresh_token"] == "refresh-1"
  assert seen_codes[0].user_code == "ABCD-1234"
  device_form = parse_qs(requests[1].content.decode())
  assert device_form["client_id"] == [DEFAULT_XAI_OAUTH_CLIENT_ID]
  assert device_form["scope"] == [DEFAULT_XAI_OAUTH_SCOPE]
  token_form = parse_qs(requests[2].content.decode())
  assert token_form["device_code"] == ["device-1"]
  assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_refresh_rotates_tokens_and_updates_store(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  old = _record()
  save_xai_token_record(settings.store_path, old)

  def handler(request: httpx.Request) -> httpx.Response:
    pending = load_xai_token_record(settings.store_path)
    assert pending["refresh_pending"] is True
    assert pending["refresh_token"] == "refresh-1"
    form = parse_qs(request.content.decode())
    assert form["grant_type"] == ["refresh_token"]
    assert form["refresh_token"] == ["refresh-1"]
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
  try:
    refreshed = _run(
      refresh_xai_oauth_token(old, settings=settings, client=client, force=True)
    )
  finally:
    _run(client.aclose())
  assert refreshed["access_token"] == "access-2"
  assert refreshed["refresh_token"] == "refresh-2"
  stored = load_xai_token_record(settings.store_path)
  assert stored["refresh_token"] == "refresh-2"
  assert "refresh_pending" not in stored


def test_lost_response_quarantines_and_second_refresh_never_posts(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  posts: list[str] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    raise httpx.ReadTimeout("response lost", request=request)

  transport = httpx.MockTransport(handler)
  _install_retry_transport(monkeypatch, transport)

  async def refresh() -> None:
    async with httpx.AsyncClient(transport=transport) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(httpx.ReadTimeout):
    _run(refresh())
  assert posts == ["refresh-1", "refresh-1"]
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True
  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(refresh())
  assert posts == ["refresh-1", "refresh-1"]


@pytest.mark.parametrize(
  "exception_type",
  [httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout],
)
def test_presend_failure_clears_marker_and_next_refresh_succeeds(
  tmp_path: Path,
  exception_type: type[httpx.TransportError],
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  posts: list[str] = []

  async def failed(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    raise exception_type("not sent", request=request)

  async def first() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(failed)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(exception_type):
    _run(first())
  assert load_xai_token_record(settings.store_path) == original
  assert not xai_oauth._STORE_REFRESH_CONSUMED

  async def succeeded(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  async def second() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(succeeded)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  _run(second())
  assert posts == ["refresh-1", "refresh-1"]
  assert load_xai_token_record(settings.store_path)["refresh_token"] == "refresh-2"


def test_presend_then_lost_response_quarantines_on_second_invocation(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  posts: list[str] = []
  exceptions = [httpx.ConnectError, httpx.ReadTimeout, httpx.ReadTimeout]

  async def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    exception_type = exceptions.pop(0)
    raise exception_type("transport failure", request=request)

  transport = httpx.MockTransport(handler)
  _install_retry_transport(monkeypatch, transport)

  async def refresh() -> None:
    async with httpx.AsyncClient(transport=transport) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(httpx.ConnectError):
    _run(refresh())
  assert load_xai_token_record(settings.store_path) == original
  with pytest.raises(httpx.ReadTimeout):
    _run(refresh())
  assert posts == ["refresh-1", "refresh-1", "refresh-1"]
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True
  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(refresh())
  assert posts == ["refresh-1", "refresh-1", "refresh-1"]


def test_grace_recovery_saves_successor_after_lost_response(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record(device_authorization_endpoint="https://auth.x.ai/oauth2/device/code")
  save_xai_token_record(settings.store_path, original)
  first_posts: list[str] = []
  retry_posts: list[str] = []

  async def first_handler(request: httpx.Request) -> httpx.Response:
    first_posts.append(_token_form(request)["refresh_token"][0])
    raise httpx.ReadTimeout("response lost", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    retry_posts.append(_token_form(request)["refresh_token"][0])
    assert request.headers["accept"] == "application/json"
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> dict:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      return await refresh_xai_oauth_token(
        original,
        settings=settings,
        client=client,
        force=True,
      )

  refreshed = _run(run())
  assert first_posts == ["refresh-1"]
  assert retry_posts == ["refresh-1"]
  assert refreshed["refresh_token"] == "refresh-2"
  assert refreshed["token_endpoint"] == original["token_endpoint"]
  assert (
    refreshed["device_authorization_endpoint"]
    == original["device_authorization_endpoint"]
  )
  stored = load_xai_token_record(settings.store_path)
  assert stored["refresh_token"] == "refresh-2"
  assert "refresh_pending" not in stored


def test_grace_recovery_invalid_grant_keeps_marker(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  posts: list[str] = []

  async def first_handler(request: httpx.Request) -> httpx.Response:
    posts.append("first")
    raise httpx.ReadError("response lost", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    posts.append("retry")
    return httpx.Response(400, json={"error": "invalid_grant"})

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(RuntimeError, match="invalid_grant"):
    _run(run())
  assert posts == ["first", "retry"]
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True


def test_grace_recovery_double_ambiguous_stops_after_one_retry(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  posts: list[str] = []

  async def first_handler(request: httpx.Request) -> httpx.Response:
    posts.append("first")
    raise httpx.ReadTimeout("first response lost", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    posts.append("retry")
    raise httpx.ReadTimeout("retry response lost", request=request)

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(httpx.ReadTimeout, match="retry response lost"):
    _run(run())
  assert posts == ["first", "retry"]
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True


def test_grace_recovery_skips_retry_when_initial_budget_is_too_small(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  retry_posts = 0
  monotonic_values = iter((100.0, 104.5))
  real_time = xai_oauth.time

  class TimeProxy:
    def monotonic(self) -> float:
      return next(monotonic_values)

    def __getattr__(self, name: str):
      return getattr(real_time, name)

  monkeypatch.setattr(xai_oauth, "time", TimeProxy())

  async def first_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("original ambiguous", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    nonlocal retry_posts
    retry_posts += 1
    return httpx.Response(500)

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(httpx.ReadTimeout, match="original ambiguous"):
    _run(run())
  assert retry_posts == 0
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True


def test_grace_recovery_charges_client_construction_to_budget(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  retry_posts = 0
  created: list[tuple[httpx.AsyncClient, dict]] = []
  monotonic_values = iter((100.0, 100.5, 104.5))
  real_time = xai_oauth.time

  class TimeProxy:
    def monotonic(self) -> float:
      return next(monotonic_values)

    def __getattr__(self, name: str):
      return getattr(real_time, name)

  monkeypatch.setattr(xai_oauth, "time", TimeProxy())

  async def first_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("original ambiguous", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    nonlocal retry_posts
    retry_posts += 1
    return httpx.Response(500)

  _install_retry_transport(
    monkeypatch,
    httpx.MockTransport(retry_handler),
    created=created,
  )

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(httpx.ReadTimeout, match="original ambiguous"):
    _run(run())
  assert len(created) == 1
  assert created[0][0].is_closed
  assert retry_posts == 0
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True


def test_grace_retry_deadline_cancels_only_local_retry_task(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  retry_started = asyncio.Event()
  retry_cancelled = asyncio.Event()
  never_release = asyncio.Event()
  monkeypatch.setattr(xai_oauth, "_GRACE_HARD_BUDGET_S", 0.05)
  monkeypatch.setattr(xai_oauth, "_MIN_RETRY_BUDGET_S", 0.001)

  async def first_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("response lost", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    retry_started.set()
    try:
      await never_release.wait()
    except asyncio.CancelledError:
      retry_cancelled.set()
      raise
    raise AssertionError("retry unexpectedly released")

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      with pytest.raises(asyncio.TimeoutError):
        await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)
      assert retry_started.is_set()
      assert retry_cancelled.is_set()
      assert asyncio.current_task() is not None
      assert not asyncio.current_task().cancelled()

  _run(run())
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True


def test_delayed_first_post_success_beyond_grace_budget_is_preserved(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  first_posts = 0
  retry_posts = 0
  monkeypatch.setattr(xai_oauth, "_GRACE_HARD_BUDGET_S", 0.01)

  async def first_handler(request: httpx.Request) -> httpx.Response:
    nonlocal first_posts
    first_posts += 1
    await asyncio.sleep(0.03)
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    nonlocal retry_posts
    retry_posts += 1
    return httpx.Response(500)

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> dict:
    async with httpx.AsyncClient(
      transport=httpx.MockTransport(first_handler),
      timeout=1.0,
    ) as client:
      return await refresh_xai_oauth_token(
        original,
        settings=settings,
        client=client,
        force=True,
      )

  assert _run(run())["refresh_token"] == "refresh-2"
  assert first_posts == 1
  assert retry_posts == 0
  assert "refresh_pending" not in load_xai_token_record(settings.store_path)


@pytest.mark.parametrize("exception_type", xai_oauth._PRESEND_EXC)
@pytest.mark.parametrize("was_present", [False, True])
def test_every_presend_failure_keeps_existing_cleanup_semantics(
  tmp_path: Path,
  exception_type: type[httpx.TransportError],
  was_present: bool,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  canonical = os.path.realpath(settings.store_path)
  if was_present:
    xai_oauth._STORE_REFRESH_CONSUMED.add(canonical)
  posts = 0

  async def handler(request: httpx.Request) -> httpx.Response:
    nonlocal posts
    posts += 1
    raise exception_type("provably pre-send", request=request)

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(exception_type):
    _run(run())
  assert posts == 1
  assert load_xai_token_record(settings.store_path) == original
  assert (canonical in xai_oauth._STORE_REFRESH_CONSUMED) is was_present


@pytest.mark.parametrize(
  "exception_type",
  [httpx.ReadTimeout, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError],
)
def test_ambiguous_transport_errors_each_trigger_one_grace_retry(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  exception_type: type[httpx.TransportError],
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  posts: list[str] = []

  async def first_handler(request: httpx.Request) -> httpx.Response:
    posts.append("first")
    raise exception_type("ambiguous", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    posts.append("retry")
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> dict:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      return await refresh_xai_oauth_token(
        original,
        settings=settings,
        client=client,
        force=True,
      )

  assert _run(run())["refresh_token"] == "refresh-2"
  assert posts == ["first", "retry"]


def test_explicit_refresh_reject_is_not_grace_retried(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  first_posts = 0
  retry_posts = 0

  async def first_handler(request: httpx.Request) -> httpx.Response:
    nonlocal first_posts
    first_posts += 1
    return httpx.Response(400, json={"error": "invalid_grant"})

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    nonlocal retry_posts
    retry_posts += 1
    return httpx.Response(500)

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(RuntimeError, match="invalid_grant"):
    _run(run())
  assert first_posts == 1
  assert retry_posts == 0
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True


def test_first_post_success_never_constructs_grace_client(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  created: list[tuple[httpx.AsyncClient, dict]] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  _install_retry_transport(
    monkeypatch,
    httpx.MockTransport(lambda request: httpx.Response(500)),
    created=created,
  )

  async def run() -> dict:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      return await refresh_xai_oauth_token(
        original,
        settings=settings,
        client=client,
        force=True,
      )

  assert _run(run())["refresh_token"] == "refresh-2"
  assert created == []
  assert "refresh_pending" not in load_xai_token_record(settings.store_path)


def test_restart_pending_marker_still_blocks_before_any_post(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  save_xai_token_record(settings.store_path, _record(refresh_pending=True))
  posts = 0

  async def handler(request: httpx.Request) -> httpx.Response:
    nonlocal posts
    posts += 1
    return httpx.Response(500)

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(_record(), settings=settings, client=client, force=True)

  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(run())
  assert posts == 0


def test_cancelled_error_during_first_post_propagates_without_retry(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  retry_posts = 0

  async def first_handler(request: httpx.Request) -> httpx.Response:
    raise asyncio.CancelledError

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    nonlocal retry_posts
    retry_posts += 1
    return httpx.Response(500)

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      with pytest.raises(asyncio.CancelledError):
        await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  _run(run())
  assert retry_posts == 0
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True


def test_cancelled_error_during_grace_retry_propagates_untouched(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  retry_posts = 0

  async def first_handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("response lost", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    nonlocal retry_posts
    retry_posts += 1
    raise asyncio.CancelledError

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(first_handler)) as client:
      with pytest.raises(asyncio.CancelledError):
        await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  _run(run())
  assert retry_posts == 1
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True


def test_success_without_rotated_refresh_token_stays_quarantined(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  posts: list[str] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(200, json={"access_token": "access-2", "expires_in": 3600})

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(RuntimeError, match="missing refresh_token"):
    _run(run())
  stored = load_xai_token_record(settings.store_path)
  assert stored["refresh_pending"] is True
  assert stored["refresh_token"] == "refresh-1"
  assert posts == ["refresh-1"]


def test_invalid_grant_retains_marker_and_subsequent_refresh_is_gated(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  posts: list[str] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(400, json={"error": "invalid_grant"})

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(RuntimeError, match="invalid_grant"):
    _run(run())
  assert load_xai_token_record(settings.store_path)["refresh_pending"] is True
  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(run())
  assert posts == ["refresh-1"]


def test_rotated_token_save_failure_retries_and_retains_marker(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)
  real_save = xai_oauth.save_xai_token_record
  save_calls = 0
  posts: list[str] = []

  def fail_rotated_save(path: Path, record: dict) -> None:
    nonlocal save_calls
    save_calls += 1
    if record.get("refresh_token") == "refresh-2":
      raise OSError("rotated save failed")
    real_save(path, record)

  async def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  monkeypatch.setattr(xai_oauth, "save_xai_token_record", fail_rotated_save)

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

  with pytest.raises(OSError, match="rotated save failed"):
    _run(run())
  assert save_calls == 1 + xai_oauth._SAVE_ATTEMPTS
  stored = load_xai_token_record(settings.store_path)
  assert stored["refresh_pending"] is True
  assert stored["refresh_token"] == "refresh-1"
  assert posts == ["refresh-1"]


@pytest.mark.parametrize("marker", ["refresh_pending", "reauth_required"])
def test_bootstrap_marked_caller_is_quarantined_without_post(
  tmp_path: Path,
  marker: str,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  posts: list[str] = []

  async def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(500)

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(
        _record(**{marker: True}),
        settings=settings,
        client=client,
        force=True,
      )

  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(run())
  assert posts == []


def test_preexisting_pending_marker_fails_closed_without_transport_entry(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  save_xai_token_record(settings.store_path, _record(refresh_pending=True))
  entered = False

  async def handler(request: httpx.Request) -> httpx.Response:
    nonlocal entered
    entered = True
    return httpx.Response(500)

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(_record(), settings=settings, client=client, force=True)

  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(run())
  assert entered is False


@pytest.mark.parametrize("initial_state", ["refresh_pending", "reauth_required"])
def test_login_clears_refresh_quarantine_and_logout_tombstone(
  tmp_path: Path,
  initial_state: str,
) -> None:
  store = tmp_path / "oauth.json"
  if initial_state == "refresh_pending":
    save_xai_token_record(store, _record(refresh_pending=True))
  else:
    async def write_tombstone() -> None:
      async with xai_oauth._store_lock(store):
        xai_oauth._write_reauth_tombstone(store)

    _run(write_tombstone())

  async def handler(request: httpx.Request) -> httpx.Response:
    if request.method == "GET":
      return httpx.Response(
        200,
        json={
          "device_authorization_endpoint": "https://auth.x.ai/oauth2/device/code",
          "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
      )
    if request.url.path.endswith("/device/code"):
      return httpx.Response(
        200,
        json={
          "device_code": "device-1",
          "user_code": "ABCD",
          "verification_uri": "https://accounts.x.ai/device",
          "expires_in": 900,
          "interval": 1,
        },
      )
    return httpx.Response(
      200,
      json={
        "access_token": "access-login",
        "refresh_token": "refresh-login",
        "expires_in": 3600,
      },
    )

  async def login() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await login_xai_device_code(config={"auth_store_path": str(store)}, client=client)

  _run(login())
  stored = load_xai_token_record(store)
  assert stored["access_token"] == "access-login"
  assert stored["refresh_token"] == "refresh-login"
  assert "refresh_pending" not in stored
  assert "reauth_required" not in stored


@pytest.mark.parametrize("marker", ["refresh_pending", "reauth_required"])
def test_provider_reports_marked_oauth_store_unavailable(
  tmp_path: Path,
  marker: str,
) -> None:
  store = tmp_path / "oauth.json"
  if marker == "refresh_pending":
    save_xai_token_record(store, _record(refresh_pending=True))
  else:
    async def write_tombstone() -> None:
      async with xai_oauth._store_lock(store):
        xai_oauth._write_reauth_tombstone(store)

    _run(write_tombstone())
  provider = XAIProvider()
  config = {"auth_mode": "oauth", "auth_store_path": str(store)}
  assert provider.has_active_credential(config) is False


def test_provider_reports_unmarked_oauth_store_available(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record())
  provider = XAIProvider()
  assert provider.has_active_credential({
    "auth_mode": "oauth",
    "auth_store_path": str(store),
  }) is True


def test_recreated_client_refresh_is_gated_but_valid_cached_access_can_infer(
  tmp_path: Path,
) -> None:
  store = tmp_path / "oauth.json"
  refresh_posts: list[str] = []
  inference_posts = 0

  async def handler(request: httpx.Request) -> httpx.Response:
    nonlocal inference_posts
    if request.url.path.endswith("/oauth2/token"):
      refresh_posts.append(_token_form(request)["refresh_token"][0])
      return httpx.Response(500)
    inference_posts += 1
    return _completed_response("ok")

  provider = XAIProvider()
  config = {
    "auth_mode": "oauth",
    "auth_store_path": str(store),
    "_transport": httpx.MockTransport(handler),
  }
  save_xai_token_record(store, _record(expires_at=1, refresh_pending=True))
  expiring_client = provider.create_client(config)
  assert refresh_posts == []

  async def drive_expiring() -> None:
    try:
      async for _event in provider.stream(
        expiring_client,
        {"model": "grok-4.5", "input": "test", "stream": True},
      ):
        pass
    finally:
      await provider.close_client(expiring_client)

  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(drive_expiring())
  assert refresh_posts == []
  assert inference_posts == 0

  save_xai_token_record(store, _record(refresh_pending=True))
  cached_client = provider.create_client(config)

  async def drive_cached() -> None:
    try:
      async for _event in provider.stream(
        cached_client,
        {"model": "grok-4.5", "input": "test", "stream": True},
      ):
        pass
    finally:
      await provider.close_client(cached_client)

  _run(drive_cached())
  assert refresh_posts == []
  assert inference_posts == 1


def test_ambiguous_refresh_guard_blocks_bootstrap_after_store_loss(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  cached = _record()
  save_xai_token_record(settings.store_path, cached)
  posts: list[str] = []

  async def lost(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    raise httpx.ReadTimeout("response lost", request=request)

  lost_transport = httpx.MockTransport(lost)
  _install_retry_transport(monkeypatch, lost_transport)

  async def refresh(transport: httpx.AsyncBaseTransport) -> None:
    async with httpx.AsyncClient(transport=transport) as client:
      await refresh_xai_oauth_token(cached, settings=settings, client=client, force=True)

  with pytest.raises(httpx.ReadTimeout):
    _run(refresh(lost_transport))
  settings.store_path.unlink()

  async def should_not_post(request: httpx.Request) -> httpx.Response:
    posts.append("unexpected")
    return httpx.Response(500)

  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(refresh(httpx.MockTransport(should_not_post)))
  assert posts == ["refresh-1", "refresh-1"]


def test_successful_refresh_guard_blocks_stale_second_client_after_store_loss(
  tmp_path: Path,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  client_a_cached = _record()
  client_b_cached = dict(client_a_cached)
  save_xai_token_record(settings.store_path, client_a_cached)
  posts: list[str] = []

  async def success(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  async def refresh(record: dict, transport: httpx.AsyncBaseTransport) -> None:
    async with httpx.AsyncClient(transport=transport) as client:
      await refresh_xai_oauth_token(record, settings=settings, client=client, force=True)

  _run(refresh(client_a_cached, httpx.MockTransport(success)))
  assert os.path.realpath(settings.store_path) in xai_oauth._STORE_REFRESH_CONSUMED
  settings.store_path.unlink()

  async def should_not_post(request: httpx.Request) -> httpx.Response:
    posts.append("unexpected")
    return httpx.Response(500)

  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(refresh(client_b_cached, httpx.MockTransport(should_not_post)))
  assert posts == ["refresh-1"]


def test_first_bootstrap_presend_failure_removes_own_guard_and_allows_retry(
  tmp_path: Path,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  cached = _record()
  posts: list[str] = []

  async def presend(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    raise httpx.ConnectError("not sent", request=request)

  async def refresh(transport: httpx.AsyncBaseTransport) -> None:
    async with httpx.AsyncClient(transport=transport) as client:
      await refresh_xai_oauth_token(cached, settings=settings, client=client, force=True)

  with pytest.raises(httpx.ConnectError):
    _run(refresh(httpx.MockTransport(presend)))
  assert not xai_oauth._STORE_REFRESH_CONSUMED
  settings.store_path.unlink()

  async def success(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  _run(refresh(httpx.MockTransport(success)))
  assert posts == ["refresh-1", "refresh-1"]


def test_presend_failure_never_resets_prior_consuming_history(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  stale_client_b = _record()
  save_xai_token_record(settings.store_path, stale_client_b)
  posts: list[str] = []

  async def first_success(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  async def refresh(record: dict, transport: httpx.AsyncBaseTransport) -> None:
    async with httpx.AsyncClient(transport=transport) as client:
      await refresh_xai_oauth_token(record, settings=settings, client=client, force=True)

  _run(refresh(stale_client_b, httpx.MockTransport(first_success)))

  async def presend(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    raise httpx.ConnectError("not sent", request=request)

  with pytest.raises(httpx.ConnectError):
    _run(refresh(load_xai_token_record(settings.store_path), httpx.MockTransport(presend)))
  canonical = os.path.realpath(settings.store_path)
  assert canonical in xai_oauth._STORE_REFRESH_CONSUMED
  assert "refresh_pending" not in load_xai_token_record(settings.store_path)
  settings.store_path.unlink()

  async def should_not_post(request: httpx.Request) -> httpx.Response:
    posts.append("unexpected")
    return httpx.Response(500)

  with pytest.raises(RuntimeError, match="refresh did not complete previously"):
    _run(refresh(stale_client_b, httpx.MockTransport(should_not_post)))
  assert posts == ["refresh-1", "refresh-2"]


def test_provider_refreshes_once_on_401_then_retries_response(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record())
  response_attempts = 0
  authorization_headers: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    nonlocal response_attempts
    if request.url.path.endswith("/oauth2/token"):
      return httpx.Response(
        200,
        json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
      )
    response_attempts += 1
    authorization_headers.append(str(request.headers.get("authorization")))
    if response_attempts == 1:
      return httpx.Response(401, json={"error": {"message": "expired"}})
    payload = {
      "type": "response.completed",
      "response": {"status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    }
    return httpx.Response(200, text=f"data: {json.dumps(payload)}\n\n")

  provider = XAIProvider()
  config = {
    "auth_mode": "oauth",
    "auth_store_path": str(store),
    "_transport": httpx.MockTransport(handler),
  }
  client = provider.create_client(config)

  async def collect():
    try:
      return [event async for event in provider.stream(client, {"model": "grok-4.5", "stream": True})]
    finally:
      await provider.close_client(client)

  events = _run(collect())
  assert response_attempts == 2
  assert authorization_headers == ["Bearer access-1", "Bearer access-2"]
  assert events[-1].type == "message_end"
  assert load_xai_token_record(store)["refresh_token"] == "refresh-2"


def test_untrusted_discovery_override_is_rejected() -> None:
  with pytest.raises(ValueError, match="untrusted discovery URL"):
    resolve_xai_oauth_settings({"oauth_discovery_url": "https://evil.example/.well-known/openid"})


def test_proactive_refresh_adopts_fresh_store_without_posting(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  stored = _record(refresh_token="refresh-store")
  save_xai_token_record(settings.store_path, stored)
  posts: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(500)

  async def run() -> dict:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      return await refresh_xai_oauth_token(
        _record(refresh_token="refresh-caller"),
        settings=settings,
        client=client,
        force=False,
      )

  assert _run(run()) == stored
  assert posts == []


def test_independent_clients_serialize_and_never_reuse_refresh_token(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  initial = _record(expires_at=1)
  save_xai_token_record(settings.store_path, initial)

  async def run() -> None:
    transport = _GatedRefreshTransport()
    original_store_lock = xai_oauth._store_lock
    lock_attempts = 0
    second_lock_attempted = asyncio.Event()

    @asynccontextmanager
    async def observed_store_lock(path: Path):
      nonlocal lock_attempts
      lock_attempts += 1
      if lock_attempts == 2:
        second_lock_attempted.set()
      async with original_store_lock(path):
        yield

    monkeypatch.setattr(xai_oauth, "_store_lock", observed_store_lock)
    async with (
      httpx.AsyncClient(transport=transport) as first_client,
      httpx.AsyncClient(transport=transport) as second_client,
    ):
      first = asyncio.create_task(
        refresh_xai_oauth_token(initial, settings=settings, client=first_client, force=True)
      )
      await transport.first_posted.wait()
      second_started = asyncio.Event()

      async def second_refresh() -> dict:
        second_started.set()
        return await refresh_xai_oauth_token(
          initial,
          settings=settings,
          client=second_client,
          force=True,
        )

      second = asyncio.create_task(second_refresh())
      await second_started.wait()
      await second_lock_attempted.wait()
      assert transport.posts == ["refresh-1"]
      transport.release_first.set()
      await asyncio.gather(first, second)
    assert transport.posts == ["refresh-1", "refresh-2"]
    assert len(transport.posts) == len(set(transport.posts))

  _run(run())
  assert load_xai_token_record(settings.store_path)["refresh_token"] == "refresh-3"


def test_concurrent_proactive_refresh_on_one_client_rotates_once(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record(expires_at=1))

  async def run() -> list[str]:
    transport = _GatedRefreshTransport()
    provider = XAIProvider()
    client = provider.create_client({
      "auth_mode": "oauth",
      "auth_store_path": str(store),
      "_transport": transport,
    })
    state = provider._client_state[client]
    try:
      first = asyncio.create_task(provider._refresh_oauth_state(state, client, force=False))
      await transport.first_posted.wait()
      second = asyncio.create_task(provider._refresh_oauth_state(state, client, force=False))
      transport.release_first.set()
      await asyncio.gather(first, second)
      return transport.posts
    finally:
      await provider.close_client(client)

  assert _run(run()) == ["refresh-1"]


def test_concurrent_reactive_401_refreshes_once_when_access_changes(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record())

  async def run() -> list[str]:
    transport = _GatedRefreshTransport()
    provider = XAIProvider()
    client = provider.create_client({
      "auth_mode": "oauth",
      "auth_store_path": str(store),
      "_transport": transport,
    })
    state = provider._client_state[client]
    try:
      first = asyncio.create_task(
        provider._refresh_oauth_state(
          state,
          client,
          force=True,
          rejected_token="access-1",
        )
      )
      await transport.first_posted.wait()
      second = asyncio.create_task(
        provider._refresh_oauth_state(
          state,
          client,
          force=True,
          rejected_token="access-1",
        )
      )
      transport.release_first.set()
      await asyncio.gather(first, second)
      assert state["token"] == "access-2"
      return transport.posts
    finally:
      await provider.close_client(client)

  assert _run(run()) == ["refresh-1"]


def test_reactive_same_access_token_forces_second_safe_rotation(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record())

  async def run() -> list[str]:
    transport = _GatedRefreshTransport(same_access=True)
    provider = XAIProvider()
    client = provider.create_client({
      "auth_mode": "oauth",
      "auth_store_path": str(store),
      "_transport": transport,
    })
    state = provider._client_state[client]
    try:
      first = asyncio.create_task(
        provider._refresh_oauth_state(
          state,
          client,
          force=True,
          rejected_token="access-1",
        )
      )
      await transport.first_posted.wait()
      second = asyncio.create_task(
        provider._refresh_oauth_state(
          state,
          client,
          force=True,
          rejected_token="access-1",
        )
      )
      transport.release_first.set()
      await asyncio.gather(first, second)
      return transport.posts
    finally:
      await provider.close_client(client)

  posts = _run(run())
  assert posts == ["refresh-1", "refresh-2"]
  assert len(posts) == len(set(posts))


def test_short_lifetime_refresh_uses_new_store_token_not_stale_caller(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  initial = _record(expires_at=1)
  save_xai_token_record(settings.store_path, initial)

  async def run() -> list[str]:
    transport = _GatedRefreshTransport(expires_in=30)
    async with (
      httpx.AsyncClient(transport=transport) as first_client,
      httpx.AsyncClient(transport=transport) as second_client,
    ):
      first = asyncio.create_task(
        refresh_xai_oauth_token(initial, settings=settings, client=first_client)
      )
      await transport.first_posted.wait()
      second = asyncio.create_task(
        refresh_xai_oauth_token(initial, settings=settings, client=second_client)
      )
      transport.release_first.set()
      await asyncio.gather(first, second)
    return transport.posts

  posts = _run(run())
  assert posts == ["refresh-1", "refresh-2"]
  assert posts.count("refresh-1") == 1


def test_config_overlay_chimera_posts_store_refresh_token(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record(refresh_token="refresh-store", expires_at=1))
  monkeypatch.setenv("XAI_REFRESH_TOKEN", "refresh-caller")
  caller, settings = oauth_record_from_config({"auth_store_path": str(store)})
  assert caller["refresh_token"] == "refresh-caller"
  posts: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(
      200,
      json={"access_token": "access-new", "refresh_token": "refresh-new", "expires_in": 3600},
    )

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(caller, settings=settings, client=client)

  _run(run())
  assert posts == ["refresh-store"]


def test_bootstrap_uses_caller_only_when_store_absent_and_invalid_json_raises(
  tmp_path: Path,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  caller = _record(refresh_token="refresh-bootstrap")
  posts: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(
      200,
      json={"access_token": "access-new", "refresh_token": "refresh-new", "expires_in": 3600},
    )

  async def refresh() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(caller, settings=settings, client=client)

  _run(refresh())
  assert posts == ["refresh-bootstrap"]
  settings.store_path.write_text("{invalid", encoding="utf-8")
  with pytest.raises(RuntimeError, match="Invalid xAI OAuth token store JSON"):
    _run(refresh())
  assert posts == ["refresh-bootstrap"]


def test_present_invalid_store_fails_closed_without_posting_caller(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  save_xai_token_record(settings.store_path, _record(issuer="https://wrong.example"))
  posts: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(500)

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(
        _record(refresh_token="refresh-stale-caller"),
        settings=settings,
        client=client,
        force=True,
      )

  with pytest.raises(RuntimeError, match="present but not valid"):
    _run(run())
  assert posts == []


@pytest.mark.parametrize(
  "invalid",
  [
    {"issuer": "https://wrong.example"},
    {"client_id": "wrong-client"},
    {"access_token": "  "},
    {"refresh_token": "\t"},
    {"expires_at": 0},
    {"expires_at": float("nan")},
    {"expires_at": float("inf")},
    {"expires_at": float("-inf")},
    {"expires_at": "not-a-number"},
    {"token_endpoint": 42},
    {"token_endpoint": "https://evil.example/token"},
  ],
)
def test_invalid_present_records_fail_closed_and_never_post(
  tmp_path: Path,
  invalid: dict,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  stored = _record(**invalid)
  save_xai_token_record(settings.store_path, stored)
  assert _record_is_valid(stored, settings) is False
  posts: list[str] = []

  def handler(request: httpx.Request) -> httpx.Response:
    posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(500)

  async def run() -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      await refresh_xai_oauth_token(
        _record(refresh_token="refresh-caller"),
        settings=settings,
        client=client,
        force=True,
      )

  with pytest.raises(RuntimeError, match="present but not valid"):
    _run(run())
  assert posts == []


def test_caller_cancellation_during_successful_grace_retry_waits_for_save(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record(expires_at=1))

  class RetryLifecycleTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
      self.accepted = asyncio.Event()
      self.release_response = asyncio.Event()
      self.closed = asyncio.Event()
      self.in_flight = False
      self.aborted = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
      self.in_flight = True
      self.accepted.set()
      await self.release_response.wait()
      self.in_flight = False
      return httpx.Response(
        200,
        json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
      )

    async def aclose(self) -> None:
      if self.in_flight:
        self.aborted = True
        self.release_response.set()
      self.closed.set()

  class LifecycleProvider(XAIProvider):
    def __init__(self) -> None:
      super().__init__()
      self.close_started = asyncio.Event()

    async def close_client(self, client, timeout: float = 2.0) -> None:
      self.close_started.set()
      await super().close_client(client, timeout=timeout)

  retry_transport = RetryLifecycleTransport()
  _install_retry_transport(monkeypatch, retry_transport)

  async def first_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/oauth2/token"):
      raise httpx.ReadTimeout("response lost", request=request)
    return _completed_response()

  async def run() -> None:
    provider = LifecycleProvider()
    client = provider.create_client({
      "auth_mode": "oauth",
      "auth_store_path": str(store),
      "_transport": httpx.MockTransport(first_handler),
    })

    async def collect_with_close() -> None:
      try:
        async for _event in provider.stream(
          client,
          {"model": "grok-4.5", "input": "summarize", "stream": True},
        ):
          pass
      finally:
        await provider.close_client(client)

    task = asyncio.create_task(collect_with_close())
    await retry_transport.accepted.wait()
    task.cancel()
    cancellation_processed = asyncio.Event()
    asyncio.get_running_loop().call_soon(cancellation_processed.set)
    await cancellation_processed.wait()
    assert not task.done()
    assert not provider.close_started.is_set()
    assert not retry_transport.closed.is_set()
    retry_transport.release_response.set()
    with pytest.raises(asyncio.CancelledError):
      await task
    assert load_xai_token_record(store)["refresh_token"] == "refresh-2"
    assert provider.close_started.is_set()
    assert retry_transport.closed.is_set()
    assert retry_transport.aborted is False

  _run(run())


def test_grace_retry_timeout_is_isolated_from_caller_client_and_pool(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  store = tmp_path / "oauth.json"
  settings = resolve_xai_oauth_settings({"auth_store_path": str(store)})
  original = _record()
  save_xai_token_record(store, original)
  monkeypatch.setattr(xai_oauth, "_GRACE_HARD_BUDGET_S", 0.05)
  monkeypatch.setattr(xai_oauth, "_MIN_RETRY_BUDGET_S", 0.001)

  class CallerTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
      self.posts: list[str] = []
      self.closed = False
      self.aborted = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
      self.posts.append(_token_form(request)["refresh_token"][0])
      if len(self.posts) == 1:
        raise httpx.ReadTimeout("response lost", request=request)
      return httpx.Response(
        200,
        json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
      )

    async def aclose(self) -> None:
      self.closed = True

  retry_started = asyncio.Event()
  retry_cancelled = asyncio.Event()
  never_release = asyncio.Event()

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    retry_started.set()
    try:
      await never_release.wait()
    except asyncio.CancelledError:
      retry_cancelled.set()
      raise
    raise AssertionError("retry unexpectedly released")

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> None:
    transport = CallerTransport()
    client = httpx.AsyncClient(transport=transport)
    try:
      with pytest.raises(asyncio.TimeoutError):
        await refresh_xai_oauth_token(
          original,
          settings=settings,
          client=client,
          force=True,
        )
      assert retry_started.is_set()
      assert retry_cancelled.is_set()
      assert not client.is_closed
      assert transport.closed is False
      assert transport.aborted is False

      save_xai_token_record(store, original)
      refreshed = await refresh_xai_oauth_token(
        original,
        settings=settings,
        client=client,
        force=True,
      )
      assert refreshed["refresh_token"] == "refresh-2"
      assert transport.posts == ["refresh-1", "refresh-1"]
      assert not client.is_closed
      assert transport.closed is False
      assert transport.aborted is False
    finally:
      await client.aclose()
    assert transport.closed is True

  _run(run())


def test_discovery_first_post_and_grace_retry_use_expected_timeout_scopes(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  store = tmp_path / "oauth.json"
  settings = resolve_xai_oauth_settings({"auth_store_path": str(store)})
  original = _record(token_endpoint=None)
  save_xai_token_record(store, original)
  caller_timeouts: list[tuple[str, dict]] = []
  retry_timeouts: list[dict] = []
  created: list[tuple[httpx.AsyncClient, dict]] = []

  async def caller_handler(request: httpx.Request) -> httpx.Response:
    caller_timeouts.append((request.method, dict(request.extensions["timeout"])))
    if request.method == "GET":
      return httpx.Response(
        200,
        json={
          "device_authorization_endpoint": "https://auth.x.ai/oauth2/device/code",
          "token_endpoint": "https://auth.x.ai/oauth2/token",
        },
      )
    raise httpx.ReadTimeout("response lost", request=request)

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    retry_timeouts.append(dict(request.extensions["timeout"]))
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  _install_retry_transport(
    monkeypatch,
    httpx.MockTransport(retry_handler),
    created=created,
  )

  async def run() -> dict:
    timeout = httpx.Timeout(17.0, connect=13.0)
    async with httpx.AsyncClient(
      transport=httpx.MockTransport(caller_handler),
      timeout=timeout,
    ) as client:
      return await refresh_xai_oauth_token(
        original,
        settings=settings,
        client=client,
        force=True,
      )

  assert _run(run())["refresh_token"] == "refresh-2"
  assert [method for method, _timeout in caller_timeouts] == ["GET", "POST"]
  for _method, timeout in caller_timeouts:
    assert timeout == {"connect": 13.0, "read": 17.0, "write": 17.0, "pool": 17.0}
  assert len(created) == 1
  retry_client_timeout = created[0][1]["timeout"]
  assert 1.0 <= retry_client_timeout.connect <= xai_oauth._GRACE_HARD_BUDGET_S
  assert retry_client_timeout.connect == retry_client_timeout.read
  assert retry_client_timeout.connect == retry_client_timeout.write
  assert retry_client_timeout.connect == retry_client_timeout.pool
  assert len(retry_timeouts) == 1
  assert retry_timeouts[0]["connect"] == retry_client_timeout.connect
  assert retry_timeouts[0]["read"] == retry_client_timeout.read


@pytest.mark.parametrize("entry_path", ["proactive", "reactive"])
def test_provider_stream_entry_paths_drive_grace_recovery(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  entry_path: str,
) -> None:
  store = tmp_path / "oauth.json"
  expires_at = 1 if entry_path == "proactive" else 4_000_000_000
  save_xai_token_record(store, _record(expires_at=expires_at))
  caller_refresh_posts: list[str] = []
  retry_posts: list[str] = []
  inference_attempts = 0

  async def caller_handler(request: httpx.Request) -> httpx.Response:
    nonlocal inference_attempts
    if request.url.path.endswith("/oauth2/token"):
      caller_refresh_posts.append(_token_form(request)["refresh_token"][0])
      raise httpx.ReadTimeout("response lost", request=request)
    inference_attempts += 1
    if entry_path == "reactive" and inference_attempts == 1:
      return httpx.Response(401, json={"error": {"message": "expired"}})
    return _completed_response("recovered")

  async def retry_handler(request: httpx.Request) -> httpx.Response:
    retry_posts.append(_token_form(request)["refresh_token"][0])
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  _install_retry_transport(monkeypatch, httpx.MockTransport(retry_handler))

  async def run() -> list:
    provider = XAIProvider()
    client = provider.create_client({
      "auth_mode": "oauth",
      "auth_store_path": str(store),
      "_transport": httpx.MockTransport(caller_handler),
    })
    try:
      return [
        event
        async for event in provider.stream(
          client,
          {"model": "grok-4.5", "input": "test", "stream": True},
        )
      ]
    finally:
      await provider.close_client(client)

  events = _run(run())
  assert caller_refresh_posts == ["refresh-1"]
  assert retry_posts == ["refresh-1"]
  assert inference_attempts == (1 if entry_path == "proactive" else 2)
  assert events[-1].type == "message_end"
  stored = load_xai_token_record(store)
  assert stored["refresh_token"] == "refresh-2"
  assert "refresh_pending" not in stored


def test_cancelled_summarizer_keeps_client_open_until_rotated_token_is_saved(
  tmp_path: Path,
) -> None:
  store = tmp_path / "oauth.json"
  save_xai_token_record(store, _record(expires_at=1))

  class LifecycleTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
      self.accepted = asyncio.Event()
      self.release_response = asyncio.Event()
      self.closed = asyncio.Event()
      self.in_flight = False
      self.aborted = False
      self.posts: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
      if not request.url.path.endswith("/oauth2/token"):
        return _completed_response()
      self.posts.append(_token_form(request)["refresh_token"][0])
      self.in_flight = True
      self.accepted.set()
      await self.release_response.wait()
      self.in_flight = False
      if self.aborted:
        raise httpx.ReadError("refresh aborted by client close", request=request)
      return httpx.Response(
        200,
        json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
      )

    async def aclose(self) -> None:
      if self.in_flight:
        self.aborted = True
        self.release_response.set()
      self.closed.set()

  class LifecycleProvider(XAIProvider):
    def __init__(self) -> None:
      super().__init__()
      self.close_started = asyncio.Event()

    async def close_client(self, client, timeout: float = 2.0) -> None:
      self.close_started.set()
      await super().close_client(client, timeout=timeout)

  async def run() -> None:
    transport = LifecycleTransport()
    provider = LifecycleProvider()
    client = provider.create_client({
      "auth_mode": "oauth",
      "auth_store_path": str(store),
      "_transport": transport,
    })

    async def collect_with_close() -> None:
      try:
        async for _event in provider.stream(
          client,
          {"model": "grok-4.5", "input": "summarize", "stream": True},
        ):
          pass
      finally:
        await provider.close_client(client)

    task = asyncio.create_task(collect_with_close())
    await transport.accepted.wait()
    task.cancel()
    cancellation_processed = asyncio.Event()
    asyncio.get_running_loop().call_soon(cancellation_processed.set)
    await cancellation_processed.wait()
    assert not task.done()
    assert not provider.close_started.is_set()
    assert not transport.closed.is_set()
    transport.release_response.set()
    with pytest.raises(asyncio.CancelledError):
      await task
    assert load_xai_token_record(store)["refresh_token"] == "refresh-2"
    assert provider.close_started.is_set()
    assert transport.closed.is_set()
    assert transport.aborted is False

    next_posts: list[str] = []

    def next_handler(request: httpx.Request) -> httpx.Response:
      next_posts.append(_token_form(request)["refresh_token"][0])
      return httpx.Response(
        200,
        json={"access_token": "access-3", "refresh_token": "refresh-3", "expires_in": 3600},
      )

    settings = resolve_xai_oauth_settings({"auth_store_path": str(store)})
    async with httpx.AsyncClient(transport=httpx.MockTransport(next_handler)) as client:
      await refresh_xai_oauth_token(
        load_xai_token_record(store),
        settings=settings,
        client=client,
        force=True,
      )
    assert next_posts == ["refresh-2"]
    assert "refresh-1" not in next_posts
    assert load_xai_token_record(store)["refresh_token"] == "refresh-3"

  _run(run())


def test_cancellation_during_rotated_save_finishes_safe_commit(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  store = tmp_path / "oauth.json"
  original = _record()
  save_xai_token_record(store, original)
  settings = resolve_xai_oauth_settings({"auth_store_path": str(store)})
  real_save = xai_oauth.save_xai_token_record
  refresh_caller: asyncio.Task | None = None

  def cancel_during_rotated_save(path: Path, record: dict) -> None:
    if record.get("refresh_token") == "refresh-2":
      assert refresh_caller is not None
      refresh_caller.cancel()
    real_save(path, record)

  monkeypatch.setattr(xai_oauth, "save_xai_token_record", cancel_during_rotated_save)

  async def handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
      200,
      json={"access_token": "access-2", "refresh_token": "refresh-2", "expires_in": 3600},
    )

  async def run() -> None:
    nonlocal refresh_caller

    async def invoke(client: httpx.AsyncClient) -> None:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      refresh_caller = asyncio.create_task(invoke(client))
      with pytest.raises(asyncio.CancelledError):
        await refresh_caller

  _run(run())
  stored = load_xai_token_record(store)
  assert stored["refresh_token"] == "refresh-2"
  assert "refresh_pending" not in stored


def test_cancellation_during_presend_cleanup_never_leaves_clear_but_consumed_state(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  store = tmp_path / "oauth.json"
  original = _record()
  save_xai_token_record(store, original)
  settings = resolve_xai_oauth_settings({"auth_store_path": str(store)})
  real_save = xai_oauth.save_xai_token_record
  refresh_caller: asyncio.Task | None = None
  writes = 0

  def cancel_during_clear(path: Path, record: dict) -> None:
    nonlocal writes
    writes += 1
    if writes == 2:
      assert "refresh_pending" not in record
      assert refresh_caller is not None
      refresh_caller.cancel()
    real_save(path, record)

  monkeypatch.setattr(xai_oauth, "save_xai_token_record", cancel_during_clear)

  async def handler(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("not sent", request=request)

  async def run() -> None:
    nonlocal refresh_caller

    async def invoke(client: httpx.AsyncClient) -> None:
      await refresh_xai_oauth_token(original, settings=settings, client=client, force=True)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
      refresh_caller = asyncio.create_task(invoke(client))
      with pytest.raises(asyncio.CancelledError):
        await refresh_caller

  _run(run())
  assert load_xai_token_record(store) == original
  assert not xai_oauth._STORE_REFRESH_CONSUMED


def test_save_failure_after_cancelled_accepted_post_is_retrieved_and_logged(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  caplog: pytest.LogCaptureFixture,
) -> None:
  store = tmp_path / "oauth.json"
  original = _record()
  save_xai_token_record(store, original)
  settings = resolve_xai_oauth_settings({"auth_store_path": str(store)})

  async def run() -> tuple[list[dict], list[str]]:
    transport = _GatedRefreshTransport()
    client = httpx.AsyncClient(transport=transport)
    loop = asyncio.get_running_loop()
    unhandled: list[dict] = []
    prior_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    real_save = xai_oauth.save_xai_token_record

    def fail_save(path: Path, record: dict) -> None:
      if record.get("refresh_token") == "refresh-2":
        raise OSError("durable save failed")
      real_save(path, record)

    monkeypatch.setattr(xai_oauth, "save_xai_token_record", fail_save)
    try:
      task = asyncio.create_task(
        refresh_xai_oauth_token(
          original,
          settings=settings,
          client=client,
          force=True,
        )
      )
      await transport.first_posted.wait()
      task.cancel()
      cancellation_processed = asyncio.Event()
      asyncio.get_running_loop().call_soon(cancellation_processed.set)
      await cancellation_processed.wait()
      assert not task.done()
      transport.release_first.set()
      with pytest.raises(asyncio.CancelledError):
        await task
      return unhandled, transport.posts
    finally:
      loop.set_exception_handler(prior_handler)
      await client.aclose()

  caplog.set_level(logging.ERROR, logger=xai_oauth.__name__)
  unhandled, posts = _run(run())
  assert posts == ["refresh-1"]
  stored = load_xai_token_record(store)
  assert stored["refresh_token"] == "refresh-1"
  assert stored["refresh_pending"] is True
  assert "durable save failed" in caplog.text
  assert "token store may be stale" in caplog.text
  assert not any("Task exception was never retrieved" in str(item) for item in unhandled)


def test_cancellation_while_waiting_for_store_flock_releases_fd(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  store = tmp_path / "oauth.json"

  async def run() -> None:
    poll_entered = asyncio.Event()
    never_release = asyncio.Event()

    async def gated_sleep(_delay: float) -> None:
      poll_entered.set()
      await never_release.wait()

    async with _store_lock(store):
      fd_count = len(os.listdir("/dev/fd"))
      monkeypatch.setattr(xai_oauth.asyncio, "sleep", gated_sleep)

      async def wait_for_lock() -> None:
        async with _store_lock(store):
          raise AssertionError("cancelled waiter unexpectedly acquired lock")

      waiter = asyncio.create_task(wait_for_lock())
      await poll_entered.wait()
      waiter.cancel()
      with pytest.raises(asyncio.CancelledError):
        await waiter
      assert len(os.listdir("/dev/fd")) == fd_count

    async with _store_lock(store):
      pass

  _run(run())


def test_refresh_500_keeps_store_quarantined_and_lock_reacquirable(tmp_path: Path) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  original = _record()
  save_xai_token_record(settings.store_path, original)

  async def run() -> None:
    transport = _GatedRefreshTransport(response_status=500)
    transport.release_first.set()
    async with httpx.AsyncClient(transport=transport) as client:
      with pytest.raises(RuntimeError, match=r"xAI OAuth refresh failed \(500\)"):
        await refresh_xai_oauth_token(
          original,
          settings=settings,
          client=client,
          force=True,
        )
    async with _store_lock(settings.store_path):
      pass

  _run(run())
  stored = load_xai_token_record(settings.store_path)
  assert stored["refresh_token"] == "refresh-1"
  assert stored["refresh_pending"] is True


def test_login_and_refresh_saves_are_serialized_without_torn_write(tmp_path: Path) -> None:
  store = tmp_path / "oauth.json"
  original = _record()
  save_xai_token_record(store, original)
  settings = resolve_xai_oauth_settings({"auth_store_path": str(store)})

  class LoginRefreshTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
      self.refresh_posted = asyncio.Event()
      self.release_refresh = asyncio.Event()
      self.login_token_issued = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
      if request.method == "GET":
        return httpx.Response(
          200,
          json={
            "device_authorization_endpoint": "https://auth.x.ai/oauth2/device/code",
            "token_endpoint": "https://auth.x.ai/oauth2/token",
          },
        )
      form = _token_form(request)
      if request.url.path.endswith("/device/code"):
        return httpx.Response(
          200,
          json={
            "device_code": "device-1",
            "user_code": "ABCD",
            "verification_uri": "https://accounts.x.ai/device",
            "expires_in": 900,
            "interval": 1,
          },
        )
      if form.get("grant_type") == ["refresh_token"]:
        self.refresh_posted.set()
        await self.release_refresh.wait()
        return httpx.Response(
          200,
          json={"access_token": "access-refresh", "refresh_token": "refresh-2", "expires_in": 3600},
        )
      self.login_token_issued.set()
      return httpx.Response(
        200,
        json={"access_token": "access-login", "refresh_token": "refresh-login", "expires_in": 3600},
      )

  async def run() -> None:
    transport = LoginRefreshTransport()
    async with (
      httpx.AsyncClient(transport=transport) as refresh_client,
      httpx.AsyncClient(transport=transport) as login_client,
    ):
      refresh = asyncio.create_task(
        refresh_xai_oauth_token(
          original,
          settings=settings,
          client=refresh_client,
          force=True,
        )
      )
      await transport.refresh_posted.wait()
      login = asyncio.create_task(
        login_xai_device_code(
          config={"auth_store_path": str(store)},
          client=login_client,
        )
      )
      await transport.login_token_issued.wait()
      pending = load_xai_token_record(store)
      assert pending["refresh_token"] == original["refresh_token"]
      assert pending["refresh_pending"] is True
      transport.release_refresh.set()
      await asyncio.gather(refresh, login)

  _run(run())
  final = load_xai_token_record(store)
  assert final["access_token"] == "access-login"
  assert final["refresh_token"] == "refresh-login"
  assert _record_is_valid(final, settings)


def test_second_refresh_cannot_post_until_first_save_releases_lock(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  settings = resolve_xai_oauth_settings({"auth_store_path": str(tmp_path / "oauth.json")})
  initial = _record()
  save_xai_token_record(settings.store_path, initial)

  async def run() -> None:
    transport = _GatedRefreshTransport()
    original_store_lock = xai_oauth._store_lock
    lock_attempts = 0
    second_lock_attempted = asyncio.Event()

    @asynccontextmanager
    async def observed_store_lock(path: Path):
      nonlocal lock_attempts
      lock_attempts += 1
      if lock_attempts == 2:
        second_lock_attempted.set()
      async with original_store_lock(path):
        yield

    monkeypatch.setattr(xai_oauth, "_store_lock", observed_store_lock)
    async with (
      httpx.AsyncClient(transport=transport) as first_client,
      httpx.AsyncClient(transport=transport) as second_client,
    ):
      first = asyncio.create_task(
        refresh_xai_oauth_token(initial, settings=settings, client=first_client, force=True)
      )
      await transport.first_posted.wait()
      second_entered = asyncio.Event()

      async def second_refresh() -> None:
        second_entered.set()
        await refresh_xai_oauth_token(
          initial,
          settings=settings,
          client=second_client,
          force=True,
        )

      second = asyncio.create_task(second_refresh())
      await second_entered.wait()
      await second_lock_attempted.wait()
      assert transport.posts == ["refresh-1"]
      transport.release_first.set()
      await asyncio.gather(first, second)
      assert transport.posts == ["refresh-1", "refresh-2"]

  _run(run())


def test_cross_process_refreshers_capture_same_record_but_never_post_same_token(
  tmp_path: Path,
) -> None:
  store = tmp_path / "oauth.json"
  settings = resolve_xai_oauth_settings({"auth_store_path": str(store)})
  save_xai_token_record(store, _record(expires_at=time.time() + 30))
  server_state = {
    "current": "refresh-1",
    "counter": 1,
    "posts": [],
  }
  server_lock = threading.Lock()

  class CountingHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
      length = int(self.headers.get("Content-Length", "0"))
      form = parse_qs(self.rfile.read(length).decode())
      posted = form["refresh_token"][0]
      with server_lock:
        server_state["posts"].append(posted)
        if posted != server_state["current"]:
          status = 400
          payload = {"error": "invalid_grant"}
        else:
          server_state["counter"] += 1
          server_state["current"] = f"refresh-{server_state['counter']}"
          status = 200
          payload = {
            "access_token": f"access-{server_state['counter']}",
            "refresh_token": server_state["current"],
            "expires_in": 30,
          }
      body = json.dumps(payload).encode()
      self.send_response(status)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
      return

  server = ThreadingHTTPServer(("127.0.0.1", 0), CountingHandler)
  server_thread = threading.Thread(target=server.serve_forever, daemon=True)
  server_thread.start()
  context = multiprocessing.get_context("spawn")
  barrier = context.Barrier(2)
  captured = context.Queue()
  server_url = f"http://127.0.0.1:{server.server_port}/token"
  workers = [
    context.Process(
      target=_cross_process_refresh_worker,
      args=(str(store), server_url, barrier, captured),
    )
    for _ in range(2)
  ]
  try:
    for worker in workers:
      worker.start()
    captured_tokens = [captured.get(timeout=10), captured.get(timeout=10)]
    for worker in workers:
      worker.join(timeout=15)
    assert captured_tokens == ["refresh-1", "refresh-1"]
    assert [worker.exitcode for worker in workers] == [0, 0]
  finally:
    for worker in workers:
      if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
    server.shutdown()
    server.server_close()
    server_thread.join(timeout=5)

  assert server_state["posts"] == ["refresh-1", "refresh-2"]
  assert len(server_state["posts"]) == len(set(server_state["posts"]))
  final = load_xai_token_record(store)
  assert final["refresh_token"] == "refresh-3"
  assert _record_is_valid(final, settings)
