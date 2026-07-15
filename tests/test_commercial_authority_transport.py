import asyncio
import hashlib
import hmac
import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from agent_gateway.commercial_authority_cache import CommercialAuthorityStateCache
from agent_gateway.commercial_authority_client import HttpCommercialAuthorityClient
from agent_gateway.commercial_authority_subscriber import CommercialAuthoritySubscriber


def test_client_signs_exact_request_and_parses_token_free_snapshot() -> None:
  secret = b"s" * 32
  context_id = uuid4()

  def post(url, *, content, headers, timeout):
    assert url.endswith("/internal/commercial/live-context")
    assert json.loads(content) == {
      "context_id": str(context_id), "environment": "prod",
    }
    digest = hashlib.sha256(content).hexdigest()
    message = (
      "POST\n/internal/commercial/live-context\n1700000000\n"
      f"{headers['x-hank-request-nonce']}\n{digest}"
    ).encode()
    assert headers["x-hank-request-signature"] == (
      "v1=" + hmac.new(secret, message, hashlib.sha256).hexdigest()
    )
    return httpx.Response(200, json={
      "environment": "prod", "context_id": str(context_id), "active": True,
      "entitlement_revision": 9, "commercial_account_id": 7,
      "mcp_token_id": None,
    }, request=httpx.Request("POST", url))

  client = HttpCommercialAuthorityClient(
    base_url="https://risk.test", environment="prod", key_id="gateway", secret=secret,
    clock=lambda: 1700000000, post=post,
  )
  snapshot = client.load(context_id)
  assert snapshot.active is True
  assert snapshot.entitlement_revision == 9


def test_client_signs_and_parses_current_agreement_terms_revision() -> None:
  secret = b"s" * 32
  agreement_id = uuid4()

  def post(url, *, content, headers, timeout):
    assert url.endswith("/internal/commercial/agreement-terms")
    assert json.loads(content) == {
      "agreement_id": str(agreement_id), "environment": "prod",
    }
    digest = hashlib.sha256(content).hexdigest()
    message = (
      "POST\n/internal/commercial/agreement-terms\n1700000000\n"
      f"{headers['x-hank-request-nonce']}\n{digest}"
    ).encode()
    assert headers["x-hank-request-signature"] == (
      "v1=" + hmac.new(secret, message, hashlib.sha256).hexdigest()
    )
    return httpx.Response(200, json={
      "environment": "prod", "agreement_id": str(agreement_id),
      "current_terms_revision": 4,
    }, request=httpx.Request("POST", url))

  client = HttpCommercialAuthorityClient(
    base_url="https://risk.test", environment="prod", key_id="gateway",
    secret=secret, clock=lambda: 1700000000, post=post,
  )
  assert client.current_agreement_terms_revision(agreement_id) == 4


@pytest.mark.parametrize("revision", [0, -1, True, "4", 1.5])
def test_client_rejects_invalid_agreement_terms_revision(revision) -> None:
  agreement_id = uuid4()

  def post(url, **kwargs):
    return httpx.Response(200, json={
      "environment": "prod", "agreement_id": str(agreement_id),
      "current_terms_revision": revision,
    }, request=httpx.Request("POST", url))

  client = HttpCommercialAuthorityClient(
    base_url="https://risk.test", environment="prod", key_id="gateway",
    secret=b"s" * 32, post=post,
  )
  with pytest.raises(ValueError, match="revision"):
    client.current_agreement_terms_revision(agreement_id)


def test_client_rejects_wrong_agreement_identity() -> None:
  agreement_id = uuid4()

  def post(url, **kwargs):
    return httpx.Response(200, json={
      "environment": "prod", "agreement_id": str(uuid4()),
      "current_terms_revision": 4,
    }, request=httpx.Request("POST", url))

  client = HttpCommercialAuthorityClient(
    base_url="https://risk.test", environment="prod", key_id="gateway",
    secret=b"s" * 32, post=post,
  )
  with pytest.raises(ValueError, match="wrong agreement"):
    client.current_agreement_terms_revision(agreement_id)


@pytest.mark.parametrize("endpoint", ["live_context", "agreement_terms"])
def test_client_rejects_wrong_authority_environment(endpoint) -> None:
  identity = uuid4()

  def post(url, **kwargs):
    if endpoint == "live_context":
      document = {
        "environment": "staging", "context_id": str(identity), "active": True,
        "entitlement_revision": 9, "commercial_account_id": 7,
        "mcp_token_id": None,
      }
    else:
      document = {
        "environment": "staging", "agreement_id": str(identity),
        "current_terms_revision": 4,
      }
    return httpx.Response(
      200, json=document, request=httpx.Request("POST", url)
    )

  client = HttpCommercialAuthorityClient(
    base_url="https://risk.test", environment="prod", key_id="gateway",
    secret=b"s" * 32, post=post,
  )
  with pytest.raises(ValueError, match="environment"):
    if endpoint == "live_context":
      client.load(identity)
    else:
      client.current_agreement_terms_revision(identity)


def test_startup_catch_up_consumes_every_page_before_returning(tmp_path: Path) -> None:
  context_id = uuid4()
  pages = iter([
    {"events": [{
      "sequence_id": 2, "kind": "context", "commercial_account_id": 7,
      "entitlement_revision": 3, "context_id": str(context_id), "token_id": None,
      "environment": "dev", "occurred_at": "2026-07-12T12:00:00Z",
    }], "next_sequence": 2, "high_water_sequence": 4},
    {"events": [{
      "sequence_id": 4, "kind": "entitlement", "commercial_account_id": 7,
      "entitlement_revision": 4, "context_id": None, "token_id": None,
      "environment": "dev", "occurred_at": "2026-07-12T12:00:01Z",
    }], "next_sequence": 4, "high_water_sequence": 4},
  ])

  class Client:
    def fetch(self, cursor):
      return next(pages)

  cache = CommercialAuthorityStateCache(lambda _: pytest.fail("loader must not run"))
  path = tmp_path / "cursor.json"
  asyncio.run(CommercialAuthoritySubscriber(
    client=Client(), cache=cache, cursor_path=path,
  ).catch_up())
  assert json.loads(path.read_text()) == {"sequence": 4}
  assert cache.resolve_context_state(context_id).active is False
  assert path.stat().st_mode & 0o077 == 0


def test_startup_catch_up_rejects_non_progressing_feed(tmp_path: Path) -> None:
  class Client:
    def fetch(self, cursor):
      return {"events": [], "next_sequence": cursor, "high_water_sequence": 3}

  cache = CommercialAuthorityStateCache(lambda _: pytest.fail("loader must not run"))
  with pytest.raises(ValueError, match="no catch-up progress"):
    asyncio.run(CommercialAuthoritySubscriber(
      client=Client(), cache=cache, cursor_path=tmp_path / "cursor.json",
    ).catch_up())


def test_subscriber_health_tracks_success_failure_and_staleness(tmp_path: Path) -> None:
  now = [10.0]

  class Client:
    fail = False

    def fetch(self, cursor):
      if self.fail:
        raise RuntimeError("authority unavailable")
      return {"events": [], "next_sequence": cursor, "high_water_sequence": cursor}

  client = Client()
  cache = CommercialAuthorityStateCache(lambda _: pytest.fail("loader must not run"))
  subscriber = CommercialAuthoritySubscriber(
    client=client,
    cache=cache,
    cursor_path=tmp_path / "cursor.json",
    monotonic=lambda: now[0],
  )
  asyncio.run(subscriber.catch_up())
  assert subscriber.health(max_staleness_seconds=5.0)["ok"] is True

  now[0] = 16.0
  stale = subscriber.health(max_staleness_seconds=5.0)
  assert stale["ok"] is False
  assert stale["last_success_age_seconds"] == 6.0

  async def fail_once() -> None:
    client.fail = True
    stop = asyncio.Event()

    async def stop_after_failure(seen_stop, _seconds):
      seen_stop.set()

    subscriber._wait = stop_after_failure
    await subscriber.run_forever(stop)

  asyncio.run(fail_once())
  failed = subscriber.health(max_staleness_seconds=30.0)
  assert failed["ok"] is False
  assert failed["consecutive_failures"] == 1
  assert failed["last_error_type"] == "RuntimeError"


def test_subscriber_readiness_fails_while_live_feed_is_behind_high_water(
  tmp_path: Path,
) -> None:
  class Client:
    def fetch(self, cursor):
      return {"events": [], "next_sequence": cursor, "high_water_sequence": cursor}

  cache = CommercialAuthorityStateCache(lambda _: pytest.fail("loader must not run"))
  subscriber = CommercialAuthoritySubscriber(
    client=Client(),
    cache=cache,
    cursor_path=tmp_path / "cursor.json",
  )

  async def scenario() -> None:
    await subscriber.catch_up()
    stop = asyncio.Event()
    polls = 0

    async def consume_page(_cursor):
      nonlocal polls
      polls += 1
      if polls == 1:
        return 1, 2
      assert subscriber.health(max_staleness_seconds=30.0)["ok"] is False
      return 2, 2

    async def stop_when_caught_up(seen_stop, _seconds):
      seen_stop.set()

    subscriber._consume_page = consume_page
    subscriber._wait = stop_when_caught_up
    await subscriber.run_forever(stop)

  asyncio.run(scenario())
  assert subscriber.health(max_staleness_seconds=30.0)["ok"] is True


@pytest.mark.parametrize("active,revision,account", [
  ("false", 2, 7), (1, 2, 7), (True, True, 7), (True, 2, True),
])
def test_client_rejects_fail_open_snapshot_types(active, revision, account) -> None:
  context_id = uuid4()
  def post(url, **kwargs):
    return httpx.Response(200, json={
      "environment": "prod", "context_id": str(context_id), "active": active,
      "entitlement_revision": revision, "commercial_account_id": account,
      "mcp_token_id": None,
    }, request=httpx.Request("POST", url))
  client = HttpCommercialAuthorityClient(
    base_url="https://risk.test", environment="prod", key_id="gateway",
    secret=b"s" * 32, post=post,
  )
  with pytest.raises(ValueError):
    client.load(context_id)


@pytest.mark.parametrize("environment,url", [
  ("prod", "http://risk.internal"), ("staging", "http://127.0.0.1:8000"),
  ("dev", "http://risk.internal"), ("dev", "http://localhost.evil.example"),
  ("dev", "http://127.0.0.1.evil.example"),
])
def test_client_rejects_insecure_non_dev_loopback(environment, url) -> None:
  with pytest.raises(ValueError, match="HTTPS"):
    HttpCommercialAuthorityClient(
      base_url=url, environment=environment, key_id="gateway", secret=b"s" * 32,
    )
