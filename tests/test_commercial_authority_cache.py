from uuid import uuid4
import json
from threading import Event, Thread

from agent_gateway.commercial_authority_cache import (
  CommercialAuthorityInvalidation,
  CommercialAuthoritySnapshot,
  CommercialAuthorityStateCache,
)


def test_push_evicts_account_and_rejects_stale_loader_revision() -> None:
  context_id = uuid4()
  token_id = uuid4()
  calls = []

  def load(identity):
    calls.append(identity)
    return CommercialAuthoritySnapshot(
      context_id=identity,
      active=True,
      entitlement_revision=42,
      commercial_account_id=7,
      token_id=token_id,
    )

  cache = CommercialAuthorityStateCache(load)
  assert cache.resolve_context_state(context_id).active
  assert cache.resolve_context_state(context_id).active
  assert calls == [context_id]

  cache.apply_invalidation(CommercialAuthorityInvalidation(
    kind="entitlement",
    commercial_account_id=7,
    entitlement_revision=43,
  ))
  state = cache.resolve_context_state(context_id)
  assert not state.active
  assert state.entitlement_revision == 43
  assert calls == [context_id, context_id]


def test_context_revocation_is_immediate_and_never_reloaded() -> None:
  context_id = uuid4()
  calls = []

  def load(identity):
    calls.append(identity)
    return CommercialAuthoritySnapshot(identity, True, 9, 7, uuid4())

  cache = CommercialAuthorityStateCache(load)
  assert cache.resolve_context_state(context_id).active
  cache.apply_invalidation(CommercialAuthorityInvalidation(
    kind="context",
    commercial_account_id=7,
    entitlement_revision=9,
    context_id=context_id,
  ))
  assert not cache.resolve_context_state(context_id).active
  assert calls == [context_id]


def test_token_push_evicts_only_matching_cached_authority() -> None:
  first, second = uuid4(), uuid4()
  first_token, second_token = uuid4(), uuid4()
  calls = []

  def load(identity):
    calls.append(identity)
    return CommercialAuthoritySnapshot(
      identity, True, 5, 7 if identity == first else 8,
      first_token if identity == first else second_token,
    )

  cache = CommercialAuthorityStateCache(load)
  cache.resolve_context_state(first)
  cache.resolve_context_state(second)
  cache.apply_invalidation(CommercialAuthorityInvalidation(
    kind="token", commercial_account_id=7, entitlement_revision=5,
    token_id=first_token,
  ))
  cache.resolve_context_state(first)
  cache.resolve_context_state(second)
  assert calls == [first, second, first]


def test_notification_parser_accepts_all_risk_channels_and_rejects_bad_payload() -> None:
  context_id, token_id = uuid4(), uuid4()
  cache = CommercialAuthorityStateCache(lambda identity: CommercialAuthoritySnapshot(
    identity, True, 4, 7, token_id
  ))
  cache.resolve_context_state(context_id)
  cache.apply_notification(
    "commercial_execution_context_invalidation",
    json.dumps({
      "commercial_account_id": 7,
      "context_id": str(context_id),
      "token_id": str(token_id),
      "entitlement_revision": 4,
    }),
  )
  assert not cache.resolve_context_state(context_id).active

  for channel, payload in (
    ("commercial_mcp_token_invalidation", {
      "commercial_account_id": 7, "token_id": str(token_id),
      "entitlement_revision": 5,
    }),
    ("commercial_entitlement_invalidation", {
      "commercial_account_id": 7, "revision": 6,
    }),
  ):
    cache.apply_notification(channel, json.dumps(payload))

  try:
    cache.apply_notification("unknown", "{}")
  except ValueError:
    pass
  else:
    raise AssertionError("unknown invalidation channel must fail closed")


def test_invalidation_racing_loader_never_returns_stale_active() -> None:
  context_id, token_id = uuid4(), uuid4()
  loading, release = Event(), Event()
  result = []

  def load(identity):
    loading.set()
    assert release.wait(timeout=5)
    return CommercialAuthoritySnapshot(identity, True, 5, 7, token_id)

  cache = CommercialAuthorityStateCache(load)
  worker = Thread(target=lambda: result.append(cache.resolve_context_state(context_id)))
  worker.start()
  assert loading.wait(timeout=5)
  cache.apply_invalidation(CommercialAuthorityInvalidation(
    kind="context", commercial_account_id=7, entitlement_revision=5,
    context_id=context_id,
  ))
  release.set()
  worker.join(timeout=5)
  assert not worker.is_alive()
  assert result and not result[0].active


def test_all_cache_indexes_are_bounded() -> None:
  cache = CommercialAuthorityStateCache(
    lambda identity: CommercialAuthoritySnapshot(identity, True, 1, 1, uuid4()),
    max_entries=2,
  )
  for account_id in range(1, 5):
    cache.apply_invalidation(CommercialAuthorityInvalidation(
      kind="context", commercial_account_id=account_id,
      entitlement_revision=1, context_id=uuid4(),
    ))
  assert len(cache._revision_floor_by_account) == 2
  assert len(cache._revoked_contexts) == 2
