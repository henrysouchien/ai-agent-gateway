"""Bounded token-free live-authority cache with pushed invalidation fences."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import time
from threading import RLock
from typing import Callable, Literal
from uuid import UUID

from .commercial_claims import CommercialContextState


@dataclass(frozen=True)
class CommercialAuthoritySnapshot:
  context_id: UUID
  active: bool
  entitlement_revision: int
  commercial_account_id: int | None
  token_id: UUID | None

  def __post_init__(self) -> None:
    if type(self.context_id) is not UUID or (
      self.token_id is not None and type(self.token_id) is not UUID
    ):
      raise ValueError("commercial authority UUID identity is invalid")
    if type(self.active) is not bool:
      raise ValueError("commercial authority active state must be boolean")
    if type(self.entitlement_revision) is not int or self.entitlement_revision < 0:
      raise ValueError("commercial authority revision cannot be negative")
    if self.commercial_account_id is not None and (
      type(self.commercial_account_id) is not int or self.commercial_account_id <= 0
    ):
      raise ValueError("commercial account identity must be positive")


@dataclass(frozen=True)
class CommercialAuthorityInvalidation:
  kind: Literal["context", "token", "entitlement", "agreement", "emergency"]
  commercial_account_id: int
  entitlement_revision: int
  context_id: UUID | None = None
  token_id: UUID | None = None

  def __post_init__(self) -> None:
    if self.kind not in {"context", "token", "entitlement", "agreement", "emergency"}:
      raise ValueError("commercial invalidation kind is invalid")
    if (
      self.context_id is not None and type(self.context_id) is not UUID
    ) or (self.token_id is not None and type(self.token_id) is not UUID):
      raise ValueError("commercial invalidation UUID identity is invalid")
    if (
      type(self.commercial_account_id) is not int
      or type(self.entitlement_revision) is not int
      or self.commercial_account_id <= 0
      or self.entitlement_revision <= 0
    ):
      raise ValueError("commercial invalidation identity is invalid")
    if self.kind == "context" and self.context_id is None:
      raise ValueError("context invalidation requires context identity")
    if self.kind == "token" and self.token_id is None:
      raise ValueError("token invalidation requires token identity")


@dataclass(frozen=True)
class _Cached:
  snapshot: CommercialAuthoritySnapshot
  loaded_at: float


class CommercialAuthorityStateCache:
  def __init__(
    self,
    loader: Callable[[UUID], CommercialAuthoritySnapshot],
    *,
    ttl_seconds: float = 30.0,
    max_entries: int = 10_000,
    monotonic: Callable[[], float] = time.monotonic,
  ) -> None:
    if ttl_seconds <= 0 or max_entries <= 0:
      raise ValueError("commercial authority cache bounds must be positive")
    self._loader = loader
    self._ttl = float(ttl_seconds)
    self._max_entries = int(max_entries)
    self._monotonic = monotonic
    self._entries: OrderedDict[UUID, _Cached] = OrderedDict()
    self._revision_floor_by_account: OrderedDict[int, int] = OrderedDict()
    self._revoked_contexts: OrderedDict[UUID, None] = OrderedDict()
    self._generation = 0
    self._lock = RLock()

  def resolve_context_state(self, context_id: UUID) -> CommercialContextState:
    for _attempt in range(2):
      with self._lock:
        if context_id in self._revoked_contexts:
          return CommercialContextState(active=False, entitlement_revision=0)
        now = self._monotonic()
        cached = self._entries.get(context_id)
        if cached is not None and now - cached.loaded_at < self._ttl:
          floor = self._floor(cached.snapshot.commercial_account_id)
          if cached.snapshot.entitlement_revision >= floor:
            self._entries.move_to_end(context_id)
            return self._state(cached.snapshot)
          self._entries.pop(context_id, None)
        generation = self._generation
      snapshot = self._loader(context_id)
      if snapshot.context_id != context_id:
        raise ValueError("commercial authority loader returned wrong context")
      with self._lock:
        if generation != self._generation:
          continue
        floor = self._floor(snapshot.commercial_account_id)
        if snapshot.entitlement_revision < floor:
          return CommercialContextState(active=False, entitlement_revision=floor)
        self._entries[context_id] = _Cached(snapshot=snapshot, loaded_at=now)
        self._entries.move_to_end(context_id)
        self._trim(self._entries)
        return self._state(snapshot)
    return CommercialContextState(active=False, entitlement_revision=0)

  def apply_invalidation(self, event: CommercialAuthorityInvalidation) -> None:
    with self._lock:
      self._generation += 1
      current = self._revision_floor_by_account.get(event.commercial_account_id, 0)
      self._revision_floor_by_account[event.commercial_account_id] = max(
        current, event.entitlement_revision
      )
      self._revision_floor_by_account.move_to_end(event.commercial_account_id)
      self._trim(self._revision_floor_by_account)
      if event.kind == "context" and event.context_id is not None:
        self._revoked_contexts[event.context_id] = None
        self._revoked_contexts.move_to_end(event.context_id)
        self._trim(self._revoked_contexts)
      for context_id, cached in tuple(self._entries.items()):
        snapshot = cached.snapshot
        if (
          snapshot.commercial_account_id == event.commercial_account_id
          or (event.context_id is not None and context_id == event.context_id)
          or (event.token_id is not None and snapshot.token_id == event.token_id)
        ):
          self._entries.pop(context_id, None)

  def apply_notification(self, channel: str, payload: str) -> None:
    """Validate a risk control-plane notification before applying its fence."""
    kinds = {
      "commercial_execution_context_invalidation": "context",
      "commercial_mcp_token_invalidation": "token",
      "commercial_entitlement_invalidation": "entitlement",
    }
    kind = kinds.get(channel)
    if kind is None:
      raise ValueError("unknown commercial invalidation channel")
    try:
      value = json.loads(payload)
    except json.JSONDecodeError as exc:
      raise ValueError("commercial invalidation payload is invalid") from exc
    if not isinstance(value, dict):
      raise ValueError("commercial invalidation payload must be an object")
    account_id = value.get("commercial_account_id")
    revision = value.get("entitlement_revision", value.get("revision"))
    if isinstance(account_id, bool) or not isinstance(account_id, int):
      raise ValueError("commercial invalidation account is invalid")
    if isinstance(revision, bool) or not isinstance(revision, int):
      raise ValueError("commercial invalidation revision is invalid")
    try:
      context_id = UUID(value["context_id"]) if value.get("context_id") else None
      token_id = UUID(value["token_id"]) if value.get("token_id") else None
    except (TypeError, ValueError) as exc:
      raise ValueError("commercial invalidation UUID is invalid") from exc
    self.apply_invalidation(CommercialAuthorityInvalidation(
      kind=kind,  # type: ignore[arg-type]
      commercial_account_id=account_id,
      entitlement_revision=revision,
      context_id=context_id,
      token_id=token_id,
    ))

  def _floor(self, account_id: int | None) -> int:
    return self._revision_floor_by_account.get(account_id, 0) if account_id else 0

  def _trim(self, values: OrderedDict) -> None:
    while len(values) > self._max_entries:
      values.popitem(last=False)

  @staticmethod
  def _state(snapshot: CommercialAuthoritySnapshot) -> CommercialContextState:
    return CommercialContextState(
      active=snapshot.active,
      entitlement_revision=snapshot.entitlement_revision,
    )


__all__ = [
  "CommercialAuthorityInvalidation",
  "CommercialAuthoritySnapshot",
  "CommercialAuthorityStateCache",
]
