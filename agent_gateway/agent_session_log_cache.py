from __future__ import annotations

import threading

ActiveFileIdentity = tuple[int, int, int, int]


class ActiveFileOffsetCache:
  """Seq-to-byte offset cache for the mutable active session-log file."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._seq_offsets: dict[int, int] = {}
    self._max_cached_seq = 0
    self._complete = False
    self._file_identity: ActiveFileIdentity | None = None

  def _refresh_locked(self, active_identity: ActiveFileIdentity | None) -> None:
    if self._file_identity == active_identity:
      return
    self._seq_offsets.clear()
    self._max_cached_seq = 0
    self._complete = False
    self._file_identity = active_identity

  def starting_offset_for_seq(
    self,
    after_seq: int | None,
    *,
    active_identity: ActiveFileIdentity | None,
  ) -> int:
    if after_seq is None or after_seq <= 1:
      return 0
    with self._lock:
      self._refresh_locked(active_identity)
      if after_seq in self._seq_offsets:
        return self._seq_offsets[after_seq]
      if self._complete and after_seq > self._max_cached_seq:
        return active_identity[2] if active_identity is not None else 0
      if self._max_cached_seq > 0 and after_seq > self._max_cached_seq:
        return self._seq_offsets.get(self._max_cached_seq, 0)
    return 0

  def update(
    self,
    *,
    seq: int,
    offset: int,
    active_identity: ActiveFileIdentity | None,
  ) -> None:
    with self._lock:
      self._refresh_locked(active_identity)
      existing = self._seq_offsets.get(seq)
      if existing is None or offset < existing:
        self._seq_offsets[seq] = offset
      if seq > self._max_cached_seq:
        self._max_cached_seq = seq

  def mark_complete(
    self,
    *,
    active_identity: ActiveFileIdentity | None,
    current_identity: ActiveFileIdentity | None,
    file_size: int,
  ) -> None:
    if active_identity is None or current_identity != active_identity or current_identity[2] != file_size:
      return
    with self._lock:
      self._refresh_locked(active_identity)
      self._complete = True

  def clear(self) -> None:
    with self._lock:
      self._seq_offsets.clear()
      self._max_cached_seq = 0
      self._complete = False
      self._file_identity = None


__all__ = ["ActiveFileIdentity", "ActiveFileOffsetCache"]
