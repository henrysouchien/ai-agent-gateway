"""Durable per-user standing approval preference; the policy's one input."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
from threading import RLock
import time
from typing import cast

from .approval_policy import (
  APPROVAL_PREFERENCES,
  DEFAULT_APPROVAL_PREFERENCE,
  ApprovalPreference,
)


@dataclass(frozen=True, kw_only=True)
class StandingApprovalPreference:
  """One user's answer, and whether it was stored or is the product default."""

  user_id: str
  preference: ApprovalPreference
  updated_at: int | None
  source: str


class ApprovalPreferenceStore:
  """SQLite preference store keyed by the user the approval belongs to.

  Per-user and account-wide, like the model preference the taskpane ``/model``
  writes: a session is not the owner, because the answer has to survive a new
  Excel pane, a new CLI session and every autonomous run the user launches.
  """

  def __init__(self, path: str | Path) -> None:
    self.path = Path(path).expanduser().resolve(strict=False)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = RLock()
    # ``with connection`` scopes the transaction only; ``closing`` releases
    # the sqlite handle deterministically instead of at garbage collection.
    with closing(self._connect()) as connection, connection:
      connection.execute("PRAGMA journal_mode=WAL")
      connection.execute("PRAGMA synchronous=FULL")
      connection.execute(
        """
        CREATE TABLE IF NOT EXISTS approval_preferences (
          user_id TEXT NOT NULL PRIMARY KEY,
          preference TEXT NOT NULL,
          updated_at INTEGER NOT NULL
        )
        """
      )
    os.chmod(self.path, 0o600)

  def _connect(self) -> sqlite3.Connection:
    connection = sqlite3.connect(self.path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection

  @staticmethod
  def _user(user_id: str) -> str:
    user = str(user_id or "").strip()
    if not user:
      raise ValueError("approval preference identity requires user_id")
    return user

  def get(self, *, user_id: str) -> StandingApprovalPreference:
    """Return this user's standing answer, or the product default."""

    user = self._user(user_id)
    with self._lock, closing(self._connect()) as connection, connection:
      row = connection.execute(
        """
        SELECT preference, updated_at
        FROM approval_preferences
        WHERE user_id = ?
        """,
        (user,),
      ).fetchone()
    if row is None:
      return StandingApprovalPreference(
        user_id=user,
        preference=DEFAULT_APPROVAL_PREFERENCE,
        updated_at=None,
        source="default",
      )
    # The row was written by ``put``, which is this value's admission boundary;
    # reading it back never re-judges what this store itself produced.
    return StandingApprovalPreference(
      user_id=user,
      preference=cast(ApprovalPreference, str(row["preference"])),
      updated_at=int(row["updated_at"]),
      source="stored",
    )

  def put(
    self,
    *,
    user_id: str,
    preference: str,
  ) -> StandingApprovalPreference:
    """Store one of the two supported answers for this user."""

    user = self._user(user_id)
    stored = admit_approval_preference(preference)
    updated_at = int(time.time())
    with self._lock, closing(self._connect()) as connection, connection:
      connection.execute(
        """
        INSERT INTO approval_preferences (user_id, preference, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT (user_id) DO UPDATE SET
          preference = excluded.preference,
          updated_at = excluded.updated_at
        """,
        (user, stored, updated_at),
      )
    return StandingApprovalPreference(
      user_id=user,
      preference=stored,
      updated_at=updated_at,
      source="stored",
    )


def admit_approval_preference(value: object) -> ApprovalPreference:
  """Admit a caller's requested answer: the two supported states, nothing else."""

  text = str(value or "").strip()
  if text in APPROVAL_PREFERENCES:
    return cast(ApprovalPreference, text)
  raise ValueError(f"unsupported approval preference: {text!r}")


__all__ = [
  "ApprovalPreferenceStore",
  "StandingApprovalPreference",
  "admit_approval_preference",
]
