"""Durable account-wide model preferences; preferences are never authority."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
from threading import RLock
import time

from .capability_binding import ModelSelectionIntent
from .model_registry import CAPABILITY_IDS


class ModelPreferenceStore:
  """SQLite preference store keyed by tenant, account, and capability."""

  def __init__(self, path: str | Path) -> None:
    self.path = Path(path).expanduser().resolve(strict=False)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = RLock()
    with self._connect() as connection:
      connection.execute("PRAGMA journal_mode=WAL")
      connection.execute("PRAGMA synchronous=FULL")
      connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_preferences (
          tenant_id TEXT NOT NULL,
          actor_id TEXT NOT NULL,
          capability_id TEXT NOT NULL,
          model_key TEXT NOT NULL,
          effort TEXT,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY (tenant_id, actor_id, capability_id)
        )
        """
      )
    os.chmod(self.path, 0o600)

  def _connect(self) -> sqlite3.Connection:
    connection = sqlite3.connect(self.path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection

  @staticmethod
  def _identity(tenant_id: str, actor_id: str, capability_id: str) -> tuple[str, str, str]:
    tenant = str(tenant_id or "").strip()
    actor = str(actor_id or "").strip()
    capability = str(capability_id or "").strip()
    if not tenant or not actor:
      raise ValueError("preference identity requires tenant_id and actor_id")
    if capability not in CAPABILITY_IDS:
      raise ValueError(f"unknown preference capability: {capability!r}")
    return tenant, actor, capability

  def get(
    self,
    *,
    tenant_id: str,
    actor_id: str,
    capability_id: str,
  ) -> ModelSelectionIntent | None:
    tenant, actor, capability = self._identity(
      tenant_id,
      actor_id,
      capability_id,
    )
    with self._lock, self._connect() as connection:
      row = connection.execute(
        """
        SELECT model_key, effort
        FROM model_preferences
        WHERE tenant_id = ? AND actor_id = ? AND capability_id = ?
        """,
        (tenant, actor, capability),
      ).fetchone()
    if row is None:
      return None
    return ModelSelectionIntent(
      model_key=str(row["model_key"]),
      effort=str(row["effort"]) if row["effort"] is not None else None,
      source="saved_preference",
    )

  def put(
    self,
    *,
    tenant_id: str,
    actor_id: str,
    capability_id: str,
    model_key: str,
    effort: str | None,
  ) -> ModelSelectionIntent:
    tenant, actor, capability = self._identity(
      tenant_id,
      actor_id,
      capability_id,
    )
    intent = ModelSelectionIntent(
      model_key=model_key,
      effort=effort,
      source="saved_preference",
    )
    with self._lock, self._connect() as connection:
      connection.execute(
        """
        INSERT INTO model_preferences (
          tenant_id, actor_id, capability_id, model_key, effort, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (tenant_id, actor_id, capability_id) DO UPDATE SET
          model_key = excluded.model_key,
          effort = excluded.effort,
          updated_at = excluded.updated_at
        """,
        (
          tenant,
          actor,
          capability,
          intent.model_key,
          intent.effort,
          int(time.time()),
        ),
      )
    return intent

  def delete(
    self,
    *,
    tenant_id: str,
    actor_id: str,
    capability_id: str,
  ) -> bool:
    tenant, actor, capability = self._identity(
      tenant_id,
      actor_id,
      capability_id,
    )
    with self._lock, self._connect() as connection:
      cursor = connection.execute(
        """
        DELETE FROM model_preferences
        WHERE tenant_id = ? AND actor_id = ? AND capability_id = ?
        """,
        (tenant, actor, capability),
      )
    return cursor.rowcount > 0


__all__ = ["ModelPreferenceStore"]
