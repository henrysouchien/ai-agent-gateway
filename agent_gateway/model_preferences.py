"""Durable account-wide model preferences; preferences are never authority."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3
from threading import RLock
import time
from typing import Literal
from uuid import UUID

from model_authority.binding import ModelSelectionIntent
from model_authority.capabilities import CAPABILITY_IDS


@dataclass(frozen=True, slots=True)
class ModelPresetReference:
  name: str
  policy_revision: str
  updated_at: int


class StoredPreferenceError(ValueError):
  """A stored preference or preset column whose SQLite type is not the schema's.

  ``scope`` names which actor-visible row it is (``preference`` with its
  ``capability_id``, ``user_preset`` or ``firm_preset``) so the HTTP boundary
  can name the repair that row's owner has. The stored bytes are not changed.
  """

  def __init__(
    self,
    message: str,
    *,
    column: str,
    scope: Literal["preference", "user_preset", "firm_preset"],
    capability_id: str | None = None,
  ) -> None:
    super().__init__(message)
    self.column = column
    self.scope = scope
    self.capability_id = capability_id


@dataclass(frozen=True, slots=True)
class _StoredRow:
  """One stored row parsed once into typed columns.

  SQLite affinity lets a BLOB or number sit in a TEXT column, and a BLOB in an
  INTEGER column; neither is ever a key, a name or a timestamp.
  """

  row: sqlite3.Row
  scope: Literal["preference", "user_preset", "firm_preset"]
  capability_id: str | None = None

  def _refuse(self, column: str, expected: str, value: object) -> StoredPreferenceError:
    return StoredPreferenceError(
      f"stored preference column {column} must be {expected}, got {type(value).__name__}",
      column=column,
      scope=self.scope,
      capability_id=self.capability_id,
    )

  def text(self, column: str) -> str:
    value = self.row[column]
    if type(value) is not str:
      raise self._refuse(column, "text", value)
    return value

  def optional_text(self, column: str) -> str | None:
    return None if self.row[column] is None else self.text(column)

  def integer(self, column: str) -> int:
    value = self.row[column]
    if type(value) is not int:
      raise self._refuse(column, "an integer", value)
    return value


class ModelPreferenceStore:
  """SQLite preference store keyed by tenant, account, and capability."""

  def __init__(
    self,
    path: str | Path,
    *,
    known_preset_names: frozenset[str] = frozenset(),
  ) -> None:
    self.path = Path(path).expanduser().resolve(strict=False)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = RLock()
    # Supplied by the policy owner; references never contain custom role maps.
    self.known_preset_names = frozenset(known_preset_names)
    # ``with connection`` scopes the transaction only; ``closing`` releases
    # the sqlite handle deterministically instead of at garbage collection.
    with closing(self._connect()) as connection, connection:
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
      connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_model_presets (
          tenant_id TEXT NOT NULL,
          actor_id TEXT NOT NULL,
          name TEXT NOT NULL,
          policy_revision TEXT NOT NULL,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY (tenant_id, actor_id)
        )
        """
      )
      connection.execute(
        """
        CREATE TABLE IF NOT EXISTS firm_model_presets (
          commercial_account_id TEXT PRIMARY KEY NOT NULL,
          name TEXT NOT NULL,
          policy_revision TEXT NOT NULL,
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
    with self._lock, closing(self._connect()) as connection, connection:
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
    stored = _StoredRow(row, "preference", capability)
    return ModelSelectionIntent(
      model_key=stored.text("model_key"),
      effort=stored.optional_text("effort"),
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
    with self._lock, closing(self._connect()) as connection, connection:
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
    with self._lock, closing(self._connect()) as connection, connection:
      cursor = connection.execute(
        """
        DELETE FROM model_preferences
        WHERE tenant_id = ? AND actor_id = ? AND capability_id = ?
        """,
        (tenant, actor, capability),
      )
      deleted = cursor.rowcount > 0
    return deleted

  @staticmethod
  def _preset_identity(tenant_id: str, actor_id: str) -> tuple[str, str]:
    tenant = str(tenant_id or "").strip()
    actor = str(actor_id or "").strip()
    if not tenant or not actor:
      raise ValueError("preference identity requires tenant_id and actor_id")
    return tenant, actor

  @staticmethod
  def _preset_reference(
    row: sqlite3.Row | None,
    scope: Literal["user_preset", "firm_preset"],
  ) -> ModelPresetReference | None:
    if row is None:
      return None
    stored = _StoredRow(row, scope)
    return ModelPresetReference(
      name=stored.text("name"),
      policy_revision=stored.text("policy_revision"),
      updated_at=stored.integer("updated_at"),
    )

  def get_user_preset(
    self, *, tenant_id: str, actor_id: str,
  ) -> ModelPresetReference | None:
    identity = self._preset_identity(tenant_id, actor_id)
    with self._lock, closing(self._connect()) as connection, connection:
      row = connection.execute(
        """
        SELECT name, policy_revision, updated_at FROM user_model_presets
        WHERE tenant_id = ? AND actor_id = ?
        """,
        identity,
      ).fetchone()
    return self._preset_reference(row, "user_preset")

  def put_user_preset(
    self, *, tenant_id: str, actor_id: str, name: str, policy_revision: str,
  ) -> ModelPresetReference:
    identity = self._preset_identity(tenant_id, actor_id)
    reference = ModelPresetReference(name, policy_revision, int(time.time()))
    with self._lock, closing(self._connect()) as connection, connection:
      connection.execute(
        """
        INSERT INTO user_model_presets (
          tenant_id, actor_id, name, policy_revision, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (tenant_id, actor_id) DO UPDATE SET
          name = excluded.name,
          policy_revision = excluded.policy_revision,
          updated_at = excluded.updated_at
        """,
        (*identity, reference.name, reference.policy_revision, reference.updated_at),
      )
    return reference

  def delete_user_preset(self, *, tenant_id: str, actor_id: str) -> bool:
    identity = self._preset_identity(tenant_id, actor_id)
    with self._lock, closing(self._connect()) as connection, connection:
      cursor = connection.execute(
        "DELETE FROM user_model_presets WHERE tenant_id = ? AND actor_id = ?",
        identity,
      )
      return cursor.rowcount > 0

  @staticmethod
  def _firm_identity(commercial_account_id: UUID) -> str:
    # The caller obtains this UUID from a verified commercial claim, not a
    # tenant id or an authenticated tenant role.
    if not isinstance(commercial_account_id, UUID):
      raise ValueError("firm preset identity requires a commercial_account_id UUID")
    return str(commercial_account_id)

  def get_firm_preset(
    self, *, commercial_account_id: UUID,
  ) -> ModelPresetReference | None:
    identity = self._firm_identity(commercial_account_id)
    with self._lock, closing(self._connect()) as connection, connection:
      row = connection.execute(
        """
        SELECT name, policy_revision, updated_at FROM firm_model_presets
        WHERE commercial_account_id = ?
        """,
        (identity,),
      ).fetchone()
    return self._preset_reference(row, "firm_preset")

  def put_firm_preset(
    self, *, commercial_account_id: UUID, name: str, policy_revision: str,
  ) -> ModelPresetReference:
    """The commercial-account owner authorizes writes before calling this store."""
    identity = self._firm_identity(commercial_account_id)
    reference = ModelPresetReference(name, policy_revision, int(time.time()))
    with self._lock, closing(self._connect()) as connection, connection:
      connection.execute(
        """
        INSERT INTO firm_model_presets (
          commercial_account_id, name, policy_revision, updated_at
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT (commercial_account_id) DO UPDATE SET
          name = excluded.name,
          policy_revision = excluded.policy_revision,
          updated_at = excluded.updated_at
        """,
        (identity, reference.name, reference.policy_revision, reference.updated_at),
      )
    return reference

  def delete_firm_preset(self, *, commercial_account_id: UUID) -> bool:
    identity = self._firm_identity(commercial_account_id)
    with self._lock, closing(self._connect()) as connection, connection:
      cursor = connection.execute(
        "DELETE FROM firm_model_presets WHERE commercial_account_id = ?",
        (identity,),
      )
      return cursor.rowcount > 0


__all__ = ["ModelPreferenceStore", "ModelPresetReference", "StoredPreferenceError"]
