from __future__ import annotations

import pytest

from agent_gateway.model_preferences import ModelPreferenceStore


def test_preferences_are_account_wide_per_capability(tmp_path) -> None:
  store = ModelPreferenceStore(tmp_path / "preferences.sqlite3")
  stored = store.put(
    tenant_id="tenant-a",
    actor_id="user-7",
    capability_id="session.driver",
    model_key="anthropic.claude-opus-5",
    effort="high",
  )

  assert stored.source == "saved_preference"
  assert store.get(
    tenant_id="tenant-a",
    actor_id="user-7",
    capability_id="session.driver",
  ) == stored
  assert store.get(
    tenant_id="tenant-a",
    actor_id="another-user",
    capability_id="session.driver",
  ) is None
  assert store.get(
    tenant_id="tenant-a",
    actor_id="user-7",
    capability_id="plan.author",
  ) is None


def test_delete_preserves_other_account_preferences(tmp_path) -> None:
  store = ModelPreferenceStore(tmp_path / "preferences.sqlite3")
  for actor in ("user-7", "user-8"):
    store.put(
      tenant_id="tenant-a",
      actor_id=actor,
      capability_id="session.driver",
      model_key="anthropic.claude-opus-5",
      effort=None,
    )

  assert store.delete(
    tenant_id="tenant-a",
    actor_id="user-7",
    capability_id="session.driver",
  )
  assert store.get(
    tenant_id="tenant-a",
    actor_id="user-7",
    capability_id="session.driver",
  ) is None
  assert store.get(
    tenant_id="tenant-a",
    actor_id="user-8",
    capability_id="session.driver",
  ) is not None


def test_sqlite_connections_close_deterministically(tmp_path, monkeypatch) -> None:
  import sqlite3

  store = ModelPreferenceStore(tmp_path / "preferences.sqlite3")
  opened: list[sqlite3.Connection] = []
  original_connect = ModelPreferenceStore._connect

  def tracking_connect(self):
    connection = original_connect(self)
    opened.append(connection)
    return connection

  monkeypatch.setattr(ModelPreferenceStore, "_connect", tracking_connect)

  store.put(
    tenant_id="tenant-a",
    actor_id="user-7",
    capability_id="session.driver",
    model_key="anthropic.claude-opus-5",
    effort="high",
  )
  assert store.get(
    tenant_id="tenant-a",
    actor_id="user-7",
    capability_id="session.driver",
  ) is not None
  assert store.delete(
    tenant_id="tenant-a",
    actor_id="user-7",
    capability_id="session.driver",
  )

  assert len(opened) == 3
  for connection in opened:
    with pytest.raises(sqlite3.ProgrammingError):
      connection.execute("SELECT 1")
