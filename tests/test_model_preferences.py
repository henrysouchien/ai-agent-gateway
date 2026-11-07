from __future__ import annotations

import pytest

from agent_gateway.model_preferences import ModelPreferenceStore, StoredPreferenceError


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


def test_stored_blob_columns_are_refused_not_stringified(tmp_path) -> None:
  import sqlite3
  from uuid import uuid4

  store = ModelPreferenceStore(tmp_path / "preferences.sqlite3")
  account = uuid4()
  store.put(
    tenant_id="tenant-a", actor_id="user-7", capability_id="session.driver",
    model_key="anthropic.claude-opus-5", effort="high",
  )
  store.put_user_preset(
    tenant_id="tenant-a", actor_id="user-7", name="economy", policy_revision="candidate-1",
  )
  store.put_firm_preset(commercial_account_id=account, name="economy", policy_revision="candidate-1")
  with sqlite3.connect(store.path) as connection:
    connection.execute("UPDATE model_preferences SET model_key=x'3432'")
    connection.execute("UPDATE user_model_presets SET name=x'3432'")
    connection.execute("UPDATE firm_model_presets SET policy_revision=x'3432'")

  with pytest.raises(ValueError, match="model_key must be text"):
    store.get(tenant_id="tenant-a", actor_id="user-7", capability_id="session.driver")
  with pytest.raises(ValueError, match="name must be text"):
    store.get_user_preset(tenant_id="tenant-a", actor_id="user-7")
  with pytest.raises(ValueError, match="policy_revision must be text"):
    store.get_firm_preset(commercial_account_id=account)

  # Each refusal names its column, row scope and capability for the HTTP boundary.
  with pytest.raises(StoredPreferenceError) as driver_error:
    store.get(tenant_id="tenant-a", actor_id="user-7", capability_id="session.driver")
  assert (driver_error.value.column, driver_error.value.scope, driver_error.value.capability_id) == (
    "model_key", "preference", "session.driver",
  )
  with pytest.raises(StoredPreferenceError) as preset_error:
    store.get_user_preset(tenant_id="tenant-a", actor_id="user-7")
  assert (preset_error.value.column, preset_error.value.scope) == ("name", "user_preset")
  with pytest.raises(StoredPreferenceError) as firm_error:
    store.get_firm_preset(commercial_account_id=account)
  assert (firm_error.value.column, firm_error.value.scope) == ("policy_revision", "firm_preset")

  # Stored bytes are unchanged by the refusals.
  with sqlite3.connect(store.path) as connection:
    assert connection.execute("SELECT model_key FROM model_preferences").fetchone() == (b"42",)
