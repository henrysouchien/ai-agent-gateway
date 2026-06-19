# ruff: noqa: E402

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.mcp_client as mcp_client_module
from agent_gateway.mcp_client import _JsonFileKeyValue
from agent_gateway.mcp_client_oauth_storage import JsonFileKeyValue


def _run(coro):
  return asyncio.run(coro)


def test_json_file_key_value_round_trips_values_and_bulk_helpers(tmp_path: Path) -> None:
  storage = JsonFileKeyValue(tmp_path / "tokens.json")

  _run(storage.put("alpha", {"token": "a"}))
  _run(storage.put_many(["beta", "gamma"], [{"token": "b"}, {"token": "c"}]))

  assert _run(storage.get("alpha")) == {"token": "a"}
  assert _run(storage.get_many(["alpha", "beta", "missing"])) == [
    {"token": "a"},
    {"token": "b"},
    None,
  ]
  assert _run(storage.delete_many(["alpha", "gamma", "missing"])) == 2
  assert _run(storage.get_many(["alpha", "beta", "gamma"])) == [None, {"token": "b"}, None]


def test_json_file_key_value_isolates_custom_collections_and_sets_private_mode(
  tmp_path: Path,
) -> None:
  path = tmp_path / "tokens.json"
  storage = JsonFileKeyValue(path, default_collection="default-coll")

  _run(storage.put("same-key", {"token": "default"}))
  _run(storage.put("same-key", {"token": "custom"}, collection="custom-coll"))

  assert _run(storage.get("same-key")) == {"token": "default"}
  assert _run(storage.get("same-key", collection="custom-coll")) == {"token": "custom"}
  assert path.stat().st_mode & 0o777 == 0o600


def test_json_file_key_value_handles_missing_malformed_and_non_dict_files(
  tmp_path: Path,
) -> None:
  path = tmp_path / "tokens.json"
  storage = JsonFileKeyValue(path)

  assert storage._load() == {}

  path.write_text("{not-json", encoding="utf-8")
  assert storage._load() == {}

  path.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
  assert storage._load() == {}


def test_json_file_key_value_expires_and_removes_stale_entries(tmp_path: Path) -> None:
  class FakeClockStorage(JsonFileKeyValue):
    now = 100.0

    def _time(self) -> float:
      return self.now

  storage = FakeClockStorage(tmp_path / "tokens.json")

  _run(storage.put("expired", {"token": "old"}, ttl=5))
  _run(storage.put("fresh", {"token": "new"}, ttl=20))
  FakeClockStorage.now = 107.0

  assert _run(storage.ttl_many(["expired", "fresh", "missing"])) == [
    (None, None),
    ({"token": "new"}, 13.0),
    (None, None),
  ]
  assert _run(storage.ttl("expired")) == (None, None)
  assert _run(storage.get("expired")) is None
  assert "expired" not in storage._load().get("default", {})


def test_json_file_key_value_rejects_mismatched_bulk_put_lengths(tmp_path: Path) -> None:
  storage = JsonFileKeyValue(tmp_path / "tokens.json")

  try:
    _run(storage.put_many(["one"], []))
  except ValueError as exc:
    assert str(exc) == "keys and values must have the same length"
  else:
    raise AssertionError("put_many should reject mismatched inputs")


def test_parent_json_file_key_value_routes_replace_and_time_through_parent_module(
  monkeypatch,
  tmp_path: Path,
) -> None:
  replacements = []
  fake_time = SimpleNamespace(current=200.0)

  def fake_replace(tmp_path_arg, path_arg):
    replacements.append((Path(tmp_path_arg).name, Path(path_arg).name))
    Path(path_arg).write_text(Path(tmp_path_arg).read_text(encoding="utf-8"), encoding="utf-8")
    Path(tmp_path_arg).unlink()

  monkeypatch.setattr(mcp_client_module, "os", SimpleNamespace(replace=fake_replace))
  monkeypatch.setattr(mcp_client_module, "time", SimpleNamespace(time=lambda: fake_time.current))
  storage = _JsonFileKeyValue(tmp_path / "tokens.json")

  _run(storage.put("alpha", {"token": "a"}, ttl=10))
  assert replacements == [("tokens.json.tmp", "tokens.json")]

  fake_time.current = 205.0
  value, ttl = _run(storage.ttl("alpha"))

  assert value == {"token": "a"}
  assert ttl == 5.0
  assert isinstance(storage, JsonFileKeyValue)


def test_parent_json_file_key_value_active_value_keeps_class_level_call_shape(
  monkeypatch,
) -> None:
  monkeypatch.setattr(mcp_client_module, "time", SimpleNamespace(time=lambda: 100.0))

  assert _JsonFileKeyValue._active_value({"value": {"token": "a"}, "expires_at": 101.0}) == {
    "token": "a"
  }
  assert _JsonFileKeyValue._active_value({"value": {"token": "a"}, "expires_at": 99.0}) is None
