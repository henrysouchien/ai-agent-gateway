from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, SupportsFloat


class JsonFileKeyValue:
  """Minimal AsyncKeyValue-compatible JSON store for FastMCP OAuth tokens."""

  def __init__(self, path: Path | str, *, default_collection: str = "default") -> None:
    self._path = Path(path).expanduser()
    self._default_collection = default_collection

  def _collection(self, collection: str | None) -> str:
    return str(collection or self._default_collection)

  def _load(self) -> dict[str, dict[str, dict[str, Any]]]:
    try:
      data = json.loads(self._path.read_text(encoding="utf-8"))
    except FileNotFoundError:
      return {}
    except (OSError, json.JSONDecodeError):
      return {}
    return data if isinstance(data, dict) else {}

  def _replace_file(self, tmp_path: Path, path: Path) -> None:
    os.replace(tmp_path, path)

  def _time(self) -> float:
    return time.time()

  def _save(self, data: dict[str, dict[str, dict[str, Any]]]) -> None:
    self._path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp_path = tempfile.mkstemp(
      prefix=f".{self._path.name}.",
      suffix=".tmp",
      dir=self._path.parent,
    )
    tmp_path = Path(raw_tmp_path)
    try:
      with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(data, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
      tmp_path.chmod(0o600)
      self._replace_file(tmp_path, self._path)
      self._path.chmod(0o600)
    finally:
      try:
        tmp_path.unlink()
      except FileNotFoundError:
        pass

  def _active_value(self, entry: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
      return None
    expires_at = entry.get("expires_at")
    if expires_at is not None:
      try:
        if float(expires_at) <= self._time():
          return None
      except (TypeError, ValueError):
        return None
    value = entry.get("value")
    return dict(value) if isinstance(value, dict) else None

  async def get(self, key: str, *, collection: str | None = None) -> dict[str, Any] | None:
    data = self._load()
    coll = self._collection(collection)
    value = self._active_value(data.get(coll, {}).get(str(key)))
    if value is None and str(key) in data.get(coll, {}):
      await self.delete(str(key), collection=coll)
    return value

  async def ttl(
    self,
    key: str,
    *,
    collection: str | None = None,
  ) -> tuple[dict[str, Any] | None, float | None]:
    data = self._load()
    coll = self._collection(collection)
    entry = data.get(coll, {}).get(str(key))
    value = self._active_value(entry)
    if value is None:
      if str(key) in data.get(coll, {}):
        await self.delete(str(key), collection=coll)
      return None, None
    expires_at = entry.get("expires_at") if isinstance(entry, dict) else None
    ttl_seconds = None if expires_at is None else max(float(expires_at) - self._time(), 0.0)
    return value, ttl_seconds

  async def put(
    self,
    key: str,
    value: Mapping[str, Any],
    *,
    collection: str | None = None,
    ttl: SupportsFloat | None = None,
  ) -> None:
    data = self._load()
    coll = self._collection(collection)
    expires_at = None if ttl is None else self._time() + float(ttl)
    data.setdefault(coll, {})[str(key)] = {
      "value": dict(value),
      "expires_at": expires_at,
    }
    self._save(data)

  async def delete(self, key: str, *, collection: str | None = None) -> bool:
    data = self._load()
    coll = self._collection(collection)
    bucket = data.get(coll, {})
    existed = str(key) in bucket
    if existed:
      bucket.pop(str(key), None)
      if not bucket:
        data.pop(coll, None)
      self._save(data)
    return existed

  async def get_many(
    self,
    keys: Sequence[str],
    *,
    collection: str | None = None,
  ) -> list[dict[str, Any] | None]:
    return [await self.get(str(key), collection=collection) for key in keys]

  async def ttl_many(
    self,
    keys: Sequence[str],
    *,
    collection: str | None = None,
  ) -> list[tuple[dict[str, Any] | None, float | None]]:
    return [await self.ttl(str(key), collection=collection) for key in keys]

  async def put_many(
    self,
    keys: Sequence[str],
    values: Sequence[Mapping[str, Any]],
    *,
    collection: str | None = None,
    ttl: SupportsFloat | None = None,
  ) -> None:
    if len(keys) != len(values):
      raise ValueError("keys and values must have the same length")
    for key, value in zip(keys, values):
      await self.put(str(key), value, collection=collection, ttl=ttl)

  async def delete_many(self, keys: Sequence[str], *, collection: str | None = None) -> int:
    deleted = 0
    for key in keys:
      if await self.delete(str(key), collection=collection):
        deleted += 1
    return deleted


__all__ = ["JsonFileKeyValue"]
