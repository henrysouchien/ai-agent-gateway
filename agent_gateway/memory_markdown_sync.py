from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
import inspect
import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
  from .memory import MemoryStore
  from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
    FileSystemEventHandler as _WatchdogEventHandler,
    FileSystemMovedEvent,
  )
  from watchdog.observers import Observer
  from watchdog.observers.api import BaseObserver
else:
  MemoryStore = Any
  try:
    from watchdog.events import (
      FileSystemEvent,
      FileSystemEventHandler,
      FileSystemEventHandler as _WatchdogEventHandler,
      FileSystemMovedEvent,
    )
    from watchdog.observers import Observer
  except Exception:  # pragma: no cover - exercised when watchdog is unavailable
    FileSystemEvent = object
    FileSystemEventHandler = object
    FileSystemMovedEvent = object
    _WatchdogEventHandler = object
    Observer = None


log = logging.getLogger("agent_gateway.memory")

_SYNC_HEADER_PREFIX = "<!-- gateway-memory:"
_TEMP_SUFFIXES = {".tmp", ".swp", ".swx", ".bak", ".part"}
_WRITING_LOCK = threading.Lock()
_WRITING_PATHS: set[Path] = set()
_WatchCallback = Callable[
  [dict[str, Any]],
  Coroutine[Any, Any, Any] | None,
]


def _normalize_path(path: Path | str) -> Path:
  return Path(path).resolve(strict=False)


@contextmanager
def _writing_path(path: Path | str):
  writing_lock = _WRITING_LOCK
  writing_paths = _WRITING_PATHS
  normalized = _normalize_path(path)
  with writing_lock:
    writing_paths.add(normalized)
  try:
    yield
  finally:
    with writing_lock:
      writing_paths.discard(normalized)


def _is_self_write(path: Path) -> bool:
  normalized = _normalize_path(path)
  with _WRITING_LOCK:
    return normalized in _WRITING_PATHS


def _is_ignored_path(path: Path) -> bool:
  name = path.name.lower()
  if name.startswith("."):
    return True
  return path.suffix.lower() in _TEMP_SUFFIXES


def _is_markdown_path(path: Path) -> bool:
  return path.suffix.lower() == ".md"


def _slugify(value: str) -> str:
  slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
  return slug or "entity"


def _atomic_write(path: Path, content: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_suffix(path.suffix + ".tmp")
  with _writing_path(path):
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


class _SyncEventHandler(_WatchdogEventHandler):
  def __init__(self, manager: "MarkdownSyncManager") -> None:
    super().__init__()
    self._manager = manager

  def on_created(self, event: FileSystemEvent) -> None:
    self._manager._handle_watch_path(Path(os.fsdecode(event.src_path)))

  def on_modified(self, event: FileSystemEvent) -> None:
    self._manager._handle_watch_path(Path(os.fsdecode(event.src_path)))

  def on_deleted(self, event: FileSystemEvent) -> None:
    self._manager._handle_watch_path(Path(os.fsdecode(event.src_path)))

  def on_moved(self, event: FileSystemMovedEvent) -> None:
    self._manager._handle_watch_path(Path(os.fsdecode(event.src_path)))
    self._manager._handle_watch_path(Path(os.fsdecode(event.dest_path)))


class MarkdownSyncManager:
  """Synchronize `MemoryStore` entities to and from markdown files."""

  def __init__(self, store: MemoryStore, workspace_dir: str | Path):
    self._store = store
    self._workspace_dir = Path(workspace_dir)
    self._observer: BaseObserver | None = None
    self._timers: dict[Path, threading.Timer] = {}
    self._timers_lock = threading.Lock()
    self._watch_callback: _WatchCallback | None = None

  def _entity_path(self, name: str, entity_type: str) -> Path:
    return self._workspace_dir / _slugify(entity_type) / f"{_slugify(name)}.md"

  def _render_entity_file(self, entity: dict[str, Any]) -> str:
    metadata = {
      "name": str(entity["name"]),
      "entity_type": str(entity["entity_type"]),
      "tags": list(entity.get("tags") or []),
    }
    header = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    content = str(entity["content"]).rstrip()
    if content:
      return f"{_SYNC_HEADER_PREFIX} {header} -->\n\n{content}\n"
    return f"{_SYNC_HEADER_PREFIX} {header} -->\n"

  def _parse_entity_file(self, path: Path) -> dict[str, Any] | None:
    try:
      raw = path.read_text(encoding="utf-8")
    except OSError as exc:
      log.warning("Failed to read memory markdown file %s: %s", path, exc)
      return None

    lines = raw.splitlines()
    name = path.stem
    entity_type = path.parent.name
    tags: list[str] = []
    content = raw

    if lines and lines[0].startswith(_SYNC_HEADER_PREFIX) and lines[0].rstrip().endswith("-->"):
      payload = lines[0][len(_SYNC_HEADER_PREFIX):]
      payload = payload.rsplit("-->", 1)[0].strip()
      try:
        metadata = json.loads(payload)
      except json.JSONDecodeError as exc:
        log.warning("Failed to parse memory markdown header %s: %s", path, exc)
        return None
      name = str(metadata.get("name") or name).strip() or name
      entity_type = str(metadata.get("entity_type") or entity_type).strip() or entity_type
      tags = [
        str(tag).strip()
        for tag in metadata.get("tags", [])
        if str(tag).strip()
      ]
      content = "\n".join(lines[1:]).lstrip("\n")

    return {
      "name": name,
      "entity_type": entity_type,
      "tags": tags,
      "content": content.rstrip(),
      "path": path,
    }

  async def import_from_files(self, glob_pattern: str = "**/*.md") -> dict:
    self._workspace_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
      path for path in self._workspace_dir.glob(glob_pattern)
      if path.is_file() and _is_markdown_path(path)
    )

    actual_relative_paths = {
      path.relative_to(self._workspace_dir).as_posix()
      for path in files
    }
    upserted = 0
    imported_files: list[str] = []

    for path in files:
      parsed = self._parse_entity_file(path)
      if parsed is None:
        continue
      imported_files.append(path.relative_to(self._workspace_dir).as_posix())
      existing = await self._store.get(parsed["name"], parsed["entity_type"])
      existing_tags = sorted(existing.get("tags", [])) if existing else []
      incoming_tags = sorted(parsed["tags"])
      if existing is None or existing["content"] != parsed["content"] or existing_tags != incoming_tags:
        await self._store.upsert(
          parsed["name"],
          parsed["entity_type"],
          parsed["content"],
          tags=parsed["tags"],
        )
        upserted += 1

    deleted = 0
    for entity in await self._store.list_entities():
      expected_path = self._entity_path(entity["name"], entity["entity_type"])
      rel_path = expected_path.relative_to(self._workspace_dir).as_posix()
      if rel_path in actual_relative_paths:
        continue
      if await self._store.delete(entity["name"], entity["entity_type"]):
        deleted += 1

    return {"upserted": upserted, "deleted": deleted, "files": imported_files}

  async def export_to_files(self) -> dict:
    self._workspace_dir.mkdir(parents=True, exist_ok=True)
    entities = await self._store.list_entities()
    written_files: list[str] = []
    expected_paths: set[Path] = set()

    for entity in entities:
      path = self._entity_path(entity["name"], entity["entity_type"])
      expected_paths.add(path.resolve(strict=False))
      _atomic_write(path, self._render_entity_file(entity))
      written_files.append(path.relative_to(self._workspace_dir).as_posix())

    for path in sorted(self._workspace_dir.rglob("*.md")):
      if path.resolve(strict=False) in expected_paths:
        continue
      try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
      except (IndexError, OSError):
        first_line = ""
      if not first_line.startswith(_SYNC_HEADER_PREFIX):
        continue
      with _writing_path(path):
        path.unlink(missing_ok=True)

    return {"written": len(written_files), "files": written_files}

  def _handle_watch_path(self, path: Path) -> None:
    normalized = _normalize_path(path)
    if _is_ignored_path(normalized) or _is_self_write(normalized):
      return
    if not _is_markdown_path(normalized):
      return
    try:
      normalized.relative_to(self._workspace_dir.resolve())
    except ValueError:
      return
    self._schedule_import()

  def _schedule_import(self) -> None:
    key = self._workspace_dir.resolve()
    with self._timers_lock:
      existing = self._timers.pop(key, None)
      if existing is not None:
        existing.cancel()
      timer = threading.Timer(0.5, self._run_import_timer)
      timer.daemon = True
      self._timers[key] = timer
      timer.start()

  def _run_import_timer(self) -> None:
    key = self._workspace_dir.resolve()
    callback = self._watch_callback
    try:
      summary = asyncio.run(self.import_from_files())
      if callback is not None:
        result = callback(summary)
        if inspect.isawaitable(result):
          asyncio.run(result)
    except Exception as exc:  # pragma: no cover - defensive runtime path
      log.warning("Markdown sync watcher import failed: %s", exc)
    finally:
      with self._timers_lock:
        self._timers.pop(key, None)

  def watch(self, callback: _WatchCallback | None = None) -> None:
    self._watch_callback = callback
    if Observer is None:
      log.warning("Markdown sync watch disabled: watchdog dependency unavailable")
      return
    if self._observer is not None:
      return

    self._workspace_dir.mkdir(parents=True, exist_ok=True)
    observer = Observer()
    observer.schedule(_SyncEventHandler(self), str(self._workspace_dir), recursive=True)
    observer.start()
    self._observer = observer

  def stop_watch(self) -> None:
    with self._timers_lock:
      for timer in self._timers.values():
        timer.cancel()
      self._timers.clear()

    observer = self._observer
    self._observer = None
    if observer is not None:
      observer.stop()
      observer.join(timeout=5.0)
