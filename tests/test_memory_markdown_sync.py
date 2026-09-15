import asyncio
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import get_type_hints

import pytest

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.memory as memory_module  # noqa: E402
import agent_gateway.memory_markdown_sync as memory_markdown_sync_module  # noqa: E402
from agent_gateway.memory import MarkdownSyncManager, MemoryStore  # noqa: E402


def _run(coro):
  return asyncio.run(coro)


def test_corrupt_memory_store_refuses_without_moving_or_replacing_file(
  tmp_path: Path,
) -> None:
  path = tmp_path / "memory.db"
  original = b"not a sqlite database\x00user-recovery-data"
  path.write_bytes(original)

  with pytest.raises(memory_module.MemoryUnavailableError, match="preserved in place"):
    MemoryStore(path)

  assert path.read_bytes() == original
  assert sorted(item.name for item in tmp_path.iterdir()) == ["memory.db"]


def test_markdown_sync_round_trips_through_memory_import_surface(tmp_path: Path) -> None:
  store = MemoryStore(tmp_path / "memory.db")
  manager = MarkdownSyncManager(store, tmp_path / "workspace")
  try:
    _run(store.upsert("Apple Inc.", "company profile", "Initial content", tags=["ticker:AAPL"]))

    exported = _run(manager.export_to_files())

    assert exported == {"written": 1, "files": ["company-profile/Apple-Inc.md"]}
    path = tmp_path / "workspace" / "company-profile" / "Apple-Inc.md"
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("<!-- gateway-memory:")
    assert "Initial content" in raw

    path.write_text(raw.replace("Initial content", "Updated content"), encoding="utf-8")
    imported = _run(manager.import_from_files())

    assert imported == {"upserted": 1, "deleted": 0, "files": ["company-profile/Apple-Inc.md"]}
    updated = _run(store.get("Apple Inc.", "company profile"))
    assert updated is not None
    assert updated["content"] == "Updated content"
    assert updated["tags"] == ["ticker:AAPL"]

    path.unlink()
    deleted = _run(manager.import_from_files())

    assert deleted == {"upserted": 0, "deleted": 1, "files": []}
    assert _run(store.get("Apple Inc.", "company profile")) is None
  finally:
    manager.stop_watch()
    store.close()


def test_markdown_sync_refuses_malformed_import_and_preserves_unowned_export_files(
  tmp_path: Path,
) -> None:
  import_store = MemoryStore(tmp_path / "import-memory.db")
  import_workspace = tmp_path / "import-workspace"
  import_manager = MarkdownSyncManager(import_store, import_workspace)
  malformed_path = import_workspace / "notes" / "malformed.md"
  malformed_content = "<!-- gateway-memory: {not-json} -->\n\nkeep me\n"

  export_store = MemoryStore(tmp_path / "export-memory.db")
  export_workspace = tmp_path / "export-workspace"
  export_manager = MarkdownSyncManager(export_store, export_workspace)
  unowned_path = export_workspace / "notes" / "unowned.md"
  unowned_content = "# User-owned note\n\nkeep me\n"
  owned_path = export_workspace / "notes" / "stale.md"
  owned_content = (
    '<!-- gateway-memory: {"entity_type": "note", "name": "stale", "tags": []} -->\n'
    "\nStale content\n"
  )

  try:
    malformed_path.parent.mkdir(parents=True)
    malformed_path.write_text(malformed_content, encoding="utf-8")

    imported = _run(import_manager.import_from_files())

    assert imported == {"upserted": 0, "deleted": 0, "files": []}
    assert malformed_path.read_text(encoding="utf-8") == malformed_content

    unowned_path.parent.mkdir(parents=True)
    unowned_path.write_text(unowned_content, encoding="utf-8")
    owned_path.write_text(owned_content, encoding="utf-8")

    exported = _run(export_manager.export_to_files())

    assert exported == {"written": 0, "files": []}
    assert unowned_path.read_text(encoding="utf-8") == unowned_content
    assert not owned_path.exists()
  finally:
    import_manager.stop_watch()
    import_store.close()
    export_manager.stop_watch()
    export_store.close()


def test_markdown_sync_self_writes_do_not_schedule_import(tmp_path: Path) -> None:
  store = MemoryStore(tmp_path / "memory.db")
  manager = MarkdownSyncManager(store, tmp_path / "workspace")
  path = tmp_path / "workspace" / "note.md"
  try:
    path.parent.mkdir(parents=True)
    with memory_module._writing_path(path):
      assert memory_module._is_self_write(path) is True
      manager._handle_watch_path(path)

    assert memory_module._is_self_write(path) is False
    assert manager._timers == {}
  finally:
    manager.stop_watch()
    store.close()


def test_markdown_sync_watch_uses_canonical_observer_binding(
  monkeypatch,
  tmp_path: Path,
) -> None:
  class _FakeObserver:
    def __init__(self) -> None:
      self.scheduled: tuple[object, str, bool] | None = None
      self.started = False
      self.stopped = False
      self.join_timeout: float | None = None

    def schedule(self, handler: object, path: str, *, recursive: bool) -> None:
      self.scheduled = (handler, path, recursive)

    def start(self) -> None:
      self.started = True

    def stop(self) -> None:
      self.stopped = True

    def join(self, timeout: float | None = None) -> None:
      self.join_timeout = timeout

  store = MemoryStore(tmp_path / "memory.db")
  manager = MarkdownSyncManager(store, tmp_path / "workspace")
  try:
    monkeypatch.setattr(memory_markdown_sync_module, "Observer", _FakeObserver)

    manager.watch()

    observer = manager._observer
    assert isinstance(observer, _FakeObserver)
    assert observer.scheduled is not None
    handler, path, recursive = observer.scheduled
    assert isinstance(handler, memory_module._SyncEventHandler)
    assert path == str(tmp_path / "workspace")
    assert recursive is True
    assert observer.started is True

    manager.stop_watch()

    assert observer.stopped is True
    assert observer.join_timeout == 5.0
  finally:
    manager.stop_watch()
    store.close()


def test_markdown_sync_type_hints_resolve_memory_store() -> None:
  hints = get_type_hints(MarkdownSyncManager.__init__)

  assert hints["store"] is MemoryStore


def test_markdown_sync_helpers_use_canonical_writing_state(
  monkeypatch,
  tmp_path: Path,
) -> None:
  path = tmp_path / "workspace" / "note.md"
  writing_paths: set[Path] = set()
  monkeypatch.setattr(memory_markdown_sync_module, "_WRITING_LOCK", threading.Lock())
  monkeypatch.setattr(memory_markdown_sync_module, "_WRITING_PATHS", writing_paths)

  with memory_module._writing_path(path):
    assert path.resolve(strict=False) in writing_paths
    assert memory_module._is_self_write(path) is True

  assert writing_paths == set()
  assert memory_module._is_self_write(path) is False

  with pytest.raises(RuntimeError, match="intentional writing failure"):
    with memory_module._writing_path(path):
      assert path.resolve(strict=False) in writing_paths
      raise RuntimeError("intentional writing failure")

  assert memory_markdown_sync_module._WRITING_PATHS == set()
  assert memory_module._is_self_write(path) is False


def test_markdown_sync_watch_uses_canonical_temp_suffixes_and_workspace_boundary(
  monkeypatch,
  tmp_path: Path,
) -> None:
  store = MemoryStore(tmp_path / "memory.db")
  manager = MarkdownSyncManager(store, tmp_path / "workspace")
  scheduled: list[bool] = []
  try:
    manager._schedule_import = lambda: scheduled.append(True)
    monkeypatch.setattr(memory_markdown_sync_module, "_TEMP_SUFFIXES", {".md"})

    manager._handle_watch_path(tmp_path / "workspace" / "note.md")

    assert scheduled == []

    monkeypatch.setattr(memory_markdown_sync_module, "_TEMP_SUFFIXES", set())
    manager._handle_watch_path(tmp_path / "workspace" / "note.md")

    assert scheduled == [True]

    manager._handle_watch_path(tmp_path / "outside.md")

    assert scheduled == [True]
  finally:
    manager.stop_watch()
    store.close()


def test_markdown_sync_atomic_write_uses_canonical_writing_path(
  monkeypatch,
  tmp_path: Path,
) -> None:
  calls: list[Path] = []

  @contextmanager
  def _fake_writing_path(path: Path):
    calls.append(path)
    yield

  path = tmp_path / "workspace" / "note.md"
  monkeypatch.setattr(memory_markdown_sync_module, "_writing_path", _fake_writing_path)

  memory_module._atomic_write(path, "hello")

  assert calls == [path]
  assert path.read_text(encoding="utf-8") == "hello"
