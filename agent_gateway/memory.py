from __future__ import annotations

import copy
import inspect
import logging
import math
import re
import sqlite3
import threading
import time
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Protocol, TypeVar, runtime_checkable

from . import memory_markdown_sync as _memory_markdown_sync
from .memory_markdown_sync import (  # noqa: F401
  FileSystemEvent,
  FileSystemEventHandler,
  FileSystemMovedEvent,
  MarkdownSyncManager,
  Observer,
  _SYNC_HEADER_PREFIX,
  _SyncEventHandler,
  _TEMP_SUFFIXES,
  _WRITING_LOCK,
  _WRITING_PATHS,
  _atomic_write,
  _is_ignored_path,
  _is_markdown_path,
  _is_self_write,
  _normalize_path,
  _slugify,
  _writing_path,
)

log = logging.getLogger("agent_gateway.memory")

T = TypeVar("T")
_CORRUPTION_SIGNATURES = (
  "malformed",
  "corrupt",
  "not a database",
  "quick_check failed",
  "disk image",
)


class MemoryUnavailableError(RuntimeError):
  """Raised when durable memory cannot be opened without risking user data."""


@runtime_checkable
class EmbeddingProvider(Protocol):
  """Protocol for pluggable embedding backends used by `MemoryStore`."""

  async def embed(self, text: str) -> list[float]:
    """Generate an embedding vector for a single text."""

  async def embed_batch(self, texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for multiple texts."""

  @property
  def dimensions(self) -> int:
    """Embedding vector dimensions."""


def cosine_similarity(a: list[float], b: list[float]) -> float:
  if not a or not b:
    return 0.0
  n = min(len(a), len(b))
  if n <= 0:
    return 0.0
  dot = sum(float(a[idx]) * float(b[idx]) for idx in range(n))
  norm_a = math.sqrt(sum(float(a[idx]) * float(a[idx]) for idx in range(n)))
  norm_b = math.sqrt(sum(float(b[idx]) * float(b[idx]) for idx in range(n)))
  if norm_a == 0.0 or norm_b == 0.0:
    return 0.0
  return dot / (norm_a * norm_b)


def rank_by_similarity(
  query_embedding: list[float],
  candidates: list[tuple[T, list[float]]],
) -> list[tuple[T, float]]:
  ranked = [
    (item, cosine_similarity(query_embedding, candidate_embedding))
    for item, candidate_embedding in candidates
  ]
  ranked.sort(key=lambda row: row[1], reverse=True)
  return ranked


def _now_ts() -> float:
  return time.time()


class MemoryStore:
  """SQLite-backed persistent memory store.

  Entities are keyed by `(name, entity_type)` and may optionally carry vector
  embeddings plus tags for semantic or keyword search.
  """

  def __init__(
    self,
    db_path: str | Path,
    *,
    embedding_fn: Callable[[str], Any] | EmbeddingProvider | None = None,
  ):
    self._db_path = Path(db_path)
    self._conn: sqlite3.Connection | None = None
    self._db_lock = threading.RLock()
    self._startup_diagnostics: dict[str, Any] = {
      "status": "not_started",
      "db_path": str(self._db_path),
    }
    self._embedding_fn = embedding_fn
    self._ensure_db()

  def set_embedding_provider(
    self,
    embedding_fn: Callable[[str], Any] | EmbeddingProvider | None,
  ) -> None:
    self._embedding_fn = embedding_fn

  @contextmanager
  def _safe_write(self):
    """Roll back on any exception so the shared connection never leaks a transaction."""
    conn = self._ensure_db()
    try:
      yield conn
    except BaseException:
      conn.rollback()
      raise

  def _is_corruption(self, exc: sqlite3.DatabaseError) -> bool:
    msg = str(exc).lower()
    return any(signature in msg for signature in _CORRUPTION_SIGNATURES)

  def startup_diagnostics(self) -> dict[str, Any]:
    with self._db_lock:
      return copy.deepcopy(self._startup_diagnostics)

  def _ensure_db(self) -> sqlite3.Connection:
    with self._db_lock:
      if self._conn is None:
        started_at = time.time()
        started_perf = time.perf_counter()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = None
        try:
          quick_check_ms = 0.0
          if self._db_path.exists():
            quick_check_perf = time.perf_counter()
            readonly_uri = f"file:{self._db_path.resolve()}?mode=ro"
            with sqlite3.connect(readonly_uri, uri=True) as readonly:
              result = readonly.execute("PRAGMA quick_check(1)").fetchone()
            quick_check_ms = round((time.perf_counter() - quick_check_perf) * 1000, 1)
            if not result or result[0] != "ok":
              raise sqlite3.DatabaseError("quick_check failed")
          connect_perf = time.perf_counter()
          conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
          conn.row_factory = sqlite3.Row
          conn.execute("PRAGMA journal_mode=WAL")
          conn.execute("PRAGMA busy_timeout=5000")
          conn.execute("PRAGMA foreign_keys=ON")
          connect_ms = round((time.perf_counter() - connect_perf) * 1000, 1)
          self._conn = conn
          migrate_perf = time.perf_counter()
          self._maybe_migrate()
          migrate_ms = round((time.perf_counter() - migrate_perf) * 1000, 1)
          total_ms = round((time.perf_counter() - started_perf) * 1000, 1)
          self._startup_diagnostics = {
            "status": "ready",
            "db_path": str(self._db_path),
            "started_at": started_at,
            "duration_ms": total_ms,
            "connect_ms": connect_ms,
            "quick_check_ms": quick_check_ms,
            "migration_ms": migrate_ms,
          }
        except sqlite3.DatabaseError as exc:
          if conn is not None:
            try:
              conn.close()
            except Exception:
              pass
          self._conn = None
          if self._is_corruption(exc) and self._db_path.exists():
            log.error(
              "Memory DB corruption detected; refusing to replace or move user data: %s",
              self._db_path,
            )
            self._startup_diagnostics = {
              "status": "unavailable",
              "db_path": str(self._db_path),
              "started_at": started_at,
              "duration_ms": round((time.perf_counter() - started_perf) * 1000, 1),
              "error": str(exc),
              "preserved_in_place": True,
            }
            raise MemoryUnavailableError(
              f"memory store is unavailable; corrupt data was preserved in place at {self._db_path}"
            ) from exc
          self._startup_diagnostics = {
            "status": "failed",
            "db_path": str(self._db_path),
            "started_at": started_at,
            "duration_ms": round((time.perf_counter() - started_perf) * 1000, 1),
            "error": str(exc),
          }
          raise
      return self._conn

  def _maybe_migrate(self) -> None:
    conn = self._conn
    if conn is None:
      raise RuntimeError("Database connection not initialized")

    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        content TEXT NOT NULL,
        embedding BLOB,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        UNIQUE(name, entity_type)
      )
      """
    )
    conn.execute(
      """
      CREATE TABLE IF NOT EXISTS entity_tags (
        entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        tag TEXT NOT NULL,
        UNIQUE(entity_id, tag)
      )
      """
    )
    conn.execute(
      "CREATE INDEX IF NOT EXISTS idx_entities_type_name ON entities(entity_type, name)"
    )
    conn.execute(
      "CREATE INDEX IF NOT EXISTS idx_entities_updated ON entities(updated_at DESC)"
    )
    conn.execute(
      "CREATE INDEX IF NOT EXISTS idx_entity_tags_tag ON entity_tags(tag)"
    )
    conn.commit()

  def _serialize_embedding(self, vector: list[float]) -> bytes:
    return sqlite3.Binary(array("f", [float(value) for value in vector]).tobytes())

  def _deserialize_embedding(self, blob: bytes | bytearray | memoryview) -> list[float]:
    values = array("f")
    values.frombytes(bytes(blob))
    return list(values)

  async def _embed_text(self, text: str) -> list[float] | None:
    if self._embedding_fn is None:
      return None

    if isinstance(self._embedding_fn, EmbeddingProvider):
      return await self._embedding_fn.embed(text)

    result = self._embedding_fn(text)
    if inspect.isawaitable(result):
      result = await result
    if result is None:
      return None
    return [float(value) for value in result]

  async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
    if not texts or self._embedding_fn is None:
      return []

    if isinstance(self._embedding_fn, EmbeddingProvider):
      vectors = await self._embedding_fn.embed_batch(texts)
      return [[float(value) for value in vector] for vector in vectors]

    vectors: list[list[float]] = []
    for text in texts:
      vector = await self._embed_text(text)
      if vector is None:
        vectors.append([])
      else:
        vectors.append(vector)
    return vectors

  def _clean_tags(self, tags: list[str] | None) -> list[str]:
    if not tags:
      return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for tag in tags:
      value = str(tag).strip()
      if not value or value in seen:
        continue
      seen.add(value)
      cleaned.append(value)
    return cleaned

  def _keyword_score(self, query: str, row: sqlite3.Row, tags: list[str]) -> float:
    haystack = " ".join([
      str(row["name"] or ""),
      str(row["entity_type"] or ""),
      str(row["content"] or ""),
      " ".join(tags),
    ]).lower()
    needle = query.strip().lower()
    if not needle or not haystack:
      return 0.0

    score = 0.0
    if needle == str(row["name"] or "").strip().lower():
      score += 2.0
    if needle in haystack:
      score += 1.0

    terms = [term for term in re.findall(r"[a-z0-9]+", needle) if term]
    if not terms:
      return score

    matched_terms = sum(1 for term in terms if term in haystack)
    return score + (matched_terms / len(terms))

  def _build_tag_filter_sql(self, tags: list[str]) -> tuple[str, list[Any]]:
    clean_tags = self._clean_tags(tags)
    if not clean_tags:
      return "", []
    placeholders = ",".join("?" for _ in clean_tags)
    sql = f"""
      AND e.id IN (
        SELECT entity_id
        FROM entity_tags
        WHERE tag IN ({placeholders})
        GROUP BY entity_id
        HAVING COUNT(DISTINCT tag) = ?
      )
    """
    return sql, [*clean_tags, len(clean_tags)]

  def _fetch_rows(
    self,
    *,
    entity_type: str | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
    order_by: str = "e.entity_type ASC, e.name ASC",
    limit: int | None = None,
  ) -> list[sqlite3.Row]:
    conn = self._ensure_db()
    where_parts = ["1 = 1"]
    params: list[Any] = []

    if entity_type is not None:
      where_parts.append("e.entity_type = ?")
      params.append(entity_type.strip())

    if name is not None:
      where_parts.append("e.name = ?")
      params.append(name.strip())

    tag_sql, tag_params = self._build_tag_filter_sql(tags or [])
    params.extend(tag_params)

    limit_sql = ""
    if limit is not None:
      limit_sql = " LIMIT ?"
      params.append(limit)

    sql = f"""
      SELECT e.id, e.name, e.entity_type, e.content, e.embedding, e.created_at, e.updated_at
      FROM entities e
      WHERE {' AND '.join(where_parts)}
      {tag_sql}
      ORDER BY {order_by}
      {limit_sql}
    """
    return conn.execute(sql, params).fetchall()

  def _load_tags(self, entity_ids: list[int]) -> dict[int, list[str]]:
    if not entity_ids:
      return {}
    conn = self._ensure_db()
    placeholders = ",".join("?" for _ in entity_ids)
    rows = conn.execute(
      f"""
      SELECT entity_id, tag
      FROM entity_tags
      WHERE entity_id IN ({placeholders})
      ORDER BY tag ASC
      """,
      entity_ids,
    ).fetchall()

    tags_by_entity: dict[int, list[str]] = {entity_id: [] for entity_id in entity_ids}
    for row in rows:
      tags_by_entity.setdefault(int(row["entity_id"]), []).append(str(row["tag"]))
    return tags_by_entity

  def _row_to_dict(self, row: sqlite3.Row, tags: list[str]) -> dict[str, Any]:
    return {
      "id": int(row["id"]),
      "name": str(row["name"]),
      "entity_type": str(row["entity_type"]),
      "content": str(row["content"]),
      "tags": list(tags),
      "created_at": float(row["created_at"]),
      "updated_at": float(row["updated_at"]),
    }

  async def upsert(
    self,
    name: str,
    entity_type: str,
    content: str,
    tags: list[str] | None = None,
  ) -> int:
    clean_name = name.strip()
    clean_type = entity_type.strip()
    clean_content = content.strip()
    if not clean_name or not clean_type or not clean_content:
      raise ValueError("name, entity_type, and content are required")

    embedding = await self._embed_text(clean_content)
    serialized = self._serialize_embedding(embedding) if embedding else None
    clean_tags = self._clean_tags(tags)
    now = _now_ts()

    with self._db_lock:
      with self._safe_write() as conn:
        conn.execute(
          """
          INSERT INTO entities (name, entity_type, content, embedding, created_at, updated_at)
          VALUES (?, ?, ?, ?, ?, ?)
          ON CONFLICT(name, entity_type) DO UPDATE SET
            content = excluded.content,
            embedding = excluded.embedding,
            updated_at = excluded.updated_at
          """,
          (clean_name, clean_type, clean_content, serialized, now, now),
        )
        row = conn.execute(
          """
          SELECT id
          FROM entities
          WHERE name = ? AND entity_type = ?
          """,
          (clean_name, clean_type),
        ).fetchone()
        if row is None:
          raise RuntimeError("failed to upsert entity")

        entity_id = int(row["id"])
        if tags is not None:
          conn.execute("DELETE FROM entity_tags WHERE entity_id = ?", (entity_id,))
          if clean_tags:
            conn.executemany(
              """
              INSERT OR IGNORE INTO entity_tags (entity_id, tag)
              VALUES (?, ?)
              """,
              [(entity_id, tag) for tag in clean_tags],
            )
        conn.commit()
        return entity_id

  async def search(
    self,
    query: str,
    *,
    entity_type: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
  ) -> list[dict]:
    needle = query.strip()
    if not needle:
      return []

    clean_limit = max(1, min(int(limit), 500))
    with self._db_lock:
      rows = self._fetch_rows(
        entity_type=entity_type,
        tags=tags,
        order_by="e.updated_at DESC, e.id DESC",
      )
      tags_by_entity = self._load_tags([int(row["id"]) for row in rows])

    query_embedding = await self._embed_text(needle)
    results: list[dict[str, Any]] = []
    for row in rows:
      entity_id = int(row["id"])
      entity_tags = tags_by_entity.get(entity_id, [])
      score: float
      if query_embedding is not None and row["embedding"] is not None:
        score = cosine_similarity(
          query_embedding,
          self._deserialize_embedding(row["embedding"]),
        )
      else:
        score = self._keyword_score(needle, row, entity_tags)
        if score <= 0.0:
          continue
      item = self._row_to_dict(row, entity_tags)
      item["score"] = float(score)
      results.append(item)

    results.sort(
      key=lambda row: (
        float(row.get("score") or 0.0),
        float(row.get("updated_at") or 0.0),
        str(row.get("name") or ""),
      ),
      reverse=True,
    )
    return results[:clean_limit]

  async def get(self, name: str, entity_type: str | None = None) -> dict | None:
    clean_name = name.strip()
    if not clean_name:
      return None

    with self._db_lock:
      rows = self._fetch_rows(
        name=clean_name,
        entity_type=entity_type,
        order_by="e.updated_at DESC, e.id DESC",
        limit=1,
      )
      if not rows:
        return None
      row = rows[0]
      tags_by_entity = self._load_tags([int(row["id"])])
      return self._row_to_dict(row, tags_by_entity.get(int(row["id"]), []))

  async def delete(self, name: str, entity_type: str | None = None) -> bool:
    clean_name = name.strip()
    if not clean_name:
      raise ValueError("name is required")

    with self._db_lock:
      with self._safe_write() as conn:
        if entity_type is None:
          cursor = conn.execute(
            "DELETE FROM entities WHERE name = ?",
            (clean_name,),
          )
        else:
          cursor = conn.execute(
            "DELETE FROM entities WHERE name = ? AND entity_type = ?",
            (clean_name, entity_type.strip()),
          )
        conn.commit()
        return cursor.rowcount > 0

  async def list_entities(
    self,
    entity_type: str | None = None,
    tags: list[str] | None = None,
  ) -> list[dict]:
    with self._db_lock:
      rows = self._fetch_rows(entity_type=entity_type, tags=tags)
      tags_by_entity = self._load_tags([int(row["id"]) for row in rows])
      return [
        self._row_to_dict(row, tags_by_entity.get(int(row["id"]), []))
        for row in rows
      ]

  def close(self) -> None:
    with self._db_lock:
      if self._conn is not None:
        self._conn.close()
        self._conn = None


_memory_markdown_sync.MemoryStore = MemoryStore
