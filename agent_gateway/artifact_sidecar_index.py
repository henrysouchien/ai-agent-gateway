from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .artifact_paths import canonicalize_ticker


INDEX_VERSION = 1
_INDEX_RELATIVE_PATH = ("artifacts", "_index", "artifact_sidecars.sqlite3")


def register_dashboard_artifact_sidecar(
  *,
  workspace_dir: Path,
  artifact: Any,
  sidecar_path: Path,
  payload_path: Path,
  user_id: str = "",
) -> None:
  effective_user_id = _effective_user_id(workspace_dir, user_id=user_id)
  _upsert_index_row(
    workspace_dir=workspace_dir,
    row={
      "user_id": effective_user_id,
      "artifact_kind": "dashboard",
      "artifact_id": str(artifact.artifact_id),
      "artifact_ref": _relative_to_workspace(workspace_dir, sidecar_path),
      "payload_ref": _relative_to_workspace(workspace_dir, payload_path),
      "scope": "ticker" if artifact.ticker else "portfolio",
      "scope_label": artifact.scope_label,
      "ticker": artifact.ticker,
      "skill": artifact.source_skill,
      "contract_name": getattr(artifact, "contract_name", "DashboardArtifact"),
      "purpose": None,
      "research_file_id": artifact.research_file_id,
      "control_run_id": artifact.control_run_id,
      "origin_kind": artifact.origin_kind,
      "visibility": artifact.visibility,
      "origin_ref": _json_or_none(artifact.origin_ref),
      "classification_source": _classification_source(artifact),
      "created_ts": artifact.ts,
      "updated_ts": _mtime_ts(sidecar_path),
      "sidecar_mtime_ns": sidecar_path.stat().st_mtime_ns,
      "content_hash": _sha256_file(sidecar_path),
      "index_version": INDEX_VERSION,
    },
  )


def register_html_artifact_sidecar(
  *,
  workspace_dir: Path,
  artifact: Any,
  sidecar_path: Path,
  content_path: Path,
  user_id: str = "",
) -> None:
  effective_user_id = _effective_user_id(workspace_dir, user_id=user_id)
  _upsert_index_row(
    workspace_dir=workspace_dir,
    row={
      "user_id": effective_user_id,
      "artifact_kind": "html",
      "artifact_id": str(artifact.artifact_id),
      "artifact_ref": _relative_to_workspace(workspace_dir, sidecar_path),
      "payload_ref": _relative_to_workspace(workspace_dir, content_path),
      "scope": "ticker" if artifact.ticker else "portfolio",
      "scope_label": None,
      "ticker": artifact.ticker,
      "skill": artifact.source_skill,
      "contract_name": getattr(artifact, "contract_name", "HtmlArtifact"),
      "purpose": artifact.purpose,
      "research_file_id": artifact.research_file_id,
      "control_run_id": artifact.control_run_id,
      "origin_kind": artifact.origin_kind,
      "visibility": artifact.visibility,
      "origin_ref": _json_or_none(artifact.origin_ref),
      "classification_source": _classification_source(artifact),
      "created_ts": artifact.ts,
      "updated_ts": _mtime_ts(sidecar_path),
      "sidecar_mtime_ns": sidecar_path.stat().st_mtime_ns,
      "content_hash": _sha256_file(sidecar_path),
      "index_version": INDEX_VERSION,
    },
  )


def register_skill_artifact_sidecar(
  *,
  workspace_dir: Path,
  sidecar_path: Path,
  payload: dict[str, Any] | None = None,
  user_id: str = "",
) -> None:
  effective_user_id = _effective_user_id(workspace_dir, user_id=user_id)
  sidecar = Path(sidecar_path)
  payload = dict(payload) if payload is not None else _read_json_object(sidecar)
  path_fields = _skill_artifact_path_fields(workspace_dir=workspace_dir, sidecar_path=sidecar, payload=payload)
  research_file_id = _int_or_none(payload.get("research_file_id"))
  _upsert_index_row(
    workspace_dir=workspace_dir,
    row={
      "user_id": effective_user_id,
      "artifact_kind": "skill_artifact",
      "artifact_id": path_fields["artifact_id"],
      "artifact_ref": path_fields["artifact_ref"],
      "payload_ref": _skill_payload_ref(workspace_dir=workspace_dir, payload=payload),
      "scope": path_fields["scope"],
      "scope_label": payload.get("scope_label"),
      "ticker": path_fields["ticker"],
      "skill": path_fields["skill"],
      "contract_name": payload.get("contract_name"),
      "purpose": payload.get("purpose"),
      "research_file_id": research_file_id,
      "control_run_id": _text_or_none(payload.get("control_run_id")),
      "origin_kind": payload.get("origin_kind"),
      "visibility": payload.get("visibility"),
      "origin_ref": _json_or_none(payload.get("origin_ref")),
      "classification_source": _payload_classification_source(payload, research_file_id=research_file_id),
      "created_ts": _text_or_none(payload.get("created_at")) or _text_or_none(payload.get("ts")) or _mtime_ts(sidecar),
      "updated_ts": _mtime_ts(sidecar),
      "sidecar_mtime_ns": sidecar.stat().st_mtime_ns,
      "content_hash": _sha256_file(sidecar),
      "index_version": INDEX_VERSION,
    },
  )


def register_ui_blocks_payload_sidecar(
  *,
  workspace_dir: Path,
  user_id: str,
  ui_blocks_id: str,
  path: Path,
  session_id: str,
  turn_key: str,
  emission_index: int,
  ts: float,
  manifest_digest: str,
) -> None:
  del manifest_digest  # The digest stays in the envelope; the index has no digest column.
  effective_user_id = _effective_user_id(workspace_dir, user_id=user_id)
  envelope_path = _ensure_under_workspace(path, workspace_dir)
  envelope_ts = datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
  _upsert_index_row(
    workspace_dir=workspace_dir,
    row={
      "user_id": effective_user_id,
      "artifact_kind": "ui_blocks",
      "artifact_id": str(ui_blocks_id),
      "artifact_ref": str(ui_blocks_id),
      "payload_ref": _relative_to_workspace(workspace_dir, envelope_path),
      "scope": None,
      "scope_label": None,
      "ticker": None,
      "skill": None,
      "contract_name": "hank_ui_blocks.v1",
      "purpose": None,
      "research_file_id": None,
      "control_run_id": None,
      "origin_kind": "chat",
      "visibility": "default",
      "origin_ref": _json_or_none({
        "session_id": str(session_id),
        "turn_key": str(turn_key),
        "emission_index": int(emission_index),
      }),
      "classification_source": "ui_blocks_store",
      "created_ts": envelope_ts,
      "updated_ts": envelope_ts,
      "sidecar_mtime_ns": envelope_path.stat().st_mtime_ns,
      "content_hash": _sha256_file(envelope_path),
      "index_version": INDEX_VERSION,
    },
  )


def reconcile_ui_blocks_index(users_root: Path) -> None:
  root = Path(users_root).resolve()
  if not root.is_dir():
    return
  for user_dir in sorted(root.iterdir(), key=lambda item: item.name):
    workspace = user_dir / "workspace"
    if (
      user_dir.is_symlink()
      or workspace.is_symlink()
      or not user_dir.is_dir()
      or not workspace.is_dir()
    ):
      continue
    user_id = user_dir.name
    rows = list_artifact_sidecar_index_rows(
      workspace_dir=workspace,
      artifact_kind="ui_blocks",
      user_id=user_id,
    )
    files_dir = workspace / "artifacts" / "_ui_blocks"
    files = (
      {
        path.stem: path
        for path in sorted(files_dir.glob("*.json"), key=lambda item: item.name)
        if path.is_file() and not path.is_symlink()
      }
      if files_dir.is_dir()
      else {}
    )

    for row in rows:
      if str(row["artifact_ref"]) not in files:
        _delete_index_row(
          workspace_dir=workspace,
          user_id=user_id,
          artifact_kind="ui_blocks",
          artifact_ref=str(row["artifact_ref"]),
        )

    rows_by_ref = {str(row["artifact_ref"]): row for row in rows}
    for ui_blocks_id, path in files.items():
      try:
        envelope = _read_json_object(path)
        if str(envelope.get("ui_blocks_id") or "") != ui_blocks_id:
          raise ValueError("ui_blocks_id does not match filename")
        session_id = str(envelope["session_id"])
        turn_key = str(envelope["turn_key"])
        emission_index = int(envelope["emission_index"])
        ts = float(envelope["ts"])
        manifest_digest = str(envelope["manifest_digest"])
      except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        if ui_blocks_id in rows_by_ref:
          mark_artifact_sidecar_index_row_stale(
            workspace_dir=workspace,
            artifact_kind="ui_blocks",
            artifact_ref=ui_blocks_id,
            error="corrupt_envelope",
            user_id=user_id,
          )
        else:
          _upsert_minimal_stale_ui_blocks_row(
            workspace_dir=workspace,
            user_id=user_id,
            ui_blocks_id=ui_blocks_id,
            path=path,
          )
        continue
      register_ui_blocks_payload_sidecar(
        workspace_dir=workspace,
        user_id=user_id,
        ui_blocks_id=ui_blocks_id,
        path=path,
        session_id=session_id,
        turn_key=turn_key,
        emission_index=emission_index,
        ts=ts,
        manifest_digest=manifest_digest,
      )


def artifact_sidecar_index_path(workspace_dir: Path) -> Path:
  return _ensure_under_workspace(Path(workspace_dir).resolve().joinpath(*_INDEX_RELATIVE_PATH), workspace_dir)


def get_artifact_sidecar_index_row(
  *,
  workspace_dir: Path,
  artifact_kind: str,
  artifact_id: str,
  user_id: str = "",
) -> dict[str, Any] | None:
  db_path = artifact_sidecar_index_path(workspace_dir)
  if not db_path.is_file():
    return None
  effective_user_id = _effective_user_id(workspace_dir, user_id=user_id)
  with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    row = conn.execute(
      """
      SELECT * FROM artifact_sidecar_index
      WHERE user_id=? AND artifact_kind=? AND artifact_id=?
      """,
      (effective_user_id, str(artifact_kind), str(artifact_id)),
    ).fetchone()
  return dict(row) if row is not None else None


def get_artifact_sidecar_index_row_by_ref(
  *,
  workspace_dir: Path,
  artifact_kind: str,
  artifact_ref: str,
  user_id: str = "",
) -> dict[str, Any] | None:
  db_path = artifact_sidecar_index_path(workspace_dir)
  if not db_path.is_file():
    return None
  effective_user_id = _effective_user_id(workspace_dir, user_id=user_id)
  with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    row = conn.execute(
      """
      SELECT * FROM artifact_sidecar_index
      WHERE user_id=? AND artifact_kind=? AND artifact_ref=?
      """,
      (effective_user_id, str(artifact_kind), str(artifact_ref)),
    ).fetchone()
  return dict(row) if row is not None else None


def list_artifact_sidecar_index_rows(
  *,
  workspace_dir: Path,
  artifact_kind: str | None = None,
  user_id: str = "",
) -> list[dict[str, Any]]:
  db_path = artifact_sidecar_index_path(workspace_dir)
  if not db_path.is_file():
    return []
  effective_user_id = _effective_user_id(workspace_dir, user_id=user_id)
  with sqlite3.connect(db_path) as conn:
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    if artifact_kind is None:
      rows = conn.execute(
        """
        SELECT * FROM artifact_sidecar_index
        WHERE user_id=?
        ORDER BY artifact_kind, artifact_ref
        """,
        (effective_user_id,),
      ).fetchall()
    else:
      rows = conn.execute(
        """
        SELECT * FROM artifact_sidecar_index
        WHERE user_id=? AND artifact_kind=?
        ORDER BY artifact_ref
        """,
        (effective_user_id, str(artifact_kind)),
      ).fetchall()
  return [dict(row) for row in rows]


def mark_artifact_sidecar_index_row_stale(
  *,
  workspace_dir: Path,
  artifact_kind: str,
  artifact_ref: str,
  error: str,
  user_id: str = "",
) -> bool:
  db_path = artifact_sidecar_index_path(workspace_dir)
  if not db_path.is_file():
    return False
  effective_user_id = _effective_user_id(workspace_dir, user_id=user_id)
  with sqlite3.connect(db_path) as conn:
    _ensure_schema(conn)
    cursor = conn.execute(
      """
      UPDATE artifact_sidecar_index
      SET stale_ts=?, last_error=?
      WHERE user_id=? AND artifact_kind=? AND artifact_ref=?
      """,
      (_now_ts(), str(error), effective_user_id, str(artifact_kind), str(artifact_ref)),
    )
  return cursor.rowcount > 0


def _upsert_index_row(*, workspace_dir: Path, row: dict[str, Any]) -> None:
  db_path = artifact_sidecar_index_path(workspace_dir)
  db_path.parent.mkdir(parents=True, exist_ok=True)
  values = _normalized_row(row)
  with sqlite3.connect(db_path) as conn:
    _ensure_schema(conn)
    conn.execute(
      """
      INSERT INTO artifact_sidecar_index (
        user_id,
        artifact_kind,
        artifact_id,
        artifact_ref,
        payload_ref,
        scope,
        scope_label,
        ticker,
        skill,
        contract_name,
        purpose,
        research_file_id,
        control_run_id,
        origin_kind,
        visibility,
        origin_ref,
        classification_source,
        created_ts,
        updated_ts,
        sidecar_mtime_ns,
        content_hash,
        index_version,
        last_seen_ts,
        stale_ts,
        last_error
      ) VALUES (
        :user_id,
        :artifact_kind,
        :artifact_id,
        :artifact_ref,
        :payload_ref,
        :scope,
        :scope_label,
        :ticker,
        :skill,
        :contract_name,
        :purpose,
        :research_file_id,
        :control_run_id,
        :origin_kind,
        :visibility,
        :origin_ref,
        :classification_source,
        :created_ts,
        :updated_ts,
        :sidecar_mtime_ns,
        :content_hash,
        :index_version,
        :last_seen_ts,
        NULL,
        NULL
      )
      ON CONFLICT(user_id, artifact_kind, artifact_ref) DO UPDATE SET
        artifact_ref=excluded.artifact_ref,
        payload_ref=excluded.payload_ref,
        scope=excluded.scope,
        scope_label=excluded.scope_label,
        ticker=excluded.ticker,
        skill=excluded.skill,
        contract_name=excluded.contract_name,
        purpose=excluded.purpose,
        research_file_id=excluded.research_file_id,
        control_run_id=excluded.control_run_id,
        origin_kind=excluded.origin_kind,
        visibility=excluded.visibility,
        origin_ref=excluded.origin_ref,
        classification_source=excluded.classification_source,
        created_ts=excluded.created_ts,
        updated_ts=excluded.updated_ts,
        sidecar_mtime_ns=excluded.sidecar_mtime_ns,
        content_hash=excluded.content_hash,
        index_version=excluded.index_version,
        last_seen_ts=excluded.last_seen_ts,
        stale_ts=NULL,
        last_error=NULL
      """,
      values,
    )


def _upsert_minimal_stale_ui_blocks_row(
  *,
  workspace_dir: Path,
  user_id: str,
  ui_blocks_id: str,
  path: Path,
) -> None:
  db_path = artifact_sidecar_index_path(workspace_dir)
  db_path.parent.mkdir(parents=True, exist_ok=True)
  safe_path = _ensure_under_workspace(path, workspace_dir)
  info = safe_path.stat()
  mtime = datetime.fromtimestamp(info.st_mtime, tz=timezone.utc).isoformat()
  columns = (
    "user_id", "artifact_kind", "artifact_id", "artifact_ref", "payload_ref",
    "scope", "scope_label", "ticker", "skill", "contract_name", "purpose",
    "research_file_id", "control_run_id", "origin_kind", "visibility", "origin_ref",
    "classification_source", "created_ts", "updated_ts", "sidecar_mtime_ns",
    "content_hash", "index_version", "last_seen_ts", "stale_ts", "last_error",
  )
  row = {
    "user_id": _effective_user_id(workspace_dir, user_id=user_id),
    "artifact_kind": "ui_blocks",
    "artifact_id": ui_blocks_id,
    "artifact_ref": ui_blocks_id,
    "payload_ref": _relative_to_workspace(workspace_dir, safe_path),
    "created_ts": mtime,
    "updated_ts": mtime,
    "sidecar_mtime_ns": info.st_mtime_ns,
    "content_hash": _sha256_file(safe_path),
    "index_version": INDEX_VERSION,
    "stale_ts": _now_ts(),
    "last_error": "corrupt_envelope",
  }
  values = {column: row.get(column) for column in columns}
  with sqlite3.connect(db_path) as conn:
    _ensure_schema(conn)
    conn.execute(
      f"""
      INSERT INTO artifact_sidecar_index ({', '.join(columns)})
      VALUES ({', '.join(':' + column for column in columns)})
      ON CONFLICT(user_id, artifact_kind, artifact_ref) DO UPDATE SET
        artifact_id=excluded.artifact_id,
        payload_ref=excluded.payload_ref,
        created_ts=excluded.created_ts,
        updated_ts=excluded.updated_ts,
        sidecar_mtime_ns=excluded.sidecar_mtime_ns,
        content_hash=excluded.content_hash,
        index_version=excluded.index_version,
        stale_ts=excluded.stale_ts,
        last_error=excluded.last_error
      """,
      values,
    )


def _delete_index_row(
  *,
  workspace_dir: Path,
  user_id: str,
  artifact_kind: str,
  artifact_ref: str,
) -> None:
  db_path = artifact_sidecar_index_path(workspace_dir)
  if not db_path.is_file():
    return
  with sqlite3.connect(db_path) as conn:
    _ensure_schema(conn)
    conn.execute(
      "DELETE FROM artifact_sidecar_index WHERE user_id=? AND artifact_kind=? AND artifact_ref=?",
      (_effective_user_id(workspace_dir, user_id=user_id), artifact_kind, artifact_ref),
    )


def _ensure_schema(conn: sqlite3.Connection) -> None:
  conn.execute(
    """
    CREATE TABLE IF NOT EXISTS artifact_sidecar_index (
      user_id TEXT NOT NULL,
      artifact_kind TEXT NOT NULL,
      artifact_id TEXT NOT NULL,
      artifact_ref TEXT NOT NULL,
      payload_ref TEXT,
      scope TEXT,
      scope_label TEXT,
      ticker TEXT,
      skill TEXT,
      contract_name TEXT,
      purpose TEXT,
      research_file_id INTEGER,
      control_run_id TEXT,
      origin_kind TEXT,
      visibility TEXT,
      origin_ref TEXT,
      classification_source TEXT,
      created_ts TEXT,
      updated_ts TEXT,
      sidecar_mtime_ns INTEGER,
      content_hash TEXT,
      index_version INTEGER NOT NULL,
      last_seen_ts TEXT,
      stale_ts TEXT,
      last_error TEXT,
      PRIMARY KEY (user_id, artifact_kind, artifact_ref)
    )
    """
  )
  conn.execute(
    """
    CREATE INDEX IF NOT EXISTS artifact_sidecar_index_kind_id
    ON artifact_sidecar_index(user_id, artifact_kind, artifact_id)
    """
  )
  conn.execute(
    """
    CREATE INDEX IF NOT EXISTS artifact_sidecar_index_file_updated
    ON artifact_sidecar_index(user_id, research_file_id, updated_ts DESC)
    """
  )
  conn.execute(
    """
    CREATE INDEX IF NOT EXISTS artifact_sidecar_index_ticker_skill_updated
    ON artifact_sidecar_index(user_id, ticker, skill, updated_ts DESC)
    """
  )
  conn.execute(
    """
    CREATE INDEX IF NOT EXISTS artifact_sidecar_index_kind_updated
    ON artifact_sidecar_index(user_id, artifact_kind, updated_ts DESC)
    """
  )
  conn.execute(
    """
    CREATE INDEX IF NOT EXISTS artifact_sidecar_index_classification_updated
    ON artifact_sidecar_index(user_id, origin_kind, visibility, updated_ts DESC)
    """
  )


def _normalized_row(row: dict[str, Any]) -> dict[str, Any]:
  values = dict(row)
  values["user_id"] = str(values.get("user_id") or "")
  values["artifact_kind"] = str(values.get("artifact_kind") or "")
  values["artifact_id"] = str(values.get("artifact_id") or "")
  values["artifact_ref"] = str(values.get("artifact_ref") or "")
  values["index_version"] = int(values.get("index_version") or INDEX_VERSION)
  values["last_seen_ts"] = values.get("last_seen_ts") or _now_ts()
  return values


def _workspace_user_id(workspace_dir: Path) -> str:
  parts = Path(workspace_dir).resolve().parts
  if len(parts) >= 3 and parts[-1] == "workspace" and parts[-3] == "users":
    return parts[-2]
  return ""


def _effective_user_id(workspace_dir: Path, *, user_id: str) -> str:
  explicit = str(user_id or "").strip()
  inferred = _workspace_user_id(workspace_dir)
  if explicit and inferred and explicit != inferred:
    raise ValueError("artifact index user_id does not match workspace owner")
  return explicit or inferred


def _classification_source(artifact: Any) -> str:
  if artifact.origin_kind is not None or artifact.visibility is not None or artifact.origin_ref is not None:
    return "sidecar"
  if artifact.research_file_id is not None:
    return "unresolved_research_file"
  return "legacy_default"


def _payload_classification_source(payload: dict[str, Any], *, research_file_id: int | None) -> str:
  if payload.get("origin_kind") is not None or payload.get("visibility") is not None or payload.get("origin_ref") is not None:
    return "sidecar"
  if research_file_id is not None or "research_file_id" in payload:
    return "unresolved_research_file"
  return "legacy_default"


def _skill_artifact_path_fields(
  *,
  workspace_dir: Path,
  sidecar_path: Path,
  payload: dict[str, Any],
) -> dict[str, str | None]:
  artifact_ref = _relative_to_workspace(workspace_dir, sidecar_path)
  parts = artifact_ref.split("/")
  if len(parts) != 4 or parts[0] != "artifacts" or not parts[-1].endswith(".json"):
    raise ValueError("skill artifact sidecar must be under artifacts/{ticker}/{skill}/ or artifacts/_portfolio/{skill}/")
  artifact_id = _text_or_none(payload.get("artifact_id")) or Path(parts[-1]).stem
  if parts[1] == "_portfolio":
    return {
      "artifact_ref": artifact_ref,
      "artifact_id": artifact_id,
      "scope": "portfolio",
      "ticker": None,
      "skill": _text_or_none(payload.get("skill")) or parts[2],
    }
  if parts[1].startswith("_"):
    raise ValueError("reserved artifact directory is not a skill artifact sidecar")
  raw_ticker = _text_or_none(payload.get("ticker")) or parts[1]
  try:
    indexed_ticker = canonicalize_ticker(raw_ticker)
  except ValueError:
    # Preserve indexability of legacy invalid records; exact readers still
    # reject invalid request tickers at their public lookup boundary.
    indexed_ticker = raw_ticker.strip().upper()
  return {
    "artifact_ref": artifact_ref,
    "artifact_id": artifact_id,
    "scope": "ticker",
    "ticker": indexed_ticker,
    "skill": _text_or_none(payload.get("skill")) or parts[2],
  }


def _skill_payload_ref(*, workspace_dir: Path, payload: dict[str, Any]) -> str | None:
  raw_ref = _text_or_none(payload.get("binary_artifact_path") or payload.get("payload_ref"))
  if raw_ref is None:
    return None
  candidate = Path(raw_ref)
  if candidate.is_absolute():
    return _relative_to_workspace(workspace_dir, candidate)
  return _relative_to_workspace(workspace_dir, Path(workspace_dir) / raw_ref)


def _relative_to_workspace(workspace_dir: Path, path: Path) -> str:
  return _ensure_under_workspace(path, workspace_dir).relative_to(Path(workspace_dir).resolve()).as_posix()


def _ensure_under_workspace(path: Path, workspace_dir: Path) -> Path:
  workspace_root = Path(workspace_dir).resolve()
  resolved_path = Path(path).resolve()
  try:
    resolved_path.relative_to(workspace_root)
  except ValueError as exc:
    raise ValueError("resolved artifact index path escapes workspace") from exc
  return resolved_path


def _sha256_file(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_or_none(value: Any) -> str | None:
  if value is None:
    return None
  return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_json_object(path: Path) -> dict[str, Any]:
  with Path(path).open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
  if not isinstance(payload, dict):
    raise ValueError("artifact sidecar payload must be a JSON object")
  return payload


def _int_or_none(value: Any) -> int | None:
  if value is None:
    return None
  try:
    return int(value)
  except (TypeError, ValueError):
    return None


def _text_or_none(value: Any) -> str | None:
  if value is None:
    return None
  text = str(value).strip()
  return text or None


def _mtime_ts(path: Path) -> str:
  return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _now_ts() -> str:
  return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
  "artifact_sidecar_index_path",
  "get_artifact_sidecar_index_row",
  "get_artifact_sidecar_index_row_by_ref",
  "list_artifact_sidecar_index_rows",
  "mark_artifact_sidecar_index_row_stale",
  "register_dashboard_artifact_sidecar",
  "register_html_artifact_sidecar",
  "register_skill_artifact_sidecar",
  "register_ui_blocks_payload_sidecar",
  "reconcile_ui_blocks_index",
]
