from __future__ import annotations

import json as json_mod
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .agent_session_log import _atomic_write_sidecar
from .events import event_to_dict
from .product_config import gateway_product_id
from .session import GatewaySession, SessionStream
from .session_recap import compute_recap, compute_recap_from_events
from .server_models import _SIDECAR_SLUG_RE


def _now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _sidecar_slug(value: str | None) -> str | None:
  if value is None:
    return None
  slug = _SIDECAR_SLUG_RE.sub("-", str(value).strip().lower().replace("_", "-")).strip("-")[:64]
  return slug or None


def _maybe_write_chat_log_meta(
  transcript_dir: Path,
  session_id: str,
  *,
  user_id: str | None,
  channel: str | None,
  atomic_write_sidecar: Callable[[Path, Dict[str, Any]], None] = _atomic_write_sidecar,
  now_iso: Callable[[], str] = _now_iso,
  product_id_resolver: Callable[[], str | None] = gateway_product_id,
  sidecar_slug: Callable[[str | None], str | None] = _sidecar_slug,
) -> None:
  meta_path = transcript_dir / f"{session_id}.meta.json"
  if meta_path.exists():
    try:
      now = time.time()
      os.utime(meta_path, (now, now))
    except OSError:
      pass
    return
  atomic_write_sidecar(
    meta_path,
    {
      "schema_version": 1,
      "agent_session_id": session_id,
      "agent_id": None,
      "user_id": user_id,
      "product_id": product_id_resolver() or None,
      "file_kind": None,
      "channel": sidecar_slug(channel),
      "profile": None,
      "created_at": now_iso(),
    },
  )


def _write_transcript(
  transcript_dir: Optional[Path],
  session_id: str,
  entry: Dict[str, Any],
  *,
  user_id: str | None = None,
  channel: str | None = None,
  write_chat_log_meta: Callable[..., None] = _maybe_write_chat_log_meta,
  warn: Callable[..., None] | None = None,
) -> None:
  if transcript_dir is None:
    return
  try:
    write_chat_log_meta(transcript_dir, session_id, user_id=user_id, channel=channel)
  except Exception:
    if warn is not None:
      warn("Chat log sidecar write failed for %s (telemetry-only)", session_id, exc_info=True)
  payload = dict(entry)
  payload["ts"] = time.time()
  path = transcript_dir / f"{session_id}.jsonl"
  try:
    with open(path, "a", encoding="utf-8") as handle:
      handle.write(json_mod.dumps(payload, default=str) + "\n")
  except Exception:
    pass
  try:
    meta_path = transcript_dir / f"{session_id}.meta.json"
    if meta_path.exists():
      transcript_mtime = path.stat().st_mtime
      os.utime(meta_path, (transcript_mtime, transcript_mtime))
  except OSError:
    pass


def _cleanup_old_transcripts(
  transcript_dir: Optional[Path],
  retention_days: int,
  *,
  now: float | None = None,
) -> int:
  if transcript_dir is None or retention_days <= 0 or not transcript_dir.exists():
    return 0

  cutoff = (time.time() if now is None else now) - (retention_days * 86400)
  removed = 0
  transcript_freshness: dict[str, float] = {}
  for transcript in transcript_dir.glob("*.jsonl"):
    try:
      transcript_freshness[transcript.name.removesuffix(".jsonl")] = transcript.stat().st_mtime
    except OSError:
      continue

  for path in transcript_dir.iterdir():
    if not path.is_file():
      continue
    if path.suffix != ".jsonl" and not path.name.endswith(".meta.json"):
      continue
    try:
      effective_mtime = path.stat().st_mtime
      if path.name.endswith(".meta.json"):
        session_key = path.name.removesuffix(".meta.json")
        effective_mtime = max(effective_mtime, transcript_freshness.get(session_key, 0.0))
      if effective_mtime <= cutoff:
        path.unlink()
        removed += 1
    except OSError:
      continue
  return removed


def _compute_session_recap_payload(
  session: GatewaySession,
  active_turn: SessionStream,
  *,
  trigger: str,
) -> Dict[str, Any]:
  recap = compute_recap(
    active_turn.event_log,
    session_id=session.session_id,
    started_at=float(session.created_at),
    trigger=trigger,  # type: ignore[arg-type]
    usage=getattr(session, "cached_usage", None),
  )
  return event_to_dict(recap)


def _read_session_transcript_events(
  transcript_dir: Optional[Path],
  session_id: str,
) -> list[Dict[str, Any]]:
  if transcript_dir is None:
    return []
  path = transcript_dir / f"{session_id}.jsonl"
  if not path.exists():
    return []

  events: list[Dict[str, Any]] = []
  try:
    lines = path.read_text(encoding="utf-8").splitlines()
  except OSError:
    return []

  for line in lines:
    if not line.strip():
      continue
    try:
      payload = json_mod.loads(line)
    except json_mod.JSONDecodeError:
      continue
    if not isinstance(payload, dict):
      continue
    if payload.get("type") == "session_recap":
      continue
    events.append(dict(payload))
  return events


def _compute_cumulative_session_recap_payload(
  session: GatewaySession,
  active_turn: SessionStream | None,
  transcript_dir: Optional[Path],
  *,
  trigger: str,
  event_for_wire: Callable[[Any, Any], Dict[str, Any]],
) -> Dict[str, Any]:
  events = _read_session_transcript_events(transcript_dir, session.session_id)
  if active_turn is not None:
    written_seqs = active_turn.transcript_written_seqs if transcript_dir is not None else set()
    for entry in active_turn.event_log.entries:
      if entry.seq in written_seqs:
        continue
      event = event_for_wire(entry, active_turn.event_log)
      if event.get("type") == "session_recap":
        continue
      events.append(event)

  recap = compute_recap_from_events(
    events,
    session_id=session.session_id,
    started_at=float(session.created_at),
    trigger=trigger,  # type: ignore[arg-type]
    usage=getattr(session, "cached_usage", None),
  )
  return event_to_dict(recap)
