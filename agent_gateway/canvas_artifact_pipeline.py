"""Validation, compilation, persistence, and event pipeline for Canvas artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Sequence

from schema.canvas_artifact import CanvasArtifact, StaticExports
from schema.thesis_shared_slice import SourceRecord

from . import canvas_kit_contract
from .canvas_artifact_events import emit_canvas_artifact_ready
from .canvas_artifact_store import write_canvas_artifact
from .canvas_build_environment import (
  CanvasBuildFailure,
  CanvasBuildPreflight,
  build_canvas_bundle,
  build_canvas_bundle_async,
)


_SCRIPT_END_RE = re.compile(br"</script", re.IGNORECASE)
_BARE_IMPORT_RE = re.compile(br"\b(?:import\s|require\s*\()")


def _failure(stage: str, code: str, message: str, repair_hint: str) -> dict[str, Any]:
  return {
    "validation_failed": {
      "stage": stage,
      "diagnostics": [{"code": code, "message": message, "repair_hint": repair_hint}],
    }
  }


def emit_canvas_artifact(
  *,
  workspace_dir: Path,
  preflight: CanvasBuildPreflight,
  title: str,
  purpose: str,
  summary: str,
  tsx_source: str,
  copy_as_markdown: str,
  source_skill: str,
  skill_run_id: str,
  ticker: str | None = None,
  session_id: str | None = None,
  sources: Sequence[SourceRecord] = (),
  copy_as_prompt: str | None = None,
  copy_as_json: dict[str, Any] | None = None,
  research_file_id: int | None = None,
  control_run_id: str | None = None,
  user_id: str = "",
  emit_event: Callable[[dict[str, Any]], None] | None = None,
  _built_bundle: bytes | None = None,
) -> dict[str, Any]:
  """Run the locked stage sequence and return accepted or normalized diagnostics."""

  source_bytes = tsx_source.encode("utf-8")
  limit_values = canvas_kit_contract.limits()
  if len(source_bytes) > limit_values["source_max_bytes"]:
    return _failure(
      "size_cap", "source_size_cap_exceeded",
      f"Canvas source is {len(source_bytes)} bytes; maximum is {limit_values['source_max_bytes']}.",
      "Reduce repeated source and keep analytical data compact.",
    )
  if _built_bundle is None:
    try:
      bundle = build_canvas_bundle(tsx_source, preflight)
    except CanvasBuildFailure as exc:
      return exc.payload()
  else:
    bundle = _built_bundle
  if len(bundle) > limit_values["bundle_max_bytes"]:
    return _failure(
      "bundle_size_cap", "bundle_size_cap_exceeded",
      f"Canvas bundle is {len(bundle)} bytes; maximum is {limit_values['bundle_max_bytes']}.",
      "Reduce embedded data or source complexity.",
    )
  if _SCRIPT_END_RE.search(source_bytes) or _SCRIPT_END_RE.search(bundle):
    return _failure(
      "bundle", "script_end_forbidden",
      "Canvas bundle contains a case-insensitive </script raw-text terminator.",
      "Remove the raw-text terminator from string data and retry.",
    )
  if _BARE_IMPORT_RE.search(bundle):
    return _failure(
      "bundle", "bare_import_survived",
      "Canvas bundle retained a bare module import.",
      "Use only the three Canvas runtime imports.",
    )

  now = datetime.now(timezone.utc)
  artifact_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(8)}"
  artifact = CanvasArtifact(
    artifact_id=artifact_id,
    title=title,
    purpose=purpose,
    source_ref=f"{artifact_id}.tsx",
    source_digest=hashlib.sha256(source_bytes).hexdigest(),
    bundle_ref=f"{artifact_id}.bundle.js",
    bundle_digest=hashlib.sha256(bundle).hexdigest(),
    toolchain_version=preflight.toolchain_version,
    kit_contract_version=canvas_kit_contract.contract_version(),
    summary=summary,
    ticker=ticker,
    session_id=session_id,
    source_skill=source_skill,
    sources=list(sources),
    exports=StaticExports(
      copy_as_prompt=copy_as_prompt,
      copy_as_markdown=copy_as_markdown,
      copy_as_json=copy_as_json,
    ),
    ts=now.isoformat(),
    research_file_id=research_file_id,
    control_run_id=control_run_id,
    origin_kind=None if research_file_id is not None else "product",
    visibility=None if research_file_id is not None else "default",
  )
  write_canvas_artifact(
    workspace_dir=workspace_dir, artifact=artifact, source=tsx_source,
    bundle=bundle, user_id=user_id,
  )
  event = emit_canvas_artifact_ready(
    artifact_id=artifact_id, skill_run_id=skill_run_id, ticker=ticker,
    scope="ticker" if ticker else "portfolio",
  )
  if emit_event is not None:
    emit_event(event)
  return {
    "artifact_id": artifact_id,
    "status": "ok",
    "artifact_path": f"artifacts/_canvas/{artifact_id}.json",
    "bundle_digest": artifact.bundle_digest,
  }


async def emit_canvas_artifact_async(**kwargs: Any) -> dict[str, Any]:
  """Cancellation-safe live-handler entry point with the same locked pipeline."""

  source = str(kwargs["tsx_source"])
  source_bytes = source.encode("utf-8")
  source_cap = canvas_kit_contract.limits()["source_max_bytes"]
  if len(source_bytes) > source_cap:
    return _failure(
      "size_cap", "source_size_cap_exceeded",
      f"Canvas source is {len(source_bytes)} bytes; maximum is {source_cap}.",
      "Reduce repeated source and keep analytical data compact.",
    )
  try:
    bundle = await build_canvas_bundle_async(source, kwargs["preflight"])
  except CanvasBuildFailure as exc:
    return exc.payload()
  return emit_canvas_artifact(**kwargs, _built_bundle=bundle)


__all__ = ["emit_canvas_artifact", "emit_canvas_artifact_async"]
