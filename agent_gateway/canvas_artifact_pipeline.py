"""Validation, compilation, persistence, and event pipeline for Canvas artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Mapping, Sequence

from pydantic import ValidationError

from schema.canvas_artifact import (
  CanvasArtifact,
  CanvasArtifactPurpose,
  StaticExports,
)
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


_SOURCE_REPAIR_HINT = (
  "Repair the named field against the sources schema in the tool description; "
  "do not delete the source ledger."
)


def _coerce_sources(
  raw_sources: Any,
) -> tuple[list[SourceRecord] | None, dict[str, Any] | None]:
  """Convert caller-authored source payloads into the typed ledger, or reject them."""

  if raw_sources is None:
    return [], None
  # Imported below the empty-input returns so a source-free canvas emit keeps working in a
  # gateway-only topology, where the api package is not on sys.path.
  from research.source_identity import normalize_source_record_payload

  if not isinstance(raw_sources, (list, tuple)):
    return None, _failure(
      "sources", "sources_not_a_list",
      f"sources must be an array of source records; received {type(raw_sources).__name__}.",
      _SOURCE_REPAIR_HINT,
    )
  records: list[SourceRecord] = []
  for index, value in enumerate(raw_sources):
    try:
      records.append(SourceRecord.model_validate(normalize_source_record_payload(value)))
    except (ValidationError, ValueError, TypeError) as exc:
      return None, _failure(
        "sources", "source_record_invalid",
        f"sources[{index}]: {exc}",
        _SOURCE_REPAIR_HINT,
      )
  return records, None


def _validate_inputs(
  tsx_source: str, raw_sources: Any,
) -> tuple[list[SourceRecord] | None, dict[str, Any] | None]:
  """Run the caller-input stages (`sources`, then `size_cap`) shared by both entry points."""

  sources, failure = _coerce_sources(raw_sources)
  if failure is not None:
    return None, failure
  source_bytes = len(tsx_source.encode("utf-8"))
  source_cap = canvas_kit_contract.limits()["source_max_bytes"]
  if source_bytes > source_cap:
    return None, _failure(
      "size_cap", "source_size_cap_exceeded",
      f"Canvas source is {source_bytes} bytes; maximum is {source_cap}.",
      "Reduce repeated source and keep analytical data compact.",
    )
  return sources, None


def emit_canvas_artifact(
  *,
  workspace_dir: Path,
  preflight: CanvasBuildPreflight,
  title: str,
  purpose: CanvasArtifactPurpose,
  summary: str,
  tsx_source: str,
  copy_as_markdown: str,
  source_skill: str,
  skill_run_id: str,
  ticker: str | None = None,
  session_id: str | None = None,
  sources: Sequence[Mapping[str, Any]] = (),
  copy_as_prompt: str | None = None,
  copy_as_json: dict[str, Any] | None = None,
  research_file_id: int | None = None,
  control_run_id: str | None = None,
  user_id: str = "",
  emit_event: Callable[[dict[str, Any]], None] | None = None,
  _built_bundle: bytes | None = None,
) -> dict[str, Any]:
  """Run the locked stage sequence and return accepted or normalized diagnostics."""

  source_records, failure = _validate_inputs(tsx_source, sources)
  if failure is not None:
    return failure
  source_bytes = tsx_source.encode("utf-8")
  limit_values = canvas_kit_contract.limits()
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
    sources=list(source_records or []),
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
  # Validated here to reject a malformed ledger before the expensive bundle build. The
  # coerced records are discarded and emit_canvas_artifact coerces the same payload again;
  # that is the same owner running an idempotent, non-mutating projection twice, not a
  # second writer, so it is left as duplicated work rather than threaded through.
  _, failure = _validate_inputs(source, kwargs.get("sources"))
  if failure is not None:
    return failure
  try:
    bundle = await build_canvas_bundle_async(source, kwargs["preflight"])
  except CanvasBuildFailure as exc:
    return exc.payload()
  return emit_canvas_artifact(**kwargs, _built_bundle=bundle)


__all__ = ["emit_canvas_artifact", "emit_canvas_artifact_async"]
