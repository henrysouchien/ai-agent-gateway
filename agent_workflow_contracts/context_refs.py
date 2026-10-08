"""Exact workflow refs shared by pipeline stages and named child admission.

The producer's readback mints artifact refs; consumers render only the keys their
skill declares. No artifact lookup or model-authored reference copying lives here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol


class RequiredContext(Protocol):
  @property
  def required_context(self) -> tuple[str, ...]: ...


def read_back_refs(primary_fms_result: Mapping[str, Any]) -> dict[str, Any]:
  """Name a door's exact refs, including the three-field skill-artifact ref.

  The consumer's own door validates the ref. Skill artifact sidecar fields are
  not part of its identity and must not travel in the minted ref. A door that
  published a ModelVersion names it as ``model_version_ref``, so a later
  publication replaces an earlier one and a consumer prices the version the
  workflow published last; the build report also names the SourceBundle it
  compiled as ``source_bundle_id``.

  A skill-artifact read (``get_skill_artifact``) returns the sidecar the caller
  selected at ``artifact``; it names only that artifact's three-field ref. Its
  ``*_ref`` keys and published ModelVersion are what the artifact recorded when
  it was written, not what the workflow holds now, so they are not minted.
  """
  artifact = primary_fms_result.get("artifact")
  refs: dict[str, Any] = _skill_artifact_ref(artifact) if isinstance(artifact, Mapping) else {}
  readback = primary_fms_result.get("readback")
  if not isinstance(readback, Mapping):
    return refs
  refs.update(
    (str(key), value)
    for key, value in readback.items()
    if str(key).endswith("_ref") and value is not None
  )
  refs.update(_skill_artifact_ref(readback))
  refs.update(_published_model_version_refs(readback))
  return refs


def _skill_artifact_ref(sidecar: Mapping[str, Any]) -> dict[str, Any]:
  """``<skill>_ref`` as exactly ``{skill, artifact_id, content_sha256}``."""
  skill = sidecar.get("skill")
  artifact_id = sidecar.get("artifact_id")
  content_sha256 = sidecar.get("content_sha256")
  if not (
    isinstance(skill, str)
    and skill
    and isinstance(artifact_id, str)
    and artifact_id
    and isinstance(content_sha256, str)
    and content_sha256
  ):
    return {}
  return {
    f"{skill.replace('-', '_')}_ref": {
      "skill": skill,
      "artifact_id": artifact_id,
      "content_sha256": content_sha256,
    },
  }


def _published_model_version_refs(readback: Mapping[str, Any]) -> dict[str, Any]:
  """The exact ModelVersion a door published, and the build's SourceBundle.

  Two producers publish a ModelVersion: the build-model report door, whose
  ``run-bound/1`` typed payload names the build's version and SourceBundle (its
  failed arm names neither), and a model-writer door, whose sidecar provenance
  names the successor it published with ``write_target: model_version``.
  """
  refs: dict[str, Any] = {}
  typed_outputs = readback.get("typed_outputs")
  if isinstance(typed_outputs, Mapping) and typed_outputs.get("schema_version") == "run-bound/1":
    model_version = typed_outputs.get("model_version")
    if isinstance(model_version, Mapping):
      refs["model_version_ref"] = {
        "model_id": model_version.get("model_id"),
        "version": model_version.get("version"),
      }
    source = typed_outputs.get("source")
    if isinstance(source, Mapping) and source.get("bundle_id") is not None:
      refs["source_bundle_id"] = source.get("bundle_id")
  provenance = readback.get("provenance")
  if isinstance(provenance, Mapping) and provenance.get("write_target") == "model_version":
    refs["model_version_ref"] = {
      "model_id": provenance.get("model_id"),
      "version": provenance.get("model_version"),
    }
  return refs


def declared_ref_lines(
  skill_definition: RequiredContext,
  refs: Mapping[str, Any],
) -> tuple[str, ...]:
  """Render declared refs from the caller's server-held bag, without lookup.

  Missing-key policy belongs to the caller: pipeline context doors answer for
  absent upstream refs; named child admission refuses before spawning a child.
  """
  bound = [key for key in skill_definition.required_context if key in refs]
  if not bound:
    return ()
  return (
    "Workflow-injected refs from prior tool results; pass each one unchanged where this skill names it:",
    *(
      f"{key}={json.dumps(refs[key], sort_keys=True, separators=(',', ':'))}"
      for key in bound
    ),
  )
