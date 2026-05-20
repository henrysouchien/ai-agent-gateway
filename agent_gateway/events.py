from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias, Union


RunId: TypeAlias = str
Confidence: TypeAlias = Literal["HIGH", "MEDIUM", "LOW"]
DataSource: TypeAlias = Literal["live", "fixture"]
ArtifactErrorCode: TypeAlias = Literal[
  "yaml_parse",
  "validation",
  "missing_contract",
  "schema_drift",
  "other",
]
ArtifactUnavailableReason: TypeAlias = Literal[
  "no_runs_yet",
  "stale",
  "fixture_only",
  "auth_blocked",
]
AggregateTriggerKind: TypeAlias = Literal["artifact_ready", "tool_response"]


@dataclass(frozen=True)
class SkillRunStartedEvent:
  skill_run_id: RunId
  skill: str
  ticker: str
  ts: float
  type: Literal["skill_run_started"] = field(default="skill_run_started", init=False)


@dataclass(frozen=True)
class VerdictEmittedEvent:
  skill_run_id: RunId
  skill: str
  ticker: str
  verdict_token: str
  confidence: Confidence | None
  materiality_cushion: float | None
  one_line_summary: str
  ts: float
  type: Literal["verdict_emitted"] = field(default="verdict_emitted", init=False)


@dataclass(frozen=True)
class ArtifactReadyEvent:
  skill_run_id: RunId
  ticker: str
  skill: str
  artifact_id: str
  artifact_path: str
  binary_artifact_path: str | None
  contract_name: str
  data_source: DataSource
  ts: float
  type: Literal["artifact_ready"] = field(default="artifact_ready", init=False)


@dataclass(frozen=True)
class AggregateReadyTrigger:
  kind: AggregateTriggerKind
  source: str


@dataclass(frozen=True)
class AggregateReadyEvent:
  skill_run_id: RunId
  ticker: str
  view_model_id: str
  trigger: AggregateReadyTrigger
  sources_complete: bool
  ts: float
  type: Literal["aggregate_ready"] = field(default="aggregate_ready", init=False)


@dataclass(frozen=True)
class ArtifactFailedEvent:
  skill_run_id: RunId
  ticker: str
  skill: str
  error_code: ArtifactErrorCode
  error_detail: str
  source_path: str
  ts: float
  type: Literal["artifact_failed"] = field(default="artifact_failed", init=False)


@dataclass(frozen=True)
class ArtifactUnavailableEvent:
  ticker: str
  skill: str
  reason: ArtifactUnavailableReason
  affordance: str
  ts: float
  type: Literal["artifact_unavailable"] = field(default="artifact_unavailable", init=False)


TypedEvent = Union[
  SkillRunStartedEvent,
  VerdictEmittedEvent,
  ArtifactReadyEvent,
  AggregateReadyEvent,
  ArtifactFailedEvent,
  ArtifactUnavailableEvent,
]

TYPED_EVENT_TYPES = frozenset(
  {
    "skill_run_started",
    "verdict_emitted",
    "artifact_ready",
    "aggregate_ready",
    "artifact_failed",
    "artifact_unavailable",
  }
)

RUN_SCOPED_EVENT_TYPES = frozenset(
  {
    "skill_run_started",
    "verdict_emitted",
    "artifact_ready",
    "aggregate_ready",
    "artifact_failed",
  }
)


def event_to_dict(event: TypedEvent) -> dict[str, Any]:
  payload = asdict(event)
  event_type = payload.pop("type")
  return {"type": event_type, **payload}


def event_from_dict(payload: dict[str, Any]) -> TypedEvent:
  event_type = payload.get("type")
  if event_type == "skill_run_started":
    return SkillRunStartedEvent(
      skill_run_id=str(payload["skill_run_id"]),
      skill=str(payload["skill"]),
      ticker=str(payload["ticker"]),
      ts=float(payload["ts"]),
    )
  if event_type == "verdict_emitted":
    confidence = payload.get("confidence")
    return VerdictEmittedEvent(
      skill_run_id=str(payload["skill_run_id"]),
      skill=str(payload["skill"]),
      ticker=str(payload["ticker"]),
      verdict_token=str(payload["verdict_token"]),
      confidence=str(confidence) if confidence is not None else None,  # type: ignore[arg-type]
      materiality_cushion=_optional_float(payload.get("materiality_cushion")),
      one_line_summary=str(payload["one_line_summary"]),
      ts=float(payload["ts"]),
    )
  if event_type == "artifact_ready":
    return ArtifactReadyEvent(
      skill_run_id=str(payload["skill_run_id"]),
      ticker=str(payload["ticker"]),
      skill=str(payload["skill"]),
      artifact_id=str(payload["artifact_id"]),
      artifact_path=str(payload["artifact_path"]),
      binary_artifact_path=_optional_str(payload.get("binary_artifact_path")),
      contract_name=str(payload["contract_name"]),
      data_source=str(payload["data_source"]),  # type: ignore[arg-type]
      ts=float(payload["ts"]),
    )
  if event_type == "aggregate_ready":
    trigger = payload["trigger"]
    if not isinstance(trigger, dict):
      raise ValueError("aggregate_ready.trigger must be a mapping")
    return AggregateReadyEvent(
      skill_run_id=str(payload["skill_run_id"]),
      ticker=str(payload["ticker"]),
      view_model_id=str(payload["view_model_id"]),
      trigger=AggregateReadyTrigger(
        kind=str(trigger["kind"]),  # type: ignore[arg-type]
        source=str(trigger["source"]),
      ),
      sources_complete=bool(payload["sources_complete"]),
      ts=float(payload["ts"]),
    )
  if event_type == "artifact_failed":
    return ArtifactFailedEvent(
      skill_run_id=str(payload["skill_run_id"]),
      ticker=str(payload["ticker"]),
      skill=str(payload["skill"]),
      error_code=str(payload["error_code"]),  # type: ignore[arg-type]
      error_detail=str(payload["error_detail"]),
      source_path=str(payload["source_path"]),
      ts=float(payload["ts"]),
    )
  if event_type == "artifact_unavailable":
    return ArtifactUnavailableEvent(
      ticker=str(payload["ticker"]),
      skill=str(payload["skill"]),
      reason=str(payload["reason"]),  # type: ignore[arg-type]
      affordance=str(payload["affordance"]),
      ts=float(payload["ts"]),
    )
  raise ValueError(f"Unknown typed event: {event_type!r}")


def _optional_float(value: Any) -> float | None:
  if value is None:
    return None
  return float(value)


def _optional_str(value: Any) -> str | None:
  if value is None:
    return None
  return str(value)


__all__ = [
  "AggregateReadyEvent",
  "AggregateReadyTrigger",
  "ArtifactErrorCode",
  "ArtifactFailedEvent",
  "ArtifactReadyEvent",
  "ArtifactUnavailableEvent",
  "ArtifactUnavailableReason",
  "Confidence",
  "DataSource",
  "RUN_SCOPED_EVENT_TYPES",
  "RunId",
  "SkillRunStartedEvent",
  "TYPED_EVENT_TYPES",
  "TypedEvent",
  "VerdictEmittedEvent",
  "event_from_dict",
  "event_to_dict",
]
