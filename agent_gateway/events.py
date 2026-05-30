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
ApprovalOutcome: TypeAlias = Literal["approved", "denied", "timeout"]
ApprovalDecisionSource: TypeAlias = Literal[
  "user_approved",
  "user_denied",
  "headless_auto_deny",
  "headless_hook_approved",
  "session_cache_approved",
  "approval_timeout",
]


@dataclass(frozen=True)
class SkillRunStartedEvent:
  skill_run_id: RunId
  skill: str
  ticker: str | None
  ts: float
  scope: Literal["ticker", "portfolio"] = "ticker"
  portfolio_id: str | None = None
  type: Literal["skill_run_started"] = field(default="skill_run_started", init=False)


@dataclass(frozen=True)
class VerdictEmittedEvent:
  skill_run_id: RunId
  skill: str
  ticker: str | None
  verdict_token: str
  confidence: Confidence | None
  materiality_cushion: float | None
  one_line_summary: str
  ts: float
  scope: Literal["ticker", "portfolio"] = "ticker"
  portfolio_id: str | None = None
  type: Literal["verdict_emitted"] = field(default="verdict_emitted", init=False)


@dataclass(frozen=True)
class ArtifactReadyEvent:
  skill_run_id: RunId
  ticker: str | None
  skill: str
  artifact_id: str
  artifact_path: str
  binary_artifact_path: str | None
  contract_name: str
  data_source: DataSource
  ts: float
  scope: Literal["ticker", "portfolio"] = "ticker"
  portfolio_id: str | None = None
  type: Literal["artifact_ready"] = field(default="artifact_ready", init=False)


@dataclass(frozen=True)
class TypedRecommendationsExtractedEvent:
  skill: str
  workflow_name: str
  scope: Literal["ticker", "portfolio"]
  ticker: str | None
  portfolio_id: str | None
  recommendations_count: int
  verdict_code: str | None
  validation_errors: list[str]
  warnings: list[str]
  source_artifact_path: str
  ts: float
  type: Literal["typed_recommendations_extracted"] = field(default="typed_recommendations_extracted", init=False)


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


@dataclass(frozen=True)
class ToolApprovalRequestEvent:
  tool_call_id: str
  nonce: str
  tool_name: str
  tool_input: dict[str, Any]
  resolved_qualifier: str
  reason: str
  allow_persistent_approval: bool
  ts: float
  type: Literal["tool_approval_request"] = field(default="tool_approval_request", init=False)


@dataclass(frozen=True)
class ToolApprovalDecidedEvent:
  tool_call_id: str
  tool_name: str
  outcome: ApprovalOutcome
  decision_source: ApprovalDecisionSource
  allow_tool_type_applied: bool
  ts: float
  type: Literal["tool_approval_decided"] = field(default="tool_approval_decided", init=False)


TypedEvent = Union[
  SkillRunStartedEvent,
  VerdictEmittedEvent,
  ArtifactReadyEvent,
  TypedRecommendationsExtractedEvent,
  AggregateReadyEvent,
  ArtifactFailedEvent,
  ArtifactUnavailableEvent,
  ToolApprovalRequestEvent,
  ToolApprovalDecidedEvent,
]

TYPED_EVENT_TYPES = frozenset(
  {
    "skill_run_started",
    "verdict_emitted",
    "artifact_ready",
    "typed_recommendations_extracted",
    "aggregate_ready",
    "artifact_failed",
    "artifact_unavailable",
    "tool_approval_request",
    "tool_approval_decided",
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
      ticker=_optional_str(payload.get("ticker")),
      ts=float(payload["ts"]),
      scope=str(payload.get("scope", "ticker")),  # type: ignore[arg-type]
      portfolio_id=_optional_str(payload.get("portfolio_id")),
    )
  if event_type == "verdict_emitted":
    confidence = payload.get("confidence")
    return VerdictEmittedEvent(
      skill_run_id=str(payload["skill_run_id"]),
      skill=str(payload["skill"]),
      ticker=_optional_str(payload.get("ticker")),
      verdict_token=str(payload["verdict_token"]),
      confidence=str(confidence) if confidence is not None else None,  # type: ignore[arg-type]
      materiality_cushion=_optional_float(payload.get("materiality_cushion")),
      one_line_summary=str(payload["one_line_summary"]),
      ts=float(payload["ts"]),
      scope=str(payload.get("scope", "ticker")),  # type: ignore[arg-type]
      portfolio_id=_optional_str(payload.get("portfolio_id")),
    )
  if event_type == "artifact_ready":
    return ArtifactReadyEvent(
      skill_run_id=str(payload["skill_run_id"]),
      ticker=_optional_str(payload.get("ticker")),
      skill=str(payload["skill"]),
      artifact_id=str(payload["artifact_id"]),
      artifact_path=str(payload["artifact_path"]),
      binary_artifact_path=_optional_str(payload.get("binary_artifact_path")),
      contract_name=str(payload["contract_name"]),
      data_source=str(payload["data_source"]),  # type: ignore[arg-type]
      ts=float(payload["ts"]),
      scope=str(payload.get("scope", "ticker")),  # type: ignore[arg-type]
      portfolio_id=_optional_str(payload.get("portfolio_id")),
    )
  if event_type == "typed_recommendations_extracted":
    return TypedRecommendationsExtractedEvent(
      skill=str(payload["skill"]),
      workflow_name=str(payload["workflow_name"]),
      scope=str(payload["scope"]),  # type: ignore[arg-type]
      ticker=_optional_str(payload.get("ticker")),
      portfolio_id=_optional_str(payload.get("portfolio_id")),
      recommendations_count=int(payload["recommendations_count"]),
      verdict_code=_optional_str(payload.get("verdict_code")),
      validation_errors=_string_list(payload.get("validation_errors")),
      warnings=_string_list(payload.get("warnings")),
      source_artifact_path=str(payload["source_artifact_path"]),
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
  if event_type == "tool_approval_request":
    tool_input = payload.get("tool_input", {})
    if not isinstance(tool_input, dict):
      raise ValueError("tool_approval_request.tool_input must be a mapping")
    return ToolApprovalRequestEvent(
      tool_call_id=str(payload["tool_call_id"]),
      nonce=str(payload["nonce"]),
      tool_name=str(payload["tool_name"]),
      tool_input=tool_input,
      resolved_qualifier=str(payload.get("resolved_qualifier", "")),
      reason=str(payload.get("reason", "")),
      allow_persistent_approval=bool(payload.get("allow_persistent_approval", False)),
      ts=_event_ts(payload),
    )
  if event_type == "tool_approval_decided":
    return ToolApprovalDecidedEvent(
      tool_call_id=str(payload["tool_call_id"]),
      tool_name=str(payload["tool_name"]),
      outcome=_approval_outcome(payload["outcome"]),
      decision_source=_approval_decision_source(payload["decision_source"]),
      allow_tool_type_applied=bool(payload.get("allow_tool_type_applied", False)),
      ts=_event_ts(payload),
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


def _string_list(value: Any) -> list[str]:
  if value is None:
    return []
  if not isinstance(value, list):
    raise ValueError("typed event list field must be a list")
  return [str(item) for item in value]


def _event_ts(payload: dict[str, Any]) -> float:
  if payload.get("ts") is None:
    return 0.0
  return float(payload["ts"])


def _approval_outcome(value: Any) -> ApprovalOutcome:
  if value in {"approved", "denied", "timeout"}:
    return str(value)  # type: ignore[return-value]
  raise ValueError(f"Unknown approval outcome: {value!r}")


def _approval_decision_source(value: Any) -> ApprovalDecisionSource:
  if value in {
    "user_approved",
    "user_denied",
    "headless_auto_deny",
    "headless_hook_approved",
    "session_cache_approved",
    "approval_timeout",
  }:
    return str(value)  # type: ignore[return-value]
  raise ValueError(f"Unknown approval decision source: {value!r}")


__all__ = [
  "AggregateReadyEvent",
  "AggregateReadyTrigger",
  "ApprovalDecisionSource",
  "ApprovalOutcome",
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
  "ToolApprovalDecidedEvent",
  "ToolApprovalRequestEvent",
  "TypedRecommendationsExtractedEvent",
  "VerdictEmittedEvent",
  "event_from_dict",
  "event_to_dict",
]
