from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, Union

if TYPE_CHECKING:
  from .multi_user.billing import SessionUsageSummary


RunId: TypeAlias = str
Confidence: TypeAlias = Literal["HIGH", "MEDIUM", "LOW"]
DataSource: TypeAlias = Literal["live", "fixture"]
ArtifactErrorCode: TypeAlias = Literal[
  "validation",
  "missing_contract",
  "schema_drift",
  "tool_write_failed",
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
  "delegated_auto_approved",
  "user_denied",
  "relay_policy_denied",
  "headless_auto_deny",
  "headless_hook_approved",
  "session_cache_approved",
  "approval_timeout",
]
RecapTrigger: TypeAlias = Literal["turn_end", "explicit", "session_gc"]
RecapFailureType: TypeAlias = Literal[
  "terminal_error",
  "artifact_failed",
  "artifact_unavailable",
  "budget_exceeded",
  "max_turns_reached",
]
DEFAULT_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS: frozenset[int] = frozenset({DEFAULT_SCHEMA_VERSION})


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
class SkillResultCapturedEvent:
  skill_run_id: RunId
  skill: str
  ticker: str | None
  exit_code: int
  outcome: str
  status: str
  gate_code: str | None
  artifact_refs: list[str]
  proposal_ids: list[str]
  verdict_echo: dict[str, Any] | None
  fms_results: list[dict[str, Any]]
  artifact_events: list[dict[str, Any]]
  output_memory_file: str | None
  cost_usd: float | None
  duration_s: float | None
  error: str | None
  warnings: list[str]
  type: Literal["skill_result_captured"] = field(default="skill_result_captured", init=False)


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
class ArtifactUpdatedEvent:
  skill_run_id: RunId
  ticker: str
  skill: str
  artifact_id: str
  contract_name: str
  partial_view_model: dict[str, Any]
  ts: float
  type: Literal["artifact_updated"] = field(default="artifact_updated", init=False)


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
  ticker: str | None
  skill: str
  error_code: ArtifactErrorCode
  error_detail: str
  source_path: str | None
  ts: float
  tool_call_id: str | None = None
  type: Literal["artifact_failed"] = field(default="artifact_failed", init=False)


@dataclass(frozen=True)
class ArtifactUnavailableEvent:
  ticker: str | None
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


@dataclass(frozen=True)
class RecapArtifact:
  artifact_id: str
  skill: str
  contract_name: str
  ticker: str | None
  artifact_path: str
  emitted_at_seq: int
  ts: float


@dataclass(frozen=True)
class RecapVerdict:
  skill_run_id: str
  skill: str
  ticker: str
  verdict_token: str
  confidence: Confidence | None
  materiality_cushion: float | None
  one_line_summary: str
  emitted_at_seq: int
  ts: float


@dataclass(frozen=True)
class RecapApproval:
  tool_call_id: str
  tool_name: str
  outcome: ApprovalOutcome
  decision_source: ApprovalDecisionSource
  allow_tool_type_applied: bool
  emitted_at_seq: int
  ts: float


@dataclass(frozen=True)
class ToolCallsSummary:
  total_calls: int
  successes: int
  errors: int
  by_tool_name: dict[str, int]
  by_server: dict[str, int]


@dataclass(frozen=True)
class RecapFailure:
  failure_type: RecapFailureType
  detail: str
  emitted_at_seq: int
  ts: float


@dataclass(frozen=True)
class SessionRecapEvent:
  session_id: str
  seq_range: tuple[int, int]
  started_at: float
  ended_at: float
  trigger: RecapTrigger
  artifacts: list[RecapArtifact]
  verdicts: list[RecapVerdict]
  approvals: list[RecapApproval]
  tool_calls_summary: ToolCallsSummary
  failures: list[RecapFailure]
  usage: SessionUsageSummary | None
  ts: float
  type: Literal["session_recap"] = field(default="session_recap", init=False)


TypedEvent = Union[
  SkillRunStartedEvent,
  SkillResultCapturedEvent,
  ArtifactReadyEvent,
  ArtifactUpdatedEvent,
  TypedRecommendationsExtractedEvent,
  AggregateReadyEvent,
  ArtifactFailedEvent,
  ArtifactUnavailableEvent,
  ToolApprovalRequestEvent,
  ToolApprovalDecidedEvent,
  SessionRecapEvent,
]

TYPED_EVENT_TYPES = frozenset(
  {
    "skill_run_started",
    "skill_result_captured",
    "artifact_ready",
    "artifact_updated",
    "typed_recommendations_extracted",
    "aggregate_ready",
    "artifact_failed",
    "artifact_unavailable",
    "tool_approval_request",
    "tool_approval_decided",
    "session_recap",
  }
)

RUN_SCOPED_EVENT_TYPES = frozenset(
  {
    "skill_run_started",
    "skill_result_captured",
    "artifact_ready",
    "artifact_updated",
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
  if event_type == "skill_result_captured":
    verdict_echo = payload.get("verdict_echo")
    return SkillResultCapturedEvent(
      skill_run_id=str(payload["skill_run_id"]),
      skill=str(payload["skill"]),
      ticker=_optional_str(payload.get("ticker")),
      exit_code=int(payload["exit_code"]),
      outcome=str(payload["outcome"]),
      status=str(payload.get("status") or payload["outcome"]),
      gate_code=_optional_str(payload.get("gate_code")),
      artifact_refs=_string_list(payload.get("artifact_refs")),
      proposal_ids=_string_list(payload.get("proposal_ids")),
      verdict_echo=dict(verdict_echo) if isinstance(verdict_echo, dict) else None,
      fms_results=_mapping_list(payload.get("fms_results", []), "skill_result_captured.fms_results"),
      artifact_events=_mapping_list(
        payload.get("artifact_events", []),
        "skill_result_captured.artifact_events",
      ),
      output_memory_file=_optional_str(payload.get("output_memory_file")),
      cost_usd=_optional_float(payload.get("cost_usd")),
      duration_s=_optional_float(payload.get("duration_s")),
      error=_optional_str(payload.get("error")),
      warnings=_string_list(payload.get("warnings")),
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
  if event_type == "artifact_updated":
    partial_view_model = payload.get("partial_view_model", {})
    if not isinstance(partial_view_model, dict):
      raise ValueError("artifact_updated.partial_view_model must be a mapping")
    return ArtifactUpdatedEvent(
      skill_run_id=str(payload["skill_run_id"]),
      ticker=str(payload["ticker"]),
      skill=str(payload["skill"]),
      artifact_id=str(payload["artifact_id"]),
      contract_name=str(payload["contract_name"]),
      partial_view_model=partial_view_model,
      ts=float(payload["ts"]),
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
      ticker=_optional_str(payload.get("ticker")),
      skill=str(payload["skill"]),
      error_code=str(payload["error_code"]),  # type: ignore[arg-type]
      error_detail=str(payload["error_detail"]),
      source_path=_optional_str(payload.get("source_path")),
      ts=float(payload["ts"]),
      tool_call_id=_optional_str(payload.get("tool_call_id")),
    )
  if event_type == "artifact_unavailable":
    return ArtifactUnavailableEvent(
      ticker=_optional_str(payload.get("ticker")),
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
  if event_type == "session_recap":
    tool_calls_summary = payload["tool_calls_summary"]
    if not isinstance(tool_calls_summary, dict):
      raise ValueError("session_recap.tool_calls_summary must be a mapping")
    return SessionRecapEvent(
      session_id=str(payload["session_id"]),
      seq_range=_int_pair(payload["seq_range"]),
      started_at=float(payload["started_at"]),
      ended_at=float(payload["ended_at"]),
      trigger=_recap_trigger(payload["trigger"]),
      artifacts=[
        _recap_artifact(item)
        for item in _mapping_list(payload["artifacts"], "session_recap.artifacts")
      ],
      verdicts=[
        _recap_verdict(item)
        for item in _mapping_list(payload["verdicts"], "session_recap.verdicts")
      ],
      approvals=[
        _recap_approval(item)
        for item in _mapping_list(payload["approvals"], "session_recap.approvals")
      ],
      tool_calls_summary=ToolCallsSummary(
        total_calls=int(tool_calls_summary["total_calls"]),
        successes=int(tool_calls_summary["successes"]),
        errors=int(tool_calls_summary["errors"]),
        by_tool_name=_int_dict(
          tool_calls_summary["by_tool_name"],
          "session_recap.tool_calls_summary.by_tool_name",
        ),
        by_server=_int_dict(
          tool_calls_summary["by_server"],
          "session_recap.tool_calls_summary.by_server",
        ),
      ),
      failures=[
        _recap_failure(item)
        for item in _mapping_list(payload["failures"], "session_recap.failures")
      ],
      usage=_session_usage_summary(payload.get("usage")),
      ts=_event_ts(payload),
    )
  raise ValueError(f"Unknown typed event: {event_type!r}")


def _recap_artifact(payload: dict[str, Any]) -> RecapArtifact:
  return RecapArtifact(
    artifact_id=str(payload["artifact_id"]),
    skill=str(payload["skill"]),
    contract_name=str(payload["contract_name"]),
    ticker=_optional_str(payload.get("ticker")),
    artifact_path=str(payload["artifact_path"]),
    emitted_at_seq=int(payload["emitted_at_seq"]),
    ts=float(payload["ts"]),
  )


def _recap_verdict(payload: dict[str, Any]) -> RecapVerdict:
  return RecapVerdict(
    skill_run_id=str(payload["skill_run_id"]),
    skill=str(payload["skill"]),
    ticker=str(payload["ticker"]),
    verdict_token=str(payload["verdict_token"]),
    confidence=_confidence(payload.get("confidence")),
    materiality_cushion=_optional_float(payload.get("materiality_cushion")),
    one_line_summary=str(payload["one_line_summary"]),
    emitted_at_seq=int(payload["emitted_at_seq"]),
    ts=float(payload["ts"]),
  )


def _recap_approval(payload: dict[str, Any]) -> RecapApproval:
  return RecapApproval(
    tool_call_id=str(payload["tool_call_id"]),
    tool_name=str(payload["tool_name"]),
    outcome=_approval_outcome(payload["outcome"]),
    decision_source=_approval_decision_source(payload["decision_source"]),
    allow_tool_type_applied=bool(payload["allow_tool_type_applied"]),
    emitted_at_seq=int(payload["emitted_at_seq"]),
    ts=float(payload["ts"]),
  )


def _recap_failure(payload: dict[str, Any]) -> RecapFailure:
  return RecapFailure(
    failure_type=_recap_failure_type(payload["failure_type"]),
    detail=str(payload["detail"]),
    emitted_at_seq=int(payload["emitted_at_seq"]),
    ts=float(payload["ts"]),
  )


def _session_usage_summary(value: Any) -> SessionUsageSummary | None:
  from .multi_user.billing import SessionUsageSummary as _SessionUsageSummary

  if value is None:
    return None
  if isinstance(value, _SessionUsageSummary):
    return value
  if not isinstance(value, dict):
    raise ValueError("session_recap.usage must be a mapping or null")
  return _SessionUsageSummary(
    user_id=str(value["user_id"]),
    session_id=str(value["session_id"]),
    request_id=str(value["request_id"]),
    input_tokens=int(value["input_tokens"]),
    output_tokens=int(value["output_tokens"]),
    cache_read_tokens=int(value["cache_read_tokens"]),
    cache_creation_tokens=int(value["cache_creation_tokens"]),
    cost=float(value["cost"]),
    turns=int(value["turns"]),
    channel=_optional_str(value.get("channel")),
    started_at=float(value["started_at"]),
    ended_at=float(value["ended_at"]),
    drain_complete=bool(value.get("drain_complete", True)),
    in_flight_task_count=int(value.get("in_flight_task_count", 0)),
    product_id=_optional_str(value.get("product_id")),
    model=_optional_str(value.get("model")),
    provider=_optional_str(value.get("provider")),
    context_surfaces=_mapping_list(value.get("context_surfaces", []), "session_recap.usage.context_surfaces"),
  )


def _mapping_list(value: Any, field_name: str) -> list[dict[str, Any]]:
  if not isinstance(value, list):
    raise ValueError(f"{field_name} must be a list")
  if not all(isinstance(item, dict) for item in value):
    raise ValueError(f"{field_name} entries must be mappings")
  return value


def _int_pair(value: Any) -> tuple[int, int]:
  if not isinstance(value, (list, tuple)) or len(value) != 2:
    raise ValueError("session_recap.seq_range must contain exactly two integers")
  return (int(value[0]), int(value[1]))


def _int_dict(value: Any, field_name: str) -> dict[str, int]:
  if not isinstance(value, dict):
    raise ValueError(f"{field_name} must be a mapping")
  return {str(key): int(count) for key, count in value.items()}


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


def _confidence(value: Any) -> Confidence | None:
  if value is None:
    return None
  if value in {"HIGH", "MEDIUM", "LOW"}:
    return str(value)  # type: ignore[return-value]
  raise ValueError(f"Unknown recap confidence: {value!r}")


def _approval_outcome(value: Any) -> ApprovalOutcome:
  if value in {"approved", "denied", "timeout"}:
    return str(value)  # type: ignore[return-value]
  raise ValueError(f"Unknown approval outcome: {value!r}")


def _approval_decision_source(value: Any) -> ApprovalDecisionSource:
  if value in {
    "user_approved",
    "delegated_auto_approved",
    "user_denied",
    "relay_policy_denied",
    "headless_auto_deny",
    "headless_hook_approved",
    "session_cache_approved",
    "approval_timeout",
  }:
    return str(value)  # type: ignore[return-value]
  raise ValueError(f"Unknown approval decision source: {value!r}")


def _recap_trigger(value: Any) -> RecapTrigger:
  if value in {"turn_end", "explicit", "session_gc"}:
    return str(value)  # type: ignore[return-value]
  raise ValueError(f"Unknown session recap trigger: {value!r}")


def _recap_failure_type(value: Any) -> RecapFailureType:
  if value in {
    "terminal_error",
    "artifact_failed",
    "artifact_unavailable",
    "budget_exceeded",
    "max_turns_reached",
  }:
    return str(value)  # type: ignore[return-value]
  raise ValueError(f"Unknown session recap failure type: {value!r}")


__all__ = [
  "AggregateReadyEvent",
  "AggregateReadyTrigger",
  "ApprovalDecisionSource",
  "ApprovalOutcome",
  "ArtifactErrorCode",
  "ArtifactFailedEvent",
  "ArtifactReadyEvent",
  "ArtifactUpdatedEvent",
  "ArtifactUnavailableEvent",
  "ArtifactUnavailableReason",
  "Confidence",
  "DataSource",
  "DEFAULT_SCHEMA_VERSION",
  "RUN_SCOPED_EVENT_TYPES",
  "RecapApproval",
  "RecapArtifact",
  "RecapFailure",
  "RecapFailureType",
  "RecapTrigger",
  "RecapVerdict",
  "RunId",
  "SessionRecapEvent",
  "SkillResultCapturedEvent",
  "SkillRunStartedEvent",
  "SUPPORTED_SCHEMA_VERSIONS",
  "TYPED_EVENT_TYPES",
  "TypedEvent",
  "ToolCallsSummary",
  "ToolApprovalDecidedEvent",
  "ToolApprovalRequestEvent",
  "TypedRecommendationsExtractedEvent",
  "event_from_dict",
  "event_to_dict",
]
