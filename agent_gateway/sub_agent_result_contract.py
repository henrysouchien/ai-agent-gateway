from __future__ import annotations

import datetime
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import (
  Annotated,
  Any,
  Literal,
  Mapping,
  Protocol,
  Sequence,
  TYPE_CHECKING,
  TypeAlias,
  runtime_checkable,
)
from uuid import UUID

from agent_workflow_contracts import (
  ActivityHandle,
  AnalyticalOutcome,
  AttemptRef,
  CanonicalProjection,
  ContentHandle,
  ContractRef,
  EvidenceObservation,
  ExecutionSettlement,
  LogicalTaskRef,
  NamedArtifact,
  ResultRequirement,
  TaskObservation,
  TaskResult,
  TaskResultProvenance,
  TaskResultValues,
  TERMINAL_NARRATIVE_CONTRACT,
  TranscriptHandle,
  UsageObservation,
  canonical_json_bytes,
  sha256_digest,
)
from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
  field_validator,
  model_validator,
  StringConstraints,
)

if TYPE_CHECKING:
  from agent.skills.authoring.learning_draft import (
    LEARNING_DRAFT_BODY_MAX_BYTES,
  )

DEFAULT_SUBAGENT_RETURN_CONTRACT = "report-base-v1"
EXPLORE_FINDINGS_RETURN_CONTRACT = "explore-findings-v1"
VERIFY_FINDING_RETURN_CONTRACT = "verify-finding-v1"
ARTIFACT_BRIEF_RETURN_CONTRACT = "artifact-brief-v1"
LEARNING_REPORT_RETURN_CONTRACT = "learning-report-v1"
FUNDAMENTAL_RESEARCH_PROPOSAL_RETURN_CONTRACT = (
  "fundamental-research-proposal-v1"
)

# These semantic caps replace the legacy assumption that compactness can be
# recovered by truncating one serialized response after the child finishes.
REPORT_SUMMARY_MAX_CHARS = 2_000
REPORT_FINDING_CLAIM_MAX_CHARS = 1_200
REPORT_FINDINGS_MAX_ITEMS = 12
REPORT_CONFIDENCE_MAX_CHARS = 64
REPORT_ARTIFACTS_MAX_ITEMS = 32
REPORT_CAVEAT_MAX_CHARS = 600
REPORT_CAVEATS_MAX_ITEMS = 8
REPORT_REFERENCE_VALUE_MAX_CHARS = 1_024
REPORT_SERIALIZED_MAX_BYTES = 24_000
CHILD_EVIDENCE_EXTERNALIZATION_MAX_BYTES = 64 * 1024 * 1024
CHILD_EVIDENCE_EXTERNALIZATION_MAX_NODES = 100_000
CHILD_EVIDENCE_EXTERNALIZATION_MAX_DEPTH = 64

def _learning_draft_body_max_bytes() -> int:
  from agent.skills.authoring.learning_draft import (
    LEARNING_DRAFT_BODY_MAX_BYTES,
  )

  return LEARNING_DRAFT_BODY_MAX_BYTES


def __getattr__(name: str) -> Any:
  if name == "LEARNING_DRAFT_BODY_MAX_BYTES":
    return _learning_draft_body_max_bytes()
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def report_contract_ref(contract_name: str) -> ContractRef:
  """Return the immutable wire identity for one gateway projection contract."""

  normalized = str(contract_name or "").strip()
  if not normalized:
    raise ValueError("report contract name must be non-empty")
  return ContractRef(
    namespace="agent-gateway",
    name=normalized,
    version="1.0",
    digest=sha256_digest({
      "namespace": "agent-gateway",
      "name": normalized,
      "version": "1.0",
    }),
  )


def terminal_narrative_content_handle(
  reference: FinalNarrativeArtifactReference,
) -> ContentHandle:
  """Project a private artifact locator to the public canonical handle."""

  if not isinstance(reference, FinalNarrativeArtifactReference):
    raise TypeError("terminal narrative requires a typed artifact reference")
  return ContentHandle(
    content_id=f"sha256:{reference.content_sha256}",
    content_sha256=reference.content_sha256,
    content_chars=reference.content_chars,
    content_bytes=reference.content_bytes,
    contract=TERMINAL_NARRATIVE_CONTRACT,
    media_type=reference.media_type,
    encoding="utf-8",
    retention="durable",
  )


def canonical_projection(
  *,
  contract: ContractRef,
  value: Any,
) -> CanonicalProjection:
  """Bind one exact JSON projection to its immutable content identity."""

  encoded = canonical_json_bytes(value)
  decoded = encoded.decode("utf-8")
  content_sha256 = hashlib.sha256(encoded).hexdigest()
  return CanonicalProjection(
    contract=contract,
    content=ContentHandle(
      content_id=f"sha256:{content_sha256}",
      content_sha256=content_sha256,
      content_chars=len(decoded),
      content_bytes=len(encoded),
      contract=contract,
      media_type="application/json",
      encoding="utf-8",
      retention="durable",
    ),
    inline_view=value,
  )


def _usage_observation(value: Mapping[str, Any] | None) -> UsageObservation:
  usage = value or {}

  def _counter(*names: str) -> int:
    for name in names:
      raw = usage.get(name)
      if type(raw) is int and raw >= 0:
        return raw
    return 0

  raw_cost = usage.get("cost_usd", usage.get("estimated_cost", 0.0))
  cost_usd = (
    float(raw_cost)
    if (
      not isinstance(raw_cost, bool)
      and isinstance(raw_cost, (int, float))
      and math.isfinite(float(raw_cost))
      and float(raw_cost) >= 0
    )
    else 0.0
  )

  return UsageObservation(
    input_tokens=_counter("input_tokens", "prompt_tokens"),
    output_tokens=_counter("output_tokens", "completion_tokens"),
    cached_input_tokens=_counter(
      "cached_input_tokens",
      "cache_read_input_tokens",
    ),
    tool_calls=_counter("tool_calls"),
    cost_usd=cost_usd,
  )


def build_task_result(
  *,
  logical_task: LogicalTaskRef,
  attempt: AttemptRef,
  requirement: ResultRequirement,
  provenance: TaskResultProvenance,
  execution: ExecutionSettlement,
  outcome: AnalyticalOutcome | None,
  terminal_narrative: FinalNarrativeArtifactReference | None,
  projection: CanonicalProjection | None,
  artifacts: Sequence[NamedArtifact] = (),
  observed_sources: Sequence[Any] = (),
  tools_used: Sequence[str] = (),
  usage: Mapping[str, Any] | None = None,
) -> TaskResult:
  """Build and validate the sole canonical child settlement envelope.

  Result-mode validation belongs at acquisition, while :class:`TaskResult`
  remains reusable by deterministic operations that have no agent narrative.
  No prose is parsed to infer an outcome or projection.
  """

  narrative_handle = (
    terminal_narrative_content_handle(terminal_narrative)
    if terminal_narrative is not None
    else None
  )
  successful = execution.status == "succeeded"
  if (
    successful
    and requirement.terminal_narrative == "required"
    and narrative_handle is None
  ):
    raise ValueError("result contract requires an exact terminal narrative")
  if requirement.terminal_narrative == "forbidden" and narrative_handle is not None:
    raise ValueError("result contract forbids a terminal narrative")
  if requirement.projection is None:
    if projection is not None:
      raise ValueError("result contract does not declare a projection")
  else:
    if successful and requirement.projection.required and projection is None:
      raise ValueError("result contract requires a canonical projection")
    if projection is not None and projection.contract != requirement.projection.contract:
      raise ValueError("projection contract does not match result requirement")
  if successful and requirement.outcome.required:
    if outcome is None:
      raise ValueError("result contract requires an analytical outcome")
    if outcome.assessment_source != requirement.outcome.source:
      raise ValueError("outcome assessment source does not match result requirement")
  elif outcome is not None and requirement.outcome.source != "none":
    if outcome.assessment_source != requirement.outcome.source:
      raise ValueError("optional outcome source does not match result requirement")
  if requirement.outcome.source == "none" and outcome is not None:
    if not (
      outcome.disposition == "not_assessed"
      and outcome.assessment_source == "none"
    ):
      raise ValueError("non-assessing result may only carry not_assessed")
  if execution.status != "succeeded" and (
    narrative_handle is not None or projection is not None or artifacts
  ):
    raise ValueError("non-successful execution cannot publish canonical values")

  values = TaskResultValues(
    terminal_narrative=narrative_handle,
    projection=projection,
    artifacts=tuple(artifacts),
  )
  evidence_value = EvidenceObservation(
    observed_sources=tuple(observed_sources),
    tools_used=tuple(dict.fromkeys(
      str(tool).strip() for tool in tools_used if str(tool).strip()
    )),
  )
  usage_payload = dict(usage or {})
  usage_payload.setdefault("tool_calls", len(tuple(tools_used)))
  observation_value = TaskObservation(
    transcript=TranscriptHandle(
      kind="child_transcript",
      owner_id=attempt.physical_task_id,
    ),
    activity=ActivityHandle(
      kind="child_activity",
      owner_id=attempt.physical_task_id,
    ),
    usage=_usage_observation(usage_payload),
  )
  identity_payload = {
    "schema_version": "2.0",
    "logical_task": logical_task.model_dump(mode="json"),
    "attempt": attempt.model_dump(mode="json"),
    "execution": execution.model_dump(mode="json"),
    "outcome": (
      outcome.model_dump(mode="json") if outcome is not None else None
    ),
    "evidence": evidence_value.model_dump(mode="json"),
    "values": values.model_dump(mode="json"),
    "observation": observation_value.model_dump(mode="json"),
    "provenance": provenance.model_dump(mode="json"),
  }
  result = TaskResult(
    task_result_id=f"task-result:{sha256_digest(identity_payload)}",
    logical_task=logical_task,
    attempt=attempt,
    execution=execution,
    outcome=outcome,
    evidence=evidence_value,
    values=values,
    observation=observation_value,
    provenance=provenance,
  )
  return result


NonEmptyReferenceValue: TypeAlias = Annotated[
  str,
  StringConstraints(strip_whitespace=True, min_length=1, max_length=REPORT_REFERENCE_VALUE_MAX_CHARS),
]
SummaryText: TypeAlias = Annotated[
  str,
  StringConstraints(strip_whitespace=True, min_length=1, max_length=REPORT_SUMMARY_MAX_CHARS),
]
FindingClaim: TypeAlias = Annotated[
  str,
  StringConstraints(strip_whitespace=True, min_length=1, max_length=REPORT_FINDING_CLAIM_MAX_CHARS),
]
ConfidenceText: TypeAlias = Annotated[
  str,
  StringConstraints(strip_whitespace=True, min_length=1, max_length=REPORT_CONFIDENCE_MAX_CHARS),
]
CaveatText: TypeAlias = Annotated[
  str,
  StringConstraints(strip_whitespace=True, min_length=1, max_length=REPORT_CAVEAT_MAX_CHARS),
]


class CitationReference(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  kind: Literal["citation"] = "citation"
  handle: NonEmptyReferenceValue
  retention: Literal["durable"] = "durable"


class FmsArtifactReference(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  kind: Literal["fms_artifact"] = "fms_artifact"
  artifact_ref: NonEmptyReferenceValue
  retention: Literal["durable"] = "durable"


class ArtifactReference(BaseModel):
  """Generic durable artifact-store reference.

  This is intentionally distinct from :class:`FmsArtifactReference`: promoted
  spill evidence is a platform artifact and must not acquire FMS semantics
  merely because both references are durable.
  """

  model_config = ConfigDict(extra="forbid", frozen=True)

  kind: Literal["artifact"] = "artifact"
  artifact_ref: NonEmptyReferenceValue
  retention: Literal["durable"] = "durable"


class FinalNarrativeArtifactReference(BaseModel):
  """Immutable exact terminal assistant narrative promoted at settlement."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  kind: Literal["final_narrative"] = "final_narrative"
  artifact_id: NonEmptyReferenceValue
  artifact_ref: NonEmptyReferenceValue
  content_sha256: Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
  ]
  content_chars: int = Field(ge=1)
  content_bytes: int = Field(ge=1)
  terminal_event_seq: int = Field(ge=1, le=9_007_199_254_740_991)
  media_type: Literal["text/plain; charset=utf-8"] = (
    "text/plain; charset=utf-8"
  )
  retention: Literal["durable"] = "durable"

  @model_validator(mode="after")
  def _validate_artifact_identity(self) -> FinalNarrativeArtifactReference:
    if not self.artifact_id.startswith("sha256:") or len(self.artifact_id) != 71:
      raise ValueError("final narrative artifact_id must be a sha256 identity")
    return self


class SpillFileReference(BaseModel):
  """Internal spill-store input retained for protected FMS overflow handling."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  kind: Literal["spill_file"] = "spill_file"
  path: NonEmptyReferenceValue
  retention: Literal["transient"] = "transient"


class PlanJournalReference(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)

  kind: Literal["plan_journal"] = "plan_journal"
  run_id: NonEmptyReferenceValue
  artifact_id: NonEmptyReferenceValue
  retention: Literal["run"] = "run"


ReportReference: TypeAlias = Annotated[
  (
    CitationReference
    | FmsArtifactReference
    | ArtifactReference
    | PlanJournalReference
  ),
  Field(discriminator="kind"),
]


class Finding(BaseModel):
  model_config = ConfigDict(extra="forbid")

  claim: FindingClaim
  evidence_ref: ReportReference | None = None
  confidence: ConfidenceText | None = None


class ReportBase(BaseModel):
  """Common projection implemented by every child report contract.

  ``extra="allow"`` is intentional: consumers of the common projection can
  validate specialized report instances without discarding their additional
  fields. The default compact contract below closes the shape again.
  """

  model_config = ConfigDict(extra="allow")

  summary: SummaryText
  findings: list[Finding] = Field(default_factory=list, max_length=REPORT_FINDINGS_MAX_ITEMS)
  artifacts: list[ReportReference] = Field(default_factory=list, max_length=REPORT_ARTIFACTS_MAX_ITEMS)
  caveats: list[CaveatText] = Field(default_factory=list, max_length=REPORT_CAVEATS_MAX_ITEMS)


class CompactReport(ReportBase):
  """Default report contract for generic and otherwise undeclared profiles."""

  model_config = ConfigDict(extra="forbid")


class ExploreFindingsReport(ReportBase):
  """Compact coverage report returned by the purpose-named explore role."""

  model_config = ConfigDict(extra="forbid")

  coverage: list[FindingClaim] = Field(max_length=REPORT_FINDINGS_MAX_ITEMS)
  follow_ups: list[FindingClaim] = Field(max_length=REPORT_CAVEATS_MAX_ITEMS)


class VerifyFindingReport(ReportBase):
  """Adversarial verdict for one claim under review."""

  model_config = ConfigDict(extra="forbid")

  target_claim: FindingClaim
  verdict: Literal[
    "supported",
    "weakened",
    "refuted",
    "insufficient_evidence",
  ]
  recommended_action: Literal["keep", "revise", "drop", "investigate"]


class ArtifactBriefReport(ReportBase):
  """Typed compression of one document, spill, or durable artifact."""

  model_config = ConfigDict(extra="forbid")

  artifact_title: FindingClaim
  key_points: list[FindingClaim] = Field(max_length=REPORT_FINDINGS_MAX_ITEMS)
  omissions: list[CaveatText] = Field(max_length=REPORT_CAVEATS_MAX_ITEMS)


class FundamentalResearchProposalReport(ReportBase):
  """Complete source-bearing judgment for the parent-owned FMS door.

  The child is intentionally responsible only for research and proposal
  construction.  The report door validates that the complete handoff object
  is present; the mutation-capable parent remains responsible for invoking
  ``fms_propose_fundamental_research`` and returning its durable receipt.
  """

  model_config = ConfigDict(extra="forbid")

  judgment: dict[str, Any]

  @field_validator("judgment")
  @classmethod
  def _validate_complete_judgment(
    cls,
    value: dict[str, Any],
  ) -> dict[str, Any]:
    required_fields = {
      "ticker",
      "verdict",
      "confidence",
      "research_date",
      "step_synthesis",
      "sources",
      "data_gaps",
      "step_coverage",
      "recommended_next_action",
    }
    missing = sorted(required_fields - set(value))
    if missing:
      raise ValueError(
        "fundamental-research judgment is missing required handoff fields: "
        + ", ".join(missing)
      )
    ticker = value.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
      raise ValueError("fundamental-research judgment ticker must be non-empty")
    if not isinstance(value.get("step_synthesis"), dict):
      raise ValueError(
        "fundamental-research judgment step_synthesis must be an object"
      )
    if (
      value.get("company_diligence") is not None
      and not isinstance(value.get("company_diligence"), dict)
    ):
      raise ValueError(
        "fundamental-research judgment company_diligence must be an object when present"
      )
    for field_name in ("sources", "data_gaps"):
      items = value.get(field_name)
      if not isinstance(items, list) or any(
        not isinstance(item, dict) for item in items
      ):
        raise ValueError(
          f"fundamental-research judgment {field_name} must be a list of objects"
        )
    return value


class LearningMemoryWrite(BaseModel):
  """One immutable learning-memory note claimed by the fork."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  path: NonEmptyReferenceValue
  summary: FindingClaim


class LearningDraftReference(BaseModel):
  """Reference to a learning draft written by the learning workflow."""

  model_config = ConfigDict(extra="forbid", frozen=True)

  path: NonEmptyReferenceValue
  draft_status: Literal["clean", "flagged"]
  findings: list[
    Literal[
      "description_too_short",
      "body_empty",
      "body_missing_markdown_heading",
      "amendment_missing_existing_reference",
    ]
  ] = Field(
    default_factory=list,
    max_length=4,
  )


class LearningSkillDraftCandidate(BaseModel):
  """Draft proposal before write, or its file reference after write."""

  model_config = ConfigDict(extra="forbid")

  name: Annotated[
    str,
    StringConstraints(
      strip_whitespace=True,
      min_length=1,
      max_length=64,
      pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    ),
  ]
  description: FindingClaim
  body: str | LearningDraftReference
  kind: Literal["new_skill", "amendment"]
  references_existing: NonEmptyReferenceValue | None = None

  @field_validator("body")
  @classmethod
  def _bound_raw_draft_body(
    cls,
    value: str | LearningDraftReference,
  ) -> str | LearningDraftReference:
    if isinstance(value, str):
      body_size = len(value.encode("utf-8"))
      body_max_bytes = _learning_draft_body_max_bytes()
      if body_size > body_max_bytes:
        raise ValueError(
          "body exceeds "
          f"{body_max_bytes} raw UTF-8 bytes ({body_size})"
        )
    return value


class LearningReport(ReportBase):
  """Structured outcome of a post-session self-learning fork."""

  model_config = ConfigDict(extra="forbid")

  decision: Literal["memory_update", "skill_draft", "both", "no_op"]
  memory_writes: list[LearningMemoryWrite] = Field(
    default_factory=list,
    max_length=REPORT_ARTIFACTS_MAX_ITEMS,
  )
  skill_draft_candidate: LearningSkillDraftCandidate | None = None
  rationale: SummaryText


def _evidence_fits_externalization_bound(
  *,
  usage: Mapping[str, Any] | None,
  tools_used: Sequence[str] | None,
  fms_results: Sequence[Mapping[str, Any]] | None,
  artifact_events: Sequence[Mapping[str, Any]] | None,
  max_bytes: int = CHILD_EVIDENCE_EXTERNALIZATION_MAX_BYTES,
) -> bool:
  remaining = max_bytes
  node_count = 0
  active_containers: set[int] = set()

  def _visit(
    value: Any,
    depth: int,
    *,
    mapping_key: bool = False,
  ) -> bool:
    nonlocal node_count, remaining
    node_count += 1
    if (
      node_count > CHILD_EVIDENCE_EXTERNALIZATION_MAX_NODES
      or depth > CHILD_EVIDENCE_EXTERNALIZATION_MAX_DEPTH
    ):
      return False
    if (
      mapping_key
      and isinstance(
        value,
        (Mapping, list, tuple, set, frozenset),
      )
    ):
      return False

    if value is None:
      remaining -= 4
    elif isinstance(value, bool):
      remaining -= 5
    elif isinstance(value, Enum):
      projected_value = value.value
      if projected_value is value:
        return False
      return _visit(
        projected_value,
        depth + 1,
        mapping_key=mapping_key,
      )
    elif isinstance(value, str):
      remaining = _subtract_json_utf8_string(
        remaining,
        value,
      )
    elif isinstance(value, (bytes, bytearray)):
      try:
        decoded = bytes(value).decode("utf-8")
      except UnicodeDecodeError:
        return False
      remaining = _subtract_json_utf8_string(
        remaining,
        decoded,
      )
    elif isinstance(value, int):
      decimal_digits = max(
        1,
        math.ceil(value.bit_length() * math.log10(2)),
      )
      integer_digit_limit = sys.get_int_max_str_digits()
      if (
        integer_digit_limit > 0
        and decimal_digits > integer_digit_limit
      ):
        return False
      remaining -= decimal_digits + (1 if value < 0 else 0)
    elif isinstance(value, float):
      remaining -= 24
    elif isinstance(value, Mapping):
      if (
        node_count + (2 * len(value))
        > CHILD_EVIDENCE_EXTERNALIZATION_MAX_NODES
      ):
        return False
      container_id = id(value)
      if container_id in active_containers:
        return False
      active_containers.add(container_id)
      remaining -= 2 + max(0, len(value) - 1)
      try:
        for key, item in value.items():
          if not _visit(
            key,
            depth + 1,
            mapping_key=True,
          ):
            return False
          if not _visit(item, depth + 1):
            return False
      finally:
        active_containers.discard(container_id)
    elif isinstance(value, (list, tuple)):
      if (
        node_count + len(value)
        > CHILD_EVIDENCE_EXTERNALIZATION_MAX_NODES
      ):
        return False
      container_id = id(value)
      if container_id in active_containers:
        return False
      active_containers.add(container_id)
      remaining -= 2 + max(0, len(value) - 1)
      try:
        for item in value:
          if not _visit(item, depth + 1):
            return False
      finally:
        active_containers.discard(container_id)
    elif isinstance(value, (set, frozenset)):
      # Set iteration is hash-seed dependent and cannot define one canonical
      # durable artifact identity. Producers must use an ordered sequence.
      return False
    elif isinstance(
      value,
      (
        datetime.date,
        datetime.time,
        datetime.timedelta,
        Decimal,
        Path,
        UUID,
      ),
    ):
      remaining = _subtract_json_utf8_string(
        remaining,
        str(value),
      )
    else:
      return False
    return remaining >= 0

  return all(
    _visit(value, 0)
    for value in (
      usage or {},
      tools_used or (),
      fms_results or (),
      artifact_events or (),
    )
  )


def _subtract_json_utf8_string(
  remaining: int,
  value: str,
) -> int:
  remaining -= 2
  for character in value:
    codepoint = ord(character)
    if character in {'"', "\\"}:
      remaining -= 2
    elif codepoint < 0x20:
      remaining -= 6
    elif codepoint < 0x80:
      remaining -= 1
    else:
      try:
        remaining -= len(character.encode("utf-8"))
      except UnicodeEncodeError:
        return -1
    if remaining < 0:
      break
  return remaining


def child_evidence_fits_externalization_bound(
  *,
  usage: Mapping[str, Any] | None,
  tools_used: Sequence[str] | None,
  fms_results: Sequence[Mapping[str, Any]] | None,
  artifact_events: Sequence[Mapping[str, Any]] | None,
) -> bool:
  """Check the canonical evidence resource contract without copying it."""

  try:
    return _evidence_fits_externalization_bound(
      usage=usage,
      tools_used=tools_used,
      fms_results=fms_results,
      artifact_events=artifact_events,
    )
  except Exception:
    # Evidence is untrusted tool output. A custom Mapping/Sequence may raise
    # from len(), iteration, or scalar projection; inspection failure must be
    # indistinguishable from an over-bound value and must never preserve the
    # raw object for a later copy or serialization attempt.
    return False


ContractValidationErrorCode: TypeAlias = Literal[
  "unknown_contract",
  "invalid_report",
  "report_too_large",
]


@dataclass(frozen=True)
class ContractValidationError:
  code: ContractValidationErrorCode
  message: str
  details: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContractValidationResult:
  report: ReportBase | None = None
  error: ContractValidationError | None = None

  def __post_init__(self) -> None:
    if (self.report is None) == (self.error is None):
      raise ValueError("ContractValidationResult requires exactly one of report or error")

  @property
  def ok(self) -> bool:
    return self.report is not None


@runtime_checkable
class ReportContractAdapter(Protocol):
  @property
  def contract_name(self) -> str: ...

  @property
  def report_type(self) -> type[ReportBase]: ...

  def validate_report(self, payload: Mapping[str, Any]) -> ContractValidationResult: ...


@runtime_checkable
class ReportContractRegistry(Protocol):
  def resolve(self, contract_name: str) -> ReportContractAdapter | None: ...

  def validate_report(
    self,
    contract_name: str,
    payload: Mapping[str, Any],
  ) -> ContractValidationResult: ...


def canonical_report_size_bytes(report: ReportBase) -> int:
  payload = report.model_dump(mode="json", serialize_as_any=True)
  return len(
    json.dumps(
      payload,
      ensure_ascii=True,
      allow_nan=False,
      separators=(",", ":"),
      sort_keys=True,
    ).encode("utf-8")
  )


__all__ = [
  "ARTIFACT_BRIEF_RETURN_CONTRACT",
  "ArtifactReference",
  "ArtifactBriefReport",
  "CHILD_EVIDENCE_EXTERNALIZATION_MAX_BYTES",
  "CHILD_EVIDENCE_EXTERNALIZATION_MAX_DEPTH",
  "CHILD_EVIDENCE_EXTERNALIZATION_MAX_NODES",
  "CitationReference",
  "CompactReport",
  "ContractValidationError",
  "ContractValidationErrorCode",
  "ContractValidationResult",
  "DEFAULT_SUBAGENT_RETURN_CONTRACT",
  "EXPLORE_FINDINGS_RETURN_CONTRACT",
  "ExploreFindingsReport",
  "Finding",
  "FinalNarrativeArtifactReference",
  "FUNDAMENTAL_RESEARCH_PROPOSAL_RETURN_CONTRACT",
  "FundamentalResearchProposalReport",
  "FmsArtifactReference",
  "LEARNING_DRAFT_BODY_MAX_BYTES",
  "LEARNING_REPORT_RETURN_CONTRACT",
  "LearningDraftReference",
  "LearningMemoryWrite",
  "LearningReport",
  "LearningSkillDraftCandidate",
  "PlanJournalReference",
  "REPORT_ARTIFACTS_MAX_ITEMS",
  "REPORT_CAVEATS_MAX_ITEMS",
  "REPORT_FINDINGS_MAX_ITEMS",
  "REPORT_SERIALIZED_MAX_BYTES",
  "ReportBase",
  "ReportContractAdapter",
  "ReportContractRegistry",
  "ReportReference",
  "VERIFY_FINDING_RETURN_CONTRACT",
  "VerifyFindingReport",
  "canonical_report_size_bytes",
  "canonical_projection",
  "build_task_result",
  "child_evidence_fits_externalization_bound",
  "report_contract_ref",
  "terminal_narrative_content_handle",
]
