"""Canonical, dependency-neutral agent/workflow wire models.

The models in this module contain no storage locators, credentials, or runtime
objects.  They are immutable wire values suitable for durable events and API
boundaries.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Annotated, Any, Literal, Mapping, TypeAlias

from pydantic import (
  BaseModel,
  ConfigDict,
  Field,
  JsonValue,
  StringConstraints,
  field_validator,
  model_validator,
)


class WireModel(BaseModel):
  model_config = ConfigDict(extra="forbid", frozen=True)


Name = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9._-]{0,127}$")]
Version = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
HexDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS = 262_144
OpaqueId = Annotated[
  str,
  StringConstraints(strip_whitespace=True, min_length=1, max_length=512),
]
NonEmptyText = Annotated[
  str,
  StringConstraints(
    strip_whitespace=True,
    min_length=1,
    max_length=AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS,
  ),
]

COMMON_CHILD_OPERATION_INSTRUCTIONS = (
  "You are a focused sub-agent working on behalf of another agent. Complete "
  "the admitted objective thoroughly. If evidence access fails or returns "
  "suspicious data, identify the limitation explicitly instead of silently "
  "proceeding. You cannot delegate to another agent."
)


def compose_operation_instructions(instructions: str) -> str:
  """Append the common child contract within the exact wire-text bound."""

  if type(instructions) is not str:
    raise TypeError("operation instructions must be a string")
  methodology = instructions.strip()
  if not methodology:
    raise ValueError("operation instructions must be non-empty")
  composed = f"{methodology}\n\n{COMMON_CHILD_OPERATION_INSTRUCTIONS}"
  if len(composed) > AGENT_OPERATION_INSTRUCTIONS_MAX_CHARS:
    raise ValueError(
      "composed operation instructions exceed the 262144-character wire bound"
    )
  return composed


_RAW_PATH = re.compile(r"^(?:/|~/|\.{1,2}/|file://|[A-Za-z]:[\\/])")
_SECRET_KEY = re.compile(
  r"(?:^|[_-])(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|"
  r"credential|private[_-]?key|authorization)(?:$|[_-])",
  re.IGNORECASE,
)


def canonical_json_bytes(value: Any) -> bytes:
  """Return the canonical UTF-8 JSON representation used for wire digests."""

  if isinstance(value, BaseModel):
    value = value.model_dump(mode="json", exclude_none=False)
  return json.dumps(
    value,
    ensure_ascii=False,
    allow_nan=False,
    sort_keys=True,
    separators=(",", ":"),
  ).encode("utf-8")


def sha256_digest(value: Any) -> str:
  return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def _reject_raw_path(value: str, *, field_name: str) -> str:
  if _RAW_PATH.match(value):
    raise ValueError(f"{field_name} must be a logical identifier, not a raw path")
  if "\x00" in value or "\n" in value or "\r" in value:
    raise ValueError(f"{field_name} must be a single logical identifier")
  return value


def _validate_safe_literal(value: JsonValue, *, key: str | None = None) -> None:
  if key is not None and _SECRET_KEY.search(key):
    raise ValueError("literal selectors cannot carry secret-bearing fields")
  if isinstance(value, dict):
    for child_key, child in value.items():
      _validate_safe_literal(child, key=child_key)
  elif isinstance(value, list):
    for child in value:
      _validate_safe_literal(child)


class ContractRef(WireModel):
  namespace: Name
  name: Name
  version: Version
  digest: Digest


SELECTED_CONTENT_UTF8_CONTRACT = ContractRef(
  namespace="agent-gateway",
  name="selected-content-utf8",
  version="1.0",
  digest="sha256:e8598f8c941b84145dbe018508c52f2c620867e8134f66c72c63e061da996aa9",
)


TERMINAL_NARRATIVE_CONTRACT = ContractRef(
  namespace="agent-gateway",
  name="terminal-narrative",
  version="1.0",
  digest=sha256_digest({
    "namespace": "agent-gateway",
    "name": "terminal-narrative",
    "version": "1.0",
    "canonical_encoding": "utf-8",
  }),
)


class AgentOperationRef(ContractRef):
  """Full immutable identity of an executable operation."""


class CapabilityBind(WireModel):
  """Complete, secret-free execution identity for one admitted capability."""

  schema_version: Literal["1.0"]
  capability_id: OpaqueId
  model_key: OpaqueId
  provider: OpaqueId
  upstream_model: OpaqueId
  adapter: OpaqueId
  protocol_profile: OpaqueId
  route: OpaqueId
  effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
  credential_principal: Literal["user", "service"]
  credential_ref: OpaqueId
  run_mode: Literal["interactive", "fleet", "batch", "autonomous", "cron"]
  registry_revision: Version
  policy_revision: Version
  selection_source: Literal[
    "explicit_user",
    "saved_preference",
    "channel_default",
    "capability_default",
    "internal_policy",
    "parent_binding",
  ]

  @field_validator("provider")
  @classmethod
  def _provider_family(cls, value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "agent-sdk":
      raise ValueError("provider must name a credential provider family")
    return normalized

  @field_validator(
    "capability_id",
    "model_key",
    "upstream_model",
    "adapter",
    "protocol_profile",
    "route",
    "credential_ref",
    "registry_revision",
    "policy_revision",
  )
  @classmethod
  def _canonical_bind_text(cls, value: str) -> str:
    return value.strip()

  def receipt(self) -> dict[str, str]:
    return self.model_dump(mode="json")

  @classmethod
  def from_receipt(cls, receipt: object) -> CapabilityBind:
    return cls.model_validate(receipt)


class ContentHandle(WireModel):
  content_id: Digest
  content_sha256: HexDigest
  content_bytes: int = Field(ge=0)
  content_chars: int | None = Field(default=None, ge=0)
  contract: ContractRef
  media_type: Annotated[str, StringConstraints(min_length=1, max_length=255)]
  encoding: Annotated[str, StringConstraints(min_length=1, max_length=32)] | None = None
  retention: Literal["durable", "workflow", "session", "transient"]

  @model_validator(mode="after")
  def _validate_content(self) -> ContentHandle:
    if self.content_id != f"sha256:{self.content_sha256}":
      raise ValueError("content_id must equal sha256:<content_sha256>")
    media = self.media_type.lower()
    textual = media.startswith("text/") or "json" in media or "xml" in media
    if textual and (self.content_chars is None or self.encoding is None):
      raise ValueError("textual content requires content_chars and encoding")
    return self


class ContentReadGrant(WireModel):
  kind: Literal["content_read_grant"] = "content_read_grant"
  grant_id: OpaqueId
  content_id: Digest
  scope: Literal["this_task", "direct_parent"]
  principal_id: OpaqueId


class ContextViewPolicy(WireModel):
  preferred: Literal["inline_exact", "semantic_excerpt", "content_read"]
  max_bytes: int = Field(ge=1)
  on_overflow: Literal["content_read", "semantic_excerpt", "fail"]


class InvocationArgumentSelector(WireModel):
  kind: Literal["invocation_argument_selector"] = "invocation_argument_selector"
  argument_name: Name


class LiteralSelector(WireModel):
  kind: Literal["literal_selector"] = "literal_selector"
  value: JsonValue

  @field_validator("value")
  @classmethod
  def _literal_is_not_authority(cls, value: JsonValue) -> JsonValue:
    _validate_safe_literal(value)
    return value


class SelectedContentSelector(WireModel):
  kind: Literal["selected_content_selector"] = "selected_content_selector"
  input_name: Name


class NodeValueSelector(WireModel):
  kind: Literal["node_value_selector"] = "node_value_selector"
  node_id: OpaqueId
  value_kind: Literal["narrative", "projection", "artifact"]
  artifact_name: Name | None = None
  projection_path: tuple[Name, ...] = ()

  @model_validator(mode="after")
  def _validate_value_selector(self) -> NodeValueSelector:
    _reject_raw_path(self.node_id, field_name="node_id")
    if self.value_kind == "artifact" and self.artifact_name is None:
      raise ValueError("artifact selection requires artifact_name")
    if self.value_kind != "artifact" and self.artifact_name is not None:
      raise ValueError("artifact_name is valid only for artifact selection")
    if self.value_kind != "projection" and self.projection_path:
      raise ValueError("projection_path is valid only for projection selection")
    return self


class PhaseOutputSelector(WireModel):
  kind: Literal["phase_output_selector"] = "phase_output_selector"
  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  output_name: Name


class DurableArtifactSelector(WireModel):
  kind: Literal["durable_artifact_selector"] = "durable_artifact_selector"
  artifact_id: OpaqueId

  @field_validator("artifact_id")
  @classmethod
  def _logical_artifact_id(cls, value: str) -> str:
    return _reject_raw_path(value, field_name="artifact_id")


RequestedDataSelector: TypeAlias = Annotated[
  InvocationArgumentSelector
  | LiteralSelector
  | SelectedContentSelector
  | NodeValueSelector
  | PhaseOutputSelector
  | DurableArtifactSelector,
  Field(discriminator="kind"),
]


class RequestedDataRef(WireModel):
  kind: Literal["requested_data"] = "requested_data"
  name: Name
  selector: RequestedDataSelector
  expected_contract: ContractRef
  context_policy: ContextViewPolicy | None = None


class OwnerBinding(WireModel):
  tenant_id: OpaqueId
  workflow_run_id: OpaqueId | None = None
  session_id: OpaqueId | None = None
  invocation_id: OpaqueId | None = None

  @model_validator(mode="after")
  def _has_scope(self) -> OwnerBinding:
    if not (self.workflow_run_id or self.session_id or self.invocation_id):
      raise ValueError("owner binding requires workflow, session, or invocation scope")
    return self


class AdmittedDataRef(WireModel):
  kind: Literal["admitted_data"] = "admitted_data"
  request: RequestedDataRef
  source_kind: Literal[
    "invocation_argument",
    "literal",
    "selected_content",
    "node_value",
    "phase_output",
    "durable_artifact",
  ]
  logical_source_id: OpaqueId
  owner: OwnerBinding
  actual_contract: ContractRef
  content: ContentHandle
  read_grant: ContentReadGrant | None = None

  @field_validator("logical_source_id")
  @classmethod
  def _logical_source(cls, value: str) -> str:
    return _reject_raw_path(value, field_name="logical_source_id")

  @model_validator(mode="after")
  def _contracts_match(self) -> AdmittedDataRef:
    if self.request.expected_contract != self.actual_contract:
      raise ValueError("actual contract must match the admitted expected contract")
    if self.content.contract != self.actual_contract:
      raise ValueError("content contract must match actual contract")
    if self.read_grant is not None and self.read_grant.content_id != self.content.content_id:
      raise ValueError("content read grant must address admitted content")
    return self


class ContextSourceRef(WireModel):
  kind: Literal["admitted_data_ref"] = "admitted_data_ref"
  logical_source_id: OpaqueId
  content_id: Digest


class InlineExactContextView(WireModel):
  kind: Literal["inline_exact"] = "inline_exact"
  source: ContextSourceRef
  content: JsonValue
  content_bytes: int = Field(ge=0)
  complete: Literal[True] = True


class SemanticExcerptContextView(WireModel):
  kind: Literal["semantic_excerpt"] = "semantic_excerpt"
  source: ContextSourceRef
  content: NonEmptyText
  content_bytes: int = Field(ge=1)
  selection_query: NonEmptyText
  complete: Literal[False] = False


class ContentReadContextView(WireModel):
  kind: Literal["content_read"] = "content_read"
  source: ContextSourceRef
  read_grant: ContentReadGrant
  complete: Literal[False] = False

  @model_validator(mode="after")
  def _grant_matches(self) -> ContentReadContextView:
    if self.source.content_id != self.read_grant.content_id:
      raise ValueError("context read grant must address source content")
    return self


ContextView: TypeAlias = Annotated[
  InlineExactContextView | SemanticExcerptContextView | ContentReadContextView,
  Field(discriminator="kind"),
]
ContextMaterialization: TypeAlias = ContextView


class AdmittedInputBinding(WireModel):
  name: Name
  source: AdmittedDataRef
  context: ContextMaterialization

  @model_validator(mode="after")
  def _context_matches(self) -> AdmittedInputBinding:
    if self.context.source.logical_source_id != self.source.logical_source_id:
      raise ValueError("materialized context must address admitted source")
    if self.context.source.content_id != self.source.content.content_id:
      raise ValueError("materialized context must address admitted content")
    return self


class TypedInputCapabilityBinding(WireModel):
  kind: Literal["typed_input"] = "typed_input"
  capability: OpaqueId
  input_name: Name
  input_contract: ContractRef


class LiveToolCapabilityBinding(WireModel):
  kind: Literal["live_tool"] = "live_tool"
  capability: OpaqueId
  route_id: OpaqueId
  tool_ids: tuple[OpaqueId, ...] = Field(min_length=1)


CapabilityBinding: TypeAlias = Annotated[
  TypedInputCapabilityBinding | LiveToolCapabilityBinding,
  Field(discriminator="kind"),
]


class ToolGrantEntry(WireModel):
  tool_id: OpaqueId
  route_id: OpaqueId
  effect: Literal["read", "propose", "write", "external_effect"]


class ToolGrant(WireModel):
  kind: Literal["tool_grant"] = "tool_grant"
  grant_id: OpaqueId
  tools: tuple[ToolGrantEntry, ...] = ()
  digest: Digest

  @model_validator(mode="after")
  def _unique_tools(self) -> ToolGrant:
    ids = [tool.tool_id for tool in self.tools]
    if len(ids) != len(set(ids)):
      raise ValueError("tool grant cannot contain duplicate tool IDs")
    return self


class WorkspaceGrant(WireModel):
  kind: Literal["workspace_grant"] = "workspace_grant"
  workspace_id: OpaqueId
  scope: Literal["read_only", "workspace_write", "model_write"]


class SemanticCapabilityRequirement(WireModel):
  name: OpaqueId
  required: bool = True
  binding_modes: tuple[Literal["typed_input", "live_tool"], ...] = Field(min_length=1)
  compatible_input_contracts: tuple[ContractRef, ...] = ()

  @model_validator(mode="after")
  def _typed_contracts_are_exact_and_canonical(
    self,
  ) -> SemanticCapabilityRequirement:
    identities = tuple(
      (item.namespace, item.name, item.version, item.digest)
      for item in self.compatible_input_contracts
    )
    if tuple(sorted(set(identities))) != identities:
      raise ValueError(
        "compatible_input_contracts must be sorted exact identities"
      )
    if self.compatible_input_contracts and "typed_input" not in self.binding_modes:
      raise ValueError(
        "compatible_input_contracts require typed_input binding mode"
      )
    return self


# ---------------------------------------------------------------------------
# One resolver, one artifact (T3-I11): the platform tool catalog, the exact
# authority an operation resolves to, and the visible Left when it cannot.
# ---------------------------------------------------------------------------


class ExecutionIdentity(WireModel):
  """The exact identity one admitted operation executes under (D-B6-1).

  Credential *material* never rides here: ``credential_handle_id`` names a
  server-held handle, never a secret.
  """

  kind: Literal["execution_identity"] = "execution_identity"
  tenant_id: OpaqueId
  credential_handle_id: OpaqueId | None = None


class CatalogToolEntry(WireModel):
  """One platform tool, described exactly once.

  ``capability`` is **singular** (D-B5-2), matching ``SemanticToolRoute``:
  a tool route targets at most one semantic capability.  ``effect`` is
  ``None`` when the platform cannot resolve one — such a tool is describable
  but never becomes authority.  ``success_signal`` / ``source_identity`` are
  declarative descriptors (never callables) read at the dispatch boundary.
  """

  kind: Literal["catalog_tool"] = "catalog_tool"
  tool_id: OpaqueId
  canonical_name: OpaqueId
  effect: Literal["read", "propose", "write", "external_effect"] | None = None
  server_id: OpaqueId | None = None
  capability: OpaqueId | None = None
  idempotent: bool | None = None
  success_signal: dict[str, JsonValue] | None = None
  source_identity: dict[str, JsonValue] | None = None


class PlatformToolCatalog(WireModel):
  """The exact snapshot of describable platform tools at one instant."""

  kind: Literal["platform_tool_catalog"] = "platform_tool_catalog"
  tools: tuple[CatalogToolEntry, ...] = ()

  @model_validator(mode="after")
  def _sorted_unique_tools(self) -> PlatformToolCatalog:
    ids = tuple(entry.tool_id for entry in self.tools)
    if tuple(sorted(set(ids))) != ids:
      raise ValueError("platform catalog tools must be sorted and unique")
    return self

  def entry(self, tool_id: str) -> CatalogToolEntry | None:
    for item in self.tools:
      if item.tool_id == tool_id:
        return item
    return None


class UnsatisfiedCapability(WireModel):
  """One declared semantic capability the live platform cannot satisfy."""

  kind: Literal["unsatisfied_capability"] = "unsatisfied_capability"
  capability: OpaqueId
  required: bool = True
  reason: Literal[
    "no_compatible_route",
    "unregistered_capability",
    "ambiguous_route",
  ]
  detail: Annotated[str, StringConstraints(min_length=1, max_length=2_048)]


class OperationUnavailable(WireModel):
  """The Left of ``resolve_operation_authority`` — a **visible** offer.

  An operation the platform cannot currently authorize is said out loud, not
  dropped: this is the value that replaces every silent catalog drop.
  """

  kind: Literal["operation_unavailable"] = "operation_unavailable"
  operation_name: Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
  ]
  code: Literal[
    "invalid_metadata",
    "missing_route",
    "policy_denial",
    "credential_unavailable",
    "version_digest_mismatch",
  ]
  detail: Annotated[str, StringConstraints(min_length=1, max_length=2_048)]
  unsatisfied: tuple[UnsatisfiedCapability, ...] = ()


class ResolvedAuthority(WireModel):
  """The frozen authority one operation resolved to, credentials excluded.

  ``bindings`` and ``grant`` are exactly the fields ``TaskAdmissionAuthority``
  projects; ``routes`` carries the catalog entries the grant was cut from, so
  the dispatcher allowlist is derived, never re-discovered.
  """

  kind: Literal["resolved_authority"] = "resolved_authority"
  operation_name: Annotated[
    str,
    StringConstraints(min_length=1, max_length=128),
  ]
  grant: ToolGrant
  bindings: tuple[CapabilityBinding, ...] = ()
  routes: tuple[CatalogToolEntry, ...] = ()
  identity: ExecutionIdentity | None = None

  @model_validator(mode="after")
  def _routes_cover_the_grant(self) -> ResolvedAuthority:
    route_ids = tuple(route.tool_id for route in self.routes)
    if tuple(sorted(set(route_ids))) != route_ids:
      raise ValueError("resolved routes must be sorted and unique")
    if set(route_ids) != {entry.tool_id for entry in self.grant.tools}:
      raise ValueError("resolved routes must be exactly the granted tools")
    return self


class EvidencePort(WireModel):
  """One declared multi-valued evidence input on an operation contract.

  An operation that cannot gather evidence itself declares where upstream
  results enter.  The catalog owns the fact that the port exists and its
  cardinality floor; the plan author selects only which upstream results
  feed it.
  """

  name: Name
  min_selections: int = Field(default=1, ge=1, le=8)
  max_selections: int = Field(default=8, ge=1, le=8)

  @model_validator(mode="after")
  def _floor_within_ceiling(self) -> EvidencePort:
    if self.max_selections < self.min_selections:
      raise ValueError("evidence port max_selections must be >= min_selections")
    return self


class AgentOperationSnapshot(WireModel):
  operation: AgentOperationRef
  methodology: ContractRef
  prompt: ContractRef
  description: Annotated[str, StringConstraints(min_length=1, max_length=2_048)]
  instructions: NonEmptyText
  execution_class: OpaqueId
  required_capabilities: tuple[SemanticCapabilityRequirement, ...] = ()
  workspace_scope: Literal["read_only", "workspace_write", "model_write"]
  required_context: tuple[Name, ...] = Field(default=(), max_length=8)
  # Absent means none: the field is omitted from dumps when empty so durable
  # snapshots and digests recorded before evidence ports existed replay
  # byte-identically.
  evidence_ports: tuple[EvidencePort, ...] = Field(
    default=(),
    max_length=4,
    exclude_if=lambda value: not value,
  )
  resumable: bool = False
  result_modes: tuple[Literal["narrative"], ...] = Field(min_length=1)
  projection_contracts: tuple[ContractRef, ...] = ()

  @model_validator(mode="after")
  def _agent_result_is_a_terminal_message(self) -> AgentOperationSnapshot:
    if self.result_modes != ("narrative",) or self.projection_contracts:
      raise ValueError(
        "agent operations return a normal terminal message; typed result "
        "records are materialized by runtime code"
      )
    port_names = tuple(port.name for port in self.evidence_ports)
    if tuple(sorted(set(port_names))) != port_names:
      raise ValueError("evidence_ports must be sorted and unique by name")
    if set(port_names) & set(self.required_context):
      raise ValueError("evidence port names cannot collide with required_context")
    return self


class AgentResumeMechanics(WireModel):
  """Exact admitted continuation mechanics for one logical child task."""

  resumable: bool
  max_chain_depth: int = Field(ge=0, le=100)
  transcript_strategy: Literal["durable_reconstruction"]
  prompt_strategy: Literal["reuse_exact"]
  tool_grant_strategy: Literal["reissue_exact"]
  control_message_strategy: Literal["admitted_exact"]

  @model_validator(mode="after")
  def _depth_matches_resumability(self) -> AgentResumeMechanics:
    if self.resumable == (self.max_chain_depth == 0):
      raise ValueError(
        "resumable execution requires a positive chain depth and "
        "non-resumable execution requires zero"
      )
    return self


class AgentExecutionSnapshot(WireModel):
  """Exact prompt and mechanics consumed by one admitted child attempt."""

  system_prompt: NonEmptyText
  admission_date: Annotated[
    str,
    StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}$"),
  ]
  persisted_methodology_state: JsonValue | None
  result_instructions: NonEmptyText
  max_turns: int | None = Field(default=None, ge=1, le=100)
  timeout_seconds: float | None = Field(
    default=None,
    gt=0,
    allow_inf_nan=False,
  )
  client_timeout_seconds: float = Field(gt=0, allow_inf_nan=False)
  max_tokens: int = Field(ge=1, le=256_000)
  cost_observation_threshold_usd: float | None = Field(
    default=None,
    gt=0,
    allow_inf_nan=False,
    description=(
      "Telemetry-only runaway-cost observation threshold. This value never "
      "authorizes, admits, interrupts, gates, or settles execution."
    ),
  )
  max_budget_usd: float | None = Field(
    default=None,
    gt=0,
    allow_inf_nan=False,
    exclude_if=lambda value: value is None,
  )
  resume_mechanics: AgentResumeMechanics
  resume_instruction: NonEmptyText | None = None

  @field_validator("max_budget_usd", mode="before")
  @classmethod
  def _validate_max_budget_usd(cls, value: Any) -> float | None:
    if value is None:
      return None
    if (
      isinstance(value, bool)
      or not isinstance(value, int | float)
      or not math.isfinite(float(value))
      or float(value) <= 0
    ):
      raise ValueError("max_budget_usd must be a finite positive number")
    return float(value)

  @model_validator(mode="after")
  def _prompt_contains_admitted_dynamic_instructions(
    self,
  ) -> AgentExecutionSnapshot:
    if self.admission_date not in self.system_prompt:
      raise ValueError("system prompt must contain its exact admission date")
    if self.result_instructions not in self.system_prompt:
      raise ValueError("system prompt must contain exact result instructions")
    if self.persisted_methodology_state is not None:
      _validate_safe_literal(self.persisted_methodology_state)
    return self


class ProjectionRequirement(WireModel):
  contract: ContractRef
  required: bool = True


class OutcomeRequirement(WireModel):
  required: Literal[False] = False
  source: Literal["none"] = "none"


class ResultRequirement(WireModel):
  """Runtime policy for an agent's ordinary terminal assistant message.

  This is code-owned admission metadata, not a response schema for the model.
  Any later structured projection must be produced by a deterministic runtime
  materializer or a separately invoked domain tool.
  """

  mode: Literal["narrative"] = "narrative"
  projection: None = None
  terminal_narrative: Literal["required"] = "required"
  outcome: OutcomeRequirement

  @model_validator(mode="after")
  def _mode_is_coherent(self) -> ResultRequirement:
    if self.outcome != OutcomeRequirement():
      raise ValueError("agent terminal messages cannot author typed outcomes")
    return self


class OutcomeRoute(WireModel):
  disposition: Literal["complete", "partial", "insufficient_evidence", "blocked", "not_assessed"]
  action: Literal["settle", "continue", "skip_dependents", "fail"]


class OutcomePolicy(WireModel):
  routes: tuple[OutcomeRoute, ...] = Field(min_length=1)

  @model_validator(mode="after")
  def _one_route_each(self) -> OutcomePolicy:
    values = [route.disposition for route in self.routes]
    if len(values) != len(set(values)):
      raise ValueError("outcome policy cannot duplicate dispositions")
    return self


class TaskSettlementAcceptancePolicy(WireModel):
  kind: Literal["task_settlement"] = "task_settlement"
  execution_statuses: tuple[Literal["succeeded", "failed", "interrupted", "cancelled", "skipped"], ...] = Field(min_length=1)
  outcome_dispositions: tuple[Literal["complete", "partial", "insufficient_evidence", "blocked", "not_assessed"], ...] = ()
  allow_missing_outcome: bool = False


class ValidationVerdictAcceptancePolicy(WireModel):
  kind: Literal["validation_verdict"] = "validation_verdict"
  execution_statuses: tuple[Literal["succeeded"], ...] = ("succeeded",)
  projection_contract: ContractRef
  accepted_verdicts: tuple[Literal["accepted", "needs_revision", "rejected", "insufficient_evidence"], ...] = Field(min_length=1)


DependencyAcceptancePolicy: TypeAlias = Annotated[
  TaskSettlementAcceptancePolicy | ValidationVerdictAcceptancePolicy,
  Field(discriminator="kind"),
]


class OrdinaryDelegationTaskRef(WireModel):
  kind: Literal["ordinary_delegation"] = "ordinary_delegation"
  delegation_id: OpaqueId
  operation: AgentOperationRef


class WorkflowNodeTaskRef(WireModel):
  kind: Literal["workflow_node"] = "workflow_node"
  workflow_run_id: OpaqueId
  plan_id: OpaqueId
  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  node_id: OpaqueId
  item_key: OpaqueId | None = None
  operation: AgentOperationRef


LogicalTaskRef: TypeAlias = Annotated[
  OrdinaryDelegationTaskRef | WorkflowNodeTaskRef,
  Field(discriminator="kind"),
]


class AttemptRef(WireModel):
  attempt_number: int = Field(ge=1)
  attempt_id: OpaqueId
  physical_task_id: OpaqueId
  restart_of_attempt_id: OpaqueId | None = None
  resume_of_task_id: OpaqueId | None = None

  @model_validator(mode="after")
  def _lineage(self) -> AttemptRef:
    if self.restart_of_attempt_id == self.attempt_id:
      raise ValueError("attempt cannot restart itself")
    if self.resume_of_task_id == self.physical_task_id:
      raise ValueError("attempt cannot resume itself")
    return self


class AdmittedWorkflowNodeIdentity(WireModel):
  workflow_run_id: OpaqueId
  plan_id: OpaqueId
  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  node_id: OpaqueId
  item_key: OpaqueId | None = None


NonExecutionReason: TypeAlias = Literal[
  "dependency_not_accepted",
  "required_input_unavailable",
  "workflow_cancelled",
  "admission_denied",
]


class ExecuteTaskDisposition(WireModel):
  kind: Literal["execute"] = "execute"


class SettleWithoutExecutionDisposition(WireModel):
  kind: Literal["settle_without_execution"] = "settle_without_execution"
  reason: NonExecutionReason
  unavailable_input_names: tuple[Name, ...]

  @model_validator(mode="after")
  def _unavailable_inputs_match_reason(self) -> SettleWithoutExecutionDisposition:
    names = self.unavailable_input_names
    if len(names) != len(set(names)):
      raise ValueError("unavailable input names must be unique")
    if names != tuple(sorted(names)):
      raise ValueError("unavailable input names must use canonical sorted order")
    if self.reason == "required_input_unavailable":
      if not names:
        raise ValueError(
          "required_input_unavailable requires unavailable input names"
        )
    elif names:
      raise ValueError(
        "unavailable input names require required_input_unavailable reason"
      )
    return self


TaskExecutionDisposition: TypeAlias = Annotated[
  ExecuteTaskDisposition | SettleWithoutExecutionDisposition,
  Field(discriminator="kind"),
]


class AdmittedTask(WireModel):
  schema_version: Literal["1.0"] = "1.0"
  admitted_task_id: OpaqueId
  logical_task: LogicalTaskRef
  attempt: AttemptRef
  objective: JsonValue
  workflow_identity: AdmittedWorkflowNodeIdentity | None = None
  execution_disposition: TaskExecutionDisposition
  execution_snapshot: AgentExecutionSnapshot | None
  operation: AgentOperationSnapshot
  inputs: tuple[AdmittedInputBinding, ...] = ()
  capability_bindings: tuple[CapabilityBinding, ...] = ()
  tool_grant: ToolGrant
  content_read_grants: tuple[ContentReadGrant, ...] = ()
  workspace_grant: WorkspaceGrant
  model_bind: CapabilityBind | None
  result_requirement: ResultRequirement
  outcome_policy: OutcomePolicy
  admitted_plan_digest: Digest | None = None
  admitted_task_digest: Digest
  model_bind_digest: Digest
  capability_binding_digest: Digest
  tool_grant_digest: Digest

  @model_validator(mode="after")
  def _identity_and_grants(self) -> AdmittedTask:
    _validate_safe_literal(self.objective)
    if isinstance(self.execution_disposition, ExecuteTaskDisposition):
      if self.execution_snapshot is None:
        raise ValueError("executed tasks require an exact execution snapshot")
      if (
        self.execution_snapshot.resume_mechanics.resumable
        != self.operation.resumable
      ):
        raise ValueError(
          "execution resume mechanics must match operation resumability"
        )
      is_resume = self.attempt.resume_of_task_id is not None
      if is_resume != (self.execution_snapshot.resume_instruction is not None):
        raise ValueError(
          "resume attempts require one exact admitted resume instruction"
        )
    elif self.execution_snapshot is not None:
      raise ValueError("non-execution tasks cannot carry execution mechanics")
    if isinstance(self.logical_task, WorkflowNodeTaskRef):
      if self.workflow_identity is None:
        raise ValueError("workflow tasks require workflow_identity")
      if self.admitted_plan_digest is None:
        raise ValueError("workflow tasks require admitted_plan_digest")
      logical = self.logical_task
      expected = (
        logical.workflow_run_id,
        logical.plan_id,
        logical.phase_number,
        logical.revision,
        logical.node_id,
        logical.item_key,
      )
      actual = (
        self.workflow_identity.workflow_run_id,
        self.workflow_identity.plan_id,
        self.workflow_identity.phase_number,
        self.workflow_identity.revision,
        self.workflow_identity.node_id,
        self.workflow_identity.item_key,
      )
      if actual != expected:
        raise ValueError("workflow identity must match logical task")
    elif self.workflow_identity is not None or self.admitted_plan_digest is not None:
      raise ValueError("ordinary delegations cannot carry workflow plan identity")
    if self.operation.operation != self.logical_task.operation:
      raise ValueError("operation snapshot must match logical task")
    if self.model_bind_digest != sha256_digest(self.model_bind):
      raise ValueError("model-bind digest must match admitted task provenance")
    binding_payload = [
      binding.model_dump(mode="json") for binding in self.capability_bindings
    ]
    if self.capability_binding_digest != sha256_digest(binding_payload):
      raise ValueError(
        "capability-binding digest must match admitted task provenance"
      )
    if self.tool_grant.digest != self.tool_grant_digest:
      raise ValueError("tool grant digest must match admitted task provenance")
    admitted_payload = self.model_dump(
      mode="json",
      exclude={"admitted_task_digest"},
    )
    if self.admitted_task_digest != sha256_digest(admitted_payload):
      raise ValueError(
        "admitted-task digest must cover the exact admitted task snapshot"
      )
    read_ids = [grant.grant_id for grant in self.content_read_grants]
    if len(read_ids) != len(set(read_ids)):
      raise ValueError("content read grants must be unique")
    if isinstance(self.execution_disposition, ExecuteTaskDisposition):
      if self.model_bind is None:
        raise ValueError("execute disposition requires a model bind")
    else:
      if self.model_bind is not None:
        raise ValueError("settle_without_execution cannot carry a model bind")
      if self.inputs:
        raise ValueError("settle_without_execution cannot carry admitted inputs")
      if self.capability_bindings:
        raise ValueError(
          "settle_without_execution cannot carry capability bindings"
        )
      if self.content_read_grants:
        raise ValueError(
          "settle_without_execution cannot carry content read grants"
        )
      if self.tool_grant.tools:
        raise ValueError("settle_without_execution cannot carry tool authority")
      if self.workspace_grant.scope != "read_only":
        raise ValueError(
          "settle_without_execution requires a read-only workspace grant"
        )
    return self


class ExecutionSettlement(WireModel):
  status: Literal["succeeded", "failed", "interrupted", "cancelled", "skipped"]
  terminal_reason: NonEmptyText | None = None

  @model_validator(mode="after")
  def _terminal_reason(self) -> ExecutionSettlement:
    if self.status == "succeeded" and self.terminal_reason is not None:
      raise ValueError("successful execution cannot have a terminal reason")
    if self.status != "succeeded" and self.terminal_reason is None:
      raise ValueError("non-successful execution requires a terminal reason")
    return self


class CitationEvidenceRef(WireModel):
  kind: Literal["citation"] = "citation"
  citation_id: OpaqueId


class ContentEvidenceRef(WireModel):
  kind: Literal["content"] = "content"
  content: ContentHandle


class ObservedSourceEvidenceRef(WireModel):
  """A typed record of a source a child actually read.

  This is an observation, not a citation: runtime code derives it from the
  child's durable tool events (source envelopes and suppressed-context source
  observations). ``excerpt_handle_id`` is present only when the run's ledger
  actually minted that handle; it is never authored by an agent.
  """

  kind: Literal["observed_source"] = "observed_source"
  source_kind: OpaqueId
  document_id: OpaqueId
  produced_by_tool: OpaqueId | None = None
  source_url: NonEmptyText | None = None
  excerpt_handle_id: OpaqueId | None = None


EvidenceRef: TypeAlias = Annotated[
  CitationEvidenceRef | ContentEvidenceRef | ObservedSourceEvidenceRef,
  Field(discriminator="kind"),
]


class AnalyticalOutcome(WireModel):
  disposition: Literal["complete", "partial", "insufficient_evidence", "blocked", "not_assessed"]
  assessment_source: Literal["domain_tool", "mechanically_derived", "none"]
  assessment_rationale: NonEmptyText | None = None
  unmet_requirements: tuple[NonEmptyText, ...] = ()
  evidence_refs: tuple[EvidenceRef, ...] = ()

  @model_validator(mode="after")
  def _outcome_is_coherent(self) -> AnalyticalOutcome:
    if self.disposition == "not_assessed":
      if self.assessment_source != "none":
        raise ValueError("not_assessed outcome must have source none")
    elif self.assessment_source == "none":
      raise ValueError("assessed outcome must name its assessment source")
    if self.disposition == "complete" and self.unmet_requirements:
      raise ValueError("complete outcome cannot have unmet requirements")
    if self.disposition in {"partial", "insufficient_evidence", "blocked"} and self.assessment_rationale is None:
      raise ValueError("incomplete analytical outcomes require a rationale")
    return self


class EvidenceObservation(WireModel):
  observed_sources: tuple[EvidenceRef, ...] = ()
  tools_used: tuple[OpaqueId, ...] = ()


class CanonicalProjection(WireModel):
  contract: ContractRef
  content: ContentHandle
  inline_view: JsonValue | None = None

  @model_validator(mode="after")
  def _projection_contract(self) -> CanonicalProjection:
    if self.contract != self.content.contract:
      raise ValueError("projection content must use projection contract")
    if self.inline_view is not None:
      raw = canonical_json_bytes(self.inline_view)
      if self.content.content_bytes != len(raw):
        raise ValueError("projection inline view byte count must match content handle")
      if self.content.content_sha256 != hashlib.sha256(raw).hexdigest():
        raise ValueError("projection inline view digest must match content handle")
    return self


class NamedArtifact(WireModel):
  name: Name
  content: ContentHandle


class TaskResultValues(WireModel):
  terminal_narrative: ContentHandle | None = None
  projection: CanonicalProjection | None = None
  artifacts: tuple[NamedArtifact, ...] = ()


class TranscriptHandle(WireModel):
  kind: Literal["child_transcript", "workflow_transcript"]
  owner_id: OpaqueId


class ActivityHandle(WireModel):
  kind: Literal["child_activity", "workflow_activity"]
  owner_id: OpaqueId


class UsageObservation(WireModel):
  input_tokens: int = Field(default=0, ge=0)
  output_tokens: int = Field(default=0, ge=0)
  cached_input_tokens: int = Field(default=0, ge=0)
  tool_calls: int = Field(default=0, ge=0)
  cost_usd: float = Field(default=0.0, ge=0)

  @field_validator("cost_usd")
  @classmethod
  def _finite_cost(cls, value: float) -> float:
    if not math.isfinite(value):
      raise ValueError("cost_usd must be finite")
    return value


class TaskObservation(WireModel):
  transcript: TranscriptHandle
  activity: ActivityHandle
  usage: UsageObservation = Field(default_factory=UsageObservation)


class TaskResultProvenance(WireModel):
  admitted_task_digest: Digest
  model_bind_digest: Digest
  capability_binding_digest: Digest
  tool_grant_digest: Digest


class TaskResult(WireModel):
  schema_version: Literal["2.0"] = "2.0"
  task_result_id: OpaqueId
  logical_task: LogicalTaskRef
  attempt: AttemptRef
  execution: ExecutionSettlement
  outcome: AnalyticalOutcome | None = None
  evidence: EvidenceObservation = Field(default_factory=EvidenceObservation)
  values: TaskResultValues = Field(default_factory=TaskResultValues)
  observation: TaskObservation
  provenance: TaskResultProvenance

  @model_validator(mode="after")
  def _settlement_invariants(self) -> TaskResult:
    if self.execution.status == "skipped" and self.outcome is not None:
      raise ValueError("skipped execution cannot carry analytical outcome")
    if self.execution.status == "succeeded" and not (
      self.values.terminal_narrative
      or self.values.projection
      or self.values.artifacts
    ):
      raise ValueError("successful result requires a canonical value")
    if self.values.terminal_narrative is not None:
      if not self.values.terminal_narrative.media_type.lower().startswith("text/"):
        raise ValueError("terminal narrative must be textual content")
    return self


def terminal_task_result(
  admitted_task: AdmittedTask,
  *,
  status: Literal["failed", "interrupted", "cancelled", "skipped"],
  reason: str,
  tools_used: tuple[str, ...] = (),
  usage: UsageObservation | None = None,
) -> TaskResult:
  """Build one deterministic non-success result from exact task admission.

  The constructor publishes no result values or analytical outcome.  Every
  authority provenance digest and observation owner is derived from the
  admitted task instead of being independently supplied by the caller.
  """

  if status not in {"failed", "interrupted", "cancelled", "skipped"}:
    raise ValueError("terminal task result requires a non-success status")
  execution = ExecutionSettlement(status=status, terminal_reason=reason)
  usage_observation = usage or UsageObservation()
  if isinstance(
    admitted_task.execution_disposition,
    SettleWithoutExecutionDisposition,
  ):
    expected_status = (
      "cancelled"
      if admitted_task.execution_disposition.reason == "workflow_cancelled"
      else "skipped"
    )
    if status != expected_status:
      raise ValueError(
        "settle_without_execution status must match its admitted reason"
      )
    if tools_used or usage_observation != UsageObservation():
      raise ValueError(
        "settle_without_execution cannot report executed tools or usage"
      )
  observation = TaskObservation(
    transcript=TranscriptHandle(
      kind="child_transcript",
      owner_id=admitted_task.attempt.physical_task_id,
    ),
    activity=ActivityHandle(
      kind="child_activity",
      owner_id=admitted_task.attempt.physical_task_id,
    ),
    usage=usage_observation,
  )
  provenance = TaskResultProvenance(
    admitted_task_digest=admitted_task.admitted_task_digest,
    model_bind_digest=admitted_task.model_bind_digest,
    capability_binding_digest=admitted_task.capability_binding_digest,
    tool_grant_digest=admitted_task.tool_grant_digest,
  )
  base_payload = {
    "schema_version": "2.0",
    "logical_task": admitted_task.logical_task.model_dump(mode="json"),
    "attempt": admitted_task.attempt.model_dump(mode="json"),
    "execution": execution.model_dump(mode="json"),
    "outcome": None,
    "evidence": EvidenceObservation(tools_used=tools_used).model_dump(mode="json"),
    "values": TaskResultValues().model_dump(mode="json"),
    "observation": observation.model_dump(mode="json"),
    "provenance": provenance.model_dump(mode="json"),
  }
  result_digest = sha256_digest(base_payload).removeprefix("sha256:")
  return TaskResult(
    task_result_id=f"task-result:{result_digest}",
    **base_payload,
  )


class TaskResultRef(WireModel):
  task_result_id: OpaqueId
  logical_task: LogicalTaskRef
  attempt: AttemptRef

  @classmethod
  def from_result(cls, result: TaskResult) -> TaskResultRef:
    return cls(
      task_result_id=result.task_result_id,
      logical_task=result.logical_task,
      attempt=result.attempt,
    )


class ParentResultPolicy(WireModel):
  preferred: Literal[
    "terminal_narrative_inline_exact",
    "projection_inline",
    "authored_summary_with_result_handle",
    "result_handle",
  ]
  max_inline_bytes: int = Field(ge=1)
  on_overflow: Literal["authored_summary_with_result_handle", "result_handle", "fail"]


class TerminalNarrativeInlineExact(WireModel):
  kind: Literal["terminal_narrative_inline_exact"] = "terminal_narrative_inline_exact"
  source: ContentHandle
  content: str
  complete: Literal[True] = True

  @model_validator(mode="after")
  def _exact_content(self) -> TerminalNarrativeInlineExact:
    raw = self.content.encode(self.source.encoding or "utf-8")
    if self.source.content_bytes != len(raw):
      raise ValueError("inline narrative byte count must match source")
    if self.source.content_chars != len(self.content):
      raise ValueError("inline narrative character count must match source")
    if self.source.content_sha256 != hashlib.sha256(raw).hexdigest():
      raise ValueError("inline narrative digest must match source")
    return self


class ProjectionInline(WireModel):
  kind: Literal["projection_inline"] = "projection_inline"
  source: ContentHandle
  contract: ContractRef
  value: JsonValue
  complete: Literal[True] = True

  @model_validator(mode="after")
  def _exact_projection(self) -> ProjectionInline:
    if self.contract != self.source.contract:
      raise ValueError("projection contract must match source")
    raw = canonical_json_bytes(self.value)
    if self.source.content_bytes != len(raw):
      raise ValueError("inline projection byte count must match source")
    if self.source.content_sha256 != hashlib.sha256(raw).hexdigest():
      raise ValueError("inline projection digest must match source")
    return self


class AuthoredSummaryWithResultHandle(WireModel):
  kind: Literal["authored_summary_with_result_handle"] = "authored_summary_with_result_handle"
  summary: NonEmptyText
  source: ContentHandle
  read_grant: ContentReadGrant

  @model_validator(mode="after")
  def _grant_matches(self) -> AuthoredSummaryWithResultHandle:
    if self.source.content_id != self.read_grant.content_id:
      raise ValueError("read grant must address result content")
    if self.read_grant.scope != "direct_parent":
      raise ValueError("parent materialization requires direct-parent read scope")
    return self


class ResultHandle(WireModel):
  kind: Literal["result_handle"] = "result_handle"
  source: ContentHandle
  read_grant: ContentReadGrant

  @model_validator(mode="after")
  def _grant_matches(self) -> ResultHandle:
    if self.source.content_id != self.read_grant.content_id:
      raise ValueError("read grant must address result content")
    if self.read_grant.scope != "direct_parent":
      raise ValueError("parent materialization requires direct-parent read scope")
    return self


ParentResultMaterialization: TypeAlias = Annotated[
  TerminalNarrativeInlineExact
  | ProjectionInline
  | AuthoredSummaryWithResultHandle
  | ResultHandle,
  Field(discriminator="kind"),
]


class SettlementProjection(WireModel):
  execution_status: Literal["succeeded", "failed", "interrupted", "cancelled", "skipped"]
  outcome_disposition: Literal["complete", "partial", "insufficient_evidence", "blocked", "not_assessed"] | None = None


class ChildEvidenceProjection(WireModel):
  """The bounded record of what one child actually read, for its parent.

  This is a projection over the child's already-durable
  ``TaskResult.evidence`` — never agent-authored, never a new ledger. The
  parent runtime uses it to seed its own citation registry so a document the
  child read is citable without re-retrieval; it is stripped before the
  result reaches the model, which sees the resulting source map instead.
  """

  observed_sources: tuple[ObservedSourceEvidenceRef, ...] = Field(
    default=(),
    max_length=256,
  )
  evidence_tools: tuple[OpaqueId, ...] = Field(default=(), max_length=128)

  @model_validator(mode="after")
  def _projection_observes_something(self) -> ChildEvidenceProjection:
    if not self.observed_sources and not self.evidence_tools:
      raise ValueError("child evidence projection must record an observation")
    return self


class AgentCompletionEnvelope(WireModel):
  schema_version: Literal["1.0"] = "1.0"
  message_id: OpaqueId
  task_result_ref: TaskResultRef
  settlement_projection: SettlementProjection
  parent_materialization: ParentResultMaterialization
  # Absent means none: the field is omitted from dumps when empty so durable
  # completion events and digests recorded before child evidence existed
  # replay byte-identically.
  child_evidence: ChildEvidenceProjection | None = Field(
    default=None,
    exclude_if=lambda value: value is None,
  )


def _inline_content_bytes(value: JsonValue, handle: ContentHandle) -> bytes:
  if isinstance(value, str) and handle.media_type.lower().startswith("text/"):
    return value.encode(handle.encoding or "utf-8")
  return canonical_json_bytes(value)


class PublishedInlineView(WireModel):
  kind: Literal["inline_exact"] = "inline_exact"
  value: JsonValue
  complete: Literal[True] = True


class PublishedOutput(WireModel):
  name: Name
  output_id: OpaqueId
  contract: ContractRef
  content: ContentHandle
  inline_view: PublishedInlineView | None = None

  @field_validator("output_id")
  @classmethod
  def _logical_output_id(cls, value: str) -> str:
    _reject_raw_path(value, field_name="output_id")
    if not value.startswith("wout:"):
      raise ValueError("published output ID must be a logical wout: identity")
    return value

  @model_validator(mode="after")
  def _published_value(self) -> PublishedOutput:
    if self.contract != self.content.contract:
      raise ValueError("published output content must use output contract")
    if self.content.retention != "durable":
      raise ValueError("published output content must be durable")
    if self.inline_view is not None:
      raw = _inline_content_bytes(self.inline_view.value, self.content)
      if len(raw) != self.content.content_bytes:
        raise ValueError("published inline view byte count must match content")
      if hashlib.sha256(raw).hexdigest() != self.content.content_sha256:
        raise ValueError("published inline view digest must match content")
      if isinstance(self.inline_view.value, str) and self.content.content_chars is not None:
        if len(self.inline_view.value) != self.content.content_chars:
          raise ValueError("published inline view character count must match content")
    return self


class PublishedOutputRef(WireModel):
  output_id: OpaqueId
  contract: ContractRef
  content: ContentHandle

  @model_validator(mode="after")
  def _ref_value(self) -> PublishedOutputRef:
    _reject_raw_path(self.output_id, field_name="output_id")
    if not self.output_id.startswith("wout:"):
      raise ValueError("published output reference must use a logical wout: identity")
    if self.contract != self.content.contract:
      raise ValueError("published output reference contract must match content")
    if self.content.retention != "durable":
      raise ValueError("published output reference must be durable")
    return self

  @classmethod
  def from_output(cls, output: PublishedOutput) -> PublishedOutputRef:
    return cls(output_id=output.output_id, contract=output.contract, content=output.content)


WORKFLOW_CONTENT_MAX_SEQUENCE = 9_007_199_254_740_991
WORKFLOW_CONTENT_PAGE_MAX_BYTES = 32_000

WorkflowContentView = Literal["paged_exact_content"]


class WorkflowContentError(RuntimeError):
  """Canonical workflow content could not be read or verified."""


class WorkflowContentNotFoundError(WorkflowContentError):
  """The requested content is not authorized for this principal and owner."""


class WorkflowContentIntegrityError(WorkflowContentError):
  """Stored content conflicts with its canonical handle."""


# The content-page models were rebased from the api-side WorkflowModel
# (extra="forbid", frozen=True, allow_inf_nan=False) onto WireModel; they
# carry no float fields, so the dropped allow_inf_nan constraint is
# behavior-neutral. A future float field here must restore it locally.
class WorkflowContentCursor(WireModel):
  """Character cursor that never splits a UTF-8 code point."""

  after_char: int = Field(ge=1, le=WORKFLOW_CONTENT_MAX_SEQUENCE)


class GrantContentPageAuthorization(WireModel):
  """Derivative page authority from one admitted task/parent grant."""

  kind: Literal["content_read_grant"] = "content_read_grant"
  grant_id: str = Field(min_length=1, max_length=512)


class PublishedOutputPageAuthorization(WireModel):
  """Derivative page authority from one exact owned publication."""

  kind: Literal["published_output"] = "published_output"
  workflow_run_id: str = Field(min_length=1, max_length=512)
  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  output_id: str = Field(min_length=1, max_length=1_024)


WorkflowContentPageAuthorization = (
  GrantContentPageAuthorization | PublishedOutputPageAuthorization
)


class WorkflowContentPage(WireModel):
  """One explicitly derivative page of an exact canonical content value."""

  view: Literal["paged_exact_content"] = "paged_exact_content"
  source: ContentHandle
  authorization: WorkflowContentPageAuthorization
  after_char: int = Field(default=0, ge=0, le=WORKFLOW_CONTENT_MAX_SEQUENCE)
  content: str
  next_cursor: WorkflowContentCursor | None = None
  end: bool
  complete_source: bool

  @model_validator(mode="after")
  def _coherent_page(self) -> WorkflowContentPage:
    source_chars = self.source.content_chars
    if source_chars is None:
      raise ValueError("workflow content paging requires textual content")
    delivered_end = self.after_char + len(self.content)
    if delivered_end > source_chars:
      raise ValueError("workflow content page exceeds the source length")
    if self.end != (self.next_cursor is None):
      raise ValueError("workflow content end must agree with next_cursor")
    if self.end:
      if delivered_end != source_chars:
        raise ValueError("terminal workflow content page must reach content end")
    elif (
      not self.content
      or self.next_cursor is None
      or self.next_cursor.after_char != delivered_end
    ):
      raise ValueError("workflow content cursor must exactly continue the page")
    if self.complete_source != (self.after_char == 0 and self.end):
      raise ValueError("complete_source must identify a whole-source page")
    if len(canonical_json_bytes(self.model_dump(mode="json"))) > (
      WORKFLOW_CONTENT_PAGE_MAX_BYTES
    ):
      raise ValueError("workflow content page exceeds its byte limit")
    return self


def verified_text(source: ContentHandle, payload: bytes) -> str:
  if type(payload) is not bytes:
    raise WorkflowContentIntegrityError("content store must return exact bytes")
  if source.content_chars is None or source.encoding is None:
    raise WorkflowContentIntegrityError(
      "workflow content paging requires a textual handle"
    )
  if len(payload) != source.content_bytes:
    raise WorkflowContentIntegrityError("workflow content byte size changed")
  if hashlib.sha256(payload).hexdigest() != source.content_sha256:
    raise WorkflowContentIntegrityError("workflow content digest changed")
  try:
    text = payload.decode(source.encoding)
  except (LookupError, UnicodeDecodeError) as exc:
    raise WorkflowContentIntegrityError(
      "workflow content encoding is invalid"
    ) from exc
  if len(text) != source.content_chars:
    raise WorkflowContentIntegrityError(
      "workflow content character size changed"
    )
  return text


# The exact code-owned presentation bound for one inline published-output
# view.  A delivery summary larger than this many UTF-8 bytes cannot be
# presented inline; the runtime then delivers the exact primary output alone
# with an explicit delivery warning.  The bound is serialized on
# ``WorkflowDeliverySpec`` so plan authors see the contract instead of a
# hidden runtime constant.
PUBLISHED_OUTPUT_INLINE_MAX_BYTES = 8_000


class WorkflowDeliverySpecV1(WireModel):
  """Historical read-only spec whose durable wire has no version field."""

  presentation: Literal["attachment", "inline"]
  primary_selector: Name
  summary_selector: Name | None = None
  additional_selectors: tuple[Name, ...] = ()
  summary_inline_max_bytes: int = PUBLISHED_OUTPUT_INLINE_MAX_BYTES

  @model_validator(mode="after")
  def _delivery_selection(self) -> WorkflowDeliverySpecV1:
    if self.presentation == "attachment" and self.summary_selector is None:
      raise ValueError("attachment delivery requires a summary selector")
    if self.summary_inline_max_bytes != PUBLISHED_OUTPUT_INLINE_MAX_BYTES:
      raise ValueError(
        "summary_inline_max_bytes is the code-owned presentation bound; "
        f"it must be exactly {PUBLISHED_OUTPUT_INLINE_MAX_BYTES}"
      )
    selectors = [self.primary_selector, *self.additional_selectors]
    if self.summary_selector is not None:
      selectors.append(self.summary_selector)
    if len(selectors) != len(set(selectors)):
      raise ValueError("delivery selectors must be unique")
    return self


DELIVERY_PREVIEW_POLICY_VERSION = "deterministic_text_prefix/v1"
DELIVERY_PREVIEW_MAX_BYTES = 8_000


class WorkflowDeliverySpecV2(WireModel):
  """Explicit deterministic-preview policy recorded by a version-2 start."""

  # Required without a default: an absent-version historical payload must
  # never be promoted into the version-2 branch by model defaults.
  schema_version: Literal["2.0"]
  presentation: Literal["attachment", "inline"]
  primary_selector: Name
  additional_selectors: tuple[Name, ...] = ()
  preview_policy_version: Literal["deterministic_text_prefix/v1"]
  preview_max_bytes: int

  @model_validator(mode="after")
  def _delivery_selection(self) -> WorkflowDeliverySpecV2:
    if self.preview_policy_version != DELIVERY_PREVIEW_POLICY_VERSION:
      raise ValueError("delivery preview policy version is unsupported")
    if self.preview_max_bytes != DELIVERY_PREVIEW_MAX_BYTES:
      raise ValueError(
        "preview_max_bytes is the code-owned version-2 presentation bound; "
        f"it must be exactly {DELIVERY_PREVIEW_MAX_BYTES}"
      )
    selectors = [self.primary_selector, *self.additional_selectors]
    if len(selectors) != len(set(selectors)):
      raise ValueError("delivery selectors must be unique")
    return self


WorkflowDeliverySpec: TypeAlias = WorkflowDeliverySpecV1 | WorkflowDeliverySpecV2


def parse_workflow_delivery_spec(value: object) -> WorkflowDeliverySpec:
  """Parse exactly one absent-version v1 or explicit-version v2 spec."""

  if isinstance(value, (WorkflowDeliverySpecV1, WorkflowDeliverySpecV2)):
    return value
  if not isinstance(value, Mapping):
    raise ValueError("workflow delivery spec must be an object")
  version = value.get("schema_version")
  if "schema_version" not in value:
    return WorkflowDeliverySpecV1.model_validate(value)
  if version == "2.0":
    return WorkflowDeliverySpecV2.model_validate(value)
  raise ValueError("workflow delivery spec has an unsupported schema_version")


class AuthoredDeliverySummary(WireModel):
  text: NonEmptyText
  source: PublishedOutputRef

  @model_validator(mode="after")
  def _summary_is_exact(self) -> AuthoredDeliverySummary:
    if not self.source.content.media_type.lower().startswith("text/"):
      raise ValueError("delivery summary must reference text content")
    raw = self.text.encode(self.source.content.encoding or "utf-8")
    if len(raw) != self.source.content.content_bytes:
      raise ValueError("summary bytes must match source content")
    if len(self.text) != self.source.content.content_chars:
      raise ValueError("summary characters must match source content")
    if hashlib.sha256(raw).hexdigest() != self.source.content.content_sha256:
      raise ValueError("summary digest must match source content")
    return self


class DeliveryPrimary(WireModel):
  name: Name
  published_output_ref: PublishedOutputRef


class DeliveryAdditionalOutput(WireModel):
  name: Name
  published_output_ref: PublishedOutputRef


class DeliveryEnvelopeV1(WireModel):
  """Historical authored-summary delivery envelope."""

  schema_version: Literal["1.0"]
  workflow_run_id: OpaqueId
  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  summary: AuthoredDeliverySummary | None = None
  primary: DeliveryPrimary
  additional_outputs: tuple[DeliveryAdditionalOutput, ...] = ()

  @model_validator(mode="after")
  def _one_atomic_revision(self) -> DeliveryEnvelopeV1:
    refs = [self.primary.published_output_ref]
    if self.summary is not None:
      refs.append(self.summary.source)
    refs.extend(item.published_output_ref for item in self.additional_outputs)
    ids = [ref.output_id for ref in refs]
    if len(ids) != len(set(ids)):
      raise ValueError("delivery outputs must be distinct")
    prefix = (
      f"wout:{self.workflow_run_id}:phase:{self.phase_number}:"
      f"revision:{self.revision}:"
    )
    if any(not ref.output_id.startswith(prefix) for ref in refs):
      raise ValueError("delivery outputs must belong to one workflow phase revision")
    return self


class DeliveryPreview(WireModel):
  """Non-authoritative exact UTF-8 prefix of one published primary output."""

  kind: Literal["deterministic_text_preview"]
  text: Annotated[str, StringConstraints(max_length=262_144)]
  source_start_byte: Literal[0]
  source_end_byte: int = Field(ge=0)
  source_total_bytes: int = Field(ge=0)
  complete: bool
  omitted_bytes: int = Field(ge=0)

  @model_validator(mode="after")
  def _exact_interval(self) -> DeliveryPreview:
    preview_bytes = self.text.encode("utf-8")
    if len(preview_bytes) != self.source_end_byte:
      raise ValueError("delivery preview byte range must match its UTF-8 text")
    if self.source_end_byte > DELIVERY_PREVIEW_MAX_BYTES:
      raise ValueError("delivery preview exceeds the version-2 byte bound")
    if self.source_end_byte > self.source_total_bytes:
      raise ValueError("delivery preview cannot exceed its exact source")
    if self.omitted_bytes != self.source_total_bytes - self.source_end_byte:
      raise ValueError("delivery preview omitted bytes must match its byte range")
    if self.complete != (self.omitted_bytes == 0):
      raise ValueError("delivery preview completeness must match omitted bytes")
    return self


class DeliveryPrimaryV2(WireModel):
  name: Name
  published_output_ref: PublishedOutputRef
  preview: DeliveryPreview

  @model_validator(mode="after")
  def _preview_matches_primary(self) -> DeliveryPrimaryV2:
    content = self.published_output_ref.content
    if not content.media_type.lower().startswith("text/"):
      raise ValueError("delivery preview requires a textual primary output")
    if (content.encoding or "").lower() != "utf-8":
      raise ValueError("delivery preview requires an exact UTF-8 primary output")
    if self.preview.source_total_bytes != content.content_bytes:
      raise ValueError("delivery preview total must match primary content bytes")
    return self


class DeliveryEnvelopeV2(WireModel):
  """Canonical deterministic-preview delivery envelope for version-2 runs."""

  schema_version: Literal["2.0"]
  workflow_run_id: OpaqueId
  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  primary: DeliveryPrimaryV2
  additional_outputs: tuple[DeliveryAdditionalOutput, ...] = ()

  @model_validator(mode="after")
  def _one_atomic_revision(self) -> DeliveryEnvelopeV2:
    refs = [self.primary.published_output_ref]
    refs.extend(item.published_output_ref for item in self.additional_outputs)
    ids = [ref.output_id for ref in refs]
    if len(ids) != len(set(ids)):
      raise ValueError("delivery outputs must be distinct")
    prefix = (
      f"wout:{self.workflow_run_id}:phase:{self.phase_number}:"
      f"revision:{self.revision}:"
    )
    if any(not ref.output_id.startswith(prefix) for ref in refs):
      raise ValueError("delivery outputs must belong to one workflow phase revision")
    return self


DeliveryEnvelope: TypeAlias = Annotated[
  DeliveryEnvelopeV1 | DeliveryEnvelopeV2,
  Field(discriminator="schema_version"),
]


def parse_delivery_envelope(value: object) -> DeliveryEnvelope:
  """Parse one explicitly versioned canonical delivery envelope."""

  if isinstance(value, (DeliveryEnvelopeV1, DeliveryEnvelopeV2)):
    return value
  if not isinstance(value, Mapping):
    raise ValueError("delivery envelope must be an object")
  version = value.get("schema_version")
  if version == "1.0":
    return DeliveryEnvelopeV1.model_validate(value)
  if version == "2.0":
    return DeliveryEnvelopeV2.model_validate(value)
  raise ValueError("delivery envelope has an unsupported schema_version")


class DeliveryFailure(WireModel):
  code: Name
  message: Annotated[str, StringConstraints(min_length=1, max_length=4_096)]
  missing_outputs: tuple[Name, ...] = ()


class DeliveryWarning(WireModel):
  """Explicit presentation degradation on an otherwise complete delivery."""

  code: Name
  message: Annotated[str, StringConstraints(min_length=1, max_length=4_096)]
  omitted_outputs: tuple[Name, ...] = ()


class DeliverySettlement(WireModel):
  status: Literal["complete", "failed", "not_required"]
  phase_number: int | None = Field(default=None, ge=1)
  revision: int | None = Field(default=None, ge=1)
  spec: WorkflowDeliverySpec | None = None
  envelope: DeliveryEnvelope | None = None
  failure: DeliveryFailure | None = None
  warning: DeliveryWarning | None = None

  @model_validator(mode="after")
  def _settlement_shape(self) -> DeliverySettlement:
    attempted = self.phase_number is not None or self.revision is not None
    if attempted and (self.phase_number is None or self.revision is None):
      raise ValueError("delivery phase and revision must be present together")
    if self.status == "not_required":
      if (
        attempted
        or self.spec is not None
        or self.envelope is not None
        or self.failure is not None
        or self.warning is not None
      ):
        raise ValueError("not-required delivery cannot carry attempted delivery state")
      return self
    if self.phase_number is None or self.revision is None or self.spec is None:
      raise ValueError("attempted delivery requires phase, revision, and spec")
    if self.status == "complete":
      if self.envelope is None or self.failure is not None:
        raise ValueError("complete delivery requires envelope and forbids failure")
      v1_pair = isinstance(self.spec, WorkflowDeliverySpecV1) and isinstance(
        self.envelope, DeliveryEnvelopeV1
      )
      v2_pair = isinstance(self.spec, WorkflowDeliverySpecV2) and isinstance(
        self.envelope, DeliveryEnvelopeV2
      )
      if not (v1_pair or v2_pair):
        raise ValueError("delivery spec and envelope versions must match")
      if (
        self.envelope.phase_number != self.phase_number
        or self.envelope.revision != self.revision
      ):
        raise ValueError("delivery envelope must match settlement revision")
      if v1_pair and (
        self.spec.summary_selector is not None
        and self.envelope.summary is None
        and self.warning is None
      ):
        raise ValueError(
          "delivery without its admitted authored summary requires an "
          "explicit delivery warning"
        )
      if v2_pair:
        preview = self.envelope.primary.preview
        if preview.source_end_byte > self.spec.preview_max_bytes:
          raise ValueError("delivery preview exceeds its recorded byte bound")
        if preview.complete != (
          preview.source_total_bytes <= self.spec.preview_max_bytes
        ):
          raise ValueError(
            "delivery preview completeness disagrees with its recorded policy"
          )
        if self.warning is not None:
          raise ValueError("version-2 delivery forbids authored-summary warnings")
    elif self.envelope is not None or self.failure is None:
      raise ValueError("failed delivery requires failure and forbids envelope")
    if self.warning is not None:
      if not isinstance(self.spec, WorkflowDeliverySpecV1):
        raise ValueError("delivery warning requires a version-1 spec")
      if self.status != "complete" or self.envelope is None:
        raise ValueError("a delivery warning is only legal on a complete delivery")
      if not isinstance(self.envelope, DeliveryEnvelopeV1):
        raise ValueError("delivery warning requires a version-1 envelope")
      if self.envelope.summary is not None:
        raise ValueError("a delivery warning must explain an omitted authored summary")
      if (
        self.spec is None
        or self.spec.summary_selector is None
        or self.warning.omitted_outputs != (self.spec.summary_selector,)
      ):
        raise ValueError("a delivery warning must name the omitted summary selector")
    return self


class AdmittedPlanRef(WireModel):
  workflow_run_id: OpaqueId
  plan_id: OpaqueId
  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  digest: Digest


class TerminalPhaseRevision(WireModel):
  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)


class ContinuationState(WireModel):
  status: Literal["not_available", "available", "requested", "exhausted"]
  next_phase_number: int | None = Field(default=None, ge=1)

  @model_validator(mode="after")
  def _continuation_target(self) -> ContinuationState:
    if self.status in {"available", "requested"} and self.next_phase_number is None:
      raise ValueError("available or requested continuation requires next phase")
    if self.status in {"not_available", "exhausted"} and self.next_phase_number is not None:
      raise ValueError("terminal continuation state cannot name next phase")
    return self


WorkflowViewState = Literal[
  "authoring",
  "running",
  "awaiting_action",
  "cancel_requested",
  "terminal",
]
WorkflowViewLegalAction = Literal["observe", "continue", "finish", "cancel"]


class WorkflowViewPhase(WireModel):
  """The current admitted phase's identity and settlement progress."""

  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  is_terminal: bool
  settled_nodes: int = Field(ge=0)
  total_nodes: int = Field(ge=0)


class WorkflowViewNodeState(WireModel):
  """One admitted node's state, embedding its settlement when settled.

  ``settlement.outcome_disposition`` is pinned absent until B-3 populates
  outcome qualifiers; the embed exists now so that population is reshape-free
  (design A-M5-before-B-3 ordering).
  """

  phase_number: int = Field(ge=1)
  revision: int = Field(ge=1)
  node_id: OpaqueId
  status: Literal[
    "pending",
    "ready",
    "running",
    "completed_unpublished",
    "restart_requested",
    "resume_requested",
    "stop_requested",
    "interrupted",
    "settled",
  ]
  attempt_number: int | None = Field(default=None, ge=1)
  task_id: OpaqueId | None = None
  action_required: Literal["publish_result", "restart", "retry", "resume"] | None = None
  settlement: SettlementProjection | None = None


class WorkflowAuthorFailureView(WireModel):
  """One non-accepted plan-author operation result, never masked (T2-I08)."""

  authoring_operation_id: OpaqueId
  phase_number: int = Field(ge=1)
  proposed_revision: int = Field(ge=1)
  status: Literal["failed", "interrupted", "cancelled"]
  terminal_reason: Literal[
    "aborted",
    "failed_validation",
    "no_viable_plan",
    "rate_limited",
    "author_provider_unavailable",
    "interrupted",
    "cancelled",
  ]
  stop_reason: str | None = Field(default=None, min_length=1, max_length=128)
  failure_detail: str | None = Field(default=None, min_length=1, max_length=512)
  attempt_count: int = Field(ge=0)
  latest_validation_summary: tuple[JsonValue, ...] = ()


class WorkflowRecoveryHint(WireModel):
  """Actionable recovery classification for the latest author failure.

  ``relaunch_budget_field`` names the exact launch knob — the
  ``author_output_budget_tokens`` tool field — whenever a larger authoring
  budget would make a relaunch viable (T2-I08).
  """

  retryability: Literal[
    "retryable_relaunch_larger_budget",
    "retryable_after_backoff",
    "not_retryable",
  ]
  relaunch_budget_field: Literal["author_output_budget_tokens"] | None = None

  @model_validator(mode="after")
  def _budget_field_names_the_knob(self) -> WorkflowRecoveryHint:
    needs_field = self.retryability == "retryable_relaunch_larger_budget"
    if needs_field != (self.relaunch_budget_field is not None):
      raise ValueError(
        "relaunch_budget_field must name author_output_budget_tokens exactly "
        "when the hint is retryable_relaunch_larger_budget"
      )
    return self


class WorkflowAnomalyView(WireModel):
  """The active (D15) recorded anomaly parking one run at its boundary."""

  anomaly_id: OpaqueId
  origin: Literal["phase_drive", "continuation_drive", "retry_drive"]
  exception_class: str = Field(min_length=1, max_length=128)
  message: str = Field(min_length=1, max_length=512)
  phase_number: int | None = Field(default=None, ge=1)
  revision: int | None = Field(default=None, ge=1)


class WorkflowContinuationAcceptedView(WireModel):
  """The open durable continuation-authoring bracket (A-M4, T2-S06)."""

  phase_number: int = Field(ge=2)
  revision: int = Field(ge=1)


class WorkflowOutputReadRecipe(WireModel):
  """Executable read recipe for one published output (same shape as
  ``WorkflowOutputAttachment.read``)."""

  action: Literal["output"] = "output"
  workflow_run_id: OpaqueId
  output_id: OpaqueId


class WorkflowView(WireModel):
  """The one canonical caller-facing projection of a workflow run (T2-I07).

  Every ``workflow_run`` surface renders this view; a fact the view carries
  cannot be omitted by any surface.
  """

  workflow_run_id: OpaqueId
  workflow_name: OpaqueId
  state: WorkflowViewState
  execution_status: Literal[
    "authoring",
    "running",
    "succeeded",
    "failed",
    "interrupted",
    "cancelled",
  ]
  delivery_status: Literal["pending", "complete", "failed", "not_required"]
  terminal_status: Literal["succeeded", "failed", "interrupted", "cancelled"] | None = None
  terminal_reason: str | None = Field(default=None, min_length=1, max_length=2_048)
  cancellation_reason: str | None = Field(default=None, min_length=1, max_length=2_048)
  continuation_accepted: WorkflowContinuationAcceptedView | None = None
  latest_anomaly: WorkflowAnomalyView | None = None
  latest_author_failure: WorkflowAuthorFailureView | None = None
  recovery_hint: WorkflowRecoveryHint | None = None
  legal_actions: tuple[WorkflowViewLegalAction, ...] = ()
  observation_seq: int = Field(ge=0)
  max_phases: int = Field(ge=1)
  phase: WorkflowViewPhase | None = None
  node_states: tuple[WorkflowViewNodeState, ...] = ()
  published_output_reads: tuple[WorkflowOutputReadRecipe, ...] = ()
  admitted_plan_ref: AdmittedPlanRef | None = None
  terminal_phase_revision: TerminalPhaseRevision | None = None
  estimated_cost_usd: float = Field(ge=0)
  admitted_cost_estimate_usd: float = Field(ge=0)
  author_cost_usd: float = Field(ge=0)

  @model_validator(mode="after")
  def _view_identity(self) -> WorkflowView:
    if (self.state == "terminal") != (self.terminal_status is not None):
      raise ValueError("terminal state and terminal_status must agree")
    if self.admitted_plan_ref is not None and (
      self.admitted_plan_ref.workflow_run_id != self.workflow_run_id
    ):
      raise ValueError("admitted plan must belong to workflow view")
    if self.terminal_phase_revision is not None:
      if self.admitted_plan_ref is None:
        raise ValueError(
          "terminal phase revision requires its admitted plan reference"
        )
      if (
        self.admitted_plan_ref.phase_number
        != self.terminal_phase_revision.phase_number
        or self.admitted_plan_ref.revision != self.terminal_phase_revision.revision
      ):
        raise ValueError("terminal phase revision must match admitted plan reference")
    if self.recovery_hint is not None and self.latest_author_failure is None:
      raise ValueError("recovery hint requires its author failure")
    for recipe in self.published_output_reads:
      if recipe.workflow_run_id != self.workflow_run_id:
        raise ValueError("published output read must belong to workflow view")
    return self


class WorkflowResult(WireModel):
  schema_version: Literal["2.0"] = "2.0"
  # ``workflow_run_id`` stays top-level (D-T2-8): it is load-bearing for the
  # gateway attachment and continuation parsers on every rendered payload.
  workflow_run_id: OpaqueId
  view: WorkflowView
  node_results: tuple[TaskResultRef, ...] = ()
  published_outputs: tuple[PublishedOutput, ...] = ()
  delivery: DeliverySettlement | None = None
  transcript: TranscriptHandle
  activity: ActivityHandle
  usage_observation: UsageObservation = Field(default_factory=UsageObservation)
  continuation_state: ContinuationState

  @model_validator(mode="after")
  def _aggregate_identity(self) -> WorkflowResult:
    if self.view.workflow_run_id != self.workflow_run_id:
      raise ValueError("workflow view must belong to workflow result")
    if self.delivery is not None:
      if self.view.delivery_status != self.delivery.status:
        raise ValueError("view delivery status must match delivery settlement")
    elif self.view.delivery_status not in {"pending", "not_required"}:
      raise ValueError(
        "workflow result without a delivery settlement requires a pending or "
        "not-required view delivery status"
      )
    ids = [output.output_id for output in self.published_outputs]
    if len(ids) != len(set(ids)):
      raise ValueError("workflow result cannot duplicate published output IDs")
    by_id = {output.output_id: PublishedOutputRef.from_output(output) for output in self.published_outputs}
    if self.delivery is not None and self.delivery.envelope is not None:
      envelope = self.delivery.envelope
      if envelope.workflow_run_id != self.workflow_run_id:
        raise ValueError("delivery envelope must belong to workflow result")
      refs = [envelope.primary.published_output_ref]
      if isinstance(envelope, DeliveryEnvelopeV1) and envelope.summary is not None:
        refs.append(envelope.summary.source)
      refs.extend(item.published_output_ref for item in envelope.additional_outputs)
      if any(by_id.get(ref.output_id) != ref for ref in refs):
        raise ValueError("delivery must reference exact published outputs")
      if isinstance(envelope, DeliveryEnvelopeV2):
        publication = next(
          (
            output
            for output in self.published_outputs
            if output.output_id == envelope.primary.published_output_ref.output_id
          ),
          None,
        )
        inline = publication.inline_view if publication is not None else None
        if envelope.primary.preview.complete and inline is not None and (
          _inline_content_bytes(inline.value, publication.content)
          != envelope.primary.preview.text.encode("utf-8")
        ):
          raise ValueError(
            "complete delivery preview conflicts with exact inline primary"
          )
    return self


__all__ = [
  name
  for name, value in tuple(globals().items())
  if isinstance(value, type)
  and issubclass(value, BaseModel)
  and value.__module__ == __name__
  and value is not WireModel
] + [
  "DELIVERY_PREVIEW_MAX_BYTES",
  "DELIVERY_PREVIEW_POLICY_VERSION",
  "DeliveryEnvelope",
  "PUBLISHED_OUTPUT_INLINE_MAX_BYTES",
  "SELECTED_CONTENT_UTF8_CONTRACT",
  "WORKFLOW_CONTENT_MAX_SEQUENCE",
  "WORKFLOW_CONTENT_PAGE_MAX_BYTES",
  "WorkflowContentError",
  "WorkflowContentIntegrityError",
  "WorkflowContentNotFoundError",
  "WorkflowContentPageAuthorization",
  "WorkflowContentView",
  "WorkflowDeliverySpec",
  "RequestedDataSelector",
  "ContextView",
  "ContextMaterialization",
  "CapabilityBinding",
  "DependencyAcceptancePolicy",
  "NonExecutionReason",
  "TaskExecutionDisposition",
  "LogicalTaskRef",
  "EvidenceRef",
  "ParentResultMaterialization",
  "canonical_json_bytes",
  "sha256_digest",
  "parse_delivery_envelope",
  "parse_workflow_delivery_spec",
  "terminal_task_result",
  "verified_text",
]
