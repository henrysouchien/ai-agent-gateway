from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from fastapi import Body
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_gateway.control_run_lifecycle import ControlRunState
from agent_gateway.thinking import parse_effort

ChatRunState = ControlRunState
AutonomousRunState = ControlRunState
_CONTROL_CONTEXT_AUTHORITY_FIELDS = frozenset({
  "account_id",
  "account_ids",
  "api_key",
  "auth",
  "authorization",
  "channel",
  "credential",
  "credential_id",
  "credential_ids",
  "credentials",
  "dev_mode",
  "dispatch_scope",
  "email",
  "owner",
  "owner_id",
  "owner_user_id",
  "portfolio",
  "portfolio_id",
  "portfolio_name",
  "refresh_token",
  "risk_user_id",
  "route",
  "route_id",
  "token",
  "user",
  "user_email",
  "user_id",
})
_CONTROL_CONTEXT_AUTHORITY_KEYS = frozenset(
  "".join(character.lower() for character in field if character.isalnum())
  for field in _CONTROL_CONTEXT_AUTHORITY_FIELDS
)


def _context_key_is_authority_field(key: Any) -> bool:
  normalized = "".join(character.lower() for character in str(key) if character.isalnum())
  return normalized in _CONTROL_CONTEXT_AUTHORITY_KEYS


def _context_authority_paths(value: Any, *, path: str = "context") -> list[str]:
  paths: list[str] = []
  if isinstance(value, dict):
    for key, nested_value in value.items():
      key_text = str(key)
      child_path = f"{path}.{key_text}"
      if _context_key_is_authority_field(key_text):
        paths.append(child_path)
      paths.extend(_context_authority_paths(nested_value, path=child_path))
    return paths
  if isinstance(value, list):
    for index, nested_value in enumerate(value):
      paths.extend(_context_authority_paths(nested_value, path=f"{path}[{index}]"))
  return paths


def _reject_context_authority_fields(value: dict[str, Any]) -> dict[str, Any]:
  paths = _context_authority_paths(value)
  if paths:
    preview = ", ".join(paths[:5])
    if len(paths) > 5:
      preview += f", ... (+{len(paths) - 5} more)"
    raise ValueError(
      "context must not include portfolio, account, owner, credential, token, "
      f"route, channel, or dispatch-scope authority fields: {preview}"
    )
  return value


class VerdictSummaryResponse(BaseModel):
  verdict_token: str
  confidence: str | None
  one_line_summary: str
  skill_run_id: str


class StagedProposalResponse(BaseModel):
  """A proposal staged by the run that still requires an apply step (LH-26)."""

  proposal_id: str
  status: str
  requires_apply: bool = True
  expires_at: str | None = None
  subcommand: str | None = None
  ticker: str | None = None
  research_file_id: int | None = None
  skill_run_id: str | None = None


class PendingApprovalResponse(BaseModel):
  pending_id: str
  approval_id: str
  tool_name: str
  tool_input: dict[str, Any]
  planned_change: dict[str, Any] | None = None
  resolved_qualifier: str | None
  reason: str | None
  allow_persistent_approval: bool
  requested_at: str


class ToolResultSummaryResponse(BaseModel):
  """Bounded server-observed summary of the latest completed tool call."""

  tool_call_id: str
  tool_name: str
  status: str | None = None
  succeeded: bool
  subcommand: str | None = None
  gate_code: str | None = None
  artifact_ref: str | None = None
  proposal_id: str | None = None
  verdict: str | None = None
  stage_receipt_status: str | None = None
  error_code: str | None = None
  error_message: str | None = None
  error_recoverable: bool | None = None


class AutonomousResultReference(BaseModel):
  model_config = ConfigDict(extra="forbid")

  kind: Literal["skill_run", "artifact", "proposal", "output_memory"]
  ref: str = Field(..., min_length=1)
  skill_run_id: str | None = None


class AutonomousTerminalReceipt(BaseModel):
  model_config = ConfigDict(extra="forbid")

  run_id: str = Field(..., min_length=1)
  disposition: Literal[
    "completed",
    "budget_limited",
    "failed",
    "interrupted",
    "cancelled",
  ]
  exit_code: int | None
  error: str | None
  terminal_reason: Literal["writer_lease_already_held"] | None
  completed_at: str
  log_ref: str = Field(..., min_length=1)
  result_refs: list[AutonomousResultReference] = Field(default_factory=list)

  @model_validator(mode="after")
  def _validate_terminal_outcome(self) -> AutonomousTerminalReceipt:
    if self.disposition == "completed" and (
      self.exit_code != 0 or self.error is not None
    ):
      raise ValueError(
        "completed terminal receipt requires exit_code=0 and error=null"
      )
    if (
      self.terminal_reason is not None
      and self.disposition != "completed"
    ):
      raise ValueError(
        "terminal receipt reason requires completed disposition"
      )
    return self


class DispatchScope(BaseModel):
  """Redacted, browser-safe structured scope for control-plane dispatches."""

  model_config = ConfigDict(extra="forbid")

  kind: Literal["portfolio"]
  source: Literal["active_default", "user_selected"]
  portfolio_name: str
  portfolio_id: str | None = None
  display_name: str | None = None

  @field_validator("portfolio_name")
  @classmethod
  def _validate_portfolio_name(cls, value: str) -> str:
    if not value.strip():
      raise ValueError("portfolio_name must be a non-empty string")
    if len(value) > 256:
      raise ValueError("portfolio_name must be 256 characters or fewer")
    return value

  @field_validator("portfolio_id", "display_name")
  @classmethod
  def _validate_optional_scope_string(cls, value: str | None, info: Any) -> str | None:
    if value is None:
      return value
    if not value.strip():
      raise ValueError(f"{info.field_name} must be omitted, null, or a non-empty string")
    if len(value) > 256:
      raise ValueError(f"{info.field_name} must be 256 characters or fewer")
    return value


class ChatRunResponse(BaseModel):
  kind: Literal["chat"]
  run_id: str
  session_id: str
  agent: Literal["hank"]
  channel: str
  user_id: str
  owner_user_id: str | None = None
  raw_user_id: str | None = None
  user_slug: str | None = None
  risk_user_id: int | None = None
  user_email: str | None = None
  user_aliases: list[str] = Field(default_factory=list)
  identity_status: str | None = None
  state: ChatRunState
  started_at: str
  ended_at: str | None
  cost_usd: float | None
  max_budget_usd: float | None = None
  initial_message: str
  skill_run_ids: list[str]
  current_verdict: VerdictSummaryResponse | None
  pending_approval: PendingApprovalResponse | None
  latest_tool_result: ToolResultSummaryResponse | None = None
  dispatch_scope: DispatchScope | None = None


class AutonomousRunResponse(BaseModel):
  kind: Literal["autonomous"]
  run_id: str
  task_id: str
  agent: Literal["hank"]
  profile: str
  mode: str
  skill: str | None
  task: str | None
  ticker: str | None
  channel: str
  user_id: str
  owner_user_id: str | None = None
  raw_user_id: str | None = None
  user_slug: str | None = None
  risk_user_id: int | None = None
  user_email: str | None = None
  user_aliases: list[str] = Field(default_factory=list)
  identity_status: str | None = None
  state: AutonomousRunState
  exit_code: int | None = None
  error: str | None = None
  terminal_receipt: AutonomousTerminalReceipt | None = None
  messageable: bool = False
  started_at: str
  ended_at: str | None
  cost_usd: float | None
  max_budget_usd: float | None = None
  skill_run_ids: list[str]
  current_verdict: VerdictSummaryResponse | None
  staged_proposals: list[StagedProposalResponse] = Field(default_factory=list)
  resumable: bool = False
  resumed_from: str | None = None
  resumed_as: list[str] = Field(default_factory=list)
  latest_resume_run_id: str | None = None
  dispatch_scope: DispatchScope | None = None
  schedule_id: str | None = None
  schedule_name: str | None = None


class ChatMessage(BaseModel):
  role: str
  content: str


class ChatDispatchRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  kind: Literal["chat"]
  message: str = Field(..., min_length=1)
  channel: str = Field(..., min_length=1)
  model_key: str | None = None
  effort: str | None = None
  catalog_revision: str | None = None
  skill: str | None = None
  ticker: str | None = None
  deadline_sec: int | None = Field(default=None, ge=1)
  max_budget_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
  context: dict[str, Any] = Field(default_factory=dict)
  dispatch_scope: DispatchScope | None = None

  @field_validator("context")
  @classmethod
  def _reject_context_authority_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
    return _reject_context_authority_fields(value)

  @field_validator("effort", mode="before")
  @classmethod
  def _normalize_effort(cls, value: Any) -> str | None:
    parsed = parse_effort(value)
    return parsed.value if parsed is not None else None

  @model_validator(mode="after")
  def _require_complete_selection(self) -> "ChatDispatchRequest":
    if self.effort is not None and self.model_key is None:
      raise ValueError("effort requires an explicit model_key")
    if self.catalog_revision is not None and self.model_key is None:
      raise ValueError("catalog_revision requires an explicit model_key")
    return self

  @field_validator("max_budget_usd", mode="before")
  @classmethod
  def _reject_coerced_max_budget(cls, value: Any) -> Any:
    if value is None:
      return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError("max_budget_usd must be a finite positive number")
    return value


class AutonomousDispatchRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  kind: Literal["autonomous"]
  profile: str | None = None
  mode: Literal["once", "task", "skill"] | None = None
  skill: str | None = None
  task: str | None = None
  ticker: str | None = None
  context: str | None = None
  channel: str | None = None
  max_budget_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
  dispatch_scope: DispatchScope | None = None

  @field_validator("max_budget_usd", mode="before")
  @classmethod
  def _reject_coerced_max_budget(cls, value: Any) -> Any:
    if value is None:
      return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError("max_budget_usd must be a finite positive number")
    return value

  @model_validator(mode="after")
  def _require_exclusive_mode_payload(self) -> "AutonomousDispatchRequest":
    if self.mode is None:
      return self
    skill = (self.skill or "").strip()
    task = (self.task or "").strip()
    ticker = (self.ticker or "").strip()
    context = (self.context or "").strip()
    if self.mode == "once":
      if skill or task or ticker or context:
        raise ValueError("mode='once' does not accept skill, task, ticker, or context")
      return self
    if self.mode == "task":
      if not task:
        raise ValueError("mode='task' requires task")
      if skill:
        raise ValueError("mode='task' does not accept skill")
      return self
    if not skill:
      raise ValueError("mode='skill' requires skill")
    if task:
      raise ValueError("mode='skill' does not accept task")
    return self


ControlRunDispatchRequest = Annotated[
  Union[ChatDispatchRequest, AutonomousDispatchRequest],
  Body(discriminator="kind"),
]


class AutonomousDispatchResponse(BaseModel):
  run: AutonomousRunResponse
  task_id: str
  run_id: str
  log_path: str
  started_at: int
  cmd: list[str]
  resumed_from: str | None = None


class ChatDispatchResponse(BaseModel):
  run: ChatRunResponse
  chat_session_token: str
  chat_session_id: str
  chat_session_expires_at: int


RunDispatchResponse = Union[ChatDispatchResponse, AutonomousDispatchResponse]
RunResponse = Union[ChatRunResponse, AutonomousRunResponse]


class RunsListResponse(BaseModel):
  runs: list[RunResponse]


class RunLogsResponse(BaseModel):
  run_id: str
  log_lines: list[str]
  more_available: bool


class ChatContinuationRequest(BaseModel):
  model_config = ConfigDict(extra="forbid")

  messages: list[ChatMessage]
  request_id: str | None = None
  context: dict[str, Any] = Field(default_factory=dict)
  model_key: str | None = None
  effort: str | None = None
  catalog_revision: str | None = None
  deadline_sec: int | None = Field(default=None, ge=1)

  @field_validator("context")
  @classmethod
  def _reject_context_authority_fields(cls, value: dict[str, Any]) -> dict[str, Any]:
    return _reject_context_authority_fields(value)

  @field_validator("effort", mode="before")
  @classmethod
  def _normalize_effort(cls, value: Any) -> str | None:
    parsed = parse_effort(value)
    return parsed.value if parsed is not None else None

  @model_validator(mode="after")
  def _require_complete_selection(self) -> "ChatContinuationRequest":
    if self.effort is not None and self.model_key is None:
      raise ValueError("effort requires an explicit model_key")
    if self.catalog_revision is not None and self.model_key is None:
      raise ValueError("catalog_revision requires an explicit model_key")
    return self


class AutonomousRunMessageRequest(BaseModel):
  message: str = Field(..., min_length=1)
  message_id: str | None = None


class AutonomousResumeRequest(BaseModel):
  context: str | None = None
  message: str | None = None
  request_id: str | None = None


RunMessageRequest = Union[ChatContinuationRequest, AutonomousRunMessageRequest]


class RunEnvelopeResponse(BaseModel):
  run: RunResponse
  message_id: str | None = None
  delivery_status: Literal["delivered", "duplicate"] | None = None


__all__ = [
  "AutonomousDispatchRequest",
  "AutonomousDispatchResponse",
  "AutonomousResumeRequest",
  "AutonomousResultReference",
  "AutonomousRunMessageRequest",
  "AutonomousRunResponse",
  "AutonomousRunState",
  "AutonomousTerminalReceipt",
  "ChatContinuationRequest",
  "ChatDispatchRequest",
  "ChatDispatchResponse",
  "ChatMessage",
  "ChatRunResponse",
  "ChatRunState",
  "ControlRunDispatchRequest",
  "DispatchScope",
  "PendingApprovalResponse",
  "RunDispatchResponse",
  "RunEnvelopeResponse",
  "RunLogsResponse",
  "RunMessageRequest",
  "RunResponse",
  "RunsListResponse",
  "ToolResultSummaryResponse",
  "VerdictSummaryResponse",
]
