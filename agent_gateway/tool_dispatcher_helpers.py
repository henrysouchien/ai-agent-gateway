from __future__ import annotations

import asyncio
import importlib
import json
import logging
from collections.abc import Sequence as AbcSequence
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Literal, Mapping, Optional, Sequence, Tuple

from . import approval_settings
from .event_log import EventLog

RELAY_POLICY_DENIED_SUB_CODE = "relay_policy_denied"
RELAY_POLICY_DENIED_MESSAGE = (
  "Denied automatically by relay chat policy — the user did not see or decline this request. "
  "Do not tell the user they denied it. To run this tool, the user must send the request from "
  "the Excel taskpane composer, where they can approve it interactively."
)


ToolResult = Tuple[Optional[Any], Optional[Dict[str, Any]]]
NeedsApprovalCallback = Callable[[str, Dict[str, Any], str], bool]
ApprovalKeyQualifier = Callable[[str, Dict[str, Any]], str]
ToolResult.__doc__ = "Standard tool return type: `(result, error)`."

PlanningIdentity = Literal["change_set", "reviewed_change_binding"]


class TrustedToolPlanError(RuntimeError):
  """Raised when a local planning hook violates the exact-plan contract."""


class PlannedWritePlanningRejected(RuntimeError):
  """Carry one trusted handler rejection through planning without losing guidance."""

  def __init__(self, *, result: Any | None = None, error: Dict[str, Any] | None = None):
    if (result is None) == (error is None):
      raise ValueError("planned-write rejection requires exactly one result or error")
    super().__init__("planned-write preparation rejected the requested input")
    self.result = result
    self.error = error

  def tool_result(self) -> ToolResult:
    return self.result, self.error


_MODEL_STATE_STORE_IDS = frozenset({
  "projection_override",
  "valuation_override",
  "workbook",
})


def _enum_value(value: Any) -> Any:
  raw = getattr(value, "value", value)
  return raw if isinstance(raw, (str, int, float, bool)) or raw is None else str(raw)


def _effect_review_row(effect: Any) -> dict[str, Any]:
  payload = effect.payload
  row: dict[str, Any] = {
    "effect_id": effect.effect_id,
    "kind": _enum_value(effect.kind),
    "criticality": _enum_value(effect.criticality),
    "depends_on": list(effect.depends_on),
  }
  for field_name in (
    "store_id",
    "target_key",
    "research_file_id",
    "ticker",
    "proposal_id",
    "queue",
    "transaction_key",
    "target_path",
    "promotion_id",
    "base_hash",
    "target_hash",
    "expected_content_digest",
    "op_hash",
  ):
    value = getattr(payload, field_name, None)
    if value is not None:
      row[field_name] = _enum_value(value)
  semantic_intent = getattr(payload, "semantic_intent", None)
  if semantic_intent is not None:
    row["semantic_intent_digest"] = getattr(
      semantic_intent,
      "content_digest",
      None,
    )
  content = getattr(payload, "content", None)
  if content is not None:
    row["content_digest"] = getattr(content, "content_digest", None)
  return row


def _workbook_write_review(
  contract: Any,
  effect: Any,
  *,
  snapshot_store_ids: frozenset[str],
) -> dict[str, Any]:
  payload = effect.payload
  if type(payload) is not contract.StoreWritePayload:
    raise TrustedToolPlanError("workbook write effect has the wrong payload type")
  content = payload.content
  if type(content) is not contract.InlinePayload:
    raise TrustedToolPlanError(
      "workbook write approval requires inline exact execution payload"
    )
  try:
    execution = json.loads(content.content.decode("utf-8"))
  except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    raise TrustedToolPlanError(
      "workbook write approval payload is not valid JSON"
    ) from exc
  if not isinstance(execution, dict):
    raise TrustedToolPlanError("workbook write approval payload must be an object")
  operations = execution.get("operations")
  if not isinstance(operations, list) or not all(
    isinstance(operation, dict) for operation in operations
  ):
    raise TrustedToolPlanError(
      "workbook write approval payload lacks exact operations"
    )
  mutation_execution = dict(execution)
  expected_readback = mutation_execution.pop("expected_readback", None)
  return {
    "effect_id": effect.effect_id,
    "target_key": payload.target_key,
    "base_hash": payload.base_hash,
    "target_hash": payload.target_hash,
    "precommit_snapshot_present": payload.store_id in snapshot_store_ids,
    "operation_count": len(operations),
    "execution": mutation_execution,
    "execution_payload_digest": content.content_digest,
    "expected_readback_digest": (
      contract.CanonicalPayload.from_value(expected_readback).content_digest
      if expected_readback is not None
      else None
    ),
  }


def _model_writer_undo_review(
  contract: Any,
  change_set: Any,
  *,
  snapshot_store_ids: frozenset[str],
  model_write_store_ids: frozenset[str],
) -> dict[str, Any]:
  scope = "workbook_and_projection_state"
  if not model_write_store_ids:
    return {
      "scope": scope,
      "status": "not_required",
      "reason": "plan_has_no_workbook_or_projection_state_write",
    }

  persist_runner_module = f"{contract.__name__.rsplit('.', 1)[0]}.persist_runner"
  try:
    persist_runner = importlib.import_module(persist_runner_module)
  except ModuleNotFoundError as exc:
    if exc.name not in {
      persist_runner_module,
      persist_runner_module.split(".")[0],
    }:
      raise
    raise TrustedToolPlanError(
      "durable Undo capability contract is unavailable"
    ) from exc
  durable_subcommands = getattr(
    persist_runner,
    "_DURABLE_UNDO_SUBCOMMANDS",
    None,
  )
  if not isinstance(durable_subcommands, frozenset):
    raise TrustedToolPlanError("durable Undo capability contract is invalid")

  missing_snapshots = sorted(model_write_store_ids.difference(snapshot_store_ids))
  subcommand = str(change_set.intent.subcommand)
  if subcommand in durable_subcommands and not missing_snapshots:
    return {
      "scope": scope,
      "status": "durable_token_after_commit",
      "tool_name": "fms_undo_model_writer_commit",
      "issued_when": "authorized_commit_succeeds",
      "covers_store_ids": sorted(model_write_store_ids),
    }
  return {
    "scope": scope,
    "status": "not_available_for_this_subcommand",
    "reason": (
      "precommit_snapshot_missing"
      if missing_snapshots
      else "subcommand_does_not_issue_durable_undo"
    ),
    "missing_snapshot_store_ids": missing_snapshots,
  }


def _planning_contract_module_for_identity(
  identity: object,
  *module_names: str,
) -> Any:
  module_name = type(identity).__module__
  if module_name not in module_names:
    raise TrustedToolPlanError("planned-write identity came from an unsupported owner")
  try:
    return importlib.import_module(module_name)
  except ModuleNotFoundError as exc:
    if exc.name not in {module_name, module_name.split(".")[0]}:
      raise
    raise TrustedToolPlanError(
      "planned-write identity contract is unavailable"
    ) from exc


@dataclass(frozen=True, slots=True)
class TrustedToolPlan:
  """Opaque trusted carrier for one exact local-tool plan.

  The identity and executable payload never enter model-controlled tool input.
  Exact type, digest, and payload linkage are revalidated on construction.
  """

  identity_source: PlanningIdentity
  identity: object = field(repr=False)
  prepared: object = field(repr=False)
  change_set_id: str
  change_hash: str
  base_vector_hash: str
  reviewed_change_binding_digest: str | None = None
  review_reference: dict[str, Any] | None = None
  execution_semantics_digest: str | None = None

  @classmethod
  def create(
    cls,
    *,
    identity_source: str,
    identity: object,
    prepared: object,
  ) -> "TrustedToolPlan":
    if identity_source == "change_set":
      contract = _planning_contract_module_for_identity(
        identity,
        "fms.core.change_set",
        "api.fms.core.change_set",
      )
      if type(identity) is not contract.ChangeSet:
        raise TrustedToolPlanError("change_set planner returned the wrong identity type")
      identity.verify_identity()
      if getattr(prepared, "change_set", None) is not identity:
        raise TrustedToolPlanError("prepared FMS payload is not linked to the exact ChangeSet")
      return cls(
        identity_source="change_set",
        identity=identity,
        prepared=prepared,
        change_set_id=identity.change_set_id,
        change_hash=identity.change_hash,
        base_vector_hash=contract.compute_base_vector_hash(identity.base_vector),
      )
    if identity_source == "reviewed_change_binding":
      contract = _planning_contract_module_for_identity(
        identity,
        "research.reviewed_change_binding",
        "api.research.reviewed_change_binding",
      )
      if type(identity) is not contract.ReviewedChangeBinding:
        raise TrustedToolPlanError(
          "reviewed_change_binding planner returned the wrong identity type"
        )
      identity.verify_identity()
      if getattr(prepared, "binding", None) is not identity:
        raise TrustedToolPlanError(
          "prepared reviewed-change payload is not linked to the exact binding"
        )
      review_reference = (
        None
        if identity.review_reference is None
        else identity.review_reference.to_dict()
      )
      return cls(
        identity_source="reviewed_change_binding",
        identity=identity,
        prepared=prepared,
        change_set_id=identity.change_set_id,
        change_hash=identity.change_hash,
        base_vector_hash=identity.base_vector_hash,
        reviewed_change_binding_digest=identity.reviewed_change_binding_digest,
        review_reference=review_reference,
        execution_semantics_digest=identity.execution_semantics_digest,
      )
    raise TrustedToolPlanError(f"unsupported planning identity: {identity_source!r}")

  def approval_identity(self) -> dict[str, Any]:
    """Return the canonical additive approval-row identity mapping."""

    self.verify_integrity()

    return {
      "identity_source": self.identity_source,
      "change_set_id": self.change_set_id,
      "change_hash": self.change_hash,
      "base_vector_hash": self.base_vector_hash,
      "reviewed_change_binding_digest": self.reviewed_change_binding_digest,
      "review_reference": (
        None if self.review_reference is None else dict(self.review_reference)
      ),
      "execution_semantics_digest": self.execution_semantics_digest,
    }

  def approval_review(self) -> dict[str, Any]:
    """Return a safe operator review derived from the exact executable plan."""

    self.verify_integrity()
    if self.identity_source != "change_set":
      raise TrustedToolPlanError(
        "planned-change approval review requires a ChangeSet identity"
      )
    contract = _planning_contract_module_for_identity(
      self.identity,
      "fms.core.change_set",
      "api.fms.core.change_set",
    )
    change_set = self.identity
    snapshot_store_ids = frozenset(
      str(effect.payload.store_id)
      for effect in change_set.effects
      if type(effect.payload) is contract.SnapshotPayload
    )
    model_write_store_ids = frozenset(
      str(effect.payload.store_id)
      for effect in change_set.effects
      if (
        type(effect.payload) is contract.StoreWritePayload
        and str(effect.payload.store_id) in _MODEL_STATE_STORE_IDS
      )
    )
    workbook_writes = [
      _workbook_write_review(
        contract,
        effect,
        snapshot_store_ids=snapshot_store_ids,
      )
      for effect in change_set.effects
      if (
        type(effect.payload) is contract.StoreWritePayload
        and str(effect.payload.store_id) == "workbook"
      )
    ]
    return {
      "schema_version": "planned-change-review.v1",
      "change_set_id": change_set.change_set_id,
      "change_hash": change_set.change_hash,
      "intent": {"subcommand": change_set.intent.subcommand},
      "target": {
        "ticker": change_set.target.ticker,
        "research_file_id": change_set.target.research_file_id,
        "workspace": change_set.target.workspace,
        "scope": _enum_value(change_set.target.scope),
      },
      "commit_strategy": _enum_value(change_set.commit_strategy),
      "effects": [
        _effect_review_row(effect) for effect in change_set.effects
      ],
      "precommit_snapshot_store_ids": sorted(snapshot_store_ids),
      "workbook": {
        "will_write": bool(workbook_writes),
        "operation_count": sum(
          int(write["operation_count"]) for write in workbook_writes
        ),
        "writes": workbook_writes,
      },
      "undo": _model_writer_undo_review(
        contract,
        change_set,
        snapshot_store_ids=snapshot_store_ids,
        model_write_store_ids=model_write_store_ids,
      ),
    }

  def verify_integrity(self) -> None:
    """Reject identity drift, prepared-payload substitution, or carrier mutation."""

    rebuilt = type(self).create(
      identity_source=self.identity_source,
      identity=self.identity,
      prepared=self.prepared,
    )
    comparable_fields = (
      "change_set_id",
      "change_hash",
      "base_vector_hash",
      "reviewed_change_binding_digest",
      "review_reference",
      "execution_semantics_digest",
    )
    if any(getattr(self, field_name) != getattr(rebuilt, field_name) for field_name in comparable_fields):
      raise TrustedToolPlanError("trusted planned-write identity carrier was mutated")

  def verify_approval_request(self, request: object) -> None:
    """Require the durable row to remain bound to this exact planned identity."""

    expected = self.approval_identity()
    actual = {field_name: getattr(request, field_name, None) for field_name in expected}
    if actual != expected:
      raise TrustedToolPlanError("approval row is not bound to the exact trusted plan")


def _approval_queue_timeout_seconds(expiry_seconds: float | int | None) -> float:
  return min(float(expiry_seconds or 600), approval_settings.approval_wait_seconds())


def active_local_tool_schema(
  get_tool_definitions: Callable[[], Sequence[Mapping[str, Any]]] | None,
  tool_name: str,
) -> tuple[Mapping[str, Any] | None, Dict[str, Any] | None]:
  if get_tool_definitions is None:
    return None, None
  try:
    tool_definitions = get_tool_definitions()
  except Exception as exc:
    return None, {
      "code": "tool_schema_unavailable",
      "message": f"Could not load active tool schema for local tool '{tool_name}': {exc}",
      "details": {"tool_name": tool_name},
      "fix": "Retry after the active tool catalog is available.",
    }

  for definition in tool_definitions or []:
    if not isinstance(definition, Mapping):
      continue
    if str(definition.get("name") or "") != tool_name:
      continue
    schema = definition.get("input_schema") or definition.get("parameters")
    if isinstance(schema, Mapping):
      return schema, None
    return None, None

  return None, {
    "code": "tool_not_advertised",
    "message": f"Local tool '{tool_name}' is not advertised in the active tool definitions for this run",
    "details": {"tool_name": tool_name},
    "fix": "Call only tools present in the active tool catalog.",
  }


def json_type_name(value: Any) -> str:
  if value is None:
    return "null"
  if isinstance(value, bool):
    return "boolean"
  if isinstance(value, dict):
    return "object"
  if isinstance(value, list):
    return "array"
  if isinstance(value, str):
    return "string"
  if isinstance(value, int):
    return "integer"
  if isinstance(value, float):
    return "number"
  return type(value).__name__


def matches_json_type(value: Any, expected_type: Any) -> bool:
  if isinstance(expected_type, AbcSequence) and not isinstance(expected_type, str):
    return any(matches_json_type(value, item) for item in expected_type)
  if expected_type == "object":
    return isinstance(value, dict)
  if expected_type == "array":
    return isinstance(value, list)
  if expected_type == "string":
    return isinstance(value, str)
  if expected_type == "boolean":
    return isinstance(value, bool)
  if expected_type == "integer":
    return isinstance(value, int) and not isinstance(value, bool)
  if expected_type == "number":
    return isinstance(value, (int, float)) and not isinstance(value, bool)
  if expected_type == "null":
    return value is None
  return True


def format_expected_type(expected_type: Any) -> str:
  if isinstance(expected_type, AbcSequence) and not isinstance(expected_type, str):
    return "|".join(str(item) for item in expected_type)
  return str(expected_type)


def tool_input_schema_error(
  tool_name: str,
  *,
  message: str,
  details: Dict[str, Any],
) -> Dict[str, Any]:
  return {
    "code": "invalid_tool_input_schema",
    "message": f"Tool '{tool_name}' input does not match its active input_schema: {message}",
    "details": {"tool_name": tool_name, **details},
    "fix": "Call the tool with a JSON object matching its active input_schema, including required fields.",
  }


def validate_against_local_schema(
  tool_name: str,
  tool_input: Any,
  schema: Mapping[str, Any],
  *,
  json_type_name_fn: Callable[[Any], str] = json_type_name,
  matches_json_type_fn: Callable[[Any, Any], bool] = matches_json_type,
  format_expected_type_fn: Callable[[Any], str] = format_expected_type,
  tool_input_schema_error_fn: Callable[..., Dict[str, Any]] = tool_input_schema_error,
) -> Dict[str, Any] | None:
  expects_object = (
    schema.get("type") == "object"
    or "properties" in schema
    or "required" in schema
    or "additionalProperties" in schema
  )
  if expects_object and not isinstance(tool_input, dict):
    got = json_type_name_fn(tool_input)
    return tool_input_schema_error_fn(
      tool_name,
      message=f"expected object, got {got}",
      details={"expected": "object", "got": got},
    )
  if not isinstance(tool_input, dict):
    return None

  required = [
    str(item)
    for item in (schema.get("required") or [])
    if isinstance(item, str) and item
  ]
  missing = [name for name in required if name not in tool_input]

  raw_properties = schema.get("properties")
  properties = raw_properties if isinstance(raw_properties, Mapping) else {}
  unexpected: list[str] = []
  if schema.get("additionalProperties") is False:
    allowed = set(str(key) for key in properties.keys())
    unexpected = sorted(str(key) for key in tool_input.keys() if str(key) not in allowed)

  type_errors: list[dict[str, str]] = []
  for schema_field, field_schema in properties.items():
    field_name = str(schema_field)
    if field_name not in tool_input or not isinstance(field_schema, Mapping):
      continue
    expected_type = field_schema.get("type")
    if expected_type is None:
      continue
    value = tool_input[field_name]
    if not matches_json_type_fn(value, expected_type):
      type_errors.append({
        "field": field_name,
        "expected": format_expected_type_fn(expected_type),
        "got": json_type_name_fn(value),
      })

  if not missing and not unexpected and not type_errors:
    return None

  parts: list[str] = []
  if missing:
    parts.append(f"missing required field(s): {', '.join(missing)}")
  if unexpected:
    parts.append(f"unexpected field(s): {', '.join(unexpected)}")
  if type_errors:
    first = type_errors[0]
    parts.append(
      f"field '{first['field']}' expected {first['expected']}, got {first['got']}"
    )
  return tool_input_schema_error_fn(
    tool_name,
    message="; ".join(parts),
    details={
      "missing": missing,
      "unexpected": unexpected,
      "type_errors": type_errors,
    },
  )


def validate_local_tool_input(
  tool_call_id: str,
  tool_name: str,
  tool_input: Any,
  *,
  local_tool_handlers: Mapping[str, Callable[..., Awaitable[ToolResult]]],
  get_tool_definitions: Callable[[], Sequence[Mapping[str, Any]]] | None,
  event_log: EventLog | None,
  active_local_tool_schema_fn: Callable[[str], tuple[Mapping[str, Any] | None, Dict[str, Any] | None]] | None = None,
  validate_against_local_schema_fn: Callable[[str, Any, Mapping[str, Any]], Dict[str, Any] | None] | None = None,
) -> Dict[str, Any] | None:
  if tool_name not in local_tool_handlers or get_tool_definitions is None:
    return None

  schema, schema_error = (
    active_local_tool_schema_fn(tool_name)
    if active_local_tool_schema_fn is not None
    else active_local_tool_schema(get_tool_definitions, tool_name)
  )
  error = schema_error
  if error is None and schema is not None:
    validate_schema = validate_against_local_schema_fn or validate_against_local_schema
    error = validate_schema(tool_name, tool_input, schema)
  if error is not None and event_log is not None:
    event_log.append(
      {
        "type": "tool_input_validation_failed",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "code": error.get("code"),
        "message": error.get("message"),
        "details": error.get("details"),
      }
    )
  return error


@dataclass
class TransportApprovalRequest:
  """Approval payload sent from the dispatcher to a client or UI layer."""

  tool_call_id: str
  nonce: str
  tool_name: str
  tool_input: Dict[str, Any]
  resolved_qualifier: str = ""
  timeout: float = 120.0
  reason: str = ""
  allow_persistent_approval: bool = True


@dataclass
class TransportApprovalResult:
  """Result returned after a user approves or denies a tool call."""

  approved: bool
  allow_tool_type: bool = False
  denied_by: str | None = None


ApprovalCallback = Callable[[TransportApprovalRequest], Awaitable[Optional[TransportApprovalResult]]]
LocalToolHandler = Callable[..., Awaitable[ToolResult]]

# Transport-layer aliases for callback-based integrations.
ApprovalRequest = TransportApprovalRequest
ApprovalDecision = TransportApprovalResult


def resolve_denied_provenance(denied_by: str | None) -> tuple[str, dict[str, Any]]:
  if denied_by == "relay_policy":
    return (
      RELAY_POLICY_DENIED_SUB_CODE,
      {
        "code": "user_denied",
        "sub_code": RELAY_POLICY_DENIED_SUB_CODE,
        "message": RELAY_POLICY_DENIED_MESSAGE,
      },
    )
  return "user_denied", {"code": "user_denied", "message": "User denied execution"}


@dataclass
class InterceptContext:
  """Context passed to a `ToolInterceptor` before tool execution."""

  tool_call_id: str
  tool_name: str
  tool_input: Dict[str, Any]
  session_id: str


@dataclass
class InterceptDecision:
  """Interceptor outcome.

  `action` should be one of `allow`, `warn`, `ask`, or `deny`.

  - allow: proceed normally
  - warn: proceed, but attach the message as a policy warning in the result
  - ask: route through the dispatcher's request_approval callback before
    executing; in headless contexts, resolves to deny unless an
    `on_headless_ask` hook overrides
  - deny: block and return an error to the model
  """

  action: str  # "allow", "deny", "warn", "ask"
  message: str = ""
  code: str = "interceptor"

  def __post_init__(self) -> None:
    if self.action not in {"allow", "deny", "warn", "ask"}:
      raise ValueError(f"Invalid interceptor action: {self.action!r}")


@dataclass
class InterceptResult:
  """Structured return from `_run_interceptors()`."""

  proceed: bool
  warnings: list[str] = field(default_factory=list)
  error: dict[str, Any] | None = None
  pending_ask: InterceptDecision | None = None


ToolInterceptor = Callable[[InterceptContext], Awaitable[InterceptDecision]]
ToolInterceptor.__doc__ = (
  "Async policy hook that receives `InterceptContext` and returns an `InterceptDecision`."
)
HeadlessAskCallback = Callable[[InterceptContext, InterceptDecision], Awaitable[str] | str]


async def run_interceptors(
  tool_call_id: str,
  tool_name: str,
  tool_input: Dict[str, Any],
  *,
  interceptors: Sequence[ToolInterceptor],
  event_log: EventLog | None,
  session_id: str,
  log: logging.Logger,
) -> InterceptResult:
  """Run interceptor chain and return a structured result."""
  if not interceptors:
    return InterceptResult(proceed=True)

  ctx = InterceptContext(
    tool_call_id=tool_call_id,
    tool_name=tool_name,
    tool_input=tool_input,
    session_id=session_id,
  )
  warnings: list[str] = []
  pending_ask: InterceptDecision | None = None

  for interceptor in interceptors:
    try:
      decision = await interceptor(ctx)
    except Exception as exc:
      is_critical = getattr(interceptor, "__intercept_critical__", False)
      if is_critical:
        log.error("Critical interceptor %s failed: %s — denying", interceptor, exc)
        if event_log is not None:
          event_log.append(
            {
              "type": "interceptor_decision",
              "tool_call_id": tool_call_id,
              "tool_name": tool_name,
              "action": "deny",
              "code": "interceptor_error",
              "message": f"Critical safety interceptor failed: {exc}",
            }
          )
        return InterceptResult(
          proceed=False,
          error={
            "code": "interceptor_error",
            "message": f"Safety check failed due to an internal error. Tool '{tool_name}' was blocked.",
          },
        )
      log.warning("Interceptor %s error (non-fatal): %s", interceptor, exc)
      continue

    if decision.action == "deny":
      if event_log is not None:
        event_log.append(
          {
            "type": "interceptor_decision",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "action": "deny",
            "code": decision.code,
            "message": decision.message,
          }
        )
      return InterceptResult(
        proceed=False,
        error={
          "code": decision.code,
          "message": decision.message or f"Tool '{tool_name}' was blocked by a runtime policy",
        },
      )
    if decision.action == "warn":
      warnings.append(decision.message)
      if event_log is not None:
        event_log.append(
          {
            "type": "interceptor_decision",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "action": "warn",
            "code": decision.code,
            "message": decision.message,
          }
        )
      continue
    if decision.action == "ask":
      if event_log is not None:
        event_log.append(
          {
            "type": "interceptor_decision",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "action": "ask",
            "code": decision.code,
            "message": decision.message,
          }
        )
      if pending_ask is None:
        pending_ask = decision

  return InterceptResult(proceed=True, warnings=warnings, pending_ask=pending_ask)


@dataclass
class ToolExecutionContext:
  """Context object passed to local tool handlers.

  Local tools can call `emit()` to send additional structured events into the
  active `EventLog`, for example incremental stdout chunks from code execution.
  """

  tool_call_id: str
  tool_name: str
  event_log: EventLog | None
  resolved_qualifier: str = ""
  abort_event: asyncio.Event | None = None
  skill_run_id: str | None = None
  step_id: str | None = None
  workspace_dir: str | None = None
  batch_id: int | str | None = None
  request_id: str | None = None
  run_id: str | None = None
  user_id: str | None = None
  approval_id: str | None = None
  approval_chain_id: str | None = None
  durable_business_model_payload: bytes | None = field(default=None, repr=False)
  trusted_plan: TrustedToolPlan | None = field(default=None, repr=False)

  def emit(self, event: Dict[str, Any]) -> None:
    """Append a custom event to the active event log."""
    if self.event_log is not None:
      self.event_log.append(event)

  @property
  def aborted(self) -> bool:
    """Return True when the owning chat stream has disconnected."""
    return self.abort_event is not None and self.abort_event.is_set()

  async def wait_aborted(self) -> None:
    """Wait until the owning chat stream disconnects."""
    if self.abort_event is None:
      await asyncio.Future()
    else:
      await self.abort_event.wait()
