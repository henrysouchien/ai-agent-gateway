"""Dependency-neutral executable registry for versioned tool policies.

The application supplies implementations. This module owns only immutable
bindings, closed reference-parameter schemas, exact call DTOs, and typed
execution. It never imports application policy or installs a global registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from agent_workflow_contracts.tool_registration import (
  PolicyKind,
  RegisteredToolIdentity,
  ToolRegistrationCatalog,
  VersionedPolicyRef,
  validate_tool_registration_catalog,
  validate_registered_tool_identity,
  validate_versioned_policy_ref,
)


ToolPolicyImplementationKey: TypeAlias = tuple[PolicyKind, str, str]
PolicyParameterValidator: TypeAlias = Callable[[Mapping[str, object]], None]
DispatchOutcome = Literal[
  "ok",
  "cancelled",
  "error_timeout",
  "error_rate_limited",
  "error_transport",
  "error_semantic",
]
_DISPATCH_OUTCOMES = frozenset({
  "ok",
  "cancelled",
  "error_timeout",
  "error_rate_limited",
  "error_transport",
  "error_semantic",
})
_POLICY_KINDS = frozenset({
  "approval_predicate",
  "approval_cache_key",
  "redaction",
  "planning",
  "input_preparation",
  "outcome",
  "source_identity",
  "session_injection",
})


class ToolPolicyRegistryTypeError(TypeError):
  """A registry value has the wrong exact runtime type."""


class ToolPolicyRegistryValidationError(ValueError):
  """A registry value is well typed but violates a closed invariant."""


class DuplicateToolPolicyImplementationError(ToolPolicyRegistryValidationError):
  """More than one implementation claims the same exact identity."""


class MissingToolPolicyImplementationError(LookupError):
  """No implementation matches an exact policy identity."""


class ToolPolicyKindMismatchError(ToolPolicyRegistryValidationError):
  """A kind-specific executor received a different policy kind."""


class ToolPolicyParameterError(ToolPolicyRegistryValidationError):
  """Reference parameters do not match the implementation's closed schema."""


class ToolPolicyResultError(ToolPolicyRegistryValidationError):
  """An implementation returned a value outside its kind's result contract."""


class ToolInputPreparationError(ValueError):
  """Preparation rejected a call with a safe model-facing tool error."""

  def __init__(self, error: Mapping[str, object]) -> None:
    if not isinstance(error, Mapping):
      raise ToolPolicyRegistryTypeError("preparation error must be a mapping")
    detached = _freeze_json(error, field_name="preparation error")
    assert isinstance(detached, Mapping)
    code = detached.get("code")
    message = detached.get("message")
    if type(code) is not str or not code:
      raise ToolPolicyRegistryValidationError(
        "preparation error code must be non-empty exact str"
      )
    if type(message) is not str or not message:
      raise ToolPolicyRegistryValidationError(
        "preparation error message must be non-empty exact str"
      )
    self.error = detached
    super().__init__(message)

  def materialize_error(self) -> dict[str, object]:
    return PreparedToolCall(self.error).materialize_input()


def _text(value: object, *, field_name: str) -> str:
  if type(value) is not str:
    raise ToolPolicyRegistryTypeError(f"{field_name} must be an exact str")
  if not value or value != value.strip():
    raise ToolPolicyRegistryValidationError(
      f"{field_name} must be non-empty trimmed text"
    )
  return value


def _freeze_json(value: object, *, field_name: str) -> object:
  if isinstance(value, Mapping):
    frozen: dict[str, object] = {}
    for key, item in value.items():
      if type(key) is not str:
        raise ToolPolicyRegistryTypeError(
          f"{field_name} mapping keys must be exact strings"
        )
      frozen[key] = _freeze_json(item, field_name=field_name)
    return MappingProxyType(dict(sorted(frozen.items())))
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_json(item, field_name=field_name) for item in value)
  if type(value) is float:
    if not math.isfinite(value):
      raise ToolPolicyRegistryValidationError(
        f"{field_name} floats must be finite"
      )
    return value
  if value is None or type(value) in {bool, int, str}:
    return value
  raise ToolPolicyRegistryTypeError(f"{field_name} must be JSON-like")


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
  if not isinstance(value, Mapping):
    raise ToolPolicyRegistryTypeError(f"{field_name} must be a mapping")
  frozen = _freeze_json(value, field_name=field_name)
  assert isinstance(frozen, Mapping)
  return frozen


def _materialize_mapping(value: Mapping[str, object]) -> dict[str, object]:
  def materialize(item: object) -> object:
    if isinstance(item, Mapping):
      return {key: materialize(child) for key, child in item.items()}
    if type(item) is tuple:
      return [materialize(child) for child in item]
    return item

  return {key: materialize(item) for key, item in value.items()}


def _optional_mapping(
  value: object | None,
  *,
  field_name: str,
) -> Mapping[str, object] | None:
  return None if value is None else _mapping(value, field_name=field_name)


def _registered_identity(
  value: object,
  *,
  field_name: str,
) -> RegisteredToolIdentity:
  if type(value) is not RegisteredToolIdentity:
    raise ToolPolicyRegistryTypeError(
      f"{field_name} must be an exact RegisteredToolIdentity"
    )
  return validate_registered_tool_identity(value)


def reject_policy_parameters(parameters: Mapping[str, object]) -> None:
  """Exact validator for implementation identities that accept no params."""

  if not isinstance(parameters, Mapping):
    raise ToolPolicyRegistryTypeError("policy parameters must be a mapping")
  if parameters:
    raise ToolPolicyParameterError("policy parameters must be empty")


@dataclass(frozen=True, slots=True)
class ApprovalPredicateCall:
  """Approval input; ``trusted_context`` is intentionally runtime-opaque."""

  identity: RegisteredToolIdentity
  prepared_input: Mapping[str, object]
  trusted_context: object | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "identity",
      _registered_identity(self.identity, field_name="approval identity"),
    )
    object.__setattr__(
      self,
      "prepared_input",
      _mapping(self.prepared_input, field_name="prepared_input"),
    )

  @property
  def registration_key(self) -> str:
    return self.identity.registration_key

  @property
  def tool_name(self) -> str:
    return self.identity.logical_name

  @property
  def logical_server_id(self) -> str | None:
    return self.identity.logical_server_id


@dataclass(frozen=True, slots=True)
class ApprovalCacheKeyCall:
  identity: RegisteredToolIdentity
  prepared_input: Mapping[str, object]
  exact_backend: str | None = None
  prepared_plan: Mapping[str, object] | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "identity",
      _registered_identity(self.identity, field_name="approval cache identity"),
    )
    object.__setattr__(
      self,
      "prepared_input",
      _mapping(self.prepared_input, field_name="prepared_input"),
    )
    if self.exact_backend is not None:
      _text(self.exact_backend, field_name="exact_backend")
    object.__setattr__(
      self,
      "prepared_plan",
      _optional_mapping(self.prepared_plan, field_name="prepared_plan"),
    )

  @property
  def registration_key(self) -> str:
    return self.identity.registration_key

  @property
  def tool_name(self) -> str:
    return self.identity.logical_name

  @property
  def logical_server_id(self) -> str | None:
    return self.identity.logical_server_id


@dataclass(frozen=True, slots=True)
class RedactionCall:
  """Redaction input; ``trusted_context`` is intentionally runtime-opaque."""

  identity: RegisteredToolIdentity
  prepared_input: Mapping[str, object]
  trusted_context: object | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "identity",
      _registered_identity(self.identity, field_name="redaction identity"),
    )
    object.__setattr__(
      self,
      "prepared_input",
      _mapping(self.prepared_input, field_name="prepared_input"),
    )

  @property
  def registration_key(self) -> str:
    return self.identity.registration_key

  @property
  def tool_name(self) -> str:
    return self.identity.logical_name

  @property
  def logical_server_id(self) -> str | None:
    return self.identity.logical_server_id


@dataclass(frozen=True, slots=True)
class PlanningCall:
  """Planning input; ``trusted_context`` is intentionally runtime-opaque."""

  identity: RegisteredToolIdentity
  prepared_input: Mapping[str, object]
  trusted_context: object | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "identity",
      _registered_identity(self.identity, field_name="planning identity"),
    )
    object.__setattr__(
      self,
      "prepared_input",
      _mapping(self.prepared_input, field_name="prepared_input"),
    )

  @property
  def registration_key(self) -> str:
    return self.identity.registration_key

  @property
  def tool_name(self) -> str:
    return self.identity.logical_name

  @property
  def logical_server_id(self) -> str | None:
    return self.identity.logical_server_id


@dataclass(frozen=True, slots=True)
class InputPreparationCall:
  """Route input; ``trusted_context`` is intentionally runtime-opaque."""

  identity: RegisteredToolIdentity
  raw_input: Mapping[str, object]
  trusted_context: object | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "identity",
      _registered_identity(self.identity, field_name="input identity"),
    )
    object.__setattr__(
      self,
      "raw_input",
      _mapping(self.raw_input, field_name="raw_input"),
    )

  @property
  def registration_key(self) -> str:
    return self.identity.registration_key

  @property
  def tool_name(self) -> str:
    return self.identity.logical_name

  @property
  def logical_server_id(self) -> str | None:
    return self.identity.logical_server_id


@dataclass(frozen=True, slots=True)
class OutcomeCall:
  """Outcome input; the raw provider ``result`` remains runtime-opaque."""

  result: object
  error: Mapping[str, object] | None = None
  semantic_error: Mapping[str, object] | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "error",
      _optional_mapping(self.error, field_name="error"),
    )
    object.__setattr__(
      self,
      "semantic_error",
      _optional_mapping(self.semantic_error, field_name="semantic_error"),
    )


@dataclass(frozen=True, slots=True)
class SourceIdentityCall:
  """Source input with the exact input executed for this provider result."""

  identity: RegisteredToolIdentity
  result: object
  tool_input: Mapping[str, object]
  exposed_tool_name: str

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "identity",
      _registered_identity(self.identity, field_name="source identity"),
    )
    object.__setattr__(
      self,
      "tool_input",
      _mapping(self.tool_input, field_name="source tool_input"),
    )
    _text(self.exposed_tool_name, field_name="source exposed_tool_name")

  @property
  def registration_key(self) -> str:
    return self.identity.registration_key

  @property
  def tool_name(self) -> str:
    return self.identity.logical_name

  @property
  def logical_server_id(self) -> str | None:
    return self.identity.logical_server_id


@dataclass(frozen=True, slots=True)
class SessionInjectionCall:
  """Session input; ``trusted_context`` is intentionally runtime-opaque."""

  logical_server_id: str
  prepared_input: Mapping[str, object]
  trusted_context: object | None = None

  def __post_init__(self) -> None:
    _text(self.logical_server_id, field_name="logical_server_id")
    object.__setattr__(
      self,
      "prepared_input",
      _mapping(self.prepared_input, field_name="prepared_input"),
    )


@dataclass(frozen=True, slots=True)
class PlanDecision:
  """Closed planning result containing only authorized intent or plan data."""

  kind: Literal["none", "authorized_intent", "prepared_plan"]
  authorized_intent: Mapping[str, object] | None = None
  prepared_plan: Mapping[str, object] | None = None

  def __post_init__(self) -> None:
    if type(self.kind) is not str:
      raise ToolPolicyRegistryTypeError("plan decision kind must be exact str")
    if self.kind not in {"none", "authorized_intent", "prepared_plan"}:
      raise ToolPolicyRegistryValidationError("plan decision kind is unsupported")
    intent = _optional_mapping(
      self.authorized_intent,
      field_name="authorized_intent",
    )
    plan = _optional_mapping(self.prepared_plan, field_name="prepared_plan")
    if self.kind == "none" and (intent is not None or plan is not None):
      raise ToolPolicyRegistryValidationError(
        "plan decision none must not carry intent or plan data"
      )
    if self.kind == "authorized_intent" and (intent is None or plan is not None):
      raise ToolPolicyRegistryValidationError(
        "authorized_intent decision requires only authorized intent"
      )
    if self.kind == "prepared_plan" and plan is None:
      raise ToolPolicyRegistryValidationError(
        "prepared_plan decision requires an exact prepared plan"
      )
    if self.kind == "prepared_plan" and intent is not None:
      raise ToolPolicyRegistryValidationError(
        "prepared_plan decision must not also carry authorized intent"
      )
    object.__setattr__(self, "authorized_intent", intent)
    object.__setattr__(self, "prepared_plan", plan)


@dataclass(frozen=True, slots=True)
class PreparedToolCall:
  """Route-owned prepared input plus the exact selected code backend."""

  prepared_input: Mapping[str, object]
  exact_backend: str | None = None

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "prepared_input",
      _mapping(self.prepared_input, field_name="prepared_input"),
    )
    if self.exact_backend is not None:
      _text(self.exact_backend, field_name="exact_backend")

  def materialize_input(self) -> dict[str, object]:
    """Return a fresh mutable JSON value for one dispatch attempt."""

    return _materialize_mapping(self.prepared_input)


@dataclass(frozen=True, slots=True)
class SessionInjectionResult:
  """Trusted server input and transport metadata after session injection."""

  tool_input: Mapping[str, object]
  transport_metadata: Mapping[str, object]

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "tool_input",
      _mapping(self.tool_input, field_name="tool_input"),
    )
    object.__setattr__(
      self,
      "transport_metadata",
      _mapping(self.transport_metadata, field_name="transport_metadata"),
    )


@dataclass(frozen=True, slots=True)
class RedactionResult:
  """Detached redacted tool input."""

  tool_input: Mapping[str, object]

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "tool_input",
      _mapping(self.tool_input, field_name="tool_input"),
    )

  def materialize_input(self) -> dict[str, object]:
    """Return detached provider-wire JSON for the redacted input."""

    return _materialize_mapping(self.tool_input)


@dataclass(frozen=True, slots=True)
class SourceIdentityResult:
  """Detached exact source identities extracted from a provider result."""

  identities: tuple[Mapping[str, object], ...]

  def __post_init__(self) -> None:
    if type(self.identities) is not tuple:
      raise ToolPolicyRegistryTypeError("source identities must be an exact tuple")
    object.__setattr__(
      self,
      "identities",
      tuple(_mapping(item, field_name="source identity") for item in self.identities),
    )


ApprovalPredicateImplementation: TypeAlias = Callable[
  [VersionedPolicyRef, ApprovalPredicateCall], bool
]
ApprovalCacheKeyImplementation: TypeAlias = Callable[
  [VersionedPolicyRef, ApprovalCacheKeyCall], str
]
RedactionImplementation: TypeAlias = Callable[
  [VersionedPolicyRef, RedactionCall], RedactionResult
]
PlanningImplementation: TypeAlias = Callable[
  [VersionedPolicyRef, PlanningCall], PlanDecision
]
InputPreparationImplementation: TypeAlias = Callable[
  [VersionedPolicyRef, InputPreparationCall], PreparedToolCall
]
OutcomeImplementation: TypeAlias = Callable[
  [VersionedPolicyRef, OutcomeCall], DispatchOutcome
]
SourceIdentityImplementation: TypeAlias = Callable[
  [VersionedPolicyRef, SourceIdentityCall], SourceIdentityResult
]
SessionInjectionImplementation: TypeAlias = Callable[
  [VersionedPolicyRef, SessionInjectionCall], SessionInjectionResult
]
_Implementation: TypeAlias = (
  ApprovalPredicateImplementation
  | ApprovalCacheKeyImplementation
  | RedactionImplementation
  | PlanningImplementation
  | InputPreparationImplementation
  | OutcomeImplementation
  | SourceIdentityImplementation
  | SessionInjectionImplementation
)


def _ref(value: object, *, field_name: str) -> VersionedPolicyRef:
  if type(value) is not VersionedPolicyRef:
    raise ToolPolicyRegistryTypeError(
      f"{field_name} must be an exact VersionedPolicyRef"
    )
  return validate_versioned_policy_ref(value)


def _ref_key(reference: VersionedPolicyRef) -> ToolPolicyImplementationKey:
  return (reference.kind, reference.policy_id, reference.version)


@dataclass(frozen=True, slots=True)
class ToolPolicyImplementation:
  """One immutable, schema-closed, kind-bound implementation entry."""

  kind: PolicyKind
  policy_id: str
  version: str
  parameter_validator: PolicyParameterValidator = field(
    repr=False,
    compare=False,
  )
  _implementation: _Implementation = field(repr=False, compare=False)

  def __post_init__(self) -> None:
    if type(self.kind) is not str:
      raise ToolPolicyRegistryTypeError("policy kind must be an exact str")
    if self.kind not in _POLICY_KINDS:
      raise ToolPolicyRegistryValidationError("policy kind is unsupported")
    _text(self.policy_id, field_name="policy_id")
    _text(self.version, field_name="policy version")
    if not callable(self.parameter_validator):
      raise ToolPolicyRegistryTypeError("parameter validator must be callable")
    if not callable(self._implementation):
      raise ToolPolicyRegistryTypeError("policy implementation must be callable")

  @property
  def key(self) -> ToolPolicyImplementationKey:
    return (self.kind, self.policy_id, self.version)


@dataclass(frozen=True, slots=True)
class ToolPolicyImplementationDescriptor:
  """Callable-free metadata exposed by an implementation registry."""

  kind: PolicyKind
  policy_id: str
  version: str

  def __post_init__(self) -> None:
    if type(self.kind) is not str:
      raise ToolPolicyRegistryTypeError("policy kind must be an exact str")
    if self.kind not in _POLICY_KINDS:
      raise ToolPolicyRegistryValidationError("policy kind is unsupported")
    _text(self.policy_id, field_name="policy_id")
    _text(self.version, field_name="policy version")

  @property
  def key(self) -> ToolPolicyImplementationKey:
    return (self.kind, self.policy_id, self.version)


@dataclass(frozen=True, slots=True)
class _BoundToolPolicyImplementation:
  descriptor: ToolPolicyImplementationDescriptor
  parameter_validator: PolicyParameterValidator = field(repr=False, compare=False)
  implementation: _Implementation = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True, init=False)
class ToolPolicyImplementationRegistry:
  """Immutable registry with only kind-specific, result-checked execution."""

  descriptors: tuple[ToolPolicyImplementationDescriptor, ...]
  _index: Mapping[
    ToolPolicyImplementationKey,
    _BoundToolPolicyImplementation,
  ] = field(
    init=False, repr=False, compare=False
  )

  def __init__(
    self,
    implementations: tuple[ToolPolicyImplementation, ...],
  ) -> None:
    if type(implementations) is not tuple:
      raise ToolPolicyRegistryTypeError("implementations must be an exact tuple")
    descriptors: list[ToolPolicyImplementationDescriptor] = []
    index: dict[ToolPolicyImplementationKey, _BoundToolPolicyImplementation] = {}
    for value in implementations:
      if type(value) is not ToolPolicyImplementation:
        raise ToolPolicyRegistryTypeError(
          "implementation entry must be an exact ToolPolicyImplementation"
        )
      entry = ToolPolicyImplementation(
        value.kind,
        value.policy_id,
        value.version,
        value.parameter_validator,
        value._implementation,
      )
      descriptor = ToolPolicyImplementationDescriptor(
        entry.kind,
        entry.policy_id,
        entry.version,
      )
      if descriptor.key in index:
        raise DuplicateToolPolicyImplementationError(
          "duplicate tool policy implementation: " + "/".join(descriptor.key)
        )
      descriptors.append(descriptor)
      index[descriptor.key] = _BoundToolPolicyImplementation(
        descriptor,
        entry.parameter_validator,
        entry._implementation,
      )
    object.__setattr__(self, "descriptors", tuple(descriptors))
    object.__setattr__(self, "_index", MappingProxyType(index))

  def __len__(self) -> int:
    return len(self.descriptors)

  def validate_reference(self, reference: VersionedPolicyRef) -> None:
    """Validate exact identity and complete parameters without exposing code."""

    self._resolve_bound(reference)

  def _resolve_bound(
    self,
    reference: VersionedPolicyRef,
  ) -> _BoundToolPolicyImplementation:
    canonical = _ref(reference, field_name="policy reference")
    entry = self._index.get(_ref_key(canonical))
    if entry is None:
      raise MissingToolPolicyImplementationError(
        "missing tool policy implementation: " + "/".join(_ref_key(canonical))
      )
    result = entry.parameter_validator(canonical.parameters)
    if result is not None:
      raise ToolPolicyResultError(
        "policy parameter validator must return exactly None"
      )
    return entry

  def validate_catalog(self, catalog: ToolRegistrationCatalog) -> None:
    """Validate implementation totality and ref shape, not route compatibility.

    Declaration/owner coherence is a join-composer responsibility because this
    dependency-neutral registry does not own route policy declarations.
    """

    canonical = validate_tool_registration_catalog(catalog)
    for declaration in canonical.declarations:
      semantics = declaration.semantics
      refs = [
        semantics.redaction_policy,
        semantics.planning_policy,
        semantics.input_preparation_policy,
        semantics.outcome_policy,
        semantics.source_identity_policy,
      ]
      if semantics.approval.predicate is not None:
        refs.append(semantics.approval.predicate)
      if semantics.approval.cache_key is not None:
        refs.append(semantics.approval.cache_key)
      for reference in refs:
        self._resolve_bound(reference)
    for server in canonical.servers:
      self._resolve_bound(server.session_injection_policy)

  def _bound(
    self,
    reference: VersionedPolicyRef,
    expected_kind: PolicyKind,
  ) -> tuple[VersionedPolicyRef, _BoundToolPolicyImplementation]:
    canonical = _ref(reference, field_name="policy reference")
    if canonical.kind != expected_kind:
      raise ToolPolicyKindMismatchError(
        f"expected {expected_kind}, got {canonical.kind}"
      )
    return canonical, self._resolve_bound(canonical)

  def execute_approval_predicate(
    self,
    reference: VersionedPolicyRef,
    call: ApprovalPredicateCall,
  ) -> bool:
    canonical, entry = self._bound(reference, "approval_predicate")
    if type(call) is not ApprovalPredicateCall:
      raise ToolPolicyRegistryTypeError("call must be exact ApprovalPredicateCall")
    detached = ApprovalPredicateCall(
      call.identity,
      call.prepared_input,
      call.trusted_context,
    )
    result = cast(ApprovalPredicateImplementation, entry.implementation)(
      canonical, detached
    )
    if type(result) is not bool:
      raise ToolPolicyResultError("approval predicate must return an exact bool")
    return result

  def execute_approval_cache_key(
    self,
    reference: VersionedPolicyRef,
    call: ApprovalCacheKeyCall,
  ) -> str:
    canonical, entry = self._bound(reference, "approval_cache_key")
    if type(call) is not ApprovalCacheKeyCall:
      raise ToolPolicyRegistryTypeError("call must be exact ApprovalCacheKeyCall")
    detached = ApprovalCacheKeyCall(
      call.identity,
      call.prepared_input,
      call.exact_backend,
      call.prepared_plan,
    )
    result = cast(ApprovalCacheKeyImplementation, entry.implementation)(
      canonical, detached
    )
    return _text(result, field_name="approval cache key")

  def execute_redaction(
    self,
    reference: VersionedPolicyRef,
    call: RedactionCall,
  ) -> RedactionResult:
    canonical, entry = self._bound(reference, "redaction")
    if type(call) is not RedactionCall:
      raise ToolPolicyRegistryTypeError("call must be exact RedactionCall")
    detached = RedactionCall(
      call.identity,
      call.prepared_input,
      call.trusted_context,
    )
    result = cast(RedactionImplementation, entry.implementation)(
      canonical, detached
    )
    if type(result) is not RedactionResult:
      raise ToolPolicyResultError("redaction must return exact RedactionResult")
    return RedactionResult(result.tool_input)

  def execute_planning(
    self,
    reference: VersionedPolicyRef,
    call: PlanningCall,
  ) -> PlanDecision:
    canonical, entry = self._bound(reference, "planning")
    if type(call) is not PlanningCall:
      raise ToolPolicyRegistryTypeError("call must be exact PlanningCall")
    detached = PlanningCall(
      call.identity,
      call.prepared_input,
      call.trusted_context,
    )
    result = cast(PlanningImplementation, entry.implementation)(
      canonical, detached
    )
    if type(result) is not PlanDecision:
      raise ToolPolicyResultError("planning must return exact PlanDecision")
    return PlanDecision(
      result.kind,
      result.authorized_intent,
      result.prepared_plan,
    )

  def execute_input_preparation(
    self,
    reference: VersionedPolicyRef,
    call: InputPreparationCall,
  ) -> PreparedToolCall:
    canonical, entry = self._bound(reference, "input_preparation")
    if type(call) is not InputPreparationCall:
      raise ToolPolicyRegistryTypeError("call must be exact InputPreparationCall")
    detached = InputPreparationCall(
      call.identity,
      call.raw_input,
      call.trusted_context,
    )
    result = cast(InputPreparationImplementation, entry.implementation)(
      canonical, detached
    )
    if type(result) is not PreparedToolCall:
      raise ToolPolicyResultError(
        "input preparation must return exact PreparedToolCall"
      )
    return PreparedToolCall(result.prepared_input, result.exact_backend)

  def execute_outcome(
    self,
    reference: VersionedPolicyRef,
    call: OutcomeCall,
  ) -> DispatchOutcome:
    canonical, entry = self._bound(reference, "outcome")
    if type(call) is not OutcomeCall:
      raise ToolPolicyRegistryTypeError("call must be exact OutcomeCall")
    detached = OutcomeCall(call.result, call.error, call.semantic_error)
    result = cast(OutcomeImplementation, entry.implementation)(
      canonical, detached
    )
    if type(result) is not str or result not in _DISPATCH_OUTCOMES:
      raise ToolPolicyResultError("outcome must be an exact normalized outcome")
    return cast(DispatchOutcome, result)

  def execute_source_identity(
    self,
    reference: VersionedPolicyRef,
    call: SourceIdentityCall,
  ) -> SourceIdentityResult:
    canonical, entry = self._bound(reference, "source_identity")
    if type(call) is not SourceIdentityCall:
      raise ToolPolicyRegistryTypeError("call must be exact SourceIdentityCall")
    detached = SourceIdentityCall(
      call.identity,
      call.result,
      call.tool_input,
      call.exposed_tool_name,
    )
    result = cast(SourceIdentityImplementation, entry.implementation)(
      canonical, detached
    )
    if type(result) is not SourceIdentityResult:
      raise ToolPolicyResultError(
        "source identity must return exact SourceIdentityResult"
      )
    return SourceIdentityResult(result.identities)

  def execute_session_injection(
    self,
    reference: VersionedPolicyRef,
    call: SessionInjectionCall,
  ) -> SessionInjectionResult:
    canonical, entry = self._bound(reference, "session_injection")
    if type(call) is not SessionInjectionCall:
      raise ToolPolicyRegistryTypeError("call must be exact SessionInjectionCall")
    detached = SessionInjectionCall(
      call.logical_server_id,
      call.prepared_input,
      call.trusted_context,
    )
    result = cast(SessionInjectionImplementation, entry.implementation)(
      canonical, detached
    )
    if type(result) is not SessionInjectionResult:
      raise ToolPolicyResultError(
        "session injection must return exact SessionInjectionResult"
      )
    return SessionInjectionResult(result.tool_input, result.transport_metadata)


__all__ = [
  "ApprovalCacheKeyCall",
  "ApprovalCacheKeyImplementation",
  "ApprovalPredicateCall",
  "ApprovalPredicateImplementation",
  "DispatchOutcome",
  "DuplicateToolPolicyImplementationError",
  "InputPreparationCall",
  "InputPreparationImplementation",
  "MissingToolPolicyImplementationError",
  "OutcomeCall",
  "OutcomeImplementation",
  "PlanDecision",
  "PlanningCall",
  "PlanningImplementation",
  "PolicyParameterValidator",
  "PreparedToolCall",
  "RedactionCall",
  "RedactionImplementation",
  "RedactionResult",
  "SessionInjectionCall",
  "SessionInjectionImplementation",
  "SessionInjectionResult",
  "SourceIdentityCall",
  "SourceIdentityImplementation",
  "SourceIdentityResult",
  "ToolPolicyImplementation",
  "ToolPolicyImplementationDescriptor",
  "ToolPolicyImplementationKey",
  "ToolPolicyImplementationRegistry",
  "ToolInputPreparationError",
  "ToolPolicyKindMismatchError",
  "ToolPolicyParameterError",
  "ToolPolicyRegistryTypeError",
  "ToolPolicyRegistryValidationError",
  "ToolPolicyResultError",
  "reject_policy_parameters",
]
