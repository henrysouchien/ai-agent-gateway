from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import fields, FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from agent_gateway.tool_policy_registry import (
  ApprovalCacheKeyCall,
  ApprovalPredicateCall,
  DuplicateToolPolicyImplementationError,
  InputPreparationCall,
  MissingToolPolicyImplementationError,
  OutcomeCall,
  PlanDecision,
  PlanningCall,
  PreparedToolCall,
  RedactionCall,
  RedactionResult,
  SessionInjectionCall,
  SessionInjectionResult,
  SourceIdentityCall,
  SourceIdentityResult,
  ToolPolicyImplementation,
  ToolPolicyImplementationDescriptor,
  ToolPolicyImplementationRegistry,
  ToolPolicyKindMismatchError,
  ToolPolicyParameterError,
  ToolPolicyRegistryTypeError,
  ToolPolicyResultError,
  reject_policy_parameters,
)
from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  RegisteredToolServerDescriptor,
  ToolApprovalPolicy,
  ToolIntrinsicSemantics,
  ToolRegistrationCatalog,
  ToolRegistrationDeclaration,
  VersionedPolicyRef,
)


ROOT = Path(__file__).resolve().parents[1]


def _ref(
  kind: str = "planning",
  policy_id: str = "none",
  *,
  parameters: object | None = None,
) -> VersionedPolicyRef:
  return VersionedPolicyRef(
    kind=kind,  # type: ignore[arg-type]
    policy_id=policy_id,
    version="v1",
    parameters={} if parameters is None else parameters,  # type: ignore[arg-type]
  )


def _entry(
  kind: str = "planning",
  policy_id: str = "none",
  implementation: object = lambda _ref, _call: PlanDecision("none"),
  *,
  parameter_validator: object = reject_policy_parameters,
) -> ToolPolicyImplementation:
  return ToolPolicyImplementation(
    kind,  # type: ignore[arg-type]
    policy_id,
    "v1",
    parameter_validator,  # type: ignore[arg-type]
    implementation,  # type: ignore[arg-type]
  )


def test_registry_exposes_only_frozen_callable_free_descriptors() -> None:
  entry = _entry()
  registry = ToolPolicyImplementationRegistry((entry,))

  assert registry.descriptors == (
    ToolPolicyImplementationDescriptor("planning", "none", "v1"),
  )
  assert len(registry) == 1
  assert not hasattr(registry, "implementations")
  assert not hasattr(registry, "resolve")
  assert not hasattr(registry, "__iter__")
  assert not hasattr(registry.descriptors[0], "implementation")
  with pytest.raises(FrozenInstanceError):
    registry.descriptors = ()  # type: ignore[misc]


def test_registry_rejects_non_exact_inputs_and_duplicate_identity() -> None:
  entry = _entry()
  with pytest.raises(ToolPolicyRegistryTypeError, match="exact tuple"):
    ToolPolicyImplementationRegistry([entry])  # type: ignore[arg-type]
  with pytest.raises(ToolPolicyRegistryTypeError, match="exact ToolPolicyImplementation"):
    ToolPolicyImplementationRegistry((object(),))  # type: ignore[arg-type]
  with pytest.raises(DuplicateToolPolicyImplementationError):
    ToolPolicyImplementationRegistry((entry, _entry()))
  with pytest.raises(ToolPolicyRegistryTypeError, match="validator"):
    _entry(parameter_validator=object())


def _validate_status_equals(parameters: Mapping[str, object]) -> None:
  if set(parameters) != {"field", "values"}:
    raise ToolPolicyParameterError("status-equals parameters are not exact")
  if type(parameters["field"]) is not str:
    raise ToolPolicyParameterError("field must be an exact string")
  values = parameters["values"]
  if type(values) is not tuple or not values:
    raise ToolPolicyParameterError("values must be a non-empty exact tuple")
  if any(type(value) is not str or not value for value in values):
    raise ToolPolicyParameterError("values entries must be non-empty strings")


def test_full_parameter_validator_checks_nested_call_site_shape() -> None:
  registry = ToolPolicyImplementationRegistry((
    _entry(
      "outcome",
      "status-equals",
      lambda _ref, _call: "ok",
      parameter_validator=_validate_status_equals,
    ),
  ))
  registry.validate_reference(
    _ref(
      "outcome",
      "status-equals",
      parameters={"field": "status", "values": ["success"]},
    )
  )
  with pytest.raises(ToolPolicyParameterError, match="not exact"):
    registry.validate_reference(
      _ref("outcome", "status-equals", parameters={"field": "status"})
    )
  with pytest.raises(ToolPolicyParameterError, match="non-empty"):
    registry.validate_reference(
      _ref(
        "outcome",
        "status-equals",
        parameters={"field": "status", "values": []},
      )
    )


def test_private_lookup_drives_missing_kind_and_result_checks() -> None:
  registry = ToolPolicyImplementationRegistry((
    _entry("approval_predicate", "predicate", lambda _ref, _call: "yes"),
    _entry("outcome", "outcome", lambda _ref, _call: "invalid"),
  ))
  with pytest.raises(MissingToolPolicyImplementationError):
    registry.validate_reference(_ref("planning", "missing"))
  with pytest.raises(ToolPolicyKindMismatchError):
    registry.execute_outcome(
      _ref("approval_predicate", "predicate"),
      OutcomeCall({}),
    )
  with pytest.raises(ToolPolicyResultError, match="exact bool"):
    registry.execute_approval_predicate(
      _ref("approval_predicate", "predicate"),
      ApprovalPredicateCall(RegisteredToolIdentity("local_handler", "tool"), {}),
    )
  with pytest.raises(ToolPolicyResultError, match="normalized outcome"):
    registry.execute_outcome(_ref("outcome", "outcome"), OutcomeCall({}))


def test_exact_result_envelopes_are_required_and_detached() -> None:
  backing = {"nested": ["before"]}
  source = {"source_id": "source-1", "nested": ["before"]}
  registry = ToolPolicyImplementationRegistry((
    _entry("redaction", "redact", lambda _ref, _call: RedactionResult(backing)),
    _entry(
      "planning",
      "plan",
      lambda _ref, _call: PlanDecision("prepared_plan", prepared_plan=backing),
    ),
    _entry(
      "input_preparation",
      "prepare",
      lambda _ref, _call: PreparedToolCall(backing, "python"),
    ),
    _entry(
      "source_identity",
      "source",
      lambda _ref, _call: SourceIdentityResult((source,)),
    ),
    _entry(
      "session_injection",
      "session",
      lambda _ref, _call: SessionInjectionResult(backing, {"session": "live"}),
    ),
  ))

  redacted = registry.execute_redaction(
    _ref("redaction", "redact"),
    RedactionCall(RegisteredToolIdentity("local_handler", "tool"), {}),
  )
  planned = registry.execute_planning(
    _ref("planning", "plan"),
    PlanningCall(RegisteredToolIdentity("local_handler", "tool"), {}),
  )
  prepared = registry.execute_input_preparation(
    _ref("input_preparation", "prepare"),
    InputPreparationCall(RegisteredToolIdentity("mcp", "tool", "server"), {}),
  )
  sources = registry.execute_source_identity(
    _ref("source_identity", "source"),
    SourceIdentityCall(
      RegisteredToolIdentity("local_handler", "tool"),
      {},
      {},
      "tool",
    ),
  )
  injected = registry.execute_session_injection(
    _ref("session_injection", "session"),
    SessionInjectionCall("server", {}),
  )
  backing["nested"].append("after")
  source["nested"].append("after")

  assert type(redacted) is RedactionResult
  assert redacted.tool_input["nested"] == ("before",)
  assert type(planned) is PlanDecision
  assert planned.prepared_plan["nested"] == ("before",)  # type: ignore[index]
  assert type(prepared) is PreparedToolCall
  assert prepared.exact_backend == "python"
  assert prepared.prepared_input["nested"] == ("before",)
  assert type(sources) is SourceIdentityResult
  assert sources.identities[0]["nested"] == ("before",)
  assert type(injected) is SessionInjectionResult
  assert injected.tool_input["nested"] == ("before",)
  for mapping in (
    redacted.tool_input,
    planned.prepared_plan,
    prepared.prepared_input,
    sources.identities[0],
    injected.tool_input,
    injected.transport_metadata,
  ):
    assert type(mapping) is MappingProxyType


def test_redaction_call_requires_and_detaches_exact_registered_identity() -> None:
  identity = RegisteredToolIdentity("addin_relay", "write_cells")
  call = RedactionCall(identity, {})
  object.__setattr__(identity, "logical_name", "mutated")

  assert call.identity is not identity
  assert call.identity == RegisteredToolIdentity("addin_relay", "write_cells")
  with pytest.raises(ToolPolicyRegistryTypeError, match="RegisteredToolIdentity"):
    RedactionCall(object(), {})  # type: ignore[arg-type]


def test_tool_scoped_calls_derive_route_properties_from_one_exact_identity() -> None:
  identities = (
    RegisteredToolIdentity("local_handler", "same-name"),
    RegisteredToolIdentity("addin_relay", "same-name"),
    RegisteredToolIdentity("mcp", "same-name", "server"),
  )
  calls = tuple(ApprovalPredicateCall(identity, {}) for identity in identities)

  assert len({call.registration_key for call in calls}) == 3
  assert {call.tool_name for call in calls} == {"same-name"}
  assert tuple(call.logical_server_id for call in calls) == (None, None, "server")
  for call in calls:
    assert call.registration_key == call.identity.registration_key
    assert {field.name for field in fields(call)} == {
      "identity",
      "prepared_input",
      "trusted_context",
    }


@pytest.mark.parametrize(
  "factory",
  [
    lambda: ApprovalPredicateCall("registration", {}),  # pyright: ignore[reportArgumentType]  # negative: non-identity route string rejection
    lambda: ApprovalCacheKeyCall("registration", {}),  # pyright: ignore[reportArgumentType]  # negative: non-identity route string rejection
    lambda: RedactionCall("tool", {}),  # pyright: ignore[reportArgumentType]  # negative: non-identity route string rejection
    lambda: PlanningCall("registration", {}),  # pyright: ignore[reportArgumentType]  # negative: non-identity route string rejection
    lambda: InputPreparationCall("registration", {}),  # pyright: ignore[reportArgumentType]  # negative: non-identity route string rejection
    lambda: SourceIdentityCall("tool", {}, {}, "tool"),  # pyright: ignore[reportArgumentType]  # negative: non-identity route string rejection
  ],
)
def test_tool_scoped_calls_cannot_construct_separable_route_strings(
  factory,
) -> None:
  with pytest.raises(ToolPolicyRegistryTypeError, match="RegisteredToolIdentity"):
    factory()


def test_plan_decision_discriminants_are_mutually_exclusive() -> None:
  with pytest.raises(
    ValueError,
    match="must not also carry authorized intent",
  ):
    PlanDecision(
      "prepared_plan",
      authorized_intent={"intent": "write"},
      prepared_plan={"operations": []},
    )


def test_cache_key_and_input_call_route_fields_are_exact() -> None:
  observed: list[InputPreparationCall] = []
  registry = ToolPolicyImplementationRegistry((
    _entry(
      "approval_cache_key",
      "cache",
      lambda _ref, call: f"{call.registration_key}:cache",
    ),
    _entry(
      "input_preparation",
      "prepare",
      lambda _ref, call: (
        observed.append(call) or PreparedToolCall(call.raw_input, None)
      ),
    ),
  ))
  cache_identity = RegisteredToolIdentity("local_handler", "tool")
  assert registry.execute_approval_cache_key(
    _ref("approval_cache_key", "cache"),
    ApprovalCacheKeyCall(cache_identity, {}),
  ) == f"{cache_identity.registration_key}:cache"
  registry.execute_input_preparation(
    _ref("input_preparation", "prepare"),
    InputPreparationCall(RegisteredToolIdentity("mcp", "tool", "server"), {}),
  )
  assert observed[0].tool_name == "tool"
  assert observed[0].logical_server_id == "server"


def _catalog() -> ToolRegistrationCatalog:
  declaration = ToolRegistrationDeclaration(
    RegisteredToolIdentity("mcp", "tool", "server"),
    ToolIntrinsicSemantics(
      effect="read",
      idempotent=True,
      semantic_capability="tool:tool",
      approval=ToolApprovalPolicy(
        "predicate",
        predicate=_ref("approval_predicate", "predicate"),
        cache_key=_ref("approval_cache_key", "cache"),
      ),
      audience="ordinary",
      redaction_policy=_ref("redaction", "redact"),
      planning_policy=_ref("planning", "plan"),
      input_preparation_policy=_ref("input_preparation", "prepare"),
      outcome_policy=_ref("outcome", "outcome"),
      source_identity_policy=_ref("source_identity", "source"),
    ),
  )
  server = RegisteredToolServerDescriptor(
    "server",
    "transport",
    1,
    {},
    _ref("session_injection", "session"),
  )
  return ToolRegistrationCatalog((declaration,), (server,))


def test_catalog_totality_uses_private_validated_lookup() -> None:
  entries = (
    _entry("approval_predicate", "predicate", lambda _ref, _call: False),
    _entry("approval_cache_key", "cache", lambda _ref, _call: "cache"),
    _entry("redaction", "redact", lambda _ref, _call: RedactionResult({})),
    _entry("planning", "plan"),
    _entry(
      "input_preparation",
      "prepare",
      lambda _ref, _call: PreparedToolCall({}),
    ),
    _entry("outcome", "outcome", lambda _ref, _call: "ok"),
    _entry(
      "source_identity",
      "source",
      lambda _ref, _call: SourceIdentityResult(()),
    ),
    _entry(
      "session_injection",
      "session",
      lambda _ref, _call: SessionInjectionResult({}, {}),
    ),
  )
  ToolPolicyImplementationRegistry(entries).validate_catalog(_catalog())
  with pytest.raises(MissingToolPolicyImplementationError):
    ToolPolicyImplementationRegistry(entries[:-1]).validate_catalog(_catalog())


def test_module_has_strict_imports_and_no_global_or_generic_execution() -> None:
  path = ROOT / "agent_gateway" / "tool_policy_registry.py"
  tree = ast.parse(path.read_text(encoding="utf-8"))
  allowed_roots = {
    "__future__",
    "collections",
    "dataclasses",
    "math",
    "types",
    "typing",
    "agent_workflow_contracts",
  }
  imports: set[str] = set()
  global_registry_calls: list[ast.Call] = []
  public_method_names: set[str] = set()
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      imports.update(alias.name.split(".", 1)[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      imports.add((node.module or "").split(".", 1)[0])
    elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
      public_method_names.add(node.name)
    elif (
      isinstance(node, (ast.Assign, ast.AnnAssign))
      and isinstance(node.value, ast.Call)
      and isinstance(node.value.func, ast.Name)
      and node.value.func.id == "ToolPolicyImplementationRegistry"
    ):
      global_registry_calls.append(node.value)

  assert imports <= allowed_roots
  assert global_registry_calls == []
  assert "execute" not in public_method_names
  assert "resolve" not in public_method_names
