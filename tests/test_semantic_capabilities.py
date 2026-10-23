from __future__ import annotations

import hashlib

import pytest

from agent_gateway.semantic_capabilities import (
  SemanticCapabilityCompilationError,
  SemanticCapabilityRegistry,
  SemanticCapabilitySpec,
  SemanticToolRoute,
  compile_semantic_capabilities,
)
from agent_workflow_contracts import (
  AdmittedDataRef,
  AdmittedInputBinding,
  ContentHandle,
  ContextSourceRef,
  ContractRef,
  InlineExactContextView,
  InvocationArgumentSelector,
  OwnerBinding,
  RequestedDataRef,
  SemanticCapabilityRequirement,
  TypedInputCapabilityBinding,
)


def _contract(digest_char: str) -> ContractRef:
  return ContractRef(
    namespace="workflow",
    name="research-report",
    version="1.0",
    digest=f"sha256:{digest_char * 64}",
  )


def _input(contract: ContractRef) -> AdmittedInputBinding:
  raw = b"{}"
  content_hash = hashlib.sha256(raw).hexdigest()
  request = RequestedDataRef(
    name="source",
    selector=InvocationArgumentSelector(argument_name="source"),
    expected_contract=contract,
  )
  source = AdmittedDataRef(
    request=request,
    source_kind="invocation_argument",
    logical_source_id="invocation:source",
    owner=OwnerBinding(tenant_id="tenant", invocation_id="invocation"),
    actual_contract=contract,
    content=ContentHandle(
      content_id=f"sha256:{content_hash}",
      content_sha256=content_hash,
      content_bytes=len(raw),
      content_chars=len(raw.decode()),
      contract=contract,
      media_type="application/json",
      encoding="utf-8",
      retention="workflow",
    ),
  )
  return AdmittedInputBinding(
    name="source",
    source=source,
    context=InlineExactContextView(
      source=ContextSourceRef(
        logical_source_id=source.logical_source_id,
        content_id=source.content.content_id,
      ),
      content={},
      content_bytes=len(raw),
    ),
  )


def test_typed_compatibility_requires_full_registered_contract_identity() -> None:
  accepted = _contract("a")
  drifted = _contract("b")
  registry = SemanticCapabilityRegistry((SemanticCapabilitySpec(
    name="research.source/v1",
    typed_input_contracts=(accepted,),
  ),))
  requirement = SemanticCapabilityRequirement(
    name="research.source/v1",
    binding_modes=("typed_input",),
    compatible_input_contracts=(accepted,),
  )

  compiled = compile_semantic_capabilities(
    (requirement,),
    grant_id="grant:accepted",
    inputs=(_input(accepted),),
    registry=registry,
  )
  binding = compiled.capability_bindings[0]
  assert isinstance(binding, TypedInputCapabilityBinding)
  assert binding.input_contract == accepted

  with pytest.raises(
    SemanticCapabilityCompilationError,
    match="no compatible admitted route",
  ):
    compile_semantic_capabilities(
      (requirement,),
      grant_id="grant:drifted",
      inputs=(_input(drifted),),
      registry=registry,
    )


def test_operation_snapshot_does_not_authorize_an_unregistered_actual_contract() -> None:
  accepted = _contract("e")
  drifted = _contract("f")
  requirement = SemanticCapabilityRequirement(
    name="research.source/v1",
    binding_modes=("typed_input",),
    compatible_input_contracts=(accepted,),
  )

  compile_semantic_capabilities(
    (requirement,),
    grant_id="grant:operation-exact",
    inputs=(_input(accepted),),
  )
  with pytest.raises(
    SemanticCapabilityCompilationError,
    match="no compatible admitted route",
  ):
    compile_semantic_capabilities(
      (requirement,),
      grant_id="grant:operation-drift",
      inputs=(_input(drifted),),
    )


def test_untagged_tool_is_not_bound_to_first_of_ambiguous_requirements() -> None:
  registry = SemanticCapabilityRegistry((
    SemanticCapabilitySpec(
      name="evidence.primary/v1",
      live_tool_effects=frozenset({"read"}),
    ),
    SemanticCapabilitySpec(
      name="evidence.secondary/v1",
      live_tool_effects=frozenset({"read"}),
    ),
  ))
  requirements = tuple(
    SemanticCapabilityRequirement(
      name=name,
      binding_modes=("live_tool",),
    )
    for name in ("evidence.primary/v1", "evidence.secondary/v1")
  )

  # The ambiguous route is granted but never positionally bound, so a required
  # capability it cannot be uniquely assigned to is honestly unsatisfied.
  with pytest.raises(
    SemanticCapabilityCompilationError,
    match="has no compatible admitted route",
  ):
    compile_semantic_capabilities(
      requirements,
      grant_id="grant:ambiguous",
      tool_routes=(SemanticToolRoute(tool_id="file_read", effect="read"),),
      registry=registry,
    )

  optional = tuple(
    SemanticCapabilityRequirement(
      name=requirement.name, required=False, binding_modes=("live_tool",),
    )
    for requirement in requirements
  )
  compiled = compile_semantic_capabilities(
    optional,
    grant_id="grant:ambiguous",
    tool_routes=(SemanticToolRoute(tool_id="file_read", effect="read"),),
    registry=registry,
  )
  assert compiled.capability_bindings == ()
  assert [(entry.tool_id, entry.route_id) for entry in compiled.tool_grant.tools] == [
    ("file_read", "tool:file_read"),
  ]


def test_tagged_route_for_an_undeclared_capability_is_granted_unbound() -> None:
  registry = SemanticCapabilityRegistry((
    SemanticCapabilitySpec(
      name="evidence.primary/v1",
      live_tool_effects=frozenset({"read"}),
    ),
  ))
  compiled = compile_semantic_capabilities(
    (SemanticCapabilityRequirement(
      name="evidence.primary/v1", binding_modes=("live_tool",),
    ),),
    grant_id="grant:undeclared",
    tool_routes=(
      SemanticToolRoute(tool_id="a_read", effect="read", capability="evidence.primary/v1"),
      SemanticToolRoute(tool_id="b_write", effect="write", capability="workspace.write/v1"),
    ),
    registry=registry,
  )
  assert {entry.tool_id: entry.route_id for entry in compiled.tool_grant.tools} == {
    "a_read": "semantic:evidence.primary/v1",
    "b_write": "tool:b_write",
  }
  assert [binding.capability for binding in compiled.capability_bindings] == [
    "evidence.primary/v1",
  ]
