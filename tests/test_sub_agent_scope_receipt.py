from __future__ import annotations

import pytest

from agent_gateway.skills import (
  SkillProfile,
  compile_agent_operation,
  generic_explore_profile,
)
from agent_gateway.capability_resolution import (
  derive_dispatcher_allowlist,
  granted_tool_ids,
)
from agent_gateway.sub_agent_scope_receipt import (
  OperationToolAdmissionError,
  admit_operation_tools,
  parse_tool_grant,
  reissue_tool_grant,
  scopes_from_tool_grant,
)
from agent_workflow_contracts import (
  ExecutionIdentity,
  OperationUnavailable,
  ResolvedAuthority,
)


class _Mcp:
  def __init__(self) -> None:
    self.routes = {
      "propose_record": "records",
      "web_search": "web",
      "write_record": "records",
    }

  def is_mcp_tool(self, name: str) -> bool:
    return name in self.routes

  def get_server_for_tool(self, name: str) -> str | None:
    return self.routes.get(name)

  def get_original_tool_name(self, name: str) -> str:
    return name


def _operation():
  return compile_agent_operation(
    generic_explore_profile(),
    execution_class="node.explore",
  )


def _effect(tool_id: str, _server: str | None, _local: bool) -> str | None:
  return {
    "file_read": "read",
    "propose_record": "propose",
    "web_search": "read",
    "write_record": "write",
  }.get(tool_id)


def test_model_writer_with_artifact_only_contract_compiles_proposal_authority() -> None:
  profile = SkillProfile(
    name="artifact-model-build",
    system_prompt="Build one immutable model artifact.",
    agent_callable=True,
    mutation_mode="model_writer",
    metadata={
      "semantic_metadata": {
        "allowed_effects": ["artifact_write", "read"],
        "tool_refs": [
          {"kind": "local", "tool_id": "file_read"},
          {
            "kind": "mcp",
            "server_id": "records",
            "tool_id": "propose_record",
          },
        ],
        "capability_requirements": [
          {
            "name": "artifact.propose/v1",
            "required": True,
            "binding_modes": ["live_tool"],
          },
          {
            "name": "corpus.read/v1",
            "required": True,
            "binding_modes": ["live_tool"],
          },
        ],
      },
    },
  )
  operation = compile_agent_operation(
    profile,
    execution_class="node.implement",
  )

  assert operation.workspace_scope == "workspace_write"
  assert {
    requirement.name for requirement in operation.required_capabilities
  } == {"artifact.propose/v1", "corpus.read/v1"}
  authority = admit_operation_tools(
    operation,
    grant_id="grant:artifact-model-build",
    operation_tool_ids=("file_read", "propose_record"),
    definitions=(
      {"name": "file_read"},
      {"name": "propose_record"},
    ),
    local_tool_handlers={"file_read": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
  )

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == frozenset({"file_read", "propose_record"})
  assert {
    binding.capability for binding in authority.bindings
  } == {"artifact.propose/v1", "corpus.read/v1"}


def test_semantic_route_compiles_exact_read_only_tool_grant() -> None:
  authority = admit_operation_tools(
    _operation(),
    grant_id="grant:task-1",
    operation_tool_ids=("file_read", "web_search", "write_record"),
    definitions=(
      {"name": "file_read"},
      {"name": "web_search"},
      {"name": "write_record"},
      {"name": "undeclared_support"},
    ),
    local_tool_handlers={"file_read": object(), "undeclared_support": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
  )

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == frozenset({"file_read", "web_search"})
  assert [item.effect for item in authority.grant.tools] == ["read", "read"]
  assert derive_dispatcher_allowlist(authority) == {
    "web": frozenset({"web_search"}),
  }
  # Granular, not one coarse row: the workspace corpus and the open web are
  # separate declared domains bound to separate routes (B-8).
  assert {binding.capability for binding in authority.bindings} == {
    "corpus.read/v1",
    "web.read/v1",
  }
  assert parse_tool_grant(authority.grant) == authority.grant


def test_required_semantic_capability_fails_closed_as_a_visible_offer() -> None:
  """B-6: the Left is returned and named, never a silent drop or a raise.

  The unsatisfiable operation used to raise ``OperationToolAdmissionError``
  from inside the admission helper. It now resolves to the visible
  ``OperationUnavailable`` the ``run_agent`` boundary projects verbatim as
  ``operation_unavailable``, with the unmet capability named.
  """

  profile = SkillProfile(
    name="filings-only",
    system_prompt="Read filings.",
    agent_callable=True,
    mutation_mode="read_only",
    metadata={
      "semantic_metadata": {
        "tool_refs": [
          {
            "kind": "mcp",
            "server_id": "research-corpus-mcp",
            "tool_id": "filings_search",
          }
        ],
        "capability_requirements": [
          {
            "name": "filings.read/v1",
            "required": True,
            "binding_modes": ["live_tool"],
          }
        ],
      },
    },
  )
  # Coherent against its own ceiling; the *parent* is what cannot route it.
  operation = compile_agent_operation(profile, execution_class="node.explore")

  unavailable = admit_operation_tools(
    operation,
    grant_id="grant:task-2",
    operation_tool_ids=("filings_search",),
    definitions=({"name": "write_record"},),
    local_tool_handlers={},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
  )

  assert isinstance(unavailable, OperationUnavailable)
  assert unavailable.code == "missing_route"
  assert "no compatible admitted route" in unavailable.detail
  assert [item.capability for item in unavailable.unsatisfied] == [
    "filings.read/v1",
  ]


def test_resolved_authority_carries_the_threaded_execution_identity() -> None:
  """B-6/D-B6-1: the identity an operation executes under rides the authority."""

  identity = ExecutionIdentity(
    tenant_id="tenant-1",
    credential_handle_id="credential-1",
  )
  authority = admit_operation_tools(
    _operation(),
    grant_id="grant:task-identity",
    operation_tool_ids=("file_read", "web_search"),
    definitions=({"name": "file_read"}, {"name": "web_search"}),
    local_tool_handlers={"file_read": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
    identity=identity,
  )

  assert isinstance(authority, ResolvedAuthority)
  assert authority.identity == identity


def test_caller_exclusions_are_a_resolver_input_not_a_later_subtraction() -> None:
  """D-B6-2: an excluded tool is never granted, so nothing takes it back."""

  authority = admit_operation_tools(
    _operation(),
    grant_id="grant:task-excluded",
    operation_tool_ids=("file_read", "web_search"),
    definitions=({"name": "file_read"}, {"name": "web_search"}),
    local_tool_handlers={"file_read": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
    exclusions=frozenset({"file_read"}),
  )

  assert isinstance(authority, ResolvedAuthority)
  assert granted_tool_ids(authority) == frozenset({"web_search"})
  assert derive_dispatcher_allowlist(authority) == {"web": frozenset({"web_search"})}


def test_persisted_grant_digest_is_verified() -> None:
  authority = admit_operation_tools(
    _operation(),
    grant_id="grant:task-3",
    operation_tool_ids=("file_read", "web_search"),
    definitions=({"name": "file_read"}, {"name": "web_search"}),
    local_tool_handlers={"file_read": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
  )
  assert isinstance(authority, ResolvedAuthority)
  payload = authority.grant.model_dump(mode="json")
  payload["digest"] = "sha256:" + "0" * 64

  with pytest.raises(OperationToolAdmissionError, match="digest mismatch"):
    parse_tool_grant(payload)


def test_resume_scopes_come_only_from_persisted_grant() -> None:
  authority = admit_operation_tools(
    _operation(),
    grant_id="grant:task-4",
    operation_tool_ids=("file_read", "web_search"),
    definitions=({"name": "file_read"}, {"name": "web_search"}),
    local_tool_handlers={"file_read": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
  )
  assert isinstance(authority, ResolvedAuthority)
  reissued = reissue_tool_grant(
    authority.grant,
    grant_id="grant:task-4-resume",
  )

  tools, mcp_scope = scopes_from_tool_grant(
    reissued,
    local_tool_handlers={"file_read": object()},
    mcp_client=_Mcp(),
  )

  assert tools == frozenset({"file_read", "web_search"})
  assert mcp_scope == {"web": {"web_search"}}
  assert reissued.grant_id == "grant:task-4-resume"
  assert reissued.digest != authority.grant.digest
