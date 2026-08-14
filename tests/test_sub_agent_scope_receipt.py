from __future__ import annotations

import pytest

from agent_gateway.skills import (
  SkillProfile,
  compile_agent_operation,
  generic_explore_profile,
)
from agent_gateway.sub_agent_scope_receipt import (
  OperationToolAdmissionError,
  admit_operation_tools,
  parse_tool_grant,
  reissue_tool_grant,
  scopes_from_tool_grant,
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
  } == {"artifact.propose/v1", "research-evidence.read/v1"}
  admission = admit_operation_tools(
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

  assert admission.tool_ids == frozenset({"file_read", "propose_record"})
  assert {
    binding.capability for binding in admission.capability_bindings
  } == {"artifact.propose/v1", "research-evidence.read/v1"}


def test_semantic_route_compiles_exact_read_only_tool_grant() -> None:
  admission = admit_operation_tools(
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

  assert admission.tool_ids == frozenset({"file_read", "web_search"})
  assert [item.effect for item in admission.tool_grant.tools] == ["read", "read"]
  assert admission.mcp_tools_by_server == {
    "web": frozenset({"web_search"}),
  }
  assert admission.capability_bindings[0].capability == (
    "research-evidence.read/v1"
  )
  assert parse_tool_grant(admission.tool_grant) == admission.tool_grant


def test_required_semantic_capability_fails_closed_without_live_route() -> None:
  with pytest.raises(OperationToolAdmissionError, match="no compatible admitted route"):
    admit_operation_tools(
      _operation(),
      grant_id="grant:task-2",
      operation_tool_ids=("write_record",),
      definitions=({"name": "write_record"},),
      local_tool_handlers={},
      mcp_client=_Mcp(),
      effect_resolver=_effect,
    )


def test_persisted_grant_digest_is_verified() -> None:
  admission = admit_operation_tools(
    _operation(),
    grant_id="grant:task-3",
    operation_tool_ids=("file_read",),
    definitions=({"name": "file_read"},),
    local_tool_handlers={"file_read": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
  )
  payload = admission.tool_grant.model_dump(mode="json")
  payload["digest"] = "sha256:" + "0" * 64

  with pytest.raises(OperationToolAdmissionError, match="digest mismatch"):
    parse_tool_grant(payload)


def test_resume_scopes_come_only_from_persisted_grant() -> None:
  admission = admit_operation_tools(
    _operation(),
    grant_id="grant:task-4",
    operation_tool_ids=("file_read", "web_search"),
    definitions=({"name": "file_read"}, {"name": "web_search"}),
    local_tool_handlers={"file_read": object()},
    mcp_client=_Mcp(),
    effect_resolver=_effect,
  )
  reissued = reissue_tool_grant(
    admission.tool_grant,
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
  assert reissued.digest != admission.tool_grant.digest
