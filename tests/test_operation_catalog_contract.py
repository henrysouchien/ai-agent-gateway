from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from agent_gateway.operation_catalog import (
  AgentOperationCatalog,
  OperationRuntimePolicy,
  ResolvedOperationRuntime,
)
from agent_workflow_contracts import (
  AgentOperationRef,
  AgentOperationSnapshot,
  ContractRef,
)


_DIGEST = "sha256:" + "a" * 64


def _operation_ref() -> AgentOperationRef:
  return AgentOperationRef(
    namespace="skills",
    name="filing-review",
    version="1.0",
    digest=_DIGEST,
  )


def _snapshot(**overrides: Any) -> AgentOperationSnapshot:
  payload: dict[str, Any] = {
    "operation": _operation_ref(),
    "methodology": ContractRef(
      namespace="skills",
      name="filing-review-methodology",
      version="1.0",
      digest=_DIGEST,
    ),
    "prompt": ContractRef(
      namespace="skills",
      name="filing-review-prompt",
      version="1.0",
      digest=_DIGEST,
    ),
    "description": "Review a filing.",
    "instructions": "Review the selected filing and report the findings.",
    "execution_class": "research",
    "workspace_scope": "read_only",
    "required_context": ("ticker",),
    "resumable": False,
    "result_modes": ("narrative",),
  }
  return AgentOperationSnapshot(**{**payload, **overrides})


def _policy_kwargs() -> dict[str, Any]:
  return {
    "semantic_scope": "ticker",
    "state_class": "advisor-no-state",
    "persist_state": False,
    "resumable": False,
    "resume_mcp_session_reset_ok": False,
    "state_dir": None,
    "mutation_mode": "read_only",
    "run_mode": "full",
    "exact_tool_ids": {"file_read", "mcp__filings-mcp__get_filing"},
    "mcp_tools_by_server": {"filings-mcp": {"get_filing"}},
    "runtime_server_refs": {"filings-mcp", "research-corpus-mcp"},
    "session_inject_servers": {"filings-mcp"},
    "timeout_overrides": {"filings-mcp": 45},
    "extra_excluded_tools": {"file_write"},
    "tool_packs": ("research",),
    "tool_packs_enabled": True,
    "model": "analysis-model",
    "provider": "openai",
    "max_turns": 12,
    "timeout_seconds": 300.0,
    "max_tokens": 8_000,
    "max_budget_usd": 2.5,
    "effort": "high",
    "max_retries": 2,
    "max_structured_reads": 8,
    "initial_message": "Reviewing the filing.",
    "delivery_label": "Filing review",
  }


def _policy(**overrides: Any) -> OperationRuntimePolicy:
  return OperationRuntimePolicy(**{**_policy_kwargs(), **overrides})


def test_runtime_policy_copies_and_deeply_freezes_input_collections() -> None:
  exact_tool_ids = {"file_read", "mcp__filings-mcp__get_filing"}
  mcp_tool_ids = {"get_filing"}
  mcp_tools = {"filings-mcp": mcp_tool_ids}
  runtime_servers = {"filings-mcp", "research-corpus-mcp"}
  timeout_overrides = {"filings-mcp": 45}

  policy = _policy(
    exact_tool_ids=exact_tool_ids,
    mcp_tools_by_server=mcp_tools,
    runtime_server_refs=runtime_servers,
    timeout_overrides=timeout_overrides,
  )
  exact_tool_ids.add("file_write")
  mcp_tool_ids.add("get_sections")
  mcp_tools["research-corpus-mcp"] = {"search"}
  runtime_servers.add("new-server")
  timeout_overrides["filings-mcp"] = 1

  assert policy.exact_tool_ids == frozenset({
    "file_read",
    "mcp__filings-mcp__get_filing",
  })
  assert policy.mcp_tools_by_server == {
    "filings-mcp": frozenset({"get_filing"}),
  }
  assert policy.runtime_server_refs == frozenset({
    "filings-mcp",
    "research-corpus-mcp",
  })
  assert policy.timeout_overrides == {"filings-mcp": 45}
  with pytest.raises(TypeError):
    policy.mcp_tools_by_server["filings-mcp"] = frozenset()  # type: ignore[index]
  with pytest.raises(TypeError):
    policy.timeout_overrides["filings-mcp"] = 1  # type: ignore[index]
  with pytest.raises(FrozenInstanceError):
    policy.max_turns = 1  # type: ignore[misc]


@pytest.mark.parametrize(
  ("overrides", "match"),
  [
    ({"semantic_scope": "account"}, "semantic_scope"),
    ({"state_class": "mutable"}, "state_class"),
    ({"mutation_mode": "dry_run"}, "mutation_mode"),
    ({"run_mode": "automatic"}, "run_mode"),
    ({"effort": "extreme"}, "effort"),
    ({"persist_state": 1}, "persist_state"),
    ({"max_turns": True}, "max_turns"),
    ({"max_turns": 0}, "max_turns"),
    ({"max_retries": -1}, "max_retries"),
    ({"timeout_seconds": float("inf")}, "timeout_seconds"),
    ({"max_budget_usd": float("nan")}, "max_budget_usd"),
    ({"state_dir": "../outside"}, "state_dir"),
    ({"state_dir": "state//filing-review"}, "state_dir"),
    ({"tool_packs": ("zeta", "alpha")}, "tool_packs"),
  ],
)
def test_runtime_policy_rejects_noncanonical_values(
  overrides: dict[str, Any],
  match: str,
) -> None:
  with pytest.raises((TypeError, ValueError), match=match):
    _policy(**overrides)


def test_runtime_policy_requires_truthful_state_lifecycle() -> None:
  with pytest.raises(ValueError, match="cannot persist"):
    _policy(persist_state=True)
  with pytest.raises(ValueError, match="session reset"):
    _policy(
      resumable=True,
      resume_mcp_session_reset_ok=False,
      session_inject_servers={"filings-mcp"},
    )


@pytest.mark.parametrize(
  "overrides",
  [
    {
      "exact_tool_ids": {"file_read"},
      "mcp_tools_by_server": {"filings-mcp": {"get_filing"}},
    },
    {
      "exact_tool_ids": {
        "file_read",
        "mcp__filings-mcp__get_filing",
        "mcp__filings-mcp__get_sections",
      },
    },
  ],
)
def test_runtime_policy_requires_exact_mcp_projection(
  overrides: dict[str, Any],
) -> None:
  with pytest.raises(ValueError, match="project exactly"):
    _policy(**overrides)


def test_runtime_policy_separates_server_availability_from_tool_grants() -> None:
  policy = _policy()

  assert "research-corpus-mcp" in policy.runtime_server_refs
  assert "research-corpus-mcp" not in policy.mcp_tools_by_server
  assert not any(
    tool_id.startswith("mcp__research-corpus-mcp__")
    for tool_id in policy.exact_tool_ids
  )


@pytest.mark.parametrize(
  ("overrides", "match"),
  [
    (
      {"runtime_server_refs": {"research-corpus-mcp"}},
      "exact MCP tool server",
    ),
    (
      {"session_inject_servers": {"unavailable-mcp"}},
      "subset of runtime_server_refs",
    ),
    (
      {"timeout_overrides": {"unavailable-mcp": 10}},
      "runtime server refs",
    ),
    (
      {"timeout_overrides": {"filings-mcp": True}},
      "positive integer",
    ),
  ],
)
def test_runtime_policy_rejects_invalid_server_relations(
  overrides: dict[str, Any],
  match: str,
) -> None:
  with pytest.raises(ValueError, match=match):
    _policy(**overrides)


@pytest.mark.parametrize(
  "excluded_tool_id",
  ["mcp__filings-mcp__get_filing", "get_filing"],
)
def test_runtime_policy_rejects_contradictory_exclusions(
  excluded_tool_id: str,
) -> None:
  with pytest.raises(ValueError, match="must not overlap"):
    _policy(extra_excluded_tools={excluded_tool_id})


@pytest.mark.parametrize(
  "excluded_tool_id",
  ["not a tool!", "mcp__BadServer__get_filing", "mcp__missing_tool"],
)
def test_runtime_policy_rejects_malformed_exclusion_identifiers(
  excluded_tool_id: str,
) -> None:
  with pytest.raises(ValueError, match="invalid excluded"):
    _policy(extra_excluded_tools={excluded_tool_id})


def test_resolved_runtime_requires_exact_package_owned_types() -> None:
  resolved = ResolvedOperationRuntime(snapshot=_snapshot(), policy=_policy())

  assert resolved.snapshot.operation == _operation_ref()
  with pytest.raises(TypeError, match="AgentOperationSnapshot"):
    ResolvedOperationRuntime(snapshot=object(), policy=_policy())  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="OperationRuntimePolicy"):
    ResolvedOperationRuntime(snapshot=_snapshot(), policy=object())  # type: ignore[arg-type]


def test_resolved_runtime_requires_one_resumability_answer() -> None:
  with pytest.raises(ValueError, match="agree on resumability"):
    ResolvedOperationRuntime(
      snapshot=_snapshot(resumable=False),
      policy=_policy(
        resumable=True,
        session_inject_servers=frozenset(),
      ),
    )


@pytest.mark.parametrize(
  ("mutation_mode", "workspace_scope"),
  [
    ("read_only", "workspace_write"),
    ("preview", "read_only"),
    ("apply", "model_write"),
    ("thesis_writer", "workspace_write"),
  ],
)
def test_resolved_runtime_rejects_incoherent_workspace_scope(
  mutation_mode: str,
  workspace_scope: str,
) -> None:
  with pytest.raises(ValueError, match="workspace_scope"):
    ResolvedOperationRuntime(
      snapshot=_snapshot(workspace_scope=workspace_scope),
      policy=_policy(mutation_mode=mutation_mode),
    )


@pytest.mark.parametrize("workspace_scope", ["workspace_write", "model_write"])
def test_model_writer_accepts_both_supported_workspace_scopes(
  workspace_scope: str,
) -> None:
  resolved = ResolvedOperationRuntime(
    snapshot=_snapshot(workspace_scope=workspace_scope),
    policy=_policy(mutation_mode="model_writer"),
  )

  assert resolved.snapshot.workspace_scope == workspace_scope


class _FakeCatalog:
  def __init__(self, resolved: ResolvedOperationRuntime) -> None:
    self._resolved = resolved

  def resolve_operation(
    self,
    selector: AgentOperationRef | dict[str, Any] | None,
  ) -> ResolvedOperationRuntime:
    del selector
    return self._resolved

  def list_callable_operations_with_descriptions(
    self,
  ) -> tuple[tuple[AgentOperationRef, str], ...]:
    return ((self._resolved.snapshot.operation, "Review a filing."),)


def test_catalog_protocol_is_structural_and_returns_the_exact_runtime_pair() -> None:
  resolved = ResolvedOperationRuntime(snapshot=_snapshot(), policy=_policy())
  catalog = _FakeCatalog(resolved)

  assert isinstance(catalog, AgentOperationCatalog)
  assert catalog.resolve_operation(None) is resolved
  assert catalog.list_callable_operations_with_descriptions() == (
    (_operation_ref(), "Review a filing."),
  )


def test_gateway_operation_catalog_has_no_application_import_boundary() -> None:
  source_path = (
    Path(__file__).parents[1] / "agent_gateway" / "operation_catalog.py"
  )
  tree = ast.parse(source_path.read_text(encoding="utf-8"))
  imported_roots: set[str] = set()
  relative_imports: list[ast.ImportFrom] = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      if node.level:
        relative_imports.append(node)
      if node.module:
        imported_roots.add(node.module.split(".", 1)[0])

  assert not relative_imports
  assert imported_roots <= {
    "__future__",
    "agent_workflow_contracts",
    "collections",
    "dataclasses",
    "math",
    "pathlib",
    "re",
    "types",
    "typing",
  }
