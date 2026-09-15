from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

from agent_workflow_contracts.tool_registration import (
  AmbiguousToolRegistrationError,
  McpInputPreparationRoute,
  RegisteredToolIdentity,
  RegisteredToolServerDescriptor,
  ToolApprovalPolicy,
  ToolIntrinsicSemantics,
  ToolRegistrationCatalog,
  ToolRegistrationContractError,
  ToolRegistrationDeclaration,
  UnknownToolRegistrationError,
  VersionedPolicyRef,
  index_mcp_input_preparation_routes,
  validate_tool_registration_catalog,
  validate_versioned_policy_ref,
)


ROOT = Path(__file__).resolve().parents[1]


def _input_preparation_route(
  server: str = "market-data-mcp",
  name: str = "fetch_financials",
) -> McpInputPreparationRoute:
  return McpInputPreparationRoute(
    logical_server_id=server,
    logical_name=name,
    mode="scalar",
    keys=("symbol",),
  )


def test_input_preparation_index_defaults_empty_and_requires_exact_identity() -> None:
  assert index_mcp_input_preparation_routes() == {}

  route = _input_preparation_route()
  index = index_mcp_input_preparation_routes((route,))

  assert isinstance(index, MappingProxyType)
  assert index[("market-data-mcp", "fetch_financials")] is route
  assert ("edgar-parser-mcp", "fetch_financials") not in index
  with pytest.raises(TypeError):
    index[("other", "route")] = route  # pyright: ignore[reportIndexIssue]  # negative: frozen route index mutation


def test_input_preparation_index_rejects_duplicates_and_non_routes() -> None:
  with pytest.raises(ToolRegistrationContractError, match="duplicate"):
    index_mcp_input_preparation_routes((
      _input_preparation_route(),
      _input_preparation_route(),
    ))
  with pytest.raises(TypeError, match="exact McpInputPreparationRoute"):
    index_mcp_input_preparation_routes((object(),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
  ("changes", "message"),
  [
    ({"mode": "list"}, "unsupported MCP input-preparation mode"),
    ({"resolver": "network-symbol-lookup"}, "unsupported MCP input-preparation resolver"),
    ({"keys": ("symbol", "ticker")}, "scalar requires exactly 1 key"),
    (
      {"mode": "comma-separated", "keys": ("symbols", "tickers")},
      "comma-separated requires exactly 1 key",
    ),
    (
      {"mode": "consistent-present-keys", "keys": ("symbol",)},
      "consistent-present-keys requires exactly 2 key",
    ),
    (
      {
        "mode": "consistent-present-keys",
        "keys": ("symbol", "ticker", "security"),
      },
      "consistent-present-keys requires exactly 2 key",
    ),
    (
      {"mode": "consistent-present-keys", "keys": ("ticker", "symbol")},
      "consistent-present-keys requires exact keys",
    ),
    (
      {"logical_name": "tool-registration:fetch_financials"},
      "reserved tool-registration namespace",
    ),
  ],
)
def test_input_preparation_route_rejects_values_outside_closed_grammar(
  changes: dict[str, object],
  message: str,
) -> None:
  values: dict[str, object] = {
    "logical_server_id": "market-data-mcp",
    "logical_name": "fetch_financials",
    "mode": "scalar",
    "keys": ("symbol",),
    "resolver": "sec-native-symbol-cached-only",
  }
  values.update(changes)

  with pytest.raises(ToolRegistrationContractError, match=message):
    McpInputPreparationRoute(**values)  # type: ignore[arg-type]


def test_input_preparation_route_rejects_incoherent_consistent_keys() -> None:
  with pytest.raises(ToolRegistrationContractError, match="must not contain duplicates"):
    McpInputPreparationRoute(
      logical_server_id="market-data-mcp",
      logical_name="fetch_company_profile",
      mode="consistent-present-keys",
      keys=("symbol", "symbol"),
    )


def _policy(
  kind: str,
  policy_id: str | None = None,
  *,
  parameters: object | None = None,
) -> VersionedPolicyRef:
  return VersionedPolicyRef(
    kind=kind,  # type: ignore[arg-type]
    policy_id=policy_id or f"{kind}.identity",
    version="1",
    parameters={} if parameters is None else parameters,  # type: ignore[arg-type]
  )


def _semantics(
  *,
  effect: str = "read",
  idempotent: bool = True,
  approval: ToolApprovalPolicy | None = None,
) -> ToolIntrinsicSemantics:
  return ToolIntrinsicSemantics(
    effect=effect,  # type: ignore[arg-type]
    idempotent=idempotent,
    semantic_capability="corpus.read/v1",
    approval=approval or ToolApprovalPolicy(mode="never"),
    audience="ordinary",
    redaction_policy=_policy("redaction"),
    planning_policy=_policy("planning"),
    input_preparation_policy=_policy("input_preparation"),
    outcome_policy=_policy("outcome"),
    source_identity_policy=_policy("source_identity"),
  )


def _declaration(
  route_kind: str,
  name: str,
  *,
  server_id: str | None = None,
) -> ToolRegistrationDeclaration:
  return ToolRegistrationDeclaration(
    identity=RegisteredToolIdentity(
      route_kind=route_kind,  # type: ignore[arg-type]
      logical_name=name,
      logical_server_id=server_id,
    ),
    semantics=_semantics(),
  )


def _server(
  logical_server_id: str = "research-mcp",
) -> RegisteredToolServerDescriptor:
  return RegisteredToolServerDescriptor(
    logical_server_id=logical_server_id,
    transport_server_id=f"{logical_server_id}-transport",
    default_timeout_seconds=30,
    per_tool_timeout_seconds={},
    session_injection_policy=_policy("session_injection"),
  )


def test_identity_is_exact_and_registration_key_is_stable() -> None:
  identity = RegisteredToolIdentity(
    route_kind="mcp",
    logical_name="filings_read",
    logical_server_id="research-mcp",
  )

  assert identity.materialize() == {
    "logical_name": "filings_read",
    "logical_server_id": "research-mcp",
    "route_kind": "mcp",
  }
  assert identity.registration_key == (
    "tool-registration:sha256:"
    "5537da967aeee295794fd0956ec03615f482a876673be1358552b0b1764a8e0d"
  )
  assert not hasattr(identity, "__dict__")
  with pytest.raises(FrozenInstanceError):
    identity.logical_name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
  ("route_kind", "name", "server_id", "error_type"),
  [
    ("mcp", "tool", None, TypeError),
    ("local_handler", "tool", "server", ToolRegistrationContractError),
    ("addin_relay", "tool", "server", ToolRegistrationContractError),
    ("unknown", "tool", None, ToolRegistrationContractError),
    ("mcp", " padded ", "server", ToolRegistrationContractError),
    ("mcp", "tool", " padded ", ToolRegistrationContractError),
    ("mcp", "tool-registration:sha256:hostile", "server", ToolRegistrationContractError),
  ],
)
def test_identity_rejects_incoherent_or_reserved_routes(
  route_kind: object,
  name: object,
  server_id: object,
  error_type: type[Exception],
) -> None:
  with pytest.raises(error_type):
    RegisteredToolIdentity(
      route_kind=route_kind,  # type: ignore[arg-type]
      logical_name=name,  # type: ignore[arg-type]
      logical_server_id=server_id,  # type: ignore[arg-type]
    )


def test_policy_parameters_are_deeply_frozen_detached_and_freshly_materialized() -> None:
  source = {
    "keys": ["ticker", "period"],
    "shape": {"required": True},
  }
  policy = _policy("input_preparation", parameters=source)
  source["keys"].append("mutated")
  source["shape"]["required"] = False

  assert type(policy.parameters) is MappingProxyType
  assert policy.parameters["keys"] == ("ticker", "period")
  frozen_shape = policy.parameters["shape"]
  assert type(frozen_shape) is MappingProxyType
  assert frozen_shape["required"] is True
  first = policy.materialize()
  second = policy.materialize()
  assert first == second
  assert first is not second
  assert first["parameters"] is not second["parameters"]


@pytest.mark.parametrize(
  "parameters",
  [
    {"bad": {"set"}},
    {1: "bad"},
    {"bad": float("inf")},
    {"bad": lambda: None},
  ],
)
def test_policy_parameters_reject_non_json_or_nonfinite_values(
  parameters: object,
) -> None:
  with pytest.raises((TypeError, ToolRegistrationContractError)):
    _policy("planning", parameters=parameters)


def test_policy_validator_rebuilds_hostile_mapping_proxy_backing() -> None:
  backing = {"mode": "exact"}
  policy = _policy("planning", parameters={"mode": "safe"})
  object.__setattr__(policy, "parameters", MappingProxyType(backing))

  canonical = validate_versioned_policy_ref(policy)
  backing["mode"] = "mutated"

  assert canonical is not policy
  assert canonical.materialize()["parameters"] == {"mode": "exact"}


def test_approval_policy_requires_coherent_typed_references() -> None:
  predicate = _policy("approval_predicate")
  cache_key = _policy("approval_cache_key")
  approval = ToolApprovalPolicy(
    mode="predicate",
    predicate=predicate,
    cache_key=cache_key,
  )
  assert approval.materialize()["predicate"] == predicate.materialize()

  with pytest.raises(ToolRegistrationContractError, match="requires"):
    ToolApprovalPolicy(mode="predicate")
  with pytest.raises(ToolRegistrationContractError, match="only predicate"):
    ToolApprovalPolicy(mode="always", predicate=predicate)
  with pytest.raises(ToolRegistrationContractError, match="cache=never"):
    ToolApprovalPolicy(mode="never", cache_key=cache_key)
  with pytest.raises(ToolRegistrationContractError, match="wrong policy kind"):
    ToolApprovalPolicy(mode="predicate", predicate=_policy("planning"))
  with pytest.raises(TypeError, match="exact VersionedPolicyRef"):
    ToolApprovalPolicy(
      mode="predicate",
      predicate=object(),  # type: ignore[arg-type]
    )
  with pytest.raises(TypeError, match="exact VersionedPolicyRef"):
    ToolApprovalPolicy(
      mode="always",
      cache_key=object(),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
  ("field_name", "replacement", "error_type"),
  [
    ("effect", "unknown", ToolRegistrationContractError),
    ("idempotent", 1, TypeError),
    ("semantic_capability", "", ToolRegistrationContractError),
    ("audience", "unknown", ToolRegistrationContractError),
    ("declaration_mode_exception", "unknown", ToolRegistrationContractError),
  ],
)
def test_intrinsic_semantics_are_total_and_exact(
  field_name: str,
  replacement: object,
  error_type: type[Exception],
) -> None:
  values = {
    "effect": "read",
    "idempotent": True,
    "semantic_capability": "corpus.read/v1",
    "approval": ToolApprovalPolicy(mode="never"),
    "audience": "ordinary",
    "redaction_policy": _policy("redaction"),
    "planning_policy": _policy("planning"),
    "input_preparation_policy": _policy("input_preparation"),
    "outcome_policy": _policy("outcome"),
    "source_identity_policy": _policy("source_identity"),
    "declaration_mode_exception": None,
  }
  values[field_name] = replacement
  with pytest.raises(error_type):
    ToolIntrinsicSemantics(**values)


def test_intrinsic_policy_kinds_cannot_be_swapped() -> None:
  with pytest.raises(ToolRegistrationContractError, match="wrong policy kind"):
    ToolIntrinsicSemantics(
      effect="read",
      idempotent=True,
      semantic_capability="corpus.read/v1",
      approval=ToolApprovalPolicy(mode="never"),
      audience="ordinary",
      redaction_policy=_policy("planning"),
      planning_policy=_policy("planning"),
      input_preparation_policy=_policy("input_preparation"),
      outcome_policy=_policy("outcome"),
      source_identity_policy=_policy("source_identity"),
    )


def test_preview_state_write_exception_requires_state_write_effect() -> None:
  values = {
    "effect": "read",
    "idempotent": True,
    "semantic_capability": "corpus.read/v1",
    "approval": ToolApprovalPolicy(mode="never"),
    "audience": "ordinary",
    "redaction_policy": _policy("redaction"),
    "planning_policy": _policy("planning"),
    "input_preparation_policy": _policy("input_preparation"),
    "outcome_policy": _policy("outcome"),
    "source_identity_policy": _policy("source_identity"),
    "declaration_mode_exception": "preview_allows_state_write",
  }
  with pytest.raises(ToolRegistrationContractError, match="state_write effect"):
    ToolIntrinsicSemantics(**values)

  values["effect"] = "state_write"
  assert (
    ToolIntrinsicSemantics(**values).declaration_mode_exception
    == "preview_allows_state_write"
  )


def test_server_descriptor_detaches_timeouts_and_requires_session_policy() -> None:
  timeouts = {"filings_read": 45}
  server = RegisteredToolServerDescriptor(
    logical_server_id="research-mcp",
    transport_server_id="research-transport",
    default_timeout_seconds=30,
    per_tool_timeout_seconds=timeouts,
    session_injection_policy=_policy("session_injection"),
  )
  timeouts["filings_read"] = 1

  assert server.default_timeout_seconds == 30.0
  assert type(server.per_tool_timeout_seconds) is MappingProxyType
  assert server.per_tool_timeout_seconds == {"filings_read": 45.0}
  with pytest.raises(ToolRegistrationContractError, match="wrong policy kind"):
    RegisteredToolServerDescriptor(
      logical_server_id="research-mcp",
      transport_server_id="research-transport",
      default_timeout_seconds=30,
      per_tool_timeout_seconds={},
      session_injection_policy=_policy("planning"),
    )


@pytest.mark.parametrize("timeout", [True, 0, -1, float("inf")])
def test_server_descriptor_rejects_invalid_timeouts(timeout: object) -> None:
  with pytest.raises((TypeError, ToolRegistrationContractError)):
    RegisteredToolServerDescriptor(
      logical_server_id="research-mcp",
      transport_server_id="research-transport",
      default_timeout_seconds=timeout,  # type: ignore[arg-type]
      per_tool_timeout_seconds={},
      session_injection_policy=_policy("session_injection"),
    )


def test_catalog_detaches_sorts_and_resolves_exact_routes() -> None:
  local = _declaration("local_handler", "memory_read")
  same_bare_mcp = _declaration(
    "mcp",
    "memory_read",
    server_id="research-mcp",
  )
  catalog = ToolRegistrationCatalog(
    declarations=(same_bare_mcp, local),
    servers=(_server(),),
  )

  assert catalog.declarations == (local, same_bare_mcp)
  assert catalog.by_identity(local.identity) is not local
  assert catalog.resolve_local("memory_read").identity == local.identity
  assert catalog.resolve_mcp("research-mcp", "memory_read").identity == (
    same_bare_mcp.identity
  )
  assert catalog.by_registration_key(local.identity.registration_key).identity == (
    local.identity
  )
  assert catalog.server("research-mcp").transport_server_id == (
    "research-mcp-transport"
  )


def test_catalog_rejects_duplicate_or_serverless_mcp_declarations() -> None:
  local = _declaration("local_handler", "memory_read")
  with pytest.raises(ToolRegistrationContractError, match="duplicate tool"):
    ToolRegistrationCatalog(declarations=(local, local), servers=())

  mcp = _declaration("mcp", "filings_read", server_id="research-mcp")
  with pytest.raises(ToolRegistrationContractError, match="unregistered"):
    ToolRegistrationCatalog(declarations=(mcp,), servers=())

  with pytest.raises(ToolRegistrationContractError, match="duplicate logical server"):
    ToolRegistrationCatalog(declarations=(), servers=(_server(), _server()))


def test_catalog_rejects_timeout_for_unregistered_or_other_server_tool() -> None:
  server = RegisteredToolServerDescriptor(
    logical_server_id="research-mcp",
    transport_server_id="research-transport",
    default_timeout_seconds=30,
    per_tool_timeout_seconds={"other_tool": 45},
    session_injection_policy=_policy("session_injection"),
  )
  declaration = _declaration(
    "mcp",
    "filings_read",
    server_id="research-mcp",
  )
  with pytest.raises(ToolRegistrationContractError, match="unregistered tool"):
    ToolRegistrationCatalog(
      declarations=(declaration,),
      servers=(server,),
    )


def test_catalog_reports_unknown_and_ambiguous_source_selectors() -> None:
  local = _declaration("local_handler", "read_cells")
  addin = _declaration("addin_relay", "read_cells")
  catalog = ToolRegistrationCatalog(
    declarations=(local, addin),
    servers=(),
  )

  with pytest.raises(AmbiguousToolRegistrationError):
    catalog.resolve_local("read_cells")
  with pytest.raises(UnknownToolRegistrationError):
    catalog.resolve_local("missing")
  with pytest.raises(UnknownToolRegistrationError):
    catalog.by_registration_key("tool-registration:sha256:" + "0" * 64)


def test_catalog_validator_reconstructs_forged_nested_records() -> None:
  declaration = _declaration("local_handler", "memory_read")
  catalog = ToolRegistrationCatalog(declarations=(declaration,), servers=())
  forged_declaration = catalog.declarations[0]
  forged_policy = forged_declaration.semantics.planning_policy
  backing = {"mode": "exact"}
  object.__setattr__(forged_policy, "parameters", MappingProxyType(backing))

  canonical = validate_tool_registration_catalog(catalog)
  backing["mode"] = "mutated"

  assert canonical is not catalog
  assert canonical.declarations[0].semantics.planning_policy.materialize()[
    "parameters"
  ] == {"mode": "exact"}


def test_contract_module_is_stdlib_only_not_reexported_and_imports_no_gateway() -> None:
  source = (
    ROOT / "agent_workflow_contracts" / "tool_registration.py"
  ).read_text(encoding="utf-8")
  allowed = {
    "__future__": {"annotations"},
    "collections.abc": {"Iterable", "Mapping"},
    "dataclasses": {"dataclass", "field"},
    "types": {"MappingProxyType"},
    "typing": {"Literal"},
  }
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
      assert all(alias.asname is None for alias in node.names)
      assert {alias.name for alias in node.names}.issubset(
        {"hashlib", "json", "math"}
      )
      continue
    if not isinstance(node, ast.ImportFrom):
      continue
    assert node.level == 0
    assert node.module in allowed
    assert all(alias.asname is None for alias in node.names)
    assert {alias.name for alias in node.names}.issubset(allowed[node.module])

  package_init = (
    ROOT / "agent_workflow_contracts" / "__init__.py"
  ).read_text(encoding="utf-8")
  assert "tool_registration" not in package_init

  env = dict(os.environ)
  env["PYTHONPATH"] = str(ROOT)
  probe = subprocess.run(
    [
      sys.executable,
      "-c",
      (
        "import sys; "
        "import agent_workflow_contracts.tool_registration; "
        "assert not any(name == 'agent_gateway' or "
        "name.startswith('agent_gateway.') for name in sys.modules)"
      ),
    ],
    cwd=ROOT,
    env=env,
    check=False,
    capture_output=True,
    text=True,
  )
  assert probe.returncode == 0, probe.stderr
