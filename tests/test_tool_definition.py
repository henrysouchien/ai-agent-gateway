from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from agent_gateway.tool_definition import (
  LiveToolRouteBinding,
  OriginatedToolDefinition,
  validate_live_tool_route_binding,
  validate_originated_tool_definition,
)


ROOT = Path(__file__).resolve().parents[1]


def _definition() -> dict[str, object]:
  return {
    "name": "sample_tool",
    "description": "Sample tool.",
    "input_schema": {
      "type": "object",
      "properties": {
        "ticker": {"type": "string", "enum": ["AAPL", "MSFT"]},
      },
      "required": ["ticker"],
    },
  }


def test_originated_tool_definition_is_deeply_frozen_and_detached() -> None:
  source = _definition()
  record = OriginatedToolDefinition(
    definition=source,
    origin="mcp",
    server_id="research-mcp",
  )
  source["name"] = "mutated"
  source_schema = source["input_schema"]
  assert isinstance(source_schema, dict)
  source_schema["required"] = ["changed"]

  assert record.name == "sample_tool"
  assert type(record.definition) is MappingProxyType
  frozen_schema = record.definition["input_schema"]
  assert type(frozen_schema) is MappingProxyType
  assert frozen_schema["required"] == ("ticker",)
  with pytest.raises(TypeError):
    record.definition["name"] = "changed"  # type: ignore[index]
  with pytest.raises(TypeError):
    frozen_schema["type"] = "array"  # type: ignore[index]
  with pytest.raises(FrozenInstanceError):
    record.origin = "local"  # type: ignore[misc]
  assert not hasattr(record, "__dict__")


def test_materialize_returns_fresh_deep_provider_dictionaries() -> None:
  record = OriginatedToolDefinition(
    definition=_definition(),
    origin="local",
    server_id=None,
  )
  first = record.materialize()
  second = record.materialize()

  assert first == second == _definition()
  assert first is not second
  assert first["input_schema"] is not second["input_schema"]
  first["name"] = "changed"
  first["input_schema"]["required"].append("other")
  assert second == _definition()
  assert record.name == "sample_tool"
  assert "origin" not in second
  assert "server_id" not in second


@pytest.mark.parametrize(
  ("origin", "server_id", "error_type"),
  [
    ("local", "server", ValueError),
    ("mcp", None, TypeError),
    ("mcp", "", ValueError),
    ("mcp", " padded ", ValueError),
    ("unknown", None, ValueError),
    (1, None, TypeError),
  ],
)
def test_origin_and_server_coherence_is_exact(
  origin: object,
  server_id: object,
  error_type: type[Exception],
) -> None:
  with pytest.raises(error_type):
    OriginatedToolDefinition(
      definition=_definition(),
      origin=origin,  # type: ignore[arg-type]
      server_id=server_id,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
  ("definition", "error_type"),
  [
    ({}, TypeError),
    ({"name": 1}, TypeError),
    ({"name": ""}, ValueError),
    ({"name": "tool", 1: "value"}, TypeError),
    ({"name": "tool", "invalid": {"set"}}, TypeError),
    ([{"name": "tool"}], TypeError),
  ],
)
def test_definition_requires_deep_json_like_mapping_and_exact_name(
  definition: object,
  error_type: type[Exception],
) -> None:
  with pytest.raises(error_type):
    OriginatedToolDefinition(
      definition=definition,  # type: ignore[arg-type]
      origin="local",
      server_id=None,
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_definition_rejects_nonfinite_floats_anywhere(value: float) -> None:
  with pytest.raises(ValueError, match="finite"):
    OriginatedToolDefinition(
      definition={"name": "tool", "input_schema": {"default": value}},
      origin="local",
      server_id=None,
    )


def test_public_validator_refuses_forged_postconstruction_origin_state() -> None:
  record = OriginatedToolDefinition(
    definition=_definition(),
    origin="local",
    server_id=None,
  )
  object.__setattr__(record, "origin", "foreign")
  object.__setattr__(record, "server_id", "BadServer")

  with pytest.raises(ValueError, match="origin"):
    validate_originated_tool_definition(record)


@pytest.mark.parametrize(
  "definition",
  [
    {"name": "sample_tool"},
    MappingProxyType({
      "name": "sample_tool",
      "input_schema": MappingProxyType({"default": float("inf")}),
    }),
  ],
)
def test_public_validator_refuses_forged_definition_graphs(
  definition: object,
) -> None:
  record = OriginatedToolDefinition(
    definition=_definition(),
    origin="local",
    server_id=None,
  )
  object.__setattr__(record, "definition", definition)

  with pytest.raises((TypeError, ValueError)):
    validate_originated_tool_definition(record)


def test_public_validator_returns_a_detached_canonical_record() -> None:
  nested_backing = {"type": "object"}
  root_backing = {
    "name": "sample_tool",
    "input_schema": MappingProxyType(nested_backing),
  }
  record = OriginatedToolDefinition(
    definition=_definition(),
    origin="mcp",
    server_id="My_Server",
  )
  object.__setattr__(record, "definition", MappingProxyType(root_backing))

  canonical = validate_originated_tool_definition(record)
  root_backing["name"] = "mutated"
  nested_backing["type"] = "array"

  assert canonical is not record
  assert canonical.name == "sample_tool"
  assert canonical.server_id == "My_Server"
  assert canonical.materialize()["input_schema"] == {"type": "object"}


def test_live_tool_route_binding_is_exact_frozen_and_detached() -> None:
  definition = OriginatedToolDefinition(
    definition=_definition(),
    origin="mcp",
    server_id="market-data-mcp",
  )
  binding = LiveToolRouteBinding(
    originated_definition=definition,
    route_kind="logical",
    logical_name="sample_tool",
    transport_server_id="FMP_Server:1",
    provider_original_name="fmp_sample",
    provider_id="fmp",
  )

  assert binding.exposed_name == "sample_tool"
  assert binding.logical_server_id == "market-data-mcp"
  assert binding.materialize_provider_definition() == _definition()
  assert binding.originated_definition is not definition
  with pytest.raises(FrozenInstanceError):
    binding.provider_id = "other"  # type: ignore[misc]
  assert not hasattr(binding, "__dict__")

  canonical = validate_live_tool_route_binding(binding)
  assert canonical == binding
  assert canonical is not binding
  assert canonical.originated_definition is not binding.originated_definition


def test_physical_route_requires_exact_original_and_transport_identity() -> None:
  definition = OriginatedToolDefinition(
    definition={"name": "prefix__sample_tool"},
    origin="mcp",
    server_id="My_Server",
  )
  binding = LiveToolRouteBinding(
    originated_definition=definition,
    route_kind="physical",
    logical_name="sample_tool",
    transport_server_id="My_Server",
    provider_original_name="sample_tool",
    provider_id=None,
  )
  assert binding.exposed_name == "prefix__sample_tool"

  with pytest.raises(ValueError, match="transport"):
    LiveToolRouteBinding(
      originated_definition=definition,
      route_kind="physical",
      logical_name="sample_tool",
      transport_server_id="other",
      provider_original_name="sample_tool",
      provider_id=None,
    )

  with pytest.raises(ValueError, match="provider_original"):
    LiveToolRouteBinding(
      originated_definition=definition,
      route_kind="physical",
      logical_name="other",
      transport_server_id="My_Server",
      provider_original_name="sample_tool",
      provider_id=None,
    )


@pytest.mark.parametrize(
  ("overrides", "error_type"),
  [
    ({"route_kind": "unknown"}, ValueError),
    ({"route_kind": 1}, TypeError),
    ({"logical_name": " padded "}, ValueError),
    ({"transport_server_id": ""}, ValueError),
    ({"provider_original_name": 1}, TypeError),
    ({"provider_id": ""}, ValueError),
  ],
)
def test_live_tool_route_binding_validates_route_scalars(
  overrides: dict[str, object],
  error_type: type[Exception],
) -> None:
  values: dict[str, object] = {
    "route_kind": "logical",
    "logical_name": "sample_tool",
    "transport_server_id": "transport-mcp",
    "provider_original_name": "provider_sample",
    "provider_id": None,
  }
  values.update(overrides)
  with pytest.raises(error_type):
    LiveToolRouteBinding(
      originated_definition=OriginatedToolDefinition(
        definition=_definition(),
        origin="mcp",
        server_id="logical-mcp",
      ),
      route_kind=values["route_kind"],  # type: ignore[arg-type]
      logical_name=values["logical_name"],  # type: ignore[arg-type]
      transport_server_id=values["transport_server_id"],  # type: ignore[arg-type]
      provider_original_name=values["provider_original_name"],  # type: ignore[arg-type]
      provider_id=values["provider_id"],  # type: ignore[arg-type]
    )


def test_live_tool_route_binding_refuses_local_definition_and_forged_state() -> None:
  with pytest.raises(ValueError, match="MCP definition"):
    LiveToolRouteBinding(
      originated_definition=OriginatedToolDefinition(
        definition=_definition(),
        origin="local",
        server_id=None,
      ),
      route_kind="logical",
      logical_name="sample_tool",
      transport_server_id="transport-mcp",
      provider_original_name="provider_sample",
      provider_id=None,
    )

  binding = LiveToolRouteBinding(
    originated_definition=OriginatedToolDefinition(
      definition=_definition(),
      origin="mcp",
      server_id="logical-mcp",
    ),
    route_kind="logical",
    logical_name="sample_tool",
    transport_server_id="transport-mcp",
    provider_original_name="provider_sample",
    provider_id=None,
  )
  object.__setattr__(binding, "logical_name", "forged")
  with pytest.raises(ValueError, match="logical_name"):
    validate_live_tool_route_binding(binding)

  with pytest.raises(ValueError, match="exposed_name"):
    LiveToolRouteBinding(
      originated_definition=OriginatedToolDefinition(
        definition={"name": " padded "},
        origin="mcp",
        server_id="logical-mcp",
      ),
      route_kind="physical",
      logical_name="padded",
      transport_server_id="logical-mcp",
      provider_original_name="padded",
      provider_id=None,
    )


@pytest.mark.parametrize(
  "reserved_field",
  ["exposed_name", "logical_name", "provider_original_name"],
)
def test_live_tool_route_binding_rejects_reserved_tool_names(
  reserved_field: str,
) -> None:
  reserved = "tool-registration:sha256:" + "0" * 64
  exposed_name = reserved if reserved_field == "exposed_name" else "safe_tool"
  logical_name = reserved if reserved_field == "logical_name" else "safe_tool"
  provider_original_name = (
    reserved
    if reserved_field == "provider_original_name"
    else logical_name
  )
  with pytest.raises(ValueError, match="reserved"):
    LiveToolRouteBinding(
      originated_definition=OriginatedToolDefinition(
        definition={"name": exposed_name},
        origin="mcp",
        server_id="physical-mcp",
      ),
      route_kind="physical",
      logical_name=logical_name,
      transport_server_id="physical-mcp",
      provider_original_name=provider_original_name,
      provider_id=None,
    )


def test_contract_module_is_stdlib_only_and_not_reexported() -> None:
  source = (ROOT / "agent_gateway" / "tool_definition.py").read_text(
    encoding="utf-8"
  )
  allowed = {
    "__future__": {"annotations"},
    "collections.abc": {"Mapping"},
    "dataclasses": {"dataclass"},
    "types": {"MappingProxyType"},
    "typing": {"Any", "Literal"},
  }
  for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Import):
      assert all(alias.asname is None for alias in node.names)
      assert {alias.name for alias in node.names}.issubset({"math"})
      continue
    if not isinstance(node, ast.ImportFrom):
      continue
    assert node.level == 0
    assert node.module in allowed
    assert all(alias.asname is None for alias in node.names)
    assert {alias.name for alias in node.names}.issubset(allowed[node.module])
  package_init = (ROOT / "agent_gateway" / "__init__.py").read_text(
    encoding="utf-8"
  )
  assert "tool_definition" not in package_init
