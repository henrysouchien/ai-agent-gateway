"""Dependency-neutral originated tool-definition contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal


ToolDefinitionOrigin = Literal["local", "mcp"]
_ORIGINS = frozenset({"local", "mcp"})
LiveToolRouteKind = Literal["physical", "logical"]
_LIVE_ROUTE_KINDS = frozenset({"physical", "logical"})
_RESERVED_TOOL_NAME_PREFIX = "tool-registration:"


def _validate_origin_server(
  origin: object,
  server_id: object,
) -> None:
  if type(origin) is not str:
    raise TypeError("origin must be an exact str")
  if origin not in _ORIGINS:
    raise ValueError("origin must be 'local' or 'mcp'")
  if origin == "local":
    if server_id is not None:
      raise ValueError("local tool definitions must not declare server_id")
  elif type(server_id) is not str:
    raise TypeError("MCP tool definitions require an exact str server_id")
  elif not server_id or server_id != server_id.strip():
    raise ValueError("MCP server_id must be non-empty trimmed text")


def _freeze_json(value: object) -> object:
  if isinstance(value, Mapping):
    frozen: dict[str, object] = {}
    for key, item in value.items():
      if type(key) is not str:
        raise TypeError("tool definition mapping keys must be exact strings")
      frozen[key] = _freeze_json(item)
    return MappingProxyType(frozen)
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_json(item) for item in value)
  if type(value) is float:
    if not math.isfinite(value):
      raise ValueError("tool definition floats must be finite")
    return value
  if value is None or type(value) in {bool, int, str}:
    return value
  raise TypeError("tool definition values must be JSON-like")


def _materialize_json(value: object) -> object:
  if isinstance(value, Mapping):
    return {key: _materialize_json(item) for key, item in value.items()}
  if type(value) is tuple:
    return [_materialize_json(item) for item in value]
  return value


def _validate_frozen_json(value: object) -> None:
  if type(value) is MappingProxyType:
    for key, item in value.items():
      if type(key) is not str:
        raise TypeError("tool definition mapping keys must be exact strings")
      _validate_frozen_json(item)
    return
  if type(value) is tuple:
    for item in value:
      _validate_frozen_json(item)
    return
  if type(value) is float:
    if not math.isfinite(value):
      raise ValueError("tool definition floats must be finite")
    return
  if value is None or type(value) in {bool, int, str}:
    return
  raise TypeError("tool definition must retain its deeply frozen JSON shape")


@dataclass(frozen=True, slots=True)
class OriginatedToolDefinition:
  """One deeply immutable provider tool definition with exact origin."""

  definition: Mapping[str, Any]
  origin: ToolDefinitionOrigin
  server_id: str | None

  def __post_init__(self) -> None:
    _validate_origin_server(self.origin, self.server_id)

    frozen = _freeze_json(self.definition)
    if not isinstance(frozen, Mapping):
      raise TypeError("definition must be a mapping")
    name = frozen.get("name")
    if type(name) is not str:
      raise TypeError("definition name must be an exact str")
    if not name:
      raise ValueError("definition name must be non-empty")
    object.__setattr__(self, "definition", frozen)

  @property
  def name(self) -> str:
    """Return the validated exposed provider name."""

    name = self.definition["name"]
    if type(name) is not str:
      raise TypeError("definition name must be an exact str")
    if not name:
      raise ValueError("definition name must be non-empty")
    return name

  def materialize(self) -> dict[str, Any]:
    """Return a fresh deeply mutable provider-shaped dictionary."""

    materialized = _materialize_json(self.definition)
    assert type(materialized) is dict
    return materialized


def validate_originated_tool_definition(
  value: object,
) -> OriginatedToolDefinition:
  """Validate an exact record, including state forged after construction."""

  if type(value) is not OriginatedToolDefinition:
    raise TypeError("tool definition records must use the exact contract")
  _validate_origin_server(value.origin, value.server_id)
  if type(value.definition) is not MappingProxyType:
    raise TypeError("tool definition must retain its deeply frozen mapping")
  _validate_frozen_json(value.definition)
  return OriginatedToolDefinition(
    definition=value.materialize(),
    origin=value.origin,
    server_id=value.server_id,
  )


def _exact_route_text(value: object, *, field_name: str) -> str:
  if type(value) is not str:
    raise TypeError(f"{field_name} must be an exact str")
  if not value or value != value.strip():
    raise ValueError(f"{field_name} must be non-empty trimmed text")
  return value


def _exact_route_tool_name(value: object, *, field_name: str) -> str:
  name = _exact_route_text(value, field_name=field_name)
  if name.startswith(_RESERVED_TOOL_NAME_PREFIX):
    raise ValueError(
      f"{field_name} uses the reserved tool-registration namespace"
    )
  return name


@dataclass(frozen=True, slots=True)
class LiveToolRouteBinding:
  """One immutable MCP definition bound to its exact live route identities."""

  originated_definition: OriginatedToolDefinition
  route_kind: LiveToolRouteKind
  logical_name: str
  transport_server_id: str
  provider_original_name: str
  provider_id: str | None

  def __post_init__(self) -> None:
    definition = validate_originated_tool_definition(
      self.originated_definition
    )
    if definition.origin != "mcp" or definition.server_id is None:
      raise ValueError("live tool route bindings require an MCP definition")
    if type(self.route_kind) is not str:
      raise TypeError("route_kind must be an exact str")
    if self.route_kind not in _LIVE_ROUTE_KINDS:
      raise ValueError("route_kind must be 'physical' or 'logical'")
    exposed_name = _exact_route_tool_name(
      definition.name,
      field_name="exposed_name",
    )
    logical_name = _exact_route_tool_name(
      self.logical_name,
      field_name="logical_name",
    )
    transport_server_id = _exact_route_text(
      self.transport_server_id,
      field_name="transport_server_id",
    )
    provider_original_name = _exact_route_tool_name(
      self.provider_original_name,
      field_name="provider_original_name",
    )
    if self.provider_id is not None:
      _exact_route_text(self.provider_id, field_name="provider_id")

    if self.route_kind == "physical":
      if definition.server_id != transport_server_id:
        raise ValueError(
          "physical route owner must equal transport_server_id"
        )
      if logical_name != provider_original_name:
        raise ValueError(
          "physical logical_name must equal provider_original_name"
        )
    elif logical_name != exposed_name:
      raise ValueError("logical route logical_name must equal exposed_name")

    object.__setattr__(self, "originated_definition", definition)

  @property
  def exposed_name(self) -> str:
    """Return the provider-facing name in the frozen definition."""

    return self.originated_definition.name

  @property
  def logical_server_id(self) -> str:
    """Return the public server identity that owns the exposed route."""

    server_id = self.originated_definition.server_id
    assert server_id is not None
    return server_id

  def materialize_provider_definition(self) -> dict[str, Any]:
    """Return a fresh provider-shaped tool definition."""

    return self.originated_definition.materialize()


def validate_live_tool_route_binding(
  value: object,
) -> LiveToolRouteBinding:
  """Validate and detach an exact live MCP route binding."""

  if type(value) is not LiveToolRouteBinding:
    raise TypeError("live tool route bindings must use the exact contract")
  return LiveToolRouteBinding(
    originated_definition=value.originated_definition,
    route_kind=value.route_kind,
    logical_name=value.logical_name,
    transport_server_id=value.transport_server_id,
    provider_original_name=value.provider_original_name,
    provider_id=value.provider_id,
  )


__all__ = [
  "LiveToolRouteBinding",
  "LiveToolRouteKind",
  "OriginatedToolDefinition",
  "ToolDefinitionOrigin",
  "validate_live_tool_route_binding",
  "validate_originated_tool_definition",
]
