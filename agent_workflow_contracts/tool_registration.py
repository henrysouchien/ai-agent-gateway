"""Dependency-neutral static tool-registration contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Literal


ToolRouteKind = Literal["local_handler", "addin_relay", "mcp"]
ToolEffect = Literal[
  "read",
  "pure_transform",
  "support",
  "preview",
  "artifact_write",
  "state_write",
  "external_write",
  "portfolio_config",
  "irreversible",
]
ToolAudience = Literal["ordinary", "internal_only"]
PolicyKind = Literal[
  "approval_predicate",
  "approval_cache_key",
  "redaction",
  "planning",
  "input_preparation",
  "outcome",
  "source_identity",
  "session_injection",
]
ApprovalMode = Literal["never", "always", "predicate"]
DeclarationModeException = Literal["preview_allows_state_write"]

_ROUTE_KINDS = frozenset({"local_handler", "addin_relay", "mcp"})
_TOOL_EFFECTS = frozenset({
  "read",
  "pure_transform",
  "support",
  "preview",
  "artifact_write",
  "state_write",
  "external_write",
  "portfolio_config",
  "irreversible",
})
_TOOL_AUDIENCES = frozenset({"ordinary", "internal_only"})
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
_APPROVAL_MODES = frozenset({"never", "always", "predicate"})
_DECLARATION_MODE_EXCEPTIONS = frozenset({"preview_allows_state_write"})
_RESERVED_TOOL_NAME_PREFIX = "tool-registration:"
_MCP_INPUT_PREPARATION_MODES = frozenset({
  "comma-separated",
  "consistent-present-keys",
  "scalar",
})
_SEC_NATIVE_SYMBOL_RESOLVER = "sec-native-symbol-cached-only"


class ToolRegistrationContractError(ValueError):
  """A static registration value is incoherent."""


class UnknownToolRegistrationError(LookupError):
  """No static declaration matches an exact selector."""


class AmbiguousToolRegistrationError(LookupError):
  """A source-facing selector matches more than one exact registration."""


def _exact_trimmed_text(value: object, *, field_name: str) -> str:
  if type(value) is not str:
    raise TypeError(f"{field_name} must be an exact str")
  if not value or value != value.strip():
    raise ToolRegistrationContractError(
      f"{field_name} must be non-empty trimmed text"
    )
  return value


def _tool_name(value: object, *, field_name: str) -> str:
  name = _exact_trimmed_text(value, field_name=field_name)
  if name.startswith(_RESERVED_TOOL_NAME_PREFIX):
    raise ToolRegistrationContractError(
      f"{field_name} uses the reserved tool-registration namespace"
    )
  return name


def _freeze_json(value: object, *, field_name: str) -> object:
  if isinstance(value, Mapping):
    frozen: dict[str, object] = {}
    for key, item in value.items():
      if type(key) is not str:
        raise TypeError(f"{field_name} mapping keys must be exact strings")
      frozen[key] = _freeze_json(item, field_name=field_name)
    return MappingProxyType(dict(sorted(frozen.items())))
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_json(item, field_name=field_name) for item in value)
  if type(value) is float:
    if not math.isfinite(value):
      raise ToolRegistrationContractError(f"{field_name} floats must be finite")
    return value
  if value is None or type(value) in {bool, int, str}:
    return value
  raise TypeError(f"{field_name} values must be JSON-like")


def _materialize_json(value: object) -> object:
  if isinstance(value, Mapping):
    return {key: _materialize_json(item) for key, item in value.items()}
  if type(value) is tuple:
    return [_materialize_json(item) for item in value]
  return value


def _canonical_json_bytes(value: object) -> bytes:
  return json.dumps(
    value,
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
  ).encode("utf-8")


def _validate_timeout(value: object, *, field_name: str) -> float:
  if type(value) is int:
    timeout = float(value)
  elif type(value) is float:
    timeout = value
  else:
    raise TypeError(f"{field_name} must be an exact int or float")
  if not math.isfinite(timeout) or timeout <= 0:
    raise ToolRegistrationContractError(f"{field_name} must be finite and positive")
  return timeout


@dataclass(frozen=True, slots=True)
class RegisteredToolIdentity:
  """One exact logical tool identity, independent of live exposure."""

  route_kind: ToolRouteKind
  logical_name: str
  logical_server_id: str | None = None

  def __post_init__(self) -> None:
    if type(self.route_kind) is not str:
      raise TypeError("route_kind must be an exact str")
    if self.route_kind not in _ROUTE_KINDS:
      raise ToolRegistrationContractError("route_kind is unsupported")
    _tool_name(self.logical_name, field_name="logical_name")
    if self.route_kind == "mcp":
      _exact_trimmed_text(
        self.logical_server_id,
        field_name="logical_server_id",
      )
    elif self.logical_server_id is not None:
      raise ToolRegistrationContractError(
        "local and add-in identities must not declare logical_server_id"
      )

  @property
  def registration_key(self) -> str:
    digest = hashlib.sha256(
      _canonical_json_bytes(self.materialize())
    ).hexdigest()
    return f"tool-registration:sha256:{digest}"

  def materialize(self) -> dict[str, str | None]:
    return {
      "logical_name": self.logical_name,
      "logical_server_id": self.logical_server_id,
      "route_kind": self.route_kind,
    }


@dataclass(frozen=True, slots=True)
class McpInputPreparationRoute:
  """One exact dependency-neutral MCP input-preparation route."""

  logical_server_id: str
  logical_name: str
  mode: str
  keys: tuple[str, ...]
  resolver: str = _SEC_NATIVE_SYMBOL_RESOLVER

  def __post_init__(self) -> None:
    _exact_trimmed_text(
      self.logical_server_id,
      field_name="logical_server_id",
    )
    _tool_name(self.logical_name, field_name="logical_name")
    _exact_trimmed_text(self.mode, field_name="mode")
    _exact_trimmed_text(self.resolver, field_name="resolver")
    if type(self.keys) is not tuple or not self.keys:
      raise TypeError("keys must be an exact non-empty tuple")
    if any(
      type(key) is not str or not key or key != key.strip()
      for key in self.keys
    ):
      raise ToolRegistrationContractError(
        "keys must contain non-empty trimmed strings"
      )
    if len(set(self.keys)) != len(self.keys):
      raise ToolRegistrationContractError("keys must not contain duplicates")
    if self.mode not in _MCP_INPUT_PREPARATION_MODES:
      raise ToolRegistrationContractError(
        f"unsupported MCP input-preparation mode: {self.mode}"
      )
    if self.resolver != _SEC_NATIVE_SYMBOL_RESOLVER:
      raise ToolRegistrationContractError(
        f"unsupported MCP input-preparation resolver: {self.resolver}"
      )
    expected_key_count = 2 if self.mode == "consistent-present-keys" else 1
    if len(self.keys) != expected_key_count:
      raise ToolRegistrationContractError(
        f"MCP input-preparation mode {self.mode} requires exactly "
        f"{expected_key_count} key(s)"
      )
    if (
      self.mode == "consistent-present-keys"
      and self.keys != ("symbol", "ticker")
    ):
      raise ToolRegistrationContractError(
        "consistent-present-keys requires exact keys ('symbol', 'ticker')"
      )


McpInputPreparationRouteIndex = Mapping[
  tuple[str, str],
  McpInputPreparationRoute,
]


def index_mcp_input_preparation_routes(
  routes: Iterable[McpInputPreparationRoute] = (),
) -> McpInputPreparationRouteIndex:
  """Validate and freeze exact injected routes; the generic default is empty."""

  if isinstance(routes, (str, bytes)):
    raise TypeError("routes must be an iterable of exact route objects")
  index: dict[tuple[str, str], McpInputPreparationRoute] = {}
  for route in routes:
    if type(route) is not McpInputPreparationRoute:
      raise TypeError("routes must contain exact McpInputPreparationRoute values")
    key = (route.logical_server_id, route.logical_name)
    if key in index:
      raise ToolRegistrationContractError(
        f"duplicate MCP input-preparation route: {key}"
      )
    index[key] = route
  return MappingProxyType(dict(sorted(index.items())))


@dataclass(frozen=True, slots=True)
class VersionedPolicyRef:
  """One immutable semantic-policy implementation reference."""

  kind: PolicyKind
  policy_id: str
  version: str
  parameters: Mapping[str, object] = field(default_factory=dict)

  def __post_init__(self) -> None:
    if type(self.kind) is not str:
      raise TypeError("policy kind must be an exact str")
    if self.kind not in _POLICY_KINDS:
      raise ToolRegistrationContractError("policy kind is unsupported")
    _exact_trimmed_text(self.policy_id, field_name="policy_id")
    _exact_trimmed_text(self.version, field_name="policy version")
    frozen = _freeze_json(self.parameters, field_name="policy parameters")
    if not isinstance(frozen, Mapping):
      raise TypeError("policy parameters must be a mapping")
    object.__setattr__(self, "parameters", frozen)

  def materialize(self) -> dict[str, object]:
    return {
      "kind": self.kind,
      "parameters": _materialize_json(self.parameters),
      "policy_id": self.policy_id,
      "version": self.version,
    }


@dataclass(frozen=True, slots=True)
class ToolApprovalPolicy:
  """Intrinsic approval rule plus its safe cache-key strategy."""

  mode: ApprovalMode
  predicate: VersionedPolicyRef | None = None
  cache_key: VersionedPolicyRef | None = None

  def __post_init__(self) -> None:
    if type(self.mode) is not str:
      raise TypeError("approval mode must be an exact str")
    if self.mode not in _APPROVAL_MODES:
      raise ToolRegistrationContractError("approval mode is unsupported")
    if self.mode == "predicate":
      if self.predicate is None:
        raise ToolRegistrationContractError(
          "predicate approval requires a predicate reference"
        )
    elif self.predicate is not None:
      raise ToolRegistrationContractError(
        "only predicate approval may declare a predicate reference"
      )
    if self.predicate is not None:
      if type(self.predicate) is not VersionedPolicyRef:
        raise TypeError("approval predicate must be an exact VersionedPolicyRef")
      if self.predicate.kind != "approval_predicate":
        raise ToolRegistrationContractError(
          "approval predicate reference has the wrong policy kind"
        )
    if self.mode == "never" and self.cache_key is not None:
      raise ToolRegistrationContractError("approval=never requires cache=never")
    if self.cache_key is not None:
      if type(self.cache_key) is not VersionedPolicyRef:
        raise TypeError("approval cache key must be an exact VersionedPolicyRef")
      if self.cache_key.kind != "approval_cache_key":
        raise ToolRegistrationContractError(
          "approval cache reference has the wrong policy kind"
        )

  def materialize(self) -> dict[str, object]:
    return {
      "cache_key": self.cache_key.materialize() if self.cache_key else None,
      "mode": self.mode,
      "predicate": self.predicate.materialize() if self.predicate else None,
    }


@dataclass(frozen=True, slots=True)
class ToolIntrinsicSemantics:
  """Complete static semantics for one strict product registration."""

  effect: ToolEffect
  idempotent: bool
  semantic_capability: str
  approval: ToolApprovalPolicy
  audience: ToolAudience
  redaction_policy: VersionedPolicyRef
  planning_policy: VersionedPolicyRef
  input_preparation_policy: VersionedPolicyRef
  outcome_policy: VersionedPolicyRef
  source_identity_policy: VersionedPolicyRef
  declaration_mode_exception: DeclarationModeException | None = None

  def __post_init__(self) -> None:
    if type(self.effect) is not str:
      raise TypeError("effect must be an exact str")
    if self.effect not in _TOOL_EFFECTS:
      raise ToolRegistrationContractError("effect is unsupported")
    if type(self.idempotent) is not bool:
      raise TypeError("idempotent must be an exact bool")
    _exact_trimmed_text(
      self.semantic_capability,
      field_name="semantic_capability",
    )
    if type(self.approval) is not ToolApprovalPolicy:
      raise TypeError("approval must be an exact ToolApprovalPolicy")
    if type(self.audience) is not str:
      raise TypeError("audience must be an exact str")
    if self.audience not in _TOOL_AUDIENCES:
      raise ToolRegistrationContractError("audience is unsupported")
    expected_kinds = (
      (self.redaction_policy, "redaction", "redaction_policy"),
      (self.planning_policy, "planning", "planning_policy"),
      (
        self.input_preparation_policy,
        "input_preparation",
        "input_preparation_policy",
      ),
      (self.outcome_policy, "outcome", "outcome_policy"),
      (
        self.source_identity_policy,
        "source_identity",
        "source_identity_policy",
      ),
    )
    for policy, expected_kind, field_name in expected_kinds:
      if type(policy) is not VersionedPolicyRef:
        raise TypeError(f"{field_name} must be an exact VersionedPolicyRef")
      if policy.kind != expected_kind:
        raise ToolRegistrationContractError(
          f"{field_name} has the wrong policy kind"
        )
    if self.declaration_mode_exception is not None:
      if type(self.declaration_mode_exception) is not str:
        raise TypeError("declaration_mode_exception must be an exact str")
      if self.declaration_mode_exception not in _DECLARATION_MODE_EXCEPTIONS:
        raise ToolRegistrationContractError(
          "declaration_mode_exception is unsupported"
        )
      if self.effect != "state_write":
        raise ToolRegistrationContractError(
          "preview state-write exception requires state_write effect"
        )

  def materialize(self) -> dict[str, object]:
    return {
      "approval": self.approval.materialize(),
      "audience": self.audience,
      "declaration_mode_exception": self.declaration_mode_exception,
      "effect": self.effect,
      "idempotent": self.idempotent,
      "input_preparation_policy": self.input_preparation_policy.materialize(),
      "outcome_policy": self.outcome_policy.materialize(),
      "planning_policy": self.planning_policy.materialize(),
      "redaction_policy": self.redaction_policy.materialize(),
      "semantic_capability": self.semantic_capability,
      "source_identity_policy": self.source_identity_policy.materialize(),
    }


@dataclass(frozen=True, slots=True)
class ToolRegistrationDeclaration:
  """One exact logical identity and its registration-owned semantics."""

  identity: RegisteredToolIdentity
  semantics: ToolIntrinsicSemantics

  def __post_init__(self) -> None:
    if type(self.identity) is not RegisteredToolIdentity:
      raise TypeError("identity must be an exact RegisteredToolIdentity")
    if type(self.semantics) is not ToolIntrinsicSemantics:
      raise TypeError("semantics must be exact ToolIntrinsicSemantics")

  def materialize(self) -> dict[str, object]:
    return {
      "identity": self.identity.materialize(),
      "semantics": self.semantics.materialize(),
    }


@dataclass(frozen=True, slots=True)
class RegisteredToolServerDescriptor:
  """Stable server-level facts shared by its registered MCP tools."""

  logical_server_id: str
  transport_server_id: str
  default_timeout_seconds: float
  per_tool_timeout_seconds: Mapping[str, float]
  session_injection_policy: VersionedPolicyRef

  def __post_init__(self) -> None:
    _exact_trimmed_text(self.logical_server_id, field_name="logical_server_id")
    _exact_trimmed_text(
      self.transport_server_id,
      field_name="transport_server_id",
    )
    object.__setattr__(
      self,
      "default_timeout_seconds",
      _validate_timeout(
        self.default_timeout_seconds,
        field_name="default_timeout_seconds",
      ),
    )
    if not isinstance(self.per_tool_timeout_seconds, Mapping):
      raise TypeError("per_tool_timeout_seconds must be a mapping")
    timeouts: dict[str, float] = {}
    for name, timeout in self.per_tool_timeout_seconds.items():
      tool_name = _tool_name(name, field_name="per-tool timeout name")
      timeouts[tool_name] = _validate_timeout(
        timeout,
        field_name=f"timeout for {tool_name}",
      )
    object.__setattr__(
      self,
      "per_tool_timeout_seconds",
      MappingProxyType(dict(sorted(timeouts.items()))),
    )
    if type(self.session_injection_policy) is not VersionedPolicyRef:
      raise TypeError(
        "session_injection_policy must be an exact VersionedPolicyRef"
      )
    if self.session_injection_policy.kind != "session_injection":
      raise ToolRegistrationContractError(
        "session_injection_policy has the wrong policy kind"
      )

  def materialize(self) -> dict[str, object]:
    return {
      "default_timeout_seconds": self.default_timeout_seconds,
      "logical_server_id": self.logical_server_id,
      "per_tool_timeout_seconds": dict(self.per_tool_timeout_seconds),
      "session_injection_policy": self.session_injection_policy.materialize(),
      "transport_server_id": self.transport_server_id,
    }


def validate_registered_tool_identity(
  value: object,
) -> RegisteredToolIdentity:
  if type(value) is not RegisteredToolIdentity:
    raise TypeError("value must be an exact RegisteredToolIdentity")
  return RegisteredToolIdentity(
    route_kind=value.route_kind,
    logical_name=value.logical_name,
    logical_server_id=value.logical_server_id,
  )


def validate_versioned_policy_ref(value: object) -> VersionedPolicyRef:
  if type(value) is not VersionedPolicyRef:
    raise TypeError("value must be an exact VersionedPolicyRef")
  return VersionedPolicyRef(
    kind=value.kind,
    policy_id=value.policy_id,
    version=value.version,
    parameters=value.parameters,
  )


def validate_tool_approval_policy(value: object) -> ToolApprovalPolicy:
  if type(value) is not ToolApprovalPolicy:
    raise TypeError("value must be an exact ToolApprovalPolicy")
  return ToolApprovalPolicy(
    mode=value.mode,
    predicate=(
      validate_versioned_policy_ref(value.predicate)
      if value.predicate is not None
      else None
    ),
    cache_key=(
      validate_versioned_policy_ref(value.cache_key)
      if value.cache_key is not None
      else None
    ),
  )


def validate_tool_intrinsic_semantics(
  value: object,
) -> ToolIntrinsicSemantics:
  if type(value) is not ToolIntrinsicSemantics:
    raise TypeError("value must be exact ToolIntrinsicSemantics")
  return ToolIntrinsicSemantics(
    effect=value.effect,
    idempotent=value.idempotent,
    semantic_capability=value.semantic_capability,
    approval=validate_tool_approval_policy(value.approval),
    audience=value.audience,
    redaction_policy=validate_versioned_policy_ref(value.redaction_policy),
    planning_policy=validate_versioned_policy_ref(value.planning_policy),
    input_preparation_policy=validate_versioned_policy_ref(
      value.input_preparation_policy
    ),
    outcome_policy=validate_versioned_policy_ref(value.outcome_policy),
    source_identity_policy=validate_versioned_policy_ref(
      value.source_identity_policy
    ),
    declaration_mode_exception=value.declaration_mode_exception,
  )


def validate_tool_registration_declaration(
  value: object,
) -> ToolRegistrationDeclaration:
  if type(value) is not ToolRegistrationDeclaration:
    raise TypeError("value must be an exact ToolRegistrationDeclaration")
  return ToolRegistrationDeclaration(
    identity=validate_registered_tool_identity(value.identity),
    semantics=validate_tool_intrinsic_semantics(value.semantics),
  )


def validate_registered_tool_server_descriptor(
  value: object,
) -> RegisteredToolServerDescriptor:
  if type(value) is not RegisteredToolServerDescriptor:
    raise TypeError("value must be an exact RegisteredToolServerDescriptor")
  return RegisteredToolServerDescriptor(
    logical_server_id=value.logical_server_id,
    transport_server_id=value.transport_server_id,
    default_timeout_seconds=value.default_timeout_seconds,
    per_tool_timeout_seconds=value.per_tool_timeout_seconds,
    session_injection_policy=validate_versioned_policy_ref(
      value.session_injection_policy
    ),
  )


def _identity_sort_key(
  identity: RegisteredToolIdentity,
) -> tuple[str, str, str]:
  return (
    identity.route_kind,
    identity.logical_server_id or "",
    identity.logical_name,
  )


@dataclass(frozen=True, slots=True)
class ToolRegistrationCatalog:
  """Immutable static declarations indexed by exact logical identity."""

  declarations: tuple[ToolRegistrationDeclaration, ...]
  servers: tuple[RegisteredToolServerDescriptor, ...]
  _by_identity: Mapping[RegisteredToolIdentity, ToolRegistrationDeclaration] = (
    field(init=False, repr=False, compare=False)
  )
  _by_registration_key: Mapping[str, ToolRegistrationDeclaration] = field(
    init=False,
    repr=False,
    compare=False,
  )
  _servers_by_logical_id: Mapping[str, RegisteredToolServerDescriptor] = field(
    init=False,
    repr=False,
    compare=False,
  )

  def __post_init__(self) -> None:
    if type(self.declarations) is not tuple:
      raise TypeError("declarations must be an exact tuple")
    if type(self.servers) is not tuple:
      raise TypeError("servers must be an exact tuple")

    servers_by_id: dict[str, RegisteredToolServerDescriptor] = {}
    canonical_servers: list[RegisteredToolServerDescriptor] = []
    for raw_server in self.servers:
      server = validate_registered_tool_server_descriptor(raw_server)
      if server.logical_server_id in servers_by_id:
        raise ToolRegistrationContractError(
          f"duplicate logical server registration: {server.logical_server_id}"
        )
      servers_by_id[server.logical_server_id] = server
      canonical_servers.append(server)

    by_identity: dict[RegisteredToolIdentity, ToolRegistrationDeclaration] = {}
    by_key: dict[str, ToolRegistrationDeclaration] = {}
    canonical_declarations: list[ToolRegistrationDeclaration] = []
    for raw_declaration in self.declarations:
      declaration = validate_tool_registration_declaration(raw_declaration)
      identity = declaration.identity
      if identity in by_identity:
        raise ToolRegistrationContractError(
          f"duplicate tool registration: {identity.materialize()}"
        )
      if (
        identity.route_kind == "mcp"
        and identity.logical_server_id not in servers_by_id
      ):
        raise ToolRegistrationContractError(
          "MCP declaration references an unregistered logical server"
        )
      key = identity.registration_key
      if key in by_key:
        raise ToolRegistrationContractError(
          "distinct registrations produced the same registration key"
        )
      by_identity[identity] = declaration
      by_key[key] = declaration
      canonical_declarations.append(declaration)

    for server in canonical_servers:
      registered_names = {
        declaration.identity.logical_name
        for declaration in canonical_declarations
        if declaration.identity.route_kind == "mcp"
        and declaration.identity.logical_server_id == server.logical_server_id
      }
      unknown_timeout_names = (
        set(server.per_tool_timeout_seconds) - registered_names
      )
      if unknown_timeout_names:
        unknown = ", ".join(sorted(unknown_timeout_names))
        raise ToolRegistrationContractError(
          "per-tool timeout references an unregistered tool on "
          f"{server.logical_server_id}: {unknown}"
        )

    canonical_declarations.sort(key=lambda item: _identity_sort_key(item.identity))
    canonical_servers.sort(key=lambda item: item.logical_server_id)
    object.__setattr__(self, "declarations", tuple(canonical_declarations))
    object.__setattr__(self, "servers", tuple(canonical_servers))
    object.__setattr__(self, "_by_identity", MappingProxyType(by_identity))
    object.__setattr__(self, "_by_registration_key", MappingProxyType(by_key))
    object.__setattr__(
      self,
      "_servers_by_logical_id",
      MappingProxyType(servers_by_id),
    )

  def by_identity(
    self,
    identity: RegisteredToolIdentity,
  ) -> ToolRegistrationDeclaration:
    canonical = validate_registered_tool_identity(identity)
    try:
      return self._by_identity[canonical]
    except KeyError as exc:
      raise UnknownToolRegistrationError(
        f"unknown tool registration: {canonical.materialize()}"
      ) from exc

  def by_registration_key(self, key: str) -> ToolRegistrationDeclaration:
    _exact_trimmed_text(key, field_name="registration key")
    try:
      return self._by_registration_key[key]
    except KeyError as exc:
      raise UnknownToolRegistrationError(
        f"unknown tool registration key: {key}"
      ) from exc

  def resolve_local(self, logical_name: str) -> ToolRegistrationDeclaration:
    name = _tool_name(logical_name, field_name="logical_name")
    matches = tuple(
      declaration
      for declaration in self.declarations
      if declaration.identity.route_kind in {"local_handler", "addin_relay"}
      and declaration.identity.logical_name == name
    )
    if not matches:
      raise UnknownToolRegistrationError(
        f"unknown local/add-in tool registration: {name}"
      )
    if len(matches) != 1:
      raise AmbiguousToolRegistrationError(
        f"ambiguous local/add-in tool registration: {name}"
      )
    return matches[0]

  def resolve_mcp(
    self,
    logical_server_id: str,
    logical_name: str,
  ) -> ToolRegistrationDeclaration:
    return self.by_identity(
      RegisteredToolIdentity(
        route_kind="mcp",
        logical_name=logical_name,
        logical_server_id=logical_server_id,
      )
    )

  def server(self, logical_server_id: str) -> RegisteredToolServerDescriptor:
    server_id = _exact_trimmed_text(
      logical_server_id,
      field_name="logical_server_id",
    )
    try:
      return self._servers_by_logical_id[server_id]
    except KeyError as exc:
      raise UnknownToolRegistrationError(
        f"unknown registered tool server: {server_id}"
      ) from exc


def validate_tool_registration_catalog(
  value: object,
) -> ToolRegistrationCatalog:
  if type(value) is not ToolRegistrationCatalog:
    raise TypeError("value must be an exact ToolRegistrationCatalog")
  return ToolRegistrationCatalog(
    declarations=value.declarations,
    servers=value.servers,
  )


__all__ = [
  "AmbiguousToolRegistrationError",
  "ApprovalMode",
  "DeclarationModeException",
  "McpInputPreparationRoute",
  "McpInputPreparationRouteIndex",
  "PolicyKind",
  "RegisteredToolIdentity",
  "RegisteredToolServerDescriptor",
  "ToolApprovalPolicy",
  "ToolAudience",
  "ToolEffect",
  "ToolIntrinsicSemantics",
  "ToolRegistrationCatalog",
  "ToolRegistrationContractError",
  "ToolRegistrationDeclaration",
  "ToolRouteKind",
  "UnknownToolRegistrationError",
  "VersionedPolicyRef",
  "validate_registered_tool_identity",
  "index_mcp_input_preparation_routes",
  "validate_registered_tool_server_descriptor",
  "validate_tool_approval_policy",
  "validate_tool_intrinsic_semantics",
  "validate_tool_registration_catalog",
  "validate_tool_registration_declaration",
  "validate_versioned_policy_ref",
]
