"""Authoritative product model registry and model-selection policy.

The registry owns stable execution identity.  The policy owns defaults.  They
are deliberately ordinary, immutable deployment configuration rather than a
runtime service, and neither contains credential material.

Registry and policy data are authored as typed deployment artifacts
(``product-model-registry/v1`` and ``product-model-selection/v1``) packaged
under ``agent_gateway/model_authority/``.  Both artifacts are loaded and fully
admitted exactly once at process construction (import of this module) and any
missing, unparsable, unknown-field, or incoherent artifact fails startup.

Deployment selection: a deployment may substitute an alternative admitted
artifact file by setting ``AGENT_GATEWAY_MODEL_REGISTRY_FILE`` and/or
``AGENT_GATEWAY_MODEL_SELECTION_FILE`` to a file path.  This is deployment
configuration naming WHICH artifact to load — the same admission gate runs on
whatever is selected, fail-closed — it is never model or default authority
itself.  The default is the packaged artifact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

import yaml

from .thinking import ThinkingLevel


CapabilityExposure: TypeAlias = Literal["user_selectable", "internal"]
ModelLifecycle: TypeAlias = Literal[
  "active",
  "hidden",
  "deprecated",
  "disabled",
  "revoked",
]
SelectionSource: TypeAlias = Literal[
  "explicit_user",
  "saved_preference",
  "channel_default",
  "capability_default",
  "internal_policy",
  "parent_binding",
]

CORE_CAPABILITY_IDS = frozenset({
  "session.driver",
  "plan.author",
  "node.explore",
  "node.implement",
  "node.mutate",
  "node.fork",
  "node.verify",
  "node.choose",
  "citation.review",
})
RISK_CAPABILITY_IDS = frozenset({
  "risk.completion",
  "risk.interpretation",
  "risk.peer_generation",
  "risk.asset_classification",
  "risk.overview_editorial",
  "risk.document_ingest",
})
INVESTMENT_CAPABILITY_IDS = frozenset({
  "investment.research_agent",
  "investment.quant_worker",
  "investment.newsletter",
  "investment.earnings_transcript",
  "investment.biotech_review",
})
CAPABILITY_IDS = CORE_CAPABILITY_IDS | RISK_CAPABILITY_IDS | INVESTMENT_CAPABILITY_IDS

# --- Capability execution designation -----------------------------------------
#
# Typed constant consumed together with the registry artifact: it names which
# serving process executes each capability.  The gateway server process executes
# the core conversational/orchestration capabilities.  The ``risk.*`` workload
# capabilities are executed inside the Risk serving process (its own protocol
# implementations resolve them via ``providers/model_authority.py`` in that
# repository), and the ``investment.*`` workload capabilities are executed by
# the Investment worker processes.  Registry entries that serve only
# externally-executed capabilities are admitted registry facts — they carry the
# authoritative execution identity for those processes — but they are excluded
# from THIS process's executable set by explicit designation, never by silently
# skipping an entry whose adapter happens to be missing.

CapabilityExecutionProcess: TypeAlias = Literal["gateway", "risk", "investment"]

CAPABILITY_EXECUTION_PROCESS: Mapping[str, CapabilityExecutionProcess] = (
  MappingProxyType(
    {
      **{capability_id: "gateway" for capability_id in CORE_CAPABILITY_IDS},
      **{capability_id: "risk" for capability_id in RISK_CAPABILITY_IDS},
      **{
        capability_id: "investment"
        for capability_id in INVESTMENT_CAPABILITY_IDS
      },
    }
  )
)
if set(CAPABILITY_EXECUTION_PROCESS) != CAPABILITY_IDS:  # pragma: no cover
  raise ValueError(
    "CAPABILITY_EXECUTION_PROCESS must designate every known capability"
  )
if (
  len(CORE_CAPABILITY_IDS) + len(RISK_CAPABILITY_IDS) + len(INVESTMENT_CAPABILITY_IDS)
  != len(CAPABILITY_IDS)
):  # pragma: no cover
  # Overlapping process sets would let a later dict-merge entry silently
  # reassign a capability's serving process.
  raise ValueError("capability process sets must be disjoint")

GATEWAY_EXECUTED_CAPABILITY_IDS = frozenset(
  capability_id
  for capability_id, process in CAPABILITY_EXECUTION_PROCESS.items()
  if process == "gateway"
)
EXTERNALLY_EXECUTED_CAPABILITY_IDS = (
  CAPABILITY_IDS - GATEWAY_EXECUTED_CAPABILITY_IDS
)

_EXPOSURES = frozenset({"user_selectable", "internal"})
_LIFECYCLES = frozenset({"active", "hidden", "deprecated", "disabled", "revoked"})
_FEATURES = frozenset({"tools", "vision", "streaming", "structured_output"})


def _text(value: object, *, field: str) -> str:
  normalized = str(value or "").strip()
  if not normalized:
    raise ValueError(f"{field} must be non-empty")
  return normalized


def _effort(value: object, *, field: str) -> str:
  normalized = _text(value, field=field).lower()
  try:
    return ThinkingLevel(normalized).value
  except ValueError as exc:
    raise ValueError(f"invalid {field}: {normalized!r}") from exc


@dataclass(frozen=True, slots=True)
class ModelRegistryEntry:
  key: str
  label: str
  provider: str
  upstream_model: str
  adapter: str
  protocol_profile: str
  route: str
  lifecycle: ModelLifecycle
  capabilities: Mapping[str, CapabilityExposure]
  supported_efforts: frozenset[str]
  default_effort: str
  features: frozenset[str]
  reported_identities: frozenset[str]

  def __post_init__(self) -> None:
    key = _text(self.key, field="model key")
    label = _text(self.label, field=f"{key}.label")
    provider = _text(self.provider, field=f"{key}.provider").lower()
    upstream_model = _text(self.upstream_model, field=f"{key}.upstream_model")
    adapter = _text(self.adapter, field=f"{key}.adapter")
    protocol_profile = _text(
      self.protocol_profile,
      field=f"{key}.protocol_profile",
    )
    route = _text(self.route, field=f"{key}.route")
    lifecycle = _text(self.lifecycle, field=f"{key}.lifecycle").lower()
    if lifecycle not in _LIFECYCLES:
      raise ValueError(f"unknown {key}.lifecycle: {lifecycle}")

    capabilities = dict(self.capabilities)
    if not capabilities:
      raise ValueError(f"{key}.capabilities must be non-empty")
    for capability_id, exposure in capabilities.items():
      if capability_id not in CAPABILITY_IDS:
        raise ValueError(f"{key} has unknown capability: {capability_id}")
      if exposure not in _EXPOSURES:
        raise ValueError(f"{key} has unknown exposure: {exposure}")
    if lifecycle in {"hidden", "deprecated", "disabled", "revoked"} and (
      "user_selectable" in capabilities.values()
    ):
      raise ValueError(f"{lifecycle} model {key} cannot be user-selectable")

    supported_efforts = frozenset(
      _effort(value, field=f"{key}.supported_efforts")
      for value in self.supported_efforts
    )
    if not supported_efforts:
      raise ValueError(f"{key}.supported_efforts must be non-empty")
    default_effort = _effort(self.default_effort, field=f"{key}.default_effort")
    if default_effort not in supported_efforts:
      raise ValueError(f"{key}.default_effort must be supported")

    features = frozenset(self.features)
    unknown_features = features - _FEATURES
    if unknown_features:
      raise ValueError(
        f"{key} has unknown features: {', '.join(sorted(unknown_features))}"
      )
    reported_identities = frozenset(
      _text(value, field=f"{key}.reported_identities")
      for value in self.reported_identities
    )
    if upstream_model not in reported_identities:
      raise ValueError(f"{key}.reported_identities must include upstream_model")

    object.__setattr__(self, "key", key)
    object.__setattr__(self, "label", label)
    object.__setattr__(self, "provider", provider)
    object.__setattr__(self, "upstream_model", upstream_model)
    object.__setattr__(self, "adapter", adapter)
    object.__setattr__(self, "protocol_profile", protocol_profile)
    object.__setattr__(self, "route", route)
    object.__setattr__(self, "lifecycle", lifecycle)
    object.__setattr__(self, "capabilities", MappingProxyType(capabilities))
    object.__setattr__(self, "supported_efforts", supported_efforts)
    object.__setattr__(self, "default_effort", default_effort)
    object.__setattr__(self, "features", features)
    object.__setattr__(self, "reported_identities", reported_identities)


@dataclass(frozen=True, slots=True)
class AdapterRouteSupport:
  """Protocol facts declared by one installed adapter implementation.

  Declarations come from the adapter classes themselves
  (``ModelProvider.adapter_route_support``) — they describe what the installed
  code actually implements and are never a second hand-maintained model
  catalog.
  """

  adapter: str
  provider: str
  protocol_profiles: frozenset[str]
  routes: frozenset[str]

  def __post_init__(self) -> None:
    adapter = _text(self.adapter, field="adapter support adapter")
    provider = _text(self.provider, field=f"{adapter}.provider").lower()
    protocol_profiles = frozenset(
      _text(value, field=f"{adapter}.protocol_profiles")
      for value in self.protocol_profiles
    )
    routes = frozenset(
      _text(value, field=f"{adapter}.routes") for value in self.routes
    )
    if not protocol_profiles:
      raise ValueError(f"{adapter} must declare at least one protocol profile")
    if not routes:
      raise ValueError(f"{adapter} must declare at least one route")
    object.__setattr__(self, "adapter", adapter)
    object.__setattr__(self, "provider", provider)
    object.__setattr__(self, "protocol_profiles", protocol_profiles)
    object.__setattr__(self, "routes", routes)

  def supports(self, entry: ModelRegistryEntry) -> bool:
    return (
      entry.adapter == self.adapter
      and entry.provider == self.provider
      and entry.protocol_profile in self.protocol_profiles
      and entry.route in self.routes
    )


@dataclass(frozen=True, slots=True)
class ProductModelRegistry:
  schema: Literal["product-model-registry/v1"]
  revision: str
  models: Mapping[str, ModelRegistryEntry]

  def __post_init__(self) -> None:
    if self.schema != "product-model-registry/v1":
      raise ValueError(f"unsupported model registry schema: {self.schema!r}")
    revision = _text(self.revision, field="registry revision")
    models = dict(self.models)
    if not models:
      raise ValueError("model registry must be non-empty")
    execution_identities: dict[tuple[str, str, str, str, str], str] = {}
    for key, entry in models.items():
      if key != entry.key:
        raise ValueError("model registry key must match entry.key")
      identity = (
        entry.provider,
        entry.upstream_model,
        entry.adapter,
        entry.protocol_profile,
        entry.route,
      )
      prior = execution_identities.setdefault(identity, key)
      if prior != key:
        raise ValueError(
          f"model keys {prior!r} and {key!r} duplicate one execution identity"
        )
    object.__setattr__(self, "revision", revision)
    object.__setattr__(self, "models", MappingProxyType(models))

  def require(self, key: str) -> ModelRegistryEntry:
    normalized = _text(key, field="model key")
    try:
      return self.models[normalized]
    except KeyError as exc:
      raise KeyError(f"unknown model key: {normalized}") from exc

  def admit_adapter_support(
    self,
    supports: Mapping[str, AdapterRouteSupport],
    *,
    executed_capability_ids: frozenset[str] | None = None,
  ) -> None:
    """Require declared adapter support for every entry this process executes.

    ``supports`` is gathered from installed adapter declarations
    (``installed_adapter_route_support``), never hand-maintained.
    ``executed_capability_ids`` is the explicit designation of which
    capabilities the admitting process executes (for the gateway server,
    ``GATEWAY_EXECUTED_CAPABILITY_IDS``).  Entries all of whose capabilities
    are designated externally executed are admitted registry facts for the
    executing process, not local execution obligations.  ``None`` — the
    fail-closed default — requires installed support for every entry.
    """
    if executed_capability_ids is None:
      executed = CAPABILITY_IDS
    else:
      executed = frozenset(executed_capability_ids)
      unknown = executed - CAPABILITY_IDS
      if unknown:
        raise ValueError(
          "executed_capability_ids has unknown capabilities: "
          f"{', '.join(sorted(unknown))}"
        )
    for entry in self.models.values():
      if not (set(entry.capabilities) & executed):
        continue
      support = supports.get(entry.adapter)
      if support is None or not support.supports(entry):
        raise ValueError(
          f"registry entry {entry.key!r} has no installed adapter/profile/route support"
        )


@dataclass(frozen=True, slots=True)
class CapabilityDefault:
  kind: Literal["model", "inherit_parent"]
  model_key: str | None = None
  effort: str | None = None

  def __post_init__(self) -> None:
    if self.kind == "inherit_parent":
      if self.model_key is not None or self.effort is not None:
        raise ValueError("inherit_parent cannot declare model_key or effort")
      return
    if self.kind != "model":
      raise ValueError(f"unknown capability default kind: {self.kind!r}")
    object.__setattr__(
      self,
      "model_key",
      _text(self.model_key, field="capability default model_key"),
    )
    object.__setattr__(
      self,
      "effort",
      _effort(self.effort, field="capability default effort"),
    )


@dataclass(frozen=True, slots=True)
class CapabilitySelectionPolicy:
  capability_id: str
  default: CapabilityDefault
  by_channel: Mapping[str, CapabilityDefault]
  allowed_model_keys: frozenset[str]
  allow_saved_preference: bool = False
  allow_explicit_user: bool = False
  allow_authenticated_run_override: bool = False

  def __post_init__(self) -> None:
    capability_id = _text(self.capability_id, field="capability_id")
    if capability_id not in CAPABILITY_IDS:
      raise ValueError(f"unknown capability_id: {capability_id}")
    by_channel = {
      _text(channel, field=f"{capability_id}.channel"): channel_default
      for channel, channel_default in dict(self.by_channel).items()
    }
    allowed_model_keys = frozenset(
      _text(value, field=f"{capability_id}.allowed_model_keys")
      for value in self.allowed_model_keys
    )
    if not allowed_model_keys:
      raise ValueError(f"{capability_id}.allowed_model_keys must be non-empty")
    if self.default.kind == "model" and self.default.model_key not in allowed_model_keys:
      raise ValueError(f"{capability_id} default must be allowed")
    if self.allow_saved_preference and not self.allow_explicit_user:
      raise ValueError("saved preferences require explicit user selection")
    if self.allow_explicit_user and capability_id != "session.driver":
      raise ValueError("only session.driver accepts conversational user selection")
    if self.allow_authenticated_run_override and capability_id != "plan.author":
      raise ValueError(f"{capability_id} cannot accept a run-level override")
    object.__setattr__(self, "capability_id", capability_id)
    object.__setattr__(self, "by_channel", MappingProxyType(by_channel))
    object.__setattr__(self, "allowed_model_keys", allowed_model_keys)


@dataclass(frozen=True, slots=True)
class ProductModelSelectionPolicy:
  schema: Literal["product-model-selection/v1"]
  revision: str
  capabilities: Mapping[str, CapabilitySelectionPolicy]

  def __post_init__(self) -> None:
    if self.schema != "product-model-selection/v1":
      raise ValueError(f"unsupported model policy schema: {self.schema!r}")
    revision = _text(self.revision, field="policy revision")
    capabilities = dict(self.capabilities)
    if set(capabilities) != CAPABILITY_IDS:
      missing = CAPABILITY_IDS - set(capabilities)
      unexpected = set(capabilities) - CAPABILITY_IDS
      raise ValueError(
        "model policy must define the complete capability set "
        f"(missing={sorted(missing)}, unexpected={sorted(unexpected)})"
      )
    for capability_id, policy in capabilities.items():
      if capability_id != policy.capability_id:
        raise ValueError("model policy key must match capability_id")
    object.__setattr__(self, "revision", revision)
    object.__setattr__(self, "capabilities", MappingProxyType(capabilities))

  def admit_registry(self, registry: ProductModelRegistry) -> None:
    for capability_id, policy in self.capabilities.items():
      for key in policy.allowed_model_keys:
        entry = registry.require(key)
        if entry.lifecycle in {"deprecated", "disabled", "revoked"}:
          raise ValueError(
            f"{capability_id} allows {entry.lifecycle} model {key} for new selection"
          )
        if capability_id not in entry.capabilities:
          raise ValueError(f"{key} is not qualified for {capability_id}")
      if policy.default.kind == "model":
        entry = registry.require(policy.default.model_key or "")
        if entry.lifecycle in {"deprecated", "disabled", "revoked"}:
          raise ValueError(
            f"{capability_id} default uses {entry.lifecycle} model {entry.key}"
          )
        if policy.default.effort not in entry.supported_efforts:
          raise ValueError(
            f"{capability_id} default effort is unsupported by {entry.key}"
          )
      for channel, channel_default in policy.by_channel.items():
        if channel_default.kind == "inherit_parent":
          continue
        entry = registry.require(channel_default.model_key or "")
        if entry.key not in policy.allowed_model_keys:
          raise ValueError(
            f"{capability_id} channel {channel!r} default is not allowed"
          )
        if channel_default.effort not in entry.supported_efforts:
          raise ValueError(
            f"{capability_id} channel {channel!r} effort is unsupported"
          )


MODEL_REGISTRY_FILE_ENV_VAR = "AGENT_GATEWAY_MODEL_REGISTRY_FILE"
MODEL_SELECTION_FILE_ENV_VAR = "AGENT_GATEWAY_MODEL_SELECTION_FILE"
_MODEL_AUTHORITY_DIR = Path(__file__).with_name("model_authority")
DEFAULT_MODEL_REGISTRY_ARTIFACT = (
  _MODEL_AUTHORITY_DIR / "product-model-registry.yaml"
)
DEFAULT_MODEL_SELECTION_ARTIFACT = (
  _MODEL_AUTHORITY_DIR / "product-model-selection.yaml"
)

_REGISTRY_DOCUMENT_FIELDS = frozenset({"schema", "revision", "models"})
_REGISTRY_ENTRY_FIELDS = frozenset({
  "key",
  "label",
  "provider",
  "upstream_model",
  "adapter",
  "protocol_profile",
  "route",
  "lifecycle",
  "capabilities",
  "supported_efforts",
  "default_effort",
  "features",
  "reported_identities",
})
_POLICY_DOCUMENT_FIELDS = frozenset({"schema", "revision", "capabilities"})
_POLICY_CAPABILITY_REQUIRED_FIELDS = frozenset({
  "default",
  "by_channel",
  "allowed_model_keys",
})
_POLICY_CAPABILITY_OPTIONAL_FIELDS = frozenset({
  "allow_saved_preference",
  "allow_explicit_user",
  "allow_authenticated_run_override",
})
_POLICY_DEFAULT_REQUIRED_FIELDS = frozenset({"kind"})
_POLICY_DEFAULT_OPTIONAL_FIELDS = frozenset({"model_key", "effort"})


class _UniqueKeySafeLoader(yaml.SafeLoader):
  """SafeLoader that rejects duplicate mapping keys at every nesting level.

  The default SafeLoader silently keeps the last value for a repeated key,
  which would let an artifact replace an authored value without failing
  admission.  Duplicate keys must fail construction loudly instead.
  """

  def construct_mapping(self, node, deep=False):  # type: ignore[no-untyped-def]
    if not isinstance(node, yaml.MappingNode):
      raise yaml.constructor.ConstructorError(
        None,
        None,
        f"expected a mapping node, but found {node.id}",
        node.start_mark,
      )
    self.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
      key = self.construct_object(key_node, deep=deep)
      try:
        hash(key)
      except TypeError as exc:
        raise yaml.constructor.ConstructorError(
          "while constructing a mapping",
          node.start_mark,
          f"found unhashable key: {key!r}",
          key_node.start_mark,
        ) from exc
      if key in mapping:
        raise yaml.constructor.ConstructorError(
          "while constructing a mapping",
          node.start_mark,
          f"found duplicate mapping key: {key!r}",
          key_node.start_mark,
        )
      mapping[key] = self.construct_object(value_node, deep=deep)
    return mapping


def _artifact_document(path: Path) -> dict[str, object]:
  try:
    raw = path.read_text(encoding="utf-8")
  except OSError as exc:
    raise ValueError(f"model authority artifact is unreadable: {path}: {exc}") from exc
  try:
    document = yaml.load(raw, Loader=_UniqueKeySafeLoader)  # noqa: S506
  except yaml.YAMLError as exc:
    raise ValueError(f"model authority artifact is invalid YAML: {path}: {exc}") from exc
  if not isinstance(document, dict):
    raise ValueError(f"model authority artifact must be a mapping: {path}")
  return document


def _artifact_fields(
  value: object,
  *,
  required: frozenset[str],
  optional: frozenset[str] = frozenset(),
  context: str,
) -> dict[str, object]:
  if not isinstance(value, dict):
    raise ValueError(f"{context} must be a mapping")
  keys = {str(key) for key in value}
  unknown = keys - required - optional
  if unknown:
    raise ValueError(f"{context} has unknown fields: {', '.join(sorted(unknown))}")
  missing = required - keys
  if missing:
    raise ValueError(f"{context} is missing fields: {', '.join(sorted(missing))}")
  return {str(key): item for key, item in value.items()}


def _artifact_string(value: object, *, context: str) -> str:
  if not isinstance(value, str):
    raise ValueError(f"{context} must be a string")
  return value


def _artifact_string_list(value: object, *, context: str) -> tuple[str, ...]:
  if not isinstance(value, list) or not all(
    isinstance(item, str) for item in value
  ):
    raise ValueError(f"{context} must be a list of strings")
  if len(set(value)) != len(value):
    raise ValueError(f"{context} has duplicate items")
  return tuple(value)


def _artifact_string_mapping(value: object, *, context: str) -> dict[str, str]:
  if not isinstance(value, dict) or not all(
    isinstance(key, str) and isinstance(item, str) for key, item in value.items()
  ):
    raise ValueError(f"{context} must map strings to strings")
  return dict(value)


def _artifact_bool(value: object, *, context: str) -> bool:
  if not isinstance(value, bool):
    raise ValueError(f"{context} must be a boolean")
  return value


def _registry_entry_from_artifact(
  value: object,
  *,
  context: str,
) -> ModelRegistryEntry:
  fields = _artifact_fields(value, required=_REGISTRY_ENTRY_FIELDS, context=context)
  return ModelRegistryEntry(
    key=_artifact_string(fields["key"], context=f"{context}.key"),
    label=_artifact_string(fields["label"], context=f"{context}.label"),
    provider=_artifact_string(fields["provider"], context=f"{context}.provider"),
    upstream_model=_artifact_string(
      fields["upstream_model"],
      context=f"{context}.upstream_model",
    ),
    adapter=_artifact_string(fields["adapter"], context=f"{context}.adapter"),
    protocol_profile=_artifact_string(
      fields["protocol_profile"],
      context=f"{context}.protocol_profile",
    ),
    route=_artifact_string(fields["route"], context=f"{context}.route"),
    lifecycle=_artifact_string(  # type: ignore[arg-type]
      fields["lifecycle"],
      context=f"{context}.lifecycle",
    ),
    capabilities=_artifact_string_mapping(  # type: ignore[arg-type]
      fields["capabilities"],
      context=f"{context}.capabilities",
    ),
    supported_efforts=frozenset(
      _artifact_string_list(
        fields["supported_efforts"],
        context=f"{context}.supported_efforts",
      )
    ),
    default_effort=_artifact_string(
      fields["default_effort"],
      context=f"{context}.default_effort",
    ),
    features=frozenset(
      _artifact_string_list(fields["features"], context=f"{context}.features")
    ),
    reported_identities=frozenset(
      _artifact_string_list(
        fields["reported_identities"],
        context=f"{context}.reported_identities",
      )
    ),
  )


def load_model_registry(path: str | Path) -> ProductModelRegistry:
  """Parse and construct one admitted ``product-model-registry/v1`` artifact.

  Every schema violation — unknown field, missing field, malformed value,
  duplicate key, or incoherent lifecycle/exposure — raises ``ValueError``.
  """
  artifact_path = Path(path)
  context = f"model registry artifact {artifact_path}"
  document = _artifact_fields(
    _artifact_document(artifact_path),
    required=_REGISTRY_DOCUMENT_FIELDS,
    context=context,
  )
  models_raw = document["models"]
  if not isinstance(models_raw, list):
    raise ValueError(f"{context}.models must be a list of entries")
  models: dict[str, ModelRegistryEntry] = {}
  for index, raw_entry in enumerate(models_raw):
    entry = _registry_entry_from_artifact(
      raw_entry,
      context=f"{context}.models[{index}]",
    )
    if entry.key in models:
      raise ValueError(f"{context} has duplicate model key: {entry.key}")
    models[entry.key] = entry
  return ProductModelRegistry(
    schema=_artifact_string(  # type: ignore[arg-type]
      document["schema"],
      context=f"{context}.schema",
    ),
    revision=_artifact_string(document["revision"], context=f"{context}.revision"),
    models=models,
  )


def _capability_default_from_artifact(
  value: object,
  *,
  context: str,
) -> CapabilityDefault:
  fields = _artifact_fields(
    value,
    required=_POLICY_DEFAULT_REQUIRED_FIELDS,
    optional=_POLICY_DEFAULT_OPTIONAL_FIELDS,
    context=context,
  )
  kind = fields["kind"]
  model_key = fields.get("model_key")
  effort = fields.get("effort")
  if model_key is not None:
    model_key = _artifact_string(model_key, context=f"{context}.model_key")
  if effort is not None:
    effort = _artifact_string(effort, context=f"{context}.effort")
  return CapabilityDefault(
    kind=_artifact_string(kind, context=f"{context}.kind"),  # type: ignore[arg-type]
    model_key=model_key,
    effort=effort,
  )


def _capability_policy_from_artifact(
  capability_id: str,
  value: object,
  *,
  context: str,
) -> CapabilitySelectionPolicy:
  fields = _artifact_fields(
    value,
    required=_POLICY_CAPABILITY_REQUIRED_FIELDS,
    optional=_POLICY_CAPABILITY_OPTIONAL_FIELDS,
    context=context,
  )
  by_channel_raw = fields["by_channel"]
  if not isinstance(by_channel_raw, dict) or not all(
    isinstance(channel, str) for channel in by_channel_raw
  ):
    raise ValueError(f"{context}.by_channel must map channel names to defaults")
  by_channel = {
    channel: _capability_default_from_artifact(
      channel_default,
      context=f"{context}.by_channel[{channel!r}]",
    )
    for channel, channel_default in by_channel_raw.items()
  }
  return CapabilitySelectionPolicy(
    capability_id=capability_id,
    default=_capability_default_from_artifact(
      fields["default"],
      context=f"{context}.default",
    ),
    by_channel=by_channel,
    allowed_model_keys=frozenset(
      _artifact_string_list(
        fields["allowed_model_keys"],
        context=f"{context}.allowed_model_keys",
      )
    ),
    allow_saved_preference=_artifact_bool(
      fields.get("allow_saved_preference", False),
      context=f"{context}.allow_saved_preference",
    ),
    allow_explicit_user=_artifact_bool(
      fields.get("allow_explicit_user", False),
      context=f"{context}.allow_explicit_user",
    ),
    allow_authenticated_run_override=_artifact_bool(
      fields.get("allow_authenticated_run_override", False),
      context=f"{context}.allow_authenticated_run_override",
    ),
  )


def load_model_selection_policy(path: str | Path) -> ProductModelSelectionPolicy:
  """Parse and construct one admitted ``product-model-selection/v1`` artifact.

  Every schema violation raises ``ValueError``; the complete capability set is
  required and unknown fields are rejected at every level.
  """
  artifact_path = Path(path)
  context = f"model selection artifact {artifact_path}"
  document = _artifact_fields(
    _artifact_document(artifact_path),
    required=_POLICY_DOCUMENT_FIELDS,
    context=context,
  )
  capabilities_raw = document["capabilities"]
  if not isinstance(capabilities_raw, dict) or not all(
    isinstance(capability_id, str) for capability_id in capabilities_raw
  ):
    raise ValueError(f"{context}.capabilities must map capability ids to policies")
  capabilities = {
    capability_id: _capability_policy_from_artifact(
      capability_id,
      raw_policy,
      context=f"{context}.capabilities[{capability_id!r}]",
    )
    for capability_id, raw_policy in capabilities_raw.items()
  }
  return ProductModelSelectionPolicy(
    schema=_artifact_string(  # type: ignore[arg-type]
      document["schema"],
      context=f"{context}.schema",
    ),
    revision=_artifact_string(document["revision"], context=f"{context}.revision"),
    capabilities=capabilities,
  )


def _selected_artifact_path(env_var: str, default_path: Path) -> Path:
  configured = os.environ.get(env_var, "").strip()
  if not configured:
    return default_path
  return Path(configured)


MODEL_REGISTRY_ARTIFACT_PATH = _selected_artifact_path(
  MODEL_REGISTRY_FILE_ENV_VAR,
  DEFAULT_MODEL_REGISTRY_ARTIFACT,
)
MODEL_SELECTION_ARTIFACT_PATH = _selected_artifact_path(
  MODEL_SELECTION_FILE_ENV_VAR,
  DEFAULT_MODEL_SELECTION_ARTIFACT,
)

INITIAL_MODEL_REGISTRY = load_model_registry(MODEL_REGISTRY_ARTIFACT_PATH)

# Adapter support is admitted from installed adapter declarations, never a
# hand-maintained table.  ``agent_gateway/__init__`` admits the loaded INITIAL
# artifacts against ``providers.installed_adapter_route_support()`` for the
# gateway-executed capability set at import (this module cannot import
# ``providers`` without a cycle), and ``create_gateway_app`` re-runs the full
# closure over whatever registry the server is actually configured with.

INITIAL_MODEL_SELECTION_POLICY = load_model_selection_policy(
  MODEL_SELECTION_ARTIFACT_PATH
)
INITIAL_MODEL_SELECTION_POLICY.admit_registry(INITIAL_MODEL_REGISTRY)


__all__ = [
  "AdapterRouteSupport",
  "CAPABILITY_EXECUTION_PROCESS",
  "CAPABILITY_IDS",
  "CORE_CAPABILITY_IDS",
  "CapabilityDefault",
  "CapabilityExecutionProcess",
  "CapabilityExposure",
  "CapabilitySelectionPolicy",
  "DEFAULT_MODEL_REGISTRY_ARTIFACT",
  "DEFAULT_MODEL_SELECTION_ARTIFACT",
  "EXTERNALLY_EXECUTED_CAPABILITY_IDS",
  "GATEWAY_EXECUTED_CAPABILITY_IDS",
  "INITIAL_MODEL_REGISTRY",
  "INITIAL_MODEL_SELECTION_POLICY",
  "INVESTMENT_CAPABILITY_IDS",
  "MODEL_REGISTRY_ARTIFACT_PATH",
  "MODEL_REGISTRY_FILE_ENV_VAR",
  "MODEL_SELECTION_ARTIFACT_PATH",
  "MODEL_SELECTION_FILE_ENV_VAR",
  "ModelLifecycle",
  "ModelRegistryEntry",
  "ProductModelRegistry",
  "ProductModelSelectionPolicy",
  "RISK_CAPABILITY_IDS",
  "SelectionSource",
  "load_model_registry",
  "load_model_selection_policy",
]
