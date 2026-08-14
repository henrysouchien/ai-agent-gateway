"""Authoritative product model registry and model-selection policy.

The registry owns stable execution identity.  The policy owns defaults.  They
are deliberately ordinary, immutable deployment configuration rather than a
runtime service, and neither contains credential material.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias

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
  adapter: str
  provider: str
  protocol_profiles: frozenset[str]
  routes: frozenset[str]

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
  ) -> None:
    for entry in self.models.values():
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


_REASONING = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
_ADAPTIVE = frozenset({"low", "medium", "high", "xhigh", "max"})
_NONE = frozenset({"none"})
_XAI = frozenset({"low", "medium", "high"})


def _entry(
  key: str,
  label: str,
  provider: str,
  upstream_model: str,
  adapter: str,
  protocol_profile: str,
  route: str,
  capabilities: Mapping[str, CapabilityExposure],
  supported_efforts: frozenset[str],
  default_effort: str,
  *,
  features: frozenset[str] = frozenset({"tools", "streaming"}),
  reported_identities: frozenset[str] | None = None,
) -> ModelRegistryEntry:
  lifecycle: ModelLifecycle = (
    "active"
    if "user_selectable" in capabilities.values()
    else "hidden"
  )
  return ModelRegistryEntry(
    key=key,
    label=label,
    provider=provider,
    upstream_model=upstream_model,
    adapter=adapter,
    protocol_profile=protocol_profile,
    route=route,
    lifecycle=lifecycle,
    capabilities=capabilities,
    supported_efforts=supported_efforts,
    default_effort=default_effort,
    features=features,
    reported_identities=reported_identities or frozenset({upstream_model}),
  )


_DRIVER_AND_NODES = {
  "session.driver": "user_selectable",
  "plan.author": "internal",
  "node.explore": "internal",
  "node.implement": "internal",
  "node.mutate": "internal",
  "node.fork": "internal",
  "node.verify": "internal",
  "node.choose": "internal",
}
_DRIVER = {
  "session.driver": "user_selectable",
  "plan.author": "internal",
  "node.fork": "internal",
}

_MODEL_ENTRIES = (
  _entry("anthropic.claude-fable-5", "Fable 5", "anthropic", "claude-fable-5", "anthropic.messages", "messages.adaptive", "anthropic.public", _DRIVER_AND_NODES, _ADAPTIVE, "high"),
  _entry(
    "anthropic.claude-haiku-4-5",
    "Haiku 4.5",
    "anthropic",
    "claude-haiku-4-5",
    "anthropic.messages",
    "messages.standard",
    "anthropic.public",
    _DRIVER,
    _NONE,
    "none",
    reported_identities=frozenset({
      "claude-haiku-4-5",
      "claude-haiku-4-5-20251001",
    }),
  ),
  _entry("anthropic.claude-mythos-5", "Mythos 5", "anthropic", "claude-mythos-5", "anthropic.messages", "messages.adaptive", "anthropic.public", _DRIVER_AND_NODES, _ADAPTIVE, "high"),
  _entry("anthropic.claude-opus-5", "Opus 5", "anthropic", "claude-opus-5", "anthropic.messages", "messages.adaptive", "anthropic.public", _DRIVER_AND_NODES, _REASONING, "high"),
  _entry("anthropic.claude-sonnet-5", "Sonnet 5", "anthropic", "claude-sonnet-5", "anthropic.messages", "messages.adaptive", "anthropic.public", _DRIVER_AND_NODES, _REASONING, "high"),
  _entry("openai.gpt-5-6", "GPT-5.6", "openai", "gpt-5.6", "openai.responses", "responses.reasoning", "openai.public", {**_DRIVER, "investment.quant_worker": "internal"}, _REASONING, "medium"),
  _entry("codex.gpt-5-6-luna", "GPT-5.6 Luna", "codex", "gpt-5.6-luna", "codex.responses", "codex.reasoning", "codex.chatgpt", _DRIVER, _REASONING, "medium"),
  _entry("codex.gpt-5-6-sol", "GPT-5.6 Sol", "codex", "gpt-5.6-sol", "codex.responses", "codex.reasoning", "codex.chatgpt", _DRIVER, _REASONING, "medium"),
  _entry("codex.gpt-5-6-terra", "GPT-5.6 Terra", "codex", "gpt-5.6-terra", "codex.responses", "codex.reasoning", "codex.chatgpt", _DRIVER, _REASONING, "medium"),
  _entry("xai.grok-4-5", "Grok 4.5", "xai", "grok-4.5", "xai.responses", "responses.reasoning", "xai.public", _DRIVER, _XAI, "medium"),
  _entry("anthropic.claude-sonnet-4-6-sdk", "Claude Sonnet 4.6 (SDK)", "anthropic", "claude-sonnet-4-6", "anthropic.sdk.messages", "messages.standard", "anthropic.byok", {"risk.completion": "internal", "risk.interpretation": "internal", "investment.research_agent": "internal"}, _NONE, "none"),
  _entry("anthropic.claude-haiku-4-5-20251001-sdk", "Claude Haiku 4.5 2025-10-01 (SDK)", "anthropic", "claude-haiku-4-5-20251001", "anthropic.sdk.messages", "messages.standard", "anthropic.byok", {"risk.asset_classification": "internal", "risk.overview_editorial": "internal"}, _NONE, "none"),
  _entry("anthropic.claude-haiku-4-5-20251001-gateway", "Claude Haiku 4.5 2025-10-01 (Gateway)", "anthropic", "claude-haiku-4-5-20251001", "anthropic.messages", "messages.standard", "anthropic.public", {"investment.newsletter": "internal", "investment.earnings_transcript": "internal"}, _NONE, "none"),
  _entry("anthropic.claude-opus-4-8-oauth", "Claude Opus 4.8 (OAuth)", "anthropic", "claude-opus-4-8", "anthropic.sdk.messages", "messages.oauth", "anthropic.oauth", {"risk.document_ingest": "internal"}, _NONE, "none", features=frozenset({"vision"})),
  _entry("anthropic.claude-sonnet-4-20250514-sdk", "Claude Sonnet 4 2025-05-14 (SDK)", "anthropic", "claude-sonnet-4-20250514", "anthropic.sdk.messages", "messages.standard", "anthropic.service", {"investment.biotech_review": "internal"}, _NONE, "none"),
  _entry("openai.gpt-5-4-mini-sdk", "GPT-5.4 Mini (SDK)", "openai", "gpt-5.4-mini", "openai.sdk.chat_completions", "chat_completions.standard", "openai.service", {"risk.peer_generation": "internal"}, _NONE, "none"),
)

INITIAL_MODEL_REGISTRY = ProductModelRegistry(
  schema="product-model-registry/v1",
  revision="2026-08-13.1",
  models={entry.key: entry for entry in _MODEL_ENTRIES},
)

INITIAL_ADAPTER_ROUTE_SUPPORT = MappingProxyType({
  "anthropic.messages": AdapterRouteSupport(
    adapter="anthropic.messages",
    provider="anthropic",
    protocol_profiles=frozenset({"messages.standard", "messages.adaptive"}),
    routes=frozenset({"anthropic.public"}),
  ),
  "anthropic.sdk.messages": AdapterRouteSupport(
    adapter="anthropic.sdk.messages",
    provider="anthropic",
    protocol_profiles=frozenset({"messages.standard", "messages.oauth"}),
    routes=frozenset({"anthropic.byok", "anthropic.oauth", "anthropic.service"}),
  ),
  "openai.responses": AdapterRouteSupport(
    adapter="openai.responses",
    provider="openai",
    protocol_profiles=frozenset({"responses.reasoning"}),
    routes=frozenset({"openai.public"}),
  ),
  "openai.sdk.chat_completions": AdapterRouteSupport(
    adapter="openai.sdk.chat_completions",
    provider="openai",
    protocol_profiles=frozenset({"chat_completions.standard"}),
    routes=frozenset({"openai.service"}),
  ),
  "codex.responses": AdapterRouteSupport(
    adapter="codex.responses",
    provider="codex",
    protocol_profiles=frozenset({"codex.reasoning"}),
    routes=frozenset({"codex.chatgpt"}),
  ),
  "xai.responses": AdapterRouteSupport(
    adapter="xai.responses",
    provider="xai",
    protocol_profiles=frozenset({"responses.reasoning"}),
    routes=frozenset({"xai.public"}),
  ),
})
INITIAL_MODEL_REGISTRY.admit_adapter_support(INITIAL_ADAPTER_ROUTE_SUPPORT)


def _model_default(key: str, effort: str) -> CapabilityDefault:
  return CapabilityDefault(kind="model", model_key=key, effort=effort)


_ALL_DRIVER_KEYS = frozenset(
  entry.key
  for entry in _MODEL_ENTRIES
  if entry.capabilities.get("session.driver") == "user_selectable"
)
_OPUS_NODE_KEYS = frozenset({
  "anthropic.claude-opus-5",
  "anthropic.claude-sonnet-5",
  "anthropic.claude-fable-5",
  "anthropic.claude-mythos-5",
})

_POLICIES = {
  "session.driver": CapabilitySelectionPolicy("session.driver", _model_default("anthropic.claude-opus-5", "high"), {}, _ALL_DRIVER_KEYS, allow_saved_preference=True, allow_explicit_user=True),
  "plan.author": CapabilitySelectionPolicy("plan.author", CapabilityDefault("inherit_parent"), {}, _ALL_DRIVER_KEYS, allow_authenticated_run_override=True),
  "node.explore": CapabilitySelectionPolicy("node.explore", _model_default("anthropic.claude-opus-5", "high"), {}, _OPUS_NODE_KEYS),
  "node.implement": CapabilitySelectionPolicy("node.implement", _model_default("anthropic.claude-opus-5", "high"), {}, _OPUS_NODE_KEYS),
  "node.mutate": CapabilitySelectionPolicy("node.mutate", _model_default("anthropic.claude-opus-5", "high"), {}, _OPUS_NODE_KEYS),
  "node.fork": CapabilitySelectionPolicy("node.fork", CapabilityDefault("inherit_parent"), {}, _ALL_DRIVER_KEYS),
  "node.verify": CapabilitySelectionPolicy("node.verify", _model_default("anthropic.claude-opus-5", "high"), {}, _OPUS_NODE_KEYS),
  "node.choose": CapabilitySelectionPolicy("node.choose", _model_default("anthropic.claude-opus-5", "high"), {}, _OPUS_NODE_KEYS),
  "risk.completion": CapabilitySelectionPolicy("risk.completion", _model_default("anthropic.claude-sonnet-4-6-sdk", "none"), {}, frozenset({"anthropic.claude-sonnet-4-6-sdk"})),
  "risk.interpretation": CapabilitySelectionPolicy("risk.interpretation", _model_default("anthropic.claude-sonnet-4-6-sdk", "none"), {}, frozenset({"anthropic.claude-sonnet-4-6-sdk"})),
  "risk.peer_generation": CapabilitySelectionPolicy("risk.peer_generation", _model_default("openai.gpt-5-4-mini-sdk", "none"), {}, frozenset({"openai.gpt-5-4-mini-sdk"})),
  "risk.asset_classification": CapabilitySelectionPolicy("risk.asset_classification", _model_default("anthropic.claude-haiku-4-5-20251001-sdk", "none"), {}, frozenset({"anthropic.claude-haiku-4-5-20251001-sdk"})),
  "risk.overview_editorial": CapabilitySelectionPolicy("risk.overview_editorial", _model_default("anthropic.claude-haiku-4-5-20251001-sdk", "none"), {}, frozenset({"anthropic.claude-haiku-4-5-20251001-sdk"})),
  "risk.document_ingest": CapabilitySelectionPolicy("risk.document_ingest", _model_default("anthropic.claude-opus-4-8-oauth", "none"), {}, frozenset({"anthropic.claude-opus-4-8-oauth"})),
  "investment.research_agent": CapabilitySelectionPolicy("investment.research_agent", _model_default("anthropic.claude-sonnet-4-6-sdk", "none"), {}, frozenset({"anthropic.claude-sonnet-4-6-sdk"})),
  "investment.quant_worker": CapabilitySelectionPolicy("investment.quant_worker", _model_default("openai.gpt-5-6", "high"), {}, frozenset({"openai.gpt-5-6"})),
  "investment.newsletter": CapabilitySelectionPolicy("investment.newsletter", _model_default("anthropic.claude-haiku-4-5-20251001-gateway", "none"), {}, frozenset({"anthropic.claude-haiku-4-5-20251001-gateway"})),
  "investment.earnings_transcript": CapabilitySelectionPolicy("investment.earnings_transcript", _model_default("anthropic.claude-haiku-4-5-20251001-gateway", "none"), {}, frozenset({"anthropic.claude-haiku-4-5-20251001-gateway"})),
  "investment.biotech_review": CapabilitySelectionPolicy("investment.biotech_review", _model_default("anthropic.claude-sonnet-4-20250514-sdk", "none"), {}, frozenset({"anthropic.claude-sonnet-4-20250514-sdk"})),
}

INITIAL_MODEL_SELECTION_POLICY = ProductModelSelectionPolicy(
  schema="product-model-selection/v1",
  revision="2026-08-13.1",
  capabilities=_POLICIES,
)
INITIAL_MODEL_SELECTION_POLICY.admit_registry(INITIAL_MODEL_REGISTRY)


__all__ = [
  "AdapterRouteSupport",
  "CAPABILITY_IDS",
  "CORE_CAPABILITY_IDS",
  "CapabilityDefault",
  "CapabilityExposure",
  "CapabilitySelectionPolicy",
  "INITIAL_MODEL_REGISTRY",
  "INITIAL_MODEL_SELECTION_POLICY",
  "INITIAL_ADAPTER_ROUTE_SUPPORT",
  "INVESTMENT_CAPABILITY_IDS",
  "ModelLifecycle",
  "ModelRegistryEntry",
  "ProductModelRegistry",
  "ProductModelSelectionPolicy",
  "RISK_CAPABILITY_IDS",
  "SelectionSource",
]
