from __future__ import annotations

import re
from typing import Any, AsyncIterator, Iterable, cast

from model_authority.bind import CapabilityBind
from model_authority.binding import AuthContext, CredentialPrincipal, CredentialHandle, RunMode
from model_authority.capabilities import CAPABILITY_IDS
from model_authority.registry import ModelRegistryEntry, ProductModelRegistry
from model_authority.selection import (
  CapabilityDefault,
  CapabilitySelectionPolicy,
  ProductModelSelectionPolicy,
)
from model_authority.binding import CapabilityEffort
from agent_gateway.capability_execution import (
  BoundCapabilityExecution,
  CapabilityExecutionResolver,
  MaterializedCredential,
)
from agent_gateway.providers import CostEstimate, ModelInfo, ModelProvider, StreamEvent
from model_authority.thinking import ThinkingLevel
from model_authority.schema import SCHEMA
from model_authority.thinking import EffortResolution
from gateway_test_support.model_defaults import compat_for_profile


_DEFAULT_MODELS = (
  ("anthropic", "claude-sonnet-4-6"),
  ("anthropic", "claude-opus-4-6"),
  ("anthropic", "claude-opus-4-7"),
  ("anthropic", "claude-opus-4-8"),
  ("anthropic", "claude-sonnet-5"),
  ("anthropic", "claude-fable-5"),
  ("anthropic", "claude-mythos-5"),
  ("anthropic", "claude-haiku-4-5"),
  ("anthropic", "claude-default"),
  ("openai", "gpt-5"),
  ("openai", "gpt-5.6"),
  ("openai", "gpt-worker"),
)
_SUPPORTED_EFFORTS: frozenset[CapabilityEffort] = frozenset({
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
})


class _ExactTestProvider(ModelProvider):
  def __init__(self, name: str) -> None:
    self.name = name

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    auth_mode = str(config.get("auth_mode") or "").strip().lower()
    if auth_mode == "api":
      return bool(str(config.get("api_key") or "").strip())
    if auth_mode == "oauth":
      return bool(str(config.get("auth_token") or "").strip())
    return False

  def get_model_info(self, model: str) -> ModelInfo:
    return ModelInfo(
      id=model,
      provider=self.name,
      max_output_tokens=64_000,
      supports_thinking=True,
    )


class _RunnerTestProviderAdapter(ModelProvider):
  """Fill the provider interface around narrow runner-test doubles."""

  def __init__(self, wrapped: Any) -> None:
    self._wrapped = wrapped
    self.name = str(getattr(wrapped, "name", "stub") or "stub")

  def __getattr__(self, name: str) -> Any:
    return getattr(self._wrapped, name)

  def has_active_credential(self, config: dict[str, Any]) -> bool:
    callback = getattr(self._wrapped, "has_active_credential", None)
    if callable(callback):
      return bool(callback(config))
    return bool(
      str(config.get("api_key") or "").strip()
      or str(config.get("auth_token") or "").strip()
    )

  def create_client(
    self,
    config: dict[str, Any],
    *,
    timeout: float | None = None,
  ) -> Any:
    return self._wrapped.create_client(config, timeout=timeout)

  async def close_client(self, client: Any, timeout: float = 2.0) -> None:
    callback = self._wrapped.close_client
    try:
      await callback(client, timeout=timeout)
    except TypeError:
      await callback(client)

  def get_model_info(self, model: str) -> ModelInfo:
    callback = getattr(self._wrapped, "get_model_info", None)
    raw = callback(model) if callable(callback) else None
    if isinstance(raw, ModelInfo):
      return raw
    return ModelInfo(
      id=str(
        getattr(raw, "id", None)
        or getattr(raw, "model_id", None)
        or model
      ),
      provider=self.name,
      context_window=int(
        getattr(raw, "context_window", 200_000) or 200_000
      ),
      max_output_tokens=int(
        getattr(raw, "max_output_tokens", 16_384) or 16_384
      ),
      supports_thinking=bool(
        getattr(raw, "supports_thinking", True)
      ),
      supports_native_compaction=bool(
        getattr(raw, "supports_native_compaction", False)
      ),
    )

  def resolve_effort(
    self,
    *,
    requested: ThinkingLevel,
    model: str,
    model_info: ModelInfo,
    max_tokens: int,
    **request_context: Any,
  ) -> EffortResolution:
    callback = getattr(self._wrapped, "resolve_effort", None)
    if callback is not None:
      return callback(
        requested=requested,
        model=model,
        model_info=model_info,
        max_tokens=max_tokens,
        **request_context,
      )
    return super().resolve_effort(
      requested=requested,
      model=model,
      model_info=model_info,
      max_tokens=max_tokens,
      **request_context,
    )

  def build_request_params(self, **kwargs: Any) -> dict[str, Any]:
    return self._wrapped.build_request_params(**kwargs)

  def normalize_messages(
    self,
    messages: list[dict[str, Any]],
    model_info: ModelInfo,
  ) -> list[dict[str, Any]]:
    callback = getattr(self._wrapped, "normalize_messages", None)
    if callback is not None:
      return callback(messages, model_info)
    return list(messages)

  async def stream(
    self,
    client: Any,
    params: dict[str, Any],
  ) -> AsyncIterator[StreamEvent]:
    async for event in self._wrapped.stream(client, params):
      yield event

  def is_retryable_error(self, exc: Exception) -> bool:
    callback = getattr(self._wrapped, "is_retryable_error", None)
    return bool(callback(exc)) if callable(callback) else False

  def is_context_length_error(self, exc: Exception) -> bool:
    callback = getattr(self._wrapped, "is_context_length_error", None)
    return bool(callback(exc)) if callable(callback) else False

  def estimate_cost(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> CostEstimate:
    callback = getattr(self._wrapped, "estimate_cost", None)
    if callback is not None:
      try:
        return callback(
          model,
          input_tokens,
          output_tokens,
          cache_read_tokens=cache_read_tokens,
          cache_creation_tokens=cache_creation_tokens,
        )
      except TypeError:
        pass
    return super().estimate_cost(
      model,
      input_tokens,
      output_tokens,
      cache_read_tokens=cache_read_tokens,
      cache_creation_tokens=cache_creation_tokens,
    )


def stub_capability_execution_resolver(
  *,
  default_provider: str = "anthropic",
  default_model: str = "claude-sonnet-4-6",
  default_effort: str = "none",
  extra_models: Iterable[tuple[str, str]] = (),
  run_mode: str = "interactive",
  default_adapter: str | None = None,
  default_protocol_profile: str | None = None,
  default_route: str | None = None,
) -> CapabilityExecutionResolver:
  """Build a strict, deterministic resolver with inert credential handles."""

  default_identity = (default_provider, default_model)
  identities = tuple(dict.fromkeys((*_DEFAULT_MODELS, *extra_models, default_identity)))

  def _key(provider: str, model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    return f"test.{provider}.{slug}"

  def _entry(provider: str, model: str) -> ModelRegistryEntry:
    is_default = (provider, model) == default_identity
    return ModelRegistryEntry(
      key=_key(provider, model),
      label=f"{provider} {model}",
      provider=provider,
      upstream_model=model,
      adapter=(default_adapter if is_default and default_adapter else f"test.{provider}"),
      protocol_profile=(
        default_protocol_profile
        if is_default and default_protocol_profile
        else "responses.reasoning"
      ),
      route=(default_route if is_default and default_route else "test.in_process"),
      lifecycle="active",
      capabilities={
        capability_id: (
          "user_selectable"
          if capability_id == "session.driver"
          else "internal"
        )
        for capability_id in CAPABILITY_IDS
      },
      supported_efforts=_SUPPORTED_EFFORTS,
      default_effort="none",
      features=frozenset({"tools", "streaming"}),
      reported_identities=frozenset({model}),
      compat=compat_for_profile(
        default_protocol_profile
        if is_default and default_protocol_profile
        else "responses.reasoning"
      ),
    )
  entries = {
    _key(provider, model): _entry(provider, model)
    for provider, model in identities
  }
  registry = ProductModelRegistry(
    schema=SCHEMA,
    revision="test-capability-execution.1",
    models=entries,
  )
  default_key = _key(*default_identity)
  policy = ProductModelSelectionPolicy(
    schema=SCHEMA,
    revision="test-capability-execution.1",
    capabilities={
      capability_id: CapabilitySelectionPolicy(
        capability_id=capability_id,
        default=CapabilityDefault(
          kind="model",
          model_key=default_key,
          effort=default_effort,
        ),
        by_channel={},
        allowed_model_keys=frozenset(entries),
        allow_saved_preference=(capability_id == "session.driver"),
        allow_explicit_user=(capability_id == "session.driver"),
        allow_authenticated_run_override=(
          capability_id == "plan.author"
        ),
      )
      for capability_id in CAPABILITY_IDS
    },
  )
  providers = {
    provider: _ExactTestProvider(provider)
    for provider, _model in identities
  }
  user_handles = {
    provider: CredentialHandle(
      handle_id=f"test-user:{provider}",
      provider=provider,
      principal="user",
      tenant_id="test-tenant",
      actor_id="test-actor",
    )
    for provider in providers
  }
  service_handles = {
    provider: CredentialHandle(
      handle_id=f"test-service:{provider}",
      provider=provider,
      principal="service",
      tenant_id="test-tenant",
      actor_id=None,
    )
    for provider in providers
  }
  material_by_identity = {
    id(handle): MaterializedCredential(
      handle=handle,
      auth_config={
        "api_key": "test-secret",
        "auth_mode": "api",
        "provider": handle.provider,
      },
    )
    for handle in (*user_handles.values(), *service_handles.values())
  }

  def _materialize(handle: CredentialHandle) -> MaterializedCredential:
    return material_by_identity[id(handle)]

  return CapabilityExecutionResolver(
    registry=registry,
    selection_policy=policy,
    auth_context=AuthContext(
      run_mode=cast(RunMode, run_mode),
      actor_id="test-actor",
      tenant_id="test-tenant",
      user_provider_handles=user_handles,
      service_provider_handles=service_handles,
      entitled_capabilities=CAPABILITY_IDS,
      entitled_model_keys=frozenset(entries),
      run_scoped_user_providers=(
        frozenset(providers)
        if run_mode != "interactive"
        else frozenset()
      ),
    ),
    credential_materializer=_materialize,
    adapter_resolver=lambda adapter: providers[
      next(
        entry.provider
        for entry in entries.values()
        if entry.adapter == adapter
      )
    ],
  )


def stub_bound_capability_execution(
  *,
  provider: ModelProvider,
  model: str,
  effort: CapabilityEffort,
  auth_config: dict[str, Any] | None = None,
  capability_id: str = "session.driver",
  run_mode: RunMode = "interactive",
  credential_principal: CredentialPrincipal = "service",
) -> BoundCapabilityExecution:
  """Wrap a behavioral test provider in one strict immutable execution."""

  provider_name = str(getattr(provider, "name", "") or "").strip().lower()
  if provider_name == "agent-sdk":
    provider_name = "anthropic"
  if not provider_name:
    raise ValueError("test provider must declare a provider name")
  normalized_model = str(model or "").strip()
  if not normalized_model:
    raise ValueError("test execution requires an explicit model")
  config = dict(auth_config or {})
  forbidden = {
    "model",
    "model_key",
    "effort",
    "thinking",
    "thinking_enabled_requested",
    "execution_transport",
  } & set(config)
  if forbidden:
    raise ValueError(
      "test execution auth_config must not contain model-selection fields: "
      + ", ".join(sorted(forbidden))
    )
  config.update({
    "provider": provider_name,
    "auth_mode": str(config.get("auth_mode") or "api").strip().lower(),
  })
  if config["auth_mode"] == "api":
    config.setdefault("api_key", "test-secret")
  elif config["auth_mode"] == "oauth":
    config.setdefault("auth_token", "test-secret")
  bind = CapabilityBind(
    schema_version="1.0",
    capability_id=capability_id,
    model_key=f"test.{provider_name}.{re.sub(r'[^a-z0-9]+', '-', normalized_model.lower()).strip('-')}",
    provider=provider_name,
    upstream_model=normalized_model,
    adapter=(
      "anthropic.agent_sdk"
      if str(getattr(provider, "name", "")).strip().lower() == "agent-sdk"
      else f"test.{provider_name}"
    ),
    protocol_profile="responses.reasoning",
    route="test.in_process",
    effort=effort,
    credential_principal=credential_principal,
    credential_ref=f"test-{credential_principal}:{provider_name}",
    run_mode=run_mode,
    registry_revision="test-capability-execution.1",
    policy_revision="test-capability-execution.1",
    selection_source=(
      "capability_default"
      if capability_id == "session.driver"
      else "internal_policy"
    ),
  )
  entry = ModelRegistryEntry(
    key=bind.model_key,
    label=f"{provider_name} {normalized_model}",
    provider=bind.provider,
    upstream_model=bind.upstream_model,
    adapter=bind.adapter,
    protocol_profile=bind.protocol_profile,
    route=bind.route,
    lifecycle="active",
    capabilities={capability_id: "internal"},
    supported_efforts=frozenset({effort}),
    default_effort=effort,
    features=frozenset({"tools", "streaming"}),
    reported_identities=frozenset({normalized_model}),
    compat=compat_for_profile(bind.protocol_profile),
  )
  registry = ProductModelRegistry(
    schema=SCHEMA,
    revision="test-capability-execution.1",
    models={entry.key: entry},
  )
  return BoundCapabilityExecution(
    bind=bind,
    registry=registry,
    adapter=provider,
    auth_config=config,
  )


def stub_runner_capability_execution(
  *,
  provider: Any,
  model: str,
  effort: CapabilityEffort,
  auth_config: dict[str, Any] | None = None,
  capability_id: str = "session.driver",
  run_mode: RunMode = "interactive",
) -> BoundCapabilityExecution:
  """Bind a narrow runner test double through an exact immutable identity."""

  source_config = dict(auth_config or {})
  adapted_provider = (
    provider
    if isinstance(provider, ModelProvider)
    else _RunnerTestProviderAdapter(provider)
  )
  return stub_bound_capability_execution(
    provider=adapted_provider,
    model=model,
    effort=effort,
    auth_config=source_config,
    capability_id=capability_id,
    run_mode=run_mode,
  )


__all__ = [
  "stub_bound_capability_execution",
  "stub_capability_execution_resolver",
  "stub_runner_capability_execution",
]
