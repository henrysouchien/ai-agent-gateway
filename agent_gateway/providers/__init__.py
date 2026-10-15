from types import MappingProxyType
from typing import Mapping

from model_authority.capabilities import GATEWAY_EXECUTED_CAPABILITY_IDS
from model_authority.current import INITIAL_MODEL_REGISTRY
from model_authority.registry import AdapterRouteSupport
from .agent_sdk import AgentSDKConfig
from .anthropic import AnthropicProvider
from .base import CostEstimate, ModelInfo, ModelProvider, StreamEvent, truncate_to_last_compaction
from .codex import CodexProvider
from .copilot import CopilotResponsesProvider
from .google import GoogleGenerateContentProvider
from .google_antigravity import GoogleAntigravityProvider
from .kimi_code import KimiCodeProvider
from .muse import MuseResponsesProvider
from .ollama import OllamaLocalProvider
from .ollama_cloud import OllamaCloudProvider
from .openai import OpenAIProvider
from .xai import XAIProvider
from .credential_pool import (
  ProviderCredentialPool,
  get_provider_credential_pool,
  get_provider_pool_hooks,
  register_provider_pool_hooks,
  rotate_provider_credential,
)


_INSTALLED_PROVIDER_CLASSES: tuple[type[ModelProvider], ...] = (
  AnthropicProvider,
  CodexProvider,
  CopilotResponsesProvider,
  GoogleGenerateContentProvider,
  GoogleAntigravityProvider,
  KimiCodeProvider,
  MuseResponsesProvider,
  OllamaLocalProvider,
  OllamaCloudProvider,
  OpenAIProvider,
  XAIProvider,
)


def installed_adapter_providers() -> Mapping[str, type[ModelProvider]]:
  """Map each declared adapter id to the installed class implementing it.

  Built from the adapters' own ``adapter_route_support`` declarations — the
  installed code is the authority for what this package can execute; there is
  no hand-maintained adapter table.
  """

  classes: dict[str, type[ModelProvider]] = {}
  for provider_class in _INSTALLED_PROVIDER_CLASSES:
    declaration = provider_class.adapter_route_support()
    if declaration is None:
      raise ValueError(
        f"installed provider {provider_class.__name__} declares no adapter support"
      )
    if declaration.adapter in classes:
      raise ValueError(
        f"duplicate installed adapter declaration: {declaration.adapter!r}"
      )
    classes[declaration.adapter] = provider_class
  return MappingProxyType(classes)


def installed_adapter_route_support() -> Mapping[str, AdapterRouteSupport]:
  """Protocol support declared by the adapters installed in this package."""

  supports: dict[str, AdapterRouteSupport] = {}
  for adapter_id, provider_class in installed_adapter_providers().items():
    declaration = provider_class.adapter_route_support()
    assert declaration is not None  # installed_adapter_providers guarantees it
    supports[adapter_id] = declaration
  return MappingProxyType(supports)


# Admit application adapter support when providers load, not when a standalone
# controller imports a dependency-free package helper.
INITIAL_MODEL_REGISTRY.admit_adapter_support(
  installed_adapter_route_support(),
  executed_capability_ids=GATEWAY_EXECUTED_CAPABILITY_IDS,
)


__all__ = [
  "AgentSDKConfig",
  "AnthropicProvider",
  "CodexProvider",
  "CopilotResponsesProvider",
  "CostEstimate",
  "GoogleAntigravityProvider",
  "GoogleGenerateContentProvider",
  "KimiCodeProvider",
  "ModelInfo",
  "ModelProvider",
  "MuseResponsesProvider",
  "OllamaCloudProvider",
  "OllamaLocalProvider",
  "OpenAIProvider",
  "ProviderCredentialPool",
  "StreamEvent",
  "XAIProvider",
  "get_provider_credential_pool",
  "get_provider_pool_hooks",
  "installed_adapter_providers",
  "installed_adapter_route_support",
  "register_provider_pool_hooks",
  "rotate_provider_credential",
  "truncate_to_last_compaction",
]
