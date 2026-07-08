from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._provider_utils import sub_agent_default_model
from .task_registry import CoordinatorConfig, ProviderResolver


@dataclass(frozen=True)
class SubAgentModelResolution:
  resolved_provider: Any | None
  provider_name: str | None
  model: str


def resolve_sub_agent_model(
  *,
  raw_provider: str | None,
  raw_model: str | None,
  profile_model: str | None,
  effective_allowed_models: set[str] | None,
  provider_resolver: ProviderResolver | None,
  effective_coordinator: CoordinatorConfig | None,
  default_model: str,
  invalid_model_skill: str | None = None,
  skill_specific_invalid_model: bool = False,
  default_model_selector: Callable[[set[str] | None], str | None] = sub_agent_default_model,
) -> tuple[SubAgentModelResolution | None, dict[str, str] | None]:
  if raw_provider is None and effective_coordinator is not None and effective_coordinator.default_worker_provider:
    raw_provider = effective_coordinator.default_worker_provider

  effective_provider_resolver = provider_resolver
  if effective_provider_resolver is None and effective_coordinator is not None:
    effective_provider_resolver = effective_coordinator.provider_resolver

  resolved = None
  if raw_provider:
    if effective_provider_resolver is None:
      return None, {
        "code": "provider_not_supported",
        "message": f"Provider '{raw_provider}' requested but no provider_resolver configured",
      }
    try:
      resolved = effective_provider_resolver(raw_provider)
    except Exception as exc:
      return None, {"code": "invalid_provider", "message": str(exc)}

  if resolved is not None:
    effective_allowed = resolved.allowed_models
    effective_model = (
      raw_model
      or profile_model
      or default_model_selector(effective_allowed)
      or (
        effective_coordinator.default_worker_model
        if effective_coordinator is not None
        else None
      )
      or resolved.default_model
      or default_model
    )
  else:
    effective_allowed = effective_allowed_models
    effective_model = (
      raw_model
      or profile_model
      or default_model_selector(effective_allowed)
      or (
        effective_coordinator.default_worker_model
        if effective_coordinator is not None
        else None
      )
      or default_model
    )

  if effective_allowed and effective_model not in effective_allowed:
    if invalid_model_skill is not None and (skill_specific_invalid_model or resolved is None):
      return None, {
        "code": "invalid_input",
        "message": f"Invalid model '{effective_model}' for skill '{invalid_model_skill}'",
      }
    return None, {"code": "invalid_input", "message": f"Invalid model: {effective_model}"}

  return SubAgentModelResolution(
    resolved_provider=resolved,
    provider_name=raw_provider,
    model=effective_model,
  ), None


__all__ = [
  "SubAgentModelResolution",
  "resolve_sub_agent_model",
]
