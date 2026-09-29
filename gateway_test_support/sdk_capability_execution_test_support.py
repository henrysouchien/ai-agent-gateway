from __future__ import annotations

from dataclasses import replace
from agent_gateway.capability_execution import BoundCapabilityExecution

from gateway_test_support.capability_execution_test_support import (
  stub_capability_execution_resolver,
)


def stub_sdk_capability_execution(
  *,
  capability_id: str = "session.driver",
  provider: str = "anthropic",
  model: str = "claude-sonnet-5",
  effort: str = "none",
  auth_mode: str = "api",
  api_key: str = "test-secret",
  auth_token: str = "",
) -> BoundCapabilityExecution:
  """Return a real, immutable capability execution for SDK runner tests."""

  resolver = stub_capability_execution_resolver(
    default_provider=provider,
    default_model=model,
    default_effort=effort,
    default_adapter="anthropic.agent_sdk",
    default_protocol_profile="messages.adaptive",
  )
  execution = resolver.resolve(capability_id)
  return replace(
    execution,
    auth_config={
      **dict(execution.auth_config),
      "auth_mode": auth_mode,
      "api_key": api_key,
      "auth_token": auth_token,
    },
  )


__all__ = ["stub_sdk_capability_execution"]
