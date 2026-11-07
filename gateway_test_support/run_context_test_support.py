"""Explicit registry declarations for dispatcher unit-test contexts."""

from typing import Any

from agent_gateway.approval_policy import RunContext
from agent_gateway.registry_scope import DeclaredRegistryScope


def run_context_for_test(
  *,
  registry_scope: DeclaredRegistryScope,
  user_id: str = "test-user",
  request_id: str = "test-request",
  **fields: Any,
) -> RunContext:
  return RunContext(
    registry_scope=registry_scope,
    user_id=user_id,
    request_id=request_id,
    **fields,
  )
