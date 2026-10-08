import asyncio
import sys
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = Path(__file__).resolve().parents[1]
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

import agent_gateway.autonomous as autonomous  # noqa: E402
from agent_workflow_contracts import AgentOperationRef  # noqa: E402
from agent_gateway import BoundCapabilityExecution  # noqa: E402
from agent_gateway.session import GatewaySession  # noqa: E402
from agent_gateway.operation_catalog import (  # noqa: E402
  AgentOperationCatalog,
  ResolvedOperationRuntime,
)
from gateway_test_support.capability_execution_test_support import (  # noqa: E402
  stub_capability_execution_resolver,
)


def _run(coro):
  return asyncio.run(coro)


class _EmptyOperationCatalog(AgentOperationCatalog):
  def resolve_operation(
    self,
    selector: AgentOperationRef | Mapping[str, Any] | None,
  ) -> ResolvedOperationRuntime:
    _ = selector
    raise FileNotFoundError("no operations")

  def list_callable_operations_with_descriptions(
    self,
  ) -> Sequence[tuple[AgentOperationRef, str]]:
    return []


def _bound_execution() -> dict[str, Any]:
  resolver = stub_capability_execution_resolver(run_mode="autonomous")
  resolved = resolver.resolve("session.driver")
  execution = BoundCapabilityExecution(
    bind=resolved.bind,
    registry=resolved.registry,
    adapter=resolved.adapter,
    auth_config={
      **resolved.auth_config,
      "max_tokens": 16_000,
    },
  )
  return {
    "capability_execution": execution,
    "admitted_skill_execution_limits": None,
    "capability_execution_resolver": resolver,
    "session": GatewaySession(
      session_id="autonomous-mcp-test",
      api_key_hash="test",
      created_at=1,
      expires_at=2,
      user_id="alice",
      role="owner",
    ),
  }



def test_run_autonomous_rejects_skills_dir_with_operation_catalog() -> None:
  async def _invoke() -> None:
    await autonomous.run_autonomous(
      "You are helpful.",
      "Run the task.",
      **_bound_execution(),
      skills_dir="skills",
      operation_catalog=_EmptyOperationCatalog(),
      user_id="alice",
      billing_mode="byok",
      rate_table_version="unknown",
    )

  try:
    _run(_invoke())
  except ValueError as exc:
    assert "either skills_dir or operation_catalog" in str(exc)
  else:
    raise AssertionError("expected mutually exclusive operation sources")


