from __future__ import annotations

# ruff: noqa: E402

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
for path in (ROOT, PKG_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from agent_gateway.capability_binding import ModelSelectionIntent
from agent_gateway.capability_execution import derive_batch_capability_execution
from tests.capability_execution_test_support import (
  stub_capability_execution_resolver,
)


def test_interactive_user_derivation_preserves_exact_binding_and_provenance() -> None:
  parent_resolver = stub_capability_execution_resolver()
  parent_execution = parent_resolver.resolve(
    "session.driver",
    explicit_intent=ModelSelectionIntent(
      model_key="test.openai.gpt-5",
      effort="high",
      source="explicit_user",
    ),
  )

  batch_resolver, batch_execution = derive_batch_capability_execution(
    parent_resolver,
    parent_execution,
  )

  assert batch_resolver.auth_context.run_mode == "batch"
  assert batch_resolver.auth_context.actor_id == parent_resolver.auth_context.actor_id
  assert batch_resolver.auth_context.tenant_id == parent_resolver.auth_context.tenant_id
  assert batch_resolver.auth_context.run_scoped_user_providers == frozenset({"openai"})
  assert batch_resolver.registry is parent_resolver.registry
  assert batch_resolver.selection_policy is parent_resolver.selection_policy
  assert batch_resolver.credential_materializer is parent_resolver.credential_materializer
  assert batch_resolver.adapter_resolver is parent_resolver.adapter_resolver
  assert batch_execution.bind == parent_execution.bind.model_copy(
    update={"run_mode": "batch"}
  )
  assert batch_execution.adapter is parent_execution.adapter
  assert batch_execution.auth_config["api_key"] == parent_execution.auth_config["api_key"]


def test_service_derivation_remains_service_scoped() -> None:
  base = stub_capability_execution_resolver()
  parent_resolver = replace(
    base,
    auth_context=replace(
      base.auth_context,
      user_provider_handles={},
      run_scoped_user_providers=frozenset(),
      allow_service_for_interactive=True,
    ),
  )
  parent_execution = parent_resolver.resolve("session.driver")
  batch_resolver, batch_execution = derive_batch_capability_execution(
    parent_resolver,
    parent_execution,
  )

  assert parent_execution.bind.credential_principal == "service"
  assert batch_execution.bind.credential_principal == "service"
  assert batch_execution.bind.run_mode == "batch"
  assert batch_execution.bind.upstream_model == parent_execution.bind.upstream_model
  assert parent_execution.bind.provider not in (
    batch_resolver.auth_context.run_scoped_user_providers
  )
