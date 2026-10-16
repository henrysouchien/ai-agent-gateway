from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from model_authority.loader import PACKAGED_AUTHORITY_DIR, load_model_authority
from model_authority.selection import TierModelsDefault, TierSelectionPolicy

from .test_sub_agent import _CapabilityResolver, _Runner, _handler, _operation, _write_operation
from agent_gateway.sub_agent_helpers import make_run_agent_tool_def
from gateway_test_support.host_policy import owner_session_host_policy
from gateway_test_support.capability_execution_test_support import stub_capability_execution_resolver


@pytest.fixture(autouse=True)
def _read_routes(owner_session_host_policy: Any) -> None:
  owner_session_host_policy.get_local_tool_effect = lambda _name: "read"
  owner_session_host_policy.get_read_capability = lambda name, _server: (
    "web.read/v1" if name == "web_search" else None
  )


def _invoke(tmp_path: Path, resolver: _CapabilityResolver, **fields: Any):
  loader = _write_operation(tmp_path, execution_class="node.explore")
  runner = _Runner(capability_execution=resolver.resolve("session.driver"))
  handler = _handler(runner, loader=loader, resolver=resolver)
  resolver.calls.clear()
  result, error = asyncio.run(handler({
    "operation": _operation(loader),
    "objective": "Review filing evidence.",
    **fields,
  }))
  return runner, result, error


def test_public_selection_schema() -> None:
  schema = make_run_agent_tool_def()["input_schema"]
  assert schema["properties"]["role"]["type"] == "string"
  assert set(schema["properties"]["effort"]["enum"]) == {
    "none", "minimal", "low", "medium", "high", "xhigh", "max",
  }
  assert not {"oneOf", "anyOf", "allOf"} & set(schema)
  assert "required" not in schema


def test_spawn_role_and_effort_reach_selection(tmp_path: Path) -> None:
  resolver = _CapabilityResolver()
  model_key = resolver.selection_policy.capabilities["node.explore"].default.model_key
  assert model_key is not None
  object.__setattr__(resolver, "selection_policy", replace(
    resolver.selection_policy,
    tiers={"deep": TierSelectionPolicy(
      TierModelsDefault((model_key,), "high"), frozenset({model_key}),
    )},
  ))
  runner, result, error = _invoke(tmp_path, resolver, role="deep", effort="low")
  assert error is None
  assert result is not None
  assert resolver.calls[0]["role"] == "deep"
  assert resolver.calls[0]["effort"] == "low"
  assert runner.background_calls[0]["capability_bind_receipt"]["effort"] == "low"


def test_omitted_selection_keeps_operation_effort(tmp_path: Path) -> None:
  resolver = _CapabilityResolver()
  runner, result, error = _invoke(tmp_path, resolver)
  assert error is None
  assert result is not None
  assert resolver.calls[0]["role"] is None
  assert resolver.calls[0]["effort"] is None
  assert runner.background_calls[0]["capability_bind_receipt"]["effort"] == "high"


def test_omitted_fields_use_operation_selection(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
  from agent_gateway.admission import AdmissionOwners, SpawnRequest, admit
  from agent_gateway.admission import ParentContext

  resolver = _CapabilityResolver()
  model_key = resolver.selection_policy.capabilities["node.explore"].default.model_key
  assert model_key is not None
  object.__setattr__(resolver, "selection_policy", replace(
    resolver.selection_policy,
    tiers={"deep": TierSelectionPolicy(
      TierModelsDefault((model_key,), "high"), frozenset({model_key}),
    )},
  ))

  def operation_admit(
    request: SpawnRequest, parent: ParentContext, *, owners: AdmissionOwners,
  ):
    assert request.role is None
    assert request.effort is None
    return admit(request, parent, owners=replace(
      owners, operation=replace(owners.operation, role="deep", effort="medium"),
    ))

  monkeypatch.setattr("agent_gateway.sub_agent.admit", operation_admit)
  runner, result, error = _invoke(tmp_path, resolver)
  assert error is None
  assert result is not None
  assert resolver.calls[0]["role"] == "deep"
  assert resolver.calls[0]["effort"] == "medium"
  assert runner.background_calls[0]["capability_bind_receipt"]["effort"] == "medium"


@pytest.mark.parametrize("fields", [
  {"role": 1}, {"effort": False}, {"effort": "unsupported"}, {"extra": "deep"},
])
def test_invalid_selection_input(tmp_path: Path, fields: dict[str, Any]) -> None:
  resolver = _CapabilityResolver()
  runner, result, error = _invoke(tmp_path, resolver, **fields)
  assert result is None
  assert error is not None and error["code"] == "invalid_input"
  assert resolver.calls == []
  assert runner.background_calls == []


@pytest.mark.parametrize("fields, refusal", [
  ({"role": "no-such-role"}, "capability_role_unknown"),
  ({"effort": "low"}, None),
])
def test_packaged_policy_selection(
  tmp_path: Path, fields: dict[str, Any], refusal: str | None,
) -> None:
  resolver = _CapabilityResolver()
  provider = stub_capability_execution_resolver().resolve("session.driver").provider
  object.__setattr__(resolver, "adapter_resolver", lambda _adapter: provider)
  authority = load_model_authority(PACKAGED_AUTHORITY_DIR)
  object.__setattr__(resolver, "registry", authority.registry)
  object.__setattr__(resolver, "selection_policy", authority.selection_policy)
  object.__setattr__(resolver, "auth_context", replace(
    resolver.auth_context, entitled_model_keys=frozenset(authority.registry.models),
  ))
  runner, result, error = _invoke(tmp_path, resolver, **fields)
  if refusal is not None:
    assert result is None
    assert error is not None and error["code"] == refusal
    assert runner.background_calls == []
  else:
    assert error is None
    assert result is not None
    assert runner.background_calls[0]["capability_bind_receipt"]["effort"] == "low"
