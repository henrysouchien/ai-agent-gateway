from __future__ import annotations

from types import MappingProxyType

import pytest

from agent_gateway.semantic_capability_routing import (
  ARTIFACT_PROPOSE,
  COMPUTATION_EXECUTE,
  CORPUS_READ,
  FILINGS_READ,
  MARKET_DATA_READ,
  SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT,
  SEMANTIC_CAPABILITY_CORRECTIONS_BY_IDENTITY,
  STATE_MUTATE,
  WEB_READ,
  WORKSPACE_WRITE,
  semantic_capability_for_registration,
)
from agent_workflow_contracts.tool_registration import (
  RegisteredToolIdentity,
  ToolRegistrationContractError,
)


def _local(name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(route_kind="local_handler", logical_name=name)


def _addin(name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(route_kind="addin_relay", logical_name=name)


def _mcp(server: str, name: str) -> RegisteredToolIdentity:
  return RegisteredToolIdentity(
    route_kind="mcp",
    logical_server_id=server,
    logical_name=name,
  )


class _Text(str):
  pass


def _forged_identity(
  route_kind: object,
  logical_name: object,
  logical_server_id: object,
) -> RegisteredToolIdentity:
  identity = object.__new__(RegisteredToolIdentity)
  object.__setattr__(identity, "route_kind", route_kind)
  object.__setattr__(identity, "logical_name", logical_name)
  object.__setattr__(identity, "logical_server_id", logical_server_id)
  return identity


@pytest.mark.parametrize(
  ("identity", "effect", "expected"),
  [
    (_local("web_fetch"), "read", WEB_READ),
    (_mcp("research-corpus-mcp", "filings_search"), "read", FILINGS_READ),
    (_mcp("market-data-mcp", "get_quote"), "read", MARKET_DATA_READ),
    (_local("unknown_read"), "support", CORPUS_READ),
    (
      _mcp("model-engine", "unknown_transform"),
      "pure_transform",
      COMPUTATION_EXECUTE,
    ),
    (_local("preview"), "preview", ARTIFACT_PROPOSE),
    (
      _mcp("portfolio-writes-mcp", "artifact"),
      "artifact_write",
      ARTIFACT_PROPOSE,
    ),
    (_local("local_write"), "state_write", WORKSPACE_WRITE),
    (_addin("addin_write"), "state_write", WORKSPACE_WRITE),
    (_mcp("finance-cli-mcp", "server_write"), "state_write", STATE_MUTATE),
    (_mcp("alerts", "notify"), "external_write", STATE_MUTATE),
    (
      _mcp("portfolio-config-mcp", "configure"),
      "portfolio_config",
      STATE_MUTATE,
    ),
    (
      _mcp("portfolio-trades-mcp", "execute_trade"),
      "irreversible",
      STATE_MUTATE,
    ),
    (_mcp("timesfm", "timesfm_forecast"), "read", COMPUTATION_EXECUTE),
  ],
)
def test_registration_capability_preserves_current_versioned_vocabulary(
  identity: RegisteredToolIdentity,
  effect: str,
  expected: str,
) -> None:
  actual = semantic_capability_for_registration(identity, effect)  # type: ignore[arg-type]

  assert actual == expected
  assert actual.endswith("/v1")
  assert not actual.startswith("tool:")


def test_tool_capability_overrides_do_not_leak_across_same_bare_routes() -> None:
  assert semantic_capability_for_registration(
    _mcp("research-corpus-mcp", "filings_search"),
    "read",
  ) == FILINGS_READ
  assert semantic_capability_for_registration(
    _local("filings_search"),
    "read",
  ) == CORPUS_READ
  assert semantic_capability_for_registration(
    _mcp("market-data-mcp", "filings_search"),
    "read",
  ) == MARKET_DATA_READ

  assert semantic_capability_for_registration(
    _mcp("timesfm", "timesfm_forecast"),
    "read",
  ) == COMPUTATION_EXECUTE
  assert semantic_capability_for_registration(
    _local("timesfm_forecast"),
    "read",
  ) == CORPUS_READ
  assert semantic_capability_for_registration(
    _mcp("market-data-mcp", "timesfm_forecast"),
    "read",
  ) == MARKET_DATA_READ


def test_registration_capability_requires_exact_contract_values() -> None:
  identity = _local("web_fetch")

  with pytest.raises(TypeError, match="exact RegisteredToolIdentity"):
    semantic_capability_for_registration(identity.materialize(), "read")  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="effect must be an exact str"):
    semantic_capability_for_registration(identity, None)  # type: ignore[arg-type]
  with pytest.raises(ToolRegistrationContractError, match="unsupported"):
    semantic_capability_for_registration(identity, "unknown")  # type: ignore[arg-type]


@pytest.mark.parametrize(
  "identity",
  [
    _forged_identity(
      _Text("mcp"),
      "filings_search",
      "research-corpus-mcp",
    ),
    _forged_identity(
      "mcp",
      _Text("filings_search"),
      "research-corpus-mcp",
    ),
    _forged_identity(
      "mcp",
      "filings_search",
      _Text("research-corpus-mcp"),
    ),
    _forged_identity("mcp", "filings_search", None),
  ],
)
def test_registration_capability_revalidates_forged_exact_class_identities(
  identity: RegisteredToolIdentity,
) -> None:
  with pytest.raises(TypeError):
    semantic_capability_for_registration(identity, "read")


def test_registration_capability_corrections_are_frozen_and_exact() -> None:
  assert isinstance(SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT, MappingProxyType)
  assert isinstance(SEMANTIC_CAPABILITY_CORRECTIONS_BY_IDENTITY, MappingProxyType)
  assert dict(SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT) == {
    "irreversible": STATE_MUTATE,
    "portfolio_config": STATE_MUTATE,
  }
  assert dict(SEMANTIC_CAPABILITY_CORRECTIONS_BY_IDENTITY) == {
    _mcp("timesfm", "timesfm_forecast"): COMPUTATION_EXECUTE,
  }
  with pytest.raises(TypeError):
    SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT["read"] = CORPUS_READ  # type: ignore[index]
