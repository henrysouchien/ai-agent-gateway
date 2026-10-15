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
  STATE_MUTATE,
  TRANSCRIPTS_READ,
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
  ("identity", "effect", "declared", "expected"),
  [
    (_local("web_fetch"), "read", WEB_READ, WEB_READ),
    (
      _mcp("research-corpus-mcp", "filings_search"),
      "read",
      FILINGS_READ,
      FILINGS_READ,
    ),
    (_mcp("market-data-mcp", "get_quote"), "read", MARKET_DATA_READ, MARKET_DATA_READ),
    (_local("unknown_read"), "support", None, CORPUS_READ),
    (
      _mcp("model-engine", "unknown_transform"),
      "pure_transform",
      COMPUTATION_EXECUTE,
      COMPUTATION_EXECUTE,
    ),
    (_local("preview"), "preview", None, ARTIFACT_PROPOSE),
    (
      _mcp("portfolio-writes-mcp", "artifact"),
      "artifact_write",
      None,
      ARTIFACT_PROPOSE,
    ),
    (_local("local_write"), "state_write", None, WORKSPACE_WRITE),
    (_addin("addin_write"), "state_write", None, WORKSPACE_WRITE),
    (_mcp("finance-cli-mcp", "server_write"), "state_write", None, STATE_MUTATE),
    (_mcp("alerts", "notify"), "external_write", None, STATE_MUTATE),
    (
      _mcp("portfolio-config-mcp", "configure"),
      "portfolio_config",
      None,
      STATE_MUTATE,
    ),
    (
      _mcp("portfolio-trades-mcp", "execute_trade"),
      "irreversible",
      None,
      STATE_MUTATE,
    ),
  ],
)
def test_registration_capability_preserves_current_versioned_vocabulary(
  identity: RegisteredToolIdentity,
  effect: str,
  declared: str | None,
  expected: str,
) -> None:
  actual = semantic_capability_for_registration(
    identity,
    effect,  # type: ignore[arg-type]
    read_capability=declared,
  )

  assert actual == expected
  assert actual.endswith("/v1")
  assert not actual.startswith("tool:")


def test_read_capability_is_the_owner_declaration_never_a_routing_list() -> None:
  # The same identity serves whatever its registration owner declares; with no
  # declaration a read still serves corpus.read, never "satisfies anything".
  identity = _mcp("research-corpus-mcp", "filings_search")
  assert semantic_capability_for_registration(
    identity,
    "read",
    read_capability=TRANSCRIPTS_READ,
  ) == TRANSCRIPTS_READ
  assert semantic_capability_for_registration(
    identity,
    "read",
    read_capability=None,
  ) == CORPUS_READ
  # Write-like effects derive from the effect; a read declaration does not move them.
  assert semantic_capability_for_registration(
    identity,
    "state_write",
    read_capability=FILINGS_READ,
  ) == STATE_MUTATE


def test_registration_capability_requires_exact_contract_values() -> None:
  identity = _local("web_fetch")

  with pytest.raises(TypeError, match="exact RegisteredToolIdentity"):
    semantic_capability_for_registration(identity.materialize(), "read", read_capability=None)  # type: ignore[arg-type]
  with pytest.raises(TypeError, match="effect must be an exact str"):
    semantic_capability_for_registration(identity, None, read_capability=None)  # type: ignore[arg-type]
  with pytest.raises(ToolRegistrationContractError, match="unsupported"):
    semantic_capability_for_registration(identity, "unknown", read_capability=None)  # type: ignore[arg-type]


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
    semantic_capability_for_registration(identity, "read", read_capability=None)


def test_registration_capability_corrections_are_frozen_and_exact() -> None:
  assert isinstance(SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT, MappingProxyType)
  assert dict(SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT) == {
    "irreversible": STATE_MUTATE,
    "portfolio_config": STATE_MUTATE,
  }
  with pytest.raises(TypeError):
    SEMANTIC_CAPABILITY_CORRECTIONS_BY_EFFECT["read"] = CORPUS_READ  # type: ignore[index]
