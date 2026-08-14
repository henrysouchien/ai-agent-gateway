from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
PKG_DIR = ROOT / "packages" / "agent-gateway"
if str(PKG_DIR) not in sys.path:
  sys.path.insert(0, str(PKG_DIR))

from agent_gateway.sub_agent_capability import (  # noqa: E402
  DELEGATION_ROLE_CAPABILITIES,
  DelegationRoleResolutionError,
  is_delegation_profile,
  resolve_delegation_role_capability,
  resolve_sub_agent_capability,
)


@pytest.mark.parametrize(
  ("validated_agent_name", "expected"),
  [
    ("explore", "node.explore"),
    ("research-digest", "node.explore"),
    ("research_digests", "node.explore"),
    ("summarize", "node.explore"),
    ("summarize artifact", "node.explore"),
    ("verify", "node.verify"),
    ("verify_finding", "node.verify"),
    ("adversarial-refute", "node.verify"),
  ],
)
def test_validated_agent_aliases_map_to_canonical_capabilities(
  validated_agent_name: str,
  expected: str,
) -> None:
  assert resolve_sub_agent_capability(
    validated_agent_name=validated_agent_name,
    mutation_mode=None,
  ) == expected


@pytest.mark.parametrize("mode", ["apply", "model_writer", "thesis-writer"])
def test_mutating_skill_mode_maps_to_node_mutate(mode: str) -> None:
  assert resolve_sub_agent_capability(
    validated_agent_name="summarize",
    mutation_mode=mode,
  ) == "node.mutate"


@pytest.mark.parametrize(
  ("validated_agent_name", "expected"),
  [
    ("research-digest", "node.explore"),
    ("verify_finding", "node.verify"),
    ("arbitrary-skill", "node.implement"),
    (None, "node.implement"),
  ],
)
def test_agent_alias_or_default_selects_capability(
  validated_agent_name: str | None,
  expected: str,
) -> None:
  assert resolve_sub_agent_capability(
    validated_agent_name=validated_agent_name,
    mutation_mode=None,
  ) == expected


@pytest.mark.parametrize(
  ("role", "expected"),
  sorted(DELEGATION_ROLE_CAPABILITIES.items()),
)
def test_explicit_delegation_role_map_is_exhaustive_and_normalized(
  role: str,
  expected: str,
) -> None:
  assert resolve_delegation_role_capability(role) == expected
  assert resolve_delegation_role_capability(role.replace("-", "_")) == expected


@pytest.mark.parametrize("role", ["", "  ", "arbitrary-role", "node.implement"])
def test_explicit_unmapped_delegation_role_is_refused(role: str) -> None:
  with pytest.raises(
    DelegationRoleResolutionError,
    match="Unmapped delegation role",
  ):
    resolve_delegation_role_capability(role)


@pytest.mark.parametrize(
  ("agent_name", "role", "expected"),
  [
    ("summarize-artifact", None, True),
    ("custom-curated-profile", "verify-finding", True),
    ("earnings-review", None, False),
  ],
)
def test_delegation_profile_predicate_covers_role_or_registered_alias(
  agent_name: str,
  role: str | None,
  expected: bool,
) -> None:
  assert is_delegation_profile(
    validated_agent_name=agent_name,
    delegation_role=role,
  ) is expected


def test_trusted_declared_role_controls_registered_skill_capability() -> None:
  assert resolve_sub_agent_capability(
    validated_agent_name="custom-curated-profile",
    mutation_mode=None,
    delegation_role="verify_finding",
  ) == "node.verify"


def test_trusted_declared_unknown_role_is_refused() -> None:
  with pytest.raises(DelegationRoleResolutionError):
    resolve_sub_agent_capability(
      validated_agent_name="custom-curated-profile",
      mutation_mode=None,
      delegation_role="unknown-role",
    )


def test_mutation_metadata_precedes_declared_delegation_role() -> None:
  assert resolve_sub_agent_capability(
    validated_agent_name="custom-curated-profile",
    mutation_mode="apply",
    delegation_role="verify-finding",
  ) == "node.mutate"


def test_mutation_metadata_does_not_hide_unmapped_delegation_role() -> None:
  with pytest.raises(DelegationRoleResolutionError):
    resolve_sub_agent_capability(
      validated_agent_name="custom-curated-profile",
      mutation_mode="apply",
      delegation_role="unknown-role",
    )
