"""The mechanical outcome derivation table (B-3, T3-I08).

``derive_mechanical_outcome`` is total and closed over design §4.3's ladder and
reads authority **only** from the values frozen at admission — never the
ambient tool catalog or the live tool surface.
"""

from __future__ import annotations

import inspect

from agent_workflow_contracts import (
  LiveToolCapabilityBinding,
  ToolGrant,
  ToolGrantEntry,
  sha256_digest,
)

from agent_gateway.mechanical_outcome import (
  derive_mechanical_outcome,
  source_capability_tool_ids,
)


def _grant(*tool_ids: str) -> ToolGrant:
  return ToolGrant(
    grant_id="grant-1",
    tools=tuple(
      ToolGrantEntry(tool_id=tool_id, route_id="route-1", effect="read")
      for tool_id in tool_ids
    ),
    digest=sha256_digest({"grant": list(tool_ids)}),
  )


def _read_binding(capability: str, *tool_ids: str) -> LiveToolCapabilityBinding:
  return LiveToolCapabilityBinding(
    capability=capability,
    route_id="route-1",
    tool_ids=tuple(tool_ids),
  )


def _derive(**overrides):
  kwargs = {
    "grant": _grant("web_search"),
    "bindings": (_read_binding("research-web.read/v1", "web_search"),),
    "failures": {},
    "sources": (),
    "narrative_present": True,
    "turns_exhausted": False,
    "missing_inputs": (),
  }
  kwargs.update(overrides)
  return derive_mechanical_outcome(**kwargs)


def test_absent_admitted_authority_derives_no_outcome() -> None:
  # The pinned reading of an absent ``task_entry``: no admission was in scope,
  # so no assessment occurred. Never a false ``complete``.
  assert _derive(grant=None, bindings=()) is None


def test_absent_narrative_derives_no_outcome() -> None:
  # Settlement already failed; there is nothing to qualify.
  assert _derive(narrative_present=False) is None


def test_turns_exhausted_derives_partial_with_the_display_carrier() -> None:
  outcome = _derive(turns_exhausted=True)

  assert outcome is not None
  assert outcome.disposition == "partial"
  assert outcome.assessment_source == "mechanically_derived"
  assert outcome.assessment_rationale
  # ``ExecutionSettlement`` forbids a terminal_reason on ``succeeded``, so the
  # exhaustion fact must ride the outcome.
  assert outcome.unmet_requirements == ("turns_exhausted",)


def test_turns_exhausted_precedes_a_clean_retrieval_record() -> None:
  outcome = _derive(turns_exhausted=True, failures={"web_search": (3, 0)})

  assert outcome is not None
  assert outcome.disposition == "partial"


def test_all_source_capability_retrievals_failed_derives_insufficient() -> None:
  outcome = _derive(failures={"web_search": (0, 2)})

  assert outcome is not None
  assert outcome.disposition == "insufficient_evidence"
  assert outcome.assessment_source == "mechanically_derived"
  assert outcome.assessment_rationale
  assert outcome.unmet_requirements == ("web_search",)


def test_some_source_capability_retrievals_failed_derives_partial() -> None:
  outcome = _derive(failures={"web_search": (2, 1)})

  assert outcome is not None
  assert outcome.disposition == "partial"
  assert outcome.unmet_requirements == ("web_search",)


def test_clean_run_derives_complete_without_unmet_requirements() -> None:
  outcome = _derive(failures={"web_search": (2, 0)})

  assert outcome is not None
  assert outcome.disposition == "complete"
  assert outcome.assessment_source == "mechanically_derived"
  # The contract validator forbids unmet requirements on ``complete``.
  assert outcome.unmet_requirements == ()


def test_no_retrieval_attempted_derives_complete_not_insufficient() -> None:
  # "All retrievals failed" needs at least one retrieval. A node that never
  # called a source capability produced no evidence-poverty fact.
  outcome = _derive(failures={})

  assert outcome is not None
  assert outcome.disposition == "complete"


def test_failures_outside_the_grant_never_qualify_the_outcome() -> None:
  # The ambient catalog is not authority: a tool the child was never granted
  # cannot make its assessment evidence-poor.
  outcome = _derive(failures={"ungranted_tool": (0, 5)})

  assert outcome is not None
  assert outcome.disposition == "complete"


def test_non_read_capability_failures_never_qualify_the_outcome() -> None:
  outcome = _derive(
    grant=_grant("model_write"),
    bindings=(_read_binding("model.write/v1", "model_write"),),
    failures={"model_write": (0, 3)},
  )

  assert outcome is not None
  assert outcome.disposition == "complete"


def test_missing_inputs_derive_partial_and_total_absence_derives_blocked() -> None:
  # D-B3-1 keeps this arm unreachable at HEAD; it stays total regardless.
  partial = _derive(
    missing_inputs=("upstream_report",),
    failures={"web_search": (1, 0)},
  )
  assert partial is not None
  assert partial.disposition == "partial"
  assert partial.unmet_requirements == ("upstream_report",)

  blocked = _derive(missing_inputs=("upstream_report",), failures={})
  assert blocked is not None
  assert blocked.disposition == "blocked"
  assert blocked.assessment_rationale
  assert blocked.unmet_requirements == ("upstream_report",)


def test_source_capability_membership_is_the_grant_intersected_bindings() -> None:
  grant = _grant("web_search", "file_read")
  bindings = (
    _read_binding("research-web.read/v1", "web_search", "not_granted"),
    _read_binding("model.write/v1", "file_read"),
  )

  assert source_capability_tool_ids(grant=grant, bindings=bindings) == (
    frozenset({"web_search"})
  )


def test_derivation_signature_names_only_admitted_authority_inputs() -> None:
  # T3-I08 as a shape contract: the derivation takes the admitted grant and
  # bindings plus runtime facts. No catalog, registry, dispatcher, or session
  # parameter may appear — those are the ambient reads the RC2 defect made.
  parameters = set(
    inspect.signature(derive_mechanical_outcome).parameters
  )

  assert parameters == {
    "grant",
    "bindings",
    "failures",
    "sources",
    "narrative_present",
    "turns_exhausted",
    "missing_inputs",
  }


def test_derivation_module_imports_no_catalog_or_runtime_surface() -> None:
  from pathlib import Path

  source = (
    Path(inspect.getfile(derive_mechanical_outcome)).read_text(encoding="utf-8")
  )
  for forbidden in (
    "tool_dispatcher",
    "mcp_client",
    "get_tool_definitions",
    "snapshot_platform_catalog",
    "tool_catalog",
  ):
    assert forbidden not in source
