"""Derive one child's analytical outcome from granted authority and facts.

Part II's outcome qualifier is *mechanical* (design §4.3, T3-I08): it is a
total function of the authority frozen at admission plus the runtime facts the
child actually produced.  It never reads the ambient tool catalog or the live
tool surface — that read is exactly the RC2 defect, because the ambient surface
drifts after admission and would let a node be judged against capabilities it
was never granted.

**B-5 rename note.**  ``ResolvedAuthority`` does not exist at this HEAD.  The
granted capability↔tool mapping available to the settlement site today is
:class:`~agent_workflow_contracts.ToolGrant` (``AdmittedTask.tool_grant``) plus
``AdmittedTask.capability_bindings`` — both frozen at admission and sealed by
``AdmittedTask._identity_and_grants``, hence semantically identical to
``ResolvedAuthority.{grant,bindings}``.  When B-5 lands, ``grant`` and
``bindings`` collapse into one ``authority: ResolvedAuthority`` parameter read
for the same two fields: a mechanical rename with no logic change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agent_workflow_contracts import AnalyticalOutcome, ToolGrant


# Only retrieval capabilities can fail in a way that makes an assessment
# evidence-poor.  The capability vocabulary marks them with the read verb.
_SOURCE_CAPABILITY_SUFFIX = ".read/v1"

_RATIONALE_TURNS_EXHAUSTED = (
  "the child reached its turn ceiling before completing the objective"
)
_RATIONALE_MISSING_INPUTS = (
  "required upstream inputs were unavailable at admission"
)
_RATIONALE_ALL_INPUTS_MISSING = (
  "every required upstream input was unavailable and no evidence was observed"
)
_RATIONALE_ALL_RETRIEVALS_FAILED = (
  "every granted source-capability retrieval failed"
)
_RATIONALE_SOME_RETRIEVALS_FAILED = (
  "some granted source-capability retrievals failed"
)


def source_capability_tool_ids(
  *,
  grant: ToolGrant,
  bindings: Sequence[object],
) -> frozenset[str]:
  """Return the granted tool IDs bound to a source-reading capability.

  Membership is read from the admitted bindings only, then intersected with
  the admitted grant.  A binding that names a tool the grant does not carry is
  not a retrieval the child could ever have performed.
  """

  granted = {str(entry.tool_id) for entry in grant.tools}
  bound: set[str] = set()
  for binding in bindings:
    if getattr(binding, "kind", None) != "live_tool":
      continue
    capability = str(getattr(binding, "capability", "") or "")
    if not capability.endswith(_SOURCE_CAPABILITY_SUFFIX):
      continue
    for tool_id in getattr(binding, "tool_ids", ()) or ():
      bound.add(str(tool_id))
  return frozenset(bound & granted)


def derive_mechanical_outcome(
  *,
  grant: ToolGrant | None,
  bindings: Sequence[object] = (),
  failures: Mapping[str, tuple[int, int]],
  sources: Sequence[Mapping[str, object]] = (),
  narrative_present: bool,
  turns_exhausted: bool,
  missing_inputs: tuple[str, ...] = (),
) -> AnalyticalOutcome | None:
  """Derive the mechanical ``AnalyticalOutcome``, or ``None`` if unassessed.

  Total and closed over the design §4.3 ladder:

  * no admitted authority (``grant is None``) → ``None``.  There was no
    admission to assess against, so no assessment occurred.  This is the
    pinned reading of an absent ``task_entry`` at the settlement sites.
  * narrative absent → ``None``.  Settlement already failed; there is nothing
    to qualify.
  * ``turns_exhausted`` → ``partial``.
  * missing inputs → ``partial``; nothing left to work with → ``blocked``.
  * all granted source-capability retrievals failed → ``insufficient_evidence``.
  * some failed → ``partial``.
  * otherwise → ``complete``.

  ``failures`` is the ``fold_dispatch_failures`` map of ``tool_id →
  (ok, failed)``.  Rationale strings are code-stamped: the contract validator
  requires one on every non-``complete`` disposition and forbids
  ``unmet_requirements`` on ``complete``.
  """

  if grant is None or not narrative_present:
    return None

  if turns_exhausted:
    return AnalyticalOutcome(
      disposition="partial",
      assessment_source="mechanically_derived",
      assessment_rationale=_RATIONALE_TURNS_EXHAUSTED,
      unmet_requirements=("turns_exhausted",),
    )

  source_tool_ids = source_capability_tool_ids(grant=grant, bindings=bindings)
  attempted = 0
  succeeded = 0
  for tool_id in sorted(source_tool_ids):
    counts = failures.get(tool_id)
    if counts is None:
      continue
    ok_count, failed_count = int(counts[0]), int(counts[1])
    attempted += ok_count + failed_count
    succeeded += ok_count

  if missing_inputs:
    # "All missing" is read from the facts this derivation actually holds:
    # nothing was observed and no granted retrieval succeeded, so the node had
    # nothing to work with. The arm is unreachable at this HEAD (D-B3-1 passes
    # ``missing_inputs=()`` at the executed settlement site).
    if not sources and succeeded == 0:
      return AnalyticalOutcome(
        disposition="blocked",
        assessment_source="mechanically_derived",
        assessment_rationale=_RATIONALE_ALL_INPUTS_MISSING,
        unmet_requirements=tuple(str(name) for name in missing_inputs),
      )
    return AnalyticalOutcome(
      disposition="partial",
      assessment_source="mechanically_derived",
      assessment_rationale=_RATIONALE_MISSING_INPUTS,
      unmet_requirements=tuple(str(name) for name in missing_inputs),
    )

  if attempted and succeeded == 0:
    return AnalyticalOutcome(
      disposition="insufficient_evidence",
      assessment_source="mechanically_derived",
      assessment_rationale=_RATIONALE_ALL_RETRIEVALS_FAILED,
      unmet_requirements=tuple(
        tool_id
        for tool_id in sorted(source_tool_ids)
        if tool_id in failures
      ),
    )
  if attempted and succeeded < attempted:
    return AnalyticalOutcome(
      disposition="partial",
      assessment_source="mechanically_derived",
      assessment_rationale=_RATIONALE_SOME_RETRIEVALS_FAILED,
      unmet_requirements=tuple(
        tool_id
        for tool_id in sorted(source_tool_ids)
        if failures.get(tool_id, (0, 0))[1]
      ),
    )
  return AnalyticalOutcome(
    disposition="complete",
    assessment_source="mechanically_derived",
  )


__all__ = [
  "derive_mechanical_outcome",
  "source_capability_tool_ids",
]
