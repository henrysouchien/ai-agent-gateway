from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from model_authority.bind import CapabilityBind

FORK_SCOPE_RECEIPT_VERSION = "2"
SIDE_QUEST_FORK_KIND = "side_quest"
ForkDecision = Literal["allow", "deny"]

_FIELDS = frozenset({
  "version",
  "fork_kind",
  "tool_decisions",
  "capability_bind",
  "tenant_id",
  "billing_mode",
  "resolved_budget_usd",
  "max_turns",
  "suffix_ceiling",
})
_DECISION_FIELDS = frozenset({"tool", "decision", "reason"})


def _required_text(value: object, *, field_name: str) -> str:
  text = str(value or "").strip()
  if not text:
    raise ValueError(f"fork scope receipt {field_name} must be non-empty")
  return text


@dataclass(frozen=True, slots=True)
class ForkToolDecision:
  tool: str
  decision: ForkDecision
  reason: str

  def __init__(
    self,
    tool: object,
    decision: object,
    reason: object,
  ) -> None:
    normalized_tool = _required_text(
      tool,
      field_name="tool decision tool",
    )
    if decision not in {"allow", "deny"}:
      raise ValueError("fork tool decision must be allow or deny")
    normalized_decision: ForkDecision = (
      "allow" if decision == "allow" else "deny"
    )
    normalized_reason = _required_text(
      reason,
      field_name="tool decision reason",
    )
    object.__setattr__(self, "tool", normalized_tool)
    object.__setattr__(self, "decision", normalized_decision)
    object.__setattr__(self, "reason", normalized_reason)

  def to_dict(self) -> dict[str, str]:
    return {
      "tool": self.tool,
      "decision": self.decision,
      "reason": self.reason,
    }


@dataclass(frozen=True, slots=True)
class ForkScopeReceipt:
  fork_kind: Literal["side_quest"]
  tool_decisions: tuple[ForkToolDecision, ...]
  capability_bind: CapabilityBind
  tenant_id: str
  billing_mode: str
  resolved_budget_usd: float
  max_turns: int
  suffix_ceiling: int
  version: str = FORK_SCOPE_RECEIPT_VERSION

  def __init__(
    self,
    fork_kind: object,
    tool_decisions: tuple[ForkToolDecision, ...],
    capability_bind: CapabilityBind,
    tenant_id: object,
    billing_mode: object,
    resolved_budget_usd: object,
    max_turns: object,
    suffix_ceiling: object,
    version: object = FORK_SCOPE_RECEIPT_VERSION,
  ) -> None:
    if version != FORK_SCOPE_RECEIPT_VERSION:
      raise ValueError("fork scope receipt version must be '2'")
    if fork_kind != SIDE_QUEST_FORK_KIND:
      raise ValueError("F1 supports only side_quest forks")
    names = [decision.tool for decision in tool_decisions]
    if names != sorted(names) or len(names) != len(set(names)):
      raise ValueError(
        "fork scope receipt tool decisions must be unique and sorted"
      )
    if not isinstance(capability_bind, CapabilityBind):
      raise TypeError("fork scope receipt requires a CapabilityBind")
    if capability_bind.capability_id != "node.fork":
      raise ValueError("fork scope receipt capability bind must be node.fork")
    normalized_tenant_id = _required_text(
      tenant_id,
      field_name="tenant_id",
    )
    normalized_billing_mode = _required_text(
      billing_mode,
      field_name="billing_mode",
    )
    if (
      isinstance(resolved_budget_usd, bool)
      or not isinstance(resolved_budget_usd, (int, float))
      or not math.isfinite(float(resolved_budget_usd))
      or float(resolved_budget_usd) <= 0
    ):
      raise ValueError("fork scope receipt budget must be finite and positive")
    if (
      isinstance(max_turns, bool)
      or not isinstance(max_turns, int)
      or max_turns <= 0
    ):
      raise ValueError(
        "fork scope receipt max_turns must be a positive integer"
      )
    if (
      isinstance(suffix_ceiling, bool)
      or not isinstance(suffix_ceiling, int)
      or suffix_ceiling <= 0
    ):
      raise ValueError(
        "fork scope receipt suffix_ceiling must be a positive integer"
      )
    object.__setattr__(self, "version", FORK_SCOPE_RECEIPT_VERSION)
    object.__setattr__(self, "fork_kind", SIDE_QUEST_FORK_KIND)
    object.__setattr__(self, "tool_decisions", tool_decisions)
    object.__setattr__(self, "capability_bind", capability_bind)
    object.__setattr__(self, "tenant_id", normalized_tenant_id)
    object.__setattr__(self, "billing_mode", normalized_billing_mode)
    object.__setattr__(
      self,
      "resolved_budget_usd",
      resolved_budget_usd,
    )
    object.__setattr__(self, "max_turns", max_turns)
    object.__setattr__(self, "suffix_ceiling", suffix_ceiling)

  def to_dict(self) -> dict[str, Any]:
    return {
      "version": self.version,
      "fork_kind": self.fork_kind,
      "tool_decisions": [
        decision.to_dict() for decision in self.tool_decisions
      ],
      "capability_bind": self.capability_bind.to_json(),
      "tenant_id": self.tenant_id,
      "billing_mode": self.billing_mode,
      "resolved_budget_usd": float(self.resolved_budget_usd),
      "max_turns": self.max_turns,
      "suffix_ceiling": self.suffix_ceiling,
    }


def parse_fork_scope_receipt(raw: object) -> ForkScopeReceipt:
  if not isinstance(raw, Mapping):
    raise ValueError("fork scope receipt must be a mapping")
  fields = frozenset(raw)
  if fields != _FIELDS:
    missing = sorted(_FIELDS - fields)
    extra = sorted(fields - _FIELDS)
    details = []
    if missing:
      details.append("missing " + ", ".join(missing))
    if extra:
      details.append("unexpected " + ", ".join(extra))
    raise ValueError(
      "fork scope receipt fields are invalid"
      + (f": {'; '.join(details)}" if details else "")
    )
  raw_decisions = raw.get("tool_decisions")
  if not isinstance(raw_decisions, Sequence) or isinstance(
    raw_decisions,
    (str, bytes),
  ):
    raise ValueError("fork scope receipt tool_decisions must be a list")
  decisions: list[ForkToolDecision] = []
  for raw_decision in raw_decisions:
    if not isinstance(raw_decision, Mapping):
      raise ValueError("fork scope receipt tool decision must be a mapping")
    if frozenset(raw_decision) != _DECISION_FIELDS:
      raise ValueError("fork scope receipt tool decision fields are invalid")
    decisions.append(ForkToolDecision(
      tool=raw_decision["tool"],
      decision=raw_decision["decision"],
      reason=raw_decision["reason"],
    ))
  return ForkScopeReceipt(
    version=raw["version"],
    fork_kind=raw["fork_kind"],
    tool_decisions=tuple(decisions),
    capability_bind=CapabilityBind.from_json(raw["capability_bind"]),
    tenant_id=raw["tenant_id"],
    billing_mode=raw["billing_mode"],
    resolved_budget_usd=raw["resolved_budget_usd"],
    max_turns=raw["max_turns"],
    suffix_ceiling=raw["suffix_ceiling"],
  )


def fork_scope_receipt_dict(
  *,
  tool_decisions: Sequence[ForkToolDecision],
  capability_bind: CapabilityBind,
  tenant_id: str,
  billing_mode: str,
  resolved_budget_usd: float,
  max_turns: int,
  suffix_ceiling: int,
) -> dict[str, Any]:
  receipt = ForkScopeReceipt(
    fork_kind=SIDE_QUEST_FORK_KIND,
    tool_decisions=tuple(sorted(
      copy.deepcopy(tuple(tool_decisions)),
      key=lambda item: item.tool,
    )),
    capability_bind=capability_bind,
    tenant_id=tenant_id,
    billing_mode=billing_mode,
    resolved_budget_usd=resolved_budget_usd,
    max_turns=max_turns,
    suffix_ceiling=suffix_ceiling,
  )
  return receipt.to_dict()


__all__ = [
  "FORK_SCOPE_RECEIPT_VERSION",
  "ForkScopeReceipt",
  "ForkToolDecision",
  "SIDE_QUEST_FORK_KIND",
  "fork_scope_receipt_dict",
  "parse_fork_scope_receipt",
]
