from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ThinkingLevel(str, Enum):
  """Provider-agnostic requested reasoning intensity."""

  NONE = "none"
  MINIMAL = "minimal"
  LOW = "low"
  MEDIUM = "medium"
  HIGH = "high"
  XHIGH = "xhigh"
  MAX = "max"


@dataclass(frozen=True)
class EffortResolution:
  requested: ThinkingLevel
  effective: ThinkingLevel
  thinking_enabled_effective: bool
  payload_fragments: Mapping[str, Any]

  def __post_init__(self) -> None:
    object.__setattr__(self, "payload_fragments", MappingProxyType(dict(self.payload_fragments)))


def parse_effort(value: Any, *, field_name: str = "effort", blank_is_unset: bool = False) -> ThinkingLevel | None:
  if value is None:
    return None
  if isinstance(value, ThinkingLevel):
    return value
  text = str(value).strip().lower()
  if not text:
    if blank_is_unset:
      return None
    raise ValueError(f"{field_name} must not be blank")
  try:
    return ThinkingLevel(text)
  except ValueError as exc:
    allowed = ", ".join(level.value for level in ThinkingLevel)
    raise ValueError(f"invalid {field_name}={value!r}; expected one of: {allowed}") from exc


def parse_thinking(value: Any, *, field_name: str = "thinking", blank_is_unset: bool = False) -> bool | None:
  if value is None:
    return None
  if isinstance(value, bool):
    return value
  text = str(value).strip().lower()
  if not text:
    if blank_is_unset:
      return None
    raise ValueError(f"{field_name} must not be blank")
  if text in {"1", "true", "yes", "on"}:
    return True
  if text in {"0", "false", "no", "off"}:
    return False
  raise ValueError(f"invalid {field_name}={value!r}; expected true or false")


def resolve_effort_pair(
  *,
  effort: Any = None,
  thinking: Any = None,
  field_prefix: str = "",
  blank_is_unset: bool = False,
) -> ThinkingLevel | None:
  effort_name = f"{field_prefix}effort" if field_prefix else "effort"
  thinking_name = f"{field_prefix}thinking" if field_prefix else "thinking"
  parsed_effort = parse_effort(effort, field_name=effort_name, blank_is_unset=blank_is_unset)
  parsed_thinking = parse_thinking(thinking, field_name=thinking_name, blank_is_unset=blank_is_unset)
  if parsed_thinking is None:
    return parsed_effort
  alias_effort = ThinkingLevel.HIGH if parsed_thinking else ThinkingLevel.NONE
  if parsed_effort is not None and parsed_effort != alias_effort:
    raise ValueError(
      f"conflicting {effort_name}={parsed_effort.value!r} and {thinking_name}={parsed_thinking!r}"
    )
  return parsed_effort or alias_effort


def canonical_effort_config(config: Mapping[str, Any], *, default: ThinkingLevel = ThinkingLevel.HIGH) -> dict[str, Any]:
  """Migrate a config layer to canonical requested effort and drop its alias."""
  normalized = dict(config)
  requested = resolve_effort_pair(effort=normalized.get("effort"), thinking=normalized.get("thinking"))
  requested = requested or default
  normalized.pop("thinking", None)
  normalized["effort"] = requested.value
  normalized["thinking_enabled_requested"] = requested != ThinkingLevel.NONE
  return normalized


def clamp_effort(requested: ThinkingLevel, supported: tuple[ThinkingLevel, ...]) -> ThinkingLevel:
  if not supported:
    return ThinkingLevel.NONE
  order = tuple(ThinkingLevel)
  requested_index = order.index(requested)
  candidates = [level for level in supported if order.index(level) <= requested_index]
  return max(candidates, key=order.index) if candidates else min(supported, key=order.index)
