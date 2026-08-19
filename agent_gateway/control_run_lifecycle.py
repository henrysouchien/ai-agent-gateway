from __future__ import annotations

from types import MappingProxyType
from typing import Literal, TypedDict, cast


CONTROL_RUN_CONTRACT_VERSION = "control-run-v1"


ControlRunState = Literal[
  "starting",
  "queued",
  "waiting",
  "running",
  "approval_pending",
  "completed",
  "budget_limited",
  "failed",
  "interrupted",
  "cancelled",
]


class ControlRunStateClassification(TypedDict):
  active: bool
  terminal: bool
  resumable: bool
  cancellable: bool


CONTROL_RUN_STATE_CLASSIFICATION: MappingProxyType[str, ControlRunStateClassification] = MappingProxyType(
  {
    "starting": {"active": True, "terminal": False, "resumable": False, "cancellable": True},
    "queued": {"active": True, "terminal": False, "resumable": False, "cancellable": True},
    "waiting": {"active": True, "terminal": False, "resumable": False, "cancellable": True},
    "running": {"active": True, "terminal": False, "resumable": False, "cancellable": True},
    "approval_pending": {
      "active": True,
      "terminal": False,
      "resumable": False,
      "cancellable": True,
    },
    "completed": {"active": False, "terminal": True, "resumable": False, "cancellable": False},
    "budget_limited": {"active": False, "terminal": True, "resumable": False, "cancellable": False},
    "failed": {"active": False, "terminal": True, "resumable": True, "cancellable": False},
    "interrupted": {"active": False, "terminal": True, "resumable": True, "cancellable": False},
    "cancelled": {"active": False, "terminal": True, "resumable": True, "cancellable": False},
  }
)

_INTERNAL_AUTONOMOUS_STATE_ALIASES: MappingProxyType[str, ControlRunState] = MappingProxyType(
  {
    "finished": "completed",
    "killed": "cancelled",
    "budget_exceeded": "budget_limited",
    "blocked": "failed",
    "remediating": "running",
  }
)
_AUTONOMOUS_RUN_RESUMABLE_INTERNAL_STATES = frozenset({"failed", "interrupted", "killed", "cancelled"})
_AUTONOMOUS_RUN_MESSAGEABLE_INTERNAL_STATES = frozenset({"running", "waiting", "approval_pending"})


def _states_where(classifier: str) -> frozenset[str]:
  return frozenset(
    state
    for state, classification in CONTROL_RUN_STATE_CLASSIFICATION.items()
    if classification[classifier]
  )


def _normalized_state_text(value: object) -> str | None:
  if not isinstance(value, str):
    return None
  normalized = value.strip().lower()
  return normalized or None


CONTROL_RUN_STATES = frozenset(CONTROL_RUN_STATE_CLASSIFICATION)
CONTROL_ACTIVE_RUN_STATES = _states_where("active")
CONTROL_TERMINAL_RUN_STATES = _states_where("terminal")
CONTROL_RESUMABLE_RUN_STATES = _states_where("resumable")
CONTROL_CANCELLABLE_RUN_STATES = _states_where("cancellable")

# Chat-kind control runs are continuable conversations. Unlike autonomous runs,
# a completed chat response is still a valid parent for the next user message.
CONTROL_CHAT_MESSAGEABLE_RUN_STATES = CONTROL_ACTIVE_RUN_STATES | frozenset({"completed"})
CONTROL_CHAT_TOKEN_FORGET_STATES = CONTROL_TERMINAL_RUN_STATES - CONTROL_CHAT_MESSAGEABLE_RUN_STATES


def coerce_control_run_state(value: object) -> ControlRunState | None:
  normalized = _normalized_state_text(value)
  if normalized is None:
    return None
  alias = _INTERNAL_AUTONOMOUS_STATE_ALIASES.get(normalized)
  if alias is not None:
    return alias
  if normalized in CONTROL_RUN_STATES:
    return cast(ControlRunState, normalized)
  return None


def canonical_control_run_state(value: object, *, default: ControlRunState = "running") -> ControlRunState:
  return coerce_control_run_state(value) or default


def is_control_run_state(value: object) -> bool:
  return isinstance(value, str) and value in CONTROL_RUN_STATES


def is_control_run_active_state(value: object) -> bool:
  return isinstance(value, str) and value in CONTROL_ACTIVE_RUN_STATES


def is_control_run_terminal_state(value: object) -> bool:
  return isinstance(value, str) and value in CONTROL_TERMINAL_RUN_STATES


def is_control_run_resumable_state(value: object) -> bool:
  return isinstance(value, str) and value in CONTROL_RESUMABLE_RUN_STATES


def is_control_run_cancellable_state(value: object) -> bool:
  return isinstance(value, str) and value in CONTROL_CANCELLABLE_RUN_STATES


def is_control_chat_messageable_state(value: object) -> bool:
  return isinstance(value, str) and value in CONTROL_CHAT_MESSAGEABLE_RUN_STATES


def should_forget_control_chat_token(value: object) -> bool:
  return isinstance(value, str) and value in CONTROL_CHAT_TOKEN_FORGET_STATES


def is_autonomous_run_internal_resumable_state(value: object) -> bool:
  normalized = _normalized_state_text(value)
  return normalized in _AUTONOMOUS_RUN_RESUMABLE_INTERNAL_STATES


def is_autonomous_run_internal_messageable_state(value: object) -> bool:
  normalized = _normalized_state_text(value)
  return normalized in _AUTONOMOUS_RUN_MESSAGEABLE_INTERNAL_STATES
