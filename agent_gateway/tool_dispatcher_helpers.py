from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from . import approval_settings
from .event_log import EventLog

RELAY_POLICY_DENIED_SUB_CODE = "relay_policy_denied"
RELAY_POLICY_DENIED_MESSAGE = (
  "Denied automatically by relay chat policy — the user did not see or decline this request. "
  "Do not tell the user they denied it. To run this tool, the user must send the request from "
  "the Excel taskpane composer, where they can approve it interactively."
)


ToolResult = Tuple[Optional[Any], Optional[Dict[str, Any]]]
NeedsApprovalCallback = Callable[[str, Dict[str, Any], str], bool]
ApprovalKeyQualifier = Callable[[str, Dict[str, Any]], str]
ToolResult.__doc__ = "Standard tool return type: `(result, error)`."


def _approval_queue_timeout_seconds(expiry_seconds: float | int | None) -> float:
  return min(float(expiry_seconds or 600), approval_settings.approval_wait_seconds())


@dataclass
class TransportApprovalRequest:
  """Approval payload sent from the dispatcher to a client or UI layer."""

  tool_call_id: str
  nonce: str
  tool_name: str
  tool_input: Dict[str, Any]
  resolved_qualifier: str = ""
  timeout: float = 120.0
  reason: str = ""
  allow_persistent_approval: bool = True


@dataclass
class TransportApprovalResult:
  """Result returned after a user approves or denies a tool call."""

  approved: bool
  allow_tool_type: bool = False
  denied_by: str | None = None


ApprovalCallback = Callable[[TransportApprovalRequest], Awaitable[Optional[TransportApprovalResult]]]
LocalToolHandler = Callable[..., Awaitable[ToolResult]]

# Transport-layer aliases for callback-based integrations.
ApprovalRequest = TransportApprovalRequest
ApprovalDecision = TransportApprovalResult


def resolve_denied_provenance(denied_by: str | None) -> tuple[str, dict[str, Any]]:
  if denied_by == "relay_policy":
    return (
      RELAY_POLICY_DENIED_SUB_CODE,
      {
        "code": "user_denied",
        "sub_code": RELAY_POLICY_DENIED_SUB_CODE,
        "message": RELAY_POLICY_DENIED_MESSAGE,
      },
    )
  return "user_denied", {"code": "user_denied", "message": "User denied execution"}


@dataclass
class InterceptContext:
  """Context passed to a `ToolInterceptor` before tool execution."""

  tool_call_id: str
  tool_name: str
  tool_input: Dict[str, Any]
  session_id: str


@dataclass
class InterceptDecision:
  """Interceptor outcome.

  `action` should be one of `allow`, `warn`, `ask`, or `deny`.

  - allow: proceed normally
  - warn: proceed, but attach the message as a policy warning in the result
  - ask: route through the dispatcher's request_approval callback before
    executing; in headless contexts, resolves to deny unless an
    `on_headless_ask` hook overrides
  - deny: block and return an error to the model
  """

  action: str  # "allow", "deny", "warn", "ask"
  message: str = ""
  code: str = "interceptor"

  def __post_init__(self) -> None:
    if self.action not in {"allow", "deny", "warn", "ask"}:
      raise ValueError(f"Invalid interceptor action: {self.action!r}")


@dataclass
class InterceptResult:
  """Structured return from `_run_interceptors()`."""

  proceed: bool
  warnings: list[str] = field(default_factory=list)
  error: dict[str, Any] | None = None
  pending_ask: InterceptDecision | None = None


ToolInterceptor = Callable[[InterceptContext], Awaitable[InterceptDecision]]
ToolInterceptor.__doc__ = (
  "Async policy hook that receives `InterceptContext` and returns an `InterceptDecision`."
)
HeadlessAskCallback = Callable[[InterceptContext, InterceptDecision], Awaitable[str] | str]


@dataclass
class ToolExecutionContext:
  """Context object passed to local tool handlers.

  Local tools can call `emit()` to send additional structured events into the
  active `EventLog`, for example incremental stdout chunks from code execution.
  """

  tool_call_id: str
  tool_name: str
  event_log: EventLog
  resolved_qualifier: str = ""
  abort_event: asyncio.Event | None = None
  skill_run_id: str | None = None
  workspace_dir: str | None = None

  def emit(self, event: Dict[str, Any]) -> None:
    """Append a custom event to the active event log."""
    self.event_log.append(event)

  @property
  def aborted(self) -> bool:
    """Return True when the owning chat stream has disconnected."""
    return self.abort_event is not None and self.abort_event.is_set()

  async def wait_aborted(self) -> None:
    """Wait until the owning chat stream disconnects."""
    if self.abort_event is None:
      await asyncio.Future()
    else:
      await self.abort_event.wait()
