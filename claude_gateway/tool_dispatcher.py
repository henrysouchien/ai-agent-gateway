from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from .event_log import EventLog

if TYPE_CHECKING:
  from .mcp_client import McpClientManager


log = logging.getLogger("claude_gateway.dispatcher")


ToolResult = Tuple[Optional[Any], Optional[Dict[str, Any]]]
NeedsApprovalCallback = Callable[[str, Dict[str, Any], str], bool]
ApprovalKeyQualifier = Callable[[str, Dict[str, Any]], str]


@dataclass
class ApprovalRequest:
  tool_call_id: str
  nonce: str
  tool_name: str
  tool_input: Dict[str, Any]
  resolved_qualifier: str = ""
  timeout: float = 120.0


@dataclass
class ApprovalDecision:
  approved: bool
  allow_tool_type: bool = False


ApprovalCallback = Callable[[ApprovalRequest], Awaitable[Optional[ApprovalDecision]]]
LocalToolHandler = Callable[..., Awaitable[ToolResult]]


@dataclass
class InterceptContext:
  tool_call_id: str
  tool_name: str
  tool_input: Dict[str, Any]
  session_id: str


@dataclass
class InterceptDecision:
  action: str  # "allow", "deny", "warn"
  message: str = ""
  code: str = "interceptor"


ToolInterceptor = Callable[[InterceptContext], Awaitable[InterceptDecision]]


@dataclass
class ToolExecutionContext:
  tool_call_id: str
  tool_name: str
  event_log: EventLog
  resolved_qualifier: str = ""

  def emit(self, event: Dict[str, Any]) -> None:
    self.event_log.append(event)


class ToolDispatcher:
  def __init__(
    self,
    mcp_client: "McpClientManager",
    local_tool_handlers: Dict[str, LocalToolHandler] | None = None,
    needs_approval: Callable[..., bool] | None = None,
    request_approval: ApprovalCallback | None = None,
    approved_tool_types: Set[str] | None = None,
    event_log: EventLog | None = None,
    approval_key_qualifier: ApprovalKeyQualifier | None = None,
    interceptors: Sequence[ToolInterceptor] | None = None,
    session_id: str = "",
  ) -> None:
    self._mcp = mcp_client
    self._local = local_tool_handlers or {}
    self._needs_approval = self._normalize_needs_approval(needs_approval)
    self._request_approval = request_approval
    self._approved_tool_types = approved_tool_types if approved_tool_types is not None else set()
    self._event_log = event_log
    self._approval_key_qualifier = approval_key_qualifier
    self._interceptors: Sequence[ToolInterceptor] = list(interceptors or [])
    self._session_id = session_id

  async def _run_interceptors(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
  ) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Run interceptor chain. Returns (proceed, warnings, error_dict)."""
    if not self._interceptors:
      return True, [], None

    ctx = InterceptContext(
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_input=tool_input,
      session_id=self._session_id,
    )
    warnings: List[str] = []

    for interceptor in self._interceptors:
      try:
        decision = await interceptor(ctx)
      except Exception as exc:
        is_critical = getattr(interceptor, "__intercept_critical__", False)
        if is_critical:
          log.error("Critical interceptor %s failed: %s — denying", interceptor, exc)
          if self._event_log is not None:
            self._event_log.append(
              {
                "type": "interceptor_decision",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "action": "deny",
                "code": "interceptor_error",
                "message": f"Critical safety interceptor failed: {exc}",
              }
            )
          return False, [], {
            "code": "interceptor_error",
            "message": f"Safety check failed due to an internal error. Tool '{tool_name}' was blocked.",
          }
        log.warning("Interceptor %s error (non-fatal): %s", interceptor, exc)
        continue

      if decision.action == "deny":
        if self._event_log is not None:
          self._event_log.append(
            {
              "type": "interceptor_decision",
              "tool_call_id": tool_call_id,
              "tool_name": tool_name,
              "action": "deny",
              "code": decision.code,
              "message": decision.message,
            }
          )
        return False, [], {
          "code": decision.code,
          "message": decision.message or f"Tool '{tool_name}' was blocked by a runtime policy",
        }
      if decision.action == "warn":
        warnings.append(decision.message)
        if self._event_log is not None:
          self._event_log.append(
            {
              "type": "interceptor_decision",
              "tool_call_id": tool_call_id,
              "tool_name": tool_name,
              "action": "warn",
              "code": decision.code,
              "message": decision.message,
            }
          )

    return True, warnings, None

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    call_index: int = 0,
  ) -> ToolResult:
    proceed, warnings, intercept_error = await self._run_interceptors(
      tool_call_id,
      tool_name,
      tool_input,
    )
    if not proceed:
      return None, intercept_error

    qualifier = ""
    if self._approval_key_qualifier is not None:
      try:
        qualifier = self._approval_key_qualifier(tool_name, tool_input) or ""
      except Exception:
        qualifier = ""

    if self._should_request_approval(tool_name, tool_input, qualifier):
      if self._request_approval is None:
        return None, {
          "code": "approval_required",
          "message": f"Tool '{tool_name}' requires approval but no approval handler is configured",
        }
      decision = await self._request_approval(
        ApprovalRequest(
          tool_call_id=tool_call_id,
          nonce=os.urandom(8).hex(),
          tool_name=tool_name,
          tool_input=tool_input,
          resolved_qualifier=qualifier,
        )
      )
      if decision is None:
        return None, {"code": "approval_timeout", "message": "User did not respond within timeout"}
      if not decision.approved:
        return None, {"code": "user_denied", "message": "User denied execution"}
      if decision.allow_tool_type:
        self._approved_tool_types.add(self._qualified_key(tool_name, qualifier))

    result: Optional[Any]
    error: Optional[Dict[str, Any]]
    if tool_name in self._local:
      tool_ctx = None
      if self._event_log is not None:
        tool_ctx = ToolExecutionContext(
          tool_call_id=tool_call_id,
          tool_name=tool_name,
          event_log=self._event_log,
          resolved_qualifier=qualifier,
        )
      result, error = await self._local[tool_name](tool_input, call_index=call_index, tool_ctx=tool_ctx)
    elif self._mcp.is_mcp_tool(tool_name):
      result, error = await self._mcp.call_tool(tool_name, tool_input)
    else:
      result, error = None, {"code": "unknown_tool", "message": f"Unknown tool: {tool_name}"}

    if warnings and error is None and result is not None and isinstance(result, dict):
      result = dict(result)
      result["_interceptor_warnings"] = warnings

    return result, error

  @staticmethod
  def _normalize_needs_approval(
    needs_approval: Callable[..., bool] | None,
  ) -> NeedsApprovalCallback:
    if needs_approval is None:
      return lambda _name, _tool_input, _qualifier: False

    try:
      arg_count = len(inspect.signature(needs_approval).parameters)
    except (TypeError, ValueError):
      return needs_approval  # type: ignore[return-value]

    if arg_count == 1:
      return lambda name, _tool_input, _qualifier: needs_approval(name)
    if arg_count == 2:
      return lambda name, tool_input, _qualifier: needs_approval(name, tool_input)
    return needs_approval  # type: ignore[return-value]

  @staticmethod
  def _qualified_key(tool_name: str, qualifier: str) -> str:
    return f"{tool_name}:{qualifier}" if qualifier else tool_name

  def _should_request_approval(
    self,
    tool_name: str,
    tool_input: Dict[str, Any],
    qualifier: str,
  ) -> bool:
    qualified_key = self._qualified_key(tool_name, qualifier)
    if qualified_key in self._approved_tool_types:
      return False
    if not qualifier and tool_name in self._approved_tool_types:
      return False
    return self._needs_approval(tool_name, tool_input, qualifier)
