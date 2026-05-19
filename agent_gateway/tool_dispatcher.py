from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from .event_log import EventLog

if TYPE_CHECKING:
  from .mcp_client import McpClientManager


log = logging.getLogger("agent_gateway.dispatcher")


ToolResult = Tuple[Optional[Any], Optional[Dict[str, Any]]]
NeedsApprovalCallback = Callable[[str, Dict[str, Any], str], bool]
ApprovalKeyQualifier = Callable[[str, Dict[str, Any]], str]
ToolResult.__doc__ = "Standard tool return type: `(result, error)`."


@dataclass
class ApprovalRequest:
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
class ApprovalDecision:
  """Result returned after a user approves or denies a tool call."""

  approved: bool
  allow_tool_type: bool = False


ApprovalCallback = Callable[[ApprovalRequest], Awaitable[Optional[ApprovalDecision]]]
LocalToolHandler = Callable[..., Awaitable[ToolResult]]


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


class ToolDispatcher:
  """Route tool calls to local handlers or MCP servers.

  The dispatcher is the policy boundary between model output and real tool
  execution. For each tool call it can:

  1. run interceptors
  2. request human approval
  3. execute a local Python handler
  4. fall back to an MCP server tool
  5. return structured warnings or errors
  """

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
    should_avoid_permission_prompts: bool = False,
    on_headless_ask: HeadlessAskCallback | None = None,
    mcp_session_inject_servers: set[str] | None = None,
    mcp_meta_inject_servers: frozenset[str] | None = None,
    user_id: str | None = None,
    risk_user_id: int | None = None,
    channel: str | None = None,
    role: str | None = None,
    credentials_resolver_active: bool = False,
    session_cache_denied_tools: frozenset[str] | None = None,
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
    self._should_avoid_permission_prompts = should_avoid_permission_prompts
    self._on_headless_ask = on_headless_ask
    self._mcp_session_inject_servers = mcp_session_inject_servers or set()
    self._mcp_meta_inject_servers = mcp_meta_inject_servers or frozenset()
    self._user_id = user_id
    self._risk_user_id = risk_user_id
    self._channel = channel
    self._role = role or "owner"
    self._credentials_resolver_active = credentials_resolver_active
    self._session_cache_denied = session_cache_denied_tools or frozenset()
    self._mcp_accepts_abort_event = self._callable_accepts_kw(getattr(self._mcp, "call_tool", None), "abort_event")

  async def _run_interceptors(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
  ) -> InterceptResult:
    """Run interceptor chain and return a structured result."""
    if not self._interceptors:
      return InterceptResult(proceed=True)

    ctx = InterceptContext(
      tool_call_id=tool_call_id,
      tool_name=tool_name,
      tool_input=tool_input,
      session_id=self._session_id,
    )
    warnings: List[str] = []
    pending_ask: InterceptDecision | None = None

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
          return InterceptResult(
            proceed=False,
            error={
              "code": "interceptor_error",
              "message": f"Safety check failed due to an internal error. Tool '{tool_name}' was blocked.",
            },
          )
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
        return InterceptResult(
          proceed=False,
          error={
            "code": decision.code,
            "message": decision.message or f"Tool '{tool_name}' was blocked by a runtime policy",
          },
        )
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
        continue
      if decision.action == "ask":
        if self._event_log is not None:
          self._event_log.append(
            {
              "type": "interceptor_decision",
              "tool_call_id": tool_call_id,
              "tool_name": tool_name,
              "action": "ask",
              "code": decision.code,
              "message": decision.message,
            }
          )
        if pending_ask is None:
          pending_ask = decision

    return InterceptResult(proceed=True, warnings=warnings, pending_ask=pending_ask)

  async def dispatch(
    self,
    tool_call_id: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    call_index: int = 0,
    abort_event: asyncio.Event | None = None,
  ) -> ToolResult:
    """Execute one tool call and return `(result, error)`.

    Args:
      tool_call_id: Provider-emitted tool id.
      tool_name: Tool name selected by the model.
      tool_input: JSON-like tool payload.
      call_index: Zero-based tool index for the current turn.

    Returns:
      A tuple of `(result, error)` where exactly one side is usually `None`.

    Notes:
      - Local handlers receive `tool_ctx` and `call_index` keyword arguments.
      - Approved tool types are cached in-session through `allow_tool_type`.
      - Interceptor warnings are attached to successful dict results under
        `_interceptor_warnings`.
    """
    if abort_event is not None and abort_event.is_set():
      raise asyncio.CancelledError()

    ir = await self._run_interceptors(
      tool_call_id,
      tool_name,
      tool_input,
    )
    if not ir.proceed:
      return None, ir.error

    qualifier = ""
    if self._approval_key_qualifier is not None:
      try:
        qualifier = self._approval_key_qualifier(tool_name, tool_input) or ""
      except Exception:
        qualifier = ""

    static_needs_approval = self._should_request_approval(tool_name, tool_input, qualifier)
    dynamic_ask = ir.pending_ask is not None

    if static_needs_approval or dynamic_ask:
      if self._should_avoid_permission_prompts:
        if static_needs_approval:
          reason_text = (
            ir.pending_ask.message
            if ir.pending_ask is not None
            else f"Tool '{tool_name}' requires static approval in headless context"
          )
          if self._event_log is not None:
            self._event_log.append(
              {
                "type": "headless_auto_deny",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "reason": reason_text,
                "source": "static",
              }
            )
          return None, {
            "code": "headless_auto_deny",
            "message": f"Tool '{tool_name}' blocked (static approval required): {reason_text}",
          }

        hook_result = "deny"
        if self._on_headless_ask is not None and ir.pending_ask is not None:
          headless_ctx = InterceptContext(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_input=tool_input,
            session_id=self._session_id,
          )
          try:
            raw = self._on_headless_ask(headless_ctx, ir.pending_ask)
            if inspect.isawaitable(raw):
              raw = await raw
            hook_result = raw if raw in ("allow", "deny") else "deny"
          except Exception as exc:
            log.warning("Headless ask hook failed: %s — auto-denying", exc)
            hook_result = "deny"

        if hook_result != "allow":
          reason_text = (
            ir.pending_ask.message
            if ir.pending_ask is not None
            else "Approval required in headless context"
          )
          if self._event_log is not None:
            self._event_log.append(
              {
                "type": "headless_auto_deny",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "reason": reason_text,
                "source": "interceptor",
              }
            )
          return None, {
            "code": "headless_auto_deny",
            "message": f"Tool '{tool_name}' blocked: {reason_text}",
          }
      else:
        if self._request_approval is None:
          return None, {
            "code": "approval_required",
            "message": f"Tool '{tool_name}' requires approval but no approval handler is configured",
          }
        allow_persistent = not dynamic_ask
        approval_reason = ir.pending_ask.message if ir.pending_ask is not None else ""
        decision = await self._request_approval(
          ApprovalRequest(
            tool_call_id=tool_call_id,
            nonce=os.urandom(8).hex(),
            tool_name=tool_name,
            tool_input=tool_input,
            resolved_qualifier=qualifier,
            reason=approval_reason,
            allow_persistent_approval=allow_persistent,
          )
        )
        if decision is None:
          return None, {"code": "approval_timeout", "message": "User did not respond within timeout"}
        if not decision.approved:
          return None, {"code": "user_denied", "message": "User denied execution"}
        if decision.allow_tool_type and allow_persistent and tool_name not in self._session_cache_denied:
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
          abort_event=abort_event,
        )
      result, error = await self._local[tool_name](tool_input, call_index=call_index, tool_ctx=tool_ctx)
    elif self._mcp.is_mcp_tool(tool_name):
      server = self._mcp.get_server_for_tool(tool_name)
      if server and server in self._mcp_meta_inject_servers:
        resolved_risk_user_id = self._risk_user_id
        if resolved_risk_user_id is None and self._user_id is not None and str(self._user_id).isdigit():
          resolved_risk_user_id = int(str(self._user_id))
        if self._credentials_resolver_active and not resolved_risk_user_id:
          raise RuntimeError("MCP meta user_id is required in strict mode")
        meta = {
          "session_id": self._session_id,
          "user_id": str(resolved_risk_user_id) if resolved_risk_user_id is not None else None,
          "channel": self._channel,
          "role": self._role,
        }
        result, error = await self._call_mcp_tool(tool_name, tool_input, meta=meta, abort_event=abort_event)
      elif server and server in self._mcp_session_inject_servers:
        tool_input = {**tool_input, "_session_id": self._session_id}
        result, error = await self._call_mcp_tool(tool_name, tool_input, abort_event=abort_event)
      else:
        result, error = await self._call_mcp_tool(tool_name, tool_input, abort_event=abort_event)
    else:
      result, error = None, {"code": "unknown_tool", "message": f"Unknown tool: {tool_name}"}

    if ir.warnings and error is None and result is not None and isinstance(result, dict):
      result = dict(result)
      result["_interceptor_warnings"] = ir.warnings

    return result, error

  def requires_approval(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
    """Return True if dispatching this tool would block on user approval."""
    if self._request_approval is None:
      return False
    qualifier = ""
    if self._approval_key_qualifier is not None:
      try:
        qualifier = self._approval_key_qualifier(tool_name, tool_input) or ""
      except Exception:
        qualifier = ""
    return self._should_request_approval(tool_name, tool_input, qualifier)

  async def _call_mcp_tool(
    self,
    tool_name: str,
    tool_input: Dict[str, Any],
    *,
    meta: Dict[str, Any] | None = None,
    abort_event: asyncio.Event | None = None,
  ) -> ToolResult:
    kwargs: Dict[str, Any] = {}
    if meta is not None:
      kwargs["meta"] = meta
    if abort_event is not None and self._mcp_accepts_abort_event:
      kwargs["abort_event"] = abort_event
    return await self._mcp.call_tool(tool_name, tool_input, **kwargs)

  @staticmethod
  def _callable_accepts_kw(callback: Any, keyword: str) -> bool:
    if callback is None:
      return False
    try:
      params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
      return False
    return keyword in params or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())

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
    if tool_name not in self._session_cache_denied:
      qualified_key = self._qualified_key(tool_name, qualifier)
      if qualified_key in self._approved_tool_types:
        return False
      if not qualifier and tool_name in self._approved_tool_types:
        return False
    return self._needs_approval(tool_name, tool_input, qualifier)
