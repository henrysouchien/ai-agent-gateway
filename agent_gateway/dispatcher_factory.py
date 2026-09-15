from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, overload

from .approval_route import (
  NO_APPROVAL_ROUTE,
  ApprovalRoute,
  DurableLocalApprovalRoute,
  NoApprovalRoute,
)
from .session import GatewaySession
from .policy_imports import resolve_effective_role
from .tool_dispatcher import (
  ApprovalKeyQualifier,
  ToolDispatcher,
)


@dataclass(frozen=True)
class GatewayDispatcherDeps:
  mcp_client: Any
  approval_store: Any
  approval_policy: Any
  mcp_meta_inject_servers: frozenset[str]
  tool_registration_catalog: Any | None = None
  tool_policy_implementations: Any | None = None
  redaction_context_factory: Any | None = None


@dataclass(frozen=True)
class InvocationPrincipal:
  session: GatewaySession
  approval_key_qualifier: ApprovalKeyQualifier | None = field(
    default=None,
    repr=False,
  )
  session_kind: str = field(init=False)
  user_id: str | None = field(init=False)
  risk_user_id: int | None = field(init=False)
  role: str | None = field(init=False)
  capabilities: frozenset[str] = field(init=False)
  channel: str | None = field(init=False)

  def __post_init__(self) -> None:
    object.__setattr__(
      self,
      "session_kind",
      str(getattr(self.session, "kind", "chat")),
    )
    object.__setattr__(self, "user_id", getattr(self.session, "user_id", None))
    object.__setattr__(
      self,
      "risk_user_id",
      getattr(self.session, "risk_user_id", None),
    )
    object.__setattr__(
      self,
      "role",
      resolve_effective_role(getattr(self.session, "role", None)),
    )
    object.__setattr__(
      self,
      "capabilities",
      frozenset(getattr(self.session, "capabilities", frozenset()) or frozenset()),
    )
    object.__setattr__(self, "channel", getattr(self.session, "channel", None))

  @classmethod
  def from_session(
    cls,
    session: GatewaySession,
    *,
    approval_key_qualifier: ApprovalKeyQualifier | None = None,
  ) -> "InvocationPrincipal":
    return cls(
      session=session,
      approval_key_qualifier=approval_key_qualifier,
    )


class DispatcherConstructionError(Exception):
  pass


_MISSING = object()
_SERVER_OWNED_PASSTHROUGH_KEYS = frozenset({
  "approval_key_qualifier",
  "approval_predicate_context_factory",
  "channel",
  "input_preparation_context_factory",
  "mcp_client",
  "mcp_meta_inject_servers",
  "risk_user_id",
  "role",
  "should_avoid_permission_prompts",
  "tool_policy_implementations",
  "tool_registration_catalog",
  "registered_approval_overlay",
  "redaction_context_factory",
  "user_id",
})


@overload
def build_tool_dispatcher(
  deps: GatewayDispatcherDeps,
  *,
  principal: InvocationPrincipal,
  profile: Literal["interactive"],
  event_log: Any,
  session_id: str,
  request_approval: Any,
  needs_approval: Any,
  approved_tool_types: set[str],
  local_tool_handlers: dict[str, Any],
  interceptors: Any = None,
  get_tool_definitions: Any = None,
  commercial_mcp_servers: frozenset[str] | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  session_cache_denied_tools: frozenset[str] | None = None,
  run_context: Any = None,
  excel_wrap: bool = False,
  channel_registry: Any = None,
  execute_addin: Any = None,
  channel_context: str | None = None,
  tool_packs: dict[str, dict[str, Any]] | None = None,
  input_preparation_context_factory: Any = None,
  approval_predicate_context_factory: Any = None,
  registered_approval_overlay: Any = None,
  addin_input_preparation_context_factory: Any = None,
  **passthrough: Any,
) -> Any: ...


@overload
def build_tool_dispatcher(
  deps: GatewayDispatcherDeps,
  *,
  principal: InvocationPrincipal,
  profile: Literal["chat_embedded"],
  event_log: Any,
  session_id: str,
  request_approval: Any,
  needs_approval: Any,
  approved_tool_types: set[str],
  local_tool_handlers: dict[str, Any],
  interceptors: Any = None,
  get_tool_definitions: Any = None,
  commercial_mcp_servers: frozenset[str] | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  session_cache_denied_tools: frozenset[str] | None = None,
  run_context: Any = None,
  excel_wrap: Literal[True],
  channel_registry: Any = None,
  execute_addin: Any = None,
  channel_context: str | None = None,
  tool_packs: dict[str, dict[str, Any]] | None = None,
  input_preparation_context_factory: Any = None,
  approval_predicate_context_factory: Any = None,
  registered_approval_overlay: Any = None,
  addin_input_preparation_context_factory: Any = None,
  **passthrough: Any,
) -> Any: ...


@overload
def build_tool_dispatcher(
  deps: GatewayDispatcherDeps,
  *,
  principal: InvocationPrincipal,
  profile: Literal["chat_embedded"],
  event_log: Any,
  session_id: str,
  request_approval: Any,
  needs_approval: Any,
  approved_tool_types: set[str],
  local_tool_handlers: dict[str, Any],
  interceptors: Any = None,
  get_tool_definitions: Any = None,
  commercial_mcp_servers: frozenset[str] | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  session_cache_denied_tools: frozenset[str] | None = None,
  run_context: Any = None,
  excel_wrap: Literal[False] = False,
  channel_registry: Any = None,
  execute_addin: Any = None,
  channel_context: str | None = None,
  tool_packs: dict[str, dict[str, Any]] | None = None,
  input_preparation_context_factory: Any = None,
  approval_predicate_context_factory: Any = None,
  registered_approval_overlay: Any = None,
  addin_input_preparation_context_factory: Any = None,
  **passthrough: Any,
) -> ToolDispatcher: ...


@overload
def build_tool_dispatcher(
  deps: GatewayDispatcherDeps,
  *,
  principal: InvocationPrincipal,
  profile: Literal["chat_embedded", "interactive"],
  event_log: Any,
  session_id: str,
  request_approval: Any,
  needs_approval: Any,
  approved_tool_types: set[str],
  local_tool_handlers: dict[str, Any],
  interceptors: Any = None,
  get_tool_definitions: Any = None,
  commercial_mcp_servers: frozenset[str] | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  session_cache_denied_tools: frozenset[str] | None = None,
  run_context: Any = None,
  excel_wrap: bool = False,
  channel_registry: Any = None,
  execute_addin: Any = None,
  channel_context: str | None = None,
  tool_packs: dict[str, dict[str, Any]] | None = None,
  input_preparation_context_factory: Any = None,
  approval_predicate_context_factory: Any = None,
  registered_approval_overlay: Any = None,
  addin_input_preparation_context_factory: Any = None,
  **passthrough: Any,
) -> Any: ...


def build_tool_dispatcher(
  deps: GatewayDispatcherDeps,
  *,
  principal: InvocationPrincipal,
  profile: Literal["chat_embedded", "interactive"],
  event_log: Any,
  session_id: str,
  request_approval: Any,
  needs_approval: Any,
  approved_tool_types: set[str],
  local_tool_handlers: dict[str, Any],
  interceptors: Any = None,
  get_tool_definitions: Any = None,
  commercial_mcp_servers: frozenset[str] | None = None,
  mcp_session_inject_servers: set[str] | None = None,
  session_cache_denied_tools: frozenset[str] | None = None,
  run_context: Any = None,
  excel_wrap: bool = False,
  channel_registry: Any = None,
  execute_addin: Any = None,
  channel_context: str | None = None,
  tool_packs: dict[str, dict[str, Any]] | None = None,
  input_preparation_context_factory: Any = None,
  approval_predicate_context_factory: Any = None,
  registered_approval_overlay: Any = None,
  addin_input_preparation_context_factory: Any = None,
  **passthrough: Any,
) -> Any:
  if (deps.tool_registration_catalog is None) != (
    deps.tool_policy_implementations is None
  ):
    raise DispatcherConstructionError(
      "registered tool catalog and policy implementations must be paired"
    )
  if principal.session_kind != getattr(principal.session, "kind", "chat"):
    raise DispatcherConstructionError(
      "invocation principal session kind does not match its authenticated session"
    )
  if profile not in {"chat_embedded", "interactive"}:
    raise DispatcherConstructionError(
      f"unsupported dispatcher construction profile: {profile!r}"
    )

  dispatcher_cls = passthrough.pop("_tool_dispatcher_cls", ToolDispatcher)
  excel_dispatcher_cls = passthrough.pop(
    "_excel_tool_dispatcher_cls",
    None,
  )
  supplied_session = passthrough.pop("session", _MISSING)
  if (
    supplied_session is not _MISSING
    and supplied_session is not principal.session
  ):
    raise DispatcherConstructionError(
      "dispatcher session must be the authenticated principal session"
    )
  if profile == "chat_embedded" and supplied_session is not _MISSING:
    raise DispatcherConstructionError(
      "chat_embedded dispatchers do not accept session wiring"
    )
  forbidden_keys = _SERVER_OWNED_PASSTHROUGH_KEYS.intersection(passthrough)
  if forbidden_keys:
    raise DispatcherConstructionError(
      "server-owned dispatcher kwargs cannot be overridden: "
      + ", ".join(sorted(forbidden_keys))
    )

  # The run's approval route is decided exactly here: this is the only place
  # holding the profile, the authenticated principal session and this
  # process's ledger and policy at the same instant. chat_embedded holds no
  # session and reaches no decider, so its route is 'none' whatever resources
  # this process happens to own.
  admitted_route: ApprovalRoute = NO_APPROVAL_ROUTE
  if (
    profile == "interactive"
    and deps.approval_store is not None
    and deps.approval_policy is not None
  ):
    admitted_route = DurableLocalApprovalRoute(
      deps.approval_store,
      deps.approval_policy,
      principal.session,
    )

  base_kwargs: dict[str, Any] = {
    "mcp_client": deps.mcp_client,
    "local_tool_handlers": local_tool_handlers,
    "needs_approval": needs_approval,
    "request_approval": request_approval,
    "approved_tool_types": approved_tool_types,
    "event_log": event_log,
  }
  if deps.tool_registration_catalog is not None:
    base_kwargs["tool_registration_catalog"] = (
      deps.tool_registration_catalog
    )
    base_kwargs["tool_policy_implementations"] = (
      deps.tool_policy_implementations
    )
    base_kwargs["input_preparation_context_factory"] = (
      input_preparation_context_factory
    )
    base_kwargs["approval_predicate_context_factory"] = (
      approval_predicate_context_factory
    )
    base_kwargs["registered_approval_overlay"] = (
      registered_approval_overlay
    )
    base_kwargs["redaction_context_factory"] = (
      deps.redaction_context_factory
    )

  if profile == "chat_embedded":
    if interceptors is not None:
      base_kwargs["interceptors"] = interceptors
    base_kwargs["session_id"] = session_id
    base_kwargs["mcp_session_inject_servers"] = mcp_session_inject_servers
    if principal.approval_key_qualifier is not None:
      base_kwargs["approval_key_qualifier"] = principal.approval_key_qualifier
    base_kwargs["session_cache_denied_tools"] = session_cache_denied_tools
    base_kwargs["get_tool_definitions"] = get_tool_definitions
    base_kwargs.update(passthrough)
    base_kwargs["commercial_mcp_servers"] = commercial_mcp_servers
  else:
    if principal.approval_key_qualifier is not None:
      base_kwargs["approval_key_qualifier"] = principal.approval_key_qualifier
    base_kwargs["interceptors"] = interceptors
    base_kwargs["session_id"] = session_id
    base_kwargs["user_id"] = principal.user_id
    base_kwargs["risk_user_id"] = principal.risk_user_id
    base_kwargs["channel"] = principal.channel
    base_kwargs["role"] = principal.role
    base_kwargs["mcp_meta_inject_servers"] = deps.mcp_meta_inject_servers
    if mcp_session_inject_servers is not None:
      base_kwargs["mcp_session_inject_servers"] = mcp_session_inject_servers
    base_kwargs.update(passthrough)
    base_kwargs["session_cache_denied_tools"] = session_cache_denied_tools
    if isinstance(admitted_route, NoApprovalRoute):
      # No route, so nothing carries the session: pass it as before.
      base_kwargs["session"] = principal.session
    base_kwargs["run_context"] = run_context
    base_kwargs["get_tool_definitions"] = get_tool_definitions
    if commercial_mcp_servers is not None:
      base_kwargs["commercial_mcp_servers"] = commercial_mcp_servers

  base_kwargs["mcp_meta_inject_servers"] = deps.mcp_meta_inject_servers
  base_kwargs["should_avoid_permission_prompts"] = False
  base_kwargs["approval_route"] = admitted_route

  base_dispatcher = dispatcher_cls(**base_kwargs)
  should_wrap = excel_wrap or profile == "interactive"
  if not should_wrap:
    return base_dispatcher
  if excel_dispatcher_cls is None:
    raise DispatcherConstructionError(
      "interactive dispatcher construction requires an Excel wrapper class"
    )
  wrapper_kwargs = {
    "base": base_dispatcher,
    "channel_registry": channel_registry,
    "execute_addin": execute_addin,
    "channel_context": channel_context,
    "tool_packs": tool_packs,
  }
  if addin_input_preparation_context_factory is not None:
    wrapper_kwargs["addin_input_preparation_context_factory"] = (
      addin_input_preparation_context_factory
    )
  return excel_dispatcher_cls(**wrapper_kwargs)


__all__ = [
  "DispatcherConstructionError",
  "GatewayDispatcherDeps",
  "InvocationPrincipal",
  "build_tool_dispatcher",
]
