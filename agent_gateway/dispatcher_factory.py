from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from .session import GatewaySession
from .policy_imports import resolve_effective_role
from .tool_dispatcher import (
  ApprovalKeyQualifier,
  ToolDispatcher,
)

if TYPE_CHECKING:
  from agent.interactive.tool_dispatcher import ExcelToolDispatcher
else:
  ExcelToolDispatcher = Any


@dataclass(frozen=True)
class GatewayDispatcherDeps:
  mcp_client: Any
  approval_store: Any
  approval_policy: Any
  mcp_meta_inject_servers: frozenset[str]


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
  "channel",
  "mcp_client",
  "mcp_meta_inject_servers",
  "risk_user_id",
  "role",
  "should_avoid_permission_prompts",
  "user_id",
})


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
  **passthrough: Any,
) -> ToolDispatcher | ExcelToolDispatcher:
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
  supplied_store = passthrough.pop("store", _MISSING)
  supplied_policy = passthrough.pop("policy", _MISSING)
  if (
    supplied_session is not _MISSING
    and supplied_session is not principal.session
  ):
    raise DispatcherConstructionError(
      "dispatcher session must be the authenticated principal session"
    )
  if profile == "chat_embedded" and (
    supplied_session is not _MISSING
    or supplied_store is not _MISSING
    or supplied_policy is not _MISSING
  ):
    raise DispatcherConstructionError(
      "chat_embedded dispatchers do not accept session/store/policy wiring"
    )
  forbidden_keys = _SERVER_OWNED_PASSTHROUGH_KEYS.intersection(passthrough)
  if forbidden_keys:
    raise DispatcherConstructionError(
      "server-owned dispatcher kwargs cannot be overridden: "
      + ", ".join(sorted(forbidden_keys))
    )

  base_kwargs: dict[str, Any] = {
    "mcp_client": deps.mcp_client,
    "local_tool_handlers": local_tool_handlers,
    "needs_approval": needs_approval,
    "request_approval": request_approval,
    "approved_tool_types": approved_tool_types,
    "event_log": event_log,
  }

  if profile == "chat_embedded":
    if interceptors is not None:
      base_kwargs["interceptors"] = interceptors
    base_kwargs["session_id"] = session_id
    base_kwargs["mcp_session_inject_servers"] = mcp_session_inject_servers
    base_kwargs["approval_key_qualifier"] = principal.approval_key_qualifier
    base_kwargs["session_cache_denied_tools"] = session_cache_denied_tools
    base_kwargs["get_tool_definitions"] = get_tool_definitions
    base_kwargs.update(passthrough)
    base_kwargs["commercial_mcp_servers"] = commercial_mcp_servers
  else:
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
    base_kwargs["session"] = principal.session
    base_kwargs["store"] = (
      deps.approval_store
      if supplied_store is _MISSING
      else supplied_store
    )
    base_kwargs["policy"] = (
      deps.approval_policy
      if supplied_policy is _MISSING
      else supplied_policy
    )
    base_kwargs["run_context"] = run_context
    base_kwargs["get_tool_definitions"] = get_tool_definitions
    if commercial_mcp_servers is not None:
      base_kwargs["commercial_mcp_servers"] = commercial_mcp_servers

  base_kwargs["mcp_meta_inject_servers"] = deps.mcp_meta_inject_servers
  base_kwargs["should_avoid_permission_prompts"] = False

  base_dispatcher = dispatcher_cls(**base_kwargs)
  should_wrap = excel_wrap or profile == "interactive"
  if not should_wrap:
    return base_dispatcher
  if excel_dispatcher_cls is None:
    raise DispatcherConstructionError(
      "interactive dispatcher construction requires an Excel wrapper class"
    )
  return excel_dispatcher_cls(
    base=base_dispatcher,
    channel_registry=channel_registry,
    execute_addin=execute_addin,
    channel_context=channel_context,
    tool_packs=tool_packs,
  )


__all__ = [
  "DispatcherConstructionError",
  "GatewayDispatcherDeps",
  "InvocationPrincipal",
  "build_tool_dispatcher",
]
