"""Load state is a fold (T3-I12).

Which MCP servers a session may route, which of their tools are still deferred
behind ``load_tools``, and which tools the dispatcher will actually admit were
three *mutable views* — ``loaded_mcp_servers``, ``deferred_mcp_tools`` and
``allowed_mcp_tools_by_server`` — each with its own writers and its own rule.
Nothing kept them agreed.  A pack could be reported loaded (its tools removed
from the deferred set, its server added to the loaded set, so the model sees the
tools) while the dispatcher allowlist never learned the grant, and the very next
call was rejected: the pack-loaded-but-rejected desync.

This module replaces the three views with one session-owned fact and two pure
derivations:

``McpActivationFold``
    The single writer.  One ``record`` per activation, in order, never removed.
    The live session and its runners share this exact instance.

``derive_live_surface``
    The whole prompt-facing surface — active servers, deferred tool names,
    deferred tool ids, and the per-server allowlist — computed together from
    ``(profile, server catalog, fold, denied)``.  Because the three
    projections come out of one function over one fact, they cannot
    disagree: a tool is advertised exactly when it is allowed.  No channel
    tier withholds a server; only a profile's registered tool ceiling does.

``LiveToolSurface``
    The immutable result. Interactive prompt, catalog, and dispatcher readers
    consume its exact per-server map; delegation projects its separate frozen
    ``ResolvedAuthority`` without converting between authority shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, TypeVar, TypedDict

_SetElementT = TypeVar("_SetElementT")


# Activation sources, recorded for provenance only. The fold's arithmetic never
# branches on them: one activation is one activation.
ACTIVATION_SOURCE_LOAD_TOOLS = "load_tools"
ACTIVATION_SOURCE_RUN_AGENT = "run_agent"

RunAgentMcpActivationErrorCode = Literal[
  "mcp_server_denied",
  "mcp_server_unavailable",
  "mcp_tool_unavailable",
]


class RunAgentMcpActivationError(TypedDict):
  """Visible refusal returned by a named-operation MCP activator."""

  code: RunAgentMcpActivationErrorCode
  message: str


class McpActivationError(ValueError):
  """An activation record was not an exact server/tool fact."""


def _exact_server_id(server_id: Any) -> str:
  text = str(server_id or "").strip()
  if not text:
    raise McpActivationError("mcp activation requires a non-empty server id")
  return text


def _exact_tool_names(tools: Any) -> frozenset[str]:
  if tools is None:
    return frozenset()
  if isinstance(tools, str):
    raise McpActivationError("mcp activation tools must be a collection, not a string")
  names: set[str] = set()
  for tool in tools:
    text = str(tool or "").strip()
    if text:
      names.add(text)
  return frozenset(names)


@dataclass(frozen=True)
class McpServerActivation:
  """One session activation fact.

  ``whole_server`` records an activation that granted the server's entire
  advertised surface (the ``load_tools(servers=[...])`` and undeclared-tools
  ``run_agent`` shapes).  A scoped activation names its tools exactly.
  """

  server_id: str
  tools: frozenset[str]
  whole_server: bool
  source: str

  def __post_init__(self) -> None:
    if not self.server_id:
      raise McpActivationError("mcp activation requires a non-empty server id")


class McpActivationFold:
  """Append-only MCP load state for one session.

  There is no removal and no in-place edit: activation is monotone within a
  session, which is what makes "advertised" and "allowed" derivable from the
  same fact instead of maintained in parallel.
  """

  __slots__ = ("_records",)

  def __init__(self, records: Iterable[McpServerActivation] = ()) -> None:
    self._records: list[McpServerActivation] = []
    for record in records:
      if not isinstance(record, McpServerActivation):
        raise McpActivationError("fold records must be McpServerActivation")
      self._records.append(record)

  def record(
    self,
    server_id: Any,
    *,
    tools: Iterable[Any] | None = None,
    whole_server: bool = False,
    source: str = "",
  ) -> McpServerActivation:
    """Append one activation and return it."""

    activation = McpServerActivation(
      server_id=_exact_server_id(server_id),
      tools=_exact_tool_names(tools),
      whole_server=bool(whole_server) or tools is None,
      source=str(source or ""),
    )
    self._records.append(activation)
    return activation

  @property
  def records(self) -> tuple[McpServerActivation, ...]:
    return tuple(self._records)

  @property
  def activated_servers(self) -> frozenset[str]:
    return frozenset(record.server_id for record in self._records)

  @property
  def whole_servers(self) -> frozenset[str]:
    return frozenset(
      record.server_id for record in self._records if record.whole_server
    )

  def granted_tools(self, server_id: str) -> frozenset[str]:
    """Return the tools explicitly granted for one server (empty if whole)."""

    wanted = str(server_id or "").strip()
    granted: set[str] = set()
    for record in self._records:
      if record.server_id == wanted:
        granted.update(record.tools)
    return frozenset(granted)

  @property
  def granted_tool_names(self) -> frozenset[str]:
    """Every explicitly granted tool name, across servers."""

    names: set[str] = set()
    for record in self._records:
      names.update(record.tools)
    return frozenset(names)

  def covers(
    self,
    server_id: Any,
    *,
    tools: Iterable[Any] | None = None,
    whole_server: bool = False,
  ) -> bool:
    """Whether this activation would change nothing.

    Activation is monotone, so a repeat of a grant the fold already carries is
    a no-op. Callers use this to avoid redundant records.
    """

    wanted = str(server_id or "").strip()
    if wanted not in self.activated_servers:
      return False
    if (bool(whole_server) or tools is None) and wanted not in self.whole_servers:
      return False
    return _exact_tool_names(tools) <= self.granted_tools(wanted) | (
      frozenset() if wanted not in self.whole_servers else _exact_tool_names(tools)
    )

  # --- staged writes ----------------------------------------------------
  # The SDK runner must not leave load state behind when the query rebuild
  # that would advertise the new tools fails.  It runs the handler, takes the
  # new records back, and re-appends them verbatim only on commit.  This is
  # the *only* sanctioned removal, it happens under the manager's lock before
  # any reader runs, and it keeps the fold the single writer instead of
  # snapshotting three collections and restoring them by hand.

  def checkpoint(self) -> int:
    """Return a mark that :meth:`take_since` can roll back to."""

    return len(self._records)

  def take_since(self, mark: int) -> tuple[McpServerActivation, ...]:
    """Remove and return every record appended after ``mark``."""

    if mark < 0 or mark > len(self._records):
      raise McpActivationError("activation checkpoint is out of range")
    taken = tuple(self._records[mark:])
    del self._records[mark:]
    return taken

  def extend(self, records: Iterable[McpServerActivation]) -> None:
    """Re-append previously taken records, in order."""

    for record in records:
      if not isinstance(record, McpServerActivation):
        raise McpActivationError("fold records must be McpServerActivation")
      self._records.append(record)

  def __len__(self) -> int:
    return len(self._records)

  def __bool__(self) -> bool:
    return bool(self._records)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, McpActivationFold):
      return NotImplemented
    return self._records == other._records

  def __repr__(self) -> str:  # pragma: no cover - diagnostics only
    return f"McpActivationFold({self._records!r})"


def _freeze_surface_value(value: Any) -> Any:
  if isinstance(value, Mapping):
    return MappingProxyType({
      key: _freeze_surface_value(item)
      for key, item in value.items()
    })
  if isinstance(value, (list, tuple)):
    return tuple(_freeze_surface_value(item) for item in value)
  if isinstance(value, (set, frozenset)):
    return frozenset(_freeze_surface_value(item) for item in value)
  return value


@dataclass(frozen=True, slots=True)
class LiveToolSurface:
  """The whole derived MCP surface for one session, from one fold."""

  active_servers: frozenset[str]
  server_catalog: Mapping[str, Any]
  deferred_mcp_tools: frozenset[str]
  deferred_mcp_tool_ids: frozenset[str]
  allowed_mcp_tools_by_server: Mapping[str, frozenset[str]]

  def __post_init__(self) -> None:
    object.__setattr__(self, "active_servers", frozenset(self.active_servers))
    object.__setattr__(
      self,
      "server_catalog",
      _freeze_surface_value(self.server_catalog),
    )
    object.__setattr__(
      self,
      "deferred_mcp_tools",
      frozenset(self.deferred_mcp_tools),
    )
    object.__setattr__(
      self,
      "deferred_mcp_tool_ids",
      frozenset(self.deferred_mcp_tool_ids),
    )
    object.__setattr__(
      self,
      "allowed_mcp_tools_by_server",
      MappingProxyType({
        str(server_name): frozenset(tool_names)
        for server_name, tool_names in self.allowed_mcp_tools_by_server.items()
      }),
    )


def live_tool_surface(
  *,
  active_servers: Iterable[str] = (),
  deferred_mcp_tools: Iterable[str] = (),
  deferred_mcp_tool_ids: Iterable[str] = (),
  allowed_mcp_tools_by_server: Mapping[str, Iterable[str]] | None = None,
  server_catalog: Mapping[str, Any] | None = None,
) -> LiveToolSurface:
  """State a derived surface directly, for callers with their own rule.

  The autonomous and preflight paths compute their deferred population from a
  different ceiling than the interactive one; they still hand every reader one
  surface value rather than several collections.
  """

  return LiveToolSurface(
    active_servers=frozenset(str(name) for name in active_servers),
    server_catalog=dict(server_catalog or {}),
    deferred_mcp_tools=frozenset(str(name) for name in deferred_mcp_tools),
    deferred_mcp_tool_ids=frozenset(str(name) for name in deferred_mcp_tool_ids),
    allowed_mcp_tools_by_server={
      str(server_name): frozenset(str(tool) for tool in tool_names)
      for server_name, tool_names in (allowed_mcp_tools_by_server or {}).items()
    },
  )


def _catalog_tool_names(payload: Any) -> set[str]:
  tools = payload.get("tools", []) if isinstance(payload, Mapping) else []
  return {
    str(tool)
    for tool in tools
    if isinstance(tool, str)
  }


def derive_live_surface(
  *,
  profile: Any | None,
  server_catalog: Mapping[str, Any],
  activation_fold: McpActivationFold | None,
  denied_mcp_servers: Iterable[str] | None = None,
) -> LiveToolSurface:
  """Derive the live MCP surface from the profile's registration and the fold.

  Every connected server the profile is not denied is active; no channel
  tier decides which servers a session sees. The denied set is the
  profile's own registration plus the operator-only audience
  (``api/agent/shared/mcp_audience.py``), an authorization partition.

  A server advertises its whole tool list unless the profile registers a
  default-visible set for it in ``core_mcp_tools`` (and does not mark it
  unscoped); the fold's grants widen that set, which is what ``load_tools``
  is for. Separately, a profile may register a tool **ceiling** for a server
  in ``mcp_tool_ceilings``: what that profile may ever reach there. The
  ceiling bounds every route that writes the fold — ``load_tools``, a
  ``run_agent`` declared activation, stage and node admission — because they
  all read this one derivation; no grant and no whole-server activation
  passes it. The community profile's invite users never see the owner's
  portfolio reads or shared-corpus writes this way, and no loader has to be
  withheld to keep it so.
  """

  denied_servers = _denied_servers(profile, denied_mcp_servers)
  fold = activation_fold if activation_fold is not None else McpActivationFold()
  default_servers = {
    str(server_name)
    for server_name in (getattr(profile, "mcp_default_mcp_servers", frozenset()) or frozenset())
  }
  catalog = {
    str(server_name): payload
    for server_name, payload in server_catalog.items()
    if str(server_name) not in denied_servers
  }
  unscoped_servers = {
    str(server_name)
    for server_name in getattr(profile, "unscoped_active_mcp_servers", frozenset()) or frozenset()
  }
  registered_defaults: dict[str, set[str]] = {
    str(server_name): {str(tool_name) for tool_name in tool_names}
    for server_name, tool_names in (getattr(profile, "core_mcp_tools", {}) or {}).items()
    if str(server_name) not in unscoped_servers
  }
  registered_ceilings: dict[str, set[str]] = {
    str(server_name): {str(tool_name) for tool_name in tool_names}
    for server_name, tool_names in (getattr(profile, "mcp_tool_ceilings", {}) or {}).items()
  }

  # A server the profile defers starts outside the active set and lists its
  # tools as loadable until the fold's first grant activates it. Deferral is
  # what a cold session is shown, never what it may reach: the allowed scope
  # below is the authorization partition and dispatch scope
  # (server_policies._profile_mcp_scope_denies), and is the same deferred or
  # not.
  cold_servers = default_servers - set(fold.activated_servers)
  withheld_names: set[str] = set()
  withheld_tool_pairs: set[tuple[str, str]] = set()
  allowed_mcp_tools_by_server: dict[str, frozenset[str]] = {}
  visible_names: set[str] = set(fold.granted_tool_names)
  for server_name, payload in catalog.items():
    server_tool_names = _catalog_tool_names(payload)
    default_visible = registered_defaults.get(server_name)
    if default_visible is None or server_name in fold.whole_servers:
      allowed = set(server_tool_names)
    else:
      allowed = server_tool_names & (default_visible | set(fold.granted_tools(server_name)))
    ceiling = registered_ceilings.get(server_name)
    if ceiling is not None:
      allowed &= ceiling
    allowed_mcp_tools_by_server[server_name] = frozenset(allowed)
    withheld = server_tool_names - allowed
    if server_name in cold_servers:
      withheld = set(server_tool_names)
    else:
      visible_names |= allowed
    for tool_name in withheld:
      withheld_names.add(tool_name)
      withheld_tool_pairs.add((server_name, tool_name))

  # The withheld set is a flat name set: a name allowed on any visible server
  # is visible everywhere it is advertised.
  return LiveToolSurface(
    active_servers=frozenset(
      (set(catalog) | set(fold.activated_servers)) - denied_servers - cold_servers
    ),
    server_catalog=catalog,
    deferred_mcp_tools=frozenset(withheld_names - visible_names),
    deferred_mcp_tool_ids=frozenset(
      f"mcp__{server_name}__{tool_name}"
      for server_name, tool_name in withheld_tool_pairs
    ),
    allowed_mcp_tools_by_server=allowed_mcp_tools_by_server,
  )


def _denied_servers(
  profile: Any | None,
  denied_mcp_servers: Iterable[str] | None,
) -> set[str]:
  return set(
    denied_mcp_servers
    if denied_mcp_servers is not None
    else getattr(profile, "denied_mcp_servers", frozenset()) or frozenset()
  )


class DerivedToolNames(AbstractSet):
  """A read-only set recomputed from the fold on every read.

  It wears the ``Set`` interface every prompt and catalog reader already uses
  (``in``, ``|``, ``-``, ``set(...)``, iteration) and deliberately has no
  ``update``, ``difference_update`` or ``clear``: a second writer is a
  ``AttributeError``, not a silent desync.
  """

  __slots__ = ("_compute",)

  def __init__(self, compute: Callable[[], AbstractSet]) -> None:
    self._compute = compute

  @classmethod
  def _from_iterable(
    cls,
    iterable: Iterable[_SetElementT],
  ) -> set[_SetElementT]:
    # Set algebra over a derived view produces an ordinary set, not another
    # view: the result is a value, and values do not re-derive.
    return set(iterable)

  def _value(self) -> frozenset[str]:
    return frozenset(self._compute())

  def __contains__(self, item: object) -> bool:
    return item in self._value()

  def __iter__(self):
    return iter(sorted(self._value()))

  def __len__(self) -> int:
    return len(self._value())

  def __eq__(self, other: object) -> bool:
    if isinstance(other, (set, frozenset, DerivedToolNames, AbstractSet)):
      return set(self._value()) == set(other)
    return NotImplemented

  def __hash__(self) -> int:  # pragma: no cover - equality is by value
    return hash(self._value())

  def __repr__(self) -> str:  # pragma: no cover - diagnostics only
    return f"DerivedToolNames({sorted(self._value())!r})"


class DerivedToolScope(Mapping):
  """A read-only per-server tool scope recomputed from the fold on every read."""

  __slots__ = ("_compute",)

  def __init__(self, compute: Callable[[], Mapping[str, AbstractSet]]) -> None:
    self._compute = compute

  def _value(self) -> Mapping[str, frozenset[str]]:
    return {
      str(server_name): frozenset(tool_names)
      for server_name, tool_names in self._compute().items()
    }

  def __getitem__(self, key: str) -> frozenset[str]:
    return self._value()[key]

  def __iter__(self):
    return iter(self._value())

  def __len__(self) -> int:
    return len(self._value())

  def __repr__(self) -> str:  # pragma: no cover - diagnostics only
    return f"DerivedToolScope({dict(self._value())!r})"


class SessionToolSurface:
  """The live MCP surface of one session, derived on every read.

  Everything the interactive runtime used to keep in three mutable collections
  is a method call on this object.  It owns no state beyond the inputs of the
  derivation; the only state is the session's activation fold.
  """

  __slots__ = (
    "_session",
    "_profile",
    "_mcp_client_manager",
    "_denied_mcp_servers",
    "_defers_mcp_tools",
    "_fold",
  )

  def __init__(
    self,
    *,
    session: Any,
    profile: Any | None,
    mcp_client_manager: Any,
    denied_mcp_servers: Iterable[str] | None = None,
    defers_mcp_tools: bool = True,
    activation_fold: McpActivationFold | None = None,
  ) -> None:
    self._session = session
    self._profile = profile
    self._mcp_client_manager = mcp_client_manager
    self._denied_mcp_servers = (
      None if denied_mcp_servers is None else set(denied_mcp_servers)
    )
    self._defers_mcp_tools = bool(defers_mcp_tools)
    self._fold = activation_fold

  @property
  def defers_mcp_tools(self) -> bool:
    """Whether this surface holds tools back behind ``load_tools``."""

    return self._defers_mcp_tools

  @property
  def activation_fold(self) -> McpActivationFold:
    if self._fold is not None:
      return self._fold
    fold = getattr(self._session, "mcp_activation_fold", None)
    if isinstance(fold, McpActivationFold):
      return fold
    self._fold = McpActivationFold()
    return self._fold

  def record_activation(
    self,
    server_id: str,
    *,
    tools: Iterable[Any] | None = None,
    whole_server: bool = False,
    source: str = "",
  ) -> McpServerActivation:
    """Append one activation to the shared session fold.

    This is the only write on the interactive path: the surface, the deferred
    set and the dispatcher allowlist all move together because they are all
    read back out of this one record.
    """

    fold = self.activation_fold
    if fold.covers(server_id, tools=tools, whole_server=whole_server):
      return McpServerActivation(
        server_id=_exact_server_id(server_id),
        tools=_exact_tool_names(tools),
        whole_server=bool(whole_server) or tools is None,
        source=str(source or ""),
      )
    return fold.record(
      server_id,
      tools=tools,
      whole_server=whole_server,
      source=source,
    )

  def _server_catalog(self) -> Mapping[str, Any]:
    get_server_catalog = getattr(self._mcp_client_manager, "get_server_catalog", None)
    if not callable(get_server_catalog):
      return {}
    catalog = get_server_catalog()
    return catalog if isinstance(catalog, Mapping) else {}

  def surface(self) -> LiveToolSurface:
    return derive_live_surface(
      profile=self._profile,
      server_catalog=self._server_catalog(),
      activation_fold=self.activation_fold,
      denied_mcp_servers=self._denied_mcp_servers,
    )

  @property
  def active_servers(self) -> DerivedToolNames:
    return DerivedToolNames(lambda: self.surface().active_servers)

  @property
  def deferred_mcp_tools(self) -> DerivedToolNames:
    return DerivedToolNames(lambda: self.surface().deferred_mcp_tools)

  @property
  def deferred_mcp_tool_ids(self) -> DerivedToolNames:
    return DerivedToolNames(lambda: self.surface().deferred_mcp_tool_ids)

  @property
  def allowed_mcp_tools_by_server(self) -> DerivedToolScope:
    return DerivedToolScope(lambda: self.surface().allowed_mcp_tools_by_server)

  @property
  def known_mcp_tools_by_server(self) -> DerivedToolScope:
    """Connected catalog routes, independent of session activation."""

    return DerivedToolScope(lambda: {
      server_name: _catalog_tool_names(payload)
      for server_name, payload in self.surface().server_catalog.items()
    })


class _EmptyMcpClientManager:
  """No catalog, because a detached surface has no live MCP client."""

  @staticmethod
  def get_server_catalog() -> dict[str, Any]:
    return {}


def detached_tool_surface(
  activated_servers: Iterable[str] = (),
  *,
  defers_mcp_tools: bool = False,
) -> SessionToolSurface:
  """A surface with no session and no durable log behind it.

  Callers that build handlers outside a live session (tests, one-shot tool
  assembly) still need somewhere for an activation to land.  It lands in a
  fold nobody else reads, which is the point: there is no second writer of
  anyone's load state.
  """

  fold = McpActivationFold()
  for server_name in sorted(
    {str(name).strip() for name in activated_servers if str(name or "").strip()}
  ):
    fold.record(server_name, tools=(), source="detached")
  return SessionToolSurface(
    session=None,
    profile=None,
    mcp_client_manager=_EmptyMcpClientManager(),
    denied_mcp_servers=(),
    defers_mcp_tools=defers_mcp_tools,
    activation_fold=fold,
  )


def coerce_tool_surface(value: Any) -> Any:
  """Accept a surface, or a plain collection of already-activated servers.

  A surface is anything that answers the five questions the handler asks —
  ``activation_fold``, ``deferred_mcp_tools``, ``defers_mcp_tools``,
  ``record_activation`` and ``allowed_mcp_tools_by_server`` (what a load
  reached) — so the autonomous runtime can supply its own
  derivation over the same fold.  ``defers_mcp_tools`` is checked here because
  ``_make_load_tools_handler`` reads it before anything else, to decide whether
  pack loads take the deferral branch at all; a surface missing it would fail
  at handler-build time instead of here.
  """

  if value is None:
    return detached_tool_surface()
  if isinstance(value, (set, frozenset, list, tuple)):
    return detached_tool_surface(value)
  if all(
    hasattr(value, attribute)
    for attribute in (
      "activation_fold",
      "deferred_mcp_tools",
      "defers_mcp_tools",
      "record_activation",
      "allowed_mcp_tools_by_server",
    )
  ):
    return value
  raise TypeError(
    "tool surface must expose an activation fold or be a collection of server names"
  )


__all__ = [
  "ACTIVATION_SOURCE_LOAD_TOOLS",
  "ACTIVATION_SOURCE_RUN_AGENT",
  "DerivedToolNames",
  "DerivedToolScope",
  "LiveToolSurface",
  "McpActivationError",
  "McpActivationFold",
  "McpServerActivation",
  "RunAgentMcpActivationError",
  "RunAgentMcpActivationErrorCode",
  "SessionToolSurface",
  "coerce_tool_surface",
  "derive_live_surface",
  "detached_tool_surface",
  "live_tool_surface",
]
