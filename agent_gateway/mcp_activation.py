"""Load state is a fold (T3-I12).

Which MCP servers a session may route, which of their tools are still deferred
behind ``load_tools``, and which tools the dispatcher will actually admit were
three *mutable views* — ``loaded_mcp_servers``, ``deferred_mcp_tools`` and
``allowed_mcp_tools_by_server`` — each with its own writers and its own rule.
Nothing kept them agreed.  A pack could be reported loaded (its tools removed
from the deferred set, its server added to the loaded set, so the model sees the
tools) while the dispatcher allowlist never learned the grant, and the very next
call was rejected: the pack-loaded-but-rejected desync.

This module replaces the three views with one append-only fact and two pure
derivations:

``McpActivationFold``
    The single writer.  One ``record`` per activation, in order, never removed.
    It is the in-memory projection of the durable ``mcp_server_activated``
    session-log events, so a replay of the log rebuilds exactly the live state
    (:func:`fold_mcp_activations`).

``derive_live_surface``
    The whole prompt-facing surface — active servers, deferred tool names,
    deferred tool ids, and the per-server allowlist — computed together from
    ``(profile, channel tier, server catalog, fold, denied)``.  Because the
    three projections come out of one function over one fact, they cannot
    disagree: a tool is advertised exactly when it is allowed.

``derive_dispatcher_allowlist``
    Lives in :mod:`agent_gateway.capability_resolution` and accepts either a
    frozen ``ResolvedAuthority`` (the delegation path) or a
    :class:`LiveToolSurface` (the interactive path), so the dispatcher's scope
    has one derivation for both.

The durable event is **session-log-only** (D-B7-1): it is deliberately absent
from ``event_adapter.V1_WIRE_EVENT_TYPES`` and ``V1_FIELD_PROJECTION``, so no
client contract changes and an older binary replaying a newer log simply ignores
an event type it does not know — degrade, not corrupt.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence, Set as AbstractSet
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

MCP_SERVER_ACTIVATED_EVENT = "mcp_server_activated"

# Activation sources, recorded for provenance only. The fold's arithmetic never
# branches on them: one activation is one activation.
ACTIVATION_SOURCE_LOAD_TOOLS = "load_tools"
ACTIVATION_SOURCE_RUN_AGENT = "run_agent"
ACTIVATION_SOURCE_STAGE_SCOPE = "stage_tool_scope"


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
  """One durable activation fact.

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
    a no-op.  Callers use this to keep the durable log to the activations that
    actually moved the surface.
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


def mcp_server_activated_event(
  *,
  server_id: str,
  tools: Iterable[Any] | None = None,
  whole_server: bool = False,
  source: str = "",
  agent_name: str | None = None,
  profile_name: str | None = None,
  error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
  """Build the durable ``mcp_server_activated`` payload.

  A refused activation is recorded too — with ``error`` and no tools — so the
  log says why a declared server never became routable instead of leaving a
  silent gap.  The fold ignores errored records by construction: they are built
  through :func:`fold_mcp_activations`'s error branch, never appended.
  """

  payload: dict[str, Any] = {
    "type": MCP_SERVER_ACTIVATED_EVENT,
    "server_id": _exact_server_id(server_id),
  }
  if agent_name:
    payload["agent_name"] = str(agent_name)
  if profile_name:
    payload["profile_name"] = str(profile_name)
  if source:
    payload["source"] = str(source)
  if error is not None:
    payload["error"] = {
      "code": str(error.get("code") or ""),
      "message": str(error.get("message") or ""),
    }
    return payload
  payload["tools"] = sorted(_exact_tool_names(tools))
  payload["whole_server"] = bool(whole_server) or tools is None
  return payload


def fold_mcp_activations(events: Iterable[Mapping[str, Any]]) -> McpActivationFold:
  """Rebuild the live fold from durable ``mcp_server_activated`` events."""

  fold = McpActivationFold()
  for event in events:
    if not isinstance(event, Mapping):
      continue
    if str(event.get("type") or "") != MCP_SERVER_ACTIVATED_EVENT:
      continue
    if event.get("error") is not None:
      continue
    server_id = str(event.get("server_id") or "").strip()
    if not server_id:
      continue
    raw_tools = event.get("tools")
    tools = raw_tools if isinstance(raw_tools, Sequence) and not isinstance(raw_tools, str) else ()
    fold.record(
      server_id,
      tools=tools,
      whole_server=bool(event.get("whole_server")),
      source=str(event.get("source") or ""),
    )
  return fold


@dataclass(frozen=True)
class LiveToolSurface:
  """The whole derived MCP surface for one session, from one fold."""

  active_servers: frozenset[str]
  server_catalog: dict[str, Any]
  deferred_mcp_tools: frozenset[str]
  deferred_mcp_tool_ids: frozenset[str]
  allowed_mcp_tools_by_server: dict[str, frozenset[str]]


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
  channel_context: str | None,
  channel_tiers: Mapping[str | None, Mapping[str, Any]],
  server_catalog: Mapping[str, Any],
  activation_fold: McpActivationFold | None,
  denied_mcp_servers: Iterable[str] | None = None,
  stage_tool_scope: Any | None = None,
) -> LiveToolSurface:
  """Derive the live MCP surface from the activation fold.

  ``profile=None`` means "no profile ceiling in force": every active server is
  fully advertised and nothing is deferred, which is what a scope-less caller
  used to get by passing ``deferred_mcp_tools=None``.

  A ``stage_tool_scope`` replaces the profile ceiling outright: the stage's
  exact per-server map is the allowlist, and only the stage's own servers
  contribute deferred names.
  """

  if stage_tool_scope is not None:
    return _stage_scoped_surface(
      stage_tool_scope=stage_tool_scope,
      channel_tiers=channel_tiers,
      channel_context=channel_context,
      server_catalog=server_catalog,
      activation_fold=activation_fold,
      denied_mcp_servers=denied_mcp_servers,
      profile=profile,
    )

  tier = channel_tiers.get(channel_context, channel_tiers[None])
  denied_servers = set(
    denied_mcp_servers
    if denied_mcp_servers is not None
    else getattr(profile, "denied_mcp_servers", frozenset()) or frozenset()
  )
  fold = activation_fold if activation_fold is not None else McpActivationFold()
  active_servers = (
    set(tier.get("always", set())) | set(fold.activated_servers)
  ) - denied_servers
  catalog = {
    str(server_name): payload
    for server_name, payload in server_catalog.items()
    if str(server_name) not in denied_servers
  }

  unscoped_active_servers = {
    str(server_name)
    for server_name in getattr(profile, "unscoped_active_mcp_servers", frozenset()) or frozenset()
    if isinstance(server_name, str)
  }
  core_tools_by_server: dict[str, set[str]] = {
    str(server_name): {
      str(tool_name)
      for tool_name in tool_names
      if isinstance(tool_name, str)
    }
    for server_name, tool_names in (getattr(profile, "core_mcp_tools", {}) or {}).items()
  }

  withheld_names: set[str] = set()
  deferred_tool_pairs: set[tuple[str, str]] = set()
  allowed_mcp_tools_by_server: dict[str, frozenset[str]] = {}
  active_ungated_tool_names: set[str] = set()
  whole_servers = fold.whole_servers

  for server_name in catalog:
    server_tool_names = _catalog_tool_names(catalog.get(server_name, {}))
    if server_name in active_servers:
      if profile is None or server_name in unscoped_active_servers:
        allowed_mcp_tools_by_server[server_name] = frozenset(server_tool_names)
        active_ungated_tool_names.update(server_tool_names)
        continue
      granted = set(core_tools_by_server.get(server_name, set()))
      if server_name in whole_servers:
        granted |= server_tool_names
      granted |= set(fold.granted_tools(server_name))
      allowed = server_tool_names & granted
      allowed_mcp_tools_by_server[server_name] = frozenset(allowed)
      server_deferred_tools = server_tool_names - allowed
    else:
      server_deferred_tools = server_tool_names

    withheld_names |= server_deferred_tools
    for tool_name in server_deferred_tools:
      deferred_tool_pairs.add((server_name, tool_name))

  # An explicitly granted or ungated tool name is visible everywhere it is
  # advertised. Both subtractions preserve the pre-fold behaviour exactly: the
  # deferred set was always a flat name set, and `load_tools` removed loaded
  # names from it globally.
  visible_names = active_ungated_tool_names | set(fold.granted_tool_names)
  withheld_names -= visible_names
  return LiveToolSurface(
    active_servers=frozenset(active_servers),
    server_catalog=catalog,
    deferred_mcp_tools=frozenset(withheld_names),
    deferred_mcp_tool_ids=frozenset(
      f"mcp__{server_name}__{tool_name}"
      for server_name, tool_name in deferred_tool_pairs
    ),
    allowed_mcp_tools_by_server=allowed_mcp_tools_by_server,
  )


def _stage_scoped_surface(
  *,
  stage_tool_scope: Any,
  channel_tiers: Mapping[str | None, Mapping[str, Any]],
  channel_context: str | None,
  server_catalog: Mapping[str, Any],
  activation_fold: McpActivationFold | None,
  denied_mcp_servers: Iterable[str] | None,
  profile: Any | None,
) -> LiveToolSurface:
  """The stage route's exact scope, in the shape every reader already takes."""

  tier = channel_tiers.get(channel_context, channel_tiers[None])
  denied_servers = set(
    denied_mcp_servers
    if denied_mcp_servers is not None
    else getattr(profile, "denied_mcp_servers", frozenset()) or frozenset()
  )
  fold = activation_fold if activation_fold is not None else McpActivationFold()
  stage_servers = {
    str(server_name)
    for server_name in getattr(stage_tool_scope, "mcp_server_names", ()) or ()
  }
  allowed_mcp_tools_by_server = {
    str(server_name): frozenset(str(tool_name) for tool_name in tool_names)
    for server_name, tool_names in (
      getattr(stage_tool_scope, "mcp_tools_by_server", {}) or {}
    ).items()
  }
  catalog = {
    str(server_name): payload
    for server_name, payload in server_catalog.items()
    if str(server_name) not in denied_servers
  }
  withheld_names: set[str] = set()
  deferred_tool_ids: set[str] = set()
  for server_name in stage_servers:
    allowed = allowed_mcp_tools_by_server.get(server_name, frozenset())
    for tool_name in sorted(_catalog_tool_names(catalog.get(server_name, {}))):
      if tool_name in allowed:
        continue
      withheld_names |= {tool_name}
      deferred_tool_ids |= {f"mcp__{server_name}__{tool_name}"}
  active_servers = (
    set(tier.get("always", set())) | set(fold.activated_servers) | stage_servers
  ) - denied_servers
  return LiveToolSurface(
    active_servers=frozenset(active_servers),
    server_catalog=catalog,
    deferred_mcp_tools=frozenset(withheld_names),
    deferred_mcp_tool_ids=frozenset(deferred_tool_ids),
    allowed_mcp_tools_by_server=allowed_mcp_tools_by_server,
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
  def _from_iterable(cls, iterable: Iterable[str]) -> set[str]:
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
    "_channel_context",
    "_channel_tiers",
    "_mcp_client_manager",
    "_denied_mcp_servers",
    "_stage_tool_scope",
    "_defers_mcp_tools",
    "_fold",
    "_session_log",
  )

  def __init__(
    self,
    *,
    session: Any,
    profile: Any | None,
    channel_context: str | None,
    channel_tiers: Mapping[str | None, Mapping[str, Any]],
    mcp_client_manager: Any,
    denied_mcp_servers: Iterable[str] | None = None,
    stage_tool_scope: Any | None = None,
    defers_mcp_tools: bool = True,
    activation_fold: McpActivationFold | None = None,
    session_log: Any | None = None,
  ) -> None:
    self._session = session
    self._profile = profile
    self._channel_context = channel_context
    self._channel_tiers = channel_tiers
    self._mcp_client_manager = mcp_client_manager
    self._denied_mcp_servers = (
      None if denied_mcp_servers is None else set(denied_mcp_servers)
    )
    self._stage_tool_scope = stage_tool_scope
    self._defers_mcp_tools = bool(defers_mcp_tools)
    self._fold = activation_fold
    self._session_log = session_log

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
    """Append one activation, durably where a session log is bound.

    This is the only write on the interactive path: the surface, the deferred
    set and the dispatcher allowlist all move together because they are all
    read back out of this one record.

    Known asymmetry: the durable event is appended here, at record time, while
    `LoadToolsSDKTransactionManager` may `take_since` the record back out of
    the live fold and only re-`extend` it on commit. A discarded SDK load
    therefore leaves an `mcp_server_activated` event behind that the live fold
    no longer carries. This is latent — `fold_mcp_activations` has no
    production caller, so nothing rehydrates a fold from the log today — but
    the append must move behind the commit before any replay reader lands, or
    the log stops agreeing with the fold.
    """

    fold = self.activation_fold
    if fold.covers(server_id, tools=tools, whole_server=whole_server):
      return McpServerActivation(
        server_id=_exact_server_id(server_id),
        tools=_exact_tool_names(tools),
        whole_server=bool(whole_server) or tools is None,
        source=str(source or ""),
      )
    activation = fold.record(
      server_id,
      tools=tools,
      whole_server=whole_server,
      source=source,
    )
    append_sync = getattr(self._session_log, "append_sync", None)
    if callable(append_sync):
      append_sync(
        mcp_server_activated_event(
          server_id=activation.server_id,
          tools=sorted(activation.tools),
          whole_server=activation.whole_server,
          source=activation.source,
        )
      )
    return activation

  def _server_catalog(self) -> Mapping[str, Any]:
    get_server_catalog = getattr(self._mcp_client_manager, "get_server_catalog", None)
    if not callable(get_server_catalog):
      return {}
    catalog = get_server_catalog()
    return catalog if isinstance(catalog, Mapping) else {}

  def surface(self) -> LiveToolSurface:
    return derive_live_surface(
      profile=self._profile,
      channel_context=self._channel_context,
      channel_tiers=self._channel_tiers,
      server_catalog=self._server_catalog(),
      activation_fold=self.activation_fold,
      denied_mcp_servers=self._denied_mcp_servers,
      stage_tool_scope=self._stage_tool_scope,
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


class _EmptyMcpClientManager:
  """No catalog, because a detached surface has no live MCP client."""

  @staticmethod
  def get_server_catalog() -> dict[str, Any]:
    return {}


_DETACHED_CHANNEL_TIERS: Mapping[str | None, Mapping[str, Any]] = MappingProxyType({
  None: MappingProxyType({"always": frozenset(), "defer": frozenset()}),
})


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
    channel_context=None,
    channel_tiers=_DETACHED_CHANNEL_TIERS,
    mcp_client_manager=_EmptyMcpClientManager(),
    denied_mcp_servers=(),
    defers_mcp_tools=defers_mcp_tools,
    activation_fold=fold,
  )


def coerce_tool_surface(value: Any) -> Any:
  """Accept a surface, or a plain collection of already-activated servers.

  A surface is anything that answers the four questions the handler asks —
  ``activation_fold``, ``deferred_mcp_tools``, ``defers_mcp_tools`` and
  ``record_activation`` — so the autonomous runtime can supply its own
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
    )
  ):
    return value
  raise TypeError(
    "tool surface must expose an activation fold or be a collection of server names"
  )


__all__ = [
  "ACTIVATION_SOURCE_LOAD_TOOLS",
  "ACTIVATION_SOURCE_RUN_AGENT",
  "ACTIVATION_SOURCE_STAGE_SCOPE",
  "DerivedToolNames",
  "DerivedToolScope",
  "LiveToolSurface",
  "MCP_SERVER_ACTIVATED_EVENT",
  "McpActivationError",
  "McpActivationFold",
  "McpServerActivation",
  "SessionToolSurface",
  "coerce_tool_surface",
  "derive_live_surface",
  "detached_tool_surface",
  "fold_mcp_activations",
  "live_tool_surface",
  "mcp_server_activated_event",
]
