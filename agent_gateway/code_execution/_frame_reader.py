"""Host frame reader: nested tool calls from a running `code_execute` cell.

A cell's bridged calls reach session tools through inherited pipes (subprocess)
or container stdio (Docker), not a socket or loopback endpoint
(execution-boundary design §8; clients plan §5.3/§5.5).
`FrameReader` is session-owned and rebound by each request bundle to its current
dispatcher. Each cell captures that binding and owns its background channel;
session disposal awaits every remaining registration. Each `code_execute` call
opens one `CellChannel`, registered under a random call-scoped token for the
outer `tool_call_id`. Backend activation begins only when the cell executes;
completion revokes it before output collection. The same token authenticates
native calls through `lookup_active_cell`, sharing the channel's grant, ordinal,
ledger and source settlement. A non-executing token answers `bad_cell_token`.

Frames, one JSON object per line. Child -> host, every frame carries
`call_id` (child correlation id) and `cell_token`:

- `hello{protocol}` -> `{protocol, parent_call_id, operations, tools}` (`tools`:
  the granted, code-callable tool names the client binds as `hank.<tool>`)
- `catalog{parent_call_id, request: "search", query}` -> list of `{name, description}`
- `catalog{parent_call_id, request: "describe", operation}` -> `{name, description, input_schema}`
- `derive{parent_call_id, name, value, expression, inputs}` -> `derive_result` (the published row)

A successful `tool_result` also carries `source_refs`, the issued keys of the
rows the call recorded; `derive` resolves them against the session's issued rows.

Host -> child replies echo `type` and `call_id` and carry `ok` plus `result` or
`error{code, message}`.

An operation is callable iff `operation ∈ owning tool_grant ∩ operations
registered as code-callable` (execution-boundary design §8): the grant is the
bound dispatcher's enforced catalog for the turn (`tool_definitions`) plus
the definitions the product binds as activatable this session
(`activatable_tool_definitions` — the set its `load_tools` authority would
admit, derived by that authority's owner, never a hand list here), the
code-callable set is the registration the product binds beside it. A call to
an activatable-but-not-yet-loaded operation first runs the bound
`activate_tool` (the product's `load_tools` activation/binding mechanism)
and then dispatches through the same prepared dispatcher path as any other
nested call, so approval and authority still apply. There is no
mutation-class filter here; the dispatcher's own policy and validation run on
`dispatch` as for any call. Every nested call is recorded in the channel's
ledger under the outer id (its `policy_class` is telemetry, read after the
decision); `summary()` is what the outer result carries. Nested payloads go
only to the child; the outer result's complete inline `sources` come from
the dispatcher's producer-owned settlement, supplemented by the product's
bound richer-handle extractor. They are deduplicated by citation-registry
identity (`source_key`, bound beside the extractor), each stamped with the
operation that produced it (`produced_by_tool`, canonical name).
"""
from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import os
import secrets
import time
import weakref
from typing import AbstractSet, Any, Awaitable, Callable, Dict, List, Mapping, Optional
from agent_workflow_contracts.models import AdmittedTenant

from ..policy_imports import load_server_policy_helpers, resolve_server_policy_tool_class
from ..sdk_runner_helpers import server_for_tool
from ..runner_background_tasks import jobs_wait
from ..task_registry import TaskState
from ..tool_dispatch_declarations import canonical_dispatch_tool_name

from ..capability_execution import (
  execute_leaf_model_call,
  list_model_call_operations,
  lookup_model_call_operation,
)
from ..tool_dispatch_classification import (
  _materialize_source_identity,
  classify_semantic_tool_error,
  resolve_dispatch_entry,
)
from ..tool_policy_registry import PreparedToolCall, ToolInputPreparationError
from ..tool_dispatcher_helpers import ToolExecutionContext
from ._frame_codec import CellBudget, MAX_FRAME_BYTES, ProtocolError, decode_frame, encode_frame
log = logging.getLogger(__name__)

BRIDGE_FD_IN_ENV = "AGENT_CODE_EXECUTE_BRIDGE_FD_IN"
BRIDGE_FD_OUT_ENV = "AGENT_CODE_EXECUTE_BRIDGE_FD_OUT"
CELL_TOKEN_ENV = "AGENT_CODE_EXECUTE_CELL_TOKEN"
BRIDGE_PROTOCOL = 1

# Product-owned `(operation, result) -> source handles` for one nested result,
# and the citation registry's identity for one handle (`sources` dedup key).
SourceExtractor = Callable[[str, Any], List[Dict[str, Any]]]
SourceKey = Callable[[Dict[str, Any]], Any]
CatalogMetadata = Callable[[str], Mapping[str, object]]
# Product-owned decision for one exact granted operation name.
CodeCallable = Callable[[str], bool]
# Product-owned definitions `load_tools` could activate for this session,
# and the activation of one such operation through that same mechanism.
ActivatableToolDefinitions = Callable[[], List[Dict[str, Any]]]
ActivateTool = Callable[[str], Awaitable[bool]]

# Catalog previews are presentation only; source identities are never capped.
_DESCRIPTION_PREVIEW_CHARS = 300

# The bridge operations this reader answers beyond `tool_call`/`catalog`. One
# registration: the cell's client generates `hank.<name>(...)` from the rows the
# `hello` reply carries, and the `code_execute` prompt names them from here. An
# op lowers to the same authority the direct door uses; it is not a tool and is
# never in a grant or an MCP policy set.
RUN_AGENT_OPERATION = "run_agent"
BRIDGE_OPERATIONS: tuple[Dict[str, Any], ...] = (
  {
    "name": "agent",
    "arguments": ["objective", "operation", "handles", "role", "effort", "ticker", "research_file_id"],
    "description": (
      "Submit a delegated job through the same admission as run_agent; "
      "returns its task_id at once. Wait on it with hank.wait."
    ),
  },
  {
    "name": "wait",
    "arguments": ["task_ids", "timeout"],
    "description": (
      "Block until the named jobs settle or timeout (seconds) elapses; "
      "returns each job's state and, once completed, its error and "
      "task_result_id (for agent jobs). A result reaches you as its background task "
      "notification after the cell returns, or by get_background_result(task_id=...) "
      "when notifications are off. Jobs keep running when the wait is interrupted."
    ),
  },
  {
    "name": "completion",
    "arguments": ["operation", "material", "role", "effort", "evidence_handles"],
    "description": (
      "Submit a leaf model completion under a registered model-call operation; "
      "returns its task_id at once. Wait on it with hank.wait."
    ),
  },
  {
    "name": "judge",
    "arguments": ["operation", "state", "questions", "role", "effort", "evidence_handles"],
    "description": (
      "Submit a leaf judgment under a registered judgment operation and questions; "
      "returns its task_id at once. Wait on it with hank.wait."
    ),
  },
  {
    "name": "workpool",
    "arguments": ["items", "operation", "mode", "role", "effort", "context_handles", "timeout"],
    "description": (
      "Submit a batch of items as one P4 controller job drawing from the per-runner six-slot queue; "
      "returns the controller task_id, submitted item keys, and item task_ids. Wait on it with hank.wait."
    ),
  },
  {
    "name": "judge_batch",
    "arguments": ["operation", "states", "questions", "role", "effort", "evidence_handles", "wait", "timeout"],
    "description": (
      "Apply questions across keyed states under a registered judgment operation; "
      "returns keyed judgments under the registered model-call operation. Wait on it with hank.wait when wait=false."
    ),
  },
)
# Every cell-to-host request a cell may send: the three protocol frames and
# one frame per registered bridge operation. A backend's frame reader routes
# exactly these to the bridge, so a new operation needs no reader edit.
REQUEST_FRAME_TYPES: frozenset[str] = frozenset(
  {"hello", "catalog", "tool_call", "derive", *(str(row["name"]) for row in BRIDGE_OPERATIONS)}
)
DEFAULT_WAIT_TIMEOUT_S = 600.0
# `jobs.wait` reads the executing runner's task registry; the product binds
# an accessor because the runner that owns a cell's jobs exists after the
# session's reader does.
JobRunner = Callable[[], Any]

_READERS: weakref.WeakSet[FrameReader] = weakref.WeakSet()


def lookup_active_cell(token: str) -> ActiveCell | None:
  """Resolve a live execution token against its session-owned registration."""
  for reader in _READERS:
    cell = reader._cells.get(token)
    if cell is not None and reader.is_registered(cell):
      return ActiveCell(cell)
  return None


class ActiveCell:
  """Native calls share the executing channel's authority and settlement."""

  def __init__(self, cell: CellChannel) -> None:
    self._cell = cell

  @property
  def outer_call_id(self) -> str:
    return self._cell.outer_call_id

  @property
  def principal(self) -> dict[str, Any]:
    return dict(self._cell._principal)

  def installed_routes(self) -> list[dict[str, Any]]:
    cell = self._cell
    return [
      cell._route_row(name, definition)
      for name, definition in cell._granted().items()
      if cell._code_callable is not None and cell._code_callable(name)
    ]

  def start_native(self, operation: str, tool_input: dict) -> dict:
    cell = self._cell
    if not cell._reader.is_registered(cell):
      raise ToolInputPreparationError({"code": "bad_cell_token", "message": "cell has ended"})
    dispatcher = cell._dispatcher
    if dispatcher is None:
      raise ToolInputPreparationError({"code": "bridge_unavailable", "message": f"{operation!r} is not code-callable here"})
    refusal = cell._refusal(dispatcher, operation)
    if refusal is not None:
      raise ToolInputPreparationError({"code": refusal, "message": f"{operation!r} is not code-callable here"})
    # The native producer has already executed these kwargs. Capture them as
    # the MCP transport does for settlement, without re-running preparation.
    prepared = PreparedToolCall(tool_input)
    entry = cell._new_entry(operation)
    entry.update(native=True, server=_server_of(operation), policy_class=_policy_class(operation))
    cell._native_calls[entry["nested_call_id"]] = (time.monotonic(), prepared, dispatcher)
    return entry

  def native_entry(self, nested_call_id: str) -> dict | None:
    """Return this cell's pending native ledger row, never a detached copy."""
    if nested_call_id not in self._cell._native_calls:
      return None
    return next(entry for entry in self._cell.ledger if entry["nested_call_id"] == nested_call_id)

  def settle_native(
    self, entry: dict, *, result: dict | None, error: dict | None,
    provenance: Mapping[str, Any] | None = None,
  ) -> None:
    cell = self._cell
    started, prepared, dispatcher = cell._native_calls.pop(entry["nested_call_id"])
    operation = entry["operation"]
    dispatch_entry = resolve_dispatch_entry(
      operation, origin=dispatcher.route_origin_for_tool(operation),
      server=entry["server"] or None, original_tool_name=canonical_dispatch_tool_name(operation),
    )
    semantic_error = classify_semantic_tool_error(result)
    settlement = dispatcher.settle_tool_result(
      operation, dispatch_entry, result, error, semantic_error,
      prepared_call=prepared,
    )
    failure = error if error is not None else semantic_error
    entry.update(
      duration_ms=_elapsed_ms(started), ok=failure is None,
      error_code=str(failure.get("code") or "tool_error") if failure is not None else None,
      result_bytes=len(json.dumps(result, separators=(",", ":"), default=str).encode()) if error is None else 0,
    )
    if provenance is not None:
      entry["provenance"] = dict(provenance)
    if error is None and settlement.outcome == "ok":
      entry["source_refs"] = cell._record_sources(operation, entry["nested_call_id"], result, settlement.sources)


from ._tool_defs import CODE_EXECUTION_TOOL_NAMES


class FrameReader:
  """Session-owned reader: turn bindings and executing cell registrations."""

  def __init__(
    self,
    *,
    principal: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    tenant: AdmittedTenant | None = None,
  ) -> None:
    self._dispatcher: Any | None = None
    self._tool_definitions: Callable[[], List[Dict[str, Any]]] | None = None
    self._activatable_tool_definitions: ActivatableToolDefinitions | None = None
    self._activate_tool: ActivateTool | None = None
    self._extract_sources: SourceExtractor | None = None
    self._source_key: SourceKey | None = None
    self._code_callable: CodeCallable | None = None
    self._catalog_metadata: CatalogMetadata | None = None
    self._job_runner: JobRunner | None = None
    self._host_tools: frozenset[str] = CODE_EXECUTION_TOOL_NAMES
    self._cells: Dict[str, CellChannel] = {}
    # Every source row this session's cells issued, by `source_refs` key. Kept
    # with the session reader, which the session's kernel cleanup releases with
    # it, so a retained kernel value resolves without a refetch.
    self._issued_sources: Dict[str, Dict[str, Any]] = {}
    self._closed = False
    self._closing_cells: tuple[CellChannel, ...] = ()
    self._principal = dict(principal or {})
    self.session_id = session_id or "session"
    self.tenant = tenant
    _READERS.add(self)

  def bind_job_runner(self, job_runner: JobRunner | None) -> None:
    """Bind the accessor for the runner whose task registry `wait` reads.

    Idempotent; the latest bind wins. `job_runner()` returning None means no
    runner is executing for this session yet, and `wait` answers
    `bridge_unavailable`.
    """
    self._job_runner = job_runner

  def bind_dispatcher(
    self,
    dispatcher: Any,
    *,
    tool_definitions: Callable[[], List[Dict[str, Any]]],
    extract_sources: SourceExtractor | None = None,
    source_key: SourceKey | None = None,
    code_callable: CodeCallable | None = None,
    catalog_metadata: CatalogMetadata | None = None,
    activatable_tool_definitions: ActivatableToolDefinitions | None = None,
    activate_tool: ActivateTool | None = None,
    host_tools: AbstractSet[str] | None = None,
  ) -> None:
    """Bind the dispatcher running this session's turn; the latest bind wins.

    `code_callable(operation)` is the product registration's decision for the
    exact installed route; a nested call needs it and the grant. Unbound means
    no operation is code-callable.
    `activatable_tool_definitions()` names the not-yet-loaded operations the
    session's `load_tools` authority would admit, and `activate_tool(name)`
    activates/binds one through that mechanism before dispatch; neither
    widens the grant past that authority.
    `source_key` is the citation registry's identity for an extracted handle;
    handles with equal keys are one source on the outer result.
    """
    self._dispatcher = dispatcher
    self._tool_definitions = tool_definitions
    self._activatable_tool_definitions = activatable_tool_definitions
    self._activate_tool = activate_tool
    self._extract_sources = extract_sources
    self._source_key = source_key
    self._code_callable = code_callable
    self._catalog_metadata = catalog_metadata
    self._host_tools = (
      frozenset(host_tools) if host_tools is not None else CODE_EXECUTION_TOOL_NAMES
    )
  def active_cells(self) -> tuple[str, ...]:
    """Outer `tool_call_id`s of the cells currently registered."""
    return tuple(cell.outer_call_id for cell in self._cells.values())

  def open_cell(
    self,
    outer_call_id: str,
    *,
    abort_event: asyncio.Event | None,
    active: bool,
    tool_ctx: ToolExecutionContext | None = None,
  ) -> "CellChannel":
    """Register one cell under a fresh token and create its fd pair."""
    if self._closed:
      raise RuntimeError("the session's code execution reader is disposed")
    cell = CellChannel(self, outer_call_id, abort_event=abort_event, tool_ctx=tool_ctx)
    cell._active = active
    self._cells[cell.token] = cell
    return cell

  async def aclose(self) -> None:
    """Kill the open cells' own background jobs, then revoke every registration.

    A background cell's job runs in the session's work directory, so it is
    killed through its registry and awaited before its cell closes.
    """
    if not self._closed:
      self._closed = True
      _READERS.discard(self)
      self._closing_cells = tuple(self._cells.values())
    job_tasks: List[asyncio.Task[Any]] = []
    for cell in self._closing_cells:
      if cell.job is None:
        continue
      runner, task_id = cell.job
      registry = runner._task_registry
      registry.kill(task_id, termination_intent="cancelled")
      entry = registry.get(task_id)
      if entry is not None and entry.asyncio_task is not None:
        job_tasks.append(entry.asyncio_task)
    await asyncio.gather(*job_tasks, return_exceptions=True)
    await asyncio.gather(*(cell.aclose() for cell in self._closing_cells))
    self._issued_sources.clear()

  def is_registered(self, cell: "CellChannel") -> bool:
    return self._cells.get(cell.token) is cell and cell._active and not self._closed

  def _registered(self, cell: "CellChannel", frame: Dict[str, Any]) -> bool:
    token = frame.get("cell_token")
    if not isinstance(token, str) or token != cell.token or not self.is_registered(cell):
      return False
    parent = frame.get("parent_call_id", cell.outer_call_id if frame.get("type") == "hello" else None)
    return isinstance(parent, str) and hmac.compare_digest(parent, cell.outer_call_id)

  def _release(self, cell: "CellChannel") -> None:
    if self._cells.get(cell.token) is cell:
      del self._cells[cell.token]


class CellChannel:
  """One cell's registration, fd pair, frame loop and nested-call ledger."""

  def __init__(
    self,
    reader: FrameReader,
    outer_call_id: str,
    *,
    abort_event: asyncio.Event | None,
    tool_ctx: ToolExecutionContext | None = None,
  ) -> None:
    self._reader = reader
    self.outer_call_id = outer_call_id
    self.token = secrets.token_urlsafe(32)
    self._abort_event = abort_event
    self._tool_ctx = tool_ctx
    self._active = True
    self._principal = dict(reader._principal)
    self.session_id = reader.session_id
    self.tenant = reader.tenant
    # `(runner, task_id)` of the background job this cell runs, set only by
    # the handler that launched it; the reader's close kills it.
    self.job: tuple[Any, str] | None = None
    self._native_calls: Dict[str, tuple[float, Any, Any]] = {}
    self._dispatcher = reader._dispatcher
    self._tool_definitions = reader._tool_definitions
    self._activatable_tool_definitions = reader._activatable_tool_definitions
    self._activate_tool = reader._activate_tool
    self._extract_sources = reader._extract_sources
    self._source_key = reader._source_key
    self._code_callable = reader._code_callable
    self._job_runner = reader._job_runner
    self._catalog_metadata = reader._catalog_metadata
    self._host_tools = reader._host_tools
    self._grant: Dict[str, Dict[str, Any]] | None = None
    self.budget = CellBudget()
    # host -> child, child -> host
    self._child_in, self._host_out = os.pipe()
    self._host_in, self._child_out = os.pipe()
    self._raw_fds: set[int] = {self._child_in, self._host_out, self._host_in, self._child_out}
    self._pipe_files: List[Any] = []
    self._read_transport: asyncio.ReadTransport | None = None
    self._write_transport: asyncio.WriteTransport | None = None
    self._ordinal = 0
    self.ledger: List[Dict[str, Any]] = []
    self._sources: Dict[Any, Dict[str, Any]] = {}
    self.derived: List[Dict[str, Any]] = []
    self._inflight: set[asyncio.Task[Any]] = set()
    self._serve_task: asyncio.Task[Any] | None = None

  # --- backend-facing (`_backends._base.CellBridge`) -------------------------

  def child_env(self) -> Dict[str, str]:
    env = {
      BRIDGE_FD_IN_ENV: str(self._child_in),
      BRIDGE_FD_OUT_ENV: str(self._child_out),
      CELL_TOKEN_ENV: self.token,
    }
    try:
      from ..investment_capability_claim import (
        investment_capability_signing_available,
        issue_risk_job_capability_claim,
      )
      if investment_capability_signing_available():
        granted_ops = [
          name for name in self._granted()
          if self._code_callable is not None and self._code_callable(name)
        ]
        if self.tenant is not None:
          claim = issue_risk_job_capability_claim(
            user_id=str(self._principal.get("user_id") or "user"),
            tenant=self.tenant,
            session_id=self.session_id,
            granted_operations=granted_ops,
            cell_token=self.token,
          )
          env["AGENT_JOB_CAPABILITY_CLAIM"] = claim
    except Exception:
      pass
    return env

  def child_fds(self) -> tuple[int, ...]:
    return (self._child_in, self._child_out)

  def fds(self) -> tuple[int, ...]:
    """All four pipe fds this channel created (host and child ends)."""
    return (self._child_in, self._host_out, self._host_in, self._child_out)

  def activate(self) -> None:
    """Begin execution only after the backend owns its execution slot."""
    self._active = True
    self._active = self._reader.is_registered(self)

  def deactivate(self) -> None:
    """Revoke authentication at execution end, retaining the result ledger."""
    self._active = False

  def close_child_ends(self) -> None:
    self._close_raw(self._child_in, self._child_out)

  def close(self) -> None:
    """End the registration and close every fd; idempotent."""
    self.deactivate()
    self._reader._release(self)
    # A pipe transport may already have closed itself (the child closed its
    # end); closing it again touches a detached loop.
    if self._read_transport is not None and not self._read_transport.is_closing():
      self._read_transport.close()
    self._read_transport = None
    if self._write_transport is not None and not self._write_transport.is_closing():
      self._write_transport.abort()
    self._write_transport = None
    # An aborted transport closes its pipe on the next loop turn; close it now
    # so the fds are gone when the cell's call returns.
    for pipe_file in self._pipe_files:
      pipe_file.close()
    self._pipe_files.clear()
    self._close_raw(*tuple(self._raw_fds))

  async def aclose(self) -> None:
    """Close this registration, cancel nested work, and await cancellation."""
    self.close()
    current = asyncio.current_task()
    tasks = set(self._inflight)
    if self._serve_task is not None:
      tasks.add(self._serve_task)
    if current is not None:
      tasks.discard(current)
    for task in tasks:
      task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)

  async def serve(self) -> None:
    """Answer frames until the child closes its end or this task is cancelled."""
    self._serve_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    stream = asyncio.StreamReader(limit=MAX_FRAME_BYTES)
    host_in = os.fdopen(self._host_in, "rb", buffering=0)
    self._raw_fds.discard(self._host_in)
    self._pipe_files.append(host_in)
    host_out = os.fdopen(self._host_out, "wb", buffering=0)
    self._raw_fds.discard(self._host_out)
    self._pipe_files.append(host_out)
    read_transport, _ = await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(stream), host_in)
    self._read_transport = read_transport
    write_transport, write_protocol = await loop.connect_write_pipe(
      lambda: asyncio.StreamReaderProtocol(asyncio.StreamReader()),
      host_out,
    )
    self._write_transport = write_transport
    writer = asyncio.StreamWriter(write_transport, write_protocol, None, loop)
    while True:
      try:
        line = await stream.readline()
      except ValueError:
        message = str(self.budget.fail("frame exceeds the 2 MiB bound"))
        writer.write(encode_frame(_error_reply(None, None, "protocol_error", message)))
        await writer.drain()
        return
      if not line:
        return
      try:
        writer.write(await self.handle_frame(line))
        await writer.drain()
      except (BrokenPipeError, ConnectionResetError):
        return

  def summary(self) -> Dict[str, Any]:
    """`nested_calls`, `sources` and `derived` for the outer result; empty when nothing was called or derived."""
    summary: Dict[str, Any] = {}
    if self.ledger:
      summary["nested_calls"] = [dict(entry) for entry in self.ledger]
      summary["sources"] = list(self._sources.values())
    if self.derived:
      summary["derived"] = [dict(row) for row in self.derived]
    return summary

  # --- frame handling --------------------------------------------------------

  def decode_frame(self, line: bytes) -> Dict[str, Any]:
    """Decode and charge one inbound frame, including kernel control frames."""
    return decode_frame(line, self.budget)

  async def handle_frame(self, line: bytes) -> bytes:
    """Decode one raw request and return its budgeted wire reply."""
    try:
      frame = self.decode_frame(line)
    except ProtocolError as exc:
      return self._protocol_error_wire({}, exc)
    return await self.handle_decoded_frame(frame)

  async def handle_decoded_frame(self, frame: Dict[str, Any]) -> bytes:
    """Answer a request already decoded and charged by this channel."""
    reply: Dict[str, Any] = {}
    task = asyncio.current_task()
    if task is not None:
      self._inflight.add(task)
    try:
      reply = await self._answer_frame(frame)
      wire = encode_frame(reply, self.budget)
      nested_id = reply.get("nested_call_id")
      if reply.get("ok") and nested_id is not None:
        for entry in self.ledger:
          if entry["nested_call_id"] == nested_id:
            envelope = encode_frame({**reply, "result": None})
            entry["result_bytes"] = len(wire) - len(envelope) + 4
            break
      return wire
    except ProtocolError as exc:
      nested_id = reply.get("nested_call_id")
      for entry in self.ledger:
        if entry["nested_call_id"] == nested_id:
          entry["ok"] = False
          entry["error_code"] = "protocol_error"
      return self._protocol_error_wire(frame, exc)
    except Exception as exc:
      log.exception("unexpected frame reader error: %s", exc)
      reply = _error_reply(frame.get("type"), frame.get("call_id"), "internal_error", str(exc))
      try:
        return encode_frame(reply, self.budget)
      except Exception:
        return self._protocol_error_wire(frame, ProtocolError(str(exc)))
    finally:
      if task is not None:
        self._inflight.discard(task)

  def _protocol_error_wire(self, frame: Dict[str, Any], exc: ProtocolError) -> bytes:
    reply = _error_reply(frame.get("type"), frame.get("call_id"), "protocol_error", str(exc))
    try:
      return encode_frame(reply, self.budget)
    except ProtocolError as bounds:
      # Untrusted correlation/type fields may themselves fill the frame.
      # A terminal protocol error has a fixed-size envelope.
      return encode_frame(_error_reply(None, None, "protocol_error", str(bounds)))

  async def _answer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    frame_type = frame.get("type")
    call_id = frame.get("call_id")
    if not self._reader._registered(self, frame):
      return _error_reply(frame_type, call_id, "bad_cell_token", "no live cell registration for this token")
    if frame_type == "hello":
      return _ok_reply(frame_type, call_id, {
        "protocol": BRIDGE_PROTOCOL,
        "parent_call_id": self.outer_call_id,
        "operations": [dict(row) for row in BRIDGE_OPERATIONS],
        "tools": self._granted_code_callable_names(),
      })
    if frame_type == "tool_call":
      return await self._tool_call(frame)
    if frame_type == "catalog":
      return self._catalog(frame)
    if frame_type == "agent":
      return await self._agent(frame)
    if frame_type == "wait":
      return await self._wait(frame)
    if frame_type == "completion":
      return await self._completion(frame)
    if frame_type == "judge":
      return await self._judge(frame)
    if frame_type == "workpool":
      return await self._workpool(frame)
    if frame_type == "judge_batch":
      return await self._judge_batch(frame)
    if frame_type == "derive":
      return self._derive(frame)
    return _error_reply(frame_type, call_id, "protocol_error", f"unknown frame type {frame_type!r}")

  def _granted_code_callable_names(self) -> List[str]:
    """Grant ∩ code-callable tool names, in grant order (hello's `tools`)."""
    if self._code_callable is None:
      return []
    return [name for name in self._granted() if self._code_callable(name)]

  def _loaded_granted(self) -> Dict[str, Dict[str, Any]]:
    """The turn's already-loaded grant by name, read live from the bound catalog."""
    tool_definitions = self._tool_definitions
    definitions = tool_definitions() if tool_definitions is not None else []
    return {
      name: definition
      for definition in definitions
      if (name := str(definition.get("name") or ""))
      and name not in self._host_tools
    }

  def _granted(self) -> Dict[str, Dict[str, Any]]:
    """Loaded grant plus the product's activatable definitions, once per cell."""
    if self._grant is None:
      grant = self._loaded_granted()
      if self._activatable_tool_definitions is not None:
        for definition in self._activatable_tool_definitions() or []:
          name = str(definition.get("name") or "")
          if name and name not in self._host_tools and name not in grant:
            grant[name] = definition
      self._grant = grant
    return self._grant

  async def _activate_if_deferred(self, operation: str) -> bool:
    """Activate one activatable operation through the bound mechanism.

    The product callback owns activation and reports loader failures unchanged.
    """
    if self._activate_tool is None:
      return False
    return await self._activate_tool(operation)

  def _route_row(self, name: str, definition: Dict[str, Any]) -> Dict[str, Any]:
    return {
      **(self._catalog_metadata(name) if self._catalog_metadata is not None else {}),
      "name": name,
      "description": str(definition.get("description") or ""),
      "input_schema": definition.get("input_schema") or {},
    }

  def _refusal(self, dispatcher: Any, operation: str) -> Optional[str]:
    """Error code, or None when `operation ∈ grant ∩ code-callable` (§8)."""
    if operation in self._host_tools:
      return "unknown_operation"
    if operation not in self._granted():
      if dispatcher.route_origin_for_tool(operation) is None:
        return "unknown_operation"
      return "not_granted"
    if self._code_callable is None or not self._code_callable(operation):
      return "not_code_callable"
    return None

  def _new_entry(self, operation: Any) -> Dict[str, Any]:
    self._ordinal += 1
    nested_call_id = f"{self.outer_call_id}.{self._ordinal}"
    entry: Dict[str, Any] = {
      "parent_call_id": self.outer_call_id,
      "nested_call_id": nested_call_id,
      "ordinal": self._ordinal,
      "operation": operation if isinstance(operation, str) else None,
      "server": "",
      "policy_class": "",
      "duration_ms": 0,
      "ok": False,
      "error_code": None,
      "result_bytes": 0,
    }
    self.ledger.append(entry)
    return entry

  async def _tool_call(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    call_id = frame.get("call_id")
    operation = frame.get("operation")
    tool_input = frame.get("input")
    entry = self._new_entry(operation)
    nested_call_id = entry["nested_call_id"]

    def _refuse(code: str, message: str) -> Dict[str, Any]:
      entry["error_code"] = code
      reply = _error_reply("tool_result", call_id, code, message)
      reply["nested_call_id"] = nested_call_id
      return reply

    if not isinstance(operation, str) or not isinstance(tool_input, dict):
      return _refuse("protocol_error", "tool_call needs a string operation and an object input")
    dispatcher = self._dispatcher
    if dispatcher is None:
      return _refuse("bridge_unavailable", "no dispatcher is bound to this session")
    refusal = self._refusal(dispatcher, operation)
    entry["server"] = _server_of(operation)
    entry["policy_class"] = _policy_class(operation)
    if refusal == "unknown_operation":
      return _refuse(refusal, f"{operation!r} is not a tool on this session's live surface")
    if refusal == "not_granted":
      return _refuse(refusal, f"{operation!r} is not granted to this turn")
    if refusal is not None:
      return _refuse(refusal, f"{operation!r} is not registered as code-callable")
    if operation not in self._loaded_granted():
      try:
        activated = await self._activate_if_deferred(operation)
      except ToolInputPreparationError as exc:
        detail = exc.materialize_error()
        return _refuse(str(detail.get("code") or "tool_error"), str(detail.get("message") or ""))
      if not activated:
        return _refuse("not_granted", f"{operation!r} is not granted to this turn")
    return await self._dispatch_granted("tool_result", call_id, entry, dispatcher, operation, tool_input)

  async def _agent(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    """`hank.agent`: the direct door, through the turn's dispatcher.

    The same `run_agent` handler admits it (`admit(SpawnRequest, ...)` from
    the executing job), so a cell gets the authority decision, binding and
    selection the model would for the same input. Only the grant is checked
    here, as for the model: the op is registered above, not code-callable.
    A cell's job is always background; `hank.wait` observes it.
    """
    call_id = frame.get("call_id")
    entry = self._new_entry(RUN_AGENT_OPERATION)
    nested_call_id = entry["nested_call_id"]

    def _refuse(code: str, message: str) -> Dict[str, Any]:
      entry["error_code"] = code
      reply = _error_reply("agent_result", call_id, code, message)
      reply["nested_call_id"] = nested_call_id
      return reply

    objective = frame.get("objective")
    if not isinstance(objective, str) or not objective.strip():
      return _refuse("protocol_error", "agent needs a non-empty string objective")
    dispatcher = self._dispatcher
    if dispatcher is None:
      return _refuse("bridge_unavailable", "no dispatcher is bound to this session")
    if RUN_AGENT_OPERATION not in self._granted():
      return _refuse("not_granted", f"{RUN_AGENT_OPERATION!r} is not granted to this turn")
    tool_input: Dict[str, Any] = {"objective": objective, "background": True}
    for key in ("operation", "handles", "role", "effort", "ticker", "research_file_id"):
      value = frame.get(key)
      if value is not None:
        tool_input[key] = value
    entry["server"] = _server_of(RUN_AGENT_OPERATION)
    entry["policy_class"] = _policy_class(RUN_AGENT_OPERATION)
    return await self._dispatch_granted(
      "agent_result", call_id, entry, dispatcher, RUN_AGENT_OPERATION, tool_input,
    )

  async def _wait(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    """`hank.wait`: `jobs.wait`, a registry read on the executing runner."""
    call_id = frame.get("call_id")
    task_ids = frame.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(item, str) for item in task_ids):
      return _error_reply("wait_result", call_id, "protocol_error", "wait needs a list of string task_ids")
    raw_timeout = frame.get("timeout")
    if raw_timeout is None:
      timeout = DEFAULT_WAIT_TIMEOUT_S
    elif isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool) and raw_timeout >= 0:
      timeout = float(raw_timeout)
    else:
      return _error_reply("wait_result", call_id, "protocol_error", "wait timeout must be a non-negative number of seconds")
    runner = self._job_runner() if self._job_runner is not None else None
    registry = getattr(runner, "_task_registry", None)
    if registry is None:
      return _error_reply("wait_result", call_id, "bridge_unavailable", "no job registry is bound to this session")
    result = await jobs_wait(
      task_ids,
      registry_lookup=registry.get,
      timeout=timeout,
      wait_fn=asyncio.wait,
    )
    return _ok_reply("wait_result", call_id, result)

  async def _completion(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    """`hank.completion`: leaf model call under a registered model-call operation."""
    call_id = frame.get("call_id")
    operation = frame.get("operation")
    if not isinstance(operation, str) or not operation.strip():
      return _error_reply("completion_result", call_id, "protocol_error", "completion needs a non-empty string operation")

    op_name = operation.strip()
    op = lookup_model_call_operation(op_name)
    if op is None or op.kind != "completion":
      registered = [o.name for o in list_model_call_operations(kind="completion")]
      return _error_reply(
        "completion_result",
        call_id,
        "unknown_operation",
        f"unknown completion operation {operation!r}; registered: {registered}",
      )

    material = frame.get("material")
    if material is not None and not isinstance(material, str):
      return _error_reply("completion_result", call_id, "protocol_error", "material must be a string")

    runner = self._job_runner() if self._job_runner is not None else None
    if runner is None:
      return _error_reply("completion_result", call_id, "bridge_unavailable", "no job runner is bound to this session")

    role = frame.get("role")
    effort = frame.get("effort")
    evidence_handles = frame.get("evidence_handles") or ()
    if not isinstance(evidence_handles, (list, tuple)):
      return _error_reply("completion_result", call_id, "protocol_error", "evidence_handles must be a list or tuple")

    started = time.monotonic()
    entry = self._new_entry(f"model_call:{op.name}")
    entry["server"] = "model_call"
    entry["policy_class"] = "read"

    try:
      launched = await execute_leaf_model_call(
        op.name,
        material=material or "",
        role=role,
        effort=effort,
        evidence_handles=evidence_handles,
        runner=runner,
        wait=False,
      )
    except Exception as exc:
      entry["duration_ms"] = _elapsed_ms(started)
      entry["error_code"] = "execution_error"
      reply = _error_reply("completion_result", call_id, "execution_error", str(exc))
      reply["nested_call_id"] = entry["nested_call_id"]
      return reply

    entry["duration_ms"] = _elapsed_ms(started)
    entry["ok"] = True
    entry["error_code"] = None
    reply = _ok_reply("completion_result", call_id, launched)
    reply["nested_call_id"] = entry["nested_call_id"]
    return reply

  async def _judge(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    """`hank.judge`: leaf judgment under a registered judgment operation."""
    call_id = frame.get("call_id")
    operation = frame.get("operation")
    if not isinstance(operation, str) or not operation.strip():
      return _error_reply("judge_result", call_id, "protocol_error", "judge needs a non-empty string operation")

    op_name = operation.strip()
    op = lookup_model_call_operation(op_name)
    if op is None or op.kind != "judgment":
      registered = [o.name for o in list_model_call_operations(kind="judgment")]
      return _error_reply(
        "judge_result",
        call_id,
        "unknown_operation",
        f"unknown judgment operation {operation!r}; registered: {registered}",
      )

    state = frame.get("state")
    questions = frame.get("questions")
    if questions is None:
      return _error_reply("judge_result", call_id, "protocol_error", "judge needs questions or rubric input")

    runner = self._job_runner() if self._job_runner is not None else None
    if runner is None:
      return _error_reply("judge_result", call_id, "bridge_unavailable", "no job runner is bound to this session")

    role = frame.get("role")
    effort = frame.get("effort")
    evidence_handles = frame.get("evidence_handles") or ()
    if not isinstance(evidence_handles, (list, tuple)):
      return _error_reply("judge_result", call_id, "protocol_error", "evidence_handles must be a list or tuple")

    started = time.monotonic()
    entry = self._new_entry(f"model_call:{op.name}")
    entry["server"] = "model_call"
    entry["policy_class"] = "read"

    try:
      launched = await execute_leaf_model_call(
        op.name,
        state=state,
        questions=questions,
        role=role,
        effort=effort,
        evidence_handles=evidence_handles,
        runner=runner,
        wait=False,
      )
    except Exception as exc:
      entry["duration_ms"] = _elapsed_ms(started)
      entry["error_code"] = "execution_error"
      reply = _error_reply("judge_result", call_id, "execution_error", str(exc))
      reply["nested_call_id"] = entry["nested_call_id"]
      return reply

    entry["duration_ms"] = _elapsed_ms(started)
    entry["ok"] = True
    entry["error_code"] = None
    reply = _ok_reply("judge_result", call_id, launched)
    reply["nested_call_id"] = entry["nested_call_id"]
    return reply

  async def _workpool(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    """`hank.workpool`: one P4 `controller` job drawing from the runner's queue.

    Submits a batch of keyed items as a controller job. Each item is admitted
    on its own through the runner's `run_agent` lowerer, drawing capacity from
    the per-runner child queue, and surviving cell or kernel loss.
    """
    started = time.monotonic()
    call_id = frame.get("call_id")
    entry = self._new_entry("workpool")
    nested_call_id = entry["nested_call_id"]
    entry["server"] = "gateway"
    entry["policy_class"] = "controller"

    def _refuse(code: str, message: str) -> Dict[str, Any]:
      entry["duration_ms"] = _elapsed_ms(started)
      entry["error_code"] = code
      reply = _error_reply("workpool_result", call_id, code, message)
      reply["nested_call_id"] = nested_call_id
      return reply

    raw_items = frame.get("items")
    if raw_items is None:
      return _refuse("protocol_error", "workpool needs items")
    if isinstance(raw_items, dict):
      items_seq: List[tuple[str, Dict[str, Any]]] = [
        (str(k), v if isinstance(v, dict) else {"objective": str(v)})
        for k, v in raw_items.items()
      ]
    elif isinstance(raw_items, (list, tuple)):
      items_seq = []
      for idx, item in enumerate(raw_items):
        if isinstance(item, dict):
          k = str(item.get("key") or item.get("item_key") or item.get("id") or item.get("name") or f"item_{idx}")
          items_seq.append((k, item))
        else:
          items_seq.append((f"item_{idx}", {"objective": str(item)}))
    else:
      return _refuse("protocol_error", "workpool items must be a list or dict")

    if not items_seq:
      return _refuse("protocol_error", "workpool items cannot be empty")

    seen_keys: set[str] = set()
    for k, _ in items_seq:
      if k in seen_keys:
        return _refuse("protocol_error", f"duplicate item key {k!r} in workpool items")
      seen_keys.add(k)

    dispatcher = self._dispatcher
    if dispatcher is None:
      return _refuse("bridge_unavailable", "no dispatcher is bound to this session")
    if RUN_AGENT_OPERATION not in self._granted():
      return _refuse("not_granted", f"{RUN_AGENT_OPERATION!r} is not granted to this turn")

    runner = self._job_runner() if self._job_runner is not None else None
    registry = getattr(runner, "_task_registry", None)
    if registry is None:
      return _refuse("bridge_unavailable", "no job registry is bound to this session")

    raw_timeout = frame.get("timeout")
    timeout = (
      float(raw_timeout)
      if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool) and raw_timeout >= 0
      else DEFAULT_WAIT_TIMEOUT_S
    )
    raw_context_handles = frame.get("context_handles") or ()
    if not isinstance(raw_context_handles, (list, tuple)) or not all(
      isinstance(h, str) for h in raw_context_handles
    ):
      return _refuse("protocol_error", "workpool context_handles must be a list of strings")
    context_handles = list(raw_context_handles)

    controller_task_id = f"wp_{secrets.token_hex(4)}"
    registry.register(task_type="controller", task_id=controller_task_id)
    registry.transition(controller_task_id, TaskState.RUNNING)

    append_durable = getattr(runner, "_append_durable_event", None)
    if append_durable is not None:
      try:
        await append_durable({
          "type": "task_registered",
          "event_schema_version": 2,
          "task_id": controller_task_id,
          "task_type": "controller",
          "started_at": time.time(),
          "metadata": {"keys": [k for k, _ in items_seq]},
        })
      except Exception:
        pass
    event_log = getattr(runner, "_log", None) or getattr(runner, "_event_log", None)
    if event_log is not None:
      event_log.append({
        "type": "task_registered",
        "event_schema_version": 2,
        "task_id": controller_task_id,
        "task_type": "controller",
        "started_at": time.time(),
        "metadata": {"keys": [k for k, _ in items_seq]},
      })

    child_jobs: Dict[str, Any] = {}
    dispatched_event = asyncio.Event()

    ctrl_ent = registry.get(controller_task_id)
    if ctrl_ent is not None:
      ctrl_ent.result = {
        "task_id": controller_task_id,
        "status": "running",
        "keys": [k for k, _ in items_seq],
        "items": child_jobs,
      }

    async def _run_workpool_controller() -> None:
      try:
        child_task_ids: List[str] = []
        for key, item_data in items_seq:
          item_input: Dict[str, Any] = {
            "objective": item_data.get("objective") or item_data.get("task") or str(item_data),
            "background": True,
          }
          if context_handles and not item_data.get("handles"):
            item_input["handles"] = list(context_handles)
          for opt in ("operation", "handles", "role", "effort", "ticker", "research_file_id"):
            val = item_data.get(opt) or frame.get(opt)
            if val is not None:
              item_input[opt] = val

          child_entry = self._new_entry(RUN_AGENT_OPERATION)
          child_entry["server"] = _server_of(RUN_AGENT_OPERATION)
          child_entry["policy_class"] = _policy_class(RUN_AGENT_OPERATION)
          child_reply = await self._dispatch_granted(
            "agent_result", None, child_entry, dispatcher, RUN_AGENT_OPERATION, item_input,
          )
          if child_reply.get("ok"):
            child_res = child_reply.get("result") or {}
            cid = child_res.get("task_id")
            child_jobs[key] = {"task_id": cid, "status": "running"}
            if cid:
              child_task_ids.append(cid)
          else:
            err = child_reply.get("error") or {}
            child_jobs[key] = {"status": "refused", "error": err.get("message") or err.get("code")}

          if ctrl_ent is not None:
            ctrl_ent.result = {
              "task_id": controller_task_id,
              "status": "running",
              "keys": [k for k, _ in items_seq],
              "items": dict(child_jobs),
            }

        dispatched_event.set()
        if child_task_ids:
          await jobs_wait(child_task_ids, registry_lookup=registry.get, timeout=timeout, wait_fn=asyncio.wait)
        settled_items: Dict[str, Any] = {}
        all_settled = True
        for k, c_info in child_jobs.items():
          cid = c_info.get("task_id")
          if cid:
            c_ent = registry.get(cid)
            if c_ent is not None:
              c_state = c_ent.state.value if hasattr(c_ent.state, "value") else str(c_ent.state)
              c_res = c_ent.result
              bounded_outcome = None
              if isinstance(c_res, dict):
                bounded_outcome = c_res.get("outcome") or c_res.get("disposition") or c_res.get("content") or c_res.get("summary")
              settled_items[k] = {
                "task_id": cid,
                "status": c_state,
                "outcome": bounded_outcome,
                "error": c_ent.error,
              }
              if c_ent.state not in {
                TaskState.COMPLETED,
                TaskState.FAILED,
                TaskState.KILLED,
                TaskState.INTERRUPTED,
                TaskState.PARKED,
              }:
                all_settled = False
            else:
              settled_items[k] = c_info
              all_settled = False
          else:
            settled_items[k] = c_info

        final_state = TaskState.COMPLETED if all_settled else TaskState.INTERRUPTED
        pool_status = "completed" if all_settled else "timed_out"
        completed_result = {
          "task_id": controller_task_id,
          "status": pool_status,
          "keys": [k for k, _ in items_seq],
          "items": settled_items,
        }
        registry.transition(
          controller_task_id,
          final_state,
          result=completed_result,
        )
        if append_durable is not None:
          try:
            await append_durable({
              "type": "task_completed",
              "event_schema_version": 2,
              "task_id": controller_task_id,
              "task_type": "controller",
              "state": final_state.value if hasattr(final_state, "value") else str(final_state),
              "final_state": final_state.value if hasattr(final_state, "value") else str(final_state),
              "completed_at": time.time(),
              "result": completed_result,
            })
          except Exception:
            pass
        if event_log is not None:
          try:
            event_log.append({
              "type": "task_completed",
              "event_schema_version": 2,
              "task_id": controller_task_id,
              "task_type": "controller",
              "state": final_state.value if hasattr(final_state, "value") else str(final_state),
              "final_state": final_state.value if hasattr(final_state, "value") else str(final_state),
              "completed_at": time.time(),
              "result": completed_result,
            })
          except Exception:
            pass
      except Exception as exc:
        log.exception("workpool controller %s execution failed", controller_task_id)
        try:
          registry.transition(
            controller_task_id,
            TaskState.FAILED,
            error={"code": "controller_error", "message": str(exc)},
          )
        except Exception:
          pass
        if append_durable is not None:
          try:
            await append_durable({
              "type": "task_completed",
              "event_schema_version": 2,
              "task_id": controller_task_id,
              "task_type": "controller",
              "state": "failed",
              "final_state": "failed",
              "completed_at": time.time(),
              "error": {"code": "controller_error", "message": str(exc)},
            })
          except Exception:
            pass
        if event_log is not None:
          try:
            event_log.append({
              "type": "task_completed",
              "event_schema_version": 2,
              "task_id": controller_task_id,
              "task_type": "controller",
              "state": "failed",
              "final_state": "failed",
              "completed_at": time.time(),
              "error": {"code": "controller_error", "message": str(exc)},
            })
          except Exception:
            pass
      finally:
        dispatched_event.set()
    if ctrl_ent is not None:
      ctrl_ent.asyncio_task = asyncio.create_task(_run_workpool_controller())

    await dispatched_event.wait()
    entry["duration_ms"] = _elapsed_ms(started)
    entry["ok"] = True
    entry["error_code"] = None
    result_payload = {
      "task_id": controller_task_id,
      "status": "running",
      "keys": [k for k, _ in items_seq],
      "jobs": child_jobs,
    }
    reply = _ok_reply("workpool_result", call_id, result_payload)
    reply["nested_call_id"] = nested_call_id
    return reply

  async def _judge_batch(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    """`hank.judge_batch`: applied questions across keyed states under a model-call op."""
    started = time.monotonic()
    call_id = frame.get("call_id")
    entry = self._new_entry("judge_batch")
    nested_call_id = entry["nested_call_id"]
    entry["server"] = "gateway"
    entry["policy_class"] = "model_call"

    def _refuse(code: str, message: str) -> Dict[str, Any]:
      entry["duration_ms"] = _elapsed_ms(started)
      entry["error_code"] = code
      reply = _error_reply("judge_batch_result", call_id, code, message)
      reply["nested_call_id"] = nested_call_id
      return reply

    operation = frame.get("operation")
    if not isinstance(operation, str) or not operation.strip():
      return _refuse("protocol_error", "judge_batch needs a non-empty string operation")

    op_name = operation.strip()
    op = lookup_model_call_operation(op_name)
    if op is None or op.kind != "judgment":
      registered = [o.name for o in list_model_call_operations(kind="judgment")]
      return _refuse(
        "unknown_operation",
        f"unknown judgment operation {operation!r}; registered: {registered}",
      )

    raw_states = frame.get("states")
    if raw_states is None:
      return _refuse("protocol_error", "judge_batch needs states")
    if isinstance(raw_states, dict):
      states_seq: List[tuple[str, Any]] = list(raw_states.items())
    elif isinstance(raw_states, (list, tuple)):
      states_seq = []
      for idx, item in enumerate(raw_states):
        if isinstance(item, (list, tuple)) and len(item) == 2:
          states_seq.append((str(item[0]), item[1]))
        elif isinstance(item, dict) and "key" in item:
          states_seq.append((str(item["key"]), item.get("state") or item))
        else:
          states_seq.append((f"item_{idx}", item))
    else:
      return _refuse("protocol_error", "judge_batch states must be a dict or list")

    if not states_seq:
      return _refuse("protocol_error", "judge_batch states cannot be empty")

    questions = frame.get("questions")
    if questions is None:
      return _refuse("protocol_error", "judge_batch needs questions or rubric input")

    evidence_handles = frame.get("evidence_handles") or ()
    if not isinstance(evidence_handles, (list, tuple)) or not all(
      isinstance(h, str) for h in evidence_handles
    ):
      return _refuse("protocol_error", "judge_batch evidence_handles must be a list of strings")

    runner = self._job_runner() if self._job_runner is not None else None
    registry = getattr(runner, "_task_registry", None)
    if registry is None:
      return _refuse("bridge_unavailable", "no job registry is bound to this session")

    role = frame.get("role")
    effort = frame.get("effort")
    wait_flag = frame.get("wait", True)
    if wait_flag is None:
      wait_flag = True

    task_id_by_key: Dict[str, str] = {}
    try:
      for key, st in states_seq:
        res = await execute_leaf_model_call(
          op_name,
          state=st,
          questions=questions,
          role=role,
          effort=effort,
          evidence_handles=evidence_handles,
          runner=runner,
          wait=False,
        )
        task_id_by_key[key] = res["task_id"]
    except Exception as exc:
      return _refuse("execution_error", str(exc))

    if wait_flag:
      raw_timeout = frame.get("timeout")
      timeout = (
        float(raw_timeout)
        if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool) and raw_timeout >= 0
        else DEFAULT_WAIT_TIMEOUT_S
      )
      await jobs_wait(list(task_id_by_key.values()), registry_lookup=registry.get, timeout=timeout, wait_fn=asyncio.wait)
      judgments: Dict[str, Any] = {}
      all_settled = True
      for key, tid in task_id_by_key.items():
        ent = registry.get(tid)
        if ent is not None:
          ent_state = ent.state.value if hasattr(ent.state, "value") else str(ent.state)
          j_row: Dict[str, Any] = {"status": ent_state}
          if ent.result:
            j_row.update(ent.result)
          if ent.error:
            j_row["error"] = ent.error
          judgments[key] = j_row
          if ent.state not in {
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.KILLED,
            TaskState.INTERRUPTED,
            TaskState.PARKED,
          }:
            all_settled = False
        else:
          judgments[key] = {"status": "unknown"}
          all_settled = False
      batch_status = "completed" if all_settled else "timed_out"
      entry["duration_ms"] = _elapsed_ms(started)
      entry["ok"] = all_settled
      entry["error_code"] = None if all_settled else "timed_out"
      reply = _ok_reply("judge_batch_result", call_id, {
        "operation": op_name,
        "status": batch_status,
        "judgments": judgments,
      })
      reply["nested_call_id"] = nested_call_id
      return reply
    else:
      entry["duration_ms"] = _elapsed_ms(started)
      entry["ok"] = True
      entry["error_code"] = None
      reply = _ok_reply("judge_batch_result", call_id, {
        "operation": op_name,
        "status": "running",
        "jobs": {k: {"task_id": tid, "status": "running"} for k, tid in task_id_by_key.items()},
      })
      reply["nested_call_id"] = nested_call_id
      return reply
  async def _dispatch_granted(
    self,
    reply_type: str,
    call_id: Any,
    entry: Dict[str, Any],
    dispatcher: Any,
    operation: str,
    tool_input: Dict[str, Any],
  ) -> Dict[str, Any]:
    """Dispatch one admitted nested call and settle it onto the ledger."""
    nested_call_id = entry["nested_call_id"]

    def _refuse(code: str, message: str) -> Dict[str, Any]:
      entry["error_code"] = code
      reply = _error_reply(reply_type, call_id, code, message)
      reply["nested_call_id"] = nested_call_id
      return reply

    started = time.monotonic()
    try:
      prepared_call = await dispatcher.prepare_tool_call_async(operation, tool_input)
      executed_call = prepared_call

      def capture_executed(call: Any) -> None:
        nonlocal executed_call
        executed_call = call

      result, error = await dispatcher.dispatch_prepared(
        nested_call_id,
        operation,
        prepared_call,
        abort_event=self._abort_event,
        skill_run_id=self._tool_ctx.skill_run_id if self._tool_ctx is not None else None,
        workspace_dir=self._tool_ctx.workspace_dir if self._tool_ctx is not None else None,
        batch_id=self._tool_ctx.batch_id if self._tool_ctx is not None else None,
        on_executed_prepared_call=capture_executed,
      )
      dispatch_entry = resolve_dispatch_entry(
        operation,
        origin=dispatcher.route_origin_for_tool(operation),
        server=entry["server"] or None,
        original_tool_name=canonical_dispatch_tool_name(operation),
      )
      semantic_error = classify_semantic_tool_error(result)
      settlement = dispatcher.settle_tool_result(
        operation,
        dispatch_entry,
        result,
        error,
        semantic_error,
        prepared_call=executed_call,
      )
    except asyncio.CancelledError:
      entry["duration_ms"] = _elapsed_ms(started)
      entry["error_code"] = "cancelled"
      raise
    except ToolInputPreparationError as exc:
      detail = exc.materialize_error()
      entry["duration_ms"] = _elapsed_ms(started)
      return _refuse(str(detail.get("code") or "tool_error"), str(detail.get("message") or ""))
    except Exception as exc:
      log.exception("code_execute nested dispatch failed for %s", nested_call_id)
      entry["duration_ms"] = _elapsed_ms(started)
      return _refuse("tool_error", f"{type(exc).__name__}: nested dispatch failed")
    entry["duration_ms"] = _elapsed_ms(started)
    if error is not None:
      code = str(error.get("code") or "tool_error") if isinstance(error, dict) else "tool_error"
      message = str(error.get("message") or "") if isinstance(error, dict) else str(error)
      return _refuse(code, message)
    # A producer error delivered as a result payload still reaches the cell
    # unchanged; the ledger records it as the failed call it is.
    entry["ok"] = semantic_error is None
    entry["error_code"] = str(semantic_error["code"]) if semantic_error is not None else None
    refs: List[str] | None = None
    if settlement.outcome == "ok":
      refs = entry["source_refs"] = self._record_sources(operation, nested_call_id, result, settlement.sources)
    reply = _ok_reply(reply_type, call_id, result)
    reply["nested_call_id"] = nested_call_id
    if refs is not None:
      reply["source_refs"] = refs
    return reply

  def _record_sources(self, operation: str, nested_call_id: str, result: Any, settled_sources: Any) -> List[str]:
    # Settlement owns identity; the extractor only enriches those rows with
    # citation presentation. It cannot introduce another source.
    identity_fields = ("document_id", "source_kind", "source_url", "provider", "endpoint", "key_fields")

    def identity_key(source: Mapping[str, Any]) -> str:
      return json.dumps(
        _materialize_source_identity({field: source.get(field) for field in identity_fields}),
        sort_keys=True, separators=(",", ":"),
      )

    identities = {
      identity_key(source): _materialize_source_identity(source)
      for source in settled_sources
    }
    presentations: List[tuple[str, Dict[str, Any]]] = []
    extract = self._extract_sources
    if extract is not None:
      try:
        for partial in extract(operation, result) or ():
          identity = identity_key(partial)
          presentations.append((identity, {
            field: value for field, value in partial.items() if field not in identity_fields
          }))
      except Exception:
        log.exception("code_execute source extraction failed for %s", nested_call_id)
    handles = [
      {**identities[identity], **presentation}
      for identity, presentation in presentations if identity in identities
    ]
    presented = {identity for identity, _presentation in presentations}
    handles.extend(source for identity, source in identities.items() if identity not in presented)
    refs: List[str] = []
    for handle in handles:
      try:
        key = (
          self._source_key(handle)
          if self._source_key is not None
          else json.dumps(handle, sort_keys=True, separators=(",", ":"), default=str)
        )
      except Exception:
        log.exception("code_execute source identity failed for %s", nested_call_id)
        continue
      # Provenance, not payload: the citation owner treats the handle as the
      # nested operation's partial, not `code_execute`'s.
      row = self._sources.setdefault(
        key, {**handle, "produced_by_tool": canonical_dispatch_tool_name(operation)},
      )
      ref = _issued_ref(key)
      self._reader._issued_sources.setdefault(ref, row)
      if ref not in refs:
        refs.append(ref)
    return refs

  def _derive(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    """`hank.derive`: publish a computed value with the inputs it was computed from.

    Each input's `source_refs` resolve against the rows this session issued;
    the published row carries those rows themselves. An input the session did
    not issue, or a literal that carries no reference, is an `input_unsourced`
    finding on the row. The row is never a source and never enters `sources`.
    """
    call_id = frame.get("call_id")
    name, expression, inputs = frame.get("name"), frame.get("expression"), frame.get("inputs")
    if not isinstance(name, str) or not isinstance(expression, str) or not isinstance(inputs, list):
      return _error_reply(
        "derive_result", call_id, "protocol_error",
        "derive needs a string name, a value, a string expression and a list of inputs",
      )
    issued = self._reader._issued_sources
    resolved: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    for index, item in enumerate(inputs):
      reference = item if isinstance(item, dict) else {"value": item}
      raw_refs = reference.get("source_refs")
      refs = list(dict.fromkeys(ref for ref in raw_refs if isinstance(ref, str))) if isinstance(raw_refs, list) else []
      sources = [dict(issued[ref]) for ref in refs if ref in issued]
      row = {
        "nested_call_id": reference.get("nested_call_id"),
        "path": reference.get("path"),
        "value": reference.get("value"),
        "sources": sources,
      }
      resolved.append(row)
      if not refs:
        reason = "the input carries no source reference; select it from a producer result with hank.input"
      elif len(sources) != len(refs):
        reason = "the input's source references were not issued by this session"
      else:
        continue
      findings.append({
        "code": "input_unsourced", "input": index,
        "nested_call_id": row["nested_call_id"], "path": row["path"], "message": reason,
      })
    published = {
      "name": name, "value": frame.get("value"), "expression": expression,
      "status": "derived", "inputs": resolved, "findings": findings,
    }
    self.derived.append(published)
    return _ok_reply("derive_result", call_id, published)

  def _catalog(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    call_id = frame.get("call_id")
    dispatcher = self._dispatcher
    if dispatcher is None or self._tool_definitions is None:
      return _error_reply("catalog", call_id, "bridge_unavailable", "no dispatcher is bound to this session")
    request = frame.get("request")
    if request == "search":
      query = str(frame.get("query") or "").lower()
      rows = []
      for name, definition in self._granted().items():
        description = str(definition.get("description") or "")
        if query not in name.lower() and query not in description.lower():
          continue
        if self._code_callable is not None and self._code_callable(name):
          metadata = self._catalog_metadata(name) if self._catalog_metadata is not None else {}
          rows.append({**metadata, "name": name, "description": description[:_DESCRIPTION_PREVIEW_CHARS]})
      return _ok_reply("catalog", call_id, rows)
    if request == "describe":
      operation = frame.get("operation")
      if not isinstance(operation, str):
        return _error_reply("catalog", call_id, "protocol_error", "describe needs a string operation")
      # A registered bridge op describes itself; it is called as
      # `hank.<name>(...)`, never through `hank.call`, so `search` (the
      # grant ∩ code-callable listing) does not carry it.
      for row in BRIDGE_OPERATIONS:
        if row["name"] == operation:
          return _ok_reply("catalog", call_id, {"route_kind": "bridge", **row})
      op = lookup_model_call_operation(operation)
      if op is not None:
        return _ok_reply("catalog", call_id, {
          "route_kind": "model_call",
          "name": op.name,
          "kind": op.kind,
          "execution_class": op.execution_class,
          "description": op.description,
          "rubric_schema": dict(op.rubric_schema) if op.rubric_schema else None,
        })
      refusal = self._refusal(dispatcher, operation)
      if refusal is not None:
        return _error_reply("catalog", call_id, refusal, f"{operation!r} is not code-callable here")
      definition = self._granted()[operation]
      return _ok_reply("catalog", call_id, self._route_row(operation, definition))
    return _error_reply("catalog", call_id, "protocol_error", f"unknown catalog request {request!r}")

  def _close_raw(self, *fds: int) -> None:
    for fd in fds:
      if fd in self._raw_fds:
        self._raw_fds.discard(fd)
        try:
          os.close(fd)
        except OSError:
          pass


def _server_of(operation: str) -> str:
  runtime_server = server_for_tool(operation)
  if runtime_server:
    return runtime_server
  _forbidden, get_server_for_policy_tool, _tool_class = load_server_policy_helpers()
  if get_server_for_policy_tool is None:
    return ""
  return str(get_server_for_policy_tool(canonical_dispatch_tool_name(operation)) or "")


def _policy_class(operation: str) -> str:
  """The policy owner's class for the ledger; telemetry, never the decision."""
  return resolve_server_policy_tool_class(
    operation,
    policy_tool_name=canonical_dispatch_tool_name(operation),
    runtime_server=server_for_tool(operation),
    default="",
  )


def _elapsed_ms(started: float) -> int:
  return int((time.monotonic() - started) * 1000)


def _ok_reply(frame_type: Any, call_id: Any, result: Any) -> Dict[str, Any]:
  return {"type": frame_type, "call_id": call_id, "ok": True, "result": result}


def _issued_ref(key: Any) -> str:
  """The opaque, stable key a source's registry identity is issued under."""
  identity = json.dumps(key, sort_keys=True, separators=(",", ":"), default=str)
  return "src_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _error_reply(frame_type: Any, call_id: Any, code: str, message: str) -> Dict[str, Any]:
  return {"type": frame_type, "call_id": call_id, "ok": False, "error": {"code": code, "message": message}}
