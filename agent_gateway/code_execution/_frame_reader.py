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

- `hello{protocol}` -> `{protocol, parent_call_id}`
- `tool_call{parent_call_id, operation, input}` -> `tool_result{nested_call_id}`
- `catalog{parent_call_id, request: "search", query}` -> list of `{name, description}`
- `catalog{parent_call_id, request: "describe", operation}` -> `{name, description, input_schema}`

Host -> child replies echo `type` and `call_id` and carry `ok` plus `result` or
`error{code, message}`.

An operation is callable iff `operation ∈ owning tool_grant ∩ operations
registered as code-callable` (execution-boundary design §8): the grant is the
bound dispatcher's enforced catalog for the turn (`tool_definitions`), the
code-callable set is the registration the product binds beside it. There is no
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
import json
import logging
import os
import secrets
import time
import weakref
from typing import Any, Callable, Dict, List, Mapping, Optional

from ..policy_imports import load_server_policy_helpers, resolve_server_policy_tool_class
from ..runner_tool_execution import _AsyncToolCallPreparer
from ..sdk_runner_helpers import server_for_tool
from ..tool_dispatch_declarations import canonical_dispatch_tool_name

from ..tool_dispatch_classification import (
  _materialize_source_identity,
  classify_semantic_tool_error,
  resolve_dispatch_entry,
)
from ..tool_policy_registry import PreparedToolCall, ToolInputPreparationError
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

# Catalog previews are presentation only; source identities are never capped.
_DESCRIPTION_PREVIEW_CHARS = 300

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
      cell._record_sources(operation, entry["nested_call_id"], result, settlement.sources)


class FrameReader:
  """Session-owned reader: turn bindings and executing cell registrations."""

  def __init__(self, *, principal: Mapping[str, Any] | None = None) -> None:
    self._dispatcher: Any | None = None
    self._tool_definitions: Callable[[], List[Dict[str, Any]]] | None = None
    self._extract_sources: SourceExtractor | None = None
    self._source_key: SourceKey | None = None
    self._code_callable: CodeCallable | None = None
    self._catalog_metadata: CatalogMetadata | None = None
    self._cells: Dict[str, CellChannel] = {}
    self._closed = False
    self._closing_cells: tuple[CellChannel, ...] = ()
    self._principal = dict(principal or {})
    _READERS.add(self)

  def bind_dispatcher(
    self,
    dispatcher: Any,
    *,
    tool_definitions: Callable[[], List[Dict[str, Any]]],
    extract_sources: SourceExtractor | None = None,
    source_key: SourceKey | None = None,
    code_callable: CodeCallable | None = None,
    catalog_metadata: CatalogMetadata | None = None,
  ) -> None:
    """Bind the dispatcher running this session's turn; the latest bind wins.

    `code_callable(operation)` is the product registration's decision for the
    exact installed route; a nested call needs it and the grant. Unbound means
    no operation is code-callable.
    `source_key` is the citation registry's identity for an extracted handle;
    handles with equal keys are one source on the outer result.
    """
    self._dispatcher = dispatcher
    self._tool_definitions = tool_definitions
    self._extract_sources = extract_sources
    self._source_key = source_key
    self._code_callable = code_callable
    self._catalog_metadata = catalog_metadata

  def active_cells(self) -> tuple[str, ...]:
    """Outer `tool_call_id`s of the cells currently registered."""
    return tuple(cell.outer_call_id for cell in self._cells.values())

  def open_cell(self, outer_call_id: str, *, abort_event: asyncio.Event | None, active: bool) -> "CellChannel":
    """Register one cell under a fresh token and create its fd pair."""
    if self._closed:
      raise RuntimeError("the session's code execution reader is disposed")
    cell = CellChannel(self, outer_call_id, abort_event=abort_event)
    cell._active = active
    self._cells[cell.token] = cell
    return cell

  async def aclose(self) -> None:
    """Revoke every registration and await its nested dispatch/pipe tasks."""
    if not self._closed:
      self._closed = True
      _READERS.discard(self)
      self._closing_cells = tuple(self._cells.values())
    await asyncio.gather(*(cell.aclose() for cell in self._closing_cells))

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

  def __init__(self, reader: FrameReader, outer_call_id: str, *, abort_event: asyncio.Event | None) -> None:
    self._reader = reader
    self.outer_call_id = outer_call_id
    self.token = secrets.token_urlsafe(32)
    self._abort_event = abort_event
    self._active = True
    self._principal = dict(reader._principal)
    self._native_calls: Dict[str, tuple[float, Any, Any]] = {}
    self._dispatcher = reader._dispatcher
    self._tool_definitions = reader._tool_definitions
    self._extract_sources = reader._extract_sources
    self._source_key = reader._source_key
    self._code_callable = reader._code_callable
    self._catalog_metadata = reader._catalog_metadata
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
    self._inflight: set[asyncio.Task[Any]] = set()
    self._serve_task: asyncio.Task[Any] | None = None

  # --- backend-facing (`_backends._base.CellBridge`) -------------------------

  def child_env(self) -> Dict[str, str]:
    return {
      BRIDGE_FD_IN_ENV: str(self._child_in),
      BRIDGE_FD_OUT_ENV: str(self._child_out),
      CELL_TOKEN_ENV: self.token,
    }

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
    """`nested_calls` and `sources` for the outer result; empty when no frame called."""
    if not self.ledger:
      return {}
    return {
      "nested_calls": [dict(entry) for entry in self.ledger],
      "sources": list(self._sources.values()),
    }

  # --- frame handling --------------------------------------------------------

  async def handle_frame(self, line: bytes, *, accounted: bool = False) -> bytes:
    """Return already-accounted wire bytes; transports must not charge twice."""
    return (await self._answer_wire(line, accounted=accounted))[1]

  async def answer_frame(self, line: bytes, *, accounted: bool = False) -> Dict[str, Any]:
    """Validate both directions, including direct (non-transport) callers.

    `accounted` means the transport already decoded this inbound frame against
    this same cell budget. It never bypasses the individual frame bound.
    """
    return (await self._answer_wire(line, accounted=accounted))[0]

  async def _answer_wire(self, line: bytes, *, accounted: bool) -> tuple[Dict[str, Any], bytes]:
    frame: Dict[str, Any] = {}
    reply: Dict[str, Any] = {}
    task = asyncio.current_task()
    if task is not None:
      self._inflight.add(task)
    try:
      frame = decode_frame(line, None if accounted else self.budget)
      if self.budget.error is not None:
        raise ProtocolError(self.budget.error)
      reply = await self._answer_frame(frame)
      wire = encode_frame(reply, self.budget)
      nested_id = reply.get("nested_call_id")
      if reply.get("ok") and nested_id is not None:
        for entry in self.ledger:
          if entry["nested_call_id"] == nested_id:
            envelope = encode_frame({**reply, "result": None})
            entry["result_bytes"] = len(wire) - len(envelope) + 4
            break
      return reply, wire
    except ProtocolError as exc:
      if len(line) > MAX_FRAME_BYTES:
        self.budget.fail(str(exc))
      nested_id = reply.get("nested_call_id")
      for entry in self.ledger:
        if entry["nested_call_id"] == nested_id:
          entry["ok"] = False
          entry["error_code"] = "protocol_error"
      reply = _error_reply(frame.get("type"), frame.get("call_id"), "protocol_error", str(exc))
      try:
        wire = encode_frame(reply, self.budget)
      except ProtocolError as bounds:
        # Untrusted correlation/type fields may themselves fill the frame.
        # A terminal protocol error has a fixed-size envelope.
        reply = _error_reply(None, None, "protocol_error", str(bounds))
        wire = encode_frame(reply)
      return reply, wire
    finally:
      if task is not None:
        self._inflight.discard(task)

  async def _answer_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
    frame_type = frame.get("type")
    call_id = frame.get("call_id")
    if not self._reader._registered(self, frame):
      return _error_reply(frame_type, call_id, "bad_cell_token", "no live cell registration for this token")
    if frame_type == "hello":
      return _ok_reply(frame_type, call_id, {"protocol": BRIDGE_PROTOCOL, "parent_call_id": self.outer_call_id})
    if frame_type == "tool_call":
      return await self._tool_call(frame)
    if frame_type == "catalog":
      return self._catalog(frame)
    return _error_reply(frame_type, call_id, "protocol_error", f"unknown frame type {frame_type!r}")

  def _granted(self) -> Dict[str, Dict[str, Any]]:
    """The turn's grant by name: the bound catalog, read once per cell."""
    if self._grant is None:
      tool_definitions = self._tool_definitions
      definitions = tool_definitions() if tool_definitions is not None else []
      self._grant = {
        str(definition["name"]): definition
        for definition in definitions
        if definition.get("name")
      }
    return self._grant

  def _route_row(self, name: str, definition: Dict[str, Any]) -> Dict[str, Any]:
    return {
      **(self._catalog_metadata(name) if self._catalog_metadata is not None else {}),
      "name": name,
      "description": str(definition.get("description") or ""),
      "input_schema": definition.get("input_schema") or {},
    }

  def _refusal(self, dispatcher: Any, operation: str) -> Optional[str]:
    """Error code, or None when `operation ∈ grant ∩ code-callable` (§8)."""
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

    started = time.monotonic()
    try:
      # Select the same preparation owner as direct runner calls. Registered
      # Excel routes resolve workbook identity asynchronously before dispatch.
      prepare_async = getattr(dispatcher, "prepare_tool_call_async", None)
      if isinstance(prepare_async, _AsyncToolCallPreparer):
        prepared_call = await prepare_async(operation, tool_input)
      else:
        prepared_call = dispatcher.prepare_tool_call(operation, tool_input)
      executed_call = prepared_call

      def capture_executed(call: Any) -> None:
        nonlocal executed_call
        executed_call = call

      result, error = await dispatcher.dispatch_prepared(
        nested_call_id,
        operation,
        prepared_call,
        abort_event=self._abort_event,
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
    if settlement.outcome == "ok":
      self._record_sources(operation, nested_call_id, result, settlement.sources)
    reply = _ok_reply("tool_result", call_id, result)
    reply["nested_call_id"] = nested_call_id
    return reply

  def _record_sources(self, operation: str, nested_call_id: str, result: Any, settled_sources: Any) -> None:
    # Settled producer identities are authoritative (including input-dependent
    # vendor ids). The product extractor may additionally supply richer handles.
    handles = list(settled_sources)
    extract = self._extract_sources
    if extract is not None:
      try:
        handles.extend(extract(operation, result) or ())
      except Exception:
        log.exception("code_execute source extraction failed for %s", nested_call_id)
    for handle in handles:
      if not isinstance(handle, Mapping):
        continue
      # Registered settlement freezes nested values; project them exactly as
      # direct dispatch records do before registry keying or inline transport.
      handle = _materialize_source_identity(handle)
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
      existing = self._sources.get(key)
      if existing is None:
        self._sources[key] = {**handle, "produced_by_tool": canonical_dispatch_tool_name(operation)}
      else:
        # Preserve settled identity fields while retaining richer handle
        # metadata (e.g. filing accession) for the same registry identity.
        self._sources[key] = {**handle, **existing}

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


def _error_reply(frame_type: Any, call_id: Any, code: str, message: str) -> Dict[str, Any]:
  return {"type": frame_type, "call_id": call_id, "ok": False, "error": {"code": code, "message": message}}
